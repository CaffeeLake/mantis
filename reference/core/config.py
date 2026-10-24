# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections.abc import Mapping
from typing import Any, Optional, Tuple, Union

# Suppress ADK warning regarding Gemini via LiteLLM to maintain unified error handling and backoff
os.environ.setdefault("ADK_SUPPRESS_GEMINI_LITELLM_WARNINGS", "true")

try:
    import litellm
    litellm.drop_params = True
    litellm.suppress_debug_info = True

    # Suppress interactive reauth popups (e.g. macOS Touch ID / WebAuthn prompts)
    # when ADC credentials require reauthentication.
    try:
        import google.oauth2.reauth
        google.oauth2.reauth.is_interactive = lambda: False
    except Exception:
        pass

    try:
        import google.oauth2.credentials
        _orig_creds_init = google.oauth2.credentials.Credentials.__init__

        def _non_interactive_creds_init(self, *args, **kwargs):
            kwargs["enable_reauth_refresh"] = False
            _orig_creds_init(self, *args, **kwargs)
            self._enable_reauth_refresh = False

        google.oauth2.credentials.Credentials.__init__ = _non_interactive_creds_init
    except Exception:
        pass

    # Patch VertexAIAnthropicConfig to properly support adaptive thinking models (Claude 4.6+)
    # when structured output (response_format) is used. LiteLLM temporarily swaps model to
    # claude-3-sonnet-20240229 to force tool-based response_format, but this mistakenly converts
    # reasoning_effort into legacy thinking={"type": "enabled"} and retains non-1.0 temperature.
    try:
        from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import VertexAIAnthropicConfig
        from litellm.llms.anthropic.chat.transformation import (
            AnthropicConfig,
            REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT,
        )

        _orig_vertex_anthropic_map_openai_params = VertexAIAnthropicConfig.map_openai_params

        def _safe_vertex_anthropic_map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            original_model = model
            provider = getattr(self, "_resolved_provider", "vertex_ai")
            is_adaptive = False
            try:
                is_adaptive = AnthropicConfig._is_adaptive_thinking_model(original_model, provider)
            except Exception:
                pass

            res = _orig_vertex_anthropic_map_openai_params(
                self,
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=drop_params,
            )

            if is_adaptive:
                raw_effort = non_default_params.get("reasoning_effort")
                if isinstance(raw_effort, dict):
                    raw_effort = raw_effort.get("effort")
                if raw_effort and raw_effort != "none":
                    mapped_effort = REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT.get(raw_effort, raw_effort)
                    res["thinking"] = {"type": "adaptive"}
                    res["output_config"] = {"effort": mapped_effort}
                elif "thinking" in res and isinstance(res["thinking"], dict) and res["thinking"].get("type") == "enabled":
                    res["thinking"] = {"type": "adaptive"}

                # If thinking is enabled on an adaptive model, drop non-1.0 temperature and top_p
                if "thinking" in res:
                    temp = res.get("temperature")
                    if temp is not None and temp != 1:
                        res.pop("temperature", None)
                    res.pop("top_p", None)
            return res

        VertexAIAnthropicConfig.map_openai_params = _safe_vertex_anthropic_map_openai_params
    except Exception:
        pass
except Exception:
    pass

try:
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient, LlmCapabilities
    from pydantic import Field, PrivateAttr
except Exception:
    LiteLlm = object
    LiteLLMClient = object
    LlmCapabilities = None

    def Field(**kwargs: Any) -> Any:
        factory = kwargs.get("default_factory")
        if callable(factory):
            return factory()
        return kwargs.get("default", None)

    def PrivateAttr(**kwargs: Any) -> Any:
        return kwargs.get("default", None)


# ---------------------------------------------------------------------------
# P6: Per-finding review verdicts.
#
# A live run pushed 14 findings through ONE campaign-level ReviewVerdict: a
# single {route, reason} pair silently judged 14 independent vulnerabilities,
# so one review opinion either promoted or dismissed all of them at once.
# ExtendedReviewVerdict mirrors core.schemas.ReviewVerdict exactly (route,
# reason, extra="forbid") and adds an OPTIONAL finding_verdicts list carrying
# one verdict per finding id. It lives HERE rather than in core.schemas
# because the ADK runtime re-validates the agent's final response text against
# the bound output schema (google.adk.utils._schema_utils.validate_schema uses
# model_validate_json), and the canonical ReviewVerdict is extra="forbid":
# binding the strict schema while asking the reviewer for finding_verdicts
# would turn every extended response into an ADK validation error. The strict
# schema in core.schemas stays untouched as the wire-compatibility baseline
# (INV-6): when this class cannot be defined, graph_loader keeps the canonical
# schema and behavior is exactly pre-P6.
try:
    from typing import Literal as _Literal
    from pydantic import BaseModel as _PydanticBaseModel, ConfigDict as _PydanticConfigDict

    class FindingVerdict(_PydanticBaseModel):
        """One reviewer verdict for ONE finding (see ExtendedReviewVerdict)."""

        # extra="ignore" and route-as-plain-str are deliberate leniency: a
        # single malformed entry must never invalidate the 13 well-formed
        # sibling verdicts beside it in the same response. The classifier's
        # coercion layer lowercases routes, drops entries it cannot use, and
        # only ever persists the known dismissal vocabulary -- an unknown
        # route can therefore route conservatively but can never be learned.
        model_config = _PydanticConfigDict(extra="ignore")
        finding_id: int = Field(description="The numeric finding id exactly as returned by get_findings.")
        route: str = Field(description="'confirmed' or 'false_positive' for THIS finding only.")
        reason: str = Field(default="", description="One sentence justifying this finding's verdict.")

    class ExtendedReviewVerdict(_PydanticBaseModel):
        """Structured reviewer verdict with optional per-finding verdicts (P6)."""

        model_config = _PydanticConfigDict(extra="forbid")
        route: _Literal["confirmed", "false_positive"]
        reason: str = Field(description="One sentence justifying the verdict.")
        # The empty-list default IS the fail-safe contract (INV-6): an absent
        # or empty list means campaign-level routing and persistence exactly
        # as before this field existed, so old prompts, old models, replayed
        # sessions, and synthesized refusal fallbacks lose nothing.
        finding_verdicts: list[FindingVerdict] = Field(
            default_factory=list,
            description=(
                "One verdict per finding id from get_findings. "
                "Empty list = the top-level verdict applies to the whole campaign."
            ),
        )
except Exception:  # pragma: no cover - only when pydantic itself is absent
    FindingVerdict = None
    ExtendedReviewVerdict = None

DEFAULT_MODEL = "vertex_ai/gemini-3.8-flash"
SUPPORTED_SANDBOXES = ("static-only", "static", "gvisor", "microsandbox", "gce")
RECOMMENDED_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "claude-opus-5",
    "vertex_ai/gemini-3.8-flash",
    "vertex_ai/gemini-3.5-flash-lite",
    "vertex_ai/claude-opus-5",
    "vertex_ai/zai_org/glm-5.2-maas",
)

PLACEHOLDER_STRINGS = {
    "YOUR_PROJECT_ID",
    "YOUR_PROJECT",
    "YOUR_GCP_PROJECT",
    "<YOUR_PROJECT_ID>",
    "<PROJECT_ID>",
    "YOUR_API_KEY",
    "<YOUR_API_KEY>",
    "TODO",
    "CHANGE_ME",
    "REPLACE_ME",
    "",
}


def is_placeholder(val: Any) -> bool:
    """Checks if a string or value represents an unconfigured placeholder."""
    if val is None:
        return True
    s = str(val).strip()
    if not s or s.upper() in PLACEHOLDER_STRINGS:
        return True
    if s.upper().startswith(("YOUR_", "<YOUR_", "CHANGE_ME", "REPLACE_ME")):
        return True
    # Token-level check for composite paths/models (e.g. openai/YOUR_API_KEY)
    for token in s.upper().split("/"):
        t = token.strip()
        if t in PLACEHOLDER_STRINGS or t.startswith(("YOUR_", "<YOUR_", "CHANGE_ME", "REPLACE_ME")):
            return True
    return False


def normalize_model_id(model_id: str) -> str:
    """Normalizes model names and routes bare gemini/claude models to vertex_ai or gemini/ based on credentials."""
    if not model_id:
        return DEFAULT_MODEL
    cleaned = model_id.strip()
    if cleaned.startswith("gemini-"):
        # If explicit Google AI Studio API key exists, route to gemini/ for LiteLLM
        if os.environ.get("GEMINI_API_KEY"):
            return f"gemini/{cleaned}"
        # If running in GCP / Vertex environment without explicit Google AI Studio API key, route to vertex_ai/
        if (
            os.environ.get("VERTEXAI_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        ):
            return f"vertex_ai/{cleaned}"
    if cleaned.startswith("claude-"):
        # Route bare claude models to vertex_ai when GCP credentials exist and no direct Anthropic API key is set
        if not os.environ.get("ANTHROPIC_API_KEY") and (
            os.environ.get("VERTEXAI_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        ):
            return f"vertex_ai/{cleaned}"
    if cleaned in (
        "glm-5.2-maas",
        "zai-org/glm-5.2-maas",
        "vertex_ai/glm-5.2-maas",
        "vertex_ai/zai_org/glm-5.2-maas",
        "vertex_ai/zai-org/glm-5.2-maas",
        "vertex_ai/openai/zai-org/glm-5.2-maas",
    ):
        return "vertex_ai/openai/zai-org/glm-5.2-maas"
    return cleaned


def is_rate_limit_error(e: Exception) -> bool:
    """Detects whether an exception represents a 429 / RateLimitError / RESOURCE_EXHAUSTED error."""
    try:
        import litellm
        if isinstance(e, (getattr(litellm, "RateLimitError", ()), getattr(litellm.exceptions, "RateLimitError", ()))):
            return True
    except Exception:
        pass
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status in (429, 503, 529):
        return True
    resp = getattr(e, "response", None)
    if resp and getattr(resp, "status_code", None) in (429, 503, 529):
        return True
    msg = str(e).lower()
    if any(k in msg for k in ("ratelimiterror", "resource_exhausted", "quota exceeded", "rate limit", "rate_limit", "overloaded", "too many requests")):
        return True
    if "429" in msg and any(k in msg for k in ("quota", "token", "limit", "exhausted", "aiplatform", "prediction")):
        return True
    return False


def is_retryable_llm_error(e: Exception) -> bool:
    """Detects whether an exception represents a retryable transient error (429, 408, 5xx, timeout, or network reset)."""
    if is_auth_error(e) or isinstance(e, (MantisAuthError, PermissionError, FileNotFoundError)):
        return False
    if is_rate_limit_error(e):
        return True
    # An empty STOP turn is retried like a 429: same backoff, same patience.
    # Its cost is metered where it is raised (see MantisEmptyResponseError).
    if isinstance(e, MantisEmptyResponseError):
        return True
    if isinstance(e, json.decoder.JSONDecodeError):
        return True
    try:
        import litellm
        timeout_classes = (
            getattr(litellm, "Timeout", ()),
            getattr(getattr(litellm, "exceptions", None), "Timeout", ()),
            getattr(litellm, "InternalServerError", ()),
            getattr(getattr(litellm, "exceptions", None), "InternalServerError", ()),
            getattr(litellm, "ServiceUnavailableError", ()),
            getattr(getattr(litellm, "exceptions", None), "ServiceUnavailableError", ()),
            getattr(litellm, "BadGatewayError", ()),
            getattr(getattr(litellm, "exceptions", None), "BadGatewayError", ()),
        )
        valid_timeout_classes = tuple(c for c in timeout_classes if isinstance(c, type))
        if valid_timeout_classes and isinstance(e, valid_timeout_classes):
            return True
    except Exception:
        pass
    if isinstance(e, (asyncio.TimeoutError, TimeoutError, ConnectionResetError, BrokenPipeError)):
        return True
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status in (408, 429, 500, 502, 503, 504, 529):
        return True
    resp = getattr(e, "response", None)
    if resp and getattr(resp, "status_code", None) in (408, 429, 500, 502, 503, 504, 529):
        return True
    msg = str(e).lower()
    if any(
        k in msg
        for k in (
            "timeout",
            "timed out",
            "connection reset",
            "connection closed",
            "sockettimeout",
            "remotepathreset",
            "broken pipe",
            "internal server error",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        )
    ):
        return True
    return False


def extract_retry_after(e: Exception) -> Optional[float]:
    """Extracts suggested retry delay in seconds from response headers if present."""
    headers = getattr(e, "headers", None)
    if not headers:
        resp = getattr(e, "response", None)
        if resp:
            headers = getattr(resp, "headers", None)
    if headers and isinstance(headers, (dict, Mapping)):
        for k, v in headers.items():
            if k.lower() == "retry-after":
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
    return None


def extract_rate_limit_detail(e: Exception) -> str:
    """Extracts a succinct summary of the rate limit reason from the exception."""
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg:
        m = re.search(r"Quota exceeded for ([^.\s]+(?:\.[^.\s]+)*)", msg)
        if m:
            metric = m.group(1).split("/")[-1]
            return f"Quota exceeded: {metric}"
        return "RESOURCE_EXHAUSTED"
    if "tokens_per_minute" in msg:
        return "Tokens per minute limit exceeded"
    if "requests_per_minute" in msg:
        return "Requests per minute limit exceeded"
    return "Rate limit (429)"


def compute_full_jitter_delay(
    attempt: int,
    initial_delay: float = 5.0,
    max_delay: float = 60.0,
    min_offset: float = 5.0,
    backoff_factor: float = 2.0,
    retry_after: Optional[float] = None,
    max_remaining: Optional[float] = None,
) -> float:
    """Computes backoff delay using full jitter with a minimum offset."""
    upper_bound = min(max_delay, max(min_offset, initial_delay * (backoff_factor ** attempt)))
    if upper_bound <= min_offset:
        delay = min_offset
    else:
        delay = random.uniform(min_offset, upper_bound)

    if retry_after is not None and retry_after > 0:
        delay = min(max_delay, max(delay, retry_after))

    if max_remaining is not None:
        delay = min(delay, max(1.0, max_remaining))

    return delay


class MantisAuthError(RuntimeError):
    """Raised when an unrecoverable LLM authentication or token refresh failure occurs."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


# Process exit status for a MantisAuthError. Distinct from 1 (a scan that
# failed) and 2 (a budget pause) so a caller can recognize "the provider
# rejected the credentials" from the status alone. Log text is not a safe
# channel for that: the pipeline echoes file names and file contents that the
# code under audit controls, so any marker string can be forged there.
EXIT_AUTH = 4


class MantisStreamingTruncationError(RuntimeError):
    """Raised when streaming tool call arguments are truncated mid-stream by output limits or premature chunk completion."""
    pass


class MantisStreamInterruptedError(RuntimeError):
    """Raised when a resilient stream fails after at least one chunk was delivered.

    Everything before the first chunk is retried transparently: no output has been
    observed, so a fresh stream is equivalent to a slow first attempt. After that,
    a transparent retry would re-sample the model and splice two different
    completions together -- silently corrupting output the consumer has already
    processed. The wrapper therefore refuses to guess: it reports how far delivery
    got and leaves the continuation policy (re-prompt, discard, resume) to the
    caller.
    """

    def __init__(self, message: str, chunks_yielded: int = 0, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.chunks_yielded = chunks_yielded
        self.original_exception = original_exception


class MantisEmptyResponseError(RuntimeError):
    """A non-streaming turn finished with STOP and no parts: no text, no tool
    call, no error code. The model billed the prompt and said nothing.

    Left alone, ADK reports the turn as MODEL_RETURNED_NO_CONTENT, the node
    moves on with nothing, and the whole campaign is marked failed over one
    empty sample. Raised inside the model layer instead, so the turn goes
    back through the same backoff loop as a 429. The difference from a 429
    is that each attempt was a billed prompt, so the response's usage is
    metered against the run budget before this is raised: the token ceiling
    bounds the worst case, as it bounds everything else. Past the patience
    window -- or past ``MANTIS_LLM_EMPTY_TURN_MAX_ATTEMPTS`` empty turns on
    the one request, whichever comes first -- the empty response is passed
    on unchanged, and the pipeline sees exactly what it saw before.
    """

    def __init__(self, message: str, response: Any = None):
        super().__init__(message)
        self.response = response


# How many empty STOP turns one request may draw before the empty turn is
# passed on. The general patience window is for faults that pass (quota,
# 5xx); a request the model has answered with nothing six times in a row is
# being answered, and the answer is nothing. A calibrator with twenty
# findings and an hour of patience each would otherwise stall a run for a
# day on a prompt the model will never fill.
DEFAULT_EMPTY_TURN_MAX_ATTEMPTS = 6


def structured_llm_request(prompt: str, system_instruction: str, schema_cls: Any) -> Any:
    """A one-shot request whose answer is a ``set_model_response`` call.

    The tool is registered with ``LlmRequest.append_tools``, which fills both
    ``config.tools`` (a ``types.Tool`` carrying the declaration) and
    ``tools_dict``. Putting the ``SetModelResponseTool`` straight into
    ``GenerateContentConfig(tools=[...])`` looks the same and is not: ADK's
    LiteLlm adapter forwards only ``types.Tool`` instances and drops anything
    else without a word, so the model was being told to call a tool it had
    never been given. Some models then answer in prose that the caller can
    still parse; some answer with nothing, and an empty turn is retried.
    """
    from google.adk.models.llm_request import LlmRequest
    from google.adk.tools.set_model_response_tool import SetModelResponseTool
    from google.genai import types

    req = LlmRequest(
        contents=[types.Content(parts=[types.Part.from_text(text=prompt)], role="user")],
        config=types.GenerateContentConfig(system_instruction=system_instruction),
    )
    req.append_tools([SetModelResponseTool(schema_cls)])
    return req


def is_empty_stop_response(response: Any) -> bool:
    """The condition ADK flags as MODEL_RETURNED_NO_CONTENT, checked on the
    response before ADK sees it. Streaming is the caller's business: a
    terminal finish-only chunk legitimately follows streamed content."""
    if getattr(response, "partial", False) or getattr(response, "error_code", None):
        return False
    from google.genai import types as genai_types

    if getattr(response, "finish_reason", None) not in (genai_types.FinishReason.STOP, "STOP"):
        return False
    content = getattr(response, "content", None)
    return not content or not getattr(content, "parts", None)


def meter_response_usage(response: Any) -> None:
    """Charges a response's usage to the run budget from inside the model
    layer. The pipeline meters from events, and a response the model layer
    swallows never becomes one; without this a retried empty turn would be
    free as far as the budget could tell. A budget ceiling crossed here
    raises BudgetExceededError, which is the point."""
    from core.context import current_run_context

    ctx = current_run_context.get()
    controller = getattr(ctx, "budget_controller", None) if ctx else None
    usage = getattr(response, "usage_metadata", None)
    total = int(getattr(usage, "total_token_count", 0) or 0) if usage else 0
    if controller is None or total <= 0:
        return
    cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
    # The weight of a cached token is the run budget's to decide (a CI run
    # counts every token at full price); this layer only reports the counts.
    controller.record_tokens(
        total, cached_count=cached,
        prompt_count=int(getattr(usage, "prompt_token_count", 0) or 0),
        output_count=int(getattr(usage, "candidates_token_count", 0) or 0))


def _observe(name: str, n: int = 1) -> None:
    """Counts a request-level event on the run's spend report (core.budget)."""
    try:
        from core.budget import observe
        observe(name, n)
    except Exception:
        pass


def context_edits_applied(response: Any) -> tuple[int, int]:
    """(edits applied, input tokens cleared) that an Anthropic response
    reports under `context_management.applied_edits`, which LiteLLM keeps in
    the message's provider_specific_fields. (0, 0) when the response says
    nothing: no editing was requested, the trigger was not reached, or the
    provider does not report it."""
    edits = 0
    cleared = 0
    try:
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        fields = getattr(message, "provider_specific_fields", None) or {}
        if not isinstance(fields, Mapping) or not fields:
            hidden = getattr(response, "_hidden_params", None) or {}
            fields = hidden.get("provider_specific_fields") or {} if isinstance(hidden, Mapping) else {}
        cm = fields.get("context_management") if isinstance(fields, Mapping) else None
        applied = cm.get("applied_edits") if isinstance(cm, Mapping) else None
        for edit in applied or []:
            if not isinstance(edit, Mapping):
                continue
            edits += 1
            cleared += int(edit.get("cleared_input_tokens") or 0)
    except Exception:
        return 0, 0
    return edits, cleared


def _observe_context_edits(response: Any) -> None:
    edits, cleared = context_edits_applied(response)
    if edits:
        _observe("context_edits_applied", edits)
    if cleared:
        _observe("context_edit_cleared_tokens", cleared)


class ContextBudgetExceededError(RuntimeError):
    """Raised before dispatch when a request cannot possibly fit the model's context window.

    Sending it anyway costs a full upload of the payload, a provider-side rejection, and --
    because the node retry layer cannot tell a deterministic overflow from a transient fault
    -- two more identical uploads of the same doomed request. The failure is deterministic in
    the request itself, so the only useful thing to do with it is refuse early and say which
    message was too big.
    """

    def __init__(self, message: str, estimated_tokens: int = 0, limit: int = 0):
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.limit = limit


# Exceptions whose outcome is fully determined by the request, so re-running the node with
# the identical request can only reproduce them. ADK's node retry cannot distinguish these
# from transient faults and will burn the full attempt budget on them.
#
# Matched by class name because several of these types live in third-party packages that may
# not be importable in every deployment, and a missing import must not silently disable the
# rule. Note this is intentionally NOT the same list as the log-suppression filter below:
# retrying an error and reporting it are separate decisions, and a context overflow must stay
# loudly visible precisely because the operator is the one who has to act on it.
_NON_RETRYABLE_EXC_NAMES = (
    "BudgetExceededError",
    "LlmCallsLimitExceededError",
    "MantisAuthError",
    "ContextBudgetExceededError",
    "ContextWindowExceededError",
    # litellm 4xx families whose outcome is fully determined by the request;
    # re-running the identical request can only reproduce the failure.
    "NotFoundError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "UnprocessableEntityError",
    "ContentPolicyViolationError",
    "UnsupportedParamsError",
)


# Characters per token. Deliberately a constant rather than a real tokenizer: tokenizing a
# multi-megabyte payload to discover that it is multi-megabyte is self-defeating, and the
# guard only needs to separate "plausibly fits" from "an order of magnitude over". Real code
# runs denser than 4 chars/token, so this UNDER-estimates and the guard errs toward letting
# a borderline request through to the provider -- the fail-open direction, which matters
# because this is a reliability control, not a security boundary.
_CHARS_PER_TOKEN = 4

# Fraction of the window a request may occupy before it is refused. Below 1.0 because the
# estimate is approximate and the response needs room too; a request at 95% of the window
# has no space left to answer in.
_CONTEXT_BUDGET_RATIO = 0.9


def _message_char_len(message: Any) -> int:
    """Total character length of one chat message, including structured content parts."""
    if isinstance(message, str):
        return len(message)
    if not isinstance(message, Mapping):
        return len(str(message))

    total = 0
    for key in ("role", "name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str):
            total += len(value)

    content = message.get("content")
    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, (list, tuple)):
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                total += len(text) if isinstance(text, str) else len(str(part))
            else:
                total += len(str(part))
    elif content is not None:
        total += len(str(content))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, (list, tuple)):
        for call in tool_calls:
            total += len(str(call))

    return total


def estimate_prompt_tokens(messages: Any, tools: Any = None) -> int:
    """Cheap upper-bound-ish token estimate for a request, without tokenizing it."""
    total_chars = 0
    if isinstance(messages, (list, tuple)):
        for message in messages:
            total_chars += _message_char_len(message)
    elif messages is not None:
        total_chars += _message_char_len(messages)

    if tools:
        try:
            total_chars += len(json.dumps(tools, default=str))
        except (TypeError, ValueError):
            total_chars += len(str(tools))

    return total_chars // _CHARS_PER_TOKEN


def resolve_context_limit(model: Any) -> Optional[int]:
    """Returns the model's max input tokens, or None when it cannot be determined.

    Fails open on purpose. An unknown window means we cannot prove the request is doomed,
    and refusing a request that would have succeeded is a worse failure than the one this
    guard exists to prevent.
    """
    try:
        import litellm
    except Exception:
        return None

    model_name = str(model)
    # Only ever input limits. litellm's get_max_tokens()/"max_tokens" report the max OUTPUT
    # tokens -- often a couple of orders of magnitude smaller than the context window -- so
    # using either as an input budget would refuse ordinary requests.
    fn = getattr(litellm, "get_max_input_tokens", None)
    if fn is not None:
        try:
            value = fn(model_name)
        except Exception:
            value = None
        if isinstance(value, int) and value > 0:
            return value

    try:
        info = litellm.get_model_info(model_name)
    except Exception:
        return None
    if isinstance(info, Mapping):
        value = info.get("max_input_tokens")
        if isinstance(value, int) and value > 0:
            return value
    return None


def _largest_message_summary(messages: Any) -> str:
    """Names the biggest contributor to an oversized request so the failure is actionable."""
    if not isinstance(messages, (list, tuple)) or not messages:
        return ""
    sizes = [(_message_char_len(m), i, m) for i, m in enumerate(messages)]
    chars, index, message = max(sizes, key=lambda item: item[0])
    role = message.get("role", "?") if isinstance(message, Mapping) else "?"
    return (
        f" Largest contributor: message {index} (role={role}) at {chars:,} characters "
        f"(~{chars // _CHARS_PER_TOKEN:,} tokens)."
    )


# Share of the model's input window that ONE tool response may occupy. The window-relative
# request guard below refuses a request that no longer fits; this is the control that keeps
# a single tool response from being the reason it does not. A quarter leaves room for the
# instruction, the conversation so far, a few more tool responses and the answer. Measured
# failure this exists for: a 1 MiB read_file (~262k tokens) and a 975k-char recorded summary
# each delivered whole into a 272k-token window, which refused the request on the spot.
TOOL_RESPONSE_WINDOW_FRACTION = 0.25

# No model-derived cap goes below this many characters (~10k tokens). The verdict stages
# read tool output looking for a bounds check; a cut tight enough to hide one reads as
# "the check is missing" (see tools/research_tools.MAX_READ_SIZE).
MIN_TOOL_RESPONSE_CHARS = 40_000

_TOOL_RESPONSE_CAP_ENV = "MANTIS_MAX_TOOL_RESPONSE_CHARS"


def resolve_tool_response_char_cap(model: Any) -> Optional[int]:
    """Characters one tool response may carry into `model`'s context, or None for no cap.

    Derived from the model's input window (``TOOL_RESPONSE_WINDOW_FRACTION`` of it, in
    characters at the guard's chars-per-token estimate) and from the operator's
    per-request input cap when that is tighter. ``MANTIS_MAX_TOOL_RESPONSE_CHARS`` is an
    absolute operator override that applies whether or not the model is known. An
    unknown model with no override means no cap -- the request guard still refuses what
    cannot fit, and the per-tool byte limits still apply -- which is the fail-open
    direction every control in this file takes.
    """
    raw = str(os.environ.get(_TOOL_RESPONSE_CAP_ENV, "") or "").strip()
    if raw:
        try:
            override = int(raw)
        except ValueError:
            override = 0
        if override > 0:
            return override

    limits = [l for l in (resolve_context_limit(model), resolve_input_token_cap()) if l]
    if not limits:
        return None
    tokens = int(min(limits) * TOOL_RESPONSE_WINDOW_FRACTION)
    return max(tokens * _CHARS_PER_TOKEN, MIN_TOOL_RESPONSE_CHARS)


# --- Anthropic context editing -----------------------------------------------
#
# Server-side clearing of a conversation's older tool results (Anthropic's
# `context_management`, edit `clear_tool_uses`). The provider applies it before
# the model reads the request, so a long stage stays inside the window while
# the request we send remains append-only -- the one form of history
# management that models binding their reasoning to its prefix accept. The
# cleared results are replaced by a placeholder the model can see, and the
# edit is reported back in the response. Measured on Vertex AI: a
# 210,105-token request was served at 529 tokens with the two tool results
# reported cleared. Claude models only: the parameter is Anthropic's, on any
# route (vertex_ai/, anthropic/, bedrock/).
#
# The trigger sits below the node context ceiling (0.8) so the provider acts
# first and the ceiling is the backstop; `clear_at_least` is sized so one edit
# buys a long stretch of further work (each edit invalidates the prompt cache
# from the edit point); `keep` leaves the newest results verbatim.
CONTEXT_EDITING_TRIGGER_RATIO = 0.5
CONTEXT_EDITING_KEEP_TOOL_USES = 5
CONTEXT_EDITING_CLEAR_RATIO = 0.2
CONTEXT_EDITING_EDIT_TYPE = "clear_tool_uses_20250919"
_CONTEXT_EDITING_ENV = "MANTIS_CONTEXT_EDITING"
_OFF_WORDS = frozenset({"off", "false", "0", "no", "disabled"})


def is_claude_model(model: Any) -> bool:
    """A Claude model on any provider route."""
    return "claude" in str(model or "").lower()


def _context_editing_off(config: Optional[dict]) -> bool:
    if str(os.environ.get(_CONTEXT_EDITING_ENV, "") or "").strip().lower() in _OFF_WORDS:
        return True
    return isinstance(config, dict) and str(config.get("context_editing", "auto") or "").strip().lower() in _OFF_WORDS


def anthropic_context_editing(model: Any, config: Optional[dict] = None) -> Optional[dict]:
    """The `context_management` parameter for `model`, or None when it must not
    be sent: not a Claude model, switched off (config `context_editing: off`
    or MANTIS_CONTEXT_EDITING=off), or a window we cannot size against."""
    if not is_claude_model(model):
        return None
    if _context_editing_off(config):
        return None
    limit = resolve_context_limit(model)
    if not limit:
        return None
    return {"edits": [{
        "type": CONTEXT_EDITING_EDIT_TYPE,
        "trigger": {"type": "input_tokens", "value": int(limit * CONTEXT_EDITING_TRIGGER_RATIO)},
        "keep": {"type": "tool_uses", "value": CONTEXT_EDITING_KEEP_TOOL_USES},
        "clear_at_least": {"type": "input_tokens", "value": int(limit * CONTEXT_EDITING_CLEAR_RATIO)},
    }]}


def client_context_editing(model: Any, config: Optional[dict] = None) -> Optional[dict]:
    """The harness-side counterpart of `anthropic_context_editing` for models
    that do not bind their reasoning to the request prefix (Gemini, OpenAI):
    the same trigger, keep and clear-at-least, applied by the harness to the
    request before dispatch (core.graph_loader). `{"trigger_tokens", "keep",
    "clear_at_least_tokens"}`, or None for a Claude model (the provider does
    it), when switched off, or a window we cannot size against."""
    if is_claude_model(model):
        return None
    if _context_editing_off(config):
        return None
    limit = resolve_context_limit(model)
    if not limit:
        return None
    return {
        "trigger_tokens": int(limit * CONTEXT_EDITING_TRIGGER_RATIO),
        "keep": CONTEXT_EDITING_KEEP_TOOL_USES,
        "clear_at_least_tokens": int(limit * CONTEXT_EDITING_CLEAR_RATIO),
    }


def context_editing_keeps(context_management: Any) -> Optional[int]:
    """How many newest tool results a `clear_tool_uses` edit keeps verbatim, or
    None when the parameter carries no such edit. What the client-side
    estimates (request guard, node context ceiling) treat as still present."""
    if not isinstance(context_management, Mapping):
        return None
    for edit in context_management.get("edits") or []:
        if isinstance(edit, Mapping) and str(edit.get("type", "")).startswith("clear_tool_uses"):
            keep = edit.get("keep") or {}
            if isinstance(keep, Mapping) and keep.get("type") == "tool_uses":
                try:
                    return max(0, int(keep.get("value", 0)))
                except (TypeError, ValueError):
                    return 0
            return 0
    return None


_CLEARED_TOOL_RESULT = "[cleared by context editing]"


def _messages_as_the_provider_sees_them(messages: Any, keep: int) -> Any:
    """The chat messages with every tool result but the newest `keep` replaced
    by the provider's placeholder -- the shape the model reads once a
    clear_tool_uses edit has fired. Tool calls themselves are untouched."""
    if not isinstance(messages, (list, tuple)):
        return messages
    tool_indexes = [i for i, m in enumerate(messages) if isinstance(m, Mapping) and m.get("role") == "tool"]
    cleared = set(tool_indexes[:-keep] if keep else tool_indexes)
    if not cleared:
        return messages
    return [
        ({**m, "content": _CLEARED_TOOL_RESULT} if i in cleared else m)
        for i, m in enumerate(messages)
    ]


def resolve_input_token_cap() -> Optional[int]:
    """An absolute per-request input ceiling from MANTIS_MAX_INPUT_TOKENS, or None.

    The window-relative guard below bounds a request by the MODEL's limit, which is
    whatever the model happens to have (a million tokens, say) and nothing at all for a
    model litellm does not know. This cap is the OPERATOR's limit: a number of tokens
    one request may carry regardless of model, so the cost of one call has a ceiling
    that does not move when the model changes and does not vanish behind a gateway.
    Unset, empty, zero, or unparsable means no cap, which is the previous behaviour.
    """
    raw = str(os.environ.get("MANTIS_MAX_INPUT_TOKENS", "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def enforce_context_budget(model: Any, messages: Any, tools: Any = None,
                           context_management: Any = None) -> None:
    """Refuses a request that cannot fit the model's context window, or that exceeds the
    operator's per-request input cap, before dispatching it.

    `context_management` is the request's Anthropic context-editing parameter, when
    any: the provider clears older tool results before the model reads the request,
    so the estimate counts those as cleared too -- otherwise this would refuse a
    request the provider was about to make fit."""
    limit = resolve_context_limit(model)
    cap = resolve_input_token_cap()
    if not limit and not cap:
        return

    keep = context_editing_keeps(context_management)
    if keep is not None:
        messages = _messages_as_the_provider_sees_them(messages, keep)
    estimated = estimate_prompt_tokens(messages, tools)
    window_budget = int(limit * _CONTEXT_BUDGET_RATIO) if limit else None
    budget = min(b for b in (window_budget, cap) if b is not None)
    if estimated <= budget:
        return

    capped = cap is not None and (window_budget is None or cap <= window_budget)
    if capped:
        reason = (f"Request to '{model}' exceeds the per-request input cap: estimated "
                  f"~{estimated:,} input tokens against MANTIS_MAX_INPUT_TOKENS={cap:,}.")
    else:
        reason = (f"Request to '{model}' is too large for its context window: estimated "
                  f"~{estimated:,} input tokens against a {limit:,}-token window "
                  f"({estimated / limit:.1f}x the window, budget {budget:,}).")
    raise ContextBudgetExceededError(
        reason
        + f"{_largest_message_summary(messages)}"
        " Refused before dispatch; retrying the identical request cannot succeed. "
        "Narrow the tool output feeding this node (for example, pass a 'directory' to "
        "list_files or a smaller range to read_file).",
        estimated_tokens=estimated,
        limit=cap if capped else limit,
    )


def is_auth_error(e: Optional[Exception]) -> bool:
    """Detects whether an exception represents an authentication or token refresh failure."""
    if e is None:
        return False
    if isinstance(e, MantisAuthError):
        return True

    # Check known exception class names
    exc_cls_name = getattr(getattr(e, "__class__", None), "__name__", "")
    if exc_cls_name in (
        "RefreshError",
        "DefaultCredentialsError",
        "ReauthError",
        "ReauthFailError",
        "ReauthSamlChallengeFailError",
        "AuthenticationError",
    ):
        return True

    # Check known exception classes
    auth_classes = []
    try:
        import google.auth.exceptions
        for name in (
            "RefreshError",
            "DefaultCredentialsError",
            "ReauthFailError",
            "ReauthSamlChallengeFailError",
            "OAuthError",
            "UserAccessTokenError",
        ):
            cls = getattr(google.auth.exceptions, name, None)
            if cls is not None and isinstance(cls, type):
                auth_classes.append(cls)
    except Exception:
        pass

    try:
        import litellm
        for name in ("AuthenticationError",):
            cls = getattr(litellm, name, None) or getattr(getattr(litellm, "exceptions", None), name, None)
            if cls is not None and isinstance(cls, type):
                auth_classes.append(cls)
    except Exception:
        pass

    valid_classes = tuple(c for c in auth_classes if isinstance(c, type))
    if valid_classes and isinstance(e, valid_classes):
        return True

    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status == 401:
        return True
    resp = getattr(e, "response", None)
    if resp and getattr(resp, "status_code", None) == 401:
        return True

    curr = e
    while curr is not None:
        msg = str(curr).lower()
        if any(
            phrase in msg
            for phrase in (
                "reauthentication is needed",
                "gcloud auth application-default login",
                "invalid_grant",
                "your default credentials were not found",
                "could not automatically determine credentials",
                "reauthentication required",
                "reauthentication challenge",
                "authenticationerror",
                "credentials are expired",
                "credentials expired",
                "failed to retrieve auth token",
                "invalid api key",
                "incorrect api key",
                "unauthenticated",
            )
        ):
            return True
        curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

    return False


def format_auth_error_message(e: Exception, model: str = "") -> str:
    """Formats an informative, clean authentication error banner without tracebacks."""
    err_text = str(e).strip()
    if err_text.startswith("=" * 10) and "[AUTHENTICATION ERROR]" in err_text:
        return err_text
    cause_msg = ""
    curr = e
    while curr is not None:
        c_str = str(curr).strip()
        if c_str and not c_str.startswith("Traceback") and not c_str.startswith("PIPELINE"):
            cause_msg = c_str
        curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

    cause_clean = cause_msg or err_text
    if "AuthenticationError:" in cause_clean:
        cause_clean = cause_clean.split("AuthenticationError:")[-1].strip()

    model_label = f" for '{model}'" if model else ""
    lines = [
        "=" * 80,
        f" ❌ [AUTHENTICATION ERROR] Cloud / LLM Authentication Failed{model_label}",
        "=" * 80,
        f"  • Cause: {cause_clean}",
    ]

    is_gcp = any(
        k in cause_clean.lower() or k in model.lower()
        for k in ("gcloud", "google", "vertex", "gemini", "adc", "application-default", "refresh_token", "rapt")
    ) or not model or model.startswith("vertex_ai/") or model.startswith("gemini/")

    is_anthropic = "anthropic" in model.lower() or "anthropic" in cause_clean.lower()
    is_openai = "openai" in model.lower() or "openai" in cause_clean.lower()

    if is_gcp:
        lines.extend([
            "  • If using Vertex AI with personal Application Default Credentials (ADC):",
            "      gcloud auth application-default login",
            "  • Or if using a Google Cloud service account key:",
            '      export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account_key.json"',
        ])
    if is_anthropic:
        lines.extend([
            "  • If using Anthropic API directly:",
            '      export ANTHROPIC_API_KEY="your-anthropic-api-key"',
        ])
    if is_openai:
        lines.extend([
            "  • If using OpenAI API directly:",
            '      export OPENAI_API_KEY="your-openai-api-key"',
        ])

    lines.append("=" * 80)
    return "\n".join(lines)


def clear_vertex_credential_caches() -> None:
    """Clears cached credentials across all LiteLLM Vertex AI handlers to force token reload."""
    try:
        import gc
        from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

        for obj in gc.get_objects():
            if isinstance(obj, VertexBase):
                mapping = getattr(obj, "_credentials_project_mapping", None)
                if isinstance(mapping, dict):
                    mapping.clear()
    except Exception:
        pass


def is_token_refreshable_auth_error(e: Exception) -> bool:
    """Checks if an auth error is due to an expired access token that can be refreshed automatically."""
    curr = e
    while curr is not None:
        msg = str(curr).lower()
        if any(
            phrase in msg
            for phrase in (
                "reauthentication is needed",
                "gcloud auth application-default login",
                "invalid_grant",
                "your default credentials were not found",
                "could not automatically determine credentials",
                "invalid api key",
                "incorrect api key",
            )
        ):
            return False
        curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if status == 401:
        return True
    resp = getattr(e, "response", None)
    if resp and getattr(resp, "status_code", None) == 401:
        return True

    curr = e
    while curr is not None:
        msg = str(curr).lower()
        if any(
            phrase in msg
            for phrase in (
                "credentials are expired",
                "credentials expired",
                "unauthenticated",
                "401",
                "token has expired",
                "access token expired",
            )
        ):
            return True
        curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)

    return False


def try_refresh_auth() -> bool:
    """Attempts to refresh Application Default Credentials and clear LiteLLM token caches."""
    clear_vertex_credential_caches()
    try:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        return getattr(creds, "valid", False)
    except Exception:
        return False


class ExpectedErrorLoggingFilter(logging.Filter):
    """Filters out noisy, multi-frame tracebacks for expected operational conditions

    (such as MantisAuthError or BudgetExceededError), while strictly preserving full
    tracebacks for unexpected bugs or runtime exceptions.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exc_val = record.exc_info[1]
            if exc_val is not None:
                if is_auth_error(exc_val) or isinstance(exc_val, MantisAuthError):
                    return False
                exc_cls_name = getattr(getattr(exc_val, "__class__", None), "__name__", "")
                if exc_cls_name in (
                    "BudgetExceededError",
                    "LlmCallsLimitExceededError",
                    "MantisAuthError",
                ):
                    return False
                cause = getattr(exc_val, "__cause__", None) or getattr(exc_val, "__context__", None)
                if cause is not None and (
                    is_auth_error(cause)
                    or isinstance(cause, MantisAuthError)
                    or getattr(getattr(cause, "__class__", None), "__name__", "") in (
                        "BudgetExceededError",
                        "LlmCallsLimitExceededError",
                        "MantisAuthError",
                    )
                ):
                    return False

        # Also suppress messages directly matching authentication failure signatures
        msg = record.getMessage() if hasattr(record, "getMessage") else str(record.msg)
        if any(
            pattern in msg
            for pattern in (
                "Failed to load vertex credentials",
                "Reauthentication is needed",
                "gcloud auth application-default login",
            )
        ):
            return False

        return True


_expected_error_filter = ExpectedErrorLoggingFilter()

if getattr(logging, "lastResort", None):
    logging.lastResort.addFilter(_expected_error_filter)

_orig_logger_add_handler = logging.Logger.addHandler


def _filtered_add_handler(self: logging.Logger, h: logging.Handler) -> None:
    h.addFilter(_expected_error_filter)
    return _orig_logger_add_handler(self, h)


logging.Logger.addHandler = _filtered_add_handler


def apply_expected_error_filter() -> None:
    """Attaches _expected_error_filter to root, all existing loggers, and LiteLLM/ADK handlers."""
    root_lg = logging.getLogger()
    root_lg.addFilter(_expected_error_filter)
    for h in root_lg.handlers:
        h.addFilter(_expected_error_filter)

    known_loggers = [
        "",
        "LiteLLM",
        "litellm",
        "litellm._logging",
        "google_adk",
        "google_adk.google.adk.runners",
        "google.adk",
        "google.adk.runners",
        "google.adk.workflow._node_runner",
        "google_adk.google.adk.workflow._node_runner",
        "google.adk.workflow",
        "google_adk.google.adk.workflow",
        "google_adk.google.adk.flows.llm_flows.base_llm_flow",
        "google.adk.flows.llm_flows.base_llm_flow",
    ]
    for name in known_loggers:
        lg = logging.getLogger(name)
        lg.addFilter(_expected_error_filter)
        for h in lg.handlers:
            h.addFilter(_expected_error_filter)

    for lg in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(lg, logging.Logger):
            lg.addFilter(_expected_error_filter)
            for h in lg.handlers:
                h.addFilter(_expected_error_filter)

    try:
        import litellm
        from litellm._logging import verbose_logger
        verbose_logger.addFilter(_expected_error_filter)
        for h in verbose_logger.handlers:
            h.addFilter(_expected_error_filter)
    except Exception:
        pass


apply_expected_error_filter()

# Disable ADK node retry on unrecoverable authentication errors or budget pauses
try:
    import google.adk.workflow.utils._retry_utils as _adk_retry_utils
    _orig_adk_should_retry_node = _adk_retry_utils._should_retry_node

    def _non_retryable_should_retry_node(
        exception: BaseException,
        retry_config: Any,
        node_state: Any,
    ) -> bool:
        if is_auth_error(exception) or isinstance(exception, MantisAuthError):
            return False
        exc_cls_name = getattr(getattr(exception, "__class__", None), "__name__", "")
        # A stage continuing in a fresh context (core.graph_loader
        # .NodeContextExhausted): retried for as long as the stage's own
        # continuation budget says so, whatever the node's attempt count --
        # the continuation IS the recovery, not a retry of a failure. The
        # exception carries the verdict so this module need not import the
        # graph loader.
        if exc_cls_name == "NodeContextExhausted":
            return retry_config is not None and bool(getattr(exception, "retryable", False))
        if exc_cls_name in _NON_RETRYABLE_EXC_NAMES:
            return False
        cause = getattr(exception, "__cause__", None) or getattr(exception, "__context__", None)
        if cause is not None and (
            is_auth_error(cause)
            or isinstance(cause, MantisAuthError)
            or getattr(getattr(cause, "__class__", None), "__name__", "") in _NON_RETRYABLE_EXC_NAMES
        ):
            return False
        return _orig_adk_should_retry_node(exception, retry_config, node_state)

    _adk_retry_utils._should_retry_node = _non_retryable_should_retry_node
except Exception:
    pass


class ResilientLiteLLMClient(LiteLLMClient):
    """LiteLLMClient with full jitter exponential backoff (min offset 5s, 1h patience) on 429/quota exhaustion.

    Streaming (stream=True) is supported on both dispatch paths with a precise
    resilience boundary: every failure up to and including the first chunk re-enters
    the same backoff/auth-refresh loop as a non-streaming call, while a failure after
    the first chunk raises MantisStreamInterruptedError instead of retrying --
    sampling is nondeterministic, so a transparent restart would splice two different
    completions together. The consumer owns any continuation policy.
    """

    async def acompletion(
        self,
        model: Any,
        messages: Any,
        tools: Any = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        import litellm

        # Before the retry loop, not inside it: an oversized request is deterministic, so
        # every pass through the loop would upload the same doomed payload again.
        enforce_context_budget(model, messages, tools, context_management=kwargs.get("context_management"))
        if kwargs.get("context_management"):
            _observe("context_edit_requests")

        if stream:
            # Returned, not awaited into a response: callers receive the async
            # iterator directly (response = await client.acompletion(..., stream=True);
            # async for chunk in response). The context budget above still runs
            # eagerly, at call time, not at first iteration.
            return self._astream_with_retry(model, messages, tools, kwargs)

        max_patience = float(os.environ.get("MANTIS_LLM_MAX_PATIENCE_SECONDS", "3600.0"))
        initial_delay = float(os.environ.get("MANTIS_LLM_RETRY_INITIAL_DELAY", "5.0"))
        max_delay = float(os.environ.get("MANTIS_LLM_RETRY_MAX_DELAY", "60.0"))
        min_offset = float(os.environ.get("MANTIS_LLM_MIN_OFFSET", "5.0"))
        backoff_factor = float(os.environ.get("MANTIS_LLM_RETRY_BACKOFF", "2.0"))

        start_time = time.time()
        attempt = 0
        auth_refreshed = False
        while True:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    **kwargs,
                )
                _observe_context_edits(response)
                return response
            except Exception as e:
                if is_auth_error(e):
                    if not auth_refreshed and is_token_refreshable_auth_error(e) and try_refresh_auth():
                        auth_refreshed = True
                        _observe("auth_refreshes")
                        print(f"\n[AUTH REFRESH] Token refreshed for '{model}'; retrying request...", file=sys.stderr, flush=True)
                        continue
                    raise MantisAuthError(
                        format_auth_error_message(e, model=str(model)),
                        original_exception=e,
                    ) from None

                elapsed = time.time() - start_time
                if not is_retryable_llm_error(e) or elapsed >= max_patience:
                    raise

                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )

                attempt += 1
                model_name = str(model)
                _observe("retries")
                if is_rate_limit_error(e):
                    _observe("rate_limit_retries")
                    err_detail = extract_rate_limit_detail(e)
                    prefix = f"[RATE LIMIT] 429 Quota Exceeded on '{model_name}'"
                else:
                    err_detail = f"{type(e).__name__}: {e}"
                    prefix = f"[LLM RETRY] Transient error on '{model_name}'"
                print(
                    f"\n{prefix}. "
                    f"Full jitter backoff: pausing {delay:.1f}s before retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(delay)

    async def _astream_with_retry(
        self,
        model: Any,
        messages: Any,
        tools: Any,
        kwargs: dict,
    ) -> Any:
        """Streams a completion with retries confined to the pre-first-chunk window.

        Everything up to and including the first chunk re-enters the full jitter
        backoff / auth-refresh loop: no output has been delivered yet, so re-opening
        the stream is indistinguishable from a slow first attempt. The patience clock
        is shared across restarts (never reset per attempt). Once a chunk has been
        yielded, a failure raises MantisStreamInterruptedError instead of retrying:
        sampling is nondeterministic, so a transparent restart would splice two
        different completions together behind the consumer's back.
        """
        import litellm

        max_patience = float(os.environ.get("MANTIS_LLM_MAX_PATIENCE_SECONDS", "3600.0"))
        initial_delay = float(os.environ.get("MANTIS_LLM_RETRY_INITIAL_DELAY", "5.0"))
        max_delay = float(os.environ.get("MANTIS_LLM_RETRY_MAX_DELAY", "60.0"))
        min_offset = float(os.environ.get("MANTIS_LLM_MIN_OFFSET", "5.0"))
        backoff_factor = float(os.environ.get("MANTIS_LLM_RETRY_BACKOFF", "2.0"))

        start_time = time.time()
        attempt = 0
        auth_refreshed = False
        while True:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    **kwargs,
                )
                stream_iter = response.__aiter__()
                try:
                    first_chunk = await stream_iter.__anext__()
                except StopAsyncIteration:
                    # An empty stream is a successfully completed (if silent) response.
                    return
            except Exception as e:
                if is_auth_error(e):
                    if not auth_refreshed and is_token_refreshable_auth_error(e) and try_refresh_auth():
                        auth_refreshed = True
                        _observe("auth_refreshes")
                        print(f"\n[AUTH REFRESH] Token refreshed for '{model}'; retrying stream request...", file=sys.stderr, flush=True)
                        continue
                    raise MantisAuthError(
                        format_auth_error_message(e, model=str(model)),
                        original_exception=e,
                    ) from None

                elapsed = time.time() - start_time
                if not is_retryable_llm_error(e) or elapsed >= max_patience:
                    raise

                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )

                attempt += 1
                model_name = str(model)
                _observe("retries")
                if is_rate_limit_error(e):
                    _observe("rate_limit_retries")
                    err_detail = extract_rate_limit_detail(e)
                    prefix = f"[RATE LIMIT] 429 Quota Exceeded on '{model_name}'"
                else:
                    err_detail = f"{type(e).__name__}: {e}"
                    prefix = f"[LLM RETRY] Transient error on '{model_name}'"
                print(
                    f"\n{prefix}. "
                    f"Full jitter backoff: pausing {delay:.1f}s before stream retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(delay)
                continue

            break

        yield first_chunk
        chunks_yielded = 1
        while True:
            try:
                chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                return
            except Exception as e:
                raise MantisStreamInterruptedError(
                    f"Stream from '{model}' failed after {chunks_yielded} chunk(s) were already delivered "
                    f"({type(e).__name__}: {e}). A transparent retry would re-sample the model and splice "
                    "two different completions together; the consumer owns the continuation policy.",
                    chunks_yielded=chunks_yielded,
                    original_exception=e,
                ) from e
            yield chunk
            chunks_yielded += 1

    def completion(
        self,
        model: Any,
        messages: Any,
        tools: Any = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        import litellm

        # Both dispatch paths are guarded: a control that only covers the async path is a
        # control that a single synchronous caller silently disables.
        enforce_context_budget(model, messages, tools, context_management=kwargs.get("context_management"))

        if stream:
            # Same boundary as the async path, for the same reason the context
            # budget guards both dispatch paths.
            return self._stream_with_retry(model, messages, tools, kwargs)

        max_patience = float(os.environ.get("MANTIS_LLM_MAX_PATIENCE_SECONDS", "3600.0"))
        initial_delay = float(os.environ.get("MANTIS_LLM_RETRY_INITIAL_DELAY", "5.0"))
        max_delay = float(os.environ.get("MANTIS_LLM_RETRY_MAX_DELAY", "60.0"))
        min_offset = float(os.environ.get("MANTIS_LLM_MIN_OFFSET", "5.0"))
        backoff_factor = float(os.environ.get("MANTIS_LLM_RETRY_BACKOFF", "2.0"))

        start_time = time.time()
        attempt = 0
        auth_refreshed = False
        while True:
            try:
                return litellm.completion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=stream,
                    **kwargs,
                )
            except Exception as e:
                if is_auth_error(e):
                    if not auth_refreshed and is_token_refreshable_auth_error(e) and try_refresh_auth():
                        auth_refreshed = True
                        print(f"\n[AUTH REFRESH] Token refreshed for '{model}'; retrying request...", file=sys.stderr, flush=True)
                        continue
                    raise MantisAuthError(
                        format_auth_error_message(e, model=str(model)),
                        original_exception=e,
                    ) from None

                elapsed = time.time() - start_time
                if not is_retryable_llm_error(e) or elapsed >= max_patience:
                    raise

                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )

                attempt += 1
                model_name = str(model)
                _observe("retries")
                if is_rate_limit_error(e):
                    _observe("rate_limit_retries")
                    err_detail = extract_rate_limit_detail(e)
                    prefix = f"[RATE LIMIT] 429 Quota Exceeded on '{model_name}'"
                else:
                    err_detail = f"{type(e).__name__}: {e}"
                    prefix = f"[LLM RETRY] Transient error on '{model_name}'"
                print(
                    f"\n{prefix}. "
                    f"Full jitter backoff: pausing {delay:.1f}s before retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)

    def _stream_with_retry(
        self,
        model: Any,
        messages: Any,
        tools: Any,
        kwargs: dict,
    ) -> Any:
        """Sync mirror of _astream_with_retry; see that method for the boundary rationale."""
        import litellm

        max_patience = float(os.environ.get("MANTIS_LLM_MAX_PATIENCE_SECONDS", "3600.0"))
        initial_delay = float(os.environ.get("MANTIS_LLM_RETRY_INITIAL_DELAY", "5.0"))
        max_delay = float(os.environ.get("MANTIS_LLM_RETRY_MAX_DELAY", "60.0"))
        min_offset = float(os.environ.get("MANTIS_LLM_MIN_OFFSET", "5.0"))
        backoff_factor = float(os.environ.get("MANTIS_LLM_RETRY_BACKOFF", "2.0"))

        start_time = time.time()
        attempt = 0
        auth_refreshed = False
        while True:
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    **kwargs,
                )
                stream_iter = iter(response)
                try:
                    first_chunk = next(stream_iter)
                except StopIteration:
                    return
            except Exception as e:
                if is_auth_error(e):
                    if not auth_refreshed and is_token_refreshable_auth_error(e) and try_refresh_auth():
                        auth_refreshed = True
                        _observe("auth_refreshes")
                        print(f"\n[AUTH REFRESH] Token refreshed for '{model}'; retrying stream request...", file=sys.stderr, flush=True)
                        continue
                    raise MantisAuthError(
                        format_auth_error_message(e, model=str(model)),
                        original_exception=e,
                    ) from None

                elapsed = time.time() - start_time
                if not is_retryable_llm_error(e) or elapsed >= max_patience:
                    raise

                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )

                attempt += 1
                model_name = str(model)
                _observe("retries")
                if is_rate_limit_error(e):
                    _observe("rate_limit_retries")
                    err_detail = extract_rate_limit_detail(e)
                    prefix = f"[RATE LIMIT] 429 Quota Exceeded on '{model_name}'"
                else:
                    err_detail = f"{type(e).__name__}: {e}"
                    prefix = f"[LLM RETRY] Transient error on '{model_name}'"
                print(
                    f"\n{prefix}. "
                    f"Full jitter backoff: pausing {delay:.1f}s before stream retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                continue

            break

        yield first_chunk
        chunks_yielded = 1
        while True:
            try:
                chunk = next(stream_iter)
            except StopIteration:
                return
            except Exception as e:
                raise MantisStreamInterruptedError(
                    f"Stream from '{model}' failed after {chunks_yielded} chunk(s) were already delivered "
                    f"({type(e).__name__}: {e}). A transparent retry would re-sample the model and splice "
                    "two different completions together; the consumer owns the continuation policy.",
                    chunks_yielded=chunks_yielded,
                    original_exception=e,
                ) from e
            yield chunk
            chunks_yielded += 1


#: Findings listed in a stage's snapshot before the list is cut.
FINDINGS_SNAPSHOT_MAX_LINES = 150
FINDINGS_SNAPSHOT_HEADER = "[STATE STORE: RECORDED FINDINGS (DO NOT RE-REPORT)]"


class ResilientLiteLlm(LiteLlm):
    """LiteLlm wrapper with resilient full jitter retry (min 5s offset, 1h patience) on quota and rate limits."""

    llm_client: LiteLLMClient = Field(default_factory=ResilientLiteLLMClient, exclude=True)

    def __init__(self, model: str, **kwargs: Any) -> None:
        if LiteLlm is not object:
            super().__init__(model=model, **kwargs)
        else:
            self.model = model
        if getattr(self, "llm_client", None) is None:
            self.llm_client = ResilientLiteLLMClient()
        if "gemini" in str(model).lower():
            try:
                import litellm
                if getattr(litellm, "vertex_ai_safety_settings", None) is None:
                    litellm.vertex_ai_safety_settings = [
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    ]
            except Exception:
                pass

    @property
    def capabilities(self) -> Any:
        """Explicitly declare output_schema_and_tools=False.

        When models have both tools and an output schema (e.g. reproducer, reviewer, critic),
        this instructs ADK's _OutputSchemaRequestProcessor to inject SetModelResponseTool.
        The agent can then conclude and submit its final structured verdict cleanly via a tool
        call (set_model_response) instead of attempting to push JSON via probe shell commands
        or becoming trapped in thinking-token loops.
        """
        if LlmCapabilities is not None:
            return LlmCapabilities(output_schema_and_tools=False)
        return getattr(super(), "capabilities", None)

    @staticmethod
    def _inject_stage_turn_reminder(llm_request: Any) -> None:
        """Injects a termination reminder or reporting directive on every turn."""
        schema_cls = getattr(getattr(llm_request, "config", None), "response_schema", None)
        has_set_model_response = False
        tools_dict = getattr(llm_request, "tools_dict", None)
        if schema_cls is None:
            if isinstance(tools_dict, dict) and "set_model_response" in tools_dict:
                has_set_model_response = True
                schema_cls = getattr(tools_dict["set_model_response"], "output_schema", None)
        else:
            if isinstance(tools_dict, dict) and "set_model_response" in tools_dict:
                has_set_model_response = True

        has_report_findings = isinstance(tools_dict, dict) and "report_findings" in tools_dict

        if schema_cls is None and not has_report_findings:
            return

        reminder_entries: list[tuple[str, str]] = []

        if schema_cls is not None:
            schema_name = getattr(schema_cls, "__name__", str(schema_cls))
            if schema_name == "ReproVerdict":
                hint = '{"route": "success" | "failed_repro", "reason": "<explanation>"}'
            elif schema_name in ("ReviewVerdict", "ExtendedReviewVerdict"):
                # Per-finding verdicts (P6): the hint shows the optional
                # finding_verdicts list so the reviewer judges each finding id
                # individually instead of pushing every finding on the target
                # through one campaign-level route.
                hint = (
                    '{"route": "confirmed" | "false_positive", "reason": "<explanation>", '
                    '"finding_verdicts": [{"finding_id": <id from get_findings>, '
                    '"route": "confirmed" | "false_positive", "reason": "<one sentence>"}, ...]} '
                    '-- include one finding_verdicts entry PER finding id when multiple findings exist; '
                    'omit or leave the list empty only when there is a single finding or none'
                )
            elif schema_name == "CriticVerdict":
                hint = '{"route": "viable" | "non_viable", "reason": "<explanation>"}'
            else:
                hint = '{"route": "...", "reason": "..."}'

            reminder_header = f"[STAGE TURN REMINDER - {schema_name}]"
            if has_set_model_response:
                reminder = (
                    f"\n\n{reminder_header}: "
                    f"If your analysis or updates for this stage are complete, you MUST STOP calling other tools now. "
                    f"Do NOT invoke run_sandbox, write_file, or dummy shell commands to signal completion or test output. "
                    f"To conclude this stage and provide your result, submit your final verdict using the 'set_model_response' tool "
                    f"(e.g. set_model_response(route=..., reason=...)) or emit it directly as raw JSON response text conforming to {schema_name}: "
                    f"{hint}"
                )
            else:
                reminder = (
                    f"\n\n{reminder_header}: "
                    f"If your analysis or updates for this stage are complete, you MUST STOP calling tools now. "
                    f"Do NOT invoke any tools or shell commands to signal completion or test output. "
                    f"To conclude this stage and provide your result, emit your final verdict directly as raw JSON response text conforming to {schema_name}: "
                    f"{hint}"
                )
            reminder_entries.append((reminder_header, reminder))

        if has_report_findings:
            reporting_header = "[STAGE DIRECTIVE - RESEARCHER]"
            reporting_directive = (
                f"\n\n{reporting_header}: "
                f"When reporting security findings via 'report_findings', call 'report_findings' directly without "
                f"emitting intermediate conversational text, analysis essays, or markdown preambles prior to the tool call. "
                f"This ensures your entire output token budget is preserved for the full structured findings payload."
            )
            reminder_entries.append((reporting_header, reporting_directive))

        # Into the instruction, never into the messages. Both reminders are
        # the same text on every call of a stage, so they belong to the
        # stable head of the prompt. Appending them to the LAST message
        # instead -- as this once did -- changed that message between one
        # request and the next (the session keeps the original), which broke
        # the provider's prompt cache at the first tool result of every call
        # (measured: 18% of input served from cache over a run; the system
        # prompt was the only prefix that still matched) and, for models that
        # bind reasoning to its prefix, rewrites the prefix under every later
        # thinking block.
        for header, text in reminder_entries:
            ResilientLiteLlm._append_instruction_once(llm_request, header, text)

    @staticmethod
    def _append_instruction_once(llm_request: Any, header: str, text: str) -> None:
        """Appends `text` to the request's system instruction unless `header`
        is already in it. The instruction is rebuilt per request from the
        agent's, so this runs on every call and lands the same bytes each
        time -- the property the prompt cache and prefix binding need."""
        config = getattr(llm_request, "config", None)
        current = str(getattr(config, "system_instruction", "") or "") if config is not None else ""
        if header in current:
            return
        append = getattr(llm_request, "append_instructions", None)
        if callable(append):
            append([text])
        elif config is not None:
            config.system_instruction = current + text

    @staticmethod
    def _inject_active_findings_state(llm_request: Any) -> None:
        """Tells a stage, once, which findings its campaign has already recorded.

        A snapshot taken at the stage's first call and held for the stage's
        lifetime, appended to the system instruction: the same bytes on every
        call, so the prompt cache holds and the prefix under retained
        reasoning never moves. Findings recorded during the stage reach the
        model through the tool results that recorded them and get_findings.

        The campaign's own findings only (the subtree of its target), fenced
        and bounded: titles are model-written text, and under --parallel the
        run's list is every sibling campaign's work. Keyed by the instruction
        it is appended to, so each stage of a campaign takes its own snapshot
        and a continued attempt takes a fresh one.
        """
        from core.context import current_run_context
        from core.database import read_findings

        ctx = current_run_context.get()
        if ctx is None or not ctx.db_path or not os.path.exists(ctx.db_path):
            return
        header = FINDINGS_SNAPSHOT_HEADER
        config = getattr(llm_request, "config", None)
        instruction = str(getattr(config, "system_instruction", "") or "") if config is not None else ""
        if header in instruction:
            return
        store = getattr(ctx, "node_context", None)
        snapshots = store.setdefault("_findings_snapshot", {}) if isinstance(store, dict) else {}
        key = hashlib.sha256(instruction.encode("utf-8", "replace")).hexdigest()
        if key not in snapshots:
            try:
                findings = read_findings(
                    ctx.db_path, run_id=ctx.run_id,
                    scope_path=(getattr(ctx, "target_file", "") or None),
                )
            except Exception:
                return
            snapshots[key] = ResilientLiteLlm._render_findings_snapshot(findings)
        block = snapshots[key]
        if block:
            ResilientLiteLlm._append_instruction_once(llm_request, header, block)

    @staticmethod
    def _render_findings_snapshot(findings: list) -> str:
        if not findings:
            return ""
        from core.llm_gateway import SecretScrubber, wrap_untrusted_content

        lines = []
        for f in findings[: FINDINGS_SNAPSHOT_MAX_LINES]:
            f_id = f.get("id")
            f_sev = f.get("severity", "UNKNOWN")
            f_fp = f.get("filepath", "")
            f_lines = f.get("line_numbers") or []
            raw_title = str(f.get("title", "")).replace("\n", " ").strip()
            clean_title = SecretScrubber.scrub(raw_title)[:120]
            f_st = f.get("status", "reported")
            lines.append(f"  • Finding #{f_id} [{f_sev}] {f_fp}:{f_lines} - {clean_title} (status: {f_st})")
        omitted = len(findings) - len(lines)
        if omitted > 0:
            lines.append(f"  … and {omitted} more; get_findings lists them all.")
        return (
            f"\n\n{FINDINGS_SNAPSHOT_HEADER}\n"
            f"The following {len(findings)} vulnerability finding(s) were already recorded for this "
            "campaign when this stage began. Do NOT call report_findings for them; report only NEW, "
            "distinct findings. Findings recorded since appear in your tool results and in get_findings.\n"
            + wrap_untrusted_content("\n".join(lines), filename="recorded_findings_snapshot")
        )

    @staticmethod
    def _sanitize_structured_response(response: Any, schema_cls: Any) -> Any:
        """Sanitizes model responses when a structured schema is required to prevent Pydantic ValidationError on refusals."""
        if schema_cls is None or getattr(response, "partial", False):
            return response

        # If the response contains function calls (e.g. set_model_response or tool calls), let ADK handle it
        parts = getattr(getattr(response, "content", None), "parts", None)
        if parts:
            if any(getattr(p, "function_call", None) is not None for p in parts):
                return response

        # Extract text content from parts
        text = ""
        if parts:
            text = "".join(
                getattr(p, "text", "") or ""
                for p in parts
                if not getattr(p, "thought", False)
            )

        # Check if the text is valid JSON according to schema_cls
        is_valid = False
        clean_text = text.strip()
        if clean_text:
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            try:
                if hasattr(schema_cls, "model_validate_json"):
                    schema_cls.model_validate_json(clean_text)
                    is_valid = True
                elif hasattr(schema_cls, "validate_json"):
                    schema_cls.validate_json(clean_text)
                    is_valid = True
            except Exception:
                is_valid = False

        from google.genai import types
        is_safety_block = getattr(response, "finish_reason", None) in (types.FinishReason.SAFETY, "SAFETY") or getattr(response, "error_code", None) == "SAFETY"
        if is_safety_block:
            is_valid = False

        if not is_valid:
            schema_name = getattr(schema_cls, "__name__", str(schema_cls))
            model_fields = getattr(schema_cls, "model_fields", None) or {}
            if schema_name not in ("ReproVerdict", "ReviewVerdict", "ExtendedReviewVerdict", "CriticVerdict") and "route" not in model_fields:
                # Non-verdict schemas (FindingCalibration, AuditPlan) have dedicated
                # prose/JSON fallback parsers in their callers; replacing their text
                # with a ReproVerdict payload would clobber those fallbacks.
                if is_safety_block:
                    if getattr(response, "content", None) is None:
                        response.content = types.Content(role="model", parts=[types.Part.from_text(text="{}")])
                    else:
                        response.content.parts = [types.Part.from_text(text="{}")]
                    if getattr(response, "finish_reason", None) in (types.FinishReason.SAFETY, "SAFETY"):
                        response.finish_reason = types.FinishReason.STOP
                    if getattr(response, "error_code", None):
                        response.error_code = None
                        response.error_message = None
                return response
            clean_snippet = text.strip().replace("\n", " ")[:200] or "Model emitted non-JSON or refusal output"
            if schema_name == "ReproVerdict":
                fallback_payload = {
                    "route": "failed_repro",
                    "reason": f"Fallback: {clean_snippet}",
                }
            elif schema_name in ("ReviewVerdict", "ExtendedReviewVerdict"):
                # Fail CLOSED like the other two verdicts. The old value
                # ("confirmed") promoted a safety-blocked or garbage response
                # into a confirmed vulnerability and sent it down the
                # critic/repro chain. The "Fallback:" reason prefix is a
                # contract with the classifier: synthesized dismissals are
                # routed but never persisted as finding status. The explicit
                # empty finding_verdicts list is the same contract at the
                # per-finding level (INV-6): a synthesized verdict reviewed
                # nothing, so it must route campaign-level and can never
                # stamp any individual finding.
                fallback_payload = {
                    "route": "false_positive",
                    "reason": f"Fallback: {clean_snippet}",
                }
                if schema_name == "ExtendedReviewVerdict":
                    # Only the extended schema tolerates this key: the
                    # canonical ReviewVerdict is extra="forbid" and ADK
                    # re-validates this synthesized text against the bound
                    # schema, so adding the key there would turn the
                    # fail-safe itself into a validation error.
                    fallback_payload["finding_verdicts"] = []
            elif schema_name == "CriticVerdict":
                fallback_payload = {
                    "route": "non_viable",
                    "reason": f"Fallback: {clean_snippet}",
                }
            else:
                fallback_payload = {
                    "route": "failed_repro",
                    "reason": f"Fallback: {clean_snippet}",
                }

            fallback_json = json.dumps(fallback_payload)
            if getattr(response, "content", None) is None:
                response.content = types.Content(role="model", parts=[types.Part.from_text(text=fallback_json)])
            else:
                response.content.parts = [types.Part.from_text(text=fallback_json)]

            if getattr(response, "finish_reason", None) in (types.FinishReason.SAFETY, "SAFETY"):
                response.finish_reason = types.FinishReason.STOP
            if getattr(response, "error_code", None):
                response.error_code = None
                response.error_message = None

            print(
                f"\n[SCHEMA RESILIENCE] Intercepted non-schema/refusal output for {schema_name}; "
                f"synthesized fallback: {fallback_json}",
                file=sys.stderr,
                flush=True,
            )

        return response

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> Any:
        self._inject_stage_turn_reminder(llm_request)
        self._inject_active_findings_state(llm_request)
        schema_cls = getattr(getattr(llm_request, "config", None), "response_schema", None)
        if schema_cls is None and isinstance(getattr(llm_request, "tools_dict", None), dict):
            if "set_model_response" in llm_request.tools_dict:
                schema_cls = getattr(llm_request.tools_dict["set_model_response"], "output_schema", None)

        max_patience = float(os.environ.get("MANTIS_LLM_MAX_PATIENCE_SECONDS", "3600.0"))
        initial_delay = float(os.environ.get("MANTIS_LLM_RETRY_INITIAL_DELAY", "5.0"))
        max_delay = float(os.environ.get("MANTIS_LLM_RETRY_MAX_DELAY", "60.0"))
        min_offset = float(os.environ.get("MANTIS_LLM_MIN_OFFSET", "5.0"))
        backoff_factor = float(os.environ.get("MANTIS_LLM_RETRY_BACKOFF", "2.0"))
        # An empty STOP turn is retried like a 429, but it is not a 429: a
        # request the model has answered with nothing this many times in a
        # row is being answered, and the answer is nothing. Past the cap the
        # empty turn is passed on (below) instead of burning the rest of the
        # patience window one identical request at a time.
        try:
            max_empty_turns = max(
                1,
                int(os.environ.get("MANTIS_LLM_EMPTY_TURN_MAX_ATTEMPTS", str(DEFAULT_EMPTY_TURN_MAX_ATTEMPTS))),
            )
        except ValueError:
            max_empty_turns = DEFAULT_EMPTY_TURN_MAX_ATTEMPTS

        start_time = time.time()
        attempt = 0
        empty_turns = 0
        auth_refreshed = False
        current_stream = stream
        while True:
            try:
                from google.genai import types as genai_types
                async for response in super().generate_content_async(llm_request, stream=current_stream):
                    if (
                        current_stream
                        and (
                            getattr(response, "error_code", None) in (getattr(genai_types.FinishReason, "MAX_TOKENS", None), "MAX_TOKENS")
                            or getattr(response, "finish_reason", None) in (getattr(genai_types.FinishReason, "MAX_TOKENS", None), "MAX_TOKENS")
                        )
                        and "Tool call arguments were truncated" in str(getattr(response, "error_message", ""))
                    ):
                        raise MantisStreamingTruncationError(str(getattr(response, "error_message", "")))
                    if not current_stream and is_empty_stop_response(response):
                        # Billed, so charged, before the turn is thrown away.
                        # A budget ceiling crossed here raises past the
                        # retry loop below, which does not retry that.
                        meter_response_usage(response)
                        raise MantisEmptyResponseError(
                            "The model returned no content (finish_reason=STOP with empty parts)",
                            response=response,
                        )
                    yield self._sanitize_structured_response(response, schema_cls)
                return
            except (json.decoder.JSONDecodeError, ValueError, MantisStreamingTruncationError) as e:
                if current_stream:
                    model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                    print(
                        f"\n[STREAM RESILIENCE] Streaming tool call parsing failed on '{model_name}' ({type(e).__name__}: {e}). "
                        f"Retrying turn with stream=False...",
                        file=sys.stderr,
                        flush=True,
                    )
                    current_stream = False
                    continue
                if not is_retryable_llm_error(e):
                    raise
                elapsed = time.time() - start_time
                if elapsed >= max_patience:
                    raise
                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )
                attempt += 1
                model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                err_detail = f"{type(e).__name__}: {e}"
                print(
                    f"\n[LLM RETRY] Transient error on '{model_name}'. "
                    f"Full jitter backoff: pausing {delay:.1f}s before retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                if current_stream and any(p in str(e).lower() for p in ("unterminated string", "jsondecodeerror", "truncated while streaming")):
                    model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                    print(
                        f"\n[STREAM RESILIENCE] Streaming tool call parsing failed on '{model_name}' ({type(e).__name__}: {e}). "
                        f"Retrying turn with stream=False...",
                        file=sys.stderr,
                        flush=True,
                    )
                    current_stream = False
                    continue

                if is_auth_error(e):
                    if not auth_refreshed and is_token_refreshable_auth_error(e) and try_refresh_auth():
                        auth_refreshed = True
                        model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                        print(f"\n[AUTH REFRESH] Token refreshed for '{model_name}'; retrying request...", file=sys.stderr, flush=True)
                        continue
                    raise MantisAuthError(
                        format_auth_error_message(
                            e,
                            model=str(getattr(self, "model", getattr(llm_request, "model", "llm"))),
                        ),
                        original_exception=e,
                    ) from None

                elapsed = time.time() - start_time
                if isinstance(e, MantisEmptyResponseError):
                    empty_turns += 1
                if isinstance(e, MantisEmptyResponseError) and (elapsed >= max_patience or empty_turns >= max_empty_turns):
                    # Out of patience, or out of empty turns: hand the empty
                    # turn on as it was, so ADK reports
                    # MODEL_RETURNED_NO_CONTENT and the campaign fails the
                    # way it did before this retry existed. --resume picks
                    # that up; an exception here would turn one known failure
                    # mode into another.
                    model_name = str(getattr(self, "model", getattr(llm_request, "model", "llm")))
                    reason = (f"{empty_turns} empty turn(s) in a row on this request (cap {max_empty_turns})"
                              if empty_turns >= max_empty_turns else f"{elapsed:.0f}s of retries")
                    print(
                        f"\n[LLM RETRY] '{model_name}' still returned no content after "
                        f"{reason}; passing the empty turn on.",
                        file=sys.stderr,
                        flush=True,
                    )
                    yield self._sanitize_structured_response(e.response, schema_cls)
                    return
                if not is_retryable_llm_error(e) or elapsed >= max_patience:
                    raise

                remaining = max_patience - elapsed
                retry_after = extract_retry_after(e)
                delay = compute_full_jitter_delay(
                    attempt=attempt,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    min_offset=min_offset,
                    backoff_factor=backoff_factor,
                    retry_after=retry_after,
                    max_remaining=remaining,
                )

                attempt += 1
                model_name = str(getattr(self, "model", llm_request.model if hasattr(llm_request, "model") else "llm"))
                if is_rate_limit_error(e):
                    err_detail = extract_rate_limit_detail(e)
                    prefix = f"[RATE LIMIT] 429 Quota Exceeded on '{model_name}'"
                else:
                    err_detail = f"{type(e).__name__}: {e}"
                    prefix = f"[LLM RETRY] Transient error on '{model_name}'"
                print(
                    f"\n{prefix}. "
                    f"Full jitter backoff: pausing {delay:.1f}s before retry (attempt {attempt}, elapsed {elapsed:.1f}s / {max_patience:.0f}s patience) [{err_detail}]...",
                    file=sys.stderr,
                    flush=True,
                )
                await asyncio.sleep(delay)


class DeferredEnvLiteLlm(ResilientLiteLlm):
    """ResilientLiteLlm that resolves cloud credentials at CALL time, not build time.

    get_llm_kwargs raises when a vertex_ai/ model has no resolvable project
    (VERTEXAI_PROJECT / GOOGLE_CLOUD_PROJECT / ADC), and graph_loader used to
    call it eagerly for every node while BUILDING the graph. That coupled two
    unrelated capabilities: constructing the workflow graph (pure local data
    flow) and holding cloud credentials (only needed to actually call the
    model). In hermetic environments -- unit tests, CI sandboxes, air-gapped
    graph review -- 9 tests failed at import/build time without ever intending
    to make an LLM call.

    This class carries a zero-argument resolver (a functools.partial closing
    over the exact same arguments graph_loader would have passed eagerly) and
    invokes it on the FIRST generate_content_async call. Resolution is
    fail-closed: if the environment still lacks credentials at call time, the
    resolver re-raises the same clear ValueError ("You must set
    VERTEXAI_PROJECT or GOOGLE_CLOUD_PROJECT...") with build-deferral context
    appended -- the call fails loudly, never silently downgrades to another
    provider. If the environment gained credentials between build and call
    (main.py setup, test fixtures, delayed ADC), resolution simply succeeds.
    """

    _env_resolver: Any = PrivateAttr(default=None)
    _env_resolved: bool = PrivateAttr(default=True)

    def __init__(self, model: str, env_resolver: Any = None, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self._env_resolver = env_resolver
        self._env_resolved = env_resolver is None

    def _resolve_deferred_env(self) -> None:
        if self._env_resolved:
            return
        try:
            resolved_model, llm_kwargs = self._env_resolver()
        except ValueError as e:
            # Same message the eager path produced, so operators and tests
            # that match on it keep working; the suffix explains why it
            # surfaced at call time instead of graph-build time.
            raise ValueError(
                f"{e} (credential resolution was deferred from graph build; "
                f"this LLM call for '{getattr(self, 'model', '?')}' cannot proceed without them)"
            ) from e
        merged = dict(llm_kwargs)
        merged.pop("model", None)
        try:
            # The construction-time model string was a display placeholder
            # (raw node/config id); swap in the fully resolved id now.
            self.model = resolved_model
        except Exception:
            pass
        additional = getattr(self, "_additional_args", None)
        if isinstance(additional, dict):
            additional.update(merged)
        self._env_resolver = None
        self._env_resolved = True

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        self._resolve_deferred_env()
        async for item in super().generate_content_async(llm_request, stream=stream):
            yield item


def resolve_model_id(
    model_id: Optional[str] = None,
    default_model: str = DEFAULT_MODEL,
    global_model_override: Optional[str] = None,
) -> str:
    """The model a node will actually call, as its normalized id.

    Precedence: global_override > MANTIS_MODEL > node > MODEL_ID > default.
    Split out of get_llm_kwargs so a caller that only needs the identity --
    the harness fingerprint stamps it on every coverage row -- gets the same
    answer whether or not the credentials get_llm_kwargs goes on to require
    are present.
    """
    if global_model_override:
        raw_model = global_model_override
    elif os.environ.get("MANTIS_MODEL"):
        raw_model = os.environ.get("MANTIS_MODEL")
    else:
        raw_model = model_id or os.environ.get("MODEL_ID") or default_model
    return normalize_model_id(raw_model)


def get_llm_kwargs(
    model_id: Optional[str] = None,
    default_model: str = DEFAULT_MODEL,
    api_base: Optional[str] = None,
    default_api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    default_timeout: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    default_reasoning_effort: Optional[str] = None,
    global_model_override: Optional[str] = None,
    config: Optional[dict] = None,
) -> Tuple[str, dict]:
    """Resolves the LLM mapping details cleanly with precedence: global_override > MANTIS_MODEL > node > MODEL_ID > default."""
    resolved_model = resolve_model_id(model_id, default_model, global_model_override)
    resolved_api_base = api_base or os.environ.get("LLM_API_BASE") or default_api_base
    raw_timeout = timeout if timeout is not None else (
        os.environ.get("LLM_TIMEOUT")
        or os.environ.get("MANTIS_TIMEOUT")
        or os.environ.get("LLM_REQUEST_TIMEOUT")
        or default_timeout
    )
    effort = reasoning_effort or os.environ.get("REASONING_EFFORT") or default_reasoning_effort
    if raw_timeout is None and (effort in ("high", "medium") or "glm" in resolved_model or "claude" in resolved_model):
        raw_timeout = 300.0

    llm_kwargs = {"model": resolved_model}
    if resolved_api_base:
        llm_kwargs["api_base"] = resolved_api_base
    if effort:
        llm_kwargs["reasoning_effort"] = str(effort).lower().strip()
    if raw_timeout is not None:
        try:
            timeout_val = float(raw_timeout)
            if timeout_val > 0:
                llm_kwargs["timeout"] = timeout_val
        except (ValueError, TypeError):
            pass

    max_tokens_val = os.environ.get("LLM_MAX_TOKENS") or os.environ.get("MANTIS_MAX_TOKENS")
    if max_tokens_val:
        try:
            llm_kwargs["max_tokens"] = int(max_tokens_val)
        except (ValueError, TypeError):
            pass
    elif effort in ("high", "medium") or "claude" in resolved_model:
        # Provide a generous output token budget so thinking tokens do not starve response text
        llm_kwargs.setdefault("max_tokens", 32768)

    if resolved_model.startswith("vertex_ai/"):
        project = os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if is_placeholder(project):
            project = None

        if not project and config and isinstance(config, dict):
            cfg_proj = config.get("project")
            if not cfg_proj or is_placeholder(cfg_proj):
                sb = config.get("sandbox")
                if isinstance(sb, dict):
                    sb_opts = sb.get("options")
                    if isinstance(sb_opts, dict):
                        cfg_proj = sb_opts.get("project")
            if cfg_proj and not is_placeholder(cfg_proj):
                project = str(cfg_proj)

        location = os.environ.get("VERTEXAI_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

        if not project:
            try:
                import google.auth
                _, project = google.auth.default()
                if is_placeholder(project):
                    project = None
            except Exception:
                pass

        if not project and not resolved_api_base:
            raise ValueError("ERROR: You must set VERTEXAI_PROJECT or GOOGLE_CLOUD_PROJECT env variables.")

        if project:
            llm_kwargs["vertex_project"] = project
            llm_kwargs["vertex_location"] = location

        if os.environ.get("VERTEX_FLEX") in ("1", "true", "True") or os.environ.get("VERTEXAI_SERVICE_TIER", "").lower() == "flex":
            headers = llm_kwargs.get("extra_headers") or {}
            headers["X-Vertex-AI-LLM-Request-Type"] = "shared"
            llm_kwargs["extra_headers"] = headers

    # Relax the category filters for Gemini models, which sometimes trigger on
    # defensive security analysis and vulnerability remediation. Applied on
    # every route that reaches a Gemini model natively (vertex_ai/ and
    # gemini/), not only Vertex: the Gemini API route used to run with the
    # provider defaults, so the same prompt could be filtered through an API
    # key and pass through a project. Keyed on the provider prefix, not on
    # the substring "gemini": a Gemini model behind an OpenAI-compatible
    # gateway (openai/gemini-*) takes OpenAI parameters, and litellm rejects
    # this one there. The litellm global stays Vertex-scoped; it only exists
    # for callers that bypass these kwargs.
    if resolved_model.startswith(("gemini/", "vertex_ai/")) and "gemini" in resolved_model:
        safety_settings = [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        ]
        llm_kwargs["safety_settings"] = safety_settings
        if resolved_model.startswith("vertex_ai/"):
            try:
                import litellm
                litellm.vertex_ai_safety_settings = safety_settings
            except Exception:
                pass

    # Anthropic context editing, Claude models only (see
    # anthropic_context_editing): the provider clears older tool results
    # before the model reads a request that has grown past the trigger.
    editing = anthropic_context_editing(resolved_model, config=config)
    if editing:
        llm_kwargs["context_management"] = editing

    return resolved_model, llm_kwargs

