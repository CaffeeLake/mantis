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

"""Anthropic context editing: the provider clears a stage's older tool
results before the model reads a request that has grown past the trigger,
so a long stage stays inside the window while the request we send stays
append-only. Sent to Claude models only; the client-side estimates (request
guard, node context ceiling) count the request the way the provider reads
it, so neither refuses or ends a stage the provider was about to serve."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core import config as C  # noqa: E402

CLAUDE = "vertex_ai/claude-example-1"
GEMINI = "vertex_ai/gemini-3.8-flash"


class ParameterTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(C._CONTEXT_EDITING_ENV, None)
        os.environ.pop("VERTEXAI_PROJECT", None)
        os.environ.pop("GOOGLE_CLOUD_PROJECT", None)

    def test_claude_models_get_a_clear_tool_uses_edit_sized_to_the_window(self):
        with patch.object(C, "resolve_context_limit", return_value=1_000_000):
            edit = C.anthropic_context_editing(CLAUDE)
        self.assertIsNotNone(edit)
        [e] = edit["edits"]
        self.assertEqual(e["type"], C.CONTEXT_EDITING_EDIT_TYPE)
        self.assertEqual(e["trigger"], {"type": "input_tokens", "value": 500_000})
        self.assertEqual(e["keep"], {"type": "tool_uses", "value": C.CONTEXT_EDITING_KEEP_TOOL_USES})
        self.assertEqual(e["clear_at_least"], {"type": "input_tokens", "value": 200_000})
        self.assertLess(C.CONTEXT_EDITING_TRIGGER_RATIO, 0.8, "the provider acts before the node context ceiling")

    def test_only_claude_models_and_only_when_on(self):
        with patch.object(C, "resolve_context_limit", return_value=1_000_000):
            for other in (GEMINI, "openai/gpt-5.3-codex", "gemini/gemini-3.6-flash", ""):
                self.assertIsNone(C.anthropic_context_editing(other), other)
            for route in ("anthropic/claude-x", "bedrock/anthropic.claude-x", "vertex_ai/claude-x"):
                self.assertIsNotNone(C.anthropic_context_editing(route), route)
            self.assertIsNone(C.anthropic_context_editing(CLAUDE, config={"context_editing": "off"}))
            self.assertIsNotNone(C.anthropic_context_editing(CLAUDE, config={"context_editing": "auto"}))
            os.environ[C._CONTEXT_EDITING_ENV] = "off"
            self.assertIsNone(C.anthropic_context_editing(CLAUDE))
            os.environ.pop(C._CONTEXT_EDITING_ENV)
        with patch.object(C, "resolve_context_limit", return_value=None):
            self.assertIsNone(C.anthropic_context_editing(CLAUDE), "no window to size against: not sent")

    def test_get_llm_kwargs_carries_it_for_claude_and_nothing_else(self):
        os.environ["VERTEXAI_PROJECT"] = "example-project"
        with patch.object(C, "resolve_context_limit", return_value=1_000_000):
            _, claude_kwargs = C.get_llm_kwargs(CLAUDE)
            _, gemini_kwargs = C.get_llm_kwargs(GEMINI)
        self.assertIn("context_management", claude_kwargs)
        self.assertEqual(C.context_editing_keeps(claude_kwargs["context_management"]), C.CONTEXT_EDITING_KEEP_TOOL_USES)
        self.assertNotIn("context_management", gemini_kwargs)

    def test_keeps_reads_the_edit_and_nothing_else(self):
        self.assertIsNone(C.context_editing_keeps(None))
        self.assertIsNone(C.context_editing_keeps({"edits": [{"type": "clear_thinking_20251015"}]}))
        self.assertEqual(C.context_editing_keeps({"edits": [{"type": "clear_tool_uses_20250919"}]}), 0)
        self.assertEqual(C.context_editing_keeps(
            {"edits": [{"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 3}}]}), 3)


class EstimatesFollowTheProviderTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("MANTIS_MAX_INPUT_TOKENS", None)

    def _messages(self, n_tools: int, chars: int):
        msgs = [{"role": "user", "content": "go"}]
        for i in range(n_tools):
            msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function",
                         "function": {"name": "read_file", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "y" * chars})
        return msgs

    def test_the_request_guard_counts_cleared_results_as_cleared(self):
        # Ten 40k-char tool results: ~100k tokens at 4 chars/token, far over a
        # 50k-token window. With the edit keeping the newest 2, the provider
        # reads ~20k tokens -- and so does the guard.
        msgs = self._messages(10, 40_000)
        edit = {"edits": [{"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 2}}]}
        with patch.object(C, "resolve_context_limit", return_value=50_000):
            with self.assertRaises(C.ContextBudgetExceededError):
                C.enforce_context_budget(CLAUDE, msgs)
            C.enforce_context_budget(CLAUDE, msgs, context_management=edit)  # fits as the provider reads it
            # The kept window alone still has to fit.
            with self.assertRaises(C.ContextBudgetExceededError):
                C.enforce_context_budget(CLAUDE, self._messages(3, 100_000), context_management=edit)

    def test_cleared_view_touches_only_older_tool_messages(self):
        msgs = self._messages(3, 10)
        seen = C._messages_as_the_provider_sees_them(msgs, keep=1)
        tools = [m for m in seen if m["role"] == "tool"]
        self.assertEqual([m["content"] for m in tools], [C._CLEARED_TOOL_RESULT, C._CLEARED_TOOL_RESULT, "y" * 10])
        self.assertEqual([m for m in seen if m["role"] != "tool"], [m for m in msgs if m["role"] != "tool"])
        self.assertEqual(msgs[2]["content"], "y" * 10, "the caller's messages are not mutated")
        self.assertIs(C._messages_as_the_provider_sees_them(msgs, keep=5), msgs, "nothing to clear: same list")

    def test_the_gateway_hands_the_request_edit_to_the_guard(self):
        import litellm

        from core.config import ResilientLiteLLMClient

        seen = {}

        def spy(model, messages, tools=None, context_management=None):
            seen["cm"] = context_management

        class _Resp:
            choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]

        edit = {"edits": [{"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 2}}]}
        with patch.object(C, "enforce_context_budget", spy), \
             patch.object(litellm, "completion", lambda **kw: _Resp()):
            ResilientLiteLLMClient().completion(model="m", messages=[{"role": "user", "content": "hi"}],
                                                context_management=edit)
        self.assertEqual(seen.get("cm"), edit)

    def test_the_node_context_ceiling_measures_the_request_as_the_provider_reads_it(self):
        from google.adk.models import LlmRequest
        from google.genai import types

        import core.graph_loader as gl

        req = LlmRequest()
        req.config.system_instruction = "be careful"
        req.contents = [types.Content(role="user", parts=[types.Part.from_text(text="go")])]
        for i in range(6):
            req.contents.append(types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="read_file", args={"n": i}))]))
            req.contents.append(types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(name="read_file", response={"result": "x" * 10_000}))]))
        full = gl._request_chars(req)
        kept2 = gl._request_chars(req, keep_tool_results=2)
        self.assertGreater(full, 60_000)
        self.assertLess(kept2, 21_000 + 1_000, "four of six results count as placeholders")
        self.assertGreater(kept2, 20_000)

        # The stage is told once, in the same words on every call.
        from core.context import RunContext, current_run_context
        token = current_run_context.set(RunContext(jail_dir="/tmp", db_path="", target_file="a.py", run_id="r"))
        self.addCleanup(current_run_context.reset, token)
        cb = gl._make_node_context_ceiling_before_model_callback("researcher", model="m", keep_tool_results=5)
        with patch("core.config.resolve_context_limit", return_value=1_000_000), \
             patch("core.config.resolve_input_token_cap", return_value=None):
            self.assertIsNone(cb(callback_context=None, llm_request=req))
            first = req.config.system_instruction
            self.assertIn("[CONTEXT EDITING]", first)
            self.assertIsNone(cb(callback_context=None, llm_request=req))
            self.assertEqual(req.config.system_instruction, first)
        plain = gl._make_node_context_ceiling_before_model_callback("researcher", model="m")
        req2 = LlmRequest()
        req2.config.system_instruction = "be careful"
        with patch("core.config.resolve_context_limit", return_value=1_000_000), \
             patch("core.config.resolve_input_token_cap", return_value=None):
            plain(callback_context=None, llm_request=req2)
        self.assertNotIn("[CONTEXT EDITING]", req2.config.system_instruction, "no editing, no notice")


if __name__ == "__main__":
    unittest.main()
