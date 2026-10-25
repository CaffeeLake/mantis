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

"""The output-token ceiling for Claude models: 64k by default, because
thinking at high effort shares it and one turn can pass 32k; and when a
model allows less, the request is retried once at the ceiling the provider
named instead of failing the stage."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.adk.models import LlmRequest, LlmResponse  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.genai import types  # noqa: E402

from core.config import (  # noqa: E402
    CLAUDE_DEFAULT_MAX_OUTPUT_TOKENS, ResilientLiteLlm, get_llm_kwargs, output_cap_from_error,
)

_FAST = {
    "MANTIS_LLM_MAX_PATIENCE_SECONDS": "5",
    "MANTIS_LLM_RETRY_INITIAL_DELAY": "0",
    "MANTIS_LLM_RETRY_MAX_DELAY": "0",
    "MANTIS_LLM_RETRY_MIN_OFFSET": "0",
}
VERTEX_MESSAGE = (
    'litellm.BadRequestError: Vertex_aiException BadRequestError - {"type":"error","error":'
    '{"type":"invalid_request_error","message":"max_tokens: 128000 > 32000, which is the maximum '
    'allowed number of output tokens for this model"}}'
)


class DefaultCeilingTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"VERTEXAI_PROJECT": "p", "VERTEXAI_LOCATION": "global"}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for var in ("LLM_MAX_TOKENS", "MANTIS_MAX_TOKENS"):
            os.environ.pop(var, None)

    def test_claude_gets_the_full_ceiling_at_every_effort(self):
        for effort in ("low", "medium", "high", "max"):
            _, kw = get_llm_kwargs(model_id="vertex_ai/claude-example", reasoning_effort=effort)
            self.assertEqual(kw["max_tokens"], CLAUDE_DEFAULT_MAX_OUTPUT_TOKENS, effort)
        self.assertEqual(CLAUDE_DEFAULT_MAX_OUTPUT_TOKENS, 128000)

    def test_other_models_keep_their_ceilings(self):
        _, kw = get_llm_kwargs(model_id="vertex_ai/gemini-example", reasoning_effort="high")
        self.assertEqual(kw["max_tokens"], 32768)
        _, kw = get_llm_kwargs(model_id="vertex_ai/gemini-example", reasoning_effort="low")
        self.assertNotIn("max_tokens", kw)

    def test_the_operator_s_ceiling_wins(self):
        with mock.patch.dict(os.environ, {"MANTIS_MAX_TOKENS": "4096"}):
            _, kw = get_llm_kwargs(model_id="vertex_ai/claude-example", reasoning_effort="max")
        self.assertEqual(kw["max_tokens"], 4096)


class OutputCapFromErrorTest(unittest.TestCase):
    def test_reads_the_ceiling_the_provider_names(self):
        self.assertEqual(output_cap_from_error(Exception(VERTEX_MESSAGE)), 32000)

    def test_anything_else_is_not_a_ceiling(self):
        self.assertIsNone(output_cap_from_error(Exception("429 Too Many Requests")))
        self.assertIsNone(output_cap_from_error(Exception("max_tokens: 1000 > 2000")), "not a rejection for asking more")
        self.assertIsNone(output_cap_from_error(Exception("max_tokens: 5 > 0")))


def _reply(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
                       finish_reason=types.FinishReason.STOP)


class RetryAtTheNamedCeilingTest(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, _FAST)
        env.start()
        self.addCleanup(env.stop)
        self.llm = ResilientLiteLlm(model="openai/fake", max_tokens=128000)
        self.request = LlmRequest(
            model="openai/fake",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text="go")])],
        )

    def _run(self, script):
        seen = []

        async def upstream(_self, llm_request, stream=False):
            seen.append(dict(_self._additional_args))
            step = script.pop(0)
            if isinstance(step, Exception):
                raise step
            yield step

        async def go():
            return [r async for r in self.llm.generate_content_async(self.request, stream=False)]

        err = io.StringIO()
        with mock.patch.object(LiteLlm, "generate_content_async", upstream), contextlib.redirect_stderr(err):
            out = asyncio.run(go())
        return out, seen, err.getvalue()

    def test_a_rejected_ceiling_is_lowered_once_and_remembered(self):
        out, seen, err = self._run([Exception(VERTEX_MESSAGE), _reply("done")])
        self.assertEqual([r.content.parts[0].text for r in out], ["done"])
        self.assertEqual([s["max_tokens"] for s in seen], [128000, 32000])
        self.assertIn("allows at most 32000 output tokens", err)
        self.assertEqual(self.llm._additional_args["max_tokens"], 32000, "every later call uses the named ceiling")

    def test_a_ceiling_no_lower_than_the_request_is_not_retried(self):
        self.llm._additional_args["max_tokens"] = 32000
        with self.assertRaises(Exception) as cm:
            self._run([Exception(VERTEX_MESSAGE)])
        self.assertIn("max_tokens: 128000 > 32000", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
