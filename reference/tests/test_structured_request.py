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

"""A hand-built structured request must actually carry its tool to the provider.

Found live: the calibrator asked the model to call ``set_model_response`` on
a request whose tool had been placed in ``GenerateContentConfig(tools=[...])``
as a bare ``SetModelResponseTool``. ADK's LiteLlm adapter forwards only
``types.Tool`` instances from there and drops the rest silently, so the
provider saw ``tools=None``; one model answered with an empty turn, every
time, and the empty-turn retry spent an hour per finding on it.
"""

import asyncio
import sys
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parents[1]
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from google.adk.models.lite_llm import _get_completion_inputs  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.tools.set_model_response_tool import SetModelResponseTool  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core.config import structured_llm_request  # noqa: E402
from core.schemas import FindingCalibration  # noqa: E402


class Verdict(BaseModel):
    score: float
    why: str


def _litellm_tools(req: LlmRequest):
    _messages, tools, _fmt, _params, _choice = asyncio.run(_get_completion_inputs(req, "vertex_ai/some-gemini"))
    return tools


class StructuredRequestTest(unittest.TestCase):
    def test_the_old_shape_is_the_bug_the_adapter_drops_a_bare_tool(self):
        bare = LlmRequest(
            contents=[types.Content(parts=[types.Part.from_text(text="calibrate")], role="user")],
            config=types.GenerateContentConfig(system_instruction="s", tools=[SetModelResponseTool(Verdict)]),
        )
        self.assertIsNone(_litellm_tools(bare), "if this starts passing, ADK changed and the helper can go")

    def test_structured_llm_request_reaches_the_provider_with_its_declaration(self):
        req = structured_llm_request("calibrate", "s", Verdict)
        tools = _litellm_tools(req)
        self.assertIsNotNone(tools)
        self.assertEqual(len(tools), 1)
        fn = tools[0]["function"]
        self.assertEqual(fn["name"], "set_model_response")
        self.assertEqual(set(fn["parameters"]["properties"]), {"score", "why"})
        # And the response sanitiser can find the schema the way it looks for it.
        self.assertIn("set_model_response", req.tools_dict)
        self.assertIs(req.tools_dict["set_model_response"].output_schema, Verdict)
        self.assertEqual(req.config.system_instruction, "s")
        self.assertEqual(req.contents[0].parts[0].text, "calibrate")

    def test_the_calibrator_schema_declares_every_field(self):
        tools = _litellm_tools(structured_llm_request("x", "s", FindingCalibration))
        props = set(tools[0]["function"]["parameters"]["properties"])
        for field in ("finding_id", "mantis_risk_score", "priority", "impact_score", "likelihood_score",
                      "inferred_exposure", "reasoning", "executive_summary"):
            self.assertIn(field, props)


if __name__ == "__main__":
    unittest.main()
