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

"""A node prompt that quotes code must reach the model as written.

Live: an evolved threat-modeler prompt contained ``f"... WHERE name='{name}'"``;
ADK rendered the instruction string as a template, looked up session state
``name``, and the stage died with ``KeyError: Context variable not found``
before its first model call. Node prompts are literal text; they are handed
to ADK as a provider, which ADK uses verbatim.
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parents[1]
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.graph_loader import LiteralInstruction, load_workflow_from_json  # noqa: E402

BRACEY = (
    "You are the threat modeler. Known sinks in this codebase:\n"
    "  database.py: conn.execute(f\"SELECT * FROM users WHERE name='{name}'\")\n"
    "  dal/repo.py: ORDER BY {column} {direction}; also {upload.filename} and {artifact.report}\n"
    "Report via record_threat_model."
)


def _agents(workflow):
    seen = {}
    for edge in workflow.edges:
        for node in (getattr(edge, "from_node", None), getattr(edge, "to_node", None)):
            agent = getattr(node, "agent", node)
            if agent is not None and hasattr(agent, "canonical_instruction"):
                seen[agent.name] = agent
    return seen


class LiteralInstructionTest(unittest.TestCase):
    def test_reads_like_the_string_it_carries(self):
        li = LiteralInstruction(BRACEY)
        self.assertIn("WHERE name='{name}'", li)
        self.assertTrue(li.startswith("You are the threat modeler"))
        self.assertEqual(str(li), BRACEY)
        self.assertEqual(li, BRACEY)
        self.assertEqual(len(li), len(BRACEY))
        self.assertEqual(li(None), BRACEY)

    def test_adk_uses_it_verbatim_where_a_string_would_be_templated(self):
        """The exact failure: ADK's canonical_instruction on a string vs on the provider."""
        from google.adk.agents import LlmAgent
        from google.adk.flows.llm_flows.prompt import _instructions_utils as iu
        from unittest import mock

        class _Ctx:  # what inject_session_state reads: an invocation with empty state
            class _IC:
                session = type("S", (), {"state": {}})()
                artifact_service = None
                app_name = "t"; user_id = "u"; session_id = "s"
            _invocation_context = _IC()
            state = {}
            agent_name = "threat_modeler"

        as_string = LlmAgent(name="t_str", model="gemini/x", instruction=BRACEY)
        as_literal = LlmAgent(name="t_lit", model="gemini/x", instruction=LiteralInstruction(BRACEY))

        text_s, bypass_s = asyncio.run(as_string.canonical_instruction(_Ctx()))
        text_l, bypass_l = asyncio.run(as_literal.canonical_instruction(_Ctx()))
        self.assertFalse(bypass_s, "a string instruction goes through state injection")
        self.assertTrue(bypass_l, "the provider bypasses it")
        self.assertEqual(text_l, BRACEY)
        # And state injection of the string form is what raised live.
        with self.assertRaises(KeyError):
            asyncio.run(iu.inject_session_state(text_s, _Ctx()))

    def test_every_agent_in_the_shipped_workflow_carries_a_literal_instruction(self):
        workflow, _ = load_workflow_from_json(str(_REF_ROOT / "workflow.json"), load_local=False)
        agents = _agents(workflow)
        self.assertGreater(len(agents), 5)
        for name, agent in agents.items():
            self.assertIsInstance(agent.instruction, LiteralInstruction, name)

    def test_an_evolved_prompt_with_braces_loads_and_renders_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            wf = json.loads((_REF_ROOT / "workflow.json").read_text())
            tm = next(n for n in wf["nodes"] if n["id"] == "threat_modeler")
            tm["system_prompt"] = BRACEY
            path = Path(tmp) / "workflow.json"
            path.write_text(json.dumps(wf))
            workflow, _ = load_workflow_from_json(str(path), load_local=False)
            agent = _agents(workflow)["threat_modeler"]
            text, bypass = asyncio.run(agent.canonical_instruction(None))
            self.assertTrue(bypass)
            self.assertIn("WHERE name='{name}'", text)
            self.assertIn("{artifact.report}", text)


if __name__ == "__main__":
    unittest.main()
