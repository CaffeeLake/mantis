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

"""A stage may fork its predecessor's conversation instead of starting
fresh (`fork_from`). What is inherited, what is not, where the callback
sits, and what the loader refuses."""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from google.adk.events import Event  # noqa: E402
from google.adk.models import LlmRequest  # noqa: E402
from google.genai import types  # noqa: E402

import core.graph_loader as gl  # noqa: E402
from core.budget import BudgetConfig, BudgetController  # noqa: E402
from core.context import RunContext, current_run_context  # noqa: E402


def _ev(author, role, parts, **kw):
    return Event(author=author, content=types.Content(role=role, parts=parts), **kw)


def _planner_events():
    return [
        _ev("user", "user", [types.Part.from_text(text="Initial Task Input: Evaluate app")]),
        _ev("planner", "model", [
            types.Part(text="thinking hard", thought=True),
            types.Part(function_call=types.FunctionCall(name="read_file", id="c1", args={"filepath": "app.py"}),
                       thought_signature=b"sig"),
        ]),
        _ev("planner", "user", [types.Part(function_response=types.FunctionResponse(
            name="read_file", id="c1", response={"result": "def route(): ..."}))]),
        _ev("planner", "model", [types.Part.from_text(text="Plan: look at route() for injection.")]),
        _ev("other_stage", "model", [types.Part.from_text(text="not the planner's")]),
        _ev("planner", "model", [types.Part.from_text(text="partial")], partial=True),
        _ev("planner", "model", [types.Part.from_text(text="failed")], error_code="SOMETHING"),
    ]


class ForkedContentsTest(unittest.TestCase):
    def test_the_parents_transcript_minus_thinking_and_others(self):
        contents = gl.forked_contents(_planner_events(), "planner")
        self.assertEqual([c.role for c in contents], ["model", "user", "model"])
        first = contents[0].parts
        self.assertEqual(len(first), 1, "the thinking part is dropped")
        self.assertIsNotNone(first[0].function_call)
        self.assertEqual(first[0].function_call.id, "c1", "the call keeps its id to pair with its result")
        self.assertIsNone(first[0].thought_signature, "no signature travels into a different prefix")
        self.assertEqual(contents[1].parts[0].function_response.id, "c1")
        self.assertEqual(contents[2].parts[0].text, "Plan: look at route() for injection.")
        texts = json.dumps([p.text for c in contents for p in c.parts if p.text])
        self.assertNotIn("not the planner's", texts)
        self.assertNotIn("Initial Task Input", texts, "the seed is the user's, not the parent's")
        self.assertNotIn("partial", texts)
        self.assertNotIn("failed", texts)

    def test_nothing_to_inherit(self):
        self.assertEqual(gl.forked_contents([], "planner"), [])
        self.assertEqual(gl.forked_contents(None, "planner"), [])
        self.assertEqual(gl.forked_contents(_planner_events(), "nobody"), [])


class ForkCallbackTest(unittest.TestCase):
    def setUp(self):
        self.ctrl = BudgetController(config=BudgetConfig(max_tokens=0, max_wall_clock_seconds=0), run_id="r")
        self.token = current_run_context.set(RunContext(jail_dir="/tmp", db_path="", run_id="r",
                                                        budget_controller=self.ctrl, node_context={}))

    def tearDown(self):
        current_run_context.reset(self.token)

    def _request(self):
        req = LlmRequest()
        req.config.system_instruction = "You are the researcher."
        req.contents = [types.Content(role="user", parts=[types.Part.from_text(text="go")])]
        return req

    def test_the_transcript_is_prepended_on_every_call_and_the_note_once(self):
        cb = gl._make_fork_before_model_callback("researcher", "planner")
        ctx = SimpleNamespace(session=SimpleNamespace(events=_planner_events()))
        req = self._request()
        self.assertIsNone(asyncio.run(cb(callback_context=ctx, llm_request=req)))
        self.assertEqual(len(req.contents), 4, "three inherited, then the node's own")
        self.assertEqual(req.contents[-1].parts[0].text, "go")
        self.assertEqual(req.config.system_instruction.count(gl.FORK_NOTICE_MARKER), 1)
        self.assertIn("'planner' stage's work", req.config.system_instruction)
        # ADK rebuilds the request each call; the same transcript goes in front again.
        again = self._request()
        again.config.system_instruction = req.config.system_instruction
        asyncio.run(cb(callback_context=ctx, llm_request=again))
        self.assertEqual(len(again.contents), 4)
        self.assertEqual(again.config.system_instruction.count(gl.FORK_NOTICE_MARKER), 1)
        self.assertEqual(self.ctrl.spend_snapshot()["context"]["forked_requests"], 2)

    def test_no_parent_events_means_no_change(self):
        cb = gl._make_fork_before_model_callback("researcher", "planner")
        req = self._request()
        asyncio.run(cb(callback_context=SimpleNamespace(session=SimpleNamespace(events=[])), llm_request=req))
        self.assertEqual(len(req.contents), 1)
        self.assertNotIn(gl.FORK_NOTICE_MARKER, req.config.system_instruction)
        asyncio.run(cb(callback_context=None, llm_request=req))
        self.assertEqual(len(req.contents), 1)

    def test_the_parents_final_reply_is_not_doubled(self):
        """ADK opens the node's turn with its predecessor's final reply as a
        user message; the inherited transcript ends with the same text."""
        cb = gl._make_fork_before_model_callback("researcher", "planner")
        ctx = SimpleNamespace(session=SimpleNamespace(events=_planner_events()))
        req = LlmRequest()
        req.contents = [types.Content(role="user", parts=[types.Part.from_text(
            text="Plan: look at route() for injection.")])]
        asyncio.run(cb(callback_context=ctx, llm_request=req))
        texts = [p.text for c in req.contents for p in c.parts if p.text]
        self.assertEqual(texts.count("Plan: look at route() for injection."), 1)
        self.assertEqual(len(req.contents), 3, "call, result, then ADK's opener")

    def test_the_ceiling_sees_the_inherited_transcript(self):
        cb = gl._make_fork_before_model_callback("researcher", "planner")
        big = [_ev("planner", "user", [types.Part(function_response=types.FunctionResponse(
            name="read_file", id=f"c{i}", response={"result": "x" * 50_000}))]) for i in range(4)]
        ctx = SimpleNamespace(session=SimpleNamespace(events=big))
        req = self._request()
        before = gl._request_chars(req)
        asyncio.run(cb(callback_context=ctx, llm_request=req))
        self.assertGreater(gl._request_chars(req), before + 190_000)


class LoaderTest(unittest.TestCase):
    def _spec(self, fork_from):
        return {
            "name": "wf",
            "config": {"default_model": "openai/gpt-5.3-codex"},
            "nodes": [
                {"id": "planner", "type": "agent", "tools": ["read_file"], "system_prompt": "plan"},
                {"id": "researcher", "type": "agent", "tools": ["read_file"], "system_prompt": "research",
                 **({"fork_from": fork_from} if fork_from else {})},
            ],
            "edges": [{"from": "START", "to": "planner"}, {"from": "planner", "to": "researcher"}],
        }

    def _load(self, spec):
        import os
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wf.json")
            Path(path).write_text(json.dumps(spec))
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
                return gl.load_workflow_from_json(path, load_local=False)

    def _agent(self, workflow, name):
        for edge in workflow.edges:
            for node in (getattr(edge, "from_node", None), getattr(edge, "to_node", None)):
                agent = getattr(node, "agent", node)
                if getattr(agent, "name", None) == name:
                    return agent
        raise AssertionError(name)

    def test_a_fork_puts_the_callback_first_and_only_on_the_forked_node(self):
        workflow, _cfg = self._load(self._spec("planner"))
        names = [getattr(cb, "__qualname__", "") for cb in self._agent(workflow, "researcher").before_model_callback]
        self.assertIn("fork_before_model", names[0], names)
        self.assertLess(names.index(next(n for n in names if "fork_before_model" in n)),
                        names.index(next(n for n in names if "node_context_ceiling" in n)))
        planner_names = [getattr(cb, "__qualname__", "") for cb in self._agent(workflow, "planner").before_model_callback]
        self.assertFalse(any("fork_before_model" in n for n in planner_names))

    def test_a_fork_is_part_of_the_harness_fingerprint(self):
        _wf, cfg_plain = self._load(self._spec(None))
        _wf, cfg_fork = self._load(self._spec("planner"))
        self.assertNotEqual(cfg_plain.get("harness_fingerprint"), cfg_fork.get("harness_fingerprint"))
        nodes = [{"id": "a", "tools": [], "model": "m", "instruction": "i"}]
        self.assertEqual(gl.compute_harness_fingerprint(nodes),
                         gl.compute_harness_fingerprint([{**nodes[0], "fork_from": ""}]),
                         "no fork, same fingerprint as before the field existed")

    def test_the_loader_refuses_a_fork_from_nowhere_or_from_itself(self):
        with self.assertRaises(Exception) as cm:
            self._load(self._spec("nobody"))
        self.assertIn("fork_from", str(cm.exception))
        with self.assertRaises(Exception) as cm:
            self._load(self._spec("researcher"))
        self.assertIn("itself", str(cm.exception))


class PlannedEvalGraphTest(unittest.IsolatedAsyncioTestCase):
    """research_eval's planned arms: the planner runs first, the researcher
    follows with include_contents='none'; forked, its requests begin with
    the planner's transcript; fresh, they do not. A recording model stands
    in for the provider."""

    async def _run(self, fork: bool):
        from unittest import mock

        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_response import LlmResponse

        import evals.research_eval as re_mod
        import evals.stage_agents as sa

        seen: list = []

        def noop() -> str:
            """A tool the planner calls once, so its transcript has tool traffic."""
            return "ok"

        class Recorder(BaseLlm):
            model: str = "dummy"

            async def generate_content_async(self, req, stream=False):
                sys_inst = str(getattr(req.config, "system_instruction", "") or "")
                texts = [p.text for c in (req.contents or []) for p in (c.parts or []) if getattr(p, "text", None)]
                calls = sum(1 for c in (req.contents or []) for p in (c.parts or []) if getattr(p, "function_call", None))
                seen.append((sys_inst, texts, calls))
                planner_first_turn = sys_inst.startswith("You are planner") and not any(
                    getattr(p, "function_response", None) for c in (req.contents or []) for p in (c.parts or []))
                if planner_first_turn:
                    yield LlmResponse(content=types.Content(role="model", parts=[
                        types.Part(function_call=types.FunctionCall(name="noop", args={}))]))
                    return
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text="done")]))

        def build(stage_name, model_id=None, reasoning_effort=None, tools_override=None, **kw):
            from google.adk.agents import Agent

            from core.graph_loader import LiteralInstruction
            extra = {}
            if kw.get("include_contents") is not None:
                extra["include_contents"] = kw["include_contents"]
            if kw.get("before_model_callback"):
                extra["before_model_callback"] = list(kw["before_model_callback"])
            instruction = f"You are {stage_name}."
            if kw.get("instruction_suffix"):
                instruction += "\n\n" + kw["instruction_suffix"]
            tools = [noop] if stage_name == "planner" else []
            return Agent(name=stage_name, model=Recorder(), instruction=LiteralInstruction(instruction),
                         tools=tools, **extra)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "t")
            (target / "app").mkdir(parents=True)
            (target / "app" / "a.py").write_text("x = 1\n")
            (target / "ground_truth.json").write_text(json.dumps({
                "format_version": "1.0",
                "findings": [{"id": "gt1", "files": ["app/a.py"], "match_any": ["anything"]}],
            }))
            with mock.patch.object(sa, "build_stage_agent", build):
                result = await re_mod.eval_researcher(
                    "dummy", "low", target_dir=target, toolset="baseline",
                    predecessor="planner", fork=fork, arm_label="x",
                )
        return seen, result

    async def test_fresh_and_forked_planned_arms(self):
        seen, result = await self._run(fork=False)
        # planner (tool call), planner (final), researcher.
        self.assertEqual([s[0].split(".")[0] for s in seen], ["You are planner", "You are planner", "You are researcher"])
        self.assertTrue(any("Initial Task Input" in t for t in seen[0][1]), "the seed reaches the planner")
        researcher = seen[2]
        self.assertFalse(any("Initial Task Input" in t for t in researcher[1]), "and not the researcher")
        self.assertEqual(researcher[2], 0, "fresh: none of the planner's tool traffic")
        self.assertEqual(researcher[1], ["done"], "only the planner's final reply, as ADK opens every node's turn")
        self.assertIn("researcher stage", researcher[0], "the briefing rides in its instruction")
        self.assertNotIn(gl.FORK_NOTICE_MARKER, researcher[0])
        self.assertEqual((result["predecessor"], result["fork"], result["shape"]), ("planner", False, "planner"))

        seen, result = await self._run(fork=True)
        researcher = seen[2]
        self.assertEqual(researcher[2], 1, "forked: the planner's tool call is in front")
        self.assertEqual(researcher[1], ["done"], "the planner's final reply once, not twice")
        self.assertIn(gl.FORK_NOTICE_MARKER, researcher[0])
        self.assertEqual((result["fork"], result["shape"]), (True, "planner+fork"))
        self.assertIn("planner", result["tokens_by_stage"])


if __name__ == "__main__":
    unittest.main()
