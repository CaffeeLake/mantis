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

"""One tool response may not be the reason a request no longer fits the model.

Reported in the field (issue 13): a 1 MiB read_file (~262k tokens) and a
975k-character recorded summary were each handed whole to a 272k-token
model, and the pre-dispatch guard refused the turn. The per-tool byte limits
are sized for million-token windows; the bound here is sized for the model
the node actually runs on.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = Path(__file__).resolve().parents[1]
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core import config as C  # noqa: E402
from core.llm_gateway import UNTRUSTED_DATA_END, wrap_untrusted_content  # noqa: E402
from core.truncation import TRUNCATION_MARKER_PREFIX, bound_fenced_text  # noqa: E402


class FencedBoundTest(unittest.TestCase):
    def test_within_limit_is_untouched(self):
        text = wrap_untrusted_content("x" * 100, filename="a.py")
        self.assertEqual(bound_fenced_text(text, 10_000), text)

    def test_a_cut_keeps_the_fence_closed_and_the_marker_outside_it(self):
        body = "".join(f"line {i}: {'y' * 40}\n" for i in range(2000))
        text = wrap_untrusted_content(body, filename="big.json")
        out = bound_fenced_text(text, 5_000, "read_file output")
        self.assertLessEqual(len(out), 5_000 + 400, "the marker is the only thing over the limit")
        self.assertTrue(out.startswith("<<<UNTRUSTED_SOURCE_CODE_DATA_START"))
        end_at = out.rfind(UNTRUSTED_DATA_END)
        marker_at = out.find(TRUNCATION_MARKER_PREFIX)
        self.assertGreater(end_at, 0)
        self.assertGreater(marker_at, end_at, "the marker follows the END fence, never sits inside it")
        self.assertIn("read_file output cut by the harness after", out)
        self.assertEqual(out.count(UNTRUSTED_DATA_END), 1)

    def test_harness_text_after_the_fence_survives(self):
        text = wrap_untrusted_content("z" * 20_000) + "\n\n[HARNESS STEERING HINT: narrow the read]"
        out = bound_fenced_text(text, 4_000)
        self.assertTrue(out.endswith("[HARNESS STEERING HINT: narrow the read]"))
        self.assertIn(TRUNCATION_MARKER_PREFIX, out)
        self.assertLess(out.find(TRUNCATION_MARKER_PREFIX), out.find("[HARNESS STEERING HINT"))

    def test_unfenced_text_is_cut_plainly(self):
        out = bound_fenced_text("a" * 10_000, 1_000, "git log")
        self.assertTrue(out.startswith("a" * 1_000 + "\n" + TRUNCATION_MARKER_PREFIX))
        self.assertIn("git log cut by the harness after 1000 of 10000 chars", out)

    def test_never_raises_on_odd_input(self):
        self.assertEqual(bound_fenced_text(None, 10), "")
        out = bound_fenced_text("abc", "bad")  # a bad limit is treated as zero
        self.assertTrue(out.startswith("\n" + TRUNCATION_MARKER_PREFIX))
        self.assertIn("after 0 of 3 chars", out)


class CapResolutionTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(C._TOOL_RESPONSE_CAP_ENV, None)
        os.environ.pop("MANTIS_MAX_INPUT_TOKENS", None)

    def test_a_known_window_yields_a_quarter_of_it_in_chars(self):
        with mock.patch.object(C, "resolve_context_limit", return_value=272_000):
            cap = C.resolve_tool_response_char_cap("openai/gpt-5.3-codex")
        self.assertEqual(cap, int(272_000 * C.TOOL_RESPONSE_WINDOW_FRACTION) * C._CHARS_PER_TOKEN)
        # The 1 MiB read from the report would have been cut to a quarter of the window...
        self.assertLess(cap, 1_048_751)
        # ...and the request that carried it fits with room for everything else.
        self.assertLess(cap // C._CHARS_PER_TOKEN, int(272_000 * C._CONTEXT_BUDGET_RATIO) // 2)

    def test_a_million_token_window_is_not_tightened_below_the_per_tool_byte_limit(self):
        from tools.research_tools import MAX_READ_SIZE
        with mock.patch.object(C, "resolve_context_limit", return_value=1_048_576):
            cap = C.resolve_tool_response_char_cap("vertex_ai/gemini-big")
        self.assertGreaterEqual(cap, MAX_READ_SIZE, "a whole-file read that fits the window still fits")

    def test_unknown_model_means_no_cap_unless_the_operator_sets_one(self):
        with mock.patch.object(C, "resolve_context_limit", return_value=None):
            self.assertIsNone(C.resolve_tool_response_char_cap("gemini/unknown-model"))
            os.environ[C._TOOL_RESPONSE_CAP_ENV] = "120000"
            self.assertEqual(C.resolve_tool_response_char_cap("gemini/unknown-model"), 120_000)
            os.environ[C._TOOL_RESPONSE_CAP_ENV] = "nonsense"
            self.assertIsNone(C.resolve_tool_response_char_cap("gemini/unknown-model"))

    def test_the_operator_input_cap_tightens_the_bound(self):
        os.environ["MANTIS_MAX_INPUT_TOKENS"] = "100000"
        with mock.patch.object(C, "resolve_context_limit", return_value=1_000_000):
            cap = C.resolve_tool_response_char_cap("vertex_ai/gemini-big")
        self.assertEqual(cap, int(100_000 * C.TOOL_RESPONSE_WINDOW_FRACTION) * C._CHARS_PER_TOKEN)

    def test_the_floor_protects_the_verdict_stages(self):
        with mock.patch.object(C, "resolve_context_limit", return_value=8_000):
            self.assertEqual(C.resolve_tool_response_char_cap("tiny"), C.MIN_TOOL_RESPONSE_CHARS)


class _Tool:
    def __init__(self, name):
        self.name = name


class AfterToolBoundTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(C._TOOL_RESPONSE_CAP_ENV, None)

    def _guard(self, tools=("read_file", "get_summary"), model="openai/gpt-5.3-codex", window=272_000):
        from core.graph_loader import _make_after_tool_guard_callback
        patcher = mock.patch.object(C, "resolve_context_limit", return_value=window)
        patcher.start()
        self.addCleanup(patcher.stop)
        return _make_after_tool_guard_callback("architect", list(tools), model=model)

    def test_the_reported_whole_file_read_is_cut_to_fit(self):
        guard = self._guard()
        big = wrap_untrusted_content("{" + '"k": "v",\n' * 110_000 + "}", filename="umsdAPI.json")
        self.assertGreater(len(big), 1_000_000)
        out = guard(_Tool("read_file"), {"filepath": "umsdAPI.json"}, None, big)
        self.assertIsInstance(out, str)
        self.assertLessEqual(len(out), 272_000 + 400)
        self.assertIn("read_file output cut by the harness", out)
        self.assertGreater(out.rfind(TRUNCATION_MARKER_PREFIX), out.rfind(UNTRUSTED_DATA_END))

    def test_the_reported_recorded_summary_is_cut_the_same_way(self):
        guard = self._guard()
        summary = wrap_untrusted_content("Canonical System Architecture\n" * 30_000, filename="recorded_summary")
        out = guard(_Tool("get_summary"), {}, None, {"result": summary})
        self.assertIsInstance(out, dict)
        self.assertIn("get_summary output cut by the harness", out["result"])
        self.assertLessEqual(len(out["result"]), 272_000 + 400)

    def test_small_responses_pass_through_untouched(self):
        guard = self._guard()
        self.assertIsNone(guard(_Tool("read_file"), {"filepath": "a.py", "start_line": 1, "end_line": 5}, None, "small"))
        self.assertIsNone(guard(_Tool("list_files"), {}, None, {"result": "[]"}))

    def test_no_model_or_unknown_window_means_the_old_behaviour(self):
        from core.graph_loader import _make_after_tool_guard_callback
        guard = _make_after_tool_guard_callback("architect", ["read_file"])
        big = "x" * 2_000_000
        self.assertIsNone(guard(_Tool("read_file"), {"filepath": "a"}, None, big))
        with mock.patch.object(C, "resolve_context_limit", return_value=None):
            guard = _make_after_tool_guard_callback("architect", ["read_file"], model="mystery/model")
            self.assertIsNone(guard(_Tool("read_file"), {"filepath": "a"}, None, big))

    def test_bound_and_steering_hint_compose(self):
        guard = self._guard(tools=("read_file", "get_function_boundary", "find_symbol"), window=200_000)
        big = wrap_untrusted_content("q" * 900_000, filename="a.py")
        last = None
        for _ in range(3):
            last = guard(_Tool("read_file"), {"filepath": "a.py"}, None, big)
        self.assertIsInstance(last, str)
        self.assertIn("HARNESS STEERING HINT", last)
        self.assertIn(TRUNCATION_MARKER_PREFIX, last)
        self.assertLess(last.find(TRUNCATION_MARKER_PREFIX), last.find("HARNESS STEERING HINT"))

    def test_every_agent_node_in_the_shipped_workflow_gets_the_guard(self):
        from core.graph_loader import load_workflow_from_json
        workflow, _cfg = load_workflow_from_json(str(_REF_ROOT / "workflow.json"), load_local=False)
        seen = {}
        for edge in workflow.edges:
            for node in (getattr(edge, "from_node", None), getattr(edge, "to_node", None)):
                agent = getattr(node, "agent", node)
                if agent is not None and hasattr(agent, "after_tool_callback") and getattr(agent, "tools", None):
                    seen[getattr(agent, "name", id(agent))] = agent
        self.assertGreater(len(seen), 5, "the shipped graph has many tool-using agents")
        for name, agent in seen.items():
            self.assertIsNotNone(agent.after_tool_callback, f"{name} has no tool-response bound")


class PinnedDependencyPreflightTest(unittest.TestCase):
    """Issue 14's traceback was ADK 2.9.1's replay barrier on a venv that had not been
    reinstalled after the pin moved to 2.11.0; the preflight now says so."""

    def test_matching_install_passes_and_a_stale_one_warns_with_the_fix(self):
        import tempfile
        from importlib import metadata
        from scripts.configure import check_pinned_dependencies

        installed = metadata.version("google-adk")
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text(f"foo==1.0 \\\n    --hash=sha256:aa\ngoogle-adk=={installed} \\\n    --hash=sha256:bb\n")
            ok, msg = check_pinned_dependencies(str(req))
            self.assertTrue(ok, msg)
            req.write_text("google-adk==0.0.1 \\\n    --hash=sha256:bb\n")
            ok, msg = check_pinned_dependencies(str(req))
            self.assertFalse(ok)
            self.assertIn(f"google-adk: installed {installed}, pinned 0.0.1", msg)
            self.assertIn("pip install --require-hashes -r", msg)
            self.assertIn("not Mantis bugs", msg)
            ok, msg = check_pinned_dependencies(str(Path(tmp) / "absent.txt"))
            self.assertTrue(ok, "a missing manifest is not a finding")

    def test_the_real_manifest_matches_this_venv(self):
        from scripts.configure import check_pinned_dependencies
        ok, msg = check_pinned_dependencies()
        self.assertTrue(ok, msg)

    def test_preflight_reports_it_without_gating(self):
        from scripts.configure import run_preflight_checks
        with mock.patch("scripts.configure.check_pinned_dependencies", return_value=(False, "stale")):
            ok, messages = run_preflight_checks({"default_model": "openai/x", "api_base": "http://127.0.0.1:9"},
                                                test_sandbox=False)
        self.assertTrue(any("[DEPENDENCY PREFLIGHT] ⚠️ WARNING: stale" in m for m in messages), messages)
        self.assertTrue(ok, "a stale pin warns; it does not refuse to run")


if __name__ == "__main__":
    unittest.main()
