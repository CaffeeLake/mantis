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

"""The spend report says how the run kept within the model window.

Whether prompt caching worked, how often a stage continued in a fresh
context or was ended at the ceiling, how many tool results were cut, how
many requests were retried: these used to be answerable only by querying
the sessions database by hand. Now they are counted where they happen and
written with --spend-report and the end-of-run summary.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core import config as C  # noqa: E402
from core.budget import (  # noqa: E402
    CONTEXT_OBSERVATIONS, REQUEST_OBSERVATIONS, BudgetConfig, BudgetController, CampaignBudgetScope,
    observe, render_spend_summary,
)
from core.context import RunContext, current_run_context  # noqa: E402


class SnapshotShapeTest(unittest.TestCase):
    def _ctrl(self) -> BudgetController:
        return BudgetController(config=BudgetConfig(max_tokens=0, max_wall_clock_seconds=0), run_id="r")

    def test_input_output_and_cache_share_are_reported(self):
        ctrl = self._ctrl()
        ctrl.record_tokens(1_000, cached_count=0, prompt_count=900, output_count=100)
        ctrl.record_tokens(2_100, cached_count=900, prompt_count=2_000, output_count=100)
        snap = ctrl.spend_snapshot()
        self.assertEqual(snap["prompt_tokens"], 2_900)
        self.assertEqual(snap["output_tokens"], 200)
        self.assertEqual(snap["cached_tokens"], 900)
        self.assertAlmostEqual(snap["cache_hit_ratio"], 900 / 2_900, places=4)
        self.assertEqual(snap["llm_calls"], 2)
        # The old fields keep their meaning.
        self.assertEqual(snap["fresh_tokens"] + snap["cached_tokens"], 3_100)

    def test_no_input_metered_means_no_ratio(self):
        ctrl = self._ctrl()
        ctrl.record_tokens(500)
        self.assertIsNone(ctrl.spend_snapshot()["cache_hit_ratio"])

    def test_every_counter_is_present_even_at_zero(self):
        snap = self._ctrl().spend_snapshot()
        self.assertEqual(set(snap["context"]), set(CONTEXT_OBSERVATIONS))
        self.assertEqual(set(snap["requests"]), set(REQUEST_OBSERVATIONS))
        self.assertTrue(all(v == 0 for v in snap["context"].values()))
        self.assertEqual(snap["other"], {})

    def test_observe_counts_and_keeps_unknown_keys_apart(self):
        ctrl = self._ctrl()
        ctrl.observe("continuations")
        ctrl.observe("continuations", 2)
        ctrl.observe("retries")
        ctrl.observe("something_new", 5)
        ctrl.observe("", 9)
        snap = ctrl.spend_snapshot()
        self.assertEqual(snap["context"]["continuations"], 3)
        self.assertEqual(snap["requests"]["retries"], 1)
        self.assertEqual(snap["other"], {"something_new": 5})

    def test_a_campaign_scope_mirrors_to_the_run(self):
        run = self._ctrl()
        scope = CampaignBudgetScope(run)
        scope.record_tokens(1_000, cached_count=600, prompt_count=800, output_count=200)
        scope.observe("tool_result_cuts")
        self.assertEqual(run.prompt_tokens, 800)
        self.assertEqual(run.output_tokens, 200)
        self.assertEqual(run.spend_snapshot()["context"]["tool_result_cuts"], 1)
        self.assertEqual(scope.spend_snapshot()["context"]["tool_result_cuts"], 1)


class ModuleObserveTest(unittest.TestCase):
    def setUp(self):
        self.token = None

    def tearDown(self):
        if self.token is not None:
            current_run_context.reset(self.token)

    def test_observe_reaches_the_current_runs_controller(self):
        ctrl = BudgetController(config=BudgetConfig(max_tokens=0, max_wall_clock_seconds=0), run_id="r")
        self.token = current_run_context.set(RunContext(jail_dir="/tmp", db_path="", budget_controller=ctrl))
        observe("ceiling_ends")
        observe("retries", 3)
        snap = ctrl.spend_snapshot()
        self.assertEqual(snap["context"]["ceiling_ends"], 1)
        self.assertEqual(snap["requests"]["retries"], 3)

    def test_observe_without_a_run_is_silent(self):
        observe("continuations")  # no context: nothing to count, nothing raised

    def test_a_tool_result_cut_is_counted_but_a_notice_alone_is_not(self):
        from core.graph_loader import _make_after_tool_guard_callback

        ctrl = BudgetController(config=BudgetConfig(max_tokens=0, max_wall_clock_seconds=0), run_id="r")
        self.token = current_run_context.set(
            RunContext(jail_dir="/tmp", db_path="", budget_controller=ctrl, node_context={}))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(C._TOOL_RESPONSE_CAP_ENV, None)
            with mock.patch.object(C, "resolve_context_limit", return_value=200_000):
                guard = _make_after_tool_guard_callback("architect", ["read_file"], model="openai/gpt-5.3-codex")
                guard(type("T", (), {"name": "read_file"})(), {"filepath": "a"}, None, "x" * 2_000_000)
                guard(type("T", (), {"name": "read_file"})(), {"filepath": "a"}, None, "small")
        self.assertEqual(ctrl.spend_snapshot()["context"]["tool_result_cuts"], 1)


class _Msg:
    def __init__(self, fields):
        self.provider_specific_fields = fields


class _Choice:
    def __init__(self, fields):
        self.message = _Msg(fields)


class _Resp:
    def __init__(self, fields=None, hidden=None):
        self.choices = [_Choice(fields or {})]
        self._hidden_params = hidden or {}


class ContextEditsTest(unittest.TestCase):
    def test_applied_edits_are_read_from_the_providers_fields(self):
        resp = _Resp({"context_management": {"applied_edits": [
            {"type": "clear_tool_uses_20250919", "cleared_tool_uses": 12, "cleared_input_tokens": 180_000},
            {"type": "clear_thinking_20251015", "cleared_input_tokens": 2_000},
        ]}})
        self.assertEqual(C.context_edits_applied(resp), (2, 182_000))

    def test_hidden_params_are_the_fallback(self):
        resp = _Resp({}, hidden={"provider_specific_fields": {"context_management": {"applied_edits": [
            {"type": "clear_tool_uses_20250919", "cleared_input_tokens": 10}]}}})
        self.assertEqual(C.context_edits_applied(resp), (1, 10))

    def test_nothing_reported_is_zero(self):
        self.assertEqual(C.context_edits_applied(_Resp()), (0, 0))
        self.assertEqual(C.context_edits_applied(object()), (0, 0))
        self.assertEqual(C.context_edits_applied(_Resp({"context_management": {"applied_edits": []}})), (0, 0))


class SummaryTest(unittest.TestCase):
    def test_the_summary_names_the_cache_share_and_the_nonzero_counters(self):
        ctrl = BudgetController(config=BudgetConfig(max_tokens=0, max_wall_clock_seconds=0), run_id="r")
        ctrl.record_tokens(10_000, cached_count=8_000, prompt_count=9_000, output_count=1_000)
        ctrl.observe("continuations", 2)
        ctrl.observe("context_edits_applied", 3)
        ctrl.observe("context_edit_cleared_tokens", 150_000)
        ctrl.observe("rate_limit_retries", 1)
        lines = render_spend_summary(ctrl.spend_snapshot())
        self.assertEqual(len(lines), 2)
        self.assertIn("9,000 input tokens (89% read from the prompt cache)", lines[0])
        self.assertIn("1,000 output tokens", lines[0])
        self.assertIn("2 stage continuation(s)", lines[1])
        self.assertIn("3 context edit(s) applied", lines[1])
        self.assertIn("150,000 input tokens cleared by context editing", lines[1])
        self.assertIn("1 rate-limit retry(ies)", lines[1])
        self.assertNotIn("ceiling", lines[1], "zero counters are not listed")

    def test_a_quiet_run_is_one_line(self):
        ctrl = BudgetController(config=BudgetConfig(max_tokens=0, max_wall_clock_seconds=0), run_id="r")
        ctrl.record_tokens(1_000, prompt_count=900, output_count=100)
        lines = render_spend_summary(ctrl.spend_snapshot())
        self.assertEqual(len(lines), 1)
        self.assertIn("0% read from the prompt cache", lines[0])

    def test_a_snapshot_without_the_split_still_renders(self):
        lines = render_spend_summary({"fresh_tokens": 400, "cached_tokens": 100, "llm_calls": 2})
        self.assertEqual(lines, ["500 tokens (100 cached), 2 LLM call(s)"])


if __name__ == "__main__":
    unittest.main()
