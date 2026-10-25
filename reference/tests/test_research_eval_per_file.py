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

"""The researcher eval's per-file campaign mode: file selection, the shape
it adds to a run's identity, the sweep carrying it, and one cell running a
campaign per file with its own target."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import sweep  # noqa: E402
from evals.research_eval import (  # noqa: E402
    DEFAULT_CAMPAIGN_FILES, DEFAULT_TARGET, GROUND_TRUTH_BASENAME, PLANNER_PROMPT_FILE, RESEARCH_PROMPT_FILE,
    campaign_shape, eval_researcher, load_research_ground_truth, select_campaign_files,
)


class CampaignShapeTest(unittest.TestCase):
    def test_whole_adds_nothing_so_earlier_fingerprints_stand(self):
        self.assertEqual(campaign_shape("whole", "gt"), "")
        self.assertEqual(campaign_shape("", ""), "")

    def test_per_file_names_the_mode_and_the_selector(self):
        self.assertEqual(campaign_shape("per-file", "gt+clean:3"), "/per-file:gt+clean:3")
        self.assertEqual(campaign_shape("per-file", ""), f"/per-file:{DEFAULT_CAMPAIGN_FILES}")

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            campaign_shape("per-function", "all")

    def test_per_file_briefings_substitute_the_file_literally(self):
        for template in (PLANNER_PROMPT_FILE, RESEARCH_PROMPT_FILE):
            self.assertIn("{filepath}", template)
            self.assertIn("app/x.py", template.replace("{filepath}", "app/x.py"))


class SelectCampaignFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = Path(self.tmp.name) / "app"
        for rel in ("a.py", "web/r.py", "services/s.py", "services/t.py", "utils/u.py", "vendor/v.py", "README.md"):
            p = self.app / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x = 1\n")
        self.gt = [
            {"id": "one", "files": ["web/r.py", "services/s.py"], "match_any": ["x"], "kind": "seeded",
             "cross_file": True, "class": "c"},
            {"id": "dead", "files": ["vendor/v.py"], "match_any": ["x"], "kind": "decoy_vendor",
             "cross_file": False, "class": "c"},
            {"id": "gone", "files": ["services/missing.py"], "match_any": ["x"], "kind": "seeded",
             "cross_file": False, "class": "c"},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_is_every_source_file(self):
        self.assertEqual(select_campaign_files(self.app, self.gt, "all"),
                         ["a.py", "services/s.py", "services/t.py", "utils/u.py", "vendor/v.py", "web/r.py"])

    def test_gt_is_the_named_files_that_exist_decoys_included(self):
        self.assertEqual(select_campaign_files(self.app, self.gt, "gt"), ["services/s.py", "vendor/v.py", "web/r.py"])

    def test_gt_plus_clean_is_deterministic_and_capped(self):
        once = select_campaign_files(self.app, self.gt, "gt+clean:2")
        again = select_campaign_files(self.app, self.gt, "gt+clean:2")
        self.assertEqual(once, again)
        self.assertEqual(len(once), 5)
        self.assertTrue({"services/s.py", "vendor/v.py", "web/r.py"} < set(once))
        self.assertEqual(len(select_campaign_files(self.app, self.gt, "gt+clean:99")), 6)

    def test_unknown_selector_is_refused(self):
        with self.assertRaises(ValueError):
            select_campaign_files(self.app, self.gt, "half")


class SweepCarriesTheModeTest(unittest.TestCase):
    def _cfg(self, out, **kw):
        base = dict(out_dir=str(out), arms={"a": "structural@planner", "b": "structural@planner+fork"},
                    seeds=[sweep.targets.DEV_SEEDS[0]], repeats=1, size="small", model="m", effort="low")
        base.update(kw)
        return sweep.SweepConfig(**base)

    def test_whole_identity_is_unchanged_so_old_sweeps_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            ident = self._cfg(Path(tmp)).identity()
        self.assertNotIn("campaigns", ident)
        self.assertNotIn("files", ident)

    def test_per_file_is_part_of_the_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._cfg(Path(tmp), campaigns="per-file", files="gt").identity()
            b = self._cfg(Path(tmp), campaigns="per-file", files="all").identity()
        self.assertEqual(a["campaigns"], "per-file")
        self.assertNotEqual(a, b)
        with self.assertRaises(ValueError):
            self._cfg(Path("x"), campaigns="per-function")

    def test_the_eval_is_told_the_mode_only_when_set(self):
        seen = []

        async def fake_eval(model, effort, **kw):
            seen.append(kw)
            return {"graded": True, "status": "complete", "recall": 1.0, "precision": 1.0, "total_tokens": 1}

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(sweep.run_sweep(self._cfg(Path(tmp) / "w"), eval_fn=fake_eval, log=lambda s: None))
            self.assertTrue(all("campaigns" not in kw for kw in seen))
            seen.clear()
            asyncio.run(sweep.run_sweep(
                self._cfg(Path(tmp) / "p", campaigns="per-file", files="gt+clean:1", concurrency=2),
                eval_fn=fake_eval, log=lambda s: None))
            self.assertEqual({(kw["campaigns"], kw["files"], kw["concurrency"]) for kw in seen},
                             {("per-file", "gt+clean:1", 2)})


class _FakeRunner:
    """Stands in for the ADK Runner: each run_async records the campaign's
    target and reports one finding in that file through the real tool."""

    targets: list = []
    apps: list = []

    def __init__(self, **kwargs):
        _FakeRunner.apps.append(kwargs.get("app"))

    async def run_async(self, session_id=None, user_id=None, new_message=None):
        from core.context import current_run_context
        from tools.research_tools import report_findings

        ctx = current_run_context.get()
        rel = Path(ctx.target_file).relative_to(Path(ctx.jail_dir) / "app").as_posix()
        _FakeRunner.targets.append(rel)
        await asyncio.sleep(0)  # let the other campaigns interleave
        cited = "app.py" if rel == "." else rel  # whole mode: the target is the tree
        report_findings({"findings": [{
            "title": f"issue in {cited}", "filepath": f"app/{cited}", "line_numbers": [1],
            "description": "the eval's fake researcher", "severity": "LOW", "cwe": "CWE-1",
        }]})
        yield mock.Mock(author="researcher", content=None, usage_metadata=None, partial=False)


class PerFileCellTest(unittest.TestCase):
    @staticmethod
    def _fake_agent(stage_name, **_kw):
        # A real (model-less) agent: the eval builds a real App around it,
        # which validates its root agent; only the Runner is faked.
        from google.adk.agents import Agent
        return Agent(name=stage_name)

    def test_a_per_file_cell_runs_one_campaign_per_file_with_its_own_target(self):
        _FakeRunner.targets = []
        gt = load_research_ground_truth(DEFAULT_TARGET / GROUND_TRUTH_BASENAME)
        with mock.patch("google.adk.runners.Runner", _FakeRunner), \
                mock.patch("evals.stage_agents.build_stage_agent", side_effect=self._fake_agent):
            result = asyncio.run(eval_researcher(
                "m", "low", toolset="baseline", campaigns="per-file", files="gt", concurrency=3,
            ))
        expected = select_campaign_files(DEFAULT_TARGET, gt, "gt")
        self.assertEqual(sorted(_FakeRunner.targets), expected)
        self.assertEqual(result["campaigns"], "per-file")
        self.assertEqual(result["campaign_files"], expected)
        self.assertEqual(result["campaigns_run"], len(expected))
        self.assertEqual(result["findings_reported"], len(expected), "every campaign's finding reached the one database")
        self.assertEqual({s["status"] for s in result["campaign_stats"]}, {"complete"})
        self.assertIn("/per-file:gt", result["shape"])
        self.assertTrue(result["graded"])

    def test_whole_mode_is_the_old_shape(self):
        _FakeRunner.targets = []
        with mock.patch("google.adk.runners.Runner", _FakeRunner), \
                mock.patch("evals.stage_agents.build_stage_agent", side_effect=self._fake_agent):
            result = asyncio.run(eval_researcher("m", "low", toolset="baseline"))
        self.assertEqual(_FakeRunner.targets, ["."])
        self.assertEqual(result["campaigns"], "whole")
        self.assertEqual(result["shape"], "solo")
        self.assertEqual(result["campaign_files"], [])
        self.assertEqual(result["campaigns_run"], 1)
        self.assertEqual(result["campaign_stats"][0]["file"], "")
        self.assertEqual(result["findings_reported"], 1)

    def test_the_eval_runs_with_production_context_caching(self):
        # main.py turns context caching on by default; for Claude that is
        # what makes ADK mark the prompt prefix cacheable, so an eval without
        # it would bill a forked transcript at a price the pipeline never pays.
        from google.adk.agents.context_cache_config import ContextCacheConfig
        _FakeRunner.apps = []
        with mock.patch("google.adk.runners.Runner", _FakeRunner), \
                mock.patch("evals.stage_agents.build_stage_agent", side_effect=self._fake_agent):
            asyncio.run(eval_researcher("m", "low", toolset="baseline"))
        self.assertEqual(len(_FakeRunner.apps), 1)
        self.assertIsInstance(_FakeRunner.apps[0].context_cache_config, ContextCacheConfig)


if __name__ == "__main__":
    unittest.main()
