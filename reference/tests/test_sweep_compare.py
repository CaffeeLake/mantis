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

"""The sweep scheduler (interleaving, completion marks, retries, identity
lock, holdout refusal) and the paired statistics behind the change gate.
No model is called: the eval function is injected."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from evals import compare, sweep  # noqa: E402
from evals.research_targets import gen  # noqa: E402


def _per_entry(detected_ids, all_ids, decoy_ids=()):
    rows = [{"id": i, "kind": "seeded", "detected": i in detected_ids} for i in all_ids]
    rows += [{"id": i, "kind": "decoy_sanitized", "detected": False} for i in decoy_ids]
    return rows


def _cell(arm, seed, repeat, recall, detected=(), total_ids=("a", "b", "c", "d"), tokens=1000, graded=True, **extra):
    c = {
        "arm": arm, "seed": seed, "repeat": repeat, "graded": graded,
        "status": "complete" if graded else "ungraded",
        "recall": recall, "detected": len(detected), "ground_truth_total": len(total_ids),
        "per_entry": _per_entry(set(detected), total_ids), "total_tokens": tokens,
        "decoys_hit": 0, "decoy_fp_rate": 0.0, "precision": 1.0, "precision_lenient": 1.0,
        "zero_findings_complete": recall == 0 and graded, "cap_hit": False, "abort_guard_rearms": 0,
        "telemetry": {"unwindowed_reads": 3, "structural_nav_calls": 0},
    }
    c.update(extra)
    return c


class ParsingTest(unittest.TestCase):
    def test_seed_and_arm_specs(self):
        self.assertEqual(sweep.parse_seeds("1000-1002"), [1000, 1001, 1002])
        self.assertEqual(sweep.parse_seeds("1000, 1005"), [1000, 1005])
        self.assertEqual(sweep.parse_seeds("dev:3"), list(gen.DEV_SEEDS)[:3])
        self.assertEqual(sweep.parse_seeds("holdout:2"), list(gen.HOLDOUT_SEEDS)[:2])
        self.assertEqual(sweep.parse_arms("baseline,structural"), {"baseline": "baseline", "structural": "structural"})
        self.assertEqual(sweep.parse_arms("ctl=baseline, nav=structural"), {"ctl": "baseline", "nav": "structural"})

    def test_cells_are_interleaved_with_repeats_outermost(self):
        order = sweep.cell_order(["A", "B"], [1, 2], 2)
        self.assertEqual(
            [(c.arm, c.seed, c.repeat) for c in order],
            [("A", 1, 0), ("B", 1, 0), ("A", 2, 0), ("B", 2, 0), ("A", 1, 1), ("B", 1, 1), ("A", 2, 1), ("B", 2, 1)],
        )
        self.assertEqual(order[0].key, "A__1__r0")


class SchedulerTest(unittest.TestCase):
    def _cfg(self, out, **kw):
        base = dict(out_dir=str(out), arms={"baseline": "baseline", "structural": "structural"},
                    seeds=[1000, 1001], repeats=2, size="small", model="m", effort="low", max_attempts=2)
        base.update(kw)
        return sweep.SweepConfig(**base)

    def test_runs_every_cell_once_skips_complete_on_resume_and_locks_identity(self):
        calls = []

        async def fake_eval(model, effort, *, target_dir, toolset, prompt_override, max_tokens, max_tool_calls, arm_label):
            calls.append((arm_label, Path(target_dir).name))
            self.assertTrue((Path(target_dir) / "ground_truth.json").exists())
            return _cell(arm_label, 0, 0, 0.5, detected=("a", "b"))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sw"
            cfg = self._cfg(out)
            summary = asyncio.run(sweep.run_sweep(cfg, eval_fn=fake_eval, log=lambda s: None))
            self.assertEqual(summary["ran"], 8)
            self.assertEqual(summary["outcomes"], {"complete": 8})
            # Interleaved: arms alternate within each (seed, repeat).
            self.assertEqual([a for a, _ in calls[:4]], ["baseline", "structural", "baseline", "structural"])
            cells = sweep.load_cells(out)
            self.assertEqual(len(cells), 8)
            self.assertTrue(all(c["complete"] for c in cells))
            manifest = json.loads((out / sweep.MANIFEST_NAME).read_text())
            self.assertEqual(set(manifest["arm_fingerprints"]), {"baseline", "structural"})
            self.assertFalse(manifest["holdout"])

            # Resume: nothing to do.
            calls.clear()
            summary2 = asyncio.run(sweep.run_sweep(cfg, eval_fn=fake_eval, log=lambda s: None))
            self.assertEqual((summary2["ran"], calls), (0, []))

            # A different identity may not reuse the directory.
            with self.assertRaises(RuntimeError):
                asyncio.run(sweep.run_sweep(self._cfg(out, effort="high"), eval_fn=fake_eval, log=lambda s: None))

    def test_ungraded_cells_are_retried_then_exhausted(self):
        attempts = {}

        async def flaky_eval(model, effort, *, arm_label, **kw):
            attempts[arm_label] = attempts.get(arm_label, 0) + 1
            if arm_label == "structural":
                return _cell(arm_label, 0, 0, 0.0, graded=False, run_errors=["boom"])
            return _cell(arm_label, 0, 0, 0.25, detected=("a",))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sw"
            cfg = self._cfg(out, seeds=[1000], repeats=1, max_attempts=2)
            summary = asyncio.run(sweep.run_sweep(cfg, eval_fn=flaky_eval, log=lambda s: None))
            self.assertEqual(attempts, {"baseline": 1, "structural": 2})
            self.assertEqual(summary["outcomes"], {"complete": 1, "retry": 1, "exhausted": 1})
            cells = {c["arm"]: c for c in sweep.load_cells(out)}
            self.assertTrue(cells["structural"]["exhausted"])
            self.assertFalse(cells["structural"]["graded"])
            self.assertEqual(cells["structural"]["attempt"], 2)
            # Resume does not re-roll an exhausted cell.
            attempts.clear()
            asyncio.run(sweep.run_sweep(cfg, eval_fn=flaky_eval, log=lambda s: None))
            self.assertEqual(attempts, {})

    def test_harness_crash_is_recorded_as_ungraded_not_raised(self):
        async def crashing_eval(*a, **kw):
            raise RuntimeError("adapter blew up")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sw"
            cfg = self._cfg(out, seeds=[1000], repeats=1, arms={"baseline": "baseline"}, max_attempts=1)
            summary = asyncio.run(sweep.run_sweep(cfg, eval_fn=crashing_eval, log=lambda s: None))
            self.assertEqual(summary["outcomes"], {"exhausted": 1})
            cell = sweep.load_cells(out)[0]
            self.assertFalse(cell["graded"])
            self.assertIn("adapter blew up", cell["run_errors"][0])

    def test_holdout_seeds_need_explicit_confirmation(self):
        async def never(*a, **kw):
            raise AssertionError("must not run")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(gen.HoldoutViolation):
                asyncio.run(sweep.run_sweep(self._cfg(Path(tmp) / "a", seeds=[gen.HOLDOUT_SEEDS[0]]), eval_fn=never, log=lambda s: None))
            summary = asyncio.run(sweep.run_sweep(
                self._cfg(Path(tmp) / "b", seeds=[gen.HOLDOUT_SEEDS[0]], confirm_holdout=True),
                eval_fn=never, dry_run=True, log=lambda s: None))
            self.assertTrue(summary["dry_run"])
            self.assertTrue(json.loads((Path(tmp) / "b" / sweep.MANIFEST_NAME).read_text())["holdout"])


class StatisticsTest(unittest.TestCase):
    def _cells(self):
        # Two seeds, two repeats. Baseline recall 0.5/0.5 on seed 1, 0.25/0.75 on seed 2;
        # candidate 0.75/0.75 and 0.5/1.0. Per-target deltas: +0.25 and +0.25.
        return [
            _cell("baseline", 1, 0, 0.5, ("a", "b"), tokens=2000), _cell("baseline", 1, 1, 0.5, ("a", "b"), tokens=2000),
            _cell("baseline", 2, 0, 0.25, ("a",), tokens=2000), _cell("baseline", 2, 1, 0.75, ("a", "b", "c"), tokens=2000),
            _cell("structural", 1, 0, 0.75, ("a", "b", "c"), tokens=1500), _cell("structural", 1, 1, 0.75, ("b", "c", "d"), tokens=1500),
            _cell("structural", 2, 0, 0.5, ("c", "d"), tokens=1500), _cell("structural", 2, 1, 1.0, ("a", "b", "c", "d"), tokens=1500),
            _cell("structural", 2, 2, 0.0, graded=False),  # a crash: excluded everywhere
        ]

    def test_paired_deltas_and_bootstrap(self):
        cells = self._cells()
        self.assertEqual(compare.per_target_mean(cells, "baseline"), {1: 0.5, 2: 0.5})
        self.assertEqual(compare.paired_deltas(cells, "baseline", "structural"), {1: 0.25, 2: 0.25})
        ci = compare.bootstrap_ci([0.25, 0.25], n_boot=200)
        self.assertEqual((ci["n"], ci["mean"], ci["lo"], ci["hi"]), (2, 0.25, 0.25, 0.25))
        ci2 = compare.bootstrap_ci([0.1, -0.1, 0.3, 0.2], n_boot=500, seed=3)
        self.assertEqual(ci2, compare.bootstrap_ci([0.1, -0.1, 0.3, 0.2], n_boot=500, seed=3))  # deterministic
        self.assertLessEqual(ci2["lo"], ci2["mean"])
        self.assertLessEqual(ci2["mean"], ci2["hi"])
        self.assertEqual(compare.bootstrap_ci([])["n"], 0)

    def test_aa_noise_floor(self):
        floor = compare.aa_noise_floor(self._cells(), "baseline")
        # seed 1: |0.5-0.5| = 0; seed 2: |0.25-0.75| = 0.5 -> mean 0.25
        self.assertEqual(floor["n_targets"], 2)
        self.assertAlmostEqual(floor["floor"], 0.25)

    def test_union_gain_measures_complementarity_against_independence(self):
        u = compare.union_gain(self._cells(), "baseline", "structural")
        self.assertEqual(u["n_pairs"], 4)
        pair = next(p for p in u["pairs"] if p["seed"] == 1 and p["repeat"] == 1)
        # baseline {a,b}, structural {b,c,d} -> union 4/4, best single 0.75, complementarity 0.25
        self.assertEqual((pair["union_recall"], pair["complementarity"]), (1.0, 0.25))
        self.assertEqual(pair["only_a"], ["a"])
        self.assertAlmostEqual(pair["expected_independent"], 1 - 0.5 * 0.25)
        self.assertAlmostEqual(pair["jaccard"], 1 / 4)

    def test_validity_and_cost(self):
        v = compare.validity_summary(self._cells(), "structural")
        self.assertEqual((v["cells"], v["graded"], v["ungraded"]), (5, 4, 1))
        self.assertAlmostEqual(v["ungraded_rate"], 0.2)
        cost = compare.cost_summary(self._cells(), "baseline")
        self.assertEqual(cost["n"], 4)
        self.assertAlmostEqual(cost["mean_tokens_per_tp"], (1000 + 1000 + 2000 + 2000 / 3) / 4)

    def test_gate_passes_a_clear_win_and_fails_a_regression(self):
        # The crashed structural cell would (rightly) fail criterion 4; drop it here.
        cells = [c for c in self._cells() if c["graded"]]
        # Make the baseline's A/A floor small so criterion 2 is decided by the delta.
        for c in cells:
            if c["arm"] == "baseline" and c["seed"] == 2:
                c["recall"] = 0.5
        gate = compare.evaluate_gate(cells, "baseline", "structural", n_boot=300)
        self.assertEqual(gate["verdict"], "PASS", json.dumps(gate["criteria"], indent=1, default=str))
        self.assertEqual(gate["criteria"]["1_recall_ci_above_zero_or_efficiency_win"]["via"], "recall")
        self.assertIn("holdout confirmation", gate["caveat"].lower())

        worse = compare.evaluate_gate(cells, "structural", "baseline", n_boot=300)
        self.assertEqual(worse["verdict"], "FAIL")
        self.assertFalse(worse["criteria"]["1_recall_ci_above_zero_or_efficiency_win"]["pass"])
        self.assertFalse(worse["criteria"]["3_worst_target_delta"]["pass"])

    def test_gate_fails_on_a_new_failure_mode_even_with_higher_recall(self):
        gate = compare.evaluate_gate(self._cells(), "baseline", "structural", n_boot=100)
        self.assertEqual(gate["verdict"], "FAIL")
        self.assertFalse(gate["criteria"]["4_no_new_false_positives_or_failures"]["pass"])
        self.assertAlmostEqual(gate["criteria"]["4_no_new_false_positives_or_failures"]["ungraded_rate"]["candidate"], 0.2)

    def test_gate_accepts_a_pure_efficiency_win(self):
        cells = []
        for seed in (1, 2, 3, 4):
            cells.append(_cell("baseline", seed, 0, 0.5, ("a", "b"), tokens=4000))
            cells.append(_cell("baseline", seed, 1, 0.5, ("a", "b"), tokens=4000))
            cells.append(_cell("cheap", seed, 0, 0.5, ("a", "b"), tokens=2000))
            cells.append(_cell("cheap", seed, 1, 0.5, ("a", "b"), tokens=2000))
        gate = compare.evaluate_gate(cells, "baseline", "cheap", n_boot=300)
        c1 = gate["criteria"]["1_recall_ci_above_zero_or_efficiency_win"]
        self.assertEqual(c1["via"], "efficiency")
        self.assertAlmostEqual(c1["tokens_per_tp_improvement_ci"]["mean"], 0.5)
        # Criterion 2 is not met by a zero-delta, so the overall gate still fails: an
        # efficiency win is a reason to look, not a licence to skip the floor.
        self.assertFalse(gate["criteria"]["2_clears_twice_aa_noise_floor"]["pass"])

    def test_report_renders_without_error(self):
        text = compare.render_report(compare.evaluate_gate(self._cells(), "baseline", "structural", n_boot=100))
        self.assertIn("verdict:", text)
        self.assertIn("A/A noise floor", text)
        self.assertIn("union:", text)


if __name__ == "__main__":
    unittest.main()
