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

"""The generated research targets: deterministic, parseable, leak-free,
holdout by construction, and scored with decoys as certain false positives."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from evals.research_eval import (  # noqa: E402
    DECOY_KIND_PREFIX,
    _copy_target,
    is_decoy,
    load_research_ground_truth,
    score_research_findings,
)
from evals.research_targets import gen  # noqa: E402


class GeneratorDeterminismTest(unittest.TestCase):
    def test_same_seed_same_bytes_different_seed_different_tree(self):
        t1, g1 = gen.generate(1001, "small")
        t2, g2 = gen.generate(1001, "small")
        self.assertEqual(t1, t2)
        self.assertEqual(g1, g2)
        t3, _ = gen.generate(1002, "small")
        self.assertNotEqual(set(t1), set(t3))

    def test_every_generated_python_file_parses(self):
        for seed in (1000, 1037, 1099, 9000, 9050):
            tree, _ = gen.generate(seed, "small")
            for rel, content in tree.items():
                if rel.endswith(".py"):
                    ast.parse(content, filename=rel)

    def test_size_knob_scales_the_tree(self):
        small, _ = gen.generate(1005, "small")
        medium, _ = gen.generate(1005, "medium")
        self.assertGreater(len(medium), 2 * len(small))
        self.assertGreater(
            sum(c.count("\n") for c in medium.values()),
            4 * sum(c.count("\n") for c in small.values()),
        )
        with self.assertRaises(ValueError):
            gen.generate(1005, "huge")


class CorpusHygieneTest(unittest.TestCase):
    """The tree must read like ordinary code; the answers live only in ground_truth.json."""

    def test_no_generated_file_names_a_flaw(self):
        for seed in (1000, 1011, 9003):
            tree, _ = gen.generate(seed, "small")
            self.assertEqual(gen.banned_words_in_tree(tree), [], f"seed {seed}")

    def test_ground_truth_files_exist_and_sinks_resolve(self):
        tree, gt = gen.generate(1003, "small")
        for entry in gt["findings"]:
            for rel in entry["files"]:
                self.assertIn(rel, tree, entry["id"])
            sink_file, _, sink_fn = entry["sink"].partition(":")
            self.assertIn(sink_file, tree)
            self.assertIn(sink_fn.split("_view")[0], tree[sink_file], entry["id"])

    def test_dead_decoys_have_no_callers(self):
        tree, gt = gen.generate(1004, "medium")
        dead = [e for e in gt["findings"] if e["kind"] == "decoy_dead"]
        self.assertTrue(dead)
        for entry in dead:
            module = Path(entry["files"][0]).stem  # legacy_<mod>
            referers = [rel for rel, c in tree.items() if module in c and rel != entry["files"][0]]
            self.assertEqual(referers, [], entry["id"])

    def test_seeded_entries_are_mostly_cross_file(self):
        tree, gt = gen.generate(1006, "small")
        seeded = [e for e in gt["findings"] if e["kind"] == "seeded"]
        cross = [e for e in seeded if e["cross_file"]]
        self.assertGreaterEqual(len(seeded), 8)
        self.assertGreaterEqual(len(cross), len(seeded) // 2)
        for e in seeded:
            self.assertEqual(e["cross_file"], len(e["files"]) > 1)

    def test_written_target_loads_through_the_eval_and_jail_excludes_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "t"
            gt = gen.write_target(target, 1007, "small")
            entries = load_research_ground_truth(target / gen.GROUND_TRUTH_BASENAME)
            self.assertEqual(len(entries), len(gt["findings"]))
            self.assertTrue(any(is_decoy(e) for e in entries))
            jail = Path(tmp) / "jail"
            jail.mkdir()
            app_dir = _copy_target(target, jail)
            self.assertFalse((app_dir / gen.GROUND_TRUTH_BASENAME).exists())
            self.assertTrue((app_dir / "app.py").exists())


class HoldoutByConstructionTest(unittest.TestCase):
    def test_seed_ranges_are_disjoint_and_classified(self):
        self.assertFalse(set(gen.DEV_SEEDS) & set(gen.HOLDOUT_SEEDS))
        self.assertEqual(gen.seed_split(gen.DEV_SEEDS[0]), "dev")
        self.assertEqual(gen.seed_split(gen.HOLDOUT_SEEDS[0]), "holdout")
        self.assertEqual(gen.seed_split(5), "other")

    def test_assert_not_holdout_refuses_holdout_seeds_only(self):
        gen.assert_not_holdout(gen.DEV_SEEDS[3], "proposer context")
        gen.assert_not_holdout(5, "proposer context")
        with self.assertRaises(gen.HoldoutViolation):
            gen.assert_not_holdout(gen.HOLDOUT_SEEDS[3], "proposer context")

    def test_dev_seeds_never_carry_holdout_only_classes_and_holdout_seeds_always_do(self):
        for seed in list(gen.DEV_SEEDS)[:20]:
            classes = {e["class"] for e in gen.generate(seed, "small")[1]["findings"]}
            self.assertFalse(classes & gen.HOLDOUT_ONLY_CLASSES, seed)
        for seed in list(gen.HOLDOUT_SEEDS)[:10]:
            classes = {e["class"] for e in gen.generate(seed, "small")[1]["findings"] if e["kind"] == "seeded"}
            self.assertTrue(classes & gen.HOLDOUT_ONLY_CLASSES, seed)

    def test_holdout_only_skin_is_refused_for_dev_seeds(self):
        for skin in gen.HOLDOUT_ONLY_SKINS:
            with self.assertRaises(gen.HoldoutViolation):
                gen.plan(gen.DEV_SEEDS[0], "small", skin)
        with self.assertRaises(ValueError):
            gen.plan(gen.DEV_SEEDS[0], "small", "not-a-skin")

    def test_cli_refuses_holdout_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                gen.main(["--seed", str(gen.HOLDOUT_SEEDS[0]), "--out", tmp])
            self.assertEqual(gen.main(["--seed", str(gen.HOLDOUT_SEEDS[0]), "--out", tmp, "--allow-holdout"]), 0)
            self.assertEqual(gen.main(["--seed", str(gen.DEV_SEEDS[0]), "--out", tmp, "--size", "small"]), 0)


class DecoyScoringTest(unittest.TestCase):
    def _entries(self):
        return [
            {"id": "s1", "files": ["dal/a.py"], "match_any": ["sql injection"], "cross_file": True, "class": "sql_injection", "cwe": "CWE-89", "kind": "seeded"},
            {"id": "s2", "files": ["web/b.py"], "match_any": ["open redirect"], "cross_file": False, "class": "open_redirect", "cwe": "CWE-601", "kind": "seeded"},
            {"id": "d1", "files": ["dal/c.py"], "match_any": ["sql injection"], "cross_file": False, "class": "sql_injection", "cwe": "CWE-89", "kind": "decoy_sanitized"},
            {"id": "d2", "files": ["tests/test_d.py"], "match_any": ["sql injection"], "cross_file": False, "class": "sql_injection", "cwe": "CWE-89", "kind": "decoy_vendor"},
        ]

    def test_decoy_hits_are_false_positives_not_detections(self):
        r = score_research_findings(
            [
                {"filepath": "dal/a.py", "title": "SQL injection in query"},
                {"filepath": "dal/c.py", "title": "SQL injection in query"},
                {"filepath": "web/zzz.py", "title": "something unmatched"},
            ],
            self._entries(),
        )
        self.assertEqual(r["ground_truth_total"], 2)      # seeded only
        self.assertEqual(r["detected"], 1)
        self.assertEqual(r["recall"], 0.5)
        self.assertEqual((r["decoys_total"], r["decoys_hit"]), (2, 1))
        self.assertEqual(r["decoy_fp_rate"], 0.5)
        self.assertEqual(r["findings_true"], 1)
        self.assertEqual(r["findings_decoy"], 1)
        self.assertAlmostEqual(r["precision"], 1 / 3, places=3)          # strict: decoy + unmatched both wrong
        self.assertAlmostEqual(r["precision_lenient"], 1 / 2, places=3)  # unmatched excluded
        kinds = {e["id"]: e["kind"] for e in r["per_entry"]}
        self.assertTrue(kinds["d1"].startswith(DECOY_KIND_PREFIX))

    def test_entries_without_kind_are_seeded(self):
        legacy = [{"id": "x", "files": ["a.py"], "match_any": ["p"], "cross_file": False, "class": "", "cwe": ""}]
        r = score_research_findings([{"filepath": "a.py", "title": "p"}], legacy)
        self.assertEqual((r["ground_truth_total"], r["detected"], r["decoys_total"]), (1, 1, 0))
        self.assertEqual(r["precision"], 1.0)

    def test_committed_corpus_has_no_decoys_yet(self):
        entries = load_research_ground_truth(_REF_ROOT / "evals" / "research_target" / "ground_truth.json")
        self.assertFalse(any(is_decoy(e) for e in entries))
        self.assertEqual(json.loads((_REF_ROOT / "evals" / "research_target" / "ground_truth.json").read_text())["format_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
