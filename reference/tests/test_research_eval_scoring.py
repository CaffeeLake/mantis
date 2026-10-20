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

"""Tests for the researcher-stage eval (evals/research_eval.py).

Covers the deterministic layer only: the finding matcher, the scorer, the
ground-truth contract against the committed corpus, the jail-copy exclusion,
and token extraction. No LLM calls.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from evals.research_eval import (
    ABORT_GUARD_MIN_TOOL_CALLS,
    ABORT_GUARD_PROMPT,
    DEFAULT_TARGET,
    GROUND_TRUTH_BASENAME,
    STRUCTURAL_NAV_TOOLS,
    TOOLSETS,
    TRANSCRIPT_PAYLOAD_LIMIT,
    ToolTelemetry,
    _clip,
    _copy_target,
    cap_exceeded,
    config_fingerprint,
    count_event_tool_calls,
    event_has_pseudo_tool_call,
    event_transcript_rows,
    extract_event_tokens,
    ground_truth_version,
    load_research_ground_truth,
    match_finding,
    read_is_windowed,
    score_research_findings,
    should_rearm_run,
    summarize_research_runs,
)


def _entry(id_, files, phrases, cross=False):
    return {
        "id": id_,
        "files": files,
        "match_any": [p.lower() for p in phrases],
        "cross_file": cross,
        "class": "",
        "cwe": "",
    }


def _finding(filepath="", title="", description="", code_paths=None):
    return {
        "filepath": filepath,
        "title": title,
        "description": description,
        "code_paths": code_paths or [],
    }


class MatchFindingTest(unittest.TestCase):
    GT = _entry("sqli", ["database.py", "app.py"], ["sql injection", "cwe-89"])

    def test_path_suffix_variants_match(self):
        for path in ("database.py", "app/database.py", "./app/database.py"):
            self.assertTrue(
                match_finding(_finding(path, "SQL injection in find_user"), self.GT),
                path,
            )

    def test_basename_collision_does_not_match(self):
        self.assertFalse(
            match_finding(_finding("app/xdatabase.py", "SQL injection"), self.GT)
        )

    def test_file_match_without_phrase_is_rejected(self):
        # Right file, wrong mechanism: naming the file is not naming the bug.
        self.assertFalse(
            match_finding(_finding("app/database.py", "Debug mode enabled"), self.GT)
        )

    def test_phrase_may_live_in_description(self):
        self.assertTrue(
            match_finding(
                _finding("app/database.py", "Injection flaw", "classic CWE-89 shape"),
                self.GT,
            )
        )

    def test_structured_cwe_field_can_carry_the_phrase_match(self):
        finding = _finding("app/database.py", "Injection in find_user")
        finding["cwe"] = "CWE-89"
        self.assertTrue(match_finding(finding, self.GT))

    def test_code_paths_can_carry_the_file_match(self):
        finding = _finding(
            "", "SQL injection", code_paths=["app/database.py:28"]
        )
        self.assertTrue(match_finding(finding, self.GT))


class ScoreResearchFindingsTest(unittest.TestCase):
    def setUp(self):
        self.gt = [
            _entry("sqli", ["database.py", "app.py"], ["sql injection"], cross=True),
            _entry("md5", ["auth.py"], ["md5", "weak hash"], cross=False),
            _entry("ssrf", ["fetcher.py", "app.py"], ["ssrf"], cross=True),
        ]

    def test_mixed_outcome_arithmetic(self):
        findings = [
            _finding("app/database.py", "SQL injection in find_user"),
            _finding("app/auth.py", "Weak password hashing", "unsalted MD5 digest"),
            _finding("database.py", "sql injection via name parameter"),  # duplicate
            _finding("app/helpers.py", "Hypothetical issue"),  # false positive
        ]
        result = score_research_findings(findings, self.gt)
        self.assertEqual(result["detected"], 2)
        self.assertAlmostEqual(result["recall"], 2 / 3, places=4)
        self.assertEqual(result["cross_file_total"], 2)
        self.assertEqual(result["cross_file_detected"], 1)
        self.assertAlmostEqual(result["cross_file_recall"], 0.5, places=4)
        self.assertEqual(result["findings_reported"], 4)
        self.assertEqual(result["findings_true"], 3)
        # Unique results: 2 detected + 1 unmatched; the duplicate counts for
        # neither, so it cannot dilute the false positive.
        self.assertAlmostEqual(result["precision"], 2 / 3, places=4)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(len(result["unmatched_findings"]), 1)
        by_id = {e["id"]: e for e in result["per_entry"]}
        self.assertTrue(by_id["sqli"]["detected"])
        self.assertFalse(by_id["ssrf"]["detected"])

    def test_no_findings_scores_zero_not_crash(self):
        result = score_research_findings([], self.gt)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["findings_true"], 0)


class GroundTruthContractTest(unittest.TestCase):
    """The committed corpus and its answer key must stay in lockstep."""

    @classmethod
    def setUpClass(cls):
        cls.entries = load_research_ground_truth(
            DEFAULT_TARGET / GROUND_TRUTH_BASENAME
        )

    def test_every_cited_file_exists_in_the_corpus(self):
        for entry in self.entries:
            for file in entry["files"]:
                self.assertTrue(
                    (DEFAULT_TARGET / file).is_file(), f"{entry['id']}: {file}"
                )

    def test_cross_file_entries_dominate(self):
        # Most entries must stay cross-file: cross-file reasoning is the
        # capability this corpus exists to measure.
        cross = [e for e in self.entries if e["cross_file"]]
        self.assertGreaterEqual(len(cross), 4, [e["id"] for e in self.entries])
        for entry in cross:
            self.assertGreaterEqual(
                len(entry["files"]), 2,
                f"{entry['id']}: cross_file entries must allow source or sink file",
            )

    def test_match_phrases_are_lowercase_nonempty(self):
        for entry in self.entries:
            self.assertTrue(entry["match_any"], entry["id"])
            for phrase in entry["match_any"]:
                self.assertEqual(phrase, phrase.lower(), entry["id"])


class CorpusLeakTest(unittest.TestCase):
    """Corpus files must read like production code, not an answer key."""

    FORBIDDEN = (
        "seeded",
        "eval target",
        "benign counterpart",
        "vulnerab",
        "attacker",
        "exploit",
        "injection",
        "traversal",
        "ssrf",
    )

    def test_corpus_files_do_not_leak_ground_truth(self):
        corpus_files = sorted(DEFAULT_TARGET.rglob("*.py"))
        self.assertGreater(len(corpus_files), 20, "corpus walk looks truncated")
        for path in corpus_files:
            text = path.read_text(encoding="utf-8").lower()
            for marker in self.FORBIDDEN:
                self.assertNotIn(
                    marker, text, f"{path.relative_to(DEFAULT_TARGET)}: {marker!r}"
                )


class JailCopyTest(unittest.TestCase):
    def test_ground_truth_is_excluded_from_the_jail(self):
        with tempfile.TemporaryDirectory(prefix="mantis_jail_test_") as tmp:
            app_dir = _copy_target(DEFAULT_TARGET, Path(tmp))
            copied = sorted(p.name for p in app_dir.iterdir())
            self.assertIn("app.py", copied)
            self.assertIn("database.py", copied)
            self.assertTrue((app_dir / "dal" / "repo.py").is_file())
            self.assertTrue((app_dir / "web" / "orders_routes.py").is_file())
            self.assertNotIn(GROUND_TRUTH_BASENAME, copied)


class ExtractEventTokensTest(unittest.TestCase):
    def test_total_token_count_preferred(self):
        event = types.SimpleNamespace(
            usage_metadata=types.SimpleNamespace(
                total_token_count=321, prompt_token_count=300, candidates_token_count=21
            )
        )
        self.assertEqual(extract_event_tokens(event), 321)

    def test_falls_back_to_prompt_plus_candidates(self):
        event = types.SimpleNamespace(
            usage_metadata=types.SimpleNamespace(
                total_token_count=None, prompt_token_count=10, candidates_token_count=5
            )
        )
        self.assertEqual(extract_event_tokens(event), 15)

    def test_absent_or_malformed_usage_counts_zero(self):
        self.assertEqual(extract_event_tokens(types.SimpleNamespace()), 0)
        self.assertEqual(
            extract_event_tokens(
                types.SimpleNamespace(
                    usage_metadata=types.SimpleNamespace(
                        total_token_count="not-a-number",
                        prompt_token_count=None,
                        candidates_token_count=None,
                    )
                )
            ),
            0,
        )


class ToolsetContractTest(unittest.TestCase):
    def test_structural_is_a_strict_superset_of_baseline(self):
        self.assertTrue(set(TOOLSETS["baseline"]) < set(TOOLSETS["structural"]))

    def test_resolve_toolset_against_live_registry(self):
        # Requires the production tool registry (and its ADK dependency); the
        # contract still holds without it, so skip rather than fail.
        try:
            from evals.research_eval import resolve_toolset
            resolved = resolve_toolset("baseline")
        except ImportError as exc:  # pragma: no cover - minimal environments
            self.skipTest(f"tool registry unavailable: {exc}")
        self.assertEqual(resolved, TOOLSETS["baseline"])
        with self.assertRaises(ValueError):
            resolve_toolset("nonexistent")

    def test_missing_tool_fails_the_arm_instead_of_degrading_it(self):
        try:
            import evals.research_eval as re_mod
            from evals.research_eval import resolve_toolset
        except ImportError as exc:  # pragma: no cover - minimal environments
            self.skipTest(f"tool registry unavailable: {exc}")
        saved = dict(re_mod.TOOLSETS)
        re_mod.TOOLSETS["degraded"] = TOOLSETS["baseline"] + ["tool_that_does_not_exist"]
        try:
            with self.assertRaises(ValueError) as cm:
                resolve_toolset("degraded")
            self.assertIn("tool_that_does_not_exist", str(cm.exception))
        finally:
            re_mod.TOOLSETS.clear()
            re_mod.TOOLSETS.update(saved)


class AbortGuardTest(unittest.TestCase):
    """The silent-abort re-arm: once, and only for runs that did nothing."""

    def test_rearms_a_toolless_zero_finding_run(self):
        self.assertTrue(should_rearm_run(0, 0, 0))
        self.assertTrue(should_rearm_run(0, ABORT_GUARD_MIN_TOOL_CALLS - 1, 0))

    def test_runs_that_reported_or_worked_score_as_they_stand(self):
        self.assertFalse(should_rearm_run(1, 0, 0))
        self.assertFalse(should_rearm_run(0, ABORT_GUARD_MIN_TOOL_CALLS, 0))

    def test_never_rearms_twice(self):
        self.assertFalse(should_rearm_run(0, 0, 1))

    def test_nudge_demands_tools_not_prose(self):
        for needle in ("list_files", "report_findings", "prose"):
            self.assertIn(needle, ABORT_GUARD_PROMPT)

    def test_tool_call_counter_never_raises(self):
        call = types.SimpleNamespace(function_call=object(), text=None)
        text = types.SimpleNamespace(function_call=None, text="plan...")
        event = types.SimpleNamespace(
            content=types.SimpleNamespace(parts=[call, text, call])
        )
        self.assertEqual(count_event_tool_calls(event), 2)
        self.assertEqual(count_event_tool_calls(types.SimpleNamespace()), 0)
        self.assertEqual(count_event_tool_calls(None), 0)

    def test_pseudo_call_rearms_even_past_the_tool_count_threshold(self):
        """A run that wrote its tool call as text ended mid-audit by accident;
        no tool-count threshold can tell that from a deliberate stop."""
        self.assertTrue(should_rearm_run(0, ABORT_GUARD_MIN_TOOL_CALLS, 0, True))
        # Reported findings or a spent re-arm still take precedence.
        self.assertFalse(should_rearm_run(1, ABORT_GUARD_MIN_TOOL_CALLS, 0, True))
        self.assertFalse(should_rearm_run(0, ABORT_GUARD_MIN_TOOL_CALLS, 1, True))

    def test_pseudo_call_detector_matches_crash_shape_not_prose(self):
        def _text_event(text):
            return types.SimpleNamespace(
                content=types.SimpleNamespace(
                    parts=[types.SimpleNamespace(text=text, function_call=None)]
                )
            )

        crash = _text_event('call:default_api:read_file{"filepath": "app/x.py"}')
        self.assertTrue(event_has_pseudo_tool_call(crash))
        prose = _text_event("Next I will call read_file on app/x.py.")
        self.assertFalse(event_has_pseudo_tool_call(prose))
        self.assertFalse(event_has_pseudo_tool_call(types.SimpleNamespace()))
        self.assertFalse(event_has_pseudo_tool_call(None))

    def test_nudge_names_the_function_calling_interface(self):
        self.assertIn("function-calling", ABORT_GUARD_PROMPT)


class TranscriptRowsTest(unittest.TestCase):
    """The JSONL trace a human reads to follow a run hop by hop."""

    def test_rows_keep_order_and_kinds(self):
        event = types.SimpleNamespace(
            author="researcher",
            content=types.SimpleNamespace(parts=[
                types.SimpleNamespace(text="tracing the coupon flow",
                                      function_call=None, function_response=None),
                types.SimpleNamespace(text=None,
                                      function_call=types.SimpleNamespace(
                                          name="find_callers",
                                          args={"symbol": "redeem"}),
                                      function_response=None),
                types.SimpleNamespace(text=None, function_call=None,
                                      function_response=types.SimpleNamespace(
                                          name="find_callers",
                                          response={"result": "2 call site(s)"})),
            ]),
        )
        rows = event_transcript_rows(event)
        self.assertEqual([r["type"] for r in rows],
                         ["text", "tool_call", "tool_response"])
        self.assertEqual(rows[1]["name"], "find_callers")
        self.assertIn("redeem", rows[1]["args"])
        self.assertIn("call site", rows[2]["response"])

    def test_clip_marks_what_it_cut(self):
        clipped = _clip("x" * (TRANSCRIPT_PAYLOAD_LIMIT + 7), TRANSCRIPT_PAYLOAD_LIMIT)
        self.assertIn("[clipped 7 chars]", clipped)
        self.assertEqual(_clip("short", TRANSCRIPT_PAYLOAD_LIMIT), "short")

    def test_malformed_event_yields_no_rows(self):
        self.assertEqual(event_transcript_rows(None), [])
        self.assertEqual(event_transcript_rows(types.SimpleNamespace()), [])


def _call_event(name, args=None):
    part = types.SimpleNamespace(
        text=None,
        function_call=types.SimpleNamespace(name=name, args=args or {}),
        function_response=None,
    )
    return types.SimpleNamespace(content=types.SimpleNamespace(parts=[part]))


def _response_event(name, response):
    part = types.SimpleNamespace(
        text=None,
        function_call=None,
        function_response=types.SimpleNamespace(name=name, response=response),
    )
    return types.SimpleNamespace(content=types.SimpleNamespace(parts=[part]))


class ToolTelemetryTest(unittest.TestCase):
    """How a run read is measured from the same events the score is."""

    def test_windowed_rule_matches_the_production_guard(self):
        self.assertFalse(read_is_windowed({"filepath": "a.py"}))
        self.assertFalse(read_is_windowed({"filepath": "a.py", "start_line": 0, "end_line": 0}))
        self.assertTrue(read_is_windowed({"filepath": "a.py", "start_line": 10, "end_line": 40}))
        self.assertTrue(read_is_windowed({"filepath": "a.py", "end_line": 40}))
        self.assertFalse(read_is_windowed(None))

    def test_counts_reads_navigation_hints_and_report_marks(self):
        t = ToolTelemetry()
        t.observe(_call_event("list_files", {"directory": "app"}), 1_000)
        t.observe(_call_event("read_file", {"filepath": "app/a.py"}), 2_000)
        t.observe(_call_event("read_file", {"filepath": "app/a.py", "start_line": 0, "end_line": 0}), 3_000)
        t.observe(_call_event("read_file", {"filepath": "app/b.py", "start_line": 5, "end_line": 50}), 4_000)
        t.observe(_call_event("find_callers", {"name": "redeem"}), 5_000)
        t.observe(_call_event("get_function_boundary", {"filepath": "app/a.py", "line": 12}), 6_000)
        t.observe(_response_event("read_file", "src...\n\n[HARNESS STEERING HINT: ...]"), 6_000)
        t.observe(_response_event("read_file", {"result": "plain"}), 6_000)
        t.observe(_call_event("report_findings", {"findings": []}), 7_500)
        t.observe(_call_event("report_findings", {"findings": []}), 9_000)
        d = t.as_dict()
        self.assertEqual(d["unwindowed_reads"], 2)
        self.assertEqual(d["windowed_reads"], 1)
        self.assertEqual(d["structural_nav_calls"], 2)
        self.assertEqual(d["list_files_calls"], 1)
        self.assertEqual(d["report_findings_calls"], 2)
        self.assertEqual(d["steering_hints_fired"], 1)
        self.assertEqual(d["unique_files_read"], 2)
        self.assertEqual(d["repeat_unwindowed_files"], 1)
        self.assertEqual(d["report_token_marks"], [7_500, 9_000])
        self.assertEqual(d["tool_counts"]["read_file"], 3)
        self.assertTrue(STRUCTURAL_NAV_TOOLS <= set(TOOLSETS["structural"]))

    def test_malformed_events_are_ignored(self):
        t = ToolTelemetry()
        t.observe(None, 0)
        t.observe(types.SimpleNamespace(), 0)
        t.observe(types.SimpleNamespace(content=types.SimpleNamespace(parts=[object()])), 0)
        self.assertEqual(t.as_dict()["tool_counts"], {})


class CapAndIdentityTest(unittest.TestCase):
    def test_caps_are_inclusive_and_none_means_uncapped(self):
        self.assertFalse(cap_exceeded(10**9, 10**6))
        self.assertFalse(cap_exceeded(999, 5, max_tokens=1000))
        self.assertTrue(cap_exceeded(1000, 5, max_tokens=1000))
        self.assertTrue(cap_exceeded(5, 20, max_tool_calls=20))
        self.assertFalse(cap_exceeded(5, 19, max_tokens=0, max_tool_calls=20))

    def test_fingerprint_changes_with_prompt_tools_model_and_answer_key(self):
        base = config_fingerprint("m", "low", ["read_file", "list_files"], "prompt", "gt1")
        self.assertEqual(len(base), 16)
        self.assertEqual(base, config_fingerprint("m", "low", ["list_files", "read_file"], "prompt", "gt1"))
        for changed in (
            config_fingerprint("m2", "low", ["read_file", "list_files"], "prompt", "gt1"),
            config_fingerprint("m", "high", ["read_file", "list_files"], "prompt", "gt1"),
            config_fingerprint("m", "low", ["read_file"], "prompt", "gt1"),
            config_fingerprint("m", "low", ["read_file", "list_files"], "prompt v2", "gt1"),
            config_fingerprint("m", "low", ["read_file", "list_files"], "prompt", "gt2"),
        ):
            self.assertNotEqual(base, changed)

    def test_ground_truth_version_tracks_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ground_truth.json"
            p.write_text('{"findings": [1]}', encoding="utf-8")
            v1 = ground_truth_version(p)
            self.assertEqual(v1, ground_truth_version(p))
            p.write_text('{"findings": [1, 2]}', encoding="utf-8")
            self.assertNotEqual(v1, ground_truth_version(p))
        self.assertEqual(len(ground_truth_version(DEFAULT_TARGET / GROUND_TRUTH_BASENAME)), 12)


class SummarizeRunsTest(unittest.TestCase):
    """Crashed runs are missing measurements, not low scores."""

    def test_ungraded_runs_leave_every_mean(self):
        runs = [
            {"graded": True, "recall": 0.5, "cross_file_recall": 0.4, "precision": 1.0,
             "total_tokens": 100, "tokens_per_detected": 10, "zero_findings_complete": False,
             "telemetry": {"unwindowed_reads": 4, "structural_nav_calls": 2},
             "ground_truth_version": "gt1", "config_fingerprint": "cfg1"},
            {"graded": False, "status": "ungraded", "recall": 0.0, "cross_file_recall": 0.0,
             "precision": 0.0, "total_tokens": 5, "run_errors": ["boom"],
             "ground_truth_version": "gt1", "config_fingerprint": "cfg1"},
            {"graded": True, "recall": 0.0, "cross_file_recall": 0.0, "precision": 0.0,
             "total_tokens": 50, "tokens_per_detected": None, "zero_findings_complete": True,
             "cap_hit": True, "abort_guard_rearms": 1,
             "telemetry": {"unwindowed_reads": 0, "structural_nav_calls": 0},
             "ground_truth_version": "gt1", "config_fingerprint": "cfg1"},
        ]
        agg = summarize_research_runs(runs)
        self.assertEqual((agg["runs"], agg["graded"], agg["ungraded"]), (3, 2, 1))
        self.assertAlmostEqual(agg["avg_recall"], 0.25)  # (0.5 + 0.0) / 2 -- the crash is not a zero
        self.assertEqual(agg["worst_recall"], 0.0)       # the real zero is
        self.assertAlmostEqual(agg["avg_tokens"], 75.0)
        self.assertEqual(agg["zero_finding_runs"], 1)
        self.assertEqual(agg["cap_hits"], 1)
        self.assertEqual(agg["rearms"], 1)
        self.assertAlmostEqual(agg["avg_unwindowed_reads"], 2.0)
        self.assertEqual(agg["ground_truth_versions"], ["gt1"])

    def test_legacy_rows_without_graded_flag_count_as_graded(self):
        agg = summarize_research_runs([{"recall": 1.0}, {"recall": 0.0}])
        self.assertEqual(agg["graded"], 2)
        self.assertAlmostEqual(agg["avg_recall"], 0.5)

    def test_mixed_versions_are_reported_not_hidden(self):
        agg = summarize_research_runs([
            {"recall": 1.0, "ground_truth_version": "a", "config_fingerprint": "x"},
            {"recall": 1.0, "ground_truth_version": "b", "config_fingerprint": "x"},
        ])
        self.assertEqual(agg["ground_truth_versions"], ["a", "b"])

    def test_empty_input(self):
        agg = summarize_research_runs([])
        self.assertEqual((agg["runs"], agg["graded"], agg["avg_recall"]), (0, 0, 0.0))


if __name__ == "__main__":
    unittest.main()
