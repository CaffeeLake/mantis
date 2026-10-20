"""Unit tests for leak tripwires, `<experience>` parsing, and stage-targeted recall."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.database import init_db, read_learnings, record_learning  # noqa: E402
from core.harness_context import (  # noqa: E402
    assert_no_grader_verdicts,
    assert_no_leak_paths,
    assert_tree_has_no_grader_verdicts,
    parse_reflection_blocks,
)
from core.memory import (  # noqa: E402
    recall,
    recall_for_stage,
    record_reflection_output,
    render_stage_learnings_for_agent,
)


class LeakTripwireTest(unittest.TestCase):
    def test_leak_and_grader_verdict_tripwires_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ground_truth.json").write_text('{"findings": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_no_leak_paths(root)
            (root / "ground_truth.json").unlink()
            assert_no_leak_paths(root)

            (root / "trace.json").write_text('{"cross_file_recall": 0.75}', encoding="utf-8")
            with self.assertRaises(ValueError):
                assert_no_grader_verdicts(root)

        with self.assertRaises(ValueError):
            assert_tree_has_no_grader_verdicts({"eval.json": '{"ground_truth_total": 5}'})

    def test_symlink_in_context_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "real.txt").write_text("ok", encoding="utf-8")
            (root / "link.txt").symlink_to(root / "real.txt")
            with self.assertRaises(ValueError):
                assert_no_leak_paths(root)
            with self.assertRaises(ValueError):
                assert_no_grader_verdicts(root)


class ReflectionParsingTest(unittest.TestCase):
    SAMPLE = """
<experience>
[
  {
    "cue": "auditing cross-file SQL or coupon state transitions",
    "lesson": "call find_callers on DAO mutation methods before reading route handlers",
    "evidence": "turn 2 find_callers('redeem_coupon') immediately isolated the unguarded caller",
    "confidence": "high",
    "target_stage": "researcher"
  },
  {
    "cue": "triaging multi-file findings in reviewer",
    "lesson": "use get_function_boundary on both caller and callee before dismissing",
    "evidence": "saved 2 false dismissals on cross-module flows",
    "confidence": "medium",
    "target_stage": "reviewer"
  }
]
</experience>
"""

    def test_parses_json_and_key_value_blocks(self):
        parsed = parse_reflection_blocks(self.SAMPLE)
        self.assertEqual(set(parsed), {"experiences"})
        self.assertEqual(len(parsed["experiences"]), 2)
        self.assertEqual(parsed["experiences"][0]["target_stage"], "researcher")

        kv = parse_reflection_blocks(
            "<experience>\ncue: any endpoint\nlesson: verify auth decorator\nevidence: turn 1\n</experience>"
        )
        self.assertEqual(len(kv["experiences"]), 1)
        self.assertEqual(kv["experiences"][0]["confidence"], "medium")
        self.assertEqual(kv["experiences"][0]["target_stage"], "global")

    def test_lessons_without_evidence_are_dropped(self):
        """An invented attribution poisons memory for every later run; no evidence, no entry."""
        no_evidence = (
            '<experience>[{"cue": "x", "lesson": "y", "target_stage": "researcher"}]</experience>'
        )
        self.assertEqual(parse_reflection_blocks(no_evidence)["experiences"], [])
        self.assertEqual(parse_reflection_blocks("")["experiences"], [])
        self.assertEqual(parse_reflection_blocks("no blocks here")["experiences"], [])

    def test_edits_blocks_are_ignored(self):
        """Single-rollout runs have no edits to attribute; the block is not parsed or stored."""
        text = (
            "<edits>\n- `find_callers` — helped — found sink\n</edits>\n"
            "<experience>\ncue: c\nlesson: l\nevidence: e\n</experience>"
        )
        parsed = parse_reflection_blocks(text)
        self.assertNotIn("edits", parsed)
        self.assertEqual(len(parsed["experiences"]), 1)

    def test_record_and_stage_targeted_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            record_learning(db, "run-0", "general", "Always check auth middleware.")
            counts = record_reflection_output(db, "run-1", self.SAMPLE)
            self.assertEqual(counts, {"experiences": 2})

            researcher_lessons = recall_for_stage(db, "researcher", include_general=False)
            self.assertEqual(len(researcher_lessons), 1)
            self.assertIn("find_callers", researcher_lessons[0]["learning"])
            self.assertIn("evidence:", researcher_lessons[0]["learning"])

            reviewer_lessons = recall_for_stage(db, "reviewer", include_general=False)
            self.assertEqual(len(reviewer_lessons), 1)
            self.assertIn("get_function_boundary", reviewer_lessons[0]["learning"])

            # Untargeted lessons (target_stage=global) are recalled with include_general=True.
            record_reflection_output(
                db,
                "run-2",
                "<experience>\ncue: any endpoint\nlesson: verify auth decorator\nevidence: turn 1\n</experience>",
            )
            with_general = recall_for_stage(db, "reviewer", include_general=True)
            self.assertTrue(any("verify auth decorator" in item["learning"] for item in with_general))

    def test_legacy_causal_edit_rows_never_reach_general_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            record_learning(db, "run-old", "causal_edit", "[EDIT:helped] find_callers — found sink")
            record_learning(db, "run-old", "codebase", "Parameterized queries missing in dal.")
            mem = recall(db)
            categories = {item["category"] for item in mem["learnings"]}
            self.assertNotIn("causal_edit", categories)
            self.assertIn("codebase", categories)


class StageModelCallbackTest(unittest.TestCase):
    def test_record_learning_deduplication_and_stage_model_callbacks(self):
        from core.context import RunContext, current_run_context
        from core.graph_loader import (
            _make_reflector_after_model_callback,
            _make_stage_experience_before_model_callback,
        )

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            ctx = RunContext(jail_dir=tmp, db_path=db, run_id="run-dedup")
            tok = current_run_context.set(ctx)
            try:
                after_model = _make_reflector_after_model_callback("reflector")

                class _Part:
                    text = (
                        "<experience>\n"
                        "cue: checking SQL sinks\n"
                        "lesson: run get_function_boundary before dismissing\n"
                        "evidence: turn 3\n"
                        "target_stage: reviewer\n"
                        "</experience>"
                    )
                    thought = False

                class _Content:
                    parts = [_Part()]

                class _Resp:
                    content = _Content()

                # Calling twice must deduplicate within the same run_id.
                after_model(None, _Resp())
                after_model(None, _Resp())
                rows = read_learnings(db, run_id="run-dedup")
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["category"], "experience:reviewer")

                rendered = render_stage_learnings_for_agent(db, "reviewer")
                self.assertIn("[STAGE EXPERIENCE - REVIEWER]", rendered)
                self.assertIn("get_function_boundary", rendered)
                self.assertEqual(render_stage_learnings_for_agent(db, "patcher"), "")

                class _Cfg:
                    system_instruction = "Base reviewer prompt."

                class _Req:
                    config = _Cfg()

                    def append_instructions(self, items):
                        self.config.system_instruction += "\n\n" + "\n\n".join(items)

                req = _Req()
                before_model = _make_stage_experience_before_model_callback("reviewer")
                with unittest.mock.patch(
                    "core.memory.render_stage_learnings_for_agent",
                    wraps=render_stage_learnings_for_agent,
                ) as spy_render:
                    before_model(None, req)
                    self.assertIn("[STAGE EXPERIENCE - REVIEWER]", req.config.system_instruction)
                    # Second turn with a fresh LlmRequest uses the closure cache
                    # instead of re-querying SQLite.
                    req2 = _Req()
                    req2.config = _Cfg()
                    before_model(None, req2)
                    self.assertIn("[STAGE EXPERIENCE - REVIEWER]", req2.config.system_instruction)
                    self.assertEqual(spy_render.call_count, 1)
            finally:
                current_run_context.reset(tok)


if __name__ == "__main__":
    unittest.main()
