"""Unit tests for Harness² disk-backed ContextTree, leak tripwires, and causal reflection."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.database import init_db, record_learning, write_findings  # noqa: E402
from core.harness_context import (  # noqa: E402
    assert_no_grader_verdicts,
    assert_no_leak_paths,
    assert_tree_has_no_grader_verdicts,
    build_context_tree,
    mission,
    parse_reflection_blocks,
    render_rollout_markdown,
)
from core.knowledge import export_to_context_tree  # noqa: E402
from core.memory import recall_for_stage, record_reflection_output  # noqa: E402


class HarnessContextTreeTest(unittest.TestCase):
    """Tests for disk-backed ContextTree, leak tripwires, and causal reflection."""

    def test_build_context_tree_and_mission_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = build_context_tree(
                out_root=tmp,
                task_id="audit_coupon",
                task_description="Audit coupon redemption flow across web and dal.",
                rollouts={
                    "r1": [
                        {
                            "author": "researcher",
                            "type": "tool_call",
                            "name": "find_callers",
                            "args": '{"symbol": "redeem"}',
                        },
                        {
                            "author": "researcher",
                            "type": "tool_response",
                            "name": "find_callers",
                            "response": "2 call sites found",
                        },
                    ]
                },
                deliverables={"r1": "# Finding 1: Double redemption race"},
                harness_files={"workflow.json": '{"name": "test"}'},
                decisions=["# Prior edit\nAdded find_callers to researcher."],
            )
            self.assertEqual(tree.rollouts, ["r1"])
            self.assertEqual(tree.deliverables, ["r1"])
            self.assertEqual(tree.harness_files, 1)
            self.assertEqual(tree.decisions, 1)
            self.assertTrue((tree.root / "rollouts" / "r1.md").is_file())
            self.assertTrue((tree.root / "deliverables" / "r1.md").is_file())

            prompt = mission(
                tree,
                role="You are the reflector.",
                focus="Attribute whether structural navigation tools helped.",
                output="<edits>...</edits>\n<experience>...</experience>",
            )
            self.assertIn("context/rollouts/", prompt)
            self.assertIn("context/deliverables/", prompt)
            self.assertIn("context/harness/", prompt)

    def test_render_rollout_markdown_marks_truncated_observations(self):
        md = render_rollout_markdown(
            [
                {
                    "author": "researcher",
                    "type": "tool_response",
                    "name": "read_file",
                    "response": "A" * 120,
                }
            ],
            label="r1",
            obs_trunc=50,
        )
        self.assertIn("[renderer truncation: 70 chars omitted]", md)

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

    def test_parse_reflection_blocks_and_stage_targeted_recall(self):
        sample = """
<edits>
- `find_callers` on researcher — helped — traced coupon redeem from web/orders_routes.py to dal/repo.py
- repeated full-file `read_file` — hurt — consumed 3 turns without narrowing sink line range
</edits>
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
        parsed = parse_reflection_blocks(sample)
        self.assertEqual(len(parsed["edits"]), 2)
        self.assertEqual(parsed["edits"][0]["verdict"], "helped")
        self.assertEqual(parsed["edits"][1]["verdict"], "hurt")
        self.assertEqual(len(parsed["experiences"]), 2)
        self.assertEqual(parsed["experiences"][0]["target_stage"], "researcher")

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            record_learning(db, "run-0", "general", "Always check auth middleware.")
            counts = record_reflection_output(db, "run-1", sample)
            self.assertEqual(counts, {"edits": 2, "experiences": 2})

            researcher_lessons = recall_for_stage(
                db, "researcher", include_general=False
            )
            self.assertEqual(len(researcher_lessons), 1)
            self.assertIn("find_callers", researcher_lessons[0]["learning"])

            reviewer_lessons = recall_for_stage(
                db, "reviewer", include_general=False
            )
            self.assertEqual(len(reviewer_lessons), 1)
            self.assertIn("get_function_boundary", reviewer_lessons[0]["learning"])

            # Default untargeted <experience> block (target_stage="global") is recalled when include_general=True
            record_reflection_output(
                db,
                "run-2",
                "<experience>\ncue: any endpoint\nlesson: verify auth decorator\nevidence: turn 1\n</experience>",
            )
            with_general = recall_for_stage(
                db, "reviewer", include_general=True
            )
            self.assertTrue(
                any("verify auth decorator" in item["learning"] for item in with_general)
            )

    def test_build_context_tree_rejects_traversal_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_context_tree(
                    out_root=tmp,
                    rollouts={"../escape": "bad"},
                )

    def test_export_to_context_tree_from_knowledge_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            write_findings(
                db,
                "src/app.py",
                [
                    {
                        "title": "SQLi in login",
                        "severity": "HIGH",
                        "status": "confirmed",
                        "description": "Unsanitized user input in query.",
                        "line_numbers": [10],
                        "filepath": "src/app.py",
                    }
                ],
                run_id="run-1",
            )
            record_learning(db, "run-1", "codebase", "Parameterized queries missing in dal.")
            tree = export_to_context_tree(db, Path(tmp) / "ctx", repo_root=tmp)
            self.assertIn("run-1", tree.deliverables)
            self.assertEqual(tree.decisions, 1)

    def test_record_learning_deduplication_and_stage_model_callbacks(self):
        from core.context import RunContext, current_run_context
        from core.database import read_learnings
        from core.graph_loader import (
            _make_reflector_after_model_callback,
            _make_stage_experience_before_model_callback,
        )
        from core.memory import recall, render_stage_learnings_for_agent

        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            ctx = RunContext(jail_dir=tmp, db_path=db, run_id="run-dedup")
            tok = current_run_context.set(ctx)
            try:
                after_model = _make_reflector_after_model_callback("reflector")

                class _Part:
                    text = (
                        "<edits>\n- `find_callers` — helped — found sink\n</edits>\n"
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

                # Calling twice must deduplicate within the same run_id
                after_model(None, _Resp())
                after_model(None, _Resp())
                rows = read_learnings(db, run_id="run-dedup")
                self.assertEqual(len(rows), 2)

                # General recall excludes causal_edit by default
                mem_default = recall(db)
                self.assertTrue(
                    all(item["category"] != "causal_edit" for item in mem_default["learnings"])
                )
                mem_with_edits = recall(db, include_causal_edits=True)
                self.assertTrue(
                    any(item["category"] == "causal_edit" for item in mem_with_edits["learnings"])
                )

                # Stage experience rendering wraps in CP-4 delimiters and injects via before_model_callback
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
