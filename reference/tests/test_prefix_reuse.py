"""Tests for cross-campaign reuse of repo-scope prefix stage artifacts.

In cross-functional mode every campaign shares the repository jail, so the
architect and threat modeler re-derive the same repo-wide artifacts on every
campaign of a run. The reuse callbacks let campaigns 2..N of one process skip
those stages once the artifact row exists. These tests pin the fail-safe
gates: any missing precondition must run the stage (callback returns None).
No LLM calls.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from core.context import RunContext, current_run_context
from core.database import init_db, record_artifact
from core.graph_loader import (
    _PREFIX_REUSE_ARTIFACTS,
    _SCAN_MODE_CROSS_FUNCTIONAL,
    _make_prefix_reuse_callbacks,
)


class _CallbackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mantis_prefix_reuse_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = os.path.realpath(self._tmp.name)
        self.db = os.path.join(self.tmp, "knowledge.db")
        init_db(self.db)

    def install_ctx(self, scan_mode=_SCAN_MODE_CROSS_FUNCTIONAL, run_id="r1"):
        ctx = RunContext(
            jail_dir=self.tmp,
            db_path=self.db,
            run_id=run_id,
            scan_mode=scan_mode,
        )
        token = current_run_context.set(ctx)
        self.addCleanup(current_run_context.reset, token)
        return ctx

    def record_summary(self, run_id="r1"):
        record_artifact(
            self.db, run_id, "summary",
            "workspace/.structured/summary.json", json.dumps({"ok": True}),
        )


class PrefixReuseCallbackTest(_CallbackTest):
    def test_skips_only_after_completion_and_artifact(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx()
        self.record_summary()
        # Campaign #1: nothing completed yet, the stage must run.
        self.assertIsNone(before())
        after()
        # Campaign #2: completed and the artifact row exists -> skip.
        content = before()
        self.assertIsNotNone(content)
        self.assertIn("Reused the summary", content.parts[0].text)

    def test_file_by_file_never_skips_or_marks(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx(scan_mode="file-by-file")
        self.record_summary()
        after()
        self.assertIsNone(before())
        # The file-by-file completion must not seed a later cross-functional
        # skip either: per-file campaigns never prove the repo-wide artifact.
        self.install_ctx(scan_mode=_SCAN_MODE_CROSS_FUNCTIONAL)
        self.assertIsNone(before())

    def test_unknown_mode_never_skips(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx(scan_mode="")
        self.record_summary()
        after()
        self.assertIsNone(before())

    def test_missing_artifact_runs_the_stage(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx()
        after()
        self.assertIsNone(before())

    def test_no_context_runs_the_stage(self):
        before, _ = _make_prefix_reuse_callbacks("architect")
        self.assertIsNone(current_run_context.get())
        self.assertIsNone(before())

    def test_fresh_process_reruns(self):
        # Completion is process-local by design: a resumed run re-derives
        # against the resynced checkout instead of trusting an old artifact.
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx()
        self.record_summary()
        after()
        self.assertIsNotNone(before())
        fresh_before, _ = _make_prefix_reuse_callbacks("architect")
        self.assertIsNone(fresh_before())

    def test_completion_is_per_run(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        self.install_ctx(run_id="r1")
        self.record_summary(run_id="r1")
        after()
        self.record_summary(run_id="r2")
        self.install_ctx(run_id="r2")
        self.assertIsNone(before())

    def test_threat_modeler_requires_its_own_artifact(self):
        before, after = _make_prefix_reuse_callbacks("threat_modeler")
        self.install_ctx()
        self.record_summary()  # wrong artifact type for this node
        after()
        self.assertIsNone(before())
        record_artifact(
            self.db, "r1", "threat_model",
            "workspace/.structured/threat_model.json", json.dumps({"ok": True}),
        )
        content = before()
        self.assertIsNotNone(content)
        self.assertIn("Reused the threat_model", content.parts[0].text)


class VocabularyPinTest(unittest.TestCase):
    def test_loader_literal_matches_main(self):
        # graph_loader mirrors the literal because importing main would be
        # circular; this pin is what keeps the two from drifting.
        import main as main_module

        self.assertEqual(
            main_module.SCAN_MODE_CROSS_FUNCTIONAL, _SCAN_MODE_CROSS_FUNCTIONAL
        )


class WiringPinTest(unittest.TestCase):
    def test_loader_attaches_reuse_callbacks(self):
        import core.graph_loader as gl

        src = inspect.getsource(gl.load_workflow_from_json)
        self.assertIn(
            "node_id in _PREFIX_REUSE_ARTIFACTS and not node_cfg.on_enter_status",
            src,
        )
        self.assertIn("_make_prefix_reuse_callbacks(node_id)", src)

    def test_reuse_covers_exactly_the_repo_scope_stages(self):
        self.assertEqual(
            _PREFIX_REUSE_ARTIFACTS,
            {"architect": "summary", "threat_modeler": "threat_model"},
        )

    def test_shipped_workflow_keeps_reuse_active(self):
        # The attach guard defers to a completion stamp; the shipped workflow
        # must not declare one on the reuse nodes, or reuse silently dies.
        workflow = json.loads(
            (Path(_REF_ROOT) / "workflow.json").read_text(encoding="utf-8")
        )
        for node in workflow["nodes"]:
            if node["id"] in _PREFIX_REUSE_ARTIFACTS:
                self.assertNotIn("on_enter_status", node)


class ContentAddressedPrefixAndToolLoopGuardTest(unittest.TestCase):
    """Tests for Harness² content-addressed prefix reuse (runmark) and tool-loop guards (c8)."""

    def test_compute_prefix_fingerprint_changes_on_prompt_or_tools_or_model(self):
        from core.graph_loader import compute_prefix_fingerprint

        fp_base = compute_prefix_fingerprint(
            "architect",
            instruction="Map architecture",
            tools=["list_files", "read_file"],
            model="gemini-3.8-flash",
            snapshot_id="snap-1",
        )
        fp_reordered = compute_prefix_fingerprint(
            "architect",
            instruction="Map architecture",
            tools=["read_file", "list_files"],
            model="gemini-3.8-flash",
            snapshot_id="snap-1",
        )
        self.assertEqual(fp_base, fp_reordered)
        fp_changed_prompt = compute_prefix_fingerprint(
            "architect",
            instruction="Map architecture v2",
            tools=["list_files", "read_file"],
            model="gemini-3.8-flash",
            snapshot_id="snap-1",
        )
        self.assertNotEqual(fp_base, fp_changed_prompt)

    def test_prefix_reuse_invalidates_when_snapshot_or_fingerprint_changes(self):
        before, after = _make_prefix_reuse_callbacks("architect")
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "knowledge.db")
            init_db(db)
            record_artifact(db, "run-1", "summary", "summary.md", "Architecture summary")
            ctx1 = RunContext(
                jail_dir=tmp,
                db_path=db,
                run_id="run-1",
                scan_mode="cross-functional",
                snapshot_id="snap-1",
                model="gemini-3.8-flash",
                prefix_fingerprints={"architect": "fp-v1"},
            )
            tok = current_run_context.set(ctx1)
            try:
                self.assertIsNone(before())
                after()
                self.assertIsNotNone(before())
            finally:
                current_run_context.reset(tok)

            # Mutated fingerprint invalidates reuse even under the same run_id.
            ctx2 = RunContext(
                jail_dir=tmp,
                db_path=db,
                run_id="run-1",
                scan_mode="cross-functional",
                snapshot_id="snap-1",
                model="gemini-3.8-flash",
                prefix_fingerprints={"architect": "fp-v2"},
            )
            tok2 = current_run_context.set(ctx2)
            try:
                self.assertIsNone(before())
            finally:
                current_run_context.reset(tok2)

    def test_after_tool_guard_steers_after_repeated_unbounded_read_file(self):
        from core.graph_loader import _make_after_tool_guard_callback

        guard = _make_after_tool_guard_callback(
            "researcher",
            ["read_file", "get_function_boundary", "find_symbol"],
        )

        class _DummyTool:
            name = "read_file"

        tool = _DummyTool()
        ctx = RunContext(jail_dir="/tmp", db_path="/tmp/db.sqlite", run_id="r1")
        tok = current_run_context.set(ctx)
        try:
            # First unbounded read returns None (keep original output).
            self.assertIsNone(guard(tool, {"filepath": "app/auth.py"}, None, "line1\nline2"))
            # Echoed defaults (0 = whole file) are an unbounded read too.
            self.assertIsNone(
                guard(
                    tool,
                    {"filepath": "app/auth.py", "start_line": 0, "end_line": 0},
                    None,
                    "line1\nline2",
                )
            )
            # Bounded read does not increment counter or trigger hint.
            self.assertIsNone(
                guard(
                    tool,
                    {"filepath": "app/auth.py", "start_line": 1, "end_line": 20},
                    None,
                    "line1\nline2",
                )
            )
            # Third unbounded read appends steering hint.
            res = guard(tool, {"filepath": "app/auth.py"}, None, "line1\nline2")
            self.assertIsInstance(res, str)
            self.assertIn("HARNESS STEERING HINT", res)
            self.assertIn("get_function_boundary", res)
        finally:
            current_run_context.reset(tok)

    def test_after_tool_guard_hint_matches_real_tool_signatures(self):
        """Every tool call the hint suggests must be one the tools accept.

        The hint reaches a model that is already looping; a parameter name the
        tool does not have turns the nudge into a validation error.
        """
        from core.graph_loader import _make_after_tool_guard_callback
        from tools import structural_tools

        guard = _make_after_tool_guard_callback(
            "researcher", ["read_file", "get_function_boundary", "find_symbol"]
        )

        class _DummyTool:
            name = "read_file"

        res = None
        for _ in range(3):
            res = guard(_DummyTool(), {"filepath": "app/auth.py"}, None, "body")
        self.assertIsInstance(res, str)

        gfb_params = set(inspect.signature(structural_tools.get_function_boundary).parameters)
        fs_params = set(inspect.signature(structural_tools.find_symbol).parameters)
        for call in re.finditer(r"(get_function_boundary|find_symbol)\(([^)]*)\)", res):
            tool_name, arg_text = call.group(1), call.group(2)
            named = set(re.findall(r"(\w+)=", arg_text))
            allowed = gfb_params if tool_name == "get_function_boundary" else fs_params
            self.assertTrue(
                named and named <= allowed,
                f"hint suggests {tool_name}({arg_text}) but it accepts {sorted(allowed)}",
            )

    def test_load_workflow_exports_computed_prefix_fingerprints(self):
        from core.graph_loader import load_workflow_from_json

        wf_path = str(Path(_REF_ROOT) / "workflow.json")
        _, cfg = load_workflow_from_json(wf_path)
        fps = cfg.get("prefix_fingerprints") or {}
        self.assertIn("architect", fps)
        self.assertIn("threat_modeler", fps)
        self.assertEqual(len(fps["architect"]), 16)
        self.assertEqual(len(fps["threat_modeler"]), 16)

    def test_prefix_reuse_persists_and_verifies_db_runmark_metadata(self):
        """After-callback stamps runmark metadata into campaign_artifacts and before-callback verifies it."""
        import json as _json
        from core.database import read_artifact_record, update_artifact_metadata
        from core.graph_loader import _make_prefix_reuse_callbacks

        before, after = _make_prefix_reuse_callbacks("architect", fingerprint="fp-v1")
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "knowledge.db")
            init_db(db)
            record_artifact(
                db,
                "run-mark",
                "summary",
                "workspace/.structured/summary.json",
                '{"components": []}',
            )
            ctx = RunContext(
                jail_dir=tmp,
                db_path=db,
                run_id="run-mark",
                scan_mode="cross-functional",
                snapshot_id="snap-abc",
                model="vertex_ai/gemini-3.8-flash",
                prefix_fingerprints={"architect": "fp-v1"},
            )
            tok = current_run_context.set(ctx)
            try:
                after()
                rec = read_artifact_record(db, artifact_type="summary", run_id="run-mark")
                self.assertIsNotNone(rec)
                meta = _json.loads(rec["metadata_json"] or "{}")
                self.assertEqual(meta.get("prefix_fingerprint"), "fp-v1")
                self.assertEqual(meta.get("snapshot_id"), "snap-abc")
                self.assertEqual(meta.get("model"), "vertex_ai/gemini-3.8-flash")

                # Matching runmark reuses artifact
                self.assertIsNotNone(before())

                # Tampered/stale DB runmark metadata invalidates reuse
                update_artifact_metadata(
                    db,
                    "run-mark",
                    artifact_type="summary",
                    metadata={"prefix_fingerprint": "fp-stale"},
                )
                self.assertIsNone(before())
            finally:
                current_run_context.reset(tok)



if __name__ == "__main__":
    unittest.main()
