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

"""Coverage remembers WHICH harness covered a slice.

The spend ledger said a file was examined and nothing about by what. After
an upgrade -- a tool added, a stronger model, a rewritten prompt, another
sandbox tier -- every slice the old configuration covered stayed covered
forever: `--resume` skipped it and `auto` planned around it. Each row now
carries a fingerprint of the resolved harness, and the places that reuse
coverage compare it: a mismatch is reported once per run by default and
re-examined with --rescan-changed-harness. Rows with no fingerprint predate
the stamp and cover for every harness (INV-6).
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.cost import (  # noqa: E402
    coverage_verdict,
    harness_mismatches,
    has_fingerprint_column,
    record_spend,
    run_ledger,
    uncovered_files,
)
from core.graph_loader import compute_harness_fingerprint  # noqa: E402

FP_A = "a" * 16
FP_B = "b" * 16
FP_C = "c" * 16


def _nodes(**overrides) -> list:
    base = {
        "researcher": {"tools": ["read_file", "list_files"], "model": "vertex_ai/gemini-3.8-flash",
                       "instruction": "Find the bug.\n\nReport it."},
        "reviewer": {"tools": ["get_findings"], "model": "vertex_ai/gemini-3.8-flash",
                     "instruction": "Judge each finding."},
        "gate": {},
    }
    for node_id, patch in overrides.items():
        base[node_id] = {**base.get(node_id, {}), **patch}
    return [{"id": k, **v} for k, v in base.items()]


class FingerprintTest(unittest.TestCase):
    def test_is_sixteen_hex_characters_and_deterministic(self):
        fp = compute_harness_fingerprint(_nodes(), default_model="m", sandbox="static-only")
        self.assertRegex(fp, r"^[0-9a-f]{16}$")
        self.assertEqual(fp, compute_harness_fingerprint(_nodes(), default_model="m", sandbox="static-only"))

    def test_order_and_whitespace_do_not_matter(self):
        base = compute_harness_fingerprint(_nodes(), default_model="m", sandbox="static-only")
        reordered = compute_harness_fingerprint(
            list(reversed(_nodes(researcher={"tools": ["list_files", "read_file"]}))),
            default_model="m", sandbox="static-only")
        self.assertEqual(base, reordered, "node and tool order are not configuration")
        rewrapped = compute_harness_fingerprint(
            _nodes(researcher={"instruction": "  Find   the bug.\nReport it.\n\n"}),
            default_model="m", sandbox="static-only")
        self.assertEqual(base, rewrapped, "a whitespace-only prompt edit is not an upgrade")

    def test_changes_when_the_harness_changes(self):
        base = compute_harness_fingerprint(_nodes(), default_model="m", sandbox="static-only")
        variants = {
            "tool added": _nodes(researcher={"tools": ["read_file", "list_files", "find_symbol"]}),
            "model changed": _nodes(researcher={"model": "vertex_ai/claude-opus-5-5"}),
            "prompt changed": _nodes(researcher={"instruction": "Find the bug.\n\nReport it carefully."}),
            "node removed": _nodes()[:-1],
        }
        seen = {base}
        for label, nodes in variants.items():
            fp = compute_harness_fingerprint(nodes, default_model="m", sandbox="static-only")
            self.assertNotIn(fp, seen, label)
            seen.add(fp)
        self.assertNotIn(compute_harness_fingerprint(_nodes(), default_model="m", sandbox="gvisor"), seen,
                         "sandbox tier")
        self.assertNotIn(compute_harness_fingerprint(_nodes(), default_model="other", sandbox="static-only"),
                         seen, "default model")

    def test_the_loader_exports_one_for_the_resolved_graph(self):
        from core.graph_loader import load_workflow_from_json

        tmp = tempfile.mkdtemp(prefix="mantis_fp_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)

        def load(prompt="Audit the file.", tools=("read_file",), **kw):
            spec = {
                "name": "fp_test",
                "config": {"default_model": "openai/m1", "api_base": "http://127.0.0.1:9"},
                "nodes": [{"id": "auditor", "type": "agent", "system_prompt": prompt, "tools": list(tools)}],
                "edges": [{"from": "START", "to": "auditor"}],
            }
            path = os.path.join(tmp, "wf.json")
            Path(path).write_text(json.dumps(spec))
            _, cfg = load_workflow_from_json(path, load_local=False, **kw)
            return cfg["harness_fingerprint"]

        base = load()
        self.assertRegex(base, r"^[0-9a-f]{16}$")
        self.assertEqual(base, load(prompt="Audit  the\nfile."), "whitespace-only edit")
        self.assertNotEqual(base, load(prompt="Audit the file for SQLi."))
        self.assertNotEqual(base, load(tools=("read_file", "find_symbol")))
        self.assertNotEqual(base, load(model_override="openai/m2"))
        self.assertNotEqual(base, load(sandbox_override="gvisor"))


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_fp_ledger_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = os.path.realpath(os.path.join(self.tmp, "repo"))
        os.makedirs(os.path.join(self.root, "lib"))
        self.db = os.path.join(self.tmp, "k.db")
        self.a = os.path.join(self.root, "a.py")
        self.b = os.path.join(self.root, "b.py")
        self.c = os.path.join(self.root, "c.py")
        self.d = os.path.join(self.root, "lib", "d.py")
        self.files = [self.a, self.b, self.c, self.d]

    def _legacy_table(self, db: str) -> None:
        """A ledger the way an older Mantis left it: no fingerprint column."""
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE campaign_spend (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, "
            "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, target TEXT, scan_mode TEXT, tokens INTEGER, "
            "llm_calls INTEGER, graph_steps INTEGER, elapsed_seconds REAL, metadata_json TEXT DEFAULT '{}')"
        )
        conn.execute("INSERT INTO campaign_spend (run_id, target, scan_mode, tokens, metadata_json) "
                     "VALUES ('old', ?, 'file-by-file', 10, '{}')", (self.a,))
        conn.commit()
        conn.close()

    def _column(self, target: str) -> list:
        conn = sqlite3.connect(self.db)
        try:
            return [r[0] for r in conn.execute(
                "SELECT harness_fingerprint FROM campaign_spend WHERE target = ? ORDER BY id", (target,))]
        finally:
            conn.close()

    def test_record_spend_writes_the_stamp_and_null_without_one(self):
        self.assertTrue(record_spend(self.db, "r1", self.a, "file-by-file", tokens=10, harness_fingerprint=FP_A))
        self.assertTrue(record_spend(self.db, "r1", self.b, "file-by-file", tokens=10))
        self.assertTrue(record_spend(self.db, "r1", self.c, "file-by-file", tokens=10, harness_fingerprint=""))
        self.assertEqual(self._column(self.a), [FP_A])
        self.assertEqual(self._column(self.b), [None], "no stamp is NULL, not an empty string")
        self.assertEqual(self._column(self.c), [None])

    def test_an_old_ledger_is_migrated_on_first_write_and_readable_before(self):
        self._legacy_table(self.db)
        conn = sqlite3.connect(self.db)
        self.assertFalse(has_fingerprint_column(conn))
        conn.close()
        # Read-only probes work against the old shape and never migrate it.
        self.assertEqual(uncovered_files(self.db, self.files), [self.b, self.c, self.d])
        self.assertEqual(uncovered_files(self.db, self.files, harness_fingerprint=FP_B), [self.b, self.c, self.d],
                         "a pre-upgrade row covers for every harness")
        self.assertEqual(harness_mismatches(self.db, self.files, harness_fingerprint=FP_B),
                         {"files": [], "fingerprints": {}})
        self.assertEqual(run_ledger(self.db, "old"), {self.a: {"failed": False, "harness_fingerprint": None}})
        conn = sqlite3.connect(self.db)
        self.assertFalse(has_fingerprint_column(conn), "reading did not alter the table")
        conn.close()
        # The first write adds the column; the old row keeps its NULL.
        self.assertTrue(record_spend(self.db, "r2", self.b, "file-by-file", tokens=10, harness_fingerprint=FP_B))
        conn = sqlite3.connect(self.db)
        self.assertTrue(has_fingerprint_column(conn))
        conn.close()
        self.assertEqual(self._column(self.a), [None])
        self.assertEqual(self._column(self.b), [FP_B])
        self.assertEqual(run_ledger(self.db, "r2"), {self.b: {"failed": False, "harness_fingerprint": FP_B}})

    def test_warn_keeps_mismatches_covered_and_enforce_sweeps_them(self):
        record_spend(self.db, "r1", self.a, "file-by-file", tokens=10)                            # legacy
        record_spend(self.db, "r2", self.b, "file-by-file", tokens=10, harness_fingerprint=FP_A)  # same
        record_spend(self.db, "r3", self.c, "file-by-file", tokens=10, harness_fingerprint=FP_B)  # other
        # d.py has no row of its own: uncovered under every policy.
        current = FP_A
        self.assertEqual(uncovered_files(self.db, self.files), [self.d], "no fingerprint: as it always was")
        self.assertEqual(uncovered_files(self.db, self.files, harness_fingerprint=current), [self.d],
                         "warn: covered by another harness still counts as covered")
        self.assertEqual(
            uncovered_files(self.db, self.files, harness_fingerprint=current, rescan_changed_harness=True),
            [self.c, self.d], "enforce: only the mismatch joins the sweep, in input order")
        self.assertEqual(harness_mismatches(self.db, self.files, harness_fingerprint=current),
                         {"files": [self.c], "fingerprints": {FP_B: 1}})
        self.assertIsNone(harness_mismatches(self.db, self.files), "nothing to compare against")
        self.assertIsNone(harness_mismatches("", self.files, harness_fingerprint=current))
        # The current harness covers c.py too: no longer a mismatch, by either row.
        record_spend(self.db, "r4", self.c, "file-by-file", tokens=10, harness_fingerprint=FP_A)
        self.assertEqual(
            uncovered_files(self.db, self.files, harness_fingerprint=current, rescan_changed_harness=True),
            [self.d])
        self.assertEqual(harness_mismatches(self.db, self.files, harness_fingerprint=current)["files"], [])

    def test_a_legacy_row_anywhere_in_the_chain_covers_for_every_harness(self):
        # An old whole-repository campaign, plus a per-file campaign by another harness.
        record_spend(self.db, "old", self.root, "whole", tokens=10)
        record_spend(self.db, "r2", self.d, "file-by-file", tokens=10, harness_fingerprint=FP_B)
        self.assertEqual(
            uncovered_files(self.db, self.files, harness_fingerprint=FP_A, rescan_changed_harness=True), [],
            "a row with no stamp covers the tree for any harness, even under enforce (INV-6)")
        self.assertEqual(harness_mismatches(self.db, self.files, harness_fingerprint=FP_A)["files"], [])

    def test_directory_coverage_by_another_harness_is_a_mismatch_for_each_file(self):
        record_spend(self.db, "r1", os.path.join(self.root, "lib"), "cross-functional", tokens=10,
                     harness_fingerprint=FP_B)
        record_spend(self.db, "r1", self.a, "file-by-file", tokens=10, harness_fingerprint=FP_C)
        stale = harness_mismatches(self.db, self.files, harness_fingerprint=FP_A)
        self.assertEqual(stale, {"files": [self.a, self.d], "fingerprints": {FP_B: 1, FP_C: 1}})
        self.assertEqual(uncovered_files(self.db, self.files, harness_fingerprint=FP_A, rescan_changed_harness=True),
                         self.files)
        self.assertEqual(uncovered_files(self.db, self.files, harness_fingerprint=FP_A), [self.b, self.c])

    def test_failed_rows_never_count_whatever_their_stamp(self):
        record_spend(self.db, "r1", self.a, "file-by-file", tokens=10, metadata={"failed": True},
                     harness_fingerprint=FP_A)
        record_spend(self.db, "r1", self.b, "file-by-file", tokens=10, metadata={"failed": True})
        self.assertEqual(uncovered_files(self.db, [self.a, self.b], harness_fingerprint=FP_A), [self.a, self.b])
        self.assertEqual(harness_mismatches(self.db, [self.a, self.b], harness_fingerprint=FP_B)["files"], [])

    def test_mismatch_is_judged_per_model(self):
        from core.database import init_db, record_run

        init_db(self.db)
        record_run(self.db, "r-flash", "vertex_ai/gemini-3.8-flash")
        record_run(self.db, "r-opus", "vertex_ai/claude-opus-5-5")
        record_spend(self.db, "r-flash", self.a, "file-by-file", tokens=10, harness_fingerprint=FP_B)
        record_spend(self.db, "r-opus", self.b, "file-by-file", tokens=10, harness_fingerprint=FP_B)
        files = [self.a, self.b]
        # For flash: a.py was covered by flash under another harness; b.py was never covered by flash.
        self.assertEqual(uncovered_files(self.db, files, model="vertex_ai/gemini-3.8-flash",
                                         harness_fingerprint=FP_A), [self.b])
        self.assertEqual(harness_mismatches(self.db, files, model="vertex_ai/gemini-3.8-flash",
                                            harness_fingerprint=FP_A)["files"], [self.a])

    def test_unreadable_ledger_is_unknowable_under_every_policy(self):
        corrupt = os.path.join(self.tmp, "corrupt.db")
        Path(corrupt).write_bytes(b"not a database")
        self.assertIsNone(uncovered_files(corrupt, self.files, harness_fingerprint=FP_A))
        self.assertIsNone(uncovered_files(corrupt, self.files, harness_fingerprint=FP_A, rescan_changed_harness=True))
        self.assertIsNone(harness_mismatches(corrupt, self.files, harness_fingerprint=FP_A))
        self.assertIsNone(run_ledger(corrupt, "r1"))
        missing = os.path.join(self.tmp, "nope.db")
        self.assertEqual(uncovered_files(missing, self.files, harness_fingerprint=FP_A, rescan_changed_harness=True),
                         self.files, "no ledger: a genuine first run")
        self.assertEqual(harness_mismatches(missing, self.files, harness_fingerprint=FP_A),
                         {"files": [], "fingerprints": {}})

    def test_coverage_verdict_rule(self):
        self.assertEqual(coverage_verdict(None, FP_A), "uncovered")
        self.assertEqual(coverage_verdict({FP_A}, FP_A), "covered")
        self.assertEqual(coverage_verdict({FP_B}, FP_A), "mismatched")
        self.assertEqual(coverage_verdict({FP_B, None}, FP_A), "covered", "a NULL row covers for every harness")
        self.assertEqual(coverage_verdict({FP_B, FP_A}, FP_A), "covered")
        self.assertEqual(coverage_verdict({FP_B}, ""), "covered", "nothing to compare against")


class KnowledgeRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_fp_kn_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = os.path.realpath(os.path.join(self.tmp, "repo"))
        os.makedirs(os.path.join(self.repo, "src"))
        self.src_db = os.path.join(self.tmp, "src.db")
        self.export = os.path.join(self.repo, ".mantis", "knowledge")

    def test_export_carries_the_stamp_and_import_preserves_it(self):
        from core import knowledge as K
        from core.database import init_db

        init_db(self.src_db)
        app = os.path.join(self.repo, "src/app.py")
        util = os.path.join(self.repo, "src/util.py")
        record_spend(self.src_db, "r1", app, "file-by-file", tokens=100, harness_fingerprint=FP_A)
        record_spend(self.src_db, "r1", util, "file-by-file", tokens=100)
        K.export_knowledge(self.src_db, self.export, repo_root=self.repo)
        cov = json.loads(Path(self.export, K.COVERAGE).read_text())
        by_path = {s["path"]: s for s in cov["scanned"]}
        self.assertEqual(by_path["src/app.py"]["harness_fingerprint"], FP_A)
        self.assertEqual(by_path["src/util.py"]["harness_fingerprint"], "", "unknown exports as empty")

        fresh = os.path.join(self.tmp, "fresh.db")
        counts = K.import_knowledge(fresh, self.export, run_id="run-new", repo_root=self.repo)
        self.assertEqual(counts["coverage_paths"], 2)
        self.assertEqual(run_ledger(fresh, "r1"), {
            app: {"failed": False, "harness_fingerprint": FP_A},
            util: {"failed": False, "harness_fingerprint": None},
        })
        # The imported ledger answers the harness question like a native one.
        self.assertEqual(uncovered_files(fresh, [app, util], harness_fingerprint=FP_B, rescan_changed_harness=True),
                         [app])
        self.assertEqual(harness_mismatches(fresh, [app, util], harness_fingerprint=FP_B),
                         {"files": [app], "fingerprints": {FP_A: 1}})
        # Re-exporting the imported database gives the same coverage back.
        again = os.path.join(self.tmp, "again")
        K.export_knowledge(fresh, again, repo_root=self.repo)
        self.assertEqual(json.loads(Path(again, K.COVERAGE).read_text())["scanned"], cov["scanned"])

    def test_a_legacy_export_without_the_field_imports_as_unknown(self):
        from core import knowledge as K

        os.makedirs(self.export)
        Path(self.export, K.MANIFEST).write_text(json.dumps({"schema_version": 1}))
        Path(self.export, K.COVERAGE).write_text(json.dumps({
            "runs": [],
            "scanned": [
                {"path": "src/app.py", "scan_mode": "file-by-file", "last_run_id": "old"},
                {"path": "src/util.py", "scan_mode": "file-by-file", "last_run_id": "old",
                 "harness_fingerprint": "DROP TABLE campaign_spend; --"},
                {"path": "src/ok.py", "scan_mode": "file-by-file", "last_run_id": "old",
                 "harness_fingerprint": FP_B},
            ],
        }))
        fresh = os.path.join(self.tmp, "fresh.db")
        counts = K.import_knowledge(fresh, self.export, run_id="run-new", repo_root=self.repo)
        self.assertEqual(counts["coverage_paths"], 3)
        ledger = run_ledger(fresh, "old")
        self.assertEqual(ledger[os.path.join(self.repo, "src/app.py")]["harness_fingerprint"], None)
        self.assertEqual(ledger[os.path.join(self.repo, "src/util.py")]["harness_fingerprint"], None,
                         "a value the exporter would not write imports as unknown")
        self.assertEqual(ledger[os.path.join(self.repo, "src/ok.py")]["harness_fingerprint"], FP_B)

    def test_export_reads_a_ledger_that_predates_the_column(self):
        from core import knowledge as K
        from core.database import init_db

        init_db(self.src_db)
        app = os.path.join(self.repo, "src/app.py")
        conn = sqlite3.connect(self.src_db)
        conn.execute(
            "CREATE TABLE campaign_spend (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, "
            "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, target TEXT, scan_mode TEXT, tokens INTEGER, "
            "llm_calls INTEGER, graph_steps INTEGER, elapsed_seconds REAL, metadata_json TEXT DEFAULT '{}')"
        )
        conn.execute("INSERT INTO campaign_spend (run_id, target, scan_mode, tokens) VALUES ('old', ?, 'whole', 5)",
                     (app,))
        conn.commit()
        conn.close()
        K.export_knowledge(self.src_db, self.export, repo_root=self.repo)
        cov = json.loads(Path(self.export, K.COVERAGE).read_text())
        self.assertEqual([(s["path"], s["harness_fingerprint"]) for s in cov["scanned"]], [("src/app.py", "")])


class WiringTest(unittest.TestCase):
    """The stamp reaches the ledger from the scan loop, and the policy reaches the CLI."""

    def test_run_context_carries_the_fingerprint(self):
        from core.context import RunContext

        self.assertEqual(RunContext(jail_dir="/tmp", db_path="/tmp/x.db").harness_fingerprint, "",
                         "unknown by default (INV-6)")

    def test_pipeline_stamps_every_spend_row_and_the_run_context(self):
        import main

        tree = ast.parse(textwrap.dedent(inspect.getsource(main.pipeline)))
        calls: dict = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in ("record_spend", "RunContext", "_stamp_member_coverage"):
                    calls.setdefault(name, []).append({kw.arg for kw in node.keywords})
        for name in ("record_spend", "RunContext", "_stamp_member_coverage"):
            self.assertTrue(calls.get(name), f"{name} is no longer called from pipeline()")
            for kwargs in calls[name]:
                self.assertIn("harness_fingerprint", kwargs, f"{name} call site drops the harness fingerprint")

    def test_member_stamps_carry_the_fingerprint(self):
        from main import _stamp_member_coverage

        tmp = tempfile.mkdtemp(prefix="mantis_fp_members_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = os.path.join(tmp, "k.db")
        group = {"targets": ["/repo/a.py", "/repo/b.py", "/repo/c.py"]}
        self.assertEqual(_stamp_member_coverage(db, "r1", group, "/repo/a.py", "cross-functional",
                                                harness_fingerprint=FP_A), 2)
        self.assertEqual(run_ledger(db, "r1"), {
            "/repo/b.py": {"failed": False, "harness_fingerprint": FP_A},
            "/repo/c.py": {"failed": False, "harness_fingerprint": FP_A},
        })

    def test_the_notice_names_counts_and_stamps(self):
        from main import _changed_harness_notice

        line = _changed_harness_notice(3, {FP_B: 2, FP_C: 1}, FP_A, False)
        self.assertEqual(
            line,
            f"3 previously covered slice(s) were examined by a different harness configuration "
            f"(fingerprints {FP_B}, {FP_C} vs {FP_A}); pass --rescan-changed-harness to re-examine them.")
        self.assertIn("re-examining them (--rescan-changed-harness).",
                      _changed_harness_notice(1, {FP_B: 1}, FP_A, True))
        self.assertIn(f"(fingerprint {FP_B} vs {FP_A})", _changed_harness_notice(1, {FP_B: 1}, FP_A, True))

    def test_cli_flag_defaults_off(self):
        from scripts.launch import build_parser

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(build_parser().parse_args(["x"]).rescan_changed_harness)
            self.assertTrue(build_parser().parse_args(["x", "--rescan-changed-harness"]).rescan_changed_harness)


class ResumeRunTest(unittest.TestCase):
    """End to end against a dummy endpoint: what a resumed run decides to skip."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_fp_e2e_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = os.path.realpath(os.path.join(self.tmp, "repo"))
        os.makedirs(self.repo)
        self.files = {}
        for name in ("a", "b", "c", "d"):
            p = os.path.join(self.repo, f"{name}.py")
            Path(p).write_text(f"def {name}():\n    return 1\n")
            self.files[name] = p
        self.db = os.path.join(self.tmp, "m.db")
        self.env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        self.env.update({"OPENAI_API_KEY": "dummy", "LLM_TIMEOUT": "3", "MANTIS_LLM_MAX_PATIENCE_SECONDS": "3"})

    def run_main(self, *extra):
        proc = subprocess.run(
            [sys.executable, str(_REF_ROOT / "main.py"), self.repo, "--scan-mode", "file-by-file", "--yes",
             "--sandbox", "static-only", "--model", "openai/nonexistent", "--api-base", "http://127.0.0.1:9",
             "--max-llm-calls", "1", "--db", self.db, *extra],
            capture_output=True, text=True, timeout=300, env=self.env, cwd=str(_REF_ROOT))
        return proc.returncode, proc.stdout + proc.stderr

    def test_a_resumed_run_reports_or_rescans_ground_another_harness_covered(self):
        run = "11111111-2222-3333-4444-555555555555"
        other = "deadbeefdeadbeef"
        record_spend(self.db, run, self.files["a"], "file-by-file", tokens=10, harness_fingerprint=other)
        record_spend(self.db, run, self.files["b"], "file-by-file", tokens=10, harness_fingerprint=other)
        record_spend(self.db, run, self.files["c"], "file-by-file", tokens=10)  # pre-upgrade row
        record_spend(self.db, run, self.files["d"], "file-by-file", tokens=10, metadata={"failed": True},
                     harness_fingerprint=other)

        # Default: still completed, said once, and the failed one is what gets rerun.
        _code, out = self.run_main("--resume", run)
        self.assertIn("2 previously covered slice(s) were examined by a different harness configuration "
                      f"(fingerprint {other} vs ", out)
        self.assertIn("pass --rescan-changed-harness to re-examine them.", out)
        self.assertEqual(out.count("previously covered slice(s)"), 1, "one line per run")
        self.assertIn(f"Resuming run {run}: 3 of 4 campaign(s) completed; 1 failed and 0 never started: 1 to scan.",
                      out)

        # Enforce: the two stamped by another harness rejoin the plan; the legacy row still counts.
        _code, out = self.run_main("--resume", run, "--rescan-changed-harness")
        self.assertIn("re-examining them (--rescan-changed-harness).", out)
        self.assertIn(f"Resuming run {run}: 1 of 4 campaign(s) completed; "
                      "1 failed, 2 examined by a different harness and 0 never started: 3 to scan.", out)
        self.assertIn("file-by-file: 3 campaign(s)", out)
        self.assertNotIn("Triggered via: " + self.files["c"], out)


if __name__ == "__main__":
    unittest.main()
