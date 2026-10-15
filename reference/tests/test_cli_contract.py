"""Tests for the CLI contract: one-shot overrides, --save-config opt-in,
--path-root hard-fail, and launcher usage exit codes.

A per-invocation flag must never silently become sticky configuration:
before these rules, `--db /tmp/scratch.db` on one experiment persisted
into workflow.local.json and every later run kept writing findings there,
and a typo'd --model was saved before validation, breaking every
subsequent run. A --path-root that cannot contain the target used to
warn-and-proceed, burning the full budget on an un-rooted catalog.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

SENTINEL_MODEL = "vertex_ai/sentinel-model-for-tests"


class ConfigPersistenceTest(unittest.TestCase):
    """CLI overrides are live for the run but only saved on explicit opt-in."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mantis_cli_contract_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(os.path.realpath(self._tmp.name))
        self.wf = self.tmp / "workflow.json"
        shutil.copyfile(Path(_REF_ROOT) / "workflow.json", self.wf)
        self.local = self.tmp / "workflow.local.json"

    def _run(self, **kwargs):
        from scripts.configure import ensure_configured_async
        return asyncio.run(
            ensure_configured_async(
                workflow_path=str(self.wf),
                auto=True,
                overrides={"default_model": SENTINEL_MODEL},
                **kwargs,
            )
        )

    def _local_text(self) -> str:
        return self.local.read_text(encoding="utf-8") if self.local.exists() else ""

    def test_overrides_are_one_shot_by_default(self):
        before = self.wf.read_bytes()
        cfg = self._run()
        # Live for THIS run...
        self.assertEqual(cfg.get("default_model"), SENTINEL_MODEL)
        # ...but never written to disk without the opt-in.
        self.assertNotIn(SENTINEL_MODEL, self._local_text())
        self.assertEqual(self.wf.read_bytes(), before)

    def test_overrides_persist_with_save_config(self):
        cfg = self._run(persist_overrides=True)
        self.assertEqual(cfg.get("default_model"), SENTINEL_MODEL)
        self.assertIn(SENTINEL_MODEL, self._local_text())
        # The tracked file still never changes; persistence targets the
        # local overlay only.
        self.assertNotIn(
            SENTINEL_MODEL, self.wf.read_text(encoding="utf-8")
        )


class PathRootHardFailTest(unittest.TestCase):
    """An impossible --path-root exits 1 before any budget is spent."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mantis_path_root_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(os.path.realpath(self._tmp.name))
        self.target_dir = self.tmp / "repo"
        self.target_dir.mkdir()
        self.target = self.target_dir / "app.py"
        self.target.write_text("def f():\n    return 1\n", encoding="utf-8")
        self.elsewhere = self.tmp / "unrelated"
        self.elsewhere.mkdir()

    def _invoke(self, path_root: str):
        # A dummy OpenAI-compatible endpoint: the launcher's preflight wants a
        # configured model before it hands over to the pipeline, and this test
        # must not depend on credentials being present. Nothing is called;
        # --path-root is rejected before the first LLM turn.
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        env["OPENAI_API_KEY"] = "dummy"
        return subprocess.run(
            [
                sys.executable,
                str(Path(_REF_ROOT) / "main.py"),
                str(self.target),
                "--path-root", path_root,
                "--db", str(self.tmp / "kb.db"),
                "--model", "openai/nonexistent",
                "--api-base", "http://127.0.0.1:9",
                "--sandbox", "static-only",
                "--no-auto-configure",
            ],
            cwd=str(self.tmp),
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

    def test_non_ancestor_dir_fails_closed(self):
        res = self._invoke(str(self.elsewhere))
        self.assertEqual(res.returncode, 1, res.stderr[-2000:])
        self.assertIn("--path-root", res.stderr)
        self.assertIn("not an ancestor", res.stderr)

    def test_file_path_root_fails_closed_with_distinct_reason(self):
        res = self._invoke(str(self.target))
        self.assertEqual(res.returncode, 1, res.stderr[-2000:])
        self.assertIn("--path-root", res.stderr)
        self.assertIn("not a directory", res.stderr)


class LauncherUsageExitTest(unittest.TestCase):
    """Usage errors exit 64 (EX_USAGE); exit 2 stays reserved for budget pause."""

    def _parse(self, argv):
        from scripts.launch import build_parser
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                parser.parse_args(argv)
        return ctx.exception.code

    def test_unknown_flag_exits_64(self):
        self.assertEqual(self._parse(["t", "--definitely-not-a-flag"]), 64)

    def test_bad_choice_exits_64(self):
        self.assertEqual(self._parse(["t", "--sandbox", "bogus"]), 64)

    def test_help_still_exits_0(self):
        self.assertEqual(self._parse(["--help"]), 0)


class SaveConfigFlagTest(unittest.TestCase):
    """--save-config exists, defaulting to off."""

    def test_launcher_flag_defaults_off(self):
        from scripts.launch import build_parser
        args = build_parser().parse_args(["target"])
        self.assertFalse(args.save_config)
        args = build_parser().parse_args(["target", "--save-config"])
        self.assertTrue(args.save_config)


class OneParserTest(unittest.TestCase):
    """main.py and run.sh (scripts/launch.py) are the same command line.

    Two parsers drifted twice: --scan-mode and --exclude existed only on
    main.py, --token-budget and --max-time only on the launcher, and the
    pause banner printed a resume command one of them rejected.
    """

    @staticmethod
    def _flags(help_text: str) -> set:
        import re
        return set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", help_text))

    def _help(self, script: str) -> str:
        res = subprocess.run([sys.executable, str(Path(_REF_ROOT) / script), "--help"],
                             capture_output=True, text=True, timeout=120, cwd=_REF_ROOT)
        self.assertEqual(res.returncode, 0, res.stderr[-2000:])
        return res.stdout

    def test_main_has_no_parser_of_its_own(self):
        import main as mantis_main
        self.assertFalse(hasattr(mantis_main, "parse_cli_args"))

    def test_both_entry_points_accept_the_same_flags(self):
        main_flags = self._flags(self._help("main.py"))
        launcher_flags = self._flags(self._help("scripts/launch.py"))
        self.assertEqual(main_flags, launcher_flags)
        for flag in ("--scan-mode", "--yes", "--exclude", "--path-root", "--resume",
                     "--token-budget", "--max-time", "--objective", "--no-context-cache"):
            self.assertIn(flag, main_flags, flag)

    def test_main_usage_errors_exit_64_too(self):
        res = subprocess.run([sys.executable, str(Path(_REF_ROOT) / "main.py"), "t", "--definitely-not-a-flag"],
                             capture_output=True, text=True, timeout=120, cwd=_REF_ROOT)
        self.assertEqual(res.returncode, 64, res.stderr[-500:])


if __name__ == "__main__":
    unittest.main()
