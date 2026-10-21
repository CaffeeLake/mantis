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

"""apply_patch on an image with no patch(1) applies the diff with the guest's python3.

Live run: ``python:3-slim`` returned ``SANDBOX-ERROR: patch(1) unavailable``
from the patcher stage, the campaign counted as failed, and the model spent
a stage writing its own patch into /usr/local/bin. The applier is exercised
here with the host's python3 against diffs ``diff -u`` produced, and the two
container environments against a scripted shell.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parents[1]
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.environments import patch_fallback as PF  # noqa: E402


def _diff_u(a: Path, b: Path, label_a: str, label_b: str) -> str:
    p = subprocess.run(["diff", "-u", "--label", label_a, "--label", label_b, str(a), str(b)],
                       capture_output=True, text=True)
    return p.stdout


class PythonApplierTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mantis_patch_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.script = self.tmp / PF.PATCH_SCRIPT_NAME
        self.script.write_text(PF.PYTHON_PATCH_SCRIPT, encoding="utf-8")
        self.repo = self.tmp / "repo"
        self.repo.mkdir()

    def _run(self, diff: str, strip: int = 1):
        (self.repo / PF.PATCH_FILE_NAME).write_text(diff, encoding="utf-8")
        shutil.copy(self.script, self.repo / PF.PATCH_SCRIPT_NAME)
        p = subprocess.run([sys.executable, *PF.fallback_argv(strip=strip)[1:]], cwd=str(self.repo),
                           capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def _write(self, rel: str, text: str) -> Path:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_git_style_diff_with_context_applies_like_patch_p1(self):
        before = "import os\n\ndef find_user(name):\n    q = f\"SELECT * FROM users WHERE name='{name}'\"\n    return conn.execute(q)\n\n\ndef other():\n    return 1\n"
        after = "import os\n\ndef find_user(name):\n    q = \"SELECT * FROM users WHERE name=?\"\n    return conn.execute(q, (name,))\n\n\ndef other():\n    return 1\n"
        self._write("dal/users.py", before)
        new = self.tmp / "new.py"; new.write_text(after)
        diff = "diff --git a/dal/users.py b/dal/users.py\nindex 1..2 100644\n" + _diff_u(self.repo / "dal/users.py", new, "a/dal/users.py", "b/dal/users.py")
        rc, out = self._run(diff)
        self.assertEqual(rc, 0, out)
        self.assertIn("patching file dal/users.py", out)
        self.assertEqual((self.repo / "dal/users.py").read_text(), after)

    def test_a_hunk_that_drifted_still_applies_and_one_that_does_not_match_fails_cleanly(self):
        body = "".join(f"line {i}\n" for i in range(1, 41))
        self._write("a.txt", body)
        new = self.tmp / "a_new.txt"; new.write_text(body.replace("line 30\n", "line thirty\n"))
        diff = _diff_u(self.repo / "a.txt", new, "a/a.txt", "b/a.txt")
        # Insert five lines at the top AFTER the diff was made: the hunk's line numbers are now stale.
        self._write("a.txt", "".join(f"new {i}\n" for i in range(5)) + body)
        rc, out = self._run(diff)
        self.assertEqual(rc, 0, out)
        self.assertIn("line thirty\n", (self.repo / "a.txt").read_text())
        # Context that exists nowhere: no file is touched, the exit is non-zero, the file is named.
        self._write("b.txt", "completely different\n")
        diff_b = diff.replace("a/a.txt", "a/b.txt").replace("b/a.txt", "b/b.txt")
        rc, out = self._run(diff_b)
        self.assertNotEqual(rc, 0)
        self.assertIn("Hunk FAILED for b.txt", out)
        self.assertEqual((self.repo / "b.txt").read_text(), "completely different\n")

    def test_new_and_deleted_files_and_no_newline_at_eof(self):
        self._write("old.txt", "gone\n")
        diff = (
            "--- /dev/null\n+++ b/pkg/new.txt\n@@ -0,0 +1,2 @@\n+hello\n+world\n\\ No newline at end of file\n"
            "--- a/old.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n"
        )
        rc, out = self._run(diff)
        self.assertEqual(rc, 0, out)
        self.assertEqual((self.repo / "pkg/new.txt").read_text(), "hello\nworld")
        self.assertFalse((self.repo / "old.txt").exists())
        self.assertIn("removing file old.txt", out)

    def test_all_or_nothing_across_files(self):
        self._write("x.txt", "x1\nx2\n")
        self._write("y.txt", "unrelated\n")
        diff = (
            "--- a/x.txt\n+++ b/x.txt\n@@ -1,2 +1,2 @@\n x1\n-x2\n+x2 changed\n"
            "--- a/y.txt\n+++ b/y.txt\n@@ -1 +1 @@\n-this context is wrong\n+whatever\n"
        )
        rc, out = self._run(diff)
        self.assertNotEqual(rc, 0)
        self.assertEqual((self.repo / "x.txt").read_text(), "x1\nx2\n", "the first file is not written when the second fails")

    def test_strip_zero_and_missing_file(self):
        self._write("z.txt", "a\n")
        rc, out = self._run("--- z.txt\n+++ z.txt\n@@ -1 +1 @@\n-a\n+b\n", strip=0)
        self.assertEqual(rc, 0, out)
        self.assertEqual((self.repo / "z.txt").read_text(), "b\n")
        rc, out = self._run("--- a/nope.txt\n+++ b/nope.txt\n@@ -1 +1 @@\n-a\n+b\n")
        self.assertNotEqual(rc, 0)
        self.assertIn("can't find file to patch: nope.txt", out)

    def test_the_script_is_plain_python3(self):
        compile(PF.PYTHON_PATCH_SCRIPT, PF.PATCH_SCRIPT_NAME, "exec")
        self.assertEqual(PF.fallback_command(), f"python3 {PF.PATCH_SCRIPT_NAME} -p1 {PF.PATCH_FILE_NAME}")
        self.assertIn("python3 fallback could not run", PF.unavailable_message("img"))

    def test_form_feed_and_zero_length_old_range_apply_like_gnu_patch(self):
        self._write("ff.c", "a\n\x0cb\nc\n")
        diff_ff = "--- a/ff.c\n+++ b/ff.c\n@@ -1,3 +1,3 @@\n a\n \x0cb\n-c\n+C\n"
        rc, out = self._run(diff_ff)
        self.assertEqual(rc, 0, out)
        self.assertEqual((self.repo / "ff.c").read_text(), "a\n\x0cb\nC\n")

        self._write("ins.txt", "".join(f"l{i}\n" for i in range(1, 6)))
        diff_ins = "--- a/ins.txt\n+++ b/ins.txt\n@@ -2,0 +3,1 @@\n+INSERTED\n"
        rc, out = self._run(diff_ins)
        self.assertEqual(rc, 0, out)
        self.assertEqual((self.repo / "ins.txt").read_text(), "l1\nl2\nINSERTED\nl3\nl4\nl5\n")

        rc_escape, _ = self._run("--- a/../../escape.txt\n+++ b/../../escape.txt\n@@ -0,0 +1 @@\n+x\n")
        self.assertNotEqual(rc_escape, 0)


class _Res:
    def __init__(self, code, out="", err=""):
        self.exit_code, self.stdout, self.stderr = code, out, err


class MicrosandboxFallbackTest(unittest.TestCase):
    """The environment's control flow, with the guest shell scripted."""

    def _env(self, responses):
        from core.environments.microsandbox_env import MicrosandboxEnvironment
        env = MicrosandboxEnvironment.__new__(MicrosandboxEnvironment)
        env.image = "mirror.gcr.io/library/python:3-slim"
        env.written = {}
        env.commands = []

        async def write_file(path, content):
            env.written[str(path)] = content if isinstance(content, str) else content.decode()

        async def execute(command, **_kw):
            env.commands.append(command)
            for prefix, res in responses:
                if command.startswith(prefix):
                    return res
            return _Res(1, "", "unexpected: " + command)
        env.write_file = write_file
        env.execute = execute
        return env

    def test_patch_present_is_used_as_before(self):
        env = self._env([("patch -p1", _Res(0, "patching file a.py\n"))])
        out = asyncio.run(env.apply_patch("--- a/a.py\n+++ b/a.py\n"))
        self.assertTrue(out.startswith("exit=0"))
        self.assertNotIn(PF.PATCH_SCRIPT_NAME, env.written)

    def test_no_patch_binary_runs_the_python_applier(self):
        env = self._env([("patch -p1", _Res(127, "", "sh: patch: not found")),
                         ("python3 " + PF.PATCH_SCRIPT_NAME, _Res(0, "patching file a.py\n"))])
        out = asyncio.run(env.apply_patch("--- a/a.py\n+++ b/a.py\n"))
        self.assertTrue(out.startswith("exit=0"), out)
        self.assertIn("patching file a.py", out)
        self.assertEqual(env.written[PF.PATCH_SCRIPT_NAME], PF.PYTHON_PATCH_SCRIPT)
        self.assertIn(PF.fallback_command(strip=1), env.commands)
        self.assertNotIn("SANDBOX-ERROR", out)

    def test_neither_patch_nor_python_is_the_old_error_with_both_named(self):
        env = self._env([("patch -p1", _Res(127)), ("python3 ", _Res(127, "", "python3: not found"))])
        out = asyncio.run(env.apply_patch("--- a/a.py\n+++ b/a.py\n"))
        self.assertTrue(out.startswith("SANDBOX-ERROR: patch(1) unavailable"), out)
        self.assertIn("python3 fallback could not run", out)

    def test_fallback_hunk_failure_is_reported_not_hidden(self):
        env = self._env([("patch -p1", _Res(127)),
                         ("python3 " + PF.PATCH_SCRIPT_NAME + " -p1", _Res(1, "", "Hunk FAILED for a.py at line 3")),
                         ("python3 " + PF.PATCH_SCRIPT_NAME + " -p0", _Res(1, "", "can't find file to patch: a/a.py"))])
        out = asyncio.run(env.apply_patch("--- a/a.py\n+++ b/a.py\n"))
        self.assertTrue(out.startswith("exit=1"), out)
        self.assertIn("Hunk FAILED for a.py", out)


class GvisorFallbackTest(unittest.TestCase):
    def _env(self, table):
        from core.environments.gvisor_env import GvisorEnvironment
        env = GvisorEnvironment.__new__(GvisorEnvironment)
        env.tool, env.container_name, env.image = "docker", "c1", "python:3-slim"
        env._workdir = Path("/workspace")
        env.written = {}
        env.calls = []

        async def _ensure():
            return None

        async def write_file(path, content):
            env.written[str(path)] = content

        def _run_cmd(argv, stdin=None, timeout=None):
            env.calls.append(argv)
            for key, res in table:
                if key in argv:
                    return res
            return 1, "unexpected " + " ".join(argv)
        env._ensure = _ensure
        env.write_file = write_file
        env._run_cmd = _run_cmd
        return env

    def test_no_patch_binary_runs_the_python_applier_in_the_workdir(self):
        env = self._env([("patch", (127, "exec: patch: not found")), (PF.PATCH_SCRIPT_NAME, (0, "patching file a.py\n"))])
        out = asyncio.run(env.apply_patch("--- a/a.py\n+++ b/a.py\n"))
        self.assertTrue(out.startswith("exit=0"), out)
        self.assertEqual(env.written[PF.PATCH_SCRIPT_NAME], PF.PYTHON_PATCH_SCRIPT)
        self.assertEqual(env.written[PF.PATCH_FILE_NAME], "--- a/a.py\n+++ b/a.py\n")
        fallback_calls = [c for c in env.calls if PF.PATCH_SCRIPT_NAME in c]
        self.assertEqual(fallback_calls[0][:3], ["docker", "exec", "c1"])

    def test_neither_available(self):
        env = self._env([("patch", (127, "")), (PF.PATCH_SCRIPT_NAME, (127, "python3: not found"))])
        out = asyncio.run(env.apply_patch("--- a/a.py\n+++ b/a.py\n"))
        self.assertTrue(out.startswith("SANDBOX-ERROR: patch(1) unavailable"), out)

    def test_rc_126_also_triggers_python_fallback(self):
        env = self._env([("patch", (126, "OCI runtime exec failed: executable file not found in $PATH")),
                         (PF.PATCH_SCRIPT_NAME, (0, "patching file a.py\n"))])
        out = asyncio.run(env.apply_patch("--- a/a.py\n+++ b/a.py\n"))
        self.assertTrue(out.startswith("exit=0"), out)


if __name__ == "__main__":
    unittest.main()
