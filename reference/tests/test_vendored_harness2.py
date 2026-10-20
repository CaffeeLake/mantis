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

"""The vendored Harness2 core under reference/third_party/harness2.

What is pinned: the tree is upstream commit VENDORED_SHA plus exactly the
patches in patches/ (reproducible by scripts/vendor_harness2.py), it imports
nothing outside the standard library, it is reached only through the
evals.evolve._vendor shim, and that shim never lets upstream's environment
defaults into a Mantis process.
"""

from __future__ import annotations

import ast
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import yaml

_REPO = Path(__file__).resolve().parent.parent.parent
_REFERENCE = _REPO / "reference"
_SCRIPTS = _REFERENCE / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import vendor_harness2  # noqa: E402
from evals.evolve import _vendor  # noqa: E402

_VENDOR_DIR = _vendor.VENDOR_DIR
_PACKAGE_DIR = _vendor.PACKAGE_DIR
_PATCHES = sorted((_VENDOR_DIR / "patches").glob("*.diff"))
_CORE_MODULES = (
    "spec", "tree", "context", "runmark", "ledger", "library", "selection",
    "proposer", "prompts", "recursion", "schedule", "judge", "report",
)
_MODIFIED_LINE = "# Modified by Mantis: see reference/third_party/harness2/VENDOR.md"

# An upstream checkout to compare against (a clone of
# https://github.com/google-research/harness2 at the SHA in VENDOR.md), named
# by MANTIS_HARNESS2_UPSTREAM. Tests that need it skip when it is unset or
# does not exist; the vendored tree carries no path to anyone's checkout.
_UPSTREAM = Path(os.environ.get("MANTIS_HARNESS2_UPSTREAM") or "/nonexistent")
_HAVE_UPSTREAM = (_UPSTREAM / "harness2" / "core" / "prompts.py").is_file()
_HAVE_PATCH = shutil.which("patch") is not None


def _snapshot(root: Path) -> dict[str, bytes]:
    """Every file under root (minus bytecode caches) by relative path."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            out[p.relative_to(root).as_posix()] = p.read_bytes()
    return out


def _copy_vendored(dest: Path) -> Path:
    shutil.copytree(_VENDOR_DIR, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def _pristine_from_committed(dest: Path) -> Path:
    """Upstream as it was before the patches: the committed tree with them reversed."""
    shutil.copytree(_PACKAGE_DIR, dest / "harness2", ignore=shutil.ignore_patterns("__pycache__"))
    for p in _PATCHES:
        subprocess.run(
            ["patch", "-p1", "-R", "-F0", "--no-backup-if-mismatch", "-d", str(dest), "-i", str(p)],
            check=True, capture_output=True,
        )
    return dest


def _first_unique_context_line(patch: Path, tree: Path) -> tuple[str, str]:
    """(relative file, line) for a non-blank context line of `patch` that occurs once in the file."""
    rel = None
    for line in patch.read_text(encoding="utf-8").splitlines():
        if line.startswith("+++ b/"):
            rel = line[len("+++ b/"):].split("\t")[0]
            body = (tree / rel).read_text(encoding="utf-8").splitlines()
        elif rel and line.startswith(" ") and line.strip() and body.count(line[1:]) == 1:
            return rel, line[1:]
    raise AssertionError(f"{patch.name} has no unique non-blank context line")


def _run_in_fresh_interpreter(code: str, cwd: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HARNESS2_")}
    env["PYTHONPATH"] = str(_REFERENCE)
    env.pop("GOOGLE_CLOUD_LOCATION", None)
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd or _REFERENCE, env=env, capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise AssertionError(f"child failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


class ShimTest(unittest.TestCase):
    def test_shim_imports_every_core_module_from_the_vendored_tree(self):
        h2 = _vendor.load_harness2()
        self.assertTrue(Path(h2.__file__).is_relative_to(_PACKAGE_DIR), h2.__file__)
        self.assertIn("harness2.config", sys.modules)
        self.assertIn(str(_VENDOR_DIR), sys.path)
        self.assertEqual(1, sys.path.count(str(_VENDOR_DIR)))
        for name in _CORE_MODULES:
            mod = __import__(f"harness2.core.{name}", fromlist=["_"])
            self.assertTrue(Path(mod.__file__).is_relative_to(_PACKAGE_DIR), name)
        # Calling it again neither duplicates the path entry nor re-imports.
        self.assertIs(h2, _vendor.load_harness2())
        self.assertEqual(1, sys.path.count(str(_VENDOR_DIR)))

    def test_manifest_sha_matches_shim_constant(self):
        self.assertEqual(_vendor.VENDORED_SHA, _vendor.vendored_sha_from_manifest())
        self.assertRegex(_vendor.VENDORED_SHA, r"^[0-9a-f]{7,40}$")

    def test_refuses_to_load_when_harness2_env_is_set(self):
        with mock.patch.dict(os.environ, {"HARNESS2_SOLVER_MODEL": "x", "HARNESS2_RUNS_DIR": "/r"}):
            with self.assertRaises(RuntimeError) as cm:
                _vendor.load_harness2()
        msg = str(cm.exception)
        self.assertIn("HARNESS2_RUNS_DIR, HARNESS2_SOLVER_MODEL", msg)
        self.assertIn("explicitly", msg)
        # Unrelated variables are no reason to refuse.
        with mock.patch.dict(os.environ, {"HARNESS2X": "1", "MANTIS_HARNESS2": "1"}):
            _vendor.load_harness2()

    def test_refuses_harness2_env_injected_by_a_dotenv_in_cwd(self):
        """config.py loads ./.env with setdefault at import; the shim must catch that too,
        and leave neither the variables nor the tainted modules behind."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".env").write_text("HARNESS2_JUDGE_MODEL=foo\nUNRELATED_FROM_DOTENV=1\n")
            got = _run_in_fresh_interpreter(
                """
                import json, os, sys
                from evals.evolve._vendor import load_harness2
                err = None
                try:
                    load_harness2()
                except RuntimeError as e:
                    err = str(e)
                print(json.dumps({
                    "err": err,
                    "env_left": sorted(k for k in os.environ
                                       if k.startswith("HARNESS2_") or k == "UNRELATED_FROM_DOTENV"),
                    "modules_left": sorted(m for m in sys.modules if m.startswith("harness2")),
                }))
                """,
                cwd=Path(tmp),
            )
        self.assertIsNotNone(got["err"])
        self.assertIn("HARNESS2_JUDGE_MODEL", got["err"])
        self.assertIn("during import", got["err"])
        self.assertEqual([], got["env_left"])
        self.assertEqual([], got["modules_left"])

    def test_import_leaves_os_environ_untouched(self):
        """config.py setdefaults GOOGLE_CLOUD_LOCATION (and whatever ./.env holds); none of
        it may survive into the Mantis process."""
        got = _run_in_fresh_interpreter(
            """
            import json, os
            before = dict(os.environ)
            from evals.evolve._vendor import load_harness2
            h2 = load_harness2()
            import harness2.config as cfg
            after = dict(os.environ)
            print(json.dumps({
                "added": sorted(set(after) - set(before)),
                "changed": sorted(k for k in before if before[k] != after.get(k)),
                "cfg_region": cfg.VERTEX_REGION,
            }))
            """
        )
        self.assertEqual([], got["added"])
        self.assertEqual([], got["changed"])
        self.assertEqual("global", got["cfg_region"])  # upstream's own default still applies inside


class RegisterSurfaceTest(unittest.TestCase):
    def setUp(self):
        _vendor.load_harness2()
        from harness2.core import prompts
        self.prompts = prompts
        self.addCleanup(prompts._SURFACE_TEXTS.pop, "mantis", None)

    def test_register_surface_resolves_a_new_name(self):
        P = self.prompts
        with self.assertRaises(ValueError):
            P._st("mantis")
        P.register_surface("mantis", surface_md="# WHAT YOU CAN CHANGE\nadk", inv="inv {deliverable}",
                           dom="dom", tts_surface="tts")
        st = P._st("mantis")
        self.assertEqual(("# WHAT YOU CAN CHANGE\nadk", "inv {deliverable}", "dom", "tts"),
                         (st.surface_md, st.inv, st.dom, st.tts_surface))
        self.assertIsInstance(st, P._SurfaceText)
        # A downstream name may be re-registered (module reloads in tests do that).
        P.register_surface("mantis", surface_md="v2", inv="i", dom="d", tts_surface="t")
        self.assertEqual("v2", P._st("mantis").surface_md)
        # The builtins are untouched.
        self.assertIs(P._SURFACE_TEXTS["opencode"].surface_md, P._SURFACE_OPENCODE)
        self.assertIs(P._SURFACE_TEXTS["codex"].surface_md, P._SURFACE_CODEX)

    def test_register_surface_refuses_builtin_names(self):
        P = self.prompts
        for builtin in ("opencode", "codex"):
            before = P._SURFACE_TEXTS[builtin]
            with self.assertRaises(ValueError) as cm:
                P.register_surface(builtin, surface_md="x", inv="x", dom="x", tts_surface="x")
            self.assertIn(builtin, str(cm.exception))
            self.assertIs(before, P._SURFACE_TEXTS[builtin])

    def test_registered_surface_feeds_the_authoring_prompts(self):
        """The point of the hook: propose() renders for the new surface."""
        P = self.prompts
        from harness2.core.spec import BenchPrompts
        P.register_surface("mantis", surface_md="# WHAT YOU CAN CHANGE\nMANTIS-SURFACE",
                           inv="INV-FOR-MANTIS {deliverable} {domain} {levers}",
                           dom="DOM-FOR-MANTIS {deliverable} {domain} {levers}",
                           tts_surface="TTS-FOR-MANTIS")
        bp = BenchPrompts(work="audit a repository", deliverable_word="audit", domain_word="security",
                          inv_levers="- a lever", dom_levers="- b lever")
        text = P.propose(bp, "INV", "mantis", k=1, index=1)
        self.assertIn("INV-FOR-MANTIS audit security - a lever", text)
        self.assertIn("MANTIS-SURFACE", text)
        self.assertNotIn("DOM-FOR-MANTIS", text)
        self.assertIn("DOM-FOR-MANTIS audit security - b lever", P.propose(bp, "DOM", "mantis"))


class VendoredTreeTest(unittest.TestCase):
    def test_license_is_apache_2_and_every_file_keeps_upstream_header(self):
        text = (_VENDOR_DIR / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)
        self.assertEqual((_REPO / "LICENSE").read_bytes(), (_VENDOR_DIR / "LICENSE").read_bytes(),
                         "the vendored LICENSE is the standard Apache-2.0 text, same as ours")
        for py in sorted(_PACKAGE_DIR.rglob("*.py")):
            head = py.read_text(encoding="utf-8")[:1200]
            self.assertIn("Copyright 2026 Google LLC", head, py.name)
            self.assertIn("Licensed under the Apache License, Version 2.0", head, py.name)

    def test_patched_files_carry_the_section_4b_notice_and_only_they_do(self):
        patched = set()
        for p in _PATCHES:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("+++ b/"):
                    patched.add(line[len("+++ b/"):].split("\t")[0])
        self.assertEqual({"harness2/core/prompts.py"}, patched)
        for py in sorted(_PACKAGE_DIR.rglob("*.py")):
            rel = py.relative_to(_VENDOR_DIR).as_posix()
            top = py.read_text(encoding="utf-8").splitlines()[:20]
            has = any(_MODIFIED_LINE in line for line in top)
            self.assertEqual(rel in patched, has, rel)

    @unittest.skipUnless(_HAVE_PATCH, "`patch` not on PATH")
    def test_patches_reverse_apply_to_the_committed_tree(self):
        """The committed file is pristine upstream plus the patch, nothing more."""
        self.assertTrue(_PATCHES, "no patches/*.diff")
        for p in _PATCHES:
            r = subprocess.run(
                ["patch", "-p1", "-R", "-F0", "--dry-run", "-d", str(_VENDOR_DIR), "-i", str(p)],
                capture_output=True, text=True,
            )
            self.assertEqual(0, r.returncode, f"{p.name}: {r.stdout}{r.stderr}")

    @unittest.skipUnless(_HAVE_PATCH and _HAVE_UPSTREAM, "upstream copy or `patch` unavailable")
    def test_patches_apply_to_pristine_upstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(_UPSTREAM / "harness2", Path(tmp) / "harness2",
                            ignore=shutil.ignore_patterns("__pycache__"))
            for p in _PATCHES:
                r = subprocess.run(
                    ["patch", "-p1", "-F0", "--dry-run", "-d", tmp, "-i", str(p)],
                    capture_output=True, text=True,
                )
                self.assertEqual(0, r.returncode, f"{p.name}: {r.stdout}{r.stderr}")

    def test_vendored_code_imports_only_stdlib_and_harness2(self):
        stdlib = set(sys.stdlib_module_names)
        offenders = []
        for py in sorted(_PACKAGE_DIR.rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative: stays inside harness2
                        continue
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top not in stdlib and top != "harness2":
                        offenders.append(f"{py.relative_to(_VENDOR_DIR)}:{node.lineno} {name}")
        self.assertEqual([], offenders)

    def test_manifest_file_list_matches_the_tree_and_the_allowlist(self):
        text = (_VENDOR_DIR / "VENDOR.md").read_text(encoding="utf-8")
        start = text.index(vendor_harness2.FILES_BEGIN) + len(vendor_harness2.FILES_BEGIN)
        end = text.index(vendor_harness2.FILES_END)
        listed = [ln.strip()[3:-1] for ln in text[start:end].splitlines() if ln.startswith("- `")]
        on_disk = sorted(p.relative_to(_VENDOR_DIR).as_posix() for p in _PACKAGE_DIR.rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
        self.assertEqual(listed, on_disk)
        self.assertEqual(listed, vendor_harness2.allowlist(_VENDOR_DIR))
        # Nothing outside the allowlist sneaks in: no benchmarks, substrate, cli, demo...
        self.assertTrue(all(f.startswith(("harness2/core/", "harness2/__init__.py", "harness2/config.py"))
                            for f in listed), listed)
        # And the manifest documents what was left out and why.
        for excluded in ("benchmarks/", "substrate/", "cli/", "install.py", "preflight.py", "demo.py"):
            self.assertIn(excluded, text)

    def test_nothing_under_third_party_is_collected_as_a_test(self):
        """`unittest discover -s tests -t .` only walks tests/; keep third_party free of
        anything that would be collected if that ever changed."""
        third_party = _REFERENCE / "third_party"
        self.assertEqual([], [p for p in third_party.rglob("test*.py")])
        self.assertFalse((third_party / "__init__.py").exists())
        self.assertFalse((_VENDOR_DIR / "__init__.py").exists())

    def test_precommit_mdformat_skips_the_vendored_tree(self):
        cfg = yaml.safe_load((_REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
        hooks = [h for repo in cfg["repos"] for h in repo["hooks"] if h["id"] == "mdformat"]
        self.assertEqual(1, len(hooks))
        self.assertEqual("^reference/third_party/", hooks[0].get("exclude"))

    def test_license_header_check_leaves_upstream_headers_alone(self):
        import check_license_headers
        self.assertIn("reference/third_party/", check_license_headers.EXCLUDE_PREFIXES)
        self.assertFalse(any("third_party" in str(p) for p in check_license_headers.tracked_source_files(_REPO)))


class VendorScriptTest(unittest.TestCase):
    def _run(self, src: Path, dest: Path, **kw) -> dict:
        return vendor_harness2.vendor(src, dest, _vendor.VENDORED_SHA, verify=False,
                                      out=io.StringIO(), **kw)

    @unittest.skipUnless(_HAVE_PATCH, "`patch` not on PATH")
    def test_idempotent_against_pristine_reconstructed_from_the_committed_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            pristine = _pristine_from_committed(Path(tmp) / "src")
            dest = _copy_vendored(Path(tmp) / "dest")
            summary = self._run(pristine, dest)
            self.assertEqual(_snapshot(_VENDOR_DIR), _snapshot(dest))
            self.assertEqual([p.name for p in _PATCHES], summary["patches"])
            self.assertEqual(_vendor.VENDORED_SHA, summary["sha"])

    @unittest.skipUnless(_HAVE_PATCH and _HAVE_UPSTREAM, "upstream copy or `patch` unavailable")
    def test_idempotent_against_the_upstream_copy(self):
        """Running the script over the real upstream checkout reproduces the committed tree
        byte for byte: no stray files, no .orig/.rej, same VENDOR.md."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = _copy_vendored(Path(tmp) / "dest")
            self._run(_UPSTREAM, dest)
            self.assertEqual(_snapshot(_VENDOR_DIR), _snapshot(dest))

    @unittest.skipUnless(_HAVE_PATCH, "`patch` not on PATH")
    def test_fails_loudly_and_leaves_dest_alone_when_a_patch_does_not_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            pristine = _pristine_from_committed(Path(tmp) / "src")
            # Drift upstream on a line the first patch uses as context.
            patch = _PATCHES[0]
            target_rel, anchor = _first_unique_context_line(patch, pristine)
            target = pristine / target_rel
            text = target.read_text(encoding="utf-8")
            target.write_text(text.replace(anchor, anchor + "  # upstream drifted", 1), encoding="utf-8")
            dest = _copy_vendored(Path(tmp) / "dest")
            before = _snapshot(dest)
            out, err = io.StringIO(), io.StringIO()
            with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
                rc = vendor_harness2.main(["--src", str(pristine), "--sha", _vendor.VENDORED_SHA,
                                           "--dest", str(dest), "--no-verify-sha"])
            self.assertEqual(1, rc, out.getvalue() + err.getvalue())
            self.assertIn(patch.name, err.getvalue())
            self.assertIn("does not apply", err.getvalue())
            self.assertEqual(before, _snapshot(dest))

    @unittest.skipUnless(_HAVE_PATCH, "`patch` not on PATH")
    def test_removes_files_that_are_no_longer_in_the_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            pristine = _pristine_from_committed(Path(tmp) / "src")
            dest = _copy_vendored(Path(tmp) / "dest")
            (dest / "harness2" / "core" / "stale.py").write_text("x = 1\n")
            (dest / "harness2" / "substrate").mkdir()
            (dest / "harness2" / "substrate" / "base.py").write_text("x = 1\n")
            self._run(pristine, dest)
            self.assertEqual(_snapshot(_VENDOR_DIR), _snapshot(dest))

    def test_rejects_a_source_that_is_not_an_upstream_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = _copy_vendored(Path(tmp) / "dest")
            with self.assertRaises(vendor_harness2.VendorError) as cm:
                self._run(Path(tmp) / "nope", dest)
            self.assertIn("harness2/__init__.py", str(cm.exception))
            with self.assertRaises(vendor_harness2.VendorError):
                vendor_harness2.vendor(_VENDOR_DIR, dest, "NOT-A-SHA", verify=False, out=io.StringIO())

    def test_verify_sha_requires_the_checkout_root_at_that_commit(self):
        if shutil.which("git") is None:
            self.skipTest("git not on PATH")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "up"
            (repo / "sub").mkdir(parents=True)
            (repo / "f").write_text("1")
            git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-C", str(repo)]
            subprocess.run(git + ["init", "-q"], check=True, capture_output=True)
            subprocess.run(git + ["add", "f"], check=True, capture_output=True)
            subprocess.run(git + ["commit", "-q", "-m", "x"], check=True, capture_output=True)
            head = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True,
                                  text=True).stdout.strip()
            self.assertEqual(head, vendor_harness2.verify_sha(repo, head[:7]))
            with self.assertRaises(vendor_harness2.VendorError) as cm:
                vendor_harness2.verify_sha(repo, "0" * 7 if not head.startswith("0" * 7) else "f" * 7)
            self.assertIn("does not start with", str(cm.exception))
            with self.assertRaises(vendor_harness2.VendorError) as cm:
                vendor_harness2.verify_sha(repo / "sub", head[:7])  # a copy inside another repo
            self.assertIn("not the root", str(cm.exception))
            with self.assertRaises(vendor_harness2.VendorError) as cm:
                vendor_harness2.verify_sha(Path(tmp), head[:7])  # not a checkout at all
            self.assertIn("--no-verify-sha", str(cm.exception))

    def test_manifest_fields_are_rewritten_and_date_is_kept_at_the_same_sha(self):
        text = (_VENDOR_DIR / "VENDOR.md").read_text(encoding="utf-8")
        files = ["harness2/__init__.py", "harness2/core/x.py"]
        new = vendor_harness2.updated_manifest(text, sha="abcdef0", date="2030-01-02", files=files)
        self.assertIn("- **Commit:** `abcdef0`\n", new)
        self.assertIn("- **Vendored:** 2030-01-02\n", new)
        self.assertIn(f"{vendor_harness2.FILES_BEGIN}\n- `harness2/__init__.py`\n- `harness2/core/x.py`\n"
                      f"{vendor_harness2.FILES_END}", new)
        self.assertEqual("abcdef0", vendor_harness2.manifest_sha(new))
        self.assertEqual("2030-01-02", vendor_harness2.manifest_date(new))
        with self.assertRaises(vendor_harness2.VendorError):
            vendor_harness2.updated_manifest("no fields here\n", sha="abcdef0", date="d", files=[])


if __name__ == "__main__":
    unittest.main()
