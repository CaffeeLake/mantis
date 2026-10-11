"""Tests for the pull-request scan driver behind the GitHub Action.

Nothing here reaches a model. The pipeline subprocess is replaced by a
python one-liner that writes the same spend rows and findings the real
scan would, so the gate, the fail-closed floors, the exit codes, and the
annotation rendering are exercised against a real git repository, a real
structural index, and a real findings database.
"""
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
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.pr_scan as pr_scan  # noqa: E402


BASE_APP = """\
def helper(x):
    return x * 2


def handler(y):
    return helper(y) + 1
"""

# One added line (new-side line 2) inside helper's body.
HEAD_APP = """\
def helper(x):
    x = x + 0
    return x * 2


def handler(y):
    return helper(y) + 1
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def make_pr_repo(tmp: str) -> tuple:
    """A repository with a base commit and a head commit that edits app.py,
    adds new.py and README.md, and deletes gone.py. Returns (repo, base_sha)."""
    repo = Path(tmp) / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "app.py").write_text(BASE_APP, encoding="utf-8")
    (repo / "gone.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text(HEAD_APP, encoding="utf-8")
    (repo / "new.py").write_text("def fresh():\n    return 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# docs\n", encoding="utf-8")
    (repo / "gone.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "head")
    return repo, base


# The stand-in for the pipeline: records the spend row the coverage ledger
# needs, optionally writes one finding, writes the spend report the driver
# asked for (unless told to leave none, as a killed process would), then
# exits with the given code. The driver's extra flags land in sys.argv.
_FAKE_SCAN = """\
import json, sys
sys.path.insert(0, {ref_root!r})
from core.cost import record_spend
from core.database import write_findings
record_spend({db!r}, "run-pr", {target!r}, "deep", tokens=5)
finding = {finding!r}
if finding:
    write_findings({db!r}, {target!r}, [finding], run_id="run-pr", status="")
argv = sys.argv[1:]
if "--spend-report" in argv and not {no_report!r}:
    path = argv[argv.index("--spend-report") + 1]
    granted = int(argv[argv.index("--max-tokens") + 1]) if "--max-tokens" in argv else 0
    spent = {tokens!r} if {tokens!r} is not None else 1000
    cached = {cached!r}
    json.dump({{"run_id": "run-pr", "tokens": spent, "fresh_tokens": spent - cached,
               "cached_tokens": cached, "cache_discount": 1.0, "llm_calls": 3,
               "graph_steps": 4, "elapsed_seconds": 0.5, "max_tokens": granted,
               "max_llm_calls": 1, "pause_trigger": {pause_trigger!r},
               "exit_code": {code}}}, open(path, "w"))
sys.exit({code})
"""


def fake_scan(finding=None, code=0, tokens=None, cached=0, pause_trigger="",
              no_report=False):
    def _cmd(state, target, max_llm_calls, model):
        return [sys.executable, "-c", _FAKE_SCAN.format(
            ref_root=str(pr_scan._REF_ROOT), db=state.db_path, target=target,
            finding=finding, code=code, tokens=tokens, cached=cached,
            pause_trigger=pause_trigger, no_report=no_report)]
    return _cmd


class DriverCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_pr_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo, self.base = make_pr_repo(self.tmp)
        self.out_json = Path(self.tmp) / "out" / "report.json"
        self.summary = Path(self.tmp) / "summary.md"
        self.gh_out = Path(self.tmp) / "gh_output"
        self.state_root = Path(self.tmp) / "state"
        self.env = {**os.environ, "GEMINI_API_KEY": "test-key"}

    def run_main(self, *extra, env=None, scan=None, daily=("--daily-token-budget", "0")):
        argv = ["--repo", str(self.repo), "--base", self.base,
                "--output-json", str(self.out_json),
                "--summary", str(self.summary),
                "--github-output", str(self.gh_out),
                "--state-dir", str(self.state_root),
                "--max-llm-calls", "1", *daily, *extra]
        buf = io.StringIO()
        cm = (mock.patch.object(pr_scan, "_scan_cmd", scan)
              if scan is not None else contextlib.nullcontext())
        with cm, contextlib.redirect_stdout(buf):
            rc = pr_scan.main(argv, env=self.env if env is None else env)
        return rc, buf.getvalue()

    def report(self):
        return json.loads(self.out_json.read_text(encoding="utf-8"))

    def commit_pr(self, files):
        """Adds another commit on top of HEAD with the given {path: content}
        (content None = symlink to the value in `links`), for adversarial
        pull-request shapes."""
        for rel, content in files.items():
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, tuple) and content[0] == "symlink":
                if p.exists() or p.is_symlink():
                    p.unlink()
                p.symlink_to(content[1])
            else:
                p.write_text(content, encoding="utf-8")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "adversarial")

    def scan_log(self):
        logs = list(self.state_root.glob("mantis-pr-scan-*/pr_scan.log"))
        self.assertEqual(len(logs), 1, logs)
        return logs[0].read_text(encoding="utf-8")


class TestChangeSet(DriverCase):
    def test_covers_edits_and_additions_not_deletions(self):
        changed, diff = pr_scan.pr_changes(self.repo, self.base)
        self.assertEqual(changed["app.py"], [2])
        self.assertIn("new.py", changed)
        self.assertIn("README.md", changed)
        self.assertNotIn("gone.py", changed)
        self.assertIn("+++ b/app.py", diff)

    def test_missing_or_unknown_base_is_infrastructure(self):
        with self.assertRaises(pr_scan.InfraError):
            pr_scan.pr_changes(self.repo, "")
        with self.assertRaises(pr_scan.InfraError):
            pr_scan.pr_changes(self.repo, "0" * 40)

    def test_revisions_are_validated_before_reaching_git(self):
        # A base or head is a SHA (or HEAD). Anything else -- in particular
        # anything that could be read as a git option -- is refused up front.
        for bad in ("--upload-pack=touch /tmp/pwned", "-", "main", "HEAD~1", "abc"):
            with self.subTest(base=bad), self.assertRaises(pr_scan.InfraError):
                pr_scan.pr_changes(self.repo, bad)
        with self.assertRaises(pr_scan.InfraError):
            pr_scan.pr_changes(self.repo, self.base, head="--output=/tmp/x")
        self.assertIn("app.py", pr_scan.pr_changes(self.repo, self.base, head="HEAD")[0])

    def test_gitattributes_cannot_hide_a_changed_file(self):
        # `evil.py -diff` makes git render the file as "Binary files differ"
        # with no +++ header or hunks. The file must still be in the change
        # set (line-less), so it is scanned and gated fail-closed.
        self.commit_pr({".gitattributes": "evil.py -diff\n",
                        "evil.py": "import os\nos.system(input())\n"})
        changed, diff = pr_scan.pr_changes(self.repo, self.base)
        self.assertIn("evil.py", changed)
        self.assertEqual(changed["evil.py"], [])
        self.assertNotIn("+++ b/evil.py", diff)

    def test_binary_detection_cannot_hide_a_changed_file(self):
        self.commit_pr({"blob.py": "x = 1\n\x00\nimport os\n"})
        changed, _ = pr_scan.pr_changes(self.repo, self.base)
        self.assertIn("blob.py", changed)

    def test_content_that_looks_like_a_diff_header_does_not_rename_the_section(self):
        # A new file whose first line is "++ b/README.md" appears in the diff
        # as "+++ b/README.md" inside the hunk. It is content, not a header.
        self.commit_pr({"evil.py": "++ b/README.md\nimport os\nos.system(input())\n"})
        changed, diff = pr_scan.pr_changes(self.repo, self.base)
        self.assertEqual(changed["evil.py"], [1, 2, 3])
        self.assertEqual(changed["README.md"], [1])  # README's own one-line change only
        kept = pr_scan.filter_diff(diff, {"evil.py", "app.py", "new.py"})
        self.assertIn("++ b/README.md\n", kept)       # the content line survives
        self.assertNotIn("+++ b/README.md\n@@", kept)  # README's real section is gone
        self.assertEqual(pr_scan.parse_unified_diff(kept)["evil.py"], [1, 2, 3])


class TestIgnoreAndFilter(unittest.TestCase):
    def test_default_patterns_cover_docs_and_images(self):
        pats = pr_scan.parse_ignore(None)
        self.assertTrue(pr_scan.is_ignored("docs/guide.md", pats))
        self.assertTrue(pr_scan.is_ignored("img/logo.png", pats))
        self.assertFalse(pr_scan.is_ignored("src/app.py", pats))
        self.assertFalse(pr_scan.is_ignored("Dockerfile", pats))

    def test_explicit_patterns_replace_defaults(self):
        pats = pr_scan.parse_ignore("vendor/*, *.gen.go")
        self.assertTrue(pr_scan.is_ignored("vendor/x.py", pats))
        self.assertTrue(pr_scan.is_ignored("a/b/types.gen.go", pats))
        self.assertFalse(pr_scan.is_ignored("README.md", pats))
        self.assertEqual(pr_scan.parse_ignore(""), [])

    def test_filter_diff_drops_whole_sections(self):
        diff = (
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n one\n+two\n"
            "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
            "@@ -1 +1,2 @@\n a\n+b\n"
        )
        kept = pr_scan.filter_diff(diff, {"x.py"})
        self.assertIn("+++ b/x.py", kept)
        self.assertNotIn("README.md", kept)
        self.assertEqual(pr_scan.parse_unified_diff(kept), {"x.py": [2]})


class TestPreflightAndCommand(unittest.TestCase):
    _BASE_ENV = {"PATH": os.environ.get("PATH", "")}

    def _preflight(self, env, model="", api_base=""):
        # The host may hold real Google credentials; the preflight must only
        # see the environment a runner would have, so ADC is forced absent.
        with mock.patch("google.auth.default", side_effect=Exception("no ADC")):
            return pr_scan.credential_preflight({**self._BASE_ENV, **env},
                                                api_base=api_base, model=model)

    def test_gemini_api_route_needs_its_key(self):
        self.assertIsNone(self._preflight({"GEMINI_API_KEY": "k"}, "gemini/gemini-3.8-flash"))
        reason = self._preflight({}, "gemini/gemini-3.8-flash")
        self.assertIn("GEMINI_API_KEY", reason)

    def test_anthropic_route_needs_its_key(self):
        self.assertIsNone(self._preflight({"ANTHROPIC_API_KEY": "k"}, "anthropic/claude-opus-5"))
        reason = self._preflight({}, "anthropic/claude-opus-5")
        self.assertIn("ANTHROPIC_API_KEY", reason)

    def test_openai_compatible_endpoint_skips_the_check(self):
        self.assertIsNone(self._preflight({}, "openai/glm-5.3", api_base="http://x/v1"))
        reason = self._preflight({}, "openai/glm-5.3")
        self.assertIn("OPENAI_API_KEY", reason)

    def test_vertex_route_needs_a_project(self):
        self.assertIsNone(self._preflight({"GOOGLE_CLOUD_PROJECT": "p"}, "vertex_ai/gemini-3.8-flash"))
        self.assertIsNone(self._preflight({"VERTEXAI_PROJECT": "p"}, "vertex_ai/claude-opus-5"))
        reason = self._preflight({}, "vertex_ai/claude-opus-5")
        self.assertIn("GOOGLE_CLOUD_PROJECT", reason)

    def test_bare_names_route_by_credentials(self):
        # A bare gemini-* id goes to the Gemini API when its key is present and
        # to Vertex otherwise, exactly as the pipeline resolves it.
        self.assertIsNone(self._preflight({"GEMINI_API_KEY": "k"}, "gemini-3.8-flash"))
        self.assertIsNone(self._preflight({"GOOGLE_CLOUD_PROJECT": "p"}, "gemini-3.8-flash"))
        reason = self._preflight({}, "gemini-3.8-flash")
        self.assertIsNotNone(reason)

    def test_preflight_leaves_the_process_environment_alone(self):
        before = dict(os.environ)
        self._preflight({"ANTHROPIC_API_KEY": "k"}, "anthropic/claude-opus-5")
        self.assertEqual(dict(os.environ), before)

    def test_scan_command_pins_static_sandbox(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        state = pr_scan.MantisState(tmp)
        cmd = pr_scan.scan_command(state, "a.py", 7, "some-model", api_base="http://x/v1")
        self.assertEqual(cmd[cmd.index("--sandbox") + 1], "static-only")
        self.assertEqual(cmd[cmd.index("--path-root") + 1], str(state.repo))
        self.assertEqual(cmd[cmd.index("--max-llm-calls") + 1], "7")
        self.assertEqual(cmd[cmd.index("--model") + 1], "some-model")
        self.assertEqual(cmd[cmd.index("--api-base") + 1], "http://x/v1")
        self.assertIn("--yes", cmd)


class TestSelection(DriverCase):
    def test_cap_is_deterministic_and_names_the_rest(self):
        state = pr_scan.MantisState(str(self.repo))
        changed = {"app.py": [2], "new.py": [1], "zzz/outside/../../etc": []}
        rels, skipped = pr_scan.select_scan_files(state, changed, 1)
        self.assertEqual(rels, ["app.py"])
        self.assertEqual(skipped, ["new.py"])
        rels, skipped = pr_scan.select_scan_files(state, changed, 0)
        self.assertEqual((rels, skipped), (["app.py", "new.py"], []))


class TestAnnotations(unittest.TestCase):
    def test_escaping_blocks_command_smuggling(self):
        line = pr_scan.annotation(
            "warning", "a\nb::error::pwned %", file="p,q:r.py", line=3, title="t:1,2")
        self.assertTrue(line.startswith("::warning file=p%2Cq%3Ar.py,line=3,title=t%3A1%2C2::"))
        self.assertIn("a%0Ab::error::pwned %25", line)
        # One physical line: a title or message cannot start a second command.
        self.assertEqual(line.count("\n"), 0)

    def _report(self, findings, fail_on="none"):
        return {"verdict": "REVIEW", "findings": findings, "fail_on": fail_on,
                "scan": {"results": [], "skipped": []}}

    def test_levels_follow_relationship_and_fail_on(self):
        hi = {"title": "sqli", "severity": "HIGH", "status": "reported",
              "filepath": "a.py", "line_numbers": [4]}
        direct = dict(hi, relationship="direct")
        elsewhere = dict(hi, relationship="unrelated_in_file")
        caller = dict(hi, relationship="caller_radius")
        lines = pr_scan.render_annotations(self._report([direct, elsewhere, caller]))
        self.assertTrue(lines[0].startswith("::warning file=a.py,line=4"))
        self.assertTrue(lines[1].startswith("::notice "))
        self.assertTrue(lines[2].startswith("::notice "))
        strict = pr_scan.render_annotations(self._report([direct], fail_on="block"))
        self.assertTrue(strict[0].startswith("::error file=a.py,line=4"))

    def test_per_level_cap_points_to_summary(self):
        many = [{"title": f"f{i}", "severity": "MEDIUM", "status": "reported",
                 "filepath": "a.py", "line_numbers": [i + 1], "relationship": "direct"}
                for i in range(13)]
        lines = pr_scan.render_annotations(self._report(many))
        warnings = [l for l in lines if l.startswith("::warning")]
        self.assertEqual(len(warnings), pr_scan.MAX_ANNOTATIONS_PER_LEVEL + 1)
        self.assertIn("3 more warning-level", warnings[-1])


class TestEndToEnd(DriverCase):
    """main() over a real repo with the pipeline subprocess stood in for."""

    def test_clean_scan_passes_and_writes_everything(self):
        rc, out = self.run_main(scan=fake_scan())
        self.assertEqual(rc, pr_scan.EXIT_OK, out)
        rep = self.report()
        self.assertEqual(rep["verdict"], "PASS", rep["reasons"])
        self.assertEqual(rep["findings"], [])
        self.assertEqual(sorted(rep["scan"]["files"]), ["app.py", "new.py"])
        self.assertEqual(rep["ignored"], ["README.md"])
        self.assertTrue(all(r["status"] == "done" for r in rep["scan"]["results"]))
        self.assertNotIn("::warning", out)
        self.assertNotIn("::error", out)
        self.assertIn("verdict=PASS", self.gh_out.read_text(encoding="utf-8"))
        summary = self.summary.read_text(encoding="utf-8")
        self.assertIn("Mantis PR scan: PASS", summary)
        self.assertIn("`README.md`", summary)
        # The blast radius saw the edit inside helper(): the index was
        # restored to repo-rooted after the per-file scans.
        self.assertTrue(any("helper" in str(b.get("function")) for b in rep["blast_radius"]),
                        rep["blast_radius"])

    def test_direct_high_finding_is_advisory_by_default_and_fails_on_block(self):
        finding = {"title": "Injection in helper", "description": "d",
                   "severity": "HIGH", "filepath": "app.py", "line_numbers": [2]}
        rc, out = self.run_main(scan=fake_scan(finding))
        self.assertEqual(rc, pr_scan.EXIT_OK)
        rep = self.report()
        self.assertEqual(rep["verdict"], "BLOCK")
        self.assertEqual(rep["findings"][0]["relationship"], "direct")
        self.assertIn("::warning file=app.py,line=2,title=Mantis%3A HIGH::", out)
        self.assertNotIn("::error", out)

        rc, out = self.run_main("--fail-on", "block", scan=fake_scan(finding))
        self.assertEqual(rc, pr_scan.EXIT_FAILED_GATE)
        self.assertIn("::error file=app.py,line=2", out)

    def test_same_file_debt_outside_change_is_review_not_block(self):
        # handler() lives at lines 6-7 of HEAD_APP: outside helper's span and
        # past the +/-3 window around changed line 2.
        finding = {"title": "Old debt in handler", "description": "d",
                   "severity": "CRITICAL", "filepath": "app.py", "line_numbers": [7]}
        rc, out = self.run_main("--fail-on", "block", scan=fake_scan(finding))
        self.assertEqual(rc, pr_scan.EXIT_OK, out)
        rep = self.report()
        self.assertEqual(rep["verdict"], "REVIEW")
        self.assertEqual(rep["findings"][0]["relationship"], "unrelated_in_file")
        self.assertIn("::notice ", out)

    def test_failed_scan_floors_review_and_fail_on_review_fails(self):
        rc, out = self.run_main(scan=fake_scan(code=1))
        self.assertEqual(rc, pr_scan.EXIT_OK)
        rep = self.report()
        self.assertEqual(rep["verdict"], "REVIEW")
        self.assertTrue(any("finished with errors" in r for r in rep["reasons"]), rep["reasons"])
        self.assertIn("title=Mantis%3A unscanned", out)
        rc, _ = self.run_main("--fail-on", "review", scan=fake_scan(code=1))
        self.assertEqual(rc, pr_scan.EXIT_FAILED_GATE)

    def test_budget_pause_is_partial_coverage_not_a_failure(self):
        # The pipeline exits 2 when it hits a ceiling: findings so far are
        # stored, nothing broke, but the file is not fully audited. That is
        # REVIEW with a reason naming the ceiling it hit -- not an error, and
        # not a PASS. The trigger comes from the pipeline's own spend report,
        # never from the log.
        rc, out = self.run_main("--fail-on", "block",
                                scan=fake_scan(code=2, pause_trigger="llm_calls_limit"))
        rep = self.report()
        self.assertTrue(all(r["status"] == "paused_at_budget" for r in rep["scan"]["results"]))
        self.assertEqual(rep["verdict"], "REVIEW", rep["reasons"])
        self.assertTrue(any("LLM call ceiling" in r for r in rep["reasons"]), rep["reasons"])
        self.assertTrue(any("max-llm-calls" in a for a in rep["next_actions"]))
        self.assertNotIn("unscanned", out)
        self.assertEqual(rc, pr_scan.EXIT_OK)
        rc, _ = self.run_main("--fail-on", "review",
                              scan=fake_scan(code=2, pause_trigger="token_budget"))
        rep = self.report()
        self.assertTrue(any("run token budget" in r for r in rep["reasons"]), rep["reasons"])
        self.assertTrue(any("max-tokens" in a for a in rep["next_actions"]))
        self.assertEqual(rc, pr_scan.EXIT_FAILED_GATE)
        # No account at all (the process died after deciding to pause): the
        # reason stays honest about not knowing which ceiling.
        rc, _ = self.run_main(scan=fake_scan(code=2, no_report=True))
        rep = self.report()
        self.assertTrue(any("a pipeline ceiling" in r for r in rep["reasons"]), rep["reasons"])
        # An account with no trigger is the pipeline's other exit 2, a
        # configuration error before any campaign ran: a failed scan.
        rc, _ = self.run_main(scan=fake_scan(code=2, pause_trigger=""))
        rep = self.report()
        self.assertTrue(all(r["status"] == "finished_with_errors" for r in rep["scan"]["results"]))
        self.assertEqual(rep["verdict"], "REVIEW")

    def test_timeout_floors_review(self):
        sleeper = lambda *a, **k: [sys.executable, "-c", "import time; time.sleep(5)"]
        rc, _ = self.run_main("--per-file-timeout", "0.5", "--max-files", "1", scan=sleeper)
        rep = self.report()
        self.assertEqual(rep["scan"]["results"][0]["status"], "timed_out")
        self.assertEqual(rep["verdict"], "REVIEW")
        self.assertEqual(rc, pr_scan.EXIT_OK)

    def test_max_files_cap_names_the_unscanned(self):
        rc, out = self.run_main("--max-files", "1", scan=fake_scan())
        rep = self.report()
        self.assertEqual(rep["scan"]["files"], ["app.py"])
        self.assertEqual(rep["scan"]["skipped"], ["new.py"])
        self.assertEqual(rep["verdict"], "REVIEW")
        self.assertTrue(any("new.py" in r for r in rep["reasons"]))
        self.assertIn("max-files cap", out)

    def test_all_ignored_passes_without_scanning(self):
        rc, out = self.run_main("--ignore", "*.py,*.md", scan=fake_scan())
        rep = self.report()
        self.assertEqual(rc, pr_scan.EXIT_OK)
        self.assertEqual(rep["verdict"], "PASS")
        self.assertEqual(rep["scan"]["results"], [])
        self.assertEqual(sorted(rep["ignored"]), ["README.md", "app.py", "new.py"])

    def test_missing_credentials_is_a_setup_failure_and_writes_nothing(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "GOOGLE_CLOUD_PROJECT", "VERTEXAI_PROJECT")}
        with mock.patch("google.auth.default", side_effect=Exception("no ADC")):
            rc, out = self.run_main("--model", "anthropic/claude-opus-5", env=env)
        self.assertEqual(rc, pr_scan.EXIT_INFRA)
        self.assertIn("::error title=Mantis%3A cannot scan::", out)
        self.assertIn("ANTHROPIC_API_KEY", out)
        self.assertFalse((self.repo / ".mantis").exists())
        self.assertFalse(self.state_root.exists())
        self.assertFalse(self.out_json.exists())
        self.assertIn("not run", self.summary.read_text(encoding="utf-8"))

    def test_rejected_credentials_are_a_setup_failure_not_a_verdict(self):
        # The pipeline exits with its reserved status when the provider
        # rejects the key. That is one setup problem, reported once; the loop
        # stops after the first file instead of burning the timeout N times.
        rejecting = lambda *a, **k: [sys.executable, "-c",
                                     f"import sys; sys.exit({pr_scan.EXIT_AUTH})"]
        rc, out = self.run_main(scan=rejecting)
        self.assertEqual(rc, pr_scan.EXIT_INFRA)
        self.assertIn("rejected the credentials", out)
        self.assertFalse(self.out_json.exists())
        self.assertIn("not run", self.summary.read_text(encoding="utf-8"))
        self.assertEqual(self.scan_log().count("=== mantis scan "), 1)

    def test_auth_banner_text_in_the_log_is_not_an_auth_failure(self):
        # The pipeline echoes file names and file contents into its log, so
        # any marker string there is the pull request author's to write. A
        # scan that prints the authentication banner but exits 1 is an
        # ordinary failed scan: advisory REVIEW, never a setup failure.
        self.commit_pr({"x [AUTHENTICATION ERROR] y.py": "x = 1\n"})
        forging = lambda *a, **k: [
            sys.executable, "-c",
            "import sys; sys.stderr.write('[AUTHENTICATION ERROR] Cloud / LLM "
            "Authentication Failed\\n'); sys.exit(1)"]
        rc, out = self.run_main(scan=forging)
        self.assertEqual(rc, pr_scan.EXIT_OK)
        self.assertEqual(self.report()["verdict"], "REVIEW")
        self.assertNotIn("rejected the credentials", out)
        self.assertGreater(self.scan_log().count("=== mantis scan "), 1)

    def test_ordinary_scan_failure_stays_advisory(self):
        # A crash that is NOT an authentication failure is a failed scan:
        # REVIEW with the file named, the job still green in advisory mode.
        crashing = lambda *a, **k: [sys.executable, "-c",
                                    "import sys; sys.stderr.write('Traceback: boom\\n'); sys.exit(1)"]
        rc, out = self.run_main(scan=crashing)
        self.assertEqual(rc, pr_scan.EXIT_OK)
        self.assertEqual(self.report()["verdict"], "REVIEW")
        self.assertNotIn("rejected the credentials", out)

    def test_nothing_is_written_under_the_checkout_even_with_a_planted_symlink(self):
        # Everything under the checkout is the pull request author's. A
        # committed symlink at the path a predictable log would use must not
        # turn the log into an append on the symlink's target.
        target = Path(self.tmp) / "victim.pth"
        self.commit_pr({".mantis/scans/pr_scan.log": ("symlink", str(target)),
                        ".mantis/knowledge.db": "not a database\n"})
        before = sorted(str(p.relative_to(self.repo)) for p in self.repo.rglob("*")
                        if ".git" not in p.parts)
        rc, _ = self.run_main(scan=fake_scan())
        self.assertEqual(rc, pr_scan.EXIT_OK)
        self.assertFalse(target.exists(), "the symlink target was written through")
        after = sorted(str(p.relative_to(self.repo)) for p in self.repo.rglob("*")
                       if ".git" not in p.parts)
        self.assertEqual(before, after, "the run created or removed files in the checkout")
        rep = self.report()
        self.assertTrue(rep["state_dir"].startswith(str(self.state_root)))
        self.assertFalse(Path(rep["state_dir"]).is_relative_to(self.repo))
        # The committed "database" was never consulted: a fresh one was made.
        self.assertTrue((Path(rep["state_dir"]) / "knowledge.db").stat().st_size > 100)

    def test_state_dir_inside_the_repository_is_refused(self):
        rc, out = self.run_main("--state-dir", str(self.repo / "sub"), scan=fake_scan())
        self.assertEqual(rc, pr_scan.EXIT_INFRA)
        self.assertIn("outside the repository", out)

    def test_hidden_file_is_still_scanned_and_gated(self):
        # .gitattributes -diff removes evil.py's hunks from the diff text. It
        # must still be scanned, and a HIGH finding in it must still BLOCK --
        # line-less, so the gate treats it as direct.
        self.commit_pr({".gitattributes": "evil.py -diff\n",
                        "evil.py": "import os\nos.system(input())\n"})
        finding = {"title": "Command injection", "description": "d",
                   "severity": "HIGH", "filepath": "evil.py", "line_numbers": [2]}
        rc, out = self.run_main("--fail-on", "block", scan=fake_scan(finding))
        rep = self.report()
        self.assertIn("evil.py", rep["scan"]["files"])
        self.assertEqual(rep["verdict"], "BLOCK", rep["reasons"])
        self.assertEqual(rc, pr_scan.EXIT_FAILED_GATE)
        self.assertIn("::error file=evil.py", out)

    def test_summary_neutralizes_hostile_names_and_titles(self):
        hostile = "a`<img src=x onerror=1>`.py"
        self.commit_pr({hostile: "x = 1\n"})
        finding = {"title": "ok\n## ✅ Mantis PR scan: PASS\n[click](https://x)",
                   "description": "d", "severity": "LOW", "filepath": hostile,
                   "line_numbers": [1]}
        self.run_main(scan=fake_scan(finding))
        summary = self.summary.read_text(encoding="utf-8")
        self.assertNotIn("<img", summary)
        self.assertNotIn("\n## ✅ Mantis PR scan: PASS", summary)
        self.assertNotIn("[click](", summary)
        level2 = [l for l in summary.splitlines() if l.startswith("## ")]
        self.assertEqual(len(level2), 1, level2)  # exactly one verdict heading
        # The real content is still there, legibly.
        self.assertIn("Mantis PR scan: REVIEW", summary)
        self.assertIn("a'", summary)

    def test_runtime_tokens_are_stripped_from_the_pipeline_environment(self):
        env = {**self.env, "GITHUB_TOKEN": "ghs_x", "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "t",
               "ACTIONS_RUNTIME_TOKEN": "r", "GH_TOKEN": "gh", "GH_ENTERPRISE_TOKEN": "ghe",
               "GEMINI_API_KEY": "keep-me"}
        probe = Path(self.tmp) / "env.json"
        dumping = lambda *a, **k: [sys.executable, "-c",
                                   f"import json,os; json.dump(dict(os.environ), open({str(probe)!r},'w'))"]
        self.run_main("--max-files", "1", env=env, scan=dumping)
        seen = json.loads(probe.read_text(encoding="utf-8"))
        for k in ("GITHUB_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_RUNTIME_TOKEN",
                  "GH_TOKEN", "GH_ENTERPRISE_TOKEN"):
            self.assertNotIn(k, seen)
        self.assertEqual(seen.get("GEMINI_API_KEY"), "keep-me")

    def test_unknown_base_is_infrastructure(self):
        argv = ["--repo", str(self.repo), "--base", "1" * 40,
                "--summary", str(self.summary)]
        with contextlib.redirect_stdout(io.StringIO()):
            rc = pr_scan.main(argv, env=self.env)
        self.assertEqual(rc, pr_scan.EXIT_INFRA)

    def test_launch_failure_everywhere_is_infrastructure(self):
        broken = lambda *a, **k: ["/nonexistent/mantis-pipeline"]
        rc, out = self.run_main(scan=broken)
        self.assertEqual(rc, pr_scan.EXIT_INFRA)
        self.assertIn("failed to launch", out)


class TestSpendArithmetic(unittest.TestCase):
    """The pure functions the run budget is made of."""

    def test_allowance_is_an_equal_share_of_what_is_left(self):
        self.assertEqual(pr_scan.allowance(1_000_000, 0, 4), 250_000)
        self.assertEqual(pr_scan.allowance(1_000_000, 100_000, 3), 300_000)  # carry-over
        self.assertEqual(pr_scan.allowance(1_000_000, 1_000_000, 2), 0)
        self.assertEqual(pr_scan.allowance(1_000_000, 1_500_000, 2), 0)      # never negative
        self.assertEqual(pr_scan.allowance(0, 0, 5), 0)                      # no budget
        self.assertEqual(pr_scan.allowance(100, 0, 0), 0)

    def test_charge_prefers_the_raw_count_and_defaults_to_the_grant(self):
        acct = {"tokens": 900, "fresh_tokens": 500, "cached_tokens": 400, "llm_calls": 2,
                "pause_trigger": ""}
        self.assertEqual(pr_scan.charge(acct, 10_000), 900)
        # A discounted figure larger than fresh+cached (a child run with a
        # different discount) is still honoured: the larger of the two.
        acct["tokens"] = 1_200
        self.assertEqual(pr_scan.charge(acct, 10_000), 1_200)
        self.assertEqual(pr_scan.charge(None, 10_000), 10_000)

    def test_spend_report_parsing_is_strict(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        p = Path(tmp) / "s.json"
        self.assertIsNone(pr_scan.read_spend_report(p))
        p.write_text("not json", encoding="utf-8")
        self.assertIsNone(pr_scan.read_spend_report(p))
        p.write_text("[1,2]", encoding="utf-8")
        self.assertIsNone(pr_scan.read_spend_report(p))
        p.write_text(json.dumps({"tokens": -5, "fresh_tokens": "x", "cached_tokens": True,
                                 "llm_calls": 2.0, "pause_trigger": None}), encoding="utf-8")
        acct = pr_scan.read_spend_report(p)
        self.assertEqual(acct, {"tokens": 0, "fresh_tokens": 0, "cached_tokens": 0,
                                "llm_calls": 2, "pause_trigger": ""})

    def test_scan_command_carries_the_grant_at_full_price(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        state = pr_scan.MantisState(tmp)
        cmd = pr_scan.scan_command(state, "a.py", 7, "m", max_tokens=12_345,
                                   spend_report="/tmp/x.json")
        self.assertEqual(cmd[cmd.index("--max-tokens") + 1], "12345")
        self.assertEqual(cmd[cmd.index("--cache-discount") + 1], "1")
        self.assertEqual(cmd[cmd.index("--spend-report") + 1], "/tmp/x.json")
        cmd = pr_scan.scan_command(state, "a.py", 7, "m", max_tokens=0)
        self.assertNotIn("--max-tokens", cmd)
        self.assertNotIn("--cache-discount", cmd)


class TestRunBudget(DriverCase):
    """The run budget end to end, with the pipeline stood in for. The PR
    fixture has two scannable files, app.py then new.py."""

    def results(self):
        return {r["file"]: r for r in self.report()["scan"]["results"]}

    def test_shares_are_equal_with_carry_over_and_charged_from_the_account(self):
        rc, _ = self.run_main("--max-tokens", "1000000", scan=fake_scan(tokens=100_000))
        res = self.results()
        self.assertEqual(res["app.py"]["granted"], 500_000)
        self.assertEqual(res["app.py"]["tokens"], 100_000)
        # new.py gets everything left, not its original half.
        self.assertEqual(res["new.py"]["granted"], 900_000)
        usage = self.report()["usage"]
        self.assertEqual(usage, {"tokens": 200_000, "llm_calls": 6, "max_tokens": 1_000_000,
                                 "files_scanned": 2, "files_over_budget": 0})
        self.assertEqual(self.report()["verdict"], "PASS")
        self.assertEqual(rc, pr_scan.EXIT_OK)
        self.assertIn("Model usage: 200,000 tokens of a 1,000,000 budget, 6 calls",
                      self.summary.read_text(encoding="utf-8"))

    def test_a_file_the_budget_cannot_reach_is_not_launched_and_floors_review(self):
        # The budget is checked after each call, so a scan can overshoot its
        # share by one call. Here the first scan does, leaving less than the
        # floor for the second file: that file is never launched (no
        # process, no spend), reported over budget, and the verdict is REVIEW
        # naming it.
        probe = Path(self.tmp) / "launched.txt"
        floor = pr_scan.MIN_SCAN_ALLOWANCE
        inner = fake_scan(tokens=floor + 5_000)

        def counting(state, target, max_llm_calls, model):
            with open(probe, "a") as fh:
                fh.write(target + "\n")
            return inner(state, target, max_llm_calls, model)

        rc, out = self.run_main("--max-tokens", str(2 * floor), scan=counting)
        res = self.results()
        self.assertEqual(res["app.py"]["status"], "done")
        self.assertEqual(res["app.py"]["granted"], floor)
        self.assertEqual(res["app.py"]["tokens"], floor + 5_000)
        self.assertEqual(res["new.py"]["status"], "over_budget")
        self.assertEqual(res["new.py"]["granted"], floor - 5_000)
        self.assertEqual(res["new.py"]["tokens"], 0)
        self.assertEqual(probe.read_text().count("\n"), 1, "new.py must not be launched")
        rep = self.report()
        self.assertEqual(rep["verdict"], "REVIEW")
        self.assertTrue(any("run token budget was spent" in r and "new.py" in r
                            for r in rep["reasons"]), rep["reasons"])
        self.assertTrue(any("max-tokens" in a for a in rep["next_actions"]))
        self.assertEqual(rep["scan"]["over_budget"], ["new.py"])
        self.assertIn("::warning title=Mantis%3A unscanned::1 changed file(s) not scanned "
                      "(token budget spent)", out)
        self.assertIn("Not scanned (token budget spent): `new.py`",
                      self.summary.read_text(encoding="utf-8"))
        self.assertEqual(rc, pr_scan.EXIT_OK)
        rc, _ = self.run_main("--fail-on", "review", "--max-tokens", str(2 * floor), scan=inner)
        self.assertEqual(rc, pr_scan.EXIT_FAILED_GATE)

    def test_a_scan_that_leaves_no_account_is_charged_its_whole_share(self):
        # Killed (timeout) or crashed before writing its report: the driver
        # cannot know what it spent, so it assumes all of it. Here the first
        # scan leaves nothing, so the second file's share is what remains of
        # the budget after the first's full grant, not after its real spend.
        calls = {"n": 0}
        silent = fake_scan(no_report=True)
        chatty = fake_scan(tokens=10)

        def alternating(state, target, max_llm_calls, model):
            calls["n"] += 1
            return (silent if calls["n"] == 1 else chatty)(state, target, max_llm_calls, model)

        self.run_main("--max-tokens", "1000000", scan=alternating)
        res = self.results()
        self.assertEqual(res["app.py"]["tokens"], 500_000)
        self.assertIsNone(res["app.py"]["llm_calls"])
        self.assertEqual(res["new.py"]["granted"], 500_000)
        usage = self.report()["usage"]
        self.assertEqual(usage["tokens"], 500_010)
        self.assertIsNone(usage["llm_calls"], "a sum with a hole in it is not a count")
        self.assertIn("| ? |", self.summary.read_text(encoding="utf-8"))

    def test_timeout_is_charged_in_full(self):
        sleeper = lambda *a, **k: [sys.executable, "-c", "import time; time.sleep(5)"]
        self.run_main("--per-file-timeout", "0.5", "--max-files", "1", "--max-tokens", "300000",
                      scan=sleeper)
        r = self.report()["scan"]["results"][0]
        self.assertEqual(r["status"], "timed_out")
        self.assertEqual(r["tokens"], 300_000)

    def test_a_launch_failure_costs_nothing(self):
        broken = lambda *a, **k: ["/nonexistent/mantis-pipeline"]
        self.run_main("--max-tokens", "1000000", scan=broken)
        # Every scan failed to launch: a setup failure, no report is written,
        # but the summary must not claim spend either.
        self.assertFalse(self.out_json.exists())

    def test_cached_tokens_are_charged_at_full_price(self):
        self.run_main("--max-tokens", "1000000", scan=fake_scan(tokens=1_000, cached=900))
        res = self.results()
        self.assertEqual(res["app.py"]["tokens"], 1_000)

    def test_no_run_budget_keeps_the_pipeline_ceiling_and_still_reports(self):
        rc, _ = self.run_main("--max-tokens", "0", scan=fake_scan(tokens=777))
        res = self.results()
        self.assertEqual(res["app.py"]["granted"], 0)
        self.assertEqual(res["app.py"]["tokens"], 777)
        usage = self.report()["usage"]
        self.assertEqual(usage["max_tokens"], 0)
        self.assertEqual(usage["tokens"], 1_554)
        self.assertIn("(no run budget)", self.summary.read_text(encoding="utf-8"))
        self.assertEqual(rc, pr_scan.EXIT_OK)

    def test_children_get_the_input_cap_and_the_grant(self):
        probe = Path(self.tmp) / "child.json"
        dumping = lambda *a, **k: [
            sys.executable, "-c",
            f"import json,os,sys; json.dump({{'env': dict(os.environ), 'argv': sys.argv}}, "
            f"open({str(probe)!r},'w'))"]
        self.run_main("--max-files", "1", "--max-tokens", "640000", "--max-input-tokens",
                      "4096", scan=dumping)
        seen = json.loads(probe.read_text(encoding="utf-8"))
        self.assertEqual(seen["env"].get("MANTIS_MAX_INPUT_TOKENS"), "4096")
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--max-tokens") + 1], "640000")
        self.assertEqual(argv[argv.index("--cache-discount") + 1], "1")
        spend = Path(argv[argv.index("--spend-report") + 1])
        self.assertTrue(str(spend).startswith(str(self.state_root)))
        self.assertFalse(str(spend).startswith(str(self.repo)))
        # 0 removes the cap from the child's environment even if the runner had one.
        self.run_main("--max-files", "1", "--max-input-tokens", "0",
                      env={**self.env, "MANTIS_MAX_INPUT_TOKENS": "99"}, scan=dumping)
        seen = json.loads(probe.read_text(encoding="utf-8"))
        self.assertNotIn("MANTIS_MAX_INPUT_TOKENS", seen["env"])

    def test_bad_budget_values_are_setup_errors(self):
        rc, out = self.run_main("--max-tokens", "lots", scan=fake_scan())
        self.assertEqual(rc, pr_scan.EXIT_INFRA)
        self.assertIn("max-tokens", out)
        rc, _ = self.run_main("--max-input-tokens", "-1", scan=fake_scan())
        self.assertEqual(rc, pr_scan.EXIT_INFRA)

    def test_defaults_are_the_documented_ones(self):
        args = pr_scan.parse_args([])
        self.assertEqual(args.max_tokens, str(pr_scan.DEFAULT_MAX_TOKENS))
        self.assertEqual(args.max_input_tokens, pr_scan.DEFAULT_MAX_INPUT_TOKENS)
        self.assertEqual(args.daily_token_budget, str(pr_scan.DEFAULT_DAILY_TOKEN_BUDGET))
        self.assertGreater(pr_scan.DEFAULT_MAX_TOKENS, 0)
        self.assertGreater(pr_scan.DEFAULT_MAX_INPUT_TOKENS, 0)
        self.assertGreater(pr_scan.DEFAULT_DAILY_TOKEN_BUDGET, pr_scan.DEFAULT_MAX_TOKENS)


class _FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeSession:
    def __init__(self, status=200, body=None, raise_exc=None):
        self.calls = []
        self._resp = _FakeResponse(status, body if body is not None else {})
        self._raise = raise_exc

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {}), timeout))
        if self._raise:
            raise self._raise
        return self._resp


class TestDailyMeter(unittest.TestCase):
    """Reading the platform's token meter: one request, strict parsing,
    actionable refusals."""

    ENV = {"GOOGLE_CLOUD_PROJECT": "p-1"}

    def test_sums_every_point_of_every_series(self):
        body = {"timeSeries": [
            {"metric": {"labels": {"type": "input"}}, "points": [{"value": {"int64Value": "1200"}}]},
            {"metric": {"labels": {"type": "output"}}, "points": [{"value": {"int64Value": "34"}},
                                                                  {"value": {"doubleValue": 6.0}}]},
        ]}
        s = _FakeSession(body=body)
        self.assertEqual(pr_scan.tokens_served_today(self.ENV, now=1_700_000_000, session=s), 1240)
        url, params, timeout = s.calls[0]
        self.assertEqual(url, "https://monitoring.googleapis.com/v3/projects/p-1/timeSeries")
        self.assertIn(pr_scan.TOKEN_COUNT_METRIC, params["filter"])
        self.assertEqual(params["interval.endTime"], "2023-11-14T22:13:20Z")
        self.assertEqual(params["interval.startTime"], "2023-11-13T22:13:20Z")
        self.assertEqual(params["aggregation.alignmentPeriod"], "86400s")
        self.assertEqual(params["aggregation.perSeriesAligner"], "ALIGN_SUM")
        self.assertEqual(params["aggregation.crossSeriesReducer"], "REDUCE_SUM")
        self.assertEqual(timeout, 30)

    def test_empty_series_is_zero_not_an_error(self):
        self.assertEqual(pr_scan.tokens_served_today(self.ENV, session=_FakeSession(body={})), 0)

    def test_refusals_are_named(self):
        with self.assertRaises(pr_scan.DailyMeterUnavailable) as c:
            pr_scan.tokens_served_today({}, session=_FakeSession())
        self.assertIn("GOOGLE_CLOUD_PROJECT", str(c.exception))
        with self.assertRaises(pr_scan.DailyMeterUnavailable) as c:
            pr_scan.tokens_served_today(self.ENV, session=_FakeSession(status=403))
        self.assertIn("roles/monitoring.viewer", str(c.exception))
        with self.assertRaises(pr_scan.DailyMeterUnavailable):
            pr_scan.tokens_served_today(self.ENV, session=_FakeSession(status=500))
        with self.assertRaises(pr_scan.DailyMeterUnavailable):
            pr_scan.tokens_served_today(self.ENV, session=_FakeSession(raise_exc=OSError("net")))
        with self.assertRaises(pr_scan.DailyMeterUnavailable):
            pr_scan.tokens_served_today(self.ENV, session=_FakeSession(body=ValueError("x")))
        with self.assertRaises(pr_scan.DailyMeterUnavailable):
            pr_scan.tokens_served_today(
                self.ENV, session=_FakeSession(body={"timeSeries": [{"points": [{"value": {"int64Value": "x"}}]}]}))

    def test_run_budget_for_today(self):
        f = pr_scan.run_budget_for_today
        self.assertEqual(f(4_000_000, 0, 0), 4_000_000)          # no daily budget
        self.assertEqual(f(0, 0, 0), 0)                          # no budgets at all
        self.assertEqual(f(4_000_000, 100_000_000, 1_000_000), 4_000_000)
        self.assertEqual(f(4_000_000, 100_000_000, 97_500_000), 2_500_000)  # cut to what is left
        self.assertEqual(f(4_000_000, 100_000_000, 100_000_000), 0)
        self.assertEqual(f(4_000_000, 100_000_000, 250_000_000), 0)        # never negative
        self.assertEqual(f(0, 100_000_000, 40_000_000), 60_000_000)        # daily alone bounds


class TestDailyBudgetInTheRun(DriverCase):
    """The daily budget end to end, with the meter stubbed."""

    def with_meter(self, used):
        return mock.patch.object(pr_scan, "tokens_served_today", lambda env, **kw: used)

    def test_a_spent_day_launches_nothing_and_floors_review(self):
        probe = Path(self.tmp) / "launched.txt"

        def counting(state, target, max_llm_calls, model):
            probe.write_text("launched")
            return fake_scan()(state, target, max_llm_calls, model)

        with self.with_meter(100_000_000):
            rc, out = self.run_main("--max-tokens", "4M", scan=counting,
                                    daily=("--daily-token-budget", "100M"))
        self.assertFalse(probe.exists(), "nothing may be launched on a spent day")
        rep = self.report()
        self.assertEqual(rep["verdict"], "REVIEW")
        self.assertEqual(sorted(rep["scan"]["over_budget"]), ["app.py", "new.py"])
        self.assertTrue(any("daily token budget is spent" in r for r in rep["reasons"]), rep["reasons"])
        self.assertTrue(any("daily-token-budget" in a for a in rep["next_actions"]))
        self.assertEqual(rep["usage"]["daily"],
                         {"budget": 100_000_000, "used_before_run": 100_000_000, "run_budget": 0})
        self.assertEqual(rep["usage"]["tokens"], 0)
        self.assertIn("Daily budget: 100,000,000 of 100,000,000", self.summary.read_text())
        self.assertIn("the daily budget left nothing for this run", self.summary.read_text())
        self.assertNotIn("(no run budget)", self.summary.read_text())
        self.assertIn("(token budget spent)", out)
        self.assertEqual(rc, pr_scan.EXIT_OK)

    def test_what_is_left_of_the_day_caps_the_run(self):
        with self.with_meter(99_000_000):
            self.run_main("--max-tokens", "4M", scan=fake_scan(tokens=10),
                          daily=("--daily-token-budget", "100M"))
        rep = self.report()
        self.assertEqual(rep["usage"]["max_tokens"], 1_000_000)
        self.assertEqual(rep["usage"]["daily"]["run_budget"], 1_000_000)
        self.assertEqual(rep["scan"]["results"][0]["granted"], 500_000)
        self.assertEqual(rep["verdict"], "PASS")
        self.assertIn("1,000,000 were left for it", self.summary.read_text())

    def test_an_ordinary_day_leaves_the_run_budget_alone(self):
        with self.with_meter(3_000_000):
            self.run_main("--max-tokens", "4M", scan=fake_scan(tokens=10),
                          daily=("--daily-token-budget", "100M"))
        rep = self.report()
        self.assertEqual(rep["usage"]["max_tokens"], 4_000_000)
        self.assertEqual(rep["verdict"], "PASS")

    def test_an_unreadable_meter_is_a_setup_failure_when_a_budget_is_set(self):
        def refuse(env, **kw):
            raise pr_scan.DailyMeterUnavailable("grant roles/monitoring.viewer")

        with mock.patch.object(pr_scan, "tokens_served_today", refuse):
            rc, out = self.run_main(scan=fake_scan(), daily=("--daily-token-budget", "100M"))
        self.assertEqual(rc, pr_scan.EXIT_INFRA)
        self.assertIn("roles/monitoring.viewer", out)
        self.assertFalse(self.out_json.exists())
        # Disabled explicitly: the meter is never consulted.
        with mock.patch.object(pr_scan, "tokens_served_today", refuse):
            rc, _ = self.run_main(scan=fake_scan(), daily=("--daily-token-budget", "0"))
        self.assertEqual(rc, pr_scan.EXIT_OK)

    def test_daily_budget_alone_bounds_an_unbounded_run(self):
        with self.with_meter(99_900_000):
            self.run_main("--max-tokens", "0", scan=fake_scan(tokens=10),
                          daily=("--daily-token-budget", "100M"))
        rep = self.report()
        self.assertEqual(rep["usage"]["max_tokens"], 100_000)
        self.assertEqual(rep["scan"]["results"][0]["granted"], 50_000)


if __name__ == "__main__":
    unittest.main()
