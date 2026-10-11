#!/usr/bin/env python3
"""Mantis pull-request scan: the driver behind the GitHub Action.

Audits the files a pull request changes with the full Mantis pipeline, then
runs the deterministic check_change gate over the PR diff and reports the
verdict as GitHub workflow annotations plus a step summary. The engine is
the MCP server's scan tier and gate, unchanged; this file only decides what
to scan, runs it synchronously, and renders the result for a CI job.

Design rules, in force everywhere in this file:

- The pull request checkout is DATA. Nothing in it is imported, executed,
  or installed; the pipeline reads it (read-only target, INV-4) and the
  sandbox is pinned to static-only so no PR-supplied code ever runs on the
  runner.
- No GitHub token. Annotations and the summary go through the workflow
  command and step-summary files; the only credentials in the job are
  whatever the chosen model provider needs, and only the pipeline
  subprocess reads them. The provider is the consumer's choice: any model
  id the pipeline accepts (the Gemini Enterprise Agent Platform, the Gemini
  API, Anthropic, or an OpenAI-compatible endpoint via --api-base).
- Fail closed on infrastructure, never on the model's mood. A changed file
  left unscanned (over the cap, over budget, timed out, or failed) floors
  the verdict at REVIEW with a reason that names it. No base commit,
  missing or rejected credentials, or nothing launching exit 3 -- a setup
  failure for the workflow's owner, never a verdict code -- so a broken
  installation is a red check rather than a silent pass. What the model
  found or did not find is reported as-is: the verdict is a function of
  findings only.
- Spend is bounded by this process, in the unit the invoice uses. The run
  has one token budget; each scan gets an equal share of what is left,
  counts every token at full price, and reports what it spent; a scan that
  reports nothing is charged its whole share; a file the budget cannot
  reach is not launched. Each request is also capped in input tokens, so
  one call cannot cost more than a known amount whatever the model's
  window. The day has a budget too, read from the platform's own meter
  before the run starts, and the run gets only what is left of it. The
  arithmetic lives here, not in the model, and nothing in the pull request
  can raise any of the numbers.

Exit codes: 0 the gate ran and --fail-on is satisfied; 1 the gate ran and
--fail-on is not; 3 setup failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.config import EXIT_AUTH  # noqa: E402
from scripts.mcp_server import (  # noqa: E402
    BLOCKING_SEVERITIES,
    VERDICT_BLOCK,
    VERDICT_PASS,
    VERDICT_REVIEW,
    MantisState,
    _scan_cmd,
    check_change,
    parse_unified_diff,
    reindex,
)

EXIT_OK = 0
EXIT_FAILED_GATE = 1
EXIT_INFRA = 3

FAIL_ON_CHOICES = ("none", "block", "review")

# GitHub renders at most this many annotations per level per step; the
# rest are only in the summary, which says so.
MAX_ANNOTATIONS_PER_LEVEL = 10

# Changed files the pipeline cannot audit and should not be billed for.
# Ignored files leave both the scan list and the gate's diff, so a
# docs-only pull request is a PASS rather than a "never scanned" REVIEW.
# Consumers replace the list with --ignore; an empty value ignores nothing.
DEFAULT_IGNORE = (
    "*.md", "*.rst", "*.txt", "*.lock", "*.sum",
    "*.svg", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.pdf",
    "*.woff", "*.woff2", "*.ttf", "*.eot",
)

# Pipeline exit statuses, mirroring scan_change's labels, plus the one the
# pipeline reserves for rejected credentials (core.config.EXIT_AUTH).
SCAN_DONE = "done"
SCAN_PAUSED = "paused_at_budget"
SCAN_FAILED = "finished_with_errors"
SCAN_TIMED_OUT = "timed_out"
SCAN_AUTH_REJECTED = "credentials_rejected"
# Never launched: the run's token budget had nothing left for it.
SCAN_OVER_BUDGET = "over_budget"

# A scan is not launched with less than this many tokens to spend. Measured
# at the default call cap, the smallest file costs about 250k: prompts and
# graph nodes before the file's size counts for anything. Below this a scan
# cannot get through its first pass; launching it would buy a pause, charged
# to the run, for a file the gate must still treat as unaudited.
MIN_SCAN_ALLOWANCE = 100_000

# Default per-request input ceiling handed to the pipeline
# (MANTIS_MAX_INPUT_TOKENS). The pipeline refuses a request above it before
# dispatch, whatever the model's window and even when litellm does not know
# the model, so one call has a cost ceiling that the run budget's one-call
# overshoot (the budget is checked after each response) cannot exceed.
DEFAULT_MAX_INPUT_TOKENS = 262_144

# Default run token budget. Tokens are counted at full price whether or not
# the provider served them from cache, so this bounds the worst-case bill
# per run rather than the expected one. Measured at the default call cap: an
# ordinary file (400 to 1,000 lines) costs 350k to 650k, a 2,000-line file
# 850k. Ten million lets a ten-file pull request finish whole and makes a
# twenty-five-file one pause its biggest files at their 400k share.
DEFAULT_MAX_TOKENS = 10_000_000

# Default budget for the project over any trailing 24 hours, read from the
# platform's own meter before a run starts. Sized so a repository's ordinary
# day never meets it and a misfiring trigger is stopped within hours rather
# than running until someone reads the invoice.
DEFAULT_DAILY_TOKEN_BUDGET = 100_000_000

# Environment the pipeline subprocess never needs and must not inherit:
# GitHub's own runtime tokens. The pipeline runs model-steered tool calls
# over attacker-authored code; nothing in that process should be able to
# mint an OIDC token or talk to the Actions service. Provider credentials
# pass through untouched -- which ones a run needs is the consumer's
# choice of model, so there is no fixed allowlist to apply.
_STRIPPED_ENV = frozenset({
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
    "ACTIONS_RUNTIME_URL",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_URL",
    "ACTIONS_RESULTS_URL",
    "ACTIONS_CACHE_URL",
    "ACTIONS_CACHE_SERVICE_V2",
})

# What a revision argument may look like. GitHub's event payload supplies
# full SHAs; a workflow_dispatch input is typed by a person. Either way the
# value reaches git as an argument, so it is validated before it does and
# every git call also terminates option parsing explicitly.
_REV_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


class InfraError(Exception):
    """The run cannot produce a verdict; the caller exits 3."""


def parse_ignore(spec: Optional[str]) -> List[str]:
    """Comma- or newline-separated fnmatch patterns; None means the default."""
    if spec is None:
        return list(DEFAULT_IGNORE)
    return [p.strip() for p in spec.replace("\n", ",").split(",") if p.strip()]


def is_ignored(rel_path: str, patterns: List[str]) -> bool:
    """Matches a pattern against the whole repo-relative path or its basename."""
    import fnmatch

    p = str(rel_path or "").replace("\\", "/")
    base = p.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(p, pat) or fnmatch.fnmatch(base, pat) for pat in patterns)


def filter_diff(diff_text: str, keep: set) -> str:
    """The diff with every per-file section whose new path is not in `keep`
    removed, so the gate and the scan list agree on the change set.

    A section's path is read from its header, which ends at the first hunk.
    Lines inside hunks are content the diffed file's author wrote, and a
    content line that reads `+++ b/other.py` must not rename the section.
    """
    out: List[str] = []
    section: List[str] = []
    section_path = ""
    in_header = False

    def flush() -> None:
        if section and section_path in keep:
            out.extend(section)

    for raw in str(diff_text or "").splitlines(keepends=True):
        if raw.startswith("diff --git "):
            flush()
            section, section_path, in_header = [raw], "", True
            continue
        if in_header and raw.startswith("@@"):
            in_header = False
        elif in_header and raw.startswith("+++ "):
            path = raw[4:].strip().split("\t")[0]
            if path.startswith("b/"):
                path = path[2:]
            section_path = "" if path == "/dev/null" else path
        section.append(raw)
    flush()
    return "".join(out)


def subprocess_env(env: Dict[str, str]) -> Dict[str, str]:
    """The environment the pipeline runs with: the step's, minus GitHub's
    own runtime tokens."""
    return {k: v for k, v in env.items() if k not in _STRIPPED_ENV}


# --- Inputs ------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True,
         text: bool = True) -> subprocess.CompletedProcess:
    # core.quotePath=false: file names reach us as bytes-as-UTF-8, not as
    # C-quoted strings that would never match what the pipeline is given.
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotePath=false", *args],
        capture_output=True, text=text, check=check,
    )


def _valid_rev(rev: str, allow_head: bool = False) -> bool:
    rev = str(rev or "").strip()
    if allow_head and rev == "HEAD":
        return True
    return bool(_REV_RE.match(rev))


def _commit_present(repo: Path, rev: str) -> bool:
    return _git(repo, "rev-parse", "--verify", "--quiet", "--end-of-options",
                f"{rev}^{{commit}}", check=False).returncode == 0


def pr_changes(repo: Path, base: str, head: str = "HEAD") -> Tuple[Dict[str, List[int]], str]:
    """What the PR changes: {repo-relative path: new-side changed lines},
    plus the unified diff text for the gate.

    Two git views are combined on purpose. `--name-status -z` is the
    authority on WHICH files changed: it does not depend on how git chose
    to render each file, so a `.gitattributes` line marking a file `-diff`,
    or a NUL byte that trips binary detection, cannot drop a file from the
    change set -- both are the pull request author's to arrange. The
    unified diff then supplies changed LINES where it has them; a file with
    no parsable hunk keeps an empty line list, which the gate treats as
    fail-closed (any finding in it is `direct`).

    On a pull_request event the checkout is GitHub's merge commit, so the
    two-dot diff against the base tip is exactly the PR's effect on the
    branch it targets. A shallow checkout may lack the base commit; one
    depth-1 fetch of that commit is enough, since a diff needs the two
    trees and no history. Deleted files are left out: nothing to scan.
    """
    base = str(base or "").strip()
    head = str(head or "HEAD").strip()
    if not base:
        raise InfraError("No base commit given: pass --base (the PR's base SHA).")
    if not _valid_rev(base):
        raise InfraError(f"--base must be a commit SHA (7 to 64 hex digits), got {base[:40]!r}.")
    if not _valid_rev(head, allow_head=True):
        raise InfraError(f"--head must be HEAD or a commit SHA, got {head[:40]!r}.")
    if not _commit_present(repo, base):
        _git(repo, "fetch", "--quiet", "--no-tags", "--depth=1", "--end-of-options",
             "origin", base, check=False)
    if not _commit_present(repo, base):
        raise InfraError(
            f"Base commit {base} is not in the checkout and could not be fetched; "
            "set actions/checkout fetch-depth: 0.")
    if not _commit_present(repo, head):
        raise InfraError(f"Head revision '{head}' is not in the checkout.")

    common = ["--no-color", "--no-ext-diff", "--no-textconv", "--find-renames",
              "--diff-filter=ACMR"]
    names = _git(repo, "diff", *common, "--name-status", "-z", "--end-of-options",
                 base, head, check=False, text=False)
    if names.returncode != 0:
        raise InfraError(f"git diff --name-status failed: "
                         f"{names.stderr.decode('utf-8', 'replace').strip()}")
    changed: Dict[str, List[int]] = {}
    fields = [f.decode("utf-8", "replace") for f in names.stdout.split(b"\0") if f]
    i = 0
    while i < len(fields):
        status = fields[i]
        if status[:1] in ("R", "C") and i + 2 < len(fields):
            changed.setdefault(fields[i + 2], [])  # the NEW path of a rename/copy
            i += 3
        elif i + 1 < len(fields):
            changed.setdefault(fields[i + 1], [])
            i += 2
        else:
            break

    hunks = _git(repo, "diff", *common, "--unified=0", "--end-of-options",
                 base, head, check=False)
    if hunks.returncode != 0:
        raise InfraError(f"git diff failed: {hunks.stderr.strip()}")
    for path, lines in parse_unified_diff(hunks.stdout).items():
        if path in changed:
            changed[path] = lines
    return changed, hunks.stdout


def fresh_state_dir(repo: Path, requested: str = "") -> Path:
    """Where this run keeps its database, index, and log: never inside the
    checkout, and never pre-existing.

    Everything under the checkout is the pull request author's: a committed
    symlink at a predictable path like `.mantis/scans/pr_scan.log` would
    turn the log into an append primitive on any file the runner can write,
    and a committed `.mantis/knowledge.db` or index cache would be trusted
    as this run's own memory. A fresh directory per run has neither
    problem. Defaults to the runner's temp directory.
    """
    base = Path(requested or os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()).resolve()
    repo = repo.resolve()
    if base == repo or repo in base.parents:
        raise InfraError(f"--state-dir must be outside the repository, got {base}.")
    state = base / f"mantis-pr-scan-{uuid.uuid4().hex[:12]}"
    state.mkdir(parents=True, exist_ok=False)
    return state


def credential_preflight(env: Optional[Dict[str, str]] = None, api_base: str = "",
                         model: str = "") -> Optional[str]:
    """Why no scan can succeed, or None.

    Provider-agnostic: the model id is resolved the way the pipeline resolves
    it (bare gemini-* and claude-* route by which credentials are present),
    then each provider is asked for its own requirements. A custom endpoint
    is trusted to carry its own authentication, so --api-base skips the
    check. Anything this cannot classify passes through; the pipeline's own
    authentication error is the backstop and is reported as a setup failure.
    """
    if api_base:
        return None
    env = os.environ if env is None else env
    try:
        from core.config import DEFAULT_MODEL, normalize_model_id
    except Exception:  # the reference tree is incomplete: let the pipeline say so
        return None
    with _scoped_environ(env):
        resolved = normalize_model_id(model or DEFAULT_MODEL)
        if resolved.startswith("vertex_ai/"):
            if any(env.get(v) for v in ("VERTEXAI_PROJECT", "GOOGLE_CLOUD_PROJECT")):
                return None
            try:
                import google.auth

                _, project = google.auth.default()
                if project:
                    return None
            except Exception:
                pass
            return (f"No Google Cloud project for '{resolved}': set GOOGLE_CLOUD_PROJECT "
                    "and provide credentials (workload identity or a service account).")
        try:
            import litellm

            check = litellm.validate_environment(model=resolved)
        except Exception:
            return None
        if check.get("keys_in_environment", True):
            return None
        missing = [str(k) for k in check.get("missing_keys") or []]
        if not missing:
            return None
        return (f"Missing credentials for '{resolved}': set "
                + ", ".join(missing) + " on the step.")


class _scoped_environ:
    """Runs a block with os.environ replaced by `env`, then restores it.

    normalize_model_id and litellm read os.environ directly; the driver
    takes an explicit env so tests can describe a runner without touching
    the process they run in.
    """

    def __init__(self, env: Dict[str, str]):
        self._env = dict(env)
        self._saved: Optional[Dict[str, str]] = None

    def __enter__(self):
        self._saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self._env)
        return self

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self._saved or {})
        return False


def select_scan_files(state: MantisState, changed: Dict[str, List[int]],
                      max_files: int) -> tuple[List[str], List[str]]:
    """Repo-relative files to scan, and the ones over the cap.

    Mirrors scan_change's admission rule: inside the repository after
    resolving symlinks, and a regular file. Order is the diff's order
    (git's: path-sorted), so the cap is deterministic for a given change.
    """
    rels: List[str] = []
    for f in changed:
        r = state.rel(f)
        if not r:
            continue
        probe = state.repo / r
        try:
            probe.resolve().relative_to(state.repo)
        except (OSError, ValueError):
            continue
        if probe.is_file():
            rels.append(r)
    rels = list(dict.fromkeys(rels))
    if max_files > 0 and len(rels) > max_files:
        return rels[:max_files], rels[max_files:]
    return rels, []


# --- The daily budget ----------------------------------------------------------------

# The platform's own count of tokens served to a project: input and output,
# every Gemini model, sampled each minute, a few minutes behind. The one
# source for "what has this project spent today" that does not depend on
# this code's own accounting or on any state between runs.
TOKEN_COUNT_METRIC = "aiplatform.googleapis.com/publisher/online_serving/token_count"
DAILY_WINDOW_SECONDS = 24 * 3600


class DailyMeterUnavailable(Exception):
    """The platform meter could not be read; the caller decides what that means."""


def _project_for_meter(env: Dict[str, str]) -> str:
    for key in ("VERTEXAI_PROJECT", "GOOGLE_CLOUD_PROJECT"):
        if env.get(key):
            return str(env[key])
    return ""


def tokens_served_today(env: Dict[str, str], now: Optional[float] = None,
                        session: Any = None) -> int:
    """Tokens the project consumed in the trailing 24 hours, from Cloud Monitoring.

    One request: the token_count metric summed across every model and both
    token types over the window, aligned to a single point. Raises
    DailyMeterUnavailable with a reason a workflow owner can act on when the
    project, credential, API, or role is missing; never guesses.
    """
    project = _project_for_meter(env)
    end = int(now if now is not None else time.time())
    start = end - DAILY_WINDOW_SECONDS
    try:
        if session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            with _scoped_environ(env):
                credentials, adc_project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/monitoring.read"])
            project = project or str(adc_project or "")
            session = AuthorizedSession(credentials)
    except Exception as exc:
        raise DailyMeterUnavailable(
            f"no Google Cloud credentials to read the project's token meter with: {exc}") from exc
    if not project:
        raise DailyMeterUnavailable(
            "no Google Cloud project: set GOOGLE_CLOUD_PROJECT on the step.")
    params = {
        "filter": f'metric.type = "{TOKEN_COUNT_METRIC}"',
        "interval.startTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        "interval.endTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end)),
        "aggregation.alignmentPeriod": f"{DAILY_WINDOW_SECONDS}s",
        "aggregation.perSeriesAligner": "ALIGN_SUM",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
        "view": "FULL",
    }
    url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries"
    try:
        resp = session.get(url, params=params, timeout=30)
    except Exception as exc:
        raise DailyMeterUnavailable(
            f"could not reach Cloud Monitoring for project '{project}': {exc}") from exc
    if resp.status_code == 403:
        raise DailyMeterUnavailable(
            f"Cloud Monitoring refused the read for project '{project}' (HTTP 403): grant "
            "the service account roles/monitoring.viewer and enable "
            "monitoring.googleapis.com, or set daily-token-budget to 0.")
    if resp.status_code != 200:
        raise DailyMeterUnavailable(
            f"Cloud Monitoring returned HTTP {resp.status_code} for project '{project}'.")
    try:
        body = resp.json()
    except ValueError as exc:
        raise DailyMeterUnavailable("Cloud Monitoring returned a non-JSON body.") from exc
    if not isinstance(body, dict):
        raise DailyMeterUnavailable("Cloud Monitoring returned an unexpected body.")
    total = 0
    for series in body.get("timeSeries") or []:
        for point in (series.get("points") or []) if isinstance(series, dict) else []:
            value = (point.get("value") or {}) if isinstance(point, dict) else {}
            raw = value.get("int64Value", value.get("doubleValue", 0))
            try:
                total += int(float(raw))
            except (TypeError, ValueError):
                raise DailyMeterUnavailable("Cloud Monitoring returned a non-numeric point.")
    return max(0, total)


def run_budget_for_today(max_tokens: int, daily_budget: int, used_today: int) -> int:
    """This run's token budget once the day's spend is taken into account.

    With a daily budget, a run may spend at most what is left of it; with a
    run budget too, the smaller of the two. Returns 0 when nothing is left,
    which the scan loop reads as "launch nothing". With no daily budget the
    run budget stands as given (0 meaning none).
    """
    if daily_budget <= 0:
        return max(0, max_tokens)
    left = max(0, daily_budget - max(0, used_today))
    if max_tokens <= 0:
        return left
    return min(max_tokens, left)


# --- The scan ----------------------------------------------------------------------


def scan_command(state: MantisState, rel_fp: str, max_llm_calls: int,
                 model: str, api_base: str = "", max_tokens: int = 0,
                 spend_report: str = "") -> List[str]:
    """scan_change's exact invocation plus the CI pins. Pure, for tests."""
    cmd = _scan_cmd(state, str(state.repo / rel_fp), max_llm_calls, model)
    # Static analysis only: the runner never executes anything the PR
    # supplied, whatever the action's own workflow.local.json says.
    cmd += ["--sandbox", "static-only"]
    if api_base:
        cmd += ["--api-base", api_base]
    if max_tokens > 0:
        # This file's share of the run budget, counted at full price: a
        # ceiling on discounted tokens would bound the bill only if every
        # token were served from cache at the discount assumed.
        cmd += ["--max-tokens", str(max_tokens), "--cache-discount", "1"]
    if spend_report:
        cmd += ["--spend-report", spend_report]
    return cmd


def allowance(budget: int, spent: int, files_left: int) -> int:
    """Tokens the next scan may spend: an equal share of what is left.

    A fair share with carry-over, so a cheap file's unspent share passes to
    the files after it and one expensive file cannot take the whole run.
    With no budget (0) there is no allowance to compute.
    """
    if budget <= 0 or files_left <= 0:
        return 0
    return max(0, budget - spent) // files_left


def read_spend_report(path: Path) -> Optional[Dict[str, Any]]:
    """The pipeline's own account of what it spent, or None if it left none."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("tokens", "fresh_tokens", "cached_tokens", "llm_calls"):
        v = data.get(key)
        out[key] = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else 0
    out["pause_trigger"] = str(data.get("pause_trigger") or "")
    return out


def charge(report: Optional[Dict[str, Any]], granted: int) -> int:
    """What a scan is charged against the run budget.

    Its own account when it left one, at the raw count (fresh plus cached:
    what a provider bills before any discount) or the budget-weighted
    count, whichever is larger. A scan that left no account, because it was
    killed or crashed before it could, is charged everything it was given:
    the budget may over-count, never under-count.
    """
    if report is None:
        return granted
    raw = report["fresh_tokens"] + report["cached_tokens"]
    return max(raw, report["tokens"])


def _classify(code: Optional[int], account: Optional[Dict[str, Any]] = None) -> str:
    if code is None:
        return SCAN_TIMED_OUT
    if code == 0:
        return SCAN_DONE
    if code == 2:
        # A graceful pause at a ceiling, with findings so far stored. Every
        # genuine pause names its trigger in the spend report; the pipeline
        # also exits 2 for a sandbox configuration error, which an account
        # with no trigger tells apart. No account at all (the report could
        # not be written) stays a pause: the conservative reading.
        if account is not None and not account.get("pause_trigger"):
            return SCAN_FAILED
        return SCAN_PAUSED
    if code == EXIT_AUTH:
        return SCAN_AUTH_REJECTED  # the provider refused the credentials
    return SCAN_FAILED


def _open_log(log_path: Path):
    """Append handle on a log that must be a regular file we created: no
    following symlinks, no reuse of anything that was already there."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(str(log_path), flags, 0o600), "ab")


def run_scans(state: MantisState, rels: List[str], max_llm_calls: int, model: str,
              api_base: str, per_file_timeout: float, log_path: Path,
              env: Optional[Dict[str, str]] = None, max_tokens: int = 0,
              max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS) -> List[Dict[str, Any]]:
    """Runs the pipeline over each file in turn, then restores the
    repo-rooted index (the pipeline re-roots the shared catalog at the one
    file it scanned, which would blind the gate's blast radius).

    Spend is bounded by the run: each scan is launched with an equal share
    of the tokens still unspent, told to count every token at full price,
    and asked to report what it spent; a scan that reports nothing is
    charged its whole share. When the share left for a file falls below
    MIN_SCAN_ALLOWANCE the file is not launched at all and is reported
    over budget. The driver does the arithmetic, not the model, and a
    pull request has no way to raise the budget.

    A scan the provider rejected for credentials -- known from the exit
    status the pipeline reserves for it, not from anything in the log --
    ends the loop: every later file would fail the same way, and the caller
    reports a setup failure rather than N failed scans.
    """
    results: List[Dict[str, Any]] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = subprocess_env(dict(os.environ if env is None else env))
    if max_input_tokens > 0:
        child_env["MANTIS_MAX_INPUT_TOKENS"] = str(int(max_input_tokens))
    else:
        child_env.pop("MANTIS_MAX_INPUT_TOKENS", None)
    spend_dir = log_path.parent / "spend"
    spend_dir.mkdir(mode=0o700, exist_ok=True)
    spent = 0
    with _open_log(log_path) as log:
        for i, rel_fp in enumerate(rels):
            granted = allowance(max_tokens, spent, len(rels) - i)
            if max_tokens > 0 and granted < MIN_SCAN_ALLOWANCE:
                log.write(f"\n=== mantis scan {rel_fp}: not launched, {granted} of "
                          f"{max_tokens} run tokens left ===\n".encode())
                results.append({"file": rel_fp, "returncode": None,
                                "status": SCAN_OVER_BUDGET, "seconds": 0.0,
                                "tokens": 0, "llm_calls": 0, "granted": granted,
                                "pause_trigger": ""})
                continue
            spend_path = spend_dir / f"{i}.json"
            cmd = scan_command(state, rel_fp, max_llm_calls, model, api_base,
                               max_tokens=granted, spend_report=str(spend_path))
            log.write(f"\n=== mantis scan {rel_fp} ===\n".encode())
            log.flush()
            started = time.time()
            code: Optional[int]
            try:
                proc = subprocess.run(
                    cmd, stdout=log, stderr=log, cwd=str(_REF_ROOT), env=child_env,
                    timeout=per_file_timeout if per_file_timeout > 0 else None)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                log.write(f"[pr_scan] timed out after {per_file_timeout:.0f}s\n".encode())
                code = None
            except Exception as exc:  # one bad file must not end the run
                log.write(f"[pr_scan] scan failed to launch: {exc!r}\n".encode())
                code = -1
            log.flush()
            account = read_spend_report(spend_path)
            # A process that never started spent nothing; everything else
            # that left no account is charged in full.
            cost = 0 if code == -1 else charge(account, granted)
            spent += cost
            results.append({
                "file": rel_fp,
                "returncode": code,
                "status": _classify(code, account),
                "seconds": round(time.time() - started, 1),
                "tokens": cost,
                "llm_calls": account["llm_calls"] if account else None,
                "granted": granted,
                "pause_trigger": account["pause_trigger"] if account else "",
            })
            if code == EXIT_AUTH:
                break
    try:
        reindex(state)
    except Exception:
        pass  # the gate reports a missing index as REVIEW on its own
    return results


# --- The verdict ---------------------------------------------------------------------


def _floor_review(report: Dict[str, Any], reason: str, action: str) -> None:
    if report["verdict"] == VERDICT_PASS:
        report["verdict"] = VERDICT_REVIEW
    report["reasons"].append(reason)
    if action not in report["next_actions"]:
        report["next_actions"].append(action)


def build_report(state: MantisState, diff: str, changed: Dict[str, List[int]],
                 scan_results: List[Dict[str, Any]], skipped: List[str],
                 log_path: Path, fail_on: str, max_tokens: int = 0,
                 daily: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """check_change over the change set, floored by what this run failed to cover.

    The gate gets both views: the diff text for changed lines, and the
    authoritative file list so a file git rendered without hunks is still
    gated (with no line information, which the gate treats as `direct`).
    """
    report = check_change(state, files=list(changed), diff=diff)
    report.setdefault("reasons", [])
    report.setdefault("next_actions", [])
    over_budget = [r["file"] for r in scan_results if r["status"] == SCAN_OVER_BUDGET]
    for r in scan_results:
        if r["status"] in (SCAN_FAILED, SCAN_TIMED_OUT):
            _floor_review(
                report,
                f"Scan of '{r['file']}' {r['status'].replace('_', ' ')}: its content is unaudited.",
                "Check the scan log and re-run the workflow.")
        elif r["status"] == SCAN_PAUSED:
            # The pipeline stops cleanly at a ceiling with whatever it found
            # so far stored, but it records no coverage for a cut-off
            # campaign, so the gate cannot treat the file as audited.
            trigger = r.get("pause_trigger") or ""
            if trigger == "token_budget":
                what, fix = "its share of the run token budget", "Raise max-tokens to let the scan finish."
            elif trigger == "llm_calls_limit":
                what, fix = "its LLM call ceiling", "Raise max-llm-calls to let the scan finish."
            else:
                what, fix = "a pipeline ceiling", "Check the scan log; raise max-tokens or max-llm-calls to let the scan finish."
            _floor_review(
                report,
                f"Scan of '{r['file']}' paused at {what} before finishing: coverage is partial.",
                fix)
    if over_budget:
        if daily and daily["run_budget"] < MIN_SCAN_ALLOWANCE:
            _floor_review(
                report,
                f"The project's daily token budget is spent ({int(daily['used_before_run'] or 0):,} "
                f"of {int(daily['budget']):,} tokens served in the last 24 hours), so "
                f"{len(over_budget)} changed file(s) were not scanned: " + ", ".join(over_budget) + ".",
                "Wait for the window to pass, or raise daily-token-budget.")
        else:
            _floor_review(
                report,
                f"{len(over_budget)} changed file(s) were not scanned because the run token "
                f"budget was spent: " + ", ".join(over_budget) + ".",
                "Raise max-tokens or split the pull request.")
    if skipped:
        _floor_review(
            report,
            f"{len(skipped)} changed file(s) over the max-files cap were not scanned: "
            + ", ".join(skipped) + ".",
            "Raise max-files or split the pull request.")
    report["scan"] = {
        "files": [r["file"] for r in scan_results],
        "results": scan_results,
        "skipped": skipped,
        "over_budget": over_budget,
        "log": str(log_path),
    }
    report["usage"] = usage_summary(scan_results, max_tokens, daily)
    report["fail_on"] = fail_on
    report["exit_code"] = exit_code_for(report["verdict"], fail_on)
    return report


def usage_summary(scan_results: List[Dict[str, Any]], max_tokens: int,
                  daily: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The run's model usage: what was charged against the budget, in total.

    `tokens` is the sum of what each scan was charged (its raw count, or its
    whole allowance when it left no account), so it is an upper bound on
    what the provider bills. `llm_calls` is None when any scan left no
    account, since a sum with a hole in it is not a count. `daily`, when a
    daily budget is in force, carries the platform's reading before the run
    and the run budget that resulted.
    """
    tokens = sum(int(r.get("tokens") or 0) for r in scan_results)
    calls: Optional[int] = 0
    for r in scan_results:
        if r["status"] == SCAN_OVER_BUDGET:
            continue
        c = r.get("llm_calls")
        if c is None:
            calls = None
            break
        calls += int(c)
    out = {
        "tokens": tokens,
        "llm_calls": calls,
        "max_tokens": int(max_tokens),
        "files_scanned": sum(1 for r in scan_results if r["status"] != SCAN_OVER_BUDGET),
        "files_over_budget": sum(1 for r in scan_results if r["status"] == SCAN_OVER_BUDGET),
    }
    if daily:
        out["daily"] = {"budget": int(daily["budget"]),
                        "used_before_run": int(daily["used_before_run"] or 0),
                        "run_budget": int(daily["run_budget"])}
    return out


def exit_code_for(verdict: str, fail_on: str) -> int:
    if fail_on == "block":
        return EXIT_FAILED_GATE if verdict == VERDICT_BLOCK else EXIT_OK
    if fail_on == "review":
        return EXIT_FAILED_GATE if verdict in (VERDICT_REVIEW, VERDICT_BLOCK) else EXIT_OK
    return EXIT_OK


# --- Rendering -----------------------------------------------------------------------


def _esc_data(s: Any) -> str:
    return str(s).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _esc_prop(s: Any) -> str:
    return _esc_data(s).replace(":", "%3A").replace(",", "%2C")


def annotation(level: str, message: str, file: str = "", line: Any = None,
               title: str = "") -> str:
    """One workflow command. Properties and data are escaped per GitHub's
    rules so a finding title cannot smuggle in a second command."""
    props = []
    if file:
        props.append(f"file={_esc_prop(file)}")
    if isinstance(line, int) and not isinstance(line, bool) and line > 0:
        props.append(f"line={line}")
    if title:
        props.append(f"title={_esc_prop(title)}")
    head = f"::{level}" + (" " + ",".join(props) if props else "")
    return f"{head}::{_esc_data(message)}"


def _severity(f: Dict[str, Any]) -> str:
    return str(f.get("severity") or f.get("priority") or "unrated").upper()


def _first_line(f: Dict[str, Any]) -> Optional[int]:
    for x in f.get("line_numbers") or []:
        if isinstance(x, int) and not isinstance(x, bool) and x > 0:
            return x
    return None


def render_annotations(report: Dict[str, Any]) -> List[str]:
    """Findings as annotations on the diff, infrastructure notes as plain ones.

    Levels: a direct HIGH/CRITICAL finding is an error when --fail-on would
    fail the job on it and a warning otherwise; other direct findings are
    warnings; same-file-elsewhere and caller-radius findings are notices.
    """
    strict = report.get("fail_on") in ("block", "review")
    buckets: Dict[str, List[str]] = {"error": [], "warning": [], "notice": []}
    for f in report.get("findings") or []:
        rel = f.get("relationship")
        sev = _severity(f)
        if rel == "direct" and sev in BLOCKING_SEVERITIES and strict:
            level = "error"
        elif rel == "direct":
            level = "warning"
        else:
            level = "notice"
        where = {"direct": "in the changed lines",
                 "unrelated_in_file": "in the same file, outside the changed lines",
                 "caller_radius": "in a caller of changed code"}.get(rel, "")
        msg = f"{f.get('title')} [{sev}, {f.get('status')}] {where}.".strip()
        if f.get("cwe"):
            msg += f" {f['cwe']}."
        buckets[level].append(annotation(
            level, msg, file=str(f.get("filepath") or ""), line=_first_line(f),
            title=f"Mantis: {sev}"))
    scan = report.get("scan") or {}
    for r in scan.get("results") or []:
        if r.get("status") in (SCAN_FAILED, SCAN_TIMED_OUT):
            buckets["warning"].append(annotation(
                "warning", f"Mantis scan of {r['file']} {r['status'].replace('_', ' ')}; "
                           f"see the scan log in the step summary.",
                title="Mantis: unscanned"))
    if scan.get("skipped"):
        buckets["warning"].append(annotation(
            "warning", f"{len(scan['skipped'])} changed file(s) not scanned (max-files cap): "
                       + ", ".join(scan["skipped"]),
            title="Mantis: unscanned"))
    if scan.get("over_budget"):
        buckets["warning"].append(annotation(
            "warning", f"{len(scan['over_budget'])} changed file(s) not scanned (token "
                       f"budget spent): " + ", ".join(scan["over_budget"]),
            title="Mantis: unscanned"))
    out: List[str] = []
    for level, lines in buckets.items():
        out.extend(lines[:MAX_ANNOTATIONS_PER_LEVEL])
        if len(lines) > MAX_ANNOTATIONS_PER_LEVEL:
            out.append(annotation(
                level, f"{len(lines) - MAX_ANNOTATIONS_PER_LEVEL} more {level}-level "
                       f"Mantis annotations are listed in the step summary."))
    return out


def _md(value: Any) -> str:
    """Untrusted text made inert for one line of markdown.

    File names come from the pull request; titles, reasons, function names,
    and statuses come from the model or the index after reading it. Any of
    them can carry backticks, brackets, HTML, or newlines, and the step
    summary renders markdown. The repository's sanitizer neutralizes all of
    those and scrubs credentials; this adds only the table-cell separator.
    """
    from core.llm_gateway import safe_markdown_span

    return safe_markdown_span(str(value if value is not None else "")).replace("|", "\\|")


def render_summary(report: Dict[str, Any]) -> str:
    verdict = report.get("verdict")
    icon = {VERDICT_PASS: "✅", VERDICT_REVIEW: "⚠️", VERDICT_BLOCK: "⛔"}.get(verdict, "")
    lines = [f"## {icon} Mantis PR scan: {verdict}", ""]
    if report.get("fail_on") == "none":
        lines.append("Advisory run: the verdict does not fail the job "
                     "(set `fail-on: block` or `review` to change that).")
        lines.append("")
    findings = report.get("findings") or []
    if findings:
        lines += ["### Findings", "",
                  "| Severity | Finding | Location | Status | Relation |",
                  "| --- | --- | --- | --- | --- |"]
        for f in findings:
            loc = str(f.get("filepath") or "")
            ln = _first_line(f)
            if ln:
                loc += f":{ln}"
            lines.append(f"| {_md(_severity(f))} | {_md(f.get('title'))} | `{_md(loc)}` "
                         f"| {_md(f.get('status'))} | {_md(f.get('relationship'))} |")
        lines.append("")
    reasons = report.get("reasons") or []
    if reasons:
        lines += ["### Why", ""] + [f"- {_md(r)}" for r in reasons] + [""]
    actions = report.get("next_actions") or []
    if actions:
        lines += ["### Next", ""] + [f"- {_md(a)}" for a in actions] + [""]
    blast = report.get("blast_radius") or []
    if blast:
        lines += ["### Changed functions", ""]
        for b in blast:
            callers = b.get("callers") or []
            tail = (f" — {len(callers)} caller(s): "
                    + ", ".join(f"`{_md(c.get('caller'))}`" for c in callers[:5])
                    + (" …" if len(callers) > 5 else "")) if callers else ""
            lines.append(f"- `{_md(b.get('function'))}` in `{_md(b.get('file'))}`{tail}")
        lines.append("")
    scan = report.get("scan") or {}
    results = scan.get("results") or []
    lines += ["### Scan", ""]
    if results:
        lines += ["| File | Result | Tokens | Calls | Seconds |", "| --- | --- | --- | --- | --- |"]
        for r in results:
            calls = r.get("llm_calls")
            lines.append(f"| `{_md(r['file'])}` | {_md(r['status'])} | {int(r.get('tokens') or 0):,} "
                         f"| {calls if calls is not None else '?'} | {r.get('seconds', '')} |")
    else:
        lines.append("No files were scanned.")
    usage = report.get("usage") or {}
    if usage:
        budget = int(usage.get("max_tokens") or 0)
        calls = usage.get("llm_calls")
        daily = usage.get("daily")
        if budget > 0:
            scope = f" of a {budget:,} budget"
        elif daily:
            scope = " (the daily budget left nothing for this run)"
        else:
            scope = " (no run budget)"
        lines.append("")
        lines.append(
            f"Model usage: {int(usage.get('tokens') or 0):,} tokens" + scope
            + (f", {calls} calls" if calls is not None else "")
            + ". Tokens are counted at full price, cached or not; a scan that left no "
              "account is charged its whole allowance.")
        if daily:
            used, cap = int(daily["used_before_run"]), int(daily["budget"])
            lines.append("")
            lines.append(
                f"Daily budget: {used:,} of {cap:,} tokens served to the project in the 24 "
                f"hours before this run; {max(0, cap - used):,} were left for it.")
    if scan.get("skipped"):
        lines.append("")
        lines.append("Not scanned (max-files cap): "
                     + ", ".join(f"`{_md(s)}`" for s in scan["skipped"]))
    if scan.get("over_budget"):
        lines.append("")
        lines.append("Not scanned (token budget spent): "
                     + ", ".join(f"`{_md(s)}`" for s in scan["over_budget"]))
    if report.get("ignored"):
        lines.append("")
        lines.append("Ignored (not code): "
                     + ", ".join(f"`{_md(s)}`" for s in report["ignored"]))
    if scan.get("log"):
        lines.append("")
        lines.append(f"Scan log: `{_md(scan['log'])}`")
    lines.append("")
    return "\n".join(lines)


def _append(path: str, text: str) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)


# --- Entry point ---------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mantis pull-request scan (GitHub Action driver)")
    p.add_argument("--repo", default=os.environ.get("GITHUB_WORKSPACE") or ".",
                   help="Checkout of the pull request (default: $GITHUB_WORKSPACE or cwd)")
    p.add_argument("--base", default="", help="Base commit of the pull request")
    p.add_argument("--head", default="HEAD", help="Head revision (default: HEAD)")
    p.add_argument("--model", default="", help="Model id; empty uses the workflow default")
    p.add_argument("--api-base", default="", help="Custom LLM API base URL")
    p.add_argument("--max-llm-calls", type=int, default=150,
                   help="LLM call ceiling per changed file (0 = workflow default)")
    p.add_argument("--max-tokens", default=str(DEFAULT_MAX_TOKENS),
                   help="Token budget for the whole run, e.g. 4M or 500k; every token "
                        "counts at full price, cached or not. Shared out equally over the "
                        "files still to scan; files the budget cannot reach are reported "
                        "unscanned. 0 removes the run budget (each scan keeps the "
                        f"pipeline's own ceiling). Default: {DEFAULT_MAX_TOKENS:,}")
    p.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS,
                   help="Input tokens one model request may carry; larger requests are "
                        "refused before dispatch. 0 leaves only the model's own window. "
                        f"Default: {DEFAULT_MAX_INPUT_TOKENS:,}")
    p.add_argument("--daily-token-budget", default=str(DEFAULT_DAILY_TOKEN_BUDGET),
                   help="Tokens the Google Cloud project may be served in any 24 hours, "
                        "read from Cloud Monitoring before the run; the run's budget is "
                        "cut to what is left, and nothing is launched when nothing is. "
                        "Needs roles/monitoring.viewer; a budget that cannot be checked is "
                        f"a setup error. 0 disables the check. Default: {DEFAULT_DAILY_TOKEN_BUDGET:,}")
    p.add_argument("--max-files", type=int, default=25,
                   help="Changed files scanned per run; the rest are reported unscanned")
    p.add_argument("--ignore", default=None,
                   help="Comma-separated fnmatch patterns of changed files to leave out "
                        "of the scan and the gate (default: common non-code files; "
                        "pass '' to ignore nothing)")
    p.add_argument("--per-file-timeout", type=float, default=1800.0,
                   help="Seconds before one file's scan is abandoned (0 = none)")
    p.add_argument("--fail-on", choices=FAIL_ON_CHOICES, default="none",
                   help="none: advisory; block: fail on BLOCK; review: fail on REVIEW or BLOCK")
    p.add_argument("--output-json", default="", help="Write the full report here")
    p.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY", ""),
                   help="Append the markdown summary here (default: $GITHUB_STEP_SUMMARY)")
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""),
                   help="Append verdict= and report= here (default: $GITHUB_OUTPUT)")
    p.add_argument("--no-annotations", action="store_true",
                   help="Do not print workflow-command annotations")
    p.add_argument("--state-dir", default="",
                   help="Parent directory for this run's database, index, and log; a fresh "
                        "subdirectory is created per run. Must be outside the repository "
                        "(default: $RUNNER_TEMP, else the system temp directory)")
    return p.parse_args(argv)


def _not_run(summary_path: str, message: str) -> int:
    """A setup failure: the error annotation, a one-line summary, exit 3."""
    print(annotation("error", message, title="Mantis: cannot scan"))
    _append(summary_path, f"## ⛔ Mantis PR scan: not run\n\n{_md(message)}\n")
    return EXIT_INFRA


def main(argv: Optional[List[str]] = None,
         env: Optional[Dict[str, str]] = None) -> int:
    args = parse_args(argv)
    env_map = dict(os.environ if env is None else env)
    repo = Path(args.repo).resolve()
    try:
        try:
            from core.budget import parse_token_budget

            max_tokens = parse_token_budget(args.max_tokens)
            daily_budget = parse_token_budget(args.daily_token_budget)
        except ValueError as exc:
            raise InfraError(f"--max-tokens / --daily-token-budget: {exc}")
        if args.max_input_tokens < 0:
            raise InfraError("--max-input-tokens cannot be negative.")
        if not repo.is_dir():
            raise InfraError(f"--repo '{args.repo}' is not a directory.")
        if _git(repo, "rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
            raise InfraError(f"--repo '{args.repo}' is not a git checkout.")
        reason = credential_preflight(env_map, api_base=args.api_base, model=args.model)
        if reason:
            raise InfraError(reason)
        # The day's spend, from the platform, before anything is launched. A
        # configured daily budget that cannot be checked is a setup failure:
        # running without the check would make the number a decoration.
        used_today: Optional[int] = None
        run_budget = max_tokens
        if daily_budget > 0:
            try:
                used_today = tokens_served_today(env_map)
            except DailyMeterUnavailable as exc:
                raise InfraError(
                    f"daily-token-budget is set but the project's token meter could not be "
                    f"read: {exc}")
            run_budget = run_budget_for_today(max_tokens, daily_budget, used_today)
        changed_all, diff = pr_changes(repo, args.base, args.head)
        state_dir = fresh_state_dir(repo, args.state_dir)
    except InfraError as exc:
        return _not_run(args.summary, str(exc))

    # All mutable state lives in the fresh directory, never in the checkout.
    state = MantisState(str(repo))
    state.home = state_dir
    state.db_path = str(state_dir / "knowledge.db")
    patterns = parse_ignore(args.ignore)
    ignored = [f for f in changed_all if is_ignored(f, patterns)]
    changed = {f: lines for f, lines in changed_all.items() if f not in ignored}
    if ignored:
        diff = filter_diff(diff, set(changed))
    rels, skipped = select_scan_files(state, changed, args.max_files)
    state.ensure_home()
    log_path = state.home / "pr_scan.log"
    daily = {"budget": daily_budget, "used_before_run": used_today,
             "run_budget": run_budget} if daily_budget > 0 else None
    if rels and daily and run_budget < MIN_SCAN_ALLOWANCE:
        # The day's budget is spent before this run: launch nothing, report
        # every file over budget. The index is still built, at no model cost,
        # so the gate can report the blast radius it can see.
        results = [{"file": r, "returncode": None, "status": SCAN_OVER_BUDGET,
                    "seconds": 0.0, "tokens": 0, "llm_calls": 0, "granted": 0,
                    "pause_trigger": ""} for r in rels]
        try:
            reindex(state)
        except Exception:
            pass
    else:
        results = run_scans(state, rels, args.max_llm_calls, args.model, args.api_base,
                            args.per_file_timeout, log_path, env=env_map,
                            max_tokens=run_budget,
                            max_input_tokens=args.max_input_tokens) if rels else []
    rejected = next((r for r in results if r["status"] == SCAN_AUTH_REJECTED), None)
    if rejected:
        return _not_run(
            args.summary,
            f"The model provider rejected the credentials (pipeline exit {EXIT_AUTH}) while "
            f"scanning '{rejected['file']}'. Check the credentials and model configured on "
            f"the workflow step; the provider's message is in the scan log: {log_path}")
    if rels and all(r["status"] == SCAN_FAILED and r["returncode"] == -1 for r in results):
        return _not_run(args.summary, f"Every scan failed to launch; see the scan log: {log_path}")

    if changed_all and not changed:
        # Every changed file matched an ignore pattern: nothing the pipeline
        # could audit, by the consumer's explicit choice. Say so and pass,
        # rather than echoing check_change's "no changed files" REVIEW.
        report = {
            "verdict": VERDICT_PASS,
            "reasons": [f"All {len(ignored)} changed file(s) match the ignore patterns; "
                        "nothing to audit."],
            "blast_radius": [], "findings": [], "next_actions": [],
            "scan": {"files": [], "results": [], "skipped": [], "over_budget": [],
                     "log": str(log_path)},
            "usage": usage_summary([], run_budget, daily),
            "fail_on": args.fail_on, "exit_code": EXIT_OK,
        }
    else:
        report = build_report(state, diff, changed, results, skipped, log_path,
                              args.fail_on, max_tokens=run_budget, daily=daily)
        if not changed_all:
            report["reasons"].insert(0, "The pull request changes no scannable files.")
    report["ignored"] = ignored
    report["state_dir"] = str(state_dir)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2, default=str),
                                          encoding="utf-8")
    if not args.no_annotations:
        for line in render_annotations(report):
            print(line)
    _append(args.summary, render_summary(report))
    _append(args.github_output,
            f"verdict={report['verdict']}\nreport={args.output_json}\n")
    usage = report.get("usage") or {}
    print(f"mantis: verdict {report['verdict']} "
          f"({len(report.get('findings') or [])} finding(s), "
          f"{usage.get('files_scanned', len(results))} file(s) scanned, "
          f"{usage.get('files_over_budget', 0)} over budget, {len(skipped)} skipped; "
          f"{int(usage.get('tokens') or 0):,} tokens)")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
