"""Campaign cost estimation and the observed-spend ledger.

WHAT THIS IS FOR
    Choosing a scan that should finish inside its token budget, instead of
    discovering at 10,000,000 tokens that the run covered 0.9% of the tree and
    stopped. The budget controller already stops the run safely; what it cannot
    do is tell the operator BEFOREHAND how much of the repository their budget
    actually buys.

WHY THE ARITHMETIC IS SHAPED THIS WAY
    Measured against the real trees before any of this was written:

        chromium    462,079 files, 3.0 GB, ~761,726,208 content tokens
        juice-shop    1,168 files,  14 MB,   ~3,512,671 content tokens

    Holding the 10M ceiling fixed and varying ONLY the assumed per-campaign
    overhead, the number of chromium files a file-by-file scan gets through is:

        5,000 tok/campaign  ->  1,999 files
       20,000 tok/campaign  ->    500 files
       50,000 tok/campaign  ->    200 files

    Those are exactly ceiling/overhead. The file contents contributed NOTHING:
    chromium's median file is 333 tokens against an overhead sixty times larger.

    So this module deliberately does NOT do byte accounting. Summing file sizes
    would be precise about the term that does not matter while guessing at the
    term that does. Cost is modelled as a flat per-campaign figure, and the file
    size is used only to catch the rare file so large it dominates its own
    campaign (chromium's largest is 20 MB, ~5.1M tokens -- half the default
    budget in a single file).

WHY THE DEFAULT IS A SEED AND NOT AN ANSWER
    The dominant term is turns-per-node, and nothing in the system has ever
    recorded it. DEFAULT_CAMPAIGN_TOKENS is therefore an assumption, not a
    measurement, and is marked as such everywhere it surfaces. `record_spend`
    exists so it stops being an assumption: once a deployment has run once,
    `observed_campaign_cost` returns ITS number, and the estimate describes the
    deployment's own workflow, models and repositories rather than ours.

    Accuracy is explicitly not the goal -- "it won't be perfect and we still
    might hit the guardrail, but if it's usually successful that's good enough".
    Every consumer must stay correct when the estimate is wrong, because it will
    be. The budget controller remains the thing that actually stops a run.

WHAT THIS MODULE MUST NEVER DO
    Refuse, cap, or discourage a scan. It reports numbers. The only place its
    output CHOOSES anything is scan_mode=auto, which is already defined as the
    mode that decides on the operator's behalf; every explicit mode runs exactly
    what was asked for and merely states what the budget covers.

RESUMPTION
    Spend is recorded per run_id and NOT carried forward into a resumed run.
    That is intentional and matches the resume contract: `--resume` starts a
    fresh budget so an operator can resume out of the box without recomputing
    anything. See `estimate_scan` and the README.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional


def _connect(db_path: str) -> sqlite3.Connection:
    """Knowledge-database connection with the shared busy timeout and journal
    settings (core.database.connect); imported lazily to keep module import
    order free of cycles."""
    from core.database import connect

    return connect(db_path)

# Characters per token. Matches core.config._CHARS_PER_TOKEN and the estimate at
# main.py's token fallback, deliberately: a second, different constant for the same
# quantity would make two parts of the system disagree about what a token costs.
_CHARS_PER_TOKEN = 4

# Seed cost for one campaign, in effective tokens, used until a deployment has
# observed its own. The first derivation multiplied a measured static floor
# (~4,729 tokens of prompts and tool declarations per single-turn pass) by an
# assumed five turns, giving 25k -- and the first live campaign measured
# 437,693: reasoning tokens and multi-turn tool loops dominate real cost, and
# the static floor captured neither. This is that observation, rounded up.
#
# It is still an ASSUMPTION -- one campaign, one model, one repository -- and
# it is used only until record_spend() has real data for this deployment,
# which replaces it after a single completed campaign.
DEFAULT_CAMPAIGN_TOKENS = 500_000

# Fraction of the budget available to campaigns. The remainder covers correlation,
# synthesis and reporting, which run after the scan loop and still have to fit.
_CAMPAIGN_BUDGET_RATIO = 0.85

# A campaign whose target file alone exceeds this share of the per-campaign cost is
# priced on its content instead of the flat rate. Below this, file size is noise
# against prompt overhead and including it adds false precision.
_OVERSIZED_FILE_RATIO = 0.5

# Ignore absurd observations when averaging. A campaign that died in its first
# second and recorded ~0 tokens is not evidence that campaigns are cheap.
_MIN_CREDIBLE_OBSERVATION = 100


class CostLedgerUnavailable(Exception):
    """Raised only by explicit ledger inspection, never by the estimator."""


@dataclass
class ScanEstimate:
    """What a budget buys. Every field is an estimate except `planned_campaigns`."""

    planned_campaigns: int
    affordable_campaigns: int
    cost_per_campaign: int
    spendable_tokens: int
    basis: str  # "observed" | "seeded"
    observations: int = 0
    oversized: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.affordable_campaigns >= self.planned_campaigns

    @property
    def coverage_fraction(self) -> float:
        if self.planned_campaigns <= 0:
            return 1.0
        return min(1.0, self.affordable_campaigns / self.planned_campaigns)

    def describe(self) -> str:
        """One line, stating numbers and never judging them.

        The basis is always named. An operator told "this budget covers 200 of
        462,079 campaigns" deserves to know whether that came from their own
        previous runs or from our guess, because the two justify very different
        levels of trust.
        """
        basis = (
            f"from {self.observations} observed campaign(s)"
            if self.basis == "observed"
            else "estimated, no observed runs yet"
        )
        if self.fits:
            return (
                f"Budget covers all {self.planned_campaigns:,} campaign(s) "
                f"at ~{self.cost_per_campaign:,} tokens each ({basis})."
            )
        return (
            f"Budget covers ~{self.affordable_campaigns:,} of "
            f"{self.planned_campaigns:,} campaign(s) "
            f"({self.coverage_fraction * 100:.1f}%) at ~{self.cost_per_campaign:,} "
            f"tokens each ({basis}). The run pauses there and resumes with --resume."
        )


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Creates the ledger table.

    Separate from campaign_artifacts on purpose. `list_files` surfaces every
    campaign_artifacts row for a run to the MODEL as a workspace listing, so
    filing cost rows there would inject operational accounting into agent context
    for no analytical benefit -- and grow it linearly in campaign count.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS campaign_spend (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            target TEXT,
            scan_mode TEXT,
            tokens INTEGER,
            llm_calls INTEGER,
            graph_steps INTEGER,
            elapsed_seconds REAL,
            metadata_json TEXT DEFAULT '{}'
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_campaign_spend_mode ON campaign_spend(scan_mode)"
    )


def _row_failed(metadata_json: Any) -> bool:
    try:
        meta = json.loads(metadata_json or "{}")
    except (TypeError, ValueError):
        return False
    return isinstance(meta, dict) and bool(meta.get("failed"))


def uncovered_files(db_path: str, files, model: str = "") -> Optional[list]:
    """The subset of `files` no recorded campaign has ever covered.

    With `model` given, covered means covered BY THAT MODEL: a file another
    model is known to have scanned is still uncovered for this one.
    Knowledge built by more than one model needs each to see every file,
    and the models differ in what they find. A campaign whose model is
    unknown (recorded before runs were, or imported from an export that
    did not say) counts as coverage for every model, as it always did:
    turning on provenance must not make an existing knowledge base re-sweep
    itself.

    A file counts as covered when some completed campaign's target was the
    file itself or an ancestor directory of it: scanning `repo/lib` covers
    everything under `lib`, and a whole-repository campaign covers the tree.
    Coverage is what this probe answers -- NOT "was this repository ever
    seen": a codebase whose `lib/` was scanned last month still has every
    file outside `lib/` uncovered, and skipping those silently is exactly
    the hole the caller uses this to close.

    A campaign that FAILED does not cover anything. Its row stays in the
    ledger (it ran the graph and cost real tokens, which the cost estimate
    wants to know), but a crashed scan looked at nothing a later run should
    trust; counting it would make `auto` skip the file forever and would let
    a change gate treat it as audited.

    Returns None when there is no ledger to consult or it cannot be read.
    The distinction is deliberate and load-bearing: a MISSING ledger (no
    file, no table) is a genuine first run and every file is uncovered, but
    an UNREADABLE one (corrupt, locked, permission-denied) is unknowable
    and returns None. The caller sweeps whatever this returns -- one
    campaign per file -- and a broken database must never be the reason a
    run becomes orders of magnitude larger than the operator expected.
    """
    if not db_path:
        return None  # No ledger to consult: coverage is unknowable.
    all_files = [str(f) for f in files]
    try:
        if not os.path.exists(db_path):
            return all_files
        conn = _connect(db_path)
        try:
            try:
                has_runs = model and conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'").fetchone()
                if has_runs:
                    # Rows of this model, and rows of no known model.
                    rows = conn.execute(
                        "SELECT s.target, s.metadata_json FROM campaign_spend s "
                        "LEFT JOIN runs r ON r.run_id = s.run_id "
                        "WHERE s.target IS NOT NULL AND (r.model = ? OR r.model IS NULL OR r.model = '')",
                        (str(model),)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT target, metadata_json FROM campaign_spend WHERE target IS NOT NULL"
                    ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    return all_files  # Ledger never written: genuine first run.
                raise
        finally:
            conn.close()
        targets = set()
        for target, metadata_json in rows:
            t = str(target or "").rstrip("/")
            if t and not _row_failed(metadata_json):
                targets.add(t)
        out = []
        for path in all_files:
            probe = path.rstrip("/")
            # Walk UP the directory chain so each check is a set lookup.
            # Comparing every file against every target is the quadratic
            # nobody notices until chromium's 462,079 files meet a ledger
            # with a few thousand rows.
            while probe and probe not in targets:
                parent = probe.rsplit("/", 1)[0] if "/" in probe else ""
                if parent == probe:
                    break
                probe = parent
            if not probe:
                out.append(path)
        return out
    except Exception:
        return None


def run_ledger(db_path: str, run_id: str) -> Optional[dict]:
    """What one run's campaigns did, as `{target: {"failed": bool}}`, the
    latest row per target winning (a retried campaign supersedes its failure).

    Returns None when the ledger cannot be read, `{}` when the run has no
    rows: a run that paused inside its first campaign recorded nothing, and
    resuming it means scanning everything, which is what an empty answer
    tells the caller. Never raises.
    """
    if not db_path or not run_id:
        return None
    try:
        if not os.path.exists(db_path):
            return {}
        conn = _connect(db_path)
        try:
            try:
                rows = conn.execute(
                    "SELECT target, metadata_json FROM campaign_spend "
                    "WHERE run_id = ? AND target IS NOT NULL ORDER BY id ASC",
                    (str(run_id),),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    return {}
                raise
        finally:
            conn.close()
        out: dict = {}
        for target, metadata_json in rows:
            t = str(target or "").rstrip("/")
            if t:
                out[t] = {"failed": _row_failed(metadata_json)}
        return out
    except Exception:
        return None


def run_scan_mode(db_path: str, run_id: str) -> str:
    """The scan mode a run's campaigns were recorded under, '' if unknown.

    The mode is not stored with the run, but every ledger row carries it,
    and a resumed run must plan the way the original did: a file-by-file
    sweep picked up without the flag would otherwise be re-resolved as
    `auto`, which plans one whole-target campaign for a small directory
    and so reruns the wrong thing. Latest row wins. Never raises.
    """
    if not db_path or not run_id:
        return ""
    try:
        if not os.path.exists(db_path):
            return ""
        conn = _connect(db_path)
        try:
            row = conn.execute(
                "SELECT scan_mode FROM campaign_spend WHERE run_id = ? "
                "AND scan_mode IS NOT NULL AND scan_mode != '' ORDER BY id DESC LIMIT 1",
                (str(run_id),),
            ).fetchone()
        finally:
            conn.close()
        return str(row[0]) if row else ""
    except Exception:
        return ""


def latest_run_id(db_path: str, target: str = "") -> str:
    """The most recent run recorded in this database, preferring runs whose
    ledger rows fall under `target` so `--resume` in one repository does not
    pick up a run over another. Falls back to the findings table for a run
    that never completed a campaign. '' when there is nothing to resume.
    Never raises.
    """
    if not db_path or not os.path.exists(db_path):
        return ""
    root = os.path.realpath(target) if target else ""
    try:
        conn = _connect(db_path)
        try:
            candidates: list = []
            for sql in (
                "SELECT run_id, target, MAX(timestamp) AS ts FROM campaign_spend "
                "WHERE run_id IS NOT NULL AND run_id != '' GROUP BY run_id",
                "SELECT run_id, '' AS target, MAX(timestamp) AS ts FROM findings "
                "WHERE run_id IS NOT NULL AND run_id != '' GROUP BY run_id",
            ):
                try:
                    candidates += [tuple(r) for r in conn.execute(sql).fetchall()]
                except sqlite3.OperationalError:
                    continue
        finally:
            conn.close()
    except Exception:
        return ""
    if not candidates:
        return ""

    def under_root(t: str) -> bool:
        if not root or not t:
            return not root
        try:
            rt = os.path.realpath(t)
            return rt == root or os.path.commonpath([rt, root]) == root
        except (ValueError, OSError):
            return False

    scoped = [c for c in candidates if under_root(str(c[1] or ""))] or candidates
    scoped.sort(key=lambda c: str(c[2] or ""), reverse=True)
    return str(scoped[0][0])


def record_spend(
    db_path: str,
    run_id: str,
    target: str,
    scan_mode: str,
    tokens: int,
    llm_calls: int = 0,
    graph_steps: int = 0,
    elapsed_seconds: float = 0.0,
    metadata: Optional[dict] = None,
) -> bool:
    """Records what one campaign actually cost. Returns True if stored.

    Never raises. This is bookkeeping: a scan that found a real vulnerability
    must not fail because the accounting table was unwritable (INV-6).
    """
    if not db_path or tokens is None:
        return False
    conn = None
    try:
        # closing(), not a bare `with`: sqlite3's connection context manager is a
        # TRANSACTION manager that commits or rolls back and leaves the handle open.
        # One leaked descriptor per campaign is 462,079 of them on a chromium sweep.
        conn = _connect(db_path)
        with conn:
            _ensure_table(conn)
            conn.execute(
                """
                INSERT INTO campaign_spend
                    (run_id, target, scan_mode, tokens, llm_calls, graph_steps,
                     elapsed_seconds, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_id),
                    str(target),
                    str(scan_mode),
                    int(tokens),
                    int(llm_calls or 0),
                    int(graph_steps or 0),
                    float(elapsed_seconds or 0.0),
                    json.dumps(metadata or {}),
                ),
            )
        return True
    except (sqlite3.Error, OSError, TypeError, ValueError):
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass


def observed_campaign_cost(
    db_path: str, scan_mode: str = "", limit: int = 200
) -> tuple[Optional[int], int]:
    """Mean observed cost of a campaign, as `(tokens, n_observations)`.

    Returns `(None, 0)` when there is nothing credible to average, which is the
    signal to fall back to the seed. Never raises.

    Prefers observations from the SAME scan mode: a file-by-file campaign reads
    one file and a cross-functional campaign ranges over a subsystem, so their
    costs are not drawn from the same population. Falls back to all modes rather
    than to the seed, since a real number from a different mode still beats a
    guess.
    """
    rows: list[Any] = []
    conn = None
    try:
        conn = _connect(db_path)
        with conn:
            _ensure_table(conn)
            cur = conn.cursor()
            # `tokens > 0` is stated alongside the credibility floor, not left to
            # it: multi-target campaigns stamp member coverage as zero-token
            # ledger rows (metadata member_of) purely so uncovered_files() sees
            # them covered. Those rows are bookkeeping, not observations --
            # averaging them in would drag the observed campaign cost toward
            # zero and inflate every affordability estimate. The floor happens
            # to exclude them today, but it is a tunable credibility threshold,
            # and the member-stamp exclusion must not depend on its value.
            if scan_mode:
                cur.execute(
                    "SELECT tokens FROM campaign_spend WHERE scan_mode = ? "
                    "AND tokens > 0 AND tokens >= ? "
                    "ORDER BY id DESC LIMIT ?",
                    (str(scan_mode), _MIN_CREDIBLE_OBSERVATION, int(limit)),
                )
                rows = [r[0] for r in cur.fetchall()]
            if not rows:
                cur.execute(
                    "SELECT tokens FROM campaign_spend WHERE tokens > 0 AND tokens >= ? "
                    "ORDER BY id DESC LIMIT ?",
                    (_MIN_CREDIBLE_OBSERVATION, int(limit)),
                )
                rows = [r[0] for r in cur.fetchall()]
    except (sqlite3.Error, OSError, TypeError, ValueError):
        return (None, 0)
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    clean = [
        int(r)
        for r in rows
        if isinstance(r, (int, float)) and r > 0 and r >= _MIN_CREDIBLE_OBSERVATION
    ]
    if not clean:
        return (None, 0)
    return (int(sum(clean) / len(clean)), len(clean))


def estimate_file_tokens(path: str) -> int:
    """Token estimate for a file's contents. 0 if it cannot be sized."""
    try:
        import os

        return int(os.path.getsize(path) // _CHARS_PER_TOKEN)
    except (OSError, TypeError, ValueError):
        return 0


def estimate_scan(
    targets: list[str],
    max_tokens: int,
    db_path: str = "",
    scan_mode: str = "",
) -> ScanEstimate:
    """Estimates how much of `targets` the token budget covers.

    Never raises: a broken estimate must not cost the scan, so every failure path
    degrades to the seeded cost with `basis="seeded"`.

    `max_tokens` is the WHOLE-RUN ceiling. A resumed run starts from zero by
    design, so this describes one process's worth of budget -- which is exactly
    what the operator is deciding about at the moment it is printed.
    """
    planned = len(targets or [])

    observed, n_obs = (None, 0)
    if db_path:
        observed, n_obs = observed_campaign_cost(db_path, scan_mode)

    if observed and observed > 0:
        cost, basis = int(observed), "observed"
    else:
        cost, basis = DEFAULT_CAMPAIGN_TOKENS, "seeded"

    try:
        spendable = int(max(0, int(max_tokens)) * _CAMPAIGN_BUDGET_RATIO)
    except (TypeError, ValueError):
        spendable = 0

    # A ceiling of 0 means the token budget is DISABLED, not that nothing is
    # affordable: without this, `spendable // cost` reads an unlimited budget
    # as "covers ~0 campaigns" -- the exact opposite of what the operator said.
    unbounded = False
    try:
        unbounded = int(max_tokens) <= 0
    except (TypeError, ValueError):
        pass

    # Name the files big enough to distort their own campaign. Rare, but at the
    # extreme a single file is half the default budget, and an operator who is
    # told "200 campaigns" deserves to know one of them is a 20 MB generated blob.
    oversized: list[str] = []
    threshold = cost * _OVERSIZED_FILE_RATIO
    for t in (targets or []):
        ft = estimate_file_tokens(t)
        if ft > threshold:
            oversized.append(t)
        if len(oversized) >= 10:
            break

    if unbounded:
        affordable = planned
    else:
        affordable = spendable // cost if cost > 0 else planned
    return ScanEstimate(
        planned_campaigns=planned,
        affordable_campaigns=int(affordable),
        cost_per_campaign=int(cost),
        spendable_tokens=int(spendable),
        basis=basis,
        observations=int(n_obs),
        oversized=oversized,
    )
