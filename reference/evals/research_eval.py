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

"""Researcher-stage eval: runs one researcher campaign over a seeded target.

Scores detection recall, cross-file recall, precision, and tokens per true
finding. The other stage benchmarks measure the stages that filter findings;
this one measures the stage that produces them. The toolset is the A/B
variable: same model, same instruction, same target, different tools.

How it works:

* The target (evals/research_target/) is copied into a disposable jail.
  ground_truth.json is excluded from the copy so the model cannot read it.
* The researcher runs with the production tool implementations and reports
  via report_findings; scoring reads the same database the pipeline uses.
* A finding matches a ground-truth entry only if it names one of the entry's
  files and mentions one of its match phrases. Duplicates of a matched entry
  do not hurt precision; deduplication is a different stage's job.
* cross_file_recall is reported separately because source-to-sink flows that
  cross files are what structural navigation tools should improve.
* Every result says how it was obtained: ``telemetry`` (windowed vs whole-file
  reads, navigation calls, loop-guard hints, tokens at each report),
  ``graded``/``status`` (a crashed run is ungraded, never a low score),
  ``cap_hit`` (stopped at --max-tokens/--max-tool-calls; the score is
  recall@cap), and ``config_fingerprint``/``ground_truth_version`` so that
  results measuring different things are never averaged together.

Usage:

    # Via the benchmark runner (model sweep, table output):
    python3 evals/run_eval.py --stage research --runs 3

    # Direct, single configuration, explicit toolset:
    python3 evals/research_eval.py --model vertex_ai/gemini-3.8-flash \\
        --effort low --toolset baseline --json-out /tmp/baseline.json

    # Two arms at equal spend (recall@cap):
    python3 evals/research_eval.py --toolset baseline   --max-tokens 1500000 ...
    python3 evals/research_eval.py --toolset structural --max-tokens 1500000 ...

    # The pipeline's file-by-file mode: one planner+researcher campaign per
    # file, each reading only its file (the rest of the tree through the
    # structural tools), all scored together. --files bounds the cost.
    python3 evals/research_eval.py --predecessor planner --fork \\
        --campaigns per-file --files gt+clean:7 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - direct CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DEFAULT_TARGET = Path(__file__).resolve().parent / "research_target"
GROUND_TRUTH_BASENAME = "ground_truth.json"

# How a cell campaigns over its target. "whole" is one campaign over the
# tree, the pipeline's whole-target scan; "per-file" is one campaign per
# source file, the pipeline's file-by-file scan, in which read_file opens
# only the scanned file and the rest of the tree is reached through the
# structural navigation tools. Both arms of a comparison run the same mode.
CAMPAIGN_MODES = ("whole", "per-file")
# Which files a per-file cell visits. A medium target has a couple of
# hundred files; the answer key names a few dozen. "gt+clean:N" visits the
# named files plus N the key does not name, chosen deterministically from
# the key, so recall is measured as a full sweep would measure it while
# precision is measured over a sample of clean files (and so reads high).
DEFAULT_CAMPAIGN_FILES = "gt+clean:7"

RESEARCH_PROMPT = (
    "You are the researcher stage of a security review. Audit the application "
    "source under the directory 'app/' for real, exploitable vulnerabilities.\n"
    "1. Call list_files('app') to enumerate the source files.\n"
    "2. Read the files and trace data flow from untrusted inputs (HTTP request "
    "parameters) to dangerous operations, including flows that cross file "
    "boundaries.\n"
    "3. Report every distinct vulnerability you confirm via report_findings, "
    "with a precise title, the file where the flaw lives (filepath), affected "
    "line numbers, and a description naming the mechanism end to end.\n"
    "Report only real flaws; do not pad with style issues or hypotheticals."
)

# The per-file campaign's briefings. {filepath} is substituted literally
# (never str.format: the path is repository-derived). The planner's replaces
# its stage instruction, which otherwise tells it to list and read the
# whole application; the researcher's plays the part of RESEARCH_PROMPT.
PLANNER_PROMPT_FILE = (
    "You are the Security Planner stage in the Mantis review pipeline. This "
    "campaign scans one file of the application under the directory 'app/': "
    "{filepath}. read_file opens only that file. Read it and record a focused, "
    "prioritized research plan with record_plan: which functions in the file "
    "to inspect, for which vulnerability classes, which callers and callees in "
    "other files the researcher must confirm through the structural tools, "
    "and why. Do not report findings yourself."
)
RESEARCH_PROMPT_FILE = (
    "You are the researcher stage of a security review. This campaign scans "
    "one file of the application under the directory 'app/': {filepath}. "
    "read_file opens only that file; the rest of the tree is reachable "
    "through find_symbol, find_callers, find_callees and get_function_boundary "
    "when they are available.\n"
    "1. Read the file and trace data flow from untrusted inputs (HTTP request "
    "parameters) to dangerous operations, following each flow into the "
    "callers and callees that live in other files.\n"
    "2. Report every distinct vulnerability you confirm via report_findings, "
    "with a precise title, the file where the flaw lives (filepath), affected "
    "line numbers, and a description naming the mechanism end to end.\n"
    "Report only real flaws; do not pad with style issues or hypotheticals."
)

# The A/B variable. Every name must be registered in tools.TOOLS;
# resolve_toolset refuses to run an arm with tools missing, because an arm
# that quietly loses tools is not that arm.
TOOLSETS: Dict[str, List[str]] = {
    "baseline": [
        "read_file",
        "write_file",
        "list_files",
        "get_summary",
        "get_threat_model",
        "get_plan",
        "report_findings",
        "get_findings",
    ],
    "structural": [
        "read_file",
        "write_file",
        "list_files",
        "get_summary",
        "get_threat_model",
        "get_plan",
        "report_findings",
        "get_findings",
        # Structural navigation over the deterministic tree-sitter catalog:
        "get_function_boundary",
        "find_symbol",
        "find_callers",
        "find_callees",
    ],
}


# --- Ground truth & scoring (pure, no LLM, unit-tested) -------------------------------


def load_research_ground_truth(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("findings") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("research ground truth must carry a non-empty 'findings' list")
    out: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"findings[{i}] is not an object")
        files = [str(f).replace("\\", "/").strip("/") for f in entry.get("files") or []]
        phrases = [str(p).lower() for p in entry.get("match_any") or [] if str(p).strip()]
        if not files or not phrases:
            raise ValueError(f"findings[{i}] needs non-empty 'files' and 'match_any'")
        out.append(
            {
                "id": str(entry.get("id") or f"gt-{i}"),
                "files": files,
                "match_any": phrases,
                "cross_file": bool(entry.get("cross_file")),
                "class": str(entry.get("class") or ""),
                "cwe": str(entry.get("cwe") or ""),
                # "seeded" entries are what recall counts. "decoy_*" entries
                # are places that look like the bug but are not one; a report
                # that matches one is a certain false positive. Absent = seeded.
                "kind": str(entry.get("kind") or "seeded"),
            }
        )
    return out


DECOY_KIND_PREFIX = "decoy_"


def is_decoy(gt_entry: Dict[str, Any]) -> bool:
    return str(gt_entry.get("kind") or "seeded").startswith(DECOY_KIND_PREFIX)


def _norm_path(path: Any) -> str:
    return str(path or "").replace("\\", "/").removeprefix("./").strip("/")


def _file_matches(finding_path: str, gt_file: str) -> bool:
    """Suffix match on path-component boundaries: 'app/database.py' matches
    'database.py'; 'db/database.py' matches 'database.py'; 'xdatabase.py'
    does not."""
    fp, gt = _norm_path(finding_path), _norm_path(gt_file)
    if not fp or not gt:
        return False
    return fp == gt or fp.endswith("/" + gt)


def match_finding(finding: Dict[str, Any], gt_entry: Dict[str, Any]) -> bool:
    """True when the finding names one of the entry's files and mentions one
    of its phrases. Phrases are checked over title, description, and cwe."""
    paths = [finding.get("filepath") or ""]
    paths.extend(str(cp).rsplit(":", 1)[0] for cp in finding.get("code_paths") or [])
    if not any(_file_matches(p, f) for p in paths for f in gt_entry["files"]):
        return False
    prose = " ".join(
        str(finding.get(key) or "") for key in ("title", "description", "cwe")
    ).lower()
    return any(phrase in prose for phrase in gt_entry["match_any"])


def score_research_findings(
    findings: Sequence[Dict[str, Any]],
    gt_entries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Scores reported findings against ground truth. Pure function.

    Ground truth has two kinds of entry. *Seeded* entries are the bugs: recall
    is the share of them detected. *Decoy* entries (``kind`` starting with
    ``decoy_``) are places that look like a bug and are not -- the same sink
    behind a sanitiser, the same code with no caller, the same pattern in a
    test or vendored file. A report that matches a decoy is a certain false
    positive, and ``decoy_fp_rate`` is the share of decoys reported.

    Precision is computed over unique results so duplicate reports of a
    matched entry neither help nor hurt. It is reported two ways because an
    unmatched report may be a hallucination or a real bug nobody seeded:
    ``precision`` (strict) counts every unmatched report as wrong;
    ``precision_lenient`` counts only decoy hits as wrong. A large gap
    between the two is a prompt to adjudicate the unmatched reports by hand.
    """
    seeded = [gt for gt in gt_entries if not is_decoy(gt)]
    decoys = [gt for gt in gt_entries if is_decoy(gt)]
    matched: Dict[str, List[int]] = {gt["id"]: [] for gt in gt_entries}
    unmatched_findings: List[Dict[str, Any]] = []
    decoy_finding_idx: set[int] = set()
    for idx, finding in enumerate(findings):
        seeded_hits = [gt["id"] for gt in seeded if match_finding(finding, gt)]
        decoy_hits = [gt["id"] for gt in decoys if match_finding(finding, gt)]
        if seeded_hits:
            # A report that names a real bug is a true report even if its
            # phrasing also brushes a decoy's file.
            for gt_id in seeded_hits:
                matched[gt_id].append(idx)
        elif decoy_hits:
            for gt_id in decoy_hits:
                matched[gt_id].append(idx)
            decoy_finding_idx.add(idx)
        else:
            unmatched_findings.append(
                {
                    "index": idx,
                    "title": str(finding.get("title") or "")[:120],
                    "filepath": _norm_path(finding.get("filepath")),
                }
            )

    detected = [gt for gt in seeded if matched[gt["id"]]]
    cross = [gt for gt in seeded if gt["cross_file"]]
    cross_detected = [gt for gt in cross if matched[gt["id"]]]
    decoys_hit = [gt for gt in decoys if matched[gt["id"]]]
    true_findings = len(findings) - len(unmatched_findings) - len(decoy_finding_idx)
    unique_results = len(detected) + len(decoys_hit) + len(unmatched_findings)
    unique_results_lenient = len(detected) + len(decoys_hit)

    return {
        "ground_truth_total": len(seeded),
        "detected": len(detected),
        "recall": round(len(detected) / len(seeded), 4) if seeded else 0.0,
        "cross_file_total": len(cross),
        "cross_file_detected": len(cross_detected),
        "cross_file_recall": round(len(cross_detected) / len(cross), 4) if cross else 0.0,
        "findings_reported": len(findings),
        "findings_true": true_findings,
        "findings_decoy": len(decoy_finding_idx),
        "decoys_total": len(decoys),
        "decoys_hit": len(decoys_hit),
        "decoy_fp_rate": round(len(decoys_hit) / len(decoys), 4) if decoys else 0.0,
        "precision": (
            round(len(detected) / unique_results, 4) if unique_results else 0.0
        ),
        "precision_lenient": (
            round(len(detected) / unique_results_lenient, 4)
            if unique_results_lenient else 0.0
        ),
        "duplicates": sum(max(0, len(v) - 1) for v in matched.values()),
        "per_entry": [
            {
                "id": gt["id"],
                "kind": gt.get("kind", "seeded"),
                "class": gt["class"],
                "cross_file": gt["cross_file"],
                "detected": bool(matched[gt["id"]]),
                "matched_findings": matched[gt["id"]],
            }
            for gt in gt_entries
        ],
        "unmatched_findings": unmatched_findings,
    }


def extract_event_tokens(event: Any) -> int:
    """Best-effort token count for one ADK/LiteLLM event. Never raises.

    Each LLM call reports its own usage, so summing per-event totals gives
    the campaign total. Missing or malformed metadata counts as zero.
    """
    try:
        usage = getattr(event, "usage_metadata", None)
        if usage is None:
            return 0
        total = getattr(usage, "total_token_count", None)
        if isinstance(total, int) and total > 0:
            return total
        prompt = getattr(usage, "prompt_token_count", None) or 0
        candidates = getattr(usage, "candidates_token_count", None) or 0
        return int(prompt) + int(candidates)
    except Exception:
        return 0


def parse_arm_spec(spec: str) -> "tuple[str, Optional[str], bool]":
    """'structural' -> ('structural', None, False); 'structural@planner' ->
    planned, fresh; 'structural@planner+fork' -> planned, forked. The shape
    after '@' is what sweep.py's arms carry beyond a toolset."""
    toolset, _, shape = str(spec or "").partition("@")
    toolset = toolset.strip() or "baseline"
    shape = shape.strip()
    if not shape or shape == "solo":
        return toolset, None, False
    predecessor, _, flag = shape.partition("+")
    if flag not in ("", "fork"):
        raise ValueError(f"unknown arm shape '{shape}': expected 'solo', '<stage>' or '<stage>+fork'")
    return toolset, predecessor.strip() or None, flag == "fork"


def campaign_shape(campaigns: str, files: str) -> str:
    """The part of a run's shape that names how it campaigned: '' for the
    whole-target mode every earlier result was measured in (so their
    fingerprints stand), '/per-file:<files>' otherwise."""
    mode = str(campaigns or "whole").strip()
    if mode not in CAMPAIGN_MODES:
        raise ValueError(f"unknown campaign mode '{mode}': expected one of {CAMPAIGN_MODES}")
    if mode == "whole":
        return ""
    return f"/{mode}:{str(files or DEFAULT_CAMPAIGN_FILES).strip()}"


def select_campaign_files(app_dir: Path, gt_entries: Sequence[Dict[str, Any]], selector: str) -> List[str]:
    """The files a per-file cell campaigns over, relative to ``app_dir``,
    sorted. ``all``: every source file. ``gt``: every file the answer key
    names (seeded and decoy alike) that exists. ``gt+clean:N``: those plus N
    files the key does not name, drawn by a generator seeded from the named
    list, so every arm and every repeat of a target visits the same files.
    """
    app_dir = Path(app_dir)
    all_files = sorted(
        str(p.relative_to(app_dir)).replace(os.sep, "/")
        for p in app_dir.rglob("*.py") if p.is_file()
    )
    sel = str(selector or DEFAULT_CAMPAIGN_FILES).strip().lower()
    if sel == "all":
        return all_files
    named = {_norm_path(f) for gt in gt_entries for f in (gt.get("files") or []) if _norm_path(f)}
    gt_files = [f for f in all_files if any(_file_matches(f, g) for g in named)]
    if sel == "gt":
        return gt_files
    m = re.fullmatch(r"gt\+clean:(\d+)", sel)
    if not m:
        raise ValueError(f"unknown file selector '{selector}': expected 'all', 'gt' or 'gt+clean:N'")
    clean = [f for f in all_files if f not in set(gt_files)]
    rng = random.Random(hashlib.sha256("\n".join(gt_files).encode("utf-8")).hexdigest())
    picked = rng.sample(clean, min(int(m.group(1)), len(clean)))
    return sorted(gt_files + picked)


def extract_event_billing(event: Any) -> "tuple[int, int]":
    """(uncached tokens, cached input tokens) for one event, from its usage
    metadata; (0, 0) when there is none. Never raises."""
    try:
        usage = getattr(event, "usage_metadata", None)
        if usage is None:
            return 0, 0
        total = int(getattr(usage, "total_token_count", None) or 0)
        if total <= 0:
            total = int(getattr(usage, "prompt_token_count", None) or 0) + int(
                getattr(usage, "candidates_token_count", None) or 0)
        cached = int(getattr(usage, "cached_content_token_count", None) or 0)
        return max(0, total - cached), cached
    except Exception:
        return 0, 0


# --- Abort guard ----------------------------------------------------------------------

# Shared with production execution (core.graph_loader / main.py) so both
# benchmark and production campaigns apply the same Harness²-inspired (c8)
# turn-1 silent-abort and pseudo-tool-call re-arm guard.
try:
    from core.graph_loader import (
        ABORT_GUARD_MIN_TOOL_CALLS,
        ABORT_GUARD_PROMPT,
        PSEUDO_TOOL_CALL_RE,
        count_event_tool_calls,
        event_has_pseudo_tool_call,
        should_rearm_run,
    )
except ImportError:  # pragma: no cover - minimal stdlib-only test environments
    ABORT_GUARD_MIN_TOOL_CALLS = 8
    ABORT_GUARD_PROMPT = (
        "Your previous reply ended the run without doing the audit: no findings "
        "were reported and no tools were called. Do not answer with prose only. "
        "Tool calls must go through the function-calling interface -- writing "
        "one out as text, like 'call:api:read_file{...}', does nothing. "
        "Resume now: call list_files('app'), read the source, and report every "
        "confirmed vulnerability via report_findings before you finish."
    )
    PSEUDO_TOOL_CALL_RE = re.compile(
        r"(?:\bcall:\w+:\w+\s*\{|<tool_call>|<function=|<minimax:tool_call|```tool_call)",
        re.IGNORECASE,
    )

    def count_event_tool_calls(event: Any) -> int:
        """Function calls carried by one ADK event. Never raises."""
        try:
            parts = getattr(getattr(event, "content", None), "parts", None) or []
            return sum(1 for part in parts if getattr(part, "function_call", None))
        except Exception:
            return 0

    def event_has_pseudo_tool_call(event: Any) -> bool:
        """True when a text part spells out a tool call instead of making one. Never raises."""
        try:
            parts = getattr(getattr(event, "content", None), "parts", None) or []
            if any(
                PSEUDO_TOOL_CALL_RE.search(getattr(part, "text", None) or "")
                for part in parts
            ):
                return True
            output = getattr(event, "output", None)
            return bool(isinstance(output, str) and PSEUDO_TOOL_CALL_RE.search(output))
        except Exception:
            return False

    def should_rearm_run(
        findings_count: int,
        tool_calls: int,
        rearms_used: int = 0,
        pseudo_tool_call: bool = False,
    ) -> bool:
        """True only for the first run that ends with nothing reported and
        nearly nothing attempted -- or that ended on a tool call written as
        text, which no tool-count threshold can tell from a deliberate stop."""
        return (
            rearms_used == 0
            and findings_count == 0
            and (tool_calls < ABORT_GUARD_MIN_TOOL_CALLS or pseudo_tool_call)
        )


# --- Transcript capture ---------------------------------------------------------------

# The transcript is what a human reads to follow a run hop by hop: which
# tools the model called, with what arguments, and what came back. The row
# format is core.transcript's, shared with main.py --transcript-out, so the
# researcher eval and the full pipeline leave the same shape behind.

from core.transcript import (  # noqa: E402
    TRANSCRIPT_PAYLOAD_LIMIT,
    TRANSCRIPT_TEXT_LIMIT,
    clip as _clip,
    event_transcript_rows,
)


# --- Behavioural telemetry (pure, no LLM, unit-tested) --------------------------------

# Recall on a 26-file target says little about how a run would fare on a
# repository that does not fit in one context window. What does transfer is
# *how* the run read: whole files or windows, by navigation or by dumping,
# and how often the loop guard had to intervene. These counters are derived
# from the same event stream the score is, cost nothing, and are the
# mechanistic signal behind any claim that a toolset change "helps at scale".

STRUCTURAL_NAV_TOOLS = frozenset(
    {"find_symbol", "find_callers", "find_callees", "get_function_boundary"}
)
STEERING_HINT_MARKER = "HARNESS STEERING HINT"


def read_is_windowed(args: Any) -> bool:
    """A read_file call is windowed only when a bound is positive.

    read_file's bounds default to 0 (= whole file) and models routinely echo
    declared defaults, so `start_line=0, end_line=0` is an un-windowed read.
    Same rule as the production tool-loop guard.
    """
    if not isinstance(args, dict):
        return False
    return bool(args.get("start_line") or args.get("end_line"))


def _response_text(resp: Any) -> str:
    payload = getattr(resp, "response", None)
    if isinstance(payload, dict):
        for key in ("result", "response", "output"):
            if isinstance(payload.get(key), str):
                return payload[key]
        return json.dumps(payload, default=str)
    return str(payload or "")


class ToolTelemetry:
    """Accumulates how a run used its tools. observe() never raises."""

    def __init__(self) -> None:
        self.tool_counts: Dict[str, int] = {}
        self.unwindowed_reads = 0
        self.windowed_reads = 0
        self.structural_nav_calls = 0
        self.list_files_calls = 0
        self.report_findings_calls = 0
        self.steering_hints_fired = 0
        self.files_read: set[str] = set()
        self._unwindowed_per_file: Dict[str, int] = {}
        # Cumulative token count at the moment each report_findings call was
        # made: the raw material for a recall-vs-tokens curve.
        self.report_token_marks: List[int] = []

    def observe(self, event: Any, cumulative_tokens: int) -> None:
        try:
            parts = getattr(getattr(event, "content", None), "parts", None) or []
            for part in parts:
                call = getattr(part, "function_call", None)
                if call is not None:
                    self._observe_call(
                        str(getattr(call, "name", "") or ""),
                        getattr(call, "args", None),
                        cumulative_tokens,
                    )
                resp = getattr(part, "function_response", None)
                if resp is not None and STEERING_HINT_MARKER in _response_text(resp):
                    self.steering_hints_fired += 1
        except Exception:  # pragma: no cover - telemetry must never sink a run
            return

    def _observe_call(self, name: str, args: Any, cumulative_tokens: int) -> None:
        if not name:
            return
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
        if name == "read_file":
            path = ""
            if isinstance(args, dict):
                path = str(args.get("filepath") or args.get("path") or "").strip()
            if path:
                self.files_read.add(path)
            if read_is_windowed(args):
                self.windowed_reads += 1
            else:
                self.unwindowed_reads += 1
                if path:
                    self._unwindowed_per_file[path] = self._unwindowed_per_file.get(path, 0) + 1
        elif name in STRUCTURAL_NAV_TOOLS:
            self.structural_nav_calls += 1
        elif name == "list_files":
            self.list_files_calls += 1
        elif name == "report_findings":
            self.report_findings_calls += 1
            self.report_token_marks.append(int(cumulative_tokens))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool_counts": dict(sorted(self.tool_counts.items())),
            "unwindowed_reads": self.unwindowed_reads,
            "windowed_reads": self.windowed_reads,
            "structural_nav_calls": self.structural_nav_calls,
            "list_files_calls": self.list_files_calls,
            "report_findings_calls": self.report_findings_calls,
            "steering_hints_fired": self.steering_hints_fired,
            "unique_files_read": len(self.files_read),
            "repeat_unwindowed_files": sum(
                1 for n in self._unwindowed_per_file.values() if n > 1
            ),
            "report_token_marks": list(self.report_token_marks),
        }


# --- Run identity and caps (pure) -----------------------------------------------------

# Two results may only be averaged if they measured the same thing. The
# fingerprint names the configuration (model, effort, tools, prompt); the
# ground-truth version names the answer key. A sweep keys its cells on both
# so that editing a match phrase or a prompt starts a new series instead of
# blending into the old one.


def ground_truth_version(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def config_fingerprint(
    model_id: str,
    effort: str,
    tool_names: Sequence[str],
    prompt_text: str,
    ground_truth: str = "",
) -> str:
    import hashlib

    payload = json.dumps(
        {
            "model": str(model_id or ""),
            "effort": str(effort or ""),
            "tools": sorted(str(t) for t in tool_names),
            "prompt": prompt_text,
            "ground_truth": ground_truth,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cap_exceeded(
    tokens: int,
    tool_calls: int,
    max_tokens: Optional[int] = None,
    max_tool_calls: Optional[int] = None,
) -> bool:
    """True once a run has spent past either cap. None means uncapped.

    A capped run is still a complete, graded run -- it is scored as
    recall@cap, which is the number that compares two arms at equal spend.
    """
    if max_tokens is not None and max_tokens > 0 and tokens >= max_tokens:
        return True
    if max_tool_calls is not None and max_tool_calls > 0 and tool_calls >= max_tool_calls:
        return True
    return False


def summarize_research_runs(run_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates repeated runs of one configuration. Pure; unit-tested.

    Only graded runs enter a mean. A crashed run is not a low score, it is a
    missing measurement -- it is counted in ``ungraded`` and excluded from
    every average, the way a judge failure is ``None`` rather than zero. A
    completed run with no findings is a real zero and does enter the mean.
    Mixed ground-truth versions or configurations are reported rather than
    silently averaged.
    """
    graded = [r for r in run_results if r.get("graded", True)]
    ungraded = len(run_results) - len(graded)

    def _mean(key: str, default: float = 0.0) -> float:
        vals = [r.get(key, default) for r in graded]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else default

    recalls = [r.get("recall", 0.0) for r in graded]
    tptf = [r.get("tokens_per_detected") for r in graded if r.get("tokens_per_detected")]
    return {
        "runs": len(run_results),
        "graded": len(graded),
        "ungraded": ungraded,
        "zero_finding_runs": sum(1 for r in graded if r.get("zero_findings_complete")),
        "cap_hits": sum(1 for r in graded if r.get("cap_hit")),
        "rearms": sum(int(r.get("abort_guard_rearms", 0) or 0) for r in graded),
        "avg_recall": _mean("recall"),
        "worst_recall": min(recalls) if recalls else 0.0,
        "avg_cross_file_recall": _mean("cross_file_recall"),
        "avg_precision": _mean("precision"),
        "avg_tokens": _mean("total_tokens"),
        "avg_tokens_per_detected": (sum(tptf) / len(tptf)) if tptf else 0,
        "avg_unwindowed_reads": (
            sum(int((r.get("telemetry") or {}).get("unwindowed_reads", 0)) for r in graded) / len(graded)
            if graded else 0.0
        ),
        "avg_structural_nav_calls": (
            sum(int((r.get("telemetry") or {}).get("structural_nav_calls", 0)) for r in graded) / len(graded)
            if graded else 0.0
        ),
        "ground_truth_versions": sorted({str(r.get("ground_truth_version", "")) for r in run_results} - {""}),
        "config_fingerprints": sorted({str(r.get("config_fingerprint", "")) for r in run_results} - {""}),
    }


# --- Jail setup -----------------------------------------------------------------------


def _copy_target(src: Path, jail_dir: Path) -> Path:
    """Copies the target into ``jail_dir``/app, excluding ground_truth.json.

    The exclusion is required: the model under evaluation must not be able
    to read the answers.
    """
    app_dir = Path(jail_dir) / "app"
    shutil.copytree(
        src,
        app_dir,
        ignore=shutil.ignore_patterns(GROUND_TRUTH_BASENAME, "__pycache__", "*.pyc"),
    )
    copied_gt = app_dir / GROUND_TRUTH_BASENAME
    if copied_gt.exists():  # pragma: no cover - belt and braces
        raise RuntimeError("ground truth leaked into the eval jail")
    from core.harness_context import assert_no_grader_verdicts, assert_no_leak_paths

    assert_no_leak_paths(app_dir)
    assert_no_grader_verdicts(app_dir)
    return app_dir


def resolve_toolset(name: str) -> List[str]:
    """Resolves a toolset name to registered tool names. Fails on gaps.

    The toolset is the independent variable. If a requested tool is not
    registered, silently running without it would turn one arm into the
    other and label the result with the wrong name -- so this raises.
    """
    from tools import TOOLS

    requested = TOOLSETS.get(name)
    if requested is None:
        raise ValueError(f"unknown toolset '{name}'; known: {sorted(TOOLSETS)}")
    missing = [t for t in requested if t not in TOOLS]
    if missing:
        raise ValueError(
            f"toolset '{name}' requests tools that are not registered: "
            f"{', '.join(missing)}. Register them or pick another toolset; "
            "an arm that quietly loses tools is not that arm."
        )
    return list(requested)


# --- The eval -------------------------------------------------------------------------


async def eval_researcher(
    model_id: str,
    effort: str,
    target_dir: Optional[Path] = None,
    toolset: str = "baseline",
    transcript_path: Optional[Path] = None,
    prompt_override: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_tool_calls: Optional[int] = None,
    arm_label: Optional[str] = None,
    predecessor: Optional[str] = None,
    fork: bool = False,
    campaigns: str = "whole",
    files: str = DEFAULT_CAMPAIGN_FILES,
    concurrency: int = 4,
) -> Dict[str, Any]:
    """Runs the researcher over the seeded target and scores it.

    With ``predecessor`` (a stage name, e.g. "planner"), that stage runs
    first in the same session as a two-node graph, the way the pipeline
    runs them: each node with include_contents="none", the seed reaching
    the first only, the researcher getting its briefing in its instruction
    and the plan through get_plan. With ``fork`` as well, the researcher's
    requests begin with the predecessor's transcript (core.graph_loader
    fork_from) instead of a fresh context. The two planned arms differ in
    nothing else, so a paired comparison isolates the fork; the
    predecessor's own tokens are reported apart (``tokens_by_stage``).

    ``campaigns`` is how the target is covered: ``whole`` is one campaign
    over the tree; ``per-file`` is one campaign per file in
    ``select_campaign_files(files)``, each with the file as its target (so
    read_file opens only that file, as in the pipeline's file-by-file
    scan), up to ``concurrency`` at a time, all reporting into one database
    that is scored once. Caps apply to the cell as a whole.

    Result validity, in the result's own words:

    * ``graded`` / ``status``: a run that crashed is ``ungraded`` -- its
      partial findings are still scored for the human reading the JSON, but
      an aggregator must leave it out of every mean and count it as a
      failure. A run that completed and reported nothing is a real zero. In
      per-file mode one crashed campaign ungrades the cell.
    * ``cap_hit``: the run was stopped at ``max_tokens`` / ``max_tool_calls``
      and its score is recall@cap. Still graded.
    * ``config_fingerprint`` / ``ground_truth_version``: what was measured
      and against which answer key. Results that differ in either must not
      be averaged together.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from core.context import current_run_context
    from core.database import init_db, read_findings
    from evals.stage_agents import build_stage_agent, install_eval_run_context

    if fork and not predecessor:
        raise ValueError("fork needs a predecessor stage to fork from")
    mode_suffix = campaign_shape(campaigns, files)
    per_file = bool(mode_suffix)
    concurrency = max(1, int(concurrency or 1))
    target_dir = Path(target_dir or DEFAULT_TARGET)
    gt_path = target_dir / GROUND_TRUTH_BASENAME
    gt_entries = load_research_ground_truth(gt_path)
    gt_version = ground_truth_version(gt_path)
    model_id = model_id or os.environ.get("EVAL_MODEL", "vertex_ai/gemini-3.8-flash")
    effort = effort or os.environ.get("EVAL_REASONING_EFFORT", "low")
    tool_names = resolve_toolset(toolset)
    prompt_text = prompt_override or (RESEARCH_PROMPT_FILE if per_file else RESEARCH_PROMPT)
    shape = f"{predecessor or 'solo'}{'+fork' if fork else ''}{mode_suffix}"
    fingerprint = config_fingerprint(model_id, effort, tool_names, prompt_text + "\n" + shape, gt_version)

    temp_dir = tempfile.mkdtemp(prefix="mantis_eval_research_")
    tokens = 0
    tokens_by_stage: Dict[str, int] = {}
    billed_tokens = 0.0
    cached_tokens = 0
    tool_calls = 0
    rearms = 0
    cap_hit = False
    pseudo_calls = False
    run_errors: List[str] = []
    campaign_stats: List[Dict[str, Any]] = []
    campaign_files: List[str] = []
    try:
        # The jail root is a subdirectory so eval.db stays outside it: the
        # jailed read_file/list_files tools must not be able to reach the
        # findings database.
        jail_root = Path(temp_dir) / "jail"
        jail_root.mkdir()
        app_dir = _copy_target(target_dir, jail_root)
        db_path = os.path.join(temp_dir, "eval.db")
        init_db(db_path)
        base_ctx, reset_token = install_eval_run_context(
            str(jail_root), db_path, target_file=str(app_dir)
        )
        if any(t in tool_names for t in STRUCTURAL_NAV_TOOLS):
            # The structural arm needs its catalog. build never raises; a
            # failed build just degrades the arm to baseline behavior.
            from core.structural_index import build_structural_index, state_dir_for_db
            build_structural_index(str(jail_root), state_dir_for_db(db_path))
        if per_file:
            campaign_files = select_campaign_files(app_dir, gt_entries, files)
            if not campaign_files:
                raise RuntimeError(f"file selector '{files}' chose no files under {app_dir}")
        telemetry = ToolTelemetry()
        transcript: List[Dict[str, Any]] = []
        session_service = InMemorySessionService()
        started = time.time()

        def _briefing(rel: str) -> "tuple[str, Optional[str], str]":
            """(researcher briefing, planner instruction or None, seed subject)
            for one campaign. Literal substitution: the path is repository
            bytes and must never be on the template side of a format."""
            if not rel:
                return prompt_text, None, "the application under 'app/'"
            shown = f"app/{rel}"
            return (
                prompt_text.replace("{filepath}", shown),
                PLANNER_PROMPT_FILE.replace("{filepath}", shown),
                shown,
            )

        async def _campaign(rel: str) -> None:
            """One campaign: its own RunContext (target_file), session and
            agents; everything it spends lands in the cell's totals."""
            nonlocal tokens, tool_calls, pseudo_calls, cap_hit, billed_tokens, cached_tokens, rearms
            target_file = str(app_dir / rel) if rel else str(app_dir)
            # replace() copies field values; the per-campaign dicts must be
            # this campaign's own, as main.py gives each campaign its own.
            ctx = dataclasses.replace(
                base_ctx, target_file=target_file, finding_evidence={}, node_context={}, read_file_counts={},
            )
            token = current_run_context.set(ctx)
            briefing, planner_instruction, subject = _briefing(rel)
            stat: Dict[str, Any] = {"file": rel or "", "tokens": 0, "tool_calls": 0, "status": "complete"}
            c_started = time.time()
            try:
                from google.adk.agents.context_cache_config import ContextCacheConfig
                from google.adk.apps.app import App

                # Production runs with context caching on (main.py
                # enable_context_cache): for Claude that is what makes ADK
                # mark the prompt prefix cacheable at all, and without it the
                # eval would bill every token of a forked transcript at full
                # price -- a cost the pipeline never pays. Gemini caches
                # implicitly either way.
                cache_config = ContextCacheConfig()
                if predecessor:
                    from google.adk.workflow import START, Edge, Workflow

                    from core.graph_loader import _make_fork_before_model_callback

                    first = build_stage_agent(
                        predecessor, model_id=model_id, reasoning_effort=effort, include_contents="none",
                        instruction_override=planner_instruction,
                    )
                    agent = build_stage_agent(
                        "researcher",
                        model_id=model_id,
                        reasoning_effort=effort,
                        tools_override=tool_names,
                        include_contents="none",
                        before_model_callback=(
                            [_make_fork_before_model_callback("researcher", predecessor)] if fork else None
                        ),
                        # Not the first node: it sees no seed message, so the
                        # briefing it would have been seeded with rides in its
                        # instruction -- in both planned arms alike.
                        instruction_suffix=briefing,
                    )
                    root = Workflow(name="eval_graph", edges=[
                        Edge(from_node=START, to_node=first), Edge(from_node=first, to_node=agent),
                    ])
                    seed_text = (
                        f"Initial Task Input: Evaluate {subject}. You are the {predecessor} "
                        "stage; do your stage's work and record it through your tools."
                    )
                else:
                    root = build_stage_agent(
                        "researcher",
                        model_id=model_id,
                        reasoning_effort=effort,
                        tools_override=tool_names,
                    )
                    seed_text = briefing
                runner = Runner(
                    app=App(name="eval_app", root_agent=root, context_cache_config=cache_config),
                    session_service=session_service,
                )
                session = await session_service.create_session(
                    app_name="eval_app", user_id="eval_user"
                )
                content = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=seed_text)],
                )

                async def _drain(message: types.Content) -> None:
                    nonlocal tokens, tool_calls, pseudo_calls, cap_hit, billed_tokens, cached_tokens
                    agen = runner.run_async(
                        session_id=session.id, user_id="eval_user", new_message=message
                    )
                    try:
                        async for event in agen:
                            ev_tokens = extract_event_tokens(event)
                            tokens += ev_tokens
                            stat["tokens"] += ev_tokens
                            author = str(getattr(event, "author", "") or "")
                            tokens_by_stage[author] = tokens_by_stage.get(author, 0) + ev_tokens
                            fresh, cached = extract_event_billing(event)
                            billed_tokens += fresh + 0.1 * cached
                            cached_tokens += cached
                            call_count = count_event_tool_calls(event)
                            tool_calls += call_count
                            stat["tool_calls"] += call_count
                            pseudo_calls = pseudo_calls or event_has_pseudo_tool_call(event)
                            telemetry.observe(event, tokens)
                            if transcript_path is not None:
                                rows = event_transcript_rows(event)
                                if rel:
                                    for row in rows:
                                        row["campaign"] = rel
                                transcript.extend(rows)
                            # Let the Nth function_call event execute its tool before
                            # breaking on its function_response (unless already past N
                            # or over the token budget).
                            effective_calls = tool_calls - call_count if call_count and tool_calls == max_tool_calls else tool_calls
                            if cap_exceeded(tokens, effective_calls, max_tokens, max_tool_calls):
                                cap_hit = True
                                break
                            if cap_hit:
                                # Another campaign hit the cell's cap.
                                break
                    finally:
                        await agen.aclose()

                try:
                    await _drain(content)
                    # The abort-guard re-arm re-sends the seed. In a planned arm
                    # that would run the whole graph -- the planner again, not
                    # just the researcher -- and charge the arm for it; a planned
                    # researcher that ended without a tool call is scored as it
                    # ended, as the pipeline's would be.
                    if not cap_hit and not predecessor and not per_file and should_rearm_run(
                        len(read_findings(db_path)), tool_calls, rearms, pseudo_calls
                    ):
                        logger.warning(
                            "Researcher run ended on a tool-less or text-written "
                            "tool call reply; re-arming once."
                        )
                        rearms += 1
                        await _drain(
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=ABORT_GUARD_PROMPT)],
                            )
                        )
                except Exception as exc:
                    from core.config import is_auth_error

                    if is_auth_error(exc):
                        # A credential problem is an operator error, not a score.
                        raise
                    logger.warning("Researcher run aborted (%s): %r", rel or "whole", exc)
                    run_errors.append((f"{rel}: " if rel else "") + repr(exc)[:200])
                    stat["status"] = "error"
                if cap_hit and stat["status"] == "complete":
                    stat["status"] = "cap"
            finally:
                stat["wall_seconds"] = round(time.time() - c_started, 1)
                campaign_stats.append(stat)
                current_run_context.reset(token)

        try:
            if per_file:
                sem = asyncio.Semaphore(concurrency)

                async def _bounded(rel: str) -> None:
                    async with sem:
                        if cap_hit:
                            campaign_stats.append({"file": rel, "tokens": 0, "tool_calls": 0,
                                                   "status": "skipped_cap", "wall_seconds": 0.0})
                            return
                        await _campaign(rel)

                # Each task carries its own copy of the context, so the
                # RunContext one campaign sets is invisible to the others.
                await asyncio.gather(*(asyncio.create_task(_bounded(rel)) for rel in campaign_files))
            else:
                await _campaign("")
            latency = time.time() - started
        finally:
            current_run_context.reset(reset_token)

        findings = read_findings(db_path)
        result = score_research_findings(findings, gt_entries)
        if transcript_path is not None:
            try:
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                transcript_path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in transcript),
                    encoding="utf-8",
                )
                result["transcript_path"] = str(transcript_path)
            except OSError as exc:
                run_errors.append(f"transcript write failed: {exc}")
        graded = not run_errors
        # The full records ride along for human adjudication: the eval
        # database is deleted with the temp dir, and transcript payloads
        # are clipped, so this is the only complete copy that survives.
        result["findings_full"] = findings
        result.update(
            {
                "arm": arm_label or (f"{toolset}:{shape}" if predecessor else toolset),
                "shape": shape,
                "predecessor": predecessor or "",
                "fork": bool(fork),
                "campaigns": "per-file" if per_file else "whole",
                "campaign_files": list(campaign_files),
                "campaigns_run": sum(1 for s in campaign_stats if s["status"] != "skipped_cap"),
                "campaign_stats": sorted(campaign_stats, key=lambda s: s["file"]),
                "toolset": toolset,
                "tools_resolved": tool_names,
                "model": model_id,
                "effort": effort,
                "config_fingerprint": fingerprint,
                "ground_truth_version": gt_version,
                "graded": graded,
                "status": "complete" if graded else "ungraded",
                "zero_findings_complete": graded and not findings,
                "cap_hit": cap_hit,
                "caps": {"max_tokens": max_tokens, "max_tool_calls": max_tool_calls},
                "latency": latency,
                "run_errors": run_errors,
                "total_tokens": tokens,
                "tokens_by_stage": tokens_by_stage,
                "researcher_tokens": tokens_by_stage.get("researcher", tokens if not predecessor else 0),
                # What a provider bills: uncached input and output at full
                # price, cached input at a tenth. A fork re-sends the parent's
                # transcript on every call, nearly all of it from cache.
                "billed_tokens": round(billed_tokens),
                "cached_tokens": cached_tokens,
                "tool_calls": tool_calls,
                "abort_guard_rearms": rearms,
                "tokens_per_detected": (
                    round(tokens / result["detected"])
                    if tokens and result["detected"]
                    else None
                ),
                "telemetry": telemetry.as_dict(),
            }
        )
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="Model ID (vertex_ai/...).")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--toolset", default="baseline", choices=sorted(TOOLSETS))
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--transcript-out", type=Path, default=None,
        help="Write a JSONL trace of prose, tool calls, and tool responses.",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=None,
        help="Replace the built-in researcher briefing with this file's text.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Stop the run once this many tokens are spent; score as recall@cap.",
    )
    parser.add_argument(
        "--max-tool-calls", type=int, default=None,
        help="Stop the run once this many tool calls are made; score as recall@cap.",
    )
    parser.add_argument(
        "--arm-label", default=None,
        help="Label this configuration carries in results (default: the toolset name).",
    )
    parser.add_argument(
        "--predecessor", default=None, choices=["planner"],
        help="Run this stage first in the same session, as a two-node graph; the researcher "
             "follows it the way the pipeline's does (include_contents=none, plan via get_plan).",
    )
    parser.add_argument(
        "--fork", action="store_true",
        help="With --predecessor: the researcher's requests begin with the predecessor's transcript "
             "(fork_from) instead of a fresh context. The arm to compare against --predecessor alone.",
    )
    parser.add_argument(
        "--campaigns", default="whole", choices=list(CAMPAIGN_MODES),
        help="'whole': one campaign over the target. 'per-file': one campaign per selected file, "
             "each reading only its file, the pipeline's file-by-file scan; scored together.",
    )
    parser.add_argument(
        "--files", default=DEFAULT_CAMPAIGN_FILES,
        help="With --campaigns per-file: 'all', 'gt' (the files the answer key names) or "
             "'gt+clean:N' (those plus N others, chosen deterministically per target).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="With --campaigns per-file: campaigns in flight at once.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("VERTEXAI_LOCATION", "global")
    prompt_override = (
        args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else None
    )
    result = await eval_researcher(
        args.model,
        args.effort,
        target_dir=args.target,
        toolset=args.toolset,
        transcript_path=args.transcript_out,
        prompt_override=prompt_override,
        max_tokens=args.max_tokens,
        max_tool_calls=args.max_tool_calls,
        arm_label=args.arm_label,
        predecessor=args.predecessor,
        fork=args.fork,
        campaigns=args.campaigns,
        files=args.files,
        concurrency=args.concurrency,
    )
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
