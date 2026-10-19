"""Diversification metrics: does this audit look like it actually searched?

WHY THIS EXISTS
---------------
There is no ground truth and there never will be. Mantis will be run on monorepos we
never see, by people who will not tell us when it missed something. We cannot measure
recall, so the honest alternative is to make the run describe its own shape and let a
human judge whether that shape is plausible.

The specific failure this is aimed at is the one that looks most like success: a run
that reports twenty findings which are really the same finding twenty times. Twenty
instances of "unvalidated input" in twenty files, all from one area, all the same
weakness class, is one observation with a large count -- and it is indistinguishable
from thorough work if you only read the total.

WHAT IT MEASURES
----------------
Four things, each a plain count or ratio over validated/structural fields:

  * SPREAD    -- how many distinct areas produced findings, against how many were
                 examined. Low spread with high counts means one area dominated.
  * VARIETY   -- how many distinct weakness classes. One CWE across everything is a
                 single observation repeated.
  * DEPTH     -- how many findings survived review rather than being dismissed. A high
                 dismissal rate means the run was generating noise.
  * SILENCE   -- how many examined areas produced nothing at all. Not a defect on its
                 own, but the number a reader needs to interpret everything else.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not score the run, grade it, or gate anything on the result. There is no
threshold at which a run is declared bad, because no such threshold is knowable: a
repository with one real defect SHOULD produce one finding in one area, and a metric
that called that failure would be worse than no metric. It reports the shape and says
what the shape could mean, in both directions. The reader decides.

It also never modifies a finding. Nothing here can change severity, status, or whether
something is reported -- it is strictly an observation about the set as a whole.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Statuses meaning "a reviewer looked at this and rejected it". Mirrors
# `memory._STATUSES_DISMISSED` plus the deduplicator's own outcome, which is a rejection
# of a duplicate rather than of a claim.
_REJECTED = frozenset({
    "false_positive", "non_viable", "sample_or_test", "duplicate_merged",
})

# Statuses meaning "nobody has ruled on this yet". `reported` is the schema default for
# every newly written finding (database.py), so a run whose triage never executed -- cut
# short by budget, or crashed -- leaves everything sitting here.
#
# This MUST stay separate from _REJECTED. Counting the default as a rejection made the
# report say "100% of findings were rejected at review; the reviewers are working" about
# a run in which no reviewer ran at all. An unreviewed claim is not a dismissed one, and
# conflating them invents a quality signal out of the absence of one.
_UNREVIEWED = frozenset({"reported"})


def _area_of(filepath: Any) -> str:
    """The directory a finding sits in, which is the unit 'spread' counts.

    Findings are per-file; counting distinct FILES would report high spread for twenty
    findings in one directory, which is exactly the concentration this is meant to
    expose. For the same reason root-level files collapse to a single area rather than
    each becoming its own.
    """
    path = str(filepath or "").replace("\\", "/").rstrip("/")
    if not path:
        return ""
    cut = path.rfind("/")
    if cut > 0:
        return path[:cut]
    # "a.py" -> "." and "/a.py" -> "/": both are one area, not one area per file.
    return "/" if cut == 0 else "."



def _produced_something(examined_item: str, finding_paths: List[str]) -> bool:
    """Did this scan target yield any finding?

    `examined_areas` holds whatever the run scanned, and that is NOT one granularity:
    file-by-file mode appends individual FILES while cross-functional mode appends DIRECTORIES.
    Findings are always per-file. Comparing the two by count reported "19 of 20 produced
    nothing" for a run in which every single file produced a finding.

    So match by containment instead: a target counts as productive if any finding sits
    at it or beneath it.
    """
    item = str(examined_item or "").replace("\\", "/").strip().rstrip("/")
    if not item:
        return False
    prefix = item + "/"
    for path in finding_paths:
        if path == item or path.startswith(prefix) or item.endswith("/" + path):
            return True
    return False


def measure(
    findings: List[Dict[str, Any]],
    examined_areas: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Describes the shape of what this run produced.

    `examined_areas` is what the run actually examined -- needed because the interesting
    quantity is findings-per-examined-target, and the findings alone cannot say how much
    ground was walked to produce them.

    Never raises (INV-6).
    """
    empty: Dict[str, Any] = {"available": False}
    try:
        rows = [f for f in (findings or []) if isinstance(f, dict)]
        examined = [str(a) for a in (examined_areas or []) if str(a or "").strip()]
        if not rows and not examined:
            return empty

        def _status(f: Dict[str, Any]) -> str:
            return str(f.get("status") or "").strip().lower()

        # Three outcomes, not two. A finding still sitting at the schema default has not
        # been rejected -- nobody has looked at it yet.
        rejected_rows = [f for f in rows if _status(f) in _REJECTED]
        kept = [f for f in rows if _status(f) not in _REJECTED]
        unreviewed = [f for f in kept if _status(f) in _UNREVIEWED]
        rejected = len(rejected_rows)

        # Shape is measured over everything still standing, unreviewed included: those
        # are claims that have not been knocked down, and excluding them would make a
        # run whose triage never executed report no shape at all.
        areas_with_findings = {_area_of(f.get("filepath")) for f in kept}
        areas_with_findings.discard("")

        weaknesses = {str(f.get("cwe") or "").strip() for f in kept}
        weaknesses.discard("")

        severities: Dict[str, int] = {}
        for f in kept:
            sev = str(f.get("severity") or "unspecified").strip().lower()
            severities[sev] = severities.get(sev, 0) + 1

        # Concentration: the share of surviving findings in the single busiest area.
        # 1.0 means everything came from one place.
        per_area: Dict[str, int] = {}
        for f in kept:
            area = _area_of(f.get("filepath"))
            if area:
                per_area[area] = per_area.get(area, 0) + 1
        busiest = max(per_area.values()) if per_area else 0
        concentration = (busiest / len(kept)) if kept else 0.0

        finding_paths = []
        for f in kept:
            p = str(f.get("filepath") or "").replace("\\", "/").strip().rstrip("/")
            if p:
                finding_paths.append(p)
        silent = sum(1 for item in examined if not _produced_something(item, finding_paths))

        # Denominator is what was actually REVIEWED. Dividing by every raised finding
        # would report a 100% rejection rate for a run in which no reviewer ran.
        reviewed = rejected + (len(kept) - len(unreviewed))

        return {
            "available": True,
            "findings_total": len(rows),
            "findings_kept": len(kept),
            "findings_rejected": rejected,
            "findings_unreviewed": len(unreviewed),
            "findings_reviewed": reviewed,
            "rejection_rate": (rejected / reviewed) if reviewed else 0.0,
            "targets_examined": len(examined),
            "areas_with_findings": len(areas_with_findings),
            "silent_targets": silent,
            "distinct_weaknesses": len(weaknesses),
            "severity_spread": severities,
            "concentration": round(concentration, 3),
        }
    except Exception as exc:
        logger.warning("Diversification metrics failed: %s", exc)
        return empty


def render_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Operator-facing lines describing the run's shape.

    Returns a list of plain strings containing no repository-derived bytes -- only
    counts and ratios -- so the caller can print them without CP-4 concerns.

    Every observation is stated with BOTH readings. "One weakness class across nine
    findings" can mean a systemic flaw worth chasing or a single observation counted
    nine times, and this module cannot tell which. Presenting only the pessimistic
    reading would train the reader to ignore it.
    """
    if not isinstance(metrics, dict) or not metrics.get("available"):
        return []

    kept = metrics.get("findings_kept", 0)
    lines = [
        f"{kept} finding(s) kept across {metrics.get('areas_with_findings', 0)} "
        f"area(s); {metrics.get('findings_rejected', 0)} rejected at review.",
    ]

    # Stated plainly, because every ratio below is weaker when triage did not finish.
    unreviewed = metrics.get("findings_unreviewed", 0)
    if unreviewed:
        lines.append(
            f"{unreviewed} finding(s) were never reviewed, so they are counted as "
            f"standing rather than as confirmed. Triage may have been cut short."
        )

    if metrics.get("targets_examined"):
        lines.append(
            f"{metrics['targets_examined']} target(s) examined, "
            f"{metrics.get('silent_targets', 0)} produced nothing."
        )

    if kept >= 3:
        distinct = metrics.get("distinct_weaknesses", 0)
        if distinct <= 1:
            lines.append(
                f"All {kept} findings share one weakness class. That is either a "
                f"systemic flaw worth pursuing everywhere, or one observation counted "
                f"{kept} times -- worth deciding which before acting on the count."
            )
        else:
            lines.append(f"{distinct} distinct weakness class(es) represented.")

        concentration = metrics.get("concentration", 0.0)
        if concentration >= 0.8 and metrics.get("areas_with_findings", 0) >= 1:
            lines.append(
                f"{concentration:.0%} of findings come from a single area. That area "
                f"may genuinely be the weak point, or it may be the only one that was "
                f"examined deeply."
            )

    # Gated on findings actually REVIEWED, not findings raised: the sentence claims the
    # reviewers were working, so it must not appear when they never ran.
    rate = metrics.get("rejection_rate", 0.0)
    if metrics.get("findings_reviewed", 0) >= 5 and rate >= 0.5:
        lines.append(
            f"{rate:.0%} of reviewed findings were rejected. The reviewers are "
            f"working, but a run that mostly produces noise is spending its budget on "
            f"triage rather than analysis."
        )

    return lines


# --- Pre-rollout harness diversity and blind candidate selection (Harness²) ---

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import random
import re
from typing import Any


BLIND_IDS: tuple[str, ...] = (
    "c_alpha",
    "c_beta",
    "c_gamma",
    "c_delta",
    "c_epsilon",
    "c_zeta",
    "c_eta",
    "c_theta",
)

_PICK_RE = re.compile(r"<pick>\s*([A-Za-z0-9_.-]+)\s*</pick>", re.IGNORECASE)
_WHY_RE = re.compile(r"<why>(.*?)</why>", re.DOTALL | re.IGNORECASE)


def workflow_to_surface_tree(spec: Any) -> dict[str, str]:
    """Converts a `WorkflowSpec` or raw workflow dict into a surface file tree.

    Args:
        spec: A `WorkflowSpec` instance, a raw workflow dict, or an existing
            surface `dict[str, str]`.

    Returns:
        Mapping of virtual surface paths (`topology.txt`,
        `nodes/<id>/tools.txt`, `nodes/<id>/prompt.txt`, `config/sandbox.txt`)
        to normalized text content.
    """
    if hasattr(spec, "model_dump"):
        data = spec.model_dump(by_alias=True, exclude_none=True)
    elif isinstance(spec, Mapping):
        if "nodes" not in spec and all(isinstance(v, str) for v in spec.values()):
            return {str(k): str(v) for k, v in spec.items()}
        data = dict(spec)
    else:
        raise TypeError(f"Unsupported workflow spec type: {type(spec)!r}")

    surface: dict[str, str] = {}
    cfg = data.get("config") or {}
    sb = (cfg.get("sandbox") or {}).get("type", "static-only")
    surface["config/sandbox.txt"] = f"sandbox={sb}\nmodel={cfg.get('default_model', '')}\n"

    edge_lines: list[str] = []
    for edge in data.get("edges") or []:
        if not isinstance(edge, Mapping):
            continue
        src = str(edge.get("from") or edge.get("from_node") or "")
        dst = str(edge.get("to") or edge.get("to_node") or "")
        route = edge.get("on")
        if route is not None:
            edge_lines.append(f"{src} -> {dst} [on={route}]")
        else:
            edge_lines.append(f"{src} -> {dst}")
    surface["topology.txt"] = "\n".join(sorted(edge_lines)) + "\n"

    for node in data.get("nodes") or []:
        if not isinstance(node, Mapping):
            continue
        nid = str(node.get("id") or "unknown")
        ntype = str(node.get("type") or "agent")
        if ntype == "classifier":
            routes = sorted(str(r) for r in (node.get("routes") or []))
            mv = node.get("max_visits", 1)
            surface[f"nodes/{nid}/classifier.txt"] = (
                f"routes={','.join(routes)}\nmax_visits={mv}\n"
            )
        else:
            tools = sorted(str(t) for t in (node.get("tools") or []))
            surface[f"nodes/{nid}/tools.txt"] = "\n".join(tools) + "\n"
            prompt = str(node.get("system_prompt") or node.get("skill") or "")
            surface[f"nodes/{nid}/prompt.txt"] = prompt.strip() + "\n"

    return surface


def structure_signature(tree: Mapping[str, str]) -> str:
    """Computes a structural hash over virtual surface filenames and topology."""
    topo = tree.get("topology.txt", "")
    keys = sorted(tree.keys())
    payload = json.dumps({"keys": keys, "topology": topo}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _line_set(tree: Mapping[str, str]) -> set[tuple[str, str]]:
    """Returns `(path, stripped_line)` pairs for non-blank lines across `tree`."""
    return {
        (rel, line.strip())
        for rel, body in tree.items()
        for line in str(body).splitlines()
        if line.strip()
    }


def _jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def harness_diversity(
    harnesses: Mapping[str, Any],
    baseline: Any | None = None,
) -> dict[str, Any]:
    """Quantifies how much a pool of candidate workflows actually differ.

    Ported from `harness2/core/tree.py:diversity`. Computes pairwise line-level
    Jaccard similarity (`1.0` = identical, `0.0` = disjoint), structural
    uniqueness, and per-candidate distance from `baseline`.
    """
    trees: dict[str, dict[str, str]] = {
        label: workflow_to_surface_tree(spec)
        for label, spec in harnesses.items()
    }
    labels = sorted(trees.keys())
    line_sets = {label: _line_set(trees[label]) for label in labels}

    pairwise: dict[str, float] = {}
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1 :]:
            pairwise[f"{l1}..{l2}"] = round(
                _jaccard(line_sets[l1], line_sets[l2]), 3
            )

    vals = list(pairwise.values())
    mean_jaccard = round(sum(vals) / len(vals), 3) if vals else 1.0

    vs_base: dict[str, dict[str, Any]] = {}
    if baseline is not None:
        base_tree = workflow_to_surface_tree(baseline)
        base_lines = _line_set(base_tree)
        for label in labels:
            cand_tree = trees[label]
            cand_lines = line_sets[label]
            touched = [
                p
                for p in sorted(set(cand_tree) | set(base_tree))
                if cand_tree.get(p) != base_tree.get(p)
            ]
            vs_base[label] = {
                "files_touched": touched,
                "n_files_touched": len(touched),
                "lines_added": len(cand_lines - base_lines),
                "lines_removed": len(base_lines - cand_lines),
                "jaccard_to_baseline": round(
                    _jaccard(cand_lines, base_lines), 3
                ),
            }

    all_paths = sorted({p for t in trees.values() for p in t})
    only_one: list[str] = []
    some: list[str] = []
    all_same: list[str] = []
    for path in all_paths:
        variants = {trees[label].get(path, "") for label in labels}
        if len(variants) == 1:
            all_same.append(path)
        elif len(labels) > 2 and Counter(
            trees[label].get(path, "") for label in labels
        ).most_common(1)[0][1] == len(labels) - 1:
            only_one.append(path)
        else:
            some.append(path)

    sigs = {label: structure_signature(trees[label]) for label in labels}
    return {
        "n_candidates": len(labels),
        "pairwise_jaccard": pairwise,
        "mean_pairwise_jaccard": mean_jaccard,
        "min_pairwise_jaccard": min(vals) if vals else 1.0,
        "max_pairwise_jaccard": max(vals) if vals else 1.0,
        "unique_structures": len(set(sigs.values())),
        "structure_signatures": sigs,
        "vs_baseline": vs_base,
        "surface_spread": {
            "only_one": only_one,
            "some": some,
            "all_same": all_same,
        },
    }


def assign_blind_ids(
    labels: Sequence[str],
    seed: str | int = "mantis_blind_select",
) -> tuple[dict[str, str], dict[str, str]]:
    """Assigns shuffled blind IDs (`c_alpha`, `c_beta`, ...) to candidate labels.

    Ported from `harness2/core/selection.py` so LLM candidate judges cannot
    anchor on provenance names (`baseline`, `inv`, `dom`, `composed`) or
    presentation order.

    Returns:
        Tuple `(blind_to_real, real_to_blind)`.
    """
    unique_labels = list(dict.fromkeys(str(label) for label in labels))
    rng = random.Random(str(seed))
    shuffled = list(unique_labels)
    rng.shuffle(shuffled)

    blind_to_real: dict[str, str] = {}
    real_to_blind: dict[str, str] = {}
    for idx, real_label in enumerate(shuffled):
        blind_id = (
            BLIND_IDS[idx] if idx < len(BLIND_IDS) else f"c_cand_{idx + 1}"
        )
        blind_to_real[blind_id] = real_label
        real_to_blind[real_label] = blind_id
    return blind_to_real, real_to_blind


def format_blind_candidates(
    candidates: Mapping[str, Any],
    blind_to_real: Mapping[str, str],
) -> str:
    """Formats candidate summaries under their blind IDs (`c_alpha`, `c_beta`, ...)."""
    blocks: list[str] = []
    for blind_id, real_label in blind_to_real.items():
        payload = candidates.get(real_label, "")
        if hasattr(payload, "model_dump"):
            body = json.dumps(
                payload.model_dump(by_alias=True, exclude_none=True),
                indent=2,
                sort_keys=True,
            )
        elif isinstance(payload, (dict, list)):
            body = json.dumps(payload, indent=2, sort_keys=True)
        else:
            body = str(payload)
        blocks.append(f"### Candidate `{blind_id}`\n```json\n{body}\n```")
    return "\n\n".join(blocks)


def resolve_blind_pick(
    judge_text: str,
    blind_to_real: Mapping[str, str],
    fallback_label: str | None = None,
) -> dict[str, str]:
    """Resolves `<pick>c_...</pick>` and `<why>...</why>` back to the real label.

    A reply that names no candidate -- no `<pick>` block and no unambiguous
    mention of exactly one blind id -- is a selection failure, not a choice.
    Following `harness2/core/selection.py`, such a failure must never be
    laundered into an arbitrary winner: when `fallback_label` (a real label,
    normally the baseline) is supplied the result carries it with
    `resolved_by="fallback"`; otherwise a `ValueError` is raised so the caller
    decides. The blind ids are shuffled, so "the first entry" is a random
    candidate and is never used.

    Returns:
        `{"blind_id", "selected", "why", "resolved_by"}` where `resolved_by` is
        one of `"pick"`, `"mention"`, or `"fallback"`.
    """
    if not blind_to_real:
        raise ValueError("blind_to_real mapping cannot be empty")
    why_match = _WHY_RE.search(judge_text or "")
    why = why_match.group(1).strip() if why_match else ""
    pick_match = _PICK_RE.search(judge_text or "")
    picked_blind = pick_match.group(1).strip().lower() if pick_match else ""
    resolved_by = "pick"
    if picked_blind not in blind_to_real:
        lower_text = (judge_text or "").lower()
        hits = [
            blind_id
            for blind_id in blind_to_real
            if re.search(rf"\b{re.escape(blind_id.lower())}\b", lower_text)
        ]
        if len(hits) == 1:
            picked_blind = hits[0]
            resolved_by = "mention"
    if picked_blind not in blind_to_real:
        if fallback_label is None:
            raise ValueError(
                "judge reply names no candidate: no <pick> block and no "
                f"unambiguous blind id among {sorted(blind_to_real)}"
            )
        real_to_blind = {real: blind for blind, real in blind_to_real.items()}
        if fallback_label not in real_to_blind:
            raise ValueError(
                f"fallback_label {fallback_label!r} is not one of the candidates "
                f"{sorted(real_to_blind)}"
            )
        picked_blind = real_to_blind[fallback_label]
        resolved_by = "fallback"
    return {
        "blind_id": picked_blind,
        "selected": blind_to_real[picked_blind],
        "why": why,
        "resolved_by": resolved_by,
    }
