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

"""Paired comparison of two sweep arms, with a gate a change must clear.

Everything here is a pure function of the cell results a sweep wrote, so it
is unit-tested on synthetic cells and re-runnable without a model.

The unit of analysis is the **target** (seed), not the run: repeats on one
target are correlated through the target, so the confidence interval is a
bootstrap over targets of the per-target mean paired difference. Two more
numbers keep a mean from lying:

* the **A/A noise floor** -- the mean absolute difference between repeats of
  the *baseline* arm on the same target. Harness2 reports the k=1
  parallel-vs-sequential gap as its noise floor "not a result"; this is the
  same idea at zero extra cost. A candidate's delta has to clear twice it.
* the **worst-target delta** -- an average can hide a regression on exactly
  the targets where navigation matters.

The gate (``evaluate_gate``) encodes the review's criteria; it returns every
criterion with its numbers so a failing gate says *why*.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # pragma: no cover - direct CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- Selection helpers -------------------------------------------------------------------


def graded_cells(cells: Iterable[Dict[str, Any]], arm: Optional[str] = None) -> List[Dict[str, Any]]:
    return [c for c in cells if c.get("graded", True) and (arm is None or c.get("arm") == arm)]


def by_seed(cells: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for c in cells:
        out.setdefault(int(c["seed"]), []).append(c)
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def per_target_mean(cells: Iterable[Dict[str, Any]], arm: str, metric: str = "recall") -> Dict[int, float]:
    """Mean of ``metric`` over graded repeats, per seed."""
    out: Dict[int, float] = {}
    for seed, group in by_seed(graded_cells(cells, arm)).items():
        vals = [float(c.get(metric) or 0.0) for c in group if isinstance(c.get(metric), (int, float))]
        if vals:
            out[seed] = _mean(vals)
    return out


# --- Paired statistics -------------------------------------------------------------------


def paired_deltas(cells: Iterable[Dict[str, Any]], baseline: str, candidate: str, metric: str = "recall") -> Dict[int, float]:
    """candidate - baseline per seed, over seeds where both arms have a graded cell."""
    cells = list(cells)
    base = per_target_mean(cells, baseline, metric)
    cand = per_target_mean(cells, candidate, metric)
    return {seed: cand[seed] - base[seed] for seed in sorted(set(base) & set(cand))}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Dict[str, Any]:
    """Percentile bootstrap CI of the mean over ``values`` (one value per target)."""
    vals = list(values)
    if not vals:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0}
    if len(vals) == 1:
        return {"n": 1, "mean": vals[0], "lo": vals[0], "hi": vals[0]}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(_mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * (n_boot - 1))]
    hi = means[int((1 - alpha / 2) * (n_boot - 1))]
    return {"n": len(vals), "mean": _mean(vals), "lo": lo, "hi": hi}


def aa_noise_floor(cells: Iterable[Dict[str, Any]], arm: str, metric: str = "recall") -> Dict[str, Any]:
    """Mean absolute difference between repeats of one arm on the same target.

    With two repeats this is |r1 - r2|; with more, the mean over all pairs.
    Averaged over targets that have at least two graded repeats.
    """
    per_seed: Dict[int, float] = {}
    for seed, group in by_seed(graded_cells(cells, arm)).items():
        vals = [float(c.get(metric) or 0.0) for c in group]
        if len(vals) < 2:
            continue
        diffs = [abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:]]
        per_seed[seed] = _mean(diffs)
    return {"n_targets": len(per_seed), "floor": _mean(list(per_seed.values())), "per_target": per_seed}


# --- Complementarity ---------------------------------------------------------------------


def detected_ids(cell: Dict[str, Any]) -> set:
    return {e["id"] for e in cell.get("per_entry") or [] if e.get("detected") and not str(e.get("kind", "seeded")).startswith("decoy_")}


def seeded_total(cell: Dict[str, Any]) -> int:
    n = cell.get("ground_truth_total")
    if isinstance(n, int) and n > 0:
        return n
    return sum(1 for e in cell.get("per_entry") or [] if not str(e.get("kind", "seeded")).startswith("decoy_"))


def union_gain(cells: Iterable[Dict[str, Any]], arm_a: str, arm_b: str) -> Dict[str, Any]:
    """Union recall of two arms per (seed, repeat), vs the better single arm and vs independence.

    ``complementarity`` = union - max(single). ``expected_independent`` =
    1 - (1 - r_a)(1 - r_b): the union two unrelated samplers would reach. Observed
    union well above that means the arms find different bugs; observed at or
    below it means the second arm is a second sample of the same distribution,
    and a second repeat of the first arm would do as well for less maintenance.
    """
    cells = list(cells)
    a_cells = {(int(c["seed"]), int(c.get("repeat", 0))): c for c in graded_cells(cells, arm_a)}
    b_cells = {(int(c["seed"]), int(c.get("repeat", 0))): c for c in graded_cells(cells, arm_b)}
    rows = []
    for key in sorted(set(a_cells) & set(b_cells)):
        ca, cb = a_cells[key], b_cells[key]
        total = seeded_total(ca) or seeded_total(cb)
        if not total:
            continue
        da, db = detected_ids(ca), detected_ids(cb)
        ra, rb = len(da) / total, len(db) / total
        union = len(da | db) / total
        inter = len(da & db)
        jaccard = inter / len(da | db) if (da | db) else 1.0
        rows.append({
            "seed": key[0], "repeat": key[1], "recall_a": ra, "recall_b": rb, "union_recall": union,
            "complementarity": union - max(ra, rb), "expected_independent": 1 - (1 - ra) * (1 - rb),
            "jaccard": jaccard,
            "only_a": sorted(da - db), "only_b": sorted(db - da),
        })
    return {
        "n_pairs": len(rows),
        "mean_union_recall": _mean([r["union_recall"] for r in rows]),
        "mean_best_single": _mean([max(r["recall_a"], r["recall_b"]) for r in rows]),
        "mean_complementarity": _mean([r["complementarity"] for r in rows]),
        "mean_expected_independent": _mean([r["expected_independent"] for r in rows]),
        "mean_jaccard": _mean([r["jaccard"] for r in rows]),
        "pairs": rows,
    }


# --- Cost and validity -------------------------------------------------------------------


def cost_summary(cells: Iterable[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    g = graded_cells(cells, arm)
    tokens = [float(c.get("total_tokens") or 0) for c in g]
    tps = [int(c.get("detected") or 0) for c in g]
    decoy_hits = [int(c.get("decoys_hit") or 0) for c in g]
    tok_per_tp = [t / n for t, n in zip(tokens, tps) if n > 0]
    penalised = [t / (n - d) for t, n, d in zip(tokens, tps, decoy_hits) if n - d > 0]
    return {
        "n": len(g),
        "mean_tokens": _mean(tokens),
        "mean_tokens_per_tp": _mean(tok_per_tp) if tok_per_tp else None,
        "mean_tokens_per_tp_penalised": _mean(penalised) if penalised else None,
        "per_target_tokens_per_tp": {
            seed: _mean([t / n for t, n in ((float(c.get("total_tokens") or 0), int(c.get("detected") or 0)) for c in group) if n > 0] or [0.0])
            for seed, group in by_seed(g).items()
        },
    }


def validity_summary(cells: Iterable[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    allc = [c for c in cells if c.get("arm") == arm]
    g = graded_cells(allc)
    return {
        "cells": len(allc),
        "graded": len(g),
        "ungraded": len(allc) - len(g),
        "ungraded_rate": (len(allc) - len(g)) / len(allc) if allc else 0.0,
        "zero_finding_rate": _mean([1.0 if c.get("zero_findings_complete") else 0.0 for c in g]),
        "cap_hit_rate": _mean([1.0 if c.get("cap_hit") else 0.0 for c in g]),
        "rearm_rate": _mean([1.0 if int(c.get("abort_guard_rearms") or 0) > 0 else 0.0 for c in g]),
        "mean_decoy_fp_rate": _mean([float(c.get("decoy_fp_rate") or 0.0) for c in g]),
        "mean_precision_strict": _mean([float(c.get("precision") or 0.0) for c in g]),
        "mean_precision_lenient": _mean([float(c.get("precision_lenient") if c.get("precision_lenient") is not None else c.get("precision") or 0.0) for c in g]),
        "mean_unwindowed_reads": _mean([float((c.get("telemetry") or {}).get("unwindowed_reads", 0)) for c in g]),
        "mean_structural_nav_calls": _mean([float((c.get("telemetry") or {}).get("structural_nav_calls", 0)) for c in g]),
    }


# --- The gate ---------------------------------------------------------------------------

GATE_WORST_TARGET_FLOOR = -0.10
GATE_EFFICIENCY_IMPROVEMENT = 0.20
GATE_DECOY_FP_TOLERANCE = 0.05
GATE_FAILURE_RATE_TOLERANCE = 0.0


def evaluate_gate(cells: Iterable[Dict[str, Any]], baseline: str, candidate: str, *, n_boot: int = 2000,
                  holdout: bool = False) -> Dict[str, Any]:
    """Applies the change gate. Returns each criterion with its numbers and a verdict.

    1. Paired-bootstrap 95% CI of mean delta-recall over targets has lower bound > 0,
       OR the CI contains 0 and tokens-per-TP improves by >= 20% with its own CI
       excluding 0 (a pure efficiency win).
    2. Mean delta-recall exceeds 2x the baseline's A/A noise floor.
    3. Worst-target delta-recall >= -0.10.
    4. Decoy false-positive rate and failure rates (ungraded, zero-finding, re-armed)
       do not increase beyond tolerance.
    """
    cells = list(cells)
    deltas = paired_deltas(cells, baseline, candidate, "recall")
    ci = bootstrap_ci(list(deltas.values()), n_boot=n_boot)
    floor = aa_noise_floor(cells, baseline, "recall")

    base_cost = cost_summary(cells, baseline)
    cand_cost = cost_summary(cells, candidate)
    eff_deltas = []
    for seed in sorted(set(base_cost["per_target_tokens_per_tp"]) & set(cand_cost["per_target_tokens_per_tp"])):
        b = base_cost["per_target_tokens_per_tp"][seed]
        c = cand_cost["per_target_tokens_per_tp"][seed]
        if b > 0 and c > 0:
            eff_deltas.append((b - c) / b)  # positive = candidate cheaper per TP
    eff_ci = bootstrap_ci(eff_deltas, n_boot=n_boot, seed=1)

    base_v = validity_summary(cells, baseline)
    cand_v = validity_summary(cells, candidate)

    c1_recall = ci["n"] >= 2 and ci["lo"] > 0
    c1_eff = (ci["n"] >= 2 and ci["lo"] <= 0 <= ci["hi"]
              and eff_ci["n"] >= 2 and eff_ci["mean"] >= GATE_EFFICIENCY_IMPROVEMENT and eff_ci["lo"] > 0)
    c2 = ci["n"] >= 2 and floor["n_targets"] >= 1 and ci["mean"] > 2 * floor["floor"]
    worst = min(deltas.values()) if deltas else 0.0
    c3 = bool(deltas) and worst >= GATE_WORST_TARGET_FLOOR
    c4_decoy = cand_v["mean_decoy_fp_rate"] <= base_v["mean_decoy_fp_rate"] + GATE_DECOY_FP_TOLERANCE
    c4_fail = all(
        cand_v[k] <= base_v[k] + GATE_FAILURE_RATE_TOLERANCE
        for k in ("ungraded_rate", "zero_finding_rate", "rearm_rate")
    )
    criteria = {
        "1_recall_ci_above_zero_or_efficiency_win": {
            "pass": bool(c1_recall or c1_eff),
            "delta_recall_ci": ci, "tokens_per_tp_improvement_ci": eff_ci,
            "via": "recall" if c1_recall else ("efficiency" if c1_eff else None),
        },
        "2_clears_twice_aa_noise_floor": {"pass": bool(c2), "delta_mean": ci["mean"], "aa_floor": floor["floor"], "floor_targets": floor["n_targets"]},
        "3_worst_target_delta": {"pass": bool(c3), "worst_target_delta": worst, "floor": GATE_WORST_TARGET_FLOOR, "per_target": deltas},
        "4_no_new_false_positives_or_failures": {
            "pass": bool(c4_decoy and c4_fail),
            "decoy_fp_rate": {"baseline": base_v["mean_decoy_fp_rate"], "candidate": cand_v["mean_decoy_fp_rate"]},
            "ungraded_rate": {"baseline": base_v["ungraded_rate"], "candidate": cand_v["ungraded_rate"]},
            "zero_finding_rate": {"baseline": base_v["zero_finding_rate"], "candidate": cand_v["zero_finding_rate"]},
            "rearm_rate": {"baseline": base_v["rearm_rate"], "candidate": cand_v["rearm_rate"]},
        },
    }
    verdict = all(c["pass"] for c in criteria.values())
    return {
        "baseline": baseline, "candidate": candidate, "holdout": holdout,
        "n_targets_paired": ci["n"],
        "verdict": "PASS" if verdict else "FAIL",
        "criteria": criteria,
        "union": union_gain(cells, baseline, candidate),
        "validity": {"baseline": base_v, "candidate": cand_v},
        "cost": {"baseline": {k: v for k, v in base_cost.items() if k != "per_target_tokens_per_tp"},
                 "candidate": {k: v for k, v in cand_cost.items() if k != "per_target_tokens_per_tp"}},
        "caveat": (
            "Holdout confirmation: one shot. Do not re-run to get a different answer."
            if holdout else
            "Dev numbers. A change that passes here still needs one holdout confirmation before it lands."
        ),
    }


# --- Rendering ---------------------------------------------------------------------------


def render_report(gate: Dict[str, Any]) -> str:
    c = gate["criteria"]
    ci = c["1_recall_ci_above_zero_or_efficiency_win"]["delta_recall_ci"]
    eff = c["1_recall_ci_above_zero_or_efficiency_win"]["tokens_per_tp_improvement_ci"]
    u = gate["union"]
    vb, vc = gate["validity"]["baseline"], gate["validity"]["candidate"]
    lines = [
        f"== {gate['candidate']} vs {gate['baseline']}  ({gate['n_targets_paired']} paired targets){'  [HOLDOUT]' if gate['holdout'] else ''}",
        f"verdict: {gate['verdict']}",
        f"  1. delta recall mean {ci['mean']:+.3f}  95% CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}]  "
        f"| tokens/TP improvement {eff['mean']:+.1%} CI [{eff['lo']:+.1%}, {eff['hi']:+.1%}]  -> {'PASS' if c['1_recall_ci_above_zero_or_efficiency_win']['pass'] else 'FAIL'}",
        f"  2. A/A noise floor {c['2_clears_twice_aa_noise_floor']['aa_floor']:.3f} over {c['2_clears_twice_aa_noise_floor']['floor_targets']} targets; need > {2 * c['2_clears_twice_aa_noise_floor']['aa_floor']:.3f}  -> {'PASS' if c['2_clears_twice_aa_noise_floor']['pass'] else 'FAIL'}",
        f"  3. worst-target delta {c['3_worst_target_delta']['worst_target_delta']:+.3f} (floor {GATE_WORST_TARGET_FLOOR:+.2f})  -> {'PASS' if c['3_worst_target_delta']['pass'] else 'FAIL'}",
        f"  4. decoy FP rate {vb['mean_decoy_fp_rate']:.3f} -> {vc['mean_decoy_fp_rate']:.3f}; ungraded {vb['ungraded_rate']:.2f} -> {vc['ungraded_rate']:.2f}; "
        f"zero-finding {vb['zero_finding_rate']:.2f} -> {vc['zero_finding_rate']:.2f}; re-armed {vb['rearm_rate']:.2f} -> {vc['rearm_rate']:.2f}  -> {'PASS' if c['4_no_new_false_positives_or_failures']['pass'] else 'FAIL'}",
        f"  union: recall {u['mean_union_recall']:.3f} vs best single {u['mean_best_single']:.3f} (complementarity {u['mean_complementarity']:+.3f}; "
        f"independence would give {u['mean_expected_independent']:.3f}; jaccard {u['mean_jaccard']:.2f}) over {u['n_pairs']} pairs",
        f"  reads: whole-file {vb['mean_unwindowed_reads']:.1f} -> {vc['mean_unwindowed_reads']:.1f}; nav calls {vb['mean_structural_nav_calls']:.1f} -> {vc['mean_structural_nav_calls']:.1f}",
        f"  {gate['caveat']}",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--candidate", default="structural")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)

    from evals.sweep import MANIFEST_NAME, load_cells

    cells = load_cells(args.sweep_dir)
    manifest = {}
    try:
        manifest = json.loads((args.sweep_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    gate = evaluate_gate(cells, args.baseline, args.candidate, n_boot=args.n_boot, holdout=bool(manifest.get("holdout")))
    if args.json:
        print(json.dumps(gate, indent=2, sort_keys=True, default=str))
    else:
        print(render_report(gate))
    return 0 if gate["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
