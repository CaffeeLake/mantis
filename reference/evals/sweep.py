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

"""Paired, resumable researcher sweeps over generated targets.

A sweep is a matrix of cells ``{arm x seed x repeat}``. Everything about how
it runs exists to keep single-run noise from reading as a result:

* **Interleaved in time.** For every (seed, repeat) the arms run back to back,
  so model-serving drift hits both arms equally instead of landing on
  whichever arm ran in the afternoon.
* **One file per cell, completion marked only after a normal return.** A cell
  directory holds ``result.json`` and, once the run came back graded, a
  ``.complete`` marker (the Harness2 runmark rule: presence of output is not
  completion). Re-running the sweep skips complete cells and re-rolls
  incomplete ones up to ``max_attempts``; after that the last result is kept
  and the cell is marked exhausted so the matrix terminates.
* **Identity-locked.** ``sweep.json`` records model, effort, arms, seeds,
  size, caps, every arm's configuration fingerprint and every target's
  ground-truth version. Resuming with a different identity is refused: two
  sweeps that measured different things must not share a directory.
* **Holdout refused by default.** Seeds in the holdout range need
  ``--confirm-holdout``; the manifest then carries ``holdout: true`` so
  ``compare.py`` labels the result as a confirmation, not a dev number.

The sweep calls ``evals.research_eval.eval_researcher`` once per cell; the
function is injectable so the scheduling logic is unit-tested without a model.

Usage::

    python3 evals/sweep.py --out runs/sweep-1 --arms baseline,structural \\
        --seeds 1000-1009 --repeats 3 --size medium --max-tokens 1500000 \\
        --model vertex_ai/gemini-3.8-flash --effort low
    python3 evals/compare.py runs/sweep-1 --baseline baseline --candidate structural
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # pragma: no cover - direct CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.research_targets import gen as targets  # noqa: E402

MANIFEST_NAME = "sweep.json"
COMPLETE_MARKER = ".complete"
RESULT_NAME = "result.json"
ATTEMPTS_NAME = "attempts.json"


# --- Configuration ---------------------------------------------------------------------


@dataclass
class SweepConfig:
    out_dir: str
    arms: Dict[str, str]                 # label -> toolset
    seeds: List[int]
    repeats: int = 3
    size: str = "medium"
    model: str = ""
    effort: str = "low"
    max_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_attempts: int = 2
    confirm_holdout: bool = False
    prompt_override: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.model:
            self.model = os.environ.get("EVAL_MODEL", "vertex_ai/gemini-3.8-flash")
        if not self.effort:
            self.effort = os.environ.get("EVAL_REASONING_EFFORT", "low")

    def identity(self) -> Dict[str, Any]:
        """The fields two sweeps must share to be the same measurement."""
        return {
            "arms": dict(sorted(self.arms.items())),
            "seeds": sorted(self.seeds),
            "repeats": self.repeats,
            "size": self.size,
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "max_tool_calls": self.max_tool_calls,
            "prompt_override": self.prompt_override or "",
        }


def parse_seeds(spec: str) -> List[int]:
    """'1000-1009', '1000,1001', 'dev', 'dev:10', 'holdout:8'."""
    spec = spec.strip()
    if spec.startswith(("dev", "holdout")):
        name, _, count = spec.partition(":")
        rng = targets.DEV_SEEDS if name == "dev" else targets.HOLDOUT_SEEDS
        seeds = list(rng)
        return seeds[: int(count)] if count else seeds
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_arms(spec: str) -> Dict[str, str]:
    """'baseline,structural' or 'ctl=baseline,nav=structural' -> {label: toolset}."""
    arms: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        label, _, toolset = part.partition("=")
        arms[label.strip()] = (toolset or label).strip()
    if len(arms) < 1:
        raise ValueError("at least one arm is required")
    return arms


# --- Cells ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    arm: str
    seed: int
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.arm}__{self.seed}__r{self.repeat}"


def cell_order(arms: Sequence[str], seeds: Sequence[int], repeats: int) -> List[Cell]:
    """Interleaved order: all arms for one (seed, repeat) before the next.

    Repeats are the outer loop so that a sweep stopped early still has one
    full repeat across every seed and arm rather than three repeats of the
    first few seeds.
    """
    cells: List[Cell] = []
    for r in range(repeats):
        for seed in seeds:
            for arm in arms:
                cells.append(Cell(arm, seed, r))
    return cells


def cell_dir(out_dir: Path, cell: Cell) -> Path:
    return Path(out_dir) / "cells" / cell.key


def is_complete(out_dir: Path, cell: Cell) -> bool:
    return (cell_dir(out_dir, cell) / COMPLETE_MARKER).exists()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def record_cell(out_dir: Path, cell: Cell, result: Dict[str, Any], *, max_attempts: int) -> str:
    """Stores a cell result; marks complete only when graded (or exhausted).

    Returns "complete", "retry" or "exhausted".
    """
    d = cell_dir(out_dir, cell)
    d.mkdir(parents=True, exist_ok=True)
    attempts = _read_json(d / ATTEMPTS_NAME, {"attempts": 0})
    attempts["attempts"] = int(attempts.get("attempts", 0)) + 1
    attempts["last_status"] = result.get("status")
    _write_json_atomic(d / ATTEMPTS_NAME, attempts)
    slim = {k: v for k, v in result.items() if k != "findings_full"}
    slim.update({"arm": cell.arm, "seed": cell.seed, "repeat": cell.repeat, "attempt": attempts["attempts"]})
    _write_json_atomic(d / RESULT_NAME, slim)
    _write_json_atomic(d / "findings_full.json", result.get("findings_full", []))
    if result.get("graded", False):
        _write_json_atomic(d / COMPLETE_MARKER, {"at": time.time(), "attempt": attempts["attempts"], "exhausted": False})
        return "complete"
    if attempts["attempts"] >= max_attempts:
        _write_json_atomic(d / COMPLETE_MARKER, {"at": time.time(), "attempt": attempts["attempts"], "exhausted": True})
        return "exhausted"
    return "retry"


def load_cells(out_dir: Path) -> List[Dict[str, Any]]:
    """Every recorded cell result (complete or not), newest attempt only."""
    out: List[Dict[str, Any]] = []
    cells_root = Path(out_dir) / "cells"
    if not cells_root.is_dir():
        return out
    for d in sorted(cells_root.iterdir()):
        res = _read_json(d / RESULT_NAME, None)
        if isinstance(res, dict):
            marker = _read_json(d / COMPLETE_MARKER, None)
            res["complete"] = marker is not None
            res["exhausted"] = bool(marker and marker.get("exhausted"))
            out.append(res)
    return out


# --- Manifest / identity lock -----------------------------------------------------------


def check_or_write_manifest(out_dir: Path, cfg: SweepConfig, extra: Dict[str, Any]) -> Dict[str, Any]:
    """Writes sweep.json on first run; on later runs refuses an identity mismatch."""
    path = Path(out_dir) / MANIFEST_NAME
    identity = cfg.identity()
    existing = _read_json(path, None)
    if existing is None:
        manifest = {"identity": identity, "created_at": time.time(), **extra}
        _write_json_atomic(path, manifest)
        return manifest
    if existing.get("identity") != identity:
        diff = {k: (existing.get("identity", {}).get(k), identity.get(k))
                for k in identity if existing.get("identity", {}).get(k) != identity.get(k)}
        raise RuntimeError(
            f"{path} belongs to a different sweep; refusing to mix measurements. "
            f"Differences (existing, requested): {diff}. Use a new --out directory."
        )
    return existing


# --- Targets ----------------------------------------------------------------------------


def ensure_target(out_dir: Path, seed: int, size: str) -> Tuple[Path, str]:
    """Generates the target for ``seed`` once; returns (dir, ground-truth version)."""
    from evals.research_eval import GROUND_TRUTH_BASENAME, ground_truth_version

    tdir = Path(out_dir) / "targets" / f"{seed}_{size}"
    gt_path = tdir / GROUND_TRUTH_BASENAME
    if not gt_path.exists():
        targets.write_target(tdir, seed, size)
    return tdir, ground_truth_version(gt_path)


# --- Running ----------------------------------------------------------------------------

EvalFn = Callable[..., Awaitable[Dict[str, Any]]]


async def run_sweep(cfg: SweepConfig, eval_fn: Optional[EvalFn] = None, *, dry_run: bool = False,
                    log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Runs (or resumes) a sweep. Returns a summary dict."""
    if eval_fn is None:
        from evals.research_eval import eval_researcher
        eval_fn = eval_researcher

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout = [s for s in cfg.seeds if targets.seed_split(s) == "holdout"]
    if holdout and not cfg.confirm_holdout:
        raise targets.HoldoutViolation(
            f"seeds {holdout} are holdout seeds; pass --confirm-holdout for a one-shot "
            "confirmation run (never for tuning, proposing, or reflecting)"
        )
    for s in cfg.seeds:
        if targets.seed_split(s) != "holdout":
            targets.assert_not_holdout(s, "sweep")

    from evals.research_eval import RESEARCH_PROMPT, config_fingerprint, resolve_toolset

    arm_tools = {label: resolve_toolset(toolset) for label, toolset in cfg.arms.items()}
    prompt_text = cfg.prompt_override or RESEARCH_PROMPT
    gt_versions: Dict[str, str] = {}
    target_dirs: Dict[int, Path] = {}
    for seed in cfg.seeds:
        tdir, gtv = ensure_target(out_dir, seed, cfg.size)
        target_dirs[seed] = tdir
        gt_versions[str(seed)] = gtv
    fingerprints = {
        label: config_fingerprint(cfg.model, cfg.effort, tools, prompt_text, "")
        for label, tools in arm_tools.items()
    }
    manifest = check_or_write_manifest(
        out_dir, cfg,
        {"holdout": bool(holdout), "arm_fingerprints": fingerprints, "ground_truth_versions": gt_versions,
         "split": sorted({targets.seed_split(s) for s in cfg.seeds})},
    )
    if manifest.get("ground_truth_versions") != gt_versions:
        raise RuntimeError("generated targets differ from the manifest's ground-truth versions; "
                           "the generator changed under this sweep. Use a new --out directory.")

    cells = cell_order(list(cfg.arms), cfg.seeds, cfg.repeats)
    todo = [c for c in cells if not is_complete(out_dir, c)]
    log(f"sweep {out_dir}: {len(cells)} cells, {len(cells) - len(todo)} complete, {len(todo)} to run"
        f"{' [HOLDOUT CONFIRMATION]' if holdout else ''}")
    if dry_run:
        for c in todo:
            log(f"  would run {c.key}")
        return {"cells": len(cells), "todo": len(todo), "ran": 0, "dry_run": True}

    ran = 0
    statuses: Dict[str, int] = {}
    for c in todo:
        for attempt in range(cfg.max_attempts):
            started = time.time()
            try:
                result = await eval_fn(
                    cfg.model, cfg.effort,
                    target_dir=target_dirs[c.seed],
                    toolset=cfg.arms[c.arm],
                    prompt_override=cfg.prompt_override,
                    max_tokens=cfg.max_tokens,
                    max_tool_calls=cfg.max_tool_calls,
                    arm_label=c.arm,
                )
            except Exception as exc:  # the eval swallows non-auth errors itself; this is a crash in the harness
                from core.config import is_auth_error
                if is_auth_error(exc):
                    raise
                result = {"graded": False, "status": "ungraded", "run_errors": [f"sweep: {exc!r}"[:300]],
                          "recall": 0.0, "precision": 0.0, "total_tokens": 0}
            result.setdefault("wall_seconds", round(time.time() - started, 1))
            outcome = record_cell(out_dir, c, result, max_attempts=cfg.max_attempts)
            ran += 1
            statuses[outcome] = statuses.get(outcome, 0) + 1
            log(f"  {c.key}: {result.get('status', '?')} recall={result.get('recall')} "
                f"tokens={result.get('total_tokens')} -> {outcome}")
            if outcome != "retry":
                break
    return {"cells": len(cells), "todo": len(todo), "ran": ran, "outcomes": statuses, "dry_run": False}


# --- CLI --------------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> SweepConfig:
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else None
    return SweepConfig(
        out_dir=str(args.out),
        arms=parse_arms(args.arms),
        seeds=parse_seeds(args.seeds),
        repeats=args.repeats,
        size=args.size,
        model=args.model or "",
        effort=args.effort,
        max_tokens=args.max_tokens,
        max_tool_calls=args.max_tool_calls,
        max_attempts=args.max_attempts,
        confirm_holdout=args.confirm_holdout,
        prompt_override=prompt,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="baseline,structural", help="label=toolset,... (label defaults to toolset)")
    parser.add_argument("--seeds", default="dev:10", help="'1000-1009', '1000,1002', 'dev[:N]', 'holdout[:N]'")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--size", default="medium")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--confirm-holdout", action="store_true",
                        help="Allow holdout seeds. One-shot confirmation only; never for tuning.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("VERTEXAI_LOCATION", "global")
    summary = asyncio.run(run_sweep(_build_config(args), dry_run=args.dry_run))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
