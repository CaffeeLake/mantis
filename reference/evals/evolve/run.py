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

"""One Harness2 iteration over a Mantis harness, and adoption of what it produced.

``evolve_task`` runs the loop upstream's ``schedule._run_one_mode`` runs,
with Mantis's adapter, proposer and selector:

    k base rollouts  ->  recursion.run_mode (2k proposals, each rolled once;
    k compositions, each rolled once)  ->  evidence-first selection among the
    finals  ->  reflection into an <experience>  ->  library entry  ->  an
    adoption record with the diff a human reads before anything changes.

Nothing here writes to the production harness. ``adopt`` copies an adopted
tree over ``reference/workflow.json`` (and prompt files) only when invoked
explicitly, after printing the diff; ``--yes`` is required to proceed, and
the result is validated through the production loader before it is written.

Cost is disclosed before a model call is made: at k=1 one iteration is
k + 2k + k = 4 pipeline rollouts plus 3 proposer sessions per task.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - direct CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evals.evolve import selection as ev_selection  # noqa: E402
from evals.evolve import surface as hsurface  # noqa: E402
from evals.evolve import surface_text  # noqa: E402
from evals.evolve._vendor import load_harness2  # noqa: E402
from evals.evolve.adapter import MantisEvalAdapter, MantisRepoAdapter, RunSettings  # noqa: E402
from evals.evolve.proposer import MantisProposer  # noqa: E402

ADOPTION_FILE = "adoption.json"


def rollouts_per_iteration(k: int) -> Dict[str, int]:
    return {"base_rollouts": k, "candidate_rollouts": 2 * k, "final_rollouts": k,
            "rollouts": 4 * k, "proposer_sessions": 3 * k}


def _run_id_of(task_id: str, label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in task_id)
    return f"{safe}--{label}"


def evolve_task(task_id: str, *, adapter, proposer, selector, work_dir: Path, k: int = 1, mode: str = "agg",
                library=None, log=print) -> Dict[str, Any]:
    """One task-level recursion step. Returns the adoption record (also written to disk)."""
    h2 = load_harness2()
    from harness2.core import library as h2lib
    from harness2.core import recursion, runmark
    from harness2.core.spec import RunView

    surface_text.register()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base_tree = adapter.base_tree()
    surface = adapter.substrate.surface()

    # 1. k base rollouts of the unmodified harness.
    base: List[RunView] = []
    for i in range(1, k + 1):
        label = f"base{i}"
        rid = _run_id_of(task_id, label)
        log(f"[{task_id}] base rollout {label} ({rid})")
        try:
            rd = adapter.execute(task_id, None, rid)
            rv = RunView(label=label, run_id=rid, run_dir=rd, ok=rd.exists() and runmark.is_complete(rd, harness_dir=None),
                         delivered=adapter.delivered(rd))
            if not rv.ok:
                rv.error = runmark.why_not(rd, harness_dir=None)
        except Exception as exc:
            rv = RunView(label=label, run_id=rid, run_dir=work_dir / "missing", ok=False, delivered=False,
                         error=f"{type(exc).__name__}: {exc}")
        base.append(rv)
    if not any(r.ok for r in base):
        record = {"task_id": task_id, "status": "base-failed", "errors": [r.error for r in base]}
        _write_adoption(work_dir, record)
        return record

    # 2. Propose, probe, compose (upstream).
    lib_dir = library.context_dir() if library else None
    result = recursion.run_mode(
        mode, task_id, k, base,
        adapter=adapter, prop=proposer, work_dir=work_dir,
        run_id_of=lambda lbl: _run_id_of(task_id, lbl),
        base_tree=base_tree, library_dir=lib_dir,
    )

    # 3. Evidence-first selection among the finals.
    finals = {label: result.rollouts[label] for label in result.finals if label in result.rollouts}
    sel = ev_selection.select(task_id, k, mode, finals, base, adapter=adapter, selector=selector, work_dir=work_dir)

    # 4. Reflection and library, only on a real pick.
    experience: Dict[str, Any] = {}
    if sel.ok:
        try:
            from harness2.core import selection as h2sel
            experience = h2sel.reflect_adoption(
                task_id, sel, result.rollouts, adapter=adapter, prop=selector, work_dir=work_dir,
                decisions={s.label: s.notes or s.error for s in result.stages})
        except Exception as exc:
            log(f"[{task_id}] reflection failed: {type(exc).__name__}: {exc}")
            experience = {}
        if library is not None and h2lib.has_experience(experience):
            chosen_tree = result.trees.get(sel.chosen_label, base_tree)
            library.add(task_id, chosen_tree, why=sel.why, label=sel.chosen_label, run_id=sel.chosen_run_id, experience=experience)

    # 5. Adoption record: the diff a human reads.
    chosen_tree = result.trees.get(sel.chosen_label, base_tree) if sel.ok else base_tree
    kept_base = sel.ok and _trees_equal(surface, base_tree, chosen_tree)
    record = {
        "task_id": task_id, "mode": mode, "k": k, "status": "ok" if sel.ok else "no-adoption",
        "selection": {k_: v for k_, v in sel.to_dict().items() if k_ != "experience"},
        "experience": experience,
        "kept_base": kept_base,
        "stages": [{"label": s.label, "dim": s.dim, "fell_back": s.fell_back, "kept_base": s.kept_base,
                    "attempts": s.attempts, "error": s.error[:200], "notes": (s.notes or "")[:2000]} for s in result.stages],
        "rollouts": {label: {"ok": rv.ok, "delivered": rv.delivered, "run_dir": str(rv.run_dir), "error": rv.error}
                     for label, rv in result.rollouts.items()},
        "ledger": result.ledger.to_dict() if result.ledger else None,
        "chosen_label": sel.chosen_label if sel.ok else "",
        "chosen_harness_dir": str(work_dir / "harnesses" / sel.chosen_label) if sel.ok else "",
        "diff": hsurface.diff_summary(base_tree, chosen_tree),
        "unified_diff": unified_diff(base_tree, chosen_tree),
        "cost": rollouts_per_iteration(k),
        "warnings": _warnings(result, sel, kept_base),
        "at": time.time(),
    }
    if sel.ok and not kept_base:
        adopted_dir = work_dir / "adopted"
        if adopted_dir.exists():
            shutil.rmtree(adopted_dir)
        from harness2.core import tree as h2tree
        h2tree.write_tree(surface, chosen_tree, adopted_dir, stubs=False)
        record["adopted_dir"] = str(adopted_dir)
    _write_adoption(work_dir, record)
    return record


def _trees_equal(surface, a: Dict[str, str], b: Dict[str, str]) -> bool:
    load_harness2()
    from harness2.core import tree as h2tree
    d = h2tree.diff_summary(surface, a, b)
    return not (d["added"] or d["removed"] or d["modified"])


def _warnings(result, sel, kept_base: bool) -> List[str]:
    out: List[str] = []
    fell = [s.label for s in result.stages if s.fell_back]
    if fell:
        out.append(f"stages fell back to the inherited harness after an invalid proposal: {fell}")
    if kept_base:
        out.append("the adopted harness equals the base: the proposer found nothing the evidence supported (KEEP_BASE)")
    if not sel.ok:
        out.append(f"no adoption: {sel.error}")
    if result.k == 1:
        out.append("k=1: one proposal per dimension and one composition; a single rollout per candidate is weak evidence")
    return out


def unified_diff(base: Dict[str, str], proposed: Dict[str, str]) -> str:
    chunks: List[str] = []
    for rel in sorted(set(base) | set(proposed)):
        a, b = base.get(rel, ""), proposed.get(rel, "")
        if a == b:
            continue
        chunks.append("".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{rel}", f"b/{rel}")))
    return "".join(chunks)


def _write_adoption(work_dir: Path, record: Dict[str, Any]) -> None:
    (Path(work_dir) / ADOPTION_FILE).write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


# --- Adoption into the production harness ----------------------------------------------


def adopt(adopted_dir: Path, workflow_path: Path, *, yes: bool, log=print) -> bool:
    """Writes an adopted tree over the production workflow. Prints the diff first; needs --yes."""
    adopted_dir, workflow_path = Path(adopted_dir), Path(workflow_path)
    base_tree = hsurface.base_tree(workflow_path)
    tree = hsurface.read_tree(adopted_dir)
    merged = dict(base_tree)
    merged.update(tree)
    diff = unified_diff(base_tree, merged)
    if not diff.strip():
        log("nothing to adopt: the harness equals the current workflow")
        return False
    log(diff)
    try:
        hsurface.validate_tree(merged, base_tree)
    except ValueError as exc:
        log(f"refusing to adopt: {exc}")
        return False
    if not yes:
        log("dry run: pass --yes to write this harness over " + str(workflow_path))
        return False
    compiled = hsurface.compile_tree(merged)
    backup = workflow_path.with_suffix(".json.bak")
    shutil.copy2(workflow_path, backup)
    workflow_path.write_text(json.dumps(compiled, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {workflow_path} (previous copy at {backup})")
    return True


# --- CLI ----------------------------------------------------------------------------------


def _make_adapter(args) -> Any:
    settings = RunSettings(model=args.model or "", effort=args.effort or "", token_budget=args.token_budget,
                           max_llm_calls=args.max_llm_calls, timeout_seconds=args.rollout_timeout)
    if args.eval_seeds:
        from evals.sweep import parse_seeds
        return MantisEvalAdapter(seeds=parse_seeds(args.eval_seeds), size=args.eval_size, workflow_path=args.workflow,
                                 runs_root=Path(args.out) / "runs", settings=settings, domain="generated")
    targets = {Path(t).name or f"task{i}": Path(t) for i, t in enumerate(args.target or [])}
    return MantisRepoAdapter(targets=targets, workflow_path=args.workflow, runs_root=Path(args.out) / "runs",
                             settings=settings, domain=args.domain or (Path(args.target[0]).name if args.target else "repo"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="One Propose-Probe-Compose iteration over a target.")
    run.add_argument("--target", action="append", help="Target directory (repeatable: one task each).")
    run.add_argument("--eval-seeds", default=None, help="Use generated dev targets instead (e.g. 'dev:3').")
    run.add_argument("--eval-size", default="small")
    run.add_argument("--workflow", type=Path, default=Path(__file__).resolve().parent.parent.parent / "workflow.json")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--k", type=int, default=1)
    run.add_argument("--mode", choices=["agg", "seq"], default="agg")
    run.add_argument("--model", default=None)
    run.add_argument("--effort", default=None)
    run.add_argument("--proposer-model", default=None)
    run.add_argument("--token-budget", type=int, default=None, help="Per-rollout token budget passed to main.py.")
    run.add_argument("--max-llm-calls", type=int, default=None)
    run.add_argument("--rollout-timeout", type=int, default=3600)
    run.add_argument("--domain", default=None)
    run.add_argument("--library", type=Path, default=None, help="Library root (default: <out>/library).")
    run.add_argument("--yes", action="store_true", help="Skip the cost confirmation.")

    ad = sub.add_parser("adopt", help="Write an adopted harness over the production workflow.")
    ad.add_argument("adopted_dir", type=Path)
    ad.add_argument("--workflow", type=Path, default=Path(__file__).resolve().parent.parent.parent / "workflow.json")
    ad.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)
    os.environ.setdefault("VERTEXAI_LOCATION", "global")

    if args.cmd == "adopt":
        return 0 if adopt(args.adopted_dir, args.workflow, yes=args.yes) else 1

    if not args.target and not args.eval_seeds:
        parser.error("run needs --target <dir> (repeatable) or --eval-seeds")
    adapter = _make_adapter(args)
    tasks = adapter.tasks()
    cost = rollouts_per_iteration(args.k)
    print(f"evolve: {len(tasks)} task(s) x k={args.k} {args.mode} -> {cost['rollouts']} pipeline rollouts and "
          f"{cost['proposer_sessions']} proposer sessions per task", flush=True)
    if not args.yes:
        answer = input("proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 2

    surface_text.register()
    from harness2.core import library as h2lib

    proposer = MantisProposer(adapter.spec.channels["proposer"], model_id=args.proposer_model or args.model or "",
                              reasoning_effort=args.effort or "", audit_dir=Path(args.out) / "proposer",
                              surface_ok=lambda rel: _on_surface(adapter, rel))
    lib_root = Path(args.library) if args.library else Path(args.out) / "library"
    records = []
    for i, task_id in enumerate(tasks):
        library = h2lib.Library(root=lib_root, domain=adapter.domain_of(task_id), surface=adapter.substrate.surface())
        library.retain(tasks[:i])
        record = evolve_task(task_id, adapter=adapter, proposer=proposer, selector=proposer,
                             work_dir=Path(args.out) / "tasks" / task_id, k=args.k, mode=args.mode, library=library)
        records.append(record)
        print(f"[{task_id}] {record['status']}: chosen={record.get('chosen_label') or '-'} kept_base={record.get('kept_base')}")
        for w in record.get("warnings", []):
            print(f"  warning: {w}")
        if record.get("adopted_dir"):
            print(f"  adopted harness: {record['adopted_dir']}  (review the diff in adoption.json, then "
                  f"`python3 -m evals.evolve.run adopt {record['adopted_dir']} --yes`)")
    return 0 if any(r.get("status") == "ok" for r in records) else 1


def _on_surface(adapter, rel: str) -> bool:
    """The proposer's write tool asks this; it is Mantis's predicate, not upstream's
    (which admits scripts/*.py on every surface)."""
    return hsurface.on_surface(rel, adapter._base_workflow)


if __name__ == "__main__":
    sys.exit(main())
