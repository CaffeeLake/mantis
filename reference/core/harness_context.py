"""Disk-backed evidence tree, leak tripwires, and causal reflection parsing.

Adapted from open-source google-research/harness2 (Apache-2.0) for the Mantis
ADK reference implementation:
  - `ContextTree`, `build_context_tree`, and `mission`: materialize full rollout
    trajectories, deliverables, and candidate harness diffs on disk so proposers
    and reflectors inspect evidence without prompt-window truncation.
  - `assert_no_leak_paths`, `assert_no_grader_verdicts`, and
    `assert_tree_has_no_grader_verdicts`: fail-closed tripwires preventing
    benchmark ground-truth files or grader verdicts from entering model context.
  - `parse_reflection_blocks`: extracts causal `<edits>` attributions
    (`helped`, `hurt`, `no observed effect`, `confounded`) and structured
    `<experience>` lessons (`cue`, `lesson`, `evidence`, `confidence`,
    `target_stage`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import fnmatch
import json
from pathlib import Path
import re
import shutil
from typing import Any


DEFAULT_LEAK_GLOBS: tuple[str, ...] = (
    "*ground_truth*.json",
    "*synthetic_dataset*.json",
    "*review_dataset*.json",
    "*critic_dataset*.json",
    "*calibrate_dataset*.json",
)

GRADER_VERDICT_MARKERS: tuple[str, ...] = (
    "deterministic_eval_summary",
    "deterministic_eval_result",
    "judge_scope_and_deterministic_status",
    "Judge findings on the rendered result",
    '"ground_truth_total"',
    '"cross_file_recall"',
    '"unmatched_findings"',
)

_MARKER_SCAN_SUFFIXES: frozenset[str] = frozenset({
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".log",
    ".py",
    ".js",
    ".sh",
    ".xml",
    ".ini",
    ".toml",
    ".patch",
    ".diff",
    "",
})

VALID_EDIT_VERDICTS: frozenset[str] = frozenset({
    "helped",
    "hurt",
    "no observed effect",
    "confounded",
})

_EDITS_RE = re.compile(r"<edits>(.*?)</edits>", re.DOTALL | re.IGNORECASE)
_EXP_RE = re.compile(
    r"<experience>(.*?)</experience>", re.DOTALL | re.IGNORECASE
)
_EDIT_BULLET_RE = re.compile(
    r"^\s*[-*]\s*(.+?)\s*(?:—|--|:|\s-\s)\s*"
    r"\*?\*?(helped|hurt|no observed effect|confounded)\*?\*?"
    r"(?:\s*(?:—|--|:|\s-\s)\s*(.*))?$",
    re.IGNORECASE,
)


@dataclasses.dataclass
class ContextTree:
    """Represents a materialized on-disk evidence tree and its contents."""

    root: Path
    task_id: str
    rollouts: list[str] = dataclasses.field(default_factory=list)
    deliverables: list[str] = dataclasses.field(default_factory=list)
    harness_files: int = 0
    library: bool = False
    pool: list[str] = dataclasses.field(default_factory=list)
    failed: list[str] = dataclasses.field(default_factory=list)
    deliverables_mode: str = "contents"
    source_files: int = 0
    decisions: int = 0

    @property
    def layout(self) -> str:
        """Returns the evidence-layout listing for existing sections."""
        lines = [
            "context/task.md                  the task as the agent received it"
        ]
        if self.source_files:
            lines.append(
                "context/source/                 task source materials"
            )
        if self.decisions:
            lines.append(
                "context/decisions/              earlier edits and decisions"
            )
        if self.rollouts:
            names = ", ".join(f"{r}.md" for r in self.rollouts[:6])
            more = (
                f" (+{len(self.rollouts) - 6} more)"
                if len(self.rollouts) > 6
                else ""
            )
            lines.append(
                f"context/rollouts/               FULL trajectories: "
                f"{names}{more}"
            )
        if self.deliverables:
            if self.deliverables_mode == "names":
                lines.append(
                    "context/deliverables/<label>/    INDEX.md lists WHICH "
                    "files each rollout produced -- names only."
                )
            else:
                lines.append(
                    "context/deliverables/<label>.md  the findings/artifacts "
                    "each rollout produced"
                )
        if self.harness_files:
            lines.append(
                f"context/harness/                the harness you are editing "
                f"({self.harness_files} files)"
            )
        if self.pool:
            lines.append(
                f"context/pool/                   {len(self.pool)} candidates, "
                "each with its harness and its rollout"
            )
        if self.library:
            lines.append(
                "context/library/                earlier tasks in this area "
                "-- browse INDEX.md first"
            )
        return "\n".join(lines)


def _clean_tree_label(raw: Any) -> str:
    """Validates a ContextTree item label against path traversal."""
    label = str(raw or "").strip()
    if not label or "/" in label or "\\" in label or ".." in label:
        raise ValueError(f"invalid context tree label: {raw!r}")
    return label


def assert_no_leak_paths(
    root: Path | str,
    patterns: Sequence[str] = DEFAULT_LEAK_GLOBS,
) -> None:
    """Rejects benchmark answer-key paths before a context reaches a model.

    Args:
        root: Directory root to scan for leaked ground-truth files.
        patterns: Glob patterns matching prohibited answer-key filenames.

    Raises:
        ValueError: If any file under `root` matches a pattern in `patterns`.
    """
    root_path = Path(root)
    if not patterns or not root_path.is_dir():
        return
    hits: list[str] = []
    for file_path in root_path.rglob("*"):
        if file_path.is_symlink():
            raise ValueError(
                f"symlink prohibited in model context: {file_path}"
            )
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root_path).as_posix()
        for pattern in patterns:
            clean_pat = str(pattern).replace("\\", "/").lstrip("./")
            if (
                fnmatch.fnmatch(rel, clean_pat)
                or fnmatch.fnmatch(rel, f"*/{clean_pat}")
                or fnmatch.fnmatch(file_path.name, clean_pat)
            ):
                hits.append(rel)
                break
    if hits:
        shown = ", ".join(sorted(hits)[:8])
        raise ValueError(
            f"answer-key path(s) would enter model context: {shown}"
        )


def assert_no_grader_verdicts(
    root: Path | str,
    markers: Sequence[str] = GRADER_VERDICT_MARKERS,
) -> None:
    """Refuses a context tree containing grader verdict output.

    Args:
        root: Directory root to scan.
        markers: Substring markers identifying grader verdict output.

    Raises:
        ValueError: If any filename or text file body contains a grader marker.
    """
    root_path = Path(root)
    if not markers or not root_path.is_dir():
        return
    for file_path in sorted(root_path.rglob("*")):
        if file_path.is_symlink():
            raise ValueError(
                f"symlink prohibited in model context: {file_path}"
            )
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root_path).as_posix()
        for marker in markers:
            if marker in rel:
                raise ValueError(
                    f"grader verdict would enter model context: {rel!r} "
                    f"matched marker {marker!r} in its NAME."
                )
        if file_path.suffix.lower() not in _MARKER_SCAN_SUFFIXES:
            continue
        try:
            body = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for marker in markers:
            if marker in body:
                where = body.index(marker)
                snippet = body[max(0, where - 60) : where + 120]
                raise ValueError(
                    f"grader verdict would enter model context: {rel!r} "
                    f"contains marker {marker!r} at offset {where} "
                    f"({snippet!r})."
                )


def assert_tree_has_no_grader_verdicts(
    tree: Mapping[str, str],
    markers: Sequence[str] = GRADER_VERDICT_MARKERS,
) -> None:
    """Checks an in-memory file mapping (`rel -> body`) for grader verdicts.

    Args:
        tree: Mapping of relative file paths to file contents.
        markers: Substring markers identifying grader verdict output.

    Raises:
        ValueError: If any path or body in `tree` contains a grader marker.
    """
    for rel in sorted(tree):
        for marker in markers:
            if marker in rel:
                raise ValueError(
                    f"harness file {rel!r} has grader-verdict marker "
                    f"{marker!r} in its NAME"
                )
        if Path(rel).suffix.lower() not in _MARKER_SCAN_SUFFIXES:
            continue
        body = str(tree[rel])
        for marker in markers:
            if marker in body:
                where = body.index(marker)
                snippet = body[max(0, where - 60) : where + 120]
                raise ValueError(
                    f"harness file {rel!r} contains grader-verdict marker "
                    f"{marker!r} at offset {where} ({snippet!r})"
                )


def render_rollout_markdown(
    transcript_rows: Sequence[Mapping[str, Any]],
    label: str = "rollout",
    obs_trunc: int = 4000,
) -> str:
    """Renders structured transcript events into a full Markdown trajectory.

    Every trajectory event is preserved in sequence; individual observations
    exceeding `obs_trunc` characters receive an explicit renderer-truncation
    marker so reflectors never mistake observation clipping for agent failure.

    Args:
        transcript_rows: Ordered sequence of transcript event dicts.
        label: Rollout identifier for the Markdown header.
        obs_trunc: Per-observation character limit before adding a marker.

    Returns:
        Formatted Markdown string representing the rollout trajectory.
    """
    lines = [f"# Rollout Trajectory: `{label}` ({len(transcript_rows)} events)"]
    for idx, row in enumerate(transcript_rows, 1):
        author = str(row.get("author") or "agent")
        ev_type = str(row.get("type") or "event")
        if ev_type == "tool_call":
            name = str(row.get("name") or "unknown_tool")
            args_text = str(row.get("args") or "{}")
            if obs_trunc > 0 and len(args_text) > obs_trunc:
                cut = len(args_text) - obs_trunc
                args_text = (
                    f"{args_text[:obs_trunc]}... "
                    f"[renderer truncation: {cut} chars omitted]"
                )
            lines.append(f"\n## Event {idx} [{author}] Tool Call: `{name}`")
            lines.append(f"```json\n{args_text}\n```")
        elif ev_type == "tool_response":
            name = str(row.get("name") or "unknown_tool")
            resp_text = str(row.get("response") or "")
            if obs_trunc > 0 and len(resp_text) > obs_trunc:
                cut = len(resp_text) - obs_trunc
                resp_text = (
                    f"{resp_text[:obs_trunc]}... "
                    f"[renderer truncation: {cut} chars omitted]"
                )
            lines.append(f"\n## Event {idx} [{author}] Tool Response: `{name}`")
            lines.append(f"```text\n{resp_text}\n```")
        else:
            text = str(row.get("text") or "")
            if obs_trunc > 0 and len(text) > obs_trunc:
                cut = len(text) - obs_trunc
                text = (
                    f"{text[:obs_trunc]}... "
                    f"[renderer truncation: {cut} chars omitted]"
                )
            lines.append(f"\n## Event {idx} [{author}] Message")
            lines.append(text)
    return "\n".join(lines) + "\n"


def build_context_tree(
    dest: Path | str | None = None,
    task_id: str = "mantis_audit",
    task_prompt: str = "",
    *,
    out_root: Path | str | None = None,
    task_description: str = "",
    rollouts: Mapping[str, Sequence[Mapping[str, Any]] | str] | None = None,
    deliverables: Mapping[str, Any] | None = None,
    harness_files: Mapping[str, str] | None = None,
    decisions: Mapping[str, str] | Sequence[str] | None = None,
    leak_globs: Sequence[str] = DEFAULT_LEAK_GLOBS,
    obs_trunc: int = 4000,
) -> ContextTree:
    """Materializes an on-disk evidence tree and validates leak tripwires.

    Args:
        dest: Directory path where the context tree will be written.
        task_id: Identifier for the audit task or campaign.
        task_prompt: Initial task prompt given to the agent.
        out_root: Alias for `dest`.
        task_description: Alias for `task_prompt`.
        rollouts: Optional mapping of rollout label to transcript rows or text.
        deliverables: Optional mapping of rollout label to deliverable text/JSON.
        harness_files: Optional mapping of relative harness path to content.
        decisions: Optional mapping or sequence of prior decision notes.
        leak_globs: Glob patterns of prohibited ground-truth files.
        obs_trunc: Per-observation truncation limit for rendered trajectories.

    Returns:
        A populated `ContextTree` instance describing the on-disk layout.
    """
    target_dir = dest if dest is not None else out_root
    if target_dir is None:
        raise ValueError("dest or out_root is required")
    prompt_text = task_prompt or task_description
    dest_path = Path(target_dir)
    if dest_path.is_symlink():
        raise ValueError(f"context tree root cannot be a symlink: {dest_path}")
    dest_path.mkdir(parents=True, exist_ok=True)
    for managed in ("task.md", "decisions", "rollouts", "deliverables", "harness"):
        child = dest_path / managed
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    (dest_path / "task.md").write_text(str(prompt_text), encoding="utf-8")

    decision_items: list[tuple[str, str]] = []
    if isinstance(decisions, Mapping):
        decision_items = [(_clean_tree_label(k), str(v)) for k, v in decisions.items()]
    elif isinstance(decisions, Sequence) and not isinstance(decisions, (str, bytes)):
        decision_items = [(f"d{idx:02d}", str(v)) for idx, v in enumerate(decisions, 1)]
    for label, notes in decision_items:
        dec_path = dest_path / "decisions" / f"{label}.md"
        dec_path.parent.mkdir(parents=True, exist_ok=True)
        dec_path.write_text(
            notes or "No decision notes returned.", encoding="utf-8"
        )

    got_rollouts: list[str] = []
    if rollouts:
        (dest_path / "rollouts").mkdir(parents=True, exist_ok=True)
        for raw_label, payload in rollouts.items():
            label = _clean_tree_label(raw_label)
            if isinstance(payload, str):
                text = payload
            else:
                text = render_rollout_markdown(
                    payload, label=label, obs_trunc=obs_trunc
                )
            (dest_path / "rollouts" / f"{label}.md").write_text(
                text, encoding="utf-8"
            )
            got_rollouts.append(label)

    got_deliverables: list[str] = []
    if deliverables:
        (dest_path / "deliverables").mkdir(parents=True, exist_ok=True)
        for raw_label, body in deliverables.items():
            label = _clean_tree_label(raw_label)
            if isinstance(body, (dict, list)):
                rendered_body = json.dumps(body, indent=2, sort_keys=True)
            else:
                rendered_body = str(body or "(no deliverable produced)")
            (dest_path / "deliverables" / f"{label}.md").write_text(
                rendered_body, encoding="utf-8"
            )
            got_deliverables.append(label)

    n_harness = 0
    if harness_files:
        assert_tree_has_no_grader_verdicts(harness_files)
        for rel, body in sorted(harness_files.items()):
            clean_rel = str(rel).strip().replace("\\", "/").lstrip("/")
            if not clean_rel or ".." in clean_rel.split("/"):
                raise ValueError(f"path escapes the harness: {rel!r}")
            target = dest_path / "harness" / clean_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(body), encoding="utf-8")
            n_harness += 1

    assert_no_leak_paths(dest_path, leak_globs)
    assert_no_grader_verdicts(dest_path)

    return ContextTree(
        root=dest_path,
        task_id=task_id,
        rollouts=got_rollouts,
        deliverables=got_deliverables,
        harness_files=n_harness,
        library=False,
        pool=[],
        failed=[],
        deliverables_mode="contents",
        source_files=0,
        decisions=len(decision_items),
    )


def mission(
    tree: ContextTree,
    *,
    job: str = "",
    role: str = "",
    focus: str = "",
    output: str = "",
) -> str:
    """Builds the short pointer prompt directing an agent to `context/`.

    Args:
        tree: The materialized `ContextTree` instance.
        job: Task-specific instructions for the proposer or reflector.
        role: Optional role header when `job` is composed from sections.
        focus: Optional focus instructions.
        output: Optional output format specification.

    Returns:
        Formatted prompt string pointing to the on-disk evidence layout.
    """
    header_blocks = [b.strip() for b in (job, role, focus, output) if b and b.strip()]
    parts = [
        "\n\n".join(header_blocks),
        "",
        "# YOUR EVIDENCE",
        "",
        "It is on disk, in `context/`. Read it -- do not answer from this "
        "message alone.",
        "",
        tree.layout,
        "",
        "Every trajectory EVENT is present: the trajectory was not clipped to "
        "fit this mission. An individual observation may contain an explicit "
        "renderer-truncation marker; that marker is ours, not behavior by the "
        "solver. Do not diagnose it as a solver failure.",
        "",
        "Read `task.md` and the current harness first. Inventory every "
        "available rollout or candidate and inspect each at least once before "
        "deciding. Cite the specific trajectory moment behind every change; "
        "one you cannot trace to a moment is speculation.",
    ]
    return "\n".join(parts)


def parse_reflection_blocks(text: str) -> dict[str, Any]:
    """Extracts causal `<edits>` and `<experience>` blocks from reflector text.

    Supports both Harness² key-value `<experience>` blocks (`cue: ...`,
    `lesson: ...`, `evidence: ...`) and JSON array/object `<experience>` blocks,
    routing lessons to specific pipeline stages via `target_stage`.
    """
    if not text or not isinstance(text, str):
        return {"edits": [], "experiences": []}

    edits_match = _EDITS_RE.search(text)
    edits_raw = edits_match.group(1).strip() if edits_match else ""
    attributions: list[dict[str, str]] = []
    if edits_raw:
        for line in edits_raw.splitlines():
            bullet_match = _EDIT_BULLET_RE.match(line)
            if bullet_match:
                target = (bullet_match.group(1) or "").strip().strip("`")
                verdict = bullet_match.group(2).strip().lower()
                reason = (bullet_match.group(3) or "").strip()
                if target and verdict in VALID_EDIT_VERDICTS:
                    attributions.append({
                        "edit": target,
                        "verdict": verdict,
                        "reason": reason,
                        "evidence": reason,
                    })

    experiences: list[dict[str, str]] = []
    allowed_keys = (
        "cue",
        "lesson",
        "evidence",
        "confidence",
        "target_stage",
    )
    for exp_match in _EXP_RE.finditer(text):
        body = exp_match.group(1).strip()
        if not body:
            continue
        parsed_json: Any = None
        if body.startswith("[") or body.startswith("{"):
            try:
                parsed_json = json.loads(body)
            except ValueError:
                parsed_json = None
        if isinstance(parsed_json, dict):
            parsed_json = [parsed_json]
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if not isinstance(item, dict):
                    continue
                entry = {
                    k: str(item.get(k) or "").strip()
                    for k in allowed_keys
                    if item.get(k) is not None
                }
                if all(entry.get(req) for req in ("cue", "lesson", "evidence")):
                    entry.setdefault("confidence", "medium")
                    entry.setdefault("target_stage", "global")
                    experiences.append(entry)
            continue

        entry = {}
        for line in body.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                clean_key = key.strip().lower().replace("-", "_")
                if clean_key in allowed_keys:
                    entry[clean_key] = val.strip()
        if all(entry.get(req) for req in ("cue", "lesson", "evidence")):
            entry.setdefault("confidence", "medium")
            entry.setdefault("target_stage", "global")
            experiences.append(entry)

    out: dict[str, Any] = {
        "edits": attributions,
        "edit_attributions": attributions,
        "edits_raw": edits_raw,
        "experiences": experiences,
    }
    if experiences:
        first = experiences[0]
        for key in allowed_keys:
            out[key] = first.get(key, "")

    assert_tree_has_no_grader_verdicts({"EXPERIENCE.md": json.dumps(out)})
    return out
