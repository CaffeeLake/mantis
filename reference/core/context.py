import contextvars
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RunContext:
    jail_dir: str
    db_path: str
    target_file: str = ""
    sandbox: object = None
    run_id: str = ""
    sandbox_executed: bool = False
    snapshot_id: str = ""
    budget_controller: object = None
    static_sandbox_attempts: int = 0
    active_node: str = ""
    # The scan mode this run was launched with ("whole", "file-by-file",
    # "cross-functional"). Empty when the caller predates the field; every
    # consumer must treat empty as "unknown" and fall back to per-campaign
    # behavior (INV-6).
    scan_mode: str = ""
    # The directory finding filepaths should be stored relative to -- the
    # enclosing repository when a scan targets a single file inside one.
    # A file-sized target cannot know its own repository; only the caller
    # does. Empty means "unknown": canonicalization falls back to the
    # jail-anchored behavior that predates the field (INV-6).
    path_root: str = ""
    # Operator exclusions (core.exclusions), relative to the target: what the
    # run does not scan and does not list. Empty means none (INV-6).
    exclude: tuple = ()
    # The model this run uses (resolved id). Empty when unknown.
    model: str = ""
    # Content-addressed prefix fingerprints by node_id (Harness² runmark).
    prefix_fingerprints: dict[str, str] = field(default_factory=dict)
    # Per-file un-windowed read_file call counters for tool-loop steering.
    read_file_counts: dict[str, int] = field(default_factory=dict)

current_run_context: contextvars.ContextVar[Optional[RunContext]] = contextvars.ContextVar(
    "current_run_context", default=None
)

