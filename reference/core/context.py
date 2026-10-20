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
    # Per-file un-windowed read_file call counters for tool-loop steering.
    read_file_counts: dict[str, int] = field(default_factory=dict)

current_run_context: contextvars.ContextVar[Optional[RunContext]] = contextvars.ContextVar(
    "current_run_context", default=None
)

