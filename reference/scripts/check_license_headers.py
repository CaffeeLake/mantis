#!/usr/bin/env python3
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

"""Checks that every code file carries the Apache-2.0 license header.

Shared by tests/test_license_headers.py and the pre-commit hook so there is
one definition of "code file" and one definition of "the header".

Usage: python3 reference/scripts/check_license_headers.py [--list]
Exit 0 when every file complies; 1 otherwise, printing the offenders.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# The header every source file must carry, modulo the copyright year(s).
HEADER_LINES = (
    "Licensed under the Apache License, Version 2.0 (the \"License\");",
    "you may not use this file except in compliance with the License.",
    "You may obtain a copy of the License at",
    "https://www.apache.org/licenses/LICENSE-2.0",
    "Unless required by applicable law or agreed to in writing, software",
    "distributed under the License is distributed on an \"AS IS\" BASIS,",
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "See the License for the specific language governing permissions and",
    "limitations under the License.",
)
COPYRIGHT_RE = re.compile(r"Copyright (\d{4})(-\d{4})? Google LLC")

# Which tracked files are "code". Only files a person wrote and another
# person might copy out on their own: Python and shell. Not data (JSON),
# prose (Markdown), configuration (YAML, TOML, requirements), generated
# lockfiles, vendored third-party trees, or the sample targets the pipeline
# audits -- those are read into model prompts verbatim, and a copyright
# banner there costs tokens on every read and misstates their provenance.
SOURCE_SUFFIXES = {".py", ".sh"}
SOURCE_BASENAMES: set[str] = set()
EXCLUDE_PREFIXES = (
    "reference/third_party/",
    "reference/evals/research_target/",
    "reference/evals/research_targets/",
    "reference/test_targets/",
    "reference/test_scan/",
)
# How many leading lines may precede the header: a shebang and an encoding
# cookie at most.
HEADER_WINDOW = 20


def tracked_source_files(repo: Path = REPO) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True
    ).stdout
    files = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="surrogateescape")
        if rel.startswith(EXCLUDE_PREFIXES):
            continue
        p = Path(rel)
        if p.suffix in SOURCE_SUFFIXES or p.name in SOURCE_BASENAMES:
            files.append(repo / p)
    return files


def has_header(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:HEADER_WINDOW]
    except OSError:
        return False
    text = "\n".join(line.lstrip("#").strip() for line in head)
    if not COPYRIGHT_RE.search(text):
        return False
    return all(line in text for line in HEADER_LINES)


def missing_headers(repo: Path = REPO) -> list[Path]:
    return [p for p in tracked_source_files(repo) if not has_header(p)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list every checked file")
    args = parser.parse_args(argv)
    files = tracked_source_files()
    if args.list:
        for p in files:
            print(p.relative_to(REPO))
    missing = [p for p in files if not has_header(p)]
    if missing:
        print(f"{len(missing)} code file(s) lack the Apache-2.0 header:", file=sys.stderr)
        for p in missing:
            print(f"  {p.relative_to(REPO)}", file=sys.stderr)
        return 1
    print(f"license headers OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
