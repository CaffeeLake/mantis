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

"""Writes skills/README.md: one row per skill, from each SKILL.md's frontmatter.

Usage: gen_skills_index.py <repo-root>
test_skills_integrity.py checks the index matches the directories, so re-run
this after adding or renaming a skill.
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
skills_dir = root / "skills"

rows = []
for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
    text = skill_md.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        sys.exit(f"{skill_md}: no frontmatter")
    fm = m.group(1)
    name = re.search(r"^name:\s*(.+)$", fm, re.M).group(1).strip()
    # description is either a one-liner or a folded block scalar (>-).
    dm = re.search(r"^description:\s*(?:>-?\s*\n((?:[ \t]+.*\n?)+)|(.+))$", fm, re.M)
    if dm.group(1) is not None:
        desc = " ".join(line.strip() for line in dm.group(1).splitlines() if line.strip())
    else:
        desc = dm.group(2).strip()
    # First sentence only: the "Use when / Don't use" routing text belongs in the skill.
    first = re.split(r"(?<=[.!?])\s+(?=[A-Z])", desc, maxsplit=1)[0].rstrip(".")
    rows.append((name, skill_md.parent.name, first))

if not rows:
    sys.exit("no skills found")

width = max(len(f"[`{n}`]({d}/SKILL.md)") for n, d, _ in rows)
out = ["# Skills", "",
       "Each directory is one skill: a `SKILL.md` an agent loads on demand, with any",
       "`references/` it needs.", "",
       "| Skill | What it does |", "| --- | --- |"]
for name, d, desc in rows:
    out.append(f"| [`{name}`]({d}/SKILL.md) | {desc}. |")
out.append("")
(skills_dir / "README.md").write_text("\n".join(out), encoding="utf-8")
print(f"wrote {skills_dir / 'README.md'} with {len(rows)} skills")
