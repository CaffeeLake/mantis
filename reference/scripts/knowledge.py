#!/usr/bin/env python3
"""Export the knowledge base to `.mantis/knowledge/`, or read one back.

    scripts/knowledge.py export --db knowledge.db --repo /path/to/repo
    scripts/knowledge.py import --db knowledge.db --repo /path/to/repo

Export is a deliberate step after a scan, not something the scan does: the
pipeline never writes into the target it audits (INV-4), and the operator
decides what gets committed. The default location is `<repo>/.mantis/knowledge`.

Import here seeds a database with findings, learnings, and coverage. The
threat model and architecture are recorded under the run id given with
--run-id (default "imported"); a run that should see them through its own
tools imports at start with `main.py --import-knowledge` instead.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.knowledge import (  # noqa: E402
    KNOWLEDGE_DIR,
    KnowledgeImportError,
    describe,
    export_knowledge,
    import_knowledge,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write the knowledge base as text")
    ex.add_argument("--db", default="knowledge.db", help="knowledge database (default: knowledge.db)")
    ex.add_argument("--repo", required=True, help="repository root the scan targeted; paths are made relative to it")
    ex.add_argument("--out", default="", help=f"output directory (default: <repo>/{KNOWLEDGE_DIR})")
    im = sub.add_parser("import", help="read a knowledge export into a database")
    im.add_argument("--db", default="knowledge.db")
    im.add_argument("--repo", required=True, help="repository root the paths are relative to")
    im.add_argument("--from", dest="src", default="", help=f"export directory (default: <repo>/{KNOWLEDGE_DIR})")
    im.add_argument("--run-id", default="imported", help="run id the threat model and architecture are recorded under")
    args = p.parse_args(argv)

    repo = os.path.realpath(args.repo)
    if not os.path.isdir(repo):
        print(f"error: --repo {args.repo} is not a directory", file=sys.stderr)
        return 2
    if args.cmd == "export":
        out = args.out or os.path.join(repo, KNOWLEDGE_DIR)
        if not os.path.exists(args.db):
            print(f"error: no database at {args.db}", file=sys.stderr)
            return 2
        counts = export_knowledge(args.db, out, repo_root=repo)
        print(f"Exported to {out}: {describe(counts)}.")
        return 0
    src = args.src or os.path.join(repo, KNOWLEDGE_DIR)
    try:
        counts = import_knowledge(args.db, src, run_id=args.run_id, repo_root=repo)
    except KnowledgeImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Imported from {src}: {describe(counts)} ({counts.get('findings_skipped', 0)} finding(s) already known).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
