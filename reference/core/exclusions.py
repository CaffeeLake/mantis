"""Operator exclusions: which parts of a scan target are not scanned.

A scan of ``repo/`` that should leave ``repo/vendor`` and every ``testdata``
directory alone is expressed as patterns relative to the target. Matching is
``fnmatch`` on paths, with gitignore's one convention worth keeping: a pattern
with no ``/`` names a basename and matches at any depth (``vendor`` excludes
``vendor/`` wherever it sits), a pattern with a ``/`` is matched against the
target-relative path (``third_party/*``, ``src/gen/*.pb.go``). A file is
excluded when the pattern matches the file itself or any directory above it,
so excluding a directory excludes its contents.

Exclusion governs what is SCANNED: which files become campaigns, which
subsystems the surveyor ranks, what the agent sees when it lists the tree.
It does not make a file unreadable: an excluded file that a scanned one calls
into can still be opened for context, and it stays in the structural index.
That is deliberate -- the operator asked not to spend effort on a directory,
not to pretend it is absent, and a caller in an excluded test directory is
still a caller.
"""
from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Iterable, Sequence, Tuple, Union


def normalize_exclude_patterns(raw: Union[None, str, Iterable[str]]) -> Tuple[str, ...]:
    """Patterns as a tuple of clean strings, from a list or a comma/newline list.

    Leading ``./`` and trailing ``/`` carry no meaning here and are dropped, so
    ``vendor/`` and ``vendor`` are the same pattern. Empty entries are dropped.
    Order is kept and duplicates removed.
    """
    if raw is None:
        return ()
    items: list[str] = []
    if isinstance(raw, str):
        parts = raw.replace("\n", ",").split(",")
    else:
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.extend(item.replace("\n", ",").split(","))
    for p in parts:
        p = p.strip().replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        p = p.strip("/")
        if p and p not in items:
            items.append(p)
    return tuple(items)


def is_excluded(rel_path: str, patterns: Sequence[str]) -> bool:
    """True if `rel_path` (relative to the scan target) matches an exclusion.

    Checked against the path and every ancestor directory, so a pattern that
    names a directory covers everything beneath it.
    """
    if not patterns:
        return False
    rel = str(rel_path or "").replace("\\", "/").strip("/")
    if not rel:
        return False
    parts = rel.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    for pat in patterns:
        if "/" in pat:
            if any(fnmatchcase(prefix, pat) for prefix in prefixes):
                return True
        else:
            if any(fnmatchcase(part, pat) for part in parts):
                return True
    return False


def filter_excluded(rel_paths: Iterable[str], patterns: Sequence[str]) -> list[str]:
    """The paths that are not excluded, in order."""
    if not patterns:
        return list(rel_paths)
    return [p for p in rel_paths if not is_excluded(p, patterns)]
