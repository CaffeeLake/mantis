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

"""The one sanctioned route to the vendored Harness2 core.

``load_harness2()`` puts ``reference/third_party/harness2`` (the vendored
tree) on ``sys.path`` once and imports ``harness2`` under two guarantees
Mantis relies on:

- ``harness2.config`` reads ``HARNESS2_*`` variables (and a ``.env`` in the
  working directory) at import time and bakes them into module constants.
  Mantis passes every setting explicitly, so the import is refused when any
  ``HARNESS2_*`` variable is set -- before the import, and again afterwards in
  case a ``.env`` injected one.
- ``harness2.config`` also ``setdefault``s into ``os.environ`` (for example
  ``GOOGLE_CLOUD_LOCATION``). The shim restores ``os.environ`` to its
  pre-import state so upstream defaults never reach the Mantis process.

``VENDORED_SHA`` is pinned to the ``Commit`` field of VENDOR.md by
tests/test_vendored_harness2.py; bump both when re-vendoring.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from types import ModuleType

#: Upstream google-research/harness2 commit the vendored tree was taken from.
VENDORED_SHA = "1cfe254"

THIRD_PARTY = Path(__file__).resolve().parent.parent.parent / "third_party"
#: The vendored tree: LICENSE, VENDOR.md, patches/ and the ``harness2`` package itself.
#: This, not THIRD_PARTY, is the sys.path root -- with THIRD_PARTY on the path the
#: directory would import as an empty namespace package named ``harness2``.
VENDOR_DIR = THIRD_PARTY / "harness2"
PACKAGE_DIR = VENDOR_DIR / "harness2"
MANIFEST = VENDOR_DIR / "VENDOR.md"
ENV_PREFIX = "HARNESS2_"

_COMMIT_RE = re.compile(r"^- \*\*Commit:\*\* `([0-9a-f]{7,40})`$", re.M)


def vendored_sha_from_manifest(path: Path = MANIFEST) -> str:
    """The ``Commit`` field of VENDOR.md, the manifest vendor_harness2.py maintains."""
    text = path.read_text(encoding="utf-8")
    m = _COMMIT_RE.search(text)
    if m is None:
        raise RuntimeError(f"{path} has no '- **Commit:** `<sha>`' line")
    return m.group(1)


def offending_env(environ=None) -> list[str]:
    """The ``HARNESS2_*`` names present in ``environ`` (default ``os.environ``), sorted."""
    env = os.environ if environ is None else environ
    return sorted(k for k in env if k.startswith(ENV_PREFIX))


def _refuse(names: list[str], when: str) -> None:
    raise RuntimeError(
        f"refusing to load the vendored harness2: {when} these {ENV_PREFIX}* variables "
        f"were set: {', '.join(names)}. Mantis passes every Harness2 setting explicitly "
        "and must not inherit upstream environment defaults; unset them (and remove any "
        ".env in the working directory that sets them)."
    )


def load_harness2() -> ModuleType:
    """Import and return the vendored ``harness2`` package (see the module docstring)."""
    bad = offending_env()
    if bad:
        _refuse(bad, "before import")

    path = str(VENDOR_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)

    already = sys.modules.get("harness2")
    if already is not None and not _is_vendored(already):
        raise RuntimeError(
            f"a non-vendored harness2 is already imported from {getattr(already, '__file__', '?')}; "
            f"Mantis uses only {PACKAGE_DIR}"
        )

    before = dict(os.environ)
    try:
        module = importlib.import_module("harness2")
        importlib.import_module("harness2.config")
    finally:
        # Undo whatever config.py setdefault'ed (its .env, GOOGLE_CLOUD_LOCATION, ...).
        leaked = {k: v for k, v in os.environ.items() if before.get(k) != v}
        for k in leaked:
            if k in before:
                os.environ[k] = before[k]
            else:
                del os.environ[k]
        for k in before.keys() - os.environ.keys():
            os.environ[k] = before[k]
    if not _is_vendored(module):
        raise RuntimeError(
            f"harness2 resolved to {getattr(module, '__file__', '?')}, not {PACKAGE_DIR}; "
            "something earlier on sys.path shadows the vendored tree"
        )
    bad = offending_env(leaked)
    if bad:
        # config.py baked the injected values into its constants: drop the modules so
        # a retry after the fix imports cleanly instead of reusing the tainted ones.
        for name in [n for n in sys.modules if n == "harness2" or n.startswith("harness2.")]:
            del sys.modules[name]
        _refuse(bad, "during import (a .env in the working directory?)")
    return module


def _is_vendored(module: ModuleType) -> bool:
    file = getattr(module, "__file__", None)
    if not file:  # a namespace package has no __file__
        return False
    try:
        return Path(file).resolve().is_relative_to(PACKAGE_DIR)
    except (OSError, ValueError):
        return False
