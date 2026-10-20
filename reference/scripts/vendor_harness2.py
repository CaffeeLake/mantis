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

"""(Re-)vendor the Harness2 core into reference/third_party/harness2.

Usage:
  python3 reference/scripts/vendor_harness2.py --src <upstream checkout> --sha <sha>
      [--dest reference/third_party/harness2] [--no-verify-sha] [--date YYYY-MM-DD]

Copies exactly the allowlist below from the upstream checkout (nothing else),
re-applies every ``patches/*.diff`` with ``patch -p1 -F0`` (a hunk that no
longer applies aborts the run and leaves the destination untouched), and
rewrites the ``Commit``, ``Vendored`` and files fields of VENDOR.md. Running
it twice at the same commit is a no-op, so a test can pin the committed tree to
upstream plus the patches.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_DEST = REPO / "reference" / "third_party" / "harness2"

# The allowlist. Files are copied byte-for-byte; directories are copied one
# level deep (``*.py`` only). Everything else upstream ships -- benchmarks,
# substrates, cli, install/preflight/demo, docs -- stays out; VENDOR.md says why.
INCLUDE_FILES = ("harness2/__init__.py", "harness2/config.py")
INCLUDE_DIRS = ("harness2/core",)

FILES_BEGIN = "<!-- vendored-files:begin -->"
FILES_END = "<!-- vendored-files:end -->"
_COMMIT_RE = re.compile(r"^- \*\*Commit:\*\* `([^`]*)`$", re.M)
_DATE_RE = re.compile(r"^- \*\*Vendored:\*\* (\S+)$", re.M)
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class VendorError(RuntimeError):
    """The run cannot proceed; the message says what to fix."""


def allowlist(src: Path) -> list[str]:
    """The exact relative paths to vendor from ``src``, sorted."""
    rels: list[str] = []
    for rel in INCLUDE_FILES:
        if not (src / rel).is_file():
            raise VendorError(f"{src / rel} is missing; is --src an upstream harness2 checkout?")
        rels.append(rel)
    for rel in INCLUDE_DIRS:
        d = src / rel
        if not d.is_dir():
            raise VendorError(f"{d} is missing; is --src an upstream harness2 checkout?")
        rels.extend(f"{rel}/{p.name}" for p in d.iterdir() if p.is_file() and p.suffix == ".py")
    return sorted(rels)


def verify_sha(src: Path, sha: str) -> str:
    """HEAD of the checkout rooted at ``src``, or VendorError when it is not ``sha``."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(src), *args], capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
        head = git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as e:
        raise VendorError(
            f"cannot read HEAD of {src} ({e}); pass --no-verify-sha if it is not a git checkout"
        ) from e
    if top != src.resolve():
        # A copy dropped inside some other repository would otherwise borrow that repo's HEAD.
        raise VendorError(
            f"{src} is not the root of a git checkout (toplevel is {top}); "
            "pass --no-verify-sha if it is a plain copy"
        )
    if not head.startswith(sha):
        raise VendorError(f"HEAD of {src} is {head}, which does not start with --sha {sha}")
    return head


def apply_patches(staging: Path, patches_dir: Path) -> list[Path]:
    """Apply every ``patches/*.diff`` to ``staging`` in name order, exact context only."""
    patches = sorted(patches_dir.glob("*.diff")) if patches_dir.is_dir() else []
    if not patches:
        return []
    patch_bin = shutil.which("patch")
    if not patch_bin:
        raise VendorError("`patch` is not on PATH; it is needed to re-apply the local patches")
    for p in patches:
        base = [patch_bin, "-p1", "-F0", "--no-backup-if-mismatch", "-d", str(staging),
                "-i", str(p.resolve())]
        for extra in (["--dry-run"], []):
            r = subprocess.run(base + extra, capture_output=True, text=True)
            if r.returncode != 0:
                raise VendorError(
                    f"{p.name} does not apply to the new upstream (patch exit {r.returncode}):\n"
                    f"{r.stdout}{r.stderr}\n"
                    "Fix the vendored file by hand, regenerate the patch, and rerun."
                )
    return patches


def updated_manifest(text: str, *, sha: str, date: str, files: list[str]) -> str:
    """VENDOR.md with its Commit, Vendored and files fields rewritten."""
    for pattern, line, what in (
        (_COMMIT_RE, f"- **Commit:** `{sha}`", "Commit"),
        (_DATE_RE, f"- **Vendored:** {date}", "Vendored"),
    ):
        text, n = pattern.subn(lambda _m, line=line: line, text)
        if n != 1:
            raise VendorError(f"VENDOR.md must contain exactly one `{what}` field; found {n}")
    block = FILES_BEGIN + "\n" + "".join(f"- `{f}`\n" for f in files) + FILES_END
    text, n = re.subn(
        re.escape(FILES_BEGIN) + r".*?" + re.escape(FILES_END),
        lambda _m: block, text, count=1, flags=re.S,
    )
    if n != 1:
        raise VendorError(f"VENDOR.md must contain the {FILES_BEGIN} ... {FILES_END} block")
    return text


def manifest_sha(text: str) -> str | None:
    m = _COMMIT_RE.search(text)
    return m.group(1) if m else None


def manifest_date(text: str) -> str | None:
    m = _DATE_RE.search(text)
    return m.group(1) if m else None


def vendor(src: Path, dest: Path, sha: str, *, verify: bool = True,
           date: str | None = None, out=sys.stdout) -> dict:
    """Do the work; returns a summary dict (also printed to ``out``)."""
    src = src.resolve()
    dest = dest.resolve()
    if not _SHA_RE.match(sha):
        raise VendorError(f"--sha must be 7-40 lowercase hex characters, got {sha!r}")
    manifest_path = dest / "VENDOR.md"
    if not manifest_path.is_file():
        raise VendorError(f"{manifest_path} is missing; it is hand-maintained and must exist")
    manifest = manifest_path.read_text(encoding="utf-8")
    head = verify_sha(src, sha) if verify else None
    files = allowlist(src)

    if date is None:
        # The date records when this commit was vendored: keep it when the
        # commit has not moved, so a rerun at the same sha changes nothing.
        kept = manifest_date(manifest) if manifest_sha(manifest) == sha else None
        date = kept or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    # Validate everything that can fail before anything in dest is touched.
    new_manifest = updated_manifest(manifest, sha=sha, date=date, files=files)
    if (src / "LICENSE").is_file():
        license_src, license_note = src / "LICENSE", "copied from upstream"
    elif (dest / "LICENSE").is_file():
        license_src, license_note = None, "upstream checkout has no LICENSE file; kept the existing one"
    else:
        raise VendorError(f"neither {src} nor {dest} has a LICENSE file; Apache-2.0 text is required")

    with tempfile.TemporaryDirectory(prefix="vendor_harness2_") as tmp:
        staging = Path(tmp)
        for rel in files:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, target)
        patches = apply_patches(staging, dest / "patches")
        # Everything applied: swap the package tree in and write the manifest.
        pkg = dest / "harness2"
        if pkg.exists():
            shutil.rmtree(pkg)
        shutil.copytree(staging / "harness2", pkg)
    if license_src is not None:
        shutil.copyfile(license_src, dest / "LICENSE")
    manifest_path.write_text(new_manifest, encoding="utf-8")
    summary = {
        "src": str(src), "dest": str(dest), "sha": sha, "head": head, "date": date,
        "files": files, "patches": [p.name for p in patches], "license": license_note,
    }
    print(f"vendored harness2 {sha}" + (f" (HEAD {head})" if head else "") + f" -> {dest}", file=out)
    print(f"  {len(files)} files:", file=out)
    for f in files:
        print(f"    {f}", file=out)
    print(f"  patches applied: {', '.join(summary['patches']) or 'none'}", file=out)
    print(f"  LICENSE: {license_note}", file=out)
    print(f"  VENDOR.md: Commit={sha} Vendored={date}", file=out)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, type=Path, help="upstream harness2 checkout")
    parser.add_argument("--sha", required=True, help="upstream commit being vendored (>= 7 hex)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"vendored tree (default: {DEFAULT_DEST.relative_to(REPO)})")
    parser.add_argument("--no-verify-sha", action="store_true",
                        help="do not require `git -C SRC rev-parse HEAD` to start with --sha")
    parser.add_argument("--date", help="YYYY-MM-DD to record as the vendoring date "
                                       "(default: keep when --sha is unchanged, else today)")
    args = parser.parse_args(argv)
    try:
        vendor(args.src, args.dest, args.sha, verify=not args.no_verify_sha, date=args.date)
    except VendorError as e:
        print(f"vendor_harness2: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
