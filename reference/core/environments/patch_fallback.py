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

"""A unified-diff applier for guest images that ship no patch(1).

``python:3-slim`` and most distroless-style images have python3 and nothing
else; the patcher's ``apply_patch`` then came back ``SANDBOX-ERROR: patch(1)
unavailable``, the campaign counted as failed, and in one live run the model
spent a stage writing its own patch(1) into ``/usr/local/bin`` to get past
it. The environments now write ``PYTHON_PATCH_SCRIPT`` into the guest
workdir and run it with the guest's python3 when ``patch`` is missing.

The script is deliberately plain Python 3 (no f-strings newer than 3.6, no
stdlib beyond ``sys``/``os``/``re``) so it runs on whatever python3 the image
has. It understands what ``patch -p1`` understands from a ``git diff``/
``diff -u``: ``---``/``+++`` headers with ``a/``/``b/`` prefixes, new and
deleted files via ``/dev/null``, ``@@`` hunks with context, ``\\ No newline
at end of file``, and a hunk that has drifted a few lines from where the
diff says it is. A hunk whose context does not match anywhere nearby fails
the whole run with a non-zero exit and names the file and line, which is
what ``patch`` would do; it never writes a half-applied file.
"""

from __future__ import annotations

PATCH_FILE_NAME = ".mantis.patch"
PATCH_SCRIPT_NAME = ".mantis_patch.py"

# Stdout/stderr lines this script prints are read by the patcher stage as
# the tool's output; keep them in patch(1)'s shape ("patching file X").
PYTHON_PATCH_SCRIPT = r'''
import os
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
LINE_RE = re.compile(r"[^\n]*\n|[^\n]+")
MAX_DRIFT = 200


def split_lf(text):
    return LINE_RE.findall(text) if text else []


def strip_path(path, strip):
    path = path.strip().split("\t")[0]
    if path == "/dev/null":
        return None
    parts = path.split("/")
    if strip and len(parts) > strip:
        parts = parts[strip:]
    rel = "/".join(p for p in parts if p and p != ".")
    if not rel or path.startswith("/") or ".." in parts:
        raise SystemExit("refusing unsafe patch path: %s" % path)
    return rel


def parse(diff_text):
    """Yields (old_path, new_path, hunks); a hunk is a list of (tag, line)."""
    lines = split_lf(diff_text)
    i, n = 0, len(lines)
    files = []
    while i < n:
        line = lines[i]
        if not line.startswith("--- "):
            i += 1
            continue
        old_raw = line[4:].rstrip("\r\n")
        i += 1
        if i >= n or not lines[i].startswith("+++ "):
            raise SystemExit("malformed diff: '---' without '+++' near line %d" % i)
        new_raw = lines[i][4:].rstrip("\r\n")
        i += 1
        hunks = []
        while i < n and lines[i].startswith("@@"):
            m = HUNK_RE.match(lines[i])
            if not m:
                raise SystemExit("malformed hunk header: %r" % lines[i].rstrip())
            old_start = int(m.group(1))
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            i += 1
            body = []
            seen_old = seen_new = 0
            while i < n and (seen_old < old_len or seen_new < new_len):
                hl = lines[i]
                if hl.startswith("\\"):
                    # "\ No newline at end of file": strip the newline we added
                    # to the previous body line.
                    if body:
                        tag, text = body[-1]
                        body[-1] = (tag, text.rstrip("\r\n"))
                    i += 1
                    continue
                tag = hl[0] if hl[0] in " +-" else " "
                text = hl[1:] if hl[0] in " +-" else hl
                body.append((tag, text))
                if tag in " -":
                    seen_old += 1
                if tag in " +":
                    seen_new += 1
                i += 1
            while i < n and lines[i].startswith("\\"):
                if body:
                    tag, text = body[-1]
                    body[-1] = (tag, text.rstrip("\r\n"))
                i += 1
            hunks.append((old_start, body))
        files.append((old_raw, new_raw, hunks))
    return files


def apply_hunks(file_lines, hunks, path):
    offset = 0
    min_pos = 0
    out = list(file_lines)
    for old_start, body in hunks:
        old = [t for tag, t in body if tag in " -"]
        new = [t for tag, t in body if tag in " +"]
        found = None
        if not old:
            want = 0 if old_start == 0 else old_start + offset
            found = min(max(want, min_pos), len(out))
        else:
            want = old_start - 1 + offset
            if old_start == 0:
                want = 0
            for drift in range(0, MAX_DRIFT + 1):
                for cand in (want - drift, want + drift) if drift else (want,):
                    if max(0, min_pos) <= cand <= len(out) - len(old) and out[cand:cand + len(old)] == old:
                        found = cand
                        break
                if found is not None:
                    break
        if found is None:
            raise SystemExit("Hunk FAILED for %s at line %d (context not found within %d lines)."
                             % (path, old_start, MAX_DRIFT))
        out[found:found + len(old)] = new
        if old:
            offset = (found - (old_start - 1)) + len(new) - len(old)
        else:
            offset += len(new)
        min_pos = found + len(new)
    return out


def main(argv):
    strip = 1
    patch_path = None
    k = 0
    while k < len(argv):
        a = argv[k]
        if a.startswith("-p") and len(a) > 2:
            strip = int(a[2:])
        elif a == "-p":
            k += 1
            strip = int(argv[k])
        elif a in ("-i", "--input"):
            k += 1
            patch_path = argv[k]
        elif not a.startswith("-"):
            patch_path = a
        k += 1
    if patch_path:
        with open(patch_path, "r", encoding="utf-8", errors="surrogateescape") as fh:
            diff_text = fh.read()
    else:
        diff_text = sys.stdin.read()

    files = parse(diff_text)
    if not files:
        sys.stderr.write("no file headers found in patch input\n")
        return 2

    # Two passes: every hunk must apply before any file is written.
    planned = []
    for old_raw, new_raw, hunks in files:
        old_path = strip_path(old_raw, strip)
        new_path = strip_path(new_raw, strip)
        target = new_path or old_path
        if old_path is None:
            current = []
        else:
            if not os.path.exists(old_path):
                sys.stderr.write("can't find file to patch: %s\n" % old_path)
                return 1
            with open(old_path, "r", encoding="utf-8", errors="surrogateescape", newline="") as fh:
                current = split_lf(fh.read())
        if new_path is None:
            rem = apply_hunks(current, hunks, target)
            if rem:
                raise SystemExit("Not deleting file %s as content differs from patch." % old_path)
            planned.append(("delete", old_path, None))
            continue
        updated = apply_hunks(current, hunks, target)
        planned.append(("write", target, updated))

    for action, path, content in planned:
        if action == "delete":
            os.remove(path)
            sys.stdout.write("removing file %s\n" % path)
            continue
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as fh:
            fh.write("".join(content))
        sys.stdout.write("patching file %s\n" % path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            sys.stderr.write(exc.code + "\n")
            sys.exit(1)
        raise
'''


def fallback_command(strip: int = 1, python: str = "python3") -> str:
    """The guest shell command that applies PATCH_FILE_NAME with the Python applier."""
    return f"{python} {PATCH_SCRIPT_NAME} -p{int(strip)} {PATCH_FILE_NAME}"


def fallback_argv(strip: int = 1, python: str = "python3") -> list[str]:
    return [python, PATCH_SCRIPT_NAME, f"-p{int(strip)}", PATCH_FILE_NAME]


def unavailable_message(image: str, detail: str = "") -> str:
    return (
        f"SANDBOX-ERROR: patch(1) unavailable in image '{image}' and the python3 fallback "
        f"could not run; install patch or python3 in the sandbox image.\n{detail}"
    )
