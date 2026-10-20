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

"""One truncation marker for every tool output the harness cuts.

A tool response that is cut short without saying so is read as the whole
observation: a reviewer who sees a file end before the bounds check concludes
the check is missing, a reproducer who sees a sandbox log end before the
sentinel concludes the PoC never reached it. Both are verdicts carrying INV-1
weight, drawn from an artefact of the harness rather than a property of the
target. Every site that cuts a tool response therefore cuts it through this
module, so the cut is announced in one recognisable form that the
verdict-stage prompts can name and `tools.sandbox_tools` can detect.

Dependency-free on purpose: it is imported by the sandbox environments, the
research tools and the evidence checker, and must never be the reason one
of them fails to import.
"""

from __future__ import annotations

from typing import Tuple

TRUNCATION_MARKER_PREFIX = "[MANTIS TRUNCATION:"

# The remediation every text cut points at. The three options cover the
# three kinds of output that get cut: a file (re-read a line range), a
# function (ask for its exact extent), a command trace (run something
# narrower so the part that matters lands inside the cap).
DEFAULT_ADVICE = (
    "re-read with start_line/end_line, get_function_boundary, or a narrower "
    "command before drawing any conclusion from the boundary."
)


def truncation_marker(
    what: str,
    limit: int,
    total: int,
    unit: str = "chars",
    advice: str = DEFAULT_ADVICE,
) -> str:
    """The marker text for a cut of `what` from `total` down to `limit` units.

    Kept separate from `truncate_with_marker` so a site whose cut is not a
    character slice (list_files caps entries) can still announce it in the
    same shape, and so a site can place the marker outside an untrusted-content
    fence while cutting the text inside it.
    """
    try:
        limit_i = max(int(limit), 0)
    except (TypeError, ValueError):
        limit_i = 0
    try:
        total_i = max(int(total), 0)
    except (TypeError, ValueError):
        total_i = limit_i
    omitted = max(total_i - limit_i, 0)
    return (
        f"{TRUNCATION_MARKER_PREFIX} {what} cut by the harness after {limit_i} of "
        f"{total_i} {unit} ({omitted} omitted). This cut is the harness's, not "
        f"the target's: {advice}]"
    )


def split_truncation(text: str, limit: int, what: str = "output") -> Tuple[str, str]:
    """Returns (kept_text, marker) where marker is "" when nothing was cut.

    The marker, when present, starts with a newline so it can be appended
    directly to the kept text or to whatever wrapper the caller puts around
    it. Never raises: a None or non-str `text` is coerced, a bad limit is
    treated as zero.
    """
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    try:
        limit_i = max(int(limit), 0)
    except (TypeError, ValueError):
        limit_i = 0
    total = len(text)
    if total <= limit_i:
        return text, ""
    return text[:limit_i], "\n" + truncation_marker(what, limit_i, total)


def truncate_with_marker(text: str, limit: int, what: str = "output") -> str:
    """`text` unchanged when within `limit`, else its first `limit` chars plus the marker."""
    kept, marker = split_truncation(text, limit, what)
    return kept + marker


def is_truncation_marker_present(text: str) -> bool:
    """Whether `text` carries a harness truncation marker anywhere."""
    if not text:
        return False
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return False
    return TRUNCATION_MARKER_PREFIX in text


def bound_fenced_text(text: str, limit: int, what: str = "tool output") -> str:
    """`text` unchanged when within `limit`, else cut to `limit` chars with the
    fence kept closed and the marker placed after it.

    A tool response that carries untrusted content is wrapped
    ``<<<UNTRUSTED_SOURCE_CODE_DATA_START ...>>> body <<<..._END>>>``, and
    whatever the harness appends after the fence (steering hints, earlier cut
    notes) is harness text. A plain character slice would cut the END fence
    off and leave the marker inside the untrusted region, where a hostile
    file could have forged it. So: the body is cut, the fence is closed again,
    the marker follows the fence, and any harness text that already followed
    the fence is kept. Text with no fence is cut like any other output.
    """
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    try:
        limit_i = max(int(limit), 0)
    except (TypeError, ValueError):
        limit_i = 0
    total = len(text)
    if total <= limit_i:
        return text

    try:
        from core.llm_gateway import UNTRUSTED_DATA_END
    except Exception:  # pragma: no cover - gateway unavailable in minimal environments
        UNTRUSTED_DATA_END = "<<<UNTRUSTED_SOURCE_CODE_DATA_END>>>"

    end_at = text.rfind(UNTRUSTED_DATA_END)
    if end_at < 0:
        return text[:limit_i] + "\n" + truncation_marker(what, limit_i, total)

    fenced = text[:end_at]
    trailer = text[end_at + len(UNTRUSTED_DATA_END):]
    # The budget pays for the fenced part; the fence and the trailer are the
    # harness's own and short.
    body_limit = max(limit_i - len(UNTRUSTED_DATA_END) - len(trailer), 0)
    if len(fenced) <= body_limit:
        return text
    kept = fenced[:body_limit].rstrip("\n")
    return (
        f"{kept}\n{UNTRUSTED_DATA_END}\n"
        f"{truncation_marker(what, len(kept), len(fenced))}"
        f"{trailer}"
    )
