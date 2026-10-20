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

"""Every tracked source file carries the Apache-2.0 header; LICENSE is intact.

Apache-2.0 section 4 requires that redistributed files carry the license
notice. The repository's LICENSE file alone does not satisfy that for a
file copied out on its own, which is how reference implementations travel.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = _REPO / "reference" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import check_license_headers as chk  # noqa: E402


class LicenseHeaderTest(unittest.TestCase):
    def test_license_file_is_apache_2(self):
        text = (_REPO / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)

    def test_every_tracked_source_file_has_the_header(self):
        missing = chk.missing_headers(_REPO)
        self.assertEqual(
            [],
            [str(p.relative_to(_REPO)) for p in missing],
            "run reference/scripts/check_license_headers.py for the list; "
            "add the header from any neighbouring file",
        )

    def test_source_set_is_code_only(self):
        rels = {str(p.relative_to(_REPO)) for p in chk.tracked_source_files(_REPO)}
        for must in ("reference/main.py", "reference/run.sh", "reference/core/memory.py"):
            self.assertIn(must, rels)
        # Configuration, data, prose and generated lockfiles are out of scope.
        for must_not in (
            "reference/requirements.txt",
            "reference/requirements.in",
            "reference/sandbox/Dockerfile",
            ".github/workflows/scorecard.yml",
            "action/action.yml",
            ".pre-commit-config.yaml",
        ):
            self.assertNotIn(must_not, rels)
        self.assertFalse(any(r.endswith((".json", ".md", ".yml", ".yaml", ".toml")) for r in rels))

    def test_sample_targets_carry_no_banner(self):
        """The audit targets under evals/ and test_*/ are read into model prompts
        verbatim; a copyright banner there costs tokens on every read_file and
        misstates the provenance of deliberately vulnerable fixtures."""
        rels = {str(p.relative_to(_REPO)) for p in chk.tracked_source_files(_REPO)}
        self.assertFalse(
            any(r.startswith(("reference/evals/research_target/", "reference/test_targets/", "reference/test_scan/")) for r in rels)
        )
        for fixture in ("reference/evals/research_target/app.py", "reference/test_scan/vulnerable.py"):
            path = _REPO / fixture
            if path.exists():
                self.assertNotIn("Licensed under the Apache", path.read_text(encoding="utf-8"), fixture)

    def test_checker_recognises_header_after_shebang_and_rejects_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.sh"
            good.write_text(
                "#!/bin/bash\n# Copyright 2026 Google LLC\n#\n"
                + "".join(f"# {line}\n" for line in chk.HEADER_LINES)
                + "\necho hi\n",
                encoding="utf-8",
            )
            self.assertTrue(chk.has_header(good))
            partial = Path(tmp) / "partial.py"
            partial.write_text("# Copyright 2026 Google LLC\nprint(1)\n", encoding="utf-8")
            self.assertFalse(chk.has_header(partial))
            wrong_holder = Path(tmp) / "holder.py"
            wrong_holder.write_text(
                "# Copyright 2026 Someone Else\n"
                + "".join(f"# {line}\n" for line in chk.HEADER_LINES),
                encoding="utf-8",
            )
            self.assertFalse(chk.has_header(wrong_holder))


if __name__ == "__main__":
    unittest.main()
