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

"""A campaign owns the findings it reports, wherever they sit.

A campaign's scope was its path: what its stages read and what its stamps
reached. A campaign told to follow a changed sink to its callers reports a
bug in another package, and under the path rule its own reviewer, critic
and reproducer never saw it and its stamps never reached it. Findings now
record the campaign that reported them, and a campaign's scope is
ownership OR location: its own reports plus what sits under its path.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.context import RunContext, current_run_context  # noqa: E402
from core.database import (  # noqa: E402
    campaign_key, init_db, read_findings, update_status, write_findings,
)

RUN = "run-own"


def _finding(path, title, status="valid", code_paths=None):
    f = {"title": title, "severity": "HIGH", "status": status, "description": f"d {title}",
         "line_numbers": [3], "filepath": path, "cwe": "CWE-89"}
    if code_paths:
        f["code_paths"] = code_paths
    return f


class OwnershipTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_own_")
        self.repo = os.path.realpath(os.path.join(self.tmp, "repo"))
        for d in ("pkg/a", "pkg/b", "cmd"):
            os.makedirs(os.path.join(self.repo, d))
        self.db = os.path.join(self.tmp, "k.db")
        init_db(self.db)
        self.token = current_run_context.set(RunContext(jail_dir=self.repo, db_path=self.db, run_id=RUN))

    def tearDown(self):
        current_run_context.reset(self.token)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _campaign(self, rel):
        return os.path.join(self.repo, rel)

    def test_the_key_is_target_relative_and_empty_for_the_root(self):
        self.assertEqual(campaign_key(self._campaign("pkg/a")), "pkg/a")
        self.assertEqual(campaign_key(self._campaign("cmd/main.go")), "cmd/main.go")
        self.assertEqual(campaign_key(self.repo), "", "the root campaign reads the whole tree anyway")
        self.assertEqual(campaign_key(""), "")
        self.assertNotIn(self.repo, campaign_key(self._campaign("pkg/a")), "no host path in the table")

    def test_a_campaign_reads_what_it_reported_and_what_sits_in_its_path(self):
        a, b = self._campaign("pkg/a"), self._campaign("pkg/b")
        # Campaign A reports in its own package and, following a caller,
        # in package B. Campaign B reports in its own package.
        write_findings(self.db, a, [_finding("pkg/a/x.go", "in a"), _finding("pkg/b/y.go", "found by a in b")], run_id=RUN)
        write_findings(self.db, b, [_finding("pkg/b/z.go", "in b")], run_id=RUN)
        rows = read_findings(self.db, run_id=RUN)
        owners = {r["title"]: r["campaign"] for r in rows}
        self.assertEqual(owners, {"in a": "pkg/a", "found by a in b": "pkg/a", "in b": "pkg/b"})

        seen_by_a = {r["title"] for r in read_findings(self.db, run_id=RUN, scope_path=a, campaign=a)}
        seen_by_b = {r["title"] for r in read_findings(self.db, run_id=RUN, scope_path=b, campaign=b)}
        self.assertEqual(seen_by_a, {"in a", "found by a in b"}, "its own reports, wherever they sit")
        self.assertEqual(seen_by_b, {"in b", "found by a in b"}, "and what sits in its path, whoever reported it")
        # The path rule alone, as before, for a caller that gives no campaign.
        self.assertEqual({r["title"] for r in read_findings(self.db, run_id=RUN, scope_path=a)}, {"in a"})
        # Ownership alone.
        self.assertEqual({r["title"] for r in read_findings(self.db, run_id=RUN, campaign=a)},
                         {"in a", "found by a in b"})

    def test_a_re_report_keeps_the_first_owner(self):
        a, b = self._campaign("pkg/a"), self._campaign("pkg/b")
        write_findings(self.db, a, [_finding("pkg/b/y.go", "shared")], run_id=RUN)
        write_findings(self.db, b, [_finding("pkg/b/y.go", "shared")], run_id=RUN)
        rows = read_findings(self.db, run_id=RUN)
        self.assertEqual(len(rows), 1, "same signature, one row")
        self.assertEqual(rows[0]["campaign"], "pkg/a")
        self.assertIn("shared", {r["title"] for r in read_findings(self.db, run_id=RUN, scope_path=b, campaign=b)},
                      "B still sees it: it sits in B's path")

    def test_stamps_reach_what_the_campaign_owns(self):
        a = self._campaign("pkg/a")
        write_findings(self.db, a, [_finding("pkg/a/x.go", "in a"), _finding("cmd/m.go", "found by a in cmd")], run_id=RUN)
        write_findings(self.db, self._campaign("pkg/b"), [_finding("pkg/b/z.go", "in b")], run_id=RUN)
        update_status(self.db, a, RUN, "static_confirmed", campaign=a)
        st = {r["title"]: r["status"] for r in read_findings(self.db, run_id=RUN)}
        self.assertEqual(st["in a"], "static_confirmed")
        self.assertEqual(st["found by a in cmd"], "static_confirmed", "owned, outside the path: stamped")
        self.assertEqual(st["in b"], "valid", "another campaign's finding, outside the path: untouched")
        # Without the campaign the path rule alone applies, as before.
        update_status(self.db, a, RUN, "dynamic_confirmed")
        st = {r["title"]: r["status"] for r in read_findings(self.db, run_id=RUN)}
        self.assertEqual(st["in a"], "dynamic_confirmed")
        self.assertEqual(st["found by a in cmd"], "static_confirmed")

    def test_rows_without_an_owner_fall_back_to_location(self):
        import sqlite3

        a = self._campaign("pkg/a")
        write_findings(self.db, a, [_finding("pkg/a/x.go", "old row")], run_id=RUN)
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE findings SET campaign = NULL")
        conn.commit()
        conn.close()
        self.assertEqual({r["title"] for r in read_findings(self.db, run_id=RUN, scope_path=a, campaign=a)}, {"old row"})
        update_status(self.db, a, RUN, "static_confirmed", campaign=a)
        self.assertEqual(read_findings(self.db, run_id=RUN)[0]["status"], "static_confirmed")

    def test_the_root_campaign_reads_everything_as_before(self):
        write_findings(self.db, self._campaign("pkg/a"), [_finding("pkg/a/x.go", "in a")], run_id=RUN)
        write_findings(self.db, self.repo, [_finding("cmd/m.go", "root's")], run_id=RUN)
        titles = {r["title"] for r in read_findings(self.db, run_id=RUN, scope_path=self.repo, campaign=self.repo)}
        self.assertEqual(titles, {"in a", "root's"})

    def test_get_findings_shows_a_campaign_its_own_out_of_path_report(self):
        import json

        import tools.research_tools as rt

        a = self._campaign("pkg/a")
        current_run_context.reset(self.token)
        self.token = current_run_context.set(RunContext(jail_dir=self.repo, db_path=self.db, run_id=RUN,
                                                        target_file=a, active_node="reviewer"))
        write_findings(self.db, a, [_finding("cmd/m.go", "found by a in cmd")], run_id=RUN)
        write_findings(self.db, self._campaign("pkg/b"), [_finding("pkg/b/z.go", "in b")], run_id=RUN)
        out = rt.get_findings()
        body = out[out.index("["):out.rindex("]") + 1]
        titles = {f["title"] for f in json.loads(body)}
        self.assertEqual(titles, {"found by a in cmd"})


if __name__ == "__main__":
    unittest.main()
