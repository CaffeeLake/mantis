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

"""The session store lives beside the knowledge database and says how big
it is.

It used to default to the installation directory whatever --db said, grew
with every run, and nothing reported it. Now an unconfigured store sits
next to knowledge.db (a resume of an older run still finds the
installation's file), and every run ends by stating the store's size,
session and run counts, so the operator can decide about pruning.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core import paths as P  # noqa: E402


def _make_store(path: str, run_ids: list, sessions_per_run: int = 2) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, app_name TEXT)")
    for rid in run_ids:
        for i in range(sessions_per_run):
            conn.execute("INSERT INTO sessions VALUES (?, ?)", (f"session_run_{rid}_{i:08x}", "mantis"))
    conn.commit()
    conn.close()


class LocationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_sess_")
        self.install = Path(self.tmp, "install")
        self.install.mkdir()
        self.data = Path(self.tmp, "data")
        self.data.mkdir()
        self.patch = mock.patch.object(P, "install_root", return_value=self.install)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unconfigured_store_sits_beside_the_knowledge_db(self):
        got = P.resolve_sessions_db_path("", str(self.data / "knowledge.db"))
        self.assertEqual(got, str(self.data / "sessions.db"))

    def test_the_default_knowledge_db_keeps_the_store_under_the_installation(self):
        got = P.resolve_sessions_db_path("", "")
        self.assertEqual(got, str(self.install / "sessions.db"))
        got = P.resolve_sessions_db_path("", "knowledge.db")
        self.assertEqual(got, str(self.install / "sessions.db"), "a relative name is the installation's, never $CWD's")

    def test_a_configured_path_wins(self):
        got = P.resolve_sessions_db_path(str(self.data / "elsewhere.db"), str(self.data / "knowledge.db"))
        self.assertEqual(got, str(self.data / "elsewhere.db"))

    def test_a_resume_of_an_older_run_finds_the_installations_store(self):
        (self.install / "sessions.db").write_bytes(b"")
        got = P.resolve_sessions_db_path("", str(self.data / "knowledge.db"), resuming=True)
        self.assertEqual(got, str(self.install / "sessions.db"))
        # Once a store exists beside the knowledge db, that is the one.
        (self.data / "sessions.db").write_bytes(b"")
        got = P.resolve_sessions_db_path("", str(self.data / "knowledge.db"), resuming=True)
        self.assertEqual(got, str(self.data / "sessions.db"))
        # And a fresh run never goes back to the installation's file.
        (self.data / "sessions.db").unlink()
        got = P.resolve_sessions_db_path("", str(self.data / "knowledge.db"), resuming=False)
        self.assertEqual(got, str(self.data / "sessions.db"))

    def test_a_symlinked_location_is_refused_like_any_database_path(self):
        outside = tempfile.mkdtemp(prefix="mantis_sess_outside_")
        self.addCleanup(shutil.rmtree, outside, True)
        link = Path(self.tmp, "link")
        link.symlink_to(outside)
        with self.assertRaises(PermissionError):
            P.resolve_sessions_db_path("", str(link / "knowledge.db"))


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_sess_sum_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_size_sessions_and_runs_are_counted(self):
        path = os.path.join(self.tmp, "sessions.db")
        _make_store(path, ["a" * 36, "b" * 36, "c" * 36], sessions_per_run=3)
        Path(path + "-wal").write_bytes(b"x" * 1000)
        summary = P.sessions_store_summary(path)
        self.assertEqual(summary["sessions"], 9)
        self.assertEqual(summary["runs"], 3)
        self.assertGreaterEqual(summary["bytes"], os.path.getsize(path) + 1000)
        note = P.render_sessions_store_note(summary)
        self.assertIn("9 session(s) from 3 run(s)", note)
        self.assertIn("--resume", note)
        self.assertIn(path, note)

    def test_a_missing_store_is_zeros_not_an_error(self):
        summary = P.sessions_store_summary(os.path.join(self.tmp, "nope.db"))
        self.assertEqual((summary["bytes"], summary["sessions"], summary["runs"]), (0, 0, 0))
        self.assertIn("0 B", P.render_sessions_store_note(summary))
        self.assertEqual(P.sessions_store_summary("")["bytes"], 0)

    def test_format_bytes(self):
        self.assertEqual(P.format_bytes(0), "0 B")
        self.assertEqual(P.format_bytes(1536), "1.5 KB")
        self.assertEqual(P.format_bytes(2_271_000_000), "2.1 GB")
        self.assertEqual(P.format_bytes("x"), "0 B")


class SpendReportTest(unittest.TestCase):
    def test_the_spend_report_carries_the_store_size(self):
        import main as main_mod

        tmp = tempfile.mkdtemp(prefix="mantis_sess_rep_")
        try:
            store = os.path.join(tmp, "sessions.db")
            _make_store(store, ["d" * 36])
            with mock.patch.dict(main_mod._SPEND_REPORT, {"controller": None, "sessions_db_path": store}):
                out = os.path.join(tmp, "spend.json")
                self.assertTrue(main_mod.write_spend_report(out, 0))
                with open(out, encoding="utf-8") as fh:
                    data = json.load(fh)
            self.assertEqual(data["sessions_db_bytes"], os.path.getsize(store))
            with mock.patch.dict(main_mod._SPEND_REPORT, {"controller": None, "sessions_db_path": ""}):
                out2 = os.path.join(tmp, "spend2.json")
                self.assertTrue(main_mod.write_spend_report(out2, 0))
                with open(out2, encoding="utf-8") as fh:
                    self.assertEqual(json.load(fh)["sessions_db_bytes"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
