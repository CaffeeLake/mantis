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

"""The structured transcript: what one ADK event becomes on disk, and what reads back.

The console echo clips tool responses to a few hundred characters; the
transcript is the record a proposer (or a person) reads afterwards, so it
must keep whole tool responses, drop thinking, and survive a crash mid-run.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

_REF_ROOT = Path(__file__).resolve().parents[1]
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.transcript import (  # noqa: E402
    TRANSCRIPT_PAYLOAD_LIMIT,
    TRANSCRIPT_TEXT_LIMIT,
    TranscriptWriter,
    clip,
    event_transcript_rows,
    read_transcript,
)


def _event(*parts, author="researcher", partial=False):
    return NS(author=author, partial=partial, content=NS(parts=list(parts)))


def _text(text, thought=False):
    return NS(text=text, thought=thought, function_call=None, function_response=None)


def _call(name, args):
    return NS(text=None, thought=False, function_call=NS(name=name, args=args), function_response=None)


def _resp(name, response):
    return NS(text=None, thought=False, function_call=None, function_response=NS(name=name, response=response))


class EventRowsTest(unittest.TestCase):
    def test_prose_calls_and_responses_flatten_in_order_and_thinking_is_dropped(self):
        ev = _event(
            _text("let me think", thought=True),
            _text("I will read the DAL."),
            _call("read_file", {"filepath": "dal/users.py"}),
            _resp("read_file", {"result": "def find_user(name):\n    pass\n"}),
        )
        rows = event_transcript_rows(ev, node="researcher")
        self.assertEqual([r["type"] for r in rows], ["text", "tool_call", "tool_response"])
        self.assertEqual(rows[0]["text"], "I will read the DAL.")
        self.assertEqual(rows[1]["name"], "read_file")
        self.assertIn("dal/users.py", rows[1]["args"])
        self.assertEqual(rows[2]["response"], "def find_user(name):\n    pass\n")
        for r in rows:
            self.assertEqual(r["author"], "researcher")
            self.assertEqual(r["node"], "researcher")
        self.assertNotIn("think", json.dumps(rows))

    def test_partial_stream_chunks_produce_no_rows(self):
        self.assertEqual(event_transcript_rows(_event(_text("par"), partial=True)), [])

    def test_a_model_error_is_one_error_row(self):
        ev = NS(author="researcher", partial=False, error_code="BadRequestError",
                error_message="400 bad tool schema", content=None)
        rows = event_transcript_rows(ev, node="researcher")
        self.assertEqual(rows, [{"author": "researcher", "node": "researcher", "type": "error",
                                 "name": "BadRequestError", "text": "400 bad tool schema"}])

    def test_a_response_that_is_not_a_string_is_serialised_not_dropped(self):
        rows = event_transcript_rows(_event(_resp("report_findings", {"count": 2, "ids": [1, 2]})))
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0]["response"]), {"count": 2, "ids": [1, 2]})
        rows = event_transcript_rows(_event(_resp("x", "plain string")))
        self.assertEqual(rows[0]["response"], "plain string")

    def test_long_payloads_are_clipped_with_a_marker_that_says_how_much(self):
        big = "x" * (TRANSCRIPT_PAYLOAD_LIMIT + 123)
        rows = event_transcript_rows(_event(_resp("read_file", {"result": big})))
        self.assertTrue(rows[0]["response"].startswith("x" * TRANSCRIPT_PAYLOAD_LIMIT))
        self.assertTrue(rows[0]["response"].endswith("... [clipped 123 chars]"))
        self.assertEqual(clip("short", 10), "short")
        rows = event_transcript_rows(_event(_text("y" * (TRANSCRIPT_TEXT_LIMIT + 1))))
        self.assertIn("[clipped 1 chars]", rows[0]["text"])

    def test_a_malformed_event_never_raises(self):
        self.assertEqual(event_transcript_rows(None), [])
        self.assertEqual(event_transcript_rows(NS(author="a", partial=False, content=None)), [])
        self.assertEqual(event_transcript_rows(NS(author="a", partial=False, content=NS(parts=[object()]))), [])


class WriterTest(unittest.TestCase):
    def test_rows_append_as_they_arrive_and_read_back_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "transcript.raw.jsonl"
            w = TranscriptWriter(path)
            w.write(event_transcript_rows(_event(_text("one")), node="a"))
            # Flushed per write: a reader mid-run sees what has happened so far.
            self.assertEqual([r["text"] for r in read_transcript(path)], ["one"])
            w.write(event_transcript_rows(_event(_call("f", {"k": 1}), _resp("f", "ok")), node="b"))
            w.write([])
            w.close()
            w.close()  # idempotent
            rows = read_transcript(path)
            self.assertEqual([r["type"] for r in rows], ["text", "tool_call", "tool_response"])
            self.assertEqual([r["node"] for r in rows], ["a", "b", "b"])
            self.assertEqual(w.rows, 3)
            # A second writer appends rather than truncates (campaigns share one file).
            w2 = TranscriptWriter(path)
            w2.write(event_transcript_rows(_event(_text("two"))))
            w2.close()
            self.assertEqual(len(read_transcript(path)), 4)

    def test_no_path_means_no_file_and_no_errors(self):
        w = TranscriptWriter(None)
        w.write(event_transcript_rows(_event(_text("x"))))
        w.close()
        self.assertEqual(w.rows, 0)
        self.assertIsNone(w.path)

    def test_reader_skips_torn_lines_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.jsonl"
            path.write_text('{"type": "text", "text": "a"}\n{"type": "text", "te\n\n{"type": "text", "text": "b"}\n')
            self.assertEqual([r["text"] for r in read_transcript(path)], ["a", "b"])
            self.assertEqual(read_transcript(Path(tmp) / "absent.jsonl"), [])


class LaunchFlagTest(unittest.TestCase):
    def test_transcript_out_is_a_launch_flag_threaded_to_the_pipeline(self):
        from scripts.launch import build_parser, run_launch
        import inspect

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(build_parser().parse_args(["x"]).transcript_out, "")
            self.assertEqual(build_parser().parse_args(["x", "--transcript-out", "/tmp/t.jsonl"]).transcript_out,
                             "/tmp/t.jsonl")
        self.assertIn("transcript_out", inspect.signature(run_launch).parameters)
        import main
        self.assertIn("transcript_out", inspect.signature(main.pipeline).parameters)
        self.assertIn("transcript", inspect.signature(main.execute_sub_task).parameters)


if __name__ == "__main__":
    unittest.main()
