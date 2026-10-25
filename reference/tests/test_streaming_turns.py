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

"""Streamed model turns: the default, the knobs that turn them off, and the
bookkeeping that must see a streamed tool call once -- on the final event
-- rather than once per partial chunk."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REF_ROOT))

from google.adk.agents.run_config import StreamingMode  # noqa: E402
from google.genai import types  # noqa: E402

from core import config as C  # noqa: E402
from core.budget import BudgetConfig, BudgetController  # noqa: E402
from core.graph_loader import count_event_tool_calls, event_has_pseudo_tool_call  # noqa: E402


class StreamingKnobTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(C.configure_llm_streaming, True)
        os.environ.pop("MANTIS_LLM_STREAM", None)

    def test_on_by_default(self):
        self.assertTrue(C.llm_streaming_enabled())

    def test_the_workflow_may_turn_it_off_and_none_leaves_it(self):
        C.configure_llm_streaming(False)
        self.assertFalse(C.llm_streaming_enabled())
        C.configure_llm_streaming(None)
        self.assertFalse(C.llm_streaming_enabled())
        C.configure_llm_streaming(True)
        self.assertTrue(C.llm_streaming_enabled())

    def test_the_environment_overrides_either_way(self):
        C.configure_llm_streaming(True)
        with mock.patch.dict(os.environ, {"MANTIS_LLM_STREAM": "0"}):
            self.assertFalse(C.llm_streaming_enabled())
        C.configure_llm_streaming(False)
        with mock.patch.dict(os.environ, {"MANTIS_LLM_STREAM": "1"}):
            self.assertTrue(C.llm_streaming_enabled())


def _event(parts, partial=False, author="researcher"):
    return mock.Mock(author=author, partial=partial, error_code=None, usage_metadata=None,
                     content=types.Content(role="model", parts=parts), actions=None, node_info=None)


def _call(name="read_file", args=None):
    return types.Part(function_call=types.FunctionCall(name=name, args=args or {"filepath": "a.py"}))


class PartialAwareHelpersTest(unittest.TestCase):
    def test_tool_calls_count_on_the_final_event_only(self):
        self.assertEqual(count_event_tool_calls(_event([_call()], partial=True)), 0)
        self.assertEqual(count_event_tool_calls(_event([_call(), _call("list_files")])), 2)

    def test_pseudo_calls_are_judged_on_final_text_only(self):
        text = types.Part.from_text(text="call:api:read_file{...}")
        self.assertFalse(event_has_pseudo_tool_call(_event([text], partial=True)))
        self.assertTrue(event_has_pseudo_tool_call(_event([text])))


class DriveCampaignPartialsTest(unittest.TestCase):
    """main._drive_campaign sees a streamed call three times: two partial
    chunks and the final event. It must print and count it once."""

    def _drive(self, events, streaming=True):
        import main as M

        class _Runner:
            agent = None

            async def run_async(self, **kwargs):
                for ev in events:
                    yield ev

        controller = BudgetController(BudgetConfig(max_tokens=0), run_id="r")
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"MANTIS_LLM_STREAM": "1" if streaming else "0"}), \
                contextlib.redirect_stdout(out):
            errored = asyncio.run(M._drive_campaign(
                _Runner(), "s", None, None, filepath="app/a.py", run_id="r", db_path="",
                status_map=None, budget_controller=controller, transcript=None,
            ))
        return errored, controller, out.getvalue()

    def test_a_streamed_call_is_counted_and_printed_once(self):
        events = [
            _event([_call(args={})], partial=True),
            _event([_call(args={"filepath": "a."})], partial=True),
            _event([_call()]),
        ]
        errored, controller, printed = self._drive(events)
        self.assertFalse(errored)
        self.assertEqual(sum(controller.node_tool_counts.values()), 1)
        self.assertEqual(printed.count("[TOOL CALL: read_file]"), 1)

    def test_streamed_text_prints_as_it_arrives_and_the_final_is_not_repeated(self):
        events = [
            _event([types.Part.from_text(text="hel")], partial=True),
            _event([types.Part.from_text(text="lo")], partial=True),
            _event([types.Part.from_text(text="hello")]),
        ]
        _errored, _controller, printed = self._drive(events)
        self.assertEqual(printed.count("hello"), 1)

    def test_the_run_config_follows_the_knob(self):
        import main as M
        seen = {}

        class _Runner:
            agent = None

            async def run_async(self, **kwargs):
                seen["mode"] = kwargs["run_config"].streaming_mode
                return
                yield  # pragma: no cover

        for streaming, expected in ((True, StreamingMode.SSE), (False, StreamingMode.NONE)):
            with mock.patch.dict(os.environ, {"MANTIS_LLM_STREAM": "1" if streaming else "0"}):
                asyncio.run(M._drive_campaign(
                    _Runner(), "s", None, None, filepath="app/a.py", run_id="r", db_path="",
                    status_map=None, budget_controller=None, transcript=None,
                ))
            self.assertEqual(seen["mode"], expected)


if __name__ == "__main__":
    unittest.main()
