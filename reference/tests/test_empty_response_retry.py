"""An empty STOP turn is retried like a 429, and charged like the call it was.

ADK reports a non-streaming turn that finishes with STOP and no parts as
MODEL_RETURNED_NO_CONTENT and the campaign is marked failed over one empty
sample. The model layer now catches that turn before ADK sees it, meters its
usage against the run budget, and sends it back through the same backoff
loop as a rate limit. Past the patience window the empty turn is passed on
unchanged. Streaming is left alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REF_ROOT))

from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402

from core.budget import BudgetConfig, BudgetController, BudgetExceededError  # noqa: E402
from core.config import (  # noqa: E402
    MantisEmptyResponseError,
    ResilientLiteLlm,
    is_empty_stop_response,
    is_retryable_llm_error,
)
from core.context import RunContext, current_run_context  # noqa: E402

_FAST = {
    "MANTIS_LLM_RETRY_INITIAL_DELAY": "0",
    "MANTIS_LLM_MIN_OFFSET": "0",
    "MANTIS_LLM_RETRY_MAX_DELAY": "0",
    "MANTIS_LLM_MAX_PATIENCE_SECONDS": "60",
}


def _usage(total: int) -> types.GenerateContentResponseUsageMetadata:
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=total - 1, candidates_token_count=1, total_token_count=total,
    )


def empty(total: int = 1000) -> LlmResponse:
    """What ADK's litellm adapter builds from a choice with no message."""
    return LlmResponse(
        content=types.Content(role="model", parts=[]),
        finish_reason=types.FinishReason.STOP,
        usage_metadata=_usage(total),
    )


def good(text: str = "done", total: int = 500, partial: bool = False) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
        finish_reason=types.FinishReason.STOP,
        partial=partial,
        usage_metadata=_usage(total),
    )


class ScriptedUpstream:
    """Stands in for LiteLlm.generate_content_async: one scripted response
    list per call, in order."""

    def __init__(self, calls):
        self.calls = [list(c) for c in calls]
        self.seen = []

    def __call__(self):
        async def generate_content_async(_self, llm_request, stream=False):
            self.seen.append(stream)
            for response in self.calls.pop(0):
                yield response
        return generate_content_async


class EmptyTurnRetryTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, _FAST)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.llm = ResilientLiteLlm(model="openai/fake")
        self.request = LlmRequest(
            model="openai/fake",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text="look at this")])],
        )

    def with_budget(self, max_tokens=0):
        controller = BudgetController(BudgetConfig(max_tokens=max_tokens), run_id="r")
        token = current_run_context.set(RunContext(jail_dir="", db_path="", budget_controller=controller))
        self.addCleanup(current_run_context.reset, token)
        return controller

    def run_turn(self, upstream, stream=False):
        async def go():
            out = []
            async for r in self.llm.generate_content_async(self.request, stream=stream):
                out.append(r)
            return out

        err = io.StringIO()
        with mock.patch.object(LiteLlm, "generate_content_async", upstream()), contextlib.redirect_stderr(err):
            return asyncio.run(go()), err.getvalue()

    def test_the_condition_is_adk_s(self):
        self.assertTrue(is_empty_stop_response(empty()))
        self.assertTrue(is_empty_stop_response(LlmResponse(finish_reason=types.FinishReason.STOP)))
        self.assertFalse(is_empty_stop_response(good()))
        self.assertFalse(is_empty_stop_response(good(partial=True)))
        blocked = empty()
        blocked.finish_reason = types.FinishReason.SAFETY
        self.assertFalse(is_empty_stop_response(blocked), "a different finish reason is a different problem")
        errored = empty()
        errored.error_code = "SOMETHING"
        self.assertFalse(is_empty_stop_response(errored))
        self.assertTrue(is_retryable_llm_error(MantisEmptyResponseError("x")))

    def test_an_empty_turn_is_retried_and_charged(self):
        controller = self.with_budget()
        upstream = ScriptedUpstream([[empty(1000)], [empty(1000)], [good()]])
        out, err = self.run_turn(upstream)
        self.assertEqual([r.content.parts[0].text for r in out], ["done"])
        self.assertEqual(len(upstream.seen), 3, "two empty turns, one answer")
        self.assertEqual(controller.accumulated_tokens, 2000, "both empty turns were billed, so both are charged")
        self.assertEqual(controller.llm_calls, 2)
        self.assertEqual(err.count("[LLM RETRY]"), 2)
        self.assertIn("returned no content", err)
        self.assertNotIn("passing the empty turn on", err)

    def test_out_of_patience_passes_the_empty_turn_on(self):
        self.with_budget()
        with mock.patch.dict(os.environ, {"MANTIS_LLM_MAX_PATIENCE_SECONDS": "0"}):
            upstream = ScriptedUpstream([[empty()], [good()]])
            out, err = self.run_turn(upstream)
        self.assertEqual(len(upstream.seen), 1, "no patience, no retry")
        self.assertEqual(len(out), 1)
        self.assertTrue(is_empty_stop_response(out[0]), "ADK gets the empty turn and flags it as before")
        self.assertIn("passing the empty turn on", err)

    def test_the_budget_ceiling_stops_the_retries(self):
        controller = self.with_budget(max_tokens=1500)
        upstream = ScriptedUpstream([[empty(1000)], [empty(1000)], [good()]])
        with self.assertRaises(BudgetExceededError):
            self.run_turn(upstream)
        self.assertEqual(len(upstream.seen), 2, "the second empty turn crossed the ceiling")
        self.assertEqual(controller.accumulated_tokens, 2000)

    def test_no_budget_in_context_still_retries(self):
        upstream = ScriptedUpstream([[empty()], [good()]])
        out, _err = self.run_turn(upstream)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(upstream.seen), 2)

    def test_streaming_is_left_alone(self):
        self.with_budget()
        final = LlmResponse(content=types.Content(role="model", parts=[]), finish_reason=types.FinishReason.STOP)
        upstream = ScriptedUpstream([[good("chunk", partial=True), final]])
        out, err = self.run_turn(upstream, stream=True)
        self.assertEqual(len(out), 2, "a finish-only chunk after streamed content is normal")
        self.assertEqual(len(upstream.seen), 1)
        self.assertNotIn("[LLM RETRY]", err)


class _EmptyTurnServer:
    """An OpenAI-compatible endpoint whose every answer is an empty STOP turn
    billed at 1000 tokens. What the bug report's model did, on demand."""

    def __init__(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                server.requests += 1
                body = json.dumps({
                    "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 999, "completion_tokens": 1, "total_tokens": 1000},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.requests = 0
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class EndToEndTest(unittest.TestCase):
    """Through ADK, litellm and HTTP: what the pipeline does with a model
    that never says anything."""

    def setUp(self):
        self.server = _EmptyTurnServer()
        self.addCleanup(self.server.close)
        self.tmp = tempfile.mkdtemp(prefix="mantis_empty_e2e_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = os.path.realpath(os.path.join(self.tmp, "repo"))
        os.makedirs(self.repo)
        self.target = os.path.join(self.repo, "a.py")
        Path(self.target).write_text("def a():\n    return 1\n")
        self.db = os.path.join(self.tmp, "m.db")

    def run_main(self, patience, *extra):
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        env.update(_FAST)
        env.update({
            "OPENAI_API_KEY": "dummy", "LLM_TIMEOUT": "10",
            "MANTIS_LLM_MAX_PATIENCE_SECONDS": str(patience),
            "MANTIS_LLM_RETRY_INITIAL_DELAY": "0.2", "MANTIS_LLM_MIN_OFFSET": "0.2", "MANTIS_LLM_RETRY_MAX_DELAY": "0.5",
        })
        proc = subprocess.run(
            [sys.executable, str(_REF_ROOT / "main.py"), self.repo, "--scan-mode", "file-by-file", "--yes",
             "--sandbox", "static-only", "--model", "openai/fake", "--api-base", self.server.url,
             "--max-llm-calls", "30", "--db", self.db, *extra],
            capture_output=True, text=True, timeout=300, env=env, cwd=str(_REF_ROOT))
        return proc.returncode, proc.stdout + proc.stderr

    def ledger(self):
        import sqlite3

        if not os.path.exists(self.db):
            return []
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute("SELECT tokens, llm_calls, metadata_json FROM campaign_spend").fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def test_past_patience_the_campaign_fails_as_before_and_every_turn_is_charged(self):
        code, out = self.run_main(2)
        self.assertIn("[MantisEmptyResponseError: The model returned no content", out)
        self.assertIn("passing the empty turn on", out)
        self.assertIn("[EVENT ERROR MODEL_RETURNED_NO_CONTENT]", out, "ADK still sees and names the empty turn")
        self.assertNotIn("BUDGET PAUSE", out)
        self.assertNotEqual(code, 0)
        rows = self.ledger()
        self.assertEqual(len(rows), 1, out[-2000:])
        tokens, calls, meta = rows[0]
        self.assertEqual(json.loads(meta), {"failed": True}, "the terminal state --resume picks up")
        self.assertGreater(self.server.requests, 6, "retried, not just passed on")
        self.assertEqual((tokens, calls), (1000 * self.server.requests, self.server.requests),
                         "every request the server saw is on the ledger: swallowed turns charged, none twice")

    def test_the_token_budget_pauses_the_run_before_patience_does(self):
        wf = json.loads(Path(_REF_ROOT, "workflow.json").read_text())
        wf["budget"]["max_tokens"] = 2500
        wf_path = os.path.join(self.tmp, "wf.json")
        Path(wf_path).write_text(json.dumps(wf))
        code, out = self.run_main(60, "--workflow", wf_path)
        self.assertEqual(code, 2, out[-2000:])
        self.assertIn("BUDGET PAUSE", out)
        self.assertIn("Accumulated tokens 3,000 reached ceiling of 2,500", out)
        self.assertIn("--resume", out)
        self.assertNotIn("MODEL_RETURNED_NO_CONTENT", out, "the budget spoke first")
        self.assertEqual(self.server.requests, 3, "the third empty turn crossed the ceiling; nothing after it")
        self.assertEqual(self.ledger(), [], "a paused campaign records no spend row, as before")


if __name__ == "__main__":
    unittest.main()
