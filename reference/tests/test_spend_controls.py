"""Spend controls the PR-scan driver leans on in the pipeline itself.

Three things, each independent of the model:

- MANTIS_MAX_INPUT_TOKENS: an absolute per-request input ceiling, enforced
  before dispatch, that holds for models litellm has no window for.
- BudgetConfig.cache_discount: the weight of a cached input token against
  max_tokens, so a caller can insist on full price (1.0) and have the token
  ceiling bound the raw count a provider bills.
- --spend-report / write_spend_report: the run's totals written on exit
  however the run ended, including the pause the ledger leaves out.

Nothing here reaches a model.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core import config as config_mod  # noqa: E402
from core.budget import (  # noqa: E402
    BudgetConfig,
    BudgetController,
    BudgetExceededError,
    CampaignBudgetScope,
    clamp_cache_discount,
)

UNKNOWN_MODEL = "totally-made-up-provider/nonexistent-model-xyz"
KNOWN_MODEL = "vertex_ai/gemini-3.5-flash-lite"


class TestInputTokenCap(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("MANTIS_MAX_INPUT_TOKENS")
        os.environ.pop("MANTIS_MAX_INPUT_TOKENS", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("MANTIS_MAX_INPUT_TOKENS", None)
        else:
            os.environ["MANTIS_MAX_INPUT_TOKENS"] = self._saved

    def test_cap_parsing(self):
        for raw in ("", "  ", "0", "-5", "abc", "1e6"):
            os.environ["MANTIS_MAX_INPUT_TOKENS"] = raw
            self.assertIsNone(config_mod.resolve_input_token_cap(), raw)
        os.environ["MANTIS_MAX_INPUT_TOKENS"] = " 4096 "
        self.assertEqual(config_mod.resolve_input_token_cap(), 4096)

    def test_unknown_model_fails_open_without_a_cap_and_closed_with_one(self):
        self.assertIsNone(config_mod.resolve_context_limit(UNKNOWN_MODEL),
                          "probe model has a window; the fail-open path is untested")
        big = [{"role": "user", "content": "x" * 400_000}]  # ~100k tokens
        config_mod.enforce_context_budget(UNKNOWN_MODEL, big)  # no cap: passes
        os.environ["MANTIS_MAX_INPUT_TOKENS"] = "50000"
        with self.assertRaises(config_mod.ContextBudgetExceededError) as caught:
            config_mod.enforce_context_budget(UNKNOWN_MODEL, big)
        self.assertEqual(caught.exception.limit, 50_000)
        self.assertIn("MANTIS_MAX_INPUT_TOKENS=50,000", str(caught.exception))
        self.assertIn("role=user", str(caught.exception))

    def test_cap_below_the_window_governs_and_above_it_defers(self):
        window = config_mod.resolve_context_limit(KNOWN_MODEL)
        self.assertIsNotNone(window)
        just_over_cap = [{"role": "user", "content": "x" * (4 * 60_000)}]  # ~60k tokens
        os.environ["MANTIS_MAX_INPUT_TOKENS"] = "50000"
        with self.assertRaises(config_mod.ContextBudgetExceededError) as caught:
            config_mod.enforce_context_budget(KNOWN_MODEL, just_over_cap)
        self.assertEqual(caught.exception.limit, 50_000)
        # A cap above the window changes nothing: the window still governs.
        os.environ["MANTIS_MAX_INPUT_TOKENS"] = str(window * 10)
        config_mod.enforce_context_budget(KNOWN_MODEL, just_over_cap)
        with self.assertRaises(config_mod.ContextBudgetExceededError) as caught:
            config_mod.enforce_context_budget(
                KNOWN_MODEL, [{"role": "user", "content": "x" * (4 * window)}])
        self.assertEqual(caught.exception.limit, window)
        self.assertIn("context window", str(caught.exception))

    def test_cap_is_enforced_on_the_dispatch_path(self):
        import litellm

        os.environ["MANTIS_MAX_INPUT_TOKENS"] = "1000"

        async def _never(*args, **kwargs):
            raise AssertionError("dispatched above the cap")

        with mock.patch.object(litellm, "acompletion", _never):
            with self.assertRaises(config_mod.ContextBudgetExceededError):
                asyncio.run(config_mod.ResilientLiteLLMClient().acompletion(
                    model=UNKNOWN_MODEL,
                    messages=[{"role": "user", "content": "x" * 40_000}], tools=None))


class TestCacheDiscount(unittest.TestCase):
    def test_config_default_and_parsing(self):
        self.assertEqual(BudgetConfig().cache_discount, 0.1)
        self.assertEqual(BudgetConfig.from_dict({}).cache_discount, 0.1)
        self.assertEqual(BudgetConfig.from_dict({"cache_discount": "1"}).cache_discount, 1.0)
        self.assertEqual(BudgetConfig.from_dict({"cache_discount": 0}).cache_discount, 0.0)
        for bad in ("1.5", -0.1, "x", None):
            with self.assertRaises(ValueError):
                clamp_cache_discount(bad)

    def test_full_price_makes_the_ceiling_bound_the_raw_count(self):
        # At the default discount, 10,000 tokens of which 9,000 are cached
        # count as 1,900: a 2,000 ceiling is not reached. At full price the
        # same call counts as 10,000 and the ceiling trips.
        lenient = BudgetController(BudgetConfig(max_tokens=2_000))
        lenient.record_tokens(10_000, cached_count=9_000)
        self.assertEqual(lenient.accumulated_tokens, 1_900)
        strict = BudgetController(BudgetConfig(max_tokens=2_000, cache_discount=1.0))
        with self.assertRaises(BudgetExceededError) as caught:
            strict.record_tokens(10_000, cached_count=9_000)
        self.assertEqual(caught.exception.trigger, "token_budget")
        self.assertEqual(strict.accumulated_tokens, 10_000)
        self.assertEqual(strict.fresh_tokens + strict.cached_tokens, 10_000)

    def test_campaign_scope_uses_the_same_discount(self):
        run = BudgetController(BudgetConfig(max_tokens=0, cache_discount=1.0))
        scope = CampaignBudgetScope(run)
        scope.record_tokens(1_000, cached_count=600)
        self.assertEqual(run.accumulated_tokens, 1_000)
        self.assertEqual(scope.accumulated_tokens, 1_000)
        # An explicit per-call discount still wins, for callers that pass one.
        scope.record_tokens(1_000, cached_count=1_000, cache_discount=0.0)
        self.assertEqual(run.accumulated_tokens, 1_000)

    def test_snapshot_reports_both_counts_and_the_trigger(self):
        ctrl = BudgetController(BudgetConfig(max_tokens=5_000, max_llm_calls=7), run_id="r1")
        ctrl.record_tokens(1_000, cached_count=500)
        snap = ctrl.spend_snapshot()
        self.assertEqual(snap["run_id"], "r1")
        self.assertEqual(snap["tokens"], 550)
        self.assertEqual(snap["fresh_tokens"], 500)
        self.assertEqual(snap["cached_tokens"], 500)
        self.assertEqual(snap["llm_calls"], 1)
        self.assertEqual(snap["max_tokens"], 5_000)
        self.assertEqual(snap["max_llm_calls"], 7)
        self.assertEqual(snap["cache_discount"], 0.1)
        self.assertEqual(snap["pause_trigger"], "")
        ctrl.pause_trigger = "token_budget"
        self.assertEqual(ctrl.spend_snapshot()["pause_trigger"], "token_budget")


class TestSpendReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mantis_spend_")
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", self.tmp]))

    def test_written_atomically_with_and_without_a_controller(self):
        import main as main_mod

        path = os.path.join(self.tmp, "spend.json")
        main_mod._SPEND_REPORT["controller"] = None
        self.assertTrue(main_mod.write_spend_report(path, 1))
        data = json.loads(Path(path).read_text())
        self.assertEqual((data["tokens"], data["llm_calls"], data["exit_code"]), (0, 0, 1))
        self.assertEqual([p for p in os.listdir(self.tmp) if p.endswith(".tmp")], [])

        ctrl = BudgetController(BudgetConfig(max_tokens=0), run_id="run-x")
        ctrl.record_tokens(1234, cached_count=34)
        ctrl.pause_trigger = "llm_calls_limit"
        main_mod._SPEND_REPORT["controller"] = ctrl
        try:
            self.assertTrue(main_mod.write_spend_report(path, 2))
            data = json.loads(Path(path).read_text())
            self.assertEqual(data["run_id"], "run-x")
            self.assertEqual(data["fresh_tokens"] + data["cached_tokens"], 1234)
            self.assertEqual(data["pause_trigger"], "llm_calls_limit")
            self.assertEqual(data["exit_code"], 2)
        finally:
            main_mod._SPEND_REPORT["controller"] = None
        self.assertFalse(main_mod.write_spend_report("", 0))
        # An unwritable path is a warning, never an exception.
        self.assertFalse(main_mod.write_spend_report(os.path.join(self.tmp, "no", "dir", "x"), 0))

    def test_cli_rejects_bad_budget_values_before_running(self):
        # Usage errors exit 64; 2 means "paused at budget" and a driver
        # must be able to tell the two apart.
        for flag, value in (("--max-tokens", "lots"), ("--token-budget", "lots"), ("--cache-discount", "2")):
            proc = subprocess.run(
                [sys.executable, str(_REF_ROOT / "main.py"), str(_REF_ROOT / "README.md"),
                 flag, value, "--yes"],
                capture_output=True, text=True, cwd=str(_REF_ROOT), timeout=120)
            self.assertEqual(proc.returncode, 64, proc.stderr[-500:])
            self.assertIn(flag.lstrip("-"), proc.stderr)

    def test_spend_report_is_written_even_when_the_run_never_starts(self):
        # No such target: the pipeline exits before a controller exists, and
        # the report still appears, with zeros, so the driver can tell "spent
        # nothing" from "killed before it could say".
        path = os.path.join(self.tmp, "spend.json")
        proc = subprocess.run(
            [sys.executable, str(_REF_ROOT / "main.py"), os.path.join(self.tmp, "missing.py"),
             "--yes", "--spend-report", path, "--sandbox", "static-only"],
            capture_output=True, text=True, cwd=str(_REF_ROOT), timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        data = json.loads(Path(path).read_text())
        self.assertEqual(data["tokens"], 0)
        self.assertEqual(data["exit_code"], proc.returncode)


if __name__ == "__main__":
    unittest.main()
