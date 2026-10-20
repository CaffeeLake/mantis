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

"""The Harness2 loop over a Mantis harness, end to end with no model:
surface round-trip and validation, the adapter over a fake pipeline, the
ADK proposer over a scripted agent, evidence-first selection, and one full
Propose-Probe-Compose iteration whose ledger identity holds."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

from core.database import init_db, write_findings  # noqa: E402
from evals.evolve import adapter as ad  # noqa: E402
from evals.evolve import run as ev_run  # noqa: E402
from evals.evolve import selection as ev_sel  # noqa: E402
from evals.evolve import surface as S  # noqa: E402
from evals.evolve import surface_text  # noqa: E402
from evals.evolve._vendor import load_harness2  # noqa: E402
from evals.evolve.proposer import MantisProposer, make_context_tools  # noqa: E402

WORKFLOW = _REF_ROOT / "workflow.json"
EDIT_MARKER = "Prefer get_function_boundary over whole-file reads."


# --- Fakes ---------------------------------------------------------------------------------


def fake_pipeline(target: Path, workflow_path: Path, run_dir: Path, settings) -> dict:
    """Stands in for main.py: leaves spend.json, knowledge.db and stdout.log behind.

    A harness whose researcher prompt carries EDIT_MARKER "finds" one more,
    machine-verified bug; every harness finds the base static one.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    researcher = next(n for n in wf["nodes"] if n["id"] == "researcher")
    improved = EDIT_MARKER in str(researcher.get("system_prompt") or "")
    run_id = f"pipe-{run_dir.name}"
    db = run_dir / "knowledge.db"
    init_db(str(db))
    findings = [{"title": "SQL injection in find_user", "severity": "HIGH", "status": "static_confirmed",
                 "description": "f-string query", "line_numbers": [12], "filepath": "dal/users.py", "cwe": "CWE-89"}]
    if improved:
        findings.append({"title": "Command injection in export", "severity": "CRITICAL", "status": "dynamic_confirmed",
                         "description": "shell=True with request data", "line_numbers": [30], "filepath": "services/export.py", "cwe": "CWE-78"})
    write_findings(str(db), "dal/users.py", findings, run_id=run_id)
    (run_dir / "spend.json").write_text(json.dumps({"run_id": run_id, "tokens": 400_000 if improved else 500_000,
                                                    "llm_calls": 20, "exit_code": 0}), encoding="utf-8")
    # The console echo clips every tool response and spills file lines as prose...
    (run_dir / "stdout.log").write_text(
        "-- mantis_vulnerability_pipeline@1/researcher@1\n"
        "[TOOL CALL: read_file] args={'filepath': 'dal/users.py'}\n"
        "[TOOL RESPONSE: read_file] <<<UNTRUSTED_SOURCE_CODE_DATA_START path=dal/users.py>>>\n"
        "def find_user(name):\n    return conn.execute(f\"SELECT * FROM users WHERE name='{name}'\")\n"
        "I will report the query flaw.\n"
        "[TOOL CALL: report_findings] args={'findings': [...]}\n"
        "[TOOL RESPONSE: report_findings] Recorded 1 finding(s).\n", encoding="utf-8")
    # ...while --transcript-out carries the complete rows (the adapter must prefer these).
    big_file = "def find_user(name):\n" + "    pass\n" * 400 + "    return conn.execute(f\"SELECT * FROM users WHERE name='{name}'\")\n"
    rows = [
        {"author": "researcher", "node": "researcher", "type": "tool_call", "name": "read_file", "args": "{'filepath': 'dal/users.py'}"},
        {"author": "researcher", "node": "researcher", "type": "tool_response", "name": "read_file", "response": big_file},
        {"author": "researcher", "node": "researcher", "type": "text", "text": "I will report the query flaw."},
        {"author": "researcher", "node": "researcher", "type": "tool_call", "name": "report_findings", "args": "{'findings': [...]}"},
        {"author": "researcher", "node": "researcher", "type": "tool_response", "name": "report_findings", "response": "Recorded 1 finding(s)."},
        {"author": "reviewer", "node": "reviewer", "type": "error", "name": "RateLimitError", "text": "429 slow down"},
    ]
    (run_dir / "transcript.raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return {"exit_code": 0, "wall_seconds": 1.0, "timed_out": False}


def scripted_agent(*, mission: str, system: str, context_root: Path, touched, label: str, timeout, read_only=False):
    """Edits like a proposer would, chooses like a selector would, reflects like a reflector would."""
    harness = context_root / "harness"
    if label.startswith("select."):
        ids = sorted(p.stem for p in (context_root / "candidates").glob("*.md"))
        return {"text": f"<why>Both reports are equivalent; the first is plainer.</why>\n```choice\n{ids[0]}\n```", "steps": 2, "tokens": 100}
    if label.startswith("experience."):
        return {"text": ("<experience>\ncue: auditing a Flask service with route -> service -> store layers\n"
                         "lesson: give the researcher get_function_boundary and tell it to prefer spans over whole files\n"
                         "evidence: the agg1 rollout's researcher reported the export sink the base missed after reading spans\n"
                         "confidence: medium\n</experience>"), "steps": 1, "tokens": 50}
    # inv / dom / agg proposals: append one line to the researcher prompt.
    p = harness / "prompts" / "researcher.md"
    if p.exists():
        p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n" + EDIT_MARKER + "\n", encoding="utf-8")
        touched.append("prompts/researcher.md")
    return {"text": "<notes>researcher: the base trajectory read dal/users.py whole three times; spans are cheaper.</notes>",
            "steps": 3, "tokens": 200}


def _adapter(tmp: Path, target: Path) -> ad.MantisRepoAdapter:
    return ad.MantisRepoAdapter(targets={"demo": target}, workflow_path=WORKFLOW, runs_root=tmp / "runs",
                                settings=ad.RunSettings(model="m", timeout_seconds=60), runner=fake_pipeline, domain="demo")


# --- Surface -------------------------------------------------------------------------------


class SurfaceTest(unittest.TestCase):
    def test_base_tree_compiles_back_to_an_equivalent_workflow(self):
        tree = S.base_tree(WORKFLOW)
        self.assertIn("workflow.json", tree)
        self.assertIn("prompts/researcher.md", tree)
        compiled = S.compile_tree(tree)
        original = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual([n["id"] for n in compiled["nodes"]], [n["id"] for n in original["nodes"]])
        researcher = next(n for n in compiled["nodes"] if n["id"] == "researcher")
        self.assertIn("Researcher", researcher["system_prompt"])

    def test_validation_names_every_problem_at_once(self):
        tree = S.base_tree(WORKFLOW)
        bad = dict(tree)
        wf = json.loads(bad["workflow.json"])
        next(n for n in wf["nodes"] if n["id"] == "researcher")["tools"].append("not_a_tool")
        wf["config"]["sandbox"]["type"] = "gce"
        bad["workflow.json"] = json.dumps(wf)
        bad["prompts/ghost.md"] = "for a node that does not exist"
        bad["prompts/critic.md"] = "x" * (S.MAX_NODE_PROMPT_CHARS + 1)
        with self.assertRaises(ValueError) as cm:
            S.validate_tree(bad, tree)
        msg = str(cm.exception)
        for needle in ("not_a_tool", "sandbox.type", "ghost.md", "critic", "cap"):
            self.assertIn(needle, msg)

    def test_surface_admits_only_the_three_file_kinds(self):
        wf = json.loads(WORKFLOW.read_text())
        self.assertTrue(S.on_surface("workflow.json", wf))
        self.assertTrue(S.on_surface("prompts/researcher.md", wf))
        self.assertTrue(S.on_surface("tool_guidance.md", wf))
        # Upstream's regex admits scripts/*.py for every surface; ours does not, and
        # validate_tree rejects a tree that carries one.
        self.assertFalse(S.on_surface("scripts/evil.py", wf))
        self.assertFalse(S.on_surface("../workflow.json", wf))
        self.assertFalse(S.on_surface("prompts/not_a_node.md", wf))
        self.assertFalse(S.on_surface("prompts/nested/researcher.md", wf))
        tree = S.base_tree(WORKFLOW)
        bad = dict(tree); bad["scripts/evil.py"] = "print(1)"
        with self.assertRaises(ValueError) as cm:
            S.validate_tree(bad, tree)
        self.assertIn("not on the editable surface", str(cm.exception))

    def test_a_proposal_may_add_a_stage_inline_and_remove_one(self):
        tree = S.base_tree(WORKFLOW)
        wf = json.loads(tree["workflow.json"])
        # Add a second researcher arm with an inline instruction, wired threat_modeler -> arm -> deduplicator.
        researcher = next(n for n in wf["nodes"] if n["id"] == "researcher")
        arm = json.loads(json.dumps(researcher)); arm["id"] = "researcher_domain"
        arm["system_prompt"] = "You are the domain researcher arm. Audit auth and trust boundaries; report via report_findings."
        wf["nodes"].append(arm)
        in_edges = [e for e in wf["edges"] if e["to"] == "researcher"]
        out_edges = [e for e in wf["edges"] if e["from"] == "researcher"]
        for e in in_edges:
            wf["edges"].append({**e, "to": "researcher_domain"})
        for e in out_edges:
            wf["edges"].append({**e, "from": "researcher_domain"})
        added = dict(tree); added["workflow.json"] = json.dumps(wf, indent=2)
        census = S.validate_tree(added, tree)
        self.assertEqual(census["agent_nodes"], len([n for n in wf["nodes"] if n.get("type", "agent") == "agent"]))
        compiled = S.compile_tree(added)
        self.assertIn("domain researcher arm", next(n for n in compiled["nodes"] if n["id"] == "researcher_domain")["system_prompt"])
        # A prompt FILE for the new stage is the wrong place, and the message says where the right one is.
        wrong = dict(added); wrong["prompts/researcher_domain.md"] = "domain arm prompt"
        with self.assertRaises(ValueError) as cm:
            S.validate_tree(wrong, tree)
        self.assertIn("system_prompt", str(cm.exception))
        self.assertFalse(S.on_surface("prompts/researcher_domain.md", json.loads(tree["workflow.json"])))
        # A new stage with no instruction at all is refused.
        mute = json.loads(json.dumps(wf)); next(n for n in mute["nodes"] if n["id"] == "researcher_domain").pop("system_prompt")
        next(n for n in mute["nodes"] if n["id"] == "researcher_domain").pop("skill", None)
        mute_tree = dict(tree); mute_tree["workflow.json"] = json.dumps(mute)
        with self.assertRaises(ValueError) as cm:
            S.validate_tree(mute_tree, tree)
        self.assertIn("no instruction", str(cm.exception))

        # Remove a stage: its (inherited, or upstream-stubbed) prompt file is ignored, not an error.
        # The removal must rewire the route that fed it; the loader has the final word on that.
        wf2 = json.loads(tree["workflow.json"])
        wf2["nodes"] = [n for n in wf2["nodes"] if n["id"] != "chainer"]
        wf2["edges"] = [e for e in wf2["edges"] if "chainer" not in (e["from"], e["to"])]
        wf2["edges"].append({"from": "repro_classifier", "to": "patcher", "on": "success"})
        removed = dict(tree); removed["workflow.json"] = json.dumps(wf2)
        S.validate_tree(removed, tree)                       # inherited prompts/chainer.md present
        removed["prompts/chainer.md"] = ""                   # upstream write_tree stubs always-on files as ""
        S.validate_tree(removed, tree)
        self.assertNotIn("chainer", {n["id"] for n in S.compile_tree(removed)["nodes"]})
        # Removing it without rewiring is refused by the production loader, with the orphaned route named.
        dangling = json.loads(tree["workflow.json"])
        dangling["nodes"] = [n for n in dangling["nodes"] if n["id"] != "chainer"]
        dangling["edges"] = [e for e in dangling["edges"] if "chainer" not in (e["from"], e["to"])]
        with self.assertRaises(ValueError) as cm:
            S.validate_tree({**tree, "workflow.json": json.dumps(dangling)}, tree)
        self.assertIn("repro_classifier", str(cm.exception))

    def test_registered_surface_text_renders_all_prompt_families(self):
        surface_text.register()
        from harness2.core import prompts as P
        from harness2.core.spec import BenchPrompts
        bp = BenchPrompts(work="audit a codebase for real vulnerabilities", domain_word="codebase")
        for text in (P.propose(bp, "INV", "mantis", k=1, index=1), P.propose(bp, "DOM", "mantis", k=1, index=1),
                     P.aggregate(bp, "mantis", k=1, index=1)):
            self.assertIn("workflow.json", text)
            self.assertIn("KEEP_BASE", text)
            self.assertIn("THE TOOL SET IS CLOSED", text)


# --- Adapter -------------------------------------------------------------------------------


class AdapterTest(unittest.TestCase):
    def test_stdout_parses_into_events_and_renders_with_truncation(self):
        events = ad.parse_stdout_to_events(
            "-- wf@1/researcher@1\nthinking...\n[TOOL CALL: read_file] args={'filepath': 'a.py'}\n"
            "[TOOL RESPONSE: read_file] " + "x" * 50 + "\n[EVENT ERROR BadRequestError] boom\n-- wf@1/reviewer@1 -> confirmed\n")
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds, ["node", "message", "tool_call", "tool_response", "error", "node"])
        self.assertEqual(events[2]["node"], "researcher")
        text = ad.render_trajectory(events, obs_trunc=10_000)
        self.assertIn("[ACTION 3] (researcher) read_file", text)
        self.assertIn("[ERROR 5]", text)
        self.assertIn("[MANTIS TRUNCATION:", ad.render_trajectory(events, obs_trunc=20))

    def test_structured_transcript_is_preferred_over_the_clipped_console(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            target = tmp / "repo"; target.mkdir(); (target / "app.py").write_text("print(1)\n")
            adapter = _adapter(tmp, target)
            rd = adapter.execute("demo", None, "demo--base1")
            summary = json.loads((rd / "summary.json").read_text())
            self.assertEqual(summary["trajectory_source"], "transcript")
            text = adapter.trajectory(rd, obs_trunc=50_000)
            # One observation holding the whole file, not an empty observation followed by file lines as prose.
            self.assertEqual(text.count("[OBSERVATION"), 2)
            head, _, tail = text.partition("[OBSERVATION 2]")
            self.assertNotIn("pass", head)
            self.assertGreaterEqual(tail.split("[MESSAGE")[0].count("    pass\n"), 300)
            messages = [blk.strip() for blk in text.split("\n\n") if blk.strip().startswith("[MESSAGE")]
            self.assertEqual(len(messages), 1)
            self.assertIn("I will report the query flaw.", messages[0])
            self.assertIn("[ERROR 6] (reviewer) RateLimitError", text)
            cmd = ad.build_command(target, rd / "workflow.json", rd, ad.RunSettings())
            self.assertIn("--transcript-out", cmd)
            self.assertIn(str(rd / "transcript.raw.jsonl"), cmd)

    def test_console_fallback_is_labelled_as_such(self):
        def no_transcript(target, workflow_path, run_dir, settings):
            out = fake_pipeline(target, workflow_path, run_dir, settings)
            (run_dir / "transcript.raw.jsonl").unlink()
            return out
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            target = tmp / "repo"; target.mkdir(); (target / "app.py").write_text("print(1)\n")
            adapter = ad.MantisRepoAdapter(targets={"demo": target}, workflow_path=WORKFLOW, runs_root=tmp / "runs",
                                           settings=ad.RunSettings(model="m"), runner=no_transcript, domain="demo")
            rd = adapter.execute("demo", None, "demo--base1")
            self.assertEqual(json.loads((rd / "summary.json").read_text())["trajectory_source"], "stdout")

    def test_a_failed_rollout_is_described_not_just_declared(self):
        def killed(target, workflow_path, run_dir, settings):
            fake_pipeline(target, workflow_path, run_dir, settings)
            return {"exit_code": -1, "wall_seconds": float(settings.timeout_seconds), "timed_out": True}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            target = tmp / "repo"; target.mkdir(); (target / "app.py").write_text("print(1)\n")
            adapter = ad.MantisRepoAdapter(targets={"demo": target}, workflow_path=WORKFLOW, runs_root=tmp / "runs",
                                           settings=ad.RunSettings(model="m", timeout_seconds=120), runner=killed, domain="demo")
            rd = adapter.execute("demo", None, "demo--base1")
            self.assertFalse(adapter.delivered(rd))
            text = adapter.describe_outcome(rd)
            self.assertIn("timed out after 120s (--rollout-timeout)", text)
            self.assertIn("at stage reviewer", text, "the last stage the transcript saw")
            self.assertIn("finding(s) were in the DB", text)
            self.assertIn("never ran", adapter.describe_outcome(tmp / "nowhere"))
        # The default is a safety net under the token budget, not a budget of its own.
        self.assertGreaterEqual(ad.DEFAULT_ROLLOUT_TIMEOUT_SECONDS, 2 * 3600)
        self.assertEqual(ad.RunSettings().timeout_seconds, ad.DEFAULT_ROLLOUT_TIMEOUT_SECONDS)

    def test_evidence_vector_weights_tiers_and_tracks_verified_keys(self):
        ev = ad.evidence_vector([
            {"status": "dynamic_confirmed", "severity": "high", "filepath": "a.py", "title": "Cmd injection"},
            {"status": "static_confirmed", "severity": "medium", "filepath": "b.py", "title": "SQLi"},
            {"status": "false_positive", "severity": "low", "filepath": "c.py", "title": "nope"},
        ])
        self.assertEqual((ev["live"], ev["dismissed"], ev["verified"]), (2, 1, 1))
        self.assertEqual(ev["verified_keys"], ["a.py::cmd injection"])
        self.assertEqual(ev["confirmed_keys"], ["a.py::cmd injection", "b.py::sqli"])
        self.assertGreater(ev["score"], 3.0)
        # Keys compare across independent rollouts: file + CWE, not a per-DB lineage or a paraphrased title.
        k1 = ad._finding_key({"filepath": "dal/users.py", "cwe": "CWE-89", "title": "SQL injection in find_user", "lineage_id": "L1"})
        k2 = ad._finding_key({"filepath": "dal/users.py", "cwe": "cwe-89", "title": "Unsanitised query string", "lineage_id": "L2"})
        self.assertEqual(k1, k2)
        self.assertEqual(k1, "dal/users.py#cwe89")

    def test_execute_is_idempotent_by_runmark_and_leaves_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            target = tmp / "repo"; target.mkdir(); (target / "app.py").write_text("print(1)\n")
            adapter = _adapter(tmp, target)
            rd = adapter.execute("demo", None, "demo--base1")
            for name in ("summary.json", "findings.json", "evidence.json", "findings.md", "trajectory.jsonl", ".adapt_complete"):
                self.assertTrue((rd / name).exists(), name)
            self.assertTrue(adapter.delivered(rd))
            self.assertIn("[ACTION", adapter.trajectory(rd, obs_trunc=10_000))
            self.assertIn("static_confirmed", adapter.deliverable_text(rd))
            stamp = (rd / "summary.json").stat().st_mtime_ns
            rd2 = adapter.execute("demo", None, "demo--base1")
            self.assertEqual(rd, rd2)
            self.assertEqual((rd / "summary.json").stat().st_mtime_ns, stamp, "a complete run must not be re-rolled")
            self.assertIsNone(adapter.score(rd, "demo"))

    def test_eval_adapter_refuses_holdout_and_scores_only_through_score(self):
        from evals.research_targets import gen
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(gen.HoldoutViolation):
                ad.MantisEvalAdapter(seeds=[gen.HOLDOUT_SEEDS[0]], workflow_path=WORKFLOW, runs_root=Path(tmp) / "r", runner=fake_pipeline)
            adapter = ad.MantisEvalAdapter(seeds=[gen.DEV_SEEDS[0]], size="small", workflow_path=WORKFLOW,
                                           runs_root=Path(tmp) / "r", runner=fake_pipeline)
            jail = adapter.target_for("seed-1000")
            self.assertTrue((jail / "app.py").exists())
            self.assertFalse((jail / "ground_truth.json").exists(), "the answer key must never reach the pipeline")
            rd = adapter.execute("seed-1000", None, "seed-1000--base1")
            scored = adapter.score(rd, "seed-1000")
            self.assertIsNotNone(scored)
            self.assertGreater(scored["n_criteria"], 0)


# --- Proposer ------------------------------------------------------------------------------


class ProposerTest(unittest.TestCase):
    def test_tools_are_jailed_and_surface_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "context"
            (root / "harness" / "prompts").mkdir(parents=True)
            (root / "harness" / "prompts" / "researcher.md").write_text("a\nb\nc\n")
            touched = []
            ls, rf, wf, df = make_context_tools(root, lambda rel: rel.startswith("prompts/") or rel == "workflow.json", touched)
            self.assertIn("harness/prompts/researcher.md", ls(""))
            self.assertEqual(rf("harness/prompts/researcher.md", 2, 3), "2: b\n3: c")
            self.assertIn("refused", rf("../../etc/passwd"))
            self.assertIn("refused", wf("scripts/x.py", "print(1)"))
            self.assertIn("wrote", wf("prompts/researcher.md", "new"))
            self.assertIn("refused", df("workflow.json"))
            self.assertEqual(touched, ["prompts/researcher.md"])

    def test_thought_parts_never_reach_the_reply_and_partials_are_skipped(self):
        import types as _t
        from evals.evolve.proposer import collect_reply_text
        mk = lambda **kw: _t.SimpleNamespace(**{"text": None, "function_call": None, "thought": False, **kw})
        ev = lambda parts, partial=False, tokens=0: _t.SimpleNamespace(
            content=_t.SimpleNamespace(parts=parts), partial=partial,
            usage_metadata=_t.SimpleNamespace(total_token_count=tokens) if tokens else None)
        out = collect_reply_text([
            ev([mk(text="let me think about the candidates...", thought=True)], tokens=10),
            ev([mk(function_call=_t.SimpleNamespace(name="read_file"))]),
            ev([mk(text="<pick>c_al")], partial=True),
            ev([mk(text="<why>plainer</why>\n```choice\nA\n```")], tokens=20),
            ev([mk(text="trailing scratchpad", thought=True)]),
        ])
        self.assertEqual(out["text"], "<why>plainer</why>\n```choice\nA\n```")
        self.assertEqual((out["steps"], out["tokens"]), (1, 30))

    def test_selector_and_reflector_sessions_are_read_only(self):
        load_harness2()
        from harness2.core.spec import ChannelSpec
        seen = {}

        def recording_agent(*, system, read_only, context_root, **kw):
            seen["read_only"], seen["system"] = read_only, system
            tools = make_context_tools(context_root, lambda rel: True, [], read_only=read_only)
            seen["tools"] = [t.__name__ for t in tools]
            return {"text": "<why>x</why>\n```choice\nA\n```"}

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Path(tmp) / "ctx"; (ctx / "harness").mkdir(parents=True)
            prop = MantisProposer(ChannelSpec("opencode", "m"), agent_runner=recording_agent)
            prop.call("choose", ctx, "select.agg1")                       # no harvest -> read-only
            self.assertTrue(seen["read_only"])
            self.assertEqual(seen["tools"], ["list_files", "read_file"])
            self.assertNotIn("edit files", seen["system"])
            prop.call("propose", ctx, "inv1.a1", harvest="context/harness")  # harvest -> editing
            self.assertFalse(seen["read_only"])
            self.assertIn("write_harness_file", seen["tools"])

    def test_call_harvests_the_edited_harness_and_audits(self):
        load_harness2()
        from harness2.core.proposer import ProposerError
        from harness2.core.spec import ChannelSpec
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Path(tmp) / "ctx"; (ctx / "harness" / "prompts").mkdir(parents=True)
            (ctx / "harness" / "prompts" / "researcher.md").write_text("base\n")
            prop = MantisProposer(ChannelSpec("opencode", "m"), audit_dir=Path(tmp) / "audit", agent_runner=scripted_agent)
            reply = prop.call("mission", ctx, "inv1.a1", harvest="context/harness")
            self.assertTrue(reply.ok)
            self.assertIn("<notes>", reply.text)
            self.assertIn(EDIT_MARKER, (reply.workspace / "prompts" / "researcher.md").read_text())
            self.assertEqual("base\n", (ctx / "harness" / "prompts" / "researcher.md").read_text(), "the original context is untouched")
            self.assertTrue((Path(tmp) / "audit" / "sessions" / "inv1.a1" / "response.md").exists())

            empty = MantisProposer(ChannelSpec("opencode", "m"), agent_runner=lambda **kw: {"text": ""})
            with self.assertRaises(ProposerError):
                empty.call("mission", ctx, "inv1.a2", max_attempts=1)

    def test_a_session_is_always_wall_clock_bounded(self):
        """Upstream calls without a timeout; a hung connection must not hang the loop."""
        load_harness2()
        from harness2.core.proposer import ProposerError
        from harness2.core.spec import ChannelSpec
        import evals.evolve.proposer as P
        seen = []

        def recording_agent(*, timeout, **kw):
            seen.append(timeout)
            return {"text": "<notes>ok</notes>"}

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Path(tmp) / "ctx"; (ctx / "harness").mkdir(parents=True)
            MantisProposer(ChannelSpec("opencode", "m"), agent_runner=recording_agent).call("m", ctx, "inv1.a1")
            self.assertEqual(seen[-1], 3600, "the channel's own timeout when the caller gives none")
            MantisProposer(ChannelSpec("opencode", "m", timeout=0), agent_runner=recording_agent).call("m", ctx, "inv1.a1")
            self.assertEqual(seen[-1], P.DEFAULT_SESSION_TIMEOUT_SECONDS)
            MantisProposer(ChannelSpec("opencode", "m"), agent_runner=recording_agent).call("m", ctx, "inv1.a1", timeout=42)
            self.assertEqual(seen[-1], 42, "an explicit timeout is respected")

            # The real session runner enforces it: a runner that never returns is cut off,
            # and the call reports a ProposerError instead of sleeping forever.
            def hanging(*, mission, system, context_root, touched, label, timeout, read_only=False):
                import asyncio
                async def forever():
                    await asyncio.sleep(3600)
                return asyncio.run(asyncio.wait_for(forever(), timeout=timeout))
            prop = MantisProposer(ChannelSpec("opencode", "m"), agent_runner=hanging)
            with self.assertRaises(ProposerError) as cm:
                prop.call("m", ctx, "inv1.a1", timeout=0.05, max_attempts=1)
            self.assertIn("TimeoutError", str(cm.exception))

    def test_the_proposer_model_has_a_request_timeout(self):
        import evals.evolve.proposer as P
        from core.config import get_llm_kwargs
        _, kw = get_llm_kwargs(model_id="openai/m1", default_timeout=P.PROPOSER_LLM_TIMEOUT_SECONDS)
        self.assertEqual(kw.get("timeout"), P.PROPOSER_LLM_TIMEOUT_SECONDS)


# --- Selection -----------------------------------------------------------------------------


class SelectionTest(unittest.TestCase):
    class _RV:
        def __init__(self, label, run_dir, ok=True, delivered=True, error=""):
            self.label, self.run_dir, self.ok, self.delivered, self.error, self.run_id = label, run_dir, ok, delivered, error, label

    class _Adapter:
        def __init__(self, evidence):
            self._ev = evidence

        def evidence(self, run_dir):
            return self._ev.get(Path(run_dir).name, {})

    def _dir(self, tmp, name, tokens):
        d = Path(tmp) / name; d.mkdir()
        (d / "summary.json").write_text(json.dumps({"tokens": tokens}))
        return d

    def test_non_regression_then_evidence_then_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._RV("base1", self._dir(tmp, "base1", 500))
            a = self._RV("agg1", self._dir(tmp, "agg1", 400))
            b = self._RV("agg2", self._dir(tmp, "agg2", 300))
            c = self._RV("agg3", self._dir(tmp, "agg3", 100), ok=False, error="died")
            adapter = self._Adapter({
                "base1": {"score": 1.0, "verified": 1, "verified_keys": ["x::k"], "confirmed_keys": ["x::k"], "live": 1, "dismissed": 0},
                "agg1": {"score": 4.0, "verified": 2, "verified_keys": ["x::k", "y::z"], "confirmed_keys": ["x::k", "y::z"], "live": 2, "dismissed": 0},
                "agg2": {"score": 6.0, "verified": 1, "verified_keys": ["y::z"], "confirmed_keys": ["y::z"], "live": 3, "dismissed": 0},   # lost x::k
            })
            table = ev_sel.rank_candidates(adapter, {"agg1": a, "agg2": b, "agg3": c}, [base])
            self.assertEqual(table["base_verified_floor"], ["x::k"])
            self.assertEqual(table["eligible"], ["agg1"])
            self.assertEqual(table["ranked"][1]["regressed"], ["x::k"])
            self.assertIn("agg3", table["dropped"])
            self.assertEqual(table["tie"], [])

    def test_static_only_regression_is_caught_by_the_score_floor(self):
        """On the shipped static-only workflow nothing reaches dynamic_confirmed, so the
        machine-verified floor is empty; a candidate that found nothing must still lose."""
        with tempfile.TemporaryDirectory() as tmp:
            base = self._RV("base1", self._dir(tmp, "base1", 500))
            a = self._RV("agg1", self._dir(tmp, "agg1", 100))
            adapter = self._Adapter({
                "base1": {"score": 7.5, "verified": 0, "verified_keys": [], "confirmed": 5,
                          "confirmed_keys": ["a.py#cwe89", "b.py#cwe78", "c.py#cwe22", "d.py#cwe918", "e.py#cwe798"], "live": 5, "dismissed": 0},
                "agg1": {"score": 0.0, "verified": 0, "verified_keys": [], "confirmed": 0, "confirmed_keys": [], "live": 0, "dismissed": 0},
            })
            table = ev_sel.rank_candidates(adapter, {"agg1": a}, [base])
            self.assertEqual(table["base_floor"]["score"], 7.5)
            self.assertEqual(len(table["base_floor"]["keys"]), 5)
            self.assertEqual(table["eligible"], [])
            self.assertTrue(table["ranked"][0]["below_base_score"])
            self.assertEqual(len(table["ranked"][0]["regressed"]), 5)
            sel = ev_sel.select("t", 1, "agg", {"agg1": a}, [base], adapter=adapter, selector=None, work_dir=Path(tmp) / "w")
            self.assertFalse(sel.ok)
            self.assertIn("below the weakest base run", sel.error)

    def test_tie_is_detected_only_within_epsilon(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._RV("base1", self._dir(tmp, "base1", 500))
            a = self._RV("agg1", self._dir(tmp, "agg1", 400))
            b = self._RV("agg2", self._dir(tmp, "agg2", 300))
            adapter = self._Adapter({"base1": {"score": 1.0, "verified_keys": [], "confirmed_keys": []},
                                     "agg1": {"score": 3.0, "verified_keys": [], "confirmed_keys": []},
                                     "agg2": {"score": 3.2, "verified_keys": [], "confirmed_keys": []}})
            table = ev_sel.rank_candidates(adapter, {"agg1": a, "agg2": b}, [base])
            self.assertEqual(sorted(table["tie"]), ["agg1", "agg2"])
            self.assertEqual(table["eligible"][0], "agg2")  # higher score first, before the tie-break


# --- The loop ------------------------------------------------------------------------------


class EvolveIterationTest(unittest.TestCase):
    def test_one_iteration_end_to_end_with_scripted_proposer(self):
        load_harness2()
        from harness2.core import library as h2lib
        from harness2.core.spec import ChannelSpec
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            target = tmp / "repo"; target.mkdir(); (target / "app.py").write_text("print(1)\n")
            adapter = _adapter(tmp, target)
            prop = MantisProposer(ChannelSpec("opencode", "m"), audit_dir=tmp / "audit", agent_runner=scripted_agent,
                                  surface_ok=lambda rel: ev_run._on_surface(adapter, rel))
            library = h2lib.Library(root=tmp / "library", domain="demo", surface=adapter.substrate.surface())
            record = ev_run.evolve_task("demo", adapter=adapter, proposer=prop, selector=prop, work_dir=tmp / "task",
                                        k=1, mode="agg", library=library, log=lambda s: None)
            self.assertEqual(record["status"], "ok", json.dumps(record.get("selection"), indent=1, default=str))
            self.assertEqual(record["chosen_label"], "agg1")
            self.assertFalse(record["kept_base"])
            self.assertEqual(record["selection"]["ok"], True)
            self.assertIn("machine-verified", record["selection"]["why"])
            # Budget identity: 4k rollouts, 3k sessions, no duplicate labels.
            self.assertEqual((record["ledger"]["R"], record["ledger"]["P"]), (4, 3))
            # The adopted tree carries the edit and the diff shows it.
            self.assertEqual(record["diff"]["modified"], ["prompts/researcher.md"])
            self.assertIn(EDIT_MARKER, record["unified_diff"])
            self.assertIn(EDIT_MARKER, (Path(record["adopted_dir"]) / "prompts" / "researcher.md").read_text())
            # Reflection produced an evidence-backed experience and the library kept it.
            self.assertTrue(record["experience"].get("evidence"))
            self.assertEqual(library.entries(), ["demo"])
            self.assertTrue((library.dir / "INDEX.md").exists())
            # The production workflow was not touched.
            self.assertNotIn(EDIT_MARKER, WORKFLOW.read_text(encoding="utf-8"))
            # Adoption is a dry run without --yes, and validates before writing.
            scratch_wf = tmp / "workflow.json"; scratch_wf.write_text(WORKFLOW.read_text(encoding="utf-8"))
            self.assertFalse(ev_run.adopt(Path(record["adopted_dir"]), scratch_wf, yes=False, log=lambda s: None))
            self.assertNotIn(EDIT_MARKER, scratch_wf.read_text())
            self.assertTrue(ev_run.adopt(Path(record["adopted_dir"]), scratch_wf, yes=True, log=lambda s: None))
            self.assertIn(EDIT_MARKER, scratch_wf.read_text())
            self.assertTrue(scratch_wf.with_suffix(".json.bak").exists())

    def test_keep_base_is_a_real_answer(self):
        load_harness2()
        from harness2.core.spec import ChannelSpec

        def keep_base_agent(*, label, **kw):
            if label.startswith("experience."):
                return {"text": "NO_ENTRY"}
            return {"text": "<notes>KEEP_BASE: the trajectories show no divergence worth a change.</notes>"}

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            target = tmp / "repo"; target.mkdir(); (target / "app.py").write_text("print(1)\n")
            adapter = _adapter(tmp, target)
            prop = MantisProposer(ChannelSpec("opencode", "m"), agent_runner=keep_base_agent)
            record = ev_run.evolve_task("demo", adapter=adapter, proposer=prop, selector=prop, work_dir=tmp / "task",
                                        k=1, mode="agg", library=None, log=lambda s: None)
            self.assertEqual(record["status"], "ok")
            self.assertTrue(record["kept_base"])
            self.assertTrue(all(s["kept_base"] for s in record["stages"]))
            self.assertNotIn("adopted_dir", record)
            self.assertTrue(any("KEEP_BASE" in w for w in record["warnings"]))


if __name__ == "__main__":
    unittest.main()
