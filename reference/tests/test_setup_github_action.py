"""scripts/setup_github_action.sh: the Setup section as one idempotent command.

Run against stub `gcloud` and `gh` binaries that record what they were asked
and answer as the real ones would. What is checked is the plan: which
resources are created on a first run, that a second run creates nothing and
converges the provider condition and the role, that an existing environment
is never replaced, that the environment is created protected, that the budget
is the project's and a second repository finds it, that a dry run touches
nothing, and that inputs which could not be safely embedded in a gcloud
argument, a CEL string or a JSON body are refused before anything is called.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REF_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = _REF_ROOT / "scripts" / "setup_github_action.sh"
SEP = "\x1f"

_STUB = r'''#!/usr/bin/env bash
# Records every call; answers like the real tool for the reads the script makes.
printf '%s\n' "$(IFS=$'\x1f'; echo "$(basename "$0")$IFS$*")" >> "$STUB_LOG"
if [[ "$1" == "api" && " $* " == *" --input "* ]]; then
  # The body file is gone by the time a test looks; keep its content.
  for ((i=1;i<=$#;i++)); do [[ "${!i}" == "--input" ]] && { j=$((i+1)); cat "${!j}" >> "$STUB_LOG.bodies"; echo >> "$STUB_LOG.bodies"; }; done
fi
case "$(basename "$0") $*" in
  "gcloud auth list"*) echo "someone@example.com" ;;
  "gcloud projects describe"*) echo "123456789" ;;
  "gcloud billing projects describe"*) echo "billingAccounts/ABC-123" ;;
  "gcloud billing budgets list"*) [[ "${STUB_EXISTS:-0}" == 1 ]] && echo "billingAccounts/ABC-123/budgets/b1" ;;
  "gcloud iam workload-identity-pools describe"*"--format=value(state)"*) echo "ACTIVE" ;;
  "gcloud iam roles describe"*) [[ "${STUB_EXISTS:-0}" == 1 ]] || exit 1 ;;
  *" describe "*) [[ "${STUB_EXISTS:-0}" == 1 ]] || exit 1 ;;
  "gh auth status"*) ;;
  "gh api repos/"*"/environments/"*" --jq "*) [[ "${STUB_ENV_EXISTS:-0}" == 1 ]] || exit 1
     case "$*" in *".protection_rules | length"*) echo "${STUB_ENV_RULES:-1}" ;; *protection_rules*) echo "${STUB_ENV_REVIEWERS:-2}" ;; *deployment_branch_policy*) echo "${STUB_ENV_POLICY:-any branch}" ;; esac ;;
  "gh api repos/"*"/environments/"*) [[ " $* " == *" --method "* ]] || { [[ "${STUB_ENV_EXISTS:-0}" == 1 ]] || exit 1; } ;;
  "gh api repos/"*" --jq .id, .owner.id, .full_name"*) printf '4242\n77\n%s\n' "${STUB_CANONICAL:-acme/widgets}" ;;
  "gh api orgs/"*"/teams/"*" --jq .id"*) echo "555" ;;
  "gh api users/"*" --jq .id"*) echo "666" ;;
esac
exit 0
'''


class SetupScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mantis_setup_gha_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name in ("gcloud", "gh"):
            p = self.bin / name
            p.write_text(_STUB)
            p.chmod(0o755)
        self.log = self.tmp / "calls.log"

    def run_script(self, *args, exists=False, env_exists=False, **stub_env):
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ.get('PATH', '')}",
                   STUB_LOG=str(self.log), STUB_EXISTS="1" if exists else "0",
                   STUB_ENV_EXISTS="1" if env_exists else "0")
        env.update({k: str(v) for k, v in stub_env.items()})
        proc = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                              timeout=120, env=env)
        return proc.returncode, proc.stdout + proc.stderr

    def calls(self):
        if not self.log.exists():
            return []
        return [line.split(SEP) for line in self.log.read_text().splitlines()]

    def calls_matching(self, *prefix):
        return [c for c in self.calls() if c[: len(prefix)] == list(prefix)]

    def bodies(self):
        p = Path(str(self.log) + ".bodies")
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []

    # --- first run ------------------------------------------------------------------

    def test_first_run_creates_everything_pinned_and_protected(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--budget-usd", "50", "--github",
                                    "--reviewer-team", "security", "--reviewer-user", "alice")
        self.assertEqual(code, 0, out)
        sa = "mantis-pr-scan@my-mantis-scan.iam.gserviceaccount.com"
        condition = ("assertion.repository_id == '4242' && assertion.repository_owner_id == '77' && "
                     "assertion.repository == 'acme/widgets' && assertion.environment == 'mantis-pr-scan'")
        principal = ("principal://iam.googleapis.com/projects/123456789/locations/global/"
                     "workloadIdentityPools/github/subject/repo:acme/widgets:environment:mantis-pr-scan")
        provider = "projects/123456789/locations/global/workloadIdentityPools/github/providers/mantis"

        enable = self.calls_matching("gcloud", "services", "enable")
        self.assertEqual(len(enable), 1)
        # The least-privilege role, created with exactly the permissions the pipeline needs.
        role = self.calls_matching("gcloud", "iam", "roles", "create", "mantisPrScan")
        self.assertEqual(len(role), 1)
        perms = next(a for a in role[0] if a.startswith("--permissions=")).split("=", 1)[1].split(",")
        self.assertIn("aiplatform.endpoints.predict", perms)
        self.assertIn("monitoring.timeSeries.list", perms)
        self.assertTrue(all(p.startswith(("aiplatform.endpoints.predict", "aiplatform.cachedContents.",
                                          "monitoring.timeSeries.list")) for p in perms), perms)
        self.assertEqual(self.calls_matching("gcloud", "iam", "roles", "update"), [])
        self.assertEqual(len(self.calls_matching("gcloud", "iam", "service-accounts", "create", "mantis-pr-scan")), 1)
        roles = [a for c in self.calls_matching("gcloud", "projects", "add-iam-policy-binding", "my-mantis-scan")
                 for a in c if a.startswith("--role=")]
        self.assertEqual(roles, ["--role=projects/my-mantis-scan/roles/mantisPrScan"],
                         "no predefined create-capable role, no monitoring.viewer")
        self.assertEqual(len(self.calls_matching("gcloud", "iam", "workload-identity-pools", "create", "github")), 1)
        create = self.calls_matching("gcloud", "iam", "workload-identity-pools", "providers", "create-oidc", "mantis")
        self.assertEqual(len(create), 1)
        self.assertIn(f"--attribute-condition={condition}", create[0])
        self.assertIn("--issuer-uri=https://token.actions.githubusercontent.com", create[0])
        # The binding names one subject, not every token the pool maps for the repository.
        bind = self.calls_matching("gcloud", "iam", "service-accounts", "add-iam-policy-binding", sa)
        self.assertEqual(len(bind), 1)
        self.assertIn(f"--member={principal}", bind[0])
        self.assertFalse(any("principalSet://" in a for a in bind[0]))
        # Budget: one per project, named for it, so a second repository finds it.
        budget = self.calls_matching("gcloud", "billing", "budgets", "create")
        self.assertEqual(len(budget), 1)
        self.assertIn("--budget-amount=50USD", budget[0])
        self.assertIn("--display-name=Mantis PR scan (my-mantis-scan)", budget[0])
        # GitHub: the environment is created WITH reviewers, then the three values.
        puts = [c for c in self.calls_matching("gh", "api", "--method", "PUT")
                if c[4] == "repos/acme/widgets/environments/mantis-pr-scan"]
        self.assertEqual(len(puts), 1)
        body = self.bodies()[0]
        self.assertEqual(body["prevent_self_review"], True)
        self.assertEqual(sorted((r["type"], r["id"]) for r in body["reviewers"]),
                         [("Team", 555), ("User", 666)])
        secrets = {c[3]: c[c.index("--body") + 1] for c in self.calls_matching("gh", "secret", "set")}
        self.assertEqual(secrets, {"GCP_WORKLOAD_IDENTITY_PROVIDER": provider,
                                   "GCP_SERVICE_ACCOUNT": sa, "GCP_PROJECT": "my-mantis-scan"})
        for needle in (provider, sa, "GCP_PROJECT", "environment: mantis-pr-scan",
                       "disable-billing-with-notifications", "required reviewers are not optional"):
            self.assertIn(needle, out)
        self.assertNotIn("<project-number>", out)
        self.assertNotIn("WARNING:", out.replace("WARNING: detaching billing", ""))

    def test_the_canonical_repository_name_is_used(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "ACME/Widgets",
                                    "--no-budget", "--github", "--reviewer-team", "security",
                                    STUB_CANONICAL="acme/widgets")
        self.assertEqual(code, 0, out)
        create = self.calls_matching("gcloud", "iam", "workload-identity-pools", "providers", "create-oidc")
        self.assertTrue(any("assertion.repository == 'acme/widgets'" in a for a in create[0]))
        self.assertIn("canonical name 'acme/widgets'", out)

    # --- second run -----------------------------------------------------------------

    def test_a_second_run_creates_nothing_and_converges(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/gadgets",
                                    "--repo-id", "9", "--owner-id", "8", "--budget-usd", "75",
                                    exists=True)
        self.assertEqual(code, 0, out)
        creates = [c for c in self.calls() if "create" in c or "create-oidc" in c]
        self.assertEqual(creates, [], creates)
        update = self.calls_matching("gcloud", "iam", "workload-identity-pools", "providers", "update-oidc", "mantis")
        self.assertEqual(len(update), 1)
        self.assertIn("--attribute-condition=assertion.repository_id == '9' && assertion.repository_owner_id == '8' "
                      "&& assertion.repository == 'acme/gadgets' && assertion.environment == 'mantis-pr-scan'",
                      update[0])
        role = self.calls_matching("gcloud", "iam", "roles", "update", "mantisPrScan")
        self.assertEqual(len(role), 1, "the role's permissions converge on this version's list")
        budget = self.calls_matching("gcloud", "billing", "budgets", "update", "billingAccounts/ABC-123/budgets/b1")
        self.assertEqual(len(budget), 1)
        self.assertIn("--budget-amount=75USD", budget[0])
        self.assertNotIn("WARNING:", out.replace("WARNING: detaching billing", ""))

    def test_an_existing_environment_is_never_replaced(self):
        # PUT on an environment resets what it omits -- the reviewers a person
        # added. The script sets the secrets and reports the protection.
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--no-budget", "--github", "--reviewer-team", "security",
                                    exists=True, env_exists=True, STUB_ENV_REVIEWERS=2)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.calls_matching("gh", "api", "--method", "PUT"), [])
        self.assertEqual(len(self.calls_matching("gh", "secret", "set")), 3)
        self.assertIn("exists: 2 required reviewer(s)", out)
        self.assertNotIn("WARNING:", out.replace("WARNING: detaching billing", ""))
        # And an existing one with no reviewers is called out, loudly.
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--no-budget", "--github", "--reviewer-team", "security",
                                    exists=True, env_exists=True, STUB_ENV_REVIEWERS=0)
        self.assertEqual(code, 0, out)
        self.assertIn("NO required reviewers", out)
        self.assertEqual(self.calls_matching("gh", "api", "--method", "PUT"), [])

    def test_an_environment_github_created_on_its_own_is_configured_as_new(self):
        # A pull request opened before setup makes GitHub create the
        # environment with no protection at all. Nothing a PUT could lose, so
        # the reviewers go on rather than a warning that they are missing.
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--no-budget", "--github", "--reviewer-team", "security",
                                    exists=True, env_exists=True, STUB_ENV_REVIEWERS=0, STUB_ENV_RULES=0)
        self.assertEqual(code, 0, out)
        self.assertIn("configured as if new", out)
        self.assertNotIn("NO required reviewers", out)
        puts = [c for c in self.calls_matching("gh", "api", "--method", "PUT")
                if c[4] == "repos/acme/widgets/environments/mantis-pr-scan"]
        self.assertEqual(len(puts), 1)
        self.assertEqual([(r["type"], r["id"]) for r in self.bodies()[0]["reviewers"]], [("Team", 555)])
        # But one with no reviewers and some other rule, or a branch policy,
        # may have been shaped by a person: left alone, as before.
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--no-budget", "--github", "--reviewer-team", "security",
                                    exists=True, env_exists=True, STUB_ENV_REVIEWERS=0, STUB_ENV_RULES=0,
                                    STUB_ENV_POLICY="selected branches")
        self.assertEqual(code, 0, out)
        self.assertIn("NO required reviewers", out)
        self.assertEqual(len([c for c in self.calls_matching("gh", "api", "--method", "PUT")]), 1, "no second PUT")

    # --- the refresh side -----------------------------------------------------------

    def test_refresh_mode_pins_main_and_sets_a_branch_rule(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--no-budget", "--github", "--refresh")
        self.assertEqual(code, 0, out)
        create = self.calls_matching("gcloud", "iam", "workload-identity-pools", "providers", "create-oidc", "mantis-refresh")
        self.assertEqual(len(create), 1)
        cond = next(a for a in create[0] if a.startswith("--attribute-condition="))
        self.assertIn("assertion.environment == 'mantis-knowledge'", cond)
        self.assertIn("assertion.ref == 'refs/heads/main'", cond)
        bind = self.calls_matching("gcloud", "iam", "service-accounts", "add-iam-policy-binding")
        self.assertTrue(any(a.endswith("/subject/repo:acme/widgets:environment:mantis-knowledge") for a in bind[0]))
        body = self.bodies()[0]
        self.assertEqual(body["deployment_branch_policy"], {"protected_branches": False, "custom_branch_policies": True})
        self.assertNotIn("reviewers", body)
        post = self.calls_matching("gh", "api", "--method", "POST",
                                   "repos/acme/widgets/environments/mantis-knowledge/deployment-branch-policies")
        self.assertEqual(len(post), 1)
        self.assertIn("name=main", post[0])
        self.assertIn("CODEOWNERS", out)
        secrets = [c for c in self.calls_matching("gh", "secret", "set") if c[c.index("--env") + 1] == "mantis-knowledge"]
        self.assertEqual(len(secrets), 3)

    # --- refusals and warnings ------------------------------------------------------

    def test_github_without_reviewers_is_refused_unless_said_out_loud(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--no-budget", "--github")
        self.assertEqual(code, 2, out)
        self.assertIn("required reviewers", out)
        self.assertEqual(self.calls(), [])
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--no-budget", "--github", "--no-reviewers")
        self.assertEqual(code, 0, out)
        self.assertIn("WARNING", out)
        self.assertIn("NO required reviewers", out)
        body = self.bodies()[0]
        self.assertEqual(body["reviewers"], [])

    def test_a_budget_decision_is_required_on_an_unbudgeted_project(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets")
        self.assertEqual(code, 2, out)
        self.assertIn("--no-budget", out)
        # Only the reads that answer "does the project have a budget" happened.
        self.assertEqual(sorted({c[1] for c in self.calls()}), ["auth", "billing", "projects"])
        self.assertFalse(any("create" in c for c in self.calls()))
        self.assertEqual(self.calls_matching("gh"), [])

    def test_a_second_repository_on_a_budgeted_project_needs_no_budget_flag(self):
        # The budget is per project. A repository added to a project that has
        # one needs neither flag, and the budget is left as it is.
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/gadgets",
                                    "--repo-id", "9", "--owner-id", "8", exists=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.calls_matching("gcloud", "billing", "budgets", "create"), [])
        self.assertEqual(self.calls_matching("gcloud", "billing", "budgets", "update"), [])
        self.assertIn("existing budget", out)
        self.assertIn("Mantis PR scan (my-mantis-scan)", out)
        self.assertNotIn("No billing budget", out)
        self.assertNotIn("WARNING:", out)

    def test_ids_come_from_gh_without_github_too(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget")
        self.assertEqual(code, 0, out)
        create = self.calls_matching("gcloud", "iam", "workload-identity-pools", "providers", "create-oidc")
        cond = next(a for a in create[0] if a.startswith("--attribute-condition="))
        self.assertIn("assertion.repository_id == '4242' && assertion.repository_owner_id == '77'", cond)
        self.assertNotIn("WARNING", out)
        # Nothing on the GitHub side was written: the lookup is the one gh call.
        self.assertEqual([c[:2] for c in self.calls_matching("gh")], [["gh", "auth"], ["gh", "api"]])

    def test_name_only_pinning_warns(self):
        (self.bin / "gh").unlink()
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget")
        self.assertEqual(code, 0, out)
        self.assertIn("WARNING", out)
        self.assertIn("by NAME only", out)
        create = self.calls_matching("gcloud", "iam", "workload-identity-pools", "providers", "create-oidc")
        cond = next(a for a in create[0] if a.startswith("--attribute-condition="))
        self.assertNotIn("repository_id", cond)
        # A dry run without gh shows the name-only plan rather than placeholders.
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget",
                                    "--dry-run")
        self.assertEqual(code, 0, out)
        self.assertNotIn("<repo-id>", out)
        self.assertIn("by NAME only", out)

    def test_a_role_override_keeps_the_meter_readable(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget",
                                    "--repo-id", "1", "--owner-id", "2", "--role", "roles/aiplatform.user")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.calls_matching("gcloud", "iam", "roles"), [])
        roles = sorted(a for c in self.calls_matching("gcloud", "projects", "add-iam-policy-binding")
                       for a in c if a.startswith("--role="))
        self.assertEqual(roles, ["--role=roles/aiplatform.user", "--role=roles/monitoring.viewer"])

    def test_dry_run_calls_nothing_and_shows_the_plan(self):
        code, out = self.run_script("--project", "my-mantis-scan", "--repo", "acme/widgets",
                                    "--environment", "prod-scan", "--budget-usd", "10",
                                    "--github", "--reviewer-team", "security", "--dry-run")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.calls(), [], "a dry run consults nothing")
        self.assertIn("DRY RUN", out)
        self.assertIn("providers create-oidc mantis", out)
        self.assertIn("assertion.environment\\ ==\\ \\'prod-scan\\'", out)
        self.assertIn("gh secret set GCP_PROJECT", out)
        self.assertIn("environment: prod-scan", out)
        self.assertIn("<project-number>", out)
        self.assertIn('"prevent_self_review":true', out)

    def test_inputs_that_cannot_be_embedded_are_refused_before_any_call(self):
        bad = [
            ("--project", "my-mantis-scan", "--no-budget"),                           # no repo
            ("--repo", "acme/widgets", "--no-budget"),                                # no project
            ("--project", "My_Project", "--repo", "acme/widgets", "--no-budget"),
            ("--project", "my-mantis-scan", "--repo", "acme", "--no-budget"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets'||true", "--no-budget"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget", "--environment", "prod scan"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget", "--environment", "x' || 'a"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--budget-usd", "lots"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--budget-usd", "5", "--no-budget"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget", "--pool", "GitHub"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget", "--repo-id", "12a"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget", "--role", "owner; rm -rf /"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget", "--github",
             "--reviewer-team", "sec team"),
            ("--project", "my-mantis-scan", "--repo", "acme/widgets", "--no-budget", "--bogus"),
        ]
        for args in bad:
            code, out = self.run_script(*args)
            self.assertEqual(code, 2, (args, out))
            self.assertIn("error:", out)
            self.assertEqual(self.calls(), [], args)

    def test_help_exits_zero(self):
        code, out = self.run_script("--help")
        self.assertEqual(code, 0, out)
        for flag in ("--budget-usd", "--no-budget", "--reviewer-team", "--refresh", "--repo-id", "--role", "--dry-run"):
            self.assertIn(flag, out)


if __name__ == "__main__":
    unittest.main()
