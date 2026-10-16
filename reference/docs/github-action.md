# Mantis GitHub Action: scan the pull request

`action/` wraps the MCP server's scan tier and deterministic gate (see
[mcp.md](mcp.md)) as a GitHub Action. On every pull request it scans the changed
files with the full Mantis pipeline, runs `check_change` over the PR diff, and
reports the verdict as annotations on the diff plus a step summary.

The action is **advisory and fail-open by default**: the job passes whatever the
verdict, the findings appear as warnings on the changed lines, and a broken
scanner -- a model outage, a setup mistake, a crash -- never stops a merge. Two
inputs change that independently once you trust it on your code: `fail-on` makes
findings fail the job; `fail-closed` makes "Mantis could not look" fail it too.

## Setup

The scan runs Gemini on the Gemini Enterprise Agent Platform (the service
formerly named Vertex AI; the API, `aiplatform.googleapis.com`, and the
`vertex_ai/` model prefix litellm uses keep their old ids). The job
authenticates with workload identity federation, so the repository holds no
long-lived key. Four things to create: a Google Cloud project that can only run
the model, a spending stop on that project, a GitHub environment that holds the
project's names and gates the job, and the workflow.

### Google Cloud

One project reserved for this, with billing attached. Replace `org/repo` with
your repository and keep the environment name if you keep it in the workflow.

```bash
PROJECT=<project>
gcloud config set project "$PROJECT"
NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')

# The model API, the project's own usage meter, and what workload identity
# needs to mint and exchange tokens.
gcloud services enable aiplatform.googleapis.com monitoring.googleapis.com \
    iamcredentials.googleapis.com iam.googleapis.com sts.googleapis.com

# A service account that can call the model and read the project's meter.
gcloud iam service-accounts create mantis-pr-scan --display-name="Mantis PR scan"
SA="mantis-pr-scan@$PROJECT.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:$SA" --role="roles/aiplatform.user"
gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:$SA" --role="roles/monitoring.viewer"

# A pool with a GitHub OIDC provider, admitting one repository and one environment.
gcloud iam workload-identity-pools create github --location=global \
    --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc mantis \
    --location=global --workload-identity-pool=github \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository == 'org/repo' && assertion.environment == 'mantis-pr-scan'"

# Let that repository's jobs impersonate the service account.
gcloud iam service-accounts add-iam-policy-binding "$SA" \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/$NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/org/repo"

# The three values GitHub needs.
gcloud iam workload-identity-pools providers describe mantis \
    --location=global --workload-identity-pool=github --format='value(name)'
echo "$SA"
echo "$PROJECT"
```

### The spending stop

The action bounds what one run spends (see Spend). What the project may spend in
a month is yours to decide and yours to enforce, because the platform will not
do it for you: Gemini on the Agent Platform draws on dynamic shared quota, which
has no per-project ceiling to lower, and a budget on its own only sends mail.
Wire the budget to something that cuts the project off:

```bash
BILLING=$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)')
gcloud pubsub topics create mantis-pr-scan-budget
gcloud billing budgets create --billing-account="${BILLING#billingAccounts/}" \
    --display-name="Mantis PR scan" --budget-amount=<amount>USD \
    --filter-projects="projects/$NUMBER" \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0 \
    --notifications-rule-pubsub-topic="projects/$PROJECT/topics/mantis-pr-scan-budget"
```

Then subscribe a function to the topic that detaches billing from the project
when the message's `costAmount` reaches `budgetAmount`. Google publishes the
function and the IAM it needs under "Automated cost control responses" in the
Cloud Billing documentation; deploy it as written. Detaching billing stops every
API in the project, which is the point, and is safe only because the project
runs nothing else. Pick `<amount>` as the most you are prepared to lose to a bug
in a month, not as a forecast. Budget data lags by hours; the stop is for a
runaway that lasts a day, and the action's own limits are for the one that lasts
a run.

### GitHub

1. Create an environment named `mantis-pr-scan` (Settings, Environments) and
   **put required reviewers on it**. This is not optional. On a `pull_request`
   event GitHub runs the pull request's own copy of the workflow file, and a job
   that declares the environment receives its secrets and can mint the identity
   token the provider accepts. Without reviewers, anyone who can push a branch
   can edit the workflow to send the credential wherever they like, remove every
   budget, or run anything at all with it, and the run happens before anyone
   reads the pull request. Branch protection and CODEOWNERS gate the merge, not
   the run; a deployment-branch rule does not help either, because the run's ref
   is the merge ref. The reviewers are the control: a person approves each run,
   and before approving reads the diff of `.github/workflows/mantis-pr-scan.yml`
   (and, if you run the action from this repository with `uses: ./action`, of
   `action/` and `reference/`). Turn on "prevent self-review"; leave
   administrator bypass off unless the administrators are the reviewers. The
   provider condition above admits tokens from this environment only, so the job
   must declare it.

2. On that environment, add three secrets: `GCP_WORKLOAD_IDENTITY_PROVIDER` (the
   provider's full resource name,
   `projects/<number>/locations/global/workloadIdentityPools/github/providers/mantis`),
   `GCP_SERVICE_ACCOUNT`, and `GCP_PROJECT`. None of them is a credential;
   keeping them out of the workflow file keeps your project ids out of the
   public record and out of the job log.

3. If the organisation allows only selected actions, allow
   `google-github-actions/auth@*`; the others are under `actions/`.

4. Add the workflow. Pin every action to a full commit SHA. This repository's
   own `.github/workflows/mantis-pr-scan.yml` is this file with
   `uses: ./action`.

   ```yaml
   name: Mantis PR scan
   on:
     pull_request:
       types: [opened, synchronize, reopened, ready_for_review]
     workflow_dispatch:
       inputs:
         base-sha:
           description: Commit to diff the checked-out ref against.
           required: true
   permissions:
     contents: read
     id-token: write
   concurrency:
     group: mantis-pr-scan-${{ github.event.pull_request.number || github.run_id }}
     cancel-in-progress: true
   jobs:
     scan:
       if: >-
         github.event_name == 'workflow_dispatch' ||
         (github.event.pull_request.head.repo.full_name == github.repository &&
          !github.event.pull_request.draft &&
          github.event.pull_request.user.type != 'Bot')
       runs-on: ubuntu-latest
       timeout-minutes: 90
       environment: mantis-pr-scan
       steps:
         - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
           with:
             fetch-depth: 0
             persist-credentials: false
         - uses: google-github-actions/auth@7c6bc770dae815cd3e89ee6cdf493a5fab2cc093 # v3.0.0
           with:
             workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
             service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}
         - uses: google/mantis/action@<commit-sha>
           env:
             GOOGLE_CLOUD_PROJECT: ${{ secrets.GCP_PROJECT }}
           with:
             model: vertex_ai/gemini-3.8-flash
             base-sha: ${{ github.event.pull_request.base.sha || inputs.base-sha }}
   ```

   `id-token: write` lets the job mint the OIDC token the auth step exchanges
   and grants nothing else. `fetch-depth: 0` makes the base commit available for
   the diff; without it the action fetches that one commit on its own, which
   works on public repositories and on private ones as long as the checkout kept
   its credentials. The `if` skips drafts until they are marked ready and skips
   bot-authored pull requests; both are runs nobody reads (see Spend).

### First run

Run the workflow by hand (Actions, Mantis PR scan, Run workflow) with `base-sha`
set to any ancestor of the branch, approve it if the environment asks, and read
the result:

- The auth step fails: the provider condition does not match (the `environment`
  claim is in the token only because the job declares `environment:`; the name
  must be the same in both places), the service account binding carries the
  wrong project number, or `iamcredentials.googleapis.com` is not enabled. The
  step's error says which.
- The action exits 3 with a setup error: the preflight found no project, the
  platform rejected the credential, or the project's token meter could not be
  read for the daily budget (the service account needs
  `roles/monitoring.viewer`). The summary names what is missing. A rejected
  credential is reported this way, once, rather than as a failed scan on every
  file.
- A verdict and a summary: done. Pull requests from branches of the repository
  now trigger it on their own; forks are skipped by the `if`.

Set `VERTEXAI_LOCATION` on the step if the model is not served from `global`.
Other pipeline knobs are plain environment variables there too: `LLM_TIMEOUT`
(seconds per model call), `REASONING_EFFORT` (`low`, `medium`, `high`), and
`LLM_MAX_TOKENS`.

The `model` input takes any id the pipeline (litellm) accepts, with that
provider's credentials in `env:` on the step; the preflight asks each provider
for its own requirements. Recall differs by model and endpoint, and the verdict
reports what the model found, only that. This document covers the Agent Platform
route alone.

## Spend

The job holds a model credential while reading attacker-influenced input, on a
platform whose quota for Gemini is dynamic and shared: there is no per-project
ceiling to lower. Whatever bounds the bill therefore has to be in the action
itself, in the unit the invoice uses, and the only thing left to the platform is
the stop of last resort. Two failure modes, two different limits: one pull
request engineered to burn money, and a day on which runs do not stop coming.

### One budget per run

`max-tokens` is the hard limit on what one run can spend. The driver, not the
model, does the arithmetic:

- Every token counts at full price, whether or not the provider served it from
  cache, so the number bounds the worst case the invoice can show rather than
  the expected one.
- The budget is shared out as the run goes: each file is launched with an equal
  share of what is still unspent, so a cheap file's unspent share passes to the
  files after it and one expensive file cannot take the whole run. Each scan is
  told its share as its own ceiling and pauses when it reaches it.
- Each scan reports what it spent; a scan that reports nothing, because it was
  killed or crashed first, is charged its whole share. The budget can
  over-count, never under-count.
- A file whose share has fallen below 100,000 tokens is not launched (the
  smallest scan costs about 250,000; see Sizing). It is reported unscanned,
  which floors the verdict at REVIEW with the file named, exactly as a timeout
  would. Nothing in the pull request can raise the budget.

The step summary prints the figure for every run: tokens per file, the run total
against the budget, and the files the budget did not reach. The JSON report
carries the same under `usage` and per file under `scan.results`.

The budget is checked after each model response, so a scan can overshoot its
share by at most one request. The next layer bounds that request.

### One ceiling per request

`max-input-tokens` caps the input of any single model request. The pipeline
refuses a larger request before dispatch, whatever the model's context window
and even for a model litellm cannot look up (an `api-base` gateway, say), where
the window-relative guard would otherwise not apply. A request the cap refuses
fails that file's scan, which is REVIEW with the file named. With the cap, the
most one run can spend is `max-tokens` plus one request of `max-input-tokens`
per file, and the most one request can cost is known before the model is chosen.

### One budget per day

The two numbers above bound a run; `daily-token-budget` bounds the day, and it
does so from the platform's own meter rather than from anything this code
remembers, because the action keeps no state between runs and should not. Before
scanning, the driver reads from Cloud Monitoring how many tokens the project was
served in the trailing 24 hours (every model, input and output, the figure the
invoice is built from) and sets this run's budget to the smaller of `max-tokens`
and what is left of the day. Nothing left: nothing is launched, every file is
reported unscanned, and the verdict is REVIEW naming the day's figure. Something
left: the run proceeds with the smaller budget and the summary says so.

The meter is per project, so every repository that authenticates to the project
shares the one budget, which is the point. The meter runs a few minutes behind,
and concurrent runs each read it at their own start, so the day's figure can be
overshot by about one run budget per run in flight plus those minutes; the
billing stop below is the backstop behind that.

A daily budget that is set but cannot be read (the role is missing, the API is
off, the route has no Google Cloud project) is a setup error, exit 3, with the
fix named: a limit that is silently skipped is not a limit. `0` turns the check
off explicitly, which is the setting for a route other than the Agent Platform.

### Fewer runs

Nothing bounds how many runs a day there are except what triggers them; the
daily budget catches the day when that goes wrong, and these make it less
likely. The workflow above trims the runs nobody reads: drafts wait until they
are marked ready, bot-authored pull requests (dependency bumps) are skipped, and
`concurrency` with `cancel-in-progress` keeps a force-push from stacking runs on
one pull request. Same-repository pull requests only means every remaining run
was started by someone with push access.

If you need a ceiling on runs in flight across the repository, add a job-level
`concurrency` group without `cancel-in-progress`; runs then queue one at a time.
GitHub keeps only one queued run per group and cancels the rest, so a burst of
pull requests leaves some with a cancelled check until their next push. Fine for
an advisory check; think twice before combining it with `fail-on`.

### The stop of last resort

Everything above is this code; if it has a bug, nothing on the platform side
stops spend on its own. The one deterministic stop the platform offers is a
billing budget wired to an action that cuts the project off, set up under The
spending stop above. It is yours to do: Mantis bounds what one run spends and
what one day spends; what the project may lose to a bug in a month is a decision
only its owner can make.

### Sizing

Measured with a Gemini model at the default `max-llm-calls` of 150, every token
at full price:

| File        | Tokens | Calls | Minutes |
| ----------- | ------ | ----- | ------- |
| 20 lines    | 261k   | 44    | 7       |
| 440 lines   | 353k   | 45    | 19      |
| 1,000 lines | 658k   | 51    | 14      |
| 2,000 lines | 853k   | 60    | 13      |

Calls stay well under the cap; tokens are what grows with the file, which is why
tokens are the unit. The floor is about 250k whatever the file (prompts and
graph nodes), and size adds roughly 300k per thousand lines. The defaults follow
from the table:

- `max-tokens: 10M`: a pull request of ten ordinary files finishes whole (1M a
  file); a twenty-five-file one gets 400k a file, enough for files up to a few
  hundred lines, and its biggest pause at their share with the verdict saying
  which. Below 100k a share is not worth launching and the file is reported
  unscanned instead.
- `daily-token-budget: 100M`: forty typical runs, or ten at the worst case, in
  any 24 hours. A repository that sees more pushes than that in a day raises it
  knowingly; a trigger that fires every minute is stopped within hours.
- `max-input-tokens: 262144`: a quarter of the smallest Gemini window and
  several times the largest request an ordinary scan makes.

Read your own summaries for a week before changing any of them. A budget that is
too small shows up as REVIEW with files named, never as a silent pass. Minutes
per file set the other limit: the job's `timeout-minutes` ends a run that is
still scanning, and the files it did not reach are REVIEW too.

## Inputs

| Input                | Default            | Meaning                                                                                                                                                                                                         |
| -------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `base-sha`           | the PR's base SHA  | Commit the diff is taken against.                                                                                                                                                                               |
| `model`              | `gemini-3.8-flash` | Model id; a bare `gemini-*` id routes to the Gemini API when `GEMINI_API_KEY` is set, otherwise to the Agent Platform when a project is.                                                                        |
| `api-base`           | empty              | Custom OpenAI-compatible endpoint. Skips the credential check.                                                                                                                                                  |
| `max-llm-calls`      | `150`              | LLM call ceiling **per changed file**; a loop guard, not the spend control.                                                                                                                                     |
| `max-tokens`         | `10M`              | Token budget **per run**, every token at full price, shared equally over the files left to scan; files it cannot reach are reported unscanned (REVIEW). `0` removes it.                                         |
| `daily-token-budget` | `100M`             | Tokens the project may be served in any 24 hours, read from Cloud Monitoring before the run; the run's budget is cut to what is left. Unreadable when set: exit 3. `0` disables.                                |
| `max-input-tokens`   | `262144`           | Input tokens one model request may carry; larger requests are refused before dispatch, for any model. `0` leaves only the model's window.                                                                       |
| `max-files`          | `25`               | Files scanned per run; the rest are reported unscanned (REVIEW).                                                                                                                                                |
| `per-file-timeout`   | `1800`             | Seconds before one file's scan is abandoned.                                                                                                                                                                    |
| `ignore`             | `default`          | Comma-separated fnmatch patterns left out of the scan and the gate. `default` is the built-in list (docs, images, lockfiles, fonts); an empty string ignores nothing.                                           |
| `fail-on`            | `none`             | What the findings do to the job. `none`: advisory. `block`: fail on BLOCK. `review`: fail on REVIEW or BLOCK.                                                                                                   |
| `fail-closed`        | `false`            | Whether Mantis being unable to look fails the job (setup failure, model outage, a changed file not scanned, a crash). `false`: never; the summary says what was not audited. `true`: green means fully audited. |
| `python-version`     | `3.13`             | Interpreter for the pipeline (3.12 or newer).                                                                                                                                                                   |

Outputs: `verdict` (`PASS`, `REVIEW`, or `BLOCK`) and `report` (path of the JSON
report with findings, blast radius, per-file scan results, and `usage`: tokens
and calls for the run against its budget).

## What the verdict means

The gate is the MCP server's `check_change`. Two things come out of a run, and
two inputs decide what each does to the job.

**The verdict** is what the findings say:

- **BLOCK**: an open HIGH or CRITICAL finding sits in the changed lines or in a
  function the change modified. A known finding follows its file through a
  rename; a known finding in a file the change deletes is a REVIEW until someone
  confirms the code is gone rather than moved.
- **REVIEW**: something needs a human: lower-severity findings in the change,
  findings elsewhere in a changed file or in a caller of changed code.
- **PASS**: nothing known intersects the change.

`fail-on` decides whether a BLOCK (or, with `review`, a REVIEW) fails the job.

**Coverage** is whether Mantis could look: `complete`, `partial`, or `none`. A
changed file over `max-files`, over the token budget, timed out, failed, or
unparseable, or a missing index or database, makes coverage partial. The verdict
shown is floored to REVIEW when coverage is incomplete, so a person sees it, and
the summary names every file and why. `fail-closed` decides whether that fails
the job; by default it does not, so a scanner that cannot look is never the
reason a merge stops. With `fail-closed: true` a green check means every changed
file was audited in full -- and means a model outage or a budget exhausted by a
sibling repository on the same project turns the check red until it is fixed.
That is a trade you make knowingly.

A setup failure (no base commit, credentials missing or rejected, the daily
meter unreadable, nothing launched) is reported as **not run**: an annotation, a
summary saying what to fix, a `verdict` output of `NOT_RUN`, and exit 3 under
`fail-closed`, exit 0 otherwise. Never a verdict on the pull request, never a
silent pass.

Annotations: direct findings are warnings (errors when `fail-on` would fail the
job on them); same-file and caller-radius findings are notices; unscanned files
are warnings. GitHub shows ten per level per step; the summary lists all of
them.

Exit codes of the underlying driver (`reference/scripts/pr_scan.py`): `0` the
job passes under the two policies, `1` it fails under them, `3` a setup failure
under `fail-closed`. What the model found, or did not find, is reported as-is.

## Threat model

The job processes attacker-influenced input (the pull request) while holding a
model credential. The design keeps those two facts apart:

- **The checkout is data.** Nothing from the pull request is imported,
  installed, or executed. Both of the action's steps run from the action's own
  directory with `PYTHONSAFEPATH` set, so a `pip/` package or a module planted
  at the root of the pull request is not what `python -m pip` or the driver
  imports. The pipeline reads the checkout; the sandbox is pinned to
  `static-only`, so dynamic exploit reproduction is off and no PR-supplied code
  runs on the runner regardless of what the repository's own Mantis
  configuration says. The workflow file is the exception, and is why the
  environment's reviewers exist (Setup, GitHub).
- **Nothing is written into the checkout.** The run's database, structural
  index, and scan log live in a fresh directory under `$RUNNER_TEMP` (or
  `--state-dir`), never under the repository, and the log is opened without
  following symlinks. A pull request can therefore neither plant a symlink where
  the run would write (which would turn the log into an append on any file the
  runner can write) nor pre-seed a database or index cache the run would trust
  as its own memory.
- **The change set does not depend on how git rendered a file.** Which files
  changed comes from `git diff --name-status -z`; the unified diff only adds
  changed lines where it has them. A `.gitattributes` line marking a file
  `-diff`, or a byte that trips binary detection, cannot remove the file from
  the scan or the gate: it is scanned, and gated without line information, which
  the gate treats as `direct`. Inside a hunk, a content line that reads
  `+++ b/other.py` is content; headers are recognized only between hunks.
- **Credential rejection is an exit status, not a log string.** The pipeline
  exits with a reserved status when the provider refuses the credentials, and
  decides that from the error's class or the provider's status code, never from
  an error message, which a model can write. The action keys on the exit status
  alone. The log echoes file names and contents the pull request author wrote,
  so no marker string in it is trusted.
- **No GitHub token.** The action never reads `github.token`. Annotations and
  the summary go through workflow commands and `GITHUB_STEP_SUMMARY`. With
  `permissions: contents: read` and `persist-credentials: false`, a compromised
  job cannot push, comment, approve, or reach other repositories. The pipeline
  subprocess does not inherit GitHub's own runtime tokens (`GITHUB_TOKEN`,
  `ACTIONS_RUNTIME_TOKEN`, `ACTIONS_ID_TOKEN_REQUEST_*`); every other variable
  on the step passes through, since which provider credentials a run needs is
  the consumer's choice of model. The credential configuration the auth step
  writes into the checkout -- which holds the URL and token for minting the
  job's identity tokens -- is moved out of the tree before the scan, and the
  pipeline never lists a `gha-creds-*.json` as a scan target. What the pipeline
  holds is the provider credential itself, for the life of the job; it holds
  nothing that reaches GitHub.
- **Output is escaped at every sink.** Annotation properties and data follow
  GitHub's workflow-command escaping, so a finding title cannot start a second
  command. Every value in the step summary that originates from the pull request
  (file names) or from the model after reading it (titles, reasons, function
  names) passes through the repository's markdown sanitizer and then has every
  angle bracket and ampersand encoded: no HTML of any kind, no headings, no
  escaping out of a code span. The JSON report is passed through the same
  credential scrubber before it is written.
- **Pinned code only, given the reviewers.** The action's own code comes from
  the commit you pin, and its Python dependencies install with
  `--require-hashes` from the pipeline's lock file; nothing is upgraded from the
  index first. The only third-party action it uses is `actions/setup-python`,
  pinned to a commit; the workflow above pins `checkout` and `auth` the same
  way. None of that binds the workflow file, which a pull request can rewrite
  and which GitHub runs as rewritten; the environment's required reviewers are
  what keep a rewritten workflow from receiving the credential.
- **No template injection.** Every input crosses into the shell through
  environment variables, never `${{ }}` interpolation inside `run:`. Revision
  arguments are validated as commit SHAs before they reach git, and every git
  call terminates option parsing explicitly.
- **Bounded spend, in tokens, by the driver.** One token budget per run, shared
  out by the driver and charged from each scan's own account (or in full when
  there is none), with an input cap on every request and the day's meter read
  again before every file; a pull request can make a run hit its budget, which
  is a coverage gap with files named, and nothing more. See Spend.

What it does **not** defend against:

- **Prompt injection into the verdict.** The pipeline reads the code it audits;
  hostile comments or strings in a pull request can try to talk the model out of
  reporting a planted defect. The deterministic layers (citation verification,
  evidence gates) raise the bar but do not remove it. This is why the default is
  advisory and why a human review stays in the loop.
- **Fork pull requests.** The scan needs the provider credential, which GitHub
  withholds from fork-triggered `pull_request` jobs by design. Keep the
  `head.repo.full_name == github.repository` condition; do not switch to
  `pull_request_target` to get credentials onto fork code.
- **Credential blast radius.** A fully compromised job holds the project's model
  credential for up to an hour. What limits it: a dedicated project that holds
  nothing else, the narrowest role that can call the model and read the meter
  (the setup script creates one; `roles/aiplatform.user` can also create
  training jobs and endpoints the token meter never sees, so prefer the custom
  role), a token that lives about an hour (workload identity; the repository
  secrets hold only names), the budget, and the platform-side stop. Treat it as
  a cost control, not an access control.

If you run the action from the same repository with `uses: ./action`, the pull
request also controls `action/` and `reference/`, so the reviewer has more to
read before approving a run (Setup, GitHub). This repository's own workflow does
exactly that. Pin to a commit anywhere else you dogfood.

## Local use

The driver runs anywhere with a checkout, the reference environment, and
application default credentials for a project with the API enabled:

```bash
cd reference && ./install.sh
gcloud auth application-default login
GOOGLE_CLOUD_PROJECT=<project> .venv/bin/python scripts/pr_scan.py \
  --repo /path/to/checkout --base <base-sha> --model vertex_ai/gemini-3.8-flash \
  --output-json /tmp/report.json
```

Without `GITHUB_STEP_SUMMARY` and `GITHUB_OUTPUT` in the environment it only
prints annotations and the one-line verdict; `--no-annotations` silences the
former.
