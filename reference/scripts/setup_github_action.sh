#!/usr/bin/env bash
# Sets up Google Cloud and GitHub for the Mantis PR scan action in one run.
#
# What the Setup section of docs/github-action.md does by hand: a project
# that can run the model, a role that can call the model and read the
# project's own usage meter and nothing else, a service account with that
# role, a workload identity pool and GitHub OIDC provider admitting one
# repository (by id) from one environment, and a binding from exactly that
# subject to the service account. No key is created anywhere. Optionally a
# billing budget, and optionally the GitHub environment -- with the
# protection the design depends on -- and its three values.
#
# Safe to run again: every step checks before it creates, the provider
# condition and the role's permissions are updated to what this version
# wants, and an environment that already exists is left as it is (its
# reviewers and branch rules included; only its secrets are set), so a run
# that failed halfway, or a repository that was renamed, converges on the
# same end state without undoing what a person configured.
#
# Usage:
#   setup_github_action.sh --project PROJECT --repo OWNER/NAME
#       (--budget-usd AMOUNT | --no-budget)
#       [--github --reviewer-team SLUG ... | --reviewer-user LOGIN ... | --no-reviewers]
#       [--refresh] [--repo-id ID --owner-id ID]
#       [--environment NAME] [--service-account NAME] [--pool NAME] [--provider NAME]
#       [--role ROLE] [--dry-run]
#
#   --budget-usd    Also create a Pub/Sub topic and a billing budget on the
#                   project at 50%, 90% and 100% of AMOUNT. The budget sends
#                   mail; turning it into a stop that detaches billing is
#                   Google's recipe and a manual step, and is only for a
#                   project that holds nothing you would miss (see the doc).
#                   One of --budget-usd and --no-budget is required.
#   --github        Also create the GitHub environment and set the three
#                   secrets on it, with the gh CLI (needs admin on the repo),
#                   and read the repository's ids for the provider condition.
#   --reviewer-team / --reviewer-user
#                   Required reviewers for the environment (repeatable). On
#                   a pull_request event GitHub runs the pull request's copy
#                   of the workflow, and a job that declares the environment
#                   gets its secrets: without reviewers, anyone who can push
#                   a branch can take the credential. Required with --github
#                   unless --no-reviewers is passed, which prints why that
#                   is a bad idea and does it anyway.
#   --refresh       Set up the knowledge-refresh workflow's side instead:
#                   environment mantis-knowledge, provider mantis-refresh,
#                   a condition that admits refs/heads/main only, and an
#                   environment whose deployment-branch rule is main only
#                   (reviewers are not needed there). Run once per mode.
#   --repo-id / --owner-id
#                   GitHub's numeric ids for the repository and its owner,
#                   pinned in the provider condition so a renamed or
#                   re-created repository of the same name is not admitted.
#                   Read with gh whenever gh is installed and logged in (a
#                   read-only call anyone can make on a public repository);
#                   pass them when it is not, or the condition pins the name
#                   alone, with a warning.
#
#   One project can serve several repositories: run this once per repository
#   with its own --provider and --service-account, and the pool, the role
#   and the budget are shared. The budget is per project (--budget-usd the
#   first time; later runs find it and need neither budget flag).
#   --role          A role for the service account instead of the custom
#                   least-privilege role this creates (for example
#                   roles/aiplatform.user, if a model needs more). The meter
#                   read still gets monitoring.timeSeries.list.
#   --dry-run       Print every command instead of running it.

set -euo pipefail

PROJECT=""
REPO=""
REPO_ID=""
OWNER_ID=""
ENVIRONMENT=""
SA_NAME="mantis-pr-scan"
POOL="github"
PROVIDER=""
BUDGET_USD=""
NO_BUDGET=0
DO_GITHUB=0
NO_REVIEWERS=0
REFRESH=0
ROLE_OVERRIDE=""
DRY_RUN=0
REVIEWER_TEAMS=()
REVIEWER_USERS=()

usage() {
    sed -n '2,62p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    echo "error: $*" >&2
    exit 2
}

warn() {
    printf '\033[1;31mWARNING:\033[0m %s\n' "$*" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT="${2:-}"; shift 2 ;;
        --repo) REPO="${2:-}"; shift 2 ;;
        --repo-id) REPO_ID="${2:-}"; shift 2 ;;
        --owner-id) OWNER_ID="${2:-}"; shift 2 ;;
        --environment) ENVIRONMENT="${2:-}"; shift 2 ;;
        --service-account) SA_NAME="${2:-}"; shift 2 ;;
        --pool) POOL="${2:-}"; shift 2 ;;
        --provider) PROVIDER="${2:-}"; shift 2 ;;
        --budget-usd) BUDGET_USD="${2:-}"; shift 2 ;;
        --no-budget) NO_BUDGET=1; shift ;;
        --github) DO_GITHUB=1; shift ;;
        --reviewer-team) REVIEWER_TEAMS+=("${2:-}"); shift 2 ;;
        --reviewer-user) REVIEWER_USERS+=("${2:-}"); shift 2 ;;
        --no-reviewers) NO_REVIEWERS=1; shift ;;
        --refresh) REFRESH=1; shift ;;
        --role) ROLE_OVERRIDE="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

if [[ $REFRESH -eq 1 ]]; then
    ENVIRONMENT="${ENVIRONMENT:-mantis-knowledge}"
    PROVIDER="${PROVIDER:-mantis-refresh}"
else
    ENVIRONMENT="${ENVIRONMENT:-mantis-pr-scan}"
    PROVIDER="${PROVIDER:-mantis}"
fi

# Every value below ends up inside a gcloud argument, a CEL string literal,
# or a JSON body, so each is held to the character set its consumer defines.
# A repository is owner/name; a project id is lowercase letters, digits and
# hyphens; names of GitHub environments and of pools, providers and service
# accounts are plain identifiers; ids are digits. Anything else is refused
# rather than quoted.
[[ -n "$PROJECT" ]] || die "--project is required"
[[ -n "$REPO" ]] || die "--repo OWNER/NAME is required"
[[ "$PROJECT" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "--project '$PROJECT' is not a project id"
[[ "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "--repo '$REPO' is not OWNER/NAME"
[[ "$ENVIRONMENT" =~ ^[A-Za-z0-9_.-]+$ ]] || die "--environment '$ENVIRONMENT' has characters the provider condition cannot hold"
[[ "$SA_NAME" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "--service-account '$SA_NAME' is not a service account id"
[[ "$POOL" =~ ^[a-z][a-z0-9-]{2,30}[a-z0-9]$ ]] || die "--pool '$POOL' is not a pool id"
[[ "$PROVIDER" =~ ^[a-z][a-z0-9-]{2,30}[a-z0-9]$ ]] || die "--provider '$PROVIDER' is not a provider id"
[[ -z "$REPO_ID" || "$REPO_ID" =~ ^[0-9]+$ ]] || die "--repo-id '$REPO_ID' is not a number"
[[ -z "$OWNER_ID" || "$OWNER_ID" =~ ^[0-9]+$ ]] || die "--owner-id '$OWNER_ID' is not a number"
[[ -z "$ROLE_OVERRIDE" || "$ROLE_OVERRIDE" =~ ^(roles/[A-Za-z0-9_.]+|projects/[a-z][a-z0-9-]+/roles/[A-Za-z0-9_.]+)$ ]] || die "--role '$ROLE_OVERRIDE' is not a role name"
for t in "${REVIEWER_TEAMS[@]+"${REVIEWER_TEAMS[@]}"}"; do
    [[ "$t" =~ ^[A-Za-z0-9_.-]+$ ]] || die "--reviewer-team '$t' is not a team slug"
done
for u in "${REVIEWER_USERS[@]+"${REVIEWER_USERS[@]}"}"; do
    [[ "$u" =~ ^[A-Za-z0-9-]+$ ]] || die "--reviewer-user '$u' is not a login"
done
if [[ -n "$BUDGET_USD" ]]; then
    [[ "$BUDGET_USD" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--budget-usd '$BUDGET_USD' is not an amount"
    [[ $NO_BUDGET -eq 0 ]] || die "--budget-usd and --no-budget contradict each other"
fi
# Neither budget flag is allowed only when the project already has its
# budget (a second repository on a shared project); checked once gcloud is
# known to work, below.
N_REVIEWERS=$(( ${#REVIEWER_TEAMS[@]} + ${#REVIEWER_USERS[@]} ))
if [[ $DO_GITHUB -eq 1 && $REFRESH -eq 0 && $N_REVIEWERS -eq 0 && $NO_REVIEWERS -eq 0 ]]; then
    die "--github needs --reviewer-team or --reviewer-user for the '$ENVIRONMENT' environment: on a pull_request event GitHub runs the pull request's copy of the workflow, and a job that declares the environment receives its secrets; required reviewers are what keep a rewritten workflow from taking the credential. Pass --no-reviewers to create it unprotected anyway."
fi

SA="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
# One topic and one budget per project, whatever the repository: the budget
# caps what the project spends, and several repositories on one project
# share it.
TOPIC="mantis-pr-scan-budget"
BUDGET_NAME="Mantis PR scan (${PROJECT})"
BUDGET_LABEL="none (--no-budget)"
[[ -n "$BUDGET_USD" ]] && BUDGET_LABEL="${BUDGET_USD} USD a month"
ROLE_ID="mantisPrScan"
# What the pipeline needs and nothing more: call the model (generateContent
# and countTokens check endpoints.predict on publisher models), manage the
# context caches it creates, and read the project's own token meter.
ROLE_PERMISSIONS="aiplatform.endpoints.predict,aiplatform.cachedContents.create,aiplatform.cachedContents.get,aiplatform.cachedContents.list,aiplatform.cachedContents.update,aiplatform.cachedContents.delete,monitoring.timeSeries.list"

# run: execute, or print under --dry-run. Arguments are printed quoted so
# the dry run is paste-able and shows exactly what would be sent.
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  $'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

# exists: a read-only describe. Under --dry-run nothing is consulted and
# everything is treated as missing, so the plan shows every create.
exists() {
    [[ $DRY_RUN -eq 0 ]] && "$@" >/dev/null 2>&1
}

command -v gcloud >/dev/null 2>&1 || die "gcloud is not installed (https://cloud.google.com/sdk/docs/install)"
if [[ $DO_GITHUB -eq 1 ]]; then
    command -v gh >/dev/null 2>&1 || die "--github needs the gh CLI (https://cli.github.com)"
    if [[ $DRY_RUN -eq 0 ]]; then
        gh auth status >/dev/null 2>&1 || die "gh is not logged in; run: gh auth login"
    fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
    NUMBER="<project-number>"
else
    gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q . \
        || die "gcloud has no active account; run: gcloud auth login"
    NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)' 2>/dev/null) \
        || die "project '$PROJECT' not found or not readable with the active gcloud account"
    [[ -n "$NUMBER" ]] || die "could not read the project number of '$PROJECT'"
fi

# The project's budget, if it has one already. Decides whether a budget
# flag is required: a repository added to a project that is already
# budgeted needs no decision; the first one does.
EXISTING_BUDGET=""
BILLING=""
if [[ $DRY_RUN -eq 0 ]]; then
    BILLING=$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)' 2>/dev/null || true)
    BILLING="${BILLING#billingAccounts/}"
    if [[ -n "$BILLING" ]]; then
        EXISTING_BUDGET=$(gcloud billing budgets list --billing-account="$BILLING" \
            --filter="displayName='${BUDGET_NAME}'" --format='value(name)' 2>/dev/null | head -n1 || true)
    fi
fi
if [[ -z "$BUDGET_USD" && $NO_BUDGET -eq 0 ]]; then
    if [[ -n "$EXISTING_BUDGET" ]]; then
        BUDGET_LABEL="the project's existing budget (unchanged)"
    else
        die "pass --budget-usd AMOUNT, or --no-budget to say explicitly that nothing on the platform side caps a month"
    fi
fi

# The repository's ids. GitHub's names are reusable -- a renamed or deleted
# repository's name can be taken by someone else, who then creates an
# environment of the same name -- and its numeric ids are not. Read with
# gh whenever gh can answer: the call is read-only and works for anyone on
# a public repository, --github or not.
if [[ -z "$REPO_ID" || -z "$OWNER_ID" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
        if command -v gh >/dev/null 2>&1; then
            REPO_ID="${REPO_ID:-<repo-id>}"; OWNER_ID="${OWNER_ID:-<owner-id>}"
        fi
    elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1 \
            && ids=$(gh api "repos/${REPO}" --jq '.id, .owner.id, .full_name' 2>/dev/null); then
        REPO_ID=$(echo "$ids" | sed -n 1p)
        OWNER_ID=$(echo "$ids" | sed -n 2p)
        CANONICAL=$(echo "$ids" | sed -n 3p)
        [[ "$REPO_ID" =~ ^[0-9]+$ && "$OWNER_ID" =~ ^[0-9]+$ ]] || die "unexpected repository ids from gh: $ids"
        if [[ -n "$CANONICAL" && "$CANONICAL" != "$REPO" ]]; then
            # The OIDC claim is case-preserving and CEL == is case-sensitive.
            echo "Using the repository's canonical name '$CANONICAL' (given: '$REPO')."
            REPO="$CANONICAL"
        fi
    elif [[ $DO_GITHUB -eq 1 ]]; then
        die "could not read repository '$REPO' with gh (is the name right, is gh logged in, are you a collaborator?)"
    fi
fi

PROVIDER_NAME="projects/${NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
SUBJECT="repo:${REPO}:environment:${ENVIRONMENT}"
PRINCIPAL="principal://iam.googleapis.com/projects/${NUMBER}/locations/global/workloadIdentityPools/${POOL}/subject/${SUBJECT}"
CONDITION="assertion.repository == '${REPO}' && assertion.environment == '${ENVIRONMENT}'"
if [[ -n "$REPO_ID" && -n "$OWNER_ID" ]]; then
    CONDITION="assertion.repository_id == '${REPO_ID}' && assertion.repository_owner_id == '${OWNER_ID}' && ${CONDITION}"
else
    warn "no --repo-id/--owner-id and gh could not supply them: the provider condition pins the repository by NAME only. A renamed or re-created repository of the same name would be admitted. Pass the ids (gh api repos/${REPO} --jq '.id, .owner.id') and re-run."
fi
if [[ $REFRESH -eq 1 ]]; then
    CONDITION="${CONDITION} && assertion.ref == 'refs/heads/main'"
fi
MAPPING="google.subject=assertion.sub,attribute.repository=assertion.repository"
if [[ -n "$ROLE_OVERRIDE" ]]; then
    MODEL_ROLE="$ROLE_OVERRIDE"
else
    MODEL_ROLE="projects/${PROJECT}/roles/${ROLE_ID}"
fi

cat <<EOF
======================================================================
Mantis PR scan: Google Cloud setup$([[ $REFRESH -eq 1 ]] && echo " (knowledge refresh)")
Project:          ${PROJECT} (${NUMBER})
Repository:       ${REPO}$([[ -n "$REPO_ID" ]] && echo " (id ${REPO_ID}, owner ${OWNER_ID})")
Environment:      ${ENVIRONMENT}
Service account:  ${SA}
Role:             ${MODEL_ROLE}
Pool / provider:  ${POOL} / ${PROVIDER}
Condition:        ${CONDITION}
Budget:           ${BUDGET_LABEL}
GitHub:           $([[ $DO_GITHUB -eq 1 ]] && echo "environment and secrets via gh" || echo "values printed at the end")
$([[ $DRY_RUN -eq 1 ]] && echo "DRY RUN: nothing is created; the commands are printed.")
======================================================================
EOF

echo "[1/6] APIs: model, usage meter, token minting and exchange"
run gcloud services enable aiplatform.googleapis.com monitoring.googleapis.com \
    iamcredentials.googleapis.com iam.googleapis.com sts.googleapis.com \
    --project="$PROJECT"

echo "[2/6] Role ${MODEL_ROLE}"
if [[ -z "$ROLE_OVERRIDE" ]]; then
    # Converged on every run: the permission list is this script's to own.
    if exists gcloud iam roles describe "$ROLE_ID" --project="$PROJECT"; then
        run gcloud iam roles update "$ROLE_ID" --project="$PROJECT" \
            --permissions="$ROLE_PERMISSIONS" --stage=GA --quiet --format=none
    else
        run gcloud iam roles create "$ROLE_ID" --project="$PROJECT" \
            --title="Mantis PR scan" \
            --description="Call the model, manage its context caches, read the project's token meter. Nothing else." \
            --permissions="$ROLE_PERMISSIONS" --stage=GA --format=none
    fi
else
    echo "      using ${ROLE_OVERRIDE} as given (the custom role is not created)"
fi

echo "[3/6] Service account ${SA}"
if exists gcloud iam service-accounts describe "$SA" --project="$PROJECT"; then
    echo "      exists"
else
    run gcloud iam service-accounts create "$SA_NAME" --project="$PROJECT" \
        --display-name="Mantis PR scan"
fi
# Bindings are idempotent: adding one that is already there changes nothing.
run gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:${SA}" --role="$MODEL_ROLE" \
    --condition=None --quiet --format=none
if [[ -n "$ROLE_OVERRIDE" ]]; then
    # A role given from outside may not read the meter; the viewer role does.
    run gcloud projects add-iam-policy-binding "$PROJECT" \
        --member="serviceAccount:${SA}" --role="roles/monitoring.viewer" \
        --condition=None --quiet --format=none
fi

echo "[4/6] Workload identity pool ${POOL}"
if exists gcloud iam workload-identity-pools describe "$POOL" --location=global --project="$PROJECT"; then
    # A pool deleted in the last 30 days still holds its name; bring it back
    # rather than fail on "already exists" for something describe cannot see.
    if [[ "$(gcloud iam workload-identity-pools describe "$POOL" --location=global --project="$PROJECT" --format='value(state)' 2>/dev/null)" == "DELETED" ]]; then
        run gcloud iam workload-identity-pools undelete "$POOL" --location=global --project="$PROJECT"
    else
        echo "      exists"
    fi
else
    run gcloud iam workload-identity-pools create "$POOL" --location=global --project="$PROJECT" \
        --display-name="GitHub Actions (Mantis)"
fi

echo "[5/6] OIDC provider ${PROVIDER}"
if exists gcloud iam workload-identity-pools providers describe "$PROVIDER" \
        --location=global --workload-identity-pool="$POOL" --project="$PROJECT"; then
    # Converge the condition on the repository and environment given, so a
    # re-run after a rename, or a typo the first time, ends in the right place.
    run gcloud iam workload-identity-pools providers update-oidc "$PROVIDER" \
        --location=global --workload-identity-pool="$POOL" --project="$PROJECT" \
        --attribute-mapping="$MAPPING" \
        --attribute-condition="$CONDITION"
else
    run gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
        --location=global --workload-identity-pool="$POOL" --project="$PROJECT" \
        --issuer-uri="https://token.actions.githubusercontent.com" \
        --attribute-mapping="$MAPPING" \
        --attribute-condition="$CONDITION"
fi

echo "[6/6] Let exactly ${SUBJECT} act as the service account"
# The binding names one subject -- this repository, this environment -- not
# every token the pool might ever map for the repository. A provider added to
# the pool later, or a condition loosened later, does not widen it.
run gcloud iam service-accounts add-iam-policy-binding "$SA" --project="$PROJECT" \
    --role="roles/iam.workloadIdentityUser" \
    --member="$PRINCIPAL" --quiet --format=none

if [[ -n "$BUDGET_USD" ]]; then
    echo "[budget] ${BUDGET_USD} USD a month on ${PROJECT}"
    [[ $DRY_RUN -eq 1 ]] && BILLING="<billing-account>"
    if [[ -z "$BILLING" ]]; then
        warn "the project has no billing account attached; nothing to budget. Attach one and re-run with --budget-usd."
    else
        if exists gcloud pubsub topics describe "$TOPIC" --project="$PROJECT"; then
            echo "      topic ${TOPIC} exists"
        else
            run gcloud pubsub topics create "$TOPIC" --project="$PROJECT"
        fi
        if [[ -n "$EXISTING_BUDGET" ]]; then
            run gcloud billing budgets update "$EXISTING_BUDGET" --billing-account="$BILLING" \
                --budget-amount="${BUDGET_USD}USD"
        else
            run gcloud billing budgets create --billing-account="$BILLING" \
                --display-name="$BUDGET_NAME" --budget-amount="${BUDGET_USD}USD" \
                --filter-projects="projects/${NUMBER}" \
                --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
                --threshold-rule=percent=1.0 \
                --notifications-rule-pubsub-topic="projects/${PROJECT}/topics/${TOPIC}"
        fi
    fi
fi

ENV_NOTE=""
if [[ $DO_GITHUB -eq 1 ]]; then
    echo "[github] environment ${ENVIRONMENT} on ${REPO}"
    ENV_EXISTS=0
    if [[ $DRY_RUN -eq 0 ]] && gh api "repos/${REPO}/environments/${ENVIRONMENT}" >/dev/null 2>&1; then
        ENV_EXISTS=1
    fi
    if [[ $ENV_EXISTS -eq 1 ]]; then
        reviewers=$(gh api "repos/${REPO}/environments/${ENVIRONMENT}" \
            --jq '[.protection_rules[]? | select(.type == "required_reviewers") | .reviewers | length] | add // 0' 2>/dev/null || echo "?")
        policy=$(gh api "repos/${REPO}/environments/${ENVIRONMENT}" \
            --jq 'if .deployment_branch_policy == null then "any branch" elif .deployment_branch_policy.protected_branches then "protected branches" else "selected branches" end' 2>/dev/null || echo "?")
        rules=$(gh api "repos/${REPO}/environments/${ENVIRONMENT}" \
            --jq '.protection_rules | length' 2>/dev/null || echo "?")
        if [[ "$rules" == "0" && "$policy" == "any branch" ]]; then
            # GitHub creates the environment, unprotected, the first time a
            # workflow names it -- a pull request opened before this script
            # ran. One with no rule at all has nothing a PUT could take away.
            echo "      exists with no protection at all (GitHub creates one the first time a workflow names it); configured as if new."
            ENV_EXISTS=0
        fi
    fi
    if [[ $ENV_EXISTS -eq 1 ]]; then
        # Never replaced: PUT on an environment resets whatever it omits,
        # and the reviewers a person added are the control the design
        # depends on. Only the secrets are set; the protection is reported.
        echo "      exists: ${reviewers} required reviewer(s), deployment branches: ${policy}. Left as it is."
        if [[ $REFRESH -eq 0 && "$reviewers" == "0" ]]; then
            ENV_NOTE="The '${ENVIRONMENT}' environment has NO required reviewers. Add them (Settings, Environments) before any secret is used: without them anyone who can push a branch can take the credential."
        fi
        if [[ $REFRESH -eq 1 && "$policy" == "any branch" ]]; then
            ENV_NOTE="The '${ENVIRONMENT}' environment allows any branch. Set its deployment branches to main only: a dispatched workflow runs from whichever branch was chosen."
        fi
    else
        body_file=$(mktemp)
        trap 'rm -f "$body_file"' EXIT
        if [[ $REFRESH -eq 1 ]]; then
            # Branch rule first, reviewers none: a scheduled job has nobody
            # to ask, and main-only is what keeps it honest.
            printf '{"deployment_branch_policy":{"protected_branches":false,"custom_branch_policies":true}}' > "$body_file"
        else
            reviewers_json=""
            for t in "${REVIEWER_TEAMS[@]+"${REVIEWER_TEAMS[@]}"}"; do
                org="${REPO%%/*}"
                if [[ $DRY_RUN -eq 1 ]]; then tid="<team-id:$t>"; else
                    tid=$(gh api "orgs/${org}/teams/${t}" --jq .id 2>/dev/null) || die "team '$t' not found in '$org'"
                fi
                reviewers_json+="${reviewers_json:+,}{\"type\":\"Team\",\"id\":${tid}}"
            done
            for u in "${REVIEWER_USERS[@]+"${REVIEWER_USERS[@]}"}"; do
                if [[ $DRY_RUN -eq 1 ]]; then uid="<user-id:$u>"; else
                    uid=$(gh api "users/${u}" --jq .id 2>/dev/null) || die "user '$u' not found"
                fi
                reviewers_json+="${reviewers_json:+,}{\"type\":\"User\",\"id\":${uid}}"
            done
            printf '{"reviewers":[%s],"prevent_self_review":true,"deployment_branch_policy":null}' "$reviewers_json" > "$body_file"
            if [[ $N_REVIEWERS -eq 0 ]]; then
                warn "creating '${ENVIRONMENT}' with NO required reviewers, as asked. Until a person adds them, anyone who can push a branch to ${REPO} can rewrite the workflow and take the credential. Do not set the secrets on a shared repository in this state."
            fi
        fi
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  $ gh api --method PUT repos/${REPO}/environments/${ENVIRONMENT} --input - <<< '$(cat "$body_file")'"
        else
            gh api --method PUT "repos/${REPO}/environments/${ENVIRONMENT}" --input "$body_file" --silent
        fi
        if [[ $REFRESH -eq 1 ]]; then
            run gh api --method POST "repos/${REPO}/environments/${ENVIRONMENT}/deployment-branch-policies" \
                -f name=main -f type=branch --silent
        fi
    fi
    run gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --repo "$REPO" --env "$ENVIRONMENT" --body "$PROVIDER_NAME"
    run gh secret set GCP_SERVICE_ACCOUNT --repo "$REPO" --env "$ENVIRONMENT" --body "$SA"
    run gh secret set GCP_PROJECT --repo "$REPO" --env "$ENVIRONMENT" --body "$PROJECT"
fi

cat <<EOF

======================================================================
Done. The three values GitHub needs (environment "${ENVIRONMENT}"):

  GCP_WORKLOAD_IDENTITY_PROVIDER  ${PROVIDER_NAME}
  GCP_SERVICE_ACCOUNT             ${SA}
  GCP_PROJECT                     ${PROJECT}
EOF
if [[ $DO_GITHUB -eq 1 ]]; then
    echo "  (set on the environment by this run)"
else
    echo "  Set them as secrets on the environment, or re-run with --github."
fi
if [[ -n "$ENV_NOTE" ]]; then
    echo
    warn "$ENV_NOTE"
fi

if [[ $REFRESH -eq 1 ]]; then
    cat <<EOF

Workflow: copy .github/workflows/mantis-knowledge-refresh.yml from the Mantis
repository. It declares environment "${ENVIRONMENT}" and refuses to run from
any ref but main; the environment's branch rule and the provider condition
enforce the same from the other two sides.

Still yours to do:
  - A CODEOWNERS entry for .mantis/knowledge/ naming a security owner: the
    pull request the refresh opens is not itself scanned, and a merged export
    decides what later scans suppress.
EOF
else
    cat <<EOF

Workflow (.github/workflows/mantis-pr-scan.yml), with every action pinned
to a commit. The environment's required reviewers are not optional: on a
pull_request event GitHub runs the pull request's copy of this file.

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
    group: mantis-pr-scan-\${{ github.event.pull_request.number || github.run_id }}
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
      environment: ${ENVIRONMENT}   # required reviewers on it: see the doc
      steps:
        - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
          with:
            fetch-depth: 0
            persist-credentials: false
        - uses: google-github-actions/auth@7c6bc770dae815cd3e89ee6cdf493a5fab2cc093 # v3.0.0
          with:
            workload_identity_provider: \${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
            service_account: \${{ secrets.GCP_SERVICE_ACCOUNT }}
        - uses: google/mantis/action@<commit-sha>
          env:
            GOOGLE_CLOUD_PROJECT: \${{ secrets.GCP_PROJECT }}
          with:
            model: vertex_ai/gemini-3.8-flash
            base-sha: \${{ github.event.pull_request.base.sha || inputs.base-sha }}

Still yours to do:
  - Before approving a run, the reviewers read the diff of the workflow file
    (and of action/ and reference/ if the action runs from this repository).
EOF
fi
if [[ -n "$BUDGET_USD" ]]; then
    cat <<EOF
  - The budget only sends mail until something acts on it. To make it a
    stop, deploy Google's function that detaches billing when the budget is
    spent, subscribed to projects/${PROJECT}/topics/${TOPIC}:
      https://docs.cloud.google.com/billing/docs/how-to/disable-billing-with-notifications
    WARNING: detaching billing shuts down every billable service in
    ${PROJECT} and can lose whatever is in it. Do this only if the project
    holds nothing but this scan's service account, pool and budget --
    nothing persistent, nothing another system depends on.
EOF
elif [[ -n "$EXISTING_BUDGET" ]]; then
    echo "  - The project's budget '${BUDGET_NAME}' is left as it is."
else
    cat <<EOF
  - No billing budget (--no-budget): nothing on the platform side caps a
    month. To add one later, re-run with --budget-usd AMOUNT; to make it a
    stop rather than mail, see
    https://docs.cloud.google.com/billing/docs/how-to/disable-billing-with-notifications
    and the warning in docs/github-action.md.
EOF
fi
if [[ -z "$ROLE_OVERRIDE" ]]; then
    cat <<EOF
  - If the first run fails with a permission the role lacks, the scan log
    names it; add it to this script's ROLE_PERMISSIONS, or pass
    --role roles/aiplatform.user to fall back to the predefined role (which
    can also create training jobs and endpoints the token meter never sees).
EOF
fi
echo "Then run the workflow once by hand (Actions, Mantis PR scan, Run workflow)."
