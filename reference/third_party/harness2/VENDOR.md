# Vendored: Harness² core

- **Upstream:** https://github.com/google-research/harness2
- **Commit:** `1cfe254`
- **Vendored:** 2026-10-20
- **License:** Apache-2.0 (see `LICENSE` in this directory; every vendored
  `.py` carries upstream's own `Copyright 2026 Google LLC` header)
- **Runtime dependencies:** none (`pyproject.toml` declares `dependencies =
  []`; the core imports only the standard library and `harness2.*`)
- **Python:** upstream declares `>=3.12,<3.14`; verified on the project venv
  (3.13)

The `Commit`, `Vendored` and `Files` fields are rewritten by
`reference/scripts/vendor_harness2.py`; everything else is hand-maintained.

## Why a vendored copy

`reference/requirements.txt` is installed with `pip --require-hashes`, which
forbids VCS requirements (`git+https://...`), and Harness2 is not published on
PyPI. A git submodule would push a `submodule update` dance onto every user
and onto the GitHub Action. Harness2's core has zero runtime dependencies and
supports Python 3.13, so a vendored copy adds nothing to the lockfile and
nothing to the install.

## Included

Only the substrate- and benchmark-independent method code. The shim
`reference/evals/evolve/_vendor.py` puts this directory on `sys.path` and is
the only sanctioned import route.

<!-- vendored-files:begin -->
- `harness2/__init__.py`
- `harness2/config.py`
- `harness2/core/__init__.py`
- `harness2/core/context.py`
- `harness2/core/judge.py`
- `harness2/core/ledger.py`
- `harness2/core/library.py`
- `harness2/core/prompts.py`
- `harness2/core/proposer.py`
- `harness2/core/recursion.py`
- `harness2/core/report.py`
- `harness2/core/runmark.py`
- `harness2/core/schedule.py`
- `harness2/core/selection.py`
- `harness2/core/spec.py`
- `harness2/core/tree.py`
<!-- vendored-files:end -->

`harness2/config.py` is included because `core/spec.py`, `core/proposer.py`
and `core/recursion.py` import it. It reads `HARNESS2_*` environment variables
and a `.env` in the current directory at import time; Mantis passes every
setting explicitly, and the shim refuses to load the package when any
`HARNESS2_*` variable is set and restores `os.environ` to its pre-import state
afterwards, so upstream's defaults never leak into a Mantis process.

## Excluded

- `harness2/benchmarks/` -- JobBench / WorkBuddy / Harvey adapters; Mantis is
  its own benchmark.
- `harness2/substrate/` -- OpenCode and Codex substrates; Mantis is its own
  substrate (ADK).
- `harness2/cli/` -- sweep / judge / report / setup / bridge entry points that
  drive upstream's directory layout and `.env`.
- `harness2/install.py`, `harness2/preflight.py` -- clone and pin upstream's
  `dependencies/` tree (opencode, codex, benchmark repos); nothing Mantis uses.
- `harness2/demo.py` -- interactive demo over the OpenCode substrate.
- `docs/`, `README.md`, `pyproject.toml` -- not needed to import the package;
  `pyproject.toml` is deliberately absent so `config.ROOT` resolves to the
  working directory rather than to this tree.

## Local patches

Applied in order by `vendor_harness2.py` with `patch -p1 -F0` (exact context;
drift fails loudly so the patch is regenerated, not fuzzed). Each patched file
carries a `# Modified by Mantis:` line under its license header, as
Apache-2.0 section 4(b) requires.

1. `patches/0001-register-surface.diff` -- `harness2/core/prompts.py`: adds
   `register_surface(name, *, surface_md, inv, dom, tts_surface)` so a
   downstream substrate can add its own prompt surface to `_SURFACE_TEXTS`
   without editing upstream. Refuses the builtin names (`opencode`, `codex`);
   if upstream adds a builtin surface, extend that refusal list when
   re-vendoring.

## Re-vendoring

1. Check out upstream at the new commit:
   `git clone https://github.com/google-research/harness2 /tmp/harness2 &&
   git -C /tmp/harness2 checkout <sha>`.
2. Run `reference/scripts/vendor_harness2.py --src /tmp/harness2 --sha <sha>`
   from the repository root. It replaces `harness2/` with exactly the files
   listed above, re-applies every `patches/*.diff`, and rewrites the `Commit`,
   `Vendored` and `Files` fields of this file. Add `--no-verify-sha` only when
   `--src` is not a git checkout.
3. If a patch no longer applies, fix the vendored file by hand, regenerate the
   patch with
   `diff -u --label a/<path> --label b/<path> <pristine> <patched>`, and rerun
   step 2 until it is clean.
4. Review the rest of this file by hand (exclusions, Python range, the
   `.env`/environment notes above) against the new upstream.
5. Run `reference/tests/test_vendored_harness2.py`; it pins the `Commit`
   field to `evals/evolve/_vendor.py:VENDORED_SHA`, so update that constant in
   the same commit.
