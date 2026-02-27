# READMECODE (Technical Context + Change History)

This document is a technical handoff/reference for developers and AI assistants.
It describes what EvoAI V6.0.0 (V6) is, how it starts, model/data flow, and major fixes made in this workspace.

## Versioning rule (project contract)

- SemVer shape: `MAJOR.MINOR.PATCH`
- `MAJOR`: increment for major structural changes (non-engine architecture shifts)
- `MINOR`: increment for feature-level/minor changes
- `PATCH`: increment for bug fixes
- Source of truth files: `version_tally.json` and `setup.cfg`
- Bump utility: `scripts/bump_version.py`

## Release history (major/minor/patch)

- `6.0.0` **MAJOR**: deep optimization pass across runtime persistence, startup latency, plugin/API routing efficiency, and startup script consolidation.
- `5.0.0` **MAJOR**: backend migration + decision layer integration + startup/runtime contract changes.
- `5.0.1` **PATCH**: TUI version label and startup checklist grid rendering updates.
- `5.1.0` **MINOR**: startup git-tag updater, loading-screen update phase, and restart-on-success behavior.
- `5.1.1` **PATCH**: startup GitHub token persistence and reuse/change/skip prompt flow.
- `5.1.2` **PATCH**: updater loop prevention after stash-pop by re-normalizing release version files.
- `5.1.3` **PATCH**: launcher token prompt handling fixed for non-interactive mode and pre-set env tokens.
- `5.1.4` **PATCH**: updater normalization extended to tracked non-runtime files to prevent partial updates.

## Latest major change (6.0.0)

- Updated `core/engine_template.py` to reduce repeated training-capture file reads and batch memory/embedding persistence with timed/turn-based flush plus process-exit flush.
- Updated `core/launcher.py` startup flow so engine loading in interactive mode does not block on updater completion.
- Updated `core/language_utils.py` with cached WordNet readiness and cached synonym lookup results.
- Updated `plugins/knowledge_plugin.py` with low-cost keyword prefiltering before semantic similarity scoring.
- Updated `core/api_server.py` route matching path to avoid per-request URL parsing overhead.
- Updated `Start_Engine.sh` to avoid unconditional pip toolchain upgrade on every launch.
- Consolidated script/tool wrappers in `scripts/run_trained_models.sh` and `tools/monitor.py`, and removed `backup_removed_files/` legacy artifacts.

## Latest minor change (5.1.0)

- Added startup git updater pipeline in `core/auto_updater.py` based on remote SemVer tags.
- Added update phase wiring in `core/launcher.py` before engine construction.
- Added dedicated update loading screen in `core/tui.py` and launcher restart action after successful update.
- Update detection does not change local version number automatically; release number still comes from `version_tally.json` / `setup.cfg`.

## Latest patch change (5.1.1)

- Fixed startup token persistence bug in `core/launcher.py` so first entered token is saved.
- Added startup choice flow to reuse saved token, change token, or skip token before boot.
- Added launcher tests for persistence and saved-token change behavior.

## Latest patch change (5.1.2)

- Fixed updater loop in `core/auto_updater.py` by enforcing release version files from the target tag after `stash pop` when needed.
- Added launcher-side update guard in `core/launcher.py` to avoid repeated restart/update cycles for the same target version.
- Added updater regression test in `tests/test_updater.py` for post-update version normalization.

## Latest patch change (5.1.3)

- Updated `core/launcher.py` to skip token prompting in non-interactive startup paths.
- Updated `core/launcher.py` to skip prompting when `GITHUB_TOKEN`/`GH_TOKEN` is already set.
- Added launcher tests for non-interactive startup behavior in `tests/test_launcher.py`.

## Latest patch change (5.1.4)

- Updated `core/auto_updater.py` to normalize tracked non-runtime files after `stash pop` using the target tag.
- Runtime-local paths under `data/` are excluded from forced restore.
- Added updater regression test in `tests/test_updater.py` for non-runtime tracked file normalization.

## 1) System purpose

EvoAI is a local assistant runtime with:
- conversation memory persisted to disk
- similarity-based recall over prior user messages
- optional local LLM generation
- optional external backend (GitHub Models by default)
- plugin short-circuit handling
- TUI + optional API server + updater hooks

Primary runtime entrypoint:
- `Start_Engine.sh` -> `core/launcher.py` -> `core/tui.py` + `core/engine_template.py`

## 2) Canonical start path

Use only:
- `Start_Engine.sh`

Status:
- `run_v6.sh` was removed intentionally.

`Start_Engine.sh` now sets practical defaults:
- `.venv311`
- `EVOAI_AUTO_MODEL_DISCOVERY=1`
- candidate lists for embedding + LLM model directories
- `EVOAI_RESPONDER=smart`
- `EVOAI_USE_THESAURUS=0`
- `EVOAI_STARTUP_STDOUT_LOGS=0`
- `EVOAI_BACKEND_PROVIDER=github`

Launcher startup order (`core/launcher.py`):
- prompts for `GITHUB_TOKEN`/`GH_TOKEN` before engine/model loading begins
- then constructs loader/engine and hands control to TUI
- TUI still supports `:key` for runtime token updates/clears

## 3) Model loading strategy

Engine model resolution now supports automatic fallback chains.

Embeddings model order:
1. `EVOAI_FINETUNED_MODEL` (if set)
2. `EVOAI_FINETUNED_MODEL_CANDIDATES` (colon-separated)
3. auto-discovered local dirs (if enabled)
4. base model name from `EVOAI_MODEL` (default `all-MiniLM-L6-v2`)

LLM model order:
1. `EVOAI_LLM_MODEL` (if set)
2. `EVOAI_LLM_MODEL_CANDIDATES` (colon-separated)
3. auto-discovered local dirs (if enabled)

Selected models are exposed via engine status keys:
- `embeddings_selected`
- `llm_selected`

Decision layer status keys:
- `decision_layer`
- `decision_depth`
- `decision_width`
- `decision_last_action`
- `decision_last_latency_ms`
- `decision_last_reason`

Decision layer module:
- `core/decision_policy.py`

External backend module:
- `core/github_backend.py`

## 4) Memory and embeddings behavior

Core files:
- `core/memory.py`
- `core/embeddings_cache.py`
- `core/engine_template.py`

Current guarantees:
- corrupt/invalid memory JSON no longer crashes startup (`load_memory` is safe)
- memory pruning is now applied in-memory as well as on-disk
- corpus text list and embedding cache are pruned in sync when old entries roll off
- embedding cache mismatch with memory size is cleared and rebuilt safely

## 5) Response pipeline (high level)

Responder mode (`simple` vs `smart`) comes from `EVOAI_RESPONDER`.

Smart responder order:
1. repetition handling
2. plugin handling (`can_handle`/`handle`)
3. GitHub Models conversational routing when enabled and authenticated (`EVOAI_COPILOT_CHAT=1`)
4. similarity memory hint
5. local LLM generation (if configured)
6. fallback simple echo + memory record

Conversation capture:
- every recorded high-quality user/assistant turn is appended to `data/custom_conversations.json`
- capture metadata is tracked in `data/conversation_capture_meta.json`
- startup launcher can check metadata threshold and trigger local retraining when `EVOAI_STARTUP_AUTO_TRAIN=1` (opt-in)

Post-processing:
- optional clarification prepend (`language_utils.clarify_if_ambiguous`)
- optional thesaurus rewrite (`language_utils.enhance_text`)

## 6) Language stability fixes

In `core/language_utils.py`:
- ambiguity detection is stricter to reduce false triggers (e.g., routine phrasing)
- thesaurus enhancement preserves quoted spans (prevents mutation of echoed user text)
- short grammar words are less aggressively replaced

These changes were added to reduce "scrambled" output and over-clarification.

## 7) Startup UI / self-test behavior

Files:
- `core/tui.py`
- `core/self_repair.py`

Behavior now:
- startup self-test has `mode="startup"` (fast checks) and `mode="repair"` (deeper path)
- TUI loading screen shows live self-test message and per-check pass/fail list
- startup stdout prints are suppressed during loader mode to avoid overlapping curses output
- API startup in TUI loader mode runs in quiet mode to avoid screen overlap/corruption
- redraw path now clears stale progress/status rows before repainting

Key env toggles:
- `EVOAI_STARTUP_SELF_TEST`
- `DISABLE_SELF_REPAIR`
- `EVOAI_REPAIR_SMOKE_ENGINE`
- `EVOAI_STARTUP_STDOUT_LOGS`

Decision layer env toggles:
- `EVOAI_ENABLE_DECISION_LAYER`
- `EVOAI_DECISION_DEPTH`
- `EVOAI_DECISION_WIDTH`
- `EVOAI_DECISION_TIMEOUT_MS`
- `EVOAI_DECISION_ALLOW_AUTONOMY`
- `EVOAI_DECISION_MAX_PROACTIVE_PER_TURN`
- `EVOAI_DECISION_AUTONOMY_COOLDOWN`

External backend env toggles:
- `EVOAI_BACKEND_PROVIDER` (`github`)
- `GITHUB_TOKEN` / `GH_TOKEN`
- `GITHUB_MODEL`
- `GITHUB_MODELS_ENDPOINT`

## 8) Training workflow and outputs

Training scripts:
- `scripts/train_personalization.py` (full embeddings + LLM dataset build)
- `scripts/train_llm_only.py` (LLM only)

Trainer implementation:
- `core/trainer.py`

Decision policy training:
- `core/trainer.py::train_decision_policy(...)`
- output dir default: `data/decision_policy`
- saved artifacts: `model.pt` + `metadata.json` (or `fallback.json` when torch unavailable)
- runtime load path: `EVOAI_DECISION_MODEL_PATH` (default `data/decision_policy/model.pt`)

Important implementation notes:
- LLM training now saves tokenizer files (`tokenizer.json`, vocab/merges/config) along with model
- embedding training has a robust fallback path if trainer-version mismatch occurs
- script import path safety was added so scripts run from repo root reliably

Current artifact locations:
- embeddings: `data/finetuned-model`
- local LLM: `data/llm_finetuned`

## 9) Known constraints

- Python 3.14 environment is not suitable for full PyTorch-based training here.
- Use `.venv311` for real training/inference with torch/transformers.
- Tiny local LLM (`sshleifer/tiny-gpt2`) is fast but low quality; useful for smoke tests, not high-quality generation.

## 10) Validation references

Representative validation done after recent changes:
- launcher tests (`tests/test_launcher.py`) pass
- engine tests (`tests/test_engine.py`) pass
- language tests (`tests/test_language.py`) pass
- trainer tests (`tests/test_trainer.py`) pass
- full suite via `tests/run_all.sh` passes

Live backend validation script:
- `scripts/live_github_backend_test.py` performs an authenticated GitHub model smoke test when token is present

## 11) Suggested future upgrades

1. Replace tiny GPT-2 with a better local model for usable generation quality.
2. Add quality benchmark prompts and save baseline outputs for regression tracking.
3. Add integration test that boots via `Start_Engine.sh` and checks selected model status.
4. Add model metadata manifest (`data/model_manifest.json`) for explicit provenance/versioning.

## 12) Documentation maintenance contract

This repository uses a two-file documentation contract:
- `README.md` = user-facing operational truth
- `READMECODE.md` = technical/system truth

Whenever engine behavior changes, update both files in the same work item.

Minimum required updates per change:
1. What changed
2. Why it changed
3. Startup/runtime impact
4. Env var or model path impact
5. Test/validation impact

If a change does not affect one of those areas, explicitly note "no change" in the relevant section.
