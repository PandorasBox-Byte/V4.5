# READMECODE (Technical Context + Change History)

This document is a technical handoff/reference for developers and AI assistants.
It describes what EvoAI V7.3.0 (V7) is, how it starts, model/data flow, and major fixes made in this workspace.

## Versioning rule (project contract)

- SemVer shape: `MAJOR.MINOR.PATCH`
- `MAJOR`: increment for major structural changes (non-engine architecture shifts)
- `MINOR`: increment for feature-level/minor changes
- `PATCH`: increment for bug fixes
- Source of truth files: `version_tally.json` and `setup.cfg`
- Bump utility: `scripts/bump_version.py`

## Release history (major/minor/patch)

- `7.0.0` **MAJOR**: autonomous governance architecture release with safety gating, policy controls, tested-apply orchestration, audit/governance APIs, and runtime policy closed-loop adaptation.
- `7.0.1` **PATCH**: updater bug fix: detect version file inconsistencies between `version_tally.json` and `setup.cfg`, use lower version when divergent.
- `7.0.2` **PATCH**: concrete updater: added file verification system with `verify_complete_state()` and auto-repair via `repair_to_remote_state()`.
- `7.0.3` **PATCH**: updater protection: excluded `core/auto_updater.py` from repairs (redundancy guarantee); added standalone `repair.sh` CLI script.
- `7.0.4` **PATCH**: git metadata filtering: excluded `.gitignore`, `.gitattributes`, `.github/` from file verification/repair for cleaner standalone distributions.
- `7.1.0` **MINOR**: CodeAssistant orchestrator: autonomous coding workflows with decision routing, CodeIntel analysis, LLM generation, validation, and safety gates.
- `7.2.0` **MINOR**: Cleanup and optimization: unified core architecture with consolidated exports, removed temp/debug files, enhanced Engine documentation, optimized repository structure.
- `7.3.0` **MINOR**: Virtual environment consolidation: removed redundant venvs (.venv, v4env), standardized on .venv311 (Python 3.11) for PyTorch/transformers compatibility, freed 229MB.
- `6.0.0` **MAJOR**: deep optimization pass across runtime persistence, startup latency, plugin/API routing efficiency, and startup script consolidation.
- `5.0.0` **MAJOR**: backend migration + decision layer integration + startup/runtime contract changes.
- `5.0.1` **PATCH**: TUI version label and startup checklist grid rendering updates.
- `5.1.0` **MINOR**: startup git-tag updater, loading-screen update phase, and restart-on-success behavior.
- `5.1.1` **PATCH**: startup GitHub token persistence and reuse/change/skip prompt flow.
- `5.1.2` **PATCH**: updater loop prevention after stash-pop by re-normalizing release version files.
- `5.1.3` **PATCH**: launcher token prompt handling fixed for non-interactive mode and pre-set env tokens.
- `5.1.4` **PATCH**: updater normalization extended to tracked non-runtime files to prevent partial updates.

## Latest minor change (7.3.0)

- Removed redundant virtual environments:
  - Deleted `.venv` (Python 3.14, 217MB) - not compatible with PyTorch
  - Deleted `v4env` (Python 3.14, 12MB) - legacy unused venv
  - Kept `.venv311` (Python 3.11, 1.2GB) - canonical environment for ML training
- Updated all venv references:
  - `tests/run_all.sh`: now checks for `.venv311` first
  - `README.md`: updated test/training examples to use `.venv311`
  - `.gitattributes`: removed v4env LFS rule
  - `core/autonomy_tools.py`: enhanced to exclude all `.venv*` directories from code analysis
- Benefits:
  - Freed 229MB disk space
  - Simplified development environment (single Python 3.11 venv)
  - Eliminated confusion about which venv to use
  - PyTorch/transformers training guaranteed to work

## Previous minor change (7.2.0)

- Unified core architecture in `core/__init__.py`:
  - Consolidated module exports for integrated system access
  - Simplified imports: Engine, DecisionPolicy, SafetyGate, CodeAssistant directly from core
  - All modules remain directly importable from their original paths
  - Compatibility shims preserved for huggingface_hub
- Enhanced Engine class documentation:
  - Comprehensive architecture overview in Engine docstring
  - Integration flow documentation (query → decision → safety → action → audit → persist)
  - Environment variable reference
  - Clear component responsibilities
- Repository cleanup:
  - Removed all temp files from data/ (tmp*, engine.pid)
  - Removed debug artifacts (llm_finetuned_debug/, update_test.txt)
  - Removed unused modules (network_scanner.py, memory.example.json)
  - Cleaned __pycache__ from source directories
  - Enhanced .gitignore with comprehensive patterns
- Optimization:
  - Removed network_scanner stub and related engine code
  - Streamlined module import structure
  - Improved repository cleanliness for standalone deployment

## Previous minor change (7.1.0)

- Added `core/code_assistant.py` implementing `CodeAssistant` orchestrator class for autonomous coding workflows.
- CodeAssistant coordinates: analyze (CodeIntel) → generate (LLM) → validate (TestedApply) → apply (safety-gated).
- Updated `core/decision_policy.py` to recognize coding intents (fix, debug, implement, refactor, optimize, review) and route to `code_assist` action.
- Updated `core/engine_template.py`:
  - Added `CodeAssistant` import and initialization
  - Added `code_assist` action handler in `respond()` method
  - Integrated with decision policy routing and autonomy governance
  - Added status tracking for code assist operations
- Maintains broad-spectrum assistant capabilities (memory, research, general LLM) while specializing coding operations.
- Decision policy keywords for coding intents: "fix", "bug", "debug", "error", "implement", "feature", "refactor", "rewrite", "optimize", "improve", "review code", "generate code", "generate function", "generate test".

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

## 10) CodeAssistant orchestrator (autonomous coding)

New module:
- `core/code_assistant.py`

Architecture:
- `CodeAssistant` class orchestrates a complete coding workflow: analyze → generate → validate → apply
- Coordinates with `CodeIntelToolkit` for codebase analysis and hotspot detection
- Integrates with engine's LLM for candidate code generation
- Uses `TestedApplyOrchestrator` for validation and rollback
- Subject to `SafetyGate` checks before generation and apply steps

Decision routing:
- `core/decision_policy.py` updated with `code_assist` action
- Detects coding intents: "fix", "bug", "debug", "error", "implement", "feature", "refactor", "rewrite", "optimize", "improve", "review code", "generate code", "write function", "generate test"
- Routes matching queries to `code_assist` with confidence threshold

Engine integration (`core/engine_template.py`):
- `CodeAssistant` instantiated during engine initialization
- `respond()` method handles `code_assist` action with full workflow execution
- Status tracking: `code_assist_step`, `code_assist_last_reason`, `code_assist_matches`
- Audit events recorded for all code assist outcomes

Workflow execution (`CodeAssistant.workflow()`):
- Step 1: Analyze user query against local codebase (CodeIntel)
- Step 2: Check generation allowed via SafetyGate
- Step 3: Generate code candidate (LLM or prompt for user)
- Step 4: Check apply allowed via SafetyGate
- Step 5: Validate candidate with TestedApply (unit tests, benchmark)
- Step 6: Return for user review or auto-apply (if enabled)

Safety guarantees:
- Generation blocked if `EVOAI_DECISION_ALLOW_AUTONOMY=0` or SafetyGate rejects
- Apply blocked if `EVOAI_AUTONOMY_PAUSED=1` or SafetyGate rejects
- All actions subject to unified autonomy governance policy

Status env toggles:
- `EVOAI_AUTONOMY_PAUSED` (pause all autonomous actions including code assist)
- `EVOAI_AUTONOMY_BUDGET_MAX` (max autonomous actions per session)
- `EVOAI_AUTONOMY_BUDGET_REMAINING` (remaining budget)

## 11) Known constraints

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
