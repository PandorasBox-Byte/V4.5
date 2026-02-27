# EvoAI V5.0.1 (V5)

Local assistant runtime with memory, semantic similarity retrieval, optional local LLM generation, optional GitHub Models backend, plugin support, API server, and startup self-test.

## Versioning tally system

- Current release: `5.0.1` (`V5`)
- Version format: `MAJOR.MINOR.PATCH`
- `MAJOR`: increment for major structural changes (non-engine architecture shifts)
- `MINOR`: increment for feature-level/minor changes
- `PATCH`: increment for bug fixes

Use:

```bash
python scripts/bump_version.py --change major --reason "describe the structural change"
python scripts/bump_version.py --change minor --reason "describe the feature update"
python scripts/bump_version.py --change patch --reason "describe the bug fix"
```

This updates both `version_tally.json` and `setup.cfg`.

## Latest patch summary (5.0.1)

- Added version number in the TUI bottom-left corner.
- Reworked startup self-test checklist from a list into a cleaner grid layout.

## Quick start (single launcher)

Use `Start_Engine.sh` as the canonical start path.

```bash
./Start_Engine.sh
```

This script:
- creates/uses `.venv311`
- installs requirements when the venv is first created (or when `EVOAI_FORCE_INSTALL=1`)
- sets `PYTHONPATH`
- prompts for `GITHUB_TOKEN` in terminal before model loading starts (skip allowed)
- enables automatic local model discovery
- defaults external backend provider to GitHub Models
- starts `core/launcher.py` (ASCII TUI)

`run_v6.sh` has been removed by design.

## Runtime behavior

- CPU-first default: `EVOAI_DEVICE=cpu`
- Fallback-safe startup: missing optional ML dependencies do not crash startup
- Smart responder support with memory + similarity recall
- Decision layer in core (`EVOAI_ENABLE_DECISION_LAYER`, default on)
- External backend provider defaults to GitHub Models (`EVOAI_BACKEND_PROVIDER=github`)
- Normal chat can use GitHub Models directly when token is present (`EVOAI_COPILOT_CHAT=1`, default)
- Startup self-test in TUI uses fast checks (no full pytest by default)
- Startup log suppression in loader mode avoids curses screen corruption
- TUI loader/API startup now runs in quiet mode to prevent overlapping text output
- High-quality user/assistant turns are captured to `data/custom_conversations.json` for future local training

## Model auto-discovery (no manual model switching required)

When `EVOAI_AUTO_MODEL_DISCOVERY=1` (default in `Start_Engine.sh`), engine startup tries available candidates automatically.

Primary variables:
- `EVOAI_FINETUNED_MODEL`
- `EVOAI_LLM_MODEL`

Candidate chain variables:
- `EVOAI_FINETUNED_MODEL_CANDIDATES`
- `EVOAI_LLM_MODEL_CANDIDATES`

Default discovered local paths include:
- `data/finetuned-model`
- `data/llm_finetuned`
- `data/llm_finetuned_debug`

## Useful environment variables

- `EVOAI_BACKEND_PROVIDER` (`github`, default `github`)
- `GITHUB_TOKEN` (or `GH_TOKEN`) for external backend auth
- `EVOAI_COPILOT_CHAT` (`1`/`0`, default `1`) to use GitHub backend for normal conversations when available
- `GITHUB_MODEL` (default `gpt-4o-mini`)
- `GITHUB_MODELS_ENDPOINT` (default `https://models.inference.ai.azure.com/chat/completions`)
- `EVOAI_TRAINING_DATA_PATH` (default `data/custom_conversations.json`)
- `EVOAI_TRAINING_META_PATH` (default `data/conversation_capture_meta.json`)
- `EVOAI_STARTUP_AUTO_TRAIN` (`1`/`0`, default `0`, opt-in)
- `EVOAI_STARTUP_TRAIN_THRESHOLD` (default `80` new captured turns)
- `EVOAI_STARTUP_TRAIN_TIMEOUT` (default `1200` seconds)
- `EVOAI_ENABLE_DECISION_LAYER` (`1`/`0`)
- `EVOAI_DECISION_DEPTH` (default `12`)
- `EVOAI_DECISION_WIDTH` (default `512`)
- `EVOAI_DECISION_TIMEOUT_MS` (default `40`)
- `EVOAI_DECISION_MODEL_PATH` (default `data/decision_policy/model.pt`)
- `EVOAI_DECISION_ALLOW_AUTONOMY` (`1`/`0`)
- `EVOAI_DECISION_MAX_PROACTIVE_PER_TURN` (default `2`)
- `EVOAI_DECISION_AUTONOMY_COOLDOWN` (default `2`)
- `EVOAI_RESPONDER` (`simple` or `smart`)
- `EVOAI_USE_THESAURUS` (`1`/`0`)
- `EVOAI_STARTUP_STDOUT_LOGS` (`1` shows startup prints, `0` keeps loader clean)
- `EVOAI_SIMILARITY_THRESHOLD` (default `0.7`)
- `EVOAI_MAX_MEMORY` (default `500`)
- `EVOAI_TORCH_THREADS` (default `8`)
- `EVOAI_ENABLE_API`, `EVOAI_API_ADDR`, `EVOAI_API_PORT`
- `EVOAI_AUTO_UPDATE_URL`
- `EVOAI_STARTUP_SELF_TEST` (`1`/`0`)

## Testing

```bash
bash tests/run_all.sh

# live backend smoke test (requires GITHUB_TOKEN or GH_TOKEN)
./.venv/bin/python scripts/live_github_backend_test.py
```

## Training (recommended path on this Mac)

Use Python 3.11 environment (`.venv311`) for real ML training.

```bash
python3.11 -m venv .venv311
./.venv311/bin/python -m pip install --upgrade pip
./.venv311/bin/pip install torch transformers==4.41.2 accelerate sentence-transformers==5.2.3 huggingface_hub==0.36.2 datasets nltk requests
./.venv311/bin/python scripts/train_personalization.py --emb-epochs 2 --llm-epochs 2 --decision-epochs 40
```

Artifacts:
- embeddings: `data/finetuned-model`
- local LLM: `data/llm_finetuned`
- decision policy: `data/decision_policy/model.pt`

## Notes on local LLM quality

Current default training base model is intentionally lightweight (`sshleifer/tiny-gpt2`) to keep CPU training quick. It loads correctly but output quality can be limited compared to larger models.

## Troubleshooting

- Engine starts but models are not loaded:
	- Confirm model folders exist: `data/finetuned-model` and/or `data/llm_finetuned`
	- Start via `./Start_Engine.sh` so auto-discovery env vars are set
	- Re-run training in `.venv311` if model folders were removed by tests/cleanup
- Startup UI text overlaps or looks corrupted:
	- Keep `EVOAI_STARTUP_STDOUT_LOGS=0` (default in `Start_Engine.sh`)
- External backend unavailable:
	- Provide `GITHUB_TOKEN` at startup prompt or set it in environment (`~/.evoai_env`)
	- You can still use `:key <token>` in TUI to update/clear during runtime
	- Confirm `EVOAI_BACKEND_PROVIDER=github`
- Local LLM output looks low quality:
	- This is expected with tiny GPT-2; prefer embeddings-first mode or train a stronger local base model
- Training fails in `.venv` (Python 3.14):
	- Use `.venv311` for training and model loading with torch/transformers

## Documentation sync policy

For every engine change (runtime, training, startup, memory, model-loading, tests):
- update this file (`README.md`) with user-facing behavior changes
- update `READMECODE.md` with technical/internal details and rationale

These two files are the canonical operational and technical references for this workspace.

## Project hygiene

Generated artifacts and caches are excluded from version control:
- virtual environments
- `__pycache__`
- `.pytest_cache`
- model/checkpoint output directories
