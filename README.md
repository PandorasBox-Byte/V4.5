````markdown
# EvoAI V4.5

Local semantic assistant using sentence-transformers.

Quick start

1. Activate the virtualenv:

```bash
source v4env/bin/activate
```

2. Run the assistant:

```bash
python launch_v4.py
```

Configuration

- `MODEL_NAME`: override model (default `all-MiniLM-L6-v2`)
- `DEVICE`: `cpu` or `cuda`
- `THREADS`: CPU threads to allocate
- `DISABLE_MEMORY_SAVE=1`: avoid persisting `data/memory.json`

Notes

- `v4env/` is included in the repository but should be removed from version control; consider recreating your venv with `instaallv4.5.sh`.
- Large binaries were migrated to Git LFS; collaborators must run `git lfs install` and `git lfs fetch --all`.EvoAI V4.5
=================

Local, CPU-first EvoAI core runtime (designed for Intel macOS).

Quick start
-----------

1. Create and activate the virtual environment (the repository includes `instaallv4.5.sh` to set up a pinned environment):

```bash
bash instaallv4.5.sh
source v4env/bin/activate
```

2. Run supervised launcher (preferred):

```bash
./run_v6.sh
```

This launches the engine and a lightweight ASCII monitor (in a `tmux` split if available, or a Terminal window on macOS).

Design notes
------------
- CPU-only: The system forces `device='cpu'` for `SentenceTransformer` and sets `torch` thread count.
- Deterministic dependencies (pinned): See `requirements.txt`. Do NOT upgrade these pinned HuggingFace packages.

Config via environment variables
-------------------------------
- `EVOAI_SIMILARITY_THRESHOLD` (default `0.7`) — similarity cutoff for reporting past user text.
- `EVOAI_MAX_MEMORY` (default `500`) — maximum retained memory entries; older entries are pruned.
- `EVOAI_TORCH_THREADS` (default `8`) — `torch.set_num_threads`.

### Training & fine‑tuning

A helper module (`core/trainer.py`) lets you fine-tune the embedding model or
LLM on custom data.  Models are saved under `data/finetuned-model` and
`data/llm_finetuned` by default.  When launching the engine you can point at a
fine‑tuned embedding model using `EVOAI_FINETUNED_MODEL` (this overrides
`EVOAI_MODEL`).

Here are a couple of simple CLI examples:

```bash
# embeddings (two pairs provided for demonstration)
python -m core.trainer embeddings --pairs "hi" "hello" "how are you" "good"

# LLM (uses dummy base model in tests)
python -m core.trainer llm --convs "hi" "hello" --base dummy
```

The `Trainer` API can also be used programmatically from Python.

Files added/modified
--------------------
- `run_v6.sh` — supervised launcher (tmux preferred).
- `core/launcher.py` — stabilized launcher that writes `data/engine.pid` and handles signals.
- `tools/monitor.py` — dependency-free ASCII monitor.
- `core/memory.py` — atomic save and pruning support.
- `core/engine_template.py` — incremental embedding cache, memory pruning, config.

Safety and reproducibility
-------------------------
- No network calls are made by the runtime itself, but `sentence-transformers` may download model weights on first run.
- Memory is saved atomically to `data/memory.json` to avoid corruption.

Next steps
----------
- Add persistent embedding cache (optional) and more structured indexing for large memories.
- Add unit tests and a CI workflow to preserve deterministic setup.

Cleanup
-------
- Obsolete installer and upgrade scripts plus legacy `launch_v4.py` were moved to `backup_removed_files/` on 2026-02-26 to keep the repository tidy.

````
