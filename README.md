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

> **Note:** the launcher no longer prompts for an OpenAI API key at startup.  If you need
> to supply or change the key, do so from the ASCII TUI after the engine has loaded (see
> instructions further down in this document).

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
- `EVOAI_USE_THESAURUS` (`1`/`0`, default `1`) — enable synonym expansion on replies and clarification prompts.
- `EVOAI_ENABLE_API` (`1`/`true`) — when set the engine will start a REST server on `EVOAI_API_ADDR`/`EVOAI_API_PORT` (defaults to `127.0.0.1:8000`).
- `EVOAI_AUTO_UPDATE_URL` — if present, the engine will fetch a JSON manifest from this URL and offer to upgrade itself (user must confirm before any changes are applied).


### OpenAI API key management

The engine no longer prompts before startup.  To set or clear the key during a
running session, use the ASCII TUI command:

```
:key <your_api_key>   # set (or change) the key
:key                 # clear existing key
```

A reminder message is displayed in the TUI history when no key is present, and
conversion to an active `OpenAIBackend` is attempted immediately when a key is
provided.  The value is stored only in the process environment and will be lost
when the engine exits.

### Thesaurus and clarification

When `EVOAI_USE_THESAURUS` is enabled the engine will consult a local
WordNet database (via NLTK) to expand replies with synonyms and alternate
phrasing.  It also watches for vague user inputs (`"it"`, "that", "thing"`) and
may prepend a question asking for clarification before answering.

The first time the feature is used the WordNet corpora will be downloaded
automatically (requires outbound network access).  If you prefer to disable
this behavioural layer you can unset the environment variable or set it to
`0`.

### Network awareness (disabled by default)

A placeholder network-scanning feature exists for conceptual completeness.
Set `EVOAI_ENABLE_NET_SCAN=permitted` to satisfy code paths; the engine will
still **not** probe the network and will merely emit a warning. This code is a
no-op to avoid any unauthorized reconnaissance and serves purely as a stub
for future authorised extensions.

### REST API & Home Assistant

A lightweight HTTP API is available to allow other programs (e.g. Home
Assistant) to interact with the engine.  Enable it by setting
`EVOAI_ENABLE_API=1` and optionally adjusting the address/port.  The server
exposes:

```
POST /chat   -> {"text": "..."}  returns {"reply": "..."}
GET  /status -> returns JSON snapshot of internal component status
```

While the engine is running you can launch the ASCII dashboard in a second
terminal or tmux pane:

```bash
# monitor via API
python core/monitor_ui.py --api http://127.0.0.1:8000
```

(The `run_v6.sh` helper already does this for you if tmux is installed.)

To integrate with Home Assistant you can use the `rest` sensor.  Example
configuration:

```yaml
sensor:
  - platform: rest
    name: EvolutionAI Status
    resource: http://localhost:8000/status
    json_attributes:
      - ready
      - embeddings_model
    value_template: "{{ value_json.ready }}"
```

The API server is intentionally minimal and runs in a background thread; it
should not be exposed to untrusted networks.

### Auto‑updater (opt‑in)

If `EVOAI_AUTO_UPDATE_URL` points to a JSON manifest of the form:

```json
{
  "version": "1.2",
  "files": [
    {"path": "core/engine_template.py", "url": "https://.../engine_template.py"}
  ]
}
```

the engine will periodically (currently on startup) fetch the manifest,
download the candidate files, run the local test suite against them, and then
prompt you with a diff.  Nothing is applied unless you type `y` at the prompt.
This mechanism is useful for keeping a self‑hosted deployment up-to-date, but
it must only point at a repository you trust.  The runtime never modifies your
code without explicit consent.

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
- **Network reconnaissance or automatic code modifications are not performed without explicit user permission.**  Any future features that "learn about surroundings" or fetch updates require configuration flags and a conscious opt-in; the engine will not scan devices or self-modify autonomously.

Next steps
----------
- Add persistent embedding cache (optional) and more structured indexing for large memories.
- Add unit tests and a CI workflow to preserve deterministic setup.

Cleanup
-------
- Obsolete installer and upgrade scripts plus legacy `launch_v4.py` were moved to `backup_removed_files/` on 2026-02-26 to keep the repository tidy.

Automated update (2026-02-26)
-----------------------------
The following changes were recorded automatically and committed to the repository.

Modified files:
- README.md
- core/engine_template.py
- core/launcher.py
- core/plugin_manager.py
- data/memory.json
- run_v6.sh
- tests/run_all.sh
- tests/test_engine.py
- tests/test_plugins.py

New / untracked files detected locally:
- .venv311/ (directory)
- Start_Engine.sh
- core/__init__.py
- core/api_server.py
- core/auto_updater.py
- core/language_utils.py
- core/monitor_ui.py
- core/network_scanner.py
- core/openai_backend.py
- core/self_repair.py
- core/tui.py
- core/update_test.txt
- plugins/__init__.py
- pyproject.toml
- scripts/persist_openai_key.sh
- setup.cfg
- tests/test_language.py
- tests/test_launcher.py
- tests/test_monitor_ui.py
- tests/test_tui.py
- tests/test_updater.py
- tools/package_mac_app.sh
- v311env/ (directory)

Notes:
- If some of the untracked files should be included in version control, review them and run `git add <file>` before committing.
- Sensitive files (API keys, virtualenvs) should not be committed — confirm before adding.

````
