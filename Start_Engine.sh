#!/usr/bin/env bash
# Start_Engine.sh — convenience script to create a venv (if missing) and run the launcher

set -euo pipefail

# Resolve repository root (script is expected to live in repo root)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

# Source persisted env file if present (created by scripts/persist_github_token.sh)
if [ -f "$HOME/.evoai_env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.evoai_env"
fi

VENV=".venv311"

# Find a suitable python executable (prefer 3.11)
PYTHON_BIN=""
for p in python3.11 python3 python; do
  if command -v "$p" >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v "$p")
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "No python interpreter found (tried python3.11, python3, python)." >&2
  exit 1
fi

# Create venv if missing
CREATED_VENV=0
if [ ! -d "$VENV" ]; then
  echo "Creating virtualenv $VENV using $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv "$VENV"
  CREATED_VENV=1
fi

# Ensure pip tooling is up-to-date only on first create or explicit request
if [ "$CREATED_VENV" -eq 1 ] || [ "${EVOAI_FORCE_TOOLCHAIN_UPGRADE:-0}" = "1" ]; then
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
fi

# Install requirements only on venv creation or when explicitly forced
if [ "$CREATED_VENV" -eq 1 ] || [ "${EVOAI_FORCE_INSTALL:-0}" = "1" ]; then
  if [ -f requirements.txt ]; then
    echo "Installing requirements into $VENV (this may take a while)..."
    "$VENV/bin/pip" install -r requirements.txt
  fi
fi

# Export PYTHONPATH so the package loads from the repo root
export PYTHONPATH="$REPO_ROOT"

# Auto-select available local models so users do not need manual switching.
export EVOAI_AUTO_MODEL_DISCOVERY="${EVOAI_AUTO_MODEL_DISCOVERY:-1}"
export EVOAI_FINETUNED_MODEL_CANDIDATES="${EVOAI_FINETUNED_MODEL_CANDIDATES:-$REPO_ROOT/data/finetuned-model:$REPO_ROOT/data/llm_finetuned_debug}"
export EVOAI_LLM_MODEL_CANDIDATES="${EVOAI_LLM_MODEL_CANDIDATES:-$REPO_ROOT/data/llm_finetuned:$REPO_ROOT/data/llm_finetuned_debug}"

# Stable runtime defaults.
export EVOAI_RESPONDER="${EVOAI_RESPONDER:-smart}"
export EVOAI_USE_THESAURUS="${EVOAI_USE_THESAURUS:-0}"
export EVOAI_STARTUP_STDOUT_LOGS="${EVOAI_STARTUP_STDOUT_LOGS:-0}"
export EVOAI_BACKEND_PROVIDER="${EVOAI_BACKEND_PROVIDER:-github}"
# Ensure TUI launches by default, even if parent shell exported a different value.
export EVOAI_FORCE_TUI="1"
# Keep brain monitor enabled by default.
export EVOAI_ENABLE_BRAIN_MONITOR="${EVOAI_ENABLE_BRAIN_MONITOR:-1}"

echo "Starting EvoAI launcher (TUI if available). Press q or Ctrl-C to quit."

# Exec the launcher so signals are forwarded and TTY is preserved
exec "$VENV/bin/python" core/launcher.py "$@"
