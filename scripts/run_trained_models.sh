#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x ".venv311/bin/python" ]; then
  echo "Missing .venv311. Create it first with Python 3.11 and install training deps." >&2
  exit 1
fi

export EVOAI_FINETUNED_MODEL="data/finetuned-model"
export EVOAI_LLM_MODEL="data/llm_finetuned"
export EVOAI_RESPONDER="smart"

exec ./.venv311/bin/python core/launcher.py
