#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export EVOAI_FINETUNED_MODEL="${EVOAI_FINETUNED_MODEL:-data/finetuned-model}"
export EVOAI_LLM_MODEL="${EVOAI_LLM_MODEL:-data/llm_finetuned}"
export EVOAI_RESPONDER="${EVOAI_RESPONDER:-smart}"

exec "$ROOT_DIR/Start_Engine.sh" "$@"
