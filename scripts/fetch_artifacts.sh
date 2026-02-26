#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--model <local-path-or-url>] [--emb <local-path-or-url>]
Environment: MODEL_URL and EMB_URL can be set to download when not passing args.
Examples:
  MODEL_URL=https://.../pytorch_model.bin EMB_URL=https://.../embeddings.pt ./scripts/fetch_artifacts.sh
  ./scripts/fetch_artifacts.sh --model /path/to/pytorch_model.bin --emb /path/to/embeddings.pt
EOF
}

MODEL_SRC=""
EMB_SRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_SRC="$2"; shift 2;;
    --emb) EMB_SRC="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

mkdir -p data/finetuned-model

download_or_copy() {
  local src="$1" dest="$2"
  if [[ -z "$src" ]]; then
    return 0
  fi
  if [[ "$src" =~ ^https?:// ]]; then
    echo "Downloading $src -> $dest"
    curl -L --fail --retry 3 -o "$dest" "$src"
  else
    echo "Copying $src -> $dest"
    cp "$src" "$dest"
  fi
}

if [[ -z "$MODEL_SRC" && -n "${MODEL_URL:-}" ]]; then
  MODEL_SRC="$MODEL_URL"
fi
if [[ -z "$EMB_SRC" && -n "${EMB_URL:-}" ]]; then
  EMB_SRC="$EMB_URL"
fi

download_or_copy "$MODEL_SRC" "data/finetuned-model/pytorch_model.bin"
download_or_copy "$EMB_SRC" "data/embeddings.pt"

echo "Done. artifacts are in data/finetuned-model/ and data/embeddings.pt"
