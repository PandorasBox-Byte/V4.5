#!/usr/bin/env bash
set -euo pipefail

PY="./v4env/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

RUNS=${1:-3}

for i in $(seq 1 "$RUNS"); do
  echo "\n=== Run $i/$RUNS ==="
  echo "Clearing persistent state (memory + embeddings)..."
  rm -f data/memory.json data/embeddings.pt

  echo "Running unit tests (verbosity)..."
  "$PY" -m unittest discover -s tests -v || {
    echo "Tests failed on run $i" >&2
    exit 1
  }

done

echo "\nAll $RUNS runs completed successfully." 
