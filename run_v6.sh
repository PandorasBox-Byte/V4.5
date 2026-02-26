#!/usr/bin/env bash

# Run EvoAI V4.5 supervised with dual-terminal monitoring.
# Behavior:
# - Prefer tmux (split panes). If absent, fallback to opening a second Terminal window (macOS).
# - Activates project venv, launches engine (via core/launcher.py) and monitor (tools/monitor.py).

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

VENV="$BASE_DIR/v4env/bin/activate"
PIDFILE="$BASE_DIR/data/engine.pid"
TMUX_SESSION="evoai_v4"

if [ ! -f "$VENV" ]; then
  echo "Virtual environment not found at $VENV"
  echo "Run instaallv4.5.sh or create venv first."
  exit 1
fi

source "$VENV"

cleanup_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      tmux kill-session -t "$TMUX_SESSION" || true
    fi
  fi
}

trap 'cleanup_tmux; exit 0' INT TERM

if command -v tmux >/dev/null 2>&1; then
  cleanup_tmux
  # enable the REST API so the monitor UI can poll it
  export EVOAI_ENABLE_API=1
  export EVOAI_API_ADDR=127.0.0.1
  export EVOAI_API_PORT=8000
  tmux new-session -d -s "$TMUX_SESSION" "cd '$BASE_DIR' && source '$VENV' && python3 core/launcher.py"
  tmux split-window -h -t "$TMUX_SESSION" "cd '$BASE_DIR' && source '$VENV' && python3 core/monitor_ui.py --api http://127.0.0.1:8000"
  echo "Attaching to tmux session '$TMUX_SESSION' (Ctrl+b then d to detach)."
  tmux attach -t "$TMUX_SESSION"
  cleanup_tmux
  exit 0
else
  # Fallback: open a new Terminal.app window for the monitor (macOS)
  if command -v osascript >/dev/null 2>&1; then
    # enable REST API for monitor UI
    export EVOAI_ENABLE_API=1
    export EVOAI_API_ADDR=127.0.0.1
    export EVOAI_API_PORT=8000
    MON_CMD="cd '$BASE_DIR' && source '$VENV' && python3 core/monitor_ui.py --api http://127.0.0.1:8000"
    osascript -e "tell application \"Terminal\" to do script \"$MON_CMD\""
    # Run engine in current terminal (so user interacts directly)
    python3 core/launcher.py
    # When engine exits, attempt to remove pidfile
    if [ -f "$PIDFILE" ]; then
      rm -f "$PIDFILE" || true
    fi
    exit 0
  else
    echo "Neither tmux nor osascript available. Please run monitor manually in another terminal:" >&2
    echo "  source v4env/bin/activate && python3 tools/monitor.py --pidfile data/engine.pid" >&2
    python3 core/launcher.py
  fi
fi
