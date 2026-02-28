#!/bin/bash
# EvoAI Updater Repair Script
# Manual CLI interface to verify and repair the workspace
# Usage: ./repair.sh [command] [options]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_ok() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Function to verify local state
verify_state() {
    local target_tag="${1:-HEAD}"
    print_status "Verifying local state against ${target_tag}..."
    
    python3 << PYTHON_SCRIPT
import sys
sys.path.insert(0, '.')
from core.auto_updater import verify_complete_state

result = verify_complete_state('$target_tag')
if result.get('error'):
    print('Error:', result.get('error'))
    sys.exit(1)

missing = result.get('missing_files', [])
divergent = result.get('divergent_files', [])
extra = result.get('extra_files', [])

if missing:
    print(f'\nMissing files ({len(missing)}):')
    for f in missing:
        print(f'  - {f}')

if divergent:
    print(f'\nDivergent files ({len(divergent)}):')
    for f in divergent:
        print(f'  - {f}')

if extra:
    print(f'\nExtra local files ({len(extra)}) [preserved]:')
    for f in extra[:5]:
        print(f'  - {f}')
    if len(extra) > 5:
        print(f'  ... and {len(extra) - 5} more')

if result.get('ok'):
    print('\n✓ Local state is complete and matches remote')
    sys.exit(0)
else:
    print(f'\n✗ Local state diverges: {len(missing)} missing, {len(divergent)} divergent')
    sys.exit(1)
PYTHON_SCRIPT
}

# Function to repair
repair() {
    local target_tag="${1:-HEAD}"
    print_status "Repairing local state from ${target_tag}..."
    
    python3 << PYTHON_SCRIPT
import sys
sys.path.insert(0, '.')
from core.auto_updater import repair_to_remote_state

def progress_cb(frac, msg):
    bar_width = 30
    filled = int(bar_width * frac)
    bar = '█' * filled + '░' * (bar_width - filled)
    print(f'\r[{bar}] {msg}', end='', flush=True)

ok, reason = repair_to_remote_state('$target_tag', progress_cb=progress_cb)
print()  # newline after progress bar

if ok:
    print('✓ Repair successful')
    sys.exit(0)
else:
    print(f'✗ Repair failed: {reason}')
    sys.exit(1)
PYTHON_SCRIPT
}

# Function to show help
show_help() {
    cat << 'EOF'
EvoAI Updater Repair Script

Usage: ./repair.sh [command] [options]

Commands:
  verify [tag]   Verify local state against a git tag (default: HEAD)
  repair [tag]   Repair local state from a git tag (default: HEAD)
  status         Show current git status (quick check)
  help           Show this help message

Examples:
  ./repair.sh verify            # Check if in sync with HEAD
  ./repair.sh repair            # Auto-repair from HEAD
  ./repair.sh verify v7.0.2     # Check against specific tag
  ./repair.sh repair v7.0.2     # Repair from specific tag

Notes:
  - The updater file (core/auto_updater.py) is protected from repairs
  - Runtime data under data/ is preserved during repairs
  - All repairs are non-interactive and suitable for automation
EOF
}

# Main logic
case "${1:-help}" in
    verify)
        if verify_state "${2:-HEAD}"; then
            exit 0
        else
            exit 1
        fi
        ;;
    repair)
        if repair "${2:-HEAD}"; then
            print_ok "Repair completed successfully"
            exit 0
        else
            print_error "Repair failed"
            exit 1
        fi
        ;;
    status)
        print_status "Git status:"
        git status --short
        echo
        verify_state HEAD || true
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac
