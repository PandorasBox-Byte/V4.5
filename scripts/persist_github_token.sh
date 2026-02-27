#!/usr/bin/env bash
# Persist a GitHub token to a secure local file and source it from your shell rc.
# Usage:
#   bash scripts/persist_github_token.sh "ghp_..."
# or interactively (safer):
#   bash scripts/persist_github_token.sh
#   (then paste the token when prompted)

set -euo pipefail

KEY="${1:-}"
if [ -z "$KEY" ]; then
  echo "Enter your GitHub token (input will be hidden):"
  read -rs KEY
  echo
fi

TARGET_ENV="$HOME/.evoai_env"
SHELL_RC="$HOME/.zshrc"

printf "# EvoAI GitHub token (created by scripts/persist_github_token.sh)\n" > "$TARGET_ENV"
printf "export GITHUB_TOKEN=\"%s\"\n" "$KEY" >> "$TARGET_ENV"
chmod 600 "$TARGET_ENV"

SOURCE_LINE='if [ -f "$HOME/.evoai_env" ]; then source "$HOME/.evoai_env"; fi'

grep -F "$TARGET_ENV" "$SHELL_RC" >/dev/null 2>&1 || {
  printf "\n# Source EvoAI environment file\n%s\n" "$SOURCE_LINE" >> "$SHELL_RC"
  echo "Added source line to $SHELL_RC"
}

echo "Stored token in $TARGET_ENV and restricted permissions (600)."
echo "Restart your shell or run: source $SHELL_RC"
