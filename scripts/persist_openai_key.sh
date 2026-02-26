#!/usr/bin/env bash
# Persist an OpenAI API key to a secure local file and source it from your shell rc.
# Usage:
#   bash scripts/persist_openai_key.sh "sk-..."
# or interactively (safer):
#   bash scripts/persist_openai_key.sh
#   (then paste the key when prompted)

set -euo pipefail

KEY="$1"
if [ -z "$KEY" ]; then
  # prompt securely
  echo "Enter your OPENAI API key (input will be hidden):"
  read -rs KEY
  echo
fi

TARGET_ENV="$HOME/.evoai_env"
SHELL_RC="$HOME/.zshrc"

printf "# EvoAI OpenAI key (created by scripts/persist_openai_key.sh)\n" > "$TARGET_ENV"
printf "export OPENAI_API_KEY=\"%s\"\n" "$KEY" >> "$TARGET_ENV"
chmod 600 "$TARGET_ENV"

# Ensure user's zshrc sources the file (idempotent)
SOURCE_LINE='if [ -f "$HOME/.evoai_env" ]; then source "$HOME/.evoai_env"; fi'

grep -F "$TARGET_ENV" "$SHELL_RC" >/dev/null 2>&1 || {
  printf "\n# Source EvoAI environment file\n%s\n" "$SOURCE_LINE" >> "$SHELL_RC"
  echo "Added source line to $SHELL_RC"
}

echo "Stored key in $TARGET_ENV and restricted permissions (600)."
echo "Restart your shell or run: source $SHELL_RC"
