#!/bin/bash

echo "=== EvoAI V4.5 Full Installer ==="
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "Installing into: $BASE_DIR"

# ------------------------
# 1. Virtual Environment
# ------------------------
if [ ! -d "v4env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv v4env
fi

source v4env/bin/activate
pip install --upgrade pip

# ------------------------
# 2. Dependencies
# ------------------------
pip uninstall -y numpy torch torchvision torchaudio || true
pip install "numpy<2"

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cpu

pip install \
    sentence-transformers==2.2.2 \
    scikit-learn \
    scipy \
    requests \
    transformers \
    tqdm

# ------------------------
# 3. Folder Structure
# ------------------------
mkdir -p core

# ------------------------
# 4. Core Engine
# ------------------------
cat > core/engine_template.py << 'EOF'
import signal
import sys
from sentence_transformers import SentenceTransformer

class Engine:
    def __init__(self):
        print("Loading model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")

    def respond(self, text):
        if not text.strip():
            return "Please enter something."
        return f"EvoAI processed: {text}"

def handle_exit(sig, frame):
    print("\nShutting down EvoAI safely...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
EOF

# ------------------------
# 5. Launcher
# ------------------------
cat > launch_v4.py << 'EOF'
from core.engine_template import Engine

def main():
    engine = Engine()
    print("EvoAI V4.5 Ready. Press Ctrl+C to exit.\n")

    while True:
        user_input = input("You: ")
        response = engine.respond(user_input)
        print("EvoAI:", response)

if __name__ == "__main__":
    main()
EOF

# ------------------------
# 6. Freeze Requirements
# ------------------------
pip freeze > requirements.txt

echo ""
echo "=== INSTALL COMPLETE ==="
echo "Run with:"
echo "source v4env/bin/activate"
echo "python launch_v4.py"
