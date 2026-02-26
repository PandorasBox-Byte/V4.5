# Backup of /upgrade_v4.5.1.sh
# Backed up on 2026-02-26

#!/bin/bash

echo "=== Upgrading EvoAI V4.5 (Intel Optimized) ==="
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

source v4env/bin/activate

echo "Installing any missing lightweight utilities..."
pip install numpy --quiet

mkdir -p data
mkdir -p core

# ----------------------------
# Persistent Memory System
# ----------------------------
cat > core/memory.py << 'EOF'
import json
import os

MEMORY_FILE = "data/memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
EOF

# ----------------------------
# Upgraded Engine
# ----------------------------
cat > core/engine_template.py << 'EOF'
import signal
import sys
import torch
from sentence_transformers import SentenceTransformer, util
from core.memory import load_memory, save_memory

class Engine:
    def __init__(self):
        print("Loading model (Intel CPU optimized)...")
        torch.set_num_threads(8)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.memory = load_memory()
        print("Model loaded.")

    def respond(self, text):
        if not text.strip():
            return "Please enter something."

        # Store conversation
        self.memory.append({"user": text})
        save_memory(self.memory)

        if len(self.memory) > 1:
            corpus = [entry["user"] for entry in self.memory[:-1]]
            corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True)
            query_embedding = self.model.encode(text, convert_to_tensor=True)

            hits = util.cos_sim(query_embedding, corpus_embeddings)
            score, idx = torch.max(hits, dim=1)

            if score.item() > 0.7:
                return f"You previously said something similar: '{corpus[idx].strip()}'"

        return f"I understand you said: '{text}'"

def handle_exit(sig, frame):
    print("\nShutting down EvoAI safely...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
EOF

# ----------------------------
# Knowledge Base File
# ----------------------------
cat > data/knowledge.txt << 'EOF'
EvoAI is a local semantic assistant running on an Intel-based Mac.
It uses sentence embeddings for similarity search.
EOF

echo ""
echo "=== UPGRADE COMPLETE ==="
echo "Run:"
echo "source v4env/bin/activate"
echo "python launch_v4.py"
