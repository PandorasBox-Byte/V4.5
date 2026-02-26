
import signal
import sys
import os
import torch
from sentence_transformers import SentenceTransformer, util
from core.memory import load_memory, save_memory


class Engine:
    def __init__(self):
        print("Loading model (Intel CPU optimized)...")
        torch.set_num_threads(int(os.environ.get("EVOAI_TORCH_THREADS", "8")))
        # Force CPU device to maintain determinism on Intel Mac
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # Configuration
        self.similarity_threshold = float(os.environ.get("EVOAI_SIMILARITY_THRESHOLD", "0.7"))
        self.max_memory_entries = int(os.environ.get("EVOAI_MAX_MEMORY", "500"))

        # Persistent memory and in-memory embedding cache
        self.memory = load_memory()
        self.corpus_texts = [entry.get("user", "") for entry in self.memory]
        self.embeddings_cache = None
        if self.corpus_texts:
            try:
                self.embeddings_cache = self.model.encode(self.corpus_texts, convert_to_tensor=True)
            except Exception:
                # Fall back to not caching if encoding fails
                self.embeddings_cache = None

        print("Model loaded.")

    def respond(self, text):
        if not text or not text.strip():
            return "Please enter something."

        # Prepare corpus embeddings (existing memory only)
        if self.embeddings_cache is None and self.corpus_texts:
            try:
                self.embeddings_cache = self.model.encode(self.corpus_texts, convert_to_tensor=True)
            except Exception:
                self.embeddings_cache = None

        # Compute query embedding and compare to existing corpus (if any)
        try:
            query_embedding = self.model.encode(text, convert_to_tensor=True)
        except Exception:
            query_embedding = None

        if query_embedding is not None and self.embeddings_cache is not None and len(self.corpus_texts) > 0:
            try:
                hits = util.cos_sim(query_embedding, self.embeddings_cache)
                score, idx = torch.max(hits, dim=1)
                if score.item() > self.similarity_threshold:
                    # idx is a tensor; convert to int
                    chosen = self.corpus_texts[int(idx.item())].strip()
                    return f"You previously said something similar: '{chosen}'"
            except Exception:
                # similarity check failed; continue
                pass

        # No high-similarity hit: append to memory, prune, save, and update cache
        self.memory.append({"user": text})
        save_memory(self.memory, max_entries=self.max_memory_entries)

        # Update corpus_texts and embeddings cache incrementally
        self.corpus_texts.append(text)
        if query_embedding is not None:
            try:
                if self.embeddings_cache is None:
                    self.embeddings_cache = query_embedding
                else:
                    self.embeddings_cache = torch.cat([self.embeddings_cache, query_embedding], dim=0)
            except Exception:
                # On failure, drop caching
                self.embeddings_cache = None

        return f"I understand you said: '{text}'"

def handle_exit(sig, frame):
    print("\nShutting down EvoAI safely...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
