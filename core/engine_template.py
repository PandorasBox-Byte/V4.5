import os
import signal
import sys
import torch
from sentence_transformers import SentenceTransformer, util
from core.memory import load_memory, save_memory
from core.embeddings_cache import load_embeddings, save_embeddings, clear_cache


class Engine:
    def __init__(self):
        print("Loading model (Intel CPU optimized)...")
        torch.set_num_threads(int(os.environ.get("EVOAI_TORCH_THREADS", "8")))

        model_name = os.environ.get("EVOAI_MODEL", "all-MiniLM-L6-v2")
        # Force CPU device to maintain determinism on Intel Mac
        device = os.environ.get("EVOAI_DEVICE", "cpu")

        self.model = SentenceTransformer(model_name, device=device)

        # Configuration
        self.similarity_threshold = float(os.environ.get("EVOAI_SIMILARITY_THRESHOLD", "0.7"))
        self.max_memory_entries = int(os.environ.get("EVOAI_MAX_MEMORY", "500"))

        # Persistent memory and in-memory embedding cache
        self.memory = load_memory()
        self.corpus_texts = [entry.get("user", "") for entry in self.memory]

        # Try to load persisted embeddings from disk (keeps consistent with memory)
        self.embeddings_cache = load_embeddings()
        if self.embeddings_cache is not None:
            try:
                self.embeddings_cache = self.embeddings_cache.to("cpu")
            except Exception:
                pass
            if self.embeddings_cache.shape[0] != len(self.corpus_texts):
                clear_cache()
                self.embeddings_cache = None

        # If no cache, compute once (best-effort)
        if self.embeddings_cache is None and self.corpus_texts:
            try:
                self.embeddings_cache = self.model.encode(self.corpus_texts, convert_to_tensor=True)
                save_embeddings(self.embeddings_cache)
            except Exception:
                self.embeddings_cache = None

        print("Model loaded.")

    def respond(self, text):
        if not text or not text.strip():
            return "Please enter something."

        # Compute query embedding
        try:
            query_embedding = self.model.encode(text, convert_to_tensor=True)
        except Exception:
            query_embedding = None

        # If we have cached embeddings, check similarity first
        if query_embedding is not None and self.embeddings_cache is not None and len(self.corpus_texts) > 0:
            try:
                hits = util.cos_sim(query_embedding, self.embeddings_cache)
                score, idx = torch.max(hits, dim=1)
                if score.item() > self.similarity_threshold:
                    chosen = self.corpus_texts[int(idx.item())].strip()
                    return f"You previously said something similar: '{chosen}'"
            except Exception:
                pass

        # No high-similarity hit: append to memory, prune, save, and update cache
        self.memory.append({"user": text})
        save_memory(self.memory, max_entries=self.max_memory_entries)

        # Update corpus_texts and the on-disk embeddings cache
        self.corpus_texts.append(text)
        if query_embedding is not None:
            try:
                if query_embedding.dim() == 1:
                    query_embedding = query_embedding.unsqueeze(0)

                if self.embeddings_cache is None:
                    self.embeddings_cache = query_embedding
                else:
                    self.embeddings_cache = torch.cat([self.embeddings_cache, query_embedding], dim=0)

                # Persist cache (best-effort)
                save_embeddings(self.embeddings_cache)
            except Exception:
                try:
                    clear_cache()
                except Exception:
                    pass
                self.embeddings_cache = None

        return f"I understand you said: '{text}'"


def handle_exit(sig, frame):
    print("\nShutting down EvoAI safely...")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit)
