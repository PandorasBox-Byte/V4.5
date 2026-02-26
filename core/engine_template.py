        if self.embeddings_cache is None and self.corpus_texts:
import os
import signal
import sys
import torch
from sentence_transformers import SentenceTransformer, util
from core.memory import load_memory, save_memory


class Engine:
    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
        device = os.getenv("DEVICE", "cpu")
        threads = int(os.getenv("THREADS", "8"))

        print(f"Loading model '{model_name}' on device='{device}' (threads={threads})...")
        torch.set_num_threads(threads)
        self.model = SentenceTransformer(model_name, device=device)
        self.memory = load_memory()
        print("Model loaded.")

    def respond(self, text):
        if not text.strip():
            return "Please enter something."

        # Store conversation
        self.memory.append({"user": text})
        if os.getenv("DISABLE_MEMORY_SAVE") != "1":
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
            try:
                self.embeddings_cache = self.model.encode(self.corpus_texts, convert_to_tensor=True)
                save_embeddings(self.embeddings_cache)
            except Exception:
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
        # Prune and save memory atomically
        save_memory(self.memory, max_entries=self.max_memory_entries)

        # Update corpus_texts and the on-disk embeddings cache
        self.corpus_texts.append(text)
        if query_embedding is not None:
            try:
                # Ensure query_embedding is a 2D tensor (1, dim)
                if query_embedding.dim() == 1:
                    query_embedding = query_embedding.unsqueeze(0)

                if self.embeddings_cache is None:
                    self.embeddings_cache = query_embedding
                else:
                    self.embeddings_cache = torch.cat([self.embeddings_cache, query_embedding], dim=0)

                # Persist cache (best-effort)
                save_embeddings(self.embeddings_cache)
            except Exception:
                # On failure, clear on-disk cache to avoid mismatches later
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
