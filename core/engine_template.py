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
