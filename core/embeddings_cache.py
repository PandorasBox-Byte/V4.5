import os
import torch
from typing import Optional

DATA_DIR = os.environ.get("EVOAI_DATA_DIR", "data")
CACHE_FILE = os.path.join(DATA_DIR, "embeddings.pt")


def load_embeddings() -> Optional[torch.Tensor]:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        data = torch.load(CACHE_FILE, map_location="cpu")
        emb = data.get("embeddings")
        return emb
    except Exception:
        return None


def save_embeddings(tensor: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        # Ensure tensor is on CPU and detached to avoid device issues
        to_save = tensor.cpu().detach() if hasattr(tensor, "cpu") else tensor
        torch.save({"embeddings": to_save}, CACHE_FILE)
    except Exception:
        # Best-effort: ignore save failures to avoid breaking runtime
        pass


def clear_cache() -> None:
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
    except Exception:
        pass
