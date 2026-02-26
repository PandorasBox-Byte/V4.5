import json
import os
import tempfile
from typing import List, Any, Optional

MEMORY_FILE = os.path.join("data", "memory.json")


def load_memory() -> List[Any]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def prune_memory(memory: List[Any], max_entries: Optional[int]) -> List[Any]:
    if not max_entries or max_entries <= 0:
        return memory
    if len(memory) <= max_entries:
        return memory
    return memory[-max_entries:]


def _atomic_write(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Use a NamedTemporaryFile to ensure atomic replace when possible
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        try:
            os.fsync(tmp.fileno())
        except OSError:
            pass
        tmp_name = tmp.name
    try:
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        os.rename(tmp_name, path)


def save_memory(memory: List[Any], max_entries: Optional[int] = None) -> None:
    if max_entries is not None:
        memory = prune_memory(memory, max_entries)
    _atomic_write(MEMORY_FILE, memory)
