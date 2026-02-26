import json
import os
import errno

MEMORY_FILE = "data/memory.json"


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def prune_memory(memory, max_entries):
    if not max_entries or max_entries <= 0:
        return memory
    if len(memory) <= max_entries:
        return memory
    return memory[-max_entries:]


def _atomic_write(path, data):
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Not critical on all platforms
            pass
    try:
        os.replace(tmp_path, path)
    except OSError:
        # Fallback for older systems
        try:
            os.remove(path)
        except OSError:
            pass
        os.rename(tmp_path, path)


def save_memory(memory, max_entries=None):
    if max_entries is not None:
        memory = prune_memory(memory, max_entries)
    _atomic_write(MEMORY_FILE, memory)
