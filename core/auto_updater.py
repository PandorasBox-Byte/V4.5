"""Simple auto-update helpers for the engine.

The implementation is deliberately conservative: it downloads a manifest
(usually a JSON file) describing available updates, fetches the new files into
a temporary directory, runs the local test suite against them, and then prompts
the user before overwriting any existing code.  Nothing is changed unless the
user explicitly agrees.

This module is used by ``core.engine_template.Engine`` when
``EVOAI_AUTO_UPDATE_URL`` is set.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import requests


def check_for_updates(manifest_url: str) -> dict | None:
    """Fetch and return the manifest located at *manifest_url*.

    The manifest should be a JSON object containing at least ``version`` and
    ``files`` keys.  ``files`` is a list of objects with ``path`` (relative to
    project root) and ``url`` telling where to download the updated file.
    """
    try:
        resp = requests.get(manifest_url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _download_file(url: str, dest: Path) -> None:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(resp.content)


def _compute_diff(old_path: Path, new_path: Path) -> str:
    try:
        import difflib

        old = old_path.read_text().splitlines(keepends=True)
        new = new_path.read_text().splitlines(keepends=True)
        diff = difflib.unified_diff(old, new, fromfile=str(old_path), tofile=str(new_path))
        return "".join(diff)
    except Exception:
        return ""  # non-critical


def _run_tests_on_dir(code_dir: Path) -> bool:
    """Execute the test suite with ``PYTHONPATH`` pointed at *code_dir*.

    Returns True if tests exit with code 0.  If the candidate update directory
    contains no Python source files, the function returns True immediately
    (nothing for us to test).
    """
    # quick check for Python files
    has_py = any(p.suffix == ".py" for p in code_dir.rglob("*.py"))
    if not has_py:
        return True
    env = os.environ.copy()
    # prepend our temporary directory to PYTHONPATH so updated modules are
    # picked up, but keep the original cwd so imports of untouched modules
    # still work during test run.
    env["PYTHONPATH"] = str(code_dir) + os.pathsep + os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    proc = shutil.which("python")
    if not proc:
        return False
    import subprocess

    result = subprocess.run([proc, "-m", "unittest", "discover", "-v"], env=env, cwd=code_dir)
    return result.returncode == 0


def prompt_user_for_update(diffs: str) -> bool:
    """Show *diffs* to the user and return True if they approve the update."""
    print("The following changes would be applied:")
    print(diffs or "(no textual diff available)")
    resp = input("Apply update? [y/N]: ").strip().lower()
    return resp == "y" or resp == "yes"


def apply_update(tempdir: Path, base_dir: str | None = None) -> None:
    """Copy files from *tempdir* into *base_dir* (or the cwd if unspecified)."""
    if base_dir is None:
        base_dir = os.getcwd()
    for root, dirs, files in os.walk(tempdir):
        for fname in files:
            src = Path(root) / fname
            rel = src.relative_to(tempdir)
            dest = Path(base_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def run_update_flow(manifest_url: str) -> bool:
    """Check the manifest and, if an update is available and approved,
    apply it.  Returns True if an update was performed (or False otherwise).

    Updates are applied relative to the project root (two levels up from this
    module) regardless of the current working directory.  This makes the
    behaviour consistent when tests run from the ``tests/`` directory.
    """
    # determine repository root based on this file's location
    orig_cwd = str(Path(__file__).resolve().parent.parent)
    manifest = check_for_updates(manifest_url)
    if not manifest:
        print("Failed to fetch update manifest.")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # download each file
        for entry in manifest.get("files", []):
            try:
                dest = tmp_path / entry["path"]
                _download_file(entry["url"], dest)
            except Exception as e:
                print(f"Failed to download {entry}: {e}")
                return False
        # run tests
        print("Running test suite against updated code...")
        if not _run_tests_on_dir(tmp_path):
            print("Tests failed on candidate update; aborting.")
            return False
        # compute diffs for user review
        diffs = []
        for entry in manifest.get("files", []):
            old = Path(orig_cwd) / entry["path"]
            new = tmp_path / entry["path"]
            if old.exists() and new.exists():
                diffs.append(_compute_diff(old, new))
        diff_text = "\n".join(diffs)
        if not prompt_user_for_update(diff_text):
            print("Update cancelled by user.")
            return False
        apply_update(tmp_path, base_dir=orig_cwd)
        print("Update applied successfully.")
        return True


def safe_run_update(manifest_url: str) -> None:
    """Wrapper for callers that should not crash if something goes wrong."""
    try:
        run_update_flow(manifest_url)
    except Exception as e:
        print(f"Auto-update failed: {e}")
