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
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen


def _http_get(url: str, timeout: int = 10) -> bytes:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def check_for_updates(manifest_url: str) -> dict | None:
    """Fetch and return the manifest located at *manifest_url*.

    The manifest should be a JSON object containing at least ``version`` and
    ``files`` keys.  ``files`` is a list of objects with ``path`` (relative to
    project root) and ``url`` telling where to download the updated file.
    """
    try:
        body = _http_get(manifest_url, timeout=10)
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


@dataclass
class GitUpdateResult:
    checked: bool = False
    update_available: bool = False
    updated: bool = False
    success: bool = True
    failed_reason: str = ""
    local_version: str = ""
    remote_version: str = ""
    needs_restart: bool = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 30) -> tuple[int, str, str]:
    if cwd is None:
        cwd = _repo_root()
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


_SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _parse_semver_tag(tag: str) -> tuple[int, int, int] | None:
    match = _SEMVER_TAG_RE.match(tag.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _semver_tuple_from_version(version: str) -> tuple[int, int, int] | None:
    value = version.strip().lstrip("v")
    parts = value.split(".")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return None


def get_local_version() -> str:
    root = _repo_root()
    tally = root / "version_tally.json"
    setup_cfg = root / "setup.cfg"
    
    tally_version = None
    setup_version = None

    try:
        if tally.exists():
            data = json.loads(tally.read_text(encoding="utf-8"))
            current = str(data.get("current_version", "")).strip()
            if current:
                tally_version = current
    except Exception:
        pass

    try:
        if setup_cfg.exists():
            for raw in setup_cfg.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if line.startswith("version") and "=" in line:
                    current = line.split("=", 1)[1].strip()
                    if current:
                        setup_version = current
                        break
    except Exception:
        pass

    # If both files exist and have different versions, use the lower version
    # (more conservative) to ensure updates happen when files are out of sync
    if tally_version and setup_version and tally_version != setup_version:
        tally_tuple = _semver_tuple_from_version(tally_version)
        setup_tuple = _semver_tuple_from_version(setup_version)
        if tally_tuple and setup_tuple:
            return setup_version if setup_tuple < tally_tuple else tally_version
    
    if tally_version:
        return tally_version
    if setup_version:
        return setup_version
    return "0.0.0"


def get_latest_remote_tag(remote: str = "origin") -> str | None:
    code, out, _err = _run_git(["ls-remote", "--tags", remote], timeout=45)
    if code != 0:
        return None

    best = None
    best_tuple = None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        ref = parts[1]
        if ref.endswith("^{}"):
            ref = ref[:-3]
        prefix = "refs/tags/"
        if not ref.startswith(prefix):
            continue
        tag = ref[len(prefix) :]
        parsed = _parse_semver_tag(tag)
        if parsed is None:
            continue
        if best_tuple is None or parsed > best_tuple:
            best_tuple = parsed
            best = tag
    return best


def check_git_update_available(remote: str = "origin") -> GitUpdateResult:
    result = GitUpdateResult(checked=True, success=True)
    result.local_version = get_local_version()

    local_tuple = _semver_tuple_from_version(result.local_version)
    remote_tag = get_latest_remote_tag(remote=remote)
    if not remote_tag:
        result.success = False
        result.failed_reason = "unable to read remote tags"
        return result

    result.remote_version = remote_tag.lstrip("v")
    remote_tuple = _parse_semver_tag(remote_tag)
    if local_tuple is None or remote_tuple is None:
        result.success = False
        result.failed_reason = "invalid semver format"
        return result

    result.update_available = remote_tuple > local_tuple
    return result


def _has_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def _enforce_release_version_files(target_tag: str, root: Path) -> tuple[bool, str]:
    files = ["version_tally.json", "setup.cfg"]
    code, _out, err = _run_git(["checkout", target_tag, "--", *files], cwd=root, timeout=60)
    if code != 0:
        return False, (err or "failed to restore release version files from target tag")
    return True, ""


def _is_runtime_local_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    return normalized.startswith("data/")


def _parse_porcelain_paths(status_output: str) -> list[str]:
    paths: list[str] = []
    for raw in status_output.splitlines():
        line = raw.rstrip()
        if not line or len(line) < 4:
            continue
        code = line[:2]
        payload = line[3:]
        if code == "??":
            continue
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        payload = payload.strip()
        if payload:
            paths.append(payload)
    return paths


def _normalize_tracked_nonlocal_files(target_tag: str, root: Path) -> tuple[bool, str]:
    code, status, err = _run_git(["status", "--porcelain"], cwd=root)
    if code != 0:
        return False, (err or "git status failed during normalization")

    changed = _parse_porcelain_paths(status)
    to_restore = [p for p in changed if not _is_runtime_local_path(p)]
    if not to_restore:
        return True, ""

    code, _out, err = _run_git(["checkout", target_tag, "--", *to_restore], cwd=root, timeout=60)
    if code != 0:
        return False, (err or "failed to normalize tracked files from target tag")
    return True, ""


def run_startup_git_update(
    progress_cb: Callable[[float, str], None] | None = None,
    remote: str = "origin",
) -> GitUpdateResult:
    def _progress(frac: float, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(float(frac), str(msg))
        except Exception:
            pass

    result = GitUpdateResult(checked=True, success=True)
    root = _repo_root()
    result.local_version = get_local_version()

    try:
        _progress(0.05, "Checking git updater...")

        if shutil.which("git") is None:
            result.success = False
            result.failed_reason = "git binary not found"
            _progress(1.0, "Update failed: git not installed")
            return result

        if not _has_git_repo(root):
            result.success = False
            result.failed_reason = "not a git checkout"
            _progress(1.0, "Update failed: not a git checkout")
            return result

        code, branch, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        if code != 0 or not branch or branch == "HEAD":
            result.success = False
            result.failed_reason = "detached HEAD or unknown branch"
            _progress(1.0, "Update failed: detached branch state")
            return result

        _progress(0.15, "Checking latest release tag...")
        check = check_git_update_available(remote=remote)
        result.local_version = check.local_version
        result.remote_version = check.remote_version
        result.update_available = check.update_available
        if not check.success and not check.update_available:
            result.success = False
            result.failed_reason = check.failed_reason or "remote tag check failed"
            _progress(1.0, f"Update failed: {result.failed_reason}")
            return result

        if not result.update_available:
            _progress(1.0, "No update required")
            return result

        target_tag = f"v{result.remote_version}"
        _progress(0.25, f"Update found: {target_tag}")

        code, _, err = _run_git(["fetch", "--tags", remote], cwd=root, timeout=60)
        if code != 0:
            result.success = False
            result.failed_reason = err or "git fetch failed"
            _progress(1.0, f"Update failed: {result.failed_reason}")
            return result

        _progress(0.40, "Preparing local changes (stash)...")
        code, status, err = _run_git(["status", "--porcelain"], cwd=root)
        if code != 0:
            result.success = False
            result.failed_reason = err or "git status failed"
            _progress(1.0, f"Update failed: {result.failed_reason}")
            return result

        had_changes = bool(status.strip())
        stash_created = False
        if had_changes:
            stamp = int(time.time())
            code, out, err = _run_git(
                ["stash", "push", "-u", "-m", f"evoai-autoupdate-{stamp}"],
                cwd=root,
                timeout=60,
            )
            if code != 0:
                result.success = False
                result.failed_reason = err or out or "git stash failed"
                _progress(1.0, f"Update failed: {result.failed_reason}")
                return result
            stash_created = "No local changes" not in out

        _progress(0.60, f"Applying update {target_tag}...")
        code, _, err = _run_git(["reset", "--hard", target_tag], cwd=root, timeout=60)
        if code != 0:
            result.success = False
            result.failed_reason = err or "git reset failed"
            _progress(1.0, f"Update failed: {result.failed_reason}")
            return result

        if stash_created:
            _progress(0.85, "Re-applying stashed changes...")
            code, out, err = _run_git(["stash", "pop"], cwd=root, timeout=60)
            if code != 0:
                result.success = False
                result.failed_reason = err or out or "stash pop conflict"
                _progress(1.0, f"Update failed: {result.failed_reason}")
                return result

            _progress(0.90, "Normalizing updated files...")
            ok, reason = _normalize_tracked_nonlocal_files(target_tag, root)
            if not ok:
                result.success = False
                result.failed_reason = reason
                _progress(1.0, f"Update failed: {result.failed_reason}")
                return result

            target_tuple = _parse_semver_tag(target_tag)
            local_after_pop = get_local_version()
            local_tuple = _semver_tuple_from_version(local_after_pop)
            if target_tuple is not None and (local_tuple is None or local_tuple < target_tuple):
                _progress(0.92, "Normalizing release version files...")
                ok, reason = _enforce_release_version_files(target_tag, root)
                if not ok:
                    result.success = False
                    result.failed_reason = reason
                    _progress(1.0, f"Update failed: {result.failed_reason}")
                    return result

                local_after_fix = get_local_version()
                local_fixed_tuple = _semver_tuple_from_version(local_after_fix)
                if local_fixed_tuple is None or local_fixed_tuple < target_tuple:
                    result.success = False
                    result.failed_reason = "local version still behind after update"
                    _progress(1.0, f"Update failed: {result.failed_reason}")
                    return result

            _progress(0.95, "Verifying and repairing file completeness...")
            ok, reason = repair_to_remote_state(target_tag, root=root, progress_cb=_progress)
            if not ok:
                result.success = False
                result.failed_reason = f"File repair failed: {reason}"
                _progress(1.0, f"Update failed: {result.failed_reason}")
                return result

        result.updated = True
        result.needs_restart = True
        _progress(1.0, f"Update successful: {target_tag}")
        return result
    except Exception as exc:
        result.success = False
        result.failed_reason = str(exc)
        _progress(1.0, f"Update failed: {result.failed_reason}")
        return result


def _download_file(url: str, dest: Path) -> None:
    content = _http_get(url, timeout=10)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(content)


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


def get_remote_file_list(tag: str, root: Path | None = None) -> set[str]:
    """Fetch the complete list of tracked files in the remote tag.
    
    Uses ``git ls-tree -r`` to recursively enumerate all files tracked at that tag.
    Returns a set of relative paths (using forward slashes).
    """
    if root is None:
        root = _repo_root()
    
    code, out, _err = _run_git(["ls-tree", "-r", "--name-only", tag], cwd=root, timeout=60)
    if code != 0:
        return set()
    
    files = set()
    for line in out.splitlines():
        path = line.strip()
        if path:
            files.add(path.replace("\\", "/"))
    return files


def verify_complete_state(target_tag: str, root: Path | None = None) -> dict:
    """Verify that the local working tree matches the remote tag exactly.
    
    Compares:
    - Local tracked files vs remote tracked files
    - File contents (by comparing hashes)
    
    Returns a dict with keys:
    - ``ok``: bool - True if local matches remote perfectly
    - ``missing_files``: list - files in remote but not local
    - ``extra_files``: list - files in local but not in remote
    - ``divergent_files``: list - files that exist in both but have different content
    - ``error``: str - error message if verification failed
    """
    if root is None:
        root = _repo_root()
    
    result = {
        "ok": False,
        "missing_files": [],
        "extra_files": [],
        "divergent_files": [],
        "error": None,
    }
    
    remote_files = get_remote_file_list(target_tag, root)
    if not remote_files:
        result["error"] = "Failed to fetch remote file list"
        return result
    
    local_files = set()
    for local_path in root.rglob("*"):
        if local_path.is_file() and not str(local_path.relative_to(root)).startswith(".git"):
            rel = local_path.relative_to(root)
            local_files.add(str(rel).replace("\\", "/"))
    
    remote_set = set(remote_files)
    local_set = set(local_files)
    
    missing = list(remote_set - local_set)
    extra = list(local_set - remote_set)
    result["missing_files"] = sorted(missing)
    result["extra_files"] = sorted(extra)
    
    divergent = []
    for fpath in remote_set & local_set:
        local_file = root / fpath
        
        code, remote_hash, _ = _run_git(["hash-object", f"{target_tag}:{fpath}"], cwd=root, timeout=60)
        if code != 0:
            continue
        
        try:
            local_hash = subprocess.run(
                ["git", "hash-object", str(local_file)],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            ).stdout.strip()
        except Exception:
            continue
        
        if remote_hash.strip() != local_hash:
            divergent.append(fpath)
    
    result["divergent_files"] = sorted(divergent)
    result["ok"] = not missing and not divergent
    return result


def repair_to_remote_state(
    target_tag: str,
    root: Path | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[bool, str]:
    """Repair local working tree to match remote tag state exactly.
    
    Fetches missing files and re-syncs divergent files from the remote tag.
    Preserves extra local files (runtime data, etc).
    
    Returns (success: bool, reason: str)
    """
    if root is None:
        root = _repo_root()
    
    def _progress(frac: float, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(float(frac), str(msg))
        except Exception:
            pass
    
    _progress(0.10, "Verifying local state...")
    verification = verify_complete_state(target_tag, root)
    
    if verification["ok"]:
        _progress(1.0, "Local state matches remote")
        return True, "ok"
    
    missing = verification.get("missing_files", [])
    divergent = verification.get("divergent_files", [])
    
    to_repair = list(set(missing) | set(divergent))
    if not to_repair:
        _progress(1.0, "No files to repair")
        return True, "ok"
    
    _progress(0.20, f"Repairing {len(to_repair)} file(s) from {target_tag}...")
    
    try:
        code, _out, err = _run_git(["checkout", target_tag, "--", *to_repair], cwd=root, timeout=120)
        if code != 0:
            return False, err or "Failed to checkout files from remote tag"
        
        _progress(0.90, "Verifying repair...")
        verification_after = verify_complete_state(target_tag, root)
        if not verification_after["ok"]:
            remaining = len(verification_after.get("missing_files", [])) + len(
                verification_after.get("divergent_files", [])
            )
            return False, f"Repair incomplete: {remaining} file(s) still divergent"
        
        _progress(1.0, f"Repair successful: {len(to_repair)} file(s) restored")
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def safe_run_update(manifest_url: str) -> None:
    """Wrapper for callers that should not crash if something goes wrong."""
    try:
        run_update_flow(manifest_url)
    except Exception as e:
        print(f"Auto-update failed: {e}")
