from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from core import auto_updater
from core.self_repair import SelfRepair


class TestedApplyOrchestrator:
    URL_RE = re.compile(r"https?://[^\s)\]]+")

    def __init__(self, workspace_root: str | None = None) -> None:
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.post_pytest = os.environ.get("EVOAI_TESTED_APPLY_POST_PYTEST", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.min_retention_score = float(os.environ.get("EVOAI_TESTED_APPLY_MIN_SCORE", "0.45"))
        self.benchmark_timeout_sec = int(os.environ.get("EVOAI_BENCHMARK_TIMEOUT_SEC", "120"))

    def _benchmark_command(self) -> str:
        return os.environ.get("EVOAI_BENCHMARK_COMMAND", "").strip()

    def _run_benchmark(self) -> Dict[str, Any]:
        cmd = self._benchmark_command()
        if not cmd:
            return {
                "ok": True,
                "reason": "benchmark_skipped",
                "elapsed_ms": 0.0,
            }
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workspace_root),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.benchmark_timeout_sec,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {
                "ok": proc.returncode == 0,
                "reason": "ok" if proc.returncode == 0 else "benchmark_failed",
                "elapsed_ms": elapsed_ms,
            }
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {
                "ok": False,
                "reason": "benchmark_error",
                "elapsed_ms": elapsed_ms,
            }

    def _retention_score(
        self,
        files_count: int,
        baseline_benchmark: Dict[str, Any],
        post_benchmark: Dict[str, Any],
    ) -> float:
        speed_score = 0.5
        b_ok = bool(baseline_benchmark.get("ok", False))
        p_ok = bool(post_benchmark.get("ok", False))
        b_ms = float(baseline_benchmark.get("elapsed_ms", 0.0) or 0.0)
        p_ms = float(post_benchmark.get("elapsed_ms", 0.0) or 0.0)

        if b_ok and p_ok and b_ms > 0:
            improvement = (b_ms - p_ms) / b_ms
            speed_score = max(0.0, min(1.0, 0.5 + improvement))
        elif b_ok and not p_ok:
            speed_score = 0.0

        file_penalty = min(0.30, max(0.0, float(files_count) * 0.02))
        score = max(0.0, min(1.0, (0.6 * speed_score + 0.4) - file_penalty))
        return score

    def _manifest_source(self, text: str) -> tuple[str | None, str]:
        env_url = os.environ.get("EVOAI_TESTED_APPLY_MANIFEST_URL", "").strip()
        if env_url:
            return env_url, "url"

        env_path = os.environ.get("EVOAI_TESTED_APPLY_MANIFEST_PATH", "").strip()
        if env_path:
            return env_path, "path"

        urls = self.URL_RE.findall(str(text or ""))
        if urls:
            return urls[0], "url"
        return None, ""

    def _load_manifest(self, source: str, kind: str) -> Dict[str, Any] | None:
        if kind == "url":
            return auto_updater.check_for_updates(source)
        if kind == "path":
            path = Path(source)
            if not path.is_absolute():
                path = (self.workspace_root / path).resolve()
            if not path.exists():
                return None
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data if isinstance(data, dict) else None
            except Exception:
                return None
        return None

    def _validate_manifest(self, manifest: Dict[str, Any] | None) -> tuple[bool, str, List[Dict[str, str]]]:
        if not isinstance(manifest, dict):
            return False, "manifest_missing_or_invalid", []
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            return False, "manifest_has_no_files", []
        normalized: List[Dict[str, str]] = []
        for entry in files:
            if not isinstance(entry, dict):
                continue
            rel = str(entry.get("path", "")).strip().lstrip("/")
            url = str(entry.get("url", "")).strip()
            if not rel or not url:
                continue
            normalized.append({"path": rel, "url": url})
        if not normalized:
            return False, "manifest_files_invalid", []
        return True, "ok", normalized

    def _backup_targets(self, files: List[Dict[str, str]], backup_root: Path) -> Dict[str, str]:
        state: Dict[str, str] = {}
        for entry in files:
            rel = entry["path"]
            target = self.workspace_root / rel
            snapshot = backup_root / rel
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_file():
                shutil.copy2(target, snapshot)
                state[rel] = "existing"
            else:
                state[rel] = "missing"
        return state

    def _rollback(self, files: List[Dict[str, str]], backup_root: Path, state: Dict[str, str]) -> None:
        for entry in files:
            rel = entry["path"]
            target = self.workspace_root / rel
            snapshot = backup_root / rel
            marker = state.get(rel, "missing")
            if marker == "existing" and snapshot.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot, target)
            elif marker == "missing" and target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass

    def run(self, request_text: str) -> Dict[str, Any]:
        source, kind = self._manifest_source(request_text)
        if not source:
            return {
                "ok": False,
                "summary": "Tested apply needs a manifest source. Set EVOAI_TESTED_APPLY_MANIFEST_URL or EVOAI_TESTED_APPLY_MANIFEST_PATH.",
                "reason": "manifest_not_provided",
                "files": 0,
            }

        manifest = self._load_manifest(source, kind)
        valid, reason, files = self._validate_manifest(manifest)
        if not valid:
            return {
                "ok": False,
                "summary": f"Tested apply blocked: {reason}.",
                "reason": reason,
                "files": 0,
            }

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="evoai-tested-apply-") as tmp, tempfile.TemporaryDirectory(
            prefix="evoai-tested-apply-backup-"
        ) as backup:
            tmp_path = Path(tmp)
            backup_path = Path(backup)

            for entry in files:
                try:
                    auto_updater._download_file(entry["url"], tmp_path / entry["path"])
                except Exception as exc:
                    return {
                        "ok": False,
                        "summary": f"Tested apply download failed for {entry['path']}: {exc}",
                        "reason": "download_failed",
                        "files": len(files),
                    }

            if not auto_updater._run_tests_on_dir(tmp_path):
                return {
                    "ok": False,
                    "summary": "Tested apply rejected candidate because pre-apply tests failed.",
                    "reason": "candidate_tests_failed",
                    "files": len(files),
                }

            baseline_benchmark = self._run_benchmark()

            snapshot_state = self._backup_targets(files, backup_path)
            try:
                auto_updater.apply_update(tmp_path, base_dir=str(self.workspace_root))
                post_ok, post_out = SelfRepair.run_tests(mode="startup", include_pytest=False)
                if not post_ok:
                    self._rollback(files, backup_path, snapshot_state)
                    tail = str(post_out).strip().splitlines()[-1] if str(post_out).strip() else "startup checks failed"
                    return {
                        "ok": False,
                        "summary": f"Tested apply rolled back after post-apply checks failed: {tail}",
                        "reason": "post_checks_failed",
                        "files": len(files),
                    }

                if self.post_pytest:
                    post_full_ok, post_full_out = SelfRepair.run_tests(mode="repair", include_pytest=True)
                    if not post_full_ok:
                        self._rollback(files, backup_path, snapshot_state)
                        tail = (
                            str(post_full_out).strip().splitlines()[-1]
                            if str(post_full_out).strip()
                            else "post-apply pytest failed"
                        )
                        return {
                            "ok": False,
                            "summary": f"Tested apply rolled back after full tests failed: {tail}",
                            "reason": "post_pytest_failed",
                            "files": len(files),
                        }

                post_benchmark = self._run_benchmark()
                retention_score = self._retention_score(
                    files_count=len(files),
                    baseline_benchmark=baseline_benchmark,
                    post_benchmark=post_benchmark,
                )
                if retention_score < self.min_retention_score:
                    self._rollback(files, backup_path, snapshot_state)
                    return {
                        "ok": False,
                        "summary": (
                            f"Tested apply rolled back: retention score {retention_score:.2f} "
                            f"below threshold {self.min_retention_score:.2f}."
                        ),
                        "reason": "retention_score_below_threshold",
                        "files": len(files),
                        "retention_score": retention_score,
                    }
            except Exception as exc:
                self._rollback(files, backup_path, snapshot_state)
                return {
                    "ok": False,
                    "summary": f"Tested apply rolled back due to apply error: {exc}",
                    "reason": "apply_failed",
                    "files": len(files),
                }

        elapsed = (time.perf_counter() - started) * 1000.0
        version = ""
        if isinstance(manifest, dict):
            version = str(manifest.get("version", "")).strip()
        label = f" version {version}" if version else ""
        return {
            "ok": True,
            "summary": f"Tested apply succeeded{label}: {len(files)} file(s) validated and applied in {elapsed:.0f} ms.",
            "reason": "ok",
            "files": len(files),
            "elapsed_ms": elapsed,
            "retention_score": retention_score,
            "benchmark_baseline_ms": float(baseline_benchmark.get("elapsed_ms", 0.0) or 0.0),
            "benchmark_post_ms": float(post_benchmark.get("elapsed_ms", 0.0) or 0.0),
        }
