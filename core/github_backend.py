"""Optional GitHub Models backend wrapper.

Uses token-authenticated HTTP calls to a GitHub Models compatible chat
completions endpoint. This module mirrors the sync/streaming surface used by
other backends so the engine can switch providers via environment variables.
"""
from __future__ import annotations

import json
import os
from typing import Callable


class GitHubBackend:
    def __init__(self):
        try:
            import requests
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("requests package required for GitHubBackend") from e

        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_TOKEN not set")

        self._requests = requests
        self._token = token
        self._endpoint = os.environ.get(
            "GITHUB_MODELS_ENDPOINT",
            "https://models.inference.ai.azure.com/chat/completions",
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def generate_sync(self, prompt: str, model: str | None = None, **kwargs) -> str:
        model = model or os.environ.get("GITHUB_MODEL", "gpt-4o-mini")
        timeout = int(os.environ.get("EVOAI_BACKEND_TIMEOUT", "30"))
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        resp = self._requests.post(
            self._endpoint,
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

    def generate_stream(
        self,
        prompt: str,
        chunk_callback: Callable[[str, bool], None],
        model: str | None = None,
        **kwargs,
    ):
        model = model or os.environ.get("GITHUB_MODEL", "gpt-4o-mini")
        timeout = int(os.environ.get("EVOAI_BACKEND_TIMEOUT", "30"))
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        try:
            with self._requests.post(
                self._endpoint,
                headers=self._headers(),
                json=payload,
                timeout=timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        chunk_callback("", True)
                        return None
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        if content:
                            chunk_callback(content, False)
                        if choice.get("finish_reason"):
                            chunk_callback("", True)
                            return None
        except Exception as e:  # pragma: no cover - network/credential dependent
            chunk_callback(f"(github backend error) {e}", True)
            return None

        try:
            chunk_callback("", True)
        except Exception:
            pass
        return None
