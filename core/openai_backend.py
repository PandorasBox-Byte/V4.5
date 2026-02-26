"""Optional OpenAI ChatGPT backend wrapper.

Uses the `openai` Python package when `OPENAI_API_KEY` is present in the
environment. This module provides a small sync/streaming interface so the
rest of the codebase can call into ChatGPT-style models as an alternative
to a local transformer model.
"""
from __future__ import annotations

import os
from typing import Callable


class OpenAIBackend:
    def __init__(self):
        try:
            import openai
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("openai package required for OpenAIBackend") from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        openai.api_key = api_key
        self._client = openai

    def generate_sync(self, prompt: str, model: str | None = None, **kwargs) -> str:
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        resp = self._client.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **{k: v for k, v in kwargs.items() if v is not None},
        )
        # Extract assistant content
        try:
            return resp.choices[0].message.content
        except Exception:
            # Fallback for different client shapes
            return resp.choices[0].text if hasattr(resp.choices[0], "text") else ""

    def generate_stream(self, prompt: str, chunk_callback: Callable[[str, bool], None], model: str | None = None, **kwargs):
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        try:
            stream = self._client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **{k: v for k, v in kwargs.items() if v is not None},
            )
        except Exception as e:  # pragma: no cover - network/credential dependent
            chunk_callback(f"(openai error) {e}", True)
            return None

        # stream is an iterator of events
        try:
            for event in stream:
                try:
                    # each event contains choices with delta content
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if "content" in delta:
                            chunk_callback(delta["content"], False)
                        if choice.get("finish_reason"):
                            # finished for this choice
                            chunk_callback("", True)
                            return None
                except Exception:
                    continue
        except Exception as e:  # pragma: no cover - runtime stream errors
            chunk_callback(f"(openai stream error) {e}", True)
            return None

        # ensure final callback
        try:
            chunk_callback("", True)
        except Exception:
            pass
