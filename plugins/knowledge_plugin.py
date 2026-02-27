"""A simple plugin that looks up lines in ``data/knowledge.txt``.

If the user's query semantically matches one of the knowledge lines above a
threshold, the plugin returns that line prefixed with a clarifying message.
This demonstrates the research plugin API without external dependencies.
"""

import os
import math

try:
    from sentence_transformers import util
except Exception:  # pragma: no cover - optional dependency
    class _Num:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class _CosResult:
        def __init__(self, scores):
            self.scores = scores

        def max(self, dim=1):
            if not self.scores:
                return _Num(0.0), _Num(0)
            idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
            return _Num(float(self.scores[idx])), _Num(int(idx))

    class _UtilFallback:
        @staticmethod
        def cos_sim(query_embedding, corpus_embeddings):
            q = list(query_embedding) if isinstance(query_embedding, (list, tuple)) else []
            rows = corpus_embeddings if isinstance(corpus_embeddings, list) else []
            qn = math.sqrt(sum(v * v for v in q)) or 1.0
            scores = []
            for row in rows:
                r = list(row) if isinstance(row, (list, tuple)) else []
                rn = math.sqrt(sum(v * v for v in r)) or 1.0
                dot = sum(a * b for a, b in zip(q, r))
                scores.append(dot / (qn * rn))
            return _CosResult(scores)

    util = _UtilFallback()

from core.plugin_manager import ResearchPlugin


class KnowledgePlugin(ResearchPlugin):
    name = "knowledge"

    def __init__(self):
        self._lines = []
        self._embeddings = None

        path = os.path.join("data", "knowledge.txt")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._lines = [l.strip() for l in f if l.strip()]
            except Exception:
                self._lines = []

    def _ensure_embeddings(self, engine):
        if self._embeddings is None and self._lines:
            self._embeddings = engine._encode(self._lines)

    def can_handle(self, query: str) -> bool:
        # This plugin is willing to look at every query but only returns a
        # response when it finds a strong match.
        return True

    def handle(self, query: str, engine) -> str | None:
        if not self._lines:
            return None
        self._ensure_embeddings(engine)
        try:
            q_emb = engine._encode(query)
            scores = util.cos_sim(q_emb, self._embeddings)
            score, idx = scores.max(dim=1)
            # use a slightly lower threshold so conversational queries
            # such as questions are still caught.  ``engine.similarity_threshold``
            # can be overridden for testing or tuning.
            if score.item() > min(engine.similarity_threshold, 0.5):
                line = self._lines[int(idx.item())]
                return f"According to my notes: '{line}'"
        except Exception:
            return None
        return None
