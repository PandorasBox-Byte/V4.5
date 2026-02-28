"""A plugin that looks up knowledge from structured knowledge graph and flat files.

This plugin provides multi-source knowledge lookup:
1. Queries the structured knowledge graph (knowledge_manager)
2. Falls back to legacy flat knowledge.txt for compatibility
3. Includes source citations and confidence scores
4. Triggers autonomous research for uncertain queries

Demonstrates integration of autonomous learning with existing plugin architecture.
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
        self._keywords = set()
        self._knowledge_manager = None
        self._learning_loop = None

        path = os.path.join("data", "knowledge.txt")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._lines = [l.strip() for l in f if l.strip()]
            except Exception:
                self._lines = []

        for line in self._lines:
            for token in line.lower().split():
                token = token.strip(".,!?;:'\"()[]{}")
                if len(token) >= 3:
                    self._keywords.add(token)

    def _get_knowledge_manager(self):
        """Lazy-load knowledge manager."""
        if self._knowledge_manager is None:
            try:
                from core.knowledge_manager import get_knowledge_manager
                self._knowledge_manager = get_knowledge_manager()
            except Exception:
                pass
        return self._knowledge_manager
    
    def _get_learning_loop(self):
        """Lazy-load learning loop for gap triggering."""
        if self._learning_loop is None:
            try:
                from core.learning_loop import get_learning_loop
                self._learning_loop = get_learning_loop()
            except Exception:
                pass
        return self._learning_loop

    def _ensure_embeddings(self, engine):
        if self._embeddings is None and self._lines:
            self._embeddings = engine._encode(self._lines)

    def can_handle(self, query: str) -> bool:
        # Check knowledge graph first
        km = self._get_knowledge_manager()
        if km is not None:
            try:
                results = km.search_by_topic(query, top_k=1)
                if results:
                    return True
            except Exception:
                pass
        
        # Fall back to flat file check
        if not self._lines:
            return False
        lowered = (query or "").lower()
        if not lowered.strip():
            return False
        if not self._keywords:
            return True
        for token in lowered.split():
            token = token.strip(".,!?;:'\"()[]{}")
            if token in self._keywords:
                return True
        return False

    def handle(self, query: str, engine) -> str | None:
        """Handle knowledge lookup with graph-first strategy."""
        
        # Try structured knowledge graph first
        km = self._get_knowledge_manager()
        if km is not None:
            try:
                results = km.search_by_topic(query, top_k=3)
                if results:
                    best = results[0]
                    confidence = best.get("confidence", 0.7)
                    sources = best.get("sources", [])
                    
                    # Trigger research if confidence is moderate
                    if confidence < 0.7 and confidence > 0.4:
                        learning_loop = self._get_learning_loop()
                        if learning_loop:
                            try:
                                learning_loop.trigger_gap_research(query, confidence=confidence)
                            except Exception:
                                pass
                    
                    # Format response with citations
                    response = f"From my knowledge: \"{best.get('content', '')}\" "
                    if confidence < 0.8:
                        response += f"(Confidence: {confidence:.0%}) "
                    if sources:
                        response += f"(Source: {sources[0] if isinstance(sources, list) else sources})"
                    
                    return response
            except Exception:
                pass
        
        # Fall back to flat file search
        if not self._lines:
            return None
        self._ensure_embeddings(engine)
        try:
            q_emb = engine._encode(query)
            scores = util.cos_sim(q_emb, self._embeddings)
            score, idx = scores.max(dim=1)
            if score.item() > min(engine.similarity_threshold, 0.5):
                line = self._lines[int(idx.item())]
                return f"According to my notes: '{line}'"
        except Exception:
            return None
        return None
