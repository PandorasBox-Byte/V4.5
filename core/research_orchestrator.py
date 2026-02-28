#!/usr/bin/env python3
"""Research Orchestrator - Multi-source web research coordination.

Coordinates research from multiple sources:
  - Wikipedia API
  - Stack Overflow API
  - arXiv API
  - General web scraping (allowlist-controlled)

Includes content extraction, summarization, and deduplication.

Environment Variables:
  EVOAI_RESEARCH_MAX_SOURCES: Max sources per query (default: 3)
  EVOAI_RESEARCH_MULTI_HOP: Enable multi-hop research (default: 0)
  EVOAI_RESEARCH_SUMMARIZE: Enable LLM summarization (default: 1)
"""

import os
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse


class ResearchOrchestrator:
    """Coordinates multi-source research and content extraction."""
    
    def __init__(self):
        self.max_sources = int(os.environ.get("EVOAI_RESEARCH_MAX_SOURCES", "3"))
        self.multi_hop_enabled = os.environ.get("EVOAI_RESEARCH_MULTI_HOP", "0").lower() in ("1", "true")
        self.summarize_enabled = os.environ.get("EVOAI_RESEARCH_SUMMARIZE", "1").lower() in ("1", "true")
        
        # Import dependencies (lazy loading)
        self._api_integrations = None
        self._source_validator = None
        self._local_llm = None
    
    def _get_api_integrations(self):
        """Lazy load API integrations."""
        if self._api_integrations is None:
            from core.api_integrations import get_api_integrations
            self._api_integrations = get_api_integrations()
        return self._api_integrations
    
    def _get_source_validator(self):
        """Lazy load source validator."""
        if self._source_validator is None:
            from core.source_validator import get_source_validator
            self._source_validator = get_source_validator()
        return self._source_validator
    
    def _get_local_llm(self):
        """Lazy load local LLM."""
        if self._local_llm is None:
            from core.local_llm import get_local_llm
            self._local_llm = get_local_llm()
        return self._local_llm
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query."""
        # Remove question words
        topic = re.sub(r'^(what|how|why|when|where|who|is|are|does|do)\s+', '', query.lower(), flags=re.IGNORECASE)
        # Remove trailing question marks
        topic = topic.rstrip('?').strip()
        return topic
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        if not results:
            return []
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Use first 100 chars as fingerprint
            fingerprint = result.get("content", "")[:100].lower()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                unique_results.append(result)
        
        return unique_results
    
    def _summarize_content(self, content: str, max_length: int = 500) -> str:
        """Summarize content using local LLM or simple truncation.
        
        Args:
            content: Content to summarize
            max_length: Max summary length in characters
            
        Returns:
            Summarized content
        """
        if not self.summarize_enabled:
            return content[:max_length]
        
        # Try local LLM summarization
        llm = self._get_local_llm()
        if llm.is_available():
            prompt = f"Summarize this in 2-3 sentences:\n\n{content[:2000]}"
            summary = llm.generate(prompt, max_tokens=100)
            if summary:
                return summary[:max_length]
        
        # Fallback to simple truncation
        if len(content) <= max_length:
            return content
        
        # Truncate at sentence boundary
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length // 2:
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def research_topic(self, query: str, topic: Optional[str] = None) -> Dict[str, Any]:
        """Research a topic from multiple sources.
        
        Args:
            query: User query
            topic: Extracted topic (auto-extracted if not provided)
            
        Returns:
            {
                "topic": str,
                "summary": str,
                "sources": [{url, title, confidence, content}],
                "confidence": float,
                "embedding": None  # Computed later by engine
            }
        """
        if topic is None:
            topic = self._extract_topic(query)
        
        api = self._get_api_integrations()
        validator = self._get_source_validator()
        
        all_results = []
        
        # Try Wikipedia first (highest confidence)
        wiki_result = api.search_wikipedia(topic)
        if wiki_result:
            all_results.append(wiki_result)
        
        # Try arXiv for scientific topics
        if any(word in query.lower() for word in ["quantum", "research", "study", "theory", "physics", "math", "science"]):
            arxiv_results = api.search_arxiv(topic, max_results=2)
            all_results.extend(arxiv_results)
        
        # Try Stack Overflow for programming topics
        if any(word in query.lower() for word in ["code", "program", "python", "javascript", "error", "bug", "function", "class"]):
            so_results = api.search_stackoverflow(topic, max_results=2)
            all_results.extend(so_results)
        
        # Validate and score all sources
        validated_results = []
        for result in all_results:
            is_valid, credibility = validator.validate_source(result["url"], result.get("content", ""))
            if is_valid:
                result["confidence"] = credibility
                validated_results.append(result)
        
        # Deduplicate
        unique_results = self._deduplicate_results(validated_results)
        
        # Limit to max sources
        top_results = sorted(unique_results, key=lambda x: x.get("confidence", 0), reverse=True)[:self.max_sources]
        
        if not top_results:
            return {
                "topic": topic,
                "summary": "",
                "sources": [],
                "confidence": 0.0,
                "embedding": None
            }
        
        # Combine content and summarize
        combined_content = "\n\n".join(r.get("content", "")[:500] for r in top_results)
        summary = self._summarize_content(combined_content, max_length=500)
        
        # Calculate overall confidence (weighted average by source confidence)
        total_weight = sum(r.get("confidence", 0.5) for r in top_results)
        overall_confidence = total_weight / len(top_results) if top_results else 0.0
        
        return {
            "topic": topic,
            "summary": summary,
            "sources": [
                {
                    "url": r["url"],
                    "title": r.get("title", ""),
                    "confidence": r.get("confidence", 0.5)
                }
                for r in top_results
            ],
            "confidence": overall_confidence,
            "embedding": None  # Computed later by engine
        }
    
    def multi_hop_research(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """Perform multi-hop research following references.
        
        Args:
            query: Initial query
            max_hops: Maximum number of reference hops
            
        Returns:
            Research result dict
        """
        if not self.multi_hop_enabled:
            return self.research_topic(query)
        
        # Start with initial research
        result = self.research_topic(query)
        
        # Extract topics from initial results for follow-up
        # This is simplified - a full implementation would parse references
        # For now, just return the initial result
        return result


# Singleton instance
_research_orchestrator = None


def get_research_orchestrator() -> ResearchOrchestrator:
    """Get or create singleton ResearchOrchestrator instance."""
    global _research_orchestrator
    if _research_orchestrator is None:
        _research_orchestrator = ResearchOrchestrator()
    return _research_orchestrator


if __name__ == "__main__":
    # Test research orchestrator
    ro = get_research_orchestrator()
    
    print("Research Orchestrator Test")
    
    # Test research
    result = ro.research_topic("What is quantum entanglement?")
    
    print(f"\nTopic: {result['topic']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Sources: {len(result['sources'])}")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['title']} (confidence: {source['confidence']:.2f})")
    print(f"\nSummary: {result['summary'][:200]}...")
