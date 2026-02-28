#!/usr/bin/env python3
"""Knowledge Gap Detector - Identifies what the engine doesn't know.

Analyzes response confidence to detect knowledge gaps that should trigger
autonomous learning. Tracks gap frequency and priority for curriculum planning.

Gap Detection Criteria:
  - Similarity score < 0.4: definite gap
  - Similarity score 0.4-0.6: uncertain/partial knowledge
  - Similarity score > 0.6: confident knowledge

Environment Variables:
  EVOAI_GAP_THRESHOLD: Similarity threshold for gap detection (default: 0.4)
  EVOAI_UNCERTAIN_THRESHOLD: Threshold for uncertain knowledge (default: 0.6)
  EVOAI_GAP_TRACKING_ENABLED: Enable gap tracking (default: 1)
"""

import os
import json
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter
import hashlib


class GapDetector:
    """Detects and tracks knowledge gaps from low-confidence responses."""
    
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.data_dir = Path(self.workspace_root) / "data"
        self.gaps_file = self.data_dir / "knowledge_gaps.json"
        
        # Thresholds
        self.gap_threshold = float(os.environ.get("EVOAI_GAP_THRESHOLD", "0.4"))
        self.uncertain_threshold = float(os.environ.get("EVOAI_UNCERTAIN_THRESHOLD", "0.6"))
        self.enabled = os.environ.get("EVOAI_GAP_TRACKING_ENABLED", "1").lower() in ("1", "true", "yes")
        
        # Load gaps
        self.gaps = self._load_gaps()
    
    def _load_gaps(self) -> Dict[str, Any]:
        """Load tracked knowledge gaps from JSON."""
        if not self.gaps_file.exists():
            return {"gaps": {}, "metadata": {"last_updated": time.time()}}
        
        try:
            with open(self.gaps_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"gaps": {}, "metadata": {"last_updated": time.time()}}
    
    def _save_gaps(self):
        """Save gaps to JSON file."""
        self.gaps["metadata"]["last_updated"] = time.time()
        
        try:
            with open(self.gaps_file, 'w', encoding='utf-8') as f:
                json.dump(self.gaps, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save knowledge gaps: {e}")
    
    def _extract_keywords(self, query: str, top_n: int = 5) -> List[str]:
        """Extract key terms from query using simple heuristics.
        
        In a full implementation, this would use TF-IDF or similar.
        For now, using simple keyword extraction.
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'what', 'when', 'where', 'who',
            'how', 'why', 'which', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Return most frequent (simple frequency count)
        word_counts = Counter(keywords)
        return [word for word, _ in word_counts.most_common(top_n)]
    
    def _generate_gap_id(self, topic: str) -> str:
        """Generate unique ID for a gap topic."""
        return hashlib.md5(topic.lower().encode()).hexdigest()[:12]
    
    def detect_gap(self, query: str, similarity_score: float, 
                   response: Optional[str] = None) -> Tuple[bool, str]:
        """Detect if query reveals a knowledge gap.
        
        Args:
            query: User query
            similarity_score: Similarity/confidence score (0-1)
            response: Generated response (optional, for analysis)
            
        Returns:
            (is_gap, confidence_level) where confidence_level is:
              "gap" (< 0.4), "uncertain" (0.4-0.6), "confident" (> 0.6)
        """
        if not self.enabled:
            return False, "confident"
        
        # Determine confidence level
        if similarity_score < self.gap_threshold:
            return True, "gap"
        elif similarity_score < self.uncertain_threshold:
            return True, "uncertain"
        else:
            return False, "confident"
    
    def record_gap(self, query: str, similarity_score: float, 
                   user_domain: Optional[str] = None):
        """Record a detected knowledge gap.
        
        Args:
            query: Query that revealed the gap
            similarity_score: Confidence score
            user_domain: Optional domain/topic classification
        """
        if not self.enabled:
            return
        
        # Extract topic keywords
        keywords = self._extract_keywords(query)
        if not keywords:
            return
        
        # Use primary keyword as topic
        topic = keywords[0]
        gap_id = self._generate_gap_id(topic)
        
        # Update or create gap entry
        if gap_id in self.gaps["gaps"]:
            gap = self.gaps["gaps"][gap_id]
            gap["count"] += 1
            gap["last_seen"] = time.time()
            gap["avg_confidence"] = (gap["avg_confidence"] * (gap["count"] - 1) + similarity_score) / gap["count"]
            
            # Update keywords (merge unique ones)
            existing_keywords = set(gap.get("keywords", []))
            gap["keywords"] = list(existing_keywords.union(set(keywords)))[:10]  # Keep top 10
            
            # Update queries
            gap.setdefault("example_queries", []).append(query)
            gap["example_queries"] = gap["example_queries"][-5:]  # Keep last 5
        else:
            self.gaps["gaps"][gap_id] = {
                "topic": topic,
                "count": 1,
                "first_seen": time.time(),
                "last_seen": time.time(),
                "avg_confidence": similarity_score,
                "keywords": keywords,
                "example_queries": [query],
                "user_domain": user_domain,
                "priority": 0.0,  # Computed by curriculum planner
                "researched": False
            }
        
        self._save_gaps()
    
    def calculate_priority(self, gap_id: str, domain_weight: float = 1.0) -> float:
        """Calculate priority score for a gap.
        
        Priority = frequency × recency_factor × domain_weight
        
        Args:
            gap_id: Gap ID
            domain_weight: Weight based on user's domain interest (1.0 = neutral)
            
        Returns:
            Priority score (higher = more important)
        """
        gap = self.gaps["gaps"].get(gap_id)
        if not gap:
            return 0.0
        
        # Frequency component
        frequency = gap["count"]
        
        # Recency component (decay over time)
        time_since_last = time.time() - gap["last_seen"]
        days_since = time_since_last / 86400
        recency_factor = 1.0 / (1.0 + (days_since / 7.0))  # Decay over weeks
        
        # Combined priority
        priority = frequency * recency_factor * domain_weight
        
        # Update stored priority
        gap["priority"] = priority
        
        return priority
    
    def get_top_gaps(self, n: int = 10, only_unresearched: bool = True) -> List[Dict[str, Any]]:
        """Get top N priority gaps for research.
        
        Args:
            n: Number of gaps to return
            only_unresearched: Only return gaps not yet researched
            
        Returns:
            List of gap dicts sorted by priority
        """
        # Recalculate all priorities
        for gap_id in self.gaps["gaps"]:
            self.calculate_priority(gap_id)
        
        # Filter and sort
        gaps_list = []
        for gap_id, gap in self.gaps["gaps"].items():
            if only_unresearched and gap.get("researched", False):
                continue
            gaps_list.append({**gap, "id": gap_id})
        
        # Sort by priority (descending)
        gaps_list.sort(key=lambda x: x["priority"], reverse=True)
        
        return gaps_list[:n]
    
    def mark_researched(self, gap_id: str):
        """Mark a gap as researched/resolved."""
        if gap_id in self.gaps["gaps"]:
            self.gaps["gaps"][gap_id]["researched"] = True
            self.gaps["gaps"][gap_id]["researched_at"] = time.time()
            self._save_gaps()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gap tracking statistics."""
        total_gaps = len(self.gaps["gaps"])
        researched = sum(1 for g in self.gaps["gaps"].values() if g.get("researched", False))
        
        if total_gaps == 0:
            avg_confidence = 0.0
        else:
            avg_confidence = sum(g["avg_confidence"] for g in self.gaps["gaps"].values()) / total_gaps
        
        return {
            "total_gaps": total_gaps,
            "researched": researched,
            "pending": total_gaps - researched,
            "avg_confidence": avg_confidence,
            "last_updated": self.gaps["metadata"].get("last_updated", 0)
        }
    
    def clear_old_gaps(self, days: int = 90):
        """Remove gaps not seen in X days."""
        cutoff = time.time() - (days * 86400)
        initial_count = len(self.gaps["gaps"])
        
        self.gaps["gaps"] = {
            gid: gap for gid, gap in self.gaps["gaps"].items()
            if gap.get("last_seen", 0) >= cutoff
        }
        
        removed = initial_count - len(self.gaps["gaps"])
        if removed > 0:
            print(f"Cleared {removed} old knowledge gaps")
            self._save_gaps()
        
        return removed


# Singleton instance
_gap_detector = None


def get_gap_detector(workspace_root: Optional[str] = None) -> GapDetector:
    """Get or create singleton GapDetector instance."""
    global _gap_detector
    if _gap_detector is None:
        _gap_detector = GapDetector(workspace_root)
    return _gap_detector


if __name__ == "__main__":
    # Test gap detector
    gd = get_gap_detector()
    
    print("Gap Detector Test")
    print(f"Initial stats: {gd.get_stats()}")
    
    # Simulate detecting a gap
    is_gap, level = gd.detect_gap("What is quantum entanglement?", similarity_score=0.3)
    print(f"\nGap detected: {is_gap}, level: {level}")
    
    if is_gap:
        gd.record_gap("What is quantum entanglement?", 0.3, user_domain="science")
    
    # Get top gaps
    top_gaps = gd.get_top_gaps(n=5)
    print(f"\nTop {len(top_gaps)} gaps:")
    for gap in top_gaps:
        print(f"  - {gap['topic']}: priority={gap['priority']:.2f}, count={gap['count']}")
    
    print(f"\nFinal stats: {gd.get_stats()}")
