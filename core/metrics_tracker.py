#!/usr/bin/env python3
"""Performance Metrics Tracker - Tracks system performance for evolution.

Tracks per-session metrics:
  - Response latency
  - Decision confidence  
  - User corrections/errors
  - Error rate
  - Knowledge coverage
  
Detects regressions and trends for autonomous optimization.

Stored in data/performance_metrics.jsonl (JSONL format for append-only writes)

Environment Variables:
  EVOAI_METRICS_ENABLED: Enable metrics tracking (default: 1)
  EVOAI_METRICS_REGRESSION_THRESHOLD: % drop to trigger alert (default: 10)
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
import statistics


class MetricsTracker:
    """Tracks performance metrics for autonomous evolution."""
    
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.data_dir = Path(self.workspace_root) / "data"
        self.metrics_file = self.data_dir / "performance_metrics.jsonl"
        
        self.enabled = os.environ.get("EVOAI_METRICS_ENABLED", "1").lower() in ("1", "true", "yes")
        self.regression_threshold = float(os.environ.get("EVOAI_METRICS_REGRESSION_THRESHOLD", "10"))
        
        # Current session metrics (in-memory)
        self.session = {
            "session_id": str(int(time.time())),
            "start_time": time.time(),
            "responses": 0,
            "errors": 0,
            "corrections": 0,
            "total_latency": 0.0,
            "total_confidence": 0.0,
            "actions_taken": defaultdict(int),
            "knowledge_lookups": 0,
            "knowledge_hits": 0
        }
    
    def record_response(self, latency: float, confidence: float, action: Optional[str] = None):
        """Record a response completion.
        
        Args:
            latency: Response time in seconds
            confidence: Decision confidence (0-1)
            action: Action type taken (e.g., "llm_generate", "code_assist")
        """
        if not self.enabled:
            return
        
        self.session["responses"] += 1
        self.session["total_latency"] += latency
        self.session["total_confidence"] += confidence
        
        if action:
            self.session["actions_taken"][action] += 1
    
    def record_error(self, error_type: str = "general"):
        """Record an error occurrence.
        
        Args:
            error_type: Type of error (e.g., "generation_failed", "safety_blocked")
        """
        if not self.enabled:
            return
        
        self.session["errors"] += 1
        self.session.setdefault("error_types", defaultdict(int))
        self.session["error_types"][error_type] += 1
    
    def record_correction(self):
        """Record a user correction (indicates wrong/insufficient response)."""
        if not self.enabled:
            return
        
        self.session["corrections"] += 1
    
    def record_knowledge_lookup(self, hit: bool):
        """Record a knowledge base lookup.
        
        Args:
            hit: True if knowledge was found, False if miss
        """
        if not self.enabled:
            return
        
        self.session["knowledge_lookups"] += 1
        if hit:
            self.session["knowledge_hits"] += 1
    
    def compute_session_metrics(self) -> Dict[str, Any]:
        """Compute final metrics for current session."""
        if self.session["responses"] == 0:
            return {
                "avg_latency": 0.0,
                "avg_confidence": 0.0,
                "error_rate": 0.0,
                "correction_rate": 0.0,
                "knowledge_hit_rate": 0.0
            }
        
        avg_latency = self.session["total_latency"] / self.session["responses"]
        avg_confidence = self.session["total_confidence"] / self.session["responses"]
        error_rate = self.session["errors"] / self.session["responses"]
        correction_rate = self.session["corrections"] / self.session["responses"]
        
        knowledge_hit_rate = 0.0
        if self.session["knowledge_lookups"] > 0:
            knowledge_hit_rate = self.session["knowledge_hits"] / self.session["knowledge_lookups"]
        
        return {
            "avg_latency": avg_latency,
            "avg_confidence": avg_confidence,
            "error_rate": error_rate,
            "correction_rate": correction_rate,
            "knowledge_hit_rate": knowledge_hit_rate
        }
    
    def end_session(self):
        """End current session and save metrics."""
        if not self.enabled:
            return
        
        self.session["end_time"] = time.time()
        self.session["duration"] = self.session["end_time"] - self.session["start_time"]
        
        # Compute final metrics
        final_metrics = self.compute_session_metrics()
        self.session.update(final_metrics)
        
        # Convert defaultdict to regular dict for JSON serialization
        if "actions_taken" in self.session:
            self.session["actions_taken"] = dict(self.session["actions_taken"])
        if "error_types" in self.session:
            self.session["error_types"] = dict(self.session["error_types"])
        
        # Append to JSONL file
        try:
            self.data_dir.mkdir(exist_ok=True)
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(self.session) + '\n')
        except IOError as e:
            print(f"Warning: Could not save metrics: {e}")
    
    def get_recent_sessions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get N most recent session metrics.
        
        Args:
            n: Number of sessions to retrieve
            
        Returns:
            List of session dicts, most recent first
        """
        if not self.metrics_file.exists():
            return []
        
        try:
            sessions = []
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sessions.append(json.loads(line))
            
            # Return last N sessions
            return sessions[-n:][::-1]  # Reverse for most recent first
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load metrics: {e}")
            return []
    
    def detect_regression(self, metric: str = "avg_confidence", 
                         lookback: int = 5) -> Optional[Dict[str, Any]]:
        """Detect performance regression in a specific metric.
        
        Args:
            metric: Metric to check (e.g., "avg_confidence", "error_rate")
            lookback: Number of sessions to compare
            
        Returns:
            Dict with regression info if detected, None otherwise
        """
        sessions = self.get_recent_sessions(n=lookback)
        if len(sessions) < 2:
            return None
        
        # Extract metric values
        values = [s.get(metric, 0) for s in sessions if metric in s]
        if len(values) < 2:
            return None
        
        # Compare latest vs historical average
        latest = values[-1]
        historical_avg = statistics.mean(values[:-1])
        
        # Calculate % change
        if historical_avg == 0:
            return None
        
        pct_change = ((latest - historical_avg) / historical_avg) * 100
        
        # For error_rate and correction_rate, an increase is bad
        # For other metrics, a decrease is bad
        is_regression = False
        if metric in ("error_rate", "correction_rate"):
            is_regression = pct_change > self.regression_threshold
        else:
            is_regression = pct_change < -self.regression_threshold
        
        if is_regression:
            return {
                "metric": metric,
                "latest": latest,
                "historical_avg": historical_avg,
                "pct_change": pct_change,
                "sessions_compared": len(values)
            }
        
        return None
    
    def get_trend(self, metric: str = "avg_confidence", 
                  lookback: int = 10) -> Optional[str]:
        """Get trend direction for a metric.
        
        Args:
            metric: Metric to analyze
            lookback: Number of sessions to analyze
            
        Returns:
            "improving", "declining", or "stable" (or None if insufficient data)
        """
        sessions = self.get_recent_sessions(n=lookback)
        if len(sessions) < 3:
            return None
        
        values = [s.get(metric, 0) for s in sessions if metric in s]
        if len(values) < 3:
            return None
        
        # Simple linear trend: compare first half vs second half averages
        mid = len(values) // 2
        first_half_avg = statistics.mean(values[:mid])
        second_half_avg = statistics.mean(values[mid:])
        
        if first_half_avg == 0:
            return "stable"
        
        pct_change = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        # For error_rate/correction_rate, lower is better
        if metric in ("error_rate", "correction_rate"):
            if pct_change < -5:
                return "improving"
            elif pct_change > 5:
                return "declining"
        else:
            if pct_change > 5:
                return "improving"
            elif pct_change < -5:
                return "declining"
        
        return "stable"
    
    def get_summary_stats(self, lookback: int = 30) -> Dict[str, Any]:
        """Get summary statistics across recent sessions.
        
        Args:
            lookback: Number of sessions to include
            
        Returns:
            Dict of aggregated stats
        """
        sessions = self.get_recent_sessions(n=lookback)
        if not sessions:
            return {
                "total_sessions": 0,
                "total_responses": 0,
                "avg_latency": 0.0,
                "avg_confidence": 0.0,
                "error_rate": 0.0,
                "trends": {}
            }
        
        total_responses = sum(s.get("responses", 0) for s in sessions)
        total_errors = sum(s.get("errors", 0) for s in sessions)
        
        # Average metrics across sessions
        latencies = [s.get("avg_latency", 0) for s in sessions if "avg_latency" in s]
        confidences = [s.get("avg_confidence", 0) for s in sessions if "avg_confidence" in s]
        
        return {
            "total_sessions": len(sessions),
            "total_responses": total_responses,
            "total_errors": total_errors,
            "avg_latency": statistics.mean(latencies) if latencies else 0.0,
            "avg_confidence": statistics.mean(confidences) if confidences else 0.0,
            "error_rate": (total_errors / total_responses) if total_responses > 0 else 0.0,
            "trends": {
                "confidence": self.get_trend("avg_confidence", lookback),
                "latency": self.get_trend("avg_latency", lookback),
                "error_rate": self.get_trend("error_rate", lookback)
            }
        }


# Singleton instance
_metrics_tracker = None


def get_metrics_tracker(workspace_root: Optional[str] = None) -> MetricsTracker:
    """Get or create singleton MetricsTracker instance."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker(workspace_root)
    return _metrics_tracker


if __name__ == "__main__":
    # Test metrics tracker
    mt = get_metrics_tracker()
    
    print("Metrics Tracker Test")
    
    # Simulate some responses
    mt.record_response(latency=0.5, confidence=0.8, action="llm_generate")
    mt.record_response(latency=0.3, confidence=0.9, action="simple_reply")
    mt.record_error("generation_failed")
    mt.record_knowledge_lookup(hit=True)
    mt.record_knowledge_lookup(hit=False)
    
    # Compute session metrics
    session_metrics = mt.compute_session_metrics()
    print(f"\nCurrent session metrics:")
    for key, value in session_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # End session
    mt.end_session()
    
    # Get summary stats
    summary = mt.get_summary_stats()
    print(f"\nSummary stats (last 30 sessions):")
    print(f"  Total sessions: {summary['total_sessions']}")
    print(f"  Total responses: {summary['total_responses']}")
    print(f"  Avg confidence: {summary['avg_confidence']:.3f}")
    print(f"  Error rate: {summary['error_rate']:.3f}")
