"""Error learning system for autonomous improvement.

This module detects and learns from errors to improve system performance:
- Captures user corrections and failed actions
- Detects error patterns and recurring issues
- Generates synthetic negative examples for training
- Implements heuristic adaptation for decision policy
- Triggers model retraining when error patterns emerge
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class ErrorEvent:
    """Record of an error or correction."""
    error_type: str  # "correction", "failed_action", "misclassified_intent", "knowledge_gap"
    context: str  # query or action that failed
    expected: Optional[str]  # what should have happened
    actual: str  # what actually happened
    pattern: Optional[str] = None  # detected error pattern
    related_errors: Optional[List[str]] = None  # related error IDs
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ErrorPattern:
    """Detected pattern of recurring errors."""
    pattern_id: str
    error_type: str
    pattern_description: str
    frequency: int  # how many times this pattern occurred
    severity: float = 0.5  # 0-1, impact if this error occurs
    occurrences: List[str] = field(default_factory=list)  # error IDs with this pattern
    mitigations: List[str] = field(default_factory=list)


class ErrorLearner:
    """Learns from errors to improve autonomous decision-making.
    
    Tracks errors, detects patterns, and feeds corrections back to training systems.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize error learner.
        
        Args:
            data_dir: Directory for storing error logs and patterns.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.errors_path = self.data_dir / "error_log.jsonl"
        self.patterns_path = self.data_dir / "error_patterns.json"
        self.negative_examples_path = self.data_dir / "error_examples.json"
        
        self.errors: Dict[str, ErrorEvent] = {}
        self.patterns: Dict[str, ErrorPattern] = {}
        self.error_counter = 0
        self.pattern_replan_threshold = 5  # trigger retraining after this many errors
        
        self._load_state()
        self._detect_patterns()
    
    def _load_state(self):
        """Load error history and patterns."""
        if self.errors_path.exists():
            try:
                with open(self.errors_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        error_id = entry.pop("error_id", None)
                        if error_id:
                            self.errors[error_id] = ErrorEvent(**entry)
            except Exception as e:
                logger.warning(f"Failed to load error history: {e}")
        
        if self.patterns_path.exists():
            try:
                with open(self.patterns_path, 'r') as f:
                    data = json.load(f)
                    for pattern_id, pattern_dict in data.items():
                        self.patterns[pattern_id] = ErrorPattern(**pattern_dict)
            except Exception as e:
                logger.warning(f"Failed to load error patterns: {e}")
    
    def _save_state(self):
        """Save error history and patterns."""
        try:
            with open(self.errors_path, 'a') as f:
                for error_id, error in list(self.errors.items())[-1:]:  # only new
                    entry = {
                        "error_id": error_id,
                        "error_type": error.error_type,
                        "context": error.context,
                        "expected": error.expected,
                        "actual": error.actual,
                        "pattern": error.pattern,
                        "related_errors": error.related_errors,
                        "timestamp": error.timestamp
                    }
                    f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to save error history: {e}")
        
        try:
            with open(self.patterns_path, 'w') as f:
                json.dump({
                    pattern_id: {
                        "pattern_id": p.pattern_id,
                        "error_type": p.error_type,
                        "pattern_description": p.pattern_description,
                        "frequency": p.frequency,
                        "occurrences": p.occurrences,
                        "severity": p.severity,
                        "mitigations": p.mitigations
                    }
                    for pattern_id, p in self.patterns.items()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error patterns: {e}")
    
    def record_error(self, error_type: str, context: str, expected: Optional[str] = None,
                     actual: str = "Unknown") -> str:
        """Record an error or user correction.
        
        Args:
            error_type: Type of error ("correction", "failed_action", "misclassified_intent", etc).
            context: Query/action that failed.
            expected: What should have happened.
            actual: What actually happened.
            
        Returns:
            Error ID for tracking and correlation.
        """
        error_id = f"err_{self.error_counter}_{int(datetime.now().timestamp())}"
        self.error_counter += 1
        
        error = ErrorEvent(
            error_type=error_type,
            context=context,
            expected=expected,
            actual=actual
        )
        
        self.errors[error_id] = error
        self._save_state()
        
        logger.debug(f"Recorded error {error_id}: {error_type} - {context}")
        
        return error_id
    
    def _detect_patterns(self):
        """Detect recurring error patterns from error history."""
        pattern_counts: Dict[str, List[str]] = defaultdict(list)
        
        for error_id, error in self.errors.items():
            # Pattern: same error type on similar contexts
            pattern_key = f"{error.error_type}:context_similarity"
            pattern_counts[pattern_key].append(error_id)
            
            # Pattern: specific error types
            if error.error_type == "correction":
                pattern_key = f"user_correction:{error.context[:20]}"
                pattern_counts[pattern_key].append(error_id)
        
        # Create or update patterns
        for pattern_key, error_ids in pattern_counts.items():
            if len(error_ids) >= self.pattern_replan_threshold:
                pattern_id = f"pat_{len(self.patterns)}"
                
                errors = [self.errors[eid] for eid in error_ids]
                error_type = errors[0].error_type
                
                self.patterns[pattern_id] = ErrorPattern(
                    pattern_id=pattern_id,
                    error_type=error_type,
                    pattern_description=f"Recurring {error_type} errors: {pattern_key}",
                    frequency=len(error_ids),
                    occurrences=error_ids,
                    severity=self._calculate_severity(errors)
                )
        
        self._save_state()
    
    def _calculate_severity(self, errors: List[ErrorEvent]) -> float:
        """Calculate severity of an error pattern.
        
        Args:
            errors: List of errors in the pattern.
            
        Returns:
            Severity score 0-1.
        """
        # Severity based on frequency and type
        frequency_score = min(len(errors) / 10, 1.0)
        
        # Type-based severity weights
        type_weights = {
            "correction": 0.6,
            "failed_action": 0.8,
            "misclassified_intent": 0.7,
            "knowledge_gap": 0.4
        }
        
        error_type = errors[0].error_type
        type_weight = type_weights.get(error_type, 0.5)
        
        severity = frequency_score * type_weight
        return min(severity, 1.0)
    
    def get_error_summary(self) -> Dict:
        """Get summary of recent errors and patterns.
        
        Returns:
            Dict with error statistics and high-severity patterns.
        """
        error_type_counts = defaultdict(int)
        for error in self.errors.values():
            error_type_counts[error.error_type] += 1
        
        # Identify high-severity patterns
        high_severity_patterns = [
            {
                "pattern_id": p.pattern_id,
                "description": p.pattern_description,
                "frequency": p.frequency,
                "severity": p.severity
            }
            for p in self.patterns.values()
            if p.severity > 0.6
        ]
        
        return {
            "total_errors": len(self.errors),
            "error_types": dict(error_type_counts),
            "total_patterns": len(self.patterns),
            "high_severity_patterns": high_severity_patterns,
            "requires_retraining": len(high_severity_patterns) > 0
        }
    
    def generate_negative_examples(self) -> List[Dict]:
        """Generate synthetic training examples from errors.
        
        These examples help train models to avoid repeating errors.
        
        Returns:
            List of negative example dicts with context, incorrect response, correct response.
        """
        negative_examples = []
        
        for error in self.errors.values():
            negative_example = {
                "context": error.context,
                "incorrect_response": error.actual,
                "correct_response": error.expected,
                "error_type": error.error_type,
                "timestamp": error.timestamp
            }
            negative_examples.append(negative_example)
        
        # Save to disk
        try:
            with open(self.negative_examples_path, 'w') as f:
                json.dump(negative_examples, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save negative examples: {e}")
        
        return negative_examples
    
    def get_retraining_triggers(self) -> List[str]:
        """Check if retraining is needed based on error patterns.
        
        Returns:
            List of trigger reasons (empty if no retraining needed).
        """
        triggers = []
        
        if len(self.errors) > 50:
            triggers.append("error_count_threshold_exceeded")
        
        # Check for high-severity patterns
        for pattern in self.patterns.values():
            if pattern.severity > 0.7 and pattern.frequency >= self.pattern_replan_threshold:
                triggers.append(f"high_severity_pattern: {pattern.pattern_id}")
        
        # Check for specific error type surges
        error_type_counts = defaultdict(int)
        for error in self.errors.values():
            error_type_counts[error.error_type] += 1
        
        for error_type, count in error_type_counts.items():
            if count > 20:
                triggers.append(f"surge_in_{error_type}")
        
        return triggers
    
    def get_system_health(self) -> Dict:
        """Assess overall system health based on error patterns.
        
        Returns:
            Dict with health score (0-1) and status.
        """
        summary = self.get_error_summary()
        retraining_triggers = self.get_retraining_triggers()
        
        # Calculate health score
        error_score = min(len(self.errors) / 100, 1.0)  # worse with more errors
        pattern_score = min(len(summary["high_severity_patterns"]) / 5, 1.0)
        
        health_score = max(0.0, 1.0 - (0.6 * error_score + 0.4 * pattern_score))
        
        if health_score > 0.8:
            status = "healthy"
        elif health_score > 0.6:
            status = "degraded"
        else:
            status = "critical"
        
        return {
            "health_score": health_score,
            "status": status,
            "total_errors": len(self.errors),
            "pattern_count": len(self.patterns),
            "retraining_needed": len(retraining_triggers) > 0,
            "retraining_triggers": retraining_triggers
        }


# Singleton instance
_instance: Optional[ErrorLearner] = None


def get_error_learner(data_dir: str = "data") -> ErrorLearner:
    """Get or create error learner singleton.
    
    Args:
        data_dir: Directory for storing error logs.
        
    Returns:
        ErrorLearner instance.
    """
    global _instance
    if _instance is None:
        _instance = ErrorLearner(data_dir=data_dir)
    return _instance
