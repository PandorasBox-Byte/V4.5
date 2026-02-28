"""Evolution engine for continuous autonomous improvement.

This module coordinates all autonomous learning systems to enable
continuous self-evolution with minimal human intervention:

- Metrics analysis: detect performance trends and regressions
- Error pattern detection: trigger retraining when error rate rises
- Knowledge gap closure: prioritize learning high-impact gaps
- Policy optimization: incrementally improve decision routing
- Safety gating: pause evolution if performance degrades

The evolution engine runs on a weekly cycle, analyzing system state
and proposing/implementing improvements.
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class EvolutionReport:
    """Report on system evolution during a cycle."""
    cycle_timestamp: str
    metrics_analyzed: Dict
    gaps_identified: List[str]
    errors_detected: List[str]
    proposed_actions: List[Dict]
    actions_taken: List[Dict]
    system_health: float  # 0-1, overall health score


class EvolutionEngine:
    """Coordinates autonomous system evolution and improvement.
    
    Analyzes metrics, detects issues, and triggers improvements across
    all autonomous learning subsystems.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize evolution engine.
        
        Args:
            data_dir: Directory for storing evolution state.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.evolution_history_path = self.data_dir / "evolution_history.jsonl"
        self.evolution_config_path = self.data_dir / "evolution_config.json"
        
        self.config = self._load_config()
        
        # Component references (lazy-loaded)
        self._metrics_tracker = None
        self._error_learner = None
        self._learning_loop = None
        self._continuous_trainer = None
        self._curriculum_planner = None
        self._gap_detector = None
    
    @property
    def metrics_tracker(self):
        """Lazy-load metrics tracker."""
        if self._metrics_tracker is None:
            from core.metrics_tracker import get_metrics_tracker
            self._metrics_tracker = get_metrics_tracker(str(self.data_dir))
        return self._metrics_tracker
    
    @property
    def error_learner(self):
        """Lazy-load error learner."""
        if self._error_learner is None:
            from core.error_learner import get_error_learner
            self._error_learner = get_error_learner(str(self.data_dir))
        return self._error_learner
    
    @property
    def learning_loop(self):
        """Lazy-load learning loop."""
        if self._learning_loop is None:
            from core.learning_loop import get_learning_loop
            self._learning_loop = get_learning_loop(str(self.data_dir))
        return self._learning_loop
    
    @property
    def continuous_trainer(self):
        """Lazy-load continuous trainer."""
        if self._continuous_trainer is None:
            from core.continuous_trainer import get_continuous_trainer
            self._continuous_trainer = get_continuous_trainer(str(self.data_dir))
        return self._continuous_trainer
    
    @property
    def curriculum_planner(self):
        """Lazy-load curriculum planner."""
        if self._curriculum_planner is None:
            from core.curriculum_planner import get_curriculum_planner
            self._curriculum_planner = get_curriculum_planner(str(self.data_dir))
        return self._curriculum_planner
    
    @property
    def gap_detector(self):
        """Lazy-load gap detector."""
        if self._gap_detector is None:
            from core.gap_detector import get_gap_detector
            self._gap_detector = get_gap_detector(str(self.data_dir))
        return self._gap_detector
    
    def _load_config(self) -> Dict:
        """Load evolution configuration."""
        default_config = {
            "safety_mode_enabled": True,
            "auto_training_enabled": True,
            "auto_gap_research_enabled": True,
            "performance_threshold": 0.75,
            "error_rate_threshold": 0.15,
            "max_regression_tolerance": 0.10,
            "weekly_cycle_enabled": True
        }
        
        if self.evolution_config_path.exists():
            try:
                with open(self.evolution_config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load evolution config: {e}")
        
        return default_config
    
    def run_evolution_cycle(self) -> EvolutionReport:
        """Run one evolution cycle (typically weekly).
        
        Analyzes system state and triggers improvements.
        
        Returns:
            EvolutionReport with actions taken.
        """
        cycle_timestamp = datetime.now().isoformat()
        logger.info("Starting evolution cycle")
        
        try:
            # 1. Analyze metrics
            metrics = self._analyze_metrics()
            
            # 2. Detect errors and issues
            errors = self._analyze_errors()
            
            # 3. Identify knowledge gaps
            gaps = self._identify_gaps()
            
            # 4. Propose actions
            proposed_actions = self._propose_actions(metrics, errors, gaps)
            
            # 5. Apply actions
            actions_taken = self._apply_actions(proposed_actions)
            
            # 6. Calculate system health
            system_health = self._calculate_system_health(metrics, errors)
            
            # Create report
            report = EvolutionReport(
                cycle_timestamp=cycle_timestamp,
                metrics_analyzed=metrics,
                gaps_identified=gaps,
                errors_detected=errors,
                proposed_actions=proposed_actions,
                actions_taken=actions_taken,
                system_health=system_health
            )
            
            # Log report
            self._log_evolution(report)
            
            logger.info(f"Evolution cycle complete. System health: {system_health:.2f}")
            return report
        
        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")
            raise
    
    def _analyze_metrics(self) -> Dict:
        """Analyze performance metrics to detect trends.
        
        Returns:
            Dict with metric analysis.
        """
        logger.debug("Analyzing metrics...")
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.metrics_tracker.get_summary_stats(),
            "regression_detected": False,
            "trend": None
        }
        
        try:
            # Check for regression
            if self.metrics_tracker.detect_regression(window_size=10):
                metrics["regression_detected"] = True
                logger.warning("Performance regression detected")
            
            # Get trend
            trend = self.metrics_tracker.get_trend()
            metrics["trend"] = trend
            
        except Exception as e:
            logger.error(f"Metrics analysis failed: {e}")
        
        return metrics
    
    def _analyze_errors(self) -> List[str]:
        """Analyze error patterns and system health.
        
        Returns:
            List of identified error issues.
        """
        logger.debug("Analyzing errors...")
        
        errors = []
        
        try:
            health = self.error_learner.get_system_health()
            
            if health["status"] == "critical":
                errors.append("System in critical health due to high error rate")
            elif health["status"] == "degraded":
                errors.append("System health degraded, retraining recommended")
            
            # Check for specific error patterns
            for trigger in health.get("retraining_triggers", []):
                errors.append(f"Trigger: {trigger}")
        
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
        
        return errors
    
    def _identify_gaps(self) -> List[str]:
        """Identify knowledge gaps to research.
        
        Returns:
            List of gap topics.
        """
        logger.debug("Identifying gaps...")
        
        gaps = []
        
        try:
            # Get top gaps from gap detector
            top_gaps = self.gap_detector.get_top_gaps(limit=5)
            
            for gap in top_gaps:
                gaps.append(gap["query"])
        
        except Exception as e:
            logger.error(f"Gap identification failed: {e}")
        
        return gaps
    
    def _propose_actions(self, metrics: Dict, errors: List[str], gaps: List[str]) -> List[Dict]:
        """Propose improvements based on analysis.
        
        Args:
            metrics: Analyzed metrics.
            errors: Identified errors.
            gaps: Knowledge gaps found.
            
        Returns:
            List of proposed action dicts.
        """
        logger.debug("Proposing actions...")
        
        actions = []
        
        # Propose training if errors detected
        if errors or metrics.get("regression_detected"):
            actions.append({
                "action": "trigger_training",
                "description": "Retrain models due to detected errors or regression",
                "priority": 0.9,
                "components": ["embeddings", "decision_policy"]
            })
        
        # Propose gap research
        if self.config.get("auto_gap_research_enabled", True):
            for gap in gaps[:3]:  # Top 3 gaps
                actions.append({
                    "action": "research_gap",
                    "description": f"Research knowledge gap: {gap}",
                    "priority": 0.7,
                    "gap_topic": gap
                })
        
        # Propose curriculum update
        if self.config.get("auto_training_enabled", True):
            actions.append({
                "action": "update_curriculum",
                "description": "Update learning curriculum based on performance",
                "priority": 0.6
            })
        
        return actions
    
    def _apply_actions(self, proposed_actions: List[Dict]) -> List[Dict]:
        """Apply proposed improvements.
        
        Args:
            proposed_actions: List of proposed actions.
            
        Returns:
            List of actions actually taken.
        """
        logger.debug(f"Applying {len(proposed_actions)} actions...")
        
        actions_taken = []
        
        # Check safety gate
        if self.config.get("safety_mode_enabled", True):
            health = self.error_learner.get_system_health()
            if health["status"] == "critical":
                logger.warning("Safety gate activated: system in critical state, pausing evolution")
                return []
        
        for action in proposed_actions:
            try:
                if action["action"] == "trigger_training":
                    self.continuous_trainer.train_async()
                    actions_taken.append({
                        "action": "trigger_training",
                        "status": "initiated",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif action["action"] == "research_gap":
                    self.learning_loop.trigger_gap_research(
                        action["gap_topic"],
                        confidence=0.5
                    )
                    actions_taken.append({
                        "action": "research_gap",
                        "gap_topic": action["gap_topic"],
                        "status": "queued",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif action["action"] == "update_curriculum":
                    priorities = self.curriculum_planner.get_learning_priorities()
                    actions_taken.append({
                        "action": "update_curriculum",
                        "status": "updated",
                        "top_domain": priorities[0]["domain"] if priorities else None,
                        "timestamp": datetime.now().isoformat()
                    })
            
            except Exception as e:
                logger.error(f"Failed to apply action {action['action']}: {e}")
                actions_taken.append({
                    "action": action["action"],
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return actions_taken
    
    def _calculate_system_health(self, metrics: Dict, errors: List[str]) -> float:
        """Calculate overall system health score.
        
        Args:
            metrics: Analyzed metrics.
            errors: Identified errors.
            
        Returns:
            Health score 0-1.
        """
        try:
            health = self.error_learner.get_system_health()
            base_health = health["health_score"]
            
            # Adjust for regression
            if metrics.get("regression_detected"):
                base_health *= 0.8
            
            # Adjust for error count
            if errors:
                base_health *= (1.0 - min(len(errors) * 0.1, 0.5))
            
            return max(0.0, min(base_health, 1.0))
        
        except Exception:
            return 0.5  # Default neutral
    
    def _log_evolution(self, report: EvolutionReport):
        """Log evolution cycle results."""
        try:
            entry = {
                "cycle_timestamp": report.cycle_timestamp,
                "gaps_identified_count": len(report.gaps_identified),
                "errors_detected_count": len(report.errors_detected),
                "actions_proposed": len(report.proposed_actions),
                "actions_taken": len(report.actions_taken),
                "system_health": report.system_health
            }
            
            with open(self.evolution_history_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to log evolution: {e}")
    
    def get_evolution_status(self) -> Dict:
        """Get current evolution status and health.
        
        Returns:
            Dict with status and metrics.
        """
        try:
            error_health = self.error_learner.get_system_health()
            gap_priorities = self.curriculum_planner.get_learning_priorities()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system_health": error_health["health_score"],
                "system_status": error_health["status"],
                "total_errors": error_health["total_errors"],
                "error_patterns": error_health["pattern_count"],
                "top_knowledge_gap": gap_priorities[0]["domain"] if gap_priorities else None,
                "learning_queue_size": len(self.learning_loop.learning_queue),
                "config": self.config
            }
        except Exception as e:
            logger.error(f"Failed to get evolution status: {e}")
            return {"error": str(e)}


# Singleton instance
_instance: Optional[EvolutionEngine] = None


def get_evolution_engine(data_dir: str = "data") -> EvolutionEngine:
    """Get or create evolution engine singleton.
    
    Args:
        data_dir: Directory for storing evolution state.
        
    Returns:
        EvolutionEngine instance.
    """
    global _instance
    if _instance is None:
        _instance = EvolutionEngine(data_dir=data_dir)
    return _instance
