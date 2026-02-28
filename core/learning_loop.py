"""Main learning loop for autonomous knowledge acquisition.

This module orchestrates the complete autonomous learning pipeline:
1. Detect knowledge gaps from low-confidence responses
2. Research top priority gaps using multi-source research
3. Validate research results with source credibility checks
4. Integrate validated knowledge into knowledge graph
5. Test retrieval to confirm integration
6. Queue for human review if confidence not met

Trigger modes:
- Post-response: after low-confidence response (confidence < 0.6)
- Background idle: when engine is idle for >5 minutes
- Scheduled: weekly refresh of high-priority gaps
"""

import json
import logging
import threading
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class LearningTask:
    """Task for the learning loop to process."""
    task_type: str  # "gap_research", "validation", "integration", "review"
    priority: float  # 0-1
    gap_topic: str
    research_results: Optional[List[Dict]] = None
    validation_result: Optional[Dict] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class LearningLoop:
    """Orchestrates autonomous learning pipeline.
    
    Manages the end-to-end process of detecting gaps, researching them,
    validating results, and integrating into knowledge base.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize learning loop.
        
        Args:
            data_dir: Directory for storing learning state.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.learning_queue_path = self.data_dir / "learning_queue.json"
        self.learning_history_path = self.data_dir / "learning_history.jsonl"
        
        self.learning_queue: List[LearningTask] = []
        self._load_queue()
        
        # Component references (lazy-loaded)
        self._gap_detector = None
        self._research_orchestrator = None
        self._source_validator = None
        self._knowledge_manager = None
    
    @property
    def gap_detector(self):
        """Lazy-load gap detector."""
        if self._gap_detector is None:
            from core.gap_detector import get_gap_detector
            self._gap_detector = get_gap_detector(str(self.data_dir))
        return self._gap_detector
    
    @property
    def research_orchestrator(self):
        """Lazy-load research orchestrator."""
        if self._research_orchestrator is None:
            from core.research_orchestrator import get_research_orchestrator
            self._research_orchestrator = get_research_orchestrator(str(self.data_dir))
        return self._research_orchestrator
    
    @property
    def source_validator(self):
        """Lazy-load source validator."""
        if self._source_validator is None:
            from core.source_validator import get_source_validator
            self._source_validator = get_source_validator(str(self.data_dir))
        return self._source_validator
    
    @property
    def knowledge_manager(self):
        """Lazy-load knowledge manager."""
        if self._knowledge_manager is None:
            from core.knowledge_manager import get_knowledge_manager
            self._knowledge_manager = get_knowledge_manager(str(self.data_dir))
        return self._knowledge_manager
    
    def _load_queue(self):
        """Load learning queue from disk."""
        if self.learning_queue_path.exists():
            try:
                with open(self.learning_queue_path, 'r') as f:
                    data = json.load(f)
                    self.learning_queue = [
                        LearningTask(**task) for task in data
                    ]
            except Exception as e:
                logger.warning(f"Failed to load learning queue: {e}")
    
    def _save_queue(self):
        """Save learning queue to disk."""
        try:
            with open(self.learning_queue_path, 'w') as f:
                json.dump([
                    {
                        "task_type": t.task_type,
                        "priority": t.priority,
                        "gap_topic": t.gap_topic,
                        "research_results": t.research_results,
                        "validation_result": t.validation_result,
                        "timestamp": t.timestamp
                    }
                    for t in self.learning_queue
                ], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learning queue: {e}")
    
    def trigger_gap_research(self, gap_topic: str, confidence: float = 0.5):
        """Trigger research on a detected knowledge gap.
        
        Args:
            gap_topic: Topic/query with knowledge gap.
            confidence: Current confidence level (0-1).
        """
        priority = 1.0 - confidence  # Lower confidence = higher priority
        
        task = LearningTask(
            task_type="gap_research",
            priority=priority,
            gap_topic=gap_topic
        )
        
        self.learning_queue.append(task)
        self._save_queue()
        
        logger.info(f"Queued research task for gap: {gap_topic} (priority {priority:.2f})")
    
    def process_queue(self) -> Dict:
        """Process one task from learning queue.
        
        Returns:
            Dict with processing result and status.
        """
        if not self.learning_queue:
            return {"status": "no_tasks", "message": "Learning queue is empty"}
        
        # Sort by priority (highest first)
        self.learning_queue.sort(key=lambda t: t.priority, reverse=True)
        task = self.learning_queue.pop(0)
        self._save_queue()
        
        try:
            if task.task_type == "gap_research":
                return self._process_gap_research(task)
            elif task.task_type == "validation":
                return self._process_validation(task)
            elif task.task_type == "integration":
                return self._process_integration(task)
            elif task.task_type == "review":
                return self._process_review(task)
            else:
                return {"status": "error", "message": f"Unknown task type: {task.task_type}"}
        
        except Exception as e:
            logger.error(f"Error processing task {task.task_type}: {e}")
            return {
                "status": "error",
                "task_type": task.task_type,
                "error": str(e)
            }
    
    def _process_gap_research(self, task: LearningTask) -> Dict:
        """Research a knowledge gap.
        
        Args:
            task: LearningTask with gap_topic.
            
        Returns:
            Dict with research results.
        """
        logger.info(f"Researching gap: {task.gap_topic}")
        
        try:
            # Research the gap
            research_results = self.research_orchestrator.research_topic(
                task.gap_topic,
                max_sources=5
            )
            
            if not research_results:
                return {
                    "status": "no_results",
                    "gap_topic": task.gap_topic,
                    "message": "No research results found"
                }
            
            # Create validation task
            validation_task = LearningTask(
                task_type="validation",
                priority=task.priority,
                gap_topic=task.gap_topic,
                research_results=research_results
            )
            self.learning_queue.append(validation_task)
            self._save_queue()
            
            return {
                "status": "success",
                "task_type": "gap_research",
                "gap_topic": task.gap_topic,
                "sources_found": len(research_results),
                "next_task": "validation"
            }
        
        except Exception as e:
            logger.error(f"Gap research failed for {task.gap_topic}: {e}")
            return {
                "status": "error",
                "gap_topic": task.gap_topic,
                "error": str(e)
            }
    
    def _process_validation(self, task: LearningTask) -> Dict:
        """Validate research results using source credibility.
        
        Args:
            task: LearningTask with research_results.
            
        Returns:
            Dict with validation results.
        """
        if not task.research_results:
            return {
                "status": "error",
                "message": "No research results to validate"
            }
        
        logger.info(f"Validating {len(task.research_results)} sources for {task.gap_topic}")
        
        try:
            # Validate sources
            sources_to_validate = [
                (result["url"], result["content"])
                for result in task.research_results
            ]
            
            validation_result = self.source_validator.cross_validate_sources(
                sources_to_validate,
                agreement_threshold=0.6
            )
            
            # Check if recommendation is to accept
            if validation_result["recommendation"] == "accept":
                # Create integration task
                integration_task = LearningTask(
                    task_type="integration",
                    priority=task.priority,
                    gap_topic=task.gap_topic,
                    research_results=task.research_results,
                    validation_result=validation_result
                )
                self.learning_queue.append(integration_task)
                self._save_queue()
                
                return {
                    "status": "success",
                    "task_type": "validation",
                    "consensus_score": validation_result["consensus_score"],
                    "recommendation": "accept",
                    "next_task": "integration"
                }
            
            elif validation_result["recommendation"] == "review":
                # Queue for human review
                review_task = LearningTask(
                    task_type="review",
                    priority=task.priority,
                    gap_topic=task.gap_topic,
                    research_results=task.research_results,
                    validation_result=validation_result
                )
                self.learning_queue.append(review_task)
                self._save_queue()
                
                return {
                    "status": "success",
                    "task_type": "validation",
                    "consensus_score": validation_result["consensus_score"],
                    "recommendation": "review",
                    "next_task": "human_review"
                }
            
            else:
                return {
                    "status": "validation_failed",
                    "consensus_score": validation_result["consensus_score"],
                    "recommendation": "reject",
                    "reason": validation_result.get("reason", "Low confidence")
                }
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _process_integration(self, task: LearningTask) -> Dict:
        """Integrate validated knowledge into knowledge graph.
        
        Args:
            task: LearningTask with research_results and validation_result.
            
        Returns:
            Dict with integration result.
        """
        logger.info(f"Integrating knowledge for {task.gap_topic}")
        
        try:
            # Integrate top sources into knowledge graph
            for result in task.research_results[:3]:  # Top 3 sources
                self.knowledge_manager.add_entry(
                    topic=task.gap_topic,
                    content=result["content"],
                    source_url=result["url"],
                    source_type=result["source"],
                    confidence=task.validation_result["consensus_score"]
                )
            
            # Test retrieval
            retrieved = self.knowledge_manager.search_by_topic(task.gap_topic, top_k=3)
            
            if retrieved:
                return {
                    "status": "success",
                    "task_type": "integration",
                    "gap_topic": task.gap_topic,
                    "entries_integrated": len(task.research_results[:3]),
                    "retrieval_test": "passed"
                }
            else:
                return {
                    "status": "warning",
                    "message": "Integration succeeded but retrieval test inconclusive"
                }
        
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _process_review(self, task: LearningTask) -> Dict:
        """Queue for human review.
        
        Args:
            task: LearningTask for review.
            
        Returns:
            Dict with review queue status.
        """
        review_queue_path = self.data_dir / "learning_review_queue.json"
        
        try:
            if review_queue_path.exists():
                with open(review_queue_path, 'r') as f:
                    queue = json.load(f)
            else:
                queue = []
            
            queue.append({
                "gap_topic": task.gap_topic,
                "research_results": task.research_results,
                "validation_result": task.validation_result,
                "timestamp": datetime.now().isoformat()
            })
            
            with open(review_queue_path, 'w') as f:
                json.dump(queue, f, indent=2)
            
            return {
                "status": "success",
                "task_type": "review",
                "gap_topic": task.gap_topic,
                "message": "Queued for human review"
            }
        
        except Exception as e:
            logger.error(f"Failed to queue for review: {e}")
            return {"status": "error", "error": str(e)}
    
    def _log_task(self, task: LearningTask, result: Dict):
        """Log task processing."""
        try:
            log_entry = {
                "task_type": task.task_type,
                "gap_topic": task.gap_topic,
                "priority": task.priority,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            with open(self.learning_history_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log task: {e}")


# Singleton instance
_instance: Optional[LearningLoop] = None


def get_learning_loop(data_dir: str = "data") -> LearningLoop:
    """Get or create learning loop singleton.
    
    Args:
        data_dir: Directory for storing learning state.
        
    Returns:
        LearningLoop instance.
    """
    global _instance
    if _instance is None:
        _instance = LearningLoop(data_dir=data_dir)
    return _instance
