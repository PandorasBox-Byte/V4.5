"""Curriculum planning for autonomous knowledge acquisition.

This module implements intelligent learning path planning:
- Auto-discovers user knowledge domains from interaction patterns
- Tracks learning progress per domain (coverage %)
- Prioritizes learning based on missing knowledge (gaps) and domain importance
- Implements spaced repetition scheduling for review
- Maintains learning objectives and completion status
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class LearningObjective:
    """A learning objective for a domain."""
    domain: str
    objective: str
    target_confidence: float = 0.8  # target confidence level (0-1)
    priority: float = 0.5  # 0-1, higher = more important
    status: str = "pending"  # pending, in_progress, completed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


@dataclass
class DomainProgress:
    """Learning progress for a domain."""
    domain: str
    total_interactions: int = 0
    confident_responses: int = 0  # responses with confidence >= target
    coverage: float = 0.0  # % of domain covered (0-1)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    next_review_due: str = field(default_factory=lambda: (datetime.now() + timedelta(days=7)).isoformat())


class CurriculumPlanner:
    """Plans learning curriculum autonomously.
    
    Discovers domains from user interactions, tracks learning progress,
    and prioritizes what to learn next based on gaps and importance.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize curriculum planner.
        
        Args:
            data_dir: Directory for storing learning progress and objectives.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.progress_path = self.data_dir / "learning_progress.json"
        self.objectives_path = self.data_dir / "learning_objectives.json"
        self.domain_history_path = self.data_dir / "domain_history.jsonl"
        
        self.domains: Dict[str, DomainProgress] = {}
        self.objectives: Dict[str, List[LearningObjective]] = defaultdict(list)
        self.domain_keywords: Dict[str, Set[str]] = defaultdict(set)
        
        self._load_state()
        self._initialize_default_domains()
    
    def _load_state(self):
        """Load learning progress and objectives."""
        if self.progress_path.exists():
            try:
                with open(self.progress_path, 'r') as f:
                    data = json.load(f)
                    for domain, progress_dict in data.items():
                        self.domains[domain] = DomainProgress(**progress_dict)
            except Exception as e:
                logger.warning(f"Failed to load learning progress: {e}")
        
        if self.objectives_path.exists():
            try:
                with open(self.objectives_path, 'r') as f:
                    data = json.load(f)
                    for domain, objectives_list in data.items():
                        self.objectives[domain] = [
                            LearningObjective(**obj) for obj in objectives_list
                        ]
            except Exception as e:
                logger.warning(f"Failed to load learning objectives: {e}")
    
    def _save_state(self):
        """Save learning progress and objectives."""
        try:
            with open(self.progress_path, 'w') as f:
                json.dump({
                    domain: {
                        "domain": p.domain,
                        "total_interactions": p.total_interactions,
                        "confident_responses": p.confident_responses,
                        "coverage": p.coverage,
                        "last_updated": p.last_updated,
                        "next_review_due": p.next_review_due
                    }
                    for domain, p in self.domains.items()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learning progress: {e}")
        
        try:
            with open(self.objectives_path, 'w') as f:
                json.dump({
                    domain: [
                        {
                            "domain": obj.domain,
                            "objective": obj.objective,
                            "target_confidence": obj.target_confidence,
                            "priority": obj.priority,
                            "status": obj.status,
                            "created_at": obj.created_at,
                            "completed_at": obj.completed_at
                        }
                        for obj in objectives
                    ]
                    for domain, objectives in self.objectives.items()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learning objectives: {e}")
    
    def _initialize_default_domains(self):
        """Initialize common knowledge domains."""
        default_domains = [
            "Python programming",
            "Machine Learning",
            "Natural Language Processing",
            "Data Science",
            "Software Engineering",
            "System Architecture",
            "General Knowledge"
        ]
        
        for domain in default_domains:
            if domain not in self.domains:
                self.domains[domain] = DomainProgress(domain=domain)
    
    def record_interaction(self, query: str, domain: Optional[str] = None, 
                          confidence: float = 0.5):
        """Record a user interaction to update domain progress.
        
        Args:
            query: User query/question.
            domain: Associated domain (auto-detected if not provided).
            confidence: Response confidence (0-1).
        """
        # Auto-detect domain if needed
        if domain is None:
            domain = self._detect_domain(query)
        
        # Initialize domain if new
        if domain not in self.domains:
            self.domains[domain] = DomainProgress(domain=domain)
        
        # Update progress
        progress = self.domains[domain]
        progress.total_interactions += 1
        if confidence >= 0.7:  # high confidence threshold
            progress.confident_responses += 1
        
        # Update coverage estimate
        if progress.total_interactions > 0:
            progress.coverage = min(
                progress.confident_responses / progress.total_interactions,
                1.0
            )
        
        progress.last_updated = datetime.now().isoformat()
        
        # Log interaction
        self._log_interaction(query, domain, confidence)
        
        self._save_state()
    
    def _detect_domain(self, query: str) -> str:
        """Auto-detect domain from query keywords.
        
        Args:
            query: User query.
            
        Returns:
            Detected domain name (defaults to "General Knowledge").
        """
        query_lower = query.lower()
        
        # Domain keyword mappings
        domain_keywords_map = {
            "Python programming": {"python", "pip", "django", "flask", "pandas", "numpy"},
            "Machine Learning": {"machine learning", "ml", "model", "training", "neural", "deep learning"},
            "Natural Language Processing": {"nlp", "text", "language", "nlp", "tokenize", "embedding"},
            "Data Science": {"data", "analysis", "statistics", "visualization", "csv"},
            "Software Engineering": {"software", "code", "git", "version", "design pattern"},
            "System Architecture": {"system", "architecture", "database", "distributed", "scalability"}
        }
        
        # Score each domain
        best_domain = "General Knowledge"
        best_score = 0.0
        
        for domain, keywords in domain_keywords_map.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_domain = domain
        
        return best_domain
    
    def get_learning_priorities(self) -> List[Dict]:
        """Get prioritized list of domains to focus learning on.
        
        Returns:
            List of dicts with domain, priority_score, gaps, and recommendations.
            Higher priority = should learn more about this domain.
        """
        priorities = []
        
        for domain, progress in self.domains.items():
            # Priority based on: coverage gap * domain importance
            coverage_gap = 1.0 - progress.coverage
            
            # Estimate domain importance from knowledge gaps (from gap_detector)
            # For now, use default importance and coverage gap
            domain_importance = 0.5
            
            # Time decay: if not reviewed recently, increase priority
            last_update = datetime.fromisoformat(progress.last_updated)
            days_since_update = (datetime.now() - last_update).days
            time_factor = min(1.0 + (days_since_update / 30), 2.0)  # max 2x boost
            
            priority_score = coverage_gap * domain_importance * time_factor
            
            gaps = []
            if progress.coverage < 0.5:
                gaps.append("Major knowledge gaps")
            elif progress.coverage < 0.8:
                gaps.append("Moderate knowledge gaps")
            
            if days_since_update > 30:
                gaps.append("Needs review (spaced repetition)")
            
            priorities.append({
                "domain": domain,
                "priority_score": priority_score,
                "coverage": progress.coverage,
                "total_interactions": progress.total_interactions,
                "confident_responses": progress.confident_responses,
                "gaps": gaps,
                "recommendation": self._get_recommendation(progress, coverage_gap)
            })
        
        # Sort by priority score (descending)
        priorities.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return priorities
    
    def _get_recommendation(self, progress: DomainProgress, coverage_gap: float) -> str:
        """Get learning recommendation for domain.
        
        Args:
            progress: Domain progress.
            coverage_gap: 1 - coverage (0-1).
            
        Returns:
            Recommendation text.
        """
        if coverage_gap > 0.5:
            return f"Urgent: {progress.domain} coverage only {progress.coverage:.0%}. Research key topics."
        elif coverage_gap > 0.2:
            return f"Focus on {progress.domain} to fill {coverage_gap:.0%} gap in knowledge."
        else:
            return f"Review {progress.domain} periodically to maintain knowledge."
    
    def schedule_learning(self, domain: str) -> Dict:
        """Schedule learning objectives for a domain.
        
        Args:
            domain: Domain to schedule learning for.
            
        Returns:
            Dict with scheduled objectives and timeline.
        """
        if domain not in self.domains:
            return {"error": f"Unknown domain: {domain}"}
        
        progress = self.domains[domain]
        
        # Generate default objectives based on domain
        objectives = self._generate_objectives(domain)
        
        # Schedule them
        for i, objective in enumerate(objectives):
            objective.priority = 1.0 - (i * 0.2)  # Decreasing priority
            self.objectives[domain].append(objective)
        
        # Set next review date (spaced repetition: 7 days)
        progress.next_review_due = (datetime.now() + timedelta(days=7)).isoformat()
        self._save_state()
        
        return {
            "domain": domain,
            "objectives": [
                {
                    "objective": obj.objective,
                    "priority": obj.priority,
                    "status": obj.status,
                    "target_confidence": obj.target_confidence
                }
                for obj in objectives
            ],
            "next_review": progress.next_review_due
        }
    
    def _generate_objectives(self, domain: str) -> List[LearningObjective]:
        """Generate default learning objectives for domain.
        
        Args:
            domain: Domain name.
            
        Returns:
            List of LearningObjective objects.
        """
        domain_objectives = {
            "Python programming": [
                "Master core syntax and data structures",
                "Understand OOP principles",
                "Practice functional programming",
                "Learn async/await patterns"
            ],
            "Machine Learning": [
                "Understand supervised learning fundamentals",
                "Master unsupervised learning techniques",
                "Learn model evaluation and validation",
                "Study neural network architectures"
            ],
            "Natural Language Processing": [
                "Learn tokenization and preprocessing",
                "Understand embeddings and word vectors",
                "Study transformer architectures",
                "Practice classification and generation tasks"
            ],
            "Data Science": [
                "Master data cleaning and preparation",
                "Learn exploratory data analysis",
                "Study statistical analysis techniques",
                "Practice visualization best practices"
            ]
        }
        
        objectives = domain_objectives.get(domain, [
            f"Learn fundamentals of {domain}",
            f"Master practical {domain} skills",
            f"Stay current with {domain} advances"
        ])
        
        return [LearningObjective(domain=domain, objective=obj) for obj in objectives]
    
    def mark_objective_complete(self, domain: str, objective: str):
        """Mark a learning objective as completed.
        
        Args:
            domain: Domain name.
            objective: Objective text.
        """
        if domain not in self.objectives:
            return
        
        for obj in self.objectives[domain]:
            if obj.objective == objective:
                obj.status = "completed"
                obj.completed_at = datetime.now().isoformat()
                break
        
        self._save_state()
    
    def get_due_for_review(self) -> List[str]:
        """Get domains due for spaced repetition review.
        
        Returns:
            List of domain names due for review.
        """
        due = []
        now = datetime.now()
        
        for domain, progress in self.domains.items():
            review_date = datetime.fromisoformat(progress.next_review_due)
            if now >= review_date:
                due.append(domain)
        
        return due
    
    def _log_interaction(self, query: str, domain: str, confidence: float):
        """Log learning interaction."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "domain": domain,
                "confidence": confidence
            }
            with open(self.domain_history_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")


# Singleton instance
_instance: Optional[CurriculumPlanner] = None


def get_curriculum_planner(data_dir: str = "data") -> CurriculumPlanner:
    """Get or create curriculum planner singleton.
    
    Args:
        data_dir: Directory for storing learning progress.
        
    Returns:
        CurriculumPlanner instance.
    """
    global _instance
    if _instance is None:
        _instance = CurriculumPlanner(data_dir=data_dir)
    return _instance
