"""Source validation and credibility assessment for autonomous learning.

This module provides domain-aware source validation, content quality checks,
and cross-validation logic for multi-source research. Sources are rated based
on domain trust scores, content quality heuristics, and cross-source agreement.

Trust matrix:
- Wikipedia (academic reference): 0.95
- arXiv (peer-reviewed preprints): 0.95
- .edu domains (educational institutions): 0.85
- .gov domains (government): 0.90
- Stack Overflow (community Q&A): 0.75
- General web (news, blogs): 0.50
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)


# Domain trust scores (confidence factors for source credibility)
DOMAIN_TRUST_MATRIX = {
    "wikipedia": 0.95,
    "arxiv": 0.95,
    "edu": 0.85,
    "gov": 0.90,
    "stackoverflow": 0.75,
    "github": 0.65,
    "general": 0.50,
}

# Content quality thresholds
MIN_CONTENT_LENGTH = 100  # minimum characters
MAX_AD_DENSITY = 0.15     # max fraction before marking as low quality
MIN_COHERENCE_SCORE = 0.5 # normalized 0-1


@dataclass
class SourceValidationResult:
    """Result of source validation."""
    source_url: str
    is_valid: bool
    trust_score: float  # 0-1
    quality_score: float  # 0-1
    combined_score: float  # 0-1 (weighted average)
    reason: str
    issues: List[str]


@dataclass
class ContentQualityResult:
    """Result of content quality assessment."""
    length: int
    ad_density: float  # 0-1
    coherence_score: float  # 0-1
    quality_issues: List[str]
    is_acceptable: bool


class SourceValidator:
    """Validates sources and content credibility for autonomous learning.
    
    Provides:
    - Domain-based trust scoring
    - Content quality heuristics (length, ad density, coherence)
    - Cross-validation (require multiple sources agreeing)
    - Source credibility tracking and caching
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize source validator.
        
        Args:
            data_dir: Directory for storing validation cache and logs.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.validation_cache_path = self.data_dir / "source_validation_cache.json"
        self.source_history_path = self.data_dir / "source_history.jsonl"
        
        self.validation_cache: Dict[str, Dict] = {}
        self._load_validation_cache()
    
    def _load_validation_cache(self):
        """Load cached validation results."""
        if self.validation_cache_path.exists():
            try:
                with open(self.validation_cache_path, 'r') as f:
                    self.validation_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load validation cache: {e}")
    
    def _save_validation_cache(self):
        """Save validation cache."""
        try:
            with open(self.validation_cache_path, 'w') as f:
                json.dump(self.validation_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save validation cache: {e}")
    
    def validate_source(self, url: str, content: Optional[str] = None) -> SourceValidationResult:
        """Validate a single source URL and optionally its content.
        
        Args:
            url: Source URL to validate.
            content: Optional content to perform quality checks on.
            
        Returns:
            SourceValidationResult with trust, quality, and combined scores.
        """
        # Check cache
        if url in self.validation_cache:
            cached = self.validation_cache[url]
            # Only return cache if recent (< 7 days)
            if (datetime.now().timestamp() - cached.get("timestamp", 0)) < 604800:
                return SourceValidationResult(**cached["result"])
        
        # Calculate trust score from domain
        trust_score = self._get_domain_trust_score(url)
        
        # Calculate quality score if content provided
        quality_score = 0.5  # default neutral
        quality_result = None
        if content:
            quality_result = self._assess_content_quality(content)
            quality_score = (
                0.4 * (1 - quality_result.ad_density) +  # lower ad density = higher score
                0.3 * quality_result.coherence_score +
                0.3 * min(quality_result.length / 5000, 1.0)  # longer content up to 5000 chars
            )
            quality_score = max(0.0, min(1.0, quality_score))
        
        # Combined score: weighted toward trust with quality adjustment
        combined_score = 0.7 * trust_score + 0.3 * quality_score
        
        # Determine if valid
        is_valid = combined_score >= 0.4  # threshold for acceptance
        issues = quality_result.quality_issues if quality_result else []
        
        reason = "Valid source" if is_valid else "Source credibility below threshold"
        
        result = SourceValidationResult(
            source_url=url,
            is_valid=is_valid,
            trust_score=trust_score,
            quality_score=quality_score,
            combined_score=combined_score,
            reason=reason,
            issues=issues
        )
        
        # Cache result
        self.validation_cache[url] = {
            "timestamp": datetime.now().timestamp(),
            "result": {
                "source_url": result.source_url,
                "is_valid": result.is_valid,
                "trust_score": result.trust_score,
                "quality_score": result.quality_score,
                "combined_score": result.combined_score,
                "reason": result.reason,
                "issues": result.issues
            }
        }
        self._save_validation_cache()
        
        # Log validation
        self._log_validation(result)
        
        return result
    
    def cross_validate_sources(self, 
                              sources: List[Tuple[str, str]],
                              agreement_threshold: float = 0.6) -> Dict:
        """Cross-validate multiple sources on the same topic.
        
        Requires sources to agree on key points (via user feedback or
        semantic similarity in future versions).
        
        Args:
            sources: List of (url, content) tuples.
            agreement_threshold: Require this fraction of sources to validate (0-1).
            
        Returns:
            Dict with:
            - validation_results: List of SourceValidationResult
            - consensus_score: 0-1, higher if sources agree
            - recommendation: "accept", "review", or "reject"
        """
        if not sources:
            return {
                "validation_results": [],
                "consensus_score": 0.0,
                "recommendation": "reject",
                "reason": "No sources provided"
            }
        
        results = [
            self.validate_source(url, content)
            for url, content in sources
        ]
        
        valid_count = sum(1 for r in results if r.is_valid)
        agreement_score = valid_count / len(results)
        
        # Average combined score
        avg_score = sum(r.combined_score for r in results) / len(results)
        
        # Consensus: higher if agreement threshold met and sources are credible
        consensus_score = avg_score * (
            1.0 if agreement_score >= agreement_threshold else 0.5
        )
        
        if agreement_score < agreement_threshold or avg_score < 0.5:
            recommendation = "reject"
        elif consensus_score < 0.7:
            recommendation = "review"
        else:
            recommendation = "accept"
        
        return {
            "validation_results": [
                {
                    "source_url": r.source_url,
                    "is_valid": r.is_valid,
                    "trust_score": r.trust_score,
                    "quality_score": r.quality_score,
                    "combined_score": r.combined_score,
                    "reason": r.reason,
                    "issues": r.issues
                }
                for r in results
            ],
            "consensus_score": consensus_score,
            "agreement_score": agreement_score,
            "avg_combined_score": avg_score,
            "recommendation": recommendation,
            "reason": f"{valid_count}/{len(results)} sources valid, "
                     f"agreement={agreement_score:.2f}, avg_score={avg_score:.2f}"
        }
    
    def _get_domain_trust_score(self, url: str) -> float:
        """Calculate trust score based on source domain.
        
        Args:
            url: Source URL.
            
        Returns:
            Trust score 0-1.
        """
        url_lower = url.lower()
        
        # Check specific domains
        if "wikipedia.org" in url_lower:
            return DOMAIN_TRUST_MATRIX["wikipedia"]
        elif "arxiv.org" in url_lower:
            return DOMAIN_TRUST_MATRIX["arxiv"]
        elif "stackoverflow.com" in url_lower:
            return DOMAIN_TRUST_MATRIX["stackoverflow"]
        elif "github.com" in url_lower:
            return DOMAIN_TRUST_MATRIX["github"]
        
        # Check domain extensions
        if ".edu" in url_lower:
            return DOMAIN_TRUST_MATRIX["edu"]
        elif ".gov" in url_lower:
            return DOMAIN_TRUST_MATRIX["gov"]
        
        # Default to general web trust
        return DOMAIN_TRUST_MATRIX["general"]
    
    def _assess_content_quality(self, content: str) -> ContentQualityResult:
        """Assess content quality heuristically.
        
        Args:
            content: Content to assess.
            
        Returns:
            ContentQualityResult with length, ad density, and coherence metrics.
        """
        issues = []
        
        # Length check
        length = len(content)
        if length < MIN_CONTENT_LENGTH:
            issues.append(f"Content too short ({length} chars, min {MIN_CONTENT_LENGTH})")
        
        # Ad density check (estimate ads as recurrent patterns like "buy", "click", "$")
        ad_keywords = ["buy now", "click here", "limited time", "exclusive offer", "$", "discount"]
        ad_count = sum(content.lower().count(keyword) for keyword in ad_keywords)
        ad_density = ad_count / max(len(content.split()), 1)
        
        if ad_density > MAX_AD_DENSITY:
            issues.append(f"High ad density ({ad_density:.2%})")
        
        # Coherence check (simplified: check for paragraph structure and transitions)
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
        
        if len(valid_sentences) > 0:
            # Higher coherence if content has multiple well-formed sentences
            coherence_score = min(len(valid_sentences) / 10, 1.0)
        else:
            coherence_score = 0.0
            issues.append("Low coherence (few valid sentences)")
        
        # Determine if acceptable
        is_acceptable = (
            length >= MIN_CONTENT_LENGTH and
            ad_density <= MAX_AD_DENSITY and
            coherence_score >= MIN_COHERENCE_SCORE
        )
        
        return ContentQualityResult(
            length=length,
            ad_density=min(ad_density, 1.0),
            coherence_score=coherence_score,
            quality_issues=issues,
            is_acceptable=is_acceptable
        )
    
    def _log_validation(self, result: SourceValidationResult):
        """Log source validation event."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "source_url": result.source_url,
                "is_valid": result.is_valid,
                "trust_score": result.trust_score,
                "quality_score": result.quality_score,
                "combined_score": result.combined_score
            }
            with open(self.source_history_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log validation: {e}")


# Singleton instance
_instance: Optional[SourceValidator] = None


def get_source_validator(data_dir: str = "data") -> SourceValidator:
    """Get or create source validator singleton.
    
    Args:
        data_dir: Directory for storing validation cache and logs.
        
    Returns:
        SourceValidator instance.
    """
    global _instance
    if _instance is None:
        _instance = SourceValidator(data_dir=data_dir)
    return _instance
