"""API integrations for autonomous research and knowledge gathering.

This module provides public API wrappers for:
- Wikipedia API: article summaries, search
- arXiv API: preprint search, abstracts (academic research)
- Stack Overflow API: Q&A search, accepted answers (technical problems)

All APIs are public and require no authentication. Rate limiting is implemented
to respect API terms of service (1 request/sec per source).
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

try:
    import requests
except ImportError:
    requests = None


logger = logging.getLogger(__name__)


# Rate limiting (requests per second per source)
RATE_LIMIT = 1.0

# API endpoints
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"
STACKOVERFLOW_API = "https://api.stackexchange.com/2.3/search"


@dataclass
class ResearchResult:
    """Result from a single research query."""
    source: str  # "wikipedia", "arxiv", "stackoverflow"
    title: str
    url: str
    content: str
    confidence: float  # 0-1, relevance to query
    retrieved_at: str  # ISO datetime


class APIIntegrations:
    """Manages API integrations for autonomous knowledge gathering.
    
    Provides unified search interface across Wikipedia, arXiv, and Stack Overflow
    with rate limiting, caching, and error handling.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize API integrations.
        
        Args:
            data_dir: Directory for storing API cache and logs.
        """
        if not requests:
            logger.warning("requests library not available - API calls will fail")
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.cache_path = self.data_dir / "api_cache.json"
        self.request_log_path = self.data_dir / "api_requests.jsonl"
        
        self.cache: Dict = {}
        self.last_request_time = {
            "wikipedia": 0,
            "arxiv": 0,
            "stackoverflow": 0
        }
        
        self._load_cache()
    
    def _load_cache(self):
        """Load API response cache."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load API cache: {e}")
    
    def _save_cache(self):
        """Save API response cache."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save API cache: {e}")
    
    def _rate_limit(self, source: str):
        """Enforce rate limiting for API source.
        
        Args:
            source: API source name ("wikipedia", "arxiv", "stackoverflow").
        """
        if source not in self.last_request_time:
            self.last_request_time[source] = 0
        
        elapsed = time.time() - self.last_request_time[source]
        wait_time = (1.0 / RATE_LIMIT) - elapsed
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        self.last_request_time[source] = time.time()
    
    def _log_request(self, source: str, query: str, status: str, result_count: int = 0):
        """Log API request."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "query": query,
                "status": status,
                "result_count": result_count
            }
            with open(self.request_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log API request: {e}")
    
    def search_wikipedia(self, query: str, limit: int = 3) -> List[ResearchResult]:
        """Search Wikipedia for articles related to query.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List of ResearchResult objects from Wikipedia.
        """
        if not requests:
            logger.error("requests library not available")
            return []
        
        # Check cache
        cache_key = f"wikipedia:{query}:{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit("wikipedia")
        
        try:
            # Search Wikipedia
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json"
            }
            
            response = requests.get(WIKIPEDIA_API, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("query", {}).get("search", [])[:limit]:
                title = item.get("title", "")
                pageid = item.get("pageid")
                
                # Get full page content
                content_params = {
                    "action": "query",
                    "pageids": pageid,
                    "prop": "extracts",
                    "explaintext": True,
                    "format": "json"
                }
                
                self._rate_limit("wikipedia")
                content_response = requests.get(WIKIPEDIA_API, params=content_params, timeout=10)
                content_response.raise_for_status()
                content_data = content_response.json()
                
                page = list(content_data.get("query", {}).get("pages", {}).values())[0]
                content = page.get("extract", "")[:2000]  # Limit to 2000 chars
                
                url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                
                results.append(ResearchResult(
                    source="wikipedia",
                    title=title,
                    url=url,
                    content=content,
                    confidence=0.95,  # Wikipedia articles are high quality
                    retrieved_at=datetime.now().isoformat()
                ))
            
            # Cache results
            self.cache[cache_key] = results
            self._save_cache()
            self._log_request("wikipedia", query, "success", len(results))
            
            return results
            
        except Exception as e:
            logger.error(f"Wikipedia search failed for '{query}': {e}")
            self._log_request("wikipedia", query, f"error: {str(e)[:100]}")
            return []
    
    def search_arxiv(self, query: str, limit: int = 3) -> List[ResearchResult]:
        """Search arXiv for academic papers related to query.
        
        Args:
            query: Search query (e.g., "machine learning optimization").
            limit: Maximum number of results.
            
        Returns:
            List of ResearchResult objects from arXiv.
        """
        if not requests:
            logger.error("requests library not available")
            return []
        
        # Check cache
        cache_key = f"arxiv:{query}:{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit("arxiv")
        
        try:
            # arXiv API uses OpenSearch format
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(ARXIV_API, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse Atom XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            results = []
            
            for entry in root.findall("atom:entry", ns)[:limit]:
                title = entry.find("atom:title", ns)
                title_text = title.text if title is not None else ""
                
                summary = entry.find("atom:summary", ns)
                summary_text = (summary.text if summary is not None else "")[:1000]
                
                arxiv_id = entry.find("atom:id", ns)
                arxiv_id_text = arxiv_id.text if arxiv_id is not None else ""
                url = arxiv_id_text.replace("http://arxiv.org/abs/", "https://arxiv.org/abs/")
                
                results.append(ResearchResult(
                    source="arxiv",
                    title=title_text.strip(),
                    url=url,
                    content=summary_text.strip(),
                    confidence=0.95,  # Peer-reviewed content
                    retrieved_at=datetime.now().isoformat()
                ))
            
            # Cache results
            self.cache[cache_key] = results
            self._save_cache()
            self._log_request("arxiv", query, "success", len(results))
            
            return results
            
        except Exception as e:
            logger.error(f"arXiv search failed for '{query}': {e}")
            self._log_request("arxiv", query, f"error: {str(e)[:100]}")
            return []
    
    def search_stackoverflow(self, query: str, limit: int = 3) -> List[ResearchResult]:
        """Search Stack Overflow for Q&A related to query.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List of ResearchResult objects from Stack Overflow.
        """
        if not requests:
            logger.error("requests library not available")
            return []
        
        # Check cache
        cache_key = f"stackoverflow:{query}:{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        self._rate_limit("stackoverflow")
        
        try:
            params = {
                "intitle": query,
                "sort": "relevance",
                "order": "desc",
                "site": "stackoverflow.com",
                "pagesize": limit
            }
            
            response = requests.get(STACKOVERFLOW_API, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", [])[:limit]:
                title = item.get("title", "")
                question_id = item.get("question_id")
                url = item.get("link", "")
                
                # Build content from title and score info
                is_answered = item.get("is_answered", False)
                answer_count = item.get("answer_count", 0)
                score = item.get("score", 0)
                
                content = f"Title: {title}\n"
                content += f"Answered: {is_answered}, Answers: {answer_count}, Score: {score}\n"
                if "excerpt" in item:
                    content += item["excerpt"]
                
                results.append(ResearchResult(
                    source="stackoverflow",
                    title=title,
                    url=url,
                    content=content[:1000],
                    confidence=0.75 if is_answered else 0.50,
                    retrieved_at=datetime.now().isoformat()
                ))
            
            # Cache results
            self.cache[cache_key] = results
            self._save_cache()
            self._log_request("stackoverflow", query, "success", len(results))
            
            return results
            
        except Exception as e:
            logger.error(f"Stack Overflow search failed for '{query}': {e}")
            self._log_request("stackoverflow", query, f"error: {str(e)[:100]}")
            return []
    
    def search_all_sources(self, query: str, limit: int = 3) -> Dict[str, List[ResearchResult]]:
        """Search all available sources for query.
        
        Args:
            query: Search query.
            limit: Maximum results per source.
            
        Returns:
            Dict mapping source name to list of ResearchResult objects.
        """
        return {
            "wikipedia": self.search_wikipedia(query, limit),
            "arxiv": self.search_arxiv(query, limit),
            "stackoverflow": self.search_stackoverflow(query, limit)
        }


# Singleton instance
_instance: Optional[APIIntegrations] = None


def get_api_integrations(data_dir: str = "data") -> APIIntegrations:
    """Get or create API integrations singleton.
    
    Args:
        data_dir: Directory for storing cache and logs.
        
    Returns:
        APIIntegrations instance.
    """
    global _instance
    if _instance is None:
        _instance = APIIntegrations(data_dir=data_dir)
    return _instance
