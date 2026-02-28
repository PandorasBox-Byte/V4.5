#!/usr/bin/env python3
"""Knowledge Manager - Structured knowledge graph with sources and metadata.

Replaces flat data/knowledge.txt with structured JSON knowledge graph.
Maintains backward compatibility by generating knowledge.txt for legacy plugins.

Data Structure:
  knowledge_graph.json: {
    "entries": [
      {
        "id": str,
        "topic": str,
        "content": str,
        "sources": [{"url": str, "title": str, "confidence": float}],
        "confidence": float,
        "created": timestamp,
        "last_used": timestamp,
        "use_count": int,
        "embedding": null (computed on demand),
        "related_ids": [str]
      }
    ],
    "metadata": {
      "version": "1.0",
      "last_updated": timestamp,
      "total_entries": int
    }
  }
"""

import os
import json
import time
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib


class KnowledgeManager:
    """Manages structured knowledge graph with CRUD operations."""
    
    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = workspace_root or os.getcwd()
        self.data_dir = Path(self.workspace_root) / "data"
        self.graph_file = self.data_dir / "knowledge_graph.json"
        self.legacy_file = self.data_dir / "knowledge.txt"
        self.versions_dir = self.data_dir / "knowledge_versions"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)
        
        # Load or initialize knowledge graph
        self.graph = self._load_graph()
    
    def _load_graph(self) -> Dict[str, Any]:
        """Load knowledge graph from JSON file."""
        if not self.graph_file.exists():
            return {
                "entries": [],
                "metadata": {
                    "version": "1.0",
                    "last_updated": time.time(),
                    "total_entries": 0
                }
            }
        
        try:
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                graph = json.load(f)
                # Validate structure
                if "entries" not in graph:
                    graph["entries"] = []
                if "metadata" not in graph:
                    graph["metadata"] = {"version": "1.0", "last_updated": time.time(), "total_entries": 0}
                return graph
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading knowledge graph: {e}. Starting fresh.")
            return {
                "entries": [],
                "metadata": {"version": "1.0", "last_updated": time.time(), "total_entries": 0}
            }
    
    def _save_graph(self, backup: bool = True):
        """Save knowledge graph to JSON file with atomic write."""
        # Backup current version if requested
        if backup and self.graph_file.exists():
            timestamp = int(time.time())
            backup_file = self.versions_dir / f"knowledge_graph_{timestamp}.json"
            shutil.copy2(self.graph_file, backup_file)
            
            # Keep only last 10 backups
            backups = sorted(self.versions_dir.glob("knowledge_graph_*.json"))
            for old_backup in backups[:-10]:
                old_backup.unlink()
        
        # Update metadata
        self.graph["metadata"]["last_updated"] = time.time()
        self.graph["metadata"]["total_entries"] = len(self.graph["entries"])
        
        # Atomic write using temp file
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                        dir=self.data_dir, delete=False, 
                                        suffix='.tmp') as tmp:
            json.dump(self.graph, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        
        # Replace old file
        shutil.move(tmp_path, self.graph_file)
        
        # Update legacy knowledge.txt for backward compatibility
        self._update_legacy_file()
    
    def _update_legacy_file(self):
        """Generate knowledge.txt from graph for legacy plugin compatibility."""
        try:
            with open(self.legacy_file, 'w', encoding='utf-8') as f:
                for entry in self.graph["entries"]:
                    # Format: topic: content (confidence: XX%)
                    conf_pct = int(entry.get("confidence", 0.5) * 100)
                    f.write(f"{entry['topic']}: {entry['content']} (confidence: {conf_pct}%)\n")
        except IOError as e:
            print(f"Warning: Could not update legacy knowledge.txt: {e}")
    
    def _generate_id(self, topic: str) -> str:
        """Generate unique ID from topic."""
        return hashlib.md5(topic.lower().encode()).hexdigest()[:12]
    
    def add_entry(self, topic: str, content: str, sources: List[Dict[str, Any]], 
                  confidence: float = 0.7, related_ids: Optional[List[str]] = None) -> str:
        """Add new knowledge entry.
        
        Args:
            topic: Topic/subject of the knowledge
            content: Knowledge content/summary
            sources: List of source dicts with url, title, confidence
            confidence: Overall confidence score (0-1)
            related_ids: List of related entry IDs
            
        Returns:
            Entry ID
        """
        entry_id = self._generate_id(topic)
        
        # Check if entry already exists
        existing = self.get_entry(entry_id)
        if existing:
            # Merge with existing entry
            return self.update_entry(entry_id, content=content, sources=sources, 
                                   confidence=confidence, related_ids=related_ids)
        
        entry = {
            "id": entry_id,
            "topic": topic,
            "content": content,
            "sources": sources,
            "confidence": confidence,
            "created": time.time(),
            "last_used": time.time(),
            "use_count": 0,
            "embedding": None,  # Computed on demand by engine
            "related_ids": related_ids or []
        }
        
        self.graph["entries"].append(entry)
        self._save_graph(backup=True)
        
        return entry_id
    
    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get entry by ID."""
        for entry in self.graph["entries"]:
            if entry["id"] == entry_id:
                return entry
        return None
    
    def search_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Search entries by topic keyword match."""
        topic_lower = topic.lower()
        matches = []
        for entry in self.graph["entries"]:
            if topic_lower in entry["topic"].lower() or topic_lower in entry["content"].lower():
                matches.append(entry)
        return matches
    
    def update_entry(self, entry_id: str, content: Optional[str] = None, 
                    sources: Optional[List[Dict[str, Any]]] = None,
                    confidence: Optional[float] = None,
                    related_ids: Optional[List[str]] = None) -> Optional[str]:
        """Update existing entry.
        
        Args:
            entry_id: Entry ID to update
            content: New content (merges if both exist)
            sources: New sources (appends to existing)
            confidence: New confidence (takes max if merging)
            related_ids: New related IDs (appends to existing)
            
        Returns:
            Entry ID if successful, None otherwise
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return None
        
        if content:
            # Merge content if both exist and differ
            if entry["content"] != content:
                entry["content"] = f"{entry['content']} {content}".strip()
        
        if sources:
            # Append new sources, avoiding duplicates by URL
            existing_urls = {src["url"] for src in entry["sources"]}
            for source in sources:
                if source["url"] not in existing_urls:
                    entry["sources"].append(source)
        
        if confidence is not None:
            # Take maximum confidence
            entry["confidence"] = max(entry["confidence"], confidence)
        
        if related_ids:
            # Append new related IDs
            for rid in related_ids:
                if rid not in entry["related_ids"]:
                    entry["related_ids"].append(rid)
        
        # Invalidate embedding (needs recomputation)
        entry["embedding"] = None
        
        self._save_graph(backup=True)
        return entry_id
    
    def mark_used(self, entry_id: str):
        """Mark entry as recently used (updates last_used and use_count)."""
        entry = self.get_entry(entry_id)
        if entry:
            entry["last_used"] = time.time()
            entry["use_count"] = entry.get("use_count", 0) + 1
            self._save_graph(backup=False)  # Don't backup for usage updates
    
    def prune_old_entries(self, days_unused: int = 90, min_confidence: float = 0.6):
        """Remove entries unused for X days with low confidence."""
        cutoff_time = time.time() - (days_unused * 86400)
        initial_count = len(self.graph["entries"])
        
        self.graph["entries"] = [
            entry for entry in self.graph["entries"]
            if not (entry.get("last_used", 0) < cutoff_time and 
                   entry.get("confidence", 1.0) < min_confidence)
        ]
        
        pruned = initial_count - len(self.graph["entries"])
        if pruned > 0:
            print(f"Pruned {pruned} low-quality unused knowledge entries")
            self._save_graph(backup=True)
        
        return pruned
    
    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all knowledge entries."""
        return self.graph["entries"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self.graph["entries"]:
            return {
                "total_entries": 0,
                "avg_confidence": 0.0,
                "total_sources": 0,
                "last_updated": self.graph["metadata"].get("last_updated", 0)
            }
        
        total_confidence = sum(e.get("confidence", 0.5) for e in self.graph["entries"])
        total_sources = sum(len(e.get("sources", [])) for e in self.graph["entries"])
        
        return {
            "total_entries": len(self.graph["entries"]),
            "avg_confidence": total_confidence / len(self.graph["entries"]),
            "total_sources": total_sources,
            "last_updated": self.graph["metadata"].get("last_updated", 0)
        }
    
    def export_to_legacy_format(self, output_path: Optional[str] = None) -> str:
        """Export knowledge graph to legacy flat text format."""
        output_path = output_path or str(self.legacy_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.graph["entries"]:
                f.write(f"{entry['topic']}: {entry['content']}\n")
        
        return output_path


# Singleton instance
_knowledge_manager = None


def get_knowledge_manager(workspace_root: Optional[str] = None) -> KnowledgeManager:
    """Get or create singleton KnowledgeManager instance."""
    global _knowledge_manager
    if _knowledge_manager is None:
        _knowledge_manager = KnowledgeManager(workspace_root)
    return _knowledge_manager


if __name__ == "__main__":
    # Test knowledge manager
    km = get_knowledge_manager()
    
    print("Knowledge Manager Test")
    print(f"Stats: {km.get_stats()}")
    
    # Add test entry
    entry_id = km.add_entry(
        topic="Python asyncio",
        content="Asyncio is a library to write concurrent code using async/await syntax",
        sources=[{
            "url": "https://docs.python.org/3/library/asyncio.html",
            "title": "Python asyncio documentation",
            "confidence": 0.95
        }],
        confidence=0.9
    )
    
    print(f"\nAdded entry: {entry_id}")
    print(f"New stats: {km.get_stats()}")
    
    # Search
    results = km.search_by_topic("asyncio")
    print(f"\nFound {len(results)} entries matching 'asyncio'")
