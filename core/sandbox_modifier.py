"""Safe sandbox for autonomous data and plugin modifications.

This module provides controlled CRUD operations for autonomous systems:
- Restricted to data/ and plugins/ directories (core/ is blocked)
- Validates modifications before applying (JSON, Python syntax checks)
- Audit logging of all changes
- Rollback mechanism: keeps last 10 file versions
- Approval gating: high-confidence threshold before auto-approval
"""

import json
import logging
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)


# Directories where autonomous modifications are allowed
SANDBOX_ALLOWED_DIRS = ["data", "plugins"]

# Directories that are blocked (read-only)
SANDBOX_BLOCKED_DIRS = ["core", "tests", ".git"]

# Confidence threshold for auto-approval (0-1)
AUTO_APPROVE_THRESHOLD = 0.7


@dataclass
class ModificationRequest:
    """Request to modify a file."""
    file_path: str
    operation: str  # "create", "update", "delete", "append"
    content: Optional[str] = None
    confidence: float = 0.5  # 0-1, confidence in the modification
    reason: str = "Autonomous update"
    requester: str = "system"


@dataclass
class ModificationResult:
    """Result of a modification operation."""
    success: bool
    file_path: str
    operation: str
    timestamp: str
    reason: str
    backup_created: bool = False
    backup_path: Optional[str] = None


class SandboxModifier:
    """Safely handles autonomous modifications to data/ and plugins/.
    
    Enforces boundaries, validates changes, logs all operations, and
    maintains rollback capability.
    """
    
    def __init__(self, data_dir: str = "data", approval_threshold: float = AUTO_APPROVE_THRESHOLD):
        """Initialize sandbox modifier.
        
        Args:
            data_dir: Base data directory.
            approval_threshold: Confidence threshold for auto-approval (0-1).
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.approval_threshold = approval_threshold
        self.modification_log_path = self.data_dir / "modification_log.jsonl"
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self.max_backups_per_file = 10
    
    def request_modification(self, request: ModificationRequest) -> ModificationResult:
        """Request a file modification.
        
        Args:
            request: ModificationRequest instance.
            
        Returns:
            ModificationResult with success status and details.
        """
        # Validate request
        validation_errors = self._validate_request(request)
        if validation_errors:
            result = ModificationResult(
                success=False,
                file_path=request.file_path,
                operation=request.operation,
                timestamp=datetime.now().isoformat(),
                reason=f"Validation failed: {'; '.join(validation_errors)}"
            )
            self._log_modification(result)
            return result
        
        # Check if auto-approval threshold met
        if request.confidence >= self.approval_threshold:
            return self._apply_modification(request)
        else:
            # Queue for human review
            return self._queue_for_review(request)
    
    def _validate_request(self, request: ModificationRequest) -> List[str]:
        """Validate modification request.
        
        Args:
            request: ModificationRequest instance.
            
        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        
        # Check path safety
        file_path = Path(request.file_path)
        
        # Block core/ and other protected directories
        for blocked_dir in SANDBOX_BLOCKED_DIRS:
            if blocked_dir in file_path.parts:
                errors.append(f"Cannot modify {blocked_dir}/ directory")
        
        # Allow only data/ and plugins/
        has_allowed_prefix = any(
            allowed in file_path.parts for allowed in SANDBOX_ALLOWED_DIRS
        )
        if not has_allowed_prefix:
            errors.append(f"Must be in {' or '.join(SANDBOX_ALLOWED_DIRS)}/ directory")
        
        # Validate content if provided
        if request.content:
            if request.file_path.endswith('.json'):
                try:
                    json.loads(request.content)
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON: {str(e)}")
            
            elif request.file_path.endswith('.py'):
                compile_errors = self._validate_python(request.content)
                if compile_errors:
                    errors.append(f"Python syntax error: {compile_errors}")
        
        return errors
    
    def _validate_python(self, code: str) -> Optional[str]:
        """Validate Python syntax.
        
        Args:
            code: Python code to validate.
            
        Returns:
            Error message if invalid, None if valid.
        """
        try:
            compile(code, '<string>', 'exec')
            return None
        except SyntaxError as e:
            return f"Line {e.lineno}: {e.msg}"
    
    def _apply_modification(self, request: ModificationRequest) -> ModificationResult:
        """Apply approved modification.
        
        Args:
            request: ModificationRequest instance.
            
        Returns:
            ModificationResult with success status.
        """
        file_path = Path(request.file_path)
        
        try:
            # Create backup if file exists and not appending
            backup_path = None
            if file_path.exists() and request.operation != "append":
                backup_path = self._create_backup(file_path)
            
            # Apply modification
            if request.operation == "create":
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(request.content or "")
            
            elif request.operation == "update":
                file_path.write_text(request.content or "")
            
            elif request.operation == "append":
                with open(file_path, 'a') as f:
                    f.write(request.content or "")
            
            elif request.operation == "delete":
                file_path.unlink()
            
            result = ModificationResult(
                success=True,
                file_path=request.file_path,
                operation=request.operation,
                timestamp=datetime.now().isoformat(),
                reason=request.reason,
                backup_created=backup_path is not None,
                backup_path=backup_path
            )
            
            self._log_modification(result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply modification: {e}")
            result = ModificationResult(
                success=False,
                file_path=request.file_path,
                operation=request.operation,
                timestamp=datetime.now().isoformat(),
                reason=f"Operation failed: {str(e)}"
            )
            self._log_modification(result)
            return result
    
    def _queue_for_review(self, request: ModificationRequest) -> ModificationResult:
        """Queue modification for human review.
        
        Args:
            request: ModificationRequest instance.
            
        Returns:
            ModificationResult with review status.
        """
        review_queue_path = self.data_dir / "review_queue.json"
        
        try:
            # Load existing queue
            if review_queue_path.exists():
                with open(review_queue_path, 'r') as f:
                    queue = json.load(f)
            else:
                queue = []
            
            # Add request to queue
            queue.append({
                "file_path": request.file_path,
                "operation": request.operation,
                "content": request.content,
                "confidence": request.confidence,
                "reason": request.reason,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save queue
            with open(review_queue_path, 'w') as f:
                json.dump(queue, f, indent=2)
            
            result = ModificationResult(
                success=True,
                file_path=request.file_path,
                operation=request.operation,
                timestamp=datetime.now().isoformat(),
                reason=f"Queued for review (confidence {request.confidence:.2f} < threshold {self.approval_threshold:.2f})"
            )
            
            self._log_modification(result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to queue for review: {e}")
            result = ModificationResult(
                success=False,
                file_path=request.file_path,
                operation=request.operation,
                timestamp=datetime.now().isoformat(),
                reason=f"Failed to queue: {str(e)}"
            )
            return result
    
    def _create_backup(self, file_path: Path) -> str:
        """Create backup of file before modification.
        
        Args:
            file_path: Path to file to backup.
            
        Returns:
            Path to backup file.
        """
        backup_subdir = self.backup_dir / file_path.name
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        # Find next backup number
        existing_backups = list(backup_subdir.glob("*.bak"))
        next_number = len(existing_backups) + 1
        
        backup_path = backup_subdir / f"backup_{next_number}.bak"
        
        # Copy file
        if file_path.is_file():
            shutil.copy2(file_path, backup_path)
        
        # Clean up old backups (keep only 10 most recent)
        all_backups = sorted(backup_subdir.glob("*.bak"))
        for old_backup in all_backups[:-self.max_backups_per_file]:
            old_backup.unlink()
        
        return str(backup_path)
    
    def rollback_modification(self, file_path: str, steps: int = 1) -> bool:
        """Rollback file to previous backup.
        
        Args:
            file_path: Path to file to rollback.
            steps: Number of versions to go back.
            
        Returns:
            True if rollback successful.
        """
        file_name = Path(file_path).name
        backup_subdir = self.backup_dir / file_name
        
        if not backup_subdir.exists():
            logger.warning(f"No backups found for {file_path}")
            return False
        
        try:
            # Get backups sorted by creation time
            all_backups = sorted(backup_subdir.glob("*.bak"), reverse=True)
            
            if len(all_backups) < steps:
                logger.warning(f"Cannot rollback {steps} steps (only {len(all_backups)} backups)")
                return False
            
            # Restore from backup
            backup_to_restore = all_backups[steps - 1]
            target_path = Path(file_path)
            
            shutil.copy2(backup_to_restore, target_path)
            
            logger.info(f"Rolled back {file_path} to {backup_to_restore.name}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _log_modification(self, result: ModificationResult):
        """Log modification event."""
        try:
            log_entry = {
                "timestamp": result.timestamp,
                "file_path": result.file_path,
                "operation": result.operation,
                "success": result.success,
                "reason": result.reason,
                "backup_path": result.backup_path
            }
            with open(self.modification_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log modification: {e}")
    
    def get_modification_history(self, file_path: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get history of modifications.
        
        Args:
            file_path: Optional specific file to filter on.
            limit: Maximum number of records to return.
            
        Returns:
            List of modification log entries.
        """
        if not self.modification_log_path.exists():
            return []
        
        try:
            history = []
            with open(self.modification_log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if file_path is None or entry["file_path"] == file_path:
                        history.append(entry)
            
            return history[-limit:]
        except Exception as e:
            logger.error(f"Failed to read modification history: {e}")
            return []


# Singleton instance
_instance: Optional[SandboxModifier] = None


def get_sandbox_modifier(data_dir: str = "data", 
                        approval_threshold: float = AUTO_APPROVE_THRESHOLD) -> SandboxModifier:
    """Get or create sandbox modifier singleton.
    
    Args:
        data_dir: Base data directory.
        approval_threshold: Confidence threshold for auto-approval (0-1).
        
    Returns:
        SandboxModifier instance.
    """
    global _instance
    if _instance is None:
        _instance = SandboxModifier(data_dir=data_dir, approval_threshold=approval_threshold)
    return _instance
