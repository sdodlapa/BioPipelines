"""
Diagnosis history tracking for pattern learning.

Stores diagnosis results to enable:
- Historical analysis of common errors
- Pattern frequency tracking
- Fix success rate monitoring
- Learning from resolved issues
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

from .categories import ErrorDiagnosis, ErrorCategory, FixRiskLevel

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisRecord:
    """A recorded diagnosis entry."""
    timestamp: str
    job_id: str
    workflow_name: str
    error_category: str
    root_cause: str
    confidence: float
    pattern_matched: bool
    llm_provider: Optional[str]
    suggested_fixes: List[str]
    fix_applied: Optional[str] = None
    fix_success: Optional[bool] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DiagnosisRecord":
        return cls(**data)


class DiagnosisHistory:
    """
    Manages diagnosis history storage and analysis.
    
    Stores diagnosis records in a JSON file for analysis
    and pattern learning.
    
    Example:
        history = DiagnosisHistory()
        history.record(diagnosis, job)
        
        # Get common errors
        common = history.get_common_errors(limit=10)
    """
    
    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize history manager.
        
        Args:
            history_file: Path to history JSON file
        """
        if history_file:
            self.history_file = Path(history_file)
        else:
            # Default location
            base_dir = Path(__file__).parent.parent.parent.parent
            self.history_file = base_dir / "logs" / "diagnosis_history.json"
        
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[DiagnosisRecord] = []
        self._load()
    
    def _load(self) -> None:
        """Load history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self._records = [
                        DiagnosisRecord.from_dict(r) 
                        for r in data.get("records", [])
                    ]
                logger.info(f"Loaded {len(self._records)} diagnosis records")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self._records = []
        else:
            self._records = []
    
    def _save(self) -> None:
        """Save history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    "version": "1.0",
                    "updated_at": datetime.now().isoformat(),
                    "record_count": len(self._records),
                    "records": [r.to_dict() for r in self._records],
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def record(
        self,
        diagnosis: ErrorDiagnosis,
        job_id: str,
        workflow_name: str = "Unknown",
    ) -> DiagnosisRecord:
        """
        Record a new diagnosis.
        
        Args:
            diagnosis: The ErrorDiagnosis
            job_id: Job identifier
            workflow_name: Name of the workflow
            
        Returns:
            The created DiagnosisRecord
        """
        record = DiagnosisRecord(
            timestamp=datetime.now().isoformat(),
            job_id=job_id,
            workflow_name=workflow_name,
            error_category=diagnosis.category.value,
            root_cause=diagnosis.root_cause,
            confidence=diagnosis.confidence,
            pattern_matched=diagnosis.pattern_matched,
            llm_provider=diagnosis.llm_provider_used,
            suggested_fixes=[f.description for f in diagnosis.suggested_fixes[:5]],
        )
        
        self._records.append(record)
        self._save()
        
        logger.info(f"Recorded diagnosis for job {job_id}: {diagnosis.category.value}")
        return record
    
    def update_resolution(
        self,
        job_id: str,
        fix_applied: str,
        fix_success: bool,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update a record with resolution information.
        
        Args:
            job_id: Job ID to update
            fix_applied: Description of fix applied
            fix_success: Whether the fix worked
            notes: Additional notes
            
        Returns:
            True if record was found and updated
        """
        for record in reversed(self._records):
            if record.job_id == job_id:
                record.fix_applied = fix_applied
                record.fix_success = fix_success
                record.resolution_notes = notes
                self._save()
                return True
        return False
    
    def get_common_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most common error categories.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of dicts with category and count
        """
        category_counts = defaultdict(int)
        
        for record in self._records:
            category_counts[record.error_category] += 1
        
        sorted_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {"category": cat, "count": count}
            for cat, count in sorted_categories
        ]
    
    def get_fix_success_rate(self) -> Dict[str, float]:
        """
        Get fix success rate by category.
        
        Returns:
            Dict mapping category to success rate
        """
        category_stats = defaultdict(lambda: {"success": 0, "total": 0})
        
        for record in self._records:
            if record.fix_success is not None:
                category_stats[record.error_category]["total"] += 1
                if record.fix_success:
                    category_stats[record.error_category]["success"] += 1
        
        return {
            cat: stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
            for cat, stats in category_stats.items()
        }
    
    def get_recent(self, limit: int = 20) -> List[DiagnosisRecord]:
        """
        Get recent diagnosis records.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            List of recent records
        """
        return list(reversed(self._records[-limit:]))
    
    def get_by_category(self, category: str) -> List[DiagnosisRecord]:
        """
        Get all records for a specific category.
        
        Args:
            category: Error category name
            
        Returns:
            List of matching records
        """
        return [r for r in self._records if r.error_category == category]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics.
        
        Returns:
            Dict with statistics
        """
        if not self._records:
            return {
                "total_diagnoses": 0,
                "categories": {},
                "pattern_match_rate": 0.0,
                "avg_confidence": 0.0,
            }
        
        pattern_matched = sum(1 for r in self._records if r.pattern_matched)
        total_confidence = sum(r.confidence for r in self._records)
        
        return {
            "total_diagnoses": len(self._records),
            "categories": self.get_common_errors(20),
            "pattern_match_rate": pattern_matched / len(self._records),
            "avg_confidence": total_confidence / len(self._records),
            "fix_success_rates": self.get_fix_success_rate(),
            "first_record": self._records[0].timestamp if self._records else None,
            "last_record": self._records[-1].timestamp if self._records else None,
        }
    
    def export_for_training(self, output_file: Optional[Path] = None) -> Path:
        """
        Export records in a format suitable for model training.
        
        Args:
            output_file: Output file path
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.history_file.with_suffix('.training.jsonl')
        
        with open(output_file, 'w') as f:
            for record in self._records:
                if record.fix_success is not None:  # Only records with resolution
                    training_entry = {
                        "error_category": record.error_category,
                        "root_cause": record.root_cause,
                        "fix_applied": record.fix_applied,
                        "success": record.fix_success,
                    }
                    f.write(json.dumps(training_entry) + "\n")
        
        logger.info(f"Exported {len(self._records)} records to {output_file}")
        return output_file


# Singleton instance
_history: Optional[DiagnosisHistory] = None


def get_diagnosis_history() -> DiagnosisHistory:
    """Get or create the diagnosis history singleton."""
    global _history
    
    if _history is None:
        _history = DiagnosisHistory()
    
    return _history


def record_diagnosis(
    diagnosis: ErrorDiagnosis,
    job_id: str,
    workflow_name: str = "Unknown",
) -> DiagnosisRecord:
    """
    Convenience function to record a diagnosis.
    
    Args:
        diagnosis: The ErrorDiagnosis
        job_id: Job ID
        workflow_name: Workflow name
        
    Returns:
        DiagnosisRecord
    """
    history = get_diagnosis_history()
    return history.record(diagnosis, job_id, workflow_name)
