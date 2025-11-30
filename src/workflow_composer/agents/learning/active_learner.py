"""
Active Learning System.

Identifies high-value training examples through uncertainty sampling and
failure analysis. This enables efficient human review by prioritizing
the most impactful queries for annotation.

Key Features:
- Uncertainty sampling (low confidence queries)
- Failure pattern detection
- Balanced training batch generation
- Continuous learning loop integration

Author: BioPipelines Team
Date: November 2025
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from enum import Enum
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class QueryDifficulty(Enum):
    """Classification of query difficulty."""
    EASY = "easy"           # High confidence (>0.85), correct
    MODERATE = "moderate"   # Medium confidence (0.65-0.85), correct
    HARD = "hard"           # Low confidence (0.45-0.65), correct
    UNCERTAIN = "uncertain" # Very low confidence (<0.45)
    FAILURE = "failure"     # Any confidence, incorrect result


@dataclass
class LearningSignal:
    """
    A signal from agent execution for learning.
    
    Captures information about how the agent processed a query,
    enabling analysis and improvement.
    """
    query: str
    predicted_intent: str
    predicted_entities: Dict[str, Any]
    confidence: float
    tool_selected: Optional[str] = None
    execution_success: Optional[bool] = None
    actual_intent: Optional[str] = None  # Ground truth if available
    user_corrected: bool = False
    correction: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["predicted_entities"] = json.dumps(self.predicted_entities)
        if self.correction:
            result["correction"] = json.dumps(self.correction)
        return result


class ActiveLearner:
    """
    Active learning system for identifying high-value training examples.
    
    Uses uncertainty sampling and failure analysis to prioritize queries
    for human review, maximizing the impact of annotation effort.
    
    Strategies:
    1. Uncertainty Sampling: Prioritize low-confidence queries
    2. Failure Mining: Analyze and cluster failure patterns
    3. Diversity Sampling: Ensure coverage of different query types
    4. Correction Learning: Learn from user corrections
    
    Example Usage:
        learner = ActiveLearner()
        
        # Record a signal after processing
        signal = LearningSignal(
            query="analyze human RNA data",
            predicted_intent="workflow_generation",
            predicted_entities={"organism": "human"},
            confidence=0.75,
            execution_success=True
        )
        learner.record_signal(signal)
        
        # Get queries needing review
        priority_queries = learner.get_priority_queries(n=50)
    """
    
    # Confidence thresholds for difficulty classification
    THRESHOLDS = {
        "high": 0.85,
        "medium": 0.65,
        "low": 0.45,
    }
    
    def __init__(
        self,
        db_path: str = "active_learning.db",
        auto_init: bool = True
    ):
        """
        Initialize the active learner.
        
        Args:
            db_path: Path to SQLite database
            auto_init: Whether to initialize database automatically
        """
        self.db_path = Path(db_path)
        self._failure_patterns = defaultdict(list)
        
        if auto_init:
            self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Learning signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_hash TEXT,
                    predicted_intent TEXT,
                    predicted_entities TEXT,
                    confidence REAL,
                    difficulty TEXT,
                    tool_selected TEXT,
                    execution_success BOOLEAN,
                    actual_intent TEXT,
                    user_corrected BOOLEAN DEFAULT FALSE,
                    correction TEXT,
                    latency_ms REAL,
                    timestamp DATETIME,
                    reviewed BOOLEAN DEFAULT FALSE,
                    review_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Failure patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS failure_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    pattern_description TEXT,
                    example_queries TEXT,
                    frequency INTEGER DEFAULT 1,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Training batches table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_name TEXT,
                    signal_ids TEXT,
                    difficulty_distribution TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    used_for_training BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_signals_difficulty 
                ON learning_signals(difficulty)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_signals_reviewed 
                ON learning_signals(reviewed)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_signals_confidence 
                ON learning_signals(confidence)
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def record_signal(self, signal: LearningSignal) -> int:
        """
        Record a learning signal from agent execution.
        
        Args:
            signal: LearningSignal object
            
        Returns:
            Database ID of the inserted signal
        """
        difficulty = self._assess_difficulty(signal)
        query_hash = self._hash_query(signal.query)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_signals (
                    query, query_hash, predicted_intent, predicted_entities,
                    confidence, difficulty, tool_selected, execution_success,
                    actual_intent, user_corrected, correction, latency_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.query,
                query_hash,
                signal.predicted_intent,
                json.dumps(signal.predicted_entities),
                signal.confidence,
                difficulty.value,
                signal.tool_selected,
                signal.execution_success,
                signal.actual_intent,
                signal.user_corrected,
                json.dumps(signal.correction) if signal.correction else None,
                signal.latency_ms,
                signal.timestamp.isoformat(),
            ))
            conn.commit()
            
            signal_id = cursor.lastrowid
            
            # Track failure patterns
            if difficulty == QueryDifficulty.FAILURE:
                self._track_failure_pattern(conn, signal)
            
            logger.debug(f"Recorded signal (difficulty={difficulty.value}): {signal.query[:50]}...")
            return signal_id
    
    def _assess_difficulty(self, signal: LearningSignal) -> QueryDifficulty:
        """
        Assess the difficulty of a query based on confidence and outcome.
        
        Args:
            signal: LearningSignal to assess
            
        Returns:
            QueryDifficulty classification
        """
        # Check for failure first
        is_correct = (
            signal.actual_intent is None or  # No ground truth
            signal.predicted_intent == signal.actual_intent
        )
        
        if not is_correct or signal.execution_success is False:
            return QueryDifficulty.FAILURE
        
        if signal.user_corrected:
            return QueryDifficulty.FAILURE
        
        # Classify by confidence
        confidence = signal.confidence
        
        if confidence >= self.THRESHOLDS["high"]:
            return QueryDifficulty.EASY
        elif confidence >= self.THRESHOLDS["medium"]:
            return QueryDifficulty.MODERATE
        elif confidence >= self.THRESHOLDS["low"]:
            return QueryDifficulty.HARD
        else:
            return QueryDifficulty.UNCERTAIN
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        import hashlib
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _track_failure_pattern(self, conn: sqlite3.Connection, signal: LearningSignal):
        """Track failure patterns for analysis."""
        # Simple pattern detection - could be enhanced with clustering
        pattern_type = self._detect_pattern_type(signal)
        
        cursor = conn.cursor()
        
        # Check if pattern exists
        cursor.execute('''
            SELECT id, example_queries, frequency
            FROM failure_patterns
            WHERE pattern_type = ? AND resolved = FALSE
        ''', (pattern_type,))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing pattern
            examples = json.loads(existing["example_queries"])
            if len(examples) < 10:  # Keep up to 10 examples
                examples.append(signal.query)
            
            cursor.execute('''
                UPDATE failure_patterns
                SET example_queries = ?, frequency = frequency + 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (json.dumps(examples), existing["id"]))
        else:
            # Create new pattern
            cursor.execute('''
                INSERT INTO failure_patterns (pattern_type, pattern_description, example_queries)
                VALUES (?, ?, ?)
            ''', (
                pattern_type,
                f"Failures with {pattern_type}",
                json.dumps([signal.query])
            ))
        
        conn.commit()
    
    def _detect_pattern_type(self, signal: LearningSignal) -> str:
        """Detect the type of failure pattern."""
        query_lower = signal.query.lower()
        
        # Simple heuristic pattern detection
        if "not" in query_lower or "without" in query_lower or "except" in query_lower:
            return "negation"
        elif any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return "comparison"
        elif "?" in signal.query:
            return "question"
        elif len(signal.query.split()) <= 3:
            return "short_query"
        elif len(signal.query.split()) >= 20:
            return "long_query"
        elif signal.predicted_intent == "unknown":
            return "unknown_intent"
        else:
            return "general"
    
    def get_priority_queries(
        self,
        n: int = 50,
        include_reviewed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get highest priority queries for manual review.
        
        Prioritization order:
        1. User corrections (highest value)
        2. Failures
        3. Uncertain (very low confidence)
        4. Hard (low confidence)
        
        Args:
            n: Number of queries to return
            include_reviewed: Whether to include already reviewed queries
            
        Returns:
            List of query dictionaries with priority scores
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            reviewed_filter = "" if include_reviewed else "AND reviewed = FALSE"
            
            # Priority scoring query
            cursor.execute(f'''
                SELECT *,
                    CASE difficulty
                        WHEN 'failure' THEN 100
                        WHEN 'uncertain' THEN 80
                        WHEN 'hard' THEN 60
                        WHEN 'moderate' THEN 40
                        WHEN 'easy' THEN 20
                    END +
                    CASE WHEN user_corrected = TRUE THEN 50 ELSE 0 END +
                    (1 - COALESCE(confidence, 0.5)) * 20
                    AS priority_score
                FROM learning_signals
                WHERE 1=1 {reviewed_filter}
                ORDER BY priority_score DESC, timestamp DESC
                LIMIT ?
            ''', (n,))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Parse JSON fields
                if result.get("predicted_entities"):
                    result["predicted_entities"] = json.loads(result["predicted_entities"])
                if result.get("correction"):
                    result["correction"] = json.loads(result["correction"])
                results.append(result)
            
            return results
    
    def generate_training_batch(
        self,
        batch_size: int = 100,
        difficulty_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Generate a balanced training batch emphasizing hard examples.
        
        Default distribution:
        - 40% failures
        - 30% hard
        - 20% moderate
        - 10% easy
        
        Args:
            batch_size: Total size of the batch
            difficulty_weights: Custom weights per difficulty
            
        Returns:
            Tuple of (batch data, batch_id)
        """
        if difficulty_weights is None:
            difficulty_weights = {
                "failure": 0.40,
                "uncertain": 0.10,
                "hard": 0.20,
                "moderate": 0.20,
                "easy": 0.10,
            }
        
        batch = []
        signal_ids = []
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for difficulty, weight in difficulty_weights.items():
                count = int(batch_size * weight)
                if count == 0:
                    continue
                
                cursor.execute('''
                    SELECT * FROM learning_signals
                    WHERE difficulty = ? AND reviewed = FALSE
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (difficulty, count))
                
                for row in cursor.fetchall():
                    result = dict(row)
                    if result.get("predicted_entities"):
                        result["predicted_entities"] = json.loads(result["predicted_entities"])
                    batch.append(result)
                    signal_ids.append(result["id"])
            
            # Save batch
            cursor.execute('''
                INSERT INTO training_batches (batch_name, signal_ids, difficulty_distribution)
                VALUES (?, ?, ?)
            ''', (
                f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                json.dumps(signal_ids),
                json.dumps(difficulty_weights),
            ))
            batch_id = cursor.lastrowid
            conn.commit()
        
        return batch, batch_id
    
    def mark_reviewed(
        self,
        signal_ids: List[int],
        notes: Optional[str] = None
    ):
        """
        Mark signals as reviewed.
        
        Args:
            signal_ids: List of signal IDs to mark
            notes: Optional review notes
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(signal_ids))
            cursor.execute(f'''
                UPDATE learning_signals
                SET reviewed = TRUE, review_notes = ?
                WHERE id IN ({placeholders})
            ''', [notes] + signal_ids)
            conn.commit()
    
    def get_failure_patterns(
        self,
        min_frequency: int = 2,
        include_resolved: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get detected failure patterns.
        
        Args:
            min_frequency: Minimum frequency to include
            include_resolved: Whether to include resolved patterns
            
        Returns:
            List of pattern dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            resolved_filter = "" if include_resolved else "AND resolved = FALSE"
            
            cursor.execute(f'''
                SELECT * FROM failure_patterns
                WHERE frequency >= ? {resolved_filter}
                ORDER BY frequency DESC
            ''', (min_frequency,))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get("example_queries"):
                    result["example_queries"] = json.loads(result["example_queries"])
                results.append(result)
            
            return results
    
    def resolve_pattern(self, pattern_id: int, resolution: str):
        """
        Mark a failure pattern as resolved.
        
        Args:
            pattern_id: Pattern ID
            resolution: Description of how it was resolved
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE failure_patterns
                SET resolved = TRUE, resolution = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (resolution, pattern_id))
            conn.commit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total signals
            cursor.execute("SELECT COUNT(*) FROM learning_signals")
            stats["total_signals"] = cursor.fetchone()[0]
            
            # Difficulty distribution
            cursor.execute('''
                SELECT difficulty, COUNT(*) as count
                FROM learning_signals
                GROUP BY difficulty
            ''')
            stats["difficulty_distribution"] = {
                row[0]: row[1] for row in cursor.fetchall()
            }
            
            # Review progress
            cursor.execute('''
                SELECT 
                    COUNT(*) FILTER (WHERE reviewed = TRUE) as reviewed,
                    COUNT(*) FILTER (WHERE reviewed = FALSE) as pending
                FROM learning_signals
            ''')
            row = cursor.fetchone()
            stats["reviewed"] = row[0]
            stats["pending_review"] = row[1]
            
            # Failure patterns
            cursor.execute('''
                SELECT COUNT(*), SUM(frequency)
                FROM failure_patterns
                WHERE resolved = FALSE
            ''')
            row = cursor.fetchone()
            stats["unresolved_patterns"] = row[0]
            stats["total_failures_in_patterns"] = row[1] or 0
            
            # Average confidence by difficulty
            cursor.execute('''
                SELECT difficulty, AVG(confidence) as avg_conf
                FROM learning_signals
                WHERE confidence IS NOT NULL
                GROUP BY difficulty
            ''')
            stats["avg_confidence_by_difficulty"] = {
                row[0]: round(row[1], 3) for row in cursor.fetchall()
            }
            
            return stats
    
    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """
        Suggest improvements based on failure patterns.
        
        Returns:
            List of improvement suggestions
        """
        patterns = self.get_failure_patterns(min_frequency=3)
        suggestions = []
        
        for pattern in patterns:
            pattern_type = pattern["pattern_type"]
            frequency = pattern["frequency"]
            examples = pattern.get("example_queries", [])
            
            suggestion = {
                "pattern_type": pattern_type,
                "frequency": frequency,
                "severity": "high" if frequency >= 10 else "medium" if frequency >= 5 else "low",
                "examples": examples[:3],
            }
            
            # Add specific recommendations
            if pattern_type == "negation":
                suggestion["recommendation"] = "Improve negation handling in query parser"
                suggestion["priority"] = "high"
            elif pattern_type == "short_query":
                suggestion["recommendation"] = "Add context expansion for short queries"
                suggestion["priority"] = "medium"
            elif pattern_type == "unknown_intent":
                suggestion["recommendation"] = "Expand intent vocabulary or add new intents"
                suggestion["priority"] = "high"
            elif pattern_type == "comparison":
                suggestion["recommendation"] = "Add comparative query handling"
                suggestion["priority"] = "medium"
            else:
                suggestion["recommendation"] = f"Investigate {pattern_type} failure pattern"
                suggestion["priority"] = "low"
            
            suggestions.append(suggestion)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return suggestions
