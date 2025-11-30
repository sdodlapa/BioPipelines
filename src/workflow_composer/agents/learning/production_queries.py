"""
Production Query Collector.

Collects and stores real user queries for future analysis and training.
This enables data-driven improvements by understanding how real users
interact with the system.

Features:
- Deduplication of similar queries
- Anonymization of user IDs
- Storage of parsed results for quality assessment
- Export to various formats for model training

Author: BioPipelines Team
Date: November 2025
"""
import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager


logger = logging.getLogger(__name__)


@dataclass
class ProductionQuery:
    """A real query from production usage."""
    query: str
    timestamp: datetime
    user_id_hash: str  # Anonymized user ID
    session_id: str
    parsed_intent: Optional[str] = None
    parsed_entities: Optional[Dict[str, Any]] = None
    tool_executed: Optional[str] = None
    execution_success: Optional[bool] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    user_feedback: Optional[str] = None  # 'positive', 'negative', 'corrected'
    correction: Optional[Dict[str, Any]] = None  # User's correction if any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        result["parsed_entities"] = json.dumps(self.parsed_entities) if self.parsed_entities else None
        result["correction"] = json.dumps(self.correction) if self.correction else None
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductionQuery":
        """Create from dictionary."""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if isinstance(data.get("parsed_entities"), str):
            data["parsed_entities"] = json.loads(data["parsed_entities"])
        if isinstance(data.get("correction"), str):
            data["correction"] = json.loads(data["correction"])
        return cls(**data)


class ProductionQueryCollector:
    """
    Collects queries from production for evaluation and training.
    
    This collector stores real user queries with their parsing results,
    enabling:
    - Analysis of real-world query patterns
    - Identification of failure cases
    - Generation of training data for fine-tuning
    - A/B testing of parser improvements
    
    Example Usage:
        collector = ProductionQueryCollector()
        
        # After processing a query
        query = ProductionQuery(
            query="analyze my RNA-seq data for human",
            timestamp=datetime.now(),
            user_id_hash=hash_user_id(user_id),
            session_id=session_id,
            parsed_intent="workflow_generation",
            parsed_entities={"organism": "human", "assay_type": "rna-seq"},
            tool_executed="generate_workflow",
            execution_success=True,
            confidence=0.92
        )
        collector.collect(query)
    """
    
    def __init__(
        self,
        db_path: str = "production_queries.db",
        auto_init: bool = True
    ):
        """
        Initialize the collector.
        
        Args:
            db_path: Path to SQLite database
            auto_init: Whether to initialize database automatically
        """
        self.db_path = Path(db_path)
        if auto_init:
            self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main queries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS production_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_hash TEXT UNIQUE,
                    normalized_query TEXT,
                    timestamp DATETIME,
                    user_id_hash TEXT,
                    session_id TEXT,
                    parsed_intent TEXT,
                    parsed_entities TEXT,
                    tool_executed TEXT,
                    execution_success BOOLEAN,
                    confidence REAL,
                    latency_ms REAL,
                    user_feedback TEXT,
                    correction TEXT,
                    added_to_eval BOOLEAN DEFAULT FALSE,
                    reviewed BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Index for efficient queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_queries_timestamp 
                ON production_queries(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_queries_intent 
                ON production_queries(parsed_intent)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_queries_feedback 
                ON production_queries(user_feedback)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_queries_reviewed 
                ON production_queries(reviewed)
            ''')
            
            # Sessions table for context
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id_hash TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    query_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0
                )
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
    
    def collect(self, query: ProductionQuery) -> bool:
        """
        Collect a production query.
        
        Args:
            query: ProductionQuery object to store
            
        Returns:
            True if new query was added, False if duplicate
        """
        # Generate hash for deduplication
        query_hash = self._hash_query(query.query)
        normalized = self._normalize_query(query.query)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO production_queries (
                        query, query_hash, normalized_query, timestamp,
                        user_id_hash, session_id, parsed_intent, parsed_entities,
                        tool_executed, execution_success, confidence, latency_ms,
                        user_feedback, correction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    query.query,
                    query_hash,
                    normalized,
                    query.timestamp.isoformat() if query.timestamp else None,
                    query.user_id_hash,
                    query.session_id,
                    query.parsed_intent,
                    json.dumps(query.parsed_entities) if query.parsed_entities else None,
                    query.tool_executed,
                    query.execution_success,
                    query.confidence,
                    query.latency_ms,
                    query.user_feedback,
                    json.dumps(query.correction) if query.correction else None,
                ))
                conn.commit()
                
                # Update session stats
                self._update_session(
                    conn, query.session_id, query.user_id_hash,
                    query.execution_success
                )
                
                logger.debug(f"Collected query: {query.query[:50]}...")
                return True
                
            except sqlite3.IntegrityError:
                # Duplicate query
                logger.debug(f"Duplicate query ignored: {query.query[:50]}...")
                return False
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query deduplication."""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison."""
        # Lowercase, strip whitespace, normalize spaces
        normalized = query.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized
    
    def _update_session(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        user_id_hash: str,
        success: Optional[bool]
    ):
        """Update session statistics."""
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id_hash, start_time, query_count, success_count)
            VALUES (?, ?, CURRENT_TIMESTAMP, 1, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                query_count = query_count + 1,
                success_count = success_count + ?,
                end_time = CURRENT_TIMESTAMP
        ''', (
            session_id, user_id_hash,
            1 if success else 0,
            1 if success else 0
        ))
        conn.commit()
    
    def get_unreviewed_queries(
        self,
        limit: int = 100,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get queries that haven't been reviewed yet.
        
        Args:
            limit: Maximum number to return
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            
        Returns:
            List of query dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM production_queries
                WHERE reviewed = FALSE
            '''
            params = []
            
            if min_confidence is not None:
                query += " AND confidence >= ?"
                params.append(min_confidence)
            
            if max_confidence is not None:
                query += " AND confidence <= ?"
                params.append(max_confidence)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_failed_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get queries that resulted in execution failure."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM production_queries
                WHERE execution_success = FALSE
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_low_confidence_queries(
        self,
        threshold: float = 0.7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get queries with confidence below threshold."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM production_queries
                WHERE confidence < ? AND confidence IS NOT NULL
                ORDER BY confidence ASC
                LIMIT ?
            ''', (threshold, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_corrected_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get queries where user provided corrections."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM production_queries
                WHERE user_feedback = 'corrected' AND correction IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_reviewed(self, query_ids: List[int], add_to_eval: bool = False):
        """
        Mark queries as reviewed.
        
        Args:
            query_ids: List of query IDs to mark
            add_to_eval: Whether to add to evaluation set
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(query_ids))
            cursor.execute(f'''
                UPDATE production_queries
                SET reviewed = TRUE, added_to_eval = ?
                WHERE id IN ({placeholders})
            ''', [add_to_eval] + query_ids)
            conn.commit()
    
    def export_for_training(
        self,
        output_path: str,
        format: str = "jsonl",
        only_successful: bool = True
    ) -> int:
        """
        Export queries for model training.
        
        Args:
            output_path: Path to output file
            format: Output format ('jsonl', 'csv', 'openai')
            only_successful: Only export successful executions
            
        Returns:
            Number of queries exported
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM production_queries WHERE added_to_eval = TRUE"
            if only_successful:
                query += " AND execution_success = TRUE"
            
            cursor.execute(query)
            rows = [dict(row) for row in cursor.fetchall()]
        
        output_path = Path(output_path)
        
        if format == "jsonl":
            self._export_jsonl(rows, output_path)
        elif format == "csv":
            self._export_csv(rows, output_path)
        elif format == "openai":
            self._export_openai_format(rows, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return len(rows)
    
    def _export_jsonl(self, rows: List[Dict], output_path: Path):
        """Export as JSONL."""
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    
    def _export_csv(self, rows: List[Dict], output_path: Path):
        """Export as CSV."""
        import csv
        if not rows:
            return
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    def _export_openai_format(self, rows: List[Dict], output_path: Path):
        """Export in OpenAI fine-tuning format."""
        system_prompt = """You are a bioinformatics assistant that parses user queries.
Extract: intent (workflow_generation, data_discovery, etc.), entities (organism, assay_type, etc.), and tool to use.
Respond in JSON format."""

        with open(output_path, "w") as f:
            for row in rows:
                response = {
                    "intent": row.get("parsed_intent"),
                    "entities": json.loads(row["parsed_entities"]) if row.get("parsed_entities") else {},
                    "tool": row.get("tool_executed"),
                }
                
                entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["query"]},
                        {"role": "assistant", "content": json.dumps(response)},
                    ]
                }
                f.write(json.dumps(entry) + "\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total queries
            cursor.execute("SELECT COUNT(*) FROM production_queries")
            stats["total_queries"] = cursor.fetchone()[0]
            
            # Unique queries
            cursor.execute("SELECT COUNT(DISTINCT query_hash) FROM production_queries")
            stats["unique_queries"] = cursor.fetchone()[0]
            
            # Success rate
            cursor.execute('''
                SELECT 
                    COUNT(*) FILTER (WHERE execution_success = TRUE) as success,
                    COUNT(*) FILTER (WHERE execution_success = FALSE) as failure
                FROM production_queries
                WHERE execution_success IS NOT NULL
            ''')
            row = cursor.fetchone()
            if row[0] + row[1] > 0:
                stats["success_rate"] = row[0] / (row[0] + row[1])
            else:
                stats["success_rate"] = 0.0
            
            # Average confidence
            cursor.execute('''
                SELECT AVG(confidence) FROM production_queries
                WHERE confidence IS NOT NULL
            ''')
            stats["avg_confidence"] = cursor.fetchone()[0] or 0.0
            
            # Intent distribution
            cursor.execute('''
                SELECT parsed_intent, COUNT(*) as count
                FROM production_queries
                WHERE parsed_intent IS NOT NULL
                GROUP BY parsed_intent
                ORDER BY count DESC
            ''')
            stats["intent_distribution"] = {
                row[0]: row[1] for row in cursor.fetchall()
            }
            
            # Reviewed count
            cursor.execute('''
                SELECT 
                    COUNT(*) FILTER (WHERE reviewed = TRUE) as reviewed,
                    COUNT(*) FILTER (WHERE reviewed = FALSE) as pending
                FROM production_queries
            ''')
            row = cursor.fetchone()
            stats["reviewed"] = row[0]
            stats["pending_review"] = row[1]
            
            return stats


def anonymize_user_id(user_id: str, salt: str = "") -> str:
    """
    Anonymize user ID for privacy.
    
    Args:
        user_id: Original user ID
        salt: Optional salt for additional security
        
    Returns:
        Anonymized hash
    """
    combined = f"{user_id}:{salt}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
