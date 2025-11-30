"""
Evaluation Database Infrastructure
===================================

Persistent SQLite database for storing:
- Test conversations (generated + curated)
- Evaluation results
- Experiment runs
- Metrics history
- Improvement tracking

This enables:
- Long-term trend analysis
- A/B testing of parsing methods
- Regression detection
- Performance benchmarking
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Database file location
DEFAULT_DB_PATH = Path(__file__).parent / "evaluation.db"


@dataclass
class ConversationRecord:
    """A test conversation stored in the database."""
    id: str
    name: str
    category: str
    difficulty: str
    source: str  # 'generated', 'curated', 'production'
    turns_json: str  # JSON serialized turns
    created_at: str
    tags: str  # comma-separated tags
    description: str = ""
    hash: str = ""  # Content hash for deduplication
    
    @property
    def turns(self) -> List[Dict]:
        return json.loads(self.turns_json)


@dataclass
class EvaluationResult:
    """Result of evaluating a single conversation."""
    id: int
    experiment_id: str
    conversation_id: str
    intent_accuracy: float
    entity_f1: float
    tool_accuracy: float
    latency_ms: float
    passed: bool
    turns_detail_json: str  # JSON with per-turn details
    evaluated_at: str
    parser_config: str  # JSON with parser configuration


@dataclass 
class Experiment:
    """An experiment run with specific configuration."""
    id: str
    name: str
    description: str
    parser_config_json: str  # JSON with full configuration
    started_at: str
    completed_at: Optional[str]
    total_conversations: int
    passed_conversations: int
    overall_intent_accuracy: float
    overall_entity_f1: float
    overall_tool_accuracy: float
    avg_latency_ms: float
    status: str  # 'running', 'completed', 'failed'
    notes: str = ""


class EvaluationDatabase:
    """
    SQLite database for persistent evaluation storage.
    
    Features:
    - Store thousands of test conversations
    - Track evaluation results over time
    - Compare experiments with different configurations
    - Query by category, difficulty, failure patterns
    """
    
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    source TEXT NOT NULL,
                    turns_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    tags TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    hash TEXT UNIQUE
                )
            """)
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    parser_config_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_conversations INTEGER DEFAULT 0,
                    passed_conversations INTEGER DEFAULT 0,
                    overall_intent_accuracy REAL DEFAULT 0,
                    overall_entity_f1 REAL DEFAULT 0,
                    overall_tool_accuracy REAL DEFAULT 0,
                    avg_latency_ms REAL DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    notes TEXT DEFAULT ''
                )
            """)
            
            # Evaluation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    intent_accuracy REAL NOT NULL,
                    entity_f1 REAL NOT NULL,
                    tool_accuracy REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    passed INTEGER NOT NULL,
                    turns_detail_json TEXT,
                    evaluated_at TEXT NOT NULL,
                    parser_config TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            # Metrics history for trend tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    category TEXT,
                    experiment_id TEXT,
                    notes TEXT
                )
            """)
            
            # Failure analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS failure_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    description TEXT,
                    example_queries TEXT,
                    expected_intent TEXT,
                    actual_intent TEXT,
                    frequency INTEGER DEFAULT 1,
                    first_seen TEXT,
                    last_seen TEXT,
                    status TEXT DEFAULT 'open',  -- open, fixed, wontfix
                    fix_notes TEXT
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_category ON conversations(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_difficulty ON conversations(difficulty)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_source ON conversations(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_experiment ON evaluation_results(experiment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_passed ON evaluation_results(passed)")
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _compute_hash(self, turns: List[Dict]) -> str:
        """Compute hash of conversation turns for deduplication."""
        content = json.dumps(turns, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    # =========================================================================
    # Conversation Management
    # =========================================================================
    
    def add_conversation(
        self,
        id: str,
        name: str,
        category: str,
        difficulty: str,
        turns: List[Dict],
        source: str = "generated",
        tags: List[str] = None,
        description: str = ""
    ) -> bool:
        """Add a conversation to the database."""
        turns_json = json.dumps(turns)
        conv_hash = self._compute_hash(turns)
        tags_str = ",".join(tags) if tags else ""
        
        with self._connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO conversations 
                    (id, name, category, difficulty, source, turns_json, 
                     created_at, tags, description, hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    id, name, category, difficulty, source, turns_json,
                    datetime.now().isoformat(), tags_str, description, conv_hash
                ))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Duplicate conversation
                return False
    
    def add_conversations_bulk(self, conversations: List[Dict]) -> Tuple[int, int]:
        """Add multiple conversations. Returns (added, skipped) counts."""
        added = 0
        skipped = 0
        
        with self._connection() as conn:
            for conv in conversations:
                turns_json = json.dumps(conv['turns'])
                conv_hash = self._compute_hash(conv['turns'])
                tags_str = ",".join(conv.get('tags', []))
                
                try:
                    conn.execute("""
                        INSERT INTO conversations 
                        (id, name, category, difficulty, source, turns_json, 
                         created_at, tags, description, hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        conv['id'], conv['name'], conv['category'], 
                        conv.get('difficulty', 'medium'),
                        conv.get('source', 'generated'), turns_json,
                        datetime.now().isoformat(), tags_str,
                        conv.get('description', ''), conv_hash
                    ))
                    added += 1
                except sqlite3.IntegrityError:
                    skipped += 1
            
            conn.commit()
        
        return added, skipped
    
    def get_conversations(
        self,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ConversationRecord]:
        """Query conversations with filters."""
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
        if source:
            query += " AND source = ?"
            params.append(source)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        with self._connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [ConversationRecord(**dict(row)) for row in rows]
    
    def get_conversation_count(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None
    ) -> int:
        """Get count of conversations matching filters."""
        query = "SELECT COUNT(*) FROM conversations WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        if source:
            query += " AND source = ?"
            params.append(source)
        
        with self._connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]
    
    def get_categories_summary(self) -> Dict[str, int]:
        """Get count of conversations per category."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT category, COUNT(*) as count 
                FROM conversations 
                GROUP BY category
            """)
            return {row['category']: row['count'] for row in cursor.fetchall()}
    
    # =========================================================================
    # Experiment Management
    # =========================================================================
    
    def create_experiment(
        self,
        name: str,
        description: str,
        parser_config: Dict,
    ) -> str:
        """Create a new experiment and return its ID."""
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO experiments
                (id, name, description, parser_config_json, started_at, status)
                VALUES (?, ?, ?, ?, ?, 'running')
            """, (
                exp_id, name, description,
                json.dumps(parser_config),
                datetime.now().isoformat()
            ))
            conn.commit()
        
        return exp_id
    
    def update_experiment_results(
        self,
        experiment_id: str,
        total: int,
        passed: int,
        intent_acc: float,
        entity_f1: float,
        tool_acc: float,
        avg_latency: float,
        status: str = "completed",
        notes: str = ""
    ):
        """Update experiment with final results."""
        with self._connection() as conn:
            conn.execute("""
                UPDATE experiments SET
                    completed_at = ?,
                    total_conversations = ?,
                    passed_conversations = ?,
                    overall_intent_accuracy = ?,
                    overall_entity_f1 = ?,
                    overall_tool_accuracy = ?,
                    avg_latency_ms = ?,
                    status = ?,
                    notes = ?
                WHERE id = ?
            """, (
                datetime.now().isoformat(),
                total, passed, intent_acc, entity_f1, tool_acc, avg_latency,
                status, notes, experiment_id
            ))
            conn.commit()
    
    def get_experiments(self, limit: int = 20) -> List[Experiment]:
        """Get recent experiments."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM experiments 
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
            return [Experiment(**dict(row)) for row in cursor.fetchall()]
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get a specific experiment."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,)
            )
            row = cursor.fetchone()
            return Experiment(**dict(row)) if row else None
    
    # =========================================================================
    # Evaluation Results
    # =========================================================================
    
    def add_result(
        self,
        experiment_id: str,
        conversation_id: str,
        intent_accuracy: float,
        entity_f1: float,
        tool_accuracy: float,
        latency_ms: float,
        passed: bool,
        turns_detail: List[Dict] = None,
        parser_config: Dict = None
    ):
        """Add an evaluation result."""
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO evaluation_results
                (experiment_id, conversation_id, intent_accuracy, entity_f1,
                 tool_accuracy, latency_ms, passed, turns_detail_json,
                 evaluated_at, parser_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, conversation_id, intent_accuracy, entity_f1,
                tool_accuracy, latency_ms, 1 if passed else 0,
                json.dumps(turns_detail) if turns_detail else None,
                datetime.now().isoformat(),
                json.dumps(parser_config) if parser_config else None
            ))
            conn.commit()
    
    def get_failed_results(
        self,
        experiment_id: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get failed results for analysis."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT r.*, c.name, c.category, c.turns_json
                FROM evaluation_results r
                JOIN conversations c ON r.conversation_id = c.id
                WHERE r.experiment_id = ? AND r.passed = 0
                ORDER BY r.intent_accuracy ASC
                LIMIT ?
            """, (experiment_id, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_category_results(
        self,
        experiment_id: str
    ) -> Dict[str, Dict[str, float]]:
        """Get results breakdown by category."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    c.category,
                    COUNT(*) as total,
                    SUM(r.passed) as passed,
                    AVG(r.intent_accuracy) as intent_acc,
                    AVG(r.entity_f1) as entity_f1,
                    AVG(r.tool_accuracy) as tool_acc,
                    AVG(r.latency_ms) as latency
                FROM evaluation_results r
                JOIN conversations c ON r.conversation_id = c.id
                WHERE r.experiment_id = ?
                GROUP BY c.category
            """, (experiment_id,))
            
            results = {}
            for row in cursor.fetchall():
                results[row['category']] = {
                    'total': row['total'],
                    'passed': row['passed'],
                    'pass_rate': row['passed'] / row['total'] if row['total'] > 0 else 0,
                    'intent_accuracy': row['intent_acc'],
                    'entity_f1': row['entity_f1'],
                    'tool_accuracy': row['tool_acc'],
                    'latency_ms': row['latency']
                }
            return results
    
    # =========================================================================
    # Failure Pattern Tracking
    # =========================================================================
    
    def record_failure_pattern(
        self,
        pattern_name: str,
        description: str,
        example_queries: List[str],
        expected_intent: str,
        actual_intent: str
    ):
        """Record a new failure pattern or update existing."""
        with self._connection() as conn:
            # Check if pattern exists
            cursor = conn.execute(
                "SELECT id, frequency FROM failure_patterns WHERE pattern_name = ?",
                (pattern_name,)
            )
            existing = cursor.fetchone()
            
            if existing:
                conn.execute("""
                    UPDATE failure_patterns SET
                        frequency = frequency + 1,
                        last_seen = ?,
                        example_queries = ?
                    WHERE id = ?
                """, (
                    datetime.now().isoformat(),
                    json.dumps(example_queries),
                    existing['id']
                ))
            else:
                conn.execute("""
                    INSERT INTO failure_patterns
                    (pattern_name, description, example_queries, expected_intent,
                     actual_intent, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_name, description, json.dumps(example_queries),
                    expected_intent, actual_intent,
                    datetime.now().isoformat(), datetime.now().isoformat()
                ))
            conn.commit()
    
    def get_open_failure_patterns(self) -> List[Dict]:
        """Get unresolved failure patterns."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM failure_patterns
                WHERE status = 'open'
                ORDER BY frequency DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_pattern_fixed(self, pattern_id: int, fix_notes: str):
        """Mark a failure pattern as fixed."""
        with self._connection() as conn:
            conn.execute("""
                UPDATE failure_patterns SET
                    status = 'fixed',
                    fix_notes = ?
                WHERE id = ?
            """, (fix_notes, pattern_id))
            conn.commit()
    
    # =========================================================================
    # Metrics History
    # =========================================================================
    
    def record_metric(
        self,
        metric_name: str,
        metric_value: float,
        category: Optional[str] = None,
        experiment_id: Optional[str] = None,
        notes: str = ""
    ):
        """Record a metric value for trend tracking."""
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO metrics_history
                (recorded_at, metric_name, metric_value, category, experiment_id, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                metric_name, metric_value, category, experiment_id, notes
            ))
            conn.commit()
    
    def get_metric_history(
        self,
        metric_name: str,
        days: int = 30
    ) -> List[Tuple[str, float]]:
        """Get metric history for trend analysis."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT recorded_at, metric_value
                FROM metrics_history
                WHERE metric_name = ?
                  AND recorded_at >= datetime('now', ?)
                ORDER BY recorded_at ASC
            """, (metric_name, f'-{days} days'))
            return [(row['recorded_at'], row['metric_value']) for row in cursor.fetchall()]
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of the evaluation database."""
        with self._connection() as conn:
            # Conversation counts
            conv_cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN source = 'generated' THEN 1 ELSE 0 END) as generated,
                    SUM(CASE WHEN source = 'curated' THEN 1 ELSE 0 END) as curated,
                    SUM(CASE WHEN source = 'production' THEN 1 ELSE 0 END) as production
                FROM conversations
            """)
            conv_stats = dict(conv_cursor.fetchone())
            
            # Experiment count
            exp_cursor = conn.execute("SELECT COUNT(*) as count FROM experiments")
            exp_count = exp_cursor.fetchone()['count']
            
            # Latest experiment
            latest_cursor = conn.execute("""
                SELECT * FROM experiments 
                WHERE status = 'completed'
                ORDER BY completed_at DESC LIMIT 1
            """)
            latest = latest_cursor.fetchone()
            
            # Open failure patterns
            fail_cursor = conn.execute(
                "SELECT COUNT(*) as count FROM failure_patterns WHERE status = 'open'"
            )
            open_failures = fail_cursor.fetchone()['count']
            
            return {
                'conversations': conv_stats,
                'categories': self.get_categories_summary(),
                'experiments_count': exp_count,
                'latest_experiment': dict(latest) if latest else None,
                'open_failure_patterns': open_failures,
                'generated_at': datetime.now().isoformat()
            }


# Singleton instance
_db_instance = None

def get_database() -> EvaluationDatabase:
    """Get the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = EvaluationDatabase()
    return _db_instance


if __name__ == "__main__":
    # Test the database
    db = get_database()
    
    # Add a test conversation
    db.add_conversation(
        id="test-001",
        name="Test Conversation",
        category="data_discovery",
        difficulty="easy",
        turns=[{
            "query": "Search for human RNA-seq data",
            "expected_intent": "DATA_SEARCH",
            "expected_entities": {"ORGANISM": "human", "ASSAY_TYPE": "RNA-seq"}
        }],
        source="curated",
        tags=["test", "rna-seq"]
    )
    
    print(f"Database summary: {db.generate_summary_report()}")
