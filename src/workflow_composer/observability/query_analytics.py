"""
Query Analytics
===============

Track and analyze user query patterns for insights and optimization.

Features:
- Query recording with metadata
- Analysis type distribution
- Success rate tracking
- Organism popularity
- Performance trends
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """Record of a user query."""
    query: str
    analysis_type: str
    organism: Optional[str] = None
    genome_build: Optional[str] = None
    success: bool = True
    duration_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    workflow_generated: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "analysis_type": self.analysis_type,
            "organism": self.organism,
            "genome_build": self.genome_build,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "workflow_generated": self.workflow_generated,
            "error": self.error,
        }


class QueryAnalytics:
    """
    Analyze query patterns and success rates.
    
    Tracks:
    - Query distribution by analysis type
    - Success rates by type/organism
    - Popular organisms and genomes
    - Performance trends over time
    """
    
    def __init__(self, db_path: str = "~/.biopipelines/analytics.db"):
        """
        Initialize query analytics.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    organism TEXT,
                    genome_build TEXT,
                    success INTEGER NOT NULL,
                    duration_ms REAL DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    workflow_generated INTEGER DEFAULT 0,
                    error TEXT
                )
            """)
            
            # Indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_type 
                ON queries(analysis_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_timestamp 
                ON queries(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_organism 
                ON queries(organism)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_session 
                ON queries(session_id)
            """)
            
            # Daily aggregates for trends
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_aggregates (
                    date TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    total_queries INTEGER DEFAULT 0,
                    successful_queries INTEGER DEFAULT 0,
                    avg_duration_ms REAL DEFAULT 0,
                    PRIMARY KEY (date, analysis_type)
                )
            """)
            
            conn.commit()
    
    def record_query(self, record: QueryRecord):
        """
        Record a user query.
        
        Args:
            record: QueryRecord with query details
        """
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO queries 
                    (query, analysis_type, organism, genome_build, success, 
                     duration_ms, timestamp, session_id, workflow_generated, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.query,
                    record.analysis_type,
                    record.organism,
                    record.genome_build,
                    record.success,
                    record.duration_ms,
                    record.timestamp.isoformat(),
                    record.session_id,
                    record.workflow_generated,
                    record.error,
                ))
                
                # Update daily aggregate
                date = record.timestamp.strftime("%Y-%m-%d")
                conn.execute("""
                    INSERT INTO daily_aggregates 
                    (date, analysis_type, total_queries, successful_queries, avg_duration_ms)
                    VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT(date, analysis_type) DO UPDATE SET
                        total_queries = total_queries + 1,
                        successful_queries = successful_queries + ?,
                        avg_duration_ms = (avg_duration_ms * total_queries + ?) / (total_queries + 1)
                """, (
                    date, record.analysis_type, 
                    1 if record.success else 0, record.duration_ms,
                    1 if record.success else 0, record.duration_ms,
                ))
                
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record query: {e}")
    
    def record_simple(self, query: str, analysis_type: str,
                      organism: Optional[str] = None,
                      success: bool = True, duration_ms: float = 0):
        """
        Simple query recording method.
        
        Args:
            query: User query text
            analysis_type: Type of analysis
            organism: Organism if detected
            success: Whether query was successful
            duration_ms: Processing duration
        """
        record = QueryRecord(
            query=query,
            analysis_type=analysis_type,
            organism=organism,
            success=success,
            duration_ms=duration_ms,
        )
        self.record_query(record)
    
    def get_analysis_distribution(self, days: int = 30) -> Dict[str, int]:
        """
        Get distribution of analysis types.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict mapping analysis type to count
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT analysis_type, COUNT(*) as count
                FROM queries
                WHERE timestamp > ?
                GROUP BY analysis_type
                ORDER BY count DESC
            """, (cutoff.isoformat(),)).fetchall()
            
            return {row["analysis_type"]: row["count"] for row in rows}
    
    def get_organism_distribution(self, days: int = 30) -> Dict[str, int]:
        """
        Get distribution of organisms.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict mapping organism to count
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT organism, COUNT(*) as count
                FROM queries
                WHERE timestamp > ? AND organism IS NOT NULL
                GROUP BY organism
                ORDER BY count DESC
            """, (cutoff.isoformat(),)).fetchall()
            
            return {row["organism"]: row["count"] for row in rows}
    
    def get_success_by_type(self, days: int = 30) -> Dict[str, float]:
        """
        Get success rate by analysis type.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict mapping analysis type to success rate (0-100)
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT analysis_type, 
                       AVG(CASE WHEN success THEN 100.0 ELSE 0.0 END) as rate
                FROM queries
                WHERE timestamp > ?
                GROUP BY analysis_type
            """, (cutoff.isoformat(),)).fetchall()
            
            return {row["analysis_type"]: round(row["rate"], 1) for row in rows}
    
    def get_daily_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily query trends.
        
        Args:
            days: Number of days
            
        Returns:
            List of daily statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT date,
                       SUM(total_queries) as total,
                       SUM(successful_queries) as successful,
                       AVG(avg_duration_ms) as avg_duration
                FROM daily_aggregates
                WHERE date > ?
                GROUP BY date
                ORDER BY date
            """, (cutoff.strftime("%Y-%m-%d"),)).fetchall()
            
            return [
                {
                    "date": row["date"],
                    "total_queries": row["total"],
                    "success_rate": round(row["successful"] / row["total"] * 100, 1) if row["total"] > 0 else 0,
                    "avg_duration_ms": round(row["avg_duration"], 2),
                }
                for row in rows
            ]
    
    def get_popular_queries(self, limit: int = 10, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get most common query patterns.
        
        Args:
            limit: Maximum number of results
            days: Number of days to analyze
            
        Returns:
            List of popular query patterns
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            # Group by analysis type and organism
            rows = conn.execute("""
                SELECT analysis_type, organism, COUNT(*) as count,
                       AVG(CASE WHEN success THEN 100.0 ELSE 0.0 END) as success_rate
                FROM queries
                WHERE timestamp > ?
                GROUP BY analysis_type, organism
                ORDER BY count DESC
                LIMIT ?
            """, (cutoff.isoformat(), limit)).fetchall()
            
            return [
                {
                    "analysis_type": row["analysis_type"],
                    "organism": row["organism"],
                    "count": row["count"],
                    "success_rate": round(row["success_rate"], 1),
                }
                for row in rows
            ]
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session statistics
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                       SUM(CASE WHEN workflow_generated THEN 1 ELSE 0 END) as workflows,
                       AVG(duration_ms) as avg_duration,
                       MIN(timestamp) as first_query,
                       MAX(timestamp) as last_query
                FROM queries
                WHERE session_id = ?
            """, (session_id,)).fetchone()
            
            analysis_types = conn.execute("""
                SELECT analysis_type, COUNT(*) as count
                FROM queries
                WHERE session_id = ?
                GROUP BY analysis_type
            """, (session_id,)).fetchall()
            
            return {
                "session_id": session_id,
                "total_queries": row["total"],
                "successful_queries": row["successful"],
                "workflows_generated": row["workflows"],
                "avg_duration_ms": round(row["avg_duration"] or 0, 2),
                "first_query": row["first_query"],
                "last_query": row["last_query"],
                "analysis_types": {r["analysis_type"]: r["count"] for r in analysis_types},
            }
    
    def get_error_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze errors and failures.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Error analysis data
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            # Get failure rate by analysis type
            failure_by_type = conn.execute("""
                SELECT analysis_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed
                FROM queries
                WHERE timestamp > ?
                GROUP BY analysis_type
                HAVING failed > 0
                ORDER BY failed DESC
            """, (cutoff.isoformat(),)).fetchall()
            
            # Get common errors
            common_errors = conn.execute("""
                SELECT error, COUNT(*) as count
                FROM queries
                WHERE timestamp > ? AND error IS NOT NULL
                GROUP BY error
                ORDER BY count DESC
                LIMIT 10
            """, (cutoff.isoformat(),)).fetchall()
            
            return {
                "failure_by_type": [
                    {
                        "analysis_type": r["analysis_type"],
                        "total": r["total"],
                        "failed": r["failed"],
                        "failure_rate": round(r["failed"] / r["total"] * 100, 1),
                    }
                    for r in failure_by_type
                ],
                "common_errors": [
                    {"error": r["error"], "count": r["count"]}
                    for r in common_errors
                ],
            }
    
    def get_performance_insights(self, days: int = 30) -> Dict[str, Any]:
        """
        Get performance insights.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Performance insights
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            # Slowest analysis types
            slow_types = conn.execute("""
                SELECT analysis_type, AVG(duration_ms) as avg_duration
                FROM queries
                WHERE timestamp > ? AND success = 1
                GROUP BY analysis_type
                ORDER BY avg_duration DESC
                LIMIT 5
            """, (cutoff.isoformat(),)).fetchall()
            
            # Overall stats
            overall = conn.execute("""
                SELECT COUNT(*) as total,
                       AVG(duration_ms) as avg_duration,
                       MIN(duration_ms) as min_duration,
                       MAX(duration_ms) as max_duration
                FROM queries
                WHERE timestamp > ? AND success = 1
            """, (cutoff.isoformat(),)).fetchone()
            
            return {
                "slowest_analysis_types": [
                    {
                        "analysis_type": r["analysis_type"],
                        "avg_duration_ms": round(r["avg_duration"], 2),
                    }
                    for r in slow_types
                ],
                "overall": {
                    "total_queries": overall["total"],
                    "avg_duration_ms": round(overall["avg_duration"] or 0, 2),
                    "min_duration_ms": round(overall["min_duration"] or 0, 2),
                    "max_duration_ms": round(overall["max_duration"] or 0, 2),
                },
            }
    
    def get_dashboard_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get summary data for dashboard.
        
        Args:
            days: Number of days to summarize
            
        Returns:
            Dashboard summary data
        """
        return {
            "analysis_distribution": self.get_analysis_distribution(days),
            "organism_distribution": self.get_organism_distribution(days),
            "success_by_type": self.get_success_by_type(days),
            "popular_queries": self.get_popular_queries(5, days),
            "daily_trends": self.get_daily_trends(days),
        }
    
    def cleanup_old_data(self, days: int = 90):
        """
        Remove data older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            conn.execute("DELETE FROM queries WHERE timestamp < ?",
                        (cutoff.isoformat(),))
            conn.execute("DELETE FROM daily_aggregates WHERE date < ?",
                        (cutoff.strftime("%Y-%m-%d"),))
            conn.commit()
        
        logger.info(f"Cleaned up analytics older than {days} days")
