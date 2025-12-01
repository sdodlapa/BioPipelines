"""
Provider Metrics
================

Collect and aggregate LLM provider performance metrics.

Features:
- Per-provider request tracking
- Latency and token usage
- Success/failure rates
- Error categorization
- SQLite persistence
"""

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


@dataclass
class ProviderMetrics:
    """Metrics for a single provider."""
    provider_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    last_request_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    @property
    def tokens_per_request(self) -> float:
        """Average tokens per successful request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_tokens / self.successful_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider_id": self.provider_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{self.success_rate:.1f}%",
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "tokens_per_request": round(self.tokens_per_request, 2),
            "top_errors": dict(sorted(
                self.errors_by_type.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]),
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
        }


@dataclass
class RequestRecord:
    """Record of a single request."""
    provider_id: str
    timestamp: datetime
    success: bool
    latency_ms: float
    tokens: int = 0
    error: Optional[str] = None
    model: Optional[str] = None


class MetricsCollector:
    """
    Collect and aggregate system metrics.
    
    Thread-safe metrics collection with SQLite persistence.
    """
    
    def __init__(self, db_path: str = "~/.biopipelines/metrics.db"):
        """
        Initialize metrics collector.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._provider_metrics: Dict[str, ProviderMetrics] = {}
        self._lock = threading.Lock()
        self._init_db()
        self._load_historical()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    tokens INTEGER DEFAULT 0,
                    error TEXT,
                    model TEXT
                )
            """)
            
            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_requests_provider 
                ON requests(provider_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_requests_timestamp 
                ON requests(timestamp)
            """)
            
            # Hourly aggregates table for dashboards
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hourly_aggregates (
                    provider_id TEXT NOT NULL,
                    hour TEXT NOT NULL,
                    total_requests INTEGER DEFAULT 0,
                    successful_requests INTEGER DEFAULT 0,
                    total_latency_ms REAL DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    PRIMARY KEY (provider_id, hour)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            yield conn
        finally:
            conn.close()
    
    def _load_historical(self, days: int = 7):
        """Load historical metrics from database."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT provider_id, 
                       COUNT(*) as total,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                       SUM(CASE WHEN success THEN latency_ms ELSE 0 END) as total_latency,
                       SUM(tokens) as total_tokens
                FROM requests
                WHERE timestamp > ?
                GROUP BY provider_id
            """, (cutoff.isoformat(),)).fetchall()
            
            for row in rows:
                provider_id, total, successful, total_latency, total_tokens = row
                self._provider_metrics[provider_id] = ProviderMetrics(
                    provider_id=provider_id,
                    total_requests=total,
                    successful_requests=successful or 0,
                    failed_requests=total - (successful or 0),
                    total_latency_ms=total_latency or 0,
                    total_tokens=total_tokens or 0,
                )
            
            # Load error types
            for provider_id in self._provider_metrics:
                errors = conn.execute("""
                    SELECT error, COUNT(*) as count
                    FROM requests
                    WHERE provider_id = ? AND error IS NOT NULL
                    AND timestamp > ?
                    GROUP BY error
                """, (provider_id, cutoff.isoformat())).fetchall()
                
                self._provider_metrics[provider_id].errors_by_type = dict(errors)
    
    def record_request(self, provider_id: str, success: bool,
                       latency_ms: float, tokens: int = 0,
                       error: Optional[str] = None, model: Optional[str] = None):
        """
        Record a provider request.
        
        Args:
            provider_id: Provider identifier
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            tokens: Number of tokens used
            error: Error type if failed
            model: Model used for request
        """
        timestamp = datetime.now()
        
        with self._lock:
            # Update in-memory metrics
            if provider_id not in self._provider_metrics:
                self._provider_metrics[provider_id] = ProviderMetrics(provider_id)
            
            m = self._provider_metrics[provider_id]
            m.total_requests += 1
            m.last_request_time = timestamp
            
            if success:
                m.successful_requests += 1
                m.total_latency_ms += latency_ms
                m.total_tokens += tokens
            else:
                m.failed_requests += 1
                if error:
                    error_type = self._categorize_error(error)
                    m.errors_by_type[error_type] = m.errors_by_type.get(error_type, 0) + 1
        
        # Persist to database (outside lock)
        self._save_request(provider_id, timestamp, success, latency_ms, tokens, error, model)
    
    def _categorize_error(self, error: str) -> str:
        """Categorize error into type."""
        error_lower = error.lower()
        
        if "rate limit" in error_lower or "429" in error_lower:
            return "rate_limit"
        elif "timeout" in error_lower:
            return "timeout"
        elif "connection" in error_lower:
            return "connection"
        elif "auth" in error_lower or "401" in error_lower or "403" in error_lower:
            return "auth"
        elif "model" in error_lower or "not found" in error_lower:
            return "model_not_found"
        elif "context" in error_lower or "token" in error_lower:
            return "context_length"
        else:
            return "other"
    
    def _save_request(self, provider_id: str, timestamp: datetime,
                      success: bool, latency_ms: float, tokens: int,
                      error: Optional[str], model: Optional[str]):
        """Save request to database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO requests 
                    (provider_id, timestamp, success, latency_ms, tokens, error, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (provider_id, timestamp.isoformat(), success, 
                      latency_ms, tokens, error, model))
                
                # Update hourly aggregate
                hour = timestamp.strftime("%Y-%m-%d %H:00:00")
                conn.execute("""
                    INSERT INTO hourly_aggregates 
                    (provider_id, hour, total_requests, successful_requests, 
                     total_latency_ms, total_tokens)
                    VALUES (?, ?, 1, ?, ?, ?)
                    ON CONFLICT(provider_id, hour) DO UPDATE SET
                        total_requests = total_requests + 1,
                        successful_requests = successful_requests + ?,
                        total_latency_ms = total_latency_ms + ?,
                        total_tokens = total_tokens + ?
                """, (provider_id, hour, 
                      1 if success else 0, latency_ms if success else 0, tokens,
                      1 if success else 0, latency_ms if success else 0, tokens))
                
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
    
    def get_provider_metrics(self, provider_id: str) -> Optional[ProviderMetrics]:
        """Get metrics for a specific provider."""
        return self._provider_metrics.get(provider_id)
    
    def get_all_metrics(self) -> Dict[str, ProviderMetrics]:
        """Get all provider metrics."""
        return self._provider_metrics.copy()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display."""
        providers = {}
        total_requests = 0
        total_successful = 0
        
        for pid, m in self._provider_metrics.items():
            providers[pid] = {
                "requests": m.total_requests,
                "success_rate": f"{m.success_rate:.1f}%",
                "avg_latency": f"{m.avg_latency_ms:.0f}ms",
                "tokens": m.total_tokens,
                "last_request": m.last_request_time.isoformat() if m.last_request_time else None,
            }
            total_requests += m.total_requests
            total_successful += m.successful_requests
        
        overall_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "providers": providers,
            "total_requests": total_requests,
            "overall_success_rate": f"{overall_rate:.1f}%",
            "active_providers": len(providers),
        }
    
    def get_hourly_stats(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get hourly statistics for graphs."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT hour, 
                       SUM(total_requests) as requests,
                       SUM(successful_requests) as successful,
                       AVG(CASE WHEN successful_requests > 0 
                           THEN total_latency_ms / successful_requests 
                           ELSE 0 END) as avg_latency
                FROM hourly_aggregates
                WHERE hour > ?
                GROUP BY hour
                ORDER BY hour
            """, (cutoff.isoformat(),)).fetchall()
            
            return [
                {
                    "hour": row[0],
                    "requests": row[1],
                    "success_rate": (row[2] / row[1] * 100) if row[1] > 0 else 0,
                    "avg_latency_ms": row[3] or 0,
                }
                for row in rows
            ]
    
    def get_provider_ranking(self) -> List[Dict[str, Any]]:
        """Rank providers by performance."""
        rankings = []
        
        for pid, m in self._provider_metrics.items():
            if m.total_requests > 0:
                # Score based on success rate and latency
                score = (m.success_rate * 0.7) + (max(0, 100 - m.avg_latency_ms / 10) * 0.3)
                rankings.append({
                    "provider_id": pid,
                    "score": round(score, 2),
                    "requests": m.total_requests,
                    "success_rate": m.success_rate,
                    "avg_latency_ms": round(m.avg_latency_ms, 2),
                })
        
        return sorted(rankings, key=lambda x: x["score"], reverse=True)
    
    def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            conn.execute("DELETE FROM requests WHERE timestamp < ?", 
                        (cutoff.isoformat(),))
            conn.execute("DELETE FROM hourly_aggregates WHERE hour < ?",
                        (cutoff.isoformat(),))
            conn.commit()
        
        logger.info(f"Cleaned up metrics older than {days} days")
    
    @contextmanager
    def track_request(self, provider_id: str, model: Optional[str] = None):
        """
        Context manager to track a request.
        
        Usage:
            with metrics.track_request("gemini") as tracker:
                response = await provider.generate(prompt)
                tracker.tokens = response.usage.total_tokens
        """
        tracker = RequestTracker(self, provider_id, model)
        try:
            yield tracker
            tracker._success = True
        except Exception as e:
            tracker._error = str(e)
            raise
        finally:
            tracker._complete()


class RequestTracker:
    """Helper for tracking individual requests."""
    
    def __init__(self, collector: MetricsCollector, 
                 provider_id: str, model: Optional[str] = None):
        self.collector = collector
        self.provider_id = provider_id
        self.model = model
        self.tokens = 0
        self._start_time = time.time()
        self._success = False
        self._error: Optional[str] = None
    
    def _complete(self):
        """Complete the request tracking."""
        latency_ms = (time.time() - self._start_time) * 1000
        self.collector.record_request(
            provider_id=self.provider_id,
            success=self._success,
            latency_ms=latency_ms,
            tokens=self.tokens,
            error=self._error,
            model=self.model,
        )
