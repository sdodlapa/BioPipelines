"""
Tests for Phase 2.5: Observability & Analytics
==============================================

Tests for:
- MetricsCollector provider metrics
- QueryAnalytics query tracking
- Health endpoints
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

from src.workflow_composer.observability import (
    ProviderMetrics,
    MetricsCollector,
    RequestTracker,
    QueryAnalytics,
    QueryRecord,
    health_check,
    system_health,
    get_prometheus_metrics,
    set_collectors,
)


class TestProviderMetrics:
    """Tests for ProviderMetrics dataclass."""
    
    def test_initial_values(self):
        """Test initial metric values."""
        metrics = ProviderMetrics(provider_id="gemini")
        
        assert metrics.provider_id == "gemini"
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 0.0
        assert metrics.avg_latency_ms == 0.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ProviderMetrics(
            provider_id="gemini",
            total_requests=100,
            successful_requests=85,
            failed_requests=15,
        )
        
        assert metrics.success_rate == 85.0
    
    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = ProviderMetrics(
            provider_id="gemini",
            successful_requests=10,
            total_latency_ms=1500.0,
        )
        
        assert metrics.avg_latency_ms == 150.0
    
    def test_tokens_per_request(self):
        """Test tokens per request calculation."""
        metrics = ProviderMetrics(
            provider_id="gemini",
            successful_requests=5,
            total_tokens=2500,
        )
        
        assert metrics.tokens_per_request == 500.0
    
    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = ProviderMetrics(
            provider_id="gemini",
            total_requests=100,
            successful_requests=90,
            total_tokens=5000,
            total_latency_ms=9000.0,
        )
        
        data = metrics.to_dict()
        
        assert data["provider_id"] == "gemini"
        assert data["total_requests"] == 100
        assert data["success_rate"] == "90.0%"
        assert data["avg_latency_ms"] == 100.0


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    @pytest.fixture
    def collector(self, tmp_path):
        """Create a metrics collector with temp database."""
        db_path = tmp_path / "test_metrics.db"
        return MetricsCollector(db_path=str(db_path))
    
    def test_record_successful_request(self, collector):
        """Test recording a successful request."""
        collector.record_request(
            provider_id="gemini",
            success=True,
            latency_ms=150.0,
            tokens=500,
        )
        
        metrics = collector.get_provider_metrics("gemini")
        
        assert metrics is not None
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.total_tokens == 500
    
    def test_record_failed_request(self, collector):
        """Test recording a failed request."""
        collector.record_request(
            provider_id="gemini",
            success=False,
            latency_ms=5000.0,
            error="rate limit exceeded",
        )
        
        metrics = collector.get_provider_metrics("gemini")
        
        assert metrics.failed_requests == 1
        assert "rate_limit" in metrics.errors_by_type
    
    def test_multiple_providers(self, collector):
        """Test tracking multiple providers."""
        collector.record_request("gemini", True, 100.0, tokens=100)
        collector.record_request("cerebras", True, 50.0, tokens=150)
        collector.record_request("groq", True, 75.0, tokens=200)
        
        all_metrics = collector.get_all_metrics()
        
        assert len(all_metrics) == 3
        assert "gemini" in all_metrics
        assert "cerebras" in all_metrics
        assert "groq" in all_metrics
    
    def test_dashboard_data(self, collector):
        """Test dashboard data format."""
        collector.record_request("gemini", True, 100.0, tokens=500)
        collector.record_request("gemini", True, 150.0, tokens=600)
        collector.record_request("gemini", False, 5000.0, error="timeout")
        
        data = collector.get_dashboard_data()
        
        assert "providers" in data
        assert "total_requests" in data
        assert data["total_requests"] == 3
        assert "gemini" in data["providers"]
    
    def test_error_categorization(self, collector):
        """Test error type categorization."""
        collector.record_request("test", False, 0, error="rate limit exceeded")
        collector.record_request("test", False, 0, error="connection timeout")
        collector.record_request("test", False, 0, error="401 unauthorized")
        
        metrics = collector.get_provider_metrics("test")
        
        assert "rate_limit" in metrics.errors_by_type
        assert "timeout" in metrics.errors_by_type
        assert "auth" in metrics.errors_by_type
    
    def test_request_tracker_context_manager(self, collector):
        """Test RequestTracker context manager."""
        with collector.track_request("gemini") as tracker:
            time.sleep(0.01)  # Simulate work
            tracker.tokens = 100
        
        metrics = collector.get_provider_metrics("gemini")
        
        assert metrics.successful_requests == 1
        assert metrics.total_tokens == 100
        assert metrics.avg_latency_ms >= 10  # At least 10ms
    
    def test_request_tracker_on_failure(self, collector):
        """Test RequestTracker handles failures."""
        try:
            with collector.track_request("gemini") as tracker:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        metrics = collector.get_provider_metrics("gemini")
        
        assert metrics.failed_requests == 1
    
    def test_provider_ranking(self, collector):
        """Test provider ranking by performance."""
        # Gemini: good success, medium latency
        for _ in range(10):
            collector.record_request("gemini", True, 100.0)
        
        # Cerebras: perfect success, fast
        for _ in range(10):
            collector.record_request("cerebras", True, 50.0)
        
        # Groq: some failures
        for _ in range(8):
            collector.record_request("groq", True, 75.0)
        collector.record_request("groq", False, 0, error="error")
        collector.record_request("groq", False, 0, error="error")
        
        ranking = collector.get_provider_ranking()
        
        # Cerebras should rank highest (best success + latency)
        assert ranking[0]["provider_id"] == "cerebras"


class TestQueryAnalytics:
    """Tests for QueryAnalytics."""
    
    @pytest.fixture
    def analytics(self, tmp_path):
        """Create query analytics with temp database."""
        db_path = tmp_path / "test_analytics.db"
        return QueryAnalytics(db_path=str(db_path))
    
    def test_record_query(self, analytics):
        """Test recording a query."""
        record = QueryRecord(
            query="RNA-seq analysis for human",
            analysis_type="rna-seq",
            organism="human",
            success=True,
            duration_ms=1500.0,
        )
        
        analytics.record_query(record)
        
        dist = analytics.get_analysis_distribution(1)
        assert "rna-seq" in dist
        assert dist["rna-seq"] == 1
    
    def test_record_simple(self, analytics):
        """Test simple query recording."""
        analytics.record_simple(
            query="ChIP-seq peak calling",
            analysis_type="chip-seq",
            organism="mouse",
            success=True,
            duration_ms=2000.0,
        )
        
        dist = analytics.get_analysis_distribution(1)
        assert "chip-seq" in dist
    
    def test_analysis_distribution(self, analytics):
        """Test analysis type distribution."""
        analytics.record_simple("test1", "rna-seq", success=True)
        analytics.record_simple("test2", "rna-seq", success=True)
        analytics.record_simple("test3", "chip-seq", success=True)
        analytics.record_simple("test4", "dna-seq", success=True)
        
        dist = analytics.get_analysis_distribution(1)
        
        assert dist["rna-seq"] == 2
        assert dist["chip-seq"] == 1
        assert dist["dna-seq"] == 1
    
    def test_organism_distribution(self, analytics):
        """Test organism distribution."""
        analytics.record_simple("test1", "rna-seq", organism="human")
        analytics.record_simple("test2", "rna-seq", organism="human")
        analytics.record_simple("test3", "rna-seq", organism="mouse")
        
        dist = analytics.get_organism_distribution(1)
        
        assert dist["human"] == 2
        assert dist["mouse"] == 1
    
    def test_success_by_type(self, analytics):
        """Test success rate by analysis type."""
        # RNA-seq: 80% success
        for i in range(8):
            analytics.record_simple(f"test{i}", "rna-seq", success=True)
        for i in range(2):
            analytics.record_simple(f"fail{i}", "rna-seq", success=False)
        
        # ChIP-seq: 100% success
        for i in range(5):
            analytics.record_simple(f"chip{i}", "chip-seq", success=True)
        
        rates = analytics.get_success_by_type(1)
        
        assert rates["rna-seq"] == 80.0
        assert rates["chip-seq"] == 100.0
    
    def test_popular_queries(self, analytics):
        """Test popular query patterns."""
        # Most popular
        for _ in range(10):
            analytics.record_simple("test", "rna-seq", organism="human")
        
        # Second popular
        for _ in range(5):
            analytics.record_simple("test", "chip-seq", organism="human")
        
        popular = analytics.get_popular_queries(5, 1)
        
        assert len(popular) >= 2
        assert popular[0]["count"] == 10
        assert popular[0]["analysis_type"] == "rna-seq"
    
    def test_session_stats(self, analytics):
        """Test session statistics."""
        session_id = "test-session-123"
        
        for i in range(5):
            record = QueryRecord(
                query=f"query {i}",
                analysis_type="rna-seq",
                success=True,
                duration_ms=100.0 * i,
                session_id=session_id,
            )
            analytics.record_query(record)
        
        stats = analytics.get_session_stats(session_id)
        
        assert stats["session_id"] == session_id
        assert stats["total_queries"] == 5
        assert stats["successful_queries"] == 5
    
    def test_error_analysis(self, analytics):
        """Test error analysis."""
        analytics.record_simple("test1", "rna-seq", success=False)
        
        record = QueryRecord(
            query="test",
            analysis_type="chip-seq",
            success=False,
            error="validation failed",
        )
        analytics.record_query(record)
        
        errors = analytics.get_error_analysis(1)
        
        assert len(errors["failure_by_type"]) > 0
    
    def test_dashboard_summary(self, analytics):
        """Test dashboard summary data."""
        analytics.record_simple("test1", "rna-seq", organism="human")
        analytics.record_simple("test2", "chip-seq", organism="mouse")
        
        summary = analytics.get_dashboard_summary(7)
        
        assert "analysis_distribution" in summary
        assert "organism_distribution" in summary
        assert "success_by_type" in summary
        assert "popular_queries" in summary


class TestHealthEndpoints:
    """Tests for health endpoints."""
    
    def test_basic_health_check(self):
        """Test basic health check."""
        result = health_check()
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert result["version"] == "2.0.0"
    
    def test_system_health(self):
        """Test system health check."""
        result = system_health()
        
        assert result["status"] in ["healthy", "degraded"]
        assert "resources" in result
        assert "cpu_percent" in result["resources"]
        assert "memory" in result["resources"]
        assert "disk" in result["resources"]
    
    def test_system_health_warnings(self):
        """Test system health warnings generation."""
        result = system_health()
        
        # Warnings should be a list
        assert isinstance(result.get("warnings", []), list)
    
    def test_prometheus_metrics_format(self, tmp_path):
        """Test Prometheus metrics format."""
        # Set up collectors
        db_path = tmp_path / "test.db"
        collector = MetricsCollector(db_path=str(db_path))
        analytics = QueryAnalytics(db_path=str(db_path))
        
        # Record some data
        collector.record_request("gemini", True, 100.0, tokens=500)
        analytics.record_simple("test", "rna-seq")
        
        set_collectors(metrics=collector, analytics=analytics)
        
        metrics_str = get_prometheus_metrics()
        
        # Should have HELP and TYPE comments
        assert "# HELP" in metrics_str
        assert "# TYPE" in metrics_str
        
        # Should have provider metrics
        assert "biopipelines_provider_requests" in metrics_str
        
        # Clean up
        set_collectors(None, None)


class TestIntegration:
    """Integration tests for observability."""
    
    def test_full_request_flow(self, tmp_path):
        """Test complete request tracking flow."""
        # Set up collectors
        metrics_db = tmp_path / "metrics.db"
        analytics_db = tmp_path / "analytics.db"
        
        collector = MetricsCollector(db_path=str(metrics_db))
        analytics = QueryAnalytics(db_path=str(analytics_db))
        
        # Simulate a workflow request
        with collector.track_request("gemini", model="gemini-2.0-flash") as tracker:
            # Simulate processing
            time.sleep(0.01)
            tracker.tokens = 1500
        
        # Record the query
        analytics.record_simple(
            query="RNA-seq differential expression",
            analysis_type="rna-seq",
            organism="human",
            success=True,
            duration_ms=150.0,
        )
        
        # Verify metrics
        metrics = collector.get_provider_metrics("gemini")
        assert metrics.successful_requests == 1
        assert metrics.total_tokens == 1500
        
        # Verify analytics
        dist = analytics.get_analysis_distribution(1)
        assert dist["rna-seq"] == 1
    
    def test_multiple_sessions(self, tmp_path):
        """Test tracking across sessions."""
        analytics = QueryAnalytics(db_path=str(tmp_path / "analytics.db"))
        
        # Session 1
        for i in range(3):
            record = QueryRecord(
                query=f"session1-query{i}",
                analysis_type="rna-seq",
                session_id="session-1",
            )
            analytics.record_query(record)
        
        # Session 2
        for i in range(5):
            record = QueryRecord(
                query=f"session2-query{i}",
                analysis_type="chip-seq",
                session_id="session-2",
            )
            analytics.record_query(record)
        
        # Verify session stats
        stats1 = analytics.get_session_stats("session-1")
        stats2 = analytics.get_session_stats("session-2")
        
        assert stats1["total_queries"] == 3
        assert stats2["total_queries"] == 5
        
        # Verify overall distribution
        dist = analytics.get_analysis_distribution(1)
        assert dist["rna-seq"] == 3
        assert dist["chip-seq"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
