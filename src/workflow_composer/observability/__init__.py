"""
Observability Package
=====================

Metrics collection, analytics, and health monitoring for BioPipelines.

Components:
- ProviderMetrics: Per-provider performance metrics
- MetricsCollector: Aggregates and persists metrics
- QueryAnalytics: Tracks query patterns and success rates
- Health endpoints: System and provider health monitoring
"""

from .metrics import ProviderMetrics, MetricsCollector, RequestTracker
from .query_analytics import QueryAnalytics, QueryRecord
from .health import (
    health_check,
    system_health,
    provider_health,
    get_metrics,
    get_analytics,
    get_prometheus_metrics,
    set_collectors,
)

__all__ = [
    # Metrics
    "ProviderMetrics",
    "MetricsCollector",
    "RequestTracker",
    # Analytics
    "QueryAnalytics",
    "QueryRecord",
    # Health
    "health_check",
    "system_health",
    "provider_health",
    "get_metrics",
    "get_analytics",
    "get_prometheus_metrics",
    "set_collectors",
]
