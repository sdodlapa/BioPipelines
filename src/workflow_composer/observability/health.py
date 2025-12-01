"""
Health Endpoints
================

FastAPI health and monitoring endpoints.

Endpoints:
- /health - Basic health check
- /health/providers - LLM provider health
- /health/system - System resources
- /metrics - Prometheus-style metrics
- /dashboard - Dashboard data
"""

import logging
import psutil
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Will be set by the main app
_metrics_collector = None
_query_analytics = None


def set_collectors(metrics=None, analytics=None):
    """Set the collector instances for endpoints."""
    global _metrics_collector, _query_analytics
    _metrics_collector = metrics
    _query_analytics = analytics


def health_check() -> Dict[str, Any]:
    """
    Basic health check.
    
    Returns:
        Health status dict
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "service": "biopipelines-workflow-composer",
    }


def system_health() -> Dict[str, Any]:
    """
    System resource health check.
    
    Returns:
        System health metrics
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "resources": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": disk.percent,
                },
            },
            "warnings": _get_resource_warnings(cpu_percent, memory.percent, disk.percent),
        }
    except Exception as e:
        logger.warning(f"Failed to get system health: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def _get_resource_warnings(cpu: float, memory: float, disk: float) -> list:
    """Get warnings for resource usage."""
    warnings = []
    
    if cpu > 90:
        warnings.append(f"High CPU usage: {cpu}%")
    elif cpu > 75:
        warnings.append(f"Elevated CPU usage: {cpu}%")
    
    if memory > 90:
        warnings.append(f"High memory usage: {memory}%")
    elif memory > 80:
        warnings.append(f"Elevated memory usage: {memory}%")
    
    if disk > 90:
        warnings.append(f"Low disk space: {100-disk}% free")
    elif disk > 80:
        warnings.append(f"Disk usage warning: {100-disk}% free")
    
    return warnings


async def provider_health() -> Dict[str, Any]:
    """
    Check health of all LLM providers.
    
    Returns:
        Provider health status
    """
    try:
        from ..providers import get_registry
    except ImportError:
        return {
            "status": "unavailable",
            "error": "Provider registry not available",
        }
    
    health = {}
    overall_healthy = True
    
    try:
        registry = get_registry()
        providers = registry.list_providers()
        
        for provider_config in providers:
            try:
                provider = registry.get_provider(provider_config.id)
                start = datetime.now()
                
                # Simple health check with minimal tokens
                await provider.generate("Hello", max_tokens=5)
                
                latency = (datetime.now() - start).total_seconds() * 1000
                health[provider_config.id] = {
                    "status": "healthy",
                    "latency_ms": round(latency, 2),
                }
            except Exception as e:
                health[provider_config.id] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                overall_healthy = False
    except Exception as e:
        logger.warning(f"Provider health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "providers": health,
        "healthy_count": sum(1 for p in health.values() if p["status"] == "healthy"),
        "total_count": len(health),
    }


def get_metrics() -> Dict[str, Any]:
    """
    Get system metrics.
    
    Returns:
        Metrics data
    """
    if _metrics_collector is None:
        return {"error": "Metrics collector not configured"}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "providers": _metrics_collector.get_dashboard_data(),
        "hourly": _metrics_collector.get_hourly_stats(24),
        "ranking": _metrics_collector.get_provider_ranking(),
    }


def get_analytics() -> Dict[str, Any]:
    """
    Get query analytics.
    
    Returns:
        Analytics data
    """
    if _query_analytics is None:
        return {"error": "Query analytics not configured"}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "dashboard": _query_analytics.get_dashboard_summary(7),
        "performance": _query_analytics.get_performance_insights(30),
        "errors": _query_analytics.get_error_analysis(30),
    }


def get_prometheus_metrics() -> str:
    """
    Get metrics in Prometheus format.
    
    Returns:
        Prometheus-formatted metrics string
    """
    lines = []
    
    # System metrics
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        lines.extend([
            "# HELP biopipelines_cpu_usage CPU usage percentage",
            "# TYPE biopipelines_cpu_usage gauge",
            f"biopipelines_cpu_usage {cpu}",
            "",
            "# HELP biopipelines_memory_usage Memory usage percentage",
            "# TYPE biopipelines_memory_usage gauge",
            f"biopipelines_memory_usage {memory.percent}",
            "",
        ])
    except Exception:
        pass
    
    # Provider metrics
    if _metrics_collector:
        lines.extend([
            "# HELP biopipelines_provider_requests Total requests per provider",
            "# TYPE biopipelines_provider_requests counter",
        ])
        
        for pid, m in _metrics_collector.get_all_metrics().items():
            lines.append(f'biopipelines_provider_requests{{provider="{pid}"}} {m.total_requests}')
        
        lines.extend([
            "",
            "# HELP biopipelines_provider_success_rate Success rate per provider",
            "# TYPE biopipelines_provider_success_rate gauge",
        ])
        
        for pid, m in _metrics_collector.get_all_metrics().items():
            lines.append(f'biopipelines_provider_success_rate{{provider="{pid}"}} {m.success_rate}')
        
        lines.extend([
            "",
            "# HELP biopipelines_provider_latency_avg Average latency per provider",
            "# TYPE biopipelines_provider_latency_avg gauge",
        ])
        
        for pid, m in _metrics_collector.get_all_metrics().items():
            lines.append(f'biopipelines_provider_latency_avg{{provider="{pid}"}} {m.avg_latency_ms}')
    
    # Query analytics
    if _query_analytics:
        dist = _query_analytics.get_analysis_distribution(7)
        
        lines.extend([
            "",
            "# HELP biopipelines_queries_by_type Query count by analysis type",
            "# TYPE biopipelines_queries_by_type counter",
        ])
        
        for analysis_type, count in dist.items():
            lines.append(f'biopipelines_queries_by_type{{type="{analysis_type}"}} {count}')
    
    return "\n".join(lines)


# FastAPI router if using FastAPI
try:
    from fastapi import APIRouter, Response
    
    router = APIRouter(prefix="/health", tags=["health"])
    
    @router.get("")
    async def api_health_check():
        """Basic health check endpoint."""
        return health_check()
    
    @router.get("/system")
    async def api_system_health():
        """System resource health endpoint."""
        return system_health()
    
    @router.get("/providers")
    async def api_provider_health():
        """Provider health endpoint."""
        return await provider_health()
    
    metrics_router = APIRouter(prefix="/metrics", tags=["metrics"])
    
    @metrics_router.get("")
    async def api_get_metrics():
        """Get metrics endpoint."""
        return get_metrics()
    
    @metrics_router.get("/prometheus")
    async def api_prometheus_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=get_prometheus_metrics(),
            media_type="text/plain"
        )
    
    @metrics_router.get("/analytics")
    async def api_get_analytics():
        """Query analytics endpoint."""
        return get_analytics()
    
    @metrics_router.get("/dashboard")
    async def api_dashboard():
        """Dashboard data endpoint."""
        return {
            "health": health_check(),
            "system": system_health(),
            "metrics": get_metrics() if _metrics_collector else None,
            "analytics": get_analytics() if _query_analytics else None,
        }

except ImportError:
    # FastAPI not installed
    router = None
    metrics_router = None
