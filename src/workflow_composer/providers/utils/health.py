"""
Health checking utilities for providers.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status for a provider."""
    provider_id: str
    name: str
    available: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    models_loaded: Optional[List[str]] = None
    free_tier: bool = False
    checked_at: datetime = None
    
    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "available": self.available,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "models_loaded": self.models_loaded,
            "free_tier": self.free_tier,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
        }


def check_provider(provider_id: str) -> HealthStatus:
    """
    Check health of a specific provider.
    
    Args:
        provider_id: Provider ID to check
        
    Returns:
        HealthStatus with check results
    """
    from ..registry import get_registry
    from ..router import ProviderRouter
    
    registry = get_registry()
    config = registry.get_provider_config(provider_id)
    
    if not config:
        return HealthStatus(
            provider_id=provider_id,
            name=provider_id,
            available=False,
            error=f"Unknown provider: {provider_id}",
        )
    
    if not config.is_configured():
        return HealthStatus(
            provider_id=provider_id,
            name=config.name,
            available=False,
            error="Not configured (missing API key)",
            free_tier=config.free_tier,
        )
    
    try:
        import time
        start = time.time()
        
        router = ProviderRouter()
        provider = router._create_provider(config)
        available = provider.is_available()
        latency = (time.time() - start) * 1000
        
        # Get additional info if available
        models = None
        if hasattr(provider, 'list_models'):
            try:
                models = provider.list_models()
            except Exception:
                pass
        
        return HealthStatus(
            provider_id=provider_id,
            name=config.name,
            available=available,
            latency_ms=latency,
            models_loaded=models,
            free_tier=config.free_tier,
        )
    except Exception as e:
        return HealthStatus(
            provider_id=provider_id,
            name=config.name,
            available=False,
            error=str(e),
            free_tier=config.free_tier,
        )


def check_all_providers() -> Dict[str, HealthStatus]:
    """
    Check health of all configured providers.
    
    Returns:
        Dictionary mapping provider IDs to HealthStatus
    """
    from ..registry import get_registry
    
    registry = get_registry()
    providers = registry.list_providers(configured_only=False, enabled_only=True)
    
    results = {}
    for config in providers:
        results[config.id] = check_provider(config.id)
    
    return results


def print_health_report():
    """Print a formatted health report for all providers."""
    statuses = check_all_providers()
    
    print("\n" + "=" * 60)
    print("PROVIDER HEALTH CHECK")
    print("=" * 60 + "\n")
    
    # Sort: available first, then by priority
    sorted_items = sorted(
        statuses.items(),
        key=lambda x: (not x[1].available, x[0])
    )
    
    for provider_id, status in sorted_items:
        emoji = "✅" if status.available else "❌"
        free = " [FREE]" if status.free_tier else ""
        latency = f" ({status.latency_ms:.0f}ms)" if status.latency_ms else ""
        
        print(f"{emoji} {status.name:20}{free}{latency}")
        
        if status.error:
            print(f"   Error: {status.error}")
        
        if status.models_loaded:
            models_str = ', '.join(status.models_loaded[:2])
            if len(status.models_loaded) > 2:
                models_str += f" +{len(status.models_loaded) - 2} more"
            print(f"   Models: {models_str}")
        
        print()
    
    print("=" * 60)
    available = sum(1 for s in statuses.values() if s.available)
    free_available = sum(
        1 for s in statuses.values() 
        if s.available and s.free_tier
    )
    print(f"Total: {available}/{len(statuses)} providers available")
    print(f"Free tiers available: {free_available}")
    print("=" * 60 + "\n")


async def check_provider_async(provider_id: str) -> HealthStatus:
    """Async version of check_provider."""
    return await asyncio.to_thread(check_provider, provider_id)


async def check_all_providers_async() -> Dict[str, HealthStatus]:
    """Async version of check_all_providers."""
    from ..registry import get_registry
    
    registry = get_registry()
    providers = registry.list_providers(configured_only=False, enabled_only=True)
    
    tasks = [check_provider_async(p.id) for p in providers]
    results = await asyncio.gather(*tasks)
    
    return {r.provider_id: r for r in results}
