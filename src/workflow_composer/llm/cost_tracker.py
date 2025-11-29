"""
Cost Tracker
============

Tracks costs, usage, and budgets for LLM operations.

Features:
- Per-request cost tracking
- Budget limits and alerts
- Usage statistics over time
- Provider-level cost breakdown

Usage:
    from workflow_composer.llm import CostTracker
    
    tracker = CostTracker(budget_limit=10.0)
    
    # Track a request
    tracker.track(response)
    
    # Check budget
    if tracker.is_over_budget:
        print("Warning: Over budget!")
    
    # Get summary
    print(tracker.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Cost Entry
# =============================================================================

@dataclass
class CostEntry:
    """A single cost entry."""
    timestamp: datetime
    provider: str
    model: str
    tokens_used: int
    cost: float
    latency_ms: float
    request_type: str = "complete"  # complete, ensemble, stream
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "request_type": self.request_type,
        }


# =============================================================================
# Cost Summary
# =============================================================================

@dataclass
class CostSummary:
    """Summary of costs for a period."""
    total_cost: float
    total_tokens: int
    total_requests: int
    avg_cost_per_request: float
    avg_tokens_per_request: float
    avg_latency_ms: float
    by_provider: Dict[str, float]
    by_model: Dict[str, float]
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    def __str__(self) -> str:
        lines = [
            f"Cost Summary",
            f"============",
            f"Total Cost: ${self.total_cost:.4f}",
            f"Total Tokens: {self.total_tokens:,}",
            f"Total Requests: {self.total_requests}",
            f"Avg Cost/Request: ${self.avg_cost_per_request:.4f}",
            f"Avg Tokens/Request: {self.avg_tokens_per_request:.0f}",
            f"Avg Latency: {self.avg_latency_ms:.0f}ms",
            "",
            "By Provider:",
        ]
        for provider, cost in sorted(self.by_provider.items(), key=lambda x: -x[1]):
            lines.append(f"  {provider}: ${cost:.4f}")
        
        lines.append("")
        lines.append("By Model:")
        for model, cost in sorted(self.by_model.items(), key=lambda x: -x[1]):
            lines.append(f"  {model}: ${cost:.4f}")
        
        return "\n".join(lines)


# =============================================================================
# Budget Alert
# =============================================================================

@dataclass
class BudgetAlert:
    """A budget alert event."""
    timestamp: datetime
    alert_type: str  # "warning", "exceeded", "critical"
    current_cost: float
    budget_limit: float
    message: str
    
    @property
    def percentage(self) -> float:
        if self.budget_limit <= 0:
            return 100.0
        return (self.current_cost / self.budget_limit) * 100


# =============================================================================
# Cost Tracker
# =============================================================================

class CostTracker:
    """
    Tracks costs and usage for LLM operations.
    
    Features:
    - Real-time cost tracking
    - Budget limits with warnings
    - Usage history and analytics
    - Provider/model breakdown
    
    Example:
        tracker = CostTracker(budget_limit=5.0)
        
        # Track usage
        for response in responses:
            tracker.track(response)
            
            if tracker.budget_warning:
                print(f"Warning: {tracker.budget_percentage:.0f}% of budget used")
        
        # Get summary
        print(tracker.summary())
        
        # Export history
        history = tracker.export_history()
    """
    
    def __init__(
        self,
        budget_limit: Optional[float] = None,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        history_retention_days: int = 30,
    ):
        """
        Initialize cost tracker.
        
        Args:
            budget_limit: Maximum budget in dollars (None = no limit)
            warning_threshold: Percentage (0-1) to trigger warning
            critical_threshold: Percentage (0-1) to trigger critical alert
            history_retention_days: Days to retain history
        """
        self.budget_limit = budget_limit
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.history_retention_days = history_retention_days
        
        # Usage tracking
        self._entries: List[CostEntry] = []
        self._alerts: List[BudgetAlert] = []
        
        # Aggregated stats
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._total_requests: int = 0
        self._total_latency: float = 0.0
        
        # Per-provider/model stats
        self._cost_by_provider: Dict[str, float] = defaultdict(float)
        self._cost_by_model: Dict[str, float] = defaultdict(float)
        self._tokens_by_provider: Dict[str, int] = defaultdict(int)
        
        # Alert state
        self._warning_sent = False
        self._critical_sent = False
        
        logger.debug(f"CostTracker initialized with budget=${budget_limit}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def total_cost(self) -> float:
        """Total cost incurred."""
        return self._total_cost
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self._total_tokens
    
    @property
    def total_requests(self) -> int:
        """Total number of requests."""
        return self._total_requests
    
    @property
    def budget_remaining(self) -> Optional[float]:
        """Remaining budget, or None if no limit."""
        if self.budget_limit is None:
            return None
        return max(0, self.budget_limit - self._total_cost)
    
    @property
    def budget_percentage(self) -> float:
        """Percentage of budget used."""
        if self.budget_limit is None or self.budget_limit <= 0:
            return 0.0
        return (self._total_cost / self.budget_limit) * 100
    
    @property
    def is_over_budget(self) -> bool:
        """Check if over budget."""
        if self.budget_limit is None:
            return False
        return self._total_cost >= self.budget_limit
    
    @property
    def budget_warning(self) -> bool:
        """Check if at warning threshold."""
        if self.budget_limit is None:
            return False
        return self._total_cost >= (self.budget_limit * self.warning_threshold)
    
    @property
    def budget_critical(self) -> bool:
        """Check if at critical threshold."""
        if self.budget_limit is None:
            return False
        return self._total_cost >= (self.budget_limit * self.critical_threshold)
    
    # =========================================================================
    # Tracking Methods
    # =========================================================================
    
    def track(
        self,
        response: Any,
        request_type: str = "complete",
    ) -> CostEntry:
        """
        Track a response's cost.
        
        Args:
            response: ProviderResponse or OrchestratorResponse
            request_type: Type of request
            
        Returns:
            CostEntry for this request
        """
        entry = CostEntry(
            timestamp=datetime.now(),
            provider=getattr(response, 'provider', 'unknown'),
            model=getattr(response, 'model', 'unknown'),
            tokens_used=getattr(response, 'tokens_used', 0),
            cost=getattr(response, 'cost', 0.0),
            latency_ms=getattr(response, 'latency_ms', 0.0),
            request_type=request_type,
        )
        
        self._entries.append(entry)
        
        # Update aggregates
        self._total_cost += entry.cost
        self._total_tokens += entry.tokens_used
        self._total_requests += 1
        self._total_latency += entry.latency_ms
        
        self._cost_by_provider[entry.provider] += entry.cost
        self._cost_by_model[entry.model] += entry.cost
        self._tokens_by_provider[entry.provider] += entry.tokens_used
        
        # Check budget alerts
        self._check_budget_alerts()
        
        return entry
    
    def track_manual(
        self,
        provider: str,
        model: str,
        tokens: int,
        cost: float,
        latency_ms: float = 0.0,
    ) -> CostEntry:
        """Track a manual cost entry."""
        entry = CostEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            tokens_used=tokens,
            cost=cost,
            latency_ms=latency_ms,
            request_type="manual",
        )
        
        self._entries.append(entry)
        self._total_cost += cost
        self._total_tokens += tokens
        self._total_requests += 1
        
        self._cost_by_provider[provider] += cost
        self._cost_by_model[model] += cost
        
        self._check_budget_alerts()
        
        return entry
    
    def _check_budget_alerts(self):
        """Check and generate budget alerts."""
        if self.budget_limit is None:
            return
        
        now = datetime.now()
        
        # Critical alert
        if self.budget_critical and not self._critical_sent:
            alert = BudgetAlert(
                timestamp=now,
                alert_type="critical",
                current_cost=self._total_cost,
                budget_limit=self.budget_limit,
                message=f"CRITICAL: {self.budget_percentage:.0f}% of budget used (${self._total_cost:.2f}/${self.budget_limit:.2f})"
            )
            self._alerts.append(alert)
            self._critical_sent = True
            logger.warning(alert.message)
        
        # Warning alert
        elif self.budget_warning and not self._warning_sent:
            alert = BudgetAlert(
                timestamp=now,
                alert_type="warning",
                current_cost=self._total_cost,
                budget_limit=self.budget_limit,
                message=f"WARNING: {self.budget_percentage:.0f}% of budget used (${self._total_cost:.2f}/${self.budget_limit:.2f})"
            )
            self._alerts.append(alert)
            self._warning_sent = True
            logger.warning(alert.message)
        
        # Over budget
        if self.is_over_budget:
            if not any(a.alert_type == "exceeded" for a in self._alerts):
                alert = BudgetAlert(
                    timestamp=now,
                    alert_type="exceeded",
                    current_cost=self._total_cost,
                    budget_limit=self.budget_limit,
                    message=f"EXCEEDED: Budget exceeded! ${self._total_cost:.2f} > ${self.budget_limit:.2f}"
                )
                self._alerts.append(alert)
                logger.error(alert.message)
    
    # =========================================================================
    # Summary Methods
    # =========================================================================
    
    def summary(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> CostSummary:
        """
        Get cost summary for a period.
        
        Args:
            since: Start of period (None = all time)
            until: End of period (None = now)
            
        Returns:
            CostSummary
        """
        # Filter entries by time
        entries = self._entries
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        if until:
            entries = [e for e in entries if e.timestamp <= until]
        
        if not entries:
            return CostSummary(
                total_cost=0.0,
                total_tokens=0,
                total_requests=0,
                avg_cost_per_request=0.0,
                avg_tokens_per_request=0.0,
                avg_latency_ms=0.0,
                by_provider={},
                by_model={},
                period_start=since,
                period_end=until,
            )
        
        total_cost = sum(e.cost for e in entries)
        total_tokens = sum(e.tokens_used for e in entries)
        total_latency = sum(e.latency_ms for e in entries)
        count = len(entries)
        
        by_provider = defaultdict(float)
        by_model = defaultdict(float)
        
        for e in entries:
            by_provider[e.provider] += e.cost
            by_model[e.model] += e.cost
        
        return CostSummary(
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_requests=count,
            avg_cost_per_request=total_cost / count if count else 0,
            avg_tokens_per_request=total_tokens / count if count else 0,
            avg_latency_ms=total_latency / count if count else 0,
            by_provider=dict(by_provider),
            by_model=dict(by_model),
            period_start=since or (entries[0].timestamp if entries else None),
            period_end=until or datetime.now(),
        )
    
    def today(self) -> CostSummary:
        """Get cost summary for today."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.summary(since=today_start)
    
    def this_week(self) -> CostSummary:
        """Get cost summary for this week."""
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.summary(since=week_start)
    
    def this_month(self) -> CostSummary:
        """Get cost summary for this month."""
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return self.summary(since=month_start)
    
    # =========================================================================
    # History and Export
    # =========================================================================
    
    def get_history(
        self,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[CostEntry]:
        """Get cost history with optional filters."""
        entries = self._entries
        
        if provider:
            entries = [e for e in entries if e.provider == provider]
        if model:
            entries = [e for e in entries if e.model == model]
        
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def get_alerts(self) -> List[BudgetAlert]:
        """Get all budget alerts."""
        return list(self._alerts)
    
    def export_history(self) -> List[Dict[str, Any]]:
        """Export history as list of dicts."""
        return [e.to_dict() for e in self._entries]
    
    def reset(self):
        """Reset all tracking data."""
        self._entries.clear()
        self._alerts.clear()
        self._total_cost = 0.0
        self._total_tokens = 0
        self._total_requests = 0
        self._total_latency = 0.0
        self._cost_by_provider.clear()
        self._cost_by_model.clear()
        self._tokens_by_provider.clear()
        self._warning_sent = False
        self._critical_sent = False
        logger.info("CostTracker reset")
    
    def cleanup_old_entries(self):
        """Remove entries older than retention period."""
        if self.history_retention_days <= 0:
            return
        
        cutoff = datetime.now() - timedelta(days=self.history_retention_days)
        old_count = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        removed = old_count - len(self._entries)
        
        if removed:
            logger.info(f"Cleaned up {removed} old cost entries")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CostTracker",
    "CostEntry",
    "CostSummary",
    "BudgetAlert",
]
