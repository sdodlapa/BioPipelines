"""
Tests for Ensemble and Cost Tracking
====================================

Tests ensemble strategies and CostTracker functionality.
"""

import pytest
from datetime import datetime, timedelta

from workflow_composer.llm import (
    ModelOrchestrator,
    OrchestratorResponse,
    Strategy,
    EnsembleMode,
    ProviderType,
)
from workflow_composer.llm.cost_tracker import (
    CostTracker,
    CostEntry,
    CostSummary,
    BudgetAlert,
)


# =============================================================================
# CostEntry Tests
# =============================================================================

class TestCostEntry:
    """Test CostEntry dataclass."""
    
    def test_entry_creation(self):
        """Test creating a cost entry."""
        entry = CostEntry(
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4o",
            tokens_used=1000,
            cost=0.01,
            latency_ms=250.0,
        )
        assert entry.provider == "openai"
        assert entry.model == "gpt-4o"
        assert entry.tokens_used == 1000
        assert entry.cost == 0.01
    
    def test_entry_to_dict(self):
        """Test converting entry to dict."""
        entry = CostEntry(
            timestamp=datetime.now(),
            provider="anthropic",
            model="claude-3",
            tokens_used=500,
            cost=0.005,
            latency_ms=100.0,
        )
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["provider"] == "anthropic"
        assert d["tokens_used"] == 500


# =============================================================================
# CostSummary Tests
# =============================================================================

class TestCostSummary:
    """Test CostSummary dataclass."""
    
    def test_summary_creation(self):
        """Test creating a cost summary."""
        summary = CostSummary(
            total_cost=1.50,
            total_tokens=15000,
            total_requests=10,
            avg_cost_per_request=0.15,
            avg_tokens_per_request=1500,
            avg_latency_ms=200.0,
            by_provider={"openai": 1.0, "anthropic": 0.5},
            by_model={"gpt-4o": 1.0, "claude-3": 0.5},
        )
        assert summary.total_cost == 1.50
        assert summary.total_requests == 10
    
    def test_summary_str(self):
        """Test summary string representation."""
        summary = CostSummary(
            total_cost=1.50,
            total_tokens=15000,
            total_requests=10,
            avg_cost_per_request=0.15,
            avg_tokens_per_request=1500,
            avg_latency_ms=200.0,
            by_provider={"openai": 1.50},
            by_model={"gpt-4o": 1.50},
        )
        s = str(summary)
        assert "Cost Summary" in s
        assert "$1.50" in s or "1.5" in s


# =============================================================================
# BudgetAlert Tests
# =============================================================================

class TestBudgetAlert:
    """Test BudgetAlert dataclass."""
    
    def test_alert_creation(self):
        """Test creating a budget alert."""
        alert = BudgetAlert(
            timestamp=datetime.now(),
            alert_type="warning",
            current_cost=8.0,
            budget_limit=10.0,
            message="80% of budget used",
        )
        assert alert.alert_type == "warning"
        assert alert.current_cost == 8.0
    
    def test_alert_percentage(self):
        """Test alert percentage calculation."""
        alert = BudgetAlert(
            timestamp=datetime.now(),
            alert_type="warning",
            current_cost=7.5,
            budget_limit=10.0,
            message="test",
        )
        assert alert.percentage == 75.0


# =============================================================================
# CostTracker Core Tests
# =============================================================================

class TestCostTrackerCore:
    """Test CostTracker core functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        tracker = CostTracker()
        assert tracker.budget_limit is None
        assert tracker.total_cost == 0.0
        assert tracker.total_requests == 0
    
    def test_init_with_budget(self):
        """Test initialization with budget."""
        tracker = CostTracker(budget_limit=10.0)
        assert tracker.budget_limit == 10.0
        assert tracker.budget_remaining == 10.0
    
    def test_track_manual(self):
        """Test manual tracking."""
        tracker = CostTracker()
        entry = tracker.track_manual(
            provider="openai",
            model="gpt-4o",
            tokens=1000,
            cost=0.01,
        )
        assert isinstance(entry, CostEntry)
        assert tracker.total_cost == 0.01
        assert tracker.total_tokens == 1000
        assert tracker.total_requests == 1
    
    def test_track_multiple(self):
        """Test tracking multiple requests."""
        tracker = CostTracker()
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01)
        tracker.track_manual("anthropic", "claude-3", 2000, 0.02)
        tracker.track_manual("openai", "gpt-4o", 500, 0.005)
        
        assert tracker.total_requests == 3
        assert tracker.total_tokens == 3500
        assert tracker.total_cost == pytest.approx(0.035)


# =============================================================================
# CostTracker Budget Tests
# =============================================================================

class TestCostTrackerBudget:
    """Test CostTracker budget functionality."""
    
    def test_budget_remaining(self):
        """Test budget remaining calculation."""
        tracker = CostTracker(budget_limit=10.0)
        tracker.track_manual("openai", "gpt-4o", 1000, 3.0)
        assert tracker.budget_remaining == 7.0
    
    def test_budget_percentage(self):
        """Test budget percentage calculation."""
        tracker = CostTracker(budget_limit=10.0)
        tracker.track_manual("openai", "gpt-4o", 1000, 5.0)
        assert tracker.budget_percentage == 50.0
    
    def test_is_over_budget(self):
        """Test over budget detection."""
        tracker = CostTracker(budget_limit=10.0)
        tracker.track_manual("openai", "gpt-4o", 1000, 12.0)
        assert tracker.is_over_budget is True
    
    def test_budget_warning(self):
        """Test budget warning threshold."""
        tracker = CostTracker(budget_limit=10.0, warning_threshold=0.8)
        tracker.track_manual("openai", "gpt-4o", 1000, 8.5)
        assert tracker.budget_warning is True
        assert tracker.budget_critical is False
    
    def test_budget_critical(self):
        """Test budget critical threshold."""
        tracker = CostTracker(budget_limit=10.0, critical_threshold=0.95)
        tracker.track_manual("openai", "gpt-4o", 1000, 9.6)
        assert tracker.budget_critical is True
    
    def test_no_budget_no_warnings(self):
        """Test no warnings when no budget set."""
        tracker = CostTracker()  # No budget
        tracker.track_manual("openai", "gpt-4o", 1000, 100.0)
        assert tracker.is_over_budget is False
        assert tracker.budget_warning is False


# =============================================================================
# CostTracker Summary Tests
# =============================================================================

class TestCostTrackerSummary:
    """Test CostTracker summary functionality."""
    
    def test_summary_basic(self):
        """Test basic summary."""
        tracker = CostTracker()
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01, 100.0)
        tracker.track_manual("openai", "gpt-4o", 2000, 0.02, 200.0)
        
        summary = tracker.summary()
        assert isinstance(summary, CostSummary)
        assert summary.total_cost == pytest.approx(0.03)
        assert summary.total_requests == 2
        assert summary.avg_tokens_per_request == 1500
    
    def test_summary_by_provider(self):
        """Test summary breakdown by provider."""
        tracker = CostTracker()
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01)
        tracker.track_manual("anthropic", "claude-3", 1000, 0.02)
        
        summary = tracker.summary()
        assert "openai" in summary.by_provider
        assert "anthropic" in summary.by_provider
    
    def test_summary_by_model(self):
        """Test summary breakdown by model."""
        tracker = CostTracker()
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01)
        tracker.track_manual("openai", "gpt-4o-mini", 1000, 0.001)
        
        summary = tracker.summary()
        assert "gpt-4o" in summary.by_model
        assert "gpt-4o-mini" in summary.by_model
    
    def test_summary_empty(self):
        """Test summary when no requests."""
        tracker = CostTracker()
        summary = tracker.summary()
        assert summary.total_cost == 0.0
        assert summary.total_requests == 0


# =============================================================================
# CostTracker History Tests
# =============================================================================

class TestCostTrackerHistory:
    """Test CostTracker history functionality."""
    
    def test_get_history(self):
        """Test getting history."""
        tracker = CostTracker()
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01)
        tracker.track_manual("anthropic", "claude-3", 1000, 0.02)
        
        history = tracker.get_history()
        assert len(history) == 2
    
    def test_get_history_limit(self):
        """Test history with limit."""
        tracker = CostTracker()
        for i in range(10):
            tracker.track_manual("openai", "gpt-4o", 100, 0.001)
        
        history = tracker.get_history(limit=5)
        assert len(history) == 5
    
    def test_get_history_filter_provider(self):
        """Test history filtered by provider."""
        tracker = CostTracker()
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01)
        tracker.track_manual("anthropic", "claude-3", 1000, 0.02)
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01)
        
        history = tracker.get_history(provider="openai")
        assert len(history) == 2
    
    def test_export_history(self):
        """Test exporting history."""
        tracker = CostTracker()
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01)
        
        exported = tracker.export_history()
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert isinstance(exported[0], dict)
    
    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker()
        tracker.track_manual("openai", "gpt-4o", 1000, 0.01)
        tracker.reset()
        
        assert tracker.total_cost == 0.0
        assert tracker.total_requests == 0
        assert len(tracker.get_history()) == 0


# =============================================================================
# Ensemble Mode Tests
# =============================================================================

class TestEnsembleMode:
    """Test EnsembleMode enum and usage."""
    
    def test_ensemble_modes_exist(self):
        """Test all ensemble modes exist."""
        assert EnsembleMode.VOTE.value == "vote"
        assert EnsembleMode.BEST.value == "best"
        assert EnsembleMode.MERGE.value == "merge"
        assert EnsembleMode.CONSENSUS.value == "consensus"
    
    def test_orchestrator_accepts_ensemble_mode(self):
        """Test orchestrator accepts ensemble mode."""
        orch = ModelOrchestrator(strategy=Strategy.ENSEMBLE)
        assert orch.config.ensemble_mode == EnsembleMode.BEST  # default


# =============================================================================
# Integration Tests
# =============================================================================

class TestEnsembleIntegration:
    """Integration tests for ensemble and cost tracking."""
    
    def test_track_orchestrator_response(self):
        """Test tracking an orchestrator response."""
        tracker = CostTracker()
        response = OrchestratorResponse(
            content="test",
            model="gpt-4o",
            provider="openai",
            provider_type=ProviderType.CLOUD,
            strategy_used=Strategy.AUTO,
            tokens_used=1000,
            latency_ms=250.0,
            cost=0.01,
        )
        
        entry = tracker.track(response)
        assert entry.model == "gpt-4o"
        assert entry.cost == 0.01
        assert tracker.total_cost == 0.01
