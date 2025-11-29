"""
Tests for Model Orchestrator
============================

Tests the ModelOrchestrator and strategies.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from workflow_composer.llm import (
    ModelOrchestrator,
    OrchestratorResponse,
    UsageStats,
    get_orchestrator,
    reset_orchestrator,
    Strategy,
    EnsembleMode,
    StrategyConfig,
    PRESETS,
    get_preset,
    ProviderType,
)


# =============================================================================
# Strategy Tests
# =============================================================================

class TestStrategy:
    """Test Strategy enum."""
    
    def test_strategy_values(self):
        """Test Strategy enum values."""
        assert Strategy.AUTO.value == "auto"
        assert Strategy.LOCAL_FIRST.value == "local_first"
        assert Strategy.LOCAL_ONLY.value == "local_only"
        assert Strategy.CLOUD_ONLY.value == "cloud_only"
        assert Strategy.ENSEMBLE.value == "ensemble"
        assert Strategy.PARALLEL.value == "parallel"
        assert Strategy.CASCADE.value == "cascade"
        assert Strategy.CHAIN.value == "chain"
    
    def test_ensemble_mode_values(self):
        """Test EnsembleMode enum values."""
        assert EnsembleMode.VOTE.value == "vote"
        assert EnsembleMode.BEST.value == "best"
        assert EnsembleMode.MERGE.value == "merge"
        assert EnsembleMode.CONSENSUS.value == "consensus"


# =============================================================================
# StrategyConfig Tests
# =============================================================================

class TestStrategyConfig:
    """Test StrategyConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = StrategyConfig()
        assert config.strategy == Strategy.AUTO
        assert config.fallback_enabled is True
        assert config.max_retries == 2
        assert config.timeout_seconds == 60.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StrategyConfig(
            strategy=Strategy.LOCAL_FIRST,
            fallback_enabled=False,
            max_retries=5,
        )
        assert config.strategy == Strategy.LOCAL_FIRST
        assert config.fallback_enabled is False
        assert config.max_retries == 5


# =============================================================================
# Presets Tests
# =============================================================================

class TestPresets:
    """Test preset configurations."""
    
    def test_presets_exist(self):
        """Test PRESETS dict exists and has presets."""
        assert isinstance(PRESETS, dict)
        assert "development" in PRESETS
        assert "production" in PRESETS
        assert "critical" in PRESETS
        assert "cost_optimized" in PRESETS
    
    def test_get_preset_development(self):
        """Test getting development preset."""
        config = get_preset("development")
        assert isinstance(config, StrategyConfig)
        assert config.strategy == Strategy.LOCAL_FIRST
    
    def test_get_preset_production(self):
        """Test getting production preset."""
        config = get_preset("production")
        assert config.strategy == Strategy.AUTO
    
    def test_get_preset_critical(self):
        """Test getting critical preset."""
        config = get_preset("critical")
        assert config.strategy == Strategy.ENSEMBLE
    
    def test_get_preset_invalid(self):
        """Test getting invalid preset raises error."""
        with pytest.raises(ValueError):
            get_preset("invalid_preset")


# =============================================================================
# UsageStats Tests
# =============================================================================

class TestUsageStats:
    """Test UsageStats dataclass."""
    
    def test_default_stats(self):
        """Test default statistics."""
        stats = UsageStats()
        assert stats.total_requests == 0
        assert stats.total_tokens == 0
        assert stats.total_cost == 0.0
        assert stats.local_requests == 0
        assert stats.cloud_requests == 0
    
    def test_add_local_response(self):
        """Test adding a local response."""
        stats = UsageStats()
        response = OrchestratorResponse(
            content="test",
            model="test-model",
            provider="vllm",
            provider_type=ProviderType.LOCAL,
            strategy_used=Strategy.LOCAL_FIRST,
            tokens_used=100,
            latency_ms=50.0,
            cost=0.0,
        )
        stats.add(response)
        
        assert stats.total_requests == 1
        assert stats.total_tokens == 100
        assert stats.local_requests == 1
        assert stats.cloud_requests == 0
    
    def test_add_cloud_response(self):
        """Test adding a cloud response."""
        stats = UsageStats()
        response = OrchestratorResponse(
            content="test",
            model="gpt-4o",
            provider="openai",
            provider_type=ProviderType.CLOUD,
            strategy_used=Strategy.CLOUD_ONLY,
            tokens_used=200,
            latency_ms=150.0,
            cost=0.002,
        )
        stats.add(response)
        
        assert stats.total_requests == 1
        assert stats.total_tokens == 200
        assert stats.total_cost == 0.002
        assert stats.cloud_requests == 1


# =============================================================================
# ModelOrchestrator Tests
# =============================================================================

class TestModelOrchestrator:
    """Test ModelOrchestrator class."""
    
    def test_init_default(self):
        """Test default initialization."""
        orch = ModelOrchestrator()
        assert orch.strategy == Strategy.AUTO
        assert orch.config.fallback_enabled is True
    
    def test_init_with_strategy(self):
        """Test initialization with strategy."""
        orch = ModelOrchestrator(strategy=Strategy.LOCAL_FIRST)
        assert orch.strategy == Strategy.LOCAL_FIRST
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = StrategyConfig(
            strategy=Strategy.ENSEMBLE,
            max_retries=5,
        )
        orch = ModelOrchestrator(config=config)
        assert orch.strategy == Strategy.ENSEMBLE
        assert orch.config.max_retries == 5
    
    def test_properties(self):
        """Test orchestrator properties."""
        orch = ModelOrchestrator()
        assert isinstance(orch.total_cost, float)
        assert isinstance(orch.is_local_available, bool)
        assert isinstance(orch.is_cloud_available, bool)
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        orch = ModelOrchestrator()
        orch.stats.total_requests = 10
        orch.reset_stats()
        assert orch.stats.total_requests == 0
    
    def test_clear_cache(self):
        """Test clearing cache."""
        orch = ModelOrchestrator()
        orch._cache["test"] = "value"
        orch.clear_cache()
        assert len(orch._cache) == 0
    
    def test_get_available_models(self):
        """Test getting available models."""
        orch = ModelOrchestrator()
        models = orch.get_available_models()
        assert isinstance(models, dict)
        assert "local" in models
        assert "cloud" in models


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestGetOrchestrator:
    """Test get_orchestrator factory function."""
    
    def test_get_orchestrator_default(self):
        """Test getting orchestrator with defaults."""
        reset_orchestrator()
        orch = get_orchestrator()
        assert isinstance(orch, ModelOrchestrator)
        assert orch.strategy == Strategy.AUTO
    
    def test_get_orchestrator_with_strategy(self):
        """Test getting orchestrator with strategy."""
        reset_orchestrator()
        orch = get_orchestrator(strategy=Strategy.LOCAL_FIRST)
        assert orch.strategy == Strategy.LOCAL_FIRST
    
    def test_get_orchestrator_with_preset(self):
        """Test getting orchestrator with preset."""
        orch = get_orchestrator(preset="development")
        assert orch.strategy == Strategy.LOCAL_FIRST
    
    def test_reset_orchestrator(self):
        """Test resetting default orchestrator."""
        get_orchestrator()
        reset_orchestrator()
        # Should create new instance after reset
        orch = get_orchestrator(strategy=Strategy.CLOUD_ONLY)
        assert orch.strategy == Strategy.CLOUD_ONLY


# =============================================================================
# OrchestratorResponse Tests
# =============================================================================

class TestOrchestratorResponse:
    """Test OrchestratorResponse dataclass."""
    
    def test_response_creation(self):
        """Test creating a response."""
        response = OrchestratorResponse(
            content="Generated workflow",
            model="test-model",
            provider="vllm",
            provider_type=ProviderType.LOCAL,
            strategy_used=Strategy.LOCAL_FIRST,
        )
        assert response.content == "Generated workflow"
        assert response.model == "test-model"
        assert response.strategy_used == Strategy.LOCAL_FIRST
    
    def test_response_str(self):
        """Test response string conversion."""
        response = OrchestratorResponse(
            content="Test content",
            model="test",
            provider="test",
            provider_type=ProviderType.LOCAL,
            strategy_used=Strategy.AUTO,
        )
        assert str(response) == "Test content"
    
    def test_response_metadata(self):
        """Test response metadata fields."""
        response = OrchestratorResponse(
            content="test",
            model="test",
            provider="test",
            provider_type=ProviderType.CLOUD,
            strategy_used=Strategy.ENSEMBLE,
            tokens_used=500,
            latency_ms=200.0,
            cost=0.005,
            attempts=3,
            fallback_used=True,
            cached=False,
        )
        assert response.tokens_used == 500
        assert response.latency_ms == 200.0
        assert response.cost == 0.005
        assert response.attempts == 3
        assert response.fallback_used is True
        assert response.cached is False
