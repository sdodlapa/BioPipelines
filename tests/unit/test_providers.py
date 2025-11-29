"""
Tests for LLM Providers
=======================

Tests the unified provider layer (LocalProvider, CloudProvider).
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from workflow_composer.llm.providers import (
    ProviderProtocol,
    ProviderType,
    ModelCapability,
    ModelInfo,
    ProviderResponse,
    ProviderHealth,
    ProviderError,
    ProviderUnavailableError,
    ModelNotFoundError,
    BaseProvider,
    LocalProvider,
    CloudProvider,
)


# =============================================================================
# Provider Types Tests
# =============================================================================

class TestProviderTypes:
    """Test provider type enums and dataclasses."""
    
    def test_provider_type_values(self):
        """Test ProviderType enum values."""
        assert ProviderType.LOCAL.value == "local"
        assert ProviderType.CLOUD.value == "cloud"
    
    def test_model_capability_values(self):
        """Test ModelCapability enum values."""
        assert ModelCapability.CODING.value == "coding"
        assert ModelCapability.REASONING.value == "reasoning"
        assert ModelCapability.GENERAL.value == "general"
        assert ModelCapability.CHAT.value == "chat"
    
    def test_model_info_creation(self):
        """Test ModelInfo dataclass."""
        info = ModelInfo(
            id="test-model",
            name="Test Model",
            provider="test",
            provider_type=ProviderType.LOCAL,
            capabilities=[ModelCapability.GENERAL],
            context_length=4096,
            cost_per_1k_tokens=0.0,
        )
        assert info.id == "test-model"
        assert info.provider_type == ProviderType.LOCAL
        assert ModelCapability.GENERAL in info.capabilities
    
    def test_provider_response_creation(self):
        """Test ProviderResponse dataclass."""
        response = ProviderResponse(
            content="Test response",
            model="test-model",
            provider="test",
            provider_type=ProviderType.CLOUD,
            tokens_used=100,
            latency_ms=250.0,
            cost=0.001,
        )
        assert response.content == "Test response"
        assert response.tokens_used == 100
        assert response.cost == 0.001
    
    def test_provider_health_creation(self):
        """Test ProviderHealth dataclass."""
        health = ProviderHealth(
            provider="test",
            is_healthy=True,
            available_models=5,
        )
        assert health.is_healthy is True
        assert health.available_models == 5


# =============================================================================
# Provider Exceptions Tests
# =============================================================================

class TestProviderExceptions:
    """Test provider exceptions."""
    
    def test_provider_error(self):
        """Test base ProviderError."""
        error = ProviderError("Test error", "test_provider")
        assert "Test error" in str(error)
        assert error.provider == "test_provider"
    
    def test_provider_unavailable_error(self):
        """Test ProviderUnavailableError."""
        error = ProviderUnavailableError("test", "Service down")
        assert "test" in str(error)
        assert "Service down" in str(error)
    
    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("gpt-5", "openai")
        assert "gpt-5" in str(error)


# =============================================================================
# LocalProvider Tests
# =============================================================================

class TestLocalProvider:
    """Test LocalProvider functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        provider = LocalProvider()
        assert provider.name == "local"
        assert provider.provider_type == ProviderType.LOCAL
        assert provider.prefer_vllm is True
        assert provider.enable_fallback is True
    
    def test_init_custom(self):
        """Test custom initialization."""
        provider = LocalProvider(prefer_vllm=False, enable_fallback=False)
        assert provider.prefer_vllm is False
        assert provider.enable_fallback is False
    
    def test_backends_exist(self):
        """Test that backends are initialized."""
        provider = LocalProvider()
        assert hasattr(provider, 'vllm')
        assert hasattr(provider, 'ollama')
    
    def test_get_available_backends(self):
        """Test get_available_backends returns list."""
        provider = LocalProvider()
        backends = provider.get_available_backends()
        assert isinstance(backends, list)
    
    def test_list_models_returns_list(self):
        """Test list_models returns a list."""
        provider = LocalProvider()
        models = provider.list_models()
        assert isinstance(models, list)
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check returns ProviderHealth."""
        provider = LocalProvider()
        health = await provider.health_check()
        assert isinstance(health, ProviderHealth)
        assert health.provider == "local"


# =============================================================================
# CloudProvider Tests
# =============================================================================

class TestCloudProvider:
    """Test CloudProvider functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        provider = CloudProvider()
        assert provider.name == "cloud"
        assert provider.provider_type == ProviderType.CLOUD
        assert provider.prefer_lightning is True
    
    def test_backends_exist(self):
        """Test that backends are initialized."""
        provider = CloudProvider()
        assert hasattr(provider, 'lightning')
        assert hasattr(provider, 'openai')
        assert hasattr(provider, 'anthropic')
    
    def test_get_available_backends(self):
        """Test get_available_backends returns list."""
        provider = CloudProvider()
        backends = provider.get_available_backends()
        assert isinstance(backends, list)
    
    def test_list_models_returns_list(self):
        """Test list_models returns a list of ModelInfo."""
        provider = CloudProvider()
        models = provider.list_models()
        assert isinstance(models, list)
    
    def test_get_model_cost(self):
        """Test get_model_cost returns CloudModel or None."""
        provider = CloudProvider()
        cost = provider.get_model_cost("gpt-4o")
        if cost:
            assert hasattr(cost, 'cost_per_1k_input')
            assert hasattr(cost, 'cost_per_1k_output')
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check returns ProviderHealth."""
        provider = CloudProvider()
        health = await provider.health_check()
        assert isinstance(health, ProviderHealth)
        assert health.provider == "cloud"


# =============================================================================
# Cloud Models Registry Tests
# =============================================================================

class TestCloudModels:
    """Test cloud model registry."""
    
    def test_cloud_models_exist(self):
        """Test CLOUD_MODELS registry exists and has models."""
        from workflow_composer.llm.providers import CLOUD_MODELS
        assert isinstance(CLOUD_MODELS, dict)
        assert len(CLOUD_MODELS) > 0
    
    def test_gpt4o_in_registry(self):
        """Test GPT-4o is in registry."""
        from workflow_composer.llm.providers import CLOUD_MODELS
        assert "gpt-4o" in CLOUD_MODELS
    
    def test_claude_in_registry(self):
        """Test Claude models are in registry."""
        from workflow_composer.llm.providers import CLOUD_MODELS
        assert "claude-3-5-sonnet" in CLOUD_MODELS
    
    def test_model_has_required_fields(self):
        """Test models have required fields."""
        from workflow_composer.llm.providers import CLOUD_MODELS
        for model_id, model in CLOUD_MODELS.items():
            assert hasattr(model, 'id')
            assert hasattr(model, 'name')
            assert hasattr(model, 'provider')
            assert hasattr(model, 'cost_per_1k_input')
            assert hasattr(model, 'cost_per_1k_output')
