#!/usr/bin/env python3
"""
LLM Orchestrator Usage Examples
===============================

This example demonstrates the new unified LLM orchestration layer
that provides intelligent routing between local and cloud models.

Features:
- Automatic provider selection based on task type
- Fallback strategies (local → cloud)
- Ensemble patterns for critical decisions
- Cost tracking and optimization
- Task classification for smart routing

Usage:
    python examples/orchestrator_usage.py
"""

import asyncio
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from workflow_composer.llm import (
    # Orchestrator
    get_orchestrator,
    ModelOrchestrator,
    Strategy,
    EnsembleMode,
    
    # Task Router
    TaskRouter,
    TaskType,
    
    # Providers
    LocalProvider,
    CloudProvider,
    ProviderType,
)


# =============================================================================
# Basic Usage
# =============================================================================

async def basic_usage():
    """Basic orchestrator usage with automatic provider selection."""
    print("\n" + "="*60)
    print("1. BASIC USAGE")
    print("="*60)
    
    # Get orchestrator with default AUTO strategy
    orch = get_orchestrator()
    
    # Check availability
    print(f"\nProvider Status:")
    print(f"  Local available: {orch.is_local_available}")
    print(f"  Cloud available: {orch.is_cloud_available}")
    
    # The complete() method will automatically select the best provider
    # For this example, we'll just show the configuration
    print(f"\nOrchestrator configured with strategy: {orch.strategy.value}")
    print(f"  Fallback enabled: {orch.config.fallback_enabled}")
    print(f"  Prefer cheaper: {orch.config.prefer_cheaper}")


# =============================================================================
# Strategy Selection
# =============================================================================

async def strategy_examples():
    """Examples of different routing strategies."""
    print("\n" + "="*60)
    print("2. STRATEGY EXAMPLES")
    print("="*60)
    
    # LOCAL_FIRST: Try local GPU, fallback to cloud
    print("\n[LOCAL_FIRST Strategy]")
    orch = get_orchestrator(strategy=Strategy.LOCAL_FIRST)
    print(f"  Best for: Cost optimization, GPU-intensive tasks")
    print(f"  Behavior: Local → Cloud fallback")
    
    # CLOUD_ONLY: Only use cloud models
    print("\n[CLOUD_ONLY Strategy]")
    orch = get_orchestrator(strategy=Strategy.CLOUD_ONLY)
    print(f"  Best for: When you need specific cloud models")
    print(f"  Behavior: Cloud only, no fallback")
    
    # ENSEMBLE: Use multiple models
    print("\n[ENSEMBLE Strategy]")
    orch = get_orchestrator(strategy=Strategy.ENSEMBLE)
    print(f"  Best for: Critical decisions, validation")
    print(f"  Behavior: Query multiple models, combine results")
    
    # Using presets
    print("\n[Using Presets]")
    presets = ["development", "production", "critical", "cost_optimized"]
    for preset in presets:
        orch = get_orchestrator(preset=preset)
        print(f"  {preset}: strategy={orch.strategy.value}")


# =============================================================================
# Task Router
# =============================================================================

def task_routing_examples():
    """Examples of task-based routing."""
    print("\n" + "="*60)
    print("3. TASK ROUTER EXAMPLES")
    print("="*60)
    
    router = TaskRouter()
    
    # Example prompts
    prompts = [
        "Generate a Snakemake workflow for RNA-seq analysis with DESeq2",
        "Debug this Python script that fails with IndexError",
        "What is the difference between BWA-MEM and Bowtie2?",
        "Validate this ChIP-seq workflow for correctness",
        "Summarize the quality control metrics from FastQC",
        "Why does my workflow fail with exit code 137?",
    ]
    
    print("\nTask Classification Results:")
    print("-" * 75)
    print(f"{'Task Type':<25} {'Complexity':<12} {'Provider':<8} {'Critical'}")
    print("-" * 75)
    
    for prompt in prompts:
        analysis = router.analyze(prompt)
        print(f"{analysis.task_type.value:<25} {analysis.complexity.value:<12} "
              f"{analysis.recommended_provider.value:<8} {analysis.is_critical}")
    
    print("-" * 75)


# =============================================================================
# Provider Access
# =============================================================================

async def provider_examples():
    """Direct provider access examples."""
    print("\n" + "="*60)
    print("4. DIRECT PROVIDER ACCESS")
    print("="*60)
    
    # Local provider (vLLM + Ollama)
    print("\n[Local Provider]")
    local = LocalProvider()
    print(f"  Available: {local.is_available()}")
    print(f"  Backends: {local.get_available_backends()}")
    
    # Cloud provider (Lightning.ai, OpenAI, Anthropic)
    print("\n[Cloud Provider]")
    cloud = CloudProvider()
    print(f"  Available: {cloud.is_available()}")
    print(f"  Backends: {cloud.get_available_backends()}")
    
    # List available models
    print("\n[Available Cloud Models]")
    from workflow_composer.llm import CLOUD_MODELS
    for model_id, model in list(CLOUD_MODELS.items())[:5]:
        print(f"  {model.name}: ${model.avg_cost_per_1k:.4f}/1k tokens")


# =============================================================================
# Cost Tracking
# =============================================================================

async def cost_tracking_example():
    """Example of cost tracking."""
    print("\n" + "="*60)
    print("5. COST TRACKING")
    print("="*60)
    
    orch = get_orchestrator(strategy=Strategy.LOCAL_FIRST)
    
    print("\nUsage Statistics:")
    print(f"  Total requests: {orch.stats.total_requests}")
    print(f"  Total tokens: {orch.stats.total_tokens}")
    print(f"  Total cost: ${orch.stats.total_cost:.4f}")
    print(f"  Local requests: {orch.stats.local_requests}")
    print(f"  Cloud requests: {orch.stats.cloud_requests}")
    print(f"  Fallbacks used: {orch.stats.fallbacks_used}")
    
    # Note: In real usage, these stats accumulate with each request
    print("\n  (Stats start at 0, increment with each request)")


# =============================================================================
# Async Usage Pattern
# =============================================================================

async def async_usage_pattern():
    """Pattern for using the orchestrator in async code."""
    print("\n" + "="*60)
    print("6. ASYNC USAGE PATTERN")
    print("="*60)
    
    print("""
    # Example async usage:
    
    async def generate_workflow(description: str) -> str:
        orch = get_orchestrator(strategy=Strategy.LOCAL_FIRST)
        
        # Single completion
        response = await orch.complete(
            prompt=f"Generate a Snakemake workflow for: {description}",
            temperature=0.1,
            max_tokens=4096,
        )
        
        print(f"Used provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Cost: ${response.cost:.4f}")
        print(f"Latency: {response.latency_ms:.0f}ms")
        
        return response.content
    
    # Ensemble for validation
    async def validate_workflow(workflow: str) -> str:
        orch = get_orchestrator(strategy=Strategy.ENSEMBLE)
        
        response = await orch.ensemble(
            prompt=f"Validate this workflow for errors:\\n{workflow}",
            mode=EnsembleMode.CONSENSUS,
        )
        
        return response.content
    """)


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("#" + " LLM ORCHESTRATOR EXAMPLES ".center(58) + "#")
    print("#"*60)
    
    await basic_usage()
    await strategy_examples()
    task_routing_examples()  # Sync function
    await provider_examples()
    await cost_tracking_example()
    await async_usage_pattern()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)
    print("\nFor real usage, see the docstrings in:")
    print("  - src/workflow_composer/llm/orchestrator.py")
    print("  - src/workflow_composer/llm/task_router.py")
    print("  - src/workflow_composer/llm/strategies.py")


if __name__ == "__main__":
    asyncio.run(main())
