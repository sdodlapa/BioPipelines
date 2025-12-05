"""
Routing Metrics
===============

Collects and logs metrics about LLM routing decisions.
Used for debugging, optimization, and monitoring.

Features:
- Tracks every routing decision with full context
- Logs to JSONL file for analysis
- Provides summary statistics
- Integrates with debug_routing flag in StrategyConfig

Usage:
    from workflow_composer.llm.metrics import RoutingMetrics, RoutingDecision
    
    metrics = RoutingMetrics()
    
    # Log a decision
    decision = RoutingDecision(
        task_type="code_generation",
        query_length=150,
        model_used="coder",
        provider="vllm",
        ...
    )
    metrics.log(decision)
    
    # Get summary
    summary = metrics.get_summary()
"""

import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """
    Captures full context of a routing decision.
    
    Logged for every LLM request when metrics are enabled.
    """
    
    # Request context
    task_type: str
    """Task category (e.g., 'code_generation', 'intent_parsing')."""
    
    query_length: int
    """Character count of the input query."""
    
    # Routing decision
    strategy_profile: str
    """Active strategy profile name."""
    
    model_key: str
    """Selected model key (e.g., 'coder', 'generalist')."""
    
    model_id: str
    """Actual model identifier (e.g., 'Qwen2.5-Coder-7B-Instruct-AWQ')."""
    
    provider: str
    """Provider type: 'vllm', 'deepseek', 'openai', 'anthropic', etc."""
    
    # Fallback tracking
    fallback_depth: int = 0
    """0 = primary model used, 1+ = fallback was needed."""
    
    fallback_reason: Optional[str] = None
    """Reason for fallback if fallback_depth > 0."""
    
    fallback_chain: List[str] = field(default_factory=list)
    """Models attempted before success."""
    
    # Outcome
    success: bool = True
    """Whether the request succeeded."""
    
    error_type: Optional[str] = None
    """Error type if failed (e.g., 'timeout', 'rate_limit', 'model_error')."""
    
    error_message: Optional[str] = None
    """Error message if failed."""
    
    # Performance
    latency_ms: float = 0.0
    """Total request latency in milliseconds."""
    
    time_to_first_token_ms: Optional[float] = None
    """Time to first token for streaming responses."""
    
    tokens_generated: Optional[int] = None
    """Number of tokens in the response."""
    
    # Cost
    estimated_cost: Optional[float] = None
    """Estimated cost in dollars."""
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """ISO timestamp of the decision."""
    
    request_id: Optional[str] = None
    """Unique request identifier for tracing."""
    
    session_id: Optional[str] = None
    """Session identifier if using sessions."""
    
    # Debugging context
    debug_context: Dict[str, Any] = field(default_factory=dict)
    """Additional debug information."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class RoutingMetrics:
    """
    Collects and manages routing metrics.
    
    Thread-safe for concurrent usage.
    """
    
    def __init__(
        self,
        log_file: Optional[str | Path] = None,
        enabled: bool = True,
        buffer_size: int = 100,
    ):
        """
        Initialize metrics collector.
        
        Args:
            log_file: Path to JSONL log file (default: logs/routing_metrics.jsonl)
            enabled: Whether to collect metrics
            buffer_size: Number of decisions to buffer before flush
        """
        self.enabled = enabled
        self.buffer_size = buffer_size
        self._buffer: List[RoutingDecision] = []
        self._lock = threading.Lock()
        
        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._fallback_requests = 0
        self._total_latency_ms = 0.0
        self._model_usage: Dict[str, int] = {}
        self._task_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        
        # Log file
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "routing_metrics.jsonl"
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, decision: RoutingDecision) -> None:
        """
        Log a routing decision.
        
        Thread-safe. Buffers decisions and flushes periodically.
        """
        if not self.enabled:
            return
        
        with self._lock:
            # Update statistics
            self._total_requests += 1
            self._total_latency_ms += decision.latency_ms
            
            if decision.success:
                self._successful_requests += 1
            
            if decision.fallback_depth > 0:
                self._fallback_requests += 1
            
            # Track model usage
            model_key = decision.model_key
            self._model_usage[model_key] = self._model_usage.get(model_key, 0) + 1
            
            # Track task types
            task = decision.task_type
            self._task_counts[task] = self._task_counts.get(task, 0) + 1
            
            # Track errors
            if decision.error_type:
                self._error_counts[decision.error_type] = (
                    self._error_counts.get(decision.error_type, 0) + 1
                )
            
            # Buffer decision
            self._buffer.append(decision)
            
            # Flush if buffer is full
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Write buffered decisions to log file."""
        if not self._buffer:
            return
        
        try:
            with open(self.log_file, "a") as f:
                for decision in self._buffer:
                    f.write(decision.to_json() + "\n")
            
            self._buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    def flush(self) -> None:
        """Force flush of buffered decisions."""
        with self._lock:
            self._flush_buffer()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with aggregated metrics.
        """
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._total_requests
                if self._total_requests > 0
                else 0.0
            )
            
            success_rate = (
                self._successful_requests / self._total_requests * 100
                if self._total_requests > 0
                else 0.0
            )
            
            fallback_rate = (
                self._fallback_requests / self._total_requests * 100
                if self._total_requests > 0
                else 0.0
            )
            
            return {
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "success_rate_percent": round(success_rate, 2),
                "fallback_requests": self._fallback_requests,
                "fallback_rate_percent": round(fallback_rate, 2),
                "average_latency_ms": round(avg_latency, 2),
                "model_usage": dict(self._model_usage),
                "task_counts": dict(self._task_counts),
                "error_counts": dict(self._error_counts),
            }
    
    def print_summary(self) -> None:
        """Print formatted summary to stdout."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("Routing Metrics Summary")
        print("=" * 50)
        
        print(f"\nRequests:")
        print(f"  Total:      {summary['total_requests']}")
        print(f"  Successful: {summary['successful_requests']} ({summary['success_rate_percent']}%)")
        print(f"  Fallbacks:  {summary['fallback_requests']} ({summary['fallback_rate_percent']}%)")
        
        print(f"\nPerformance:")
        print(f"  Avg Latency: {summary['average_latency_ms']} ms")
        
        if summary['model_usage']:
            print(f"\nModel Usage:")
            for model, count in sorted(summary['model_usage'].items(), key=lambda x: -x[1]):
                print(f"  {model}: {count}")
        
        if summary['task_counts']:
            print(f"\nTask Types:")
            for task, count in sorted(summary['task_counts'].items(), key=lambda x: -x[1]):
                print(f"  {task}: {count}")
        
        if summary['error_counts']:
            print(f"\nErrors:")
            for error, count in sorted(summary['error_counts'].items(), key=lambda x: -x[1]):
                print(f"  {error}: {count}")
        
        print("\n" + "=" * 50)
    
    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._flush_buffer()
            self._total_requests = 0
            self._successful_requests = 0
            self._fallback_requests = 0
            self._total_latency_ms = 0.0
            self._model_usage.clear()
            self._task_counts.clear()
            self._error_counts.clear()


# Global metrics instance
_global_metrics: Optional[RoutingMetrics] = None


def get_metrics() -> RoutingMetrics:
    """Get or create global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = RoutingMetrics()
    return _global_metrics


def log_routing_decision(decision: RoutingDecision) -> None:
    """Log a routing decision to global metrics."""
    get_metrics().log(decision)


class MetricsContext:
    """
    Context manager for timing and logging a routing decision.
    
    Usage:
        with MetricsContext(task_type="code_generation", ...) as ctx:
            response = await model.generate(prompt)
            ctx.set_result(success=True, tokens=len(response))
    """
    
    def __init__(
        self,
        task_type: str,
        query_length: int,
        strategy_profile: str,
        model_key: str,
        model_id: str,
        provider: str,
        request_id: Optional[str] = None,
    ):
        self.decision = RoutingDecision(
            task_type=task_type,
            query_length=query_length,
            strategy_profile=strategy_profile,
            model_key=model_key,
            model_id=model_id,
            provider=provider,
            request_id=request_id,
        )
        self._start_time: Optional[float] = None
    
    def __enter__(self) -> "MetricsContext":
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start_time:
            self.decision.latency_ms = (time.time() - self._start_time) * 1000
        
        if exc_type is not None:
            self.decision.success = False
            self.decision.error_type = exc_type.__name__
            self.decision.error_message = str(exc_val)[:200]
        
        log_routing_decision(self.decision)
    
    def set_result(
        self,
        success: bool = True,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
        ttft_ms: Optional[float] = None,
    ) -> None:
        """Set result details before context exits."""
        self.decision.success = success
        self.decision.tokens_generated = tokens
        self.decision.estimated_cost = cost
        self.decision.time_to_first_token_ms = ttft_ms
    
    def set_fallback(
        self,
        depth: int,
        reason: str,
        chain: List[str],
    ) -> None:
        """Record fallback information."""
        self.decision.fallback_depth = depth
        self.decision.fallback_reason = reason
        self.decision.fallback_chain = chain
    
    def add_debug_context(self, **kwargs) -> None:
        """Add debug context."""
        self.decision.debug_context.update(kwargs)


__all__ = [
    "RoutingDecision",
    "RoutingMetrics",
    "MetricsContext",
    "get_metrics",
    "log_routing_decision",
]
