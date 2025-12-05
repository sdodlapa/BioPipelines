"""
T4 Fleet Model Router for BioPipelines
=======================================

Routes requests to the appropriate vLLM server based on task type.
Provides load balancing across replicas and fallback to cloud APIs.

Architecture:
    User Request -> Router -> Local vLLM (T4) or Cloud API

Usage:
    from t4_router import T4ModelRouter
    
    router = T4ModelRouter()
    response = await router.complete("intent_parsing", "Analyze RNA-seq data")
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import yaml


class TaskCategory(Enum):
    """Task categories mapped to specialized models."""
    INTENT_PARSING = "intent"
    CODE_GENERATION = "codegen"
    CODE_VALIDATION = "validation"
    DATA_ANALYSIS = "analysis"
    MATH_STATISTICS = "math"
    BIOMEDICAL = "biomedical"
    DOCUMENTATION = "docs"
    EMBEDDINGS = "embeddings"
    SAFETY = "safety"
    ORCHESTRATION = "orchestration"  # Cloud-only


@dataclass
class ModelEndpoint:
    """Represents a model endpoint (local or cloud)."""
    name: str
    url: str
    model_id: str
    is_cloud: bool = False
    is_healthy: bool = True
    last_check: float = 0
    latency_ms: float = 0
    error_count: int = 0
    
    def mark_unhealthy(self):
        self.is_healthy = False
        self.error_count += 1
    
    def mark_healthy(self, latency_ms: float):
        self.is_healthy = True
        self.latency_ms = latency_ms
        self.last_check = time.time()
        self.error_count = 0


@dataclass
class TaskRouteConfig:
    """Configuration for routing a task category."""
    category: TaskCategory
    local_endpoints: List[ModelEndpoint] = field(default_factory=list)
    cloud_fallback: Optional[ModelEndpoint] = None
    prefer_local: bool = True
    max_retries: int = 2
    timeout_seconds: float = 30.0


class CloudProvider:
    """Cloud API providers for fallback."""
    
    DEEPSEEK = ModelEndpoint(
        name="deepseek-v3",
        url="https://api.deepseek.com/v1",
        model_id="deepseek-chat",
        is_cloud=True
    )
    
    OPENAI = ModelEndpoint(
        name="gpt-4o",
        url="https://api.openai.com/v1",
        model_id="gpt-4o",
        is_cloud=True
    )
    
    ANTHROPIC = ModelEndpoint(
        name="claude-3.5-sonnet",
        url="https://api.anthropic.com/v1",
        model_id="claude-3-5-sonnet-20241022",
        is_cloud=True
    )


class T4ModelRouter:
    """
    Routes requests to T4 vLLM servers with cloud fallback.
    
    Features:
    - Task-based routing
    - Load balancing across replicas
    - Health checking
    - Automatic failover to cloud
    - Request/response caching (optional)
    """
    
    def __init__(
        self,
        connection_dir: Optional[str] = None,
        default_cloud: str = "deepseek",
        health_check_interval: float = 60.0,
    ):
        self.connection_dir = Path(connection_dir or os.path.expanduser(
            "~/BioPipelines/.model_connections"
        ))
        self.default_cloud = default_cloud
        self.health_check_interval = health_check_interval
        
        # Route configuration per task
        self.routes: Dict[TaskCategory, TaskRouteConfig] = {}
        
        # Cloud API keys
        self.api_keys = {
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        }
        
        # Initialize routes
        self._load_local_endpoints()
        self._configure_routes()
    
    def _load_local_endpoints(self):
        """Load endpoints from connection files created by vLLM servers."""
        self.local_endpoints: Dict[str, List[ModelEndpoint]] = {}
        
        if not self.connection_dir.exists():
            print(f"Warning: Connection directory not found: {self.connection_dir}")
            return
        
        for env_file in self.connection_dir.glob("*.env"):
            try:
                config = {}
                with open(env_file) as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            config[key] = value
                
                model_key = env_file.stem
                endpoint = ModelEndpoint(
                    name=config.get("MODEL_NAME", model_key),
                    url=config.get("URL", ""),
                    model_id=config.get("MODEL_ID", ""),
                    is_cloud=False
                )
                
                # Group by base model type (remove replica suffixes)
                base_key = model_key.rstrip("0123456789").rstrip("_-r")
                if base_key not in self.local_endpoints:
                    self.local_endpoints[base_key] = []
                self.local_endpoints[base_key].append(endpoint)
                
            except Exception as e:
                print(f"Error loading {env_file}: {e}")
    
    def _configure_routes(self):
        """Configure routing for each task category."""
        
        # Map task categories to model keys
        category_to_model = {
            TaskCategory.INTENT_PARSING: "intent",
            TaskCategory.CODE_GENERATION: "codegen",
            TaskCategory.CODE_VALIDATION: "validation",
            TaskCategory.DATA_ANALYSIS: "analysis",
            TaskCategory.MATH_STATISTICS: "math",
            TaskCategory.BIOMEDICAL: "biomedical",
            TaskCategory.DOCUMENTATION: "docs",
            TaskCategory.EMBEDDINGS: "embeddings",
            TaskCategory.SAFETY: "safety",
        }
        
        # Cloud fallback preferences
        cloud_fallbacks = {
            TaskCategory.INTENT_PARSING: CloudProvider.DEEPSEEK,
            TaskCategory.CODE_GENERATION: CloudProvider.DEEPSEEK,
            TaskCategory.CODE_VALIDATION: CloudProvider.DEEPSEEK,
            TaskCategory.DATA_ANALYSIS: CloudProvider.DEEPSEEK,
            TaskCategory.MATH_STATISTICS: CloudProvider.DEEPSEEK,
            TaskCategory.BIOMEDICAL: CloudProvider.ANTHROPIC,  # Claude is good at biology
            TaskCategory.DOCUMENTATION: CloudProvider.ANTHROPIC,
            TaskCategory.EMBEDDINGS: CloudProvider.OPENAI,
            TaskCategory.SAFETY: CloudProvider.ANTHROPIC,
            TaskCategory.ORCHESTRATION: CloudProvider.DEEPSEEK,  # Cloud-only
        }
        
        for category in TaskCategory:
            model_key = category_to_model.get(category)
            local_eps = self.local_endpoints.get(model_key, []) if model_key else []
            
            self.routes[category] = TaskRouteConfig(
                category=category,
                local_endpoints=local_eps,
                cloud_fallback=cloud_fallbacks.get(category),
                prefer_local=category != TaskCategory.ORCHESTRATION,
            )
    
    def get_endpoint(self, category: TaskCategory) -> Optional[ModelEndpoint]:
        """Get the best endpoint for a task category (with load balancing)."""
        route = self.routes.get(category)
        if not route:
            return None
        
        # Filter healthy local endpoints
        healthy_local = [ep for ep in route.local_endpoints if ep.is_healthy]
        
        if healthy_local and route.prefer_local:
            # Load balance: prefer endpoint with lowest latency, with some randomness
            healthy_local.sort(key=lambda ep: ep.latency_ms + random.uniform(0, 50))
            return healthy_local[0]
        
        # Fallback to cloud
        return route.cloud_fallback
    
    async def health_check(self, endpoint: ModelEndpoint) -> bool:
        """Check if an endpoint is healthy."""
        if endpoint.is_cloud:
            return True  # Assume cloud is always available
        
        try:
            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint.url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        latency = (time.time() - start) * 1000
                        endpoint.mark_healthy(latency)
                        return True
        except Exception:
            pass
        
        endpoint.mark_unhealthy()
        return False
    
    async def health_check_all(self):
        """Check health of all endpoints."""
        tasks = []
        for route in self.routes.values():
            for endpoint in route.local_endpoints:
                tasks.append(self.health_check(endpoint))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def complete(
        self,
        task: Union[str, TaskCategory],
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a completion request to the appropriate model.
        
        Args:
            task: Task category (string or enum)
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters passed to the API
        
        Returns:
            Response dictionary with 'content', 'model', 'is_local', etc.
        """
        # Convert string to enum
        if isinstance(task, str):
            task = TaskCategory(task) if task in [t.value for t in TaskCategory] else \
                   TaskCategory[task.upper()]
        
        route = self.routes.get(task)
        if not route:
            raise ValueError(f"Unknown task category: {task}")
        
        # Try endpoints with retries
        last_error = None
        endpoints_tried = []
        
        for attempt in range(route.max_retries + 1):
            endpoint = self.get_endpoint(task)
            if not endpoint or endpoint in endpoints_tried:
                # Try cloud fallback
                endpoint = route.cloud_fallback
            
            if not endpoint:
                break
            
            endpoints_tried.append(endpoint)
            
            try:
                result = await self._call_endpoint(
                    endpoint, prompt, max_tokens, temperature, **kwargs
                )
                result["is_local"] = not endpoint.is_cloud
                result["model_name"] = endpoint.name
                result["endpoint_url"] = endpoint.url
                return result
                
            except Exception as e:
                last_error = e
                endpoint.mark_unhealthy()
                continue
        
        raise RuntimeError(
            f"All endpoints failed for {task.value}. "
            f"Tried: {[ep.name for ep in endpoints_tried]}. "
            f"Last error: {last_error}"
        )
    
    async def _call_endpoint(
        self,
        endpoint: ModelEndpoint,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Make API call to endpoint."""
        
        headers = {"Content-Type": "application/json"}
        
        # Add API key for cloud providers
        if endpoint.is_cloud:
            if "deepseek" in endpoint.name:
                headers["Authorization"] = f"Bearer {self.api_keys['deepseek']}"
            elif "gpt" in endpoint.name or "openai" in endpoint.url:
                headers["Authorization"] = f"Bearer {self.api_keys['openai']}"
            elif "claude" in endpoint.name or "anthropic" in endpoint.url:
                headers["x-api-key"] = self.api_keys["anthropic"]
                headers["anthropic-version"] = "2023-06-01"
        
        # Build request body
        body = {
            "model": endpoint.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Anthropic has different API format
        if "anthropic" in endpoint.url:
            body = {
                "model": endpoint.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            url = f"{endpoint.url}/messages"
        else:
            url = f"{endpoint.url}/chat/completions"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"API error {response.status}: {error_text}")
                
                data = await response.json()
        
        # Extract content based on API format
        if "anthropic" in endpoint.url:
            content = data["content"][0]["text"]
        else:
            content = data["choices"][0]["message"]["content"]
        
        return {
            "content": content,
            "model": endpoint.model_id,
            "usage": data.get("usage", {}),
            "raw_response": data,
        }
    
    async def embed(
        self,
        texts: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """Generate embeddings using the embeddings model."""
        
        if isinstance(texts, str):
            texts = [texts]
        
        endpoint = self.get_endpoint(TaskCategory.EMBEDDINGS)
        if not endpoint:
            raise RuntimeError("No embeddings endpoint available")
        
        headers = {"Content-Type": "application/json"}
        if endpoint.is_cloud:
            headers["Authorization"] = f"Bearer {self.api_keys['openai']}"
        
        body = {
            "model": endpoint.model_id,
            "input": texts,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint.url}/embeddings",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Embeddings error: {error_text}")
                
                data = await response.json()
        
        return {
            "embeddings": [d["embedding"] for d in data["data"]],
            "model": endpoint.model_id,
            "is_local": not endpoint.is_cloud,
        }
    
    def status(self) -> Dict[str, Any]:
        """Get status of all routes and endpoints."""
        status = {}
        
        for category, route in self.routes.items():
            local_status = []
            for ep in route.local_endpoints:
                local_status.append({
                    "name": ep.name,
                    "url": ep.url,
                    "healthy": ep.is_healthy,
                    "latency_ms": ep.latency_ms,
                    "errors": ep.error_count,
                })
            
            cloud_status = None
            if route.cloud_fallback:
                cloud_status = {
                    "name": route.cloud_fallback.name,
                    "url": route.cloud_fallback.url,
                }
            
            status[category.value] = {
                "local_endpoints": local_status,
                "cloud_fallback": cloud_status,
                "prefer_local": route.prefer_local,
            }
        
        return status


# ============================================================================
# FastAPI Service (optional, for centralized routing)
# ============================================================================

def create_app():
    """Create FastAPI app for the router service."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        return None
    
    app = FastAPI(
        title="BioPipelines T4 Model Router",
        description="Routes requests to specialized models on T4 GPUs",
        version="1.0.0",
    )
    
    router = T4ModelRouter()
    
    class CompletionRequest(BaseModel):
        task: str
        prompt: str
        max_tokens: int = 1024
        temperature: float = 0.7
    
    class EmbeddingRequest(BaseModel):
        texts: List[str]
    
    @app.on_event("startup")
    async def startup():
        await router.health_check_all()
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    @app.get("/status")
    async def status():
        return router.status()
    
    @app.post("/v1/complete")
    async def complete(request: CompletionRequest):
        try:
            result = await router.complete(
                task=request.task,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/v1/embed")
    async def embed(request: EmbeddingRequest):
        try:
            result = await router.embed(request.texts)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="T4 Model Router")
    parser.add_argument("command", choices=["status", "serve", "test"])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    if args.command == "status":
        router = T4ModelRouter()
        print(json.dumps(router.status(), indent=2))
    
    elif args.command == "serve":
        app = create_app()
        if app:
            import uvicorn
            uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.command == "test":
        async def test():
            router = T4ModelRouter()
            await router.health_check_all()
            
            # Test intent parsing
            result = await router.complete(
                TaskCategory.INTENT_PARSING,
                "I want to analyze RNA-seq data from human liver samples"
            )
            print(f"Intent result ({result['model_name']}):")
            print(result["content"][:200] + "...")
        
        asyncio.run(test())
