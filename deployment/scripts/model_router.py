#!/usr/bin/env python3
"""
BioPipelines: Multi-Node vLLM Router Service

This FastAPI service routes requests to the appropriate vLLM backend
based on task category. It reads the active server registry and
provides health-aware load balancing.

Usage:
    python model_router.py --port 8080 --registry configs/active_servers.json

Endpoints:
    POST /v1/chat/completions - Route to appropriate model
    POST /v1/completions - Route to appropriate model
    GET /health - Router health check
    GET /servers - List active servers
    GET /metrics - Prometheus-compatible metrics
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_router")


class TaskCategory(str, Enum):
    """Task categories matching the vLLM server array."""
    INTENT_PARSING = "0"      # Llama-3.2-3B
    CODE_GENERATION = "1"     # Qwen2.5-Coder-7B
    CODE_VALIDATION = "2"     # Qwen2.5-Coder-1.5B
    DATA_ANALYSIS = "3"       # Phi-3.5-mini
    ORCHESTRATION = "4"       # Qwen2.5-3B
    DOCUMENTATION = "5"       # Phi-3-mini
    MATH_STATISTICS = "6"     # Qwen2.5-Math-1.5B
    BIO_MEDICAL = "7"         # BioMistral-7B
    GENERAL = "8"             # Gemma-2-2b
    BACKUP = "9"              # Llama-3.2-1B


# Mapping from model name patterns to task categories
MODEL_TASK_MAP = {
    "llama-3.2-3b": TaskCategory.INTENT_PARSING,
    "qwen2.5-coder-7b": TaskCategory.CODE_GENERATION,
    "qwen2.5-coder-1.5b": TaskCategory.CODE_VALIDATION,
    "phi-3.5": TaskCategory.DATA_ANALYSIS,
    "qwen2.5-3b": TaskCategory.ORCHESTRATION,
    "phi-3-mini": TaskCategory.DOCUMENTATION,
    "qwen2.5-math": TaskCategory.MATH_STATISTICS,
    "biomistral": TaskCategory.BIO_MEDICAL,
    "gemma": TaskCategory.GENERAL,
    "llama-3.2-1b": TaskCategory.BACKUP,
}


@dataclass
class ServerInfo:
    """Information about a vLLM server instance."""
    task_id: str
    model: str
    host: str
    port: int
    job_id: str
    status: str = "unknown"
    last_health: Optional[datetime] = None
    latency_ms: float = 0.0
    request_count: int = 0
    error_count: int = 0


@dataclass
class RouterMetrics:
    """Router-wide metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    requests_by_task: dict = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    task_category: Optional[str] = None  # BioPipelines extension


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str
    prompt: str | list[str]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    task_category: Optional[str] = None  # BioPipelines extension


class ModelRouter:
    """Routes requests to appropriate vLLM backends."""
    
    def __init__(self, registry_path: Path, health_interval: int = 30):
        self.registry_path = registry_path
        self.health_interval = health_interval
        self.servers: dict[str, ServerInfo] = {}
        self.metrics = RouterMetrics()
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self._health_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the router and background tasks."""
        await self._load_registry()
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Router started with {len(self.servers)} servers")
        
    async def stop(self):
        """Stop the router and cleanup."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        await self.http_client.aclose()
        logger.info("Router stopped")
        
    async def _load_registry(self):
        """Load server registry from JSON file."""
        try:
            if self.registry_path.exists():
                data = json.loads(self.registry_path.read_text())
                for task_id, info in data.get("servers", {}).items():
                    self.servers[task_id] = ServerInfo(
                        task_id=task_id,
                        model=info["model"],
                        host=info["host"],
                        port=info["port"],
                        job_id=info.get("job_id", "unknown"),
                        status=info.get("status", "unknown"),
                    )
                logger.info(f"Loaded {len(self.servers)} servers from registry")
            else:
                logger.warning(f"Registry file not found: {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            
    async def _health_check_loop(self):
        """Periodically check health of all servers."""
        while True:
            try:
                await asyncio.sleep(self.health_interval)
                await self._load_registry()  # Reload in case of updates
                
                for task_id, server in self.servers.items():
                    await self._check_server_health(server)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                
    async def _check_server_health(self, server: ServerInfo):
        """Check health of a single server."""
        url = f"http://{server.host}:{server.port}/health"
        try:
            start = time.monotonic()
            response = await self.http_client.get(url, timeout=5.0)
            latency = (time.monotonic() - start) * 1000
            
            if response.status_code == 200:
                server.status = "healthy"
                server.latency_ms = latency
                server.last_health = datetime.now()
            else:
                server.status = "unhealthy"
                logger.warning(f"Server {server.task_id} unhealthy: {response.status_code}")
                
        except Exception as e:
            server.status = "unreachable"
            logger.warning(f"Server {server.task_id} unreachable: {e}")
            
    def _resolve_task_category(
        self, 
        model_name: str, 
        explicit_category: Optional[str]
    ) -> TaskCategory:
        """Determine which task category to route to."""
        # Use explicit category if provided
        if explicit_category:
            try:
                return TaskCategory(explicit_category)
            except ValueError:
                # Try by name
                for cat in TaskCategory:
                    if cat.name.lower() == explicit_category.lower():
                        return cat
                        
        # Infer from model name
        model_lower = model_name.lower()
        for pattern, category in MODEL_TASK_MAP.items():
            if pattern in model_lower:
                return category
                
        # Default to general
        return TaskCategory.GENERAL
        
    def _get_server_for_task(self, category: TaskCategory) -> Optional[ServerInfo]:
        """Get the server for a task category, with fallback."""
        task_id = category.value
        
        # Try primary server
        server = self.servers.get(task_id)
        if server and server.status == "healthy":
            return server
            
        # Fallback to backup
        backup = self.servers.get(TaskCategory.BACKUP.value)
        if backup and backup.status == "healthy":
            logger.warning(f"Using backup for task {category.name}")
            return backup
            
        # Last resort: any healthy server
        for s in self.servers.values():
            if s.status == "healthy":
                logger.warning(f"Using fallback {s.task_id} for task {category.name}")
                return s
                
        return None
        
    async def route_request(
        self,
        path: str,
        method: str,
        body: dict,
        task_category: Optional[str] = None
    ) -> tuple[int, dict]:
        """Route a request to the appropriate backend."""
        self.metrics.total_requests += 1
        
        # Determine model and category
        model_name = body.get("model", "")
        category = self._resolve_task_category(model_name, task_category)
        
        # Track by category
        cat_name = category.name
        self.metrics.requests_by_task[cat_name] = \
            self.metrics.requests_by_task.get(cat_name, 0) + 1
            
        # Get server
        server = self._get_server_for_task(category)
        if not server:
            self.metrics.failed_requests += 1
            return 503, {
                "error": f"No healthy server for task {category.name}",
                "available_servers": [
                    {"task": s.task_id, "status": s.status}
                    for s in self.servers.values()
                ]
            }
            
        # Update model name to match backend
        body["model"] = server.model
        
        # Forward request
        url = f"http://{server.host}:{server.port}{path}"
        try:
            start = time.monotonic()
            
            if method.upper() == "POST":
                response = await self.http_client.post(
                    url, 
                    json=body,
                    timeout=120.0
                )
            else:
                response = await self.http_client.get(url, timeout=30.0)
                
            latency = (time.monotonic() - start) * 1000
            server.request_count += 1
            server.latency_ms = (server.latency_ms + latency) / 2  # Moving average
            
            self.metrics.successful_requests += 1
            return response.status_code, response.json()
            
        except Exception as e:
            server.error_count += 1
            self.metrics.failed_requests += 1
            logger.error(f"Request to {server.task_id} failed: {e}")
            return 500, {"error": str(e)}


# Create FastAPI app
app = FastAPI(
    title="BioPipelines Model Router",
    description="Routes LLM requests to appropriate vLLM backends",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global router instance
router: Optional[ModelRouter] = None


@app.on_event("startup")
async def startup():
    """Initialize router on startup."""
    global router
    registry_path = Path(os.environ.get(
        "REGISTRY_PATH",
        "/home/sdodl001_odu_edu/BioPipelines/deployment/configs/active_servers.json"
    ))
    router = ModelRouter(registry_path)
    await router.start()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global router
    if router:
        await router.stop()


@app.get("/health")
async def health():
    """Router health check."""
    if not router:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    
    healthy_count = sum(1 for s in router.servers.values() if s.status == "healthy")
    return {
        "status": "healthy" if healthy_count > 0 else "degraded",
        "healthy_backends": healthy_count,
        "total_backends": len(router.servers),
    }


@app.get("/servers")
async def list_servers():
    """List all registered servers and their status."""
    if not router:
        return JSONResponse(status_code=503, content={"error": "not_ready"})
        
    return {
        "servers": [
            {
                "task_id": s.task_id,
                "task_name": TaskCategory(s.task_id).name,
                "model": s.model,
                "host": s.host,
                "port": s.port,
                "status": s.status,
                "latency_ms": round(s.latency_ms, 2),
                "request_count": s.request_count,
                "error_count": s.error_count,
            }
            for s in router.servers.values()
        ]
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics."""
    if not router:
        return JSONResponse(status_code=503, content={"error": "not_ready"})
        
    uptime = (datetime.now() - router.metrics.start_time).total_seconds()
    
    lines = [
        "# HELP router_requests_total Total requests processed",
        "# TYPE router_requests_total counter",
        f"router_requests_total {router.metrics.total_requests}",
        "",
        "# HELP router_requests_success Successful requests",
        "# TYPE router_requests_success counter",
        f"router_requests_success {router.metrics.successful_requests}",
        "",
        "# HELP router_requests_failed Failed requests",
        "# TYPE router_requests_failed counter",
        f"router_requests_failed {router.metrics.failed_requests}",
        "",
        "# HELP router_uptime_seconds Router uptime",
        "# TYPE router_uptime_seconds gauge",
        f"router_uptime_seconds {uptime:.0f}",
        "",
        "# HELP router_backends_healthy Number of healthy backends",
        "# TYPE router_backends_healthy gauge",
        f'router_backends_healthy {sum(1 for s in router.servers.values() if s.status == "healthy")}',
    ]
    
    # Per-task metrics
    for task, count in router.metrics.requests_by_task.items():
        lines.append(f'router_requests_by_task{{task="{task}"}} {count}')
        
    return Response(content="\n".join(lines), media_type="text/plain")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if not router:
        raise HTTPException(503, "Router not ready")
        
    status, response = await router.route_request(
        path="/v1/chat/completions",
        method="POST",
        body=request.model_dump(),
        task_category=request.task_category
    )
    
    if status >= 400:
        raise HTTPException(status, response)
    return response


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    if not router:
        raise HTTPException(503, "Router not ready")
        
    status, response = await router.route_request(
        path="/v1/completions",
        method="POST",
        body=request.model_dump(),
        task_category=request.task_category
    )
    
    if status >= 400:
        raise HTTPException(status, response)
    return response


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def catch_all(path: str, request: Request):
    """Forward any other requests to an appropriate backend."""
    if not router:
        raise HTTPException(503, "Router not ready")
        
    body = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except:
            pass
            
    status, response = await router.route_request(
        path=f"/{path}",
        method=request.method,
        body=body
    )
    
    if status >= 400:
        raise HTTPException(status, response)
    return response


def main():
    """Run the router service."""
    parser = argparse.ArgumentParser(description="BioPipelines Model Router")
    parser.add_argument("--port", type=int, default=8080, help="Router port")
    parser.add_argument("--host", default="0.0.0.0", help="Router host")
    parser.add_argument("--registry", default="configs/active_servers.json",
                       help="Path to server registry")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()
    
    os.environ["REGISTRY_PATH"] = str(Path(args.registry).absolute())
    
    uvicorn.run(
        "model_router:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
