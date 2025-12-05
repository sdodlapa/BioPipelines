# BioPipelines Local Model Implementation Guide
## Step-by-Step Deployment Plans for Multi-Model Inference

**Last Updated:** December 5, 2025  
**Related:** [LOCAL_MODEL_STRATEGY.md](./LOCAL_MODEL_STRATEGY.md)  
**Status:** Ready for Deployment  
**Recommended Path:** T4-Only + Cloud Hybrid (Path A2)

---

## Table of Contents

1. [Quick Start: T4-Only + Cloud](#quick-start-t4-only--cloud)
2. [Implementation Paths Overview](#implementation-paths-overview)
3. [Path A: Immediate - H100 + T4](#path-a-immediate---h100--t4)
4. [Path A2: Recommended - 10× T4 + Cloud](#path-a2-recommended---10-t4--cloud)
5. [Path B: Future - 4× L4 + 4× T4](#path-b-future---4-l4--4-t4)
6. [Path C: Alternative - Multi-H100](#path-c-alternative---multi-h100)
7. [Path D: Budget - T4 Only](#path-d-budget---t4-only)
8. [Common Infrastructure Setup](#common-infrastructure-setup)
9. [Model Serving Options](#model-serving-options)
10. [Monitoring & Observability](#monitoring--observability)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start: T4-Only + Cloud

This is the **recommended approach** for BioPipelines. It uses models that fit on a single T4 GPU (16GB) for fast, free inference, with cloud API fallback (DeepSeek-V3 at $0.27/M tokens) for complex tasks.

### Why This Approach?

| Aspect | Local T4 | Cloud Fallback |
|--------|----------|----------------|
| **Cost** | Free (power only) | $0.27-3.00/M tokens |
| **Latency** | ~100-500ms | ~500-2000ms |
| **Privacy** | Full control | Data leaves premises |
| **Availability** | Depends on SLURM | 99.9%+ |
| **Model Size** | ≤7B (FP16), ≤14B (INT8) | Any size |

### 30-Minute Quick Deploy

```bash
# 1. Clone the deployment scripts
cd ~/BioPipelines

# 2. Deploy all T4 models
./scripts/llm/deploy_t4_fleet.sh start

# 3. Check status
./scripts/llm/deploy_t4_fleet.sh status

# 4. Start the router (in a separate terminal or as service)
python -m src.workflow_composer.providers.t4_router serve --port 8080
```

### Model Allocation Summary

| GPU | Model | VRAM Used | Task |
|-----|-------|-----------|------|
| T4-1 | Llama-3.2-3B + Llama-Guard | 9.5GB | Intent + Safety |
| T4-2 | Llama-3.2-3B + BGE-M3 | 8.5GB | Intent + Embeddings |
| T4-3 | Qwen2.5-Coder-7B (INT8) | 11GB | Code Gen + Validation |
| T4-4 | Qwen2.5-Coder-7B (INT8) | 8GB | Code Gen (replica) |
| T4-5 | Phi-3.5-mini | 8GB | Data Analysis |
| T4-6 | Qwen2.5-Math-7B (INT8) | 7.5GB | Math/Statistics |
| T4-7 | BioMistral-7B (INT8) | 7.5GB | Bio/Medical |
| T4-8 | Gemma-2-9B (INT8) | 9.5GB | Documentation |
| T4-9 | Reserved | - | Hot spare |
| T4-10 | Reserved | - | Hot spare |

### Cloud Fallback Configuration

Set up API keys for fallback:

```bash
# Add to ~/.bashrc or environment
export DEEPSEEK_API_KEY="sk-..."      # Primary fallback ($0.27/M)
export ANTHROPIC_API_KEY="sk-ant-..." # For bio/medical (Claude knows biology)
export OPENAI_API_KEY="sk-..."        # For embeddings fallback
```

---

## Implementation Paths Overview

| Path | Configuration | VRAM | Status | Timeline | Best For |
|------|---------------|------|--------|----------|----------|
| **A** | 1× H100 + 2× T4 | 112GB | ✅ Ready | Week 1-2 | Immediate deployment |
| **A2** | 10× T4 (multi-node) | 160GB | ✅ Ready | Week 1-2 | No H100, max T4s |
| **B** | 4× L4 + 4× T4 | 160GB | ⏳ Pending L4s | Month 2+ | Full scaling |
| **C** | 2× H100 | 160GB | ✅ Available | As needed | High redundancy |
| **D** | 4× T4 | 64GB | ✅ Available | Anytime | Budget/testing |

---

## Understanding Multi-Node GPU Communication

### What vLLM Supports

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    vLLM Multi-GPU Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ✅ SUPPORTED: Tensor Parallelism WITHIN a Node                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Single Node                                    │  │
│  │   GPU 0 ◄──────────► GPU 1 ◄──────────► GPU 2 ◄──────────► GPU 3     │  │
│  │          NVLink/PCIe           NVLink/PCIe           NVLink/PCIe      │  │
│  │                    (Fast: 600+ GB/s with NVLink)                      │  │
│  │                                                                       │  │
│  │   vllm serve MODEL --tensor-parallel-size 4                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ❌ NOT PRACTICAL: Tensor Parallelism ACROSS Nodes                          │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   Node 1                              Node 2                          │  │
│  │   ┌─────────┐                        ┌─────────┐                      │  │
│  │   │ GPU 0   │◄─────── Network ──────►│ GPU 0   │                      │  │
│  │   └─────────┘      (Slow: 10-100 Gbps)└─────────┘                     │  │
│  │                                                                       │  │
│  │   Too slow for tensor parallelism! (100x slower than NVLink)         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ✅ SOLUTION: Independent Servers + Load Balancer                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │   Node 1          Node 2          Node 3          Node N             │  │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │  │
│  │   │ vLLM    │    │ vLLM    │    │ vLLM    │    │ vLLM    │          │  │
│  │   │ :8000   │    │ :8000   │    │ :8000   │    │ :8000   │          │  │
│  │   └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘          │  │
│  │        │              │              │              │                │  │
│  │        └──────────────┴──────────────┴──────────────┘                │  │
│  │                            │                                          │  │
│  │                    ┌───────▼───────┐                                 │  │
│  │                    │ Load Balancer │                                 │  │
│  │                    │    Router     │                                 │  │
│  │                    └───────────────┘                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Node Strategy for 10× T4

Since each T4 node has **1 GPU (16GB)**, we run:
- **10 independent vLLM servers** (one per node)
- **1 Load Balancer/Router** to distribute requests
- **Task-based routing** to send requests to appropriate model

---

## Path A2: 10× T4 Multi-Node Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PATH A2: 10× T4 Multi-Node (160GB)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │ T4-0    │ │ T4-1    │ │ T4-2    │ │ T4-3    │ │ T4-4    │               │
│  │ 16GB    │ │ 16GB    │ │ 16GB    │ │ 16GB    │ │ 16GB    │               │
│  │         │ │         │ │         │ │         │ │         │               │
│  │Qwen-7B  │ │Qwen-7B  │ │Llama-3B │ │Phi-3.5  │ │Math-7B  │               │
│  │Coder    │ │Coder    │ │Intent   │ │Analysis │ │INT8     │               │
│  │INT8     │ │INT8     │ │         │ │         │ │         │               │
│  │(replica)│ │(replica)│ │         │ │         │ │         │               │
│  │:8000    │ │:8000    │ │:8000    │ │:8000    │ │:8000    │               │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘               │
│       │          │          │          │          │                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │ T4-5    │ │ T4-6    │ │ T4-7    │ │ T4-8    │ │ T4-9    │               │
│  │ 16GB    │ │ 16GB    │ │ 16GB    │ │ 16GB    │ │ 16GB    │               │
│  │         │ │         │ │         │ │         │ │         │               │
│  │BioMist  │ │BioMist  │ │BGE-M3 + │ │Safety   │ │Orchestr │               │
│  │ral-7B  │ │ral-7B   │ │Llama-3B │ │8B INT8  │ │8B INT8  │               │
│  │INT8     │ │INT8     │ │(backup) │ │         │ │         │               │
│  │(replica)│ │(replica)│ │         │ │         │ │         │               │
│  │:8000    │ │:8000    │ │:8000    │ │:8000    │ │:8000    │               │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘               │
│       │          │          │          │          │                        │
│       └──────────┴──────────┴──────────┴──────────┘                        │
│                            │                                                │
│                    ┌───────▼───────┐                                       │
│                    │    ROUTER     │ (runs on CPU node)                    │
│                    │    :8080      │                                       │
│                    └───────────────┘                                       │
│                                                                              │
│  Model Distribution:                                                        │
│  • Coder: T4-0, T4-1 (2 replicas for load balancing)                       │
│  • Intent: T4-2                                                             │
│  • Analysis: T4-3                                                           │
│  • Math: T4-4                                                               │
│  • Bio: T4-5, T4-6 (2 replicas)                                            │
│  • Embeddings + Backup: T4-7                                                │
│  • Safety: T4-8                                                             │
│  • Orchestrator: T4-9                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Allocation Plan

| Node | GPU | Model | VRAM | Role | Port |
|------|-----|-------|------|------|------|
| T4-0 | 0 | Qwen2.5-Coder-7B (INT8) | ~8GB | Code Gen (primary) | 8000 |
| T4-1 | 0 | Qwen2.5-Coder-7B (INT8) | ~8GB | Code Gen (replica) | 8000 |
| T4-2 | 0 | Llama-3.2-3B-Instruct | ~7GB | Intent Parsing | 8000 |
| T4-3 | 0 | Phi-3.5-mini-instruct | ~8GB | Data Analysis | 8000 |
| T4-4 | 0 | Qwen2.5-Math-7B (INT8) | ~8GB | Math/Statistics | 8000 |
| T4-5 | 0 | BioMistral-7B (INT8) | ~8GB | Bio (primary) | 8000 |
| T4-6 | 0 | BioMistral-7B (INT8) | ~8GB | Bio (replica) | 8000 |
| T4-7 | 0 | Llama-3.2-3B + BGE-M3 | ~9GB | Backup + Embeddings | 8000, 8001 |
| T4-8 | 0 | Nemotron-Safety-8B (INT8) | ~9GB | Safety Filter | 8000 |
| T4-9 | 0 | Nemotron-8B-Orch (INT8) | ~9GB | Orchestration | 8000 |

### Phase 1: SLURM Job Array for All T4 Nodes

The most efficient way to manage 10 nodes is using a **SLURM job array**:

```bash
cat > deployment/scripts/start_t4_array.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=biopipe-t4
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=deployment/logs/t4-%A-%a.out
#SBATCH --error=deployment/logs/t4-%A-%a.err

# ============================================================================
# BioPipelines Multi-Node T4 Deployment
# Uses SLURM job array to start 10 independent vLLM servers
# ============================================================================

source ~/.bashrc
conda activate biopipe-inference

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/$USER/hf_cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Get task ID (0-9)
TASK_ID=$SLURM_ARRAY_TASK_ID
NODE_HOST=$(hostname)

# Log node info for router discovery
echo "${TASK_ID}:${NODE_HOST}" >> deployment/logs/t4_nodes.txt

# Model configuration based on task ID
case $TASK_ID in
    0)
        # Coder replica 1
        MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
        MODEL_NAME="coder"
        QUANTIZATION="awq"
        MAX_LEN=8192
        GPU_UTIL=0.85
        ;;
    1)
        # Coder replica 2
        MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
        MODEL_NAME="coder"
        QUANTIZATION="awq"
        MAX_LEN=8192
        GPU_UTIL=0.85
        ;;
    2)
        # Intent parsing
        MODEL="meta-llama/Llama-3.2-3B-Instruct"
        MODEL_NAME="intent"
        QUANTIZATION=""
        MAX_LEN=4096
        GPU_UTIL=0.5
        ;;
    3)
        # Data analysis
        MODEL="microsoft/Phi-3.5-mini-instruct"
        MODEL_NAME="analysis"
        QUANTIZATION=""
        MAX_LEN=4096
        GPU_UTIL=0.55
        ;;
    4)
        # Math
        MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
        MODEL_NAME="math"
        QUANTIZATION="awq"
        MAX_LEN=4096
        GPU_UTIL=0.85
        ;;
    5)
        # Bio replica 1
        MODEL="BioMistral/BioMistral-7B"
        MODEL_NAME="bio"
        QUANTIZATION="awq"
        MAX_LEN=4096
        GPU_UTIL=0.85
        ;;
    6)
        # Bio replica 2
        MODEL="BioMistral/BioMistral-7B"
        MODEL_NAME="bio"
        QUANTIZATION="awq"
        MAX_LEN=4096
        GPU_UTIL=0.85
        ;;
    7)
        # Backup intent + embeddings server
        MODEL="meta-llama/Llama-3.2-3B-Instruct"
        MODEL_NAME="backup"
        QUANTIZATION=""
        MAX_LEN=4096
        GPU_UTIL=0.45
        START_EMBEDDING=true
        ;;
    8)
        # Safety
        MODEL="nvidia/Llama-3.1-Nemotron-Safety-8B-V3"
        MODEL_NAME="safety"
        QUANTIZATION="awq"
        MAX_LEN=2048
        GPU_UTIL=0.85
        ;;
    9)
        # Orchestrator
        MODEL="nvidia/Llama-3.1-Nemotron-8B-Orchestrator"
        MODEL_NAME="orchestrator"
        QUANTIZATION="awq"
        MAX_LEN=4096
        GPU_UTIL=0.85
        ;;
esac

echo "============================================"
echo "Starting T4 node $TASK_ID: $MODEL_NAME"
echo "Host: $NODE_HOST"
echo "Model: $MODEL"
echo "Quantization: ${QUANTIZATION:-none}"
echo "============================================"

# Build vLLM command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --served-model-name $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --dtype float16 \
    --trust-remote-code"

# Add quantization if specified
if [ -n "$QUANTIZATION" ]; then
    VLLM_CMD="$VLLM_CMD --quantization $QUANTIZATION"
fi

# Start vLLM server
$VLLM_CMD &
VLLM_PID=$!

# If this is the embedding node, also start embedding server
if [ "$START_EMBEDDING" = true ]; then
    sleep 30  # Wait for vLLM to start
    python deployment/scripts/embedding_server.py \
        --model BAAI/bge-m3 \
        --port 8001 &
    EMBED_PID=$!
    echo "Started embedding server on port 8001"
fi

# Wait for main process
wait $VLLM_PID
EOF

chmod +x deployment/scripts/start_t4_array.sbatch
```

### Phase 2: Service Discovery Script

When SLURM jobs start, each node registers itself. The router needs to discover all nodes:

```bash
cat > deployment/scripts/discover_nodes.py << 'EOF'
#!/usr/bin/env python3
"""
Service Discovery for Multi-Node T4 Deployment
Reads node registrations and generates router configuration
"""

import os
import time
import json
from pathlib import Path
from collections import defaultdict

NODES_FILE = Path("deployment/logs/t4_nodes.txt")
CONFIG_FILE = Path("deployment/configs/endpoints.json")

# Model to nodes mapping (which task IDs run which model)
MODEL_MAPPING = {
    "coder": [0, 1],      # 2 replicas
    "intent": [2],
    "analysis": [3],
    "math": [4],
    "bio": [5, 6],        # 2 replicas
    "backup": [7],
    "embeddings": [7],    # Same node as backup
    "safety": [8],
    "orchestrator": [9],
}

def discover_nodes(timeout=300):
    """Wait for all nodes to register and build endpoint config."""
    print(f"Waiting for 10 T4 nodes to register (timeout: {timeout}s)...")
    
    start_time = time.time()
    nodes = {}
    
    while len(nodes) < 10 and (time.time() - start_time) < timeout:
        if NODES_FILE.exists():
            with open(NODES_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        task_id, hostname = line.split(':', 1)
                        nodes[int(task_id)] = hostname
        
        if len(nodes) < 10:
            print(f"  Found {len(nodes)}/10 nodes...")
            time.sleep(10)
    
    if len(nodes) < 10:
        print(f"WARNING: Only {len(nodes)}/10 nodes registered!")
    
    return nodes

def generate_config(nodes):
    """Generate router endpoint configuration."""
    endpoints = defaultdict(list)
    
    for model_name, task_ids in MODEL_MAPPING.items():
        for task_id in task_ids:
            if task_id in nodes:
                hostname = nodes[task_id]
                port = 8001 if model_name == "embeddings" else 8000
                endpoints[model_name].append(f"http://{hostname}:{port}")
    
    # Convert to regular dict
    config = {
        "endpoints": dict(endpoints),
        "nodes": {str(k): v for k, v in nodes.items()},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Save config
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nGenerated config: {CONFIG_FILE}")
    print(json.dumps(config, indent=2))
    
    return config

def main():
    # Clear old registrations
    if NODES_FILE.exists():
        NODES_FILE.unlink()
    NODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NODES_FILE.touch()
    
    nodes = discover_nodes()
    config = generate_config(nodes)
    
    print("\n" + "=" * 60)
    print("Service Discovery Complete!")
    print("=" * 60)
    
    for model, urls in config["endpoints"].items():
        print(f"  {model}: {len(urls)} endpoint(s)")
    
    return config

if __name__ == "__main__":
    main()
EOF

chmod +x deployment/scripts/discover_nodes.py
```

### Phase 3: Load-Balancing Router with Replica Support

```bash
cat > deployment/scripts/model_router_lb.py << 'EOF'
#!/usr/bin/env python3
"""
BioPipelines Load-Balancing Model Router
Supports multiple replicas per model with round-robin load balancing
"""

import os
import json
import asyncio
import httpx
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from collections import defaultdict
import time

app = FastAPI(title="BioPipelines Multi-Node Router")

# Load endpoint configuration
CONFIG_FILE = Path("deployment/configs/endpoints.json")

class EndpointManager:
    """Manages endpoints with load balancing and health checking."""
    
    def __init__(self):
        self.endpoints: Dict[str, List[str]] = {}
        self.healthy: Dict[str, List[str]] = {}
        self.round_robin_idx: Dict[str, int] = defaultdict(int)
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
    def load_config(self):
        """Load endpoints from config file."""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = json.load(f)
                self.endpoints = config.get("endpoints", {})
                self.healthy = {k: list(v) for k, v in self.endpoints.items()}
                print(f"Loaded {sum(len(v) for v in self.endpoints.values())} endpoints")
    
    def get_endpoint(self, model_name: str) -> Optional[str]:
        """Get next healthy endpoint for model using round-robin."""
        endpoints = self.healthy.get(model_name, [])
        if not endpoints:
            return None
        
        # Round-robin selection
        idx = self.round_robin_idx[model_name] % len(endpoints)
        self.round_robin_idx[model_name] = idx + 1
        return endpoints[idx]
    
    def get_all_endpoints(self, model_name: str) -> List[str]:
        """Get all endpoints for a model (for parallel requests)."""
        return self.healthy.get(model_name, [])
    
    async def health_check(self, client: httpx.AsyncClient):
        """Check health of all endpoints."""
        now = time.time()
        if now - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = now
        new_healthy = {}
        
        for model_name, urls in self.endpoints.items():
            healthy_urls = []
            for url in urls:
                try:
                    response = await client.get(f"{url}/health", timeout=5.0)
                    if response.status_code == 200:
                        healthy_urls.append(url)
                except Exception:
                    pass
            new_healthy[model_name] = healthy_urls
            
        self.healthy = new_healthy


# Global endpoint manager
endpoint_manager = EndpointManager()

# Task to model mapping
TASK_ROUTING = {
    "code_generation": "coder",
    "code_validation": "coder",
    "workflow_generation": "coder",
    "orchestration": "orchestrator",
    "planning": "orchestrator",
    "math": "math",
    "statistics": "math",
    "bio": "bio",
    "medical": "bio",
    "intent_parsing": "intent",
    "classification": "intent",
    "embedding": "embeddings",
    "similarity": "embeddings",
    "safety_check": "safety",
    "data_analysis": "analysis",
    "visualization": "analysis",
}


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list
    task_type: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7


# HTTP client
client = httpx.AsyncClient(timeout=120.0)


@app.on_event("startup")
async def startup():
    endpoint_manager.load_config()


async def classify_task(messages: list) -> str:
    """Use intent model to classify the task type."""
    endpoint = endpoint_manager.get_endpoint("intent")
    if not endpoint:
        return "code_generation"  # Default fallback
    
    last_message = messages[-1]["content"] if messages else ""
    
    classification_prompt = f"""Classify this request into exactly one category:
- code_generation (for code, scripts, workflows)
- math (for statistics, calculations)
- bio (for biology, genomics questions)
- data_analysis (for data interpretation, visualization)
- general (for other questions)

Request: {last_message[:500]}

Respond with only the category name:"""
    
    try:
        response = await client.post(
            f"{endpoint}/v1/chat/completions",
            json={
                "model": "intent",
                "messages": [{"role": "user", "content": classification_prompt}],
                "max_tokens": 20,
                "temperature": 0.1,
            }
        )
        result = response.json()
        category = result["choices"][0]["message"]["content"].strip().lower()
        
        for task_type in TASK_ROUTING:
            if task_type in category or category in task_type:
                return task_type
        return "code_generation"
    except Exception as e:
        print(f"Classification failed: {e}")
        return "code_generation"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Route chat completions with load balancing."""
    
    # Health check (non-blocking, periodic)
    asyncio.create_task(endpoint_manager.health_check(client))
    
    # Determine model
    if request.task_type:
        model_name = TASK_ROUTING.get(request.task_type, "coder")
    elif request.model != "auto" and request.model in endpoint_manager.endpoints:
        model_name = request.model
    else:
        task_type = await classify_task(request.messages)
        model_name = TASK_ROUTING.get(task_type, "coder")
    
    # Get endpoint (load balanced)
    endpoint = endpoint_manager.get_endpoint(model_name)
    if not endpoint:
        # Try backup
        endpoint = endpoint_manager.get_endpoint("backup")
        if not endpoint:
            raise HTTPException(status_code=503, detail=f"No healthy endpoints for {model_name}")
    
    try:
        response = await client.post(
            f"{endpoint}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
        )
        result = response.json()
        result["_routing"] = {
            "model": model_name,
            "endpoint": endpoint,
        }
        return result
    except httpx.RequestError as e:
        # Mark endpoint as unhealthy and retry
        if model_name in endpoint_manager.healthy:
            endpoint_manager.healthy[model_name] = [
                u for u in endpoint_manager.healthy[model_name] if u != endpoint
            ]
        raise HTTPException(status_code=503, detail=f"Request failed: {e}")


@app.post("/v1/embeddings")
async def embeddings(request: Dict):
    """Forward embedding requests."""
    endpoint = endpoint_manager.get_endpoint("embeddings")
    if not endpoint:
        raise HTTPException(status_code=503, detail="No embedding endpoint available")
    
    try:
        response = await client.post(f"{endpoint}/v1/embeddings", json=request)
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Embedding request failed: {e}")


@app.get("/health")
async def health():
    """Comprehensive health check."""
    await endpoint_manager.health_check(client)
    
    status = {}
    for model, endpoints in endpoint_manager.endpoints.items():
        healthy = endpoint_manager.healthy.get(model, [])
        status[model] = {
            "total": len(endpoints),
            "healthy": len(healthy),
            "endpoints": healthy,
        }
    
    all_healthy = all(s["healthy"] > 0 for s in status.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "models": status,
    }


@app.get("/models")
async def list_models():
    """List available models and their endpoints."""
    return {
        "models": list(endpoint_manager.endpoints.keys()),
        "endpoints": {
            k: len(v) for k, v in endpoint_manager.endpoints.items()
        },
        "routing": TASK_ROUTING,
    }


@app.post("/reload")
async def reload_config():
    """Reload endpoint configuration."""
    endpoint_manager.load_config()
    return {"status": "reloaded", "endpoints": len(endpoint_manager.endpoints)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

chmod +x deployment/scripts/model_router_lb.py
```

### Phase 4: Master Launch Script

```bash
cat > deployment/scripts/launch_10t4.sh << 'EOF'
#!/bin/bash
# ============================================================================
# BioPipelines 10× T4 Multi-Node Launch Script
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$DEPLOY_DIR/logs"

echo "============================================"
echo "BioPipelines 10× T4 Multi-Node Deployment"
echo "============================================"

# Step 1: Clean up old state
echo ""
echo "[1/5] Cleaning up old state..."
rm -f "$LOG_DIR/t4_nodes.txt"
rm -f "$DEPLOY_DIR/configs/endpoints.json"
mkdir -p "$LOG_DIR"
mkdir -p "$DEPLOY_DIR/configs"
touch "$LOG_DIR/t4_nodes.txt"

# Step 2: Cancel any existing jobs
echo ""
echo "[2/5] Canceling existing BioPipelines jobs..."
scancel --name=biopipe-t4 2>/dev/null || true
scancel --name=biopipe-router 2>/dev/null || true
sleep 5

# Step 3: Submit T4 job array
echo ""
echo "[3/5] Submitting T4 job array (10 nodes)..."
T4_JOB=$(sbatch --parsable "$SCRIPT_DIR/start_t4_array.sbatch")
echo "  Submitted job array: $T4_JOB"

# Step 4: Wait for nodes and run service discovery
echo ""
echo "[4/5] Waiting for nodes to start (this may take 2-5 minutes)..."
sleep 60  # Initial wait for SLURM scheduling

# Run discovery with timeout
timeout 300 python "$SCRIPT_DIR/discover_nodes.py" || {
    echo "ERROR: Service discovery timed out!"
    echo "Check job status with: squeue -u $USER"
    exit 1
}

# Step 5: Start router
echo ""
echo "[5/5] Starting load-balancing router..."
sbatch --job-name=biopipe-router \
       --partition=cpuspot \
       --cpus-per-task=4 \
       --mem=16G \
       --time=24:00:00 \
       --output="$LOG_DIR/router-%j.out" \
       --wrap="source ~/.bashrc && conda activate biopipe-inference && python $SCRIPT_DIR/model_router_lb.py"

echo ""
echo "============================================"
echo "Deployment initiated!"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f $LOG_DIR/t4-*.out"
echo "  tail -f $LOG_DIR/router-*.out"
echo ""
echo "Test with:"
echo "  curl http://\$(hostname):8080/health"
echo "  python $SCRIPT_DIR/test_deployment.py"
echo ""
EOF

chmod +x deployment/scripts/launch_10t4.sh
```

### Phase 5: Shutdown Script

```bash
cat > deployment/scripts/shutdown_10t4.sh << 'EOF'
#!/bin/bash
# ============================================================================
# BioPipelines Shutdown Script
# ============================================================================

echo "Canceling all BioPipelines jobs..."
scancel --name=biopipe-t4
scancel --name=biopipe-router

echo "Cleaning up state files..."
rm -f deployment/logs/t4_nodes.txt
rm -f deployment/configs/endpoints.json

echo "Done!"
squeue -u $USER
EOF

chmod +x deployment/scripts/shutdown_10t4.sh
```

### Complete Launch Sequence

```bash
# ============================================================================
# LAUNCH SEQUENCE FOR 10× T4 DEPLOYMENT
# ============================================================================

# 1. First-time setup (run once)
cd /home/sdodl001_odu_edu/BioPipelines
conda activate biopipe-inference
./deployment/scripts/download_models.sh

# 2. Launch all services
./deployment/scripts/launch_10t4.sh

# 3. Monitor startup
watch -n 5 'squeue -u $USER | grep biopipe'

# 4. Test deployment (after ~5 minutes)
python deployment/scripts/test_deployment.py

# 5. Shutdown when done
./deployment/scripts/shutdown_10t4.sh
```

---

## Path A: Immediate - H100 + T4 (Recommended)

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PATH A: H100 + 2× T4                                │
│                         Total: 112GB VRAM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    H100 (80GB) - Primary Node                          │ │
│  │                    Partition: h100flex                                 │ │
│  │                                                                        │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│  │  │Qwen-Coder-7B│ │Nemotron-8B  │ │Qwen-Math-7B │ │BioMistral-7B│      │ │
│  │  │   16GB      │ │   17GB      │ │   15GB      │ │   15GB      │      │ │
│  │  │             │ │             │ │             │ │             │      │ │
│  │  │ Code Gen    │ │ Orchestrate │ │ Math/Stats  │ │ Bio Domain  │      │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│  │                                                                        │ │
│  │  Total: 63GB used | KV Cache: 17GB available                          │ │
│  │  vLLM Server: ports 8000-8003                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐          │
│  │    T4-0 (16GB) - Fast Path  │  │   T4-1 (16GB) - Safety      │          │
│  │    Partition: t4flex        │  │   Partition: t4flex         │          │
│  │                             │  │                             │          │
│  │  ┌─────────┐ ┌─────────┐   │  │  ┌─────────┐ ┌─────────┐   │          │
│  │  │Llama-3B │ │ BGE-M3  │   │  │  │Safety-8B│ │Phi-3.5  │   │          │
│  │  │  7GB    │ │  2GB    │   │  │  │INT8 9GB │ │INT8 4GB │   │          │
│  │  │ Intent  │ │Embedding│   │  │  │ Filter  │ │ Backup  │   │          │
│  │  └─────────┘ └─────────┘   │  │  └─────────┘ └─────────┘   │          │
│  │                             │  │                             │          │
│  │  vLLM: port 9000            │  │  vLLM: port 9001            │          │
│  └─────────────────────────────┘  └─────────────────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Environment Setup (Day 1)

#### 1.1 Create Project Structure

```bash
# On login node
cd /home/sdodl001_odu_edu/BioPipelines

# Create deployment directories
mkdir -p deployment/{configs,scripts,logs,models}
mkdir -p deployment/configs/{vllm,nginx,systemd}
```

#### 1.2 Create Conda Environment for Inference

```bash
# Create dedicated environment
conda create -n biopipe-inference python=3.11 -y
conda activate biopipe-inference

# Install vLLM and dependencies
pip install vllm>=0.6.0
pip install transformers>=4.44.0
pip install torch>=2.4.0
pip install flash-attn --no-build-isolation
pip install huggingface_hub
pip install ray[default]  # For distributed serving

# Install additional tools
pip install httpx aiohttp  # For async requests
pip install prometheus-client  # For metrics
```

#### 1.3 Download Models

```bash
# Create model download script
cat > deployment/scripts/download_models.sh << 'EOF'
#!/bin/bash
set -e

MODEL_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
echo "Downloading models to: $MODEL_DIR"

# Primary models (H100)
echo "=== Downloading H100 models ==="
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct
huggingface-cli download nvidia/Llama-3.1-Nemotron-8B-Orchestrator
huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct
huggingface-cli download BioMistral/BioMistral-7B

# T4 models
echo "=== Downloading T4 models ==="
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
huggingface-cli download BAAI/bge-m3
huggingface-cli download microsoft/Phi-3.5-mini-instruct
huggingface-cli download nvidia/Llama-3.1-Nemotron-Safety-8B-V3

echo "=== All models downloaded ==="
EOF

chmod +x deployment/scripts/download_models.sh
```

### Phase 2: H100 Node Setup (Day 2)

#### 2.1 SLURM Job Script for H100

```bash
cat > deployment/scripts/start_h100_server.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=biopipe-h100
#SBATCH --partition=h100flex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=deployment/logs/h100-%j.out
#SBATCH --error=deployment/logs/h100-%j.err

# Activate environment
source ~/.bashrc
conda activate biopipe-inference

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HF_HOME=/scratch/$USER/hf_cache

# Get node hostname for service discovery
H100_HOST=$(hostname)
echo "H100 node: $H100_HOST" > deployment/logs/h100_host.txt

# Start vLLM with multiple models using Ray
# Model 1: Code Generation (Qwen2.5-Coder-7B)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --served-model-name coder \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.20 \
    --dtype bfloat16 \
    --trust-remote-code &

sleep 30  # Wait for first model to initialize

# Model 2: Orchestrator (Nemotron-8B)
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/Llama-3.1-Nemotron-8B-Orchestrator \
    --served-model-name orchestrator \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.21 \
    --dtype bfloat16 &

sleep 30

# Model 3: Math (Qwen2.5-Math-7B)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --served-model-name math \
    --host 0.0.0.0 \
    --port 8002 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.19 \
    --dtype bfloat16 &

sleep 30

# Model 4: Bio (BioMistral-7B)
python -m vllm.entrypoints.openai.api_server \
    --model BioMistral/BioMistral-7B \
    --served-model-name bio \
    --host 0.0.0.0 \
    --port 8003 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.19 \
    --dtype bfloat16 &

echo "All H100 models started"
echo "Ports: 8000 (coder), 8001 (orchestrator), 8002 (math), 8003 (bio)"

# Keep job running
wait
EOF

chmod +x deployment/scripts/start_h100_server.sbatch
```

#### 2.2 Alternative: Single vLLM with Multiple Models (vLLM 0.6+)

```bash
cat > deployment/scripts/start_h100_multi.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=biopipe-h100-multi
#SBATCH --partition=h100flex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=deployment/logs/h100-multi-%j.out

source ~/.bashrc
conda activate biopipe-inference

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/$USER/hf_cache

# vLLM 0.6+ supports serving multiple models
# Using OpenAI-compatible API with model routing
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --model BioMistral/BioMistral-7B \
    --model nvidia/Llama-3.1-Nemotron-8B-Orchestrator \
    --served-model-name coder,math,bio,orchestrator \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code

# Note: Multi-model in single process requires vLLM 0.6+
# Falls back to sequential loading if not supported
EOF
```

### Phase 3: T4 Nodes Setup (Day 2-3)

#### 3.1 SLURM Job Script for T4s

```bash
cat > deployment/scripts/start_t4_servers.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=biopipe-t4
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=deployment/logs/t4-%j-%N.out
#SBATCH --error=deployment/logs/t4-%j-%N.err

source ~/.bashrc
conda activate biopipe-inference

export HF_HOME=/scratch/$USER/hf_cache

# Get node rank
NODE_RANK=$SLURM_NODEID
T4_HOST=$(hostname)
echo "T4 node $NODE_RANK: $T4_HOST" >> deployment/logs/t4_hosts.txt

if [ "$NODE_RANK" -eq 0 ]; then
    # T4-0: Intent parsing + Embeddings (Fast path)
    echo "Starting T4-0: Intent + Embeddings"
    
    # Llama-3.2-3B for intent parsing
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --served-model-name intent \
        --host 0.0.0.0 \
        --port 9000 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.45 \
        --dtype float16 &
    
    sleep 20
    
    # BGE-M3 for embeddings (using sentence-transformers wrapper)
    # Note: vLLM doesn't natively support embedding models
    # Use separate embedding server
    python deployment/scripts/embedding_server.py \
        --model BAAI/bge-m3 \
        --port 9010 &
    
    wait

elif [ "$NODE_RANK" -eq 1 ]; then
    # T4-1: Safety + Backup analysis
    echo "Starting T4-1: Safety + Analysis"
    
    # Safety model (INT8 quantized)
    python -m vllm.entrypoints.openai.api_server \
        --model nvidia/Llama-3.1-Nemotron-Safety-8B-V3 \
        --served-model-name safety \
        --host 0.0.0.0 \
        --port 9001 \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.55 \
        --quantization awq \
        --dtype float16 &
    
    sleep 20
    
    # Phi-3.5-mini for backup analysis (INT8)
    python -m vllm.entrypoints.openai.api_server \
        --model microsoft/Phi-3.5-mini-instruct \
        --served-model-name analysis \
        --host 0.0.0.0 \
        --port 9002 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.25 \
        --dtype float16 &
    
    wait
fi
EOF
```

#### 3.2 Embedding Server Script

```bash
cat > deployment/scripts/embedding_server.py << 'EOF'
#!/usr/bin/env python3
"""
Simple embedding server for BGE-M3
Provides OpenAI-compatible /v1/embeddings endpoint
"""

import argparse
import asyncio
from typing import List
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI(title="Embedding Server")
model = None

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = "bge-m3"

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict

@app.on_event("startup")
async def load_model():
    global model
    model = SentenceTransformer(
        "BAAI/bge-m3",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Loaded BGE-M3 on {model.device}")

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    texts = request.input if isinstance(request.input, list) else [request.input]
    
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    data = [
        {"object": "embedding", "index": i, "embedding": emb.tolist()}
        for i, emb in enumerate(embeddings)
    ]
    
    return EmbeddingResponse(
        data=data,
        model="bge-m3",
        usage={"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)}
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "bge-m3"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
EOF

chmod +x deployment/scripts/embedding_server.py
```

### Phase 4: API Gateway & Router (Day 3-4)

#### 4.1 Request Router Service

```bash
cat > deployment/scripts/model_router.py << 'EOF'
#!/usr/bin/env python3
"""
BioPipelines Model Router
Routes requests to appropriate model servers based on task type
"""

import os
import asyncio
import httpx
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="BioPipelines Model Router")

# Model endpoints configuration
ENDPOINTS = {
    # H100 models
    "coder": os.getenv("CODER_URL", "http://h100-node:8000"),
    "orchestrator": os.getenv("ORCHESTRATOR_URL", "http://h100-node:8001"),
    "math": os.getenv("MATH_URL", "http://h100-node:8002"),
    "bio": os.getenv("BIO_URL", "http://h100-node:8003"),
    # T4 models
    "intent": os.getenv("INTENT_URL", "http://t4-node-0:9000"),
    "embeddings": os.getenv("EMBEDDINGS_URL", "http://t4-node-0:9010"),
    "safety": os.getenv("SAFETY_URL", "http://t4-node-1:9001"),
    "analysis": os.getenv("ANALYSIS_URL", "http://t4-node-1:9002"),
}

# Task to model mapping
TASK_ROUTING = {
    "code_generation": "coder",
    "code_validation": "coder",
    "workflow_generation": "coder",
    "orchestration": "orchestrator",
    "planning": "orchestrator",
    "math": "math",
    "statistics": "math",
    "bio": "bio",
    "medical": "bio",
    "intent_parsing": "intent",
    "classification": "intent",
    "embedding": "embeddings",
    "similarity": "embeddings",
    "safety_check": "safety",
    "data_analysis": "analysis",
    "visualization": "analysis",
}

class ChatRequest(BaseModel):
    model: str = "auto"  # Auto-route based on task
    messages: list
    task_type: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7

class RouterResponse(BaseModel):
    model_used: str
    endpoint: str
    response: Dict[str, Any]

# HTTP client
client = httpx.AsyncClient(timeout=120.0)

async def classify_task(messages: list) -> str:
    """Use intent model to classify the task type."""
    last_message = messages[-1]["content"] if messages else ""
    
    classification_prompt = f"""Classify this request into one category:
- code_generation
- workflow_generation
- math
- bio
- data_analysis
- general

Request: {last_message[:500]}

Category:"""
    
    try:
        response = await client.post(
            f"{ENDPOINTS['intent']}/v1/chat/completions",
            json={
                "model": "intent",
                "messages": [{"role": "user", "content": classification_prompt}],
                "max_tokens": 20,
                "temperature": 0.1,
            }
        )
        result = response.json()
        category = result["choices"][0]["message"]["content"].strip().lower()
        
        # Map to known task types
        for task_type in TASK_ROUTING:
            if task_type in category:
                return task_type
        return "code_generation"  # Default
    except Exception as e:
        print(f"Classification failed: {e}")
        return "code_generation"

async def check_safety(messages: list) -> bool:
    """Check if the request passes safety filters."""
    try:
        last_message = messages[-1]["content"] if messages else ""
        response = await client.post(
            f"{ENDPOINTS['safety']}/v1/chat/completions",
            json={
                "model": "safety",
                "messages": [{"role": "user", "content": f"Is this safe? {last_message[:500]}"}],
                "max_tokens": 10,
            }
        )
        result = response.json()
        return "unsafe" not in result["choices"][0]["message"]["content"].lower()
    except Exception:
        return True  # Fail open

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Route chat completions to appropriate model."""
    
    # Determine task type
    if request.task_type:
        task_type = request.task_type
    elif request.model != "auto" and request.model in ENDPOINTS:
        model_name = request.model
        return await forward_request(model_name, request)
    else:
        task_type = await classify_task(request.messages)
    
    # Get model for task
    model_name = TASK_ROUTING.get(task_type, "coder")
    
    # Optional: Safety check
    # if not await check_safety(request.messages):
    #     raise HTTPException(status_code=400, detail="Request blocked by safety filter")
    
    return await forward_request(model_name, request)

async def forward_request(model_name: str, request: ChatRequest) -> Dict:
    """Forward request to specific model endpoint."""
    endpoint = ENDPOINTS.get(model_name)
    if not endpoint:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    try:
        response = await client.post(
            f"{endpoint}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
        )
        result = response.json()
        result["model_used"] = model_name
        result["endpoint"] = endpoint
        return result
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Model server unavailable: {e}")

@app.post("/v1/embeddings")
async def embeddings(request: Dict):
    """Forward embedding requests."""
    try:
        response = await client.post(
            f"{ENDPOINTS['embeddings']}/v1/embeddings",
            json=request
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Embedding server unavailable: {e}")

@app.get("/health")
async def health():
    """Check health of all endpoints."""
    health_status = {}
    for name, endpoint in ENDPOINTS.items():
        try:
            response = await client.get(f"{endpoint}/health", timeout=5.0)
            health_status[name] = "healthy" if response.status_code == 200 else "unhealthy"
        except Exception:
            health_status[name] = "unreachable"
    return {"status": "ok", "endpoints": health_status}

@app.get("/models")
async def list_models():
    """List available models."""
    return {"models": list(ENDPOINTS.keys()), "routing": TASK_ROUTING}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF
```

#### 4.2 Router SLURM Job

```bash
cat > deployment/scripts/start_router.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=biopipe-router
#SBATCH --partition=cpuspot
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=deployment/logs/router-%j.out

source ~/.bashrc
conda activate biopipe-inference

# Read node hostnames
H100_HOST=$(cat deployment/logs/h100_host.txt | awk '{print $NF}')
T4_HOSTS=($(cat deployment/logs/t4_hosts.txt | awk '{print $NF}'))

# Export endpoints
export CODER_URL="http://${H100_HOST}:8000"
export ORCHESTRATOR_URL="http://${H100_HOST}:8001"
export MATH_URL="http://${H100_HOST}:8002"
export BIO_URL="http://${H100_HOST}:8003"
export INTENT_URL="http://${T4_HOSTS[0]}:9000"
export EMBEDDINGS_URL="http://${T4_HOSTS[0]}:9010"
export SAFETY_URL="http://${T4_HOSTS[1]}:9001"
export ANALYSIS_URL="http://${T4_HOSTS[1]}:9002"

echo "Router configuration:"
echo "  H100: $H100_HOST"
echo "  T4-0: ${T4_HOSTS[0]}"
echo "  T4-1: ${T4_HOSTS[1]}"

python deployment/scripts/model_router.py
EOF
```

### Phase 5: Integration with BioPipelines (Day 4-5)

#### 5.1 Update LLM Provider Configuration

```python
# src/workflow_composer/providers/local_inference.py

"""
Local Model Inference Provider
Connects BioPipelines to local vLLM servers
"""

import os
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio

@dataclass
class LocalModelConfig:
    """Configuration for local model endpoints."""
    router_url: str = "http://localhost:8080"
    timeout: float = 120.0
    
    # Direct endpoints (bypass router)
    coder_url: Optional[str] = None
    orchestrator_url: Optional[str] = None
    intent_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "LocalModelConfig":
        return cls(
            router_url=os.getenv("BIOPIPE_ROUTER_URL", "http://localhost:8080"),
            coder_url=os.getenv("BIOPIPE_CODER_URL"),
            orchestrator_url=os.getenv("BIOPIPE_ORCHESTRATOR_URL"),
            intent_url=os.getenv("BIOPIPE_INTENT_URL"),
        )


class LocalInferenceProvider:
    """Provider for local model inference via vLLM."""
    
    def __init__(self, config: Optional[LocalModelConfig] = None):
        self.config = config or LocalModelConfig.from_env()
        self.client = httpx.AsyncClient(timeout=self.config.timeout)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "auto",
        task_type: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate chat completion using local models.
        
        Args:
            messages: Chat messages
            model: Model name or "auto" for routing
            task_type: Explicit task type for routing
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            OpenAI-compatible response
        """
        response = await self.client.post(
            f"{self.config.router_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "task_type": task_type,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_tokens: int = 4096,
    ) -> str:
        """Generate code using the coder model."""
        result = await self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            task_type="code_generation",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return result["choices"][0]["message"]["content"]
    
    async def analyze_data(
        self,
        data_description: str,
        analysis_type: str = "summary",
    ) -> str:
        """Analyze data using the analysis model."""
        result = await self.chat_completion(
            messages=[{"role": "user", "content": f"Analyze: {data_description}"}],
            task_type="data_analysis",
            temperature=0.5,
        )
        return result["choices"][0]["message"]["content"]
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using BGE-M3."""
        response = await self.client.post(
            f"{self.config.router_url}/v1/embeddings",
            json={"input": texts, "model": "bge-m3"}
        )
        response.raise_for_status()
        result = response.json()
        return [item["embedding"] for item in result["data"]]
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all model endpoints."""
        response = await self.client.get(f"{self.config.router_url}/health")
        return response.json()
    
    async def close(self):
        await self.client.aclose()


# Convenience function
async def get_local_provider() -> LocalInferenceProvider:
    """Get or create local inference provider."""
    return LocalInferenceProvider()
```

#### 5.2 Update Provider Registry

```python
# Add to src/workflow_composer/providers/__init__.py

from .local_inference import (
    LocalInferenceProvider,
    LocalModelConfig,
    get_local_provider,
)

__all__ = [
    # ... existing exports ...
    "LocalInferenceProvider",
    "LocalModelConfig", 
    "get_local_provider",
]
```

### Phase 6: Testing & Validation (Day 5)

#### 6.1 Test Script

```bash
cat > deployment/scripts/test_deployment.py << 'EOF'
#!/usr/bin/env python3
"""Test the multi-model deployment."""

import asyncio
import httpx
import time

ROUTER_URL = "http://localhost:8080"

async def test_health():
    """Test health endpoint."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ROUTER_URL}/health")
        print(f"Health check: {response.json()}")
        return response.status_code == 200

async def test_code_generation():
    """Test code generation."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        start = time.time()
        response = await client.post(
            f"{ROUTER_URL}/v1/chat/completions",
            json={
                "model": "auto",
                "task_type": "code_generation",
                "messages": [
                    {"role": "user", "content": "Write a Python function to calculate GC content of a DNA sequence."}
                ],
                "max_tokens": 500,
            }
        )
        elapsed = time.time() - start
        result = response.json()
        print(f"\n=== Code Generation Test ===")
        print(f"Model used: {result.get('model_used', 'unknown')}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Response:\n{result['choices'][0]['message']['content'][:500]}...")
        return response.status_code == 200

async def test_bio_reasoning():
    """Test bio domain reasoning."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        start = time.time()
        response = await client.post(
            f"{ROUTER_URL}/v1/chat/completions",
            json={
                "model": "auto",
                "task_type": "bio",
                "messages": [
                    {"role": "user", "content": "Explain the difference between ChIP-seq and ATAC-seq."}
                ],
                "max_tokens": 500,
            }
        )
        elapsed = time.time() - start
        result = response.json()
        print(f"\n=== Bio Reasoning Test ===")
        print(f"Model used: {result.get('model_used', 'unknown')}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Response:\n{result['choices'][0]['message']['content'][:500]}...")
        return response.status_code == 200

async def test_embeddings():
    """Test embedding generation."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        start = time.time()
        response = await client.post(
            f"{ROUTER_URL}/v1/embeddings",
            json={
                "input": ["RNA-seq differential expression analysis", "ChIP-seq peak calling"],
                "model": "bge-m3"
            }
        )
        elapsed = time.time() - start
        result = response.json()
        print(f"\n=== Embedding Test ===")
        print(f"Time: {elapsed:.2f}s")
        print(f"Embedding dimension: {len(result['data'][0]['embedding'])}")
        return response.status_code == 200

async def main():
    print("=" * 60)
    print("BioPipelines Multi-Model Deployment Test")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("Code Generation", test_code_generation),
        ("Bio Reasoning", test_bio_reasoning),
        ("Embeddings", test_embeddings),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = await test_fn()
            results.append((name, "✅ PASSED" if passed else "❌ FAILED"))
        except Exception as e:
            results.append((name, f"❌ ERROR: {e}"))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
EOF
```

### Phase 7: Launch Sequence

```bash
# Complete launch sequence

# Step 1: Download models (run once)
./deployment/scripts/download_models.sh

# Step 2: Start H100 server
sbatch deployment/scripts/start_h100_server.sbatch
# Wait for job to start, check logs
sleep 60
cat deployment/logs/h100_host.txt

# Step 3: Start T4 servers
sbatch deployment/scripts/start_t4_servers.sbatch
# Wait for jobs
sleep 60
cat deployment/logs/t4_hosts.txt

# Step 4: Start router (after H100 and T4 are running)
sbatch deployment/scripts/start_router.sbatch

# Step 5: Test deployment
sleep 30
python deployment/scripts/test_deployment.py
```

---

## Path B: Future - 4× L4 + 4× T4

### Prerequisites
- Request L4 GPUs from cluster admin
- New partition `l4flex` to be created

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PATH B: 4× L4 + 4× T4 (160GB)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  NODE 1: 4× L4 (96GB)                     NODE 2: 4× T4 (64GB)              │
│  ┌─────────────────────────────────┐     ┌─────────────────────────────────┐│
│  │                                 │     │                                 ││
│  │ L4-0: Orchestrator (17GB)       │     │ T4-0: BioMistral INT8 (8GB)     ││
│  │       + Llama-3.2-3B (7GB)      │     │       + BioGPT (3GB)            ││
│  │       Port: 8000, 8001          │     │       Port: 9000, 9001          ││
│  │                                 │     │                                 ││
│  │ L4-1: Qwen-Coder-7B (16GB)      │     │ T4-1: DeepSeek-Coder INT4 (8GB) ││
│  │       + Phi-3.5 INT8 (4GB)      │     │       + BGE-base (0.5GB)        ││
│  │       Port: 8010, 8011          │     │       Port: 9010, 9011          ││
│  │                                 │     │                                 ││
│  │ L4-2: Qwen-Math-7B (15GB)       │     │ T4-2: Llama-3.2-3B (7GB)        ││
│  │       + BGE-M3 (2GB)            │     │       + Phi-3.5 (8GB)           ││
│  │       Port: 8020, 8021          │     │       Port: 9020, 9021          ││
│  │                                 │     │                                 ││
│  │ L4-3: Gemma-2-9B (19GB)         │     │ T4-3: Safety-8B INT8 (9GB)      ││
│  │       Port: 8030                │     │       Port: 9030                ││
│  │                                 │     │                                 ││
│  └─────────────────────────────────┘     └─────────────────────────────────┘│
│                                                                              │
│  Benefits:                                                                   │
│  • 10+ models with full redundancy                                          │
│  • Horizontal scaling across nodes                                          │
│  • Better fault tolerance                                                    │
│  • Load balancing per GPU                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Notes
- Same SLURM scripts as Path A, adjusted for L4 partition
- Consider using Kubernetes/Ray for multi-node orchestration
- May need InfiniBand for low-latency cross-node communication

---

## Path C: Alternative - Multi-H100

### When to Use
- Need very high throughput (>100 req/s)
- Require redundancy for production
- Running larger models (13B+)

### Configuration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PATH C: 2× H100 (160GB)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  H100-0 (80GB) - Primary                  H100-1 (80GB) - Replica           │
│  ┌─────────────────────────────────┐     ┌─────────────────────────────────┐│
│  │ All 5 models (63GB)             │     │ Same 5 models (63GB)            ││
│  │ • Coder                         │     │ • Coder (backup)                ││
│  │ • Orchestrator                  │     │ • Orchestrator (backup)         ││
│  │ • Math                          │     │ • Math (backup)                 ││
│  │ • Bio                           │     │ • Bio (backup)                  ││
│  │ • Spare: 17GB for larger model  │     │ • Spare: 17GB                   ││
│  └─────────────────────────────────┘     └─────────────────────────────────┘│
│                                                                              │
│  Load Balancer: Round-robin between H100-0 and H100-1                       │
│  Failover: Automatic redirect if one node fails                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SLURM Script

```bash
#SBATCH --partition=h100dualflex
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
```

---

## Path D: Budget - T4 Only

### When to Use
- Testing/development
- Low traffic (<10 req/min)
- Limited budget

### Configuration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PATH D: 4× T4 (64GB)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  T4-0 (16GB)          T4-1 (16GB)         T4-2 (16GB)         T4-3 (16GB)  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐│
│  │Qwen-Coder   │     │Llama-3.2-3B │     │Phi-3.5-mini │     │BioMistral   ││
│  │INT4 (4GB)   │     │(7GB)        │     │(8GB)        │     │INT8 (8GB)   ││
│  │             │     │             │     │             │     │             ││
│  │+ Qwen-1.5B  │     │+ BGE-M3     │     │+ BGE-base   │     │+ BioGPT     ││
│  │(3GB)        │     │(2GB)        │     │(0.5GB)      │     │(3GB)        ││
│  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘│
│                                                                              │
│  Limitations:                                                                │
│  • All models need INT8/INT4 quantization                                   │
│  • No dedicated orchestrator (Llama-3.2-3B handles routing)                 │
│  • Lower quality than FP16 models                                           │
│  • Cloud fallback recommended for complex tasks                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Infrastructure Setup

### Model Caching

```bash
# Set up shared model cache
export HF_HOME=/scratch/shared/hf_cache  # Or NFS mount
export TRANSFORMERS_CACHE=$HF_HOME

# Ensure cache is readable by all users
chmod -R 755 $HF_HOME
```

### Port Allocation Convention

| Port Range | Purpose |
|------------|---------|
| 8000-8099 | H100/L4 primary models |
| 8100-8199 | H100/L4 secondary models |
| 9000-9099 | T4 primary models |
| 9100-9199 | T4 secondary models |
| 8080 | API Gateway/Router |
| 9010 | Embedding server |

### Logging Setup

```bash
# Centralized logging
mkdir -p deployment/logs/{h100,t4,router,metrics}

# Log rotation (add to crontab)
0 0 * * * find deployment/logs -name "*.out" -mtime +7 -delete
```

---

## Model Serving Options

### Option 1: vLLM (Recommended)

**Pros:** Fastest, PagedAttention, continuous batching  
**Cons:** Higher memory overhead

```bash
pip install vllm>=0.6.0
vllm serve MODEL_NAME --port PORT
```

### Option 2: Text Generation Inference (TGI)

**Pros:** Lower memory, good batching  
**Cons:** Slightly slower than vLLM

```bash
docker run --gpus all -p 8000:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id MODEL_NAME
```

### Option 3: Ollama (Simplest)

**Pros:** Very easy setup, model management  
**Cons:** Less control, single model per instance

```bash
ollama serve
ollama run MODEL_NAME
```

### Comparison

| Feature | vLLM | TGI | Ollama |
|---------|------|-----|--------|
| Speed | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Memory Efficiency | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Ease of Setup | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| Multi-Model | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Production Ready | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |

---

## Monitoring & Observability

### Prometheus Metrics

```yaml
# deployment/configs/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: 
        - 'h100-node:8000'
        - 'h100-node:8001'
        - 't4-node-0:9000'
        - 't4-node-1:9001'
```

### Key Metrics to Monitor

| Metric | Alert Threshold |
|--------|-----------------|
| GPU Memory Usage | >90% |
| Request Latency P99 | >5s |
| Queue Depth | >100 |
| Error Rate | >1% |
| Tokens/Second | <100 |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Model too large | Reduce batch size or use quantization |
| Slow first request | Model loading | Use warm-up requests |
| Connection refused | Server not started | Check SLURM logs |
| Timeout errors | Network latency | Increase timeout, check firewall |

### Debug Commands

```bash
# Check GPU usage
nvidia-smi

# Check vLLM logs
tail -f deployment/logs/h100-*.out

# Test endpoint directly
curl http://node:8000/health

# Check SLURM jobs
squeue -u $USER
```

---

## Appendix: Quick Reference

### Start Everything

```bash
# 1. Submit all jobs
sbatch deployment/scripts/start_h100_server.sbatch
sbatch deployment/scripts/start_t4_servers.sbatch
sleep 60
sbatch deployment/scripts/start_router.sbatch

# 2. Verify
python deployment/scripts/test_deployment.py
```

### Stop Everything

```bash
# Cancel all BioPipelines jobs
scancel -n biopipe-h100
scancel -n biopipe-t4
scancel -n biopipe-router
```

### Environment Variables

```bash
export BIOPIPE_ROUTER_URL="http://localhost:8080"
export HF_HOME="/scratch/$USER/hf_cache"
export CUDA_VISIBLE_DEVICES="0"
```
