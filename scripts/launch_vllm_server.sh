#!/bin/bash
# ==============================================================================
# Launch vLLM Server for BioPipelines Agent Router
# ==============================================================================
# 
# This script launches a vLLM inference server for local LLM routing.
# Designed for T4 GPUs (16GB VRAM).
#
# Usage:
#   ./launch_vllm_server.sh [model]
#
# Models (optimized for T4 16GB):
#   qwen-1.5b   - Qwen2.5-1.5B-Instruct (default, fastest)
#   qwen-3b    - Qwen2.5-3B-Instruct (balanced)
#   phi-3      - Phi-3-mini-4k-instruct (3.8B, good for reasoning)
#   mistral-7b - Mistral-7B-Instruct-v0.3-AWQ (quantized, powerful)
#   biomistral - BioMistral-7B-AWQ (biomedical specialized)
#
# ==============================================================================

set -euo pipefail

# Default settings
DEFAULT_MODEL="qwen-1.5b"
PORT="${VLLM_PORT:-8000}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
HOST="${VLLM_HOST:-0.0.0.0}"

# Model mappings (all T4 compatible)
declare -A MODELS
MODELS["qwen-1.5b"]="Qwen/Qwen2.5-1.5B-Instruct"
MODELS["qwen-3b"]="Qwen/Qwen2.5-3B-Instruct"
MODELS["qwen-7b"]="Qwen/Qwen2.5-7B-Instruct-AWQ"
MODELS["phi-3"]="microsoft/Phi-3-mini-4k-instruct"
MODELS["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3"
MODELS["biomistral"]="BioMistral/BioMistral-7B-AWQ"

# Model-specific configs for T4 (16GB)
declare -A GPU_MEMORY_UTIL
GPU_MEMORY_UTIL["qwen-1.5b"]=0.5
GPU_MEMORY_UTIL["qwen-3b"]=0.7
GPU_MEMORY_UTIL["qwen-7b"]=0.9
GPU_MEMORY_UTIL["phi-3"]=0.7
GPU_MEMORY_UTIL["mistral-7b"]=0.9
GPU_MEMORY_UTIL["biomistral"]=0.9

# Parse arguments
MODEL_KEY="${1:-$DEFAULT_MODEL}"

if [[ ! ${MODELS[$MODEL_KEY]+_} ]]; then
    echo "❌ Unknown model: $MODEL_KEY"
    echo "Available models: ${!MODELS[@]}"
    exit 1
fi

MODEL="${MODELS[$MODEL_KEY]}"
GPU_UTIL="${GPU_MEMORY_UTIL[$MODEL_KEY]}"

echo "============================================================"
echo "🚀 BioPipelines vLLM Server Launcher"
echo "============================================================"
echo "Model:     $MODEL_KEY → $MODEL"
echo "Port:      $PORT"
echo "GPU:       $GPU"
echo "GPU Memory Utilization: ${GPU_UTIL}"
echo "============================================================"

# Check if already running
if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "⚠️  vLLM server already running on port ${PORT}"
    echo "    To restart, first kill it: pkill -f 'vllm.entrypoints'"
    exit 0
fi

# Create log directory
LOG_DIR="/home/sdodl001_odu_edu/BioPipelines/logs/vllm"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${MODEL_KEY}_$(date +%Y%m%d_%H%M%S).log"

# Check if we're on a GPU node
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 Checking GPU availability..."
    if nvidia-smi &> /dev/null; then
        echo "✅ GPU available:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    else
        echo "⚠️  nvidia-smi failed. You might be on a head node."
        echo "    Submit this as a SLURM job for GPU access."
        
        read -p "Submit as SLURM job? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            exec "$0" --slurm "$MODEL_KEY"
        fi
        exit 1
    fi
else
    echo "❌ nvidia-smi not found. Are you on a GPU node?"
    exit 1
fi

# Check vLLM installation
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not installed."
    echo "   Install with: pip install vllm"
    exit 1
fi

echo ""
echo "🚀 Starting vLLM server..."
echo "   Logs: $LOG_FILE"
echo ""

# Launch server
CUDA_VISIBLE_DEVICES="$GPU" python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len 4096 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    2>&1 | tee "$LOG_FILE" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo ""
echo "⏳ Waiting for server to start (this may take 1-2 minutes for model download)..."

MAX_WAIT=300
WAITED=0
while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "❌ Timeout waiting for server to start."
        echo "   Check logs: $LOG_FILE"
        exit 1
    fi
    echo "   Waiting... (${WAITED}s)"
done

echo ""
echo "✅ vLLM Server is ready!"
echo "============================================================"
echo "API Endpoint:  http://localhost:${PORT}/v1"
echo "Health Check:  http://localhost:${PORT}/health"
echo "Models:        http://localhost:${PORT}/v1/models"
echo ""
echo "Test with:"
echo "  curl http://localhost:${PORT}/v1/models"
echo ""
echo "Use in Python:"
echo "  from openai import OpenAI"
echo "  client = OpenAI(base_url='http://localhost:${PORT}/v1', api_key='x')"
echo "============================================================"

# Save PID for later
echo "$SERVER_PID" > "${LOG_DIR}/.vllm_pid"
echo "Server PID saved to ${LOG_DIR}/.vllm_pid"
