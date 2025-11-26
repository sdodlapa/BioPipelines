#!/bin/bash
# ============================================================
# vLLM Local Model Server Launcher
# ============================================================
# This script starts a vLLM server with one of the configured
# open-source models. Use as backup when API providers are
# exhausted or unavailable.
#
# Usage:
#   ./start_local_model.sh [model_name]
#
# Available models:
#   qwen-coder-32b    - Qwen2.5-Coder-32B (1 GPU)
#   deepseek-coder-v2 - DeepSeek-Coder-V2 (2 GPUs)
#   llama-3.3-70b     - Llama-3.3-70B (2 GPUs)
#   minimax-m2        - MiniMax-M2 (4 GPUs)
#   codellama-34b     - CodeLlama-34B (1 GPU)
#
# Environment Variables:
#   VLLM_PORT         - Port to serve on (default: 8000)
#   HF_HOME           - HuggingFace cache directory
# ============================================================

set -e

# Default settings
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Model configurations
declare -A MODELS=(
    ["qwen-coder-32b"]="Qwen/Qwen2.5-Coder-32B-Instruct"
    ["deepseek-coder-v2"]="deepseek-ai/DeepSeek-Coder-V2-Instruct"
    ["llama-3.3-70b"]="meta-llama/Llama-3.3-70B-Instruct"
    ["minimax-m2"]="MiniMaxAI/MiniMax-M2"
    ["codellama-34b"]="codellama/CodeLlama-34b-Instruct-hf"
)

declare -A GPUS=(
    ["qwen-coder-32b"]=1
    ["deepseek-coder-v2"]=2
    ["llama-3.3-70b"]=2
    ["minimax-m2"]=4
    ["codellama-34b"]=1
)

declare -A CONTEXT=(
    ["qwen-coder-32b"]=32768
    ["deepseek-coder-v2"]=65536
    ["llama-3.3-70b"]=32768
    ["minimax-m2"]=32768
    ["codellama-34b"]=16384
)

declare -A DTYPE=(
    ["qwen-coder-32b"]="float16"
    ["deepseek-coder-v2"]="float16"
    ["llama-3.3-70b"]="float16"
    ["minimax-m2"]="float8"
    ["codellama-34b"]="float16"
)

# Get model name from argument or prompt
MODEL_NAME="${1:-qwen-coder-32b}"

# Validate model
if [[ -z "${MODELS[$MODEL_NAME]}" ]]; then
    echo "❌ Unknown model: $MODEL_NAME"
    echo ""
    echo "Available models:"
    for model in "${!MODELS[@]}"; do
        echo "  - $model (${GPUS[$model]} GPU(s))"
    done
    exit 1
fi

MODEL_ID="${MODELS[$MODEL_NAME]}"
TP_SIZE="${GPUS[$MODEL_NAME]}"
MAX_LEN="${CONTEXT[$MODEL_NAME]}"
MODEL_DTYPE="${DTYPE[$MODEL_NAME]}"

echo "============================================================"
echo "🚀 Starting vLLM Server"
echo "============================================================"
echo ""
echo "Model:    $MODEL_NAME"
echo "HF ID:    $MODEL_ID"
echo "GPUs:     $TP_SIZE"
echo "Context:  $MAX_LEN tokens"
echo "Dtype:    $MODEL_DTYPE"
echo "Port:     $VLLM_PORT"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
    if [[ $AVAILABLE_GPUS -lt $TP_SIZE ]]; then
        echo "⚠️  Warning: Model requires $TP_SIZE GPUs, but only $AVAILABLE_GPUS available"
        echo "   Consider using SLURM to request more GPUs"
    else
        echo "✅ GPU check: $AVAILABLE_GPUS available, $TP_SIZE required"
    fi
else
    echo "⚠️  nvidia-smi not found - cannot verify GPU availability"
fi

echo ""
echo "Starting server..."
echo "============================================================"

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_LEN" \
    --dtype "$MODEL_DTYPE" \
    --gpu-memory-utilization 0.9 \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --trust-remote-code
