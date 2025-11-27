#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/vllm/slurm-%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/vllm/slurm-%j.err
# ==============================================================================
# vLLM Server SLURM Job
# ==============================================================================
#
# Submit with: sbatch vllm_server.slurm [model]
# Default model: qwen-1.5b
#
# After job starts, the server will be available on the compute node.
# Use `squeue -u $USER` to find the node, then:
#   ssh <node> -L 8000:localhost:8000
# Or configure the router to use the node's hostname.
#
# ==============================================================================

set -euo pipefail

# Get model from argument or use default
MODEL="${1:-qwen-1.5b}"

# Source conda environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/.bashrc
conda activate biopipelines

echo "============================================================"
echo "🚀 vLLM Server SLURM Job"
echo "============================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Model:       $MODEL"
echo "Start Time:  $(date)"
echo "============================================================"

# Create log directory
LOG_DIR="/home/sdodl001_odu_edu/BioPipelines/logs/vllm"
mkdir -p "$LOG_DIR"

# Save node info for clients
NODE_INFO="${LOG_DIR}/node_info.txt"
echo "SLURM_JOB_ID=$SLURM_JOB_ID" > "$NODE_INFO"
echo "SLURM_NODELIST=$SLURM_NODELIST" >> "$NODE_INFO"
echo "START_TIME=$(date -Iseconds)" >> "$NODE_INFO"
echo "MODEL=$MODEL" >> "$NODE_INFO"
echo "API_URL=http://${SLURM_NODELIST}:8000/v1" >> "$NODE_INFO"

echo "Node info saved to: $NODE_INFO"

# Check GPU
echo ""
echo "🔍 GPU Information:"
nvidia-smi

# Model mappings
declare -A MODELS
MODELS["qwen-1.5b"]="Qwen/Qwen2.5-1.5B-Instruct"
MODELS["qwen-3b"]="Qwen/Qwen2.5-3B-Instruct"
MODELS["qwen-7b"]="Qwen/Qwen2.5-7B-Instruct-AWQ"
MODELS["phi-3"]="microsoft/Phi-3-mini-4k-instruct"
MODELS["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.3"
MODELS["biomistral"]="BioMistral/BioMistral-7B-AWQ"

declare -A GPU_MEMORY_UTIL
GPU_MEMORY_UTIL["qwen-1.5b"]=0.5
GPU_MEMORY_UTIL["qwen-3b"]=0.7
GPU_MEMORY_UTIL["qwen-7b"]=0.9
GPU_MEMORY_UTIL["phi-3"]=0.7
GPU_MEMORY_UTIL["mistral-7b"]=0.9
GPU_MEMORY_UTIL["biomistral"]=0.9

MODEL_HF="${MODELS[$MODEL]}"
GPU_UTIL="${GPU_MEMORY_UTIL[$MODEL]}"

echo ""
echo "🚀 Starting vLLM server with $MODEL_HF..."
echo ""

# Start the server (runs in foreground, SLURM will manage)
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_HF" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len 4096 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# This will only run if the server stops
echo "Server stopped at $(date)"
