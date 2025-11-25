#!/bin/bash
#SBATCH --job-name=biomistral_t4
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/biomistral_t4_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/biomistral_t4_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ============================================================================
# BioMistral-7B Server on T4 GPU (16GB)
# ============================================================================
# 
# T4 GPU Specs:
#   - 16 GB GDDR6 VRAM
#   - Turing architecture  
#   - ~65 TFLOPS FP16
#   - Great for 7B models with quantization
#
# Model: BioMistral/BioMistral-7B (~14GB, fits in 16GB)
# Purpose: Biomedical intent parsing for BioPipelines
#
# Note: Uses 4-bit quantization for better memory efficiency
# ============================================================================

echo "============================================"
echo "Starting BioMistral-7B Server (T4 GPU)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

# Set up environment
export HOME=/home/sdodl001_odu_edu
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface

# Model configuration
MODEL="BioMistral/BioMistral-7B"
PORT=8000
HOST="0.0.0.0"

# Activate vLLM environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env 2>/dev/null || conda activate biopipelines

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  vLLM version: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo ""

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Save connection info for clients
CONNECTION_FILE="$HOME/BioPipelines/.biomistral_server"
echo "BIOMISTRAL_HOST=$(hostname)" > "$CONNECTION_FILE"
echo "BIOMISTRAL_PORT=$PORT" >> "$CONNECTION_FILE"
echo "BIOMISTRAL_URL=http://$(hostname):$PORT/v1" >> "$CONNECTION_FILE"
echo "SLURM_JOB_ID=$SLURM_JOB_ID" >> "$CONNECTION_FILE"

echo "Connection info saved to: $CONNECTION_FILE"
echo "Connect at: http://$(hostname):$PORT/v1"
echo ""

# Start vLLM server with quantization for T4
echo "Starting vLLM server with 4-bit quantization..."

# Try with quantization first (more memory efficient)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --trust-remote-code \
    --dtype float16 \
    --quantization awq 2>/dev/null || \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048 \
    --trust-remote-code \
    --dtype float16

echo ""
echo "Server stopped at $(date)"

# Clean up connection file
rm -f "$CONNECTION_FILE"
