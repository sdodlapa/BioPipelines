#!/bin/bash
#SBATCH --job-name=biomistral
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/biomistral_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/biomistral_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=h100flex
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ============================================================================
# BioMistral-7B Server for BioPipelines
# ============================================================================
# 
# BioMistral-7B is a biomedical-domain LLM pre-trained on PubMed Central.
# Excellent for parsing biological terminology and generating structured output.
#
# Model: BioMistral/BioMistral-7B (7B parameters, ~15GB VRAM)
# License: Apache 2.0
# ============================================================================

echo "============================================"
echo "Starting BioMistral-7B Server"
echo "Date: $(date)"
echo "Node: $(hostname)"
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
conda activate vllm_env

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo ""

# Check GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Start vLLM server
echo "Starting vLLM server..."
echo "Access at: http://$(hostname):$PORT"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto

echo ""
echo "Server stopped at $(date)"
