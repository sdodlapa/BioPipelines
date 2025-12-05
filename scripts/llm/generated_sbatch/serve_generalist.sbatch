#!/bin/bash
#SBATCH --job-name=vllm_generalist
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/vllm/generalist_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/vllm/generalist_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ============================================================================
# General purpose - intent, docs, analysis
# Model: Qwen/Qwen2.5-7B-Instruct-AWQ
# Port: 8001
# Quantization: awq
# ============================================================================

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ Starting generalist Server"
echo "║ Model: Qwen/Qwen2.5-7B-Instruct-AWQ"
echo "║ Port: 8001"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Environment setup
export HOME=/home/sdodl001_odu_edu
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface

# Activate environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env 2>/dev/null || conda activate biopipelines 2>/dev/null || {
    echo "ERROR: Could not activate conda environment"
    exit 1
}

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Save connection info for other components
CONNECTION_FILE="/home/sdodl001_odu_edu/BioPipelines/.model_connections/generalist.env"
cat > "$CONNECTION_FILE" << CONN
MODEL_KEY=generalist
MODEL_ID=Qwen/Qwen2.5-7B-Instruct-AWQ
HOST=$(hostname)
PORT=8001
URL=http://$(hostname):8001
HEALTH_URL=http://$(hostname):8001/health
SLURM_JOB_ID=$SLURM_JOB_ID
STARTED_AT=$(date -Iseconds)
CONN

echo "Connection info: $CONNECTION_FILE"
echo "API Endpoint: http://$(hostname):8001/v1"
echo ""
# Start vLLM server
echo "Starting vLLM server for Qwen/Qwen2.5-7B-Instruct-AWQ..."

VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype float16"

VLLM_CMD="$VLLM_CMD --quantization awq"

echo "Command: $VLLM_CMD"
echo ""
eval $VLLM_CMD

# Cleanup
echo ""
echo "Server stopped at $(date)"
rm -f "$CONNECTION_FILE"
