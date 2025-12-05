#!/bin/bash
#SBATCH --job-name=vllm_math
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/vllm/math_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/vllm/math_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ============================================================================
# Math and statistics
# Model: Qwen/Qwen2.5-Math-7B-Instruct-AWQ
# Port: 8003
# Quantization: awq
# ============================================================================

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ Starting math Server"
echo "║ Model: Qwen/Qwen2.5-Math-7B-Instruct-AWQ"
echo "║ Port: 8003"
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
CONNECTION_FILE="/home/sdodl001_odu_edu/BioPipelines/.model_connections/math.env"
cat > "$CONNECTION_FILE" << CONN
MODEL_KEY=math
MODEL_ID=Qwen/Qwen2.5-Math-7B-Instruct-AWQ
HOST=$(hostname)
PORT=8003
URL=http://$(hostname):8003
HEALTH_URL=http://$(hostname):8003/health
SLURM_JOB_ID=$SLURM_JOB_ID
STARTED_AT=$(date -Iseconds)
CONN

echo "Connection info: $CONNECTION_FILE"
echo "API Endpoint: http://$(hostname):8003/v1"
echo ""
# Start vLLM server
echo "Starting vLLM server for Qwen/Qwen2.5-Math-7B-Instruct-AWQ..."

VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-7B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8003 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
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
