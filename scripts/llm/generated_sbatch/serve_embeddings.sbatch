#!/bin/bash
#SBATCH --job-name=vllm_embeddings
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/vllm/embeddings_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/vllm/embeddings_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# ============================================================================
# Embeddings for RAG
# Model: BAAI/bge-m3
# Port: 8004
# Quantization: none
# ============================================================================

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ Starting embeddings Server"
echo "║ Model: BAAI/bge-m3"
echo "║ Port: 8004"
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
CONNECTION_FILE="/home/sdodl001_odu_edu/BioPipelines/.model_connections/embeddings.env"
cat > "$CONNECTION_FILE" << CONN
MODEL_KEY=embeddings
MODEL_ID=BAAI/bge-m3
HOST=$(hostname)
PORT=8004
URL=http://$(hostname):8004
HEALTH_URL=http://$(hostname):8004/health
SLURM_JOB_ID=$SLURM_JOB_ID
STARTED_AT=$(date -Iseconds)
CONN

echo "Connection info: $CONNECTION_FILE"
echo "API Endpoint: http://$(hostname):8004/v1"
echo ""
# Start embedding server
echo "Starting embedding server for BAAI/bge-m3..."
python -m vllm.entrypoints.openai.api_server \
    --model BAAI/bge-m3 \
    --host 0.0.0.0 \
    --port 8004 \
    --task embedding \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype float16

# Cleanup
echo ""
echo "Server stopped at $(date)"
rm -f "$CONNECTION_FILE"
