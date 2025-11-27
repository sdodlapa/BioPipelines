#!/bin/bash
# ============================================================================
# BioPipelines Web UI Launcher
# ============================================================================
#
# Usage:
#   ./scripts/start_gradio.sh              # Login node, cloud LLM
#   ./scripts/start_gradio.sh --gpu        # Submit GPU job (2x H100 + vLLM)
#   ./scripts/start_gradio.sh --gpu --partition t4flex --model qwen-3b
#
# ============================================================================

set -e

# Default settings
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
SHARE=""
USE_GPU=false
GPU_MODEL="${GPU_MODEL:-llama-70b}"
GPU_PARTITION="${GPU_PARTITION:-h100dualflex}"
GPU_TIME="${GPU_TIME:-8:00:00}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --share) SHARE="--share"; shift ;;
        --gpu|-g) USE_GPU=true; shift ;;
        --model|-m) GPU_MODEL="$2"; shift 2 ;;
        --partition) GPU_PARTITION="$2"; shift 2 ;;
        --time) GPU_TIME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  No options      Run on login node with cloud LLM"
            echo "  --gpu           Submit SLURM job for GPU node"
            echo "  --model MODEL   LLM model (default: llama-70b)"
            echo "  --partition P   SLURM partition (default: h100dualflex)"
            echo "  --time TIME     Job time (default: 8:00:00)"
            echo "  --port PORT     Gradio port (default: 7860)"
            echo "  --share         Create public Gradio link"
            echo ""
            echo "Partitions: t4flex (1 GPU), h100dualflex (2 GPU), h100quadflex (4 GPU)"
            echo "Models: qwen-1.5b, qwen-3b, phi-3, llama-70b, qwen-72b, deepseek-33b"
            exit 0
            ;;
        *) shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Model mappings
declare -A MODELS
MODELS["qwen-1.5b"]="Qwen/Qwen2.5-1.5B-Instruct"
MODELS["qwen-3b"]="Qwen/Qwen2.5-3B-Instruct"
MODELS["phi-3"]="microsoft/Phi-3-mini-4k-instruct"
MODELS["biomistral"]="BioMistral/BioMistral-7B-AWQ"
MODELS["llama-70b"]="meta-llama/Llama-3.3-70B-Instruct"
MODELS["qwen-72b"]="Qwen/Qwen2.5-72B-Instruct-AWQ"
MODELS["deepseek-33b"]="deepseek-ai/deepseek-coder-33b-instruct"

# Partition configs
declare -A PARTITION_GPUS PARTITION_CPUS PARTITION_MEM PARTITION_UTIL
PARTITION_GPUS["t4flex"]=1;       PARTITION_CPUS["t4flex"]=8;   PARTITION_MEM["t4flex"]="56G";  PARTITION_UTIL["t4flex"]=0.9
PARTITION_GPUS["h100flex"]=1;     PARTITION_CPUS["h100flex"]=26; PARTITION_MEM["h100flex"]="200G"; PARTITION_UTIL["h100flex"]=0.95
PARTITION_GPUS["h100dualflex"]=2; PARTITION_CPUS["h100dualflex"]=26; PARTITION_MEM["h100dualflex"]="450G"; PARTITION_UTIL["h100dualflex"]=0.95
PARTITION_GPUS["h100quadflex"]=4; PARTITION_CPUS["h100quadflex"]=52; PARTITION_MEM["h100quadflex"]="800G"; PARTITION_UTIL["h100quadflex"]=0.95

# ============================================================================
# GPU MODE: Just submit SLURM job and exit
# ============================================================================
if [ "$USE_GPU" = true ]; then
    
    # Validate model
    if [[ ! ${MODELS[$GPU_MODEL]+_} ]]; then
        echo "❌ Unknown model: $GPU_MODEL"
        echo "Available: ${!MODELS[@]}"
        exit 1
    fi
    
    MODEL_HF="${MODELS[$GPU_MODEL]}"
    NUM_GPUS="${PARTITION_GPUS[$GPU_PARTITION]:-2}"
    GPU_CPUS="${PARTITION_CPUS[$GPU_PARTITION]:-26}"
    GPU_MEM="${PARTITION_MEM[$GPU_PARTITION]:-450G}"
    GPU_UTIL="${PARTITION_UTIL[$GPU_PARTITION]:-0.95}"
    
    mkdir -p "${PROJECT_DIR}/logs"
    
    # Submit job directly
    JOB_ID=$(sbatch --parsable << SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=biopipelines-gpu
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --cpus-per-task=${GPU_CPUS}
#SBATCH --mem=${GPU_MEM}
#SBATCH --time=${GPU_TIME}
#SBATCH --output=${PROJECT_DIR}/logs/gpu_server_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/gpu_server_%j.err

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🧬 BioPipelines GPU Server                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:     \$SLURM_JOB_ID"
echo "Node:       \$SLURM_NODELIST"
echo "Partition:  ${GPU_PARTITION} (${NUM_GPUS} GPUs)"
echo "Model:      ${GPU_MODEL}"
echo "Started:    \$(date)"
echo ""

# Save connection info
cat > ${PROJECT_DIR}/logs/gpu_server_info.txt << EOF
SLURM_JOB_ID=\$SLURM_JOB_ID
SLURM_NODELIST=\$SLURM_NODELIST
VLLM_URL=http://\${SLURM_NODELIST}:${VLLM_PORT}/v1
GRADIO_URL=http://\${SLURM_NODELIST}:${PORT}
EOF

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines 2>/dev/null || conda activate base
cd ${PROJECT_DIR}

echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "=== Checking dependencies ==="
python -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || { pip install vllm -q && echo "vLLM installed"; }
python -c "import workflow_composer" 2>/dev/null || pip install -e . -q
python -c "import gradio" 2>/dev/null || pip install gradio -q
echo "Ready"
echo ""

echo "=== Loading API keys ==="
[ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=\$(cat .secrets/openai_key)
[ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=\$(cat .secrets/lightning_key)
[ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=\$(cat .secrets/google_api_key)
[ -f ".secrets/github_token" ] && export GITHUB_TOKEN=\$(cat .secrets/github_token)
echo "Done"
echo ""

echo "=== Starting vLLM (${NUM_GPUS} GPUs) ==="
python -m vllm.entrypoints.openai.api_server \\
    --model "${MODEL_HF}" \\
    --host 0.0.0.0 \\
    --port ${VLLM_PORT} \\
    --tensor-parallel-size ${NUM_GPUS} \\
    --gpu-memory-utilization ${GPU_UTIL} \\
    --max-model-len 4096 \\
    --trust-remote-code \\
    2>&1 | tee ${PROJECT_DIR}/logs/vllm_\${SLURM_JOB_ID}.log &
VLLM_PID=\$!

echo "Waiting for vLLM..."
for i in {1..120}; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM ready"
        break
    fi
    sleep 5
    echo "  [\$((i*5))s] Loading model..."
done

export VLLM_URL="http://localhost:${VLLM_PORT}/v1"
export USE_LOCAL_LLM="true"

echo ""
echo "=== Starting Gradio ==="
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  🎉 Server Ready!                                                ║"
echo "║  Gradio: http://\${SLURM_NODELIST}:${PORT}                        "
echo "║  vLLM:   http://\${SLURM_NODELIST}:${VLLM_PORT}/v1                 "
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

python -m workflow_composer.web.gradio_app --host 0.0.0.0 --port ${PORT} ${SHARE}

kill \$VLLM_PID 2>/dev/null || true
SLURM_SCRIPT
)

    echo "✅ Submitted job: $JOB_ID"
    echo ""
    echo "Monitor:  tail -f ${PROJECT_DIR}/logs/gpu_server_${JOB_ID}.out"
    echo "Status:   squeue -j $JOB_ID"
    echo "Cancel:   scancel $JOB_ID"
    exit 0
fi

# ============================================================================
# NON-GPU MODE: Run locally on login node
# ============================================================================
echo "🧬 BioPipelines - Starting on login node (cloud LLM mode)"
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines 2>/dev/null || conda activate base
cd "$PROJECT_DIR"

[ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
[ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=$(cat .secrets/lightning_key)
[ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)

python -c "import workflow_composer" 2>/dev/null || pip install -e . -q

echo "Starting Gradio on port $PORT..."
python -m workflow_composer.web.gradio_app --host "$HOST" --port "$PORT" $SHARE
