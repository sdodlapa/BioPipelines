#!/bin/bash
# ============================================================================
# BioPipelines Web UI - Login Node Launcher
# ============================================================================
#
# Start the Gradio web interface on the login node.
#
# Usage:
#   ./scripts/start_gradio.sh
#   ./scripts/start_gradio.sh --port 8080
#   ./scripts/start_gradio.sh --share
#
# ============================================================================

set -e

# Default settings
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
SHARE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --share)
            SHARE="--share"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--port PORT] [--host HOST] [--share]"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        🧬 BioPipelines - Gradio Web UI                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Activate conda environment
echo "Activating BioPipelines environment..."
if [ -d ~/envs/biopipelines ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ~/envs/biopipelines
    echo "✓ Using ~/envs/biopipelines"
elif conda env list | grep -q "^biopipelines "; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate biopipelines
    echo "✓ Using biopipelines environment"
else
    echo "⚠ BioPipelines environment not found, using base"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

# Go to project directory
cd "$PROJECT_DIR"

# Load API keys from .secrets directory
echo ""
echo "Loading API keys..."
if [ -f ".secrets/openai_key" ]; then
    export OPENAI_API_KEY=$(cat .secrets/openai_key)
    echo "✓ OpenAI API key loaded"
fi

if [ -f ".secrets/lightning_key" ]; then
    export LIGHTNING_API_KEY=$(cat .secrets/lightning_key)
    echo "✓ Lightning.ai API key loaded (30M FREE tokens/month!)"
fi

if [ -f ".secrets/anthropic_key" ]; then
    export ANTHROPIC_API_KEY=$(cat .secrets/anthropic_key)
    echo "✓ Anthropic API key loaded"
fi

if [ ! -f ".secrets/openai_key" ] && [ ! -f ".secrets/lightning_key" ]; then
    echo "⚠ No API keys found in .secrets/"
    echo "  For Lightning.ai (FREE): https://lightning.ai/models"
    echo "  Save keys to: .secrets/lightning_key or .secrets/openai_key"
fi

# Check if package is installed
echo ""
if ! python -c "import workflow_composer" 2>/dev/null; then
    echo "Installing workflow_composer package..."
    pip install -e . -q
    echo "✓ Package installed"
fi

# Verify required packages
echo "Checking dependencies..."
MISSING=""
for pkg in gradio pandas numpy pyyaml openai; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    echo "Installing missing packages:$MISSING"
    pip install $MISSING -q
    echo "✓ Dependencies installed"
fi

echo ""
echo "Starting Gradio server..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  URL:  http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python -m workflow_composer.web.gradio_app --host "$HOST" --port "$PORT" $SHARE
