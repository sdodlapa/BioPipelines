#!/bin/bash
#
# BioPipelines Nextflow Launcher
# ==============================
# 
# This script properly launches Nextflow workflows with SLURM support.
# It handles the containerized vs native execution modes.
#
# Usage:
#   ./run_nextflow.sh <workflow.nf> [nextflow options]
#
# Examples:
#   ./run_nextflow.sh main.nf -profile slurm,singularity --input samples.csv
#   ./run_nextflow.sh main.nf -profile local --input samples.csv
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Detect execution environment
detect_environment() {
    if command -v sbatch &> /dev/null; then
        echo "native"  # SLURM available natively
    elif [[ -f "/.singularity.d/runscript" ]]; then
        echo "container"  # Running inside a container
    else
        echo "local"  # Standard environment
    fi
}

# Check for Nextflow
check_nextflow() {
    if command -v nextflow &> /dev/null; then
        return 0
    fi
    
    # Try module load
    if command -v module &> /dev/null; then
        module load nextflow 2>/dev/null && return 0
    fi
    
    # Try activating conda environment
    if [[ -f "${PROJECT_ROOT}/envs/biopipelines/bin/nextflow" ]]; then
        export PATH="${PROJECT_ROOT}/envs/biopipelines/bin:$PATH"
        return 0
    fi
    
    echo "ERROR: Nextflow not found. Please install or module load nextflow."
    exit 1
}

# Main launcher
main() {
    if [[ $# -lt 1 ]]; then
        echo "Usage: $0 <workflow.nf> [nextflow options]"
        echo ""
        echo "Examples:"
        echo "  $0 main.nf -profile slurm,singularity --input samples.csv"
        echo "  $0 main.nf -profile local --input samples.csv"
        exit 1
    fi
    
    local workflow="$1"
    shift
    
    # Check if workflow exists
    if [[ ! -f "$workflow" ]]; then
        echo "ERROR: Workflow file not found: $workflow"
        exit 1
    fi
    
    # Detect environment
    local env=$(detect_environment)
    echo "Detected environment: $env"
    
    # Check Nextflow
    check_nextflow
    
    # Set up configuration
    local config_opts=""
    
    # Add project config if exists
    if [[ -f "${PROJECT_ROOT}/config/nextflow/profiles.config" ]]; then
        config_opts="-c ${PROJECT_ROOT}/config/nextflow/profiles.config"
    fi
    
    # Add workflow-specific config if exists
    local workflow_dir=$(dirname "$workflow")
    if [[ -f "${workflow_dir}/nextflow.config" ]]; then
        config_opts="${config_opts} -c ${workflow_dir}/nextflow.config"
    fi
    
    # Set work directory
    export NXF_WORK="${NXF_WORK:-/scratch/${USER}/BioPipelines/work}"
    mkdir -p "$NXF_WORK"
    
    # Launch based on environment
    case "$env" in
        native)
            echo "Launching with native SLURM support..."
            nextflow run "$workflow" $config_opts "$@"
            ;;
        container)
            echo "Launching from container (local executor recommended)..."
            # Inside container, SLURM won't work - use local executor
            nextflow run "$workflow" $config_opts -profile local "$@"
            ;;
        local)
            echo "Launching with local executor..."
            nextflow run "$workflow" $config_opts -profile local "$@"
            ;;
    esac
}

main "$@"
