#!/bin/bash
# Submit all 9 BioPipelines with test data

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_pipeline_with_container.sh"

# All 10 pipelines with test data
PIPELINES=(
    "rna-seq"
    "dna-seq"
    "chip-seq"
    "atac-seq"
    "hic"
    "scrna-seq"
    "methylation"
    "long-read"
    "metagenomics"
    "structural-variants"
)

echo "════════════════════════════════════════════════════"
echo "  Submitting All BioPipelines"
echo "════════════════════════════════════════════════════"
echo ""

for pipeline in "${PIPELINES[@]}"; do
    echo "→ Submitting $pipeline..."
    bash "$SUBMIT_SCRIPT" "$pipeline"
    echo ""
    sleep 2
done

echo "════════════════════════════════════════════════════"
echo "  All pipelines submitted!"
echo "════════════════════════════════════════════════════"
echo ""
echo "Monitor: watch -n 5 'squeue --me'"
echo "Logs: ls -lht ~/BioPipelines/logs/pipeline_runs/ | head"
echo ""
