#!/bin/bash
#SBATCH --job-name=tool_catalog
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/tool_catalog_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/tool_catalog_%j.err
#SBATCH --partition=cpuspot
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Create comprehensive tool catalog from existing containers
# Purpose: Enable AI to query which container has which tool

CONTAINER_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/images"
OUTPUT_DIR="/home/sdodl001_odu_edu/BioPipelines/data/tool_catalog"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Tool Catalog Generation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Container Directory: ${CONTAINER_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

# List of containers to inventory
CONTAINERS=(
    "rna-seq_1.0.0.sif"
    "dna-seq_1.0.0.sif"
    "atac-seq_1.0.0.sif"
    "chip-seq_1.0.0.sif"
    "scrna-seq_1.0.0.sif"
    "metagenomics_1.0.0.sif"
    "methylation_1.0.0.sif"
    "long-read_1.0.0.sif"
    "hic_1.0.0.sif"
    "structural-variants_1.0.0.sif"
    "base_1.0.0.sif"
    "workflow-engine.sif"
)

# JSON output file
JSON_OUTPUT="${OUTPUT_DIR}/tool_catalog_${TIMESTAMP}.json"
SUMMARY_OUTPUT="${OUTPUT_DIR}/tool_catalog_summary_${TIMESTAMP}.txt"

echo "{" > ${JSON_OUTPUT}
echo "  \"generated\": \"$(date -Iseconds)\"," >> ${JSON_OUTPUT}
echo "  \"containers\": {" >> ${JSON_OUTPUT}

echo "Tool Catalog Summary" > ${SUMMARY_OUTPUT}
echo "Generated: $(date)" >> ${SUMMARY_OUTPUT}
echo "========================================" >> ${SUMMARY_OUTPUT}
echo "" >> ${SUMMARY_OUTPUT}

TOTAL_TOOLS=0
FIRST_CONTAINER=true

for CONTAINER in "${CONTAINERS[@]}"; do
    CONTAINER_PATH="${CONTAINER_DIR}/${CONTAINER}"
    
    if [ ! -f "${CONTAINER_PATH}" ]; then
        echo "⚠️  Container not found: ${CONTAINER}"
        continue
    fi
    
    echo ""
    echo "Inventorying: ${CONTAINER}"
    
    # Get list of executables in conda/bin
    TOOLS_LIST=$(singularity exec ${CONTAINER_PATH} ls /opt/conda/bin/ 2>/dev/null | sort)
    TOOL_COUNT=$(echo "${TOOLS_LIST}" | wc -l)
    
    # Get container size
    CONTAINER_SIZE=$(du -h ${CONTAINER_PATH} | cut -f1)
    
    echo "  Found: ${TOOL_COUNT} tools (${CONTAINER_SIZE})"
    
    # Add to summary
    echo "${CONTAINER}" >> ${SUMMARY_OUTPUT}
    echo "  Size: ${CONTAINER_SIZE}" >> ${SUMMARY_OUTPUT}
    echo "  Tools: ${TOOL_COUNT}" >> ${SUMMARY_OUTPUT}
    echo "" >> ${SUMMARY_OUTPUT}
    
    # Add comma separator for JSON (except first entry)
    if [ "${FIRST_CONTAINER}" = false ]; then
        echo "    }," >> ${JSON_OUTPUT}
    fi
    FIRST_CONTAINER=false
    
    # Add container entry to JSON
    CONTAINER_NAME=$(basename ${CONTAINER} .sif)
    echo "    \"${CONTAINER_NAME}\": {" >> ${JSON_OUTPUT}
    echo "      \"path\": \"${CONTAINER_PATH}\"," >> ${JSON_OUTPUT}
    echo "      \"size\": \"${CONTAINER_SIZE}\"," >> ${JSON_OUTPUT}
    echo "      \"tool_count\": ${TOOL_COUNT}," >> ${JSON_OUTPUT}
    echo "      \"tools\": [" >> ${JSON_OUTPUT}
    
    # Add individual tools to JSON
    FIRST_TOOL=true
    while IFS= read -r TOOL; do
        if [ ! -z "${TOOL}" ]; then
            if [ "${FIRST_TOOL}" = false ]; then
                echo "," >> ${JSON_OUTPUT}
            fi
            echo -n "        \"${TOOL}\"" >> ${JSON_OUTPUT}
            FIRST_TOOL=false
        fi
    done <<< "${TOOLS_LIST}"
    
    echo "" >> ${JSON_OUTPUT}
    echo "      ]" >> ${JSON_OUTPUT}
    
    TOTAL_TOOLS=$((TOTAL_TOOLS + TOOL_COUNT))
done

# Close JSON
echo "    }" >> ${JSON_OUTPUT}
echo "  }," >> ${JSON_OUTPUT}
echo "  \"summary\": {" >> ${JSON_OUTPUT}
echo "    \"total_containers\": ${#CONTAINERS[@]}," >> ${JSON_OUTPUT}
echo "    \"total_tools\": ${TOTAL_TOOLS}" >> ${JSON_OUTPUT}
echo "  }" >> ${JSON_OUTPUT}
echo "}" >> ${JSON_OUTPUT}

# Add summary totals
echo "========================================" >> ${SUMMARY_OUTPUT}
echo "TOTALS:" >> ${SUMMARY_OUTPUT}
echo "  Containers: ${#CONTAINERS[@]}" >> ${SUMMARY_OUTPUT}
echo "  Total Tools: ${TOTAL_TOOLS}" >> ${SUMMARY_OUTPUT}
echo "" >> ${SUMMARY_OUTPUT}

# Create symlinks to latest versions
ln -sf tool_catalog_${TIMESTAMP}.json ${OUTPUT_DIR}/tool_catalog_latest.json
ln -sf tool_catalog_summary_${TIMESTAMP}.txt ${OUTPUT_DIR}/tool_catalog_summary_latest.txt

echo ""
echo "=========================================="
echo "✅ Tool catalog generated successfully!"
echo "JSON: ${JSON_OUTPUT}"
echo "Summary: ${SUMMARY_OUTPUT}"
echo "Total Tools: ${TOTAL_TOOLS}"
echo "=========================================="
