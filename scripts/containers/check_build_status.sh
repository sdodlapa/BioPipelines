#!/bin/bash
# Check container build status

echo "════════════════════════════════════════════════════════"
echo "Container Build Status - $(date)"
echo "════════════════════════════════════════════════════════"

echo -e "\n📦 COMPLETED CONTAINERS:"
echo "────────────────────────────────────────────────────────"
for container in /home/sdodl001_odu_edu/BioPipelines/containers/images/*.sif; do
    if [ -f "$container" ]; then
        name=$(basename "$container" .sif)
        size=$(du -h "$container" | cut -f1)
        echo "  ✓ $name ($size)"
    fi
done

COMPLETED=$(ls /home/sdodl001_odu_edu/BioPipelines/containers/images/*.sif 2>/dev/null | wc -l)
echo "────────────────────────────────────────────────────────"
echo "  Total: $COMPLETED/11 containers"

echo -e "\n🔄 RUNNING/QUEUED BUILDS:"
echo "────────────────────────────────────────────────────────"
squeue -u $USER -o "  %.10i %.20j %.10T %.10M" | grep build || echo "  None"

echo -e "\n❌ RECENT FAILURES:"
echo "────────────────────────────────────────────────────────"
sacct -u $USER --format=JobID,JobName%-20,State,ExitCode -S $(date -d '1 hour ago' +%Y-%m-%dT%H:%M) | grep -E "build.*FAILED" | tail -5 || echo "  None"

echo -e "\n════════════════════════════════════════════════════════"
if [ $COMPLETED -eq 11 ]; then
    echo "🎉 ALL CONTAINERS COMPLETE! Ready to run pipelines."
else
    REMAINING=$((11 - COMPLETED))
    echo "⏳ $REMAINING containers remaining..."
    echo "   Monitor: watch -n 30 ~/BioPipelines/scripts/containers/check_build_status.sh"
fi
echo "════════════════════════════════════════════════════════"
