#!/bin/bash -ue
echo "Testing RNA-seq container tools..." > rnaseq_tools.txt
echo "" >> rnaseq_tools.txt

echo "STAR version:" >> rnaseq_tools.txt
STAR --version 2>&1 | head -1 >> rnaseq_tools.txt || echo "STAR not found" >> rnaseq_tools.txt
echo "" >> rnaseq_tools.txt

echo "HISAT2 version:" >> rnaseq_tools.txt
hisat2 --version 2>&1 | head -1 >> rnaseq_tools.txt || echo "HISAT2 not found" >> rnaseq_tools.txt
echo "" >> rnaseq_tools.txt

echo "Salmon version:" >> rnaseq_tools.txt
salmon --version 2>&1 >> rnaseq_tools.txt || echo "Salmon not found" >> rnaseq_tools.txt
echo "" >> rnaseq_tools.txt

echo "featureCounts version:" >> rnaseq_tools.txt
featureCounts -v 2>&1 | head -1 >> rnaseq_tools.txt || echo "featureCounts not found" >> rnaseq_tools.txt
echo "" >> rnaseq_tools.txt

echo "samtools version:" >> rnaseq_tools.txt
samtools --version 2>&1 | head -1 >> rnaseq_tools.txt || echo "samtools not found" >> rnaseq_tools.txt
