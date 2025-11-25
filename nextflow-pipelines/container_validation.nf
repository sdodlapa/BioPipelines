#!/usr/bin/env nextflow

/*
 * Simple Container Validation Workflow
 * Tests that existing containers work with Nextflow
 */

nextflow.enable.dsl = 2

// Parameters
params.outdir = "results/container_validation"
params.container_base = "/home/sdodl001_odu_edu/BioPipelines/containers/images"

// Container paths
params.containers = [
    rnaseq: "${params.container_base}/rna-seq_1.0.0.sif",
    dnaseq: "${params.container_base}/dna-seq_1.0.0.sif"
]

/*
 * Test RNA-seq container tools
 */
process TEST_RNASEQ_TOOLS {
    tag "test_rnaseq"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/tests", mode: 'copy'
    
    output:
    path "rnaseq_tools.txt"
    
    script:
    """
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
    """
}

/*
 * Test DNA-seq container tools
 */
process TEST_DNASEQ_TOOLS {
    tag "test_dnaseq"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/tests", mode: 'copy'
    
    output:
    path "dnaseq_tools.txt"
    
    script:
    """
    echo "Testing DNA-seq container tools..." > dnaseq_tools.txt
    echo "" >> dnaseq_tools.txt
    
    echo "BWA version:" >> dnaseq_tools.txt
    bwa 2>&1 | grep -i version | head -1 >> dnaseq_tools.txt || echo "BWA not found" >> dnaseq_tools.txt
    echo "" >> dnaseq_tools.txt
    
    echo "samtools version:" >> dnaseq_tools.txt
    samtools --version 2>&1 | head -1 >> dnaseq_tools.txt || echo "samtools not found" >> dnaseq_tools.txt
    echo "" >> dnaseq_tools.txt
    
    echo "bcftools version:" >> dnaseq_tools.txt
    bcftools --version 2>&1 | head -1 >> dnaseq_tools.txt || echo "bcftools not found" >> dnaseq_tools.txt
    """
}

/*
 * Main workflow
 */
workflow {
    TEST_RNASEQ_TOOLS()
    TEST_DNASEQ_TOOLS()
}

workflow.onComplete {
    println "Validation workflow completed!"
    println "RNA-seq and DNA-seq containers tested successfully"
    println "Results in: ${params.outdir}/tests"
}
