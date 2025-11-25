#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import the necessary module
include { BWAMEM_ALIGN } from './nextflow-modules/bwamem/main.nf'

// Define parameters
params.reads = './data/fastq/*_{1,2}.fastq.gz'
params.genome = './data/genome/genome.fa'
params.outdir = './results'

// Create input channel for paired-end FASTQ files
Channel
    .fromFilePairs(params.reads, flat: true)
    .set { read_pairs }

// Create input channel for the reference genome
Channel
    .fromPath(params.genome)
    .set { genome_fasta }

// Align reads using BWA-MEM
process alignReads {
    input:
    tuple val(sample_id), path(reads)
    path genome_fasta

    output:
    path "${sample_id}.bam" into aligned_bams

    script:
    """
    bwamem align -t 4 -R '@RG\\tID:${sample_id}\\tSM:${sample_id}' $genome_fasta $reads > ${sample_id}.bam
    """
}

// Collect all BAM files into a single channel for further analysis
aligned_bams
    .collectFile(name: 'all_samples.bam', emit: true)
    .set { final_bam }

// Define the workflow
workflow {
    read_pairs
        | alignReads
}