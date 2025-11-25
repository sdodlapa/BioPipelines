#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import modules
include { FASTQC } from './nextflow-modules/fastqc/main.nf'
include { BISMARK_ALIGN } from './nextflow-modules/bismark/main.nf'
include { MULTIQC; MULTIQC_CUSTOM } from './nextflow-modules/multiqc/main.nf'
include { TRIM_GALORE } from './nextflow-modules/trim_galore/main.nf'

// Define parameters
params.input = './data/*.fastq'
params.genome = './genome/'
params.outdir = './results'

// Define input channel
Channel.fromPath(params.input)
    .set { fastq_files }

// Quality control with FastQC
fastq_files
    | FASTQC

// Trimming with Trim Galore
fastq_files
    | TRIM_GALORE
    | map { file -> file.name.replace('.fastq', '_trimmed.fq') }
    | set { trimmed_fastq_files }

// Alignment with Bismark
trimmed_fastq_files
    | BISMARK_ALIGN(genome: params.genome)
    | set { bismark_results }

// MultiQC report generation
bismark_results
    | MULTIQC

// Custom MultiQC report (if needed)
bismark_results
    | MULTIQC_CUSTOM

// Output aggregation
bismark_results
    | view { "Bismark alignment results: $it" }