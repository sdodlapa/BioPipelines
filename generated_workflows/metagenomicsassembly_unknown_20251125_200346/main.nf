#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Define parameters
params.input = './data/*.fastq'
params.outdir = './results'

// Import modules
include { FASTQC } from './nextflow-modules/fastqc/main.nf'
include { MEGAHIT_ASSEMBLE } from './nextflow-modules/megahit/main.nf'
include { PROKKA_ANNOTATE } from './nextflow-modules/prokka/main.nf'

// Create input channel
Channel
    .fromPath(params.input)
    .set { fastq_files }

// Run FastQC
fastqc_results = FASTQC(fastq_files)

// Run MEGAHIT for assembly
assembly_results = MEGAHIT_ASSEMBLE(fastq_files)

// Run Prokka for annotation
annotation_results = PROKKA_ANNOTATE(assembly_results)

// Save results
annotation_results.view { file -> "Annotated file: ${file}" }
annotation_results
    .map { file -> file.copyTo(file("${params.outdir}/annotations/${file.name}")) }