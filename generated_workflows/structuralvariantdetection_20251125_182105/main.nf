#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Import modules
include { BWAMEM_ALIGN } from './nextflow-modules/bwamem/main.nf'
include { SAMTOOLS_SORT; SAMTOOLS_INDEX; SAMTOOLS_FLAGSTAT; SAMTOOLS_STATS; SAMTOOLS_IDXSTATS; SAMTOOLS_VIEW; SAMTOOLS_MERGE } from './nextflow-modules/samtools/main.nf'
include { DELLY_CALL; DELLY_FILTER; DELLY_SOMATIC } from './nextflow-modules/delly/main.nf'

// Define parameters
params.input_bam = file('input/*.bam')
params.reference_genome = file('path/to/reference/genome.fa')
params.output_dir = 'results'

// Create input channel
Channel
    .fromPath(params.input_bam)
    .set { bam_files }

// Align reads (if needed, otherwise skip this step)
process align_reads {
    input:
    path bam_file from bam_files

    output:
    path "${bam_file.baseName}.sorted.bam" into sorted_bam_files

    script:
    """
    # Assuming BAM files are already aligned, we skip alignment
    cp $bam_file ${bam_file.baseName}.sorted.bam
    """
}

// Sort BAM files
SAMTOOLS_SORT {
    input:
    path bam_file from sorted_bam_files

    output:
    path "${bam_file.baseName}.sorted.bam" into sorted_bam_files
}

// Index BAM files
SAMTOOLS_INDEX {
    input:
    path bam_file from sorted_bam_files

    output:
    path "${bam_file}.bai" into indexed_bam_files
}

// Call structural variants
DELLY_CALL {
    input:
    path bam_file from sorted_bam_files
    path bai_file from indexed_bam_files
    path reference from params.reference_genome

    output:
    path "${bam_file.baseName}.bcf" into sv_calls
}

// Filter structural variants
DELLY_FILTER {
    input:
    path bcf_file from sv_calls

    output:
    path "${bcf_file.baseName}.filtered.bcf" into filtered_sv_calls
}

// Aggregate results
process aggregate_results {
    input:
    path filtered_bcf from filtered_sv_calls.collect()

    output:
    path "${params.output_dir}/all_filtered_sv_calls.bcf"

    script:
    """
    mkdir -p ${params.output_dir}
    bcftools concat -o ${params.output_dir}/all_filtered_sv_calls.bcf $filtered_bcf
    """
}