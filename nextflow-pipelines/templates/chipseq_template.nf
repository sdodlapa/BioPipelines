#!/usr/bin/env nextflow

/*
 * ChIP-seq Peak Calling Pipeline
 * Template for BioPipelines Workflow Composer
 * 
 * Generated: {{DATE}}
 * Analysis: {{ANALYSIS_TYPE}}
 * Organism: {{ORGANISM}}
 */

nextflow.enable.dsl=2

// ============================================================================
// Module Imports (relative to nextflow-pipelines/modules)
// ============================================================================

include { FASTQC         } from '../modules/qc/fastqc/main'
include { BOWTIE2_ALIGN  } from '../modules/alignment/bowtie2/main'
include { MACS2_CALLPEAK } from '../modules/peakcalling/macs2/main'
include { MULTIQC        } from '../modules/qc/multiqc/main'

// ============================================================================
// Parameters
// ============================================================================

params.treatment     = "${projectDir}/data/chip/*_treatment_R{1,2}.fastq.gz"
params.control       = "${projectDir}/data/chip/*_input_R{1,2}.fastq.gz"
params.bowtie2_index = "${projectDir}/data/references/bowtie2_index"
params.genome        = "{{GENOME_BUILD}}"
params.outdir        = "${projectDir}/results"
params.single_end    = {{SINGLE_END}}

// ============================================================================
// Input Channels
// ============================================================================

workflow {
    // Treatment samples (ChIP)
    if (params.single_end) {
        treatment_ch = Channel
            .fromPath(params.treatment, checkIfExists: true)
            .map { file -> 
                def sample_id = file.baseName.replaceAll(/_R[12].*/, '')
                tuple([id: sample_id, type: 'treatment', single_end: true], file)
            }
    } else {
        treatment_ch = Channel
            .fromFilePairs(params.treatment, checkIfExists: true)
            .map { sample_id, reads -> 
                tuple([id: sample_id, type: 'treatment', single_end: false], reads)
            }
    }
    
    // Control samples (Input)
    if (params.single_end) {
        control_ch = Channel
            .fromPath(params.control, checkIfExists: true)
            .map { file ->
                def sample_id = file.baseName.replaceAll(/_R[12].*/, '')
                tuple([id: sample_id, type: 'control', single_end: true], file)
            }
    } else {
        control_ch = Channel
            .fromFilePairs(params.control, checkIfExists: true)
            .map { sample_id, reads ->
                tuple([id: sample_id, type: 'control', single_end: false], reads)
            }
    }
    
    // Bowtie2 index
    bowtie2_index = file(params.bowtie2_index, checkIfExists: true)
    
    // ========================================================================
    // Workflow Steps
    // ========================================================================
    
    // Combine all samples for QC
    all_samples = treatment_ch.mix(control_ch)
    
    // 1. Quality Control
    FASTQC(all_samples)
    
    // 2. Alignment
    BOWTIE2_ALIGN(all_samples, bowtie2_index)
    
    // 3. Separate treatment and control BAMs
    treatment_bams = BOWTIE2_ALIGN.out.bam
        .filter { meta, bam, bai -> meta.type == 'treatment' }
    
    control_bam = BOWTIE2_ALIGN.out.bam
        .filter { meta, bam, bai -> meta.type == 'control' }
        .map { meta, bam, bai -> bam }
        .first()
    
    // 4. Peak Calling
    MACS2_CALLPEAK(treatment_bams, control_bam)
    
    // 5. MultiQC Report
    MULTIQC(
        FASTQC.out.zip
            .mix(MACS2_CALLPEAK.out.qc)
            .collect()
    )
    
    // ========================================================================
    // Completion Summary
    // ========================================================================
    
    workflow.onComplete {
        println """
        ============================================================
        ChIP-seq Pipeline Complete
        ============================================================
        Organism: {{ORGANISM}}
        Genome:   {{GENOME_BUILD}}
        
        Status:   ${workflow.success ? '✅ SUCCESS' : '❌ FAILED'}
        Duration: ${workflow.duration}
        
        Outputs:
        - Peaks:   ${params.outdir}/peaks/
        - QC:      ${params.outdir}/multiqc/
        - BAMs:    ${params.outdir}/alignments/
        ============================================================
        """.stripIndent()
    }
}
