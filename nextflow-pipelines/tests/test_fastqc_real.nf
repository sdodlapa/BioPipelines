#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Test FastQC module with actual RNA-seq data
include { FASTQC } from '../modules/qc/fastqc/main'

// Import containers config
params.container_base = '/home/sdodl001_odu_edu/BioPipelines/containers/images'
params.containers = [:]
params.containers.rnaseq = "${params.container_base}/rna-seq_1.0.0.sif"

workflow {
    // Create input channel with meta map and actual FASTQ files
    input_ch = Channel.of(
        [
            [id: 'mut_rep1', single_end: false, condition: 'mutant'],
            [
                file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/mut_rep1_R1.trimmed.fastq.gz'),
                file('/scratch/sdodl001/BioPipelines/data/processed/rna_seq/mut_rep1_R2.trimmed.fastq.gz')
            ]
        ]
    )
    
    // Run FastQC
    FASTQC(input_ch)
    
    // Display results
    FASTQC.out.html
        | map { meta, html -> "✅ FastQC HTML for ${meta.id}: ${html}" }
        | view
    
    FASTQC.out.zip
        | map { meta, zip -> "✅ FastQC ZIP for ${meta.id}: ${zip}" }
        | view
    
    FASTQC.out.versions
        | view { "📦 Versions: $it" }
}

workflow.onComplete {
    println "\n" + "="*60
    println "FastQC Module Test Complete"
    println "="*60
    println "Status:   ${workflow.success ? '✅ SUCCESS' : '❌ FAILED'}"
    println "Duration: ${workflow.duration}"
    println "Work dir: ${workflow.workDir}"
    if (workflow.success) {
        println "\n💡 Check FastQC outputs in work directory"
        println "   Find HTML reports: find ${workflow.workDir} -name '*.html'"
    }
    println "="*60
}
