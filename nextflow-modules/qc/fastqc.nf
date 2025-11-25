/*
 * FastQC Quality Control Module
 * 
 * FastQC - Quality control tool for high throughput sequence data
 * Generates quality reports for FASTQ files
 * Available in all pipeline containers
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * FastQC - Quality control check
 */
process FASTQC {
    tag "fastqc_${sample_id}"
    container "${params.containers.rnaseq}"  // FastQC available in all containers
    
    publishDir "${params.outdir}/qc/fastqc", mode: 'copy'
    
    cpus params.fastqc?.cpus ?: 2
    memory params.fastqc?.memory ?: '4.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    tuple val(sample_id), path("*_fastqc.html"), emit: html
    tuple val(sample_id), path("*_fastqc.zip"), emit: zip
    
    script:
    """
    fastqc \\
        --threads ${task.cpus} \\
        --outdir . \\
        ${reads}
    """
}

/*
 * FastQC with Custom Contaminants
 */
process FASTQC_CUSTOM {
    tag "fastqc_custom_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/fastqc", mode: 'copy'
    
    cpus params.fastqc?.cpus ?: 2
    memory params.fastqc?.memory ?: '4.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path contaminants
    path adapters
    
    output:
    tuple val(sample_id), path("*_fastqc.html"), emit: html
    tuple val(sample_id), path("*_fastqc.zip"), emit: zip
    
    script:
    def contam_opt = contaminants ? "--contaminants ${contaminants}" : ""
    def adapter_opt = adapters ? "--adapters ${adapters}" : ""
    
    """
    fastqc \\
        --threads ${task.cpus} \\
        --outdir . \\
        ${contam_opt} \\
        ${adapter_opt} \\
        ${reads}
    """
}

/*
 * FastQC for Post-Trimming QC
 */
process FASTQC_POST_TRIM {
    tag "fastqc_trimmed_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/fastqc_trimmed", mode: 'copy'
    
    cpus params.fastqc?.cpus ?: 2
    memory params.fastqc?.memory ?: '4.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    tuple val(sample_id), path("*_fastqc.html"), emit: html
    tuple val(sample_id), path("*_fastqc.zip"), emit: zip
    
    script:
    """
    fastqc \\
        --threads ${task.cpus} \\
        --outdir . \\
        --noextract \\
        ${reads}
    """
}

/*
 * Workflow: Run FastQC on all samples
 */
workflow FASTQC_WORKFLOW {
    take:
    reads_ch  // channel: [ val(sample_id), path(reads) ]
    
    main:
    FASTQC(reads_ch)
    
    emit:
    html = FASTQC.out.html
    zip = FASTQC.out.zip
}
