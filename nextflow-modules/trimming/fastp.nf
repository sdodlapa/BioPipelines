/*
 * fastp Module
 * 
 * fastp - Fast all-in-one FASTQ preprocessor
 * Quality control, adapter trimming, and filtering
 * Uses existing containers (available in most)
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * fastp - Process paired-end reads
 */
process FASTP_PE {
    tag "fastp_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed/fastp", mode: 'copy',
        pattern: "*.fastq.gz"
    publishDir "${params.outdir}/qc/fastp", mode: 'copy',
        pattern: "*.{json,html}"
    
    cpus params.fastp?.cpus ?: 4
    memory params.fastp?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    tuple val(sample_id), path("${sample_id}_R1_trimmed.fastq.gz"), path("${sample_id}_R2_trimmed.fastq.gz"), emit: reads
    path "${sample_id}_fastp.json", emit: json
    path "${sample_id}_fastp.html", emit: html
    
    script:
    """
    fastp \\
        -i ${reads[0]} \\
        -I ${reads[1]} \\
        -o ${sample_id}_R1_trimmed.fastq.gz \\
        -O ${sample_id}_R2_trimmed.fastq.gz \\
        --thread ${task.cpus} \\
        --detect_adapter_for_pe \\
        --json ${sample_id}_fastp.json \\
        --html ${sample_id}_fastp.html
    """
}

/*
 * fastp - Process single-end reads
 */
process FASTP_SE {
    tag "fastp_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed/fastp", mode: 'copy',
        pattern: "*.fastq.gz"
    publishDir "${params.outdir}/qc/fastp", mode: 'copy',
        pattern: "*.{json,html}"
    
    cpus params.fastp?.cpus ?: 4
    memory params.fastp?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    tuple val(sample_id), path("${sample_id}_trimmed.fastq.gz"), emit: reads
    path "${sample_id}_fastp.json", emit: json
    path "${sample_id}_fastp.html", emit: html
    
    script:
    """
    fastp \\
        -i ${reads} \\
        -o ${sample_id}_trimmed.fastq.gz \\
        --thread ${task.cpus} \\
        --json ${sample_id}_fastp.json \\
        --html ${sample_id}_fastp.html
    """
}

/*
 * fastp with UMI extraction
 */
process FASTP_UMI {
    tag "fastp_umi_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed/fastp", mode: 'copy',
        pattern: "*.fastq.gz"
    publishDir "${params.outdir}/qc/fastp", mode: 'copy',
        pattern: "*.{json,html}"
    
    cpus params.fastp?.cpus ?: 4
    memory params.fastp?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    val umi_loc    // "read1", "read2", "per_read", "per_index"
    val umi_len
    
    output:
    tuple val(sample_id), path("${sample_id}_R1_trimmed.fastq.gz"), path("${sample_id}_R2_trimmed.fastq.gz"), emit: reads
    path "${sample_id}_fastp.json", emit: json
    path "${sample_id}_fastp.html", emit: html
    
    script:
    """
    fastp \\
        -i ${reads[0]} \\
        -I ${reads[1]} \\
        -o ${sample_id}_R1_trimmed.fastq.gz \\
        -O ${sample_id}_R2_trimmed.fastq.gz \\
        --thread ${task.cpus} \\
        --umi \\
        --umi_loc ${umi_loc} \\
        --umi_len ${umi_len} \\
        --json ${sample_id}_fastp.json \\
        --html ${sample_id}_fastp.html
    """
}

/*
 * Workflow: fastp preprocessing pipeline
 */
workflow FASTP_PIPELINE {
    take:
    reads_ch       // channel: [ val(sample_id), path(reads) ]
    paired_end     // val: true/false
    
    main:
    if (paired_end) {
        FASTP_PE(reads_ch)
        reads_out = FASTP_PE.out.reads
        json_out = FASTP_PE.out.json
        html_out = FASTP_PE.out.html
    } else {
        FASTP_SE(reads_ch)
        reads_out = FASTP_SE.out.reads
        json_out = FASTP_SE.out.json
        html_out = FASTP_SE.out.html
    }
    
    emit:
    reads = reads_out
    json = json_out
    html = html_out
}
