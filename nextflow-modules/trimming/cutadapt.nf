/*
 * Cutadapt Module
 * 
 * Cutadapt - Removes adapter sequences from reads
 * Fast and flexible adapter trimming
 * Available in all pipeline containers
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Cutadapt - Paired-end trimming
 */
process CUTADAPT_PE {
    tag "cutadapt_pe_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed_reads", mode: 'copy'
    publishDir "${params.outdir}/trimmed_reads/logs", mode: 'copy',
        pattern: "*.log"
    
    cpus params.cutadapt?.cpus ?: 4
    memory params.cutadapt?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    val adapter_r1
    val adapter_r2
    
    output:
    tuple val(sample_id), path("${sample_id}_R{1,2}_trimmed.fastq.gz"), emit: trimmed_reads
    path "${sample_id}_cutadapt.log", emit: log
    path "${sample_id}_cutadapt.json", emit: json
    
    script:
    def quality = params.cutadapt?.quality ?: 20
    def minimum_length = params.cutadapt?.minimum_length ?: 20
    def adapter_r1_opt = adapter_r1 ? "-a ${adapter_r1}" : ""
    def adapter_r2_opt = adapter_r2 ? "-A ${adapter_r2}" : ""
    
    """
    cutadapt \\
        -j ${task.cpus} \\
        -q ${quality} \\
        -m ${minimum_length} \\
        ${adapter_r1_opt} \\
        ${adapter_r2_opt} \\
        -o ${sample_id}_R1_trimmed.fastq.gz \\
        -p ${sample_id}_R2_trimmed.fastq.gz \\
        --json ${sample_id}_cutadapt.json \\
        ${reads[0]} ${reads[1]} \\
        > ${sample_id}_cutadapt.log 2>&1
    """
}

/*
 * Cutadapt - Single-end trimming
 */
process CUTADAPT_SE {
    tag "cutadapt_se_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed_reads", mode: 'copy'
    publishDir "${params.outdir}/trimmed_reads/logs", mode: 'copy',
        pattern: "*.log"
    
    cpus params.cutadapt?.cpus ?: 4
    memory params.cutadapt?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    val adapter
    
    output:
    tuple val(sample_id), path("${sample_id}_trimmed.fastq.gz"), emit: trimmed_reads
    path "${sample_id}_cutadapt.log", emit: log
    path "${sample_id}_cutadapt.json", emit: json
    
    script:
    def quality = params.cutadapt?.quality ?: 20
    def minimum_length = params.cutadapt?.minimum_length ?: 20
    def adapter_opt = adapter ? "-a ${adapter}" : ""
    
    """
    cutadapt \\
        -j ${task.cpus} \\
        -q ${quality} \\
        -m ${minimum_length} \\
        ${adapter_opt} \\
        -o ${sample_id}_trimmed.fastq.gz \\
        --json ${sample_id}_cutadapt.json \\
        ${reads} \\
        > ${sample_id}_cutadapt.log 2>&1
    """
}

/*
 * Cutadapt - Auto-detect and remove adapters
 */
process CUTADAPT_AUTO {
    tag "cutadapt_auto_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed_reads", mode: 'copy'
    
    cpus params.cutadapt?.cpus ?: 4
    memory params.cutadapt?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    tuple val(sample_id), path("${sample_id}_*_trimmed.fastq.gz"), emit: trimmed_reads
    path "${sample_id}_cutadapt.log", emit: log
    path "${sample_id}_cutadapt.json", emit: json
    
    script:
    def quality = params.cutadapt?.quality ?: 20
    def minimum_length = params.cutadapt?.minimum_length ?: 20
    
    if (reads instanceof List) {
        // Paired-end with auto-detection
        """
        cutadapt \\
            -j ${task.cpus} \\
            -q ${quality} \\
            -m ${minimum_length} \\
            -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCA \\
            -A AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT \\
            --times 2 \\
            -o ${sample_id}_R1_trimmed.fastq.gz \\
            -p ${sample_id}_R2_trimmed.fastq.gz \\
            --json ${sample_id}_cutadapt.json \\
            ${reads[0]} ${reads[1]} \\
            > ${sample_id}_cutadapt.log 2>&1
        """
    } else {
        // Single-end with auto-detection
        """
        cutadapt \\
            -j ${task.cpus} \\
            -q ${quality} \\
            -m ${minimum_length} \\
            -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCA \\
            --times 2 \\
            -o ${sample_id}_trimmed.fastq.gz \\
            --json ${sample_id}_cutadapt.json \\
            ${reads} \\
            > ${sample_id}_cutadapt.log 2>&1
        """
    }
}

/*
 * Cutadapt - Multiple adapter removal
 */
process CUTADAPT_MULTI {
    tag "cutadapt_multi_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/trimmed_reads", mode: 'copy'
    
    cpus params.cutadapt?.cpus ?: 4
    memory params.cutadapt?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path adapter_file
    
    output:
    tuple val(sample_id), path("${sample_id}_*_trimmed.fastq.gz"), emit: trimmed_reads
    path "${sample_id}_cutadapt.log", emit: log
    
    script:
    def quality = params.cutadapt?.quality ?: 20
    def minimum_length = params.cutadapt?.minimum_length ?: 20
    
    if (reads instanceof List) {
        """
        cutadapt \\
            -j ${task.cpus} \\
            -q ${quality} \\
            -m ${minimum_length} \\
            -a file:${adapter_file} \\
            -A file:${adapter_file} \\
            -o ${sample_id}_R1_trimmed.fastq.gz \\
            -p ${sample_id}_R2_trimmed.fastq.gz \\
            ${reads[0]} ${reads[1]} \\
            > ${sample_id}_cutadapt.log 2>&1
        """
    } else {
        """
        cutadapt \\
            -j ${task.cpus} \\
            -q ${quality} \\
            -m ${minimum_length} \\
            -a file:${adapter_file} \\
            -o ${sample_id}_trimmed.fastq.gz \\
            ${reads} \\
            > ${sample_id}_cutadapt.log 2>&1
        """
    }
}

/*
 * Workflow: Standard Cutadapt pipeline with auto-detection
 */
workflow CUTADAPT_WORKFLOW {
    take:
    reads_ch   // channel: [ val(sample_id), path(reads) ]
    
    main:
    CUTADAPT_AUTO(reads_ch)
    
    emit:
    reads = CUTADAPT_AUTO.out.trimmed_reads
    logs = CUTADAPT_AUTO.out.log
    json = CUTADAPT_AUTO.out.json
}
