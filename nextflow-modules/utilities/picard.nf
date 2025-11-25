/*
 * Picard Module
 * 
 * Picard - Suite of tools for manipulating high-throughput sequencing data
 * MarkDuplicates, metrics collection, BAM manipulation
 * Available in all pipeline containers
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Picard MarkDuplicates
 */
process PICARD_MARKDUPLICATES {
    tag "markdup_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/marked_duplicates", mode: 'copy',
        pattern: "*.bam"
    publishDir "${params.outdir}/qc/picard", mode: 'copy',
        pattern: "*.metrics.txt"
    
    memory params.picard?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.marked.bam"), emit: bam
    tuple val(sample_id), path("${sample_id}.marked.bam.bai"), emit: bai
    path "${sample_id}.markdup.metrics.txt", emit: metrics
    
    script:
    def java_mem = (task.memory.toGiga() * 0.8).intValue()
    
    """
    picard -Xmx${java_mem}g MarkDuplicates \\
        INPUT=${bam} \\
        OUTPUT=${sample_id}.marked.bam \\
        METRICS_FILE=${sample_id}.markdup.metrics.txt \\
        CREATE_INDEX=true \\
        VALIDATION_STRINGENCY=LENIENT
    """
}

/*
 * Picard CollectAlignmentSummaryMetrics
 */
process PICARD_ALIGNMENT_METRICS {
    tag "alignment_metrics_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/picard", mode: 'copy'
    
    memory params.picard?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path reference
    
    output:
    tuple val(sample_id), path("${sample_id}.alignment_metrics.txt"), emit: metrics
    
    script:
    def java_mem = (task.memory.toGiga() * 0.8).intValue()
    
    """
    picard -Xmx${java_mem}g CollectAlignmentSummaryMetrics \\
        INPUT=${bam} \\
        OUTPUT=${sample_id}.alignment_metrics.txt \\
        REFERENCE_SEQUENCE=${reference}
    """
}

/*
 * Picard CollectInsertSizeMetrics
 */
process PICARD_INSERT_SIZE_METRICS {
    tag "insert_size_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/picard", mode: 'copy'
    
    memory params.picard?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.insert_size_metrics.txt"), emit: metrics
    path "${sample_id}.insert_size_histogram.pdf", emit: histogram
    
    script:
    def java_mem = (task.memory.toGiga() * 0.8).intValue()
    
    """
    picard -Xmx${java_mem}g CollectInsertSizeMetrics \\
        INPUT=${bam} \\
        OUTPUT=${sample_id}.insert_size_metrics.txt \\
        HISTOGRAM_FILE=${sample_id}.insert_size_histogram.pdf
    """
}

/*
 * Picard CollectRnaSeqMetrics
 */
process PICARD_RNASEQ_METRICS {
    tag "rnaseq_metrics_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/picard", mode: 'copy'
    
    memory params.picard?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path refflat
    path ribosomal_intervals
    
    output:
    tuple val(sample_id), path("${sample_id}.rnaseq_metrics.txt"), emit: metrics
    
    script:
    def java_mem = (task.memory.toGiga() * 0.8).intValue()
    def strand = params.picard?.strand_specificity ?: "NONE"  // "NONE", "FIRST_READ_TRANSCRIPTION_STRAND", "SECOND_READ_TRANSCRIPTION_STRAND"
    def ribo_opt = ribosomal_intervals ? "RIBOSOMAL_INTERVALS=${ribosomal_intervals}" : ""
    
    """
    picard -Xmx${java_mem}g CollectRnaSeqMetrics \\
        INPUT=${bam} \\
        OUTPUT=${sample_id}.rnaseq_metrics.txt \\
        REF_FLAT=${refflat} \\
        ${ribo_opt} \\
        STRAND_SPECIFICITY=${strand}
    """
}

/*
 * Picard CollectGcBiasMetrics
 */
process PICARD_GC_BIAS_METRICS {
    tag "gc_bias_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/picard", mode: 'copy'
    
    memory params.picard?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path reference
    
    output:
    tuple val(sample_id), path("${sample_id}.gc_bias_metrics.txt"), emit: metrics
    path "${sample_id}.gc_bias.pdf", emit: chart
    path "${sample_id}.gc_bias_summary.txt", emit: summary
    
    script:
    def java_mem = (task.memory.toGiga() * 0.8).intValue()
    
    """
    picard -Xmx${java_mem}g CollectGcBiasMetrics \\
        INPUT=${bam} \\
        OUTPUT=${sample_id}.gc_bias_metrics.txt \\
        CHART=${sample_id}.gc_bias.pdf \\
        SUMMARY_OUTPUT=${sample_id}.gc_bias_summary.txt \\
        REFERENCE_SEQUENCE=${reference}
    """
}

/*
 * Picard SortSam
 */
process PICARD_SORT {
    tag "sort_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/sorted_bams", mode: 'copy'
    
    memory params.picard?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    val sort_order  // "coordinate", "queryname"
    
    output:
    tuple val(sample_id), path("${sample_id}.sorted.bam"), emit: bam
    
    script:
    def java_mem = (task.memory.toGiga() * 0.8).intValue()
    
    """
    picard -Xmx${java_mem}g SortSam \\
        INPUT=${bam} \\
        OUTPUT=${sample_id}.sorted.bam \\
        SORT_ORDER=${sort_order}
    """
}

/*
 * Picard AddOrReplaceReadGroups
 */
process PICARD_ADD_READ_GROUPS {
    tag "add_rg_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bams_with_rg", mode: 'copy'
    
    memory params.picard?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.rg.bam"), emit: bam
    
    script:
    def java_mem = (task.memory.toGiga() * 0.8).intValue()
    
    """
    picard -Xmx${java_mem}g AddOrReplaceReadGroups \\
        INPUT=${bam} \\
        OUTPUT=${sample_id}.rg.bam \\
        RGID=${sample_id} \\
        RGSM=${sample_id} \\
        RGLB=lib1 \\
        RGPL=ILLUMINA \\
        RGPU=unit1
    """
}

/*
 * Workflow: Standard Picard QC pipeline
 */
workflow PICARD_QC_WORKFLOW {
    take:
    bam_ch      // channel: [ val(sample_id), path(bam) ]
    reference   // path: reference genome
    
    main:
    // Mark duplicates
    PICARD_MARKDUPLICATES(bam_ch)
    
    // Collect metrics
    PICARD_ALIGNMENT_METRICS(PICARD_MARKDUPLICATES.out.bam, reference)
    PICARD_INSERT_SIZE_METRICS(PICARD_MARKDUPLICATES.out.bam)
    
    emit:
    bam = PICARD_MARKDUPLICATES.out.bam
    bai = PICARD_MARKDUPLICATES.out.bai
    markdup_metrics = PICARD_MARKDUPLICATES.out.metrics
    alignment_metrics = PICARD_ALIGNMENT_METRICS.out.metrics
    insert_size_metrics = PICARD_INSERT_SIZE_METRICS.out.metrics
}
