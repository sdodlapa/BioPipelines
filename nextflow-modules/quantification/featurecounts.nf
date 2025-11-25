/*
 * featureCounts Quantification Module
 * 
 * Subread featureCounts for gene-level quantification
 * Efficient and accurate read counting from BAM files
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * featureCounts - Gene-level quantification
 */
process FEATURECOUNTS {
    tag "featureCounts_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification", mode: 'copy',
        pattern: "*.txt"
    publishDir "${params.outdir}/quantification/summary", mode: 'copy',
        pattern: "*.summary"
    
    cpus params.featureCounts?.cpus ?: 4
    memory params.featureCounts?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    val strand_specific  // 0 (unstranded), 1 (stranded), 2 (reversely stranded)
    val feature_type     // "exon" (default), "gene", "transcript"
    val attribute_type   // "gene_id" (default), "transcript_id", "gene_name"
    
    output:
    tuple val(sample_id), path("${sample_id}.counts.txt"), emit: counts
    path "${sample_id}.counts.txt.summary", emit: summary
    
    script:
    def paired = params.featureCounts?.paired ? "-p" : ""
    def feature = feature_type ?: "exon"
    def attribute = attribute_type ?: "gene_id"
    
    """
    featureCounts \\
        -T ${task.cpus} \\
        -a ${gtf} \\
        -o ${sample_id}.counts.txt \\
        -t ${feature} \\
        -g ${attribute} \\
        -s ${strand_specific} \\
        ${paired} \\
        ${bam}
    """
}

/*
 * featureCounts Multi-sample - Count multiple samples together
 */
process FEATURECOUNTS_MULTI {
    tag "featureCounts_multi"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    cpus params.featureCounts?.cpus ?: 8
    memory params.featureCounts?.memory ?: '16.GB'
    
    input:
    path bams
    path gtf
    val strand_specific
    val feature_type
    val attribute_type
    
    output:
    path "multi_sample.counts.txt", emit: counts
    path "multi_sample.counts.txt.summary", emit: summary
    
    script:
    def paired = params.featureCounts?.paired ? "-p" : ""
    def feature = feature_type ?: "exon"
    def attribute = attribute_type ?: "gene_id"
    
    """
    featureCounts \\
        -T ${task.cpus} \\
        -a ${gtf} \\
        -o multi_sample.counts.txt \\
        -t ${feature} \\
        -g ${attribute} \\
        -s ${strand_specific} \\
        ${paired} \\
        ${bams}
    """
}

/*
 * featureCounts with Metadata - Add gene/transcript metadata
 */
process FEATURECOUNTS_WITH_METADATA {
    tag "featureCounts_metadata_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    cpus params.featureCounts?.cpus ?: 4
    memory params.featureCounts?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    val strand_specific
    
    output:
    tuple val(sample_id), path("${sample_id}.counts.txt"), emit: counts
    tuple val(sample_id), path("${sample_id}.counts.detailed.txt"), emit: detailed
    path "${sample_id}.counts.txt.summary", emit: summary
    
    script:
    def paired = params.featureCounts?.paired ? "-p" : ""
    
    """
    # Standard counts
    featureCounts \\
        -T ${task.cpus} \\
        -a ${gtf} \\
        -o ${sample_id}.counts.txt \\
        -s ${strand_specific} \\
        ${paired} \\
        ${bam}
    
    # Detailed counts with extra annotations
    featureCounts \\
        -T ${task.cpus} \\
        -a ${gtf} \\
        -o ${sample_id}.counts.detailed.txt \\
        -s ${strand_specific} \\
        --extraAttributes gene_name,gene_biotype \\
        ${paired} \\
        ${bam}
    """
}

/*
 * featureCounts Fractional - Count multi-mapping reads fractionally
 */
process FEATURECOUNTS_FRACTIONAL {
    tag "featureCounts_frac_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification", mode: 'copy'
    
    cpus params.featureCounts?.cpus ?: 4
    memory params.featureCounts?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path gtf
    val strand_specific
    
    output:
    tuple val(sample_id), path("${sample_id}.counts.fractional.txt"), emit: counts
    path "${sample_id}.counts.fractional.txt.summary", emit: summary
    
    script:
    def paired = params.featureCounts?.paired ? "-p" : ""
    
    """
    featureCounts \\
        -T ${task.cpus} \\
        -a ${gtf} \\
        -o ${sample_id}.counts.fractional.txt \\
        -s ${strand_specific} \\
        -M \\
        --fraction \\
        ${paired} \\
        ${bam}
    """
}

/*
 * Workflow: Standard featureCounts pipeline
 */
workflow FEATURECOUNTS_PIPELINE {
    take:
    bam_ch           // channel: [ val(sample_id), path(bam) ]
    gtf              // path: annotation GTF
    strand_specific  // val: 0, 1, or 2
    
    main:
    FEATURECOUNTS(
        bam_ch,
        gtf,
        strand_specific,
        "exon",
        "gene_id"
    )
    
    emit:
    counts = FEATURECOUNTS.out.counts
    summary = FEATURECOUNTS.out.summary
}
