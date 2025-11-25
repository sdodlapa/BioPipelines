/*
 * MACS2 Peak Calling Module
 * 
 * MACS2 (Model-based Analysis of ChIP-Seq) - Peak caller
 * Identifies regions of genomic enrichment in ChIP-seq, ATAC-seq
 * Uses existing chip-seq or atac-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * MACS2 Peak Calling - Standard mode
 */
process MACS2_CALLPEAK {
    tag "macs2_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/peaks", mode: 'copy'
    
    cpus params.macs2?.cpus ?: 2
    memory params.macs2?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(treatment)
    tuple val(sample_id), path(control)
    val genome_size  // e.g., "hs" (human), "mm" (mouse), or numeric
    val format       // "BAM", "BED", etc.
    
    output:
    tuple val(sample_id), path("${sample_id}_peaks.narrowPeak"), emit: narrow_peaks
    tuple val(sample_id), path("${sample_id}_peaks.xls"), emit: peaks_xls
    tuple val(sample_id), path("${sample_id}_summits.bed"), emit: summits
    tuple val(sample_id), path("${sample_id}_model.r"), emit: model
    
    script:
    def control_opt = control ? "--control ${control}" : ""
    
    """
    macs2 callpeak \\
        --treatment ${treatment} \\
        ${control_opt} \\
        --format ${format} \\
        --gsize ${genome_size} \\
        --name ${sample_id} \\
        --outdir . \\
        --qvalue 0.05
    """
}

/*
 * MACS2 Broad Peak Calling - For histone marks
 */
process MACS2_CALLPEAK_BROAD {
    tag "macs2_broad_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/peaks", mode: 'copy'
    
    cpus params.macs2?.cpus ?: 2
    memory params.macs2?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(treatment)
    tuple val(sample_id), path(control)
    val genome_size
    val format
    
    output:
    tuple val(sample_id), path("${sample_id}_peaks.broadPeak"), emit: broad_peaks
    tuple val(sample_id), path("${sample_id}_peaks.gappedPeak"), emit: gapped_peaks
    tuple val(sample_id), path("${sample_id}_peaks.xls"), emit: peaks_xls
    
    script:
    def control_opt = control ? "--control ${control}" : ""
    
    """
    macs2 callpeak \\
        --treatment ${treatment} \\
        ${control_opt} \\
        --format ${format} \\
        --gsize ${genome_size} \\
        --name ${sample_id} \\
        --outdir . \\
        --broad \\
        --broad-cutoff 0.1 \\
        --qvalue 0.05
    """
}

/*
 * MACS2 for ATAC-seq - Nucleosome-free regions
 */
process MACS2_ATAC {
    tag "macs2_atac_${sample_id}"
    container "${params.containers.atacseq}"
    
    publishDir "${params.outdir}/peaks", mode: 'copy'
    
    cpus params.macs2?.cpus ?: 2
    memory params.macs2?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    val genome_size
    
    output:
    tuple val(sample_id), path("${sample_id}_peaks.narrowPeak"), emit: narrow_peaks
    tuple val(sample_id), path("${sample_id}_summits.bed"), emit: summits
    tuple val(sample_id), path("${sample_id}_peaks.xls"), emit: peaks_xls
    
    script:
    """
    macs2 callpeak \\
        --treatment ${bam} \\
        --format BAMPE \\
        --gsize ${genome_size} \\
        --name ${sample_id} \\
        --outdir . \\
        --shift -100 \\
        --extsize 200 \\
        --nomodel \\
        --keep-dup all \\
        --qvalue 0.05
    """
}

/*
 * MACS2 with Custom Parameters
 */
process MACS2_CUSTOM {
    tag "macs2_custom_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/peaks", mode: 'copy'
    
    cpus params.macs2?.cpus ?: 2
    memory params.macs2?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(treatment)
    tuple val(sample_id), path(control)
    val genome_size
    val format
    val extra_params
    
    output:
    tuple val(sample_id), path("${sample_id}_peaks.*Peak"), emit: peaks
    tuple val(sample_id), path("${sample_id}_peaks.xls"), emit: peaks_xls
    path "${sample_id}_*", emit: all_outputs
    
    script:
    def control_opt = control ? "--control ${control}" : ""
    
    """
    macs2 callpeak \\
        --treatment ${treatment} \\
        ${control_opt} \\
        --format ${format} \\
        --gsize ${genome_size} \\
        --name ${sample_id} \\
        --outdir . \\
        ${extra_params}
    """
}

/*
 * MACS2 Peak Differential Analysis
 */
process MACS2_BDGDIFF {
    tag "macs2_diff_${comparison}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/differential_peaks", mode: 'copy'
    
    cpus params.macs2?.cpus ?: 2
    memory params.macs2?.memory ?: '16.GB'
    
    input:
    tuple val(comparison), path(treat1_bg), path(ctrl1_bg), path(treat2_bg), path(ctrl2_bg)
    val cutoff
    
    output:
    tuple val(comparison), path("${comparison}_c*.bed"), emit: diff_peaks
    
    script:
    """
    macs2 bdgdiff \\
        --t1 ${treat1_bg} \\
        --c1 ${ctrl1_bg} \\
        --t2 ${treat2_bg} \\
        --c2 ${ctrl2_bg} \\
        --outdir . \\
        --o-prefix ${comparison} \\
        --cutoff ${cutoff}
    """
}

/*
 * Workflow: Standard ChIP-seq peak calling
 */
workflow MACS2_CHIPSEQ_WORKFLOW {
    take:
    treatment_ch  // channel: [ val(sample_id), path(treatment_bam) ]
    control_ch    // channel: [ val(sample_id), path(control_bam) ]
    genome_size   // val: "hs", "mm", or numeric
    
    main:
    MACS2_CALLPEAK(
        treatment_ch,
        control_ch,
        genome_size,
        "BAM"
    )
    
    emit:
    peaks = MACS2_CALLPEAK.out.narrow_peaks
    summits = MACS2_CALLPEAK.out.summits
    xls = MACS2_CALLPEAK.out.peaks_xls
}

/*
 * Workflow: ATAC-seq peak calling
 */
workflow MACS2_ATAC_WORKFLOW {
    take:
    bam_ch        // channel: [ val(sample_id), path(bam) ]
    genome_size   // val: "hs", "mm", or numeric
    
    main:
    MACS2_ATAC(bam_ch, genome_size)
    
    emit:
    peaks = MACS2_ATAC.out.narrow_peaks
    summits = MACS2_ATAC.out.summits
    xls = MACS2_ATAC.out.peaks_xls
}
