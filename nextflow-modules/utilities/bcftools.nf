/*
 * bcftools Module
 * 
 * bcftools - Utilities for variant calling and manipulating VCF/BCF files
 * Comprehensive VCF manipulation, filtering, and statistics
 * Uses existing dna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * bcftools mpileup + call - Variant calling
 */
process BCFTOOLS_CALL {
    tag "bcftools_call_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools", mode: 'copy'
    
    cpus params.bcftools?.cpus ?: 4
    memory params.bcftools?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    path reference_index
    
    output:
    tuple val(sample_id), path("${sample_id}.vcf.gz"), path("${sample_id}.vcf.gz.csi"), emit: vcf
    
    script:
    """
    bcftools mpileup \\
        -f ${reference} \\
        -Ou \\
        ${bam} | \\
    bcftools call \\
        --threads ${task.cpus} \\
        -mv \\
        -Oz \\
        -o ${sample_id}.vcf.gz
    
    bcftools index ${sample_id}.vcf.gz
    """
}

/*
 * bcftools filter - Filter variants
 */
process BCFTOOLS_FILTER {
    tag "bcftools_filter"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools/filtered", mode: 'copy'
    
    cpus params.bcftools?.cpus ?: 2
    memory params.bcftools?.memory ?: '4.GB'
    
    input:
    path vcf
    val filter_expression
    
    output:
    path "filtered.vcf.gz", emit: vcf
    path "filtered.vcf.gz.csi", emit: vcf_index
    
    script:
    """
    bcftools filter \\
        --threads ${task.cpus} \\
        -i '${filter_expression}' \\
        -Oz \\
        -o filtered.vcf.gz \\
        ${vcf}
    
    bcftools index filtered.vcf.gz
    """
}

/*
 * bcftools merge - Merge multiple VCF files
 */
process BCFTOOLS_MERGE {
    tag "bcftools_merge"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools/merged", mode: 'copy'
    
    cpus params.bcftools?.cpus ?: 4
    memory params.bcftools?.memory ?: '8.GB'
    
    input:
    path vcf_files
    
    output:
    path "merged.vcf.gz", emit: vcf
    path "merged.vcf.gz.csi", emit: vcf_index
    
    script:
    def vcf_list = vcf_files.collect { it.toString() }.join(' ')
    
    """
    bcftools merge \\
        --threads ${task.cpus} \\
        -Oz \\
        -o merged.vcf.gz \\
        ${vcf_list}
    
    bcftools index merged.vcf.gz
    """
}

/*
 * bcftools norm - Normalize variants
 */
process BCFTOOLS_NORM {
    tag "bcftools_norm"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools/normalized", mode: 'copy'
    
    cpus params.bcftools?.cpus ?: 2
    memory params.bcftools?.memory ?: '4.GB'
    
    input:
    path vcf
    path reference
    
    output:
    path "normalized.vcf.gz", emit: vcf
    path "normalized.vcf.gz.csi", emit: vcf_index
    
    script:
    """
    bcftools norm \\
        --threads ${task.cpus} \\
        -f ${reference} \\
        -m -any \\
        -Oz \\
        -o normalized.vcf.gz \\
        ${vcf}
    
    bcftools index normalized.vcf.gz
    """
}

/*
 * bcftools stats - Generate VCF statistics
 */
process BCFTOOLS_STATS {
    tag "bcftools_stats_${vcf.baseName}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools/stats", mode: 'copy'
    
    cpus params.bcftools?.cpus ?: 2
    memory params.bcftools?.memory ?: '4.GB'
    
    input:
    path vcf
    path reference
    
    output:
    path "${vcf.baseName}.stats.txt", emit: stats
    
    script:
    def ref_opt = reference ? "-F ${reference}" : ""
    
    """
    bcftools stats \\
        ${ref_opt} \\
        ${vcf} > ${vcf.baseName}.stats.txt
    """
}

/*
 * bcftools query - Extract information from VCF
 */
process BCFTOOLS_QUERY {
    tag "bcftools_query"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools/query", mode: 'copy'
    
    input:
    path vcf
    val format_string
    
    output:
    path "query_results.txt", emit: results
    
    script:
    """
    bcftools query \\
        -f '${format_string}' \\
        ${vcf} > query_results.txt
    """
}

/*
 * bcftools view - View/subset VCF files
 */
process BCFTOOLS_VIEW {
    tag "bcftools_view"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools/subset", mode: 'copy'
    
    cpus params.bcftools?.cpus ?: 2
    memory params.bcftools?.memory ?: '4.GB'
    
    input:
    path vcf
    val regions
    val samples
    
    output:
    path "subset.vcf.gz", emit: vcf
    path "subset.vcf.gz.csi", emit: vcf_index
    
    script:
    def region_opt = regions ? "-r ${regions}" : ""
    def sample_opt = samples ? "-s ${samples}" : ""
    
    """
    bcftools view \\
        --threads ${task.cpus} \\
        ${region_opt} \\
        ${sample_opt} \\
        -Oz \\
        -o subset.vcf.gz \\
        ${vcf}
    
    bcftools index subset.vcf.gz
    """
}

/*
 * bcftools annotate - Annotate VCF
 */
process BCFTOOLS_ANNOTATE {
    tag "bcftools_annotate"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools/annotated", mode: 'copy'
    
    cpus params.bcftools?.cpus ?: 2
    memory params.bcftools?.memory ?: '4.GB'
    
    input:
    path vcf
    path annotation_file
    val columns
    
    output:
    path "annotated.vcf.gz", emit: vcf
    path "annotated.vcf.gz.csi", emit: vcf_index
    
    script:
    def columns_opt = columns ? "-c ${columns}" : ""
    
    """
    bcftools annotate \\
        --threads ${task.cpus} \\
        -a ${annotation_file} \\
        ${columns_opt} \\
        -Oz \\
        -o annotated.vcf.gz \\
        ${vcf}
    
    bcftools index annotated.vcf.gz
    """
}

/*
 * bcftools consensus - Create consensus sequence
 */
process BCFTOOLS_CONSENSUS {
    tag "bcftools_consensus"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/consensus", mode: 'copy'
    
    input:
    path vcf
    path reference
    val sample_name
    
    output:
    path "${sample_name}_consensus.fa", emit: consensus
    
    script:
    def sample_opt = sample_name ? "-s ${sample_name}" : ""
    
    """
    bcftools index ${vcf}
    
    bcftools consensus \\
        ${sample_opt} \\
        -f ${reference} \\
        ${vcf} > ${sample_name}_consensus.fa
    """
}

/*
 * bcftools isec - Create intersections and complements
 */
process BCFTOOLS_ISEC {
    tag "bcftools_isec"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/bcftools/isec", mode: 'copy'
    
    cpus params.bcftools?.cpus ?: 2
    memory params.bcftools?.memory ?: '4.GB'
    
    input:
    path vcf_files
    
    output:
    path "isec_output/*", emit: results
    
    script:
    def vcf_list = vcf_files.collect { it.toString() }.join(' ')
    
    """
    mkdir -p isec_output
    
    bcftools isec \\
        --threads ${task.cpus} \\
        -p isec_output \\
        ${vcf_list}
    """
}

/*
 * Workflow: Complete bcftools variant processing pipeline
 */
workflow BCFTOOLS_PIPELINE {
    take:
    bam_ch         // channel: [ val(sample_id), path(bam), path(bai) ]
    reference      // path: reference genome
    reference_index // path: reference .fai
    
    main:
    // Variant calling
    BCFTOOLS_CALL(bam_ch, reference, reference_index)
    
    // Normalize variants
    BCFTOOLS_NORM(
        BCFTOOLS_CALL.out.vcf.map { it[1] },
        reference
    )
    
    // Filter variants (example: quality >= 20, depth >= 10)
    BCFTOOLS_FILTER(
        BCFTOOLS_NORM.out.vcf,
        'QUAL>=20 && DP>=10'
    )
    
    // Generate statistics
    BCFTOOLS_STATS(
        BCFTOOLS_FILTER.out.vcf,
        reference
    )
    
    emit:
    raw_vcf = BCFTOOLS_CALL.out.vcf
    normalized_vcf = BCFTOOLS_NORM.out.vcf
    filtered_vcf = BCFTOOLS_FILTER.out.vcf
    stats = BCFTOOLS_STATS.out.stats
}
