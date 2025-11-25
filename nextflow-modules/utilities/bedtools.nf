/*
 * BEDTools Module
 * 
 * BEDTools - Suite of utilities for genomic interval operations
 * Intersect, merge, coverage, and other genomic arithmetic
 * Available in all pipeline containers
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * BEDTools Intersect
 */
process BEDTOOLS_INTERSECT {
    tag "intersect_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/intersect", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bed_a)
    path bed_b
    val mode  // "overlap", "unique_a", "unique_b", "count"
    
    output:
    tuple val(sample_id), path("${sample_id}.intersect.bed"), emit: bed
    
    script:
    def options = ""
    if (mode == "overlap") {
        options = "-wa -wb"
    } else if (mode == "unique_a") {
        options = "-v"
    } else if (mode == "count") {
        options = "-c"
    }
    
    """
    bedtools intersect \\
        -a ${bed_a} \\
        -b ${bed_b} \\
        ${options} \\
        > ${sample_id}.intersect.bed
    """
}

/*
 * BEDTools Merge
 */
process BEDTOOLS_MERGE {
    tag "merge_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/merged", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bed)
    
    output:
    tuple val(sample_id), path("${sample_id}.merged.bed"), emit: bed
    
    script:
    def distance = params.bedtools?.merge_distance ?: 0
    
    """
    sort -k1,1 -k2,2n ${bed} | \\
    bedtools merge -d ${distance} > ${sample_id}.merged.bed
    """
}

/*
 * BEDTools Coverage
 */
process BEDTOOLS_COVERAGE {
    tag "coverage_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/coverage", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam)
    path bed
    
    output:
    tuple val(sample_id), path("${sample_id}.coverage.bed"), emit: coverage
    
    script:
    """
    bedtools coverage \\
        -a ${bed} \\
        -b ${bam} \\
        > ${sample_id}.coverage.bed
    """
}

/*
 * BEDTools Genomecov - Genome-wide coverage
 */
process BEDTOOLS_GENOMECOV {
    tag "genomecov_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/genomecov", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam)
    val output_format  // "bedgraph" or "bed"
    
    output:
    tuple val(sample_id), path("${sample_id}.*.gz"), emit: coverage
    
    script:
    def format_opt = output_format == "bedgraph" ? "-bg" : ""
    def extension = output_format == "bedgraph" ? "bedgraph" : "bed"
    
    """
    bedtools genomecov \\
        -ibam ${bam} \\
        ${format_opt} \\
        | gzip > ${sample_id}.${extension}.gz
    """
}

/*
 * BEDTools Subtract
 */
process BEDTOOLS_SUBTRACT {
    tag "subtract_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/subtract", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bed_a)
    path bed_b
    
    output:
    tuple val(sample_id), path("${sample_id}.subtract.bed"), emit: bed
    
    script:
    """
    bedtools subtract \\
        -a ${bed_a} \\
        -b ${bed_b} \\
        > ${sample_id}.subtract.bed
    """
}

/*
 * BEDTools Closest
 */
process BEDTOOLS_CLOSEST {
    tag "closest_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/closest", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bed_a)
    path bed_b
    
    output:
    tuple val(sample_id), path("${sample_id}.closest.bed"), emit: bed
    
    script:
    """
    bedtools closest \\
        -a ${bed_a} \\
        -b ${bed_b} \\
        -d \\
        > ${sample_id}.closest.bed
    """
}

/*
 * BEDTools Sort
 */
process BEDTOOLS_SORT {
    tag "sort_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/sorted", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bed)
    
    output:
    tuple val(sample_id), path("${sample_id}.sorted.bed"), emit: bed
    
    script:
    """
    bedtools sort -i ${bed} > ${sample_id}.sorted.bed
    """
}

/*
 * BEDTools BAM to BED
 */
process BEDTOOLS_BAMTOBED {
    tag "bamtobed_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/bed", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.bed"), emit: bed
    
    script:
    """
    bedtools bamtobed -i ${bam} > ${sample_id}.bed
    """
}

/*
 * BEDTools MultiCov - Coverage across multiple BAMs
 */
process BEDTOOLS_MULTICOV {
    tag "multicov"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/bedtools/multicov", mode: 'copy'
    
    input:
    path bed
    path bams
    
    output:
    path "multicov.txt", emit: coverage
    
    script:
    def bam_list = bams.collect { it.toString() }.join(' ')
    
    """
    bedtools multicov \\
        -bams ${bam_list} \\
        -bed ${bed} \\
        > multicov.txt
    """
}

/*
 * Workflow: Standard BEDTools peak analysis
 */
workflow BEDTOOLS_PEAK_WORKFLOW {
    take:
    peaks_ch   // channel: [ val(sample_id), path(peaks_bed) ]
    
    main:
    // Sort peaks
    BEDTOOLS_SORT(peaks_ch)
    
    // Merge overlapping peaks
    BEDTOOLS_MERGE(BEDTOOLS_SORT.out.bed)
    
    emit:
    sorted = BEDTOOLS_SORT.out.bed
    merged = BEDTOOLS_MERGE.out.bed
}
