/*
 * Samtools Module
 * 
 * Samtools - Suite of programs for interacting with SAM/BAM files
 * Sorting, indexing, filtering, and conversion utilities
 * Available in all pipeline containers
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Samtools Sort
 */
process SAMTOOLS_SORT {
    tag "sort_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/sorted_bams", mode: 'copy'
    
    cpus params.samtools?.cpus ?: 4
    memory params.samtools?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.sorted.bam"), emit: bam
    
    script:
    """
    samtools sort \\
        -@ ${task.cpus} \\
        -m ${task.memory.toGiga() / task.cpus}G \\
        -o ${sample_id}.sorted.bam \\
        ${bam}
    """
}

/*
 * Samtools Index
 */
process SAMTOOLS_INDEX {
    tag "index_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/sorted_bams", mode: 'copy'
    
    cpus params.samtools?.cpus ?: 2
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${bam}.bai"), emit: bai
    tuple val(sample_id), path(bam), path("${bam}.bai"), emit: bam_bai
    
    script:
    """
    samtools index -@ ${task.cpus} ${bam}
    """
}

/*
 * Samtools Flagstat
 */
process SAMTOOLS_FLAGSTAT {
    tag "flagstat_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/flagstat", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.flagstat.txt"), emit: flagstat
    
    script:
    """
    samtools flagstat ${bam} > ${sample_id}.flagstat.txt
    """
}

/*
 * Samtools Stats
 */
process SAMTOOLS_STATS {
    tag "stats_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/stats", mode: 'copy'
    
    cpus params.samtools?.cpus ?: 2
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.stats.txt"), emit: stats
    
    script:
    """
    samtools stats -@ ${task.cpus} ${bam} > ${sample_id}.stats.txt
    """
}

/*
 * Samtools Idxstats
 */
process SAMTOOLS_IDXSTATS {
    tag "idxstats_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/idxstats", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam)
    tuple val(sample_id), path(bai)
    
    output:
    tuple val(sample_id), path("${sample_id}.idxstats.txt"), emit: idxstats
    
    script:
    """
    samtools idxstats ${bam} > ${sample_id}.idxstats.txt
    """
}

/*
 * Samtools View - Filter BAM
 */
process SAMTOOLS_VIEW {
    tag "view_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/filtered_bams", mode: 'copy'
    
    cpus params.samtools?.cpus ?: 4
    memory params.samtools?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    val filter_flags  // e.g., "-F 4 -q 20" (remove unmapped, min quality 20)
    
    output:
    tuple val(sample_id), path("${sample_id}.filtered.bam"), emit: bam
    
    script:
    """
    samtools view \\
        -@ ${task.cpus} \\
        -b \\
        ${filter_flags} \\
        -o ${sample_id}.filtered.bam \\
        ${bam}
    """
}

/*
 * Samtools Merge
 */
process SAMTOOLS_MERGE {
    tag "merge_${merged_name}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/merged_bams", mode: 'copy'
    
    cpus params.samtools?.cpus ?: 4
    memory params.samtools?.memory ?: '16.GB'
    
    input:
    val merged_name
    path bams
    
    output:
    path "${merged_name}.merged.bam", emit: bam
    
    script:
    """
    samtools merge \\
        -@ ${task.cpus} \\
        ${merged_name}.merged.bam \\
        ${bams}
    """
}

/*
 * Samtools Coverage
 */
process SAMTOOLS_COVERAGE {
    tag "coverage_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/coverage", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.coverage.txt"), emit: coverage
    
    script:
    """
    samtools coverage ${bam} > ${sample_id}.coverage.txt
    """
}

/*
 * Samtools Depth
 */
process SAMTOOLS_DEPTH {
    tag "depth_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/depth", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.depth.txt.gz"), emit: depth
    
    script:
    """
    samtools depth ${bam} | gzip > ${sample_id}.depth.txt.gz
    """
}

/*
 * Workflow: Standard BAM processing pipeline
 */
workflow SAMTOOLS_PIPELINE {
    take:
    bam_ch  // channel: [ val(sample_id), path(bam) ]
    
    main:
    // Sort BAMs
    SAMTOOLS_SORT(bam_ch)
    
    // Index sorted BAMs
    SAMTOOLS_INDEX(SAMTOOLS_SORT.out.bam)
    
    // Generate QC stats
    SAMTOOLS_FLAGSTAT(SAMTOOLS_SORT.out.bam)
    SAMTOOLS_STATS(SAMTOOLS_SORT.out.bam)
    
    emit:
    bam = SAMTOOLS_SORT.out.bam
    bai = SAMTOOLS_INDEX.out.bai
    bam_bai = SAMTOOLS_INDEX.out.bam_bai
    flagstat = SAMTOOLS_FLAGSTAT.out.flagstat
    stats = SAMTOOLS_STATS.out.stats
}
