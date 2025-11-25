/*
 * Bismark Module
 * 
 * Bismark - Bisulfite-treated Read Mapper
 * Alignment and methylation calling for BS-seq data
 * Uses existing methylation container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Bismark genome preparation
 */
process BISMARK_GENOME_PREPARATION {
    tag "bismark_prep"
    container "${params.containers.methylation}"
    
    publishDir "${params.outdir}/reference/bismark", mode: 'copy'
    
    cpus params.bismark?.cpus ?: 8
    memory params.bismark?.memory ?: '32.GB'
    
    input:
    path genome_dir
    
    output:
    path "bismark_index/*", emit: index
    
    script:
    """
    mkdir -p bismark_index
    cp ${genome_dir}/* bismark_index/
    
    bismark_genome_preparation \\
        --parallel ${task.cpus} \\
        bismark_index
    """
}

/*
 * Bismark alignment
 */
process BISMARK_ALIGN {
    tag "bismark_${sample_id}"
    container "${params.containers.methylation}"
    
    publishDir "${params.outdir}/alignments/bismark", mode: 'copy'
    
    cpus params.bismark?.cpus ?: 8
    memory params.bismark?.memory ?: '32.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path genome_index
    
    output:
    tuple val(sample_id), path("${sample_id}_bismark_bt2.bam"), emit: bam
    path "${sample_id}_bismark_bt2_SE_report.txt", emit: report, optional: true
    path "${sample_id}_bismark_bt2_PE_report.txt", emit: report_pe, optional: true
    
    script:
    if (reads instanceof List) {
        """
        bismark \\
            --genome ${genome_index} \\
            --parallel ${task.cpus / 4} \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            -o . \\
            --basename ${sample_id}
        """
    } else {
        """
        bismark \\
            --genome ${genome_index} \\
            --parallel ${task.cpus / 4} \\
            ${reads} \\
            -o . \\
            --basename ${sample_id}
        """
    }
}

/*
 * Bismark methylation extraction
 */
process BISMARK_METHYLATION_EXTRACTOR {
    tag "bismark_meth_${sample_id}"
    container "${params.containers.methylation}"
    
    publishDir "${params.outdir}/methylation/bismark", mode: 'copy'
    
    cpus params.bismark?.cpus ?: 8
    memory params.bismark?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    path "${sample_id}*", emit: results
    path "CpG_context_${sample_id}*.txt", emit: cpg
    path "CHG_context_${sample_id}*.txt", emit: chg
    path "CHH_context_${sample_id}*.txt", emit: chh
    path "${sample_id}*.bedGraph", emit: bedgraph
    
    script:
    """
    bismark_methylation_extractor \\
        --parallel ${task.cpus} \\
        --gzip \\
        --bedGraph \\
        --cytosine_report \\
        ${bam}
    """
}

/*
 * Bismark deduplicate
 */
process BISMARK_DEDUPLICATE {
    tag "bismark_dedup_${sample_id}"
    container "${params.containers.methylation}"
    
    publishDir "${params.outdir}/alignments/bismark/deduplicated", mode: 'copy'
    
    memory params.bismark?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    val paired_end
    
    output:
    tuple val(sample_id), path("${sample_id}.deduplicated.bam"), emit: bam
    path "${sample_id}.deduplication_report.txt", emit: report
    
    script:
    def pe_opt = paired_end ? "-p" : "-s"
    
    """
    deduplicate_bismark \\
        ${pe_opt} \\
        --bam ${bam} \\
        --output_dir .
    
    mv *.deduplicated.bam ${sample_id}.deduplicated.bam
    mv *.deduplication_report.txt ${sample_id}.deduplication_report.txt
    """
}

/*
 * Workflow: Bismark methylation pipeline
 */
workflow BISMARK_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    genome_index  // path: bismark genome index
    paired_end    // val: true/false
    
    main:
    BISMARK_ALIGN(reads_ch, genome_index)
    BISMARK_DEDUPLICATE(BISMARK_ALIGN.out.bam, paired_end)
    BISMARK_METHYLATION_EXTRACTOR(BISMARK_DEDUPLICATE.out.bam)
    
    emit:
    bam = BISMARK_DEDUPLICATE.out.bam
    cpg = BISMARK_METHYLATION_EXTRACTOR.out.cpg
    bedgraph = BISMARK_METHYLATION_EXTRACTOR.out.bedgraph
}
