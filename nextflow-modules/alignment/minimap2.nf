/*
 * minimap2 Module
 * 
 * minimap2 - Sequence alignment for long reads (PacBio, Nanopore)
 * Fast and versatile aligner for long-read sequencing
 * Uses existing long-read container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * minimap2 Align - Align long reads
 */
process MINIMAP2_ALIGN {
    tag "minimap2_${sample_id}"
    container "${params.containers.longread}"
    
    publishDir "${params.outdir}/alignments/minimap2", mode: 'copy'
    
    cpus params.minimap2?.cpus ?: 16
    memory params.minimap2?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path reference
    val preset  // "map-pb", "map-ont", "splice", "asm5", "asm10", "asm20"
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), path("${sample_id}.bam.bai"), emit: bam
    
    script:
    """
    minimap2 \\
        -ax ${preset} \\
        -t ${task.cpus} \\
        ${reference} \\
        ${reads} | \\
    samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
    
    samtools index ${sample_id}.bam
    """
}

/*
 * minimap2 Index - Create index for reference
 */
process MINIMAP2_INDEX {
    tag "minimap2_index"
    container "${params.containers.longread}"
    
    publishDir "${params.outdir}/reference/minimap2", mode: 'copy'
    
    cpus params.minimap2?.cpus ?: 8
    memory params.minimap2?.memory ?: '32.GB'
    
    input:
    path reference
    val preset
    
    output:
    path "${reference.baseName}.mmi", emit: index
    
    script:
    """
    minimap2 \\
        -x ${preset} \\
        -t ${task.cpus} \\
        -d ${reference.baseName}.mmi \\
        ${reference}
    """
}

/*
 * minimap2 Align with Index
 */
process MINIMAP2_ALIGN_INDEX {
    tag "minimap2_idx_${sample_id}"
    container "${params.containers.longread}"
    
    publishDir "${params.outdir}/alignments/minimap2", mode: 'copy'
    
    cpus params.minimap2?.cpus ?: 16
    memory params.minimap2?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    val preset
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), path("${sample_id}.bam.bai"), emit: bam
    
    script:
    """
    minimap2 \\
        -ax ${preset} \\
        -t ${task.cpus} \\
        ${index} \\
        ${reads} | \\
    samtools sort -@ ${task.cpus} -o ${sample_id}.bam -
    
    samtools index ${sample_id}.bam
    """
}

/*
 * Workflow: minimap2 alignment pipeline
 */
workflow MINIMAP2_PIPELINE {
    take:
    reads_ch       // channel: [ val(sample_id), path(reads) ]
    reference      // path: reference genome
    preset         // val: minimap2 preset
    build_index    // val: true/false
    
    main:
    if (build_index) {
        MINIMAP2_INDEX(reference, preset)
        MINIMAP2_ALIGN_INDEX(reads_ch, MINIMAP2_INDEX.out.index, preset)
        bam_out = MINIMAP2_ALIGN_INDEX.out.bam
    } else {
        MINIMAP2_ALIGN(reads_ch, reference, preset)
        bam_out = MINIMAP2_ALIGN.out.bam
    }
    
    emit:
    bam = bam_out
}
