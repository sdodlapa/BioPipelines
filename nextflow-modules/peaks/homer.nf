/*
 * HOMER Module
 * 
 * HOMER (Hypergeometric Optimization of Motif EnRichment)
 * Motif discovery and peak annotation for ChIP-seq/ATAC-seq
 * Uses existing chip-seq or atac-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * HOMER findMotifsGenome - Motif discovery
 */
process HOMER_FINDMOTIFSGENOME {
    tag "homer_motifs_${peak_file.baseName}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/motifs/homer", mode: 'copy'
    
    cpus params.homer?.cpus ?: 8
    memory params.homer?.memory ?: '16.GB'
    
    input:
    path peak_file
    path genome
    val size
    
    output:
    path "motif_output/*", emit: results
    path "motif_output/homerResults.html", emit: html
    path "motif_output/knownResults.txt", emit: known_motifs
    
    script:
    """
    findMotifsGenome.pl \\
        ${peak_file} \\
        ${genome} \\
        motif_output \\
        -size ${size} \\
        -p ${task.cpus}
    """
}

/*
 * HOMER annotatePeaks - Annotate peaks
 */
process HOMER_ANNOTATEPEAKS {
    tag "homer_annotate_${peak_file.baseName}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/peaks/homer", mode: 'copy'
    
    cpus params.homer?.cpus ?: 4
    memory params.homer?.memory ?: '8.GB'
    
    input:
    path peak_file
    path genome
    path gtf
    
    output:
    path "${peak_file.baseName}_annotated.txt", emit: annotated
    path "${peak_file.baseName}_stats.txt", emit: stats
    
    script:
    """
    annotatePeaks.pl \\
        ${peak_file} \\
        ${genome} \\
        -gtf ${gtf} \\
        -cpu ${task.cpus} \\
        > ${peak_file.baseName}_annotated.txt \\
        2> ${peak_file.baseName}_stats.txt
    """
}

/*
 * HOMER makeTagDirectory - Create tag directory
 */
process HOMER_MAKETAGDIRECTORY {
    tag "homer_tagdir_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/homer/tagdirs", mode: 'copy'
    
    cpus params.homer?.cpus ?: 4
    memory params.homer?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}_tagdir"), emit: tagdir
    
    script:
    """
    makeTagDirectory \\
        ${sample_id}_tagdir \\
        ${bam} \\
        -format sam
    """
}

/*
 * HOMER findPeaks - Call peaks
 */
process HOMER_FINDPEAKS {
    tag "homer_peaks_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/peaks/homer", mode: 'copy'
    
    cpus params.homer?.cpus ?: 4
    memory params.homer?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(tagdir)
    val style  // "factor", "histone", "groseq", "tss", "dnase", "super", "mC"
    
    output:
    tuple val(sample_id), path("${sample_id}_peaks.txt"), emit: peaks
    
    script:
    """
    findPeaks \\
        ${tagdir} \\
        -style ${style} \\
        -o ${sample_id}_peaks.txt
    """
}

/*
 * HOMER getDifferentialPeaks - Differential peak analysis
 */
process HOMER_GETDIFFERENTIALPEAKS {
    tag "homer_diff_peaks"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/peaks/homer/differential", mode: 'copy'
    
    cpus params.homer?.cpus ?: 4
    memory params.homer?.memory ?: '8.GB'
    
    input:
    path treatment_peaks
    path control_peaks
    
    output:
    path "differential_peaks.txt", emit: peaks
    
    script:
    """
    getDifferentialPeaks \\
        ${treatment_peaks} \\
        ${control_peaks} \\
        > differential_peaks.txt
    """
}

/*
 * HOMER makeUCSCfile - Create bigWig for visualization
 */
process HOMER_MAKEUCSCFILE {
    tag "homer_bigwig_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/visualization/homer", mode: 'copy'
    
    cpus params.homer?.cpus ?: 4
    memory params.homer?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(tagdir)
    
    output:
    tuple val(sample_id), path("${sample_id}.bedGraph"), emit: bedgraph
    
    script:
    """
    makeUCSCfile \\
        ${tagdir} \\
        -o ${sample_id}.bedGraph
    """
}

/*
 * Workflow: Complete HOMER peak analysis pipeline
 */
workflow HOMER_PIPELINE {
    take:
    bam_ch        // channel: [ val(sample_id), path(bam) ]
    genome        // path: genome name (e.g., hg38, mm10) or directory
    gtf           // path: annotation GTF
    peak_style    // val: "factor", "histone", etc.
    
    main:
    // Create tag directories
    HOMER_MAKETAGDIRECTORY(bam_ch)
    
    // Call peaks
    HOMER_FINDPEAKS(HOMER_MAKETAGDIRECTORY.out.tagdir, peak_style)
    
    // Annotate peaks
    HOMER_ANNOTATEPEAKS(
        HOMER_FINDPEAKS.out.peaks.map { it[1] },
        genome,
        gtf
    )
    
    // Discover motifs
    HOMER_FINDMOTIFSGENOME(
        HOMER_FINDPEAKS.out.peaks.map { it[1] },
        genome,
        200  // default motif search size
    )
    
    // Create visualization files
    HOMER_MAKEUCSCFILE(HOMER_MAKETAGDIRECTORY.out.tagdir)
    
    emit:
    peaks = HOMER_FINDPEAKS.out.peaks
    annotated = HOMER_ANNOTATEPEAKS.out.annotated
    motifs = HOMER_FINDMOTIFSGENOME.out.results
    bedgraph = HOMER_MAKEUCSCFILE.out.bedgraph
}
