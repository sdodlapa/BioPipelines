/*
 * deepTools Module
 * 
 * deepTools - Tools for exploring deep sequencing data
 * Normalization, visualization, and analysis of ChIP-seq, RNA-seq, ATAC-seq
 * Uses existing chip-seq or atac-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * deepTools bamCoverage - Generate coverage track
 */
process DEEPTOOLS_BAMCOVERAGE {
    tag "bamcoverage_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/deeptools/coverage", mode: 'copy'
    
    cpus params.deeptools?.cpus ?: 4
    memory params.deeptools?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam)
    tuple val(sample_id), path(bai)
    val normalization  // "RPKM", "CPM", "BPM", "RPGC", "None"
    
    output:
    tuple val(sample_id), path("${sample_id}.bw"), emit: bigwig
    
    script:
    def bin_size = params.deeptools?.bin_size ?: 50
    def norm = normalization ?: "RPKM"
    
    """
    bamCoverage \\
        --bam ${bam} \\
        --outFileName ${sample_id}.bw \\
        --outFileFormat bigwig \\
        --binSize ${bin_size} \\
        --normalizeUsing ${norm} \\
        --numberOfProcessors ${task.cpus}
    """
}

/*
 * deepTools computeMatrix - Prepare data for plotting
 */
process DEEPTOOLS_COMPUTEMATRIX {
    tag "computematrix_${name}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/deeptools/matrix", mode: 'copy'
    
    cpus params.deeptools?.cpus ?: 8
    memory params.deeptools?.memory ?: '32.GB'
    
    input:
    val name
    path bigwigs
    path bed
    val mode  // "scale-regions" or "reference-point"
    
    output:
    path "${name}.matrix.gz", emit: matrix
    path "${name}.matrix.tab", emit: tab
    
    script:
    def region_size = params.deeptools?.region_size ?: 3000
    def before = params.deeptools?.before ?: 3000
    def after = params.deeptools?.after ?: 3000
    
    if (mode == "scale-regions") {
        """
        computeMatrix scale-regions \\
            --scoreFileName ${bigwigs} \\
            --regionsFileName ${bed} \\
            --outFileName ${name}.matrix.gz \\
            --outFileNameMatrix ${name}.matrix.tab \\
            --regionBodyLength ${region_size} \\
            --beforeRegionStartLength ${before} \\
            --afterRegionStartLength ${after} \\
            --numberOfProcessors ${task.cpus}
        """
    } else {
        """
        computeMatrix reference-point \\
            --scoreFileName ${bigwigs} \\
            --regionsFileName ${bed} \\
            --outFileName ${name}.matrix.gz \\
            --outFileNameMatrix ${name}.matrix.tab \\
            --beforeRegionStartLength ${before} \\
            --afterRegionStartLength ${after} \\
            --referencePoint center \\
            --numberOfProcessors ${task.cpus}
        """
    }
}

/*
 * deepTools plotHeatmap - Generate heatmap
 */
process DEEPTOOLS_PLOTHEATMAP {
    tag "plotheatmap_${name}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/deeptools/plots", mode: 'copy'
    
    memory params.deeptools?.memory ?: '16.GB'
    
    input:
    val name
    path matrix
    
    output:
    path "${name}.heatmap.png", emit: heatmap
    path "${name}.heatmap.tab", emit: data
    
    script:
    """
    plotHeatmap \\
        --matrixFile ${matrix} \\
        --outFileName ${name}.heatmap.png \\
        --outFileNameMatrix ${name}.heatmap.tab \\
        --colorMap RdYlBu \\
        --whatToShow 'heatmap and colorbar'
    """
}

/*
 * deepTools plotProfile - Generate profile plot
 */
process DEEPTOOLS_PLOTPROFILE {
    tag "plotprofile_${name}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/deeptools/plots", mode: 'copy'
    
    memory params.deeptools?.memory ?: '16.GB'
    
    input:
    val name
    path matrix
    
    output:
    path "${name}.profile.png", emit: profile
    path "${name}.profile.tab", emit: data
    
    script:
    """
    plotProfile \\
        --matrixFile ${matrix} \\
        --outFileName ${name}.profile.png \\
        --outFileNameData ${name}.profile.tab \\
        --perGroup
    """
}

/*
 * deepTools multiBamSummary - Compare multiple BAMs
 */
process DEEPTOOLS_MULTIBAMSUMMARY {
    tag "multibamsummary"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/deeptools/correlation", mode: 'copy'
    
    cpus params.deeptools?.cpus ?: 8
    memory params.deeptools?.memory ?: '32.GB'
    
    input:
    path bams
    path bais
    
    output:
    path "multiBamSummary.npz", emit: npz
    path "multiBamSummary.tab", emit: tab
    
    script:
    def bin_size = params.deeptools?.bin_size ?: 10000
    def bam_list = bams.collect { it.toString() }.join(' ')
    
    """
    multiBamSummary bins \\
        --bamfiles ${bam_list} \\
        --outFileName multiBamSummary.npz \\
        --outRawCounts multiBamSummary.tab \\
        --binSize ${bin_size} \\
        --numberOfProcessors ${task.cpus}
    """
}

/*
 * deepTools plotCorrelation - Sample correlation
 */
process DEEPTOOLS_PLOTCORRELATION {
    tag "plotcorrelation"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/deeptools/correlation", mode: 'copy'
    
    memory params.deeptools?.memory ?: '8.GB'
    
    input:
    path npz
    val method  // "pearson" or "spearman"
    
    output:
    path "correlation_heatmap.png", emit: heatmap
    path "correlation_matrix.tab", emit: matrix
    
    script:
    """
    plotCorrelation \\
        --corData ${npz} \\
        --corMethod ${method} \\
        --whatToPlot heatmap \\
        --plotFile correlation_heatmap.png \\
        --outFileCorMatrix correlation_matrix.tab \\
        --plotNumbers
    """
}

/*
 * deepTools plotPCA - Principal component analysis
 */
process DEEPTOOLS_PLOTPCA {
    tag "plotpca"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/deeptools/pca", mode: 'copy'
    
    memory params.deeptools?.memory ?: '8.GB'
    
    input:
    path npz
    
    output:
    path "pca_plot.png", emit: plot
    path "pca_data.tab", emit: data
    
    script:
    """
    plotPCA \\
        --corData ${npz} \\
        --plotFile pca_plot.png \\
        --outFileNameData pca_data.tab
    """
}

/*
 * deepTools bamCompare - Compare two BAMs
 */
process DEEPTOOLS_BAMCOMPARE {
    tag "bamcompare_${sample_id}"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/deeptools/compare", mode: 'copy'
    
    cpus params.deeptools?.cpus ?: 4
    memory params.deeptools?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam_treatment), path(bam_control)
    val operation  // "log2", "ratio", "subtract", "add"
    
    output:
    tuple val(sample_id), path("${sample_id}.compare.bw"), emit: bigwig
    
    script:
    def bin_size = params.deeptools?.bin_size ?: 50
    
    """
    bamCompare \\
        --bamfile1 ${bam_treatment} \\
        --bamfile2 ${bam_control} \\
        --outFileName ${sample_id}.compare.bw \\
        --outFileFormat bigwig \\
        --binSize ${bin_size} \\
        --operation ${operation} \\
        --normalizeUsing RPKM \\
        --numberOfProcessors ${task.cpus}
    """
}

/*
 * Workflow: Standard deepTools visualization pipeline
 */
workflow DEEPTOOLS_VISUALIZATION {
    take:
    bam_ch    // channel: [ val(sample_id), path(bam), path(bai) ]
    bed       // path: regions of interest
    
    main:
    // Generate coverage tracks
    DEEPTOOLS_BAMCOVERAGE(
        bam_ch.map { it[0..1] },
        bam_ch.map { it[0], it[2] },
        "RPKM"
    )
    
    // Compute matrix for visualization
    DEEPTOOLS_COMPUTEMATRIX(
        "analysis",
        DEEPTOOLS_BAMCOVERAGE.out.bigwig.map { it[1] }.collect(),
        bed,
        "reference-point"
    )
    
    // Generate plots
    DEEPTOOLS_PLOTHEATMAP("analysis", DEEPTOOLS_COMPUTEMATRIX.out.matrix)
    DEEPTOOLS_PLOTPROFILE("analysis", DEEPTOOLS_COMPUTEMATRIX.out.matrix)
    
    emit:
    bigwigs = DEEPTOOLS_BAMCOVERAGE.out.bigwig
    matrix = DEEPTOOLS_COMPUTEMATRIX.out.matrix
    heatmap = DEEPTOOLS_PLOTHEATMAP.out.heatmap
    profile = DEEPTOOLS_PLOTPROFILE.out.profile
}
