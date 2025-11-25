/*
 * deepTools Module
 * ================
 * Tools for the analysis of high-throughput sequencing data
 * 
 * Container: atac-seq, chip-seq
 */

process DEEPTOOLS_BAMCOVERAGE {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bam), path(bai)
    
    output:
    tuple val(meta), path("*.bigWig"), emit: bigwig
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "--normalizeUsing RPKM"
    """
    bamCoverage \\
        --bam $bam \\
        --outFileName ${prefix}.bigWig \\
        --numberOfProcessors $task.cpus \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        deeptools: \$(bamCoverage --version | sed 's/bamCoverage //')
    END_VERSIONS
    """
}

process DEEPTOOLS_COMPUTEMATRIX {
    tag "computeMatrix"
    label 'process_high'
    
    container "${params.containers.atacseq}"
    
    input:
    path bigwigs
    path bed
    
    output:
    path "*.mat.gz", emit: matrix
    path "*.mat.tab", optional: true, emit: table
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: "reference-point --referencePoint TSS -b 3000 -a 3000"
    def prefix = task.ext.prefix ?: "deeptools"
    """
    computeMatrix $args \\
        -S $bigwigs \\
        -R $bed \\
        -o ${prefix}.mat.gz \\
        --outFileNameMatrix ${prefix}.mat.tab \\
        -p $task.cpus
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        deeptools: \$(computeMatrix --version | sed 's/computeMatrix //')
    END_VERSIONS
    """
}

process DEEPTOOLS_PLOTHEATMAP {
    tag "plotHeatmap"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    path matrix
    
    output:
    path "*.pdf", emit: pdf
    path "*.png", optional: true, emit: png
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: "--colorMap RdYlBu"
    def prefix = task.ext.prefix ?: "heatmap"
    """
    plotHeatmap \\
        -m $matrix \\
        -o ${prefix}.pdf \\
        --outFileNameMatrix ${prefix}.mat.txt \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        deeptools: \$(plotHeatmap --version | sed 's/plotHeatmap //')
    END_VERSIONS
    """
}

process DEEPTOOLS_PLOTPROFILE {
    tag "plotProfile"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    path matrix
    
    output:
    path "*.pdf", emit: pdf
    path "*.png", optional: true, emit: png
    path "*.tab", optional: true, emit: table
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: ""
    def prefix = task.ext.prefix ?: "profile"
    """
    plotProfile \\
        -m $matrix \\
        -o ${prefix}.pdf \\
        --outFileNameData ${prefix}.tab \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        deeptools: \$(plotProfile --version | sed 's/plotProfile //')
    END_VERSIONS
    """
}

process DEEPTOOLS_MULTIBAMSUMMARY {
    tag "multiBamSummary"
    label 'process_high'
    
    container "${params.containers.atacseq}"
    
    input:
    path bams
    path bais
    
    output:
    path "*.npz", emit: matrix
    path "*.tab", optional: true, emit: table
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: "bins"
    def prefix = task.ext.prefix ?: "multibam_summary"
    """
    multiBamSummary $args \\
        --bamfiles $bams \\
        -o ${prefix}.npz \\
        --outRawCounts ${prefix}.tab \\
        -p $task.cpus
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        deeptools: \$(multiBamSummary --version | sed 's/multiBamSummary //')
    END_VERSIONS
    """
}

process DEEPTOOLS_PLOTCORRELATION {
    tag "plotCorrelation"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    path matrix
    
    output:
    path "*.pdf", emit: pdf
    path "*.png", optional: true, emit: png
    path "*.tab", optional: true, emit: table
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: "--corMethod spearman --whatToPlot heatmap"
    def prefix = task.ext.prefix ?: "correlation"
    """
    plotCorrelation \\
        -in $matrix \\
        -o ${prefix}.pdf \\
        --outFileCorMatrix ${prefix}.tab \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        deeptools: \$(plotCorrelation --version | sed 's/plotCorrelation //')
    END_VERSIONS
    """
}

process DEEPTOOLS_PLOTFINGERPRINT {
    tag "plotFingerprint"
    label 'process_high'
    
    container "${params.containers.atacseq}"
    
    input:
    path bams
    path bais
    
    output:
    path "*.pdf", emit: pdf
    path "*.png", optional: true, emit: png
    path "*.tab", optional: true, emit: table
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: ""
    def prefix = task.ext.prefix ?: "fingerprint"
    """
    plotFingerprint \\
        --bamfiles $bams \\
        -o ${prefix}.pdf \\
        --outRawCounts ${prefix}.tab \\
        -p $task.cpus \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        deeptools: \$(plotFingerprint --version | sed 's/plotFingerprint //')
    END_VERSIONS
    """
}

process DEEPTOOLS_PLOTPCA {
    tag "plotPCA"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    path matrix
    
    output:
    path "*.pdf", emit: pdf
    path "*.png", optional: true, emit: png
    path "*.tab", optional: true, emit: table
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: ""
    def prefix = task.ext.prefix ?: "pca"
    """
    plotPCA \\
        -in $matrix \\
        -o ${prefix}.pdf \\
        --outFileNameData ${prefix}.tab \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        deeptools: \$(plotPCA --version | sed 's/plotPCA //')
    END_VERSIONS
    """
}
