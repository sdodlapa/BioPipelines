/*
 * Picard Module
 * =============
 * BAM/SAM manipulation and metrics collection
 * 
 * Container: rna-seq, dna-seq
 */

process PICARD_MARKDUPLICATES {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.markdup.bam"), emit: bam
    tuple val(meta), path("*.markdup.bai"), emit: bai
    tuple val(meta), path("*.metrics.txt"), emit: metrics
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "ASSUME_SORTED=true REMOVE_DUPLICATES=false"
    def mem = task.memory ? "-Xmx${task.memory.toGiga()}g" : "-Xmx8g"
    """
    picard $mem MarkDuplicates \\
        INPUT=$bam \\
        OUTPUT=${prefix}.markdup.bam \\
        METRICS_FILE=${prefix}.metrics.txt \\
        CREATE_INDEX=true \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        picard: \$(picard MarkDuplicates --version 2>&1 | grep -o 'Version.*' | cut -d' ' -f2 || echo "unknown")
    END_VERSIONS
    """
}

process PICARD_COLLECTMULTIPLEMETRICS {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam), path(bai)
    path fasta
    path fai
    
    output:
    tuple val(meta), path("*_metrics"), emit: metrics
    tuple val(meta), path("*.pdf"), optional: true, emit: pdf
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def mem = task.memory ? "-Xmx${task.memory.toGiga()}g" : "-Xmx8g"
    """
    picard $mem CollectMultipleMetrics \\
        INPUT=$bam \\
        OUTPUT=${prefix} \\
        REFERENCE_SEQUENCE=$fasta \\
        PROGRAM=CollectAlignmentSummaryMetrics \\
        PROGRAM=CollectInsertSizeMetrics \\
        PROGRAM=QualityScoreDistribution \\
        PROGRAM=MeanQualityByCycle \\
        PROGRAM=CollectBaseDistributionByCycle \\
        PROGRAM=CollectGcBiasMetrics
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        picard: \$(picard CollectMultipleMetrics --version 2>&1 | grep -o 'Version.*' | cut -d' ' -f2 || echo "unknown")
    END_VERSIONS
    """
}

process PICARD_COLLECTRNASEQMETRICS {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam), path(bai)
    path refflat
    path rrna_intervals
    
    output:
    tuple val(meta), path("*.rna_metrics"), emit: metrics
    tuple val(meta), path("*.pdf"), optional: true, emit: pdf
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def mem = task.memory ? "-Xmx${task.memory.toGiga()}g" : "-Xmx8g"
    def strandedness = meta.strandedness == 'forward' ? 'FIRST_READ_TRANSCRIPTION_STRAND' : 
                       meta.strandedness == 'reverse' ? 'SECOND_READ_TRANSCRIPTION_STRAND' : 'NONE'
    def rrna = rrna_intervals ? "RIBOSOMAL_INTERVALS=$rrna_intervals" : ""
    """
    picard $mem CollectRnaSeqMetrics \\
        INPUT=$bam \\
        OUTPUT=${prefix}.rna_metrics \\
        REF_FLAT=$refflat \\
        STRAND_SPECIFICITY=$strandedness \\
        $rrna \\
        CHART_OUTPUT=${prefix}.coverage.pdf
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        picard: \$(picard CollectRnaSeqMetrics --version 2>&1 | grep -o 'Version.*' | cut -d' ' -f2 || echo "unknown")
    END_VERSIONS
    """
}

process PICARD_COLLECTWGSMETRICS {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(bam), path(bai)
    path fasta
    path fai
    path intervals
    
    output:
    tuple val(meta), path("*.wgs_metrics"), emit: metrics
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def mem = task.memory ? "-Xmx${task.memory.toGiga()}g" : "-Xmx8g"
    def interval_arg = intervals ? "INTERVALS=$intervals" : ""
    """
    picard $mem CollectWgsMetrics \\
        INPUT=$bam \\
        OUTPUT=${prefix}.wgs_metrics \\
        REFERENCE_SEQUENCE=$fasta \\
        $interval_arg
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        picard: \$(picard CollectWgsMetrics --version 2>&1 | grep -o 'Version.*' | cut -d' ' -f2 || echo "unknown")
    END_VERSIONS
    """
}

process PICARD_SORTSAM {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.sorted.bam"), emit: bam
    tuple val(meta), path("*.sorted.bai"), emit: bai
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def mem = task.memory ? "-Xmx${task.memory.toGiga()}g" : "-Xmx8g"
    def sort_order = task.ext.sort_order ?: "coordinate"
    """
    picard $mem SortSam \\
        INPUT=$bam \\
        OUTPUT=${prefix}.sorted.bam \\
        SORT_ORDER=$sort_order \\
        CREATE_INDEX=true
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        picard: \$(picard SortSam --version 2>&1 | grep -o 'Version.*' | cut -d' ' -f2 || echo "unknown")
    END_VERSIONS
    """
}

process PICARD_ADDORREPLACEREADGROUPS {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.rg.bam"), emit: bam
    tuple val(meta), path("*.rg.bai"), emit: bai
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def mem = task.memory ? "-Xmx${task.memory.toGiga()}g" : "-Xmx8g"
    """
    picard $mem AddOrReplaceReadGroups \\
        INPUT=$bam \\
        OUTPUT=${prefix}.rg.bam \\
        RGID=${meta.id} \\
        RGLB=${meta.library ?: meta.id} \\
        RGPL=${meta.platform ?: 'ILLUMINA'} \\
        RGPU=${meta.flowcell ?: 'unknown'} \\
        RGSM=${meta.sample ?: meta.id} \\
        CREATE_INDEX=true
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        picard: \$(picard AddOrReplaceReadGroups --version 2>&1 | grep -o 'Version.*' | cut -d' ' -f2 || echo "unknown")
    END_VERSIONS
    """
}
