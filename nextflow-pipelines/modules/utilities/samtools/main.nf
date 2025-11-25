/*
 * Samtools Module
 * ===============
 * SAM/BAM file manipulation utilities
 * 
 * Container: rna-seq, dna-seq, chip-seq, atac-seq
 */

process SAMTOOLS_SORT {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.sorted.bam"), emit: bam
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    samtools sort \\
        -@ ${task.cpus} \\
        -o ${prefix}.sorted.bam \\
        $bam
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(samtools --version | head -1 | sed 's/samtools //')
    END_VERSIONS
    """
}

process SAMTOOLS_INDEX {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.bai"), emit: bai
    tuple val(meta), path(bam), path("*.bai"), emit: bam_bai
    path "versions.yml", emit: versions
    
    script:
    """
    samtools index -@ ${task.cpus} $bam
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(samtools --version | head -1 | sed 's/samtools //')
    END_VERSIONS
    """
}

process SAMTOOLS_FLAGSTAT {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.flagstat"), emit: flagstat
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    samtools flagstat -@ ${task.cpus} $bam > ${prefix}.flagstat
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(samtools --version | head -1 | sed 's/samtools //')
    END_VERSIONS
    """
}

process SAMTOOLS_STATS {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.stats"), emit: stats
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    samtools stats -@ ${task.cpus} $bam > ${prefix}.stats
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(samtools --version | head -1 | sed 's/samtools //')
    END_VERSIONS
    """
}

process SAMTOOLS_IDXSTATS {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bam), path(bai)
    
    output:
    tuple val(meta), path("*.idxstats"), emit: idxstats
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    samtools idxstats $bam > ${prefix}.idxstats
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(samtools --version | head -1 | sed 's/samtools //')
    END_VERSIONS
    """
}

process SAMTOOLS_VIEW {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(input)
    path fasta
    
    output:
    tuple val(meta), path("*.bam"), emit: bam, optional: true
    tuple val(meta), path("*.cram"), emit: cram, optional: true
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def reference = fasta ? "--reference ${fasta}" : ""
    def args = task.ext.args ?: ""
    """
    samtools view \\
        -@ ${task.cpus} \\
        $reference \\
        $args \\
        -o ${prefix}.bam \\
        $input
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(samtools --version | head -1 | sed 's/samtools //')
    END_VERSIONS
    """
}

process SAMTOOLS_MERGE {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(bams)
    
    output:
    tuple val(meta), path("*.merged.bam"), emit: bam
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    samtools merge \\
        -@ ${task.cpus} \\
        ${prefix}.merged.bam \\
        $bams
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(samtools --version | head -1 | sed 's/samtools //')
    END_VERSIONS
    """
}
