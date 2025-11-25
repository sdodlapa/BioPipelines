/*
 * BEDTools Module
 * ===============
 * Genome arithmetic: operations on genomic intervals
 * 
 * Container: atac-seq, chip-seq, dna-seq
 */

process BEDTOOLS_INTERSECT {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(intervals_a)
    path intervals_b
    
    output:
    tuple val(meta), path("*.intersect.bed"), emit: bed
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-wa"
    """
    bedtools intersect \\
        -a $intervals_a \\
        -b $intervals_b \\
        $args \\
        > ${prefix}.intersect.bed
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}

process BEDTOOLS_MERGE {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bed)
    
    output:
    tuple val(meta), path("*.merged.bed"), emit: bed
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    sort -k1,1 -k2,2n $bed | bedtools merge $args > ${prefix}.merged.bed
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}

process BEDTOOLS_SORT {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bed)
    path genome_file
    
    output:
    tuple val(meta), path("*.sorted.bed"), emit: bed
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def genome_arg = genome_file ? "-g $genome_file" : ""
    """
    bedtools sort $genome_arg -i $bed > ${prefix}.sorted.bed
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}

process BEDTOOLS_COMPLEMENT {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bed)
    path genome_file
    
    output:
    tuple val(meta), path("*.complement.bed"), emit: bed
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    bedtools complement -i $bed -g $genome_file > ${prefix}.complement.bed
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}

process BEDTOOLS_GENOMECOV {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.bedgraph"), emit: bedgraph
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-bg"
    """
    bedtools genomecov \\
        -ibam $bam \\
        $args \\
        > ${prefix}.bedgraph
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}

process BEDTOOLS_BAMTOBED {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bam)
    
    output:
    tuple val(meta), path("*.bed"), emit: bed
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    bedtools bamtobed \\
        -i $bam \\
        $args \\
        > ${prefix}.bed
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}

process BEDTOOLS_GETFASTA {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bed)
    path fasta
    path fai
    
    output:
    tuple val(meta), path("*.fasta"), emit: fasta
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    bedtools getfasta \\
        -fi $fasta \\
        -bed $bed \\
        $args \\
        -fo ${prefix}.fasta
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}

process BEDTOOLS_SUBTRACT {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(intervals_a)
    path intervals_b
    
    output:
    tuple val(meta), path("*.subtract.bed"), emit: bed
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    bedtools subtract \\
        -a $intervals_a \\
        -b $intervals_b \\
        $args \\
        > ${prefix}.subtract.bed
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}

process BEDTOOLS_SLOP {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.atacseq}"
    
    input:
    tuple val(meta), path(bed)
    path genome_file
    
    output:
    tuple val(meta), path("*.slop.bed"), emit: bed
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-b 100"
    """
    bedtools slop \\
        -i $bed \\
        -g $genome_file \\
        $args \\
        > ${prefix}.slop.bed
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bedtools: \$(bedtools --version | sed 's/bedtools v//')
    END_VERSIONS
    """
}
