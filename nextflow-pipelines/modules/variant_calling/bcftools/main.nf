/*
 * BCFtools Module
 * ===============
 * Variant calling and manipulation utilities
 * 
 * Container: dna-seq, rna-seq
 */

process BCFTOOLS_CALL {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(mpileup)
    
    output:
    tuple val(meta), path("*.vcf.gz"), emit: vcf
    tuple val(meta), path("*.vcf.gz.tbi"), emit: tbi
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-mv"
    """
    bcftools call \\
        $args \\
        -Oz \\
        -o ${prefix}.vcf.gz \\
        $mpileup
    
    bcftools index -t ${prefix}.vcf.gz
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/bcftools //')
    END_VERSIONS
    """
}

process BCFTOOLS_MPILEUP {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(bam), path(bai)
    path fasta
    path fai
    
    output:
    tuple val(meta), path("*.mpileup.gz"), emit: mpileup
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    bcftools mpileup \\
        -f $fasta \\
        $args \\
        -Oz \\
        -o ${prefix}.mpileup.gz \\
        $bam
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/bcftools //')
    END_VERSIONS
    """
}

process BCFTOOLS_FILTER {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(vcf)
    
    output:
    tuple val(meta), path("*.filtered.vcf.gz"), emit: vcf
    tuple val(meta), path("*.filtered.vcf.gz.tbi"), emit: tbi
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-s LowQual -e 'QUAL<20 || DP<10'"
    """
    bcftools filter \\
        $args \\
        -Oz \\
        -o ${prefix}.filtered.vcf.gz \\
        $vcf
    
    bcftools index -t ${prefix}.filtered.vcf.gz
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/bcftools //')
    END_VERSIONS
    """
}

process BCFTOOLS_STATS {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(vcf)
    
    output:
    tuple val(meta), path("*.stats.txt"), emit: stats
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    bcftools stats $vcf > ${prefix}.stats.txt
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/bcftools //')
    END_VERSIONS
    """
}

process BCFTOOLS_NORM {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(vcf)
    path fasta
    
    output:
    tuple val(meta), path("*.norm.vcf.gz"), emit: vcf
    tuple val(meta), path("*.norm.vcf.gz.tbi"), emit: tbi
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    bcftools norm \\
        -f $fasta \\
        -Oz \\
        -o ${prefix}.norm.vcf.gz \\
        $vcf
    
    bcftools index -t ${prefix}.norm.vcf.gz
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/bcftools //')
    END_VERSIONS
    """
}

process BCFTOOLS_MERGE {
    tag "merge"
    label 'process_medium'
    
    container "${params.containers.dnaseq}"
    
    input:
    path vcfs
    path tbis
    
    output:
    path "merged.vcf.gz", emit: vcf
    path "merged.vcf.gz.tbi", emit: tbi
    path "versions.yml", emit: versions
    
    script:
    """
    bcftools merge \\
        -Oz \\
        -o merged.vcf.gz \\
        $vcfs
    
    bcftools index -t merged.vcf.gz
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/bcftools //')
    END_VERSIONS
    """
}
