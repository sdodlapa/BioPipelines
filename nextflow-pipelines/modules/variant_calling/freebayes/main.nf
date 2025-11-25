/*
 * FreeBayes Module
 * ================
 * Bayesian haplotype-based genetic polymorphism discovery
 * 
 * Container: dna-seq
 */

process FREEBAYES {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(bam), path(bai)
    path fasta
    path fai
    path targets
    path samples
    path populations
    path cnv
    
    output:
    tuple val(meta), path("*.vcf.gz"), emit: vcf
    tuple val(meta), path("*.vcf.gz.tbi"), emit: tbi
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def targets_arg = targets ? "--targets $targets" : ""
    def samples_arg = samples ? "--samples $samples" : ""
    def populations_arg = populations ? "--populations $populations" : ""
    def cnv_arg = cnv ? "--cnv-map $cnv" : ""
    """
    freebayes \\
        -f $fasta \\
        $targets_arg \\
        $samples_arg \\
        $populations_arg \\
        $cnv_arg \\
        $args \\
        $bam \\
        | bgzip -c > ${prefix}.vcf.gz
    
    tabix -p vcf ${prefix}.vcf.gz
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        freebayes: \$(freebayes --version | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}

process FREEBAYES_PARALLEL {
    tag "$meta.id"
    label 'process_high'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(bam), path(bai)
    path fasta
    path fai
    path regions
    
    output:
    tuple val(meta), path("*.vcf.gz"), emit: vcf
    tuple val(meta), path("*.vcf.gz.tbi"), emit: tbi
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    """
    freebayes-parallel \\
        $regions \\
        $task.cpus \\
        -f $fasta \\
        $args \\
        $bam \\
        | bgzip -c > ${prefix}.vcf.gz
    
    tabix -p vcf ${prefix}.vcf.gz
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        freebayes: \$(freebayes --version | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}

process FREEBAYES_SOMATIC {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.dnaseq}"
    
    input:
    tuple val(meta), path(tumor_bam), path(tumor_bai), path(normal_bam), path(normal_bai)
    path fasta
    path fai
    path targets
    
    output:
    tuple val(meta), path("*.vcf.gz"), emit: vcf
    tuple val(meta), path("*.vcf.gz.tbi"), emit: tbi
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "--pooled-continuous --min-alternate-fraction 0.03"
    def targets_arg = targets ? "--targets $targets" : ""
    """
    freebayes \\
        -f $fasta \\
        $targets_arg \\
        $args \\
        $tumor_bam $normal_bam \\
        | bgzip -c > ${prefix}.somatic.vcf.gz
    
    tabix -p vcf ${prefix}.somatic.vcf.gz
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        freebayes: \$(freebayes --version | head -1 | sed 's/.*v//')
    END_VERSIONS
    """
}
