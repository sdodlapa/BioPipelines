/*
 * FreeBayes Module
 * 
 * FreeBayes - Haplotype-based variant detector
 * Bayesian genetic variant detector for DNA sequencing
 * Uses existing dna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * FreeBayes - Call variants
 */
process FREEBAYES_CALL {
    tag "freebayes_${sample_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/freebayes", mode: 'copy'
    
    cpus params.freebayes?.cpus ?: 4
    memory params.freebayes?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    path reference
    path reference_index
    path regions
    
    output:
    tuple val(sample_id), path("${sample_id}.vcf.gz"), path("${sample_id}.vcf.gz.tbi"), emit: vcf
    
    script:
    def regions_opt = regions ? "-t ${regions}" : ""
    def ploidy = params.freebayes?.ploidy ?: 2
    def min_alt_fraction = params.freebayes?.min_alt_fraction ?: 0.2
    
    """
    freebayes \\
        -f ${reference} \\
        ${regions_opt} \\
        --ploidy ${ploidy} \\
        --min-alternate-fraction ${min_alt_fraction} \\
        ${bam} | \\
    bgzip -c > ${sample_id}.vcf.gz
    
    tabix -p vcf ${sample_id}.vcf.gz
    """
}

/*
 * FreeBayes - Call variants in parallel by regions
 */
process FREEBAYES_PARALLEL {
    tag "freebayes_parallel"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/freebayes", mode: 'copy'
    
    cpus params.freebayes?.cpus ?: 16
    memory params.freebayes?.memory ?: '32.GB'
    
    input:
    path bam_files
    path bai_files
    path reference
    path reference_index
    path regions
    
    output:
    path "cohort.vcf.gz", emit: vcf
    path "cohort.vcf.gz.tbi", emit: vcf_index
    
    script:
    def bam_list = bam_files.collect { it.toString() }.join(' ')
    def ploidy = params.freebayes?.ploidy ?: 2
    
    """
    # Create BAM list file
    echo "${bam_list}" | tr ' ' '\\n' > bam_list.txt
    
    # Run freebayes-parallel
    freebayes-parallel \\
        <(fasta_generate_regions.py ${reference_index} 100000) \\
        ${task.cpus} \\
        -f ${reference} \\
        --ploidy ${ploidy} \\
        -L bam_list.txt | \\
    bgzip -c > cohort.vcf.gz
    
    tabix -p vcf cohort.vcf.gz
    """
}

/*
 * FreeBayes - Somatic variant calling
 */
process FREEBAYES_SOMATIC {
    tag "freebayes_somatic_${tumor_id}"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/variants/freebayes/somatic", mode: 'copy'
    
    cpus params.freebayes?.cpus ?: 4
    memory params.freebayes?.memory ?: '16.GB'
    
    input:
    tuple val(tumor_id), path(tumor_bam), path(tumor_bai)
    tuple val(normal_id), path(normal_bam), path(normal_bai)
    path reference
    path reference_index
    
    output:
    tuple val(tumor_id), path("${tumor_id}_somatic.vcf.gz"), path("${tumor_id}_somatic.vcf.gz.tbi"), emit: vcf
    
    script:
    """
    freebayes \\
        -f ${reference} \\
        --pooled-continuous \\
        --pooled-discrete \\
        --genotype-qualities \\
        --report-genotype-likelihood-max \\
        --allele-balance-priors-off \\
        --min-alternate-fraction 0.05 \\
        --min-alternate-count 2 \\
        ${tumor_bam} ${normal_bam} | \\
    bgzip -c > ${tumor_id}_somatic.vcf.gz
    
    tabix -p vcf ${tumor_id}_somatic.vcf.gz
    """
}

/*
 * Workflow: FreeBayes variant calling pipeline
 */
workflow FREEBAYES_PIPELINE {
    take:
    bam_ch            // channel: [ val(sample_id), path(bam), path(bai) ]
    reference         // path: reference genome
    reference_index   // path: reference .fai
    regions           // path: optional BED file with regions
    
    main:
    FREEBAYES_CALL(bam_ch, reference, reference_index, regions)
    
    emit:
    vcf = FREEBAYES_CALL.out.vcf
}
