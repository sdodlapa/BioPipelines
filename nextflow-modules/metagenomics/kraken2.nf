/*
 * Kraken2 Module
 * 
 * Kraken2 - Taxonomic classification of metagenomic sequences
 * Ultra-fast and accurate taxonomic assignment
 * Uses existing metagenomics container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Kraken2 classify
 */
process KRAKEN2_CLASSIFY {
    tag "kraken2_${sample_id}"
    container "${params.containers.metagenomics}"
    
    publishDir "${params.outdir}/metagenomics/kraken2", mode: 'copy'
    
    cpus params.kraken2?.cpus ?: 16
    memory params.kraken2?.memory ?: '64.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path kraken2_db
    
    output:
    tuple val(sample_id), path("${sample_id}.kraken2"), emit: output
    tuple val(sample_id), path("${sample_id}.report"), emit: report
    path "${sample_id}.classified_*.fastq.gz", emit: classified_reads, optional: true
    path "${sample_id}.unclassified_*.fastq.gz", emit: unclassified_reads, optional: true
    
    script:
    def save_reads = params.kraken2?.save_reads ? "--classified-out ${sample_id}.classified#.fastq --unclassified-out ${sample_id}.unclassified#.fastq" : ""
    def confidence = params.kraken2?.confidence ?: 0.0
    
    if (reads instanceof List) {
        """
        kraken2 \\
            --db ${kraken2_db} \\
            --threads ${task.cpus} \\
            --paired \\
            --confidence ${confidence} \\
            --report ${sample_id}.report \\
            ${save_reads} \\
            --gzip-compressed \\
            ${reads[0]} ${reads[1]} \\
            > ${sample_id}.kraken2
        
        if [ -f ${sample_id}.classified_1.fastq ]; then
            gzip ${sample_id}.classified_*.fastq
            gzip ${sample_id}.unclassified_*.fastq
        fi
        """
    } else {
        """
        kraken2 \\
            --db ${kraken2_db} \\
            --threads ${task.cpus} \\
            --confidence ${confidence} \\
            --report ${sample_id}.report \\
            ${save_reads} \\
            --gzip-compressed \\
            ${reads} \\
            > ${sample_id}.kraken2
        
        if [ -f ${sample_id}.classified.fastq ]; then
            gzip ${sample_id}.classified.fastq
            gzip ${sample_id}.unclassified.fastq
        fi
        """
    }
}

/*
 * Bracken - Bayesian reestimation of abundance
 */
process BRACKEN {
    tag "bracken_${sample_id}"
    container "${params.containers.metagenomics}"
    
    publishDir "${params.outdir}/metagenomics/bracken", mode: 'copy'
    
    memory params.bracken?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(kraken_report)
    path kraken2_db
    val read_length
    val taxonomic_level
    
    output:
    tuple val(sample_id), path("${sample_id}.bracken"), emit: bracken_output
    tuple val(sample_id), path("${sample_id}.bracken_report"), emit: report
    
    script:
    def threshold = params.bracken?.threshold ?: 10
    
    """
    bracken \\
        -d ${kraken2_db} \\
        -i ${kraken_report} \\
        -o ${sample_id}.bracken \\
        -w ${sample_id}.bracken_report \\
        -r ${read_length} \\
        -l ${taxonomic_level} \\
        -t ${threshold}
    """
}

/*
 * Workflow: Kraken2 taxonomic classification
 */
workflow KRAKEN2_PIPELINE {
    take:
    reads_ch      // channel: [ val(sample_id), path(reads) ]
    kraken2_db    // path: Kraken2 database
    read_length   // val: read length for Bracken
    tax_level     // val: taxonomic level (S, G, F, etc.)
    
    main:
    KRAKEN2_CLASSIFY(reads_ch, kraken2_db)
    BRACKEN(KRAKEN2_CLASSIFY.out.report, kraken2_db, read_length, tax_level)
    
    emit:
    kraken_output = KRAKEN2_CLASSIFY.out.output
    kraken_report = KRAKEN2_CLASSIFY.out.report
    bracken_output = BRACKEN.out.bracken_output
    bracken_report = BRACKEN.out.report
}
