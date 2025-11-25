/*
 * Salmon Quantification Module
 * 
 * Salmon - Ultra-fast transcript quantification
 * Alignment-free quantification from FASTQ files
 * Uses existing rna-seq container
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * Salmon Index - Build transcriptome index
 */
process SALMON_INDEX {
    tag "salmon_index_${transcriptome.baseName}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/salmon", mode: 'copy'
    
    cpus params.salmon?.index_cpus ?: 8
    memory params.salmon?.index_memory ?: '16.GB'
    
    input:
    path transcriptome
    
    output:
    path "salmon_index", emit: index
    
    script:
    """
    salmon index \\
        -t ${transcriptome} \\
        -i salmon_index \\
        -p ${task.cpus}
    """
}

/*
 * Salmon Index with Decoys - Build index with decoy sequences
 */
process SALMON_INDEX_DECOYS {
    tag "salmon_decoy_${transcriptome.baseName}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/reference/salmon", mode: 'copy'
    
    cpus params.salmon?.index_cpus ?: 8
    memory params.salmon?.index_memory ?: '32.GB'
    
    input:
    path transcriptome
    path genome
    
    output:
    path "salmon_index", emit: index
    path "decoys.txt", emit: decoys
    
    script:
    """
    # Extract decoy names from genome
    grep "^>" ${genome} | cut -d " " -f 1 | sed 's/>//g' > decoys.txt
    
    # Combine transcriptome and genome
    cat ${transcriptome} ${genome} > gentrome.fa
    
    # Build index with decoys
    salmon index \\
        -t gentrome.fa \\
        -d decoys.txt \\
        -i salmon_index \\
        -p ${task.cpus}
    
    rm gentrome.fa
    """
}

/*
 * Salmon Quantification
 */
process SALMON_QUANT {
    tag "salmon_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/salmon", mode: 'copy'
    
    cpus params.salmon?.cpus ?: 8
    memory params.salmon?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    val lib_type  // "A" (auto), "ISR" (stranded), "ISF" (reverse stranded)
    
    output:
    tuple val(sample_id), path("${sample_id}"), emit: results
    tuple val(sample_id), path("${sample_id}/quant.sf"), emit: quant
    tuple val(sample_id), path("${sample_id}/quant.genes.sf"), emit: genes optional true
    path "${sample_id}/aux_info", emit: aux_info
    path "${sample_id}/logs", emit: logs
    
    script:
    def validate = params.salmon?.validate_mappings ? "--validateMappings" : ""
    def gene_map = params.salmon?.gene_map ? "-g ${params.salmon.gene_map}" : ""
    
    if (reads instanceof List) {
        // Paired-end
        """
        salmon quant \\
            -i ${index} \\
            -l ${lib_type} \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            -p ${task.cpus} \\
            ${validate} \\
            ${gene_map} \\
            -o ${sample_id}
        """
    } else {
        // Single-end
        """
        salmon quant \\
            -i ${index} \\
            -l ${lib_type} \\
            -r ${reads} \\
            -p ${task.cpus} \\
            ${validate} \\
            ${gene_map} \\
            -o ${sample_id}
        """
    }
}

/*
 * Salmon Alignment-based Mode - Quantify from BAM
 */
process SALMON_QUANT_BAM {
    tag "salmon_bam_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/salmon", mode: 'copy'
    
    cpus params.salmon?.cpus ?: 4
    memory params.salmon?.memory ?: '8.GB'
    
    input:
    tuple val(sample_id), path(bam)
    path transcriptome
    val lib_type
    
    output:
    tuple val(sample_id), path("${sample_id}"), emit: results
    tuple val(sample_id), path("${sample_id}/quant.sf"), emit: quant
    path "${sample_id}/logs", emit: logs
    
    script:
    """
    salmon quant \\
        -t ${transcriptome} \\
        -l ${lib_type} \\
        -a ${bam} \\
        -p ${task.cpus} \\
        -o ${sample_id}
    """
}

/*
 * Salmon with GC Bias Correction
 */
process SALMON_QUANT_GC {
    tag "salmon_gc_${sample_id}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/quantification/salmon", mode: 'copy'
    
    cpus params.salmon?.cpus ?: 8
    memory params.salmon?.memory ?: '16.GB'
    
    input:
    tuple val(sample_id), path(reads)
    path index
    path transcriptome
    val lib_type
    
    output:
    tuple val(sample_id), path("${sample_id}"), emit: results
    tuple val(sample_id), path("${sample_id}/quant.sf"), emit: quant
    path "${sample_id}/aux_info/fld.gz", emit: frag_length_dist
    path "${sample_id}/libParams", emit: lib_params
    
    script:
    if (reads instanceof List) {
        """
        salmon quant \\
            -i ${index} \\
            -l ${lib_type} \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            -p ${task.cpus} \\
            --gcBias \\
            --biasSpeedSamp 5 \\
            -o ${sample_id}
        """
    } else {
        """
        salmon quant \\
            -i ${index} \\
            -l ${lib_type} \\
            -r ${reads} \\
            -p ${task.cpus} \\
            --gcBias \\
            --biasSpeedSamp 5 \\
            -o ${sample_id}
        """
    }
}

/*
 * Workflow: Complete Salmon pipeline with indexing
 */
workflow SALMON_PIPELINE {
    take:
    reads_ch       // channel: [ val(sample_id), path(reads) ]
    transcriptome  // path: transcriptome fasta
    lib_type       // val: "A", "ISR", "ISF", etc.
    
    main:
    // Build index
    SALMON_INDEX(transcriptome)
    
    // Quantify reads
    SALMON_QUANT(reads_ch, SALMON_INDEX.out.index, lib_type)
    
    emit:
    results = SALMON_QUANT.out.results
    quant = SALMON_QUANT.out.quant
    logs = SALMON_QUANT.out.logs
}
