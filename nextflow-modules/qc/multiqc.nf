/*
 * MultiQC Module
 * 
 * MultiQC - Aggregate results from bioinformatics analyses
 * Creates unified report from multiple QC tools
 * Available in all pipeline containers
 */

// Enable DSL2
nextflow.enable.dsl = 2

/*
 * MultiQC - Aggregate QC reports
 */
process MULTIQC {
    tag "multiqc_${analysis_name}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/multiqc", mode: 'copy'
    
    memory params.multiqc?.memory ?: '8.GB'
    
    input:
    path qc_files
    val analysis_name
    
    output:
    path "${analysis_name}_multiqc_report.html", emit: html
    path "${analysis_name}_multiqc_data", emit: data
    
    script:
    """
    multiqc \\
        --filename ${analysis_name}_multiqc_report.html \\
        --force \\
        --interactive \\
        .
    """
}

/*
 * MultiQC with Custom Config
 */
process MULTIQC_CUSTOM {
    tag "multiqc_custom_${analysis_name}"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/multiqc", mode: 'copy'
    
    memory params.multiqc?.memory ?: '8.GB'
    
    input:
    path qc_files
    path config
    val analysis_name
    
    output:
    path "${analysis_name}_multiqc_report.html", emit: html
    path "${analysis_name}_multiqc_data", emit: data
    path "${analysis_name}_multiqc_plots", emit: plots optional true
    
    script:
    """
    multiqc \\
        --filename ${analysis_name}_multiqc_report.html \\
        --config ${config} \\
        --force \\
        --interactive \\
        --export \\
        .
    """
}

/*
 * MultiQC for RNA-seq Pipeline
 */
process MULTIQC_RNASEQ {
    tag "multiqc_rnaseq"
    container "${params.containers.rnaseq}"
    
    publishDir "${params.outdir}/qc/multiqc", mode: 'copy'
    
    memory params.multiqc?.memory ?: '8.GB'
    
    input:
    path fastqc_zip
    path star_logs
    path salmon_logs
    path featurecounts_summary
    
    output:
    path "rnaseq_multiqc_report.html", emit: html
    path "rnaseq_multiqc_data", emit: data
    
    script:
    """
    multiqc \\
        --filename rnaseq_multiqc_report.html \\
        --title "RNA-seq Pipeline QC Report" \\
        --comment "Quality control metrics from RNA-seq analysis" \\
        --force \\
        --interactive \\
        .
    """
}

/*
 * MultiQC for DNA-seq Pipeline
 */
process MULTIQC_DNASEQ {
    tag "multiqc_dnaseq"
    container "${params.containers.dnaseq}"
    
    publishDir "${params.outdir}/qc/multiqc", mode: 'copy'
    
    memory params.multiqc?.memory ?: '8.GB'
    
    input:
    path fastqc_zip
    path bwa_stats
    path gatk_metrics
    
    output:
    path "dnaseq_multiqc_report.html", emit: html
    path "dnaseq_multiqc_data", emit: data
    
    script:
    """
    multiqc \\
        --filename dnaseq_multiqc_report.html \\
        --title "DNA-seq Pipeline QC Report" \\
        --force \\
        --interactive \\
        .
    """
}

/*
 * MultiQC for ChIP-seq Pipeline
 */
process MULTIQC_CHIPSEQ {
    tag "multiqc_chipseq"
    container "${params.containers.chipseq}"
    
    publishDir "${params.outdir}/qc/multiqc", mode: 'copy'
    
    memory params.multiqc?.memory ?: '8.GB'
    
    input:
    path fastqc_zip
    path bowtie2_logs
    path macs2_logs
    path deeptools_logs
    
    output:
    path "chipseq_multiqc_report.html", emit: html
    path "chipseq_multiqc_data", emit: data
    
    script:
    """
    multiqc \\
        --filename chipseq_multiqc_report.html \\
        --title "ChIP-seq Pipeline QC Report" \\
        --force \\
        --interactive \\
        .
    """
}

/*
 * Workflow: Standard MultiQC aggregation
 */
workflow MULTIQC_WORKFLOW {
    take:
    qc_files_ch   // channel: path(qc_files)
    analysis_name // val: name for the analysis
    
    main:
    MULTIQC(qc_files_ch.collect(), analysis_name)
    
    emit:
    html = MULTIQC.out.html
    data = MULTIQC.out.data
}
