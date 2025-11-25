/*
 * MultiQC Module
 * ==============
 * Aggregate results from bioinformatics analyses
 * 
 * Container: rna-seq, dna-seq, atac-seq
 */

process MULTIQC {
    tag "multiqc"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    path multiqc_files
    path multiqc_config
    
    output:
    path "*multiqc_report.html", emit: report
    path "*_data", emit: data
    path "*_plots", optional: true, emit: plots
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: ""
    def config = multiqc_config ? "--config $multiqc_config" : ""
    """
    multiqc \\
        --force \\
        $config \\
        $args \\
        .
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multiqc: \$(multiqc --version | sed 's/multiqc, version //')
    END_VERSIONS
    """
}

process MULTIQC_CUSTOM {
    tag "$prefix"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    path multiqc_files
    val prefix
    path multiqc_config
    path multiqc_logo
    
    output:
    path "${prefix}_multiqc_report.html", emit: report
    path "${prefix}_multiqc_data", emit: data
    path "${prefix}_multiqc_plots", optional: true, emit: plots
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: ""
    def config = multiqc_config ? "--config $multiqc_config" : ""
    def logo = multiqc_logo ? "--cl-config 'custom_logo: \"$multiqc_logo\"'" : ""
    """
    multiqc \\
        --force \\
        $config \\
        $logo \\
        --filename ${prefix}_multiqc_report \\
        $args \\
        .
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multiqc: \$(multiqc --version | sed 's/multiqc, version //')
    END_VERSIONS
    """
}
