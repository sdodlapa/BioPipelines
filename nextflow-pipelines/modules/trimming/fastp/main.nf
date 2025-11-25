/*
 * fastp Module
 * ============
 * Ultra-fast all-in-one FASTQ preprocessor
 * 
 * Container: rna-seq, dna-seq
 */

process FASTP {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    path adapter_fasta
    val save_trimmed_fail
    val save_merged
    
    output:
    tuple val(meta), path("*.fastp.fastq.gz"), emit: reads
    tuple val(meta), path("*.json"), emit: json
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.log"), emit: log
    tuple val(meta), path("*.fail.fastq.gz"), optional: true, emit: reads_fail
    tuple val(meta), path("*.merged.fastq.gz"), optional: true, emit: reads_merged
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def adapter_list = adapter_fasta ? "--adapter_fasta $adapter_fasta" : ""
    def fail_fastq = save_trimmed_fail ? "--failed_out ${prefix}.fail.fastq.gz" : ""
    def merge_args = save_merged && !meta.single_end ? "--merge --merged_out ${prefix}.merged.fastq.gz" : ""
    
    if (meta.single_end) {
        """
        fastp \\
            --in1 $reads \\
            --out1 ${prefix}.fastp.fastq.gz \\
            --thread $task.cpus \\
            --json ${prefix}.fastp.json \\
            --html ${prefix}.fastp.html \\
            $adapter_list \\
            $fail_fastq \\
            $args \\
            2> ${prefix}.fastp.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            fastp: \$(fastp --version 2>&1 | sed -e 's/fastp //')
        END_VERSIONS
        """
    } else {
        """
        fastp \\
            --in1 ${reads[0]} \\
            --in2 ${reads[1]} \\
            --out1 ${prefix}_1.fastp.fastq.gz \\
            --out2 ${prefix}_2.fastp.fastq.gz \\
            --thread $task.cpus \\
            --json ${prefix}.fastp.json \\
            --html ${prefix}.fastp.html \\
            --detect_adapter_for_pe \\
            $adapter_list \\
            $fail_fastq \\
            $merge_args \\
            $args \\
            2> ${prefix}.fastp.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            fastp: \$(fastp --version 2>&1 | sed -e 's/fastp //')
        END_VERSIONS
        """
    }
}

process FASTP_UMI {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    val umi_loc
    val umi_len
    
    output:
    tuple val(meta), path("*.fastp.fastq.gz"), emit: reads
    tuple val(meta), path("*.json"), emit: json
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.log"), emit: log
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    def umi_location = umi_loc ?: "read1"
    def umi_length = umi_len ?: 12
    
    if (meta.single_end) {
        """
        fastp \\
            --in1 $reads \\
            --out1 ${prefix}.fastp.fastq.gz \\
            --thread $task.cpus \\
            --json ${prefix}.fastp.json \\
            --html ${prefix}.fastp.html \\
            --umi \\
            --umi_loc $umi_location \\
            --umi_len $umi_length \\
            $args \\
            2> ${prefix}.fastp.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            fastp: \$(fastp --version 2>&1 | sed -e 's/fastp //')
        END_VERSIONS
        """
    } else {
        """
        fastp \\
            --in1 ${reads[0]} \\
            --in2 ${reads[1]} \\
            --out1 ${prefix}_1.fastp.fastq.gz \\
            --out2 ${prefix}_2.fastp.fastq.gz \\
            --thread $task.cpus \\
            --json ${prefix}.fastp.json \\
            --html ${prefix}.fastp.html \\
            --detect_adapter_for_pe \\
            --umi \\
            --umi_loc $umi_location \\
            --umi_len $umi_length \\
            $args \\
            2> ${prefix}.fastp.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            fastp: \$(fastp --version 2>&1 | sed -e 's/fastp //')
        END_VERSIONS
        """
    }
}

process FASTP_DEDUP {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.dedup.fastq.gz"), emit: reads
    tuple val(meta), path("*.json"), emit: json
    tuple val(meta), path("*.html"), emit: html
    tuple val(meta), path("*.log"), emit: log
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: ""
    
    if (meta.single_end) {
        """
        fastp \\
            --in1 $reads \\
            --out1 ${prefix}.dedup.fastq.gz \\
            --thread $task.cpus \\
            --json ${prefix}.fastp.json \\
            --html ${prefix}.fastp.html \\
            --dedup \\
            $args \\
            2> ${prefix}.fastp.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            fastp: \$(fastp --version 2>&1 | sed -e 's/fastp //')
        END_VERSIONS
        """
    } else {
        """
        fastp \\
            --in1 ${reads[0]} \\
            --in2 ${reads[1]} \\
            --out1 ${prefix}_1.dedup.fastq.gz \\
            --out2 ${prefix}_2.dedup.fastq.gz \\
            --thread $task.cpus \\
            --json ${prefix}.fastp.json \\
            --html ${prefix}.fastp.html \\
            --detect_adapter_for_pe \\
            --dedup \\
            $args \\
            2> ${prefix}.fastp.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            fastp: \$(fastp --version 2>&1 | sed -e 's/fastp //')
        END_VERSIONS
        """
    }
}
