/*
 * Cutadapt Module
 * ===============
 * Adapter trimming and read filtering
 * 
 * Container: rna-seq, atac-seq
 */

process CUTADAPT {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.trimmed.fastq.gz"), emit: reads
    tuple val(meta), path("*.cutadapt.log"), emit: log
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-q 20 --minimum-length 20"
    def adapter_r1 = task.ext.adapter_r1 ?: "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
    def adapter_r2 = task.ext.adapter_r2 ?: "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"
    
    if (meta.single_end) {
        """
        cutadapt \\
            -j $task.cpus \\
            -a $adapter_r1 \\
            $args \\
            -o ${prefix}.trimmed.fastq.gz \\
            $reads \\
            > ${prefix}.cutadapt.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            cutadapt: \$(cutadapt --version)
        END_VERSIONS
        """
    } else {
        """
        cutadapt \\
            -j $task.cpus \\
            -a $adapter_r1 \\
            -A $adapter_r2 \\
            $args \\
            -o ${prefix}_1.trimmed.fastq.gz \\
            -p ${prefix}_2.trimmed.fastq.gz \\
            ${reads[0]} ${reads[1]} \\
            > ${prefix}.cutadapt.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            cutadapt: \$(cutadapt --version)
        END_VERSIONS
        """
    }
}

process CUTADAPT_HARDTRIM {
    tag "$meta.id"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    val trim_5
    val trim_3
    
    output:
    tuple val(meta), path("*.hardtrim.fastq.gz"), emit: reads
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def cut_5 = trim_5 > 0 ? "-u $trim_5" : ""
    def cut_3 = trim_3 > 0 ? "-u -$trim_3" : ""
    
    if (meta.single_end) {
        """
        cutadapt \\
            -j $task.cpus \\
            $cut_5 $cut_3 \\
            -o ${prefix}.hardtrim.fastq.gz \\
            $reads
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            cutadapt: \$(cutadapt --version)
        END_VERSIONS
        """
    } else {
        """
        cutadapt \\
            -j $task.cpus \\
            $cut_5 $cut_3 \\
            -U $trim_5 -U -$trim_3 \\
            -o ${prefix}_1.hardtrim.fastq.gz \\
            -p ${prefix}_2.hardtrim.fastq.gz \\
            ${reads[0]} ${reads[1]}
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            cutadapt: \$(cutadapt --version)
        END_VERSIONS
        """
    }
}

process CUTADAPT_POLYATRIM {
    tag "$meta.id"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    
    output:
    tuple val(meta), path("*.polyatrim.fastq.gz"), emit: reads
    tuple val(meta), path("*.polyatrim.log"), emit: log
    path "versions.yml", emit: versions
    
    script:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def args = task.ext.args ?: "-q 20 --minimum-length 20"
    
    if (meta.single_end) {
        """
        cutadapt \\
            -j $task.cpus \\
            -a "A{20}" \\
            -a "T{20}" \\
            $args \\
            -o ${prefix}.polyatrim.fastq.gz \\
            $reads \\
            > ${prefix}.polyatrim.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            cutadapt: \$(cutadapt --version)
        END_VERSIONS
        """
    } else {
        """
        cutadapt \\
            -j $task.cpus \\
            -a "A{20}" -a "T{20}" \\
            -A "A{20}" -A "T{20}" \\
            $args \\
            -o ${prefix}_1.polyatrim.fastq.gz \\
            -p ${prefix}_2.polyatrim.fastq.gz \\
            ${reads[0]} ${reads[1]} \\
            > ${prefix}.polyatrim.log
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            cutadapt: \$(cutadapt --version)
        END_VERSIONS
        """
    }
}
