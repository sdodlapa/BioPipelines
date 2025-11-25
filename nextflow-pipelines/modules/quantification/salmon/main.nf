/*
 * Salmon Module
 * =============
 * Fast transcript quantification from RNA-seq data
 * 
 * Container: rna-seq
 */

process SALMON_INDEX {
    tag "$fasta"
    label 'process_high'
    
    container "${params.containers.rnaseq}"
    
    input:
    path fasta
    path genome_fasta
    
    output:
    path "salmon_index", emit: index
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: ""
    def decoys = genome_fasta ? "-d <(grep '^>' $genome_fasta | cut -d ' ' -f 1 | sed 's/^>//') --genome $genome_fasta" : ""
    """
    salmon index \\
        -t $fasta \\
        $decoys \\
        -i salmon_index \\
        --threads $task.cpus \\
        $args
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        salmon: \$(salmon --version | sed 's/salmon //')
    END_VERSIONS
    """
}

process SALMON_QUANT {
    tag "$meta.id"
    label 'process_high'
    
    container "${params.containers.rnaseq}"
    
    input:
    tuple val(meta), path(reads)
    path index
    path gtf
    path transcript_fasta
    
    output:
    tuple val(meta), path("${meta.id}"), emit: results
    tuple val(meta), path("${meta.id}/quant.sf"), emit: quant
    tuple val(meta), path("${meta.id}/aux_info/meta_info.json"), emit: json
    path "versions.yml", emit: versions
    
    script:
    def args = task.ext.args ?: ""
    def prefix = task.ext.prefix ?: "${meta.id}"
    def lib_type = meta.strandedness == 'forward' ? 'SF' : meta.strandedness == 'reverse' ? 'SR' : 'A'
    
    if (meta.single_end) {
        """
        salmon quant \\
            --index $index \\
            --libType $lib_type \\
            -r $reads \\
            --threads $task.cpus \\
            --validateMappings \\
            -o ${prefix} \\
            $args
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            salmon: \$(salmon --version | sed 's/salmon //')
        END_VERSIONS
        """
    } else {
        """
        salmon quant \\
            --index $index \\
            --libType $lib_type \\
            -1 ${reads[0]} \\
            -2 ${reads[1]} \\
            --threads $task.cpus \\
            --validateMappings \\
            -o ${prefix} \\
            $args
        
        cat <<-END_VERSIONS > versions.yml
        "${task.process}":
            salmon: \$(salmon --version | sed 's/salmon //')
        END_VERSIONS
        """
    }
}

process SALMON_TX2GENE {
    tag "$gtf"
    label 'process_low'
    
    container "${params.containers.rnaseq}"
    
    input:
    path gtf
    
    output:
    path "tx2gene.tsv", emit: tsv
    path "versions.yml", emit: versions
    
    script:
    """
    awk '\$3=="transcript" {
        match(\$0, /gene_id "([^"]+)"/, g)
        match(\$0, /transcript_id "([^"]+)"/, t)
        if (t[1] && g[1]) print t[1] "\\t" g[1]
    }' $gtf > tx2gene.tsv
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        awk: \$(awk --version | head -1)
    END_VERSIONS
    """
}

process SALMON_MERGE {
    tag "merge"
    label 'process_medium'
    
    container "${params.containers.rnaseq}"
    
    input:
    path quant_files
    path tx2gene
    
    output:
    path "salmon_merged_gene_counts.tsv", emit: gene_counts
    path "salmon_merged_gene_tpm.tsv", emit: gene_tpm
    path "salmon_merged_transcript_counts.tsv", emit: transcript_counts
    path "salmon_merged_transcript_tpm.tsv", emit: transcript_tpm
    path "versions.yml", emit: versions
    
    script:
    """
    salmon quantmerge \\
        --quants $quant_files \\
        --column numreads \\
        -o salmon_merged_transcript_counts.tsv
    
    salmon quantmerge \\
        --quants $quant_files \\
        --column tpm \\
        -o salmon_merged_transcript_tpm.tsv
    
    # Gene-level aggregation using tx2gene
    aggregate_to_gene.py \\
        salmon_merged_transcript_counts.tsv \\
        $tx2gene \\
        salmon_merged_gene_counts.tsv
    
    aggregate_to_gene.py \\
        salmon_merged_transcript_tpm.tsv \\
        $tx2gene \\
        salmon_merged_gene_tpm.tsv
    
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        salmon: \$(salmon --version | sed 's/salmon //')
    END_VERSIONS
    """
}
