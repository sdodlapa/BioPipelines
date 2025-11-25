nextflow.enable.dsl=2

// Import the necessary modules
include { BWAMEM_ALIGN } from './nextflow-modules/bwamem/main.nf'

// Define parameters
params.reads = './data/*.fastq.gz'
params.genome = './genome/genome.fa'
params.outdir = './results'

// Create input channel for paired-end FASTQ files
Channel
    .fromFilePairs(params.reads, checkIfExists: true)
    .set { read_pairs }

// Create a channel for the reference genome
Channel
    .fromPath(params.genome)
    .set { genome_fasta }

// Align reads using BWA-MEM
process BWAMEM_ALIGN {
    input:
    tuple val(sample_id), path(reads) from read_pairs
    path genome from genome_fasta

    output:
    path "${sample_id}.bam" into aligned_bams

    script:
    """
    bwa mem -t 8 $genome ${reads[0]} ${reads[1]} | samtools view -Sb - > ${sample_id}.bam
    """
}

// Further processes would be added here for downstream analysis of Hi-C data

// Collect all BAM files into a single channel for further analysis or output
aligned_bams
    .collectFile(name: 'all_aligned_bams.tar.gz', compress: true)
    .set { final_output }

// Define the workflow
workflow {
    // Execute the alignment process
    BWAMEM_ALIGN()

    // Output the final aggregated results
    final_output.view { "Aggregated BAM files: $it" }
}