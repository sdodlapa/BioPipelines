nextflow.enable.dsl=2

// Import modules
include { FASTQC } from './nextflow-modules/fastqc/main.nf'
include { BOWTIE2_ALIGN } from './nextflow-modules/bowtie2/main.nf'
include { SAMTOOLS_SORT; SAMTOOLS_INDEX; SAMTOOLS_FLAGSTAT; SAMTOOLS_STATS; SAMTOOLS_IDXSTATS; SAMTOOLS_VIEW; SAMTOOLS_MERGE } from './nextflow-modules/samtools/main.nf'
include { MACS2_CALLPEAK } from './nextflow-modules/macs2/main.nf'
include { MULTIQC; MULTIQC_CUSTOM } from './nextflow-modules/multiqc/main.nf'
include { PICARD_MARKDUPLICATES; PICARD_COLLECTMULTIPLEMETRICS; PICARD_COLLECTRNASEQMETRICS; PICARD_COLLECTWGSMETRICS; PICARD_SORTSAM; PICARD_ADDORREPLACEREADGROUPS } from './nextflow-modules/picard/main.nf'
include { DEEPTOOLS_BAMCOVERAGE; DEEPTOOLS_COMPUTEMATRIX; DEEPTOOLS_PLOTHEATMAP; DEEPTOOLS_PLOTPROFILE; DEEPTOOLS_MULTIBAMSUMMARY; DEEPTOOLS_PLOTCORRELATION; DEEPTOOLS_PLOTFINGERPRINT; DEEPTOOLS_PLOTPCA } from './nextflow-modules/deeptools/main.nf'

// Define parameters
params.reads = './data/*_{1,2}.fastq.gz'
params.genome = './genome/genome.fa'
params.outdir = './results'

// Input channel
Channel
    .fromFilePairs(params.reads, flat: true)
    .set { read_pairs }

// FASTQC
process fastqc {
    input:
    set val(sample_id), file(reads) from read_pairs

    output:
    file("*.html") into fastqc_reports

    script:
    """
    fastqc ${reads} -o ./
    """
}

// BOWTIE2 Alignment
process bowtie2_align {
    input:
    set val(sample_id), file(reads) from read_pairs

    output:
    file("${sample_id}.bam") into aligned_bams

    script:
    """
    bowtie2 -x ${params.genome} -1 ${reads[0]} -2 ${reads[1]} | samtools view -bS - > ${sample_id}.bam
    """
}

// SAMTOOLS Sort
process samtools_sort {
    input:
    file(bam) from aligned_bams

    output:
    file("${bam.baseName}.sorted.bam") into sorted_bams

    script:
    """
    samtools sort -o ${bam.baseName}.sorted.bam ${bam}
    """
}

// SAMTOOLS Index
process samtools_index {
    input:
    file(bam) from sorted_bams

    output:
    file("${bam}.bai") into indexed_bams

    script:
    """
    samtools index ${bam}
    """
}

// MACS2 Call Peaks
process macs2_callpeak {
    input:
    file(bam) from sorted_bams

    output:
    file("*.narrowPeak") into peak_files

    script:
    """
    macs2 callpeak -t ${bam} -f BAM -g hs -n ${bam.baseName} --outdir ./
    """
}

// MULTIQC
process multiqc {
    input:
    file(fastqc_reports) from fastqc_reports
    file(peak_files) from peak_files

    output:
    file("multiqc_report.html") into multiqc_report

    script:
    """
    multiqc . -o ./
    """
}

// Output aggregation
workflow {
    fastqc()
    bowtie2_align()
    samtools_sort()
    samtools_index()
    macs2_callpeak()
    multiqc()
}