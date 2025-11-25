nextflow.enable.dsl=2

// Import modules
include { FASTQC } from './nextflow-modules/fastqc/main.nf'
include { STAR_ALIGN } from './nextflow-modules/star/main.nf'
include { SAMTOOLS_SORT; SAMTOOLS_INDEX; SAMTOOLS_FLAGSTAT; SAMTOOLS_STATS; SAMTOOLS_IDXSTATS; SAMTOOLS_VIEW; SAMTOOLS_MERGE } from './nextflow-modules/samtools/main.nf'
include { MULTIQC; MULTIQC_CUSTOM } from './nextflow-modules/multiqc/main.nf'
include { PICARD_MARKDUPLICATES; PICARD_COLLECTMULTIPLEMETRICS; PICARD_COLLECTRNASEQMETRICS; PICARD_COLLECTWGSMETRICS; PICARD_SORTSAM; PICARD_ADDORREPLACEREADGROUPS } from './nextflow-modules/picard/main.nf'
include { FEATURECOUNTS } from './nextflow-modules/featurecounts/main.nf'
include { SALMON_INDEX; SALMON_QUANT; SALMON_TX2GENE; SALMON_MERGE } from './nextflow-modules/salmon/main.nf'

// Define parameters
params.reads = 'data/*.fastq'
params.genome = 'path/to/genome'
params.gtf = 'path/to/annotations.gtf'
params.outdir = 'results'

// Create input channel
Channel
    .fromPath(params.reads)
    .set { reads_ch }

// Quality control with FastQC
reads_ch
    | FASTQC

// Align reads with STAR
reads_ch
    | STAR_ALIGN(genome: params.genome, gtf: params.gtf)
    | SAMTOOLS_SORT
    | SAMTOOLS_INDEX
    | SAMTOOLS_FLAGSTAT
    | SAMTOOLS_STATS
    | SAMTOOLS_IDXSTATS
    | PICARD_MARKDUPLICATES
    | PICARD_COLLECTRNASEQMETRICS(gtf: params.gtf)
    | FEATURECOUNTS(gtf: params.gtf)

// Run MultiQC for aggregated report
reads_ch
    | MULTIQC

// Output aggregation
reads_ch
    | view { file -> file.copyTo(params.outdir) }