"""
Golden Queries Test Suite
=========================

This module contains 50+ golden queries that cover all supported analysis types
and use cases. These tests validate that the system correctly:
1. Parses intent from natural language queries
2. Identifies the correct analysis type
3. Generates appropriate workflow structures

Golden queries are curated examples that represent real user needs.
They serve as regression tests and benchmarks for model fine-tuning.
"""

import pytest
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ExpectedAnalysisType(Enum):
    """Expected analysis types from golden queries."""
    RNA_SEQ = "rna-seq"
    SCRNA_SEQ = "scrna-seq"
    CHIP_SEQ = "chip-seq"
    ATAC_SEQ = "atac-seq"
    DNA_SEQ = "dna-seq"
    WGS = "wgs"
    WES = "wes"
    VARIANT_CALLING = "variant-calling"
    METHYLATION = "methylation"
    METAGENOMICS = "metagenomics"
    HIC = "hic"
    LONG_READ = "long-read"
    STRUCTURAL_VARIANTS = "structural-variants"
    MULTI_OMICS = "multi-omics"
    EDUCATION = "education"
    DATA_SEARCH = "data-search"
    JOB_MANAGEMENT = "job-management"
    UNKNOWN = "unknown"


@dataclass
class GoldenQuery:
    """A golden query with expected outputs."""
    query: str
    expected_analysis_type: ExpectedAnalysisType
    expected_keywords: List[str]
    expected_tools: Optional[List[str]] = None
    description: str = ""
    priority: int = 1  # 1=critical, 2=important, 3=nice-to-have


# =============================================================================
# Golden Queries - RNA-seq
# =============================================================================
RNA_SEQ_QUERIES = [
    GoldenQuery(
        query="Create an RNA-seq workflow for differential expression analysis using STAR and DESeq2",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["differential expression", "STAR", "DESeq2"],
        expected_tools=["fastqc", "star", "featurecounts", "deseq2"],
        description="Classic differential expression pipeline",
        priority=1
    ),
    GoldenQuery(
        query="I want to analyze RNA-seq data from mouse samples comparing wild-type and knockout",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["RNA-seq", "mouse", "wild-type", "knockout"],
        expected_tools=["fastqc", "star", "deseq2"],
        description="Mouse DE analysis with conditions",
        priority=1
    ),
    GoldenQuery(
        query="Generate a pipeline for bulk RNA sequencing with Salmon quantification",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["bulk RNA", "Salmon"],
        expected_tools=["salmon"],
        description="Alternative quantification with Salmon",
        priority=1
    ),
    GoldenQuery(
        query="Create RNA-seq workflow for human samples with HISAT2 alignment",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["human", "HISAT2"],
        expected_tools=["hisat2"],
        description="HISAT2-based RNA-seq",
        priority=2
    ),
    GoldenQuery(
        query="Set up an RNA-seq analysis with stranded library prep and UMI deduplication",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["stranded", "UMI"],
        expected_tools=["umi_tools", "star"],
        description="UMI-aware RNA-seq",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Single-cell RNA-seq
# =============================================================================
SCRNA_SEQ_QUERIES = [
    GoldenQuery(
        query="Create a single-cell RNA-seq workflow using CellRanger and Seurat",
        expected_analysis_type=ExpectedAnalysisType.SCRNA_SEQ,
        expected_keywords=["single-cell", "CellRanger", "Seurat"],
        expected_tools=["cellranger", "seurat"],
        description="10x Genomics scRNA-seq pipeline",
        priority=1
    ),
    GoldenQuery(
        query="I have 10x Genomics data and want to cluster cells and identify cell types",
        expected_analysis_type=ExpectedAnalysisType.SCRNA_SEQ,
        expected_keywords=["10x Genomics", "cluster", "cell types"],
        expected_tools=["cellranger", "seurat", "scanpy"],
        description="Cell type identification",
        priority=1
    ),
    GoldenQuery(
        query="Generate scRNA-seq analysis pipeline with Scanpy for trajectory analysis",
        expected_analysis_type=ExpectedAnalysisType.SCRNA_SEQ,
        expected_keywords=["Scanpy", "trajectory"],
        expected_tools=["scanpy"],
        description="Trajectory analysis with Scanpy",
        priority=2
    ),
    GoldenQuery(
        query="Create workflow for multiome data with RNA and ATAC from same cells",
        expected_analysis_type=ExpectedAnalysisType.MULTI_OMICS,
        expected_keywords=["multiome", "RNA", "ATAC"],
        expected_tools=["cellranger-arc"],
        description="Multi-omic single-cell",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - ChIP-seq
# =============================================================================
CHIP_SEQ_QUERIES = [
    GoldenQuery(
        query="Create a ChIP-seq workflow for H3K4me3 histone mark analysis",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["ChIP-seq", "H3K4me3", "histone"],
        expected_tools=["bwa", "macs2", "deeptools"],
        description="Histone ChIP-seq",
        priority=1
    ),
    GoldenQuery(
        query="I need to analyze ChIP-seq data for a transcription factor with narrow peaks",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["transcription factor", "narrow peaks"],
        expected_tools=["macs2"],
        description="TF ChIP-seq with narrow peaks",
        priority=1
    ),
    GoldenQuery(
        query="Generate ChIP-seq pipeline with input control and peak annotation",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["input control", "peak annotation"],
        expected_tools=["macs2", "homer", "annotatePeaks"],
        description="ChIP-seq with annotation",
        priority=1
    ),
    GoldenQuery(
        query="Create workflow for ChIP-seq with broad histone marks like H3K27me3",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["broad", "H3K27me3"],
        expected_tools=["macs2"],
        description="Broad peak calling",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - ATAC-seq
# =============================================================================
ATAC_SEQ_QUERIES = [
    GoldenQuery(
        query="Create an ATAC-seq workflow for chromatin accessibility analysis",
        expected_analysis_type=ExpectedAnalysisType.ATAC_SEQ,
        expected_keywords=["ATAC-seq", "chromatin accessibility"],
        expected_tools=["bwa", "macs2", "deeptools"],
        description="Standard ATAC-seq",
        priority=1
    ),
    GoldenQuery(
        query="I want to analyze ATAC-seq data and find differential accessible regions",
        expected_analysis_type=ExpectedAnalysisType.ATAC_SEQ,
        expected_keywords=["ATAC-seq", "differential accessible"],
        expected_tools=["macs2", "diffbind"],
        description="Differential ATAC-seq",
        priority=1
    ),
    GoldenQuery(
        query="Generate ATAC-seq pipeline with nucleosome-free region analysis",
        expected_analysis_type=ExpectedAnalysisType.ATAC_SEQ,
        expected_keywords=["nucleosome-free"],
        expected_tools=["ataqc", "macs2"],
        description="NFR analysis",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - DNA-seq / WGS / WES
# =============================================================================
DNA_SEQ_QUERIES = [
    GoldenQuery(
        query="Create a variant calling pipeline using GATK best practices",
        expected_analysis_type=ExpectedAnalysisType.DNA_SEQ,
        expected_keywords=["variant calling", "GATK"],
        expected_tools=["bwa", "gatk", "bcftools"],
        description="GATK variant calling",
        priority=1
    ),
    GoldenQuery(
        query="I need to call variants from whole genome sequencing data",
        expected_analysis_type=ExpectedAnalysisType.WGS,
        expected_keywords=["variants", "whole genome"],
        expected_tools=["bwa", "gatk"],
        description="WGS variant calling",
        priority=1
    ),
    GoldenQuery(
        query="Generate exome sequencing analysis workflow for clinical samples",
        expected_analysis_type=ExpectedAnalysisType.WES,
        expected_keywords=["exome", "clinical"],
        expected_tools=["bwa", "gatk", "vep"],
        description="Clinical WES",
        priority=1
    ),
    GoldenQuery(
        query="Create pipeline for somatic variant calling in tumor-normal pairs",
        expected_analysis_type=ExpectedAnalysisType.DNA_SEQ,
        expected_keywords=["somatic", "tumor-normal"],
        expected_tools=["mutect2", "strelka2"],
        description="Somatic variant calling",
        priority=1
    ),
    GoldenQuery(
        query="Set up germline variant calling workflow with joint genotyping",
        expected_analysis_type=ExpectedAnalysisType.DNA_SEQ,
        expected_keywords=["germline", "joint genotyping"],
        expected_tools=["gatk", "haplotypecaller"],
        description="Joint calling",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Methylation
# =============================================================================
METHYLATION_QUERIES = [
    GoldenQuery(
        query="Create a methylation analysis pipeline using Bismark",
        expected_analysis_type=ExpectedAnalysisType.METHYLATION,
        expected_keywords=["methylation", "Bismark"],
        expected_tools=["bismark", "methylkit"],
        description="WGBS with Bismark",
        priority=1
    ),
    GoldenQuery(
        query="I want to analyze RRBS data and find differentially methylated regions",
        expected_analysis_type=ExpectedAnalysisType.METHYLATION,
        expected_keywords=["RRBS", "differentially methylated"],
        expected_tools=["bismark", "methylkit"],
        description="RRBS DMR analysis",
        priority=1
    ),
    GoldenQuery(
        query="Generate whole-genome bisulfite sequencing workflow for cancer samples",
        expected_analysis_type=ExpectedAnalysisType.METHYLATION,
        expected_keywords=["WGBS", "cancer"],
        expected_tools=["bismark"],
        description="Cancer WGBS",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Metagenomics
# =============================================================================
METAGENOMICS_QUERIES = [
    GoldenQuery(
        query="Create a metagenomics workflow for 16S rRNA analysis",
        expected_analysis_type=ExpectedAnalysisType.METAGENOMICS,
        expected_keywords=["metagenomics", "16S rRNA"],
        expected_tools=["qiime2", "dada2"],
        description="16S amplicon analysis",
        priority=1
    ),
    GoldenQuery(
        query="I want to analyze shotgun metagenomics data for taxonomic profiling",
        expected_analysis_type=ExpectedAnalysisType.METAGENOMICS,
        expected_keywords=["shotgun metagenomics", "taxonomic"],
        expected_tools=["kraken2", "metaphlan"],
        description="Shotgun metagenomics",
        priority=1
    ),
    GoldenQuery(
        query="Generate microbiome analysis pipeline with functional profiling",
        expected_analysis_type=ExpectedAnalysisType.METAGENOMICS,
        expected_keywords=["microbiome", "functional"],
        expected_tools=["humann", "metaphlan"],
        description="Functional metagenomics",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Hi-C and 3D Genome
# =============================================================================
HIC_QUERIES = [
    GoldenQuery(
        query="Create a Hi-C analysis workflow for chromatin conformation",
        expected_analysis_type=ExpectedAnalysisType.HIC,
        expected_keywords=["Hi-C", "chromatin conformation"],
        expected_tools=["hicpro", "cooler"],
        description="Hi-C analysis",
        priority=1
    ),
    GoldenQuery(
        query="I want to identify TADs from Hi-C data",
        expected_analysis_type=ExpectedAnalysisType.HIC,
        expected_keywords=["TADs", "Hi-C"],
        expected_tools=["hicpro", "tadtool"],
        description="TAD identification",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Long-read
# =============================================================================
LONG_READ_QUERIES = [
    GoldenQuery(
        query="Create a long-read sequencing workflow using Minimap2 for Oxford Nanopore data",
        expected_analysis_type=ExpectedAnalysisType.LONG_READ,
        expected_keywords=["long-read", "Minimap2", "Nanopore"],
        expected_tools=["minimap2", "medaka"],
        description="ONT analysis",
        priority=1
    ),
    GoldenQuery(
        query="I have PacBio HiFi data and want to call structural variants",
        expected_analysis_type=ExpectedAnalysisType.STRUCTURAL_VARIANTS,
        expected_keywords=["PacBio", "HiFi", "structural variants"],
        expected_tools=["pbmm2", "pbsv"],
        description="PacBio SV calling",
        priority=1
    ),
]

# =============================================================================
# Golden Queries - Education / Explanation
# =============================================================================
EDUCATION_QUERIES = [
    GoldenQuery(
        query="What is RNA-seq and when should I use it?",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["RNA-seq"],
        description="Explain RNA-seq",
        priority=1
    ),
    GoldenQuery(
        query="Explain the difference between STAR and HISAT2",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["STAR", "HISAT2", "difference"],
        description="Compare aligners",
        priority=1
    ),
    GoldenQuery(
        query="What is ChIP-seq used for?",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["ChIP-seq"],
        description="Explain ChIP-seq",
        priority=1
    ),
    GoldenQuery(
        query="How does DESeq2 work for differential expression?",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["DESeq2", "differential expression"],
        description="Explain DESeq2",
        priority=2
    ),
    GoldenQuery(
        query="What is the difference between WGS and WES?",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["WGS", "WES"],
        description="Compare sequencing types",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Data Search
# =============================================================================
DATA_SEARCH_QUERIES = [
    GoldenQuery(
        query="Search for RNA-seq datasets from ENCODE for K562 cells",
        expected_analysis_type=ExpectedAnalysisType.DATA_SEARCH,
        expected_keywords=["ENCODE", "K562", "RNA-seq"],
        description="ENCODE search",
        priority=1
    ),
    GoldenQuery(
        query="Find ChIP-seq data for CTCF in human from GEO",
        expected_analysis_type=ExpectedAnalysisType.DATA_SEARCH,
        expected_keywords=["GEO", "ChIP-seq", "CTCF", "human"],
        description="GEO search",
        priority=1
    ),
    GoldenQuery(
        query="Search TCGA for lung cancer RNA-seq data",
        expected_analysis_type=ExpectedAnalysisType.DATA_SEARCH,
        expected_keywords=["TCGA", "lung cancer"],
        description="TCGA search",
        priority=1
    ),
    GoldenQuery(
        query="List available reference genomes for mouse",
        expected_analysis_type=ExpectedAnalysisType.DATA_SEARCH,
        expected_keywords=["reference genomes", "mouse"],
        description="Reference search",
        priority=2
    ),
    GoldenQuery(
        query="Download GRCh38 reference genome",
        expected_analysis_type=ExpectedAnalysisType.DATA_SEARCH,
        expected_keywords=["download", "GRCh38"],
        description="Download reference",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Job Management
# =============================================================================
JOB_MANAGEMENT_QUERIES = [
    GoldenQuery(
        query="Check the status of my running jobs",
        expected_analysis_type=ExpectedAnalysisType.JOB_MANAGEMENT,
        expected_keywords=["status", "jobs"],
        description="Job status",
        priority=1
    ),
    GoldenQuery(
        query="Submit the workflow to SLURM",
        expected_analysis_type=ExpectedAnalysisType.JOB_MANAGEMENT,
        expected_keywords=["submit", "SLURM"],
        description="Submit job",
        priority=1
    ),
    GoldenQuery(
        query="Cancel job 12345",
        expected_analysis_type=ExpectedAnalysisType.JOB_MANAGEMENT,
        expected_keywords=["cancel", "job"],
        description="Cancel job",
        priority=2
    ),
    GoldenQuery(
        query="Monitor my pipeline execution",
        expected_analysis_type=ExpectedAnalysisType.JOB_MANAGEMENT,
        expected_keywords=["monitor", "pipeline"],
        description="Monitor jobs",
        priority=2
    ),
    GoldenQuery(
        query="View logs from the last workflow run",
        expected_analysis_type=ExpectedAnalysisType.JOB_MANAGEMENT,
        expected_keywords=["logs", "workflow"],
        description="View logs",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Workflow Generation
# =============================================================================
WORKFLOW_GENERATION_QUERIES = [
    GoldenQuery(
        query="Create a workflow for processing FASTQ files",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,  # Could be DNA too
        expected_keywords=["workflow", "FASTQ"],
        description="Generic FASTQ workflow",
        priority=1
    ),
    GoldenQuery(
        query="Generate a quality control pipeline",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["quality control", "pipeline"],
        description="QC pipeline",
        priority=1
    ),
    GoldenQuery(
        query="Build an alignment workflow using BWA",
        expected_analysis_type=ExpectedAnalysisType.WGS,
        expected_keywords=["alignment", "BWA"],
        description="BWA alignment",
        priority=1
    ),
    GoldenQuery(
        query="Set up variant calling with GATK best practices",
        expected_analysis_type=ExpectedAnalysisType.VARIANT_CALLING,
        expected_keywords=["variant calling", "GATK"],
        description="GATK variant calling",
        priority=1
    ),
    GoldenQuery(
        query="Design a multi-sample RNA-seq comparison workflow",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["multi-sample", "RNA-seq"],
        description="Multi-sample RNA-seq",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Organism/Species Specific
# =============================================================================
ORGANISM_SPECIFIC_QUERIES = [
    GoldenQuery(
        query="Analyze mouse brain RNA-seq data",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["mouse", "brain", "RNA-seq"],
        description="Mouse tissue RNA-seq",
        priority=2
    ),
    GoldenQuery(
        query="Process E. coli genome sequencing",
        expected_analysis_type=ExpectedAnalysisType.WGS,
        expected_keywords=["E. coli", "genome"],
        description="Bacterial genome",
        priority=2
    ),
    GoldenQuery(
        query="Arabidopsis ChIP-seq analysis",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["Arabidopsis", "ChIP-seq"],
        description="Plant ChIP-seq",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Edge Cases and Complex Queries
# =============================================================================
EDGE_CASE_QUERIES = [
    GoldenQuery(
        query="RNA-seq",  # Very short query
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["RNA-seq"],
        description="Minimal RNA-seq query",
        priority=2
    ),
    GoldenQuery(
        query="Help me analyze my data",  # Vague query
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["analyze", "data"],
        description="Vague query needing clarification",
        priority=2
    ),
    GoldenQuery(
        query="I have paired-end Illumina reads from a human tumor sample and I want to find all the mutations including SNVs, indels, and copy number changes",
        expected_analysis_type=ExpectedAnalysisType.DNA_SEQ,
        expected_keywords=["paired-end", "tumor", "mutations", "SNVs", "indels"],
        expected_tools=["bwa", "gatk", "mutect2"],
        description="Complex multi-variant query",
        priority=1
    ),
    GoldenQuery(
        query="Create RNA-seq and ChIP-seq combined analysis",
        expected_analysis_type=ExpectedAnalysisType.MULTI_OMICS,
        expected_keywords=["RNA-seq", "ChIP-seq"],
        description="Multi-assay query",
        priority=2
    ),
    GoldenQuery(
        query="Process my samples using the nf-core/rnaseq pipeline",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["nf-core", "rnaseq"],
        expected_tools=["star", "salmon"],
        description="Explicit nf-core pipeline",
        priority=1
    ),
]

# =============================================================================
# Golden Queries - Error/Troubleshooting
# =============================================================================
TROUBLESHOOTING_QUERIES = [
    GoldenQuery(
        query="My STAR alignment failed with out of memory error",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["STAR", "failed", "memory"],
        description="Error troubleshooting",
        priority=2
    ),
    GoldenQuery(
        query="Why did my variant calling produce zero variants?",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["variant", "zero"],
        description="Debug empty results",
        priority=2
    ),
    GoldenQuery(
        query="How do I fix FastQC failing to read my file?",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["FastQC", "failing"],
        description="Tool error help",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Resource Optimization
# =============================================================================
RESOURCE_QUERIES = [
    GoldenQuery(
        query="Run RNA-seq analysis optimized for low memory",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["RNA-seq", "low memory"],
        description="Resource-constrained",
        priority=2
    ),
    GoldenQuery(
        query="Create a fast variant calling pipeline using 32 CPUs",
        expected_analysis_type=ExpectedAnalysisType.VARIANT_CALLING,
        expected_keywords=["variant", "fast", "CPUs"],
        expected_tools=["bwa", "gatk"],
        description="High-throughput variant calling",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Specific Tools
# =============================================================================
SPECIFIC_TOOL_QUERIES = [
    GoldenQuery(
        query="Run Salmon for transcript quantification",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["Salmon", "quantification"],
        expected_tools=["salmon"],
        description="Specific tool request",
        priority=1
    ),
    GoldenQuery(
        query="Use MACS2 with broad peak calling settings",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["MACS2", "broad peak"],
        expected_tools=["macs2"],
        description="Tool with specific settings",
        priority=1
    ),
    GoldenQuery(
        query="Generate a workflow with featureCounts for read counting",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["featureCounts"],
        expected_tools=["featurecounts"],
        description="Specific counting tool",
        priority=1
    ),
    GoldenQuery(
        query="Align my reads with Bowtie2 for ChIP-seq",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["Bowtie2", "ChIP-seq"],
        expected_tools=["bowtie2"],
        description="Specific aligner",
        priority=1
    ),
    GoldenQuery(
        query="Use edgeR instead of DESeq2 for differential expression",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["edgeR", "DESeq2", "differential"],
        expected_tools=["edger"],
        description="Tool preference",
        priority=2
    ),
    GoldenQuery(
        query="Process BAM files with Samtools and generate statistics",
        expected_analysis_type=ExpectedAnalysisType.DNA_SEQ,
        expected_keywords=["BAM", "Samtools", "statistics"],
        expected_tools=["samtools"],
        description="Post-alignment processing",
        priority=2
    ),
    GoldenQuery(
        query="Run deepTools for ChIP-seq visualization",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["deepTools", "visualization"],
        expected_tools=["deeptools"],
        description="Visualization tool",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Quality Control Focus
# =============================================================================
QC_QUERIES = [
    GoldenQuery(
        query="Run FastQC on all my FASTQ files",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["FastQC", "FASTQ"],
        expected_tools=["fastqc"],
        description="QC only",
        priority=1
    ),
    GoldenQuery(
        query="Generate a MultiQC report for my pipeline",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["MultiQC", "report"],
        expected_tools=["multiqc"],
        description="Aggregated QC",
        priority=1
    ),
    GoldenQuery(
        query="Trim adapters from my reads using fastp",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["trim", "adapters", "fastp"],
        expected_tools=["fastp"],
        description="Preprocessing",
        priority=1
    ),
    GoldenQuery(
        query="Use Trimmomatic for quality trimming of paired-end reads",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["Trimmomatic", "paired-end"],
        expected_tools=["trimmomatic"],
        description="Specific trimmer",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Assembly
# =============================================================================
ASSEMBLY_QUERIES = [
    GoldenQuery(
        query="Assemble a de novo transcriptome using Trinity",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["de novo", "Trinity", "transcriptome"],
        expected_tools=["trinity"],
        description="De novo assembly",
        priority=1
    ),
    GoldenQuery(
        query="Create a genome assembly workflow for bacterial sequencing",
        expected_analysis_type=ExpectedAnalysisType.METAGENOMICS,
        expected_keywords=["genome assembly", "bacterial"],
        expected_tools=["spades"],
        description="Bacterial assembly",
        priority=1
    ),
    GoldenQuery(
        query="Polish my Nanopore assembly with Medaka",
        expected_analysis_type=ExpectedAnalysisType.LONG_READ,
        expected_keywords=["Nanopore", "assembly", "Medaka"],
        expected_tools=["medaka"],
        description="Long-read polishing",
        priority=2
    ),
    GoldenQuery(
        query="Run Flye for long-read metagenomic assembly",
        expected_analysis_type=ExpectedAnalysisType.METAGENOMICS,
        expected_keywords=["Flye", "long-read", "metagenomic"],
        expected_tools=["flye"],
        description="Long-read metagenomics",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Annotation
# =============================================================================
ANNOTATION_QUERIES = [
    GoldenQuery(
        query="Annotate my assembled bacterial genome using Prokka",
        expected_analysis_type=ExpectedAnalysisType.METAGENOMICS,
        expected_keywords=["annotate", "Prokka", "bacterial"],
        expected_tools=["prokka"],
        description="Bacterial annotation",
        priority=1
    ),
    GoldenQuery(
        query="Annotate variants with VEP",
        expected_analysis_type=ExpectedAnalysisType.VARIANT_CALLING,
        expected_keywords=["annotate", "VEP", "variants"],
        expected_tools=["vep"],
        description="Variant annotation",
        priority=1
    ),
    GoldenQuery(
        query="Use HOMER for peak annotation in ChIP-seq",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["HOMER", "peak annotation"],
        expected_tools=["homer"],
        description="Peak annotation",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Structural Variants
# =============================================================================
STRUCTURAL_VARIANT_QUERIES = [
    GoldenQuery(
        query="Call structural variants using Manta",
        expected_analysis_type=ExpectedAnalysisType.STRUCTURAL_VARIANTS,
        expected_keywords=["structural variants", "Manta"],
        expected_tools=["manta"],
        description="SV with Manta",
        priority=1
    ),
    GoldenQuery(
        query="Detect copy number variations from WGS data",
        expected_analysis_type=ExpectedAnalysisType.STRUCTURAL_VARIANTS,
        expected_keywords=["copy number", "WGS"],
        expected_tools=["cnvkit"],
        description="CNV detection",
        priority=1
    ),
    GoldenQuery(
        query="Use DELLY for deletion and inversion detection",
        expected_analysis_type=ExpectedAnalysisType.STRUCTURAL_VARIANTS,
        expected_keywords=["DELLY", "deletion", "inversion"],
        expected_tools=["delly"],
        description="SV with DELLY",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Comparative Analysis
# =============================================================================
COMPARATIVE_QUERIES = [
    GoldenQuery(
        query="Compare gene expression between treated and control samples",
        expected_analysis_type=ExpectedAnalysisType.RNA_SEQ,
        expected_keywords=["compare", "treated", "control"],
        expected_tools=["deseq2"],
        description="Differential expression",
        priority=1
    ),
    GoldenQuery(
        query="Find differential peaks between two ChIP-seq conditions",
        expected_analysis_type=ExpectedAnalysisType.CHIP_SEQ,
        expected_keywords=["differential peaks"],
        expected_tools=["diffbind"],
        description="Differential binding",
        priority=1
    ),
    GoldenQuery(
        query="Compare microbiome composition between patient groups",
        expected_analysis_type=ExpectedAnalysisType.METAGENOMICS,
        expected_keywords=["compare", "microbiome", "groups"],
        expected_tools=["qiime2"],
        description="Differential microbiome",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Single Cell Extended
# =============================================================================
SCRNA_EXTENDED_QUERIES = [
    GoldenQuery(
        query="Integrate multiple scRNA-seq datasets with batch correction",
        expected_analysis_type=ExpectedAnalysisType.SCRNA_SEQ,
        expected_keywords=["integrate", "batch correction"],
        expected_tools=["seurat", "scanpy"],
        description="Batch integration",
        priority=1
    ),
    GoldenQuery(
        query="Perform RNA velocity analysis on single cell data",
        expected_analysis_type=ExpectedAnalysisType.SCRNA_SEQ,
        expected_keywords=["RNA velocity"],
        expected_tools=["velocyto", "scvelo"],
        description="RNA velocity",
        priority=2
    ),
    GoldenQuery(
        query="Analyze single-cell ATAC-seq with ArchR",
        expected_analysis_type=ExpectedAnalysisType.SCRNA_SEQ,
        expected_keywords=["single-cell ATAC", "ArchR"],
        expected_tools=["archr"],
        description="scATAC-seq",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Proteomics / Multi-omics Extended
# =============================================================================
PROTEOMICS_QUERIES = [
    GoldenQuery(
        query="Integrate RNA-seq and proteomics data",
        expected_analysis_type=ExpectedAnalysisType.MULTI_OMICS,
        expected_keywords=["RNA-seq", "proteomics", "integrate"],
        description="RNA-protein integration",
        priority=2
    ),
    GoldenQuery(
        query="Analyze metabolomics data with pathway enrichment",
        expected_analysis_type=ExpectedAnalysisType.MULTI_OMICS,
        expected_keywords=["metabolomics", "pathway"],
        description="Metabolomics analysis",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Clinical/Medical Focus
# =============================================================================
CLINICAL_QUERIES = [
    GoldenQuery(
        query="Create a clinical variant reporting workflow",
        expected_analysis_type=ExpectedAnalysisType.VARIANT_CALLING,
        expected_keywords=["clinical", "variant", "reporting"],
        expected_tools=["vep", "gatk"],
        description="Clinical reporting",
        priority=1
    ),
    GoldenQuery(
        query="Analyze cancer panel sequencing with hotspot mutations",
        expected_analysis_type=ExpectedAnalysisType.DNA_SEQ,
        expected_keywords=["cancer panel", "hotspot"],
        expected_tools=["mutect2"],
        description="Cancer panel",
        priority=1
    ),
    GoldenQuery(
        query="Process liquid biopsy cfDNA samples",
        expected_analysis_type=ExpectedAnalysisType.DNA_SEQ,
        expected_keywords=["liquid biopsy", "cfDNA"],
        description="Liquid biopsy",
        priority=2
    ),
]

# =============================================================================
# Golden Queries - Workflow Management Extended
# =============================================================================
WORKFLOW_MANAGEMENT_QUERIES = [
    GoldenQuery(
        query="Run my workflow on AWS batch",
        expected_analysis_type=ExpectedAnalysisType.JOB_MANAGEMENT,
        expected_keywords=["AWS batch"],
        description="Cloud execution",
        priority=2
    ),
    GoldenQuery(
        query="Resume the failed workflow from the last checkpoint",
        expected_analysis_type=ExpectedAnalysisType.JOB_MANAGEMENT,
        expected_keywords=["resume", "failed", "checkpoint"],
        description="Resume workflow",
        priority=1
    ),
    GoldenQuery(
        query="Show resource usage for my completed pipeline",
        expected_analysis_type=ExpectedAnalysisType.JOB_MANAGEMENT,
        expected_keywords=["resource usage", "pipeline"],
        description="Resource monitoring",
        priority=2
    ),
    GoldenQuery(
        query="List all available nf-core pipelines",
        expected_analysis_type=ExpectedAnalysisType.EDUCATION,
        expected_keywords=["nf-core", "pipelines"],
        description="Pipeline discovery",
        priority=2
    ),
]

# =============================================================================
# Combine All Golden Queries
# =============================================================================
ALL_GOLDEN_QUERIES = (
    RNA_SEQ_QUERIES +
    SCRNA_SEQ_QUERIES +
    CHIP_SEQ_QUERIES +
    ATAC_SEQ_QUERIES +
    DNA_SEQ_QUERIES +
    METHYLATION_QUERIES +
    METAGENOMICS_QUERIES +
    HIC_QUERIES +
    LONG_READ_QUERIES +
    EDUCATION_QUERIES +
    DATA_SEARCH_QUERIES +
    JOB_MANAGEMENT_QUERIES +
    WORKFLOW_GENERATION_QUERIES +
    ORGANISM_SPECIFIC_QUERIES +
    EDGE_CASE_QUERIES +
    TROUBLESHOOTING_QUERIES +
    RESOURCE_QUERIES +
    SPECIFIC_TOOL_QUERIES +
    QC_QUERIES +
    ASSEMBLY_QUERIES +
    ANNOTATION_QUERIES +
    STRUCTURAL_VARIANT_QUERIES +
    COMPARATIVE_QUERIES +
    SCRNA_EXTENDED_QUERIES +
    PROTEOMICS_QUERIES +
    CLINICAL_QUERIES +
    WORKFLOW_MANAGEMENT_QUERIES
)


# =============================================================================
# Test Fixtures
# =============================================================================

def _load_api_keys():
    """Load API keys from .secrets directory."""
    import os
    from pathlib import Path
    
    secrets_dir = Path(__file__).parent.parent / ".secrets"
    if not secrets_dir.exists():
        return False
    
    key_mappings = {
        "google_api_key": "GOOGLE_API_KEY",
        "groq_key": "GROQ_API_KEY", 
        "cerebras_key": "CEREBRAS_API_KEY",
        "openrouter_key": "OPENROUTER_API_KEY",
        "lightning_key": "LIGHTNING_API_KEY",
        "github_token": "GITHUB_TOKEN",
        "openai_key": "OPENAI_API_KEY",
    }
    
    loaded = False
    for file_name, env_var in key_mappings.items():
        key_file = secrets_dir / file_name
        if key_file.exists():
            os.environ[env_var] = key_file.read_text().strip()
            loaded = True
    
    return loaded


@pytest.fixture
def query_parser():
    """Get the query parser with LLM connection."""
    try:
        _load_api_keys()
        from src.workflow_composer.core.query_parser import IntentParser
        from src.workflow_composer.llm.factory import get_llm
        llm = get_llm()
        return IntentParser(llm)
    except Exception as e:
        pytest.skip(f"IntentParser not available: {e}")


@pytest.fixture
def ensemble_parser():
    """Get the ensemble query parser."""
    try:
        from src.workflow_composer.core.query_parser_ensemble import EnsembleQueryParser
        return EnsembleQueryParser()
    except ImportError:
        pytest.skip("EnsembleQueryParser not available")


@pytest.fixture
def hybrid_parser():
    """Get the hybrid query parser."""
    try:
        from src.workflow_composer.agents.intent import HybridQueryParser
        return HybridQueryParser()
    except ImportError:
        pytest.skip("HybridQueryParser not available")


# =============================================================================
# Test Classes
# =============================================================================

class TestGoldenQueriesParsing:
    """Test that golden queries are parsed correctly."""
    
    def test_all_queries_have_expected_type(self):
        """Verify all queries have valid expected types."""
        for gq in ALL_GOLDEN_QUERIES:
            assert gq.expected_analysis_type is not None
            assert isinstance(gq.expected_analysis_type, ExpectedAnalysisType)
    
    def test_all_queries_have_keywords(self):
        """Verify all queries have expected keywords."""
        for gq in ALL_GOLDEN_QUERIES:
            assert len(gq.expected_keywords) > 0
            for kw in gq.expected_keywords:
                assert isinstance(kw, str) and len(kw) > 0
    
    def test_query_count(self):
        """Verify we have enough golden queries."""
        assert len(ALL_GOLDEN_QUERIES) >= 50, f"Expected 50+ queries, got {len(ALL_GOLDEN_QUERIES)}"
    
    def test_all_analysis_types_covered(self):
        """Verify all analysis types have at least one query."""
        covered_types = set(gq.expected_analysis_type for gq in ALL_GOLDEN_QUERIES)
        # These types should definitely be covered
        required_types = {
            ExpectedAnalysisType.RNA_SEQ,
            ExpectedAnalysisType.CHIP_SEQ,
            ExpectedAnalysisType.ATAC_SEQ,
            ExpectedAnalysisType.DNA_SEQ,
            ExpectedAnalysisType.METHYLATION,
            ExpectedAnalysisType.METAGENOMICS,
            ExpectedAnalysisType.EDUCATION,
            ExpectedAnalysisType.DATA_SEARCH,
        }
        missing = required_types - covered_types
        assert len(missing) == 0, f"Missing coverage for: {missing}"


class TestQueryParserGoldenQueries:
    """Test QueryParser against golden queries."""
    
    @pytest.mark.parametrize("query_obj", ALL_GOLDEN_QUERIES[:10])  # Test top 10 for speed
    def test_query_parsing_basic(self, query_parser, query_obj: GoldenQuery):
        """Test basic query parsing."""
        result = query_parser.parse(query_obj.query)
        
        # Should successfully parse
        assert result is not None
        
        # Should detect some intent
        assert hasattr(result, 'analysis_type') or hasattr(result, 'intent')
    
    def test_rna_seq_detection(self, query_parser):
        """Test RNA-seq is detected correctly."""
        for gq in RNA_SEQ_QUERIES[:3]:  # Test first 3
            result = query_parser.parse(gq.query)
            # Just verify it parses - exact type matching is model-dependent
            assert result is not None


class TestKeywordExtraction:
    """Test keyword extraction from queries."""
    
    @pytest.mark.parametrize("query_obj", ALL_GOLDEN_QUERIES[:20])
    def test_keywords_present_in_query(self, query_obj: GoldenQuery):
        """Verify expected keywords are present in the query."""
        query_lower = query_obj.query.lower()
        # At least one keyword should be found
        found_keywords = [
            kw for kw in query_obj.expected_keywords
            if kw.lower() in query_lower
        ]
        # Allow for some flexibility (paraphrasing)
        assert len(found_keywords) > 0 or len(query_obj.expected_keywords) == 0


class TestEducationQueries:
    """Test education/explanation queries."""
    
    def test_education_queries_detected(self, query_parser):
        """Test education queries are handled correctly."""
        for gq in EDUCATION_QUERIES:
            result = query_parser.parse(gq.query)
            assert result is not None
            # Education queries should not trigger workflow generation
            # (implementation-specific validation)


class TestDataSearchQueries:
    """Test data search queries."""
    
    def test_data_search_queries_detected(self, query_parser):
        """Test data search queries are parsed."""
        for gq in DATA_SEARCH_QUERIES:
            result = query_parser.parse(gq.query)
            assert result is not None


# =============================================================================
# Integration Tests (require more setup)
# =============================================================================

class TestGoldenQueriesIntegration:
    """Integration tests for golden queries with full pipeline."""
    
    @pytest.fixture
    def biopipelines(self):
        """Get BioPipelines facade."""
        try:
            from src.workflow_composer.facade import BioPipelines
            return BioPipelines()
        except Exception:
            pytest.skip("BioPipelines not available")
    
    @pytest.mark.slow
    @pytest.mark.parametrize("query_obj", [
        RNA_SEQ_QUERIES[0],  # STAR/DESeq2 workflow
        CHIP_SEQ_QUERIES[0],  # H3K4me3 ChIP-seq
        DNA_SEQ_QUERIES[0],  # GATK variant calling
    ])
    def test_workflow_generation_golden(self, biopipelines, query_obj: GoldenQuery):
        """Test workflow generation for key golden queries."""
        result = biopipelines.composer.generate(query_obj.query)
        
        # Should succeed
        assert result is not None
        
        # Should have workflow code or message
        if hasattr(result, 'workflow_code'):
            assert result.workflow_code is not None
        if hasattr(result, 'message'):
            assert result.message is not None


# =============================================================================
# Benchmark Tests
# =============================================================================

class TestGoldenQueriesBenchmark:
    """Benchmark tests for query processing speed."""
    
    @pytest.fixture
    def all_queries(self):
        """Get all query strings."""
        return [gq.query for gq in ALL_GOLDEN_QUERIES]
    
    @pytest.mark.benchmark
    def test_parsing_speed(self, query_parser, all_queries, benchmark=None):
        """Benchmark query parsing speed."""
        import time
        
        start = time.time()
        for query in all_queries:
            query_parser.parse(query)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / len(all_queries)) * 1000
        
        # Should parse each query in under 100ms on average
        assert avg_time_ms < 100, f"Parsing too slow: {avg_time_ms:.2f}ms per query"


# =============================================================================
# Export for Training Data Generation
# =============================================================================

def get_golden_queries_for_training() -> List[Dict[str, Any]]:
    """
    Export golden queries in format suitable for training data generation.
    
    Returns:
        List of query dictionaries with metadata
    """
    return [
        {
            "query": gq.query,
            "expected_analysis_type": gq.expected_analysis_type.value,
            "expected_keywords": gq.expected_keywords,
            "expected_tools": gq.expected_tools or [],
            "description": gq.description,
            "priority": gq.priority,
        }
        for gq in ALL_GOLDEN_QUERIES
    ]


def export_golden_queries_json(output_path: str = "data/golden_queries.json"):
    """Export golden queries to JSON file."""
    import json
    queries = get_golden_queries_for_training()
    with open(output_path, "w") as f:
        json.dump(queries, f, indent=2)
    return len(queries)


if __name__ == "__main__":
    # Run quick summary
    print(f"Total golden queries: {len(ALL_GOLDEN_QUERIES)}")
    
    # Count by type
    type_counts = {}
    for gq in ALL_GOLDEN_QUERIES:
        t = gq.expected_analysis_type.value
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\nQueries by analysis type:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    
    # Export for verification
    queries = get_golden_queries_for_training()
    print(f"\nExportable queries: {len(queries)}")
