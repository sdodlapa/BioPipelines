"""
Tool Catalog Indexer
====================

Indexes comprehensive bioinformatics tool information into the knowledge base.

This indexer extracts tool information from:
- config/tool_mappings.yaml: Tool definitions with processes and containers
- config/analysis_definitions.yaml: Tool usage patterns per analysis type
- data/tool_catalog/: Container tool lists
- Built-in tool descriptions: Common bioinformatics tools

The indexed data enables:
- Tool discovery based on analysis type
- Understanding tool capabilities
- Recommending tool combinations
- Explaining tool purposes for education
"""

import logging
import yaml
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from .knowledge_base import KnowledgeBase, KnowledgeDocument, KnowledgeSource

logger = logging.getLogger(__name__)


# =============================================================================
# Comprehensive Tool Descriptions
# =============================================================================

TOOL_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    # Alignment Tools
    "star": {
        "full_name": "STAR (Spliced Transcripts Alignment to a Reference)",
        "description": "Ultra-fast RNA-seq aligner that can detect novel splice junctions and chimeric sequences. Uses suffix array-based genome index for rapid alignment.",
        "category": "alignment",
        "use_cases": ["RNA-seq alignment", "Splice junction detection", "Fusion gene detection"],
        "input_formats": ["FASTQ"],
        "output_formats": ["BAM", "SAM"],
        "homepage": "https://github.com/alexdobin/STAR",
        "citation": "Dobin et al., 2013, Bioinformatics"
    },
    "bwa": {
        "full_name": "BWA (Burrows-Wheeler Aligner)",
        "description": "Fast and accurate short read aligner based on Burrows-Wheeler transform. Supports both single-end and paired-end reads. BWA-MEM is recommended for reads >70bp.",
        "category": "alignment",
        "use_cases": ["DNA-seq alignment", "ChIP-seq alignment", "Variant calling preparation"],
        "input_formats": ["FASTQ", "FASTA"],
        "output_formats": ["SAM", "BAM"],
        "homepage": "https://bio-bwa.sourceforge.net/",
        "citation": "Li & Durbin, 2009, Bioinformatics"
    },
    "bowtie2": {
        "full_name": "Bowtie 2",
        "description": "Fast and sensitive read aligner optimized for reads longer than 50bp. Uses FM-index-based approach with support for gapped, local, and paired-end alignment.",
        "category": "alignment",
        "use_cases": ["ChIP-seq alignment", "ATAC-seq alignment", "Short read mapping"],
        "input_formats": ["FASTQ", "FASTA"],
        "output_formats": ["SAM", "BAM"],
        "homepage": "http://bowtie-bio.sourceforge.net/bowtie2/",
        "citation": "Langmead & Salzberg, 2012, Nature Methods"
    },
    "hisat2": {
        "full_name": "HISAT2 (Hierarchical Indexing for Spliced Alignment of Transcripts)",
        "description": "Graph-based RNA-seq aligner that uses a hierarchical indexing scheme for fast and sensitive alignment. Excellent for detecting known and novel splice sites.",
        "category": "alignment",
        "use_cases": ["RNA-seq alignment", "Splice site detection"],
        "input_formats": ["FASTQ"],
        "output_formats": ["SAM", "BAM"],
        "homepage": "http://daehwankimlab.github.io/hisat2/",
        "citation": "Kim et al., 2019, Nature Biotechnology"
    },
    "minimap2": {
        "full_name": "Minimap2",
        "description": "Versatile aligner for long reads (PacBio, Oxford Nanopore) and short reads. Supports splice-aware alignment for direct RNA sequencing.",
        "category": "alignment",
        "use_cases": ["Long-read alignment", "Direct RNA-seq", "Assembly polishing"],
        "input_formats": ["FASTQ", "FASTA"],
        "output_formats": ["SAM", "PAF"],
        "homepage": "https://github.com/lh3/minimap2",
        "citation": "Li, 2018, Bioinformatics"
    },
    
    # Quantification Tools
    "featurecounts": {
        "full_name": "featureCounts",
        "description": "Highly efficient read summarization tool that counts reads mapped to genomic features like genes, exons, promoters. Part of the Subread package.",
        "category": "quantification",
        "use_cases": ["Gene expression quantification", "Read counting", "RNA-seq analysis"],
        "input_formats": ["BAM", "SAM"],
        "output_formats": ["Count matrix (TSV)"],
        "homepage": "http://subread.sourceforge.net/",
        "citation": "Liao et al., 2014, Bioinformatics"
    },
    "salmon": {
        "full_name": "Salmon",
        "description": "Fast transcript-level quantification using quasi-mapping. Provides accurate abundance estimates with bias correction for GC content and sequence-specific effects.",
        "category": "quantification",
        "use_cases": ["Transcript quantification", "Isoform expression", "Differential expression"],
        "input_formats": ["FASTQ"],
        "output_formats": ["Quant files (TSV)"],
        "homepage": "https://combine-lab.github.io/salmon/",
        "citation": "Patro et al., 2017, Nature Methods"
    },
    "kallisto": {
        "full_name": "Kallisto",
        "description": "Ultra-fast RNA-seq quantification using pseudoalignment. Produces accurate transcript abundances without traditional alignment.",
        "category": "quantification",
        "use_cases": ["Transcript quantification", "Single-cell RNA-seq", "Bootstrapped abundance estimates"],
        "input_formats": ["FASTQ"],
        "output_formats": ["H5", "TSV"],
        "homepage": "https://pachterlab.github.io/kallisto/",
        "citation": "Bray et al., 2016, Nature Biotechnology"
    },
    "rsem": {
        "full_name": "RSEM (RNA-Seq by Expectation Maximization)",
        "description": "Accurate quantification of gene and isoform expression from RNA-seq. Uses statistical model to handle ambiguously mapping reads.",
        "category": "quantification",
        "use_cases": ["Gene expression", "Isoform quantification", "De novo transcriptome analysis"],
        "input_formats": ["FASTQ", "BAM"],
        "output_formats": ["Expression matrix"],
        "homepage": "https://deweylab.github.io/RSEM/",
        "citation": "Li & Dewey, 2011, BMC Bioinformatics"
    },
    "htseq": {
        "full_name": "HTSeq",
        "description": "Python framework for working with high-throughput sequencing data. htseq-count is widely used for counting reads in features.",
        "category": "quantification",
        "use_cases": ["Read counting", "Gene expression"],
        "input_formats": ["BAM", "SAM"],
        "output_formats": ["Count table"],
        "homepage": "https://htseq.readthedocs.io/",
        "citation": "Anders et al., 2015, Bioinformatics"
    },
    
    # QC Tools
    "fastqc": {
        "full_name": "FastQC",
        "description": "Quality control tool for high throughput sequence data. Produces comprehensive reports on read quality, GC content, adapter contamination, and more.",
        "category": "qc",
        "use_cases": ["Read quality assessment", "Adapter detection", "Quality metrics"],
        "input_formats": ["FASTQ", "BAM"],
        "output_formats": ["HTML report", "ZIP archive"],
        "homepage": "https://www.bioinformatics.babraham.ac.uk/projects/fastqc/",
        "citation": "Andrews, 2010"
    },
    "multiqc": {
        "full_name": "MultiQC",
        "description": "Aggregate results from multiple bioinformatics tools into a single report. Supports 100+ tools and creates interactive HTML reports.",
        "category": "qc",
        "use_cases": ["Report aggregation", "Pipeline QC summary", "Batch analysis"],
        "input_formats": ["Various tool outputs"],
        "output_formats": ["HTML", "JSON", "TSV"],
        "homepage": "https://multiqc.info/",
        "citation": "Ewels et al., 2016, Bioinformatics"
    },
    "qualimap": {
        "full_name": "Qualimap",
        "description": "Platform for quality assessment of sequencing alignment data. Provides comprehensive statistics for BAM files.",
        "category": "qc",
        "use_cases": ["Alignment QC", "Coverage analysis", "RNA-seq QC"],
        "input_formats": ["BAM"],
        "output_formats": ["HTML", "PDF"],
        "homepage": "http://qualimap.conesalab.org/",
        "citation": "García-Alcalde et al., 2012, Bioinformatics"
    },
    
    # Trimming Tools
    "trimmomatic": {
        "full_name": "Trimmomatic",
        "description": "Flexible read trimming tool for Illumina sequence data. Removes adapters and low-quality bases with various quality filtering options.",
        "category": "trimming",
        "use_cases": ["Adapter trimming", "Quality filtering", "Read preprocessing"],
        "input_formats": ["FASTQ"],
        "output_formats": ["FASTQ"],
        "homepage": "http://www.usadellab.org/cms/?page=trimmomatic",
        "citation": "Bolger et al., 2014, Bioinformatics"
    },
    "fastp": {
        "full_name": "fastp",
        "description": "Ultra-fast all-in-one FASTQ preprocessor. Performs quality control, adapter trimming, and filtering with automatic adapter detection.",
        "category": "trimming",
        "use_cases": ["All-in-one preprocessing", "Adapter trimming", "Quality filtering"],
        "input_formats": ["FASTQ"],
        "output_formats": ["FASTQ", "HTML", "JSON"],
        "homepage": "https://github.com/OpenGene/fastp",
        "citation": "Chen et al., 2018, Bioinformatics"
    },
    "cutadapt": {
        "full_name": "Cutadapt",
        "description": "Finds and removes adapter sequences, primers, and other unwanted sequences from reads. Supports various adapter types and quality trimming.",
        "category": "trimming",
        "use_cases": ["Adapter removal", "Primer trimming", "Quality trimming"],
        "input_formats": ["FASTQ", "FASTA"],
        "output_formats": ["FASTQ", "FASTA"],
        "homepage": "https://cutadapt.readthedocs.io/",
        "citation": "Martin, 2011, EMBnet.journal"
    },
    "trim_galore": {
        "full_name": "Trim Galore",
        "description": "Wrapper around Cutadapt and FastQC for consistent adapter and quality trimming. Autodetects adapters and runs FastQC.",
        "category": "trimming",
        "use_cases": ["Automatic adapter detection", "Bisulfite sequencing prep", "RRBS trimming"],
        "input_formats": ["FASTQ"],
        "output_formats": ["FASTQ"],
        "homepage": "https://www.bioinformatics.babraham.ac.uk/projects/trim_galore/",
        "citation": "Krueger, 2012"
    },
    
    # Peak Calling Tools
    "macs2": {
        "full_name": "MACS2 (Model-based Analysis of ChIP-Seq)",
        "description": "Identifies transcription factor binding sites from ChIP-seq data. Uses local Poisson distribution to model background and identify enriched regions.",
        "category": "peaks",
        "use_cases": ["ChIP-seq peak calling", "ATAC-seq peaks", "DNase-seq analysis"],
        "input_formats": ["BAM", "BED"],
        "output_formats": ["narrowPeak", "broadPeak", "BED"],
        "homepage": "https://github.com/macs3-project/MACS",
        "citation": "Zhang et al., 2008, Genome Biology"
    },
    "homer": {
        "full_name": "HOMER (Hypergeometric Optimization of Motif EnRichment)",
        "description": "Suite for ChIP-seq analysis including peak calling, motif discovery, and annotation. Excellent for identifying transcription factor binding motifs.",
        "category": "peaks",
        "use_cases": ["Peak calling", "Motif discovery", "Peak annotation", "GO analysis"],
        "input_formats": ["BAM", "BED"],
        "output_formats": ["Various"],
        "homepage": "http://homer.ucsd.edu/homer/",
        "citation": "Heinz et al., 2010, Molecular Cell"
    },
    
    # Variant Calling Tools
    "gatk": {
        "full_name": "GATK (Genome Analysis Toolkit)",
        "description": "Industry-standard variant discovery toolkit. Provides best-practice pipelines for germline and somatic variant calling with advanced filtering.",
        "category": "variant_calling",
        "use_cases": ["Germline variant calling", "Somatic variants", "Structural variants", "CNV detection"],
        "input_formats": ["BAM", "CRAM"],
        "output_formats": ["VCF", "gVCF"],
        "homepage": "https://gatk.broadinstitute.org/",
        "citation": "McKenna et al., 2010, Genome Research"
    },
    "bcftools": {
        "full_name": "BCFtools",
        "description": "Utilities for variant calling and manipulating VCFs and BCFs. Part of the samtools suite with efficient variant operations.",
        "category": "variant_calling",
        "use_cases": ["Variant calling", "VCF manipulation", "Variant filtering"],
        "input_formats": ["BAM", "VCF", "BCF"],
        "output_formats": ["VCF", "BCF"],
        "homepage": "https://samtools.github.io/bcftools/",
        "citation": "Danecek et al., 2021, GigaScience"
    },
    "freebayes": {
        "full_name": "FreeBayes",
        "description": "Bayesian haplotype-based genetic variant detector. Detects SNPs, indels, MNPs, and complex events from short-read alignments.",
        "category": "variant_calling",
        "use_cases": ["Variant calling", "Population genetics", "Pooled samples"],
        "input_formats": ["BAM"],
        "output_formats": ["VCF"],
        "homepage": "https://github.com/freebayes/freebayes",
        "citation": "Garrison & Marth, 2012, arXiv"
    },
    
    # Structural Variant Tools
    "manta": {
        "full_name": "Manta",
        "description": "Structural variant and indel caller for mapped sequencing data. Detects deletions, insertions, duplications, inversions, and translocations.",
        "category": "structural_variants",
        "use_cases": ["SV detection", "Large indels", "Inversions", "Translocations"],
        "input_formats": ["BAM", "CRAM"],
        "output_formats": ["VCF"],
        "homepage": "https://github.com/Illumina/manta",
        "citation": "Chen et al., 2016, Bioinformatics"
    },
    "delly": {
        "full_name": "Delly",
        "description": "Integrated structural variant discovery using paired-ends, split-reads, and read-depth. Supports germline and somatic SV calling.",
        "category": "structural_variants",
        "use_cases": ["SV discovery", "Deletions", "Duplications", "Inversions"],
        "input_formats": ["BAM", "CRAM"],
        "output_formats": ["VCF", "BCF"],
        "homepage": "https://github.com/dellytools/delly",
        "citation": "Rausch et al., 2012, Bioinformatics"
    },
    
    # Utility Tools
    "samtools": {
        "full_name": "SAMtools",
        "description": "Suite of programs for interacting with high-throughput sequencing data. Essential for BAM/SAM file manipulation, sorting, indexing, and viewing.",
        "category": "utilities",
        "use_cases": ["BAM sorting", "Indexing", "Flagstat", "View/filter alignments"],
        "input_formats": ["SAM", "BAM", "CRAM"],
        "output_formats": ["SAM", "BAM", "CRAM"],
        "homepage": "http://www.htslib.org/",
        "citation": "Li et al., 2009, Bioinformatics"
    },
    "bedtools": {
        "full_name": "BEDTools",
        "description": "Swiss-army knife for genome arithmetic. Performs intersections, merges, counting, and other operations on BED/BAM/VCF files.",
        "category": "utilities",
        "use_cases": ["Interval arithmetic", "Overlap detection", "Coverage", "Format conversion"],
        "input_formats": ["BED", "BAM", "VCF", "GFF"],
        "output_formats": ["BED", "BAM", "Various"],
        "homepage": "https://bedtools.readthedocs.io/",
        "citation": "Quinlan & Hall, 2010, Bioinformatics"
    },
    "picard": {
        "full_name": "Picard Tools",
        "description": "Java tools for manipulating high-throughput sequencing data. Includes duplicate marking, metrics collection, and BAM manipulation.",
        "category": "utilities",
        "use_cases": ["Mark duplicates", "Collect metrics", "BAM manipulation"],
        "input_formats": ["BAM", "SAM"],
        "output_formats": ["BAM", "Metrics files"],
        "homepage": "https://broadinstitute.github.io/picard/",
        "citation": "Broad Institute"
    },
    
    # Differential Expression
    "deseq2": {
        "full_name": "DESeq2",
        "description": "R package for differential gene expression analysis based on negative binomial distribution. Handles variance stabilization and multiple testing correction.",
        "category": "analysis",
        "use_cases": ["Differential expression", "RNA-seq analysis", "Count normalization"],
        "input_formats": ["Count matrix"],
        "output_formats": ["DE results table", "Normalized counts"],
        "homepage": "https://bioconductor.org/packages/DESeq2/",
        "citation": "Love et al., 2014, Genome Biology"
    },
    "edger": {
        "full_name": "edgeR",
        "description": "R package for differential expression analysis of digital gene expression data. Uses empirical Bayes methods and negative binomial distribution.",
        "category": "analysis",
        "use_cases": ["Differential expression", "RNA-seq", "ChIP-seq differential binding"],
        "input_formats": ["Count matrix"],
        "output_formats": ["DE results table"],
        "homepage": "https://bioconductor.org/packages/edgeR/",
        "citation": "Robinson et al., 2010, Bioinformatics"
    },
    
    # Assembly Tools
    "trinity": {
        "full_name": "Trinity",
        "description": "De novo transcriptome assembler for RNA-seq without a reference genome. Uses a three-stage approach: Inchworm, Chrysalis, and Butterfly.",
        "category": "assembly",
        "use_cases": ["De novo transcriptome assembly", "Non-model organisms", "Isoform reconstruction"],
        "input_formats": ["FASTQ"],
        "output_formats": ["FASTA (transcripts)"],
        "homepage": "https://github.com/trinityrnaseq/trinityrnaseq",
        "citation": "Grabherr et al., 2011, Nature Biotechnology"
    },
    "spades": {
        "full_name": "SPAdes",
        "description": "Versatile genome assembler supporting single-cell, metagenomics, plasmid, and RNA-seq assembly. Uses de Bruijn graphs with multiple k-mer sizes.",
        "category": "assembly",
        "use_cases": ["Genome assembly", "Metagenomics", "Plasmid assembly"],
        "input_formats": ["FASTQ"],
        "output_formats": ["FASTA (contigs/scaffolds)"],
        "homepage": "https://cab.spbu.ru/software/spades/",
        "citation": "Bankevich et al., 2012, Journal of Computational Biology"
    },
    "flye": {
        "full_name": "Flye",
        "description": "Long-read de novo genome assembler for PacBio and Oxford Nanopore data. Uses repeat graphs for high-quality assemblies.",
        "category": "assembly",
        "use_cases": ["Long-read assembly", "Metagenomics", "Structural variant resolution"],
        "input_formats": ["FASTQ", "FASTA"],
        "output_formats": ["FASTA", "GFA"],
        "homepage": "https://github.com/fenderglass/Flye",
        "citation": "Kolmogorov et al., 2019, Nature Biotechnology"
    },
    
    # Methylation Tools
    "bismark": {
        "full_name": "Bismark",
        "description": "Bisulfite read mapper and methylation caller. Aligns BS-seq reads and extracts methylation calls for CpG, CHG, and CHH contexts.",
        "category": "methylation",
        "use_cases": ["Bisulfite-seq alignment", "Methylation calling", "RRBS analysis"],
        "input_formats": ["FASTQ"],
        "output_formats": ["BAM", "Methylation reports"],
        "homepage": "https://www.bioinformatics.babraham.ac.uk/projects/bismark/",
        "citation": "Krueger & Andrews, 2011, Bioinformatics"
    },
    
    # Metagenomics Tools
    "kraken2": {
        "full_name": "Kraken 2",
        "description": "Taxonomic classification system using exact k-mer matches. Ultra-fast metagenomic sequence classification with high accuracy.",
        "category": "metagenomics",
        "use_cases": ["Taxonomic classification", "Metagenomic profiling", "Contamination screening"],
        "input_formats": ["FASTQ", "FASTA"],
        "output_formats": ["Kraken report", "Classification output"],
        "homepage": "https://ccb.jhu.edu/software/kraken2/",
        "citation": "Wood et al., 2019, Genome Biology"
    },
    "metaphlan": {
        "full_name": "MetaPhlAn",
        "description": "Metagenomic phylogenetic analysis using clade-specific marker genes. Provides species-level taxonomic profiling from metagenomes.",
        "category": "metagenomics",
        "use_cases": ["Taxonomic profiling", "Species abundance", "Microbiome analysis"],
        "input_formats": ["FASTQ", "FASTA"],
        "output_formats": ["Abundance tables"],
        "homepage": "https://huttenhower.sph.harvard.edu/metaphlan/",
        "citation": "Blanco-Míguez et al., 2023, Nature Biotechnology"
    },
    "megahit": {
        "full_name": "MEGAHIT",
        "description": "Ultra-fast metagenomic assembler using succinct de Bruijn graphs. Memory-efficient for large metagenomic datasets.",
        "category": "metagenomics",
        "use_cases": ["Metagenomic assembly", "Large-scale assembly"],
        "input_formats": ["FASTQ"],
        "output_formats": ["FASTA (contigs)"],
        "homepage": "https://github.com/voutcn/megahit",
        "citation": "Li et al., 2015, Bioinformatics"
    },
    
    # Single-cell Tools
    "cellranger": {
        "full_name": "Cell Ranger",
        "description": "10x Genomics pipeline for single-cell RNA-seq. Performs alignment, filtering, barcode counting, and UMI counting.",
        "category": "scrna",
        "use_cases": ["scRNA-seq processing", "10x Genomics data", "Gene expression"],
        "input_formats": ["FASTQ (10x)"],
        "output_formats": ["BAM", "Feature-barcode matrices"],
        "homepage": "https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/what-is-cell-ranger",
        "citation": "10x Genomics"
    },
    "seurat": {
        "full_name": "Seurat",
        "description": "R toolkit for single-cell genomics analysis. Supports QC, analysis, and exploration of scRNA-seq, spatial transcriptomics, and multimodal data.",
        "category": "scrna",
        "use_cases": ["scRNA-seq analysis", "Clustering", "Trajectory analysis", "Integration"],
        "input_formats": ["Count matrices", "H5AD"],
        "output_formats": ["Seurat objects", "Plots"],
        "homepage": "https://satijalab.org/seurat/",
        "citation": "Hao et al., 2021, Cell"
    },
    "scanpy": {
        "full_name": "Scanpy",
        "description": "Python toolkit for analyzing single-cell gene expression data. Scalable and integrates with the scverse ecosystem.",
        "category": "scrna",
        "use_cases": ["scRNA-seq analysis", "Preprocessing", "Clustering", "Trajectory inference"],
        "input_formats": ["H5AD", "Count matrices"],
        "output_formats": ["H5AD", "Plots"],
        "homepage": "https://scanpy.readthedocs.io/",
        "citation": "Wolf et al., 2018, Genome Biology"
    },
    
    # Visualization Tools
    "deeptools": {
        "full_name": "deepTools",
        "description": "User-friendly tools for exploring deep sequencing data. Creates publication-quality heatmaps, profiles, and signal tracks.",
        "category": "visualization",
        "use_cases": ["Signal visualization", "Heatmaps", "Coverage tracks", "Correlation analysis"],
        "input_formats": ["BAM", "bigWig"],
        "output_formats": ["bigWig", "PNG", "PDF"],
        "homepage": "https://deeptools.readthedocs.io/",
        "citation": "Ramírez et al., 2016, Nucleic Acids Research"
    },
    
    # Hi-C Tools  
    "juicer": {
        "full_name": "Juicer",
        "description": "Hi-C data processing pipeline. Aligns reads, generates contact matrices, and produces .hic files for visualization.",
        "category": "hic",
        "use_cases": ["Hi-C processing", "Contact matrix generation", "TAD calling"],
        "input_formats": ["FASTQ"],
        "output_formats": [".hic", "Contact matrices"],
        "homepage": "https://github.com/aidenlab/juicer",
        "citation": "Durand et al., 2016, Cell Systems"
    },
    "hicpro": {
        "full_name": "HiC-Pro",
        "description": "Optimized and flexible pipeline for Hi-C data processing. Supports various protocols and produces normalized contact maps.",
        "category": "hic",
        "use_cases": ["Hi-C processing", "Contact maps", "Quality control"],
        "input_formats": ["FASTQ"],
        "output_formats": ["Contact matrices", "Normalized maps"],
        "homepage": "https://github.com/nservant/HiC-Pro",
        "citation": "Servant et al., 2015, Genome Biology"
    },
    
    # Long-read Tools
    "racon": {
        "full_name": "Racon",
        "description": "Ultrafast consensus module for raw de novo genome assembly. Uses raw sequences and overlaps to polish assemblies.",
        "category": "polishing",
        "use_cases": ["Assembly polishing", "Consensus generation", "Error correction"],
        "input_formats": ["FASTA", "FASTQ"],
        "output_formats": ["FASTA"],
        "homepage": "https://github.com/isovic/racon",
        "citation": "Vaser et al., 2017, Genome Research"
    },
    "medaka": {
        "full_name": "Medaka",
        "description": "Neural network-based sequence polishing for Oxford Nanopore data. Creates consensus sequences with high accuracy.",
        "category": "polishing",
        "use_cases": ["ONT assembly polishing", "Consensus calling", "Variant calling"],
        "input_formats": ["FASTQ", "BAM"],
        "output_formats": ["FASTA", "VCF"],
        "homepage": "https://github.com/nanoporetech/medaka",
        "citation": "Oxford Nanopore Technologies"
    },
    "canu": {
        "full_name": "Canu",
        "description": "Long-read assembler designed for high-noise sequences from PacBio and Oxford Nanopore. Includes overlap detection, error correction, and assembly.",
        "category": "assembly",
        "use_cases": ["Long-read assembly", "Error correction", "Large genomes"],
        "input_formats": ["FASTQ", "FASTA"],
        "output_formats": ["FASTA", "GFA"],
        "homepage": "https://github.com/marbl/canu",
        "citation": "Koren et al., 2017, Genome Research"
    },
    
    # Annotation Tools
    "prokka": {
        "full_name": "Prokka",
        "description": "Rapid prokaryotic genome annotation. Identifies CDS, rRNA, tRNA, and other features with functional annotation.",
        "category": "annotation",
        "use_cases": ["Bacterial genome annotation", "Gene prediction", "Functional annotation"],
        "input_formats": ["FASTA"],
        "output_formats": ["GFF", "GenBank", "FASTA"],
        "homepage": "https://github.com/tseemann/prokka",
        "citation": "Seemann, 2014, Bioinformatics"
    },
}


class ToolCatalogIndexer:
    """
    Indexes bioinformatics tools into the knowledge base.
    
    Provides comprehensive tool information for RAG-enhanced
    workflow generation and education.
    """
    
    def __init__(self, kb: KnowledgeBase, project_root: Path = None):
        """
        Initialize indexer.
        
        Args:
            kb: Knowledge base instance
            project_root: Path to BioPipelines project root
        """
        self.kb = kb
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent.parent
        
    def index_all(self) -> Dict[str, int]:
        """
        Index all tool sources.
        
        Returns:
            Dictionary of source -> count indexed
        """
        results = {}
        
        # Index built-in descriptions
        results["tool_descriptions"] = self._index_tool_descriptions()
        
        # Index tool mappings from config
        results["tool_mappings"] = self._index_tool_mappings()
        
        # Index analysis definitions
        results["analysis_definitions"] = self._index_analysis_definitions()
        
        # Index error patterns
        results["error_patterns"] = self._index_error_patterns()
        
        logger.info(f"Indexed tools: {results}")
        return results
    
    def _index_tool_descriptions(self) -> int:
        """Index built-in tool descriptions."""
        count = 0
        
        for tool_name, info in TOOL_DESCRIPTIONS.items():
            content = self._format_tool_content(tool_name, info)
            
            doc = KnowledgeDocument(
                id=f"tool_{tool_name}",
                source=KnowledgeSource.TOOL_CATALOG,
                title=info.get("full_name", tool_name),
                content=content,
                metadata={
                    "tool_name": tool_name,
                    "category": info.get("category"),
                    "use_cases": info.get("use_cases", []),
                    "homepage": info.get("homepage"),
                },
            )
            
            self.kb.add_document(doc)
            count += 1
        
        logger.info(f"Indexed {count} tool descriptions")
        return count
    
    def _format_tool_content(self, name: str, info: Dict) -> str:
        """Format tool info into searchable content."""
        parts = [
            f"Tool: {info.get('full_name', name)}",
            f"Short name: {name}",
            f"Description: {info.get('description', '')}",
            f"Category: {info.get('category', 'unknown')}",
        ]
        
        if info.get("use_cases"):
            parts.append(f"Use cases: {', '.join(info['use_cases'])}")
        
        if info.get("input_formats"):
            parts.append(f"Input formats: {', '.join(info['input_formats'])}")
        
        if info.get("output_formats"):
            parts.append(f"Output formats: {', '.join(info['output_formats'])}")
        
        if info.get("homepage"):
            parts.append(f"Homepage: {info['homepage']}")
        
        if info.get("citation"):
            parts.append(f"Citation: {info['citation']}")
        
        return "\n".join(parts)
    
    def _index_tool_mappings(self) -> int:
        """Index tool mappings from config/tool_mappings.yaml."""
        count = 0
        
        mappings_file = self.project_root / "config" / "tool_mappings.yaml"
        if not mappings_file.exists():
            logger.warning(f"Tool mappings not found: {mappings_file}")
            return 0
        
        try:
            with open(mappings_file) as f:
                data = yaml.safe_load(f)
            
            tools = data.get("tools", {})
            
            for tool_name, tool_info in tools.items():
                # Skip if already indexed with full description
                if tool_name in TOOL_DESCRIPTIONS:
                    continue
                
                content = self._format_mapping_content(tool_name, tool_info)
                
                doc = KnowledgeDocument(
                    id=f"mapping_{tool_name}",
                    source=KnowledgeSource.TOOL_CATALOG,
                    title=tool_name.upper(),
                    content=content,
                    metadata={
                        "tool_name": tool_name,
                        "category": tool_info.get("category"),
                        "container": tool_info.get("container"),
                        "processes": tool_info.get("processes", []),
                    },
                )
                
                self.kb.add_document(doc)
                count += 1
        
        except Exception as e:
            logger.error(f"Failed to index tool mappings: {e}")
        
        logger.info(f"Indexed {count} tool mappings")
        return count
    
    def _format_mapping_content(self, name: str, info: Dict) -> str:
        """Format tool mapping into content."""
        parts = [
            f"Tool: {name.upper()}",
            f"Category: {info.get('category', 'unknown')}",
            f"Container: {info.get('container', 'unknown')}",
        ]
        
        if info.get("module"):
            parts.append(f"Nextflow module: {info['module']}")
        
        if info.get("processes"):
            parts.append(f"Processes: {', '.join(info['processes'])}")
        
        return "\n".join(parts)
    
    def _index_analysis_definitions(self) -> int:
        """Index analysis type definitions."""
        count = 0
        
        analysis_file = self.project_root / "config" / "analysis_definitions.yaml"
        if not analysis_file.exists():
            logger.warning(f"Analysis definitions not found: {analysis_file}")
            return 0
        
        try:
            with open(analysis_file) as f:
                data = yaml.safe_load(f)
            
            for analysis_type, definition in data.items():
                content = self._format_analysis_content(analysis_type, definition)
                
                doc = KnowledgeDocument(
                    id=f"analysis_{analysis_type}",
                    source=KnowledgeSource.BEST_PRACTICES,
                    title=f"Analysis: {analysis_type.replace('_', ' ').title()}",
                    content=content,
                    metadata={
                        "analysis_type": analysis_type,
                        "required_tools": definition.get("required", []),
                        "recommended_tools": definition.get("recommended", []),
                    },
                )
                
                self.kb.add_document(doc)
                count += 1
        
        except Exception as e:
            logger.error(f"Failed to index analysis definitions: {e}")
        
        logger.info(f"Indexed {count} analysis definitions")
        return count
    
    def _format_analysis_content(self, name: str, definition: Dict) -> str:
        """Format analysis definition into content."""
        parts = [
            f"Analysis Type: {name.replace('_', ' ').title()}",
            f"ID: {name}",
        ]
        
        if definition.get("required"):
            parts.append(f"Required tools: {', '.join(definition['required'])}")
        
        if definition.get("recommended"):
            parts.append(f"Recommended tools: {', '.join(definition['recommended'])}")
        
        # Add all other tool categories
        for key, value in definition.items():
            if key not in ("required", "recommended") and isinstance(value, list):
                parts.append(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
        
        return "\n".join(parts)
    
    def _index_error_patterns(self) -> int:
        """Index error patterns for troubleshooting."""
        count = 0
        
        error_file = self.project_root / "config" / "error_patterns.yaml"
        if not error_file.exists():
            logger.warning(f"Error patterns not found: {error_file}")
            return 0
        
        try:
            with open(error_file) as f:
                data = yaml.safe_load(f)
            
            patterns = data.get("patterns", [])
            if isinstance(patterns, list):
                for i, pattern in enumerate(patterns):
                    if isinstance(pattern, dict):
                        doc = KnowledgeDocument(
                            id=f"error_{i}_{hashlib.md5(str(pattern).encode()).hexdigest()[:8]}",
                            source=KnowledgeSource.ERROR_PATTERNS,
                            title=pattern.get("name", f"Error Pattern {i+1}"),
                            content=self._format_error_content(pattern),
                            metadata={
                                "tools": pattern.get("tools", []),
                                "severity": pattern.get("severity"),
                            },
                        )
                        self.kb.add_document(doc)
                        count += 1
        
        except Exception as e:
            logger.error(f"Failed to index error patterns: {e}")
        
        logger.info(f"Indexed {count} error patterns")
        return count
    
    def _format_error_content(self, pattern: Dict) -> str:
        """Format error pattern into content."""
        parts = [
            f"Error: {pattern.get('name', 'Unknown')}",
        ]
        
        if pattern.get("pattern"):
            parts.append(f"Pattern: {pattern['pattern']}")
        
        if pattern.get("cause"):
            parts.append(f"Cause: {pattern['cause']}")
        
        if pattern.get("solution"):
            parts.append(f"Solution: {pattern['solution']}")
        
        if pattern.get("tools"):
            parts.append(f"Related tools: {', '.join(pattern['tools'])}")
        
        return "\n".join(parts)


def index_tool_catalog(kb: KnowledgeBase = None, project_root: Path = None) -> Dict[str, int]:
    """
    Convenience function to index the tool catalog.
    
    Args:
        kb: Knowledge base (creates new if None)
        project_root: Project root path
        
    Returns:
        Dictionary of source -> count indexed
    """
    if kb is None:
        kb = KnowledgeBase()
    
    indexer = ToolCatalogIndexer(kb, project_root)
    return indexer.index_all()


if __name__ == "__main__":
    # Run indexing when called directly
    import logging
    logging.basicConfig(level=logging.INFO)
    
    results = index_tool_catalog()
    print(f"\nIndexing complete: {results}")
    print(f"Total indexed: {sum(results.values())}")
