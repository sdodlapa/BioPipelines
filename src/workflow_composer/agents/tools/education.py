"""
Education Tools
===============

Tools for explaining concepts, comparing samples, and teaching bioinformatics.

Enhanced with RAG-based knowledge base lookup for comprehensive responses.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import ToolResult

# Import RAG knowledge base for enhanced responses
try:
    from ..rag import KnowledgeBase, KnowledgeSource
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    KnowledgeBase = None
    KnowledgeSource = None

logger = logging.getLogger(__name__)

# Global knowledge base instance (lazy loaded)
_knowledge_base: Optional["KnowledgeBase"] = None


def get_knowledge_base() -> Optional["KnowledgeBase"]:
    """Get or initialize the knowledge base singleton."""
    global _knowledge_base
    if not RAG_AVAILABLE:
        return None
    if _knowledge_base is None:
        try:
            _knowledge_base = KnowledgeBase()
            logger.info("Knowledge base initialized for education tools")
        except Exception as e:
            logger.warning(f"Could not initialize knowledge base: {e}")
            return None
    return _knowledge_base


def search_knowledge_base(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for relevant information.
    
    Args:
        query: Search query
        limit: Maximum results to return
        
    Returns:
        List of relevant knowledge documents
    """
    kb = get_knowledge_base()
    if kb is None:
        return []
    
    try:
        # Search across all relevant sources
        results = kb.search(
            query=query,
            sources=[
                KnowledgeSource.TOOL_CATALOG,
                KnowledgeSource.NF_CORE_MODULES,
                KnowledgeSource.BEST_PRACTICES,
            ],
            limit=limit
        )
        return [
            {
                "title": doc.title,
                "content": doc.content,
                "source": doc.source.value if hasattr(doc.source, 'value') else str(doc.source),
                "relevance": getattr(doc, 'relevance_score', 0.5)
            }
            for doc in results
        ]
    except Exception as e:
        logger.warning(f"Knowledge base search failed: {e}")
        return []


# =============================================================================
# EXPLAIN_CONCEPT
# =============================================================================

EXPLAIN_CONCEPT_PATTERNS = [
    r"(?:what is|explain|describe|tell me about|how does)\s+(.+?)(?:\s+work|\s+mean|\?|$)",
    r"(?:help me understand|teach me about|define)\s+(.+)",
]


# Knowledge base for bioinformatics concepts
CONCEPT_KNOWLEDGE = {
    "fastqc": {
        "title": "FastQC",
        "category": "Quality Control",
        "description": """FastQC is a quality control tool for high throughput sequence data. 
It provides a modular set of analyses to check raw sequence data for problems before downstream analysis.""",
        "key_points": [
            "Checks per-base sequence quality",
            "Identifies adapter contamination",
            "Detects overrepresented sequences",
            "Measures GC content distribution",
            "Reports sequence duplication levels"
        ],
        "when_to_use": "Always run FastQC on raw FASTQ files before starting any analysis pipeline.",
        "related": ["MultiQC", "Trimmomatic", "fastp"]
    },
    
    "bwa": {
        "title": "BWA (Burrows-Wheeler Aligner)",
        "category": "Alignment",
        "description": """BWA is a software package for mapping low-divergent sequences against a large reference genome.
It's widely used for DNA sequencing alignment.""",
        "key_points": [
            "BWA-MEM is best for reads >70bp",
            "Creates SAM/BAM alignment files",
            "Requires indexed reference genome",
            "Supports paired-end reads",
            "Handles chimeric alignments (split reads)"
        ],
        "when_to_use": "Use BWA for DNA-seq alignment, especially for WGS, WES, and panel sequencing.",
        "related": ["Bowtie2", "HISAT2", "minimap2"]
    },
    
    "star": {
        "title": "STAR (Spliced Transcripts Alignment to a Reference)",
        "category": "Alignment",
        "description": """STAR is an ultrafast RNA-seq aligner that handles spliced alignments.
It's the recommended aligner for RNA sequencing data.""",
        "key_points": [
            "Handles splice junctions automatically",
            "Very fast but memory-intensive",
            "Can output gene counts directly",
            "Supports 2-pass alignment for novel junctions",
            "Creates chimeric output for fusion detection"
        ],
        "when_to_use": "Use STAR for all RNA-seq alignment tasks.",
        "related": ["HISAT2", "Salmon", "kallisto"]
    },
    
    "gatk": {
        "title": "GATK (Genome Analysis Toolkit)",
        "category": "Variant Calling",
        "description": """GATK is a collection of tools for variant discovery and genotyping.
Developed by the Broad Institute, it's the gold standard for germline variant calling.""",
        "key_points": [
            "HaplotypeCaller for germline variants",
            "Mutect2 for somatic variants",
            "Includes BQSR for base quality recalibration",
            "VQSR for variant quality score recalibration",
            "Best Practices workflows available"
        ],
        "when_to_use": "Use GATK for high-confidence variant calling from DNA sequencing.",
        "related": ["bcftools", "DeepVariant", "Strelka2"]
    },
    
    "deseq2": {
        "title": "DESeq2",
        "category": "Differential Expression",
        "description": """DESeq2 is an R package for differential gene expression analysis.
It uses negative binomial distribution to model count data.""",
        "key_points": [
            "Handles low-count genes appropriately",
            "Automatic normalization (size factors)",
            "Shrinkage estimation for fold changes",
            "Multiple testing correction included",
            "Works with raw counts (not FPKM/TPM)"
        ],
        "when_to_use": "Use DESeq2 for RNA-seq differential expression between conditions.",
        "related": ["edgeR", "limma-voom", "sleuth"]
    },
    
    "macs2": {
        "title": "MACS2 (Model-based Analysis of ChIP-Seq)",
        "category": "Peak Calling",
        "description": """MACS2 identifies transcription factor binding sites or histone modification 
regions from ChIP-seq and ATAC-seq data.""",
        "key_points": [
            "Models shift size from data",
            "Supports narrow and broad peak calling",
            "Handles paired-end data",
            "Outputs BED/narrowPeak format",
            "Calculates FDR for peaks"
        ],
        "when_to_use": "Use MACS2 for ChIP-seq and ATAC-seq peak calling.",
        "related": ["HOMER", "SEACR", "Genrich"]
    },
    
    "salmon": {
        "title": "Salmon",
        "category": "Quantification",
        "description": """Salmon is a fast transcript-level quantification tool.
It uses quasi-mapping for speed and bias correction for accuracy.""",
        "key_points": [
            "Alignment-free quantification",
            "Very fast (minutes for typical samples)",
            "Built-in GC and sequence bias correction",
            "Outputs TPM and counts",
            "Can be used with tximport for gene-level analysis"
        ],
        "when_to_use": "Use Salmon when you need fast transcript quantification without full alignment.",
        "related": ["kallisto", "RSEM", "featureCounts"]
    },
    
    "vcf": {
        "title": "VCF (Variant Call Format)",
        "category": "File Formats",
        "description": """VCF is a text file format for storing genetic variation data.
It's the standard format for variant calls.""",
        "key_points": [
            "Header section with metadata",
            "One variant per line",
            "CHROM, POS, ID, REF, ALT columns required",
            "QUAL and FILTER for quality info",
            "INFO and FORMAT for annotations",
            "Can be compressed with bgzip"
        ],
        "when_to_use": "VCF files are output by all variant callers and input to annotation tools.",
        "related": ["BCF", "gVCF", "MAF"]
    },
    
    "bam": {
        "title": "BAM (Binary Alignment Map)",
        "category": "File Formats",
        "description": """BAM is the binary, compressed version of SAM format for storing aligned sequences.
It's indexed for fast random access.""",
        "key_points": [
            "Binary compressed format",
            "Requires .bai index for random access",
            "Contains alignment information",
            "Can be viewed with samtools",
            "CRAM is a more compressed alternative"
        ],
        "when_to_use": "BAM files are the standard output from alignment tools.",
        "related": ["SAM", "CRAM", "FASTQ"]
    },
    
    "fpkm": {
        "title": "FPKM/TPM (Expression Units)",
        "category": "Quantification",
        "description": """FPKM and TPM are normalized expression units that account for gene length and sequencing depth.""",
        "key_points": [
            "FPKM: Fragments Per Kilobase per Million mapped",
            "TPM: Transcripts Per Million",
            "TPM is preferred (sums to 1M per sample)",
            "Both account for gene length",
            "Don't use for differential expression (use raw counts)"
        ],
        "when_to_use": "Use TPM/FPKM for comparing expression levels across genes within a sample.",
        "related": ["counts", "CPM", "RPKM"]
    },
    
    "chip-seq": {
        "title": "ChIP-seq (Chromatin Immunoprecipitation Sequencing)",
        "category": "Assay Types",
        "description": """ChIP-seq identifies DNA binding sites of proteins (transcription factors, histones).
Uses antibodies to pull down protein-DNA complexes.""",
        "key_points": [
            "Requires antibody specific to target protein",
            "Control sample is important (input or IgG)",
            "Narrow peaks for TFs, broad peaks for histones",
            "Typically 10-50M reads per sample",
            "Peak calling identifies enriched regions"
        ],
        "when_to_use": "Use ChIP-seq to map protein-DNA interactions genome-wide.",
        "related": ["ATAC-seq", "CUT&RUN", "CUT&TAG"]
    },
    
    "atac-seq": {
        "title": "ATAC-seq (Assay for Transposase-Accessible Chromatin)",
        "category": "Assay Types",
        "description": """ATAC-seq identifies open chromatin regions using Tn5 transposase.
Open chromatin indicates regulatory regions and active transcription.""",
        "key_points": [
            "Requires fewer cells than ChIP-seq",
            "No antibody needed",
            "Fragment size distribution shows nucleosome pattern",
            "Typically 50-100M reads per sample",
            "Often combined with scATAC for single-cell"
        ],
        "when_to_use": "Use ATAC-seq to profile chromatin accessibility genome-wide.",
        "related": ["ChIP-seq", "DNase-seq", "MNase-seq"]
    },
    
    # Analysis Types
    "rna-seq": {
        "title": "RNA-seq (RNA Sequencing)",
        "category": "Analysis Types",
        "description": """RNA-seq is a sequencing technique to reveal the presence and quantity 
of RNA in a biological sample at a given moment. It provides a snapshot of the transcriptome.""",
        "key_points": [
            "Quantifies gene expression levels",
            "Can detect novel transcripts and splice variants",
            "Typically 20-50M reads per sample for differential expression",
            "Strand-specific protocols preserve direction information",
            "Requires careful normalization for comparisons",
            "Can be used for single-cell (scRNA-seq) or bulk analysis"
        ],
        "when_to_use": "Use RNA-seq to study gene expression, identify differentially expressed genes, and discover novel transcripts.",
        "related": ["DESeq2", "edgeR", "STAR", "Salmon", "RSEM", "scRNA-seq"]
    },
    
    "scrna-seq": {
        "title": "scRNA-seq (Single-cell RNA Sequencing)",
        "category": "Analysis Types",
        "description": """Single-cell RNA sequencing profiles gene expression at the 
individual cell level, revealing cellular heterogeneity.""",
        "key_points": [
            "Reveals cell-to-cell variation",
            "Identifies cell types and states",
            "Common platforms: 10x Genomics, Drop-seq, Smart-seq2",
            "Requires specialized normalization methods",
            "Downstream: clustering, trajectory analysis, cell type annotation",
            "Typically millions of cells per experiment"
        ],
        "when_to_use": "Use scRNA-seq to study cellular heterogeneity, identify rare cell types, and track cell differentiation.",
        "related": ["Seurat", "Scanpy", "CellRanger", "RNA-seq"]
    },
    
    "wgs": {
        "title": "WGS (Whole Genome Sequencing)",
        "category": "Analysis Types",
        "description": """Whole genome sequencing determines the complete DNA sequence of an 
organism's genome. It captures all variants including SNPs, indels, and structural variants.""",
        "key_points": [
            "Sequences entire genome (coding + non-coding)",
            "Typically 30-50x coverage for variant calling",
            "Can detect structural variants and CNVs",
            "More expensive than WES but more comprehensive",
            "Reference genome required for alignment",
            "Detects germline and somatic variants"
        ],
        "when_to_use": "Use WGS for comprehensive variant analysis, structural variant detection, and de novo assembly.",
        "related": ["WES", "GATK", "BWA", "VCF", "Delly", "Manta"]
    },
    
    "wes": {
        "title": "WES (Whole Exome Sequencing)",
        "category": "Analysis Types",
        "description": """Whole exome sequencing targets only the protein-coding regions 
(exons) of the genome, which are about 1-2% of the total genome.""",
        "key_points": [
            "Cost-effective alternative to WGS",
            "Focuses on coding regions (exons)",
            "Captures 85% of disease-causing variants",
            "100-200x coverage typical",
            "Requires capture kit (Agilent, IDT, Twist)",
            "Misses regulatory and intronic variants"
        ],
        "when_to_use": "Use WES for clinical diagnostics and finding coding variants cost-effectively.",
        "related": ["WGS", "GATK", "BWA", "VCF"]
    },
    
    "methylation": {
        "title": "Methylation Analysis",
        "category": "Analysis Types",
        "description": """DNA methylation analysis studies epigenetic modifications, specifically 
the addition of methyl groups to DNA, typically at CpG sites.""",
        "key_points": [
            "Bisulfite sequencing (WGBS, RRBS) or methylation arrays (450K, EPIC)",
            "Detects 5-methylcytosine (5mC)",
            "Important for gene regulation and imprinting",
            "Bisulfite conversion converts unmethylated C to T",
            "Requires specialized aligners (Bismark, BSseeker)",
            "DMR (Differentially Methylated Region) analysis common"
        ],
        "when_to_use": "Use methylation analysis to study epigenetic regulation, cancer biomarkers, and developmental processes.",
        "related": ["Bismark", "WGBS", "RRBS", "methylKit", "DMR"]
    },
    
    "metagenomics": {
        "title": "Metagenomics",
        "category": "Analysis Types",
        "description": """Metagenomics studies genetic material recovered directly from 
environmental samples, analyzing microbial communities without cultivation.""",
        "key_points": [
            "16S rRNA for bacterial identification (amplicon)",
            "Shotgun metagenomics for functional profiling",
            "Reveals microbial diversity and abundance",
            "Common in microbiome studies",
            "Uses specialized databases (SILVA, Greengenes, MetaPhlAn)",
            "Can perform taxonomic and functional profiling"
        ],
        "when_to_use": "Use metagenomics to study microbial communities in gut, soil, water, and other environments.",
        "related": ["16S rRNA", "QIIME2", "Kraken2", "MetaPhlAn", "HUMAnN"]
    },
    
    "hic": {
        "title": "Hi-C (Chromosome Conformation Capture)",
        "category": "Analysis Types",
        "description": """Hi-C is a method to study the 3D organization of chromatin by 
capturing proximity between DNA regions that are close in 3D space.""",
        "key_points": [
            "Maps genome-wide chromatin interactions",
            "Identifies TADs (Topologically Associating Domains)",
            "Reveals A/B compartments",
            "Requires very deep sequencing (billions of reads)",
            "Used for scaffolding genome assemblies",
            "Specialized tools: HiC-Pro, Juicer, cooler"
        ],
        "when_to_use": "Use Hi-C to study 3D genome organization, chromatin looping, and genome scaffolding.",
        "related": ["TADs", "compartments", "Juicer", "HiC-Pro", "cooler"]
    },
    
    # Additional tools
    "trimmomatic": {
        "title": "Trimmomatic",
        "category": "Preprocessing",
        "description": """Trimmomatic is a flexible read trimming tool for Illumina sequence data. 
It removes adapters and low-quality bases.""",
        "key_points": [
            "Removes adapter sequences",
            "Trims low-quality ends (LEADING, TRAILING)",
            "Sliding window quality filtering",
            "Minimum length filtering",
            "Can process paired-end reads",
            "Multithreaded for performance"
        ],
        "when_to_use": "Use Trimmomatic to clean raw FASTQ files before alignment.",
        "related": ["FastQC", "Cutadapt", "fastp", "BBDuk"]
    },
    
    "samtools": {
        "title": "SAMtools",
        "category": "File Manipulation",
        "description": """SAMtools is a suite of programs for interacting with SAM/BAM/CRAM files. 
It's essential for most sequencing analysis pipelines.""",
        "key_points": [
            "View, sort, index BAM files",
            "Filter reads by mapping quality, flags",
            "Merge and split BAM files",
            "Calculate statistics (flagstat, idxstats)",
            "Pileup for variant calling",
            "Part of the htslib ecosystem"
        ],
        "when_to_use": "Use SAMtools for any manipulation of aligned read files.",
        "related": ["BAM", "SAM", "CRAM", "BCFtools", "htslib"]
    },
    
    "bcftools": {
        "title": "BCFtools",
        "category": "Variant Analysis",
        "description": """BCFtools is a set of utilities for variant calling and manipulating 
VCF/BCF files. It's part of the htslib ecosystem.""",
        "key_points": [
            "Variant calling with mpileup",
            "Filter, merge, annotate VCFs",
            "Query variants with expressions",
            "Statistics and QC metrics",
            "Consensus sequence generation",
            "Compare variant sets (isec)"
        ],
        "when_to_use": "Use BCFtools for variant calling and VCF file manipulation.",
        "related": ["VCF", "BCF", "SAMtools", "GATK", "VEP"]
    },
    
    "hisat2": {
        "title": "HISAT2",
        "category": "Alignment",
        "description": """HISAT2 is a fast and sensitive alignment program for mapping 
RNA-seq reads to a reference genome, with splice-aware alignment.""",
        "key_points": [
            "Fast splice-aware aligner",
            "Uses graph-based index",
            "Memory-efficient (8GB for human)",
            "Successor to TopHat2",
            "Good for general RNA-seq alignment",
            "Can use known splice sites"
        ],
        "when_to_use": "Use HISAT2 as a fast alternative to STAR for RNA-seq alignment.",
        "related": ["STAR", "RNA-seq", "StringTie", "TopHat2"]
    },
    
    "kallisto": {
        "title": "Kallisto",
        "category": "Quantification",
        "description": """Kallisto is an ultra-fast program for quantifying transcript abundances 
using pseudoalignment. It doesn't produce traditional alignments.""",
        "key_points": [
            "Pseudoalignment (not real alignment)",
            "Extremely fast (minutes for RNA-seq)",
            "Transcriptome reference only",
            "Bootstrap for uncertainty estimates",
            "Output compatible with sleuth",
            "Low memory requirements"
        ],
        "when_to_use": "Use Kallisto when you need fast transcript quantification without alignment.",
        "related": ["Salmon", "Sleuth", "tximport", "RNA-seq"]
    },
    
    "featurecounts": {
        "title": "featureCounts",
        "category": "Quantification",
        "description": """featureCounts is a highly efficient read summarization program 
that counts mapped reads for genomic features (genes, exons).""",
        "key_points": [
            "Part of Subread package",
            "Very fast and memory efficient",
            "Handles paired-end and multimapping",
            "Counts reads per gene/exon/transcript",
            "Requires BAM + GTF annotation",
            "Multiple assignment modes"
        ],
        "when_to_use": "Use featureCounts to get gene counts from aligned BAM files.",
        "related": ["HTSeq", "STAR", "Salmon", "DESeq2"]
    },
    
    "edger": {
        "title": "edgeR",
        "category": "Differential Expression",
        "description": """edgeR is an R package for differential expression analysis of 
RNA-seq count data using negative binomial distributions.""",
        "key_points": [
            "Empirical Bayes approach",
            "Handles low counts well",
            "TMM normalization",
            "Exact test and GLM methods",
            "Works with small sample sizes",
            "Similar to DESeq2 in approach"
        ],
        "when_to_use": "Use edgeR as an alternative to DESeq2 for differential expression.",
        "related": ["DESeq2", "limma", "RNA-seq", "counts"]
    },
    
    "multiqc": {
        "title": "MultiQC",
        "category": "Quality Control",
        "description": """MultiQC aggregates results from multiple tools into a single 
HTML report, making QC review efficient.""",
        "key_points": [
            "Supports 100+ bioinformatics tools",
            "Generates interactive HTML reports",
            "Combines QC metrics across samples",
            "Configurable and extensible",
            "Shows trends across samples",
            "Essential for batch QC review"
        ],
        "when_to_use": "Use MultiQC at the end of your pipeline to summarize all QC metrics.",
        "related": ["FastQC", "Qualimap", "Picard", "SAMtools"]
    },
}


def explain_concept_impl(concept: str) -> ToolResult:
    """
    Explain a bioinformatics concept.
    
    Uses static knowledge base first, then falls back to RAG-based
    knowledge base search for comprehensive answers.
    
    Args:
        concept: The concept to explain
        
    Returns:
        ToolResult with explanation
    """
    if not concept:
        return ToolResult(
            success=False,
            tool_name="explain_concept",
            error="No concept specified",
            message="""❓ **What would you like to learn about?**

I can explain:
- **Tools**: FastQC, BWA, STAR, GATK, DESeq2, MACS2, Salmon, Trimmomatic, MultiQC
- **File Formats**: VCF, BAM, FASTQ, BED
- **Assays**: ChIP-seq, ATAC-seq, RNA-seq, WGS, WES, scRNA-seq, Hi-C
- **Analysis Types**: Methylation, Metagenomics, Differential Expression
- **Metrics**: FPKM, TPM, FDR, p-value
- **Concepts**: Alignment, Variant calling, Peak calling

Just ask! Example: "what is STAR?" or "explain RNA-seq"
"""
        )
    
    # Normalize concept
    concept_lower = concept.lower().strip()
    # Remove common prefixes/suffixes for matching
    concept_clean = concept_lower.replace("what is ", "").replace("explain ", "").replace("?", "").strip()
    
    # Direct match in static knowledge
    info = None
    if concept_clean in CONCEPT_KNOWLEDGE:
        info = CONCEPT_KNOWLEDGE[concept_clean]
    else:
        # Partial match
        matches = [k for k in CONCEPT_KNOWLEDGE if concept_clean in k or k in concept_clean]
        if matches:
            info = CONCEPT_KNOWLEDGE[matches[0]]
    
    if info:
        # Found in static knowledge - format response
        key_points = "\n".join(f"- {p}" for p in info['key_points'])
        related = ", ".join(info['related'])
        
        message = f"""📚 **{info['title']}**
*Category: {info['category']}*

{info['description']}

**Key Points:**
{key_points}

**When to Use:**
{info['when_to_use']}

**Related Tools/Concepts:**
{related}

---
Need more detail? Ask about any of the related topics!
"""
        return ToolResult(
            success=True,
            tool_name="explain_concept",
            data={"concept": concept, "info": info, "source": "static"},
            message=message
        )
    
    # Not in static knowledge - try RAG knowledge base
    kb_results = search_knowledge_base(concept, limit=5)
    
    if kb_results:
        # Found relevant information in knowledge base
        logger.info(f"Found {len(kb_results)} KB results for '{concept}'")
        
        # Format KB results into a coherent response
        result_sections = []
        sources_seen = set()
        
        for result in kb_results:
            source = result.get('source', 'unknown')
            if source not in sources_seen and len(result_sections) < 3:
                sources_seen.add(source)
                content = result.get('content', '')
                title = result.get('title', concept)
                
                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                result_sections.append(f"### {title}\n{content}")
        
        if result_sections:
            formatted_results = "\n\n".join(result_sections)
            sources_list = ", ".join(sources_seen)
            
            message = f"""📚 **{concept}**
*Found in knowledge base: {sources_list}*

{formatted_results}

---
*Information retrieved from BioPipelines knowledge base. Ask follow-up questions for more details!*
"""
            return ToolResult(
                success=True,
                tool_name="explain_concept",
                data={"concept": concept, "kb_results": kb_results, "source": "knowledge_base"},
                message=message
            )
    
    # Fallback - no match in static or RAG
    # Still provide helpful context with available topics
    available_topics = list(CONCEPT_KNOWLEDGE.keys())
    topic_categories = {
        "Tools": [k for k in available_topics if CONCEPT_KNOWLEDGE[k]["category"] in ["Alignment", "Quality Control", "Variant Calling", "Quantification", "Differential Expression", "Preprocessing", "File Manipulation", "Peak Calling"]],
        "Analysis Types": [k for k in available_topics if CONCEPT_KNOWLEDGE[k]["category"] == "Analysis Types"],
        "File Formats": [k for k in available_topics if CONCEPT_KNOWLEDGE[k]["category"] == "File Formats"],
    }
    
    suggestions = []
    for cat, topics in topic_categories.items():
        if topics:
            suggestions.append(f"- **{cat}**: {', '.join(topics[:5])}")
    
    suggestions_text = "\n".join(suggestions)
    
    return ToolResult(
        success=True,
        tool_name="explain_concept",
        data={"concept": concept, "matched": False, "source": "none"},
        message=f"""🤔 **{concept}**

I don't have detailed information about "{concept}" yet.

**Suggestions:**
1. Check if it's spelled correctly
2. Try a more specific or general term
3. Search the nf-core modules documentation

**Topics I can explain:**
{suggestions_text}

**Tip:** You can also ask me to generate a workflow that uses this tool/technique,
and I'll include best practices in the pipeline code.
"""
    )


# =============================================================================
# COMPARE_SAMPLES
# =============================================================================

COMPARE_SAMPLES_PATTERNS = [
    r"compare\s+(?:samples?|conditions?|groups?)\s*(.+)?",
    r"(?:what(?:'s| is))\s+(?:the\s+)?difference\s+between\s+(.+)",
]


def compare_samples_impl(
    sample1: str = None,
    sample2: str = None,
    comparison_type: str = "general",
) -> ToolResult:
    """
    Compare samples or provide guidance on sample comparison.
    
    Args:
        sample1: First sample/condition
        sample2: Second sample/condition
        comparison_type: Type of comparison (general, expression, variants)
        
    Returns:
        ToolResult with comparison guidance
    """
    if not sample1 or not sample2:
        message = """📊 **Sample Comparison Guide**

**To compare samples, I need:**
1. Two sample names or conditions
2. The type of data (expression, variants, peaks)

**Example queries:**
- "compare treated vs control samples"
- "what's the difference between tumor and normal"
- "compare expression in samples A and B"

**Types of comparisons I can help with:**

### Expression Analysis
- Differential gene expression
- Pathway enrichment
- Gene ontology analysis

### Variant Analysis
- Shared vs unique variants
- Allele frequency differences
- Mutation signatures

### ChIP/ATAC Analysis
- Differential peaks
- Motif enrichment differences
- Chromatin state changes

Tell me what you'd like to compare!
"""
    else:
        message = f"""📊 **Comparing: {sample1} vs {sample2}**

**Recommended Analysis Workflow:**

### 1. Quality Check First
- Ensure both samples passed QC
- Check for batch effects
- Verify comparable sequencing depth

### 2. Expression Comparison (if RNA-seq)
```r
# Using DESeq2
dds <- DESeqDataSetFromMatrix(counts, colData, ~ condition)
dds <- DESeq(dds)
res <- results(dds, contrast=c("condition", "{sample1}", "{sample2}"))
```

### 3. Variant Comparison (if DNA-seq)
```bash
# Using bcftools
bcftools isec -p comparison_results {sample1}.vcf {sample2}.vcf
```

### 4. Key Metrics to Report
- Number of differentially expressed genes (padj < 0.05)
- Log2 fold change distribution
- Shared vs unique features

### 5. Visualization
- MA plots
- Volcano plots
- Heatmaps of top differences

**Would you like me to:**
1. Generate a comparison workflow?
2. Explain any analysis step?
3. Help interpret existing results?
"""
    
    return ToolResult(
        success=True,
        tool_name="compare_samples",
        data={"sample1": sample1, "sample2": sample2},
        message=message
    )


# =============================================================================
# GET_HELP
# =============================================================================

GET_HELP_PATTERNS = [
    r"^(?:help|\\?|what can you do|commands?|capabilities)$",
    r"(?:show|list)\s+(?:all\s+)?(?:commands?|capabilities|features)",
]


def get_help_impl() -> ToolResult:
    """
    Show help information about available commands.
    
    Returns:
        ToolResult with help information
    """
    message = """🤖 **BioPipelines AI Assistant**

I can help you with bioinformatics workflows! Here's what I can do:

---

### 📁 Data Discovery
- "scan my data" - Find data files in workspace
- "search for ENCODE K562 ChIP-seq" - Search public databases
- "search TCGA for lung cancer RNA-seq" - Search cancer genomics data
- "describe my files" - Analyze file contents

### 📥 Data Management
- "download ENCODE dataset ENCSR..." - Download from ENCODE
- "cleanup old data" - Remove unnecessary files
- "validate my dataset" - Check data integrity

### 🔧 Workflow Generation
- "create RNA-seq workflow" - Generate analysis pipeline
- "make ChIP-seq pipeline" - Create peak calling workflow
- "list available workflows" - Show workflow templates
- "check reference data" - Verify genome references

### 🚀 Execution
- "run the workflow" - Submit to SLURM
- "check job status" - Monitor running jobs
- "show logs" - View execution logs
- "cancel job 12345" - Stop a job

### 🔍 Analysis & Diagnostics
- "diagnose this error" - Troubleshoot problems
- "analyze my results" - Interpret output files
- "compare samples A vs B" - Comparison guidance

### 📚 Education
- "what is FastQC?" - Explain tools and concepts
- "how does BWA work?" - Learn about algorithms
- "explain ChIP-seq" - Understand assay types

---

**Tips:**
- Be specific about what you need
- I understand natural language
- Ask follow-up questions!

**Example session:**
```
You: scan my data
You: create RNA-seq workflow for the fastq files
You: run it on SLURM
You: check status
```
"""
    
    return ToolResult(
        success=True,
        tool_name="get_help",
        data={},
        message=message
    )
