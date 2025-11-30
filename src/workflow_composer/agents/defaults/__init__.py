"""
Smart Defaults + Transparency System.

This module provides intelligent default resolution for missing parameters
instead of asking clarifying questions. Professional chat agents (ChatGPT,
Claude, Copilot) rarely ask clarifying questions - they proceed with
reasonable assumptions and state them explicitly, letting users course-correct.

Philosophy:
1. Make reasonable assumptions based on context and domain knowledge
2. State assumptions explicitly in the response
3. Offer alternatives if relevant (but don't ask)
4. Let user iterate naturally

Example:
    User: "run RNA-seq analysis"
    Bad: "Which organism? Which analysis type? Which aligner?"
    Good: "Setting up RNA-seq differential expression for human (GRCh38) 
           using STAR aligner. Let me know if you need different settings."

Author: BioPipelines Team
Date: November 2025
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class DefaultConfig:
    """Configuration for smart defaults."""
    
    # Default organism (most common in bioinformatics)
    default_organism: str = "human"
    
    # Organism aliases for normalization
    organism_aliases: Dict[str, str] = field(default_factory=lambda: {
        # Human
        "homo sapiens": "human", "hs": "human", "hg38": "human", 
        "grch38": "human", "hg19": "human", "grch37": "human",
        "h. sapiens": "human",
        # Mouse
        "mus musculus": "mouse", "mm": "mouse", "mm10": "mouse",
        "mm9": "mouse", "m. musculus": "mouse", "murine": "mouse",
        # Fly
        "drosophila": "fly", "drosophila melanogaster": "fly",
        "dm6": "fly", "dm3": "fly", "d. melanogaster": "fly",
        # Worm
        "c. elegans": "worm", "caenorhabditis elegans": "worm",
        "ce11": "worm", "ce10": "worm",
        # Zebrafish
        "danio rerio": "zebrafish", "danrer11": "zebrafish",
        "danrer10": "zebrafish", "d. rerio": "zebrafish",
        # Yeast
        "saccharomyces cerevisiae": "yeast", "s. cerevisiae": "yeast",
        "saccer3": "yeast", "sc": "yeast",
        # Rat
        "rattus norvegicus": "rat", "rn6": "rat", "r. norvegicus": "rat",
        # Arabidopsis
        "arabidopsis thaliana": "arabidopsis", "a. thaliana": "arabidopsis",
        "tair10": "arabidopsis",
    })
    
    # Default analysis types per workflow
    workflow_defaults: Dict[str, str] = field(default_factory=lambda: {
        # RNA-seq variants
        "rna-seq": "differential_expression",
        "rnaseq": "differential_expression",
        "rna_seq": "differential_expression",
        "transcriptomics": "differential_expression",
        "expression": "differential_expression",
        # ChIP-seq
        "chip-seq": "peak_calling",
        "chipseq": "peak_calling",
        "chip_seq": "peak_calling",
        # ATAC-seq
        "atac-seq": "peak_calling",
        "atacseq": "peak_calling",
        "atac_seq": "peak_calling",
        # DNA-seq variants
        "dna-seq": "variant_calling",
        "dnaseq": "variant_calling",
        "wgs": "variant_calling",
        "wes": "variant_calling",
        "exome": "variant_calling",
        "genome": "variant_calling",
        # Methylation
        "methylation": "differential_methylation",
        "bisulfite": "differential_methylation",
        "wgbs": "differential_methylation",
        "rrbs": "differential_methylation",
        # Single-cell
        "scrna-seq": "clustering",
        "scrnaseq": "clustering",
        "single-cell": "clustering",
        "10x": "clustering",
        # Hi-C
        "hic": "contact_matrix",
        "hi-c": "contact_matrix",
        # Metagenomics
        "metagenomics": "taxonomic_profiling",
        "16s": "taxonomic_profiling",
        "amplicon": "taxonomic_profiling",
    })
    
    # Default aligners per workflow
    workflow_aligners: Dict[str, str] = field(default_factory=lambda: {
        "rna-seq": "STAR",
        "rnaseq": "STAR",
        "chip-seq": "BWA",
        "chipseq": "BWA",
        "atac-seq": "Bowtie2",
        "atacseq": "Bowtie2",
        "dna-seq": "BWA-MEM2",
        "dnaseq": "BWA-MEM2",
        "wgs": "BWA-MEM2",
        "wes": "BWA-MEM2",
        "methylation": "Bismark",
        "bisulfite": "Bismark",
    })
    
    # Genome versions per organism
    genome_versions: Dict[str, str] = field(default_factory=lambda: {
        "human": "GRCh38",
        "mouse": "mm10",
        "fly": "dm6",
        "worm": "ce11",
        "zebrafish": "danRer11",
        "yeast": "sacCer3",
        "rat": "rn6",
        "arabidopsis": "TAIR10",
    })
    
    # Default quality settings
    default_quality_threshold: int = 20
    default_min_read_length: int = 50
    default_threads: int = 8


@dataclass
class ResolvedDefaults:
    """Result of smart default resolution."""
    filled_params: Dict[str, Any]
    assumptions: List[str]
    explanation: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "filled_params": self.filled_params,
            "assumptions": self.assumptions,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


# Import the resolver
from .resolver import SmartDefaultResolver

__all__ = [
    "DefaultConfig",
    "ResolvedDefaults", 
    "SmartDefaultResolver",
]
