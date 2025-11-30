"""
Negation and Comparative Handling for Query Parsing.

This module detects and handles negation patterns in user queries, which are
a significant source of parsing errors. Examples:
- "RNA-seq but not mouse" → should exclude mouse
- "use STAR aligner, not HISAT2" → prefer STAR, exclude HISAT2
- "all samples except controls" → exclude controls

Author: BioPipelines Team
Date: November 2025
"""
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class NegationType(Enum):
    """Types of negation patterns."""
    NONE = "none"
    EXCLUSION = "exclusion"      # "not X", "without X", "except X"
    PREFERENCE = "preference"    # "X instead of Y", "prefer X over Y"  
    CORRECTION = "correction"    # "not X but Y", "Y not X"
    AVOIDANCE = "avoidance"      # "don't use X", "avoid X", "skip X"


@dataclass
class NegationResult:
    """Result of negation detection."""
    has_negation: bool
    negation_type: NegationType
    negated_terms: List[str] = field(default_factory=list)
    preferred_terms: List[str] = field(default_factory=list)
    original_query: str = ""
    transformed_query: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "has_negation": self.has_negation,
            "negation_type": self.negation_type.value,
            "negated_terms": self.negated_terms,
            "preferred_terms": self.preferred_terms,
            "original_query": self.original_query,
            "transformed_query": self.transformed_query,
            "confidence": self.confidence,
        }


class NegationHandler:
    """
    Detects and handles negation patterns in queries.
    
    This handler identifies various negation patterns commonly found in
    bioinformatics queries and transforms them for accurate downstream processing.
    
    Patterns handled:
    1. Exclusion: "not X", "without X", "except X", "excluding X"
    2. Preference: "X instead of Y", "X rather than Y", "prefer X over Y"
    3. Correction: "not X but Y", "Y not X"
    4. Avoidance: "don't use X", "avoid X", "skip X"
    
    Example Usage:
        handler = NegationHandler()
        result = handler.detect("analyze RNA-seq data but not mouse samples")
        # result.negated_terms = ["mouse"]
        # result.negation_type = NegationType.EXCLUSION
    """
    
    # Compiled regex patterns for efficiency
    PATTERNS: Dict[NegationType, List[re.Pattern]] = {
        # Exclusion patterns - "not X", "without X", etc.
        NegationType.EXCLUSION: [
            re.compile(
                r'\b(?:not|no|without|except|excluding|exclude)\s+(?:for\s+)?(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
            re.compile(
                r'\b(\w+(?:[-_]\w+)*)\s+excluded?\b',
                re.IGNORECASE
            ),
            re.compile(
                r'\bexcept\s+(?:for\s+)?(?:the\s+)?(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
            re.compile(
                r'\b(?:everything|all)\s+(?:but|except)\s+(?:the\s+)?(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
            # "but not X" pattern
            re.compile(
                r'\bbut\s+not\s+(?:for\s+)?(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
        ],
        
        # Preference patterns - "X instead of Y", "prefer X over Y"
        NegationType.PREFERENCE: [
            re.compile(
                r'\b(\w+(?:[-_]\w+)*)\s+(?:instead\s+of|rather\s+than|over)\s+(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
            re.compile(
                r'\bprefer\s+(\w+(?:[-_]\w+)*)\s+(?:to|over)\s+(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
            re.compile(
                r'\buse\s+(\w+(?:[-_]\w+)*)\s+not\s+(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
            re.compile(
                r'\b(\w+(?:[-_]\w+)*)\s+is\s+(?:better|preferred)\s+(?:than|over)\s+(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
        ],
        
        # Correction patterns - "not X but Y"
        NegationType.CORRECTION: [
            re.compile(
                r'\bnot\s+(\w+(?:[-_]\w+)*)\s*[,;]?\s*(?:but|use|want|need)\s+(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
            re.compile(
                r'\b(\w+(?:[-_]\w+)*)\s*[,;]?\s*not\s+(\w+(?:[-_]\w+)*)\b',
                re.IGNORECASE
            ),
            re.compile(
                r'\bactually\s+(\w+(?:[-_]\w+)*)\s*[,;]?\s*not\s+(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
        ],
        
        # Avoidance patterns - "don't use X", "avoid X"
        NegationType.AVOIDANCE: [
            re.compile(
                r"\b(?:don'?t|do\s+not|cannot|can'?t)\s+(?:use|want|need|run|execute)\s+(\w+(?:[-_]\w+)*)",
                re.IGNORECASE
            ),
            re.compile(
                r'\b(?:avoid|skip|omit|ignore)\s+(\w+(?:[-_]\w+)*)',
                re.IGNORECASE
            ),
            re.compile(
                r'\b(\w+(?:[-_]\w+)*)\s+(?:is\s+)?(?:not\s+)?(?:wanted|needed|required|supported)\b',
                re.IGNORECASE
            ),
        ],
    }
    
    # Bioinformatics-specific terms to watch for (higher relevance)
    BIO_TERMS: Dict[str, Set[str]] = {
        "organisms": {
            "human", "mouse", "rat", "fly", "worm", "zebrafish", "yeast",
            "arabidopsis", "drosophila", "elegans", "sapiens", "musculus",
            "bacteria", "viral", "fungal", "plant", "mammal", "vertebrate",
        },
        "aligners": {
            "bwa", "bowtie", "bowtie2", "star", "hisat", "hisat2", "minimap2",
            "salmon", "kallisto", "tophat", "tophat2", "subread", "bbmap",
        },
        "variant_callers": {
            "gatk", "freebayes", "deepvariant", "mutect", "mutect2", "varscan",
            "strelka", "strelka2", "bcftools", "platypus", "lofreq", "octopus",
        },
        "assemblers": {
            "spades", "megahit", "trinity", "canu", "flye", "hifiasm",
            "velvet", "abyss", "soapdenovo", "masurca", "wtdbg2",
        },
        "peak_callers": {
            "macs", "macs2", "macs3", "homer", "sicer", "peakseq", "spp",
        },
        "quantifiers": {
            "htseq", "featurecounts", "rsem", "salmon", "kallisto", "stringtie",
            "cufflinks", "deseq", "deseq2", "edger", "limma",
        },
        "file_types": {
            "fastq", "bam", "sam", "vcf", "bed", "bigwig", "gtf", "gff",
            "fasta", "fa", "gz", "cram", "bcf",
        },
        "analysis_types": {
            "alignment", "mapping", "calling", "assembly", "annotation",
            "quantification", "normalization", "clustering", "qc",
            "trimming", "filtering", "deduplication",
        },
    }
    
    # Words to ignore (too common, not meaningful)
    STOP_WORDS: Set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "can", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "so", "than", "too", "very", "just", "also",
        "now", "any", "that", "this", "it", "its", "my", "your",
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the negation handler.
        
        Args:
            strict_mode: If True, only recognize bioinformatics-specific terms.
                        If False, recognize any term that looks technical.
        """
        self.strict_mode = strict_mode
        self._all_bio_terms = self._build_bio_term_set()
    
    def _build_bio_term_set(self) -> Set[str]:
        """Build complete set of bioinformatics terms."""
        terms = set()
        for category_terms in self.BIO_TERMS.values():
            terms.update(category_terms)
        return terms
    
    def detect(self, query: str) -> NegationResult:
        """
        Detect negation patterns in query.
        
        Args:
            query: The user's input query
            
        Returns:
            NegationResult with detected negations and transformed query
        """
        if not query or not query.strip():
            return NegationResult(
                has_negation=False,
                negation_type=NegationType.NONE,
                original_query=query,
                transformed_query=query,
            )
        
        negated_terms: Set[str] = set()
        preferred_terms: Set[str] = set()
        detected_type = NegationType.NONE
        confidence = 1.0
        
        # Check each pattern type (in priority order)
        pattern_priority = [
            NegationType.CORRECTION,   # Most specific
            NegationType.PREFERENCE,
            NegationType.AVOIDANCE,
            NegationType.EXCLUSION,    # Most general
        ]
        
        for neg_type in pattern_priority:
            patterns = self.PATTERNS[neg_type]
            for pattern in patterns:
                matches = pattern.findall(query)
                for match in matches:
                    if isinstance(match, tuple):
                        # Pattern with groups (preference/correction)
                        if neg_type in (NegationType.PREFERENCE, NegationType.CORRECTION):
                            # First group is preferred, second is negated
                            pref = match[0].lower().strip()
                            neg = match[1].lower().strip()
                            if self._is_relevant_term(pref):
                                preferred_terms.add(pref)
                            if self._is_relevant_term(neg):
                                negated_terms.add(neg)
                        else:
                            # Multiple negated terms
                            for term in match:
                                term = term.lower().strip()
                                if self._is_relevant_term(term):
                                    negated_terms.add(term)
                    else:
                        # Single term
                        term = match.lower().strip()
                        if self._is_relevant_term(term):
                            negated_terms.add(term)
                    
                    # Set type only once (first match wins)
                    if detected_type == NegationType.NONE:
                        detected_type = neg_type
        
        has_negation = len(negated_terms) > 0
        
        # Calculate confidence based on term relevance
        if has_negation:
            bio_term_count = sum(1 for t in negated_terms if t in self._all_bio_terms)
            confidence = 0.7 + (0.3 * bio_term_count / max(len(negated_terms), 1))
        
        # Transform query with markers
        transformed = self._transform_query(query, negated_terms, preferred_terms)
        
        return NegationResult(
            has_negation=has_negation,
            negation_type=detected_type if has_negation else NegationType.NONE,
            negated_terms=sorted(list(negated_terms)),
            preferred_terms=sorted(list(preferred_terms)),
            original_query=query,
            transformed_query=transformed,
            confidence=confidence,
        )
    
    def _is_relevant_term(self, term: str) -> bool:
        """Check if a term is relevant for negation tracking."""
        if not term or len(term) < 2:
            return False
        
        # Skip stop words
        if term.lower() in self.STOP_WORDS:
            return False
        
        # In strict mode, only accept bio terms
        if self.strict_mode:
            return term.lower() in self._all_bio_terms
        
        # In normal mode, accept bio terms or technical-looking terms
        if term.lower() in self._all_bio_terms:
            return True
        
        # Accept terms that look technical (contain numbers, hyphens, underscores)
        if any(c in term for c in "-_") or any(c.isdigit() for c in term):
            return True
        
        # Accept longer terms (likely technical)
        if len(term) >= 4:
            return True
        
        return False
    
    def _transform_query(
        self,
        query: str,
        negated: Set[str],
        preferred: Set[str]
    ) -> str:
        """
        Add explicit markers for downstream processing.
        
        These markers help the parser understand what to exclude/prefer.
        """
        markers = []
        
        if negated:
            markers.append(f"[EXCLUDE: {', '.join(sorted(negated))}]")
        
        if preferred:
            markers.append(f"[PREFER: {', '.join(sorted(preferred))}]")
        
        if markers:
            return f"{query} {' '.join(markers)}"
        
        return query
    
    def apply_to_entities(
        self,
        entities: Dict[str, Any],
        negation_result: NegationResult
    ) -> Dict[str, Any]:
        """
        Apply negation result to extracted entities.
        
        Removes negated entities and applies preferences.
        
        Args:
            entities: Dictionary of extracted entities
            negation_result: Result from detect()
            
        Returns:
            Filtered entities dictionary
        """
        if not negation_result.has_negation:
            return entities
        
        filtered = {}
        negated_set = set(negation_result.negated_terms)
        preferred_set = set(negation_result.preferred_terms)
        
        for key, value in entities.items():
            # Get lowercase value for comparison
            if isinstance(value, str):
                value_lower = value.lower()
            elif isinstance(value, list):
                # Filter list values
                filtered_list = [
                    v for v in value
                    if str(v).lower() not in negated_set
                ]
                if filtered_list:
                    filtered[key] = filtered_list
                continue
            else:
                value_lower = str(value).lower()
            
            # Skip if value is negated
            if value_lower in negated_set:
                # Check if we have a preferred alternative
                if preferred_set:
                    # Use the first preferred term as replacement
                    filtered[key] = list(preferred_set)[0]
                # else: skip this entity entirely
                continue
            
            filtered[key] = value
        
        return filtered
    
    def get_entity_category(self, term: str) -> Optional[str]:
        """
        Get the category of a bioinformatics term.
        
        Args:
            term: The term to categorize
            
        Returns:
            Category name or None if not found
        """
        term_lower = term.lower()
        for category, terms in self.BIO_TERMS.items():
            if term_lower in terms:
                return category
        return None


# Convenience function for quick usage
def detect_negation(query: str) -> NegationResult:
    """
    Quick function to detect negation in a query.
    
    Args:
        query: The user's input query
        
    Returns:
        NegationResult with detected negations
    """
    handler = NegationHandler()
    return handler.detect(query)
