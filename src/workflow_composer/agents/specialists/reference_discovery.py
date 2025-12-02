"""
Reference Discovery Agent
=========================

Discovers relevant code references from external sources.

Capabilities:
- Search nf-core modules for existing implementations
- Find GitHub repositories with similar workflows
- Identify relevant tool implementations
- Rank references by relevance and quality
"""

import logging
import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ReferenceSource(Enum):
    """Source types for code references."""
    NF_CORE_MODULES = "nf-core/modules"
    NF_CORE_PIPELINES = "nf-core/pipelines"
    GITHUB_REPOS = "github"
    TOOL_DOCUMENTATION = "tool_docs"
    KNOWLEDGE_BASE = "knowledge_base"


@dataclass
class CodeReference:
    """A discovered code reference."""
    source: ReferenceSource
    name: str
    url: str
    description: str
    relevance_score: float  # 0.0 to 1.0
    tools: List[str] = field(default_factory=list)
    language: str = "nextflow"
    stars: int = 0
    last_updated: Optional[str] = None
    snippet: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "relevance_score": self.relevance_score,
            "tools": self.tools,
            "language": self.language,
            "stars": self.stars,
            "last_updated": self.last_updated,
            "snippet": self.snippet,
            "metadata": self.metadata,
        }


@dataclass
class ReferenceSearchResult:
    """Result from reference search."""
    query: str
    references: List[CodeReference] = field(default_factory=list)
    sources_searched: List[ReferenceSource] = field(default_factory=list)
    search_time_ms: float = 0.0
    total_found: int = 0
    
    def top_references(self, n: int = 5) -> List[CodeReference]:
        """Get top N references by relevance score."""
        sorted_refs = sorted(
            self.references, 
            key=lambda r: r.relevance_score, 
            reverse=True
        )
        return sorted_refs[:n]
    
    def by_source(self, source: ReferenceSource) -> List[CodeReference]:
        """Get references from a specific source."""
        return [r for r in self.references if r.source == source]


class ReferenceDiscoveryAgent:
    """
    Discovers relevant code references for workflow generation.
    
    Searches multiple sources:
    1. nf-core/modules - Curated Nextflow modules
    2. nf-core pipelines - Complete pipeline references
    3. GitHub - General bioinformatics repositories
    4. Local knowledge base - Previously indexed content
    
    Inspired by DeepCode's Code Reference Mining Agent.
    """
    
    # Known nf-core modules by tool name
    NF_CORE_MODULES = {
        "fastqc": {
            "name": "fastqc",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/fastqc",
            "description": "FastQC quality control",
        },
        "fastp": {
            "name": "fastp",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/fastp",
            "description": "Fast all-in-one FASTQ preprocessing",
        },
        "star": {
            "name": "star/align",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/star/align",
            "description": "STAR RNA-seq aligner",
        },
        "salmon": {
            "name": "salmon/quant",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/salmon/quant",
            "description": "Salmon transcript quantification",
        },
        "hisat2": {
            "name": "hisat2/align",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/hisat2/align",
            "description": "HISAT2 aligner",
        },
        "bowtie2": {
            "name": "bowtie2/align",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/bowtie2/align",
            "description": "Bowtie2 short read aligner",
        },
        "bwa": {
            "name": "bwa/mem",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/bwa/mem",
            "description": "BWA-MEM alignment",
        },
        "samtools": {
            "name": "samtools/sort",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/samtools/sort",
            "description": "Samtools BAM processing",
        },
        "picard": {
            "name": "picard/markduplicates",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/picard/markduplicates",
            "description": "Picard MarkDuplicates",
        },
        "gatk": {
            "name": "gatk4/haplotypecaller",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/gatk4/haplotypecaller",
            "description": "GATK HaplotypeCaller variant calling",
        },
        "macs2": {
            "name": "macs2/callpeak",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/macs2/callpeak",
            "description": "MACS2 peak calling",
        },
        "deeptools": {
            "name": "deeptools/bamcoverage",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/deeptools/bamcoverage",
            "description": "deepTools BAM coverage",
        },
        "multiqc": {
            "name": "multiqc",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/multiqc",
            "description": "MultiQC report aggregation",
        },
        "featurecounts": {
            "name": "subread/featurecounts",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/subread/featurecounts",
            "description": "featureCounts read counting",
        },
        "kallisto": {
            "name": "kallisto/quant",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/kallisto/quant",
            "description": "Kallisto transcript quantification",
        },
        "bcftools": {
            "name": "bcftools/call",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/bcftools/call",
            "description": "BCFtools variant calling",
        },
        "trimgalore": {
            "name": "trimgalore",
            "url": "https://github.com/nf-core/modules/tree/master/modules/nf-core/trimgalore",
            "description": "Trim Galore adapter trimming",
        },
    }
    
    # Known nf-core pipelines by analysis type
    NF_CORE_PIPELINES = {
        "rna-seq": {
            "name": "nf-core/rnaseq",
            "url": "https://github.com/nf-core/rnaseq",
            "description": "RNA sequencing analysis pipeline",
            "stars": 800,
        },
        "chip-seq": {
            "name": "nf-core/chipseq",
            "url": "https://github.com/nf-core/chipseq",
            "description": "ChIP-seq peak calling pipeline",
            "stars": 200,
        },
        "atac-seq": {
            "name": "nf-core/atacseq",
            "url": "https://github.com/nf-core/atacseq",
            "description": "ATAC-seq analysis pipeline",
            "stars": 150,
        },
        "dna-seq": {
            "name": "nf-core/sarek",
            "url": "https://github.com/nf-core/sarek",
            "description": "Germline and somatic variant calling",
            "stars": 400,
        },
        "variant-calling": {
            "name": "nf-core/sarek",
            "url": "https://github.com/nf-core/sarek",
            "description": "Variant calling pipeline",
            "stars": 400,
        },
        "methylation": {
            "name": "nf-core/methylseq",
            "url": "https://github.com/nf-core/methylseq",
            "description": "Methylation analysis pipeline",
            "stars": 100,
        },
        "scrna-seq": {
            "name": "nf-core/scrnaseq",
            "url": "https://github.com/nf-core/scrnaseq",
            "description": "Single-cell RNA-seq pipeline",
            "stars": 150,
        },
        "metagenomics": {
            "name": "nf-core/mag",
            "url": "https://github.com/nf-core/mag",
            "description": "Metagenome assembly pipeline",
            "stars": 180,
        },
        "amplicon": {
            "name": "nf-core/ampliseq",
            "url": "https://github.com/nf-core/ampliseq",
            "description": "Amplicon sequencing analysis",
            "stars": 120,
        },
        "hic": {
            "name": "nf-core/hic",
            "url": "https://github.com/nf-core/hic",
            "description": "Hi-C data analysis",
            "stars": 60,
        },
    }
    
    def __init__(self, router=None, knowledge_base=None, github_token: str = None):
        """
        Initialize reference discovery agent.
        
        Args:
            router: LLM router for semantic matching
            knowledge_base: Local knowledge base for cached references
            github_token: Optional GitHub token for API access
        """
        self.router = router
        self.knowledge_base = knowledge_base
        self.github_token = github_token
        self._cache: Dict[str, ReferenceSearchResult] = {}
    
    async def discover(
        self,
        query: str,
        analysis_type: str = None,
        tools: List[str] = None,
        sources: List[ReferenceSource] = None,
        limit: int = 10,
    ) -> ReferenceSearchResult:
        """
        Discover relevant code references.
        
        Args:
            query: Natural language query or workflow description
            analysis_type: Type of analysis (rna-seq, chip-seq, etc.)
            tools: List of tools to find references for
            sources: Which sources to search (default: all)
            limit: Maximum references to return
            
        Returns:
            ReferenceSearchResult with discovered references
        """
        import time
        start_time = time.time()
        
        # Check cache
        cache_key = f"{query}:{analysis_type}:{tools}"
        if cache_key in self._cache:
            logger.info("Using cached reference search result")
            return self._cache[cache_key]
        
        # Default sources
        if sources is None:
            sources = [
                ReferenceSource.NF_CORE_MODULES,
                ReferenceSource.NF_CORE_PIPELINES,
                ReferenceSource.KNOWLEDGE_BASE,
            ]
        
        references: List[CodeReference] = []
        
        # Search each source
        if ReferenceSource.NF_CORE_MODULES in sources:
            refs = await self._search_nf_core_modules(query, tools)
            references.extend(refs)
        
        if ReferenceSource.NF_CORE_PIPELINES in sources:
            refs = await self._search_nf_core_pipelines(query, analysis_type)
            references.extend(refs)
        
        if ReferenceSource.KNOWLEDGE_BASE in sources and self.knowledge_base:
            refs = await self._search_knowledge_base(query)
            references.extend(refs)
        
        if ReferenceSource.GITHUB_REPOS in sources:
            refs = await self._search_github(query, analysis_type)
            references.extend(refs)
        
        # Deduplicate by URL
        seen_urls = set()
        unique_refs = []
        for ref in references:
            if ref.url not in seen_urls:
                seen_urls.add(ref.url)
                unique_refs.append(ref)
        
        # Sort by relevance and limit
        unique_refs.sort(key=lambda r: r.relevance_score, reverse=True)
        unique_refs = unique_refs[:limit]
        
        # Create result
        elapsed_ms = (time.time() - start_time) * 1000
        result = ReferenceSearchResult(
            query=query,
            references=unique_refs,
            sources_searched=sources,
            search_time_ms=elapsed_ms,
            total_found=len(unique_refs),
        )
        
        # Cache result
        self._cache[cache_key] = result
        
        logger.info(f"Found {len(unique_refs)} references in {elapsed_ms:.1f}ms")
        return result
    
    def discover_sync(
        self,
        query: str,
        analysis_type: str = None,
        tools: List[str] = None,
    ) -> ReferenceSearchResult:
        """Synchronous version of discover."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use run in async context
                return self._discover_sync_impl(query, analysis_type, tools)
            return loop.run_until_complete(
                self.discover(query, analysis_type, tools)
            )
        except RuntimeError:
            return self._discover_sync_impl(query, analysis_type, tools)
    
    def _discover_sync_impl(
        self,
        query: str,
        analysis_type: str = None,
        tools: List[str] = None,
    ) -> ReferenceSearchResult:
        """Synchronous implementation without async."""
        references = []
        
        # Search nf-core modules
        if tools:
            for tool in tools:
                tool_lower = tool.lower()
                for key, module in self.NF_CORE_MODULES.items():
                    if key in tool_lower or tool_lower in key:
                        references.append(CodeReference(
                            source=ReferenceSource.NF_CORE_MODULES,
                            name=module["name"],
                            url=module["url"],
                            description=module["description"],
                            relevance_score=0.9,
                            tools=[tool],
                            language="nextflow",
                        ))
        
        # Search nf-core pipelines
        if analysis_type:
            analysis_lower = analysis_type.lower()
            for key, pipeline in self.NF_CORE_PIPELINES.items():
                if key in analysis_lower or analysis_lower in key:
                    references.append(CodeReference(
                        source=ReferenceSource.NF_CORE_PIPELINES,
                        name=pipeline["name"],
                        url=pipeline["url"],
                        description=pipeline["description"],
                        relevance_score=0.95,
                        tools=[],
                        language="nextflow",
                        stars=pipeline.get("stars", 0),
                    ))
        
        # Keyword search in query
        query_lower = query.lower()
        for key, module in self.NF_CORE_MODULES.items():
            if key in query_lower:
                if not any(r.name == module["name"] for r in references):
                    references.append(CodeReference(
                        source=ReferenceSource.NF_CORE_MODULES,
                        name=module["name"],
                        url=module["url"],
                        description=module["description"],
                        relevance_score=0.7,
                        tools=[key],
                        language="nextflow",
                    ))
        
        return ReferenceSearchResult(
            query=query,
            references=references,
            sources_searched=[ReferenceSource.NF_CORE_MODULES, ReferenceSource.NF_CORE_PIPELINES],
            total_found=len(references),
        )
    
    async def _search_nf_core_modules(
        self,
        query: str,
        tools: List[str] = None,
    ) -> List[CodeReference]:
        """Search nf-core modules for tool implementations."""
        references = []
        query_lower = query.lower()
        
        # Direct tool matches
        if tools:
            for tool in tools:
                tool_lower = tool.lower().replace("-", "").replace("_", "")
                for key, module in self.NF_CORE_MODULES.items():
                    key_normalized = key.replace("-", "").replace("_", "")
                    if key_normalized in tool_lower or tool_lower in key_normalized:
                        references.append(CodeReference(
                            source=ReferenceSource.NF_CORE_MODULES,
                            name=module["name"],
                            url=module["url"],
                            description=module["description"],
                            relevance_score=0.95,
                            tools=[tool],
                            language="nextflow",
                        ))
        
        # Keyword matches in query
        for key, module in self.NF_CORE_MODULES.items():
            if key in query_lower:
                # Check if already added
                if not any(r.name == module["name"] for r in references):
                    references.append(CodeReference(
                        source=ReferenceSource.NF_CORE_MODULES,
                        name=module["name"],
                        url=module["url"],
                        description=module["description"],
                        relevance_score=0.75,
                        tools=[key],
                        language="nextflow",
                    ))
        
        return references
    
    async def _search_nf_core_pipelines(
        self,
        query: str,
        analysis_type: str = None,
    ) -> List[CodeReference]:
        """Search nf-core pipelines for complete workflow references."""
        references = []
        query_lower = query.lower()
        
        # Direct analysis type match
        if analysis_type:
            analysis_lower = analysis_type.lower().replace("-", "").replace("_", "")
            for key, pipeline in self.NF_CORE_PIPELINES.items():
                key_normalized = key.replace("-", "").replace("_", "")
                if key_normalized in analysis_lower or analysis_lower in key_normalized:
                    references.append(CodeReference(
                        source=ReferenceSource.NF_CORE_PIPELINES,
                        name=pipeline["name"],
                        url=pipeline["url"],
                        description=pipeline["description"],
                        relevance_score=0.98,
                        language="nextflow",
                        stars=pipeline.get("stars", 0),
                    ))
        
        # Keyword matches
        analysis_keywords = {
            "rna": ["rna-seq", "rnaseq"],
            "chip": ["chip-seq", "chipseq"],
            "atac": ["atac-seq", "atacseq"],
            "variant": ["dna-seq", "variant-calling"],
            "wgs": ["dna-seq"],
            "wes": ["dna-seq"],
            "methyl": ["methylation"],
            "bisulfite": ["methylation"],
            "single-cell": ["scrna-seq"],
            "single cell": ["scrna-seq"],
            "10x": ["scrna-seq"],
            "metagenom": ["metagenomics"],
            "amplicon": ["amplicon"],
            "16s": ["amplicon"],
            "hic": ["hic"],
            "hi-c": ["hic"],
        }
        
        for keyword, analysis_types in analysis_keywords.items():
            if keyword in query_lower:
                for at in analysis_types:
                    if at in self.NF_CORE_PIPELINES:
                        pipeline = self.NF_CORE_PIPELINES[at]
                        if not any(r.name == pipeline["name"] for r in references):
                            references.append(CodeReference(
                                source=ReferenceSource.NF_CORE_PIPELINES,
                                name=pipeline["name"],
                                url=pipeline["url"],
                                description=pipeline["description"],
                                relevance_score=0.85,
                                language="nextflow",
                                stars=pipeline.get("stars", 0),
                            ))
        
        return references
    
    async def _search_knowledge_base(self, query: str) -> List[CodeReference]:
        """Search local knowledge base for cached references."""
        if not self.knowledge_base:
            return []
        
        references = []
        
        try:
            # Import knowledge source types
            from ..rag.knowledge_base import KnowledgeSource
            
            results = self.knowledge_base.search(
                query,
                sources=[KnowledgeSource.NF_CORE_MODULES, KnowledgeSource.TOOL_CATALOG],
                limit=5,
            )
            
            for doc in results:
                references.append(CodeReference(
                    source=ReferenceSource.KNOWLEDGE_BASE,
                    name=doc.title,
                    url=doc.metadata.get("url", ""),
                    description=doc.content[:200],
                    relevance_score=doc.metadata.get("score", 0.5),
                    language="nextflow",
                    snippet=doc.content[:500],
                    metadata=doc.metadata,
                ))
        except Exception as e:
            logger.debug(f"Knowledge base search failed: {e}")
        
        return references
    
    async def _search_github(
        self,
        query: str,
        analysis_type: str = None,
    ) -> List[CodeReference]:
        """
        Search GitHub for relevant repositories.
        
        Note: Requires github_token for API access.
        """
        if not self.github_token:
            logger.debug("GitHub token not provided, skipping GitHub search")
            return []
        
        references = []
        
        try:
            import aiohttp
            
            # Build search query
            search_terms = [query]
            if analysis_type:
                search_terms.append(analysis_type)
            search_terms.extend(["nextflow", "bioinformatics"])
            search_query = " ".join(search_terms)
            
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.github.com/search/repositories?q={search_query}&sort=stars&per_page=5"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        for item in data.get("items", []):
                            references.append(CodeReference(
                                source=ReferenceSource.GITHUB_REPOS,
                                name=item["full_name"],
                                url=item["html_url"],
                                description=item.get("description", "")[:200],
                                relevance_score=min(item["stargazers_count"] / 1000, 0.9),
                                language=item.get("language", "unknown"),
                                stars=item["stargazers_count"],
                                last_updated=item.get("updated_at"),
                            ))
        except Exception as e:
            logger.warning(f"GitHub search failed: {e}")
        
        return references
    
    def get_module_snippet(self, module_name: str) -> Optional[str]:
        """
        Get a code snippet for an nf-core module.
        
        Args:
            module_name: Name of the module (e.g., "fastqc", "star/align")
            
        Returns:
            Code snippet or None if not found
        """
        # Standard nf-core module usage patterns
        snippets = {
            "fastqc": '''
include { FASTQC } from '../modules/nf-core/fastqc/main'

FASTQC(
    reads_ch  // channel: [ val(meta), [ reads ] ]
)
''',
            "fastp": '''
include { FASTP } from '../modules/nf-core/fastp/main'

FASTP(
    reads_ch,      // channel: [ val(meta), [ reads ] ]
    adapter_fasta, // channel: [ adapter.fasta ]
    false,         // save_trimmed_fail
    false          // save_merged
)
''',
            "star/align": '''
include { STAR_ALIGN } from '../modules/nf-core/star/align/main'

STAR_ALIGN(
    reads_ch,     // channel: [ val(meta), [ reads ] ]
    star_index,   // channel: [ star_index ]
    gtf,          // channel: [ gtf ]
    false,        // star_ignore_sjdbgtf
    '',           // seq_platform
    ''            // seq_center
)
''',
            "salmon/quant": '''
include { SALMON_QUANT } from '../modules/nf-core/salmon/quant/main'

SALMON_QUANT(
    reads_ch,      // channel: [ val(meta), [ reads ] ]
    salmon_index,  // channel: [ salmon_index ]
    gtf,           // channel: [ gtf ]
    transcript_fa, // channel: [ transcript.fa ]
    false,         // alignment_mode
    ''             // lib_type
)
''',
            "bowtie2/align": '''
include { BOWTIE2_ALIGN } from '../modules/nf-core/bowtie2/align/main'

BOWTIE2_ALIGN(
    reads_ch,      // channel: [ val(meta), [ reads ] ]
    bowtie2_index, // channel: [ bowtie2_index ]
    false,         // save_unaligned
    false          // sort_bam
)
''',
            "macs2/callpeak": '''
include { MACS2_CALLPEAK } from '../modules/nf-core/macs2/callpeak/main'

MACS2_CALLPEAK(
    bam_ch,  // channel: [ val(meta), [ ip_bam ], [ control_bam ] ]
    gsize    // val: effective genome size
)
''',
            "multiqc": '''
include { MULTIQC } from '../modules/nf-core/multiqc/main'

MULTIQC(
    multiqc_files.collect(),  // channel: [ multiqc_files ]
    multiqc_config,           // channel: [ multiqc_config ]
    extra_multiqc_config,     // channel: [ extra_config ]
    multiqc_logo              // channel: [ logo ]
)
''',
        }
        
        return snippets.get(module_name.lower())
    
    def format_references_for_prompt(
        self,
        result: ReferenceSearchResult,
        include_snippets: bool = True,
    ) -> str:
        """
        Format references for inclusion in LLM prompts.
        
        Args:
            result: Search result to format
            include_snippets: Whether to include code snippets
            
        Returns:
            Formatted string for prompt inclusion
        """
        if not result.references:
            return "No relevant references found."
        
        lines = [
            f"## Discovered References ({result.total_found} found)",
            "",
        ]
        
        # Group by source
        by_source: Dict[ReferenceSource, List[CodeReference]] = {}
        for ref in result.references:
            if ref.source not in by_source:
                by_source[ref.source] = []
            by_source[ref.source].append(ref)
        
        for source, refs in by_source.items():
            lines.append(f"### {source.value}")
            lines.append("")
            
            for ref in refs:
                lines.append(f"**{ref.name}** (relevance: {ref.relevance_score:.2f})")
                lines.append(f"- URL: {ref.url}")
                lines.append(f"- {ref.description}")
                
                if include_snippets and ref.source == ReferenceSource.NF_CORE_MODULES:
                    snippet = self.get_module_snippet(ref.name)
                    if snippet:
                        lines.append(f"- Usage:")
                        lines.append("```nextflow")
                        lines.append(snippet.strip())
                        lines.append("```")
                
                lines.append("")
        
        return "\n".join(lines)
