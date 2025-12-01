"""
Planner Agent
=============

Designs workflow architecture from user queries.

Creates a structured plan with:
- Input requirements
- Processing steps
- Tool selections
- Output specifications
- QC checkpoints
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """A single step in the workflow plan."""
    name: str
    tool: str
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class WorkflowPlan:
    """Complete workflow plan."""
    name: str
    description: str
    analysis_type: str
    organism: Optional[str] = None
    genome_build: Optional[str] = None
    
    # Input specifications
    input_type: str = "fastq"  # fastq, bam, vcf, etc.
    read_type: str = "paired"  # paired, single
    
    # Steps
    steps: List[WorkflowStep] = field(default_factory=list)
    
    # QC checkpoints
    qc_checkpoints: List[str] = field(default_factory=list)
    
    # Output files
    outputs: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    estimated_runtime_hours: float = 1.0
    recommended_memory_gb: int = 16
    recommended_cpus: int = 8
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowPlan":
        """Create from dictionary."""
        steps = [WorkflowStep(**s) for s in data.pop("steps", [])]
        return cls(steps=steps, **data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowPlan":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class PlannerAgent:
    """
    Workflow architecture designer.
    
    Takes user queries and creates structured workflow plans
    that can be passed to CodeGenAgent for implementation.
    
    Can optionally use RAG from KnowledgeBase for enhanced planning.
    """
    
    SYSTEM_PROMPT = """You are a bioinformatics workflow architect expert.
Given a user query, create a detailed workflow plan in JSON format.

Your plan should include:
1. Input requirements (file types, organism, read type)
2. Processing steps in execution order
3. Tools for each step with sensible default parameters
4. Output files and formats
5. Quality control checkpoints
6. Resource estimates (memory, CPUs, runtime)

For each step, specify:
- name: Short identifier (e.g., "fastqc", "alignment", "quantification")
- tool: The actual tool name (e.g., "STAR", "salmon", "DESeq2")
- description: What this step does
- inputs: List of input channel names
- outputs: List of output channel names
- parameters: Tool-specific parameters with sensible defaults
- resources: memory, cpus, time estimates
- dependencies: Which steps must complete first

Output ONLY valid JSON matching this structure:
{
  "name": "workflow_name",
  "description": "What this workflow does",
  "analysis_type": "rna-seq|chip-seq|dna-seq|etc",
  "organism": "human|mouse|etc",
  "genome_build": "GRCh38|GRCm39|etc",
  "input_type": "fastq|bam|etc",
  "read_type": "paired|single",
  "steps": [...],
  "qc_checkpoints": ["after_alignment", "final"],
  "outputs": [{"name": "counts", "format": "tsv"}],
  "estimated_runtime_hours": 2.0,
  "recommended_memory_gb": 32,
  "recommended_cpus": 8
}"""

    def __init__(self, router=None, knowledge_base=None):
        """
        Initialize planner agent.
        
        Args:
            router: LLM provider router for generating plans
            knowledge_base: Optional KnowledgeBase for RAG enhancement
        """
        self.router = router
        self.knowledge_base = knowledge_base
        self._plan_cache = {}
    
    async def create_plan(self, query: str) -> WorkflowPlan:
        """
        Create a workflow plan from user query.
        
        Args:
            query: User's natural language query
            
        Returns:
            WorkflowPlan with structured plan
        """
        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self._plan_cache:
            logger.info("Using cached plan")
            return self._plan_cache[cache_key]
        
        if self.router is None:
            # Return a default plan if no router
            return self._create_default_plan(query)
        
        try:
            # Enhance prompt with RAG if knowledge base available
            context = await self._get_rag_context(query) if self.knowledge_base else ""
            
            # Generate plan using LLM
            rag_section = f"\n\nRelevant context from knowledge base:\n{context}" if context else ""
            prompt = f"{self.SYSTEM_PROMPT}{rag_section}\n\nUser Query: {query}"
            response = await self.router.route_async(prompt)
            
            # Extract JSON from response
            plan_json = self._extract_json(response)
            plan = WorkflowPlan.from_dict(plan_json)
            
            # Cache the plan
            self._plan_cache[cache_key] = plan
            
            return plan
            
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using default plan")
            return self._create_default_plan(query)
    
    async def _get_rag_context(self, query: str) -> str:
        """
        Get relevant context from knowledge base.
        
        Args:
            query: User query
            
        Returns:
            Formatted context string
        """
        if not self.knowledge_base:
            return ""
        
        try:
            # Import knowledge source types
            from ..rag.knowledge_base import KnowledgeSource
            
            # Search for relevant modules and tools
            results = self.knowledge_base.search(
                query, 
                sources=[KnowledgeSource.NF_CORE_MODULES, KnowledgeSource.TOOL_CATALOG],
                limit=5
            )
            
            if not results:
                return ""
            
            # Format context
            context_parts = []
            for doc in results:
                context_parts.append(f"[{doc.source.value}] {doc.title}:\n{doc.content[:500]}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.debug(f"RAG context retrieval failed: {e}")
            return ""
    
    def create_plan_sync(self, query: str) -> WorkflowPlan:
        """Synchronous version of create_plan."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, can't use run
                return self._create_default_plan(query)
            return loop.run_until_complete(self.create_plan(query))
        except RuntimeError:
            return self._create_default_plan(query)
    
    def _extract_json(self, response: str) -> dict:
        """Extract JSON from LLM response."""
        # Try to find JSON in response
        import re
        
        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try the whole response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Could not extract JSON from response")
    
    def _create_default_plan(self, query: str) -> WorkflowPlan:
        """Create a default plan based on query keywords."""
        query_lower = query.lower()
        
        # Detect analysis type
        if "rna-seq" in query_lower or "rnaseq" in query_lower or "rna seq" in query_lower:
            return self._create_rnaseq_plan(query)
        elif "chip-seq" in query_lower or "chipseq" in query_lower:
            return self._create_chipseq_plan(query)
        elif "variant" in query_lower or "dna-seq" in query_lower or "wgs" in query_lower:
            return self._create_dnaseq_plan(query)
        else:
            # Generic plan
            return self._create_generic_plan(query)
    
    def _create_rnaseq_plan(self, query: str) -> WorkflowPlan:
        """Create RNA-seq workflow plan."""
        query_lower = query.lower()
        
        # Detect organism
        organism = "human"
        genome_build = "GRCh38"
        if "mouse" in query_lower:
            organism = "mouse"
            genome_build = "GRCm39"
        
        steps = [
            WorkflowStep(
                name="fastqc",
                tool="FastQC",
                description="Quality control of raw reads",
                inputs=["reads"],
                outputs=["fastqc_reports"],
                resources={"memory": "4 GB", "cpus": 2, "time": "1h"},
            ),
            WorkflowStep(
                name="trimming",
                tool="fastp",
                description="Adapter trimming and quality filtering",
                inputs=["reads"],
                outputs=["trimmed_reads"],
                parameters={"qualified_quality_phred": 20, "length_required": 36},
                resources={"memory": "8 GB", "cpus": 4, "time": "2h"},
                dependencies=["fastqc"],
            ),
            WorkflowStep(
                name="alignment",
                tool="STAR",
                description="Align reads to reference genome",
                inputs=["trimmed_reads", "genome_index"],
                outputs=["aligned_bam"],
                parameters={"outSAMtype": "BAM SortedByCoordinate"},
                resources={"memory": "32 GB", "cpus": 8, "time": "4h"},
                dependencies=["trimming"],
            ),
            WorkflowStep(
                name="quantification",
                tool="featureCounts",
                description="Count reads per gene",
                inputs=["aligned_bam", "annotation"],
                outputs=["count_matrix"],
                parameters={"isPairedEnd": True, "countMultiMapping": False},
                resources={"memory": "8 GB", "cpus": 4, "time": "1h"},
                dependencies=["alignment"],
            ),
            WorkflowStep(
                name="multiqc",
                tool="MultiQC",
                description="Aggregate QC reports",
                inputs=["fastqc_reports", "star_logs", "featurecounts_summary"],
                outputs=["multiqc_report"],
                resources={"memory": "4 GB", "cpus": 2, "time": "30m"},
                dependencies=["quantification"],
            ),
        ]
        
        # Add differential expression if mentioned
        if "differential" in query_lower or "de" in query_lower:
            steps.append(WorkflowStep(
                name="differential_expression",
                tool="DESeq2",
                description="Differential expression analysis",
                inputs=["count_matrix", "sample_sheet"],
                outputs=["de_results", "normalized_counts", "pca_plot"],
                parameters={"alpha": 0.05, "lfcThreshold": 0},
                resources={"memory": "16 GB", "cpus": 4, "time": "1h"},
                dependencies=["quantification"],
            ))
        
        return WorkflowPlan(
            name="rnaseq_analysis",
            description="RNA-seq analysis pipeline",
            analysis_type="rna-seq",
            organism=organism,
            genome_build=genome_build,
            input_type="fastq",
            read_type="paired",
            steps=steps,
            qc_checkpoints=["after_fastqc", "after_alignment", "final"],
            outputs=[
                {"name": "count_matrix", "format": "tsv"},
                {"name": "multiqc_report", "format": "html"},
                {"name": "de_results", "format": "csv"} if "differential" in query_lower else None,
            ],
            estimated_runtime_hours=4.0,
            recommended_memory_gb=32,
            recommended_cpus=8,
        )
    
    def _create_chipseq_plan(self, query: str) -> WorkflowPlan:
        """Create ChIP-seq workflow plan."""
        steps = [
            WorkflowStep(
                name="fastqc",
                tool="FastQC",
                description="Quality control of raw reads",
                inputs=["reads"],
                outputs=["fastqc_reports"],
                resources={"memory": "4 GB", "cpus": 2},
            ),
            WorkflowStep(
                name="alignment",
                tool="Bowtie2",
                description="Align reads to reference genome",
                inputs=["reads", "genome_index"],
                outputs=["aligned_bam"],
                resources={"memory": "16 GB", "cpus": 8},
                dependencies=["fastqc"],
            ),
            WorkflowStep(
                name="peak_calling",
                tool="MACS2",
                description="Call peaks from aligned reads",
                inputs=["aligned_bam", "control_bam"],
                outputs=["peaks", "summits"],
                parameters={"gsize": "hs", "qvalue": 0.05},
                resources={"memory": "8 GB", "cpus": 4},
                dependencies=["alignment"],
            ),
            WorkflowStep(
                name="visualization",
                tool="deepTools",
                description="Generate coverage tracks and heatmaps",
                inputs=["aligned_bam", "peaks"],
                outputs=["bigwig", "heatmap"],
                resources={"memory": "16 GB", "cpus": 8},
                dependencies=["peak_calling"],
            ),
        ]
        
        return WorkflowPlan(
            name="chipseq_analysis",
            description="ChIP-seq peak calling pipeline",
            analysis_type="chip-seq",
            input_type="fastq",
            read_type="single",
            steps=steps,
            qc_checkpoints=["after_alignment", "final"],
            outputs=[
                {"name": "peaks", "format": "narrowPeak"},
                {"name": "bigwig", "format": "bw"},
            ],
            estimated_runtime_hours=3.0,
            recommended_memory_gb=16,
            recommended_cpus=8,
        )
    
    def _create_dnaseq_plan(self, query: str) -> WorkflowPlan:
        """Create DNA-seq/variant calling workflow plan."""
        steps = [
            WorkflowStep(
                name="fastqc",
                tool="FastQC",
                description="Quality control",
                inputs=["reads"],
                outputs=["fastqc_reports"],
                resources={"memory": "4 GB", "cpus": 2},
            ),
            WorkflowStep(
                name="alignment",
                tool="BWA-MEM2",
                description="Align reads to reference",
                inputs=["reads", "genome_index"],
                outputs=["aligned_bam"],
                resources={"memory": "32 GB", "cpus": 16},
                dependencies=["fastqc"],
            ),
            WorkflowStep(
                name="mark_duplicates",
                tool="Picard MarkDuplicates",
                description="Mark PCR duplicates",
                inputs=["aligned_bam"],
                outputs=["dedup_bam"],
                resources={"memory": "16 GB", "cpus": 4},
                dependencies=["alignment"],
            ),
            WorkflowStep(
                name="variant_calling",
                tool="GATK HaplotypeCaller",
                description="Call variants",
                inputs=["dedup_bam", "reference"],
                outputs=["vcf"],
                resources={"memory": "16 GB", "cpus": 8},
                dependencies=["mark_duplicates"],
            ),
            WorkflowStep(
                name="variant_filtering",
                tool="GATK VariantFiltration",
                description="Filter variants",
                inputs=["vcf"],
                outputs=["filtered_vcf"],
                dependencies=["variant_calling"],
            ),
        ]
        
        return WorkflowPlan(
            name="variant_calling",
            description="DNA-seq variant calling pipeline",
            analysis_type="dna-seq",
            input_type="fastq",
            read_type="paired",
            steps=steps,
            qc_checkpoints=["after_alignment", "after_calling", "final"],
            outputs=[
                {"name": "filtered_vcf", "format": "vcf.gz"},
            ],
            estimated_runtime_hours=8.0,
            recommended_memory_gb=32,
            recommended_cpus=16,
        )
    
    def _create_generic_plan(self, query: str) -> WorkflowPlan:
        """Create a generic plan for unknown analysis types."""
        return WorkflowPlan(
            name="custom_analysis",
            description=f"Custom analysis: {query}",
            analysis_type="custom",
            steps=[
                WorkflowStep(
                    name="qc",
                    tool="FastQC",
                    description="Quality control",
                    inputs=["reads"],
                    outputs=["qc_reports"],
                ),
            ],
            qc_checkpoints=["final"],
            outputs=[{"name": "results", "format": "various"}],
        )
