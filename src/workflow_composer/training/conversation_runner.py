"""
Conversation Runner
===================

Runs generated conversations through the actual BioPipelines system
to collect real outputs, logs, errors, and performance metrics.

This enables:
1. Real training data collection with actual system outputs
2. Gap analysis to identify weaknesses
3. Performance benchmarking
4. Error pattern identification
"""

import json
import logging
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio

from .conversation_generator import GeneratedConversation, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of processing a single conversation turn."""
    
    turn_index: int
    user_message: str
    
    # System outputs
    system_response: str = ""
    intent_parsed: Dict[str, Any] = field(default_factory=dict)
    tools_selected: List[str] = field(default_factory=list)
    workflow_generated: str = ""
    
    # Performance
    latency_ms: float = 0.0
    tokens_used: int = 0
    
    # Errors and issues
    success: bool = True
    error_message: str = ""
    error_type: str = ""
    warnings: List[str] = field(default_factory=list)
    
    # Quality signals
    intent_confidence: float = 0.0
    response_quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationResult:
    """Complete result of running a conversation through the system."""
    
    conversation_id: str
    scenario_id: str
    category: str
    analysis_type: str
    
    # Turn results
    turn_results: List[TurnResult] = field(default_factory=list)
    
    # Aggregate metrics
    total_turns: int = 0
    successful_turns: int = 0
    failed_turns: int = 0
    
    # Performance
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    
    # Outcomes
    completed_successfully: bool = False
    workflow_generated: bool = False
    final_tools: List[str] = field(default_factory=list)
    final_workflow: str = ""
    
    # Issues identified
    errors: List[Dict[str, str]] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
    
    # Timestamps
    started_at: str = ""
    completed_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['turn_results'] = [tr if isinstance(tr, dict) else tr.to_dict() for tr in self.turn_results]
        return data


@dataclass
class RunnerConfig:
    """Configuration for conversation runner."""
    
    output_dir: Path = field(default_factory=lambda: Path("data/training/run_results"))
    
    # Timeouts
    turn_timeout_seconds: float = 30.0
    conversation_timeout_seconds: float = 300.0
    
    # Retries
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    
    # Logging
    log_full_responses: bool = True
    save_intermediate: bool = True
    
    # Quality thresholds
    min_intent_confidence: float = 0.5


class ConversationRunner:
    """Runs conversations through the actual system."""
    
    def __init__(self, config: RunnerConfig = None):
        self.config = config or RunnerConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load system components
        self._load_components()
        
        # Session state
        self._current_session = {}
    
    def _load_components(self):
        """Load BioPipelines system components."""
        
        self.components = {}
        
        # Get project paths
        project_root = Path(__file__).parent.parent.parent.parent
        catalog_path = project_root / "data" / "tool_catalog"
        
        try:
            from ..core.query_parser import IntentParser, AnalysisType, ParsedIntent
            # IntentParser needs an LLM, but we can use it without for pattern matching
            self.components['intent_parser_class'] = IntentParser
            self.components['AnalysisType'] = AnalysisType
            self.components['ParsedIntent'] = ParsedIntent
            logger.info("Loaded IntentParser classes")
        except Exception as e:
            logger.warning(f"Could not load IntentParser: {e}")
        
        try:
            from ..core.tool_selector import ToolSelector
            # ToolSelector needs catalog_path
            if catalog_path.exists():
                self.components['tool_selector'] = ToolSelector(str(catalog_path))
                logger.info("Loaded ToolSelector")
            else:
                logger.warning(f"Tool catalog not found at {catalog_path}")
        except Exception as e:
            logger.warning(f"Could not load ToolSelector: {e}")
        
        try:
            from ..core.workflow_generator import WorkflowGenerator
            self.components['workflow_generator'] = WorkflowGenerator()
            logger.info("Loaded WorkflowGenerator")
        except Exception as e:
            logger.warning(f"Could not load WorkflowGenerator: {e}")
        
        try:
            from ..facade import WorkflowComposer
            self.components['composer'] = WorkflowComposer()
            logger.info("Loaded WorkflowComposer")
        except Exception as e:
            logger.warning(f"Could not load WorkflowComposer: {e}")
        
        try:
            from ..agents.orchestrator import Orchestrator
            self.components['orchestrator'] = Orchestrator()
            logger.info("Loaded Orchestrator")
        except Exception as e:
            logger.warning(f"Could not load Orchestrator: {e}")
    
    def _parse_intent_simple(self, message: str) -> Optional[Dict[str, Any]]:
        """Simple pattern-based intent parsing for training data collection."""
        
        message_lower = message.lower()
        
        # Analysis type patterns - map to config/analysis_definitions.yaml keys
        analysis_patterns = {
            # RNA-seq variants
            'rna_seq_differential_expression': ['differential expression', 'deseq', 'edger', 'degs', 'de analysis'],
            'rna_seq_basic': ['rna-seq', 'rnaseq', 'rna seq', 'transcriptom', 'gene expression'],
            'rna_seq_de_novo_assembly': ['de novo assembly', 'transcriptome assembly', 'trinity'],
            
            # ChIP-seq
            'chip_seq_peak_calling': ['chip-seq', 'chipseq', 'chip seq', 'histone', 'transcription factor', 'peak calling', 'h3k'],
            
            # ATAC-seq
            'atac_seq': ['atac-seq', 'atacseq', 'atac seq', 'chromatin accessibility', 'open chromatin'],
            
            # Variant calling
            'wgs_variant_calling': ['germline variant', 'snp', 'snv', 'wgs', 'whole genome', 'haplotypecaller'],
            'somatic_variant_calling': ['somatic', 'tumor', 'cancer', 'mutect', 'strelka'],
            'structural_variant_detection': ['structural variant', 'sv', 'cnv', 'copy number', 'manta', 'delly'],
            
            # Metagenomics
            'metagenomics_profiling': ['16s', 'microbiome', 'taxonom', 'amplicon', 'kraken', 'metaphlan'],
            'metagenomics_assembly': ['metagenom', 'metagenomic assembly', 'megahit'],
            
            # Methylation
            'bisulfite_seq_methylation': ['methylation', 'bisulfite', 'dmr', 'cpg', 'epigenom', 'wgbs', 'rrbs'],
            
            # Hi-C
            'hic_chromatin_interaction': ['hi-c', 'hic', 'chromatin interaction', '3d genome', 'tad', 'compartment'],
            
            # Single cell
            'single_cell_rna_seq': ['single-cell', 'single cell', 'scrna', '10x', 'cellranger'],
            'multi_modal_scrna': ['multiome', 'cite-seq', 'multimodal single cell'],
            
            # Spatial
            'spatial_transcriptomics': ['spatial transcriptom', 'spatial rna'],
            'spatial_visium': ['visium', '10x visium'],
            'spatial_xenium': ['xenium'],
            
            # Long read
            'long_read_rna_seq': ['nanopore rna', 'long read rna', 'direct rna'],
            'long_read_isoseq': ['isoseq', 'iso-seq', 'pacbio rna', 'isoform'],
            'long_read_assembly': ['nanopore assembly', 'pacbio assembly', 'long read assembly', 'flye', 'canu'],
        }
        
        detected_type = None
        confidence = 0.0
        
        for analysis_type, patterns in analysis_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    detected_type = analysis_type
                    confidence = 0.85
                    break
            if detected_type:
                break
        
        if not detected_type:
            # Generic analysis detection with looser matching
            if 'rna' in message_lower:
                detected_type = 'rna_seq_basic'
                confidence = 0.5
            elif 'chip' in message_lower:
                detected_type = 'chip_seq_peak_calling'
                confidence = 0.5
            elif 'variant' in message_lower or 'mutation' in message_lower:
                detected_type = 'wgs_variant_calling'
                confidence = 0.5
            elif any(word in message_lower for word in ['analyze', 'analysis', 'pipeline', 'workflow']):
                detected_type = 'generic'
                confidence = 0.4
        
        if not detected_type:
            return None
        
        # Extract organism
        organism = None
        organism_patterns = {
            'human': ['human', 'homo sapiens', 'hg38', 'grch38', 'patient', 'clinical'],
            'mouse': ['mouse', 'mus musculus', 'mm10', 'murine'],
            'arabidopsis': ['arabidopsis', 'plant', 'tair'],
            'zebrafish': ['zebrafish', 'danio rerio'],
            'drosophila': ['drosophila', 'fruit fly', 'fly'],
            'yeast': ['yeast', 'saccharomyces', 'cerevisiae'],
            'e_coli': ['e. coli', 'e coli', 'escherichia'],
        }
        
        for org, patterns in organism_patterns.items():
            for pattern in patterns:
                if pattern in message_lower:
                    organism = org
                    break
            if organism:
                break
        
        return {
            'analysis_type': detected_type,
            'organism': organism,
            'confidence': confidence,
            'raw_query': message,
        }
    
    def _generate_simple_workflow(self, intent: Dict[str, Any], tools: List[str]) -> str:
        """Generate a simple Nextflow workflow template for training data."""
        
        analysis_type = intent.get('analysis_type', 'generic')
        
        # Workflow templates by analysis type
        templates = {
            'rna_seq': '''
// RNA-seq Analysis Workflow
nextflow.enable.dsl=2

params.reads = "data/*_{R1,R2}.fastq.gz"
params.genome = "references/genome.fa"
params.gtf = "references/genes.gtf"
params.outdir = "results"

process FASTQC {{
    input:
    tuple val(sample_id), path(reads)
    
    output:
    path("*_fastqc.*")
    
    script:
    \"\"\"
    fastqc -t 4 ${{reads}}
    \"\"\"
}}

process STAR_ALIGN {{
    input:
    tuple val(sample_id), path(reads)
    path genome_index
    
    output:
    tuple val(sample_id), path("*.bam")
    
    script:
    \"\"\"
    STAR --genomeDir ${{genome_index}} \\
         --readFilesIn ${{reads[0]}} ${{reads[1]}} \\
         --outFileNamePrefix ${{sample_id}} \\
         --outSAMtype BAM SortedByCoordinate
    \"\"\"
}}

process FEATURECOUNTS {{
    input:
    tuple val(sample_id), path(bam)
    path gtf
    
    output:
    path("*.counts.txt")
    
    script:
    \"\"\"
    featureCounts -T 4 -p -a ${{gtf}} -o ${{sample_id}}.counts.txt ${{bam}}
    \"\"\"
}}

workflow {{
    reads_ch = Channel.fromFilePairs(params.reads)
    
    FASTQC(reads_ch)
    STAR_ALIGN(reads_ch, params.genome)
    FEATURECOUNTS(STAR_ALIGN.out, params.gtf)
}}
''',
            'chip_seq': '''
// ChIP-seq Analysis Workflow
nextflow.enable.dsl=2

params.reads = "data/*_{R1,R2}.fastq.gz"
params.genome = "references/genome.fa"
params.outdir = "results"

process FASTQC {{
    input:
    tuple val(sample_id), path(reads)
    
    output:
    path("*_fastqc.*")
    
    script:
    \"\"\"
    fastqc -t 4 ${{reads}}
    \"\"\"
}}

process BWA_ALIGN {{
    input:
    tuple val(sample_id), path(reads)
    path genome
    
    output:
    tuple val(sample_id), path("*.bam")
    
    script:
    \"\"\"
    bwa mem -t 8 ${{genome}} ${{reads[0]}} ${{reads[1]}} | samtools sort -o ${{sample_id}}.bam
    \"\"\"
}}

process MACS2_CALLPEAK {{
    input:
    tuple val(sample_id), path(bam)
    
    output:
    path("*.narrowPeak")
    
    script:
    \"\"\"
    macs2 callpeak -t ${{bam}} -f BAMPE -g hs -n ${{sample_id}} --outdir .
    \"\"\"
}}

workflow {{
    reads_ch = Channel.fromFilePairs(params.reads)
    
    FASTQC(reads_ch)
    BWA_ALIGN(reads_ch, params.genome)
    MACS2_CALLPEAK(BWA_ALIGN.out)
}}
''',
            'variant_calling': '''
// Variant Calling Workflow
nextflow.enable.dsl=2

params.reads = "data/*_{R1,R2}.fastq.gz"
params.genome = "references/genome.fa"
params.dbsnp = "references/dbsnp.vcf.gz"
params.outdir = "results"

process BWA_ALIGN {{
    input:
    tuple val(sample_id), path(reads)
    path genome
    
    output:
    tuple val(sample_id), path("*.bam"), path("*.bai")
    
    script:
    \"\"\"
    bwa mem -t 8 -R "@RG\\\\tID:${{sample_id}}\\\\tSM:${{sample_id}}" ${{genome}} ${{reads[0]}} ${{reads[1]}} | \\
    samtools sort -o ${{sample_id}}.bam
    samtools index ${{sample_id}}.bam
    \"\"\"
}}

process MARK_DUPLICATES {{
    input:
    tuple val(sample_id), path(bam), path(bai)
    
    output:
    tuple val(sample_id), path("*.dedup.bam"), path("*.dedup.bai")
    
    script:
    \"\"\"
    gatk MarkDuplicates -I ${{bam}} -O ${{sample_id}}.dedup.bam -M metrics.txt
    samtools index ${{sample_id}}.dedup.bam
    \"\"\"
}}

process HAPLOTYPECALLER {{
    input:
    tuple val(sample_id), path(bam), path(bai)
    path genome
    
    output:
    tuple val(sample_id), path("*.vcf.gz")
    
    script:
    \"\"\"
    gatk HaplotypeCaller -R ${{genome}} -I ${{bam}} -O ${{sample_id}}.vcf.gz
    \"\"\"
}}

workflow {{
    reads_ch = Channel.fromFilePairs(params.reads)
    
    BWA_ALIGN(reads_ch, params.genome)
    MARK_DUPLICATES(BWA_ALIGN.out)
    HAPLOTYPECALLER(MARK_DUPLICATES.out, params.genome)
}}
''',
        }
        
        # Return template or generate generic
        if analysis_type in templates:
            return templates[analysis_type].strip()
        
        # Generic workflow with provided tools
        tool_processes = []
        for tool in (tools or ['FASTQC', 'ALIGN', 'PROCESS']):
            tool_processes.append(f'''
process {tool.upper().replace("-", "_")} {{
    input:
    path(input_file)
    
    output:
    path("*.out")
    
    script:
    \"\"\"
    echo "Running {tool}"
    \"\"\"
}}''')
        
        return f'''
// {analysis_type.replace("_", " ").title()} Analysis Workflow
nextflow.enable.dsl=2

params.input = "data/*"
params.outdir = "results"

{"".join(tool_processes)}

workflow {{
    input_ch = Channel.fromPath(params.input)
    // Add workflow logic here
}}
'''.strip()

    async def process_turn(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        turn_index: int,
    ) -> TurnResult:
        """Process a single conversation turn through the system."""
        
        result = TurnResult(
            turn_index=turn_index,
            user_message=user_message,
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Parse intent - use simple pattern-based approach
            intent_parsed = self._parse_intent_simple(user_message)
            if intent_parsed:
                result.intent_parsed = intent_parsed
                result.intent_confidence = intent_parsed.get('confidence', 0.6)
            
            # Step 2: Select tools using find_tools_for_analysis
            if 'tool_selector' in self.components and result.intent_parsed:
                try:
                    analysis_type = result.intent_parsed.get('analysis_type', '')
                    tools_dict = self.components['tool_selector'].find_tools_for_analysis(analysis_type)
                    # Flatten the dict of categories to list of tools
                    all_tools = []
                    for category, tool_list in tools_dict.items():
                        for t in tool_list:
                            all_tools.append(t.name if hasattr(t, 'name') else str(t))
                    result.tools_selected = all_tools[:10]
                except Exception as e:
                    result.warnings.append(f"Tool selection issue: {str(e)}")
            
            # Step 3: Try full composer if available
            if 'composer' in self.components:
                try:
                    composer_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.components['composer'].compose,
                            user_message
                        ),
                        timeout=self.config.turn_timeout_seconds
                    )
                    
                    if composer_result:
                        result.system_response = str(composer_result.get('response', ''))
                        result.workflow_generated = composer_result.get('workflow', '')
                        result.tools_selected = composer_result.get('tools', result.tools_selected)
                        
                except asyncio.TimeoutError:
                    result.warnings.append("Composer timed out")
                except Exception as e:
                    result.warnings.append(f"Composer issue: {str(e)}")
            
            # Step 4: Generate workflow if not done yet - use simpler approach
            if not result.workflow_generated and result.intent_parsed:
                try:
                    workflow = self._generate_simple_workflow(result.intent_parsed, result.tools_selected)
                    result.workflow_generated = workflow or ""
                except Exception as e:
                    result.warnings.append(f"Workflow generation issue: {str(e)}")
            
            # Build response if not set
            if not result.system_response:
                result.system_response = self._build_response(result)
            
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.error_type = type(e).__name__
            logger.error(f"Turn processing error: {e}\n{traceback.format_exc()}")
        
        result.latency_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _build_response(self, result: TurnResult) -> str:
        """Build a response from the processing results."""
        
        parts = []
        
        if result.intent_parsed:
            analysis = result.intent_parsed.get('analysis_type', 'analysis')
            parts.append(f"I'll help you with {analysis}.")
        
        if result.tools_selected:
            tools = ", ".join(result.tools_selected[:5])
            parts.append(f"Recommended tools: {tools}.")
        
        if result.workflow_generated:
            parts.append(f"\n```nextflow\n{result.workflow_generated[:500]}\n```")
        
        return " ".join(parts) if parts else "I understand your request."
    
    async def run_conversation(
        self,
        conversation: GeneratedConversation,
    ) -> ConversationResult:
        """Run a complete conversation through the system."""
        
        result = ConversationResult(
            conversation_id=conversation.id,
            scenario_id=conversation.scenario_id,
            category=conversation.category,
            analysis_type=conversation.analysis_type,
            started_at=datetime.now().isoformat(),
        )
        
        history = []
        
        for i, turn in enumerate(conversation.turns):
            if isinstance(turn, dict):
                role = turn.get('role', '')
                content = turn.get('content', '')
            else:
                role = turn.role
                content = turn.content
            
            if role == "user":
                # Process user turn
                turn_result = await self.process_turn(
                    content,
                    history,
                    i,
                )
                result.turn_results.append(turn_result)
                
                # Update history
                history.append({"role": "user", "content": content})
                history.append({"role": "assistant", "content": turn_result.system_response})
                
                # Track metrics
                result.total_latency_ms += turn_result.latency_ms
                result.total_tokens += turn_result.tokens_used
                
                if turn_result.success:
                    result.successful_turns += 1
                else:
                    result.failed_turns += 1
                    result.errors.append({
                        "turn": i,
                        "error_type": turn_result.error_type,
                        "message": turn_result.error_message,
                    })
                
                # Track workflow generation
                if turn_result.workflow_generated:
                    result.workflow_generated = True
                    result.final_workflow = turn_result.workflow_generated
                
                # Track tools
                if turn_result.tools_selected:
                    result.final_tools = turn_result.tools_selected
        
        # Calculate aggregates
        result.total_turns = len(result.turn_results)
        if result.total_turns > 0:
            result.avg_latency_ms = result.total_latency_ms / result.total_turns
        
        result.completed_successfully = result.failed_turns == 0 and result.workflow_generated
        result.completed_at = datetime.now().isoformat()
        
        # Identify gaps
        result.gaps_identified = self._identify_gaps(result)
        
        return result
    
    def _identify_gaps(self, result: ConversationResult) -> List[str]:
        """Identify gaps and issues from conversation result."""
        
        gaps = []
        
        # Check for failed turns
        if result.failed_turns > 0:
            gaps.append(f"Had {result.failed_turns} failed turns")
        
        # Check for low confidence
        low_confidence = [
            tr for tr in result.turn_results 
            if tr.intent_confidence < self.config.min_intent_confidence
        ]
        if low_confidence:
            gaps.append(f"{len(low_confidence)} turns with low intent confidence")
        
        # Check for missing workflow
        if not result.workflow_generated:
            gaps.append("No workflow was generated")
        
        # Check for missing tools
        if not result.final_tools:
            gaps.append("No tools were identified")
        
        # Check for slow responses
        slow_turns = [tr for tr in result.turn_results if tr.latency_ms > 5000]
        if slow_turns:
            gaps.append(f"{len(slow_turns)} turns took >5 seconds")
        
        # Check for warnings
        all_warnings = []
        for tr in result.turn_results:
            all_warnings.extend(tr.warnings)
        if all_warnings:
            unique_warnings = set(all_warnings)
            gaps.append(f"Warnings: {', '.join(list(unique_warnings)[:3])}")
        
        return gaps
    
    async def run_all(
        self,
        conversations: List[GeneratedConversation],
        save_results: bool = True,
    ) -> List[ConversationResult]:
        """Run all conversations and collect results."""
        
        results = []
        
        for i, conv in enumerate(conversations):
            logger.info(f"Running conversation {i+1}/{len(conversations)}: {conv.id}")
            
            try:
                result = await self.run_conversation(conv)
                results.append(result)
                
                if self.config.save_intermediate and save_results:
                    # Save incrementally
                    self._save_result(result)
                
            except Exception as e:
                logger.error(f"Error running conversation {conv.id}: {e}")
                # Create failed result
                results.append(ConversationResult(
                    conversation_id=conv.id,
                    scenario_id=conv.scenario_id,
                    category=conv.category,
                    analysis_type=conv.analysis_type,
                    completed_successfully=False,
                    errors=[{"turn": -1, "error_type": "ConversationError", "message": str(e)}],
                ))
        
        # Save final results
        if save_results:
            self._save_all_results(results)
        
        return results
    
    def _save_result(self, result: ConversationResult):
        """Save a single result."""
        
        output_file = self.config.output_dir / f"result_{result.conversation_id}.json"
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _save_all_results(self, results: List[ConversationResult]):
        """Save all results to a combined file."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.config.output_dir / f"run_results_{timestamp}.jsonl"
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + '\n')
        
        logger.info(f"Saved {len(results)} results to {output_file}")
        
        # Also save summary
        summary = self._generate_summary(results)
        summary_file = self.config.output_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
    
    def _generate_summary(self, results: List[ConversationResult]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        
        total = len(results)
        successful = sum(1 for r in results if r.completed_successfully)
        workflow_generated = sum(1 for r in results if r.workflow_generated)
        
        # Collect all gaps
        all_gaps = []
        for r in results:
            all_gaps.extend(r.gaps_identified)
        
        gap_counts = {}
        for gap in all_gaps:
            # Normalize gap text
            key = gap.split(':')[0] if ':' in gap else gap
            gap_counts[key] = gap_counts.get(key, 0) + 1
        
        # Collect errors by type
        error_types = {}
        for r in results:
            for error in r.errors:
                etype = error.get('error_type', 'Unknown')
                error_types[etype] = error_types.get(etype, 0) + 1
        
        # Performance stats
        latencies = [r.avg_latency_ms for r in results if r.avg_latency_ms > 0]
        
        # By category
        by_category = {}
        for r in results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "successful": 0}
            by_category[cat]["total"] += 1
            if r.completed_successfully:
                by_category[cat]["successful"] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_conversations": total,
            "successful_conversations": successful,
            "success_rate": successful / total if total > 0 else 0,
            "workflow_generation_rate": workflow_generated / total if total > 0 else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "gap_frequency": gap_counts,
            "error_types": error_types,
            "by_category": by_category,
            "top_issues": sorted(gap_counts.items(), key=lambda x: -x[1])[:10],
        }


async def run_conversations(
    conversations: List[GeneratedConversation],
    output_dir: Path = None,
) -> Tuple[List[ConversationResult], Dict[str, Any]]:
    """Convenience function to run conversations."""
    
    config = RunnerConfig()
    if output_dir:
        config.output_dir = output_dir
    
    runner = ConversationRunner(config)
    results = await runner.run_all(conversations)
    summary = runner._generate_summary(results)
    
    return results, summary


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        from .conversation_generator import ConversationGenerator, GeneratorConfig
        
        # Generate some test conversations
        generator = ConversationGenerator()
        conversations = await generator.generate_dataset(num_conversations=5)
        
        # Run them
        results, summary = await run_conversations(conversations)
        
        print(f"\n=== Run Summary ===")
        print(f"Total: {summary['total_conversations']}")
        print(f"Successful: {summary['successful_conversations']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Avg Latency: {summary['avg_latency_ms']:.0f}ms")
        print(f"\nTop Issues:")
        for issue, count in summary['top_issues'][:5]:
            print(f"  - {issue}: {count}")
    
    asyncio.run(main())
