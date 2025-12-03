"""
Conversation Generator
======================

Generates realistic multi-turn dialogues (5-20 turns) that simulate
user interactions with the BioPipelines chat agent.

These conversations are used for:
1. Training data collection for LLM fine-tuning
2. System testing to identify gaps and errors
3. Evaluating conversation flow and handling
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import asyncio

from .config import GeneratorConfig

logger = logging.getLogger(__name__)


class ConversationPattern(Enum):
    """Patterns for conversation flow."""
    
    SIMPLE_DIRECT = "simple_direct"           # User knows what they want
    EXPLORATORY = "exploratory"               # User is exploring options
    CLARIFICATION_NEEDED = "clarification"    # Requires back-and-forth
    PARAMETER_REFINEMENT = "refinement"       # Iterative parameter tuning
    ERROR_RECOVERY = "error_recovery"         # Handles errors/issues
    COMPARISON = "comparison"                 # Comparing approaches
    MULTI_STEP_ANALYSIS = "multi_step"        # Complex multi-part workflow


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationScenario:
    """A scenario template for generating conversations."""
    
    id: str
    category: str
    analysis_type: str
    pattern: ConversationPattern
    initial_query: str
    expected_tools: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    difficulty: int = 1  # 1-5
    
    # Conversation flow hints
    clarification_points: List[str] = field(default_factory=list)
    refinement_options: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)


@dataclass
class GeneratedConversation:
    """A complete generated conversation."""
    
    id: str
    scenario_id: str
    pattern: ConversationPattern
    turns: List[ConversationTurn] = field(default_factory=list)
    
    # Metadata
    category: str = ""
    analysis_type: str = ""
    total_turns: int = 0
    
    # Quality signals
    completed_successfully: bool = False
    workflow_generated: bool = False
    tools_identified: List[str] = field(default_factory=list)
    
    # For analysis
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['pattern'] = self.pattern.value
        data['turns'] = [t if isinstance(t, dict) else t.to_dict() for t in self.turns]
        return data
    
    def to_training_format(self, system_prompt: str = "") -> Dict[str, Any]:
        """Convert to training format."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for turn in self.turns:
            if isinstance(turn, dict):
                messages.append({"role": turn['role'], "content": turn['content']})
            else:
                messages.append({"role": turn.role, "content": turn.content})
        
        return {"messages": messages}


# Scenario templates for different analysis types
CONVERSATION_SCENARIOS = [
    # RNA-seq scenarios
    ConversationScenario(
        id="rnaseq_de_basic",
        category="rna_seq",
        analysis_type="differential_expression",
        pattern=ConversationPattern.SIMPLE_DIRECT,
        initial_query="I want to analyze RNA-seq data to find differentially expressed genes between treated and control samples",
        expected_tools=["fastp", "star", "featurecounts", "deseq2"],
        parameters={"organism": "human", "conditions": 2, "replicates": 3},
        difficulty=1,
        clarification_points=["organism", "sequencing_type", "replicate_count"],
    ),
    ConversationScenario(
        id="rnaseq_exploratory",
        category="rna_seq",
        analysis_type="differential_expression",
        pattern=ConversationPattern.EXPLORATORY,
        initial_query="I have some sequencing data from a mouse experiment. Not sure what analysis I need.",
        expected_tools=["fastp", "star", "featurecounts", "deseq2"],
        parameters={"organism": "mouse"},
        difficulty=2,
        clarification_points=["data_type", "experimental_design", "research_question"],
    ),
    ConversationScenario(
        id="rnaseq_parameter_tuning",
        category="rna_seq",
        analysis_type="differential_expression",
        pattern=ConversationPattern.PARAMETER_REFINEMENT,
        initial_query="Set up RNA-seq analysis but I want to customize the alignment parameters",
        expected_tools=["fastp", "star", "featurecounts", "deseq2"],
        parameters={"organism": "human", "custom_star_params": True},
        difficulty=3,
        refinement_options=["alignment_stringency", "multimapping", "strand_specificity"],
    ),
    
    # ChIP-seq scenarios
    ConversationScenario(
        id="chipseq_histone",
        category="chip_seq",
        analysis_type="peak_calling",
        pattern=ConversationPattern.CLARIFICATION_NEEDED,
        initial_query="I need to analyze ChIP-seq data",
        expected_tools=["fastp", "bwa", "macs2", "deeptools"],
        parameters={"target_type": "histone"},
        difficulty=2,
        clarification_points=["target_protein", "control_samples", "peak_type"],
    ),
    ConversationScenario(
        id="chipseq_tf",
        category="chip_seq",
        analysis_type="peak_calling",
        pattern=ConversationPattern.SIMPLE_DIRECT,
        initial_query="Analyze transcription factor ChIP-seq for p53 in human cells with input control",
        expected_tools=["fastp", "bwa", "macs2", "homer"],
        parameters={"target_type": "tf", "target": "p53", "organism": "human"},
        difficulty=2,
    ),
    
    # Variant calling scenarios
    ConversationScenario(
        id="variant_germline",
        category="dna_seq",
        analysis_type="variant_calling",
        pattern=ConversationPattern.SIMPLE_DIRECT,
        initial_query="I need to identify germline variants in whole exome sequencing data from a family trio",
        expected_tools=["fastp", "bwa", "gatk", "vep"],
        parameters={"variant_type": "germline", "sample_type": "trio", "capture": "wes"},
        difficulty=2,
    ),
    ConversationScenario(
        id="variant_somatic",
        category="dna_seq",
        analysis_type="variant_calling",
        pattern=ConversationPattern.MULTI_STEP_ANALYSIS,
        initial_query="Identify somatic mutations in tumor samples compared to matched normal",
        expected_tools=["fastp", "bwa", "mutect2", "funcotator"],
        parameters={"variant_type": "somatic", "sample_type": "tumor_normal"},
        difficulty=3,
        clarification_points=["tumor_purity", "coverage", "panel_of_normals"],
    ),
    
    # ATAC-seq scenarios
    ConversationScenario(
        id="atacseq_basic",
        category="atac_seq",
        analysis_type="accessibility",
        pattern=ConversationPattern.SIMPLE_DIRECT,
        initial_query="Analyze ATAC-seq data to identify open chromatin regions in neurons",
        expected_tools=["fastp", "bowtie2", "macs2", "deeptools"],
        parameters={"cell_type": "neurons", "organism": "mouse"},
        difficulty=2,
    ),
    
    # Metagenomics scenarios
    ConversationScenario(
        id="metagenome_16s",
        category="metagenomics",
        analysis_type="taxonomic_profiling",
        pattern=ConversationPattern.CLARIFICATION_NEEDED,
        initial_query="I have microbiome sequencing data from gut samples",
        expected_tools=["dada2", "qiime2", "phyloseq"],
        parameters={"sequencing_type": "16s"},
        difficulty=2,
        clarification_points=["sequencing_method", "sample_count", "comparison_groups"],
    ),
    ConversationScenario(
        id="metagenome_shotgun",
        category="metagenomics",
        analysis_type="functional_profiling",
        pattern=ConversationPattern.MULTI_STEP_ANALYSIS,
        initial_query="Perform functional analysis of shotgun metagenomics data from soil samples",
        expected_tools=["fastp", "kraken2", "humann3", "metaphlan"],
        parameters={"sequencing_type": "shotgun", "sample_type": "soil"},
        difficulty=4,
    ),
    
    # Single-cell scenarios
    ConversationScenario(
        id="scrna_clustering",
        category="scrna_seq",
        analysis_type="cell_clustering",
        pattern=ConversationPattern.EXPLORATORY,
        initial_query="I have 10x Genomics single-cell RNA-seq data and want to identify cell types",
        expected_tools=["cellranger", "seurat", "scanpy"],
        parameters={"platform": "10x", "analysis": "clustering"},
        difficulty=3,
        clarification_points=["expected_cell_types", "doublet_rate", "batch_effects"],
    ),
    
    # Methylation scenarios
    ConversationScenario(
        id="methylation_wgbs",
        category="methylation",
        analysis_type="differential_methylation",
        pattern=ConversationPattern.PARAMETER_REFINEMENT,
        initial_query="Analyze whole genome bisulfite sequencing data for differential methylation",
        expected_tools=["bismark", "methylkit", "dmrseq"],
        parameters={"method": "wgbs"},
        difficulty=4,
        refinement_options=["dmr_size", "coverage_threshold", "context_type"],
    ),
    
    # Long-read scenarios
    ConversationScenario(
        id="longread_assembly",
        category="long_read",
        analysis_type="genome_assembly",
        pattern=ConversationPattern.COMPARISON,
        initial_query="Assemble a bacterial genome from Oxford Nanopore reads",
        expected_tools=["flye", "medaka", "quast"],
        parameters={"platform": "nanopore", "organism_type": "bacteria"},
        difficulty=3,
        refinement_options=["polishing_rounds", "hybrid_assembly", "coverage"],
    ),
    
    # Error recovery scenarios
    ConversationScenario(
        id="error_wrong_format",
        category="rna_seq",
        analysis_type="differential_expression",
        pattern=ConversationPattern.ERROR_RECOVERY,
        initial_query="My RNA-seq pipeline is failing with a BAM file error",
        expected_tools=["samtools", "picard"],
        parameters={},
        difficulty=2,
        edge_cases=["corrupted_file", "wrong_format", "missing_index"],
    ),
    ConversationScenario(
        id="error_memory",
        category="dna_seq",
        analysis_type="variant_calling",
        pattern=ConversationPattern.ERROR_RECOVERY,
        initial_query="GATK keeps running out of memory on my variant calling",
        expected_tools=["gatk"],
        parameters={},
        difficulty=3,
        edge_cases=["memory_limit", "parallel_gc", "interval_splitting"],
    ),
    
    # Comparison scenarios
    ConversationScenario(
        id="compare_aligners",
        category="rna_seq",
        analysis_type="alignment",
        pattern=ConversationPattern.COMPARISON,
        initial_query="Should I use STAR or HISAT2 for my RNA-seq alignment?",
        expected_tools=["star", "hisat2"],
        parameters={},
        difficulty=2,
    ),
]


class ConversationGenerator:
    """Generates realistic multi-turn conversations."""
    
    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self.scenarios = CONVERSATION_SCENARIOS
        self._llm_client = None
        
        # Load system components for realistic responses
        self._load_system_components()
    
    def _load_system_components(self):
        """Load system components for generating responses."""
        try:
            from ..core.query_parser import QueryParser, AnalysisType
            from ..core.tool_selector import ToolSelector
            from ..core.workflow_generator import WorkflowGenerator
            
            self.query_parser = QueryParser()
            self.tool_selector = ToolSelector()
            self.workflow_generator = WorkflowGenerator()
            self._components_loaded = True
        except ImportError as e:
            logger.warning(f"Could not load all system components: {e}")
            self._components_loaded = False
    
    def _get_llm_client(self):
        """Get or create LLM client for generation."""
        if self._llm_client is None:
            try:
                from ..agents.model_service_manager import ModelServiceManager
                manager = ModelServiceManager()
                self._llm_client = manager
            except Exception as e:
                logger.warning(f"Could not create LLM client: {e}")
        return self._llm_client
    
    def generate_user_followups(
        self, 
        scenario: ConversationScenario,
        assistant_response: str,
        turn_number: int,
        max_turns: int
    ) -> List[str]:
        """Generate realistic user follow-up messages."""
        
        followups = []
        pattern = scenario.pattern
        
        if pattern == ConversationPattern.SIMPLE_DIRECT:
            followups = [
                "That looks good. Can you generate the workflow?",
                "Perfect, let's proceed with that configuration.",
                "Yes, that's exactly what I need.",
                "Can you show me the complete pipeline?",
            ]
        
        elif pattern == ConversationPattern.EXPLORATORY:
            if turn_number < max_turns // 2:
                followups = [
                    "What are my options for this type of data?",
                    "Can you explain what each analysis would tell me?",
                    "I'm not sure which approach is best for my research question",
                    f"The data is from {scenario.parameters.get('organism', 'human')} samples",
                    "I have about 20 million reads per sample",
                ]
            else:
                followups = [
                    "I think I want to do differential expression analysis",
                    "Let's go with the standard pipeline you suggested",
                    "That makes sense, can we proceed?",
                ]
        
        elif pattern == ConversationPattern.CLARIFICATION_NEEDED:
            if scenario.clarification_points:
                point = random.choice(scenario.clarification_points)
                clarification_responses = {
                    "organism": f"It's {scenario.parameters.get('organism', 'human')} data",
                    "sequencing_type": "It's paired-end Illumina sequencing",
                    "replicate_count": "I have 3 biological replicates per condition",
                    "target_protein": "It's for histone H3K27ac",
                    "control_samples": "Yes, I have input controls for each sample",
                    "peak_type": "Looking for broad peaks",
                    "data_type": "It's RNA-seq data",
                    "experimental_design": "Treatment vs control with 4 samples each",
                    "research_question": "I want to find genes that change expression",
                    "sequencing_method": "It's 16S amplicon sequencing, V3-V4 region",
                    "sample_count": "I have 50 samples across 5 groups",
                    "comparison_groups": "Comparing healthy vs disease states",
                }
                followups = [clarification_responses.get(point, f"Let me clarify: {point}")]
            else:
                followups = [
                    "Let me provide more details...",
                    "The samples are from a case-control study",
                ]
        
        elif pattern == ConversationPattern.PARAMETER_REFINEMENT:
            if scenario.refinement_options and turn_number < max_turns - 2:
                option = random.choice(scenario.refinement_options)
                refinement_responses = {
                    "alignment_stringency": "Can we make the alignment more stringent?",
                    "multimapping": "How should we handle multi-mapping reads?",
                    "strand_specificity": "The library is strand-specific, reverse stranded",
                    "dmr_size": "I'm looking for regions at least 500bp",
                    "coverage_threshold": "Require at least 10x coverage",
                    "context_type": "Focus on CpG context only",
                    "polishing_rounds": "Can we do more polishing rounds for accuracy?",
                    "hybrid_assembly": "I also have some Illumina data for polishing",
                }
                followups = [refinement_responses.get(option, f"Can we adjust {option}?")]
            else:
                followups = [
                    "These parameters look good now",
                    "Let's run with these settings",
                ]
        
        elif pattern == ConversationPattern.ERROR_RECOVERY:
            if scenario.edge_cases:
                case = random.choice(scenario.edge_cases)
                error_responses = {
                    "corrupted_file": "The file seems to be corrupted. How do I check?",
                    "wrong_format": "I converted from CRAM, could that be the issue?",
                    "missing_index": "Do I need to create an index file?",
                    "memory_limit": "I only have 32GB RAM available",
                    "parallel_gc": "Should I adjust the garbage collection?",
                    "interval_splitting": "Can we split the analysis by chromosome?",
                }
                followups = [error_responses.get(case, "How do I fix this?")]
            else:
                followups = [
                    "I tried that but still getting errors",
                    "The error message mentions something about file format",
                ]
        
        elif pattern == ConversationPattern.COMPARISON:
            followups = [
                "What are the trade-offs between these options?",
                "Which one is faster for large datasets?",
                "Which has better accuracy in your experience?",
                "I'll go with your recommendation",
                "Let's use the one that's better for novel transcripts",
            ]
        
        elif pattern == ConversationPattern.MULTI_STEP_ANALYSIS:
            if turn_number < max_turns // 2:
                followups = [
                    "What's the first step in this pipeline?",
                    "How long will each step take approximately?",
                    "Are there any quality checks between steps?",
                    f"The {random.choice(['tumor', 'control'])} samples have higher coverage",
                ]
            else:
                followups = [
                    "Let's proceed with the full pipeline",
                    "Can you show the complete workflow now?",
                    "I understand, let's generate the final configuration",
                ]
        
        return followups
    
    async def generate_assistant_response(
        self,
        scenario: ConversationScenario,
        conversation_history: List[ConversationTurn],
        user_message: str,
    ) -> str:
        """Generate assistant response using actual system or LLM."""
        
        # Try to use actual system components first
        if self._components_loaded:
            try:
                return await self._generate_with_system(scenario, conversation_history, user_message)
            except Exception as e:
                logger.warning(f"System generation failed: {e}")
        
        # Fall back to template-based responses
        return self._generate_template_response(scenario, conversation_history, user_message)
    
    async def _generate_with_system(
        self,
        scenario: ConversationScenario,
        history: List[ConversationTurn],
        user_message: str,
    ) -> str:
        """Generate response using actual system components."""
        
        # Parse intent from the full conversation context
        full_context = "\n".join([
            f"{t.role}: {t.content}" for t in history
        ]) + f"\nuser: {user_message}"
        
        # Use query parser
        intent = self.query_parser.parse(user_message)
        
        response_parts = []
        
        # Acknowledge the user's input
        if len(history) <= 2:
            response_parts.append(f"I understand you want to perform {intent.analysis_type.value if hasattr(intent, 'analysis_type') else 'analysis'}.")
        
        # Check if we need clarification
        if scenario.pattern == ConversationPattern.CLARIFICATION_NEEDED:
            remaining_points = [p for p in scenario.clarification_points 
                              if p.lower() not in full_context.lower()]
            if remaining_points:
                point = remaining_points[0]
                questions = {
                    "organism": "What organism are your samples from?",
                    "target_protein": "What protein or histone mark are you studying?",
                    "control_samples": "Do you have control/input samples?",
                    "peak_type": "Are you looking for narrow or broad peaks?",
                    "sequencing_method": "What sequencing method was used (16S, shotgun)?",
                }
                response_parts.append(questions.get(point, f"Could you tell me more about {point}?"))
                return " ".join(response_parts)
        
        # Select tools
        tools = self.tool_selector.select_tools(intent)
        if tools:
            tool_names = [t.name if hasattr(t, 'name') else str(t) for t in tools[:5]]
            response_parts.append(f"I recommend using: {', '.join(tool_names)}.")
        
        # Generate workflow if late in conversation
        if len(history) >= 4:
            try:
                workflow = self.workflow_generator.generate(intent)
                if workflow:
                    response_parts.append(f"\nHere's the workflow:\n```nextflow\n{workflow[:500]}...\n```")
            except Exception as e:
                logger.debug(f"Workflow generation: {e}")
        
        return " ".join(response_parts) if response_parts else self._generate_template_response(scenario, history, user_message)
    
    def _generate_template_response(
        self,
        scenario: ConversationScenario,
        history: List[ConversationTurn],
        user_message: str,
    ) -> str:
        """Generate template-based response."""
        
        turn_count = len(history)
        pattern = scenario.pattern
        
        # Opening responses
        if turn_count == 0:
            openings = {
                ConversationPattern.SIMPLE_DIRECT: f"I'll set up a {scenario.analysis_type} workflow for you. Based on your requirements, I recommend using {', '.join(scenario.expected_tools[:3])}.",
                ConversationPattern.EXPLORATORY: "I'd be happy to help you analyze your data. Could you tell me more about your research question and what you're hoping to discover?",
                ConversationPattern.CLARIFICATION_NEEDED: f"I can help with {scenario.category.replace('_', ' ')} analysis. To set up the best pipeline, I need a few more details.",
                ConversationPattern.PARAMETER_REFINEMENT: f"I'll configure a {scenario.analysis_type} pipeline with customizable parameters. Let me show you the default settings first.",
                ConversationPattern.ERROR_RECOVERY: "I understand you're encountering an error. Let's troubleshoot this together. Can you share the exact error message?",
                ConversationPattern.COMPARISON: "Both tools have their strengths. Let me compare them for your specific use case.",
                ConversationPattern.MULTI_STEP_ANALYSIS: f"This analysis will involve several steps. Let me walk you through the {scenario.analysis_type} pipeline.",
            }
            return openings.get(pattern, "I'll help you set up this analysis.")
        
        # Mid-conversation responses
        if turn_count < 6:
            if pattern == ConversationPattern.CLARIFICATION_NEEDED:
                points = scenario.clarification_points
                asked = sum(1 for t in history if t.role == "assistant")
                if asked < len(points):
                    point = points[min(asked, len(points)-1)]
                    return f"Thanks for that information. Could you also tell me about {point.replace('_', ' ')}?"
            
            if pattern == ConversationPattern.PARAMETER_REFINEMENT:
                return f"I've noted that customization. The current configuration uses: {', '.join(scenario.expected_tools)}. Would you like to adjust any other parameters?"
            
            return f"Good. I'm configuring the pipeline with {', '.join(scenario.expected_tools[:2])} and other appropriate tools."
        
        # Closing responses with workflow
        tools_str = ", ".join(scenario.expected_tools)
        workflow_snippet = f"""
include {{ {scenario.expected_tools[0].upper()} }} from './modules/{scenario.expected_tools[0]}'

workflow {{
    // {scenario.analysis_type.replace('_', ' ').title()} Pipeline
    {scenario.expected_tools[0].upper()}(input_files)
}}
"""
        return f"Based on our discussion, here's your {scenario.analysis_type} workflow using {tools_str}:\n\n```nextflow{workflow_snippet}```\n\nThis pipeline is configured for your {scenario.parameters.get('organism', 'sample')} data."
    
    async def generate_conversation(
        self,
        scenario: ConversationScenario,
        min_turns: int = 5,
        max_turns: int = 15,
    ) -> GeneratedConversation:
        """Generate a complete multi-turn conversation."""
        
        conversation = GeneratedConversation(
            id=f"conv_{scenario.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scenario_id=scenario.id,
            pattern=scenario.pattern,
            category=scenario.category,
            analysis_type=scenario.analysis_type,
        )
        
        # Determine number of turns
        target_turns = random.randint(min_turns, max_turns)
        
        # Start with initial query
        conversation.turns.append(ConversationTurn(
            role="user",
            content=scenario.initial_query,
        ))
        
        # Generate conversation
        for turn_num in range(target_turns):
            # Generate assistant response
            assistant_response = await self.generate_assistant_response(
                scenario,
                conversation.turns,
                conversation.turns[-1].content if conversation.turns else "",
            )
            
            conversation.turns.append(ConversationTurn(
                role="assistant",
                content=assistant_response,
            ))
            
            # Check if we should end
            if turn_num >= target_turns - 1:
                break
            
            if "workflow" in assistant_response.lower() and "```" in assistant_response:
                # Workflow generated, maybe end or continue
                if random.random() > 0.3:
                    # User accepts
                    conversation.turns.append(ConversationTurn(
                        role="user",
                        content=random.choice([
                            "Perfect, thank you!",
                            "This looks great. I'll run this pipeline.",
                            "Thanks, this is exactly what I needed.",
                        ])
                    ))
                    conversation.workflow_generated = True
                    break
            
            # Generate user follow-up
            followups = self.generate_user_followups(
                scenario,
                assistant_response,
                turn_num,
                target_turns,
            )
            
            if followups:
                conversation.turns.append(ConversationTurn(
                    role="user",
                    content=random.choice(followups),
                ))
        
        conversation.total_turns = len(conversation.turns)
        conversation.tools_identified = scenario.expected_tools
        conversation.completed_successfully = conversation.workflow_generated or len(conversation.turns) >= min_turns
        
        return conversation
    
    async def generate_dataset(
        self,
        num_conversations: int = 100,
        output_path: Path = None,
        min_turns: int = 5,
        max_turns: int = 20,
    ) -> List[GeneratedConversation]:
        """Generate a full dataset of conversations."""
        
        output_path = output_path or self.config.output_dir / "conversations"
        output_path.mkdir(parents=True, exist_ok=True)
        
        conversations = []
        
        # Generate conversations from each scenario
        scenarios_per_batch = num_conversations // len(self.scenarios) + 1
        
        for scenario in self.scenarios:
            for i in range(scenarios_per_batch):
                if len(conversations) >= num_conversations:
                    break
                
                try:
                    # Vary the turn count
                    conv = await self.generate_conversation(
                        scenario,
                        min_turns=min_turns,
                        max_turns=max_turns,
                    )
                    conversations.append(conv)
                    
                    if len(conversations) % 10 == 0:
                        logger.info(f"Generated {len(conversations)}/{num_conversations} conversations")
                
                except Exception as e:
                    logger.error(f"Error generating conversation for {scenario.id}: {e}")
        
        # Save conversations
        output_file = output_path / f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(output_file, 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv.to_dict()) + '\n')
        
        logger.info(f"Saved {len(conversations)} conversations to {output_file}")
        
        # Also save in training format
        training_file = output_path / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(training_file, 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv.to_training_format()) + '\n')
        
        logger.info(f"Saved training format to {training_file}")
        
        return conversations


async def generate_conversations(
    num_conversations: int = 100,
    output_dir: Path = None,
) -> List[GeneratedConversation]:
    """Convenience function to generate conversations."""
    
    config = GeneratorConfig()
    if output_dir:
        config.output_dir = output_dir
    
    generator = ConversationGenerator(config)
    return await generator.generate_dataset(num_conversations)


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        conversations = await generate_conversations(num_conversations=20)
        print(f"\nGenerated {len(conversations)} conversations")
        
        # Show sample
        if conversations:
            conv = conversations[0]
            print(f"\nSample conversation ({conv.total_turns} turns):")
            for turn in conv.turns[:4]:
                print(f"  {turn.role}: {turn.content[:100]}...")
    
    asyncio.run(main())
