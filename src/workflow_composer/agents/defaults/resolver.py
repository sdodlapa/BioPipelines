"""
Smart Default Resolver.

This module fills missing parameters with intelligent defaults instead of
asking clarifying questions. It follows the design philosophy of professional
chat agents like ChatGPT and Claude.

Author: BioPipelines Team
Date: November 2025
"""
from typing import Dict, Any, Optional, List, Tuple
import logging

from . import DefaultConfig, ResolvedDefaults


logger = logging.getLogger(__name__)


class SmartDefaultResolver:
    """
    Smart default resolver - fills missing parameters with intelligent defaults.
    
    Philosophy: Professional chat agents (ChatGPT, Claude, Copilot) rarely ask
    clarifying questions. They proceed with reasonable assumptions and let
    users course-correct naturally.
    
    Example Usage:
        resolver = SmartDefaultResolver()
        result = resolver.resolve({
            "workflow_type": "rna-seq",
            "query": "analyze my RNA data"
        })
        # result.filled_params = {
        #     "workflow_type": "rna-seq",
        #     "organism": "human",
        #     "analysis_type": "differential_expression",
        #     "genome_version": "GRCh38",
        #     "aligner": "STAR",
        # }
        # result.explanation = "Note: assuming human (most common), 
        #     using differential_expression (standard for rna-seq), 
        #     genome GRCh38. Let me know if you need different settings."
    """
    
    def __init__(self, config: Optional[DefaultConfig] = None):
        """
        Initialize the resolver.
        
        Args:
            config: Configuration for defaults. If None, uses DefaultConfig().
        """
        self.config = config or DefaultConfig()
    
    def resolve(
        self,
        parsed_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ResolvedDefaults:
        """
        Fill missing parameters with smart defaults.
        
        Args:
            parsed_result: Dictionary of parsed parameters from query
            context: Optional context (e.g., previous conversation turns)
            
        Returns:
            ResolvedDefaults with filled params and human-readable explanation
        """
        filled = dict(parsed_result)
        assumptions = []
        confidence = 1.0
        
        # Extract workflow type for context-aware defaults
        workflow = self._normalize_workflow(
            filled.get("workflow_type", "") or 
            filled.get("assay_type", "") or
            filled.get("analysis_type", "")
        )
        
        # 1. Resolve organism
        organism_result = self._resolve_organism(filled, context)
        if organism_result:
            filled["organism"] = organism_result[0]
            if organism_result[1]:  # assumption made
                assumptions.append(organism_result[1])
                confidence *= 0.9
        
        # 2. Resolve analysis type based on workflow
        analysis_result = self._resolve_analysis_type(filled, workflow)
        if analysis_result:
            filled["analysis_type"] = analysis_result[0]
            if analysis_result[1]:
                assumptions.append(analysis_result[1])
                confidence *= 0.95
        
        # 3. Resolve genome version based on organism
        genome_result = self._resolve_genome_version(filled)
        if genome_result:
            filled["genome_version"] = genome_result[0]
            if genome_result[1]:
                assumptions.append(genome_result[1])
        
        # 4. Resolve aligner based on workflow
        aligner_result = self._resolve_aligner(filled, workflow)
        if aligner_result:
            filled["aligner"] = aligner_result[0]
            if aligner_result[1]:
                assumptions.append(aligner_result[1])
        
        # 5. Resolve quality parameters
        quality_result = self._resolve_quality_params(filled)
        filled.update(quality_result)
        
        # Build explanation
        explanation = self._build_explanation(assumptions)
        
        return ResolvedDefaults(
            filled_params=filled,
            assumptions=assumptions,
            explanation=explanation,
            confidence=confidence,
        )
    
    def _normalize_workflow(self, workflow: str) -> str:
        """Normalize workflow type string."""
        if not workflow:
            return ""
        
        # Lowercase and normalize common variations
        workflow = workflow.lower().strip()
        workflow = workflow.replace("_", "-").replace(" ", "-")
        
        # Map common aliases
        aliases = {
            "rnaseq": "rna-seq",
            "rna_seq": "rna-seq",
            "chipseq": "chip-seq",
            "chip_seq": "chip-seq",
            "atacseq": "atac-seq",
            "atac_seq": "atac-seq",
            "dnaseq": "dna-seq",
            "dna_seq": "dna-seq",
            "scrnaseq": "scrna-seq",
            "scrna_seq": "scrna-seq",
            "single-cell-rna-seq": "scrna-seq",
        }
        
        return aliases.get(workflow, workflow)
    
    def _resolve_organism(
        self,
        filled: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Optional[Tuple[str, Optional[str]]]:
        """
        Resolve organism parameter.
        
        Returns:
            Tuple of (organism, assumption_text) or None
        """
        current = filled.get("organism", "")
        
        if current:
            # Normalize organism name
            current_lower = current.lower().strip()
            if current_lower in self.config.organism_aliases:
                normalized = self.config.organism_aliases[current_lower]
                return (normalized, None)  # No assumption, just normalized
            return (current, None)
        
        # Check context for previously mentioned organism
        if context and context.get("organism"):
            return (context["organism"], f"using {context['organism']} (from previous context)")
        
        # Default to human (most common in bioinformatics)
        return (self.config.default_organism, f"assuming {self.config.default_organism} (most common)")
    
    def _resolve_analysis_type(
        self,
        filled: Dict[str, Any],
        workflow: str
    ) -> Optional[Tuple[str, Optional[str]]]:
        """
        Resolve analysis type based on workflow.
        
        Returns:
            Tuple of (analysis_type, assumption_text) or None
        """
        current = filled.get("analysis_type", "")
        
        if current:
            return (current, None)
        
        if not workflow:
            return None
        
        # Get default for this workflow
        if workflow in self.config.workflow_defaults:
            default_analysis = self.config.workflow_defaults[workflow]
            return (default_analysis, f"using {default_analysis} (standard for {workflow})")
        
        return None
    
    def _resolve_genome_version(
        self,
        filled: Dict[str, Any]
    ) -> Optional[Tuple[str, Optional[str]]]:
        """
        Resolve genome version based on organism.
        
        Returns:
            Tuple of (genome_version, assumption_text) or None
        """
        current = filled.get("genome_version", "") or filled.get("genome", "")
        
        if current:
            return (current, None)
        
        organism = filled.get("organism", "")
        if organism and organism in self.config.genome_versions:
            version = self.config.genome_versions[organism]
            return (version, f"genome {version}")
        
        return None
    
    def _resolve_aligner(
        self,
        filled: Dict[str, Any],
        workflow: str
    ) -> Optional[Tuple[str, Optional[str]]]:
        """
        Resolve aligner based on workflow type.
        
        Returns:
            Tuple of (aligner, assumption_text) or None
        """
        current = filled.get("aligner", "") or filled.get("alignment_tool", "")
        
        if current:
            return (current, None)
        
        if not workflow:
            return None
        
        if workflow in self.config.workflow_aligners:
            aligner = self.config.workflow_aligners[workflow]
            return (aligner, None)  # Don't mention aligner unless user asks
        
        return None
    
    def _resolve_quality_params(
        self,
        filled: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve quality parameters with sensible defaults.
        
        Returns:
            Dictionary of quality parameters
        """
        result = {}
        
        # Only set if not already present
        if "quality_threshold" not in filled and "min_quality" not in filled:
            result["quality_threshold"] = self.config.default_quality_threshold
        
        if "min_read_length" not in filled and "min_length" not in filled:
            result["min_read_length"] = self.config.default_min_read_length
        
        if "threads" not in filled and "cores" not in filled:
            result["threads"] = self.config.default_threads
        
        return result
    
    def _build_explanation(self, assumptions: List[str]) -> str:
        """
        Build human-readable explanation of assumptions.
        
        Args:
            assumptions: List of assumption strings
            
        Returns:
            Formatted explanation string
        """
        if not assumptions:
            return ""
        
        # Format assumptions nicely
        if len(assumptions) == 1:
            return f"Note: {assumptions[0]}. Let me know if you need different settings."
        
        # Join multiple assumptions
        formatted = ", ".join(assumptions[:-1])
        formatted += f", and {assumptions[-1]}"
        
        return f"Note: {formatted}. Let me know if you need different settings."
    
    def should_offer_alternatives(
        self,
        parsed_result: Dict[str, Any]
    ) -> bool:
        """
        Check if we should offer alternatives (not ask, just mention).
        
        Only when there are 2+ equally valid interpretations.
        
        Args:
            parsed_result: Dictionary of parsed parameters
            
        Returns:
            True if alternatives should be offered
        """
        # Example scenarios:
        # 1. Multiple datasets found matching criteria
        if parsed_result.get("ambiguous_matches"):
            return True
        
        # 2. Multiple valid workflows for the query
        if parsed_result.get("alternative_workflows"):
            return True
        
        return False
    
    def format_alternatives(
        self,
        alternatives: List[str],
        max_show: int = 3
    ) -> str:
        """
        Format alternatives as suggestions (not questions).
        
        Args:
            alternatives: List of alternative options
            max_show: Maximum number to show explicitly
            
        Returns:
            Formatted alternatives string
        """
        if not alternatives:
            return ""
        
        if len(alternatives) <= max_show:
            if len(alternatives) == 1:
                return f"Alternative available: {alternatives[0]}"
            else:
                alt_str = ", ".join(alternatives[:-1]) + f" or {alternatives[-1]}"
                return f"Other options available: {alt_str}"
        else:
            shown = ", ".join(alternatives[:max_show-1])
            remaining = len(alternatives) - max_show + 1
            return f"Other options: {shown}, or {remaining} others"
    
    def get_workflow_suggestions(
        self,
        query: str
    ) -> List[str]:
        """
        Get workflow suggestions based on query keywords.
        
        Args:
            query: User's input query
            
        Returns:
            List of suggested workflows
        """
        query_lower = query.lower()
        suggestions = []
        
        # Keyword to workflow mapping
        keyword_workflows = {
            ("rna", "expression", "transcript", "gene"): "rna-seq",
            ("chip", "histone", "binding", "tf"): "chip-seq",
            ("atac", "chromatin", "accessibility"): "atac-seq",
            ("variant", "snp", "mutation", "dna"): "dna-seq",
            ("methylation", "bisulfite", "5mc"): "methylation",
            ("single-cell", "scRNA", "10x", "cell"): "scrna-seq",
            ("hi-c", "contact", "3d", "chromatin"): "hic",
        }
        
        for keywords, workflow in keyword_workflows.items():
            if any(kw in query_lower for kw in keywords):
                suggestions.append(workflow)
        
        return suggestions
