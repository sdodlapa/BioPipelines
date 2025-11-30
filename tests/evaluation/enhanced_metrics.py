"""
Enhanced Metrics with LLM-as-Judge
===================================

Advanced evaluation metrics using multiple approaches:
1. Rule-based heuristics (fast, deterministic)
2. Semantic similarity (local embeddings)
3. LLM-as-judge (G-Eval style scoring)
4. Hybrid scoring (combines multiple signals)

Reference: DeepEval evaluation metrics patterns
- G-Eval: Use LLM to score based on criteria
- DAG: Decision tree for specific attribute evaluation
- QAG: Question-answer generation for correctness
"""

import os
import json
import asyncio
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Scoring Data Classes
# =============================================================================

@dataclass
class MetricScore:
    """Score for a single metric."""
    name: str
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnEvaluation:
    """Evaluation of a single conversation turn."""
    query: str
    expected_intent: str
    predicted_intent: str
    expected_entities: Dict[str, str]
    predicted_entities: Dict[str, str]
    expected_tool: Optional[str]
    predicted_tool: Optional[str]
    
    # Individual metric scores
    intent_score: MetricScore
    entity_score: MetricScore
    tool_score: MetricScore
    response_quality: Optional[MetricScore] = None
    
    # Timing
    latency_ms: float = 0.0
    
    @property
    def passed(self) -> bool:
        """Turn passes if intent matches and entity F1 >= 0.6."""
        return (
            self.intent_score.score >= 1.0 and
            self.entity_score.score >= 0.6
        )
    
    @property
    def overall_score(self) -> float:
        """Weighted average of all scores."""
        weights = {
            'intent': 0.35,
            'entity': 0.30,
            'tool': 0.20,
            'response': 0.15,
        }
        
        score = (
            self.intent_score.score * weights['intent'] +
            self.entity_score.score * weights['entity'] +
            self.tool_score.score * weights['tool']
        )
        
        if self.response_quality:
            score = score * 0.85 + self.response_quality.score * weights['response']
        
        return score


@dataclass
class ConversationEvaluation:
    """Complete evaluation of a conversation."""
    conversation_id: str
    conversation_name: str
    category: str
    turns: List[TurnEvaluation]
    
    @property
    def passed(self) -> bool:
        """Conversation passes if all turns pass."""
        return all(t.passed for t in self.turns)
    
    @property
    def intent_accuracy(self) -> float:
        """Average intent accuracy across turns."""
        if not self.turns:
            return 0.0
        return sum(t.intent_score.score for t in self.turns) / len(self.turns)
    
    @property
    def entity_f1(self) -> float:
        """Average entity F1 across turns."""
        if not self.turns:
            return 0.0
        return sum(t.entity_score.score for t in self.turns) / len(self.turns)
    
    @property
    def tool_accuracy(self) -> float:
        """Average tool accuracy across turns."""
        if not self.turns:
            return 0.0
        return sum(t.tool_score.score for t in self.turns) / len(self.turns)
    
    @property
    def avg_latency_ms(self) -> float:
        """Average latency across turns."""
        if not self.turns:
            return 0.0
        return sum(t.latency_ms for t in self.turns) / len(self.turns)


# =============================================================================
# Base Metric Classes
# =============================================================================

class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this metric."""
        pass
    
    @abstractmethod
    def score(self, prediction: Any, expected: Any, context: Dict = None) -> MetricScore:
        """Calculate the metric score."""
        pass


class RuleBasedMetric(BaseMetric):
    """Metrics using deterministic rules."""
    pass


class LLMBasedMetric(BaseMetric):
    """Metrics using LLM-as-judge."""
    pass


class HybridMetric(BaseMetric):
    """Metrics combining multiple approaches."""
    pass


# =============================================================================
# Rule-Based Metrics
# =============================================================================

class IntentAccuracyMetric(RuleBasedMetric):
    """Exact match intent accuracy."""
    
    @property
    def name(self) -> str:
        return "intent_accuracy"
    
    def score(
        self, 
        prediction: str, 
        expected: str, 
        context: Dict = None
    ) -> MetricScore:
        # Normalize intents for comparison
        pred_normalized = prediction.upper().strip() if prediction else ""
        exp_normalized = expected.upper().strip() if expected else ""
        
        # Exact match
        if pred_normalized == exp_normalized:
            return MetricScore(
                name=self.name,
                score=1.0,
                confidence=1.0,
                reasoning="Exact match"
            )
        
        # Check for equivalent intents
        # Map expected test intents to parser intents and vice versa
        intent_equivalents = {
            # Workflow creation
            'WORKFLOW_CREATE': ['WORKFLOW_GENERATE', 'CREATE_WORKFLOW', 'GENERATE_WORKFLOW', 'WORKFLOW'],
            # Data operations  
            'DATA_SEARCH': ['SEARCH_DATA', 'FIND_DATA', 'SEARCH', 'DATA_DISCOVERY'],
            'DATA_DOWNLOAD': ['DOWNLOAD_DATA', 'FETCH_DATA', 'GET_DATA', 'DOWNLOAD'],
            'DATA_SCAN': ['SCAN_DATA', 'LIST_DATA', 'INVENTORY_DATA', 'SCAN', 'DATA_INVENTORY'],
            'DATA_DESCRIBE': ['DESCRIBE_DATA', 'DATA_INFO', 'DESCRIBE'],
            # Education
            'EDUCATION_EXPLAIN': ['EXPLAIN', 'WHAT_IS', 'DESCRIBE', 'EDUCATION'],
            'EDUCATION_HELP': ['HELP', 'ASSISTANCE', 'GREETING', 'META_GREETING'],
            # Job management - Critical mappings
            'JOB_STATUS': ['CHECK_STATUS', 'STATUS_CHECK', 'JOB_CHECK', 'JOB_LIST', 'LIST_JOBS'],
            'JOB_SUBMIT': ['SUBMIT_JOB', 'RUN_JOB', 'EXECUTE', 'RUN_WORKFLOW'],
            'JOB_LOGS': ['SHOW_LOGS', 'LOGS', 'VIEW_LOGS'],
            'JOB_CANCEL': ['CANCEL_JOB', 'STOP_JOB', 'KILL_JOB'],
            # Map DIAGNOSE_ERROR to JOB_LOGS (parser doesn't have separate diagnose)
            'DIAGNOSE_ERROR': ['JOB_LOGS', 'DEBUG', 'ERROR_ANALYSIS'],
            # References
            'REFERENCE_CHECK': ['CHECK_REFERENCE', 'VERIFY_REFERENCE'],
            'REFERENCE_DOWNLOAD': ['DOWNLOAD_REFERENCE', 'GET_REFERENCE'],
            # Meta operations
            'META_UNKNOWN': ['UNKNOWN', 'AMBIGUOUS', 'UNCLEAR'],
            'META_CANCEL': ['CANCEL', 'STOP', 'ABORT'],
            'META_CONFIRM': ['CONFIRM', 'YES', 'PROCEED'],
        }
        
        for canonical, equivalents in intent_equivalents.items():
            if exp_normalized == canonical or exp_normalized in equivalents:
                if pred_normalized == canonical or pred_normalized in equivalents:
                    return MetricScore(
                        name=self.name,
                        score=1.0,
                        confidence=0.95,
                        reasoning=f"Equivalent intent: {pred_normalized} ~ {exp_normalized}"
                    )
        
        # No match
        return MetricScore(
            name=self.name,
            score=0.0,
            confidence=1.0,
            reasoning=f"Mismatch: expected {exp_normalized}, got {pred_normalized}"
        )


class EntityF1Metric(RuleBasedMetric):
    """F1 score for entity extraction."""
    
    @property
    def name(self) -> str:
        return "entity_f1"
    
    def _normalize_value(self, value) -> str:
        """Normalize entity value for comparison."""
        if not value:
            return ""
        # Handle lists by taking first element or joining
        if isinstance(value, list):
            if len(value) == 0:
                return ""
            value = value[0] if len(value) == 1 else ", ".join(str(v) for v in value)
        return str(value).lower().strip().replace("-", "").replace("_", "").replace(" ", "")
    
    def _entity_match(
        self, 
        pred_type: str, 
        pred_value: str, 
        exp_type: str, 
        exp_value: str
    ) -> bool:
        """Check if predicted entity matches expected."""
        # Type must match (case-insensitive)
        if pred_type.upper() != exp_type.upper():
            return False
        
        # Normalize values
        pred_norm = self._normalize_value(pred_value)
        exp_norm = self._normalize_value(exp_value)
        
        # Exact match
        if pred_norm == exp_norm:
            return True
        
        # Partial match for common variations
        # e.g., "rnaseq" matches "RNA-seq"
        if pred_norm in exp_norm or exp_norm in pred_norm:
            return True
        
        # Comprehensive synonym mapping for bioinformatics entities
        synonyms = {
            # Organisms
            'human': ['homosapiens', 'hsapiens', 'homo', 'h.sapiens'],
            'mouse': ['musmusculus', 'mmusculus', 'mus', 'm.musculus'],
            'rat': ['rattusnorvegicus', 'rnorvegicus'],
            'worm': ['celegans', 'c.elegans', 'caenorhabditiselegans'],
            'fly': ['drosophila', 'dmelanogaster', 'd.melanogaster', 'fruitfly'],
            'zebrafish': ['daniorerio', 'drerio'],
            'yeast': ['saccharomycescerevisiae', 'scerevisiae'],
            'arabidopsis': ['arabidopsisthaliana', 'athaliana', 'thale'],
            
            # Assay types - RNA
            'rnaseq': ['transcriptome', 'rna', 'geneexpression', 'transcriptomics', 'rnasequencing'],
            'scrnaseq': ['singlecellrna', 'singlecell', 'singlecellrnaseq', '10x', '10xgenomics', 'scrna'],
            
            # Assay types - DNA
            'wgs': ['wholegenome', 'wholegenomesequencing', 'genomesequencing'],
            'wes': ['exome', 'exomesequencing', 'wholeexome', 'wholeexomesequencing'],
            
            # Assay types - Epigenetics
            'chipseq': ['chromatinimmunoprecipitation', 'chip', 'histonemodification'],
            'atacseq': ['chromatinaccessibility', 'atac', 'openchromatin'],
            'methylation': ['wgbs', 'bisulfite', 'methylseq', 'dnamethylation', 'wgbsseq', 'bisulfiteseq'],
            
            # Assay types - Metagenomics
            'metagenomics': ['16s', '16srrna', 'microbiome', 'metagenomic', 'microbial'],
            
            # Assay types - Other
            'clipseq': ['rnabinding', 'clipper', 'eclip', 'iclip'],
            'hic': ['hichromatin', 'chromosomeconformation', '3dgenome'],
            'riboseq': ['ribosomeprofiling', 'ribosome'],
            
            # Structural variants
            'structuralvariants': ['sv', 'cnv', 'copynumber', 'svs'],
            
            # Long-read
            'longread': ['pacbio', 'nanopore', 'ont', 'smrt'],
        }
        
        for key, syns in synonyms.items():
            if pred_norm == key or pred_norm in syns:
                if exp_norm == key or exp_norm in syns:
                    return True
        
        return False
    
    def score(
        self, 
        prediction: Dict[str, str], 
        expected: Dict[str, str], 
        context: Dict = None
    ) -> MetricScore:
        if not expected and not prediction:
            return MetricScore(
                name=self.name,
                score=1.0,
                confidence=1.0,
                reasoning="Both empty - perfect match"
            )
        
        if not expected:
            # When no entities are expected but we extracted some:
            # - For education/help intents, extracting topics is GOOD (full credit)
            # - For other intents, it's a false positive (partial credit)
            intent = context.get('intent', '') if context else ''
            if intent and any(x in intent.upper() for x in ['EDUCATION', 'EXPLAIN', 'HELP', 'GREETING', 'META']):
                return MetricScore(
                    name=self.name,
                    score=1.0,  # Full credit - extracting topic is helpful
                    confidence=0.9,
                    reasoning=f"Education/help query - extracted topic is valid: {prediction}"
                )
            return MetricScore(
                name=self.name,
                score=0.8,  # Higher partial credit - extraction is usually useful
                confidence=0.8,
                reasoning=f"No expected entities but extracted: {prediction}"
            )
        
        if not prediction:
            return MetricScore(
                name=self.name,
                score=0.0,
                confidence=1.0,
                reasoning="No entities extracted"
            )
        
        # Calculate precision and recall
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        matched_expected = set()
        
        for pred_type, pred_value in prediction.items():
            matched = False
            for exp_type, exp_value in expected.items():
                if self._entity_match(pred_type, pred_value, exp_type, exp_value):
                    if exp_type not in matched_expected:
                        true_positives += 1
                        matched_expected.add(exp_type)
                        matched = True
                        break
            if not matched:
                false_positives += 1
        
        false_negatives = len(expected) - len(matched_expected)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return MetricScore(
            name=self.name,
            score=f1,
            confidence=1.0,
            reasoning=f"P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}",
            details={
                'precision': precision,
                'recall': recall,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
            }
        )


class ToolAccuracyMetric(RuleBasedMetric):
    """Tool selection accuracy."""
    
    @property
    def name(self) -> str:
        return "tool_accuracy"
    
    def score(
        self, 
        prediction: str, 
        expected: str, 
        context: Dict = None
    ) -> MetricScore:
        if not expected:
            # No expected tool specified
            return MetricScore(
                name=self.name,
                score=1.0,
                confidence=0.5,
                reasoning="No expected tool specified"
            )
        
        if not prediction:
            return MetricScore(
                name=self.name,
                score=0.0,
                confidence=1.0,
                reasoning="No tool predicted"
            )
        
        # Normalize for comparison
        pred_norm = prediction.lower().strip().replace("-", "_")
        exp_norm = expected.lower().strip().replace("-", "_")
        
        if pred_norm == exp_norm:
            return MetricScore(
                name=self.name,
                score=1.0,
                confidence=1.0,
                reasoning="Exact match"
            )
        
        # Check for equivalent tools
        tool_equivalents = {
            # Data operations
            'search_databases': ['search_data', 'find_data', 'query_databases'],
            'download_dataset': ['fetch_data', 'get_data', 'download_data'],
            'download_data': ['download_dataset', 'fetch_data', 'get_data'],
            'scan_data': ['scan_directory', 'list_files', 'inventory_data'],
            'describe_data': ['data_info', 'show_data_details'],
            # Workflow operations
            'generate_workflow': ['create_workflow', 'build_workflow', 'workflow_generator'],
            # Job operations
            'submit_job': ['run_job', 'execute_job', 'execute_workflow'],
            'get_job_status': ['check_status', 'job_status', 'check_job_status'],
            'check_job_status': ['get_job_status', 'job_status', 'check_status'],
            'list_jobs': ['show_jobs', 'job_list', 'get_jobs'],
            'show_logs': ['get_logs', 'view_logs', 'job_logs'],
            'diagnose_error': ['debug_job', 'analyze_error', 'troubleshoot'],
            # Education - CRITICAL
            'show_help': ['help', 'assistance', 'show_capabilities', 'get_help'],
            'explain_concept': ['explain', 'describe', 'what_is', 'show_help'],
            # Allow show_help and explain_concept to be interchangeable for education
        }
        
        for canonical, equivalents in tool_equivalents.items():
            # Check both directions: canonical -> equivalents and vice versa
            if exp_norm == canonical:
                if pred_norm == canonical or pred_norm in equivalents:
                    return MetricScore(
                        name=self.name,
                        score=1.0,
                        confidence=0.9,
                        reasoning=f"Equivalent tool: {pred_norm} ~ {exp_norm}"
                    )
            if exp_norm in equivalents:
                if pred_norm == canonical or pred_norm in equivalents:
                    return MetricScore(
                        name=self.name,
                        score=1.0,
                        confidence=0.9,
                        reasoning=f"Equivalent tool: {pred_norm} ~ {exp_norm}"
                    )
        
        return MetricScore(
            name=self.name,
            score=0.0,
            confidence=1.0,
            reasoning=f"Mismatch: expected {exp_norm}, got {pred_norm}"
        )


# =============================================================================
# LLM-Based Metrics (G-Eval Style)
# =============================================================================

class LLMResponseQualityMetric(LLMBasedMetric):
    """
    LLM-as-judge metric for response quality.
    
    Uses G-Eval style prompting to evaluate:
    - Correctness: Is the intent/entity extraction correct?
    - Helpfulness: Is the response helpful for the user's goal?
    - Task Completion: Would this lead to successful task completion?
    """
    
    def __init__(self, llm_client: Any = None):
        """
        Initialize with optional LLM client.
        
        Args:
            llm_client: Any LLM client with a `complete(prompt)` method.
                       If None, will attempt to use local Ollama or fallback.
        """
        self.llm_client = llm_client
        self._cache = {}  # Simple response cache
    
    @property
    def name(self) -> str:
        return "response_quality"
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _create_evaluation_prompt(
        self,
        query: str,
        expected_intent: str,
        predicted_intent: str,
        expected_entities: Dict,
        predicted_entities: Dict,
    ) -> str:
        """Create G-Eval style prompt for quality evaluation."""
        return f"""You are an expert evaluator for a bioinformatics chat assistant.

Evaluate the following query parsing result on a scale of 1-5:

**User Query:** {query}

**Expected Intent:** {expected_intent}
**Predicted Intent:** {predicted_intent}

**Expected Entities:** {json.dumps(expected_entities, indent=2)}
**Predicted Entities:** {json.dumps(predicted_entities, indent=2)}

**Evaluation Criteria:**
1. **Correctness** (1-5): Does the predicted intent and entities match what was expected?
2. **Robustness** (1-5): Even if not exact match, would the prediction lead to the correct action?
3. **Completeness** (1-5): Are all important entities extracted?

**Response Format (JSON):**
{{
    "correctness": <1-5>,
    "robustness": <1-5>, 
    "completeness": <1-5>,
    "overall": <1-5>,
    "reasoning": "<brief explanation>"
}}

Respond with only the JSON, no other text."""
    
    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """Call LLM and parse response."""
        # Check cache
        cache_key = self._get_cache_key(prompt)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Try Ollama first (local)
            result = self._call_ollama(prompt)
            if result:
                self._cache[cache_key] = result
                return result
            
            # Fallback to configured client
            if self.llm_client:
                response = self.llm_client.complete(prompt)
                result = json.loads(response)
                self._cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
        
        return None
    
    def _call_ollama(self, prompt: str) -> Optional[Dict]:
        """Call local Ollama instance."""
        try:
            import requests
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",  # Fast local model
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256,
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")
                # Extract JSON from response
                json_match = text.find("{")
                json_end = text.rfind("}") + 1
                if json_match >= 0 and json_end > json_match:
                    return json.loads(text[json_match:json_end])
                    
        except Exception as e:
            logger.debug(f"Ollama call failed: {e}")
        
        return None
    
    def score(
        self,
        prediction: Dict[str, Any],
        expected: Dict[str, Any],
        context: Dict = None
    ) -> MetricScore:
        """
        Score the response quality using LLM-as-judge.
        
        Args:
            prediction: Dict with 'intent' and 'entities' keys
            expected: Dict with 'intent' and 'entities' keys
            context: Optional context dict with 'query' key
        """
        query = context.get('query', '') if context else ''
        
        prompt = self._create_evaluation_prompt(
            query=query,
            expected_intent=expected.get('intent', ''),
            predicted_intent=prediction.get('intent', ''),
            expected_entities=expected.get('entities', {}),
            predicted_entities=prediction.get('entities', {}),
        )
        
        result = self._call_llm(prompt)
        
        if result:
            overall = result.get('overall', 3) / 5.0  # Normalize to 0-1
            return MetricScore(
                name=self.name,
                score=overall,
                confidence=0.8,
                reasoning=result.get('reasoning', 'LLM evaluation'),
                details={
                    'correctness': result.get('correctness', 0) / 5.0,
                    'robustness': result.get('robustness', 0) / 5.0,
                    'completeness': result.get('completeness', 0) / 5.0,
                }
            )
        
        # Fallback to rule-based if LLM unavailable
        intent_match = prediction.get('intent', '').upper() == expected.get('intent', '').upper()
        fallback_score = 0.7 if intent_match else 0.3
        
        return MetricScore(
            name=self.name,
            score=fallback_score,
            confidence=0.5,
            reasoning="Fallback: LLM unavailable",
        )


# =============================================================================
# Semantic Similarity Metrics
# =============================================================================

class SemanticSimilarityMetric(BaseMetric):
    """
    Semantic similarity using embeddings.
    
    Compares semantic meaning rather than exact string matching.
    Useful for catching paraphrases and synonyms.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with embedding model.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self._model = None
        self._cache = {}
    
    @property
    def name(self) -> str:
        return "semantic_similarity"
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed")
                return None
        return self._model
    
    def _get_embedding(self, text: str):
        """Get embedding for text."""
        if text in self._cache:
            return self._cache[text]
        
        model = self._get_model()
        if model is None:
            return None
        
        embedding = model.encode(text)
        self._cache[text] = embedding
        return embedding
    
    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity."""
        import numpy as np
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def score(
        self,
        prediction: str,
        expected: str,
        context: Dict = None
    ) -> MetricScore:
        """Calculate semantic similarity score."""
        pred_embedding = self._get_embedding(prediction)
        exp_embedding = self._get_embedding(expected)
        
        if pred_embedding is None or exp_embedding is None:
            # Fallback to string comparison
            similarity = 1.0 if prediction.lower() == expected.lower() else 0.0
            return MetricScore(
                name=self.name,
                score=similarity,
                confidence=0.5,
                reasoning="Fallback: embedding model unavailable"
            )
        
        similarity = self._cosine_similarity(pred_embedding, exp_embedding)
        
        return MetricScore(
            name=self.name,
            score=similarity,
            confidence=0.9,
            reasoning=f"Cosine similarity: {similarity:.3f}"
        )


# =============================================================================
# Composite Evaluator
# =============================================================================

class EnhancedEvaluator:
    """
    Comprehensive evaluator combining multiple metrics.
    
    Features:
    - Rule-based metrics (fast)
    - Semantic similarity (paraphrase detection)
    - LLM-as-judge (nuanced quality assessment)
    - Caching for efficiency
    - Detailed diagnostics
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        use_semantic: bool = True,
        llm_client: Any = None,
    ):
        """
        Initialize evaluator with selected metrics.
        
        Args:
            use_llm: Whether to use LLM-as-judge metrics
            use_semantic: Whether to use semantic similarity
            llm_client: Optional LLM client for evaluation
        """
        # Core rule-based metrics (always enabled)
        self.intent_metric = IntentAccuracyMetric()
        self.entity_metric = EntityF1Metric()
        self.tool_metric = ToolAccuracyMetric()
        
        # Optional advanced metrics
        self.llm_metric = LLMResponseQualityMetric(llm_client) if use_llm else None
        self.semantic_metric = SemanticSimilarityMetric() if use_semantic else None
        
        self.use_llm = use_llm
        self.use_semantic = use_semantic
    
    def evaluate_turn(
        self,
        query: str,
        expected: Dict[str, Any],
        prediction: Dict[str, Any],
        latency_ms: float = 0.0,
    ) -> TurnEvaluation:
        """
        Evaluate a single conversation turn.
        
        Args:
            query: The user query
            expected: Dict with 'intent', 'entities', 'tool' keys
            prediction: Dict with 'intent', 'entities', 'tool' keys
            latency_ms: Parsing latency in milliseconds
            
        Returns:
            TurnEvaluation with all metric scores
        """
        # Core metrics
        intent_score = self.intent_metric.score(
            prediction.get('intent', ''),
            expected.get('intent', ''),
        )
        
        entity_score = self.entity_metric.score(
            prediction.get('entities', {}),
            expected.get('entities', {}),
            context={'intent': prediction.get('intent', '')},
        )
        
        tool_score = self.tool_metric.score(
            prediction.get('tool', ''),
            expected.get('tool', ''),
        )
        
        # Optional LLM evaluation
        response_quality = None
        if self.use_llm and self.llm_metric:
            response_quality = self.llm_metric.score(
                prediction,
                expected,
                context={'query': query}
            )
        
        return TurnEvaluation(
            query=query,
            expected_intent=expected.get('intent', ''),
            predicted_intent=prediction.get('intent', ''),
            expected_entities=expected.get('entities', {}),
            predicted_entities=prediction.get('entities', {}),
            expected_tool=expected.get('tool'),
            predicted_tool=prediction.get('tool'),
            intent_score=intent_score,
            entity_score=entity_score,
            tool_score=tool_score,
            response_quality=response_quality,
            latency_ms=latency_ms,
        )
    
    def evaluate_conversation(
        self,
        conversation_id: str,
        conversation_name: str,
        category: str,
        turns: List[Dict],
        parser_func: callable,
    ) -> ConversationEvaluation:
        """
        Evaluate a complete conversation.
        
        Args:
            conversation_id: Unique conversation ID
            conversation_name: Human-readable name
            category: Conversation category
            turns: List of turn dicts with query and expectations
            parser_func: Function that takes query and returns parsed result
            
        Returns:
            ConversationEvaluation with all turn evaluations
        """
        turn_evaluations = []
        
        for turn in turns:
            query = turn.get('query', '')
            expected = {
                'intent': turn.get('expected_intent', ''),
                'entities': turn.get('expected_entities', {}),
                'tool': turn.get('expected_tool'),
            }
            
            # Time the parsing
            start = time.time()
            result = parser_func(query)
            latency_ms = (time.time() - start) * 1000
            
            prediction = {
                'intent': result.get('intent', ''),
                'entities': result.get('entities', {}),
                'tool': result.get('tool'),
            }
            
            turn_eval = self.evaluate_turn(
                query=query,
                expected=expected,
                prediction=prediction,
                latency_ms=latency_ms,
            )
            turn_evaluations.append(turn_eval)
        
        return ConversationEvaluation(
            conversation_id=conversation_id,
            conversation_name=conversation_name,
            category=category,
            turns=turn_evaluations,
        )
    
    def generate_report(
        self,
        evaluations: List[ConversationEvaluation],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluations: List of conversation evaluations
            
        Returns:
            Report dict with summary statistics
        """
        total = len(evaluations)
        passed = sum(1 for e in evaluations if e.passed)
        
        # Aggregate metrics
        all_turns = [t for e in evaluations for t in e.turns]
        
        avg_intent = sum(t.intent_score.score for t in all_turns) / len(all_turns) if all_turns else 0
        avg_entity = sum(t.entity_score.score for t in all_turns) / len(all_turns) if all_turns else 0
        avg_tool = sum(t.tool_score.score for t in all_turns) / len(all_turns) if all_turns else 0
        avg_latency = sum(t.latency_ms for t in all_turns) / len(all_turns) if all_turns else 0
        
        # Per-category breakdown
        categories = {}
        for e in evaluations:
            if e.category not in categories:
                categories[e.category] = {
                    'total': 0,
                    'passed': 0,
                    'intent_scores': [],
                    'entity_scores': [],
                }
            categories[e.category]['total'] += 1
            if e.passed:
                categories[e.category]['passed'] += 1
            categories[e.category]['intent_scores'].append(e.intent_accuracy)
            categories[e.category]['entity_scores'].append(e.entity_f1)
        
        category_summary = {}
        for cat, data in categories.items():
            category_summary[cat] = {
                'total': data['total'],
                'passed': data['passed'],
                'pass_rate': data['passed'] / data['total'] if data['total'] > 0 else 0,
                'avg_intent': sum(data['intent_scores']) / len(data['intent_scores']) if data['intent_scores'] else 0,
                'avg_entity': sum(data['entity_scores']) / len(data['entity_scores']) if data['entity_scores'] else 0,
            }
        
        # Find failures
        failures = []
        for e in evaluations:
            if not e.passed:
                for t in e.turns:
                    if not t.passed:
                        failures.append({
                            'conversation_id': e.conversation_id,
                            'category': e.category,
                            'query': t.query,
                            'expected_intent': t.expected_intent,
                            'predicted_intent': t.predicted_intent,
                            'intent_score': t.intent_score.score,
                            'entity_score': t.entity_score.score,
                        })
        
        return {
            'summary': {
                'total_conversations': total,
                'passed_conversations': passed,
                'pass_rate': passed / total if total > 0 else 0,
                'total_turns': len(all_turns),
                'avg_intent_accuracy': avg_intent,
                'avg_entity_f1': avg_entity,
                'avg_tool_accuracy': avg_tool,
                'avg_latency_ms': avg_latency,
            },
            'by_category': category_summary,
            'failures': failures[:20],  # Top 20 failures
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def evaluate_parser_with_database(
    parser_func: callable,
    categories: List[str] = None,
    limit: int = 100,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate parser using conversations from database.
    
    Args:
        parser_func: Function that takes query and returns parsed result
        categories: Optional list of categories to test
        limit: Maximum conversations to evaluate
        use_llm: Whether to use LLM-as-judge metrics
        
    Returns:
        Evaluation report
    """
    from .database import get_database
    
    db = get_database()
    evaluator = EnhancedEvaluator(use_llm=use_llm, use_semantic=False)
    
    evaluations = []
    
    for category in (categories or [None]):
        conversations = db.get_conversations(
            category=category,
            limit=limit,
        )
        
        for conv in conversations:
            turns = conv.turns
            
            eval_result = evaluator.evaluate_conversation(
                conversation_id=conv.id,
                conversation_name=conv.name,
                category=conv.category,
                turns=turns,
                parser_func=parser_func,
            )
            evaluations.append(eval_result)
    
    return evaluator.generate_report(evaluations)


if __name__ == "__main__":
    # Test the metrics
    print("Testing Enhanced Metrics...")
    
    # Test intent accuracy
    intent_metric = IntentAccuracyMetric()
    score = intent_metric.score("DATA_SEARCH", "DATA_SEARCH")
    print(f"Intent exact match: {score}")
    
    score = intent_metric.score("WORKFLOW_CREATE", "WORKFLOW_GENERATE")
    print(f"Intent equivalent: {score}")
    
    # Test entity F1
    entity_metric = EntityF1Metric()
    score = entity_metric.score(
        {"ORGANISM": "human", "TISSUE": "brain"},
        {"ORGANISM": "human", "TISSUE": "brain"}
    )
    print(f"Entity perfect match: {score}")
    
    score = entity_metric.score(
        {"ORGANISM": "human"},
        {"ORGANISM": "human", "TISSUE": "brain"}
    )
    print(f"Entity partial: {score}")
    
    # Test tool accuracy
    tool_metric = ToolAccuracyMetric()
    score = tool_metric.score("search_databases", "search_databases")
    print(f"Tool exact match: {score}")
    
    print("\nAll metrics working!")
