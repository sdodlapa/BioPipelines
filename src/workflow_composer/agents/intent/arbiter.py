"""
LLM Intent Arbiter
==================

Final decision layer that uses LLM to arbitrate between multiple parsing methods.

Architecture:
```
Query → [Pattern Parser, Semantic Similarity, Entity Extraction] → LLM Arbiter → Final Intent
```

The arbiter:
1. Receives results from all parsing methods
2. Only invokes LLM when there's disagreement or low confidence
3. Uses LLM to reason about context, negation, and ambiguity
4. Returns the most accurate intent with explanation

Benefits:
- Combines speed of rule-based methods with accuracy of LLM reasoning
- LLM can understand context that patterns miss
- Cost-efficient: LLM only called when needed (~20% of queries)
- Explainable: LLM provides reasoning for its decision

Usage:
    arbiter = IntentArbiter(llm_client)
    result = arbiter.arbitrate(
        query="This is not about RNA-seq, it's about ChIP-seq",
        pattern_result=("WORKFLOW_CREATE", 0.6),
        semantic_result=("WORKFLOW_CREATE", 0.7),
        entity_result=("WORKFLOW_CREATE", 0.8),
    )
    # LLM will recognize negation and return correct intent
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ArbiterStrategy(Enum):
    """Strategy for when to invoke the LLM arbiter."""
    ALWAYS = "always"              # Always use LLM (most accurate, expensive)
    ON_DISAGREEMENT = "disagreement"  # Only when methods disagree
    ON_LOW_CONFIDENCE = "low_confidence"  # When confidence < threshold
    ON_COMPLEXITY = "complexity"    # For negation, adversarial, ambiguous
    SMART = "smart"                 # Combination of above (recommended)


@dataclass
class ParserVote:
    """A single parser's vote for intent."""
    parser_name: str  # "pattern", "semantic", "entity", "keyword"
    intent: str
    confidence: float
    evidence: Optional[str] = None  # What matched/triggered
    
    def __repr__(self):
        return f"{self.parser_name}:{self.intent}({self.confidence:.2f})"


@dataclass
class ArbiterResult:
    """Result from the LLM arbiter."""
    final_intent: str
    confidence: float
    reasoning: str
    method: str  # "unanimous", "llm_arbiter", "majority_vote", "fallback"
    votes: List[ParserVote] = field(default_factory=list)
    llm_invoked: bool = False
    needs_clarification: bool = False
    clarification_prompt: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.final_intent,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "method": self.method,
            "votes": [(v.parser_name, v.intent, v.confidence) for v in self.votes],
            "llm_invoked": self.llm_invoked,
            "needs_clarification": self.needs_clarification,
            "clarification_prompt": self.clarification_prompt,
        }


# =============================================================================
# LLM TRIGGER PATTERNS (Option C)
# =============================================================================
# These patterns are AMBIGUOUS and should ALWAYS trigger LLM arbitration
# even if pattern matching returns high confidence. The pattern might match
# multiple intents, so we need LLM to disambiguate.

LLM_TRIGGER_PATTERNS = [
    # "run X --something" could be SYSTEM_COMMAND or JOB_SUBMIT
    r"run\s+\w+\s+--",
    r"execute\s+\w+\s+--",
    
    # "submit X workflow" - could be JOB_SUBMIT (run existing) or misclassified
    r"submit\s+(?:my\s+)?(?:the\s+)?\w+[-\s]?\w*\s+(?:workflow|pipeline)",
    
    # "check X" is ambiguous (status? availability? validation?)
    r"^check\s+(?:if\s+)?(?!status|job|queue)",  # But not "check status"
    r"^verify\s+",
    
    # "is X available/installed" could be REFERENCE_CHECK or SYSTEM_STATUS
    r"is\s+\w+\s+(?:installed|available|ready|present)",
    r"do\s+(?:i|we)\s+have\s+",
    
    # Version/help checks look like commands
    r"\w+\s+(?:--version|--help|-v|-h)\b",
    r"(?:version|help)\s+(?:of|for)\s+\w+",
    
    # "get X" is ambiguous (download? search? fetch?)
    r"^get\s+(?!status|logs|job)",  # But not "get status"
    
    # "find X" could be search or scan
    r"^find\s+(?:all|any|some)?\s*(?:local|my)?",
    
    # "show X" could be list, status, visualize
    r"^show\s+(?:me\s+)?(?:the\s+)?",
    
    # "what is" could be education or status
    r"^what\s+(?:is|are)\s+",
]

# =============================================================================
# HIGH RISK INTENTS (Option E)
# =============================================================================
# These intents are DANGEROUS or EXPENSIVE - always verify with LLM
# before executing, even if pattern matching is confident.

HIGH_RISK_INTENTS = {
    "JOB_SUBMIT",      # Submits to cluster, uses resources
    "JOB_CANCEL",      # Terminates running jobs
    "DATA_CLEANUP",    # Deletes data
    "DATA_DOWNLOAD",   # Downloads large files, uses bandwidth
    "REFERENCE_DOWNLOAD",  # Downloads large reference files
    "SYSTEM_RESTART",  # Restarts services
}

# Complexity indicators that suggest LLM should be invoked
COMPLEXITY_INDICATORS = {
    # Negation patterns
    "negation": [
        r"\b(?:not|don'?t|doesn'?t|won'?t|can'?t|isn'?t|aren'?t)\b",
        r"\b(?:instead|rather|but|however|actually)\b",
        r"\b(?:forget|ignore|skip|exclude)\b",
    ],
    # Adversarial/tricky patterns
    "adversarial": [
        r"this is (?:not )?about",
        r"(?:i don'?t want|not interested in)",
        r"(?:just|only) (?:search|look|find)",
        r"no (?:workflow|download|job)",
    ],
    # Conditional patterns
    "conditional": [
        r"\bif\b.+\belse\b",
        r"\bif not\b",
        r"\botherwise\b",
        r"\bthen\b",
    ],
    # Multi-step patterns
    "multi_step": [
        r"\b(?:first|then|after|before|next)\b",
        r"\b(?:and then|and also|as well)\b",
    ],
    # Ambiguous patterns
    "ambiguous": [
        r"^(?:what|how|why|when|where)\s",
        r"\?$",
        r"\bor\b",
    ],
}


class IntentArbiter:
    """
    LLM-based arbiter for intent classification.
    
    Takes votes from multiple parsing methods and uses LLM reasoning
    to determine the most accurate intent when there's disagreement
    or complexity.
    """
    
    # Prompt for LLM arbitration
    ARBITER_PROMPT = """You are an expert at understanding user intent in a bioinformatics assistant.

Given a user query and multiple parsing results, determine the most accurate intent.

USER QUERY:
"{query}"

PARSING RESULTS:
{parsing_results}

AVAILABLE INTENTS:
{available_intents}

INSTRUCTIONS:
1. Consider what the user ACTUALLY wants to do
2. Pay attention to negation words (not, don't, isn't, etc.)
3. Look for implicit intent (e.g., "lung cancer RNA data" implies DATA_SEARCH)
4. If the user says "not X, but Y" - focus on Y
5. Consider the context and entities mentioned
6. IMPORTANT: "run X --version" or "X version" is SYSTEM_COMMAND (checking tool), NOT JOB_SUBMIT (running a workflow)
7. "check reference" or "is X genome available" is REFERENCE_CHECK, NOT DATA_SEARCH
8. If the query is too vague to determine intent (e.g., "analyze this", "run it"), return:
   - intent: "META_UNKNOWN"
   - needs_clarification: true
   - clarification_prompt: A helpful question asking what they want

Respond with ONLY valid JSON:
{{"intent": "INTENT_NAME", "confidence": 0.0-1.0, "reasoning": "brief explanation", "needs_clarification": false, "clarification_prompt": null}}"""

    # Available intents (subset for prompt - avoid overwhelming LLM)
    INTENT_DESCRIPTIONS = {
        "DATA_SCAN": "Scan/list local filesystem for data",
        "DATA_SEARCH": "Search public databases (GEO, ENCODE, SRA, TCGA) for data",
        "DATA_DOWNLOAD": "Download a specific dataset by ID",
        "DATA_DESCRIBE": "Describe or summarize data contents",
        "DATA_FILTER": "Filter or subset existing data/results",
        "WORKFLOW_CREATE": "Create/generate an analysis pipeline or workflow",
        "WORKFLOW_MODIFY": "Modify/update an existing workflow (add steps, change parameters)",
        "WORKFLOW_LIST": "List available workflow templates",
        "WORKFLOW_VISUALIZE": "Show workflow diagram/DAG",
        "REFERENCE_CHECK": "Check if reference genome/annotation is available locally",
        "REFERENCE_DOWNLOAD": "Download a reference genome/annotation",
        "JOB_SUBMIT": "Submit a workflow or pipeline job to the cluster (NOT for checking tool versions)",
        "JOB_STATUS": "Check current status of running/queued jobs (quick check)",
        "JOB_WATCH": "Watch/monitor a job continuously with notifications until completion",
        "JOB_RESUBMIT": "Resubmit/retry a failed job, possibly with different parameters",
        "JOB_LOGS": "View logs or output of a job",
        "JOB_CANCEL": "Cancel/stop a running job",
        "JOB_LIST": "List all jobs or active jobs",
        "SYSTEM_COMMAND": "Run a simple shell command like checking tool version (--version, --help)",
        "SYSTEM_STATUS": "Check system health or available tools",
        "ANALYSIS_INTERPRET": "Interpret or explain analysis results",
        "DIAGNOSE_ERROR": "Debug or troubleshoot an error",
        "EDUCATION_EXPLAIN": "Explain a concept, method, or term",
        "EDUCATION_HELP": "Show help, commands, or capabilities",
        "META_CONFIRM": "Confirm/acknowledge (yes, ok, proceed)",
        "META_CANCEL": "Cancel/stop current operation",
        "META_UNKNOWN": "Cannot determine intent",
    }
    
    def __init__(
        self,
        llm_client=None,
        strategy: ArbiterStrategy = ArbiterStrategy.SMART,
        confidence_threshold: float = 0.7,
        disagreement_threshold: float = 0.3,
        use_cascade: bool = True,
    ):
        """
        Initialize the arbiter.
        
        Args:
            llm_client: LLM client with chat() or generate() method.
                        If None and use_cascade=True, uses ProviderRouter.
            strategy: When to invoke LLM arbitration
            confidence_threshold: Below this, consider low confidence
            disagreement_threshold: If votes differ by this much, there's disagreement
            use_cascade: If True and llm_client is None, use cascading ProviderRouter
        """
        self.llm_client = llm_client
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.disagreement_threshold = disagreement_threshold
        self.use_cascade = use_cascade
        
        # If no client provided but cascade enabled, use router
        if self.llm_client is None and self.use_cascade:
            try:
                from src.workflow_composer.providers.router import get_router
                self._router = get_router()
            except ImportError:
                self._router = None
        else:
            self._router = None
        
        # Compile complexity patterns
        import re
        self._complexity_patterns = {}
        for category, patterns in COMPLEXITY_INDICATORS.items():
            self._complexity_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        # Compile LLM trigger patterns (Option C)
        self._llm_trigger_patterns = [
            re.compile(p, re.IGNORECASE) for p in LLM_TRIGGER_PATTERNS
        ]
    
    def arbitrate(
        self,
        query: str,
        votes: List[ParserVote],
        context: Optional[Dict] = None,
    ) -> ArbiterResult:
        """
        Arbitrate between multiple parsing results.
        
        Args:
            query: Original user query
            votes: List of parser votes
            context: Optional conversation context
            
        Returns:
            ArbiterResult with final intent
        """
        if not votes:
            return ArbiterResult(
                final_intent="META_UNKNOWN",
                confidence=0.0,
                reasoning="No parsing results provided",
                method="fallback",
                votes=[],
            )
        
        # Check if all parsers agree
        intents = set(v.intent for v in votes if v.confidence > 0.3)
        confidences = [v.confidence for v in votes]
        max_confidence = max(confidences) if confidences else 0
        
        # Case 1: Unanimous agreement with high confidence
        if len(intents) == 1 and max_confidence >= self.confidence_threshold:
            intent = list(intents)[0]
            # But still check for complexity that might fool patterns
            if not self._should_invoke_llm(query, votes):
                return ArbiterResult(
                    final_intent=intent,
                    confidence=max_confidence,
                    reasoning="All parsers agree with high confidence",
                    method="unanimous",
                    votes=votes,
                    llm_invoked=False,
                )
        
        # Case 2: Decide if we should invoke LLM
        should_invoke = self._should_invoke_llm(query, votes)
        
        # Use LLM if available (direct client or router)
        has_llm = self.llm_client is not None or self._router is not None
        
        if should_invoke and has_llm:
            return self._invoke_llm_arbiter(query, votes, context)
        else:
            # Fall back to weighted voting
            return self._weighted_vote(query, votes)
    
    def _should_invoke_llm(self, query: str, votes: List[ParserVote]) -> bool:
        """Determine if LLM should be invoked based on strategy."""
        if self.strategy == ArbiterStrategy.ALWAYS:
            return True
        
        if self.strategy == ArbiterStrategy.ON_DISAGREEMENT:
            intents = set(v.intent for v in votes if v.confidence > 0.3)
            return len(intents) > 1
        
        if self.strategy == ArbiterStrategy.ON_LOW_CONFIDENCE:
            max_conf = max(v.confidence for v in votes) if votes else 0
            return max_conf < self.confidence_threshold
        
        if self.strategy == ArbiterStrategy.ON_COMPLEXITY:
            return self._detect_complexity(query)
        
        if self.strategy == ArbiterStrategy.SMART:
            # Combine multiple signals
            
            # 1. Check for disagreement
            intents = set(v.intent for v in votes if v.confidence > 0.3)
            has_disagreement = len(intents) > 1
            
            # 2. Check for low confidence
            max_conf = max(v.confidence for v in votes) if votes else 0
            low_confidence = max_conf < self.confidence_threshold
            
            # 3. Check for complexity
            has_complexity = self._detect_complexity(query)
            
            # 4. Check for LLM trigger patterns (Option C)
            has_trigger_pattern = self._matches_trigger_pattern(query)
            
            # 5. Check for high-risk intent (Option E)
            best_vote = max(votes, key=lambda v: v.confidence) if votes else None
            is_high_risk = best_vote and best_vote.intent in HIGH_RISK_INTENTS
            
            # 6. Check if best vote is META_UNKNOWN
            is_unknown = best_vote and best_vote.intent == "META_UNKNOWN"
            
            # Invoke LLM if any significant signal
            # Log which signal triggered for debugging
            if has_trigger_pattern:
                logger.debug(f"LLM trigger pattern matched in '{query[:50]}'")
            if is_high_risk:
                logger.debug(f"High-risk intent '{best_vote.intent}' detected, verifying with LLM")
            
            return (has_disagreement or low_confidence or has_complexity or 
                    is_unknown or has_trigger_pattern or is_high_risk)
        
        return False
    
    def _detect_complexity(self, query: str) -> bool:
        """Detect if query has complex patterns that need LLM reasoning."""
        for category, patterns in self._complexity_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    logger.debug(f"Complexity detected: {category} in '{query[:50]}'")
                    return True
        return False
    
    def _matches_trigger_pattern(self, query: str) -> bool:
        """Check if query matches any LLM trigger pattern (Option C).
        
        These are patterns that are inherently ambiguous and need
        LLM arbitration even if pattern matching is confident.
        """
        for pattern in self._llm_trigger_patterns:
            if pattern.search(query):
                return True
        return False
    
    def _invoke_llm_arbiter(
        self,
        query: str,
        votes: List[ParserVote],
        context: Optional[Dict],
    ) -> ArbiterResult:
        """Use LLM to arbitrate between parsing results.
        
        Uses cascading provider router if available, otherwise direct client.
        Rate-limited providers are automatically skipped via cascade.
        """
        # Build parsing results summary
        parsing_lines = []
        for vote in votes:
            line = f"- {vote.parser_name}: {vote.intent} (confidence: {vote.confidence:.2f})"
            if vote.evidence:
                line += f" [matched: {vote.evidence}]"
            parsing_lines.append(line)
        parsing_results = "\n".join(parsing_lines)
        
        # Build available intents
        intent_lines = [f"- {k}: {v}" for k, v in self.INTENT_DESCRIPTIONS.items()]
        available_intents = "\n".join(intent_lines)
        
        # Format prompt
        prompt = self.ARBITER_PROMPT.format(
            query=query,
            parsing_results=parsing_results,
            available_intents=available_intents,
        )
        
        system_prompt = "You are an intent classification expert. Respond only with valid JSON."
        
        try:
            # Use cascading router if available (handles rate limits automatically)
            if self._router is not None:
                response = self._router.complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=200,
                    temperature=0.1,
                )
                content = response.content
            elif self.llm_client is not None:
                # Use direct LLM client
                if hasattr(self.llm_client, 'chat'):
                    # Get correct Message class
                    try:
                        from src.workflow_composer.providers.base import Message
                    except ImportError:
                        try:
                            from workflow_composer.providers.base import Message
                        except ImportError:
                            try:
                                from workflow_composer.llm.base import Message
                            except ImportError:
                                from src.workflow_composer.llm.base import Message
                    
                    messages = [
                        Message.system(system_prompt),
                        Message.user(prompt)
                    ]
                    response = self.llm_client.chat(messages)
                    content = response.content if hasattr(response, 'content') else str(response)
                elif hasattr(self.llm_client, 'complete'):
                    response = self.llm_client.complete(prompt, system_prompt=system_prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                elif hasattr(self.llm_client, 'generate'):
                    content = self.llm_client.generate(prompt, max_tokens=200, temperature=0.1)
                else:
                    raise ValueError("LLM client must have chat(), complete(), or generate() method")
            else:
                raise ValueError("No LLM client or router available")
            
            # Parse response
            content = content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            if content.startswith("json"):
                content = content[4:].strip()
            
            data = json.loads(content)
            
            intent = data.get("intent", "META_UNKNOWN")
            confidence = float(data.get("confidence", 0.8))
            reasoning = data.get("reasoning", "LLM arbitration")
            needs_clarification = data.get("needs_clarification", False)
            clarification_prompt = data.get("clarification_prompt", "")
            
            # If LLM says META_UNKNOWN or very low confidence, needs clarification
            if intent == "META_UNKNOWN" or confidence < 0.4:
                needs_clarification = True
                if not clarification_prompt:
                    clarification_prompt = "I'm not sure what you'd like me to do. Could you please be more specific?"
            
            return ArbiterResult(
                final_intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                method="llm_arbiter",
                votes=votes,
                needs_clarification=needs_clarification,
                clarification_prompt=clarification_prompt,
                llm_invoked=True,
            )
            
        except Exception as e:
            logger.warning(f"LLM arbitration failed: {e}")
            # Fall back to weighted voting
            result = self._weighted_vote(query, votes)
            result.reasoning = f"LLM failed ({e}), using weighted vote"
            return result
    
    def _weighted_vote(self, query: str, votes: List[ParserVote]) -> ArbiterResult:
        """Fall back to weighted voting when LLM is not available/fails."""
        # Weight by parser type
        weights = {
            "pattern": 0.35,
            "semantic": 0.35,
            "entity": 0.20,
            "keyword": 0.10,
        }
        
        # Aggregate votes
        intent_scores = {}
        for vote in votes:
            weight = weights.get(vote.parser_name, 0.1)
            weighted_score = vote.confidence * weight
            if vote.intent not in intent_scores:
                intent_scores[vote.intent] = 0
            intent_scores[vote.intent] += weighted_score
        
        if not intent_scores:
            return ArbiterResult(
                final_intent="META_UNKNOWN",
                confidence=0.0,
                reasoning="No valid votes",
                method="fallback",
                votes=votes,
            )
        
        # Find best
        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]
        
        # Normalize confidence
        max_possible = sum(weights.values())
        confidence = min(best_score / max_possible, 1.0)
        
        return ArbiterResult(
            final_intent=best_intent,
            confidence=confidence,
            reasoning=f"Weighted vote: {dict(intent_scores)}",
            method="majority_vote",
            votes=votes,
            llm_invoked=False,
        )


# =============================================================================
# SIMPLE ARBITER PARSER (for internal evaluation only)
# Note: The production UnifiedIntentParser is in unified_parser.py
# =============================================================================

class SimpleArbiterParser:
    """
    Production parser that combines all methods with LLM arbiter.
    
    Flow:
    1. Run all parsers in parallel (fast)
    2. Collect votes from each method
    3. If unanimous + high confidence → return immediately
    4. Otherwise, invoke LLM arbiter for final decision
    
    This gives the best of both worlds:
    - 80% of queries resolved in <10ms (pattern matching)
    - 20% of hard cases resolved accurately by LLM
    """
    
    def __init__(
        self,
        llm_client=None,
        use_semantic: bool = True,
        arbiter_strategy: ArbiterStrategy = ArbiterStrategy.SMART,
    ):
        """Initialize the unified parser."""
        # Import dependencies
        from .parser import IntentParser, IntentType
        from .semantic import SemanticIntentClassifier, HybridQueryParser
        
        self.llm_client = llm_client
        
        # Initialize individual parsers
        self.pattern_parser = IntentParser(llm_client=None, use_semantic=False)
        
        if use_semantic:
            try:
                self.semantic_classifier = SemanticIntentClassifier(llm_client=llm_client)
            except Exception as e:
                logger.warning(f"Semantic classifier not available: {e}")
                self.semantic_classifier = None
        else:
            self.semantic_classifier = None
        
        # Initialize arbiter
        self.arbiter = IntentArbiter(
            llm_client=llm_client,
            strategy=arbiter_strategy,
        )
        
        # Intent type mapping
        self.IntentType = IntentType
    
    def parse(self, query: str, context: Optional[Dict] = None) -> ArbiterResult:
        """
        Parse query using all methods and arbiter.
        
        Args:
            query: User query
            context: Optional conversation context
            
        Returns:
            ArbiterResult with final intent and reasoning
        """
        votes = []
        
        # 1. Pattern matching (always fast)
        try:
            pattern_result = self.pattern_parser.parse(query, context)
            votes.append(ParserVote(
                parser_name="pattern",
                intent=pattern_result.primary_intent.name,
                confidence=pattern_result.confidence,
                evidence=pattern_result.matched_pattern[:50] if hasattr(pattern_result, 'matched_pattern') and pattern_result.matched_pattern else None,
            ))
        except Exception as e:
            logger.warning(f"Pattern parsing failed: {e}")
        
        # 2. Semantic similarity
        if self.semantic_classifier:
            try:
                semantic_results = self.semantic_classifier.classify(query, top_k=3)
                if semantic_results:
                    best_semantic = semantic_results[0]
                    votes.append(ParserVote(
                        parser_name="semantic",
                        intent=best_semantic[0],
                        confidence=best_semantic[1],
                    ))
            except Exception as e:
                logger.warning(f"Semantic classification failed: {e}")
        
        # 3. Entity-based inference
        try:
            entities = self.pattern_parser._entity_extractor.extract(query)
            entity_intent = self._infer_from_entities(query, entities)
            if entity_intent:
                votes.append(ParserVote(
                    parser_name="entity",
                    intent=entity_intent[0],
                    confidence=entity_intent[1],
                    evidence=f"Found entities: {[e.type for e in entities]}" if entities else None,
                ))
        except Exception as e:
            logger.warning(f"Entity inference failed: {e}")
        
        # 4. Arbiter decides
        return self.arbiter.arbitrate(query, votes, context)
    
    def _infer_from_entities(
        self, 
        query: str, 
        entities: List,
    ) -> Optional[Tuple[str, float]]:
        """Infer intent from extracted entities."""
        query_lower = query.lower()
        # Entity has a 'type' attribute, not 'entity_type'
        entity_types = {e.type.name if hasattr(e.type, 'name') else str(e.type) 
                       for e in entities}
        
        # Dataset ID + action verb → download
        if "ACCESSION" in entity_types or "DATASET_ID" in entity_types:
            if any(w in query_lower for w in ["download", "get", "fetch", "retrieve"]):
                return ("DATA_DOWNLOAD", 0.85)
            if any(w in query_lower for w in ["search", "find", "look"]):
                return ("DATA_SEARCH", 0.7)
        
        # Assay type + create/make → workflow
        if "ANALYSIS_TYPE" in entity_types or "ASSAY_TYPE" in entity_types:
            if any(w in query_lower for w in ["create", "generate", "make", "build", "workflow", "pipeline"]):
                return ("WORKFLOW_CREATE", 0.8)
            if any(w in query_lower for w in ["search", "find", "data"]):
                return ("DATA_SEARCH", 0.7)
        
        # Disease/tissue + search terms → search
        if entity_types & {"DISEASE", "TISSUE", "ORGANISM"}:
            if any(w in query_lower for w in ["data", "dataset", "find", "search", "any"]):
                return ("DATA_SEARCH", 0.7)
        
        # Job ID mentioned → likely job-related
        if "JOB_ID" in entity_types:
            if any(w in query_lower for w in ["status", "check", "how"]):
                return ("JOB_STATUS", 0.75)
            if any(w in query_lower for w in ["log", "output", "error"]):
                return ("JOB_LOGS", 0.75)
            if any(w in query_lower for w in ["cancel", "stop", "kill"]):
                return ("JOB_CANCEL", 0.8)
        
        return None


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def compare_methods(
    queries: List[Tuple[str, str]],  # (query, expected_intent)
    parser: SimpleArbiterParser,
) -> Dict[str, Any]:
    """
    Compare different parsing methods on a set of queries.
    
    Returns accuracy breakdown by method and category.
    """
    results = {
        "total": len(queries),
        "correct": 0,
        "by_method": {
            "unanimous": {"total": 0, "correct": 0},
            "llm_arbiter": {"total": 0, "correct": 0},
            "majority_vote": {"total": 0, "correct": 0},
            "fallback": {"total": 0, "correct": 0},
        },
        "llm_invocations": 0,
        "errors": [],
    }
    
    for query, expected in queries:
        try:
            result = parser.parse(query)
            predicted = result.final_intent
            method = result.method
            
            results["by_method"][method]["total"] += 1
            
            if predicted == expected:
                results["correct"] += 1
                results["by_method"][method]["correct"] += 1
            else:
                results["errors"].append({
                    "query": query[:80],
                    "expected": expected,
                    "predicted": predicted,
                    "method": method,
                    "reasoning": result.reasoning,
                })
            
            if result.llm_invoked:
                results["llm_invocations"] += 1
                
        except Exception as e:
            results["errors"].append({
                "query": query[:80],
                "expected": expected,
                "error": str(e),
            })
    
    # Calculate accuracy
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    
    for method, data in results["by_method"].items():
        if data["total"] > 0:
            data["accuracy"] = data["correct"] / data["total"]
        else:
            data["accuracy"] = 0.0
    
    return results
