"""
Query Learning & Feedback System
=================================

Implements:
1. Active Learning: Log low-confidence queries for review
2. LLM Fallback: Use LLM for ambiguous queries
3. User Feedback Loop: Correct misclassifications
4. Fine-tuning Support: Export training data for embedding fine-tuning

This module makes the query parsing system self-improving over time.
"""

import json
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QueryLog:
    """A logged query with its parse result."""
    query: str
    timestamp: str
    intent: str
    confidence: float
    parse_method: str
    entities: List[Dict[str, str]]
    slots: Dict[str, str]
    
    # Feedback (filled later)
    correct_intent: Optional[str] = None
    user_feedback: Optional[str] = None
    feedback_timestamp: Optional[str] = None
    
    # Metadata
    session_id: Optional[str] = None
    query_hash: str = ""
    
    def __post_init__(self):
        if not self.query_hash:
            self.query_hash = hashlib.md5(self.query.lower().encode()).hexdigest()[:12]
    
    @property
    def needs_review(self) -> bool:
        """Check if this query needs human review."""
        return self.confidence < 0.5 and self.correct_intent is None
    
    @property
    def was_corrected(self) -> bool:
        """Check if user provided correction."""
        return self.correct_intent is not None and self.correct_intent != self.intent
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "QueryLog":
        return cls(**data)


@dataclass
class FeedbackStats:
    """Statistics about query feedback."""
    total_queries: int = 0
    low_confidence_queries: int = 0
    corrected_queries: int = 0
    intent_accuracy: Dict[str, float] = field(default_factory=dict)
    common_corrections: List[Tuple[str, str, int]] = field(default_factory=list)


# =============================================================================
# ACTIVE LEARNING LOGGER
# =============================================================================

class QueryLogger:
    """
    Logs queries for active learning and analysis.
    
    Features:
    - Persists low-confidence queries for review
    - Tracks intent accuracy over time
    - Exports training data for fine-tuning
    """
    
    def __init__(
        self,
        log_dir: Path = None,
        confidence_threshold: float = 0.5,
        max_logs: int = 10000,
    ):
        """
        Initialize the query logger.
        
        Args:
            log_dir: Directory to store logs (default: ~/.biopipelines/query_logs)
            confidence_threshold: Log queries below this confidence
            max_logs: Maximum logs to keep (older ones archived)
        """
        self.log_dir = log_dir or Path.home() / ".biopipelines" / "query_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.confidence_threshold = confidence_threshold
        self.max_logs = max_logs
        
        self._logs: List[QueryLog] = []
        self._lock = threading.Lock()
        
        # Load existing logs
        self._load_logs()
    
    def _load_logs(self):
        """Load existing logs from disk."""
        log_file = self.log_dir / "query_logs.jsonl"
        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            self._logs.append(QueryLog.from_dict(json.loads(line)))
                logger.info(f"Loaded {len(self._logs)} query logs")
            except Exception as e:
                logger.error(f"Failed to load query logs: {e}")
    
    def _save_logs(self):
        """Save logs to disk."""
        log_file = self.log_dir / "query_logs.jsonl"
        try:
            with open(log_file, "w") as f:
                for log in self._logs[-self.max_logs:]:
                    f.write(json.dumps(log.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to save query logs: {e}")
    
    def log(
        self,
        query: str,
        intent: str,
        confidence: float,
        parse_method: str,
        entities: List[Dict] = None,
        slots: Dict = None,
        session_id: str = None,
    ) -> QueryLog:
        """
        Log a query and its parse result.
        
        Args:
            query: The user query
            intent: Detected intent
            confidence: Confidence score
            parse_method: Method used (pattern/semantic/hybrid)
            entities: Extracted entities
            slots: Extracted slots
            session_id: Optional session identifier
            
        Returns:
            The created QueryLog
        """
        log = QueryLog(
            query=query,
            timestamp=datetime.now().isoformat(),
            intent=intent,
            confidence=confidence,
            parse_method=parse_method,
            entities=entities or [],
            slots=slots or {},
            session_id=session_id,
        )
        
        with self._lock:
            # Check for duplicate (same query)
            for existing in self._logs[-100:]:  # Check recent logs
                if existing.query_hash == log.query_hash:
                    return existing  # Don't duplicate
            
            self._logs.append(log)
            
            # Periodic save for low-confidence queries
            if confidence < self.confidence_threshold:
                self._save_logs()
        
        return log
    
    def add_feedback(
        self,
        query_hash: str,
        correct_intent: str,
        feedback: str = None,
    ) -> bool:
        """
        Add user feedback for a query.
        
        Args:
            query_hash: Hash of the query to correct
            correct_intent: The correct intent
            feedback: Optional user feedback text
            
        Returns:
            True if feedback was added
        """
        with self._lock:
            for log in reversed(self._logs):
                if log.query_hash == query_hash:
                    log.correct_intent = correct_intent
                    log.user_feedback = feedback
                    log.feedback_timestamp = datetime.now().isoformat()
                    self._save_logs()
                    return True
        return False
    
    def get_low_confidence_queries(
        self,
        limit: int = 50,
        reviewed_only: bool = False,
    ) -> List[QueryLog]:
        """Get queries that need review."""
        with self._lock:
            queries = [
                log for log in self._logs
                if log.confidence < self.confidence_threshold
                and (not reviewed_only or log.correct_intent is not None)
            ]
            return queries[-limit:]
    
    def get_corrected_queries(self) -> List[QueryLog]:
        """Get queries where user provided corrections."""
        with self._lock:
            return [log for log in self._logs if log.was_corrected]
    
    def get_stats(self) -> FeedbackStats:
        """Get feedback statistics."""
        with self._lock:
            stats = FeedbackStats(
                total_queries=len(self._logs),
                low_confidence_queries=sum(1 for l in self._logs if l.confidence < self.confidence_threshold),
                corrected_queries=sum(1 for l in self._logs if l.was_corrected),
            )
            
            # Calculate per-intent accuracy
            intent_correct = defaultdict(int)
            intent_total = defaultdict(int)
            correction_counts = defaultdict(int)
            
            for log in self._logs:
                if log.correct_intent is not None:
                    intent_total[log.intent] += 1
                    if log.intent == log.correct_intent:
                        intent_correct[log.intent] += 1
                    else:
                        correction_counts[(log.intent, log.correct_intent)] += 1
            
            for intent, total in intent_total.items():
                stats.intent_accuracy[intent] = intent_correct[intent] / total if total > 0 else 0
            
            stats.common_corrections = [
                (from_i, to_i, count)
                for (from_i, to_i), count in sorted(
                    correction_counts.items(),
                    key=lambda x: -x[1]
                )[:10]
            ]
            
            return stats
    
    def export_training_data(
        self,
        output_path: Path = None,
        include_corrections: bool = True,
    ) -> Path:
        """
        Export training data for fine-tuning.
        
        Exports in format suitable for sentence-transformers training:
        - queries.jsonl: {"text": query, "label": intent}
        - pairs.jsonl: {"text1": query1, "text2": query2, "label": 1/0}
        
        Args:
            output_path: Output directory
            include_corrections: Include corrected queries with correct labels
            
        Returns:
            Path to exported data
        """
        output_path = output_path or self.log_dir / "training_export"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export labeled queries
        queries_file = output_path / "queries.jsonl"
        with open(queries_file, "w") as f:
            for log in self._logs:
                # Use corrected intent if available
                intent = log.correct_intent if log.correct_intent else log.intent
                
                # Only include high-confidence or corrected queries
                if log.confidence >= 0.7 or log.correct_intent is not None:
                    f.write(json.dumps({
                        "text": log.query,
                        "label": intent,
                        "confidence": log.confidence,
                        "corrected": log.was_corrected,
                    }) + "\n")
        
        logger.info(f"Exported training data to {output_path}")
        return output_path


# =============================================================================
# LLM FALLBACK CLASSIFIER
# =============================================================================

class LLMIntentClassifier:
    """
    Use LLM for intent classification on ambiguous queries.
    
    This is the fallback when pattern matching and semantic
    similarity both have low confidence.
    """
    
    # Prompt template for intent classification
    CLASSIFICATION_PROMPT = """You are a bioinformatics assistant. Classify the user's intent into one of these categories:

INTENTS:
- DATA_SCAN: Check/list local files and data
- DATA_SEARCH: Search public databases (GEO, ENCODE, TCGA, SRA)
- DATA_DOWNLOAD: Download a specific dataset
- WORKFLOW_CREATE: Create/generate an analysis workflow or pipeline
- WORKFLOW_VISUALIZE: Show/display workflow diagram
- JOB_SUBMIT: Run/execute/submit a job to the cluster
- JOB_STATUS: Check status of running jobs
- JOB_LOGS: View job logs or errors
- DIAGNOSE_ERROR: Debug/troubleshoot a failure
- ANALYSIS_INTERPRET: Interpret or explain results
- REFERENCE_CHECK: Check if reference genome exists
- REFERENCE_DOWNLOAD: Download reference genome
- EDUCATION_EXPLAIN: Explain a concept or term
- EDUCATION_HELP: Show help/capabilities
- COMPOSITE_CHECK_THEN_SEARCH: First check locally, then search online
- META_CONFIRM: User confirmation (yes, ok, proceed)
- META_CANCEL: User cancellation (no, stop, cancel)
- META_GREETING: Hello, hi, greetings
- META_UNKNOWN: Cannot determine intent

USER QUERY: {query}

CONTEXT (if any): {context}

Respond with ONLY a JSON object:
{{"intent": "INTENT_NAME", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

    def __init__(self, llm_client):
        """
        Initialize with an LLM client.
        
        Args:
            llm_client: LLM client with generate() method
        """
        self.llm_client = llm_client
    
    def classify(
        self,
        query: str,
        context: str = "",
    ) -> Tuple[str, float, str]:
        """
        Classify query using LLM.
        
        Args:
            query: User query
            context: Optional conversation context
            
        Returns:
            Tuple of (intent, confidence, reasoning)
        """
        if not self.llm_client:
            return "META_UNKNOWN", 0.0, "No LLM client available"
        
        prompt = self.CLASSIFICATION_PROMPT.format(
            query=query,
            context=context or "None",
        )
        
        try:
            response = self.llm_client.generate(
                prompt,
                max_tokens=150,
                temperature=0.1,  # Low temperature for consistency
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                return (
                    result.get("intent", "META_UNKNOWN"),
                    float(result.get("confidence", 0.5)),
                    result.get("reasoning", "")
                )
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
        
        return "META_UNKNOWN", 0.0, "LLM classification failed"


# =============================================================================
# FEEDBACK API
# =============================================================================

class FeedbackManager:
    """
    Manages user feedback and model improvement.
    
    Provides:
    - API for submitting feedback
    - Automatic retraining triggers
    - Feedback-based intent boosting
    """
    
    def __init__(
        self,
        query_logger: QueryLogger = None,
        retrain_threshold: int = 50,
    ):
        """
        Initialize feedback manager.
        
        Args:
            query_logger: QueryLogger instance
            retrain_threshold: Number of corrections before suggesting retrain
        """
        self.logger = query_logger or QueryLogger()
        self.retrain_threshold = retrain_threshold
        
        # In-memory corrections for real-time boosting
        self._corrections: Dict[str, str] = {}  # query_hash -> correct_intent
        self._intent_boosts: Dict[str, float] = {}  # intent -> boost factor
        
        # Load corrections from log
        self._load_corrections()
    
    def _load_corrections(self):
        """Load corrections for real-time boosting."""
        for log in self.logger.get_corrected_queries():
            self._corrections[log.query_hash] = log.correct_intent
    
    def submit_feedback(
        self,
        query: str,
        detected_intent: str,
        correct_intent: str,
        feedback_text: str = None,
    ) -> Dict[str, Any]:
        """
        Submit user feedback for a query.
        
        Args:
            query: The original query
            detected_intent: What the system detected
            correct_intent: What it should have been
            feedback_text: Optional user comment
            
        Returns:
            Status dict with feedback_id and stats
        """
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:12]
        
        # Log the correction
        success = self.logger.add_feedback(
            query_hash=query_hash,
            correct_intent=correct_intent,
            feedback=feedback_text,
        )
        
        if not success:
            # Query wasn't logged, log it now with correction
            self.logger.log(
                query=query,
                intent=detected_intent,
                confidence=0.0,
                parse_method="user_correction",
            )
            self.logger.add_feedback(
                query_hash=query_hash,
                correct_intent=correct_intent,
                feedback=feedback_text,
            )
        
        # Update in-memory corrections
        self._corrections[query_hash] = correct_intent
        
        # Check if retrain is needed
        stats = self.logger.get_stats()
        needs_retrain = stats.corrected_queries >= self.retrain_threshold
        
        return {
            "feedback_id": query_hash,
            "status": "accepted",
            "corrections_count": stats.corrected_queries,
            "needs_retrain": needs_retrain,
            "message": "Retraining recommended" if needs_retrain else "Feedback recorded",
        }
    
    def get_correction(self, query: str) -> Optional[str]:
        """
        Get cached correction for a query.
        
        Use this for instant corrections without re-parsing.
        """
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:12]
        return self._corrections.get(query_hash)
    
    def get_intent_boost(self, intent: str) -> float:
        """
        Get boost factor for an intent based on feedback patterns.
        
        Intents that are frequently corrected TO get a boost.
        Intents that are frequently corrected FROM get a penalty.
        """
        return self._intent_boosts.get(intent, 1.0)
    
    def update_boosts(self):
        """Recalculate intent boosts from feedback."""
        stats = self.logger.get_stats()
        
        boost_to = defaultdict(float)
        penalize_from = defaultdict(float)
        
        for from_intent, to_intent, count in stats.common_corrections:
            boost_to[to_intent] += count * 0.1
            penalize_from[from_intent] += count * 0.05
        
        self._intent_boosts = {}
        all_intents = set(boost_to.keys()) | set(penalize_from.keys())
        
        for intent in all_intents:
            boost = 1.0 + boost_to.get(intent, 0) - penalize_from.get(intent, 0)
            self._intent_boosts[intent] = max(0.5, min(1.5, boost))
    
    def get_pending_reviews(self, limit: int = 20) -> List[Dict]:
        """Get queries pending human review."""
        queries = self.logger.get_low_confidence_queries(limit=limit)
        return [
            {
                "query_hash": q.query_hash,
                "query": q.query,
                "detected_intent": q.intent,
                "confidence": q.confidence,
                "entities": q.entities,
                "timestamp": q.timestamp,
            }
            for q in queries
            if q.correct_intent is None
        ]
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        stats = self.logger.get_stats()
        return {
            "total_queries": stats.total_queries,
            "low_confidence": stats.low_confidence_queries,
            "corrected": stats.corrected_queries,
            "intent_accuracy": stats.intent_accuracy,
            "common_corrections": stats.common_corrections,
            "correction_rate": stats.corrected_queries / max(stats.low_confidence_queries, 1),
        }


# =============================================================================
# FINE-TUNING SUPPORT
# =============================================================================

class FineTuningExporter:
    """
    Export training data for fine-tuning sentence transformers.
    
    Generates training pairs in formats suitable for:
    - sentence-transformers (contrastive learning)
    - SetFit (few-shot classification)
    - OpenAI fine-tuning
    """
    
    def __init__(self, query_logger: QueryLogger):
        self.logger = query_logger
    
    def export_sentence_transformer_format(
        self,
        output_dir: Path,
        include_corrections: bool = True,
    ) -> Dict[str, Path]:
        """
        Export for sentence-transformers training.
        
        Creates:
        - train.jsonl: {"text": query, "label": intent}
        - Similar pairs for contrastive learning
        
        Args:
            output_dir: Output directory
            include_corrections: Include corrected queries
            
        Returns:
            Dict of file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get high-quality training examples
        examples = []
        for log in self.logger._logs:
            intent = log.correct_intent if log.correct_intent else log.intent
            if log.confidence >= 0.7 or log.correct_intent is not None:
                examples.append({"text": log.query, "label": intent})
        
        # Add from INTENT_EXAMPLES
        from .semantic import INTENT_EXAMPLES
        for intent, queries in INTENT_EXAMPLES.items():
            for query in queries:
                examples.append({"text": query, "label": intent})
        
        # Write training file
        train_file = output_dir / "train.jsonl"
        with open(train_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        
        # Generate contrastive pairs (same intent = similar, different = dissimilar)
        pairs = []
        by_intent = defaultdict(list)
        for ex in examples:
            by_intent[ex["label"]].append(ex["text"])
        
        for intent, queries in by_intent.items():
            # Positive pairs (same intent)
            for i, q1 in enumerate(queries):
                for q2 in queries[i+1:i+3]:  # Max 2 positive pairs per query
                    pairs.append({"text1": q1, "text2": q2, "label": 1})
            
            # Negative pairs (different intents)
            other_intents = [k for k in by_intent.keys() if k != intent]
            for q1 in queries[:5]:  # Sample from each intent
                for other in other_intents[:3]:
                    if by_intent[other]:
                        q2 = by_intent[other][0]
                        pairs.append({"text1": q1, "text2": q2, "label": 0})
        
        pairs_file = output_dir / "pairs.jsonl"
        with open(pairs_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        
        # Write metadata
        meta_file = output_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump({
                "num_examples": len(examples),
                "num_pairs": len(pairs),
                "intents": list(by_intent.keys()),
                "exported_at": datetime.now().isoformat(),
            }, f, indent=2)
        
        logger.info(f"Exported {len(examples)} examples and {len(pairs)} pairs to {output_dir}")
        
        return {
            "train": train_file,
            "pairs": pairs_file,
            "metadata": meta_file,
        }
    
    def export_setfit_format(self, output_dir: Path) -> Path:
        """
        Export for SetFit few-shot training.
        
        SetFit requires few labeled examples per class.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect examples by intent
        by_intent = defaultdict(list)
        
        from .semantic import INTENT_EXAMPLES
        for intent, queries in INTENT_EXAMPLES.items():
            by_intent[intent].extend(queries)
        
        # Add corrected queries
        for log in self.logger.get_corrected_queries():
            by_intent[log.correct_intent].append(log.query)
        
        # Export with 8 examples per class (SetFit default)
        train_data = []
        for intent, queries in by_intent.items():
            for query in queries[:8]:
                train_data.append({"text": query, "label": intent})
        
        train_file = output_dir / "setfit_train.jsonl"
        with open(train_file, "w") as f:
            for ex in train_data:
                f.write(json.dumps(ex) + "\n")
        
        return train_file
    
    def get_fine_tuning_instructions(self) -> str:
        """Get instructions for fine-tuning."""
        return """
# Fine-Tuning Sentence Transformers for BioPipelines Intent Classification

## Prerequisites
```bash
pip install sentence-transformers setfit datasets
```

## Option 1: Contrastive Learning with sentence-transformers

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import json

# Load training pairs
pairs = []
with open("pairs.jsonl") as f:
    for line in f:
        data = json.loads(line)
        pairs.append(InputExample(
            texts=[data["text1"], data["text2"]],
            label=float(data["label"])
        ))

# Load base model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Train with contrastive loss
train_dataloader = DataLoader(pairs, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./biopipelines-intent-model"
)
```

## Option 2: SetFit (Recommended for Few-Shot)

```python
from setfit import SetFitModel, Trainer
from datasets import Dataset
import json

# Load training data
train_data = []
with open("setfit_train.jsonl") as f:
    for line in f:
        train_data.append(json.loads(line))

dataset = Dataset.from_list(train_data)

# Train SetFit model
model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    num_iterations=20,
)
trainer.train()
model.save_pretrained("./biopipelines-setfit-model")
```

## Using the Fine-Tuned Model

```python
from workflow_composer.agents.intent import SemanticIntentClassifier

# Use custom model
classifier = SemanticIntentClassifier(
    embedding_model="./biopipelines-intent-model"
)
```
"""


# =============================================================================
# ENHANCED HYBRID PARSER WITH LEARNING
# =============================================================================

class LearningHybridParser:
    """
    Enhanced HybridQueryParser with active learning and feedback.
    
    Adds:
    - Automatic logging of low-confidence queries
    - LLM fallback for ambiguous cases
    - Real-time corrections from feedback
    - Intent boosting based on feedback patterns
    """
    
    def __init__(
        self,
        llm_client = None,
        log_queries: bool = True,
        use_llm_fallback: bool = True,
        llm_fallback_threshold: float = 0.3,
    ):
        """
        Initialize the learning parser.
        
        Args:
            llm_client: LLM client for fallback classification
            log_queries: Whether to log queries for learning
            use_llm_fallback: Use LLM for low-confidence queries
            llm_fallback_threshold: Confidence below which to use LLM
        """
        from .semantic import HybridQueryParser
        
        self.base_parser = HybridQueryParser(llm_client=llm_client)
        self.llm_client = llm_client
        self.log_queries = log_queries
        self.use_llm_fallback = use_llm_fallback
        self.llm_fallback_threshold = llm_fallback_threshold
        
        # Initialize learning components
        self.query_logger = QueryLogger() if log_queries else None
        self.feedback_manager = FeedbackManager(self.query_logger) if log_queries else None
        self.llm_classifier = LLMIntentClassifier(llm_client) if llm_client else None
    
    def parse(
        self,
        query: str,
        context: Optional[Dict] = None,
        session_id: str = None,
    ):
        """
        Parse query with learning enhancements.
        
        Args:
            query: User query
            context: Conversation context
            session_id: Session identifier for logging
            
        Returns:
            QueryParseResult with intent, entities, and metadata
        """
        # Check for cached correction first
        if self.feedback_manager:
            correction = self.feedback_manager.get_correction(query)
            if correction:
                # Return corrected intent immediately
                from .semantic import QueryParseResult
                return QueryParseResult(
                    intent=correction,
                    intent_confidence=1.0,
                    parse_method="cached_correction",
                    entities=self.base_parser.ner.extract(query),
                    slots={},
                )
        
        # Parse with base parser
        result = self.base_parser.parse(query, context)
        
        # Apply intent boosts from feedback
        if self.feedback_manager and result.intent_confidence < 0.9:
            boost = self.feedback_manager.get_intent_boost(result.intent)
            result.intent_confidence = min(result.intent_confidence * boost, 1.0)
        
        # LLM fallback for low confidence
        if (self.use_llm_fallback 
            and self.llm_classifier 
            and result.intent_confidence < self.llm_fallback_threshold):
            
            llm_intent, llm_conf, reasoning = self.llm_classifier.classify(
                query,
                context=json.dumps(context) if context else ""
            )
            
            if llm_conf > result.intent_confidence:
                result.intent = llm_intent
                result.intent_confidence = llm_conf
                result.parse_method = "llm_fallback"
        
        # Log query
        if self.query_logger:
            self.query_logger.log(
                query=query,
                intent=result.intent,
                confidence=result.intent_confidence,
                parse_method=result.parse_method,
                entities=[{"type": e.entity_type, "text": e.text, "canonical": e.canonical} 
                         for e in result.entities],
                slots=result.slots,
                session_id=session_id,
            )
        
        return result
    
    def submit_feedback(
        self,
        query: str,
        correct_intent: str,
        feedback_text: str = None,
    ) -> Dict[str, Any]:
        """Submit user feedback for a query."""
        if not self.feedback_manager:
            return {"status": "error", "message": "Logging not enabled"}
        
        # Get the detected intent
        result = self.base_parser.parse(query)
        
        return self.feedback_manager.submit_feedback(
            query=query,
            detected_intent=result.intent,
            correct_intent=correct_intent,
            feedback_text=feedback_text,
        )
    
    def get_pending_reviews(self, limit: int = 20) -> List[Dict]:
        """Get queries pending human review."""
        if not self.feedback_manager:
            return []
        return self.feedback_manager.get_pending_reviews(limit)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.feedback_manager:
            return {}
        return self.feedback_manager.get_feedback_stats()
    
    def export_training_data(self, output_dir: Path = None) -> Dict[str, Path]:
        """Export training data for fine-tuning."""
        if not self.query_logger:
            return {}
        exporter = FineTuningExporter(self.query_logger)
        return exporter.export_sentence_transformer_format(
            output_dir or Path.home() / ".biopipelines" / "training_data"
        )
