# BioPipelines Chat Agent - Comprehensive Implementation Roadmap

## Executive Summary

This document outlines a systematic approach to building a production-grade, highly accurate bioinformatics chat agent. Based on our evaluation journey (43.9% → 91.4% pass rate), we've identified key areas for improvement and expansion.

### Current State (v11 - November 2025)
| Metric | Value | Target |
|--------|-------|--------|
| Pass Rate | 91.4% | 95%+ |
| Intent Accuracy | 95.5% | 98%+ |
| Entity F1 | 93.2% | 95%+ |
| Tool Accuracy | 96.7% | 98%+ |
| Latency | 10.2ms | <100ms |

### Vision
Build a self-improving, multi-model chat agent that:
- Achieves 98%+ accuracy on diverse bioinformatics queries
- Gracefully handles ambiguity through clarification
- Learns from user interactions continuously
- Leverages both local and cloud LLMs optimally
- Scales to support multiple concurrent users

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1: Foundation Improvements](#2-phase-1-foundation-improvements)
3. [Phase 2: Multi-Model Intelligence](#3-phase-2-multi-model-intelligence)
4. [Phase 3: Continuous Learning](#4-phase-3-continuous-learning)
5. [Phase 4: Advanced Features](#5-phase-4-advanced-features)
6. [Implementation Timeline](#6-implementation-timeline)
7. [Technical Specifications](#7-technical-specifications)
8. [Success Metrics](#8-success-metrics)

---

## 1. Architecture Overview

### 1.1 Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (Gradio Web / CLI / API)                     │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        UnifiedAgent                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Intent    │  │   Entity    │  │    Tool     │             │
│  │   Parser    │  │   Extractor │  │   Router    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UnifiedEnsembleParser                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │ Rule   │ │Semantic│ │  NER   │ │  LLM   │ │  RAG   │        │
│  │ 0.25   │ │  0.30  │ │  0.20  │ │  0.15  │ │  0.10  │        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Orchestrator                         │
│  ┌─────────────────────┐    ┌─────────────────────┐            │
│  │   Local Providers   │    │   Cloud Providers   │            │
│  │  • vLLM (MiniMax)   │    │  • OpenAI (GPT-4)   │            │
│  │  • Ollama           │    │  • Anthropic        │            │
│  │  • HuggingFace      │    │  • Google Gemini    │            │
│  └─────────────────────┘    └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Tool Execution                           │
│  29 Tools across 6 Categories:                                  │
│  • Data Discovery  • Workflow Generation  • Job Management      │
│  • Reference Management  • Diagnostics  • Education             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Components

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| Intent Parser | Ensemble (5 methods) | Federated experts + fine-tuned | High |
| Entity Extraction | Pattern + NER | Knowledge graph + LLM | High |
| Confidence Handling | Fixed threshold | Adaptive clarification | Critical |
| Multi-turn Context | Basic slots | Full entity memory | High |
| Model Selection | Strategy-based | Cost-performance optimized | Medium |
| Learning | Manual iteration | Continuous RLHF | Medium |

### 1.3 Target Architecture (Post-Roadmap)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Modal Interface                        │
│         (Text / Images / Files / Voice / Code)                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Clarification Layer                          │
│  • Confidence-based routing                                     │
│  • Ambiguity detection                                          │
│  • Context-aware follow-up questions                            │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Agentic Query Processor                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Federated  │  │  Fine-tuned │  │  Knowledge  │             │
│  │   Experts   │  │   Models    │  │    Graph    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Conversation State Manager                     │
│  • Entity memory across turns                                   │
│  • Context carryover                                            │
│  • Session persistence                                          │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Continuous Learning Pipeline                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Active    │  │   RLHF      │  │   A/B       │             │
│  │  Learning   │  │  Feedback   │  │  Testing    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```


---

## 2. Phase 1: Foundation Improvements

**Timeline**: Weeks 1-4  
**Goal**: Achieve 95%+ pass rate with immediate, low-risk enhancements

### 2.1 Confidence-Based Clarification System

**Priority**: CRITICAL  
**Estimated Improvement**: +2-3% pass rate  
**Effort**: 3-5 days

#### 2.1.1 Problem Statement
Currently, the system makes deterministic decisions regardless of confidence level. Low-confidence parses often result in incorrect tool invocations.

#### 2.1.2 Implementation

**File**: `src/workflow_composer/agents/clarification/__init__.py` (NEW)

```python
"""
Confidence-based clarification system for ambiguous queries.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

class ConfidenceLevel(Enum):
    HIGH = "high"      # > 0.85 - proceed
    MEDIUM = "medium"  # 0.60 - 0.85 - soft clarification
    LOW = "low"        # 0.40 - 0.60 - strong clarification
    UNKNOWN = "unknown"  # < 0.40 - admit uncertainty

@dataclass
class ClarificationRequest:
    """Request for user clarification."""
    query: str
    confidence: float
    level: ConfidenceLevel
    suggested_intents: List[str]
    clarifying_question: str
    slot_requests: Dict[str, str]  # slot -> question

@dataclass 
class ClarificationConfig:
    """Configuration for clarification thresholds."""
    high_threshold: float = 0.85
    medium_threshold: float = 0.60
    low_threshold: float = 0.40
    max_clarifications_per_turn: int = 2
    enable_soft_clarification: bool = True
```

**File**: `src/workflow_composer/agents/clarification/clarifier.py` (NEW)

```python
"""
Clarification engine that generates context-aware follow-up questions.
"""
from typing import List, Dict, Any, Optional
from . import ClarificationRequest, ClarificationConfig, ConfidenceLevel
from ..intent.ensemble import EnsembleParseResult

class ClarificationEngine:
    """Generates clarifying questions based on parse confidence."""
    
    def __init__(self, config: Optional[ClarificationConfig] = None):
        self.config = config or ClarificationConfig()
        self._intent_questions = self._load_intent_questions()
        self._slot_questions = self._load_slot_questions()
    
    def should_clarify(self, result: EnsembleParseResult) -> bool:
        """Determine if clarification is needed."""
        level = self._get_confidence_level(result.confidence)
        return level in (ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW, 
                         ConfidenceLevel.UNKNOWN)
    
    def generate_clarification(
        self, query: str, result: EnsembleParseResult,
        context: Optional[Dict[str, Any]] = None
    ) -> ClarificationRequest:
        """Generate a clarification request for ambiguous query."""
        level = self._get_confidence_level(result.confidence)
        suggested_intents = self._get_suggested_intents(result)
        question = self._generate_question(level, result, suggested_intents)
        slot_requests = self._get_missing_slot_questions(result)
        
        return ClarificationRequest(
            query=query, confidence=result.confidence, level=level,
            suggested_intents=suggested_intents, clarifying_question=question,
            slot_requests=slot_requests
        )
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        if confidence >= self.config.high_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.config.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.config.low_threshold:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNKNOWN
    
    def _load_intent_questions(self) -> Dict[str, str]:
        return {
            "workflow_generation": "Create a bioinformatics workflow",
            "data_discovery": "Search for datasets in GEO/ENCODE/SRA",
            "job_submit": "Submit a job to the cluster",
            "job_status": "Check the status of running jobs",
            "reference_download": "Download a reference genome",
            "education": "Learn about bioinformatics concepts",
        }
    
    def _load_slot_questions(self) -> Dict[str, str]:
        return {
            "organism": "Which organism (e.g., human, mouse)?",
            "assay_type": "What type of assay (e.g., RNA-seq, ChIP-seq)?",
            "dataset_id": "Specific dataset ID (e.g., GSE12345)?",
        }
```

**Integration Point**: `src/workflow_composer/agents/unified_agent.py`

```python
# Add to UnifiedAgent.process_message()
from .clarification import ClarificationEngine, ClarificationConfig

class UnifiedAgent:
    def __init__(self, ...):
        self.clarifier = ClarificationEngine(ClarificationConfig())
    
    async def process_message(self, message: str, ...) -> AgentResponse:
        result = self._parser.parse(message)
        
        # Check if clarification needed
        if self.clarifier.should_clarify(result):
            clarification = self.clarifier.generate_clarification(message, result)
            return AgentResponse(
                response=clarification.clarifying_question,
                needs_clarification=True,
                clarification_data=clarification,
                confidence=result.confidence
            )
        # Proceed with tool execution...
```

### 2.2 Real Production Query Integration

**Priority**: HIGH  
**Estimated Improvement**: +1-2% pass rate  
**Effort**: 2-3 days

#### 2.2.1 Problem Statement
Current test conversations are synthetic. Real user queries have unpredictable patterns, typos, and domain-specific jargon.

#### 2.2.2 Implementation

**File**: `tests/evaluation/production_queries.py` (NEW)

```python
"""Collect and integrate real production queries for testing."""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class ProductionQuery:
    """A real query from production usage."""
    query: str
    timestamp: datetime
    user_id_hash: str  # Anonymized
    session_id: str
    parsed_intent: Optional[str] = None
    parsed_entities: Optional[Dict[str, str]] = None
    tool_executed: Optional[str] = None
    execution_success: Optional[bool] = None
    user_feedback: Optional[str] = None
    
class ProductionQueryCollector:
    """Collects queries from production for evaluation purposes."""
    
    def __init__(self, db_path: str = "production_queries.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS production_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                query_hash TEXT UNIQUE,
                timestamp DATETIME,
                user_id_hash TEXT,
                session_id TEXT,
                parsed_intent TEXT,
                parsed_entities TEXT,
                tool_executed TEXT,
                execution_success BOOLEAN,
                user_feedback TEXT,
                added_to_eval BOOLEAN DEFAULT FALSE,
                reviewed BOOLEAN DEFAULT FALSE
            )
        ''')
        conn.commit()
        conn.close()
    
    def collect(self, query: ProductionQuery) -> bool:
        """Collect a production query. Returns True if new, False if duplicate."""
        query_hash = hashlib.sha256(
            query.query.lower().strip().encode()
        ).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO production_queries 
                (query, query_hash, timestamp, user_id_hash, session_id,
                 parsed_intent, tool_executed, execution_success, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (query.query, query_hash, query.timestamp.isoformat(),
                  query.user_id_hash, query.session_id, query.parsed_intent,
                  query.tool_executed, query.execution_success, query.user_feedback))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
```

### 2.3 Active Learning Pipeline

**Priority**: HIGH  
**Estimated Improvement**: Continuous improvement  
**Effort**: 5-7 days

#### 2.3.1 Implementation

**File**: `src/workflow_composer/agents/learning/active_learner.py` (NEW)

```python
"""Active learning system for identifying high-value training examples."""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass

class QueryDifficulty(Enum):
    EASY = "easy"           # High confidence, correct
    MODERATE = "moderate"   # Medium confidence, correct
    HARD = "hard"           # Low confidence, correct
    FAILURE = "failure"     # Any confidence, incorrect

@dataclass
class LearningSignal:
    query: str
    predicted_intent: str
    predicted_entities: Dict[str, str]
    confidence: float
    actual_intent: str = None
    user_corrected: bool = False

class ActiveLearner:
    """Identifies high-value training examples through uncertainty sampling."""
    
    def __init__(self, db_path: str = "active_learning.db"):
        self.db_path = Path(db_path)
        self._init_db()
        self._failure_patterns = defaultdict(list)
    
    def record_signal(self, signal: LearningSignal):
        """Record a learning signal from agent execution."""
        difficulty = self._assess_difficulty(signal)
        # Store in database for later analysis
        ...
    
    def _assess_difficulty(self, signal: LearningSignal) -> QueryDifficulty:
        is_correct = (signal.actual_intent is None or 
                      signal.predicted_intent == signal.actual_intent)
        if not is_correct:
            return QueryDifficulty.FAILURE
        if signal.confidence >= 0.85:
            return QueryDifficulty.EASY
        elif signal.confidence >= 0.60:
            return QueryDifficulty.MODERATE
        return QueryDifficulty.HARD
    
    def get_priority_queries(self, n: int = 50) -> List[Dict[str, Any]]:
        """Get highest priority queries for manual review/training."""
        # Prioritizes: failures > low confidence > user corrections
        ...
    
    def generate_training_batch(self, batch_size: int = 100) -> List[Dict]:
        """Generate balanced training batch emphasizing hard examples."""
        # 40% failures, 30% hard, 20% moderate, 10% easy
        ...
```

### 2.4 Negation and Comparative Handling

**Priority**: HIGH  
**Estimated Improvement**: +1-2% pass rate  
**Effort**: 2-3 days

#### 2.4.1 Implementation

**File**: `src/workflow_composer/agents/intent/negation_handler.py` (NEW)

```python
"""Handler for negation and comparative linguistic patterns."""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class NegationResult:
    has_negation: bool
    negated_terms: List[str]
    preserved_terms: List[str]
    negation_type: str  # 'exclusion', 'preference', 'correction'

class NegationHandler:
    """Detects and handles negation patterns in queries."""
    
    NEGATION_PATTERNS = [
        (r"(?:not|no|without|except)\s+(\w+[-\w]*)", "exclusion"),
        (r"(?:don't|do not)\s+(?:want|need|use)\s+(\w+[-\w]*)", "preference"),
        (r"(?:not)\s+(\w+[-\w]*)[,\s]+(?:but|instead)\s+(\w+[-\w]*)", "correction"),
    ]
    
    def detect_negation(self, query: str) -> NegationResult:
        """Detect negation patterns in query."""
        query_lower = query.lower()
        negated_terms = []
        preserved_terms = []
        negation_type = None
        
        for pattern, neg_type in self.NEGATION_PATTERNS:
            matches = re.findall(pattern, query_lower)
            if matches:
                negation_type = neg_type
                for match in matches:
                    if isinstance(match, tuple):
                        negated_terms.append(match[0])
                        preserved_terms.append(match[1])
                    else:
                        negated_terms.append(match)
        
        return NegationResult(
            has_negation=len(negated_terms) > 0,
            negated_terms=list(set(negated_terms)),
            preserved_terms=list(set(preserved_terms)),
            negation_type=negation_type or ""
        )
    
    def transform_query(self, query: str, result: NegationResult) -> str:
        """Add explicit markers for downstream processing."""
        if not result.has_negation:
            return query
        
        negation_marker = f"[EXCLUDE: {', '.join(result.negated_terms)}]"
        return f"{query} {negation_marker}"
```

---

## 3. Phase 2: Multi-Model Intelligence

**Timeline**: Weeks 5-10  
**Goal**: Leverage multiple LLMs for optimal accuracy and cost

### 3.1 Fine-Tuned Domain-Specific LLM

**Priority**: HIGH  
**Estimated Improvement**: +3-5% pass rate  
**Effort**: 2-3 weeks

#### 3.1.1 Problem Statement
Generic LLMs lack bioinformatics domain knowledge. Fine-tuning on domain-specific data improves accuracy significantly.

#### 3.1.2 Training Data Preparation

**File**: `scripts/prepare_finetuning_data.py` (NEW)

```python
#!/usr/bin/env python3
"""
Prepare training data for fine-tuning a bioinformatics-specific LLM.
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import random

class FineTuningDataPreparer:
    """Prepares training data in various formats for fine-tuning."""
    
    def __init__(self, eval_db_path: str, output_dir: str = "finetuning_data"):
        self.eval_db = Path(eval_db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_conversations(self) -> List[Dict[str, Any]]:
        """Load conversations from evaluation database."""
        conn = sqlite3.connect(self.eval_db)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT query, expected_intent, expected_entities, 
                   expected_tool, category
            FROM conversations
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "query": row[0],
                "intent": row[1],
                "entities": json.loads(row[2]) if row[2] else {},
                "tool": row[3],
                "category": row[4]
            }
            for row in rows
        ]
    
    def create_openai_format(self, conversations: List[Dict]) -> List[Dict]:
        """Create OpenAI fine-tuning format (JSONL)."""
        formatted = []
        
        system_prompt = """You are a bioinformatics assistant that parses 
user queries and extracts:
1. Intent: The user's goal (workflow_generation, data_discovery, etc.)
2. Entities: Key entities like organism, assay_type, dataset_id
3. Tool: The appropriate tool to execute

Respond in JSON format."""

        for conv in conversations:
            response = {
                "intent": conv["intent"],
                "entities": conv["entities"],
                "tool": conv["tool"]
            }
            
            formatted.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conv["query"]},
                    {"role": "assistant", "content": json.dumps(response)}
                ]
            })
        
        return formatted
    
    def create_llama_format(self, conversations: List[Dict]) -> List[Dict]:
        """Create Llama/Alpaca fine-tuning format."""
        formatted = []
        
        for conv in conversations:
            instruction = """Parse this bioinformatics query and extract:
- intent: user's goal
- entities: key entities (organism, assay_type, dataset_id, gene, etc.)
- tool: appropriate tool to call"""
            
            response = json.dumps({
                "intent": conv["intent"],
                "entities": conv["entities"],
                "tool": conv["tool"]
            }, indent=2)
            
            formatted.append({
                "instruction": instruction,
                "input": conv["query"],
                "output": response
            })
        
        return formatted
    
    def split_train_val_test(
        self, 
        data: List[Dict], 
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> tuple:
        """Split data into train/val/test sets with stratification."""
        random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (
            data[:train_end],
            data[train_end:val_end],
            data[val_end:]
        )
    
    def save_jsonl(self, data: List[Dict], filename: str):
        """Save data in JSONL format."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Saved {len(data)} examples to {path}")
    
    def prepare_all(self):
        """Prepare data in all formats."""
        conversations = self.load_conversations()
        print(f"Loaded {len(conversations)} conversations")
        
        # OpenAI format
        openai_data = self.create_openai_format(conversations)
        train, val, test = self.split_train_val_test(openai_data)
        self.save_jsonl(train, "openai_train.jsonl")
        self.save_jsonl(val, "openai_val.jsonl")
        self.save_jsonl(test, "openai_test.jsonl")
        
        # Llama format
        llama_data = self.create_llama_format(conversations)
        train, val, test = self.split_train_val_test(llama_data)
        self.save_jsonl(train, "llama_train.jsonl")
        self.save_jsonl(val, "llama_val.jsonl")
        self.save_jsonl(test, "llama_test.jsonl")

if __name__ == "__main__":
    preparer = FineTuningDataPreparer(
        "tests/evaluation/evaluation.db",
        "finetuning_data"
    )
    preparer.prepare_all()
```

#### 3.1.3 Fine-Tuning Script

**File**: `scripts/finetune_model.py` (NEW)

```python
#!/usr/bin/env python3
"""
Fine-tune a local LLM for bioinformatics query parsing.
Supports: Llama 2/3, Mistral, Phi-2 via unsloth/transformers
"""
import argparse
from pathlib import Path

def finetune_with_unsloth(
    base_model: str,
    train_data: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """Fine-tune using unsloth (4x faster, 60% less memory)."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    
    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
    )
    
    # Load dataset
    dataset = load_dataset('json', data_files=train_data)['train']
    
    # Format for training
    def formatting_func(examples):
        texts = []
        for inst, inp, out in zip(examples['instruction'], 
                                   examples['input'], 
                                   examples['output']):
            text = f"""### Instruction:
{inst}

### Input:
{inp}

### Response:
{out}"""
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_func, batched=True)
    
    # Training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            output_dir=output_dir,
            save_strategy="epoch",
        ),
    )
    
    trainer.train()
    
    # Save final model
    model.save_pretrained_merged(
        f"{output_dir}/final",
        tokenizer,
        save_method="merged_16bit"
    )
    print(f"Model saved to {output_dir}/final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", default="finetuned_model")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    finetune_with_unsloth(
        args.base_model, args.train_data, 
        args.output_dir, args.epochs
    )
```

### 3.2 LLM-as-Judge Evaluation

**Priority**: MEDIUM  
**Estimated Improvement**: Better evaluation accuracy  
**Effort**: 3-5 days

#### 3.2.1 Implementation

**File**: `tests/evaluation/llm_judge.py` (NEW)

```python
"""
LLM-as-Judge for nuanced evaluation of query parsing.
"""
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class JudgmentResult:
    """Result from LLM judge evaluation."""
    intent_correct: bool
    intent_reasoning: str
    entities_score: float  # 0.0 to 1.0
    entities_reasoning: str
    tool_correct: bool
    tool_reasoning: str
    overall_score: float
    suggestions: List[str]

class LLMJudge:
    """Uses LLM to evaluate query parsing quality."""
    
    JUDGE_PROMPT = """You are an expert evaluator for a bioinformatics
query parsing system. Evaluate the following:

**Query**: {query}

**Expected Result**:
- Intent: {expected_intent}
- Entities: {expected_entities}
- Tool: {expected_tool}

**Actual Result**:
- Intent: {actual_intent}
- Entities: {actual_entities}
- Tool: {actual_tool}

Evaluate each component:

1. **Intent**: Is the actual intent correct or semantically equivalent?
   Consider synonyms (e.g., "workflow_generation" ≈ "create_pipeline")

2. **Entities**: Score 0.0-1.0 based on:
   - Correct entities extracted
   - Missing entities (penalize)
   - Hallucinated entities (penalize heavily)
   Consider case-insensitive matching and synonyms

3. **Tool**: Is the tool selection appropriate for the intent?

Respond in JSON:
{{
    "intent_correct": true/false,
    "intent_reasoning": "...",
    "entities_score": 0.0-1.0,
    "entities_reasoning": "...",
    "tool_correct": true/false,
    "tool_reasoning": "...",
    "overall_score": 0.0-1.0,
    "suggestions": ["improvement suggestion 1", ...]
}}"""

    def __init__(self, orchestrator):
        """Initialize with ModelOrchestrator."""
        self.orchestrator = orchestrator
    
    async def evaluate(
        self,
        query: str,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> JudgmentResult:
        """Evaluate a single query parsing result."""
        prompt = self.JUDGE_PROMPT.format(
            query=query,
            expected_intent=expected.get('intent', ''),
            expected_entities=json.dumps(expected.get('entities', {})),
            expected_tool=expected.get('tool', ''),
            actual_intent=actual.get('intent', ''),
            actual_entities=json.dumps(actual.get('entities', {})),
            actual_tool=actual.get('tool', '')
        )
        
        # Use Claude or GPT-4 for best judgment quality
        response = await self.orchestrator.generate(
            prompt=prompt,
            strategy="CLOUD_PREFERRED",
            max_tokens=1000
        )
        
        try:
            result = json.loads(response.content)
            return JudgmentResult(
                intent_correct=result.get('intent_correct', False),
                intent_reasoning=result.get('intent_reasoning', ''),
                entities_score=result.get('entities_score', 0.0),
                entities_reasoning=result.get('entities_reasoning', ''),
                tool_correct=result.get('tool_correct', False),
                tool_reasoning=result.get('tool_reasoning', ''),
                overall_score=result.get('overall_score', 0.0),
                suggestions=result.get('suggestions', [])
            )
        except json.JSONDecodeError:
            # Fallback to simple evaluation
            return self._fallback_evaluate(expected, actual)
    
    def _fallback_evaluate(
        self, expected: Dict, actual: Dict
    ) -> JudgmentResult:
        """Simple fallback evaluation if LLM parsing fails."""
        intent_correct = expected.get('intent') == actual.get('intent')
        tool_correct = expected.get('tool') == actual.get('tool')
        
        # Simple entity matching
        exp_entities = set(expected.get('entities', {}).keys())
        act_entities = set(actual.get('entities', {}).keys())
        if exp_entities:
            entities_score = len(exp_entities & act_entities) / len(exp_entities)
        else:
            entities_score = 1.0 if not act_entities else 0.5
        
        return JudgmentResult(
            intent_correct=intent_correct,
            intent_reasoning="Exact match comparison",
            entities_score=entities_score,
            entities_reasoning="Set overlap comparison",
            tool_correct=tool_correct,
            tool_reasoning="Exact match comparison",
            overall_score=(
                (1.0 if intent_correct else 0.0) * 0.4 +
                entities_score * 0.3 +
                (1.0 if tool_correct else 0.0) * 0.3
            ),
            suggestions=[]
        )
```

### 3.3 Agentic Query Processor

**Priority**: MEDIUM  
**Estimated Improvement**: +2-3% on complex queries  
**Effort**: 1-2 weeks

#### 3.3.1 Problem Statement
Single-pass parsing fails on complex queries. Multi-step reasoning with tool use improves accuracy.

#### 3.3.2 Implementation

**File**: `src/workflow_composer/agents/intent/agentic_parser.py` (NEW)

```python
"""
Agentic query parser with multi-step reasoning and tool use.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

@dataclass
class ParsingStep:
    """A single step in the parsing process."""
    action: str
    input_data: Dict[str, Any]
    output: Any
    reasoning: str

@dataclass
class AgenticParseResult:
    """Result from agentic parsing."""
    intent: str
    entities: Dict[str, str]
    tool: str
    confidence: float
    steps: List[ParsingStep]
    total_tokens: int

class AgenticQueryParser:
    """
    Multi-step query parser that can:
    1. Decompose complex queries
    2. Use tools to verify entities
    3. Self-correct through reflection
    """
    
    SYSTEM_PROMPT = """You are a bioinformatics query parser agent.
You have access to these tools:

1. VERIFY_ORGANISM: Check if a term is a valid organism name
2. VERIFY_ASSAY: Check if a term is a valid assay type
3. VERIFY_DATASET: Check if an ID is a valid dataset identifier
4. LOOKUP_SYNONYM: Find canonical name for a term
5. DECOMPOSE: Break complex query into simpler parts

For each query, think step by step:
1. Identify the primary intent
2. Extract potential entities
3. Verify ambiguous entities using tools
4. Determine the appropriate tool to call
5. Express confidence in your result

Respond with your reasoning and final result."""

    def __init__(self, orchestrator, max_steps: int = 5):
        self.orchestrator = orchestrator
        self.max_steps = max_steps
        self._tools = self._init_tools()
    
    def _init_tools(self) -> Dict[str, callable]:
        """Initialize verification tools."""
        return {
            "VERIFY_ORGANISM": self._verify_organism,
            "VERIFY_ASSAY": self._verify_assay,
            "VERIFY_DATASET": self._verify_dataset,
            "LOOKUP_SYNONYM": self._lookup_synonym,
            "DECOMPOSE": self._decompose_query,
        }
    
    async def parse(self, query: str) -> AgenticParseResult:
        """Parse query with multi-step reasoning."""
        steps = []
        context = {"query": query, "partial_result": {}}
        
        for step_num in range(self.max_steps):
            # Generate next action
            action_prompt = self._build_action_prompt(context, steps)
            response = await self.orchestrator.generate(
                prompt=action_prompt,
                system=self.SYSTEM_PROMPT,
                max_tokens=500
            )
            
            # Parse action
            action_data = self._parse_action(response.content)
            
            if action_data["action"] == "DONE":
                # Final result
                return self._build_result(action_data, steps)
            
            # Execute tool if needed
            if action_data["action"] in self._tools:
                tool_result = await self._tools[action_data["action"]](
                    action_data.get("input", {})
                )
                action_data["output"] = tool_result
            
            steps.append(ParsingStep(
                action=action_data["action"],
                input_data=action_data.get("input", {}),
                output=action_data.get("output"),
                reasoning=action_data.get("reasoning", "")
            ))
            
            # Update context
            context["steps"] = steps
            context["partial_result"].update(
                action_data.get("partial_result", {})
            )
        
        # Max steps reached - return best effort
        return self._build_result({"partial_result": context["partial_result"]}, steps)
    
    async def _verify_organism(self, input_data: Dict) -> Dict:
        """Verify if term is a valid organism."""
        term = input_data.get("term", "")
        # Use NCBI taxonomy or local database
        valid_organisms = {
            "human", "mouse", "rat", "zebrafish", "drosophila",
            "c. elegans", "arabidopsis", "yeast", "e. coli"
        }
        canonical = {
            "homo sapiens": "human", "mus musculus": "mouse",
            "danio rerio": "zebrafish", "fly": "drosophila"
        }
        
        term_lower = term.lower()
        if term_lower in valid_organisms:
            return {"valid": True, "canonical": term_lower}
        if term_lower in canonical:
            return {"valid": True, "canonical": canonical[term_lower]}
        return {"valid": False, "suggestion": None}
    
    async def _verify_assay(self, input_data: Dict) -> Dict:
        """Verify if term is a valid assay type."""
        term = input_data.get("term", "")
        assay_synonyms = {
            "rna-seq": ["rnaseq", "rna sequencing", "transcriptomics"],
            "chip-seq": ["chipseq", "chip sequencing"],
            "atac-seq": ["atacseq", "atac sequencing"],
            "wgs": ["whole genome sequencing", "genome sequencing"],
            "wes": ["whole exome sequencing", "exome sequencing"],
        }
        
        term_lower = term.lower().replace("_", "-").replace(" ", "-")
        for canonical, synonyms in assay_synonyms.items():
            if term_lower == canonical or term_lower in synonyms:
                return {"valid": True, "canonical": canonical}
        return {"valid": False, "suggestion": None}
    
    async def _verify_dataset(self, input_data: Dict) -> Dict:
        """Verify if ID is a valid dataset identifier."""
        id_val = input_data.get("id", "")
        import re
        
        patterns = [
            (r"^GSE\d{4,8}$", "GEO"),
            (r"^ENCSR[A-Z0-9]{6,10}$", "ENCODE"),
            (r"^SRR\d{6,10}$", "SRA"),
            (r"^PRJNA\d+$", "BioProject"),
        ]
        
        for pattern, db in patterns:
            if re.match(pattern, id_val, re.IGNORECASE):
                return {"valid": True, "database": db}
        return {"valid": False, "database": None}
    
    async def _lookup_synonym(self, input_data: Dict) -> Dict:
        """Look up canonical name for a term."""
        term = input_data.get("term", "")
        # Could query knowledge base or LLM
        return {"canonical": term.lower(), "confidence": 0.8}
    
    async def _decompose_query(self, input_data: Dict) -> Dict:
        """Decompose complex query into parts."""
        query = input_data.get("query", "")
        # Use LLM to decompose
        prompt = f"Decompose this query into simple parts: {query}"
        response = await self.orchestrator.generate(prompt=prompt)
        return {"parts": response.content.split("\n")}
    
    def _build_action_prompt(
        self, context: Dict, steps: List[ParsingStep]
    ) -> str:
        """Build prompt for next action."""
        steps_text = "\n".join(
            f"Step {i+1}: {s.action} - {s.reasoning}"
            for i, s in enumerate(steps)
        )
        
        return f"""Query: {context['query']}

Previous steps:
{steps_text or "None"}

Current partial result: {json.dumps(context.get('partial_result', {}))}

What is your next action? Respond in JSON:
{{
    "action": "TOOL_NAME or DONE",
    "input": {{}},  # if using a tool
    "reasoning": "why this action",
    "partial_result": {{}}  # any updates to result
}}"""

    def _parse_action(self, response: str) -> Dict:
        """Parse LLM response into action dict."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"action": "DONE", "partial_result": {}}
    
    def _build_result(
        self, action_data: Dict, steps: List[ParsingStep]
    ) -> AgenticParseResult:
        """Build final result from action data."""
        result = action_data.get("partial_result", {})
        return AgenticParseResult(
            intent=result.get("intent", "unknown"),
            entities=result.get("entities", {}),
            tool=result.get("tool", "unknown"),
            confidence=result.get("confidence", 0.5),
            steps=steps,
            total_tokens=sum(len(s.reasoning) for s in steps)  # Approximate
        )
```

### 3.4 Model Routing and Cost Optimization

**Priority**: MEDIUM  
**Estimated Improvement**: 50% cost reduction  
**Effort**: 3-5 days

#### 3.4.1 Implementation

**File**: `src/workflow_composer/llm/smart_router.py` (NEW)

```python
"""
Intelligent model routing based on query complexity and cost.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class QueryComplexity(Enum):
    SIMPLE = "simple"      # Single intent, clear entities
    MODERATE = "moderate"  # Some ambiguity
    COMPLEX = "complex"    # Multi-part, ambiguous

@dataclass
class RoutingDecision:
    """Model routing decision."""
    model: str
    provider: str
    estimated_cost: float
    reasoning: str

class SmartModelRouter:
    """Routes queries to optimal model based on complexity and cost."""
    
    # Model capabilities and costs (per 1K tokens)
    MODEL_SPECS = {
        "llama-3-8b": {
            "provider": "local",
            "cost": 0.0,
            "max_complexity": QueryComplexity.MODERATE,
            "latency_ms": 50
        },
        "mistral-7b": {
            "provider": "local", 
            "cost": 0.0,
            "max_complexity": QueryComplexity.MODERATE,
            "latency_ms": 45
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "cost": 0.00015,
            "max_complexity": QueryComplexity.COMPLEX,
            "latency_ms": 300
        },
        "gpt-4o": {
            "provider": "openai",
            "cost": 0.005,
            "max_complexity": QueryComplexity.COMPLEX,
            "latency_ms": 500
        },
        "claude-3-5-sonnet": {
            "provider": "anthropic",
            "cost": 0.003,
            "max_complexity": QueryComplexity.COMPLEX,
            "latency_ms": 400
        }
    }
    
    def __init__(self, prefer_local: bool = True, budget_limit: float = None):
        self.prefer_local = prefer_local
        self.budget_limit = budget_limit
        self._usage_tracker = {"total_cost": 0.0}
    
    def assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity for routing."""
        query_lower = query.lower()
        
        # Complex indicators
        complex_patterns = [
            " and ", " or ", " then ", "compare", "multiple",
            "several", "both", "either"
        ]
        if any(p in query_lower for p in complex_patterns):
            return QueryComplexity.COMPLEX
        
        # Moderate indicators
        moderate_patterns = [
            "?", "which", "what", "how", "best"
        ]
        if any(p in query_lower for p in moderate_patterns):
            return QueryComplexity.MODERATE
        
        return QueryComplexity.SIMPLE
    
    def route(
        self, 
        query: str, 
        require_high_accuracy: bool = False
    ) -> RoutingDecision:
        """Route query to optimal model."""
        complexity = self.assess_complexity(query)
        
        # Filter models by capability
        capable_models = [
            (name, spec) for name, spec in self.MODEL_SPECS.items()
            if spec["max_complexity"].value >= complexity.value
        ]
        
        # Sort by preference
        if self.prefer_local:
            capable_models.sort(key=lambda x: (
                0 if x[1]["provider"] == "local" else 1,
                x[1]["cost"]
            ))
        else:
            capable_models.sort(key=lambda x: x[1]["cost"])
        
        if require_high_accuracy:
            # Prefer cloud models for high accuracy
            capable_models.sort(key=lambda x: (
                0 if x[1]["provider"] != "local" else 1,
                -x[1]["cost"]  # Higher cost usually = better
            ))
        
        # Select best option
        model_name, spec = capable_models[0]
        
        return RoutingDecision(
            model=model_name,
            provider=spec["provider"],
            estimated_cost=spec["cost"],
            reasoning=f"Selected for {complexity.value} query, "
                     f"prefer_local={self.prefer_local}"
        )
```

---

## 4. Phase 3: Continuous Learning

**Timeline**: Weeks 11-16  
**Goal**: Self-improving system with RLHF and A/B testing

### 4.1 RLHF (Reinforcement Learning from Human Feedback)

**Priority**: HIGH  
**Estimated Improvement**: +5-10% over time  
**Effort**: 3-4 weeks

#### 4.1.1 Problem Statement
Static models degrade over time as user needs evolve. RLHF enables continuous improvement from real user feedback.

#### 4.1.2 Feedback Collection

**File**: `src/workflow_composer/agents/learning/feedback_collector.py` (NEW)

```python
"""
Collect and process human feedback for RLHF.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class FeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    REPHRASE = "rephrase"

@dataclass
class Feedback:
    """User feedback on an agent response."""
    query: str
    response: str
    parsed_intent: str
    parsed_entities: Dict[str, str]
    executed_tool: str
    feedback_type: FeedbackType
    correction_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    user_id_hash: str = ""
    session_id: str = ""

class FeedbackCollector:
    """Collects and stores user feedback for RLHF training."""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT,
                parsed_intent TEXT,
                parsed_entities TEXT,
                executed_tool TEXT,
                feedback_type TEXT,
                correction_data TEXT,
                timestamp DATETIME,
                user_id_hash TEXT,
                session_id TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS preference_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                chosen_response TEXT,
                rejected_response TEXT,
                margin REAL,
                source TEXT,
                timestamp DATETIME
            )
        ''')
        conn.commit()
        conn.close()
    
    def record_feedback(self, feedback: Feedback):
        """Record a piece of user feedback."""
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback
            (query, response, parsed_intent, parsed_entities, 
             executed_tool, feedback_type, correction_data, 
             timestamp, user_id_hash, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.query,
            feedback.response,
            feedback.parsed_intent,
            json.dumps(feedback.parsed_entities),
            feedback.executed_tool,
            feedback.feedback_type.value,
            json.dumps(feedback.correction_data) if feedback.correction_data else None,
            (feedback.timestamp or datetime.now()).isoformat(),
            feedback.user_id_hash,
            feedback.session_id
        ))
        conn.commit()
        conn.close()
    
    def generate_preference_pairs(self) -> int:
        """Convert feedback into preference pairs for DPO training."""
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unprocessed corrections
        cursor.execute('''
            SELECT id, query, response, correction_data
            FROM feedback
            WHERE feedback_type = 'correction' AND processed = FALSE
        ''')
        corrections = cursor.fetchall()
        
        pairs_created = 0
        for id, query, orig_response, correction_json in corrections:
            correction = json.loads(correction_json) if correction_json else {}
            corrected_response = self._build_corrected_response(correction)
            
            cursor.execute('''
                INSERT INTO preference_pairs
                (query, chosen_response, rejected_response, margin, source, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                query,
                corrected_response,
                orig_response,
                1.0,  # User correction has highest margin
                "user_correction",
                datetime.now().isoformat()
            ))
            
            cursor.execute('UPDATE feedback SET processed = TRUE WHERE id = ?', (id,))
            pairs_created += 1
        
        conn.commit()
        conn.close()
        return pairs_created
    
    def _build_corrected_response(self, correction: Dict) -> str:
        """Build corrected response from correction data."""
        import json
        return json.dumps({
            "intent": correction.get("correct_intent"),
            "entities": correction.get("correct_entities", {}),
            "tool": correction.get("correct_tool")
        })
    
    def export_for_dpo(self, output_path: str) -> int:
        """Export preference pairs in DPO training format."""
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT query, chosen_response, rejected_response
            FROM preference_pairs
        ''')
        pairs = cursor.fetchall()
        conn.close()
        
        with open(output_path, 'w') as f:
            for query, chosen, rejected in pairs:
                f.write(json.dumps({
                    "prompt": query,
                    "chosen": chosen,
                    "rejected": rejected
                }) + '\n')
        
        return len(pairs)
```

#### 4.1.3 DPO Training Script

**File**: `scripts/train_dpo.py` (NEW)

```python
#!/usr/bin/env python3
"""
Train model with Direct Preference Optimization (DPO).
"""
import argparse
from pathlib import Path

def train_dpo(
    base_model: str,
    preference_data: str,
    output_dir: str,
    epochs: int = 1,
    beta: float = 0.1
):
    """Train using DPO for preference alignment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig
    from datasets import load_dataset
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load preference dataset
    dataset = load_dataset('json', data_files=preference_data)['train']
    
    # Training config
    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        beta=beta,  # KL divergence coefficient
        logging_steps=10,
        save_strategy="epoch",
    )
    
    # Train
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(f"{output_dir}/final")
    print(f"DPO model saved to {output_dir}/final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--preference-data", required=True)
    parser.add_argument("--output-dir", default="dpo_model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    
    train_dpo(
        args.base_model, args.preference_data,
        args.output_dir, args.epochs, args.beta
    )
```

### 4.2 A/B Testing Framework

**Priority**: MEDIUM  
**Estimated Improvement**: Data-driven decisions  
**Effort**: 1-2 weeks

#### 4.2.1 Implementation

**File**: `src/workflow_composer/agents/learning/ab_testing.py` (NEW)

```python
"""
A/B testing framework for comparing parser versions.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import random
import hashlib
from enum import Enum

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"

@dataclass
class Experiment:
    """An A/B test experiment."""
    id: str
    name: str
    description: str
    control_config: Dict[str, Any]
    treatment_config: Dict[str, Any]
    traffic_split: float  # 0.0-1.0 for treatment
    status: ExperimentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_samples: int = 1000

@dataclass
class ExperimentResult:
    """Results from an A/B experiment."""
    experiment_id: str
    control_samples: int
    treatment_samples: int
    control_success_rate: float
    treatment_success_rate: float
    relative_improvement: float
    statistical_significance: float
    is_significant: bool
    recommendation: str

class ABTestingFramework:
    """Framework for running A/B tests on parser versions."""
    
    def __init__(self, db_path: str = "ab_testing.db"):
        self.db_path = Path(db_path)
        self._init_db()
        self._experiments: Dict[str, Experiment] = {}
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                control_config TEXT,
                treatment_config TEXT,
                traffic_split REAL,
                status TEXT,
                start_time DATETIME,
                end_time DATETIME,
                min_samples INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                variant TEXT,
                query TEXT,
                success BOOLEAN,
                latency_ms REAL,
                confidence REAL,
                timestamp DATETIME,
                user_id_hash TEXT,
                session_id TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def create_experiment(self, experiment: Experiment) -> str:
        """Create a new A/B test experiment."""
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO experiments
            (id, name, description, control_config, treatment_config,
             traffic_split, status, min_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment.id,
            experiment.name,
            experiment.description,
            json.dumps(experiment.control_config),
            json.dumps(experiment.treatment_config),
            experiment.traffic_split,
            experiment.status.value,
            experiment.min_samples
        ))
        conn.commit()
        conn.close()
        
        self._experiments[experiment.id] = experiment
        return experiment.id
    
    def assign_variant(
        self, 
        experiment_id: str, 
        user_id: str
    ) -> str:
        """Assign user to control or treatment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return "control"
        
        # Deterministic assignment based on user_id
        hash_val = int(hashlib.md5(
            f"{experiment_id}:{user_id}".encode()
        ).hexdigest(), 16)
        
        if (hash_val % 100) / 100 < experiment.traffic_split:
            return "treatment"
        return "control"
    
    def record_event(
        self,
        experiment_id: str,
        variant: str,
        query: str,
        success: bool,
        latency_ms: float,
        confidence: float,
        user_id_hash: str,
        session_id: str
    ):
        """Record an experiment event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO experiment_events
            (experiment_id, variant, query, success, latency_ms,
             confidence, timestamp, user_id_hash, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id, variant, query, success, latency_ms,
            confidence, datetime.now().isoformat(), 
            user_id_hash, session_id
        ))
        conn.commit()
        conn.close()
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Analyze experiment results with statistical tests."""
        from scipy import stats
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get control results
        cursor.execute('''
            SELECT COUNT(*), SUM(CASE WHEN success THEN 1 ELSE 0 END)
            FROM experiment_events
            WHERE experiment_id = ? AND variant = 'control'
        ''', (experiment_id,))
        control_total, control_success = cursor.fetchone()
        
        # Get treatment results
        cursor.execute('''
            SELECT COUNT(*), SUM(CASE WHEN success THEN 1 ELSE 0 END)
            FROM experiment_events
            WHERE experiment_id = ? AND variant = 'treatment'
        ''', (experiment_id,))
        treatment_total, treatment_success = cursor.fetchone()
        
        conn.close()
        
        # Calculate rates
        control_rate = control_success / control_total if control_total else 0
        treatment_rate = treatment_success / treatment_total if treatment_total else 0
        
        # Relative improvement
        rel_improvement = (
            (treatment_rate - control_rate) / control_rate
            if control_rate else 0
        )
        
        # Chi-square test for significance
        contingency = [
            [control_success, control_total - control_success],
            [treatment_success, treatment_total - treatment_success]
        ]
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        
        is_significant = p_value < 0.05
        
        # Generate recommendation
        if is_significant and rel_improvement > 0.02:
            recommendation = "SHIP: Treatment shows significant improvement"
        elif is_significant and rel_improvement < -0.02:
            recommendation = "REVERT: Treatment shows significant regression"
        else:
            recommendation = "CONTINUE: Need more data or no significant difference"
        
        return ExperimentResult(
            experiment_id=experiment_id,
            control_samples=control_total,
            treatment_samples=treatment_total,
            control_success_rate=control_rate,
            treatment_success_rate=treatment_rate,
            relative_improvement=rel_improvement,
            statistical_significance=1 - p_value,
            is_significant=is_significant,
            recommendation=recommendation
        )
```

### 4.3 Synthetic Data Augmentation

**Priority**: MEDIUM  
**Estimated Improvement**: +2-3% on rare cases  
**Effort**: 1 week

#### 4.3.1 Implementation

**File**: `scripts/augment_training_data.py` (NEW)

```python
#!/usr/bin/env python3
"""
Generate synthetic training data through augmentation.
"""
import json
import random
from typing import List, Dict, Any
from pathlib import Path

class DataAugmenter:
    """Augment training data with synthetic variations."""
    
    # Synonym dictionaries for augmentation
    ORGANISM_SYNONYMS = {
        "human": ["homo sapiens", "h. sapiens", "hg38", "GRCh38"],
        "mouse": ["mus musculus", "m. musculus", "mm10", "mm39"],
        "drosophila": ["fruit fly", "fly", "d. melanogaster"],
    }
    
    ASSAY_SYNONYMS = {
        "rna-seq": ["RNA sequencing", "transcriptomics", "gene expression"],
        "chip-seq": ["ChIP sequencing", "chromatin immunoprecipitation"],
        "atac-seq": ["ATAC sequencing", "chromatin accessibility"],
    }
    
    INTENT_PARAPHRASES = {
        "workflow_generation": [
            "create a {} pipeline",
            "build {} workflow",
            "set up {} analysis",
            "make a {} processing pipeline",
            "generate {} workflow",
        ],
        "data_discovery": [
            "find {} datasets",
            "search for {} data",
            "look for {} experiments",
            "get {} datasets from GEO",
            "find public {} data",
        ],
    }
    
    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        with open(self.input_file, 'r') as f:
            return [json.loads(line) for line in f]
    
    def augment_synonyms(self, example: Dict) -> List[Dict]:
        """Generate variations using synonyms."""
        augmented = []
        query = example.get("input", "")
        
        # Replace organisms
        for organism, synonyms in self.ORGANISM_SYNONYMS.items():
            if organism.lower() in query.lower():
                for syn in synonyms:
                    new_query = query.lower().replace(organism, syn)
                    new_example = example.copy()
                    new_example["input"] = new_query
                    augmented.append(new_example)
        
        # Replace assays
        for assay, synonyms in self.ASSAY_SYNONYMS.items():
            if assay.lower() in query.lower():
                for syn in synonyms:
                    new_query = query.lower().replace(assay, syn)
                    new_example = example.copy()
                    new_example["input"] = new_query
                    augmented.append(new_example)
        
        return augmented
    
    def augment_paraphrases(self, example: Dict) -> List[Dict]:
        """Generate paraphrased versions."""
        augmented = []
        output = json.loads(example.get("output", "{}"))
        intent = output.get("intent", "")
        entities = output.get("entities", {})
        
        if intent in self.INTENT_PARAPHRASES:
            assay = entities.get("assay_type", "")
            organism = entities.get("organism", "")
            
            for template in self.INTENT_PARAPHRASES[intent]:
                # Fill in template with entities
                if assay:
                    new_query = template.format(assay)
                    if organism:
                        new_query = f"{new_query} for {organism}"
                    
                    new_example = example.copy()
                    new_example["input"] = new_query
                    augmented.append(new_example)
        
        return augmented
    
    def augment_noise(self, example: Dict, noise_prob: float = 0.1) -> Dict:
        """Add realistic noise (typos, case changes)."""
        query = example.get("input", "")
        
        # Random case changes
        if random.random() < noise_prob:
            query = query.lower() if random.random() < 0.5 else query.upper()
        
        # Simple typos (swap adjacent chars)
        if random.random() < noise_prob and len(query) > 2:
            pos = random.randint(0, len(query) - 2)
            query = query[:pos] + query[pos+1] + query[pos] + query[pos+2:]
        
        new_example = example.copy()
        new_example["input"] = query
        return new_example
    
    def augment_all(self, multiplier: int = 3) -> List[Dict]:
        """Apply all augmentation strategies."""
        all_augmented = []
        
        for example in self.data:
            # Original
            all_augmented.append(example)
            
            # Synonym variations
            all_augmented.extend(self.augment_synonyms(example))
            
            # Paraphrases
            all_augmented.extend(self.augment_paraphrases(example))
            
            # Noisy versions
            for _ in range(multiplier):
                all_augmented.append(self.augment_noise(example))
        
        return all_augmented
    
    def save(self, output_file: str, augmented_data: List[Dict]):
        """Save augmented data."""
        with open(output_file, 'w') as f:
            for example in augmented_data:
                f.write(json.dumps(example) + '\n')
        print(f"Saved {len(augmented_data)} examples to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--multiplier", type=int, default=3)
    args = parser.parse_args()
    
    augmenter = DataAugmenter(args.input)
    augmented = augmenter.augment_all(args.multiplier)
    augmenter.save(args.output, augmented)
```

---

## 5. Phase 4: Advanced Features

**Timeline**: Weeks 17-24  
**Goal**: Knowledge graphs, multi-modal, and federated experts

### 5.1 Bioinformatics Knowledge Graph

**Priority**: HIGH  
**Estimated Improvement**: +3-5% on entity linking  
**Effort**: 3-4 weeks

#### 5.1.1 Problem Statement
Current entity extraction lacks biological context. A knowledge graph enables entity disambiguation and relationship inference.

#### 5.1.2 Implementation

**File**: `src/workflow_composer/knowledge/graph.py` (NEW)

```python
"""
Bioinformatics knowledge graph for entity resolution and reasoning.
"""
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class EntityType(Enum):
    ORGANISM = "organism"
    GENE = "gene"
    ASSAY = "assay"
    TOOL = "tool"
    DATABASE = "database"
    FILE_FORMAT = "file_format"
    REFERENCE_GENOME = "reference_genome"

class RelationType(Enum):
    IS_A = "is_a"                    # gene IS_A biomarker
    PART_OF = "part_of"              # exon PART_OF gene
    SYNONYM_OF = "synonym_of"        # RNA-seq SYNONYM_OF transcriptomics
    USED_FOR = "used_for"            # STAR USED_FOR RNA-seq
    REQUIRES = "requires"            # ChIP-seq REQUIRES antibody
    PRODUCES = "produces"            # STAR PRODUCES BAM
    COMPATIBLE_WITH = "compatible_with"  # GRCh38 COMPATIBLE_WITH human

@dataclass
class Entity:
    """A knowledge graph entity."""
    id: str
    name: str
    entity_type: EntityType
    aliases: List[str]
    properties: Dict[str, Any]

@dataclass
class Relation:
    """A relationship between entities."""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any]

class BioinformaticsKnowledgeGraph:
    """Knowledge graph for bioinformatics domain."""
    
    def __init__(self, data_dir: str = "data/knowledge_graph"):
        self.data_dir = Path(data_dir)
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._alias_index: Dict[str, str] = {}  # alias -> entity_id
        self._type_index: Dict[EntityType, Set[str]] = {}
        self._load_or_build()
    
    def _load_or_build(self):
        """Load existing graph or build from sources."""
        graph_file = self.data_dir / "graph.json"
        if graph_file.exists():
            self._load(graph_file)
        else:
            self._build_default_graph()
            self._save(graph_file)
    
    def _build_default_graph(self):
        """Build default knowledge graph with core entities."""
        # Organisms
        self._add_organism("human", ["homo sapiens", "h. sapiens", "hg38", "GRCh38", "GRCh37", "hg19"])
        self._add_organism("mouse", ["mus musculus", "m. musculus", "mm10", "mm39", "GRCm38", "GRCm39"])
        self._add_organism("rat", ["rattus norvegicus", "r. norvegicus", "rn6", "rn7"])
        self._add_organism("drosophila", ["fruit fly", "fly", "d. melanogaster", "dm6"])
        self._add_organism("zebrafish", ["danio rerio", "d. rerio", "GRCz11"])
        self._add_organism("yeast", ["saccharomyces cerevisiae", "s. cerevisiae"])
        self._add_organism("c_elegans", ["caenorhabditis elegans", "worm", "ce11"])
        self._add_organism("arabidopsis", ["arabidopsis thaliana", "a. thaliana", "TAIR10"])
        
        # Assay types
        self._add_assay("rna-seq", ["RNA sequencing", "transcriptomics", "gene expression", "mRNA-seq"])
        self._add_assay("chip-seq", ["ChIP sequencing", "chromatin IP", "ChIPseq"])
        self._add_assay("atac-seq", ["ATAC sequencing", "chromatin accessibility", "ATACseq"])
        self._add_assay("wgs", ["whole genome sequencing", "genome sequencing", "WGS"])
        self._add_assay("wes", ["whole exome sequencing", "exome sequencing", "WES"])
        self._add_assay("cut-and-run", ["CUT&RUN", "cleavage under targets"])
        self._add_assay("cut-and-tag", ["CUT&Tag", "cleavage under targets and tagmentation"])
        self._add_assay("hi-c", ["Hi-C", "chromatin conformation", "3C", "chromosome conformation"])
        self._add_assay("bisulfite-seq", ["bisulfite sequencing", "methylation", "WGBS", "RRBS"])
        self._add_assay("scrna-seq", ["single-cell RNA-seq", "scRNA", "10x genomics", "single cell"])
        
        # Tools
        self._add_tool("star", "STAR aligner for RNA-seq", ["rna-seq"])
        self._add_tool("hisat2", "HISAT2 aligner", ["rna-seq"])
        self._add_tool("salmon", "Salmon quasi-mapping", ["rna-seq"])
        self._add_tool("kallisto", "Kallisto pseudoalignment", ["rna-seq"])
        self._add_tool("bwa", "BWA aligner", ["wgs", "wes", "chip-seq"])
        self._add_tool("bowtie2", "Bowtie2 aligner", ["chip-seq", "atac-seq"])
        self._add_tool("macs2", "MACS2 peak caller", ["chip-seq", "atac-seq"])
        self._add_tool("deseq2", "DESeq2 differential expression", ["rna-seq"])
        self._add_tool("edger", "edgeR differential expression", ["rna-seq"])
        self._add_tool("cellranger", "Cell Ranger for 10x", ["scrna-seq"])
        self._add_tool("seurat", "Seurat single-cell analysis", ["scrna-seq"])
        
        # Databases
        self._add_database("geo", ["Gene Expression Omnibus", "NCBI GEO"], "GSE")
        self._add_database("encode", ["ENCODE project"], "ENCSR")
        self._add_database("sra", ["Sequence Read Archive", "NCBI SRA"], "SRR")
        self._add_database("arrayexpress", ["ArrayExpress"], "E-MTAB")
        self._add_database("genbank", ["GenBank", "NCBI"], None)
    
    def _add_organism(self, name: str, aliases: List[str]):
        entity = Entity(
            id=f"organism:{name}",
            name=name,
            entity_type=EntityType.ORGANISM,
            aliases=aliases,
            properties={}
        )
        self._add_entity(entity)
    
    def _add_assay(self, name: str, aliases: List[str]):
        entity = Entity(
            id=f"assay:{name}",
            name=name,
            entity_type=EntityType.ASSAY,
            aliases=aliases,
            properties={}
        )
        self._add_entity(entity)
    
    def _add_tool(self, name: str, description: str, assays: List[str]):
        entity = Entity(
            id=f"tool:{name}",
            name=name,
            entity_type=EntityType.TOOL,
            aliases=[],
            properties={"description": description}
        )
        self._add_entity(entity)
        
        for assay in assays:
            self._add_relation(Relation(
                source_id=f"tool:{name}",
                target_id=f"assay:{assay}",
                relation_type=RelationType.USED_FOR,
                properties={}
            ))
    
    def _add_database(self, name: str, aliases: List[str], id_prefix: Optional[str]):
        entity = Entity(
            id=f"database:{name}",
            name=name,
            entity_type=EntityType.DATABASE,
            aliases=aliases,
            properties={"id_prefix": id_prefix}
        )
        self._add_entity(entity)
    
    def _add_entity(self, entity: Entity):
        self._entities[entity.id] = entity
        
        # Index aliases
        for alias in [entity.name] + entity.aliases:
            self._alias_index[alias.lower()] = entity.id
        
        # Type index
        if entity.entity_type not in self._type_index:
            self._type_index[entity.entity_type] = set()
        self._type_index[entity.entity_type].add(entity.id)
    
    def _add_relation(self, relation: Relation):
        self._relations.append(relation)
    
    def resolve_entity(self, text: str) -> Optional[Entity]:
        """Resolve text to a canonical entity."""
        text_lower = text.lower().strip()
        
        # Direct alias match
        if text_lower in self._alias_index:
            entity_id = self._alias_index[text_lower]
            return self._entities.get(entity_id)
        
        # Fuzzy match (simple prefix matching)
        for alias, entity_id in self._alias_index.items():
            if alias.startswith(text_lower) or text_lower.startswith(alias):
                return self._entities.get(entity_id)
        
        return None
    
    def get_related_entities(
        self, 
        entity_id: str, 
        relation_type: Optional[RelationType] = None
    ) -> List[Tuple[Entity, RelationType]]:
        """Get entities related to given entity."""
        related = []
        for rel in self._relations:
            if rel.source_id == entity_id:
                if relation_type is None or rel.relation_type == relation_type:
                    if rel.target_id in self._entities:
                        related.append((self._entities[rel.target_id], rel.relation_type))
            elif rel.target_id == entity_id:
                if relation_type is None or rel.relation_type == relation_type:
                    if rel.source_id in self._entities:
                        related.append((self._entities[rel.source_id], rel.relation_type))
        return related
    
    def get_tools_for_assay(self, assay_name: str) -> List[Entity]:
        """Get tools compatible with an assay type."""
        assay = self.resolve_entity(assay_name)
        if not assay:
            return []
        
        tools = []
        for rel in self._relations:
            if rel.target_id == assay.id and rel.relation_type == RelationType.USED_FOR:
                if rel.source_id in self._entities:
                    tools.append(self._entities[rel.source_id])
        return tools
    
    def _load(self, path: Path):
        with open(path, 'r') as f:
            data = json.load(f)
        
        for e in data.get("entities", []):
            entity = Entity(
                id=e["id"], name=e["name"],
                entity_type=EntityType(e["entity_type"]),
                aliases=e.get("aliases", []),
                properties=e.get("properties", {})
            )
            self._add_entity(entity)
        
        for r in data.get("relations", []):
            self._add_relation(Relation(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=RelationType(r["relation_type"]),
                properties=r.get("properties", {})
            ))
    
    def _save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entities": [
                {
                    "id": e.id, "name": e.name,
                    "entity_type": e.entity_type.value,
                    "aliases": e.aliases,
                    "properties": e.properties
                }
                for e in self._entities.values()
            ],
            "relations": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relation_type": r.relation_type.value,
                    "properties": r.properties
                }
                for r in self._relations
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
```

### 5.2 Coreference Resolution

**Priority**: MEDIUM  
**Estimated Improvement**: +2-3% on multi-turn  
**Effort**: 1-2 weeks

#### 5.2.1 Implementation

**File**: `src/workflow_composer/agents/intent/coreference.py` (NEW)

```python
"""
Coreference resolution for multi-turn conversations.
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

@dataclass
class EntityMention:
    """A mention of an entity in text."""
    text: str
    start: int
    end: int
    entity_type: str
    resolved_value: Optional[str] = None

@dataclass
class ConversationContext:
    """Context from previous turns."""
    entities: Dict[str, str]  # type -> most recent value
    last_intent: Optional[str] = None
    last_tool: Optional[str] = None
    turn_count: int = 0

class CoreferenceResolver:
    """Resolves pronouns and references to previous entities."""
    
    PRONOUNS = {
        "it": ["dataset", "file", "workflow", "job"],
        "that": ["dataset", "workflow", "result", "file"],
        "this": ["dataset", "workflow", "result", "file"],
        "them": ["files", "datasets", "results"],
        "those": ["files", "datasets", "results"],
        "the same": ["organism", "assay_type", "dataset"],
    }
    
    REFERENCE_PATTERNS = [
        (r"\b(the|that|this)\s+(organism|species)\b", "organism"),
        (r"\b(the|that|this)\s+(assay|experiment|analysis)\b", "assay_type"),
        (r"\b(the|that|this)\s+(dataset|data|file)\b", "dataset_id"),
        (r"\b(the|that|this)\s+(gene|genes)\b", "gene"),
        (r"\bsame\s+(organism|species)\b", "organism"),
        (r"\bsame\s+(assay|analysis)\b", "assay_type"),
    ]
    
    def __init__(self):
        self._context = ConversationContext(entities={})
    
    def update_context(
        self, 
        entities: Dict[str, str], 
        intent: str, 
        tool: str
    ):
        """Update context with new turn information."""
        self._context.entities.update(entities)
        self._context.last_intent = intent
        self._context.last_tool = tool
        self._context.turn_count += 1
    
    def resolve(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Resolve coreferences in query.
        Returns: (resolved_query, extracted_entities)
        """
        resolved_query = query
        resolved_entities = {}
        
        # Find and resolve pronoun references
        for pronoun, entity_types in self.PRONOUNS.items():
            pattern = rf"\b{pronoun}\b"
            if re.search(pattern, query.lower()):
                # Find most likely referent
                for entity_type in entity_types:
                    if entity_type in self._context.entities:
                        value = self._context.entities[entity_type]
                        resolved_entities[entity_type] = value
                        # Optionally expand the pronoun in query
                        break
        
        # Find and resolve explicit references
        for pattern, entity_type in self.REFERENCE_PATTERNS:
            match = re.search(pattern, query.lower())
            if match:
                if entity_type in self._context.entities:
                    value = self._context.entities[entity_type]
                    resolved_entities[entity_type] = value
        
        return resolved_query, resolved_entities
    
    def get_context(self) -> ConversationContext:
        """Get current conversation context."""
        return self._context
    
    def reset_context(self):
        """Reset context for new conversation."""
        self._context = ConversationContext(entities={})
```

### 5.3 Multi-Modal Input Processing

**Priority**: LOW  
**Estimated Improvement**: New capabilities  
**Effort**: 2-3 weeks

#### 5.3.1 Implementation

**File**: `src/workflow_composer/agents/multimodal/__init__.py` (NEW)

```python
"""
Multi-modal input processing for file uploads and images.
"""
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import mimetypes

class InputType(Enum):
    TEXT = "text"
    FILE = "file"
    IMAGE = "image"
    URL = "url"

@dataclass
class MultiModalInput:
    """A multi-modal input from the user."""
    input_type: InputType
    content: Any  # Text, file path, image bytes, URL
    metadata: Dict[str, Any]

@dataclass
class ProcessedInput:
    """Processed multi-modal input ready for parsing."""
    text_representation: str
    extracted_entities: Dict[str, str]
    file_info: Optional[Dict[str, Any]] = None

class MultiModalProcessor:
    """Process various input modalities into text for parsing."""
    
    SUPPORTED_EXTENSIONS = {
        # Bioinformatics files
        ".fastq": "sequencing_reads",
        ".fq": "sequencing_reads",
        ".fastq.gz": "sequencing_reads",
        ".fasta": "sequences",
        ".fa": "sequences",
        ".bam": "alignment",
        ".sam": "alignment",
        ".vcf": "variants",
        ".bed": "regions",
        ".gtf": "annotation",
        ".gff": "annotation",
        # Workflow files
        ".nf": "nextflow_workflow",
        ".smk": "snakemake_workflow",
        ".wdl": "wdl_workflow",
        ".cwl": "cwl_workflow",
        # Data files
        ".csv": "tabular_data",
        ".tsv": "tabular_data",
        ".xlsx": "tabular_data",
        ".json": "structured_data",
        ".yaml": "configuration",
        ".yml": "configuration",
    }
    
    def process(self, input_data: MultiModalInput) -> ProcessedInput:
        """Process input into text representation."""
        if input_data.input_type == InputType.TEXT:
            return ProcessedInput(
                text_representation=input_data.content,
                extracted_entities={}
            )
        
        if input_data.input_type == InputType.FILE:
            return self._process_file(input_data)
        
        if input_data.input_type == InputType.URL:
            return self._process_url(input_data)
        
        if input_data.input_type == InputType.IMAGE:
            return self._process_image(input_data)
        
        raise ValueError(f"Unknown input type: {input_data.input_type}")
    
    def _process_file(self, input_data: MultiModalInput) -> ProcessedInput:
        """Process file upload."""
        file_path = Path(input_data.content)
        extension = "".join(file_path.suffixes).lower()
        
        file_type = self.SUPPORTED_EXTENSIONS.get(extension, "unknown")
        
        # Extract metadata based on file type
        entities = {}
        if file_type == "sequencing_reads":
            entities["file_type"] = "FASTQ"
            # Could parse header to get more info
        elif file_type == "alignment":
            entities["file_type"] = "BAM" if ".bam" in extension else "SAM"
        elif file_type == "nextflow_workflow":
            entities["workflow_engine"] = "nextflow"
        elif file_type == "snakemake_workflow":
            entities["workflow_engine"] = "snakemake"
        
        text_repr = (
            f"User uploaded file: {file_path.name} "
            f"(type: {file_type}, size: {file_path.stat().st_size if file_path.exists() else 'unknown'})"
        )
        
        return ProcessedInput(
            text_representation=text_repr,
            extracted_entities=entities,
            file_info={
                "path": str(file_path),
                "name": file_path.name,
                "type": file_type,
                "extension": extension
            }
        )
    
    def _process_url(self, input_data: MultiModalInput) -> ProcessedInput:
        """Process URL input (e.g., GEO link)."""
        url = input_data.content
        entities = {}
        
        # Extract dataset IDs from URLs
        import re
        patterns = [
            (r"GSE\d+", "dataset_id"),
            (r"ENCSR[A-Z0-9]+", "dataset_id"),
            (r"SRR\d+", "dataset_id"),
        ]
        
        for pattern, entity_type in patterns:
            match = re.search(pattern, url)
            if match:
                entities[entity_type] = match.group()
                break
        
        text_repr = f"User shared URL: {url}"
        
        return ProcessedInput(
            text_representation=text_repr,
            extracted_entities=entities
        )
    
    def _process_image(self, input_data: MultiModalInput) -> ProcessedInput:
        """Process image input (e.g., screenshot of error)."""
        # Would use OCR or vision model
        text_repr = "User uploaded an image"
        
        return ProcessedInput(
            text_representation=text_repr,
            extracted_entities={}
        )
```

---

## 6. Implementation Timeline

### 6.1 Gantt Chart Overview

```
Week:        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
             |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|

PHASE 1: FOUNDATION (Weeks 1-4)
Clarification  ████████
Prod. Queries     ██████
Active Learn.        ████████
Negation         ██████

PHASE 2: MULTI-MODEL (Weeks 5-10)
Fine-Tuning         ████████████████
LLM-as-Judge              ██████████
Agentic Parser                  ████████████
Smart Router                        ██████████

PHASE 3: CONTINUOUS LEARNING (Weeks 11-16)
RLHF                                     ████████████████
A/B Testing                                  ████████████
Data Augment.                                      ████████

PHASE 4: ADVANCED (Weeks 17-24)
Knowledge Graph                                         ████████████████
Coreference                                                   ████████
Multi-Modal                                                         ████████████
```

### 6.2 Milestone Definitions

| Milestone | Week | Deliverables | Success Criteria |
|-----------|------|--------------|------------------|
| M1: Clarification MVP | 2 | Working clarification system | Handles 80%+ low-confidence cases |
| M2: Production Integration | 4 | Query collection pipeline | 100+ real queries/day |
| M3: Fine-Tuned Model v1 | 8 | Domain-specific LLM | +3% accuracy over baseline |
| M4: RLHF Pipeline | 14 | Feedback → training loop | Automated weekly retraining |
| M5: Knowledge Graph v1 | 20 | Core entity resolution | 95%+ entity disambiguation |
| M6: Production Release | 24 | Full system deployment | 98%+ accuracy target |

### 6.3 Resource Requirements

| Resource | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|----------|---------|---------|---------|---------|
| Engineering FTE | 1 | 2 | 1.5 | 2 |
| GPU Hours/week | 0 | 100 | 50 | 20 |
| Cloud API Budget | $100/mo | $500/mo | $300/mo | $200/mo |
| Storage (GB) | 10 | 100 | 50 | 200 |

---

## 7. Technical Specifications

### 7.1 System Requirements

```yaml
# Minimum Requirements
python: ">=3.10"
memory: "16GB RAM"
storage: "100GB SSD"
gpu: "NVIDIA GPU with 16GB VRAM (for fine-tuning)"

# Recommended for Production
python: ">=3.11"
memory: "64GB RAM"
storage: "500GB NVMe SSD"
gpu: "NVIDIA A100 40GB or H100"
```

### 7.2 Dependency Matrix

```yaml
# Core Dependencies
torch: ">=2.0"
transformers: ">=4.35"
sentence-transformers: ">=2.2"
faiss-cpu: ">=1.7"  # or faiss-gpu
spacy: ">=3.5"
pydantic: ">=2.0"

# Fine-Tuning
unsloth: ">=2024.1"  # For efficient fine-tuning
trl: ">=0.7"  # DPO and RLHF
peft: ">=0.6"  # LoRA adapters
bitsandbytes: ">=0.41"  # Quantization

# Evaluation
scipy: ">=1.10"
scikit-learn: ">=1.3"
nltk: ">=3.8"

# LLM Providers
openai: ">=1.0"
anthropic: ">=0.5"
ollama: ">=0.1"
```

### 7.3 API Contracts

#### 7.3.1 Parser Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ParseResult:
    intent: str
    entities: Dict[str, str]
    tool: str
    confidence: float
    metadata: Dict[str, Any]

class ParserInterface(ABC):
    """Standard interface for all parsers."""
    
    @abstractmethod
    def parse(self, query: str) -> ParseResult:
        """Parse a query and return structured result."""
        pass
    
    @abstractmethod
    def parse_batch(self, queries: List[str]) -> List[ParseResult]:
        """Parse multiple queries efficiently."""
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold for this parser."""
        pass
```

#### 7.3.2 Learning Interface

```python
@dataclass
class TrainingExample:
    query: str
    expected_intent: str
    expected_entities: Dict[str, str]
    expected_tool: str
    difficulty: str
    source: str

class LearningInterface(ABC):
    """Standard interface for learning systems."""
    
    @abstractmethod
    def record_feedback(self, feedback: Feedback) -> None:
        """Record user feedback for learning."""
        pass
    
    @abstractmethod
    def get_training_batch(self, size: int) -> List[TrainingExample]:
        """Get a batch of training examples."""
        pass
    
    @abstractmethod
    def update_model(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """Update model with new examples, return metrics."""
        pass
```

### 7.4 Configuration Schema

```yaml
# config/agent.yaml
agent:
  name: "BioPipelines Chat Agent"
  version: "2.0"
  
  parser:
    ensemble:
      enabled: true
      weights:
        rule: 0.25
        semantic: 0.30
        ner: 0.20
        llm: 0.15
        rag: 0.10
    
    confidence:
      high_threshold: 0.85
      medium_threshold: 0.60
      low_threshold: 0.40
      clarification_enabled: true
  
  models:
    local:
      primary: "llama-3-8b"
      fallback: "mistral-7b"
      fine_tuned: "biopipelines-parser-v1"
    
    cloud:
      openai:
        model: "gpt-4o-mini"
        max_tokens: 1000
      anthropic:
        model: "claude-3-5-sonnet"
        max_tokens: 1000
    
    routing:
      strategy: "local_first"  # local_first, cloud_first, ensemble
      prefer_local: true
      fallback_to_cloud: true
  
  learning:
    active_learning:
      enabled: true
      batch_size: 100
      priority_weights:
        failure: 0.40
        hard: 0.30
        moderate: 0.20
        easy: 0.10
    
    feedback:
      collection_enabled: true
      min_feedback_for_retraining: 100
    
    ab_testing:
      enabled: true
      default_traffic_split: 0.10
  
  knowledge:
    graph:
      enabled: true
      data_dir: "data/knowledge_graph"
      auto_update: true
    
    coreference:
      enabled: true
      max_context_turns: 5

  logging:
    level: "INFO"
    structured: true
    include_query: true
    include_result: true
```

---

## 8. Success Metrics

### 8.1 Primary KPIs

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Pass Rate | 91.4% | 98%+ | Weekly evaluation runs |
| Intent Accuracy | 95.5% | 99%+ | Automated + LLM-as-Judge |
| Entity F1 | 93.2% | 97%+ | Automated with synonyms |
| Tool Accuracy | 96.7% | 99%+ | Automated mapping |
| Latency (p50) | 10.2ms | <50ms | Production monitoring |
| Latency (p99) | - | <500ms | Production monitoring |
| User Satisfaction | - | 4.5+/5 | In-app feedback |

### 8.2 Category-Specific Targets

| Category | Current | Target | Priority |
|----------|---------|--------|----------|
| data_discovery | 93.4% | 98% | High |
| workflow_generation | 87.3% | 95% | High |
| job_management | 85.2% | 95% | Medium |
| education | 98.8% | 99% | Low |
| multi_turn | 95.7% | 98% | High |
| edge_cases | 94.1% | 95% | Medium |
| adversarial | 70.0% | 85% | Low |
| ambiguous | 0.0% | N/A | Design decision |

### 8.3 Operational Metrics

```yaml
availability:
  target: 99.9%
  measurement: uptime monitoring

cost_per_query:
  local: $0.00
  cloud_fallback: <$0.01
  average: <$0.001

error_rate:
  parsing_errors: <0.1%
  tool_execution_errors: <1%
  timeout_rate: <0.1%

throughput:
  queries_per_second: 100+
  concurrent_users: 50+
```

### 8.4 Learning Metrics

| Metric | Target | Frequency |
|--------|--------|-----------|
| Feedback collection rate | 10%+ of queries | Daily |
| Positive feedback ratio | 80%+ | Weekly |
| Retraining improvement | +0.5% per cycle | Monthly |
| A/B test significance | 95% confidence | Per experiment |
| Active learning efficiency | 2x vs random | Monthly |

### 8.5 Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│                   Chat Agent Health Dashboard                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pass Rate          Intent Accuracy      Entity F1             │
│  ▓▓▓▓▓▓▓▓▓░ 91.4%   ▓▓▓▓▓▓▓▓▓▓ 95.5%   ▓▓▓▓▓▓▓▓▓░ 93.2%      │
│                                                                 │
│  Queries/Hour       Avg Latency          Error Rate            │
│  ████████░░ 850     ████░░░░░░ 10.2ms   ░░░░░░░░░░ 0.2%       │
│                                                                 │
│  Model Usage        Cost Today           Feedback/Day          │
│  Local: 95%         $0.42                ▓▓▓░░ 127             │
│  Cloud: 5%                                                      │
│                                                                 │
│  Recent Issues:                                                 │
│  • workflow_generation accuracy below target (87.3%)           │
│  • Increase in negation-related failures                        │
│                                                                 │
│  Active Experiments:                                            │
│  • clarification_v2: +2.1% (p=0.03) [Ship Ready]               │
│  • fine_tuned_llama: +1.8% (p=0.12) [Need More Data]           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Quick Reference

### A.1 File Locations

| Component | Path |
|-----------|------|
| Clarification | `src/workflow_composer/agents/clarification/` |
| Active Learning | `src/workflow_composer/agents/learning/` |
| Negation Handler | `src/workflow_composer/agents/intent/negation_handler.py` |
| Agentic Parser | `src/workflow_composer/agents/intent/agentic_parser.py` |
| Smart Router | `src/workflow_composer/llm/smart_router.py` |
| Knowledge Graph | `src/workflow_composer/knowledge/graph.py` |
| Coreference | `src/workflow_composer/agents/intent/coreference.py` |
| Multi-Modal | `src/workflow_composer/agents/multimodal/` |
| Evaluation | `tests/evaluation/` |
| Fine-Tuning Scripts | `scripts/` |

### A.2 Command Reference

```bash
# Run evaluation
python -m tests.evaluation.run_experiment --version v12

# Prepare fine-tuning data
python scripts/prepare_finetuning_data.py

# Fine-tune model
python scripts/finetune_model.py --base-model llama-3-8b --train-data train.jsonl

# Train with DPO
python scripts/train_dpo.py --base-model model_path --preference-data prefs.jsonl

# Augment data
python scripts/augment_training_data.py --input train.jsonl --output augmented.jsonl

# Review production queries
python scripts/review_production_queries.py --limit 20
```

### A.3 Integration Checklist

- [ ] Install dependencies from `requirements.txt`
- [ ] Configure `config/agent.yaml`
- [ ] Initialize knowledge graph
- [ ] Set up evaluation database
- [ ] Configure LLM providers (API keys)
- [ ] Enable logging and monitoring
- [ ] Run baseline evaluation
- [ ] Deploy to staging environment
- [ ] Run A/B test
- [ ] Graduate to production

---

*Document Version: 1.0*  
*Last Updated: November 2025*  
*Authors: BioPipelines Team*
