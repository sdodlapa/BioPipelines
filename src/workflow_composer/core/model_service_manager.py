"""
Model Orchestrator for BioPipelines Ensemble
=============================================

Automatically manages model availability:
- BioMistral-7B: GPU service (SLURM job) or CPU fallback
- BiomedBERT: On-demand CPU loading
- SciBERT: On-demand CPU loading

Strategy:
1. BERT models always available (load on first query, ~10s startup)
2. BioMistral checks if GPU server is running
3. If no GPU available, uses CPU-only ensemble or fallback to rule-based

This avoids blocking users waiting for GPU allocation.
"""

import os
import subprocess
import time
import logging
import requests
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Status of a model service."""
    UNAVAILABLE = "unavailable"
    STARTING = "starting"
    READY = "ready"
    ERROR = "error"


@dataclass
class ModelStatus:
    """Status information for a model."""
    name: str
    status: ServiceStatus
    endpoint: Optional[str] = None
    load_time_ms: Optional[float] = None
    error: Optional[str] = None
    slurm_job_id: Optional[str] = None


@dataclass  
class OrchestratorConfig:
    """Configuration for the model orchestrator."""
    # Paths
    project_dir: str = field(default_factory=lambda: os.environ.get(
        'BIOPIPELINES_HOME', 
        '/home/sdodl001_odu_edu/BioPipelines'
    ))
    
    # BioMistral settings
    biomistral_model: str = "BioMistral/BioMistral-7B"
    biomistral_port: int = 8000
    biomistral_timeout: int = 300  # 5 min max wait for GPU job
    
    # SLURM settings (T4 GPU with 16GB VRAM)
    slurm_partition: str = "t4flex"
    slurm_script: str = "scripts/llm/serve_biomistral_t4.sbatch"
    
    # Fallback behavior
    auto_start_gpu: bool = False  # Don't auto-start GPU jobs (long wait)
    cpu_fallback: bool = True     # Use CPU if no GPU
    use_rules_only: bool = False  # Pure rule-based (fastest)


class ModelOrchestrator:
    """
    Orchestrates model loading and service management.
    
    Design Philosophy:
    - BERT models: Load lazily on CPU (always available, ~10s first load)
    - BioMistral: Check for running GPU service, don't block waiting
    - Graceful degradation: GPU → CPU → Rules
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._lock = threading.Lock()
        
        # Model instances (lazy loaded)
        self._biomedbert = None
        self._scibert = None
        self._biomistral_url = None
        
        # Status tracking
        self._status: Dict[str, ModelStatus] = {
            'biomedbert': ModelStatus('BiomedBERT', ServiceStatus.UNAVAILABLE),
            'scibert': ModelStatus('SciBERT', ServiceStatus.UNAVAILABLE),
            'biomistral': ModelStatus('BioMistral-7B', ServiceStatus.UNAVAILABLE),
        }
        
        # Connection file for BioMistral
        self._connection_file = Path(self.config.project_dir) / '.biomistral_server'
        
    def get_status(self) -> Dict[str, ModelStatus]:
        """Get current status of all models."""
        # Check BioMistral
        self._check_biomistral_status()
        return self._status.copy()
    
    def _check_biomistral_status(self) -> bool:
        """Check if BioMistral GPU server is running."""
        # Check connection file
        if self._connection_file.exists():
            try:
                info = {}
                for line in self._connection_file.read_text().strip().split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        info[key] = value
                
                url = info.get('BIOMISTRAL_URL')
                job_id = info.get('SLURM_JOB_ID')
                
                if url:
                    # Test connection
                    try:
                        resp = requests.get(f"{url}/models", timeout=5)
                        if resp.status_code == 200:
                            self._biomistral_url = url
                            self._status['biomistral'] = ModelStatus(
                                'BioMistral-7B',
                                ServiceStatus.READY,
                                endpoint=url,
                                slurm_job_id=job_id
                            )
                            return True
                    except requests.exceptions.RequestException:
                        # Server exists but not responding yet
                        self._status['biomistral'] = ModelStatus(
                            'BioMistral-7B',
                            ServiceStatus.STARTING,
                            slurm_job_id=job_id
                        )
                        return False
            except Exception as e:
                logger.debug(f"Error reading connection file: {e}")
        
        # Check SLURM queue for running jobs
        try:
            result = subprocess.run(
                ['squeue', '-u', os.environ.get('USER', ''), '-n', 'biomistral_t4,biomistral_h100,biomistral', '-h', '-o', '%i %T'],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                parts = result.stdout.strip().split()
                job_id = parts[0]
                state = parts[1] if len(parts) > 1 else 'UNKNOWN'
                
                if state in ('RUNNING', 'COMPLETING'):
                    self._status['biomistral'] = ModelStatus(
                        'BioMistral-7B',
                        ServiceStatus.STARTING,
                        slurm_job_id=job_id
                    )
                elif state in ('PENDING', 'CONFIGURING'):
                    self._status['biomistral'] = ModelStatus(
                        'BioMistral-7B',
                        ServiceStatus.STARTING,
                        slurm_job_id=job_id,
                        error=f"Waiting in queue ({state})"
                    )
                return False
        except Exception as e:
            logger.debug(f"Error checking SLURM queue: {e}")
        
        self._status['biomistral'] = ModelStatus(
            'BioMistral-7B',
            ServiceStatus.UNAVAILABLE
        )
        self._biomistral_url = None
        return False
    
    def start_biomistral_gpu(self) -> Tuple[bool, str]:
        """
        Start BioMistral GPU server via SLURM.
        
        Returns:
            (success, message) tuple
        """
        # Check if already running
        if self._check_biomistral_status():
            return True, f"BioMistral already running at {self._biomistral_url}"
        
        if self._status['biomistral'].status == ServiceStatus.STARTING:
            return True, f"BioMistral starting (Job: {self._status['biomistral'].slurm_job_id})"
        
        # Submit SLURM job
        script_path = Path(self.config.project_dir) / self.config.slurm_script
        if not script_path.exists():
            return False, f"SLURM script not found: {script_path}"
        
        try:
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                # Parse job ID from "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                self._status['biomistral'] = ModelStatus(
                    'BioMistral-7B',
                    ServiceStatus.STARTING,
                    slurm_job_id=job_id
                )
                return True, f"BioMistral job submitted (Job ID: {job_id}). GPU startup may take 2-5 minutes."
            else:
                return False, f"Failed to submit job: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Timeout submitting SLURM job"
        except Exception as e:
            return False, f"Error starting BioMistral: {e}"
    
    def stop_biomistral_gpu(self) -> Tuple[bool, str]:
        """Stop BioMistral GPU server."""
        job_id = self._status['biomistral'].slurm_job_id
        
        if job_id:
            try:
                subprocess.run(['scancel', job_id], timeout=10)
                if self._connection_file.exists():
                    self._connection_file.unlink()
                self._status['biomistral'] = ModelStatus(
                    'BioMistral-7B',
                    ServiceStatus.UNAVAILABLE
                )
                self._biomistral_url = None
                return True, f"Cancelled job {job_id}"
            except Exception as e:
                return False, f"Error stopping: {e}"
        
        return False, "No running job found"
    
    def load_biomedbert(self) -> bool:
        """Load BiomedBERT model (CPU)."""
        if self._biomedbert is not None:
            return True
        
        with self._lock:
            if self._biomedbert is not None:
                return True
            
            start_time = time.time()
            try:
                from transformers import (
                    AutoTokenizer, 
                    AutoModelForTokenClassification,
                    pipeline
                )
                
                logger.info("Loading BiomedBERT for NER (CPU)...")
                
                model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
                
                # Load for token classification / NER
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # For now, use a general NER approach
                # In production, fine-tune on bio entities
                self._biomedbert = {
                    'tokenizer': tokenizer,
                    'model_name': model_name,
                    'type': 'embeddings'  # Use for embeddings/similarity
                }
                
                load_time = (time.time() - start_time) * 1000
                self._status['biomedbert'] = ModelStatus(
                    'BiomedBERT',
                    ServiceStatus.READY,
                    load_time_ms=load_time
                )
                logger.info(f"BiomedBERT loaded in {load_time:.0f}ms")
                return True
                
            except Exception as e:
                self._status['biomedbert'] = ModelStatus(
                    'BiomedBERT',
                    ServiceStatus.ERROR,
                    error=str(e)
                )
                logger.error(f"Failed to load BiomedBERT: {e}")
                return False
    
    def load_scibert(self) -> bool:
        """Load SciBERT model (CPU)."""
        if self._scibert is not None:
            return True
        
        with self._lock:
            if self._scibert is not None:
                return True
            
            start_time = time.time()
            try:
                from transformers import AutoTokenizer, AutoModel
                
                logger.info("Loading SciBERT (CPU)...")
                
                model_name = "allenai/scibert_scivocab_uncased"
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                model.eval()
                
                self._scibert = {
                    'tokenizer': tokenizer,
                    'model': model,
                    'model_name': model_name
                }
                
                load_time = (time.time() - start_time) * 1000
                self._status['scibert'] = ModelStatus(
                    'SciBERT',
                    ServiceStatus.READY,
                    load_time_ms=load_time
                )
                logger.info(f"SciBERT loaded in {load_time:.0f}ms")
                return True
                
            except Exception as e:
                self._status['scibert'] = ModelStatus(
                    'SciBERT',
                    ServiceStatus.ERROR,
                    error=str(e)
                )
                logger.error(f"Failed to load SciBERT: {e}")
                return False
    
    def get_available_strategy(self) -> str:
        """
        Determine best available parsing strategy.
        
        Returns one of:
        - 'full_ensemble': All 3 models available
        - 'bert_only': Only BERT models (CPU)
        - 'biomistral_only': Only BioMistral (GPU)
        - 'rules_only': Fall back to rule-based parsing
        """
        self._check_biomistral_status()
        
        biomistral_ready = self._status['biomistral'].status == ServiceStatus.READY
        bert_available = True  # BERT models can always be loaded on CPU
        
        if biomistral_ready and bert_available:
            return 'full_ensemble'
        elif biomistral_ready:
            return 'biomistral_only'
        elif bert_available:
            return 'bert_only'
        else:
            return 'rules_only'
    
    def get_biomistral_client(self):
        """Get OpenAI-compatible client for BioMistral."""
        if not self._biomistral_url:
            self._check_biomistral_status()
        
        if self._biomistral_url:
            try:
                from openai import OpenAI
                return OpenAI(
                    base_url=self._biomistral_url,
                    api_key="not-needed"
                )
            except ImportError:
                logger.warning("OpenAI client not installed")
        
        return None
    
    def get_biomedbert(self):
        """Get loaded BiomedBERT model."""
        if self._biomedbert is None:
            self.load_biomedbert()
        return self._biomedbert
    
    def get_scibert(self):
        """Get loaded SciBERT model."""
        if self._scibert is None:
            self.load_scibert()
        return self._scibert
    
    def preload_cpu_models(self):
        """Preload BERT models in background thread."""
        def _load():
            self.load_biomedbert()
            self.load_scibert()
        
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()
        return thread
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get human-readable status summary."""
        self.get_status()
        
        strategy = self.get_available_strategy()
        
        return {
            'strategy': strategy,
            'strategy_description': {
                'full_ensemble': '🚀 Full Ensemble (GPU + CPU)',
                'bert_only': '🔬 BERT Models Only (CPU)',
                'biomistral_only': '🧬 BioMistral Only (GPU)',
                'rules_only': '📋 Rule-Based Parsing'
            }.get(strategy, strategy),
            'models': {
                name: {
                    'status': status.status.value,
                    'endpoint': status.endpoint,
                    'load_time_ms': status.load_time_ms,
                    'error': status.error,
                    'job_id': status.slurm_job_id
                }
                for name, status in self._status.items()
            },
            'gpu_available': self._status['biomistral'].status == ServiceStatus.READY,
            'cpu_models_loaded': (
                self._status['biomedbert'].status == ServiceStatus.READY and
                self._status['scibert'].status == ServiceStatus.READY
            )
        }


# Global instance
_orchestrator: Optional[ModelOrchestrator] = None


def get_orchestrator() -> ModelOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ModelOrchestrator()
    return _orchestrator


# ============================================================================
# Integration with Intent Parser
# ============================================================================

class AdaptiveIntentParser:
    """
    DEPRECATED: Use `workflow_composer.agents.intent.UnifiedIntentParser` instead.
    
    Intent parser that adapts to available models.
    
    Automatically uses the best available strategy:
    1. Full ensemble if GPU server running
    2. BERT-enhanced rules if CPU only
    3. Pure rules as fallback
    
    .. deprecated:: 2.1.0
        Use :class:`workflow_composer.agents.intent.UnifiedIntentParser` instead.
        This class will be removed in version 3.0.0.
    """
    
    def __init__(self):
        import warnings
        warnings.warn(
            "AdaptiveIntentParser from core.model_service_manager is deprecated. "
            "Use workflow_composer.agents.intent.UnifiedIntentParser instead. "
            "This class will be removed in version 3.0.0.",
            DeprecationWarning,
            stacklevel=2
        )
        self.orchestrator = get_orchestrator()
        self._rule_parser = None
        self._ensemble_parser = None
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse user query using best available strategy.
        
        Returns:
            Dict with analysis_type, confidence, organism, tools, etc.
        """
        strategy = self.orchestrator.get_available_strategy()
        
        if strategy == 'full_ensemble':
            return self._parse_with_ensemble(query)
        elif strategy in ('bert_only', 'biomistral_only'):
            return self._parse_with_partial(query, strategy)
        else:
            return self._parse_with_rules(query)
    
    def _parse_with_rules(self, query: str) -> Dict[str, Any]:
        """Parse using rule-based parser."""
        if self._rule_parser is None:
            from .query_parser import IntentParser
            self._rule_parser = IntentParser()
        
        result = self._rule_parser.parse(query)
        result['parsing_strategy'] = 'rules_only'
        return result
    
    def _parse_with_ensemble(self, query: str) -> Dict[str, Any]:
        """Parse using full ensemble."""
        if self._ensemble_parser is None:
            from .query_parser_ensemble import EnsembleIntentParser
            self._ensemble_parser = EnsembleIntentParser()
        
        result = self._ensemble_parser.parse(query)
        return {
            'analysis_type': result.analysis_type,
            'confidence': result.confidence,
            'organism': result.organism,
            'tools': result.tools_detected,
            'entities': result.entities,
            'reasoning': result.reasoning,
            'parsing_strategy': 'full_ensemble',
            'latency_ms': result.latency_ms
        }
    
    def _parse_with_partial(self, query: str, strategy: str) -> Dict[str, Any]:
        """Parse with partial model availability."""
        # Start with rules
        result = self._parse_with_rules(query)
        result['parsing_strategy'] = strategy
        
        # Enhance with available models
        if strategy == 'bert_only':
            # Use BERT models to improve entity extraction
            entities = self._extract_entities_bert(query)
            if entities:
                result['entities'] = entities
                # Boost confidence if BERT confirms
                if self._bert_confirms_type(query, result.get('analysis_type', '')):
                    result['confidence'] = min(0.95, result.get('confidence', 0.5) + 0.15)
        
        return result
    
    def _extract_entities_bert(self, query: str) -> Dict[str, list]:
        """Extract entities using BERT models."""
        entities = {'organisms': [], 'tools': [], 'data_types': []}
        
        # Use SciBERT for scientific terms
        scibert = self.orchestrator.get_scibert()
        if scibert:
            # Simple keyword matching enhanced by embeddings
            # (Full implementation would use proper NER)
            bio_terms = {
                'organisms': ['human', 'mouse', 'rat', 'drosophila', 'zebrafish', 'yeast', 'e. coli', 'arabidopsis'],
                'tools': ['star', 'salmon', 'hisat2', 'bwa', 'bowtie2', 'minimap2', 'cellranger', 'seurat'],
                'data_types': ['fastq', 'bam', 'vcf', 'h5ad', 'anndata', '10x', 'illumina', 'nanopore']
            }
            
            query_lower = query.lower()
            for entity_type, terms in bio_terms.items():
                for term in terms:
                    if term in query_lower:
                        entities[entity_type].append(term)
        
        return entities
    
    def _bert_confirms_type(self, query: str, analysis_type: str) -> bool:
        """Check if BERT embeddings confirm the analysis type."""
        # Simplified - in production would use cosine similarity
        type_keywords = {
            'rna_seq': ['rna', 'expression', 'transcriptome', 'differential'],
            'chip_seq': ['chip', 'peak', 'histone', 'binding', 'chromatin'],
            'long_read': ['nanopore', 'pacbio', 'long-read', 'assembly'],
            'single_cell': ['single-cell', 'scrna', '10x', 'cell', 'cluster'],
        }
        
        keywords = type_keywords.get(analysis_type, [])
        query_lower = query.lower()
        
        return any(kw in query_lower for kw in keywords)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return self.orchestrator.get_status_summary()
