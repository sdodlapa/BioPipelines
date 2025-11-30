"""
Conversation Generator
======================

Generates thousands of synthetic test conversations for comprehensive
evaluation of the bioinformatics chat agent.

Strategies:
1. Template-based: Fill slots with domain vocabulary
2. Paraphrase variations: Same meaning, different wording
3. Entity substitution: Swap organisms, tissues, assays
4. Complexity gradients: Simple to complex queries
5. Edge cases: Typos, caps, special chars, long queries
6. Multi-turn chains: Realistic conversation flows
7. Adversarial: Queries designed to confuse the parser

Categories:
- data_discovery: Search, download, describe, scan
- workflow_generation: Create pipelines
- job_management: Submit, status, logs, cancel
- education: Explain concepts, help
- multi_turn: Context-dependent queries
- coreference: Pronoun resolution
- ambiguous: Vague or mixed intent
- edge_cases: Unusual inputs
- error_handling: Invalid inputs
- adversarial: Intentionally tricky
"""

import random
import itertools
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
import json
import re

try:
    from .database import get_database
except ImportError:
    from database import get_database


# =============================================================================
# Domain Vocabulary
# =============================================================================

ORGANISMS = [
    ("human", "Homo sapiens"),
    ("mouse", "Mus musculus"),
    ("rat", "Rattus norvegicus"),
    ("zebrafish", "Danio rerio"),
    ("fly", "Drosophila"),
    ("worm", "C. elegans"),
    ("yeast", "S. cerevisiae"),
    ("arabidopsis", "Arabidopsis"),
]

TISSUES = [
    "brain", "liver", "kidney", "heart", "lung", "spleen",
    "cortex", "hippocampus", "cerebellum",
    "blood", "bone marrow", "PBMC",
    "skin", "muscle", "intestine", "colon",
]

CELL_TYPES = [
    "HeLa", "HEK293", "K562", "GM12878", "IMR90", "A549", "MCF7",
    "T cell", "B cell", "macrophage", "neutrophil", "monocyte",
    "stem cell", "neuron", "astrocyte", "fibroblast",
]

ASSAY_TYPES = [
    ("RNA-seq", "rnaseq", "transcriptome", "gene expression"),
    ("ChIP-seq", "chipseq", "chromatin immunoprecipitation"),
    ("ATAC-seq", "atacseq", "chromatin accessibility"),
    ("Hi-C", "hic", "chromosome conformation"),
    ("methylation", "WGBS", "bisulfite", "DNA methylation"),
    ("scRNA-seq", "single-cell RNA", "10x genomics"),
    ("WGS", "whole genome", "genome sequencing"),
    ("WES", "exome", "whole exome"),
    ("CLIP-seq", "RIP-seq", "RNA binding"),
    ("metagenomics", "16S", "microbiome"),
]

DISEASES = [
    "cancer", "breast cancer", "lung cancer", "leukemia",
    "glioblastoma", "melanoma", "colorectal cancer",
    "Alzheimer's", "Parkinson's", "diabetes",
    "autoimmune", "COVID-19",
]

DATABASES = ["ENCODE", "GEO", "TCGA", "SRA", "GDC", "ArrayExpress"]

HISTONE_MARKS = [
    "H3K4me3", "H3K27ac", "H3K27me3", "H3K36me3",
    "H3K4me1", "H3K9me3", "H3K9ac", "H4K20me1",
]

PATHS = [
    "/data/raw", "/data/samples", "/projects/rnaseq",
    "/home/user/data", "~/experiments", "/scratch/fastq",
    "/mnt/storage/sequencing", "/data/references",
]

FILE_FORMATS = ["FASTQ", "BAM", "VCF", "BED", "BigWig", "counts"]


# =============================================================================
# Query Templates
# =============================================================================

DATA_SEARCH_TEMPLATES = [
    "Search for {organism} {tissue} {assay} data",
    "Find {assay} datasets from {organism} {tissue}",
    "Look for {organism} {assay} in {database}",
    "Search {database} for {tissue} {assay} data",
    "Are there any {organism} {tissue} {assay} datasets available?",
    "I need {assay} data from {organism} {tissue}",
    "Find {disease} {assay} samples in {database}",
    "Search for {histone} ChIP-seq in {organism} {tissue}",
    "Look for {cell_type} {assay} data",
    "Find publicly available {organism} {tissue} {assay}",
]

DATA_DOWNLOAD_TEMPLATES = [
    "Download dataset {dataset_id}",
    "Get {dataset_id}",
    "Fetch dataset {dataset_id}",
    "Download the {database} dataset {dataset_id}",
    "I want to download {dataset_id}",
    "Can you download {dataset_id} for me?",
]

DATA_SCAN_TEMPLATES = [
    "Scan {path} for {format} files",
    "What data do I have in {path}",
    "List files in {path}",
    "Check {path} for data",
    "Show me what's in {path}",
    "Scan the local data directory",
    "What samples are available locally?",
    "Inventory my data in {path}",
]

DATA_DESCRIBE_TEMPLATES = [
    "Show details for {dataset_id}",
    "Describe dataset {dataset_id}",
    "What's in {dataset_id}?",
    "Tell me about {dataset_id}",
    "Get info for {dataset_id}",
]

REFERENCE_CHECK_TEMPLATES = [
    "Do I have the {genome} reference genome?",
    "Check if {genome} reference is available",
    "Is the {genome} genome downloaded?",
    "Check for {genome} reference",
]

WORKFLOW_CREATE_TEMPLATES = [
    "Create a {assay} workflow for {organism}",
    "Generate a {assay} pipeline",
    "Build a {assay} analysis pipeline for {organism} {tissue}",
    "Set up a {assay} workflow",
    "Make a {assay} pipeline for my {organism} samples",
    "I want to run {assay} analysis on {organism} data",
    "Create a differential expression workflow for {assay}",
    "Set up variant calling for {organism}",
    "Generate a peak calling pipeline for ChIP-seq",
]

JOB_SUBMIT_TEMPLATES = [
    "Submit the workflow in {path}",
    "Run the pipeline in {path}",
    "Execute the workflow at {path}",
    "Start the job in {path}",
    "Submit my {assay} workflow",
]

JOB_STATUS_TEMPLATES = [
    "What's the status of job {job_id}?",
    "Check job {job_id}",
    "How is job {job_id} doing?",
    "Status of {job_id}",
    "Is job {job_id} complete?",
    "Check on job {job_id}",
]

JOB_LOGS_TEMPLATES = [
    "Show logs for job {job_id}",
    "Get the logs from job {job_id}",
    "What's the output of job {job_id}?",
    "View job {job_id} logs",
]

JOB_CANCEL_TEMPLATES = [
    "Cancel job {job_id}",
    "Stop job {job_id}",
    "Kill job {job_id}",
    "Abort job {job_id}",
]

JOB_LIST_TEMPLATES = [
    "List my running jobs",
    "What jobs are running?",
    "Show active jobs",
    "List all jobs",
    "What's currently running?",
]

DIAGNOSE_ERROR_TEMPLATES = [
    "Job {job_id} failed, what went wrong?",
    "Diagnose error in job {job_id}",
    "Why did job {job_id} fail?",
    "Debug job {job_id}",
    "Help me fix job {job_id}",
]

EDUCATION_EXPLAIN_TEMPLATES = [
    "What is {assay}?",
    "Explain {assay} to me",
    "How does {assay} work?",
    "Tell me about {assay}",
    "What is differential expression?",
    "Explain peak calling",
    "What is normalization?",
    "How do aligners work?",
    "What is the purpose of QC?",
    "Explain {concept}",
]

EDUCATION_CONCEPTS = [
    "RNA-seq", "ChIP-seq", "ATAC-seq", "Hi-C",
    "differential expression", "peak calling", "alignment",
    "normalization", "FDR", "p-value", "fold change",
    "quality control", "read mapping", "variant calling",
    "gene expression", "transcriptomics", "epigenetics",
]

EDUCATION_HELP_TEMPLATES = [
    "Help",
    "What can you do?",
    "Show me available commands",
    "What are my options?",
    "How do I use this?",
    "List capabilities",
    "What features are available?",
]

COMPOSITE_TEMPLATES = [
    "Check if we have {organism} {assay} data locally, otherwise search {database}",
    "Search for {assay} data and then download it",
    "Create a {assay} pipeline and submit it",
    "Find {organism} {tissue} samples and show details",
]

AMBIGUOUS_TEMPLATES = [
    "data",
    "analysis",
    "I need help with my project",
    "can you help",
    "I have some samples",
    "something with RNA",
    "process my files",
]


# =============================================================================
# Conversation Generator
# =============================================================================

class ConversationGenerator:
    """
    Generates thousands of synthetic test conversations.
    
    Features:
    - Template-based generation with slot filling
    - Paraphrase variations
    - Entity substitution
    - Complexity gradients
    - Edge case generation
    - Multi-turn conversation chains
    """
    
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self._generated_ids = set()
    
    def _generate_id(self, category: str, index: int) -> str:
        """Generate unique conversation ID."""
        prefix = {
            'data_discovery': 'DD',
            'workflow_generation': 'WF',
            'job_management': 'JM',
            'education': 'ED',
            'multi_turn': 'MT',
            'coreference': 'CR',
            'ambiguous': 'AM',
            'edge_cases': 'EC',
            'error_handling': 'EH',
            'adversarial': 'AV',
            'composite': 'CP',
        }.get(category, 'GN')
        
        conv_id = f"{prefix}-{index:04d}"
        while conv_id in self._generated_ids:
            index += 1
            conv_id = f"{prefix}-{index:04d}"
        self._generated_ids.add(conv_id)
        return conv_id
    
    def _random_organism(self) -> tuple:
        return self.random.choice(ORGANISMS)
    
    def _random_tissue(self) -> str:
        return self.random.choice(TISSUES)
    
    def _random_cell_type(self) -> str:
        return self.random.choice(CELL_TYPES)
    
    def _random_assay(self) -> tuple:
        return self.random.choice(ASSAY_TYPES)
    
    def _random_disease(self) -> str:
        return self.random.choice(DISEASES)
    
    def _random_database(self) -> str:
        return self.random.choice(DATABASES)
    
    def _random_histone(self) -> str:
        return self.random.choice(HISTONE_MARKS)
    
    def _random_path(self) -> str:
        return self.random.choice(PATHS)
    
    def _random_format(self) -> str:
        return self.random.choice(FILE_FORMATS)
    
    def _random_genome(self) -> str:
        return self.random.choice(["hg38", "hg19", "mm10", "mm39", "GRCh38"])
    
    def _random_dataset_id(self) -> str:
        patterns = [
            f"GSE{self.random.randint(10000, 200000)}",
            f"ENCSR{self.random.choice('ABCDEF')}{self.random.randint(100, 999)}{self.random.choice('ABCDEF')}{self.random.choice('ABCDEF')}{self.random.choice('ABCDEF')}",
            f"SRR{self.random.randint(1000000, 9999999)}",
        ]
        return self.random.choice(patterns)
    
    def _random_job_id(self) -> str:
        return str(self.random.randint(10000, 99999))
    
    def _fill_template(self, template: str) -> str:
        """Fill a template with random values."""
        result = template
        
        if "{organism}" in result:
            org = self._random_organism()
            # Sometimes use common name, sometimes scientific
            result = result.replace("{organism}", 
                self.random.choice([org[0], org[1]] if len(org) > 1 else [org[0]]))
        
        if "{tissue}" in result:
            result = result.replace("{tissue}", self._random_tissue())
        
        if "{cell_type}" in result:
            result = result.replace("{cell_type}", self._random_cell_type())
        
        if "{assay}" in result:
            assay = self._random_assay()
            # Use different variations
            result = result.replace("{assay}", self.random.choice(assay))
        
        if "{disease}" in result:
            result = result.replace("{disease}", self._random_disease())
        
        if "{database}" in result:
            result = result.replace("{database}", self._random_database())
        
        if "{histone}" in result:
            result = result.replace("{histone}", self._random_histone())
        
        if "{path}" in result:
            result = result.replace("{path}", self._random_path())
        
        if "{format}" in result:
            result = result.replace("{format}", self._random_format())
        
        if "{genome}" in result:
            result = result.replace("{genome}", self._random_genome())
        
        if "{dataset_id}" in result:
            result = result.replace("{dataset_id}", self._random_dataset_id())
        
        if "{job_id}" in result:
            result = result.replace("{job_id}", self._random_job_id())
        
        if "{concept}" in result:
            result = result.replace("{concept}", 
                self.random.choice(EDUCATION_CONCEPTS))
        
        return result
    
    def _apply_variation(self, query: str, difficulty: str) -> str:
        """Apply random variations based on difficulty."""
        if difficulty == "easy":
            return query
        
        variations = []
        
        if difficulty in ["medium", "hard"]:
            # Case variations
            variations.append(lambda q: q.lower())
            variations.append(lambda q: q.upper())
            variations.append(lambda q: q.capitalize())
            
            # Remove punctuation
            variations.append(lambda q: q.rstrip('?'))
            
            # Add filler words
            variations.append(lambda q: f"Please {q.lower()}")
            variations.append(lambda q: f"Can you {q.lower()}")
            variations.append(lambda q: f"I want to {q.lower()}")
        
        if difficulty == "hard":
            # Add typos
            def add_typo(q):
                if len(q) > 10:
                    pos = self.random.randint(3, len(q) - 3)
                    chars = list(q)
                    if chars[pos].isalpha():
                        chars[pos] = self.random.choice('abcdefghijklmnopqrstuvwxyz')
                    return ''.join(chars)
                return q
            variations.append(add_typo)
            
            # Extra whitespace
            variations.append(lambda q: f"  {q}  ")
            
            # Abbreviations
            variations.append(lambda q: q.replace("RNA-seq", "rnaseq"))
            variations.append(lambda q: q.replace("ChIP-seq", "chip"))
        
        if variations:
            variation = self.random.choice(variations)
            return variation(query)
        return query
    
    def _infer_entities(self, query: str) -> Dict[str, str]:
        """Infer expected entities from a query."""
        entities = {}
        query_lower = query.lower()
        
        # Organisms
        for common, scientific in ORGANISMS:
            if common.lower() in query_lower or scientific.lower() in query_lower:
                entities['ORGANISM'] = common
                break
        
        # Tissues
        for tissue in TISSUES:
            if tissue.lower() in query_lower:
                entities['TISSUE'] = tissue
                break
        
        # Assay types
        for assay_group in ASSAY_TYPES:
            for assay in assay_group:
                if assay.lower() in query_lower:
                    entities['ASSAY_TYPE'] = assay_group[0]  # Canonical form
                    break
        
        # Dataset IDs
        dataset_patterns = [
            r'GSE\d+', r'ENCSR[A-Z0-9]+', r'SRR\d+', r'TCGA-[A-Z]+'
        ]
        for pattern in dataset_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities['DATASET_ID'] = match.group()
                break
        
        # Paths
        path_pattern = r'(?:/[\w\-\.]+)+|(?:~/[\w\-\.]+)'
        match = re.search(path_pattern, query)
        if match:
            entities['PATH'] = match.group()
        
        # Job IDs
        job_patterns = [r'job\s+(\d+)', r'job_(\d+)', r'#(\d+)']
        for pattern in job_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities['JOB_ID'] = match.group(1)
                break
        
        return entities
    
    # =========================================================================
    # Generation Methods
    # =========================================================================
    
    def generate_data_search_conversations(
        self, 
        count: int = 200
    ) -> List[Dict]:
        """Generate data search conversations."""
        conversations = []
        
        for i in range(count):
            difficulty = self.random.choice(["easy", "easy", "medium", "hard"])
            template = self.random.choice(DATA_SEARCH_TEMPLATES)
            query = self._fill_template(template)
            query = self._apply_variation(query, difficulty)
            entities = self._infer_entities(query)
            
            conversations.append({
                'id': self._generate_id('data_discovery', i + 1000),
                'name': f"Data Search - {entities.get('ASSAY_TYPE', 'general')}",
                'category': 'data_discovery',
                'difficulty': difficulty,
                'source': 'generated',
                'tags': ['search', entities.get('ASSAY_TYPE', '').lower()],
                'turns': [{
                    'query': query,
                    'expected_intent': 'DATA_SEARCH',
                    'expected_entities': entities,
                    'expected_tool': 'search_databases',
                }]
            })
        
        return conversations
    
    def generate_data_download_conversations(
        self, 
        count: int = 100
    ) -> List[Dict]:
        """Generate data download conversations."""
        conversations = []
        
        for i in range(count):
            difficulty = self.random.choice(["easy", "medium"])
            template = self.random.choice(DATA_DOWNLOAD_TEMPLATES)
            query = self._fill_template(template)
            query = self._apply_variation(query, difficulty)
            entities = self._infer_entities(query)
            
            conversations.append({
                'id': self._generate_id('data_discovery', i + 2000),
                'name': f"Download - {entities.get('DATASET_ID', 'dataset')}",
                'category': 'data_discovery',
                'difficulty': difficulty,
                'source': 'generated',
                'tags': ['download'],
                'turns': [{
                    'query': query,
                    'expected_intent': 'DATA_DOWNLOAD',
                    'expected_entities': entities,
                    'expected_tool': 'download_dataset',
                }]
            })
        
        return conversations
    
    def generate_data_scan_conversations(
        self, 
        count: int = 80
    ) -> List[Dict]:
        """Generate data scan conversations."""
        conversations = []
        
        for i in range(count):
            difficulty = self.random.choice(["easy", "medium"])
            template = self.random.choice(DATA_SCAN_TEMPLATES)
            query = self._fill_template(template)
            query = self._apply_variation(query, difficulty)
            entities = self._infer_entities(query)
            
            conversations.append({
                'id': self._generate_id('data_discovery', i + 3000),
                'name': f"Scan - {entities.get('PATH', 'local')}",
                'category': 'data_discovery',
                'difficulty': difficulty,
                'source': 'generated',
                'tags': ['scan', 'local'],
                'turns': [{
                    'query': query,
                    'expected_intent': 'DATA_SCAN',
                    'expected_entities': entities,
                    'expected_tool': 'scan_data',
                }]
            })
        
        return conversations
    
    def generate_workflow_conversations(
        self, 
        count: int = 150
    ) -> List[Dict]:
        """Generate workflow creation conversations."""
        conversations = []
        
        for i in range(count):
            difficulty = self.random.choice(["easy", "medium", "hard"])
            template = self.random.choice(WORKFLOW_CREATE_TEMPLATES)
            query = self._fill_template(template)
            query = self._apply_variation(query, difficulty)
            entities = self._infer_entities(query)
            
            conversations.append({
                'id': self._generate_id('workflow_generation', i + 1000),
                'name': f"Workflow - {entities.get('ASSAY_TYPE', 'analysis')}",
                'category': 'workflow_generation',
                'difficulty': difficulty,
                'source': 'generated',
                'tags': ['workflow', entities.get('ASSAY_TYPE', '').lower()],
                'turns': [{
                    'query': query,
                    'expected_intent': 'WORKFLOW_CREATE',
                    'expected_entities': entities,
                    'expected_tool': 'generate_workflow',
                }]
            })
        
        return conversations
    
    def generate_job_management_conversations(
        self, 
        count: int = 120
    ) -> List[Dict]:
        """Generate job management conversations."""
        conversations = []
        
        templates_intents = [
            (JOB_SUBMIT_TEMPLATES, 'JOB_SUBMIT', 'submit_job'),
            (JOB_STATUS_TEMPLATES, 'JOB_STATUS', 'get_job_status'),
            (JOB_LOGS_TEMPLATES, 'JOB_LOGS', 'get_logs'),
            (JOB_CANCEL_TEMPLATES, 'JOB_CANCEL', 'cancel_job'),
            (JOB_LIST_TEMPLATES, 'JOB_LIST', 'list_jobs'),
            (DIAGNOSE_ERROR_TEMPLATES, 'DIAGNOSE_ERROR', 'diagnose_error'),
        ]
        
        per_type = count // len(templates_intents)
        
        for templates, intent, tool in templates_intents:
            for i in range(per_type):
                difficulty = self.random.choice(["easy", "medium"])
                template = self.random.choice(templates)
                query = self._fill_template(template)
                query = self._apply_variation(query, difficulty)
                entities = self._infer_entities(query)
                
                conversations.append({
                    'id': self._generate_id('job_management', len(conversations) + 1000),
                    'name': f"Job - {intent}",
                    'category': 'job_management',
                    'difficulty': difficulty,
                    'source': 'generated',
                    'tags': ['job', intent.lower()],
                    'turns': [{
                        'query': query,
                        'expected_intent': intent,
                        'expected_entities': entities,
                        'expected_tool': tool,
                    }]
                })
        
        return conversations
    
    def generate_education_conversations(
        self, 
        count: int = 100
    ) -> List[Dict]:
        """Generate education conversations."""
        conversations = []
        
        # Explain concepts
        for i in range(count * 3 // 4):
            difficulty = self.random.choice(["easy", "medium"])
            template = self.random.choice(EDUCATION_EXPLAIN_TEMPLATES)
            query = self._fill_template(template)
            query = self._apply_variation(query, difficulty)
            
            conversations.append({
                'id': self._generate_id('education', i + 1000),
                'name': f"Explain - concept",
                'category': 'education',
                'difficulty': difficulty,
                'source': 'generated',
                'tags': ['education', 'explain'],
                'turns': [{
                    'query': query,
                    'expected_intent': 'EDUCATION_EXPLAIN',
                    'expected_entities': {},
                    'expected_tool': 'explain_concept',
                }]
            })
        
        # Help requests
        for i in range(count // 4):
            template = self.random.choice(EDUCATION_HELP_TEMPLATES)
            
            conversations.append({
                'id': self._generate_id('education', i + 2000),
                'name': "Help request",
                'category': 'education',
                'difficulty': 'easy',
                'source': 'generated',
                'tags': ['education', 'help'],
                'turns': [{
                    'query': template,
                    'expected_intent': 'EDUCATION_HELP',
                    'expected_entities': {},
                    'expected_tool': 'show_help',
                }]
            })
        
        return conversations
    
    def generate_multi_turn_conversations(
        self, 
        count: int = 100
    ) -> List[Dict]:
        """Generate multi-turn conversations."""
        conversations = []
        
        # Search then download
        for i in range(count // 4):
            org = self._random_organism()
            tissue = self._random_tissue()
            assay = self._random_assay()
            dataset_id = self._random_dataset_id()
            
            conversations.append({
                'id': self._generate_id('multi_turn', i + 1000),
                'name': "Search then Download",
                'category': 'multi_turn',
                'difficulty': 'medium',
                'source': 'generated',
                'tags': ['multi-turn', 'search', 'download'],
                'turns': [
                    {
                        'query': f"Search for {org[0]} {tissue} {assay[0]} data",
                        'expected_intent': 'DATA_SEARCH',
                        'expected_entities': {
                            'ORGANISM': org[0],
                            'TISSUE': tissue,
                            'ASSAY_TYPE': assay[0]
                        },
                    },
                    {
                        'query': f"Download {dataset_id}",
                        'expected_intent': 'DATA_DOWNLOAD',
                        'expected_entities': {'DATASET_ID': dataset_id},
                    }
                ]
            })
        
        # Create workflow then submit
        for i in range(count // 4):
            org = self._random_organism()
            assay = self._random_assay()
            path = self._random_path()
            
            conversations.append({
                'id': self._generate_id('multi_turn', i + 2000),
                'name': "Create then Submit",
                'category': 'multi_turn',
                'difficulty': 'medium',
                'source': 'generated',
                'tags': ['multi-turn', 'workflow', 'submit'],
                'turns': [
                    {
                        'query': f"Create a {assay[0]} pipeline for {org[0]}",
                        'expected_intent': 'WORKFLOW_CREATE',
                        'expected_entities': {
                            'ORGANISM': org[0],
                            'ASSAY_TYPE': assay[0]
                        },
                    },
                    {
                        'query': "Now submit it",
                        'expected_intent': 'JOB_SUBMIT',
                        'expected_entities': {},
                        'context_reference': 'previous_workflow',
                    }
                ]
            })
        
        # Submit then monitor
        for i in range(count // 4):
            path = self._random_path()
            job_id = self._random_job_id()
            
            conversations.append({
                'id': self._generate_id('multi_turn', i + 3000),
                'name': "Submit then Monitor",
                'category': 'multi_turn',
                'difficulty': 'medium',
                'source': 'generated',
                'tags': ['multi-turn', 'job'],
                'turns': [
                    {
                        'query': f"Submit the workflow in {path}",
                        'expected_intent': 'JOB_SUBMIT',
                        'expected_entities': {'PATH': path},
                    },
                    {
                        'query': f"Check status of job {job_id}",
                        'expected_intent': 'JOB_STATUS',
                        'expected_entities': {'JOB_ID': job_id},
                    }
                ]
            })
        
        # Learn then apply
        for i in range(count // 4):
            assay = self._random_assay()
            
            conversations.append({
                'id': self._generate_id('multi_turn', i + 4000),
                'name': "Learn then Apply",
                'category': 'multi_turn',
                'difficulty': 'hard',
                'source': 'generated',
                'tags': ['multi-turn', 'education', 'workflow'],
                'turns': [
                    {
                        'query': f"What is {assay[0]}?",
                        'expected_intent': 'EDUCATION_EXPLAIN',
                        'expected_entities': {},
                    },
                    {
                        'query': f"Create a {assay[0]} workflow",
                        'expected_intent': 'WORKFLOW_CREATE',
                        'expected_entities': {'ASSAY_TYPE': assay[0]},
                    }
                ]
            })
        
        return conversations
    
    def generate_edge_case_conversations(
        self, 
        count: int = 80
    ) -> List[Dict]:
        """Generate edge case conversations."""
        conversations = []
        
        # All caps
        for i in range(count // 8):
            template = self.random.choice(DATA_SEARCH_TEMPLATES)
            query = self._fill_template(template).upper()
            entities = self._infer_entities(query)
            
            conversations.append({
                'id': self._generate_id('edge_cases', i + 1000),
                'name': "All Caps Query",
                'category': 'edge_cases',
                'difficulty': 'medium',
                'source': 'generated',
                'tags': ['edge-case', 'caps'],
                'turns': [{
                    'query': query,
                    'expected_intent': 'DATA_SEARCH',
                    'expected_entities': entities,
                }]
            })
        
        # Typos
        for i in range(count // 8):
            template = self.random.choice(DATA_SEARCH_TEMPLATES)
            query = self._fill_template(template)
            # Add random typos
            query = query.replace('search', 'serch').replace('data', 'daat')
            entities = self._infer_entities(query)
            
            conversations.append({
                'id': self._generate_id('edge_cases', i + 2000),
                'name': "Query with Typos",
                'category': 'edge_cases',
                'difficulty': 'hard',
                'source': 'generated',
                'tags': ['edge-case', 'typos'],
                'turns': [{
                    'query': query,
                    'expected_intent': 'DATA_SEARCH',
                    'expected_entities': entities,
                }]
            })
        
        # Very short
        short_queries = [
            ("hi", "META_GREETING"),
            ("hello", "META_GREETING"),
            ("thanks", "META_THANKS"),
            ("yes", "META_CONFIRM"),
            ("ok", "META_CONFIRM"),
            ("no", "META_CANCEL"),
            ("help", "EDUCATION_HELP"),
            ("?", "EDUCATION_HELP"),
        ]
        for i, (query, intent) in enumerate(short_queries):
            conversations.append({
                'id': self._generate_id('edge_cases', i + 3000),
                'name': f"Short Query - {query}",
                'category': 'edge_cases',
                'difficulty': 'easy',
                'source': 'generated',
                'tags': ['edge-case', 'short'],
                'turns': [{
                    'query': query,
                    'expected_intent': intent,
                    'expected_entities': {},
                }]
            })
        
        # Very long
        for i in range(count // 8):
            org = self._random_organism()
            tissue = self._random_tissue()
            assay = self._random_assay()
            
            query = f"""I am looking for publicly available {assay[0]} datasets that 
            profile {org[0]} {tissue} tissue samples from healthy donors, preferably 
            with at least two biological replicates and proper controls, ideally from 
            the ENCODE consortium or GEO database, and if possible with accompanying 
            metadata about the experimental conditions"""
            
            conversations.append({
                'id': self._generate_id('edge_cases', i + 4000),
                'name': "Very Long Query",
                'category': 'edge_cases',
                'difficulty': 'hard',
                'source': 'generated',
                'tags': ['edge-case', 'long'],
                'turns': [{
                    'query': query,
                    'expected_intent': 'DATA_SEARCH',
                    'expected_entities': {
                        'ORGANISM': org[0],
                        'TISSUE': tissue,
                        'ASSAY_TYPE': assay[0]
                    },
                }]
            })
        
        # Special characters
        for i in range(count // 8):
            org = self._random_organism()
            tissue = self._random_tissue()
            
            query = f"Find H3K4me3 ChIP-seq ({org[0]}, {tissue})"
            
            conversations.append({
                'id': self._generate_id('edge_cases', i + 5000),
                'name': "Special Characters",
                'category': 'edge_cases',
                'difficulty': 'medium',
                'source': 'generated',
                'tags': ['edge-case', 'special-chars'],
                'turns': [{
                    'query': query,
                    'expected_intent': 'DATA_SEARCH',
                    'expected_entities': {
                        'ORGANISM': org[0],
                        'TISSUE': tissue,
                    },
                }]
            })
        
        return conversations
    
    def generate_ambiguous_conversations(
        self, 
        count: int = 50
    ) -> List[Dict]:
        """Generate ambiguous conversations."""
        conversations = []
        
        for i, template in enumerate(AMBIGUOUS_TEMPLATES * (count // len(AMBIGUOUS_TEMPLATES) + 1)):
            if i >= count:
                break
            
            conversations.append({
                'id': self._generate_id('ambiguous', i + 1000),
                'name': "Ambiguous Query",
                'category': 'ambiguous',
                'difficulty': 'hard',
                'source': 'generated',
                'tags': ['ambiguous'],
                'turns': [{
                    'query': template,
                    'expected_intent': 'META_UNKNOWN',
                    'expected_entities': {},
                    'should_clarify': True,
                }]
            })
        
        return conversations
    
    def generate_adversarial_conversations(
        self, 
        count: int = 50
    ) -> List[Dict]:
        """Generate adversarial conversations designed to confuse the parser."""
        conversations = []
        
        adversarial_patterns = [
            # Intent confusion
            ("Download the RNA-seq analysis workflow", "WORKFLOW_CREATE"),  # Not download
            ("Search for a way to create a pipeline", "EDUCATION_EXPLAIN"),  # Not search
            ("Help me download the data I searched for", "DATA_DOWNLOAD"),
            ("Create a search for mouse data", "DATA_SEARCH"),  # Not create workflow
            
            # Entity extraction traps
            ("Find human data but not from liver", "DATA_SEARCH"),
            ("Search for non-RNA-seq assays", "DATA_SEARCH"),
            ("I don't want ChIP-seq, find something else", "DATA_SEARCH"),
            
            # Context traps
            ("Actually, cancel that and search instead", "DATA_SEARCH"),
            ("Wait, I meant download not search", "DATA_DOWNLOAD"),
            ("Forget the workflow, just show status", "JOB_STATUS"),
        ]
        
        for i, (query, intent) in enumerate(adversarial_patterns * (count // len(adversarial_patterns) + 1)):
            if i >= count:
                break
            
            conversations.append({
                'id': self._generate_id('adversarial', i + 1000),
                'name': f"Adversarial - {intent}",
                'category': 'adversarial',
                'difficulty': 'hard',
                'source': 'generated',
                'tags': ['adversarial', 'tricky'],
                'turns': [{
                    'query': query,
                    'expected_intent': intent,
                    'expected_entities': {},
                }]
            })
        
        return conversations
    
    def generate_all(self, total_target: int = 1000) -> List[Dict]:
        """Generate a balanced set of all conversation types."""
        # Distribution of conversation types
        distribution = {
            'data_search': 0.20,
            'data_download': 0.08,
            'data_scan': 0.07,
            'workflow': 0.12,
            'job_management': 0.10,
            'education': 0.10,
            'multi_turn': 0.12,
            'edge_cases': 0.08,
            'ambiguous': 0.05,
            'adversarial': 0.08,
        }
        
        all_conversations = []
        
        all_conversations.extend(self.generate_data_search_conversations(
            int(total_target * distribution['data_search'])))
        all_conversations.extend(self.generate_data_download_conversations(
            int(total_target * distribution['data_download'])))
        all_conversations.extend(self.generate_data_scan_conversations(
            int(total_target * distribution['data_scan'])))
        all_conversations.extend(self.generate_workflow_conversations(
            int(total_target * distribution['workflow'])))
        all_conversations.extend(self.generate_job_management_conversations(
            int(total_target * distribution['job_management'])))
        all_conversations.extend(self.generate_education_conversations(
            int(total_target * distribution['education'])))
        all_conversations.extend(self.generate_multi_turn_conversations(
            int(total_target * distribution['multi_turn'])))
        all_conversations.extend(self.generate_edge_case_conversations(
            int(total_target * distribution['edge_cases'])))
        all_conversations.extend(self.generate_ambiguous_conversations(
            int(total_target * distribution['ambiguous'])))
        all_conversations.extend(self.generate_adversarial_conversations(
            int(total_target * distribution['adversarial'])))
        
        return all_conversations


def populate_database(target_count: int = 1000):
    """Populate the database with generated conversations."""
    db = get_database()
    generator = ConversationGenerator()
    
    print(f"Generating {target_count} conversations...")
    conversations = generator.generate_all(target_count)
    
    print(f"Adding to database...")
    added, skipped = db.add_conversations_bulk(conversations)
    
    print(f"Added: {added}, Skipped (duplicates): {skipped}")
    print(f"Database summary: {db.generate_summary_report()}")
    
    return added


if __name__ == "__main__":
    populate_database(1500)  # Generate 1500 conversations
