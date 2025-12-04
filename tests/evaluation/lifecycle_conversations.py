"""
Lifecycle Conversation Test Suite
=================================

Comprehensive multi-turn (5+ turns) conversation tests covering:
1. Full workflow lifecycle: Search → Generate → Validate → Submit → Monitor → Interpret
2. Error recovery flows: Job fails → Diagnose → Fix → Resubmit → Monitor
3. Iterative refinement: Create → Review → Modify → Validate → Submit
4. Complex data discovery: Search → Filter → Compare → Download → Verify

These tests validate:
- Context retention across many turns
- Coreference resolution (it, that, the job, etc.)
- Intent transitions (education → workflow → job management)
- Error handling and recovery suggestions
- Workflow validation feedback

Usage:
    python -m tests.evaluation.lifecycle_conversations
    python -m tests.evaluation.lifecycle_conversations --scenario workflow_lifecycle
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class LifecycleCategory(Enum):
    """Categories of lifecycle tests."""
    WORKFLOW_LIFECYCLE = "workflow_lifecycle"
    ERROR_RECOVERY = "error_recovery"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    DATA_DISCOVERY_FLOW = "data_discovery_flow"
    EDUCATION_TO_ACTION = "education_to_action"
    JOB_MONITORING = "job_monitoring"


@dataclass
class ExpectedResult:
    """Expected result for a turn."""
    intent: str
    entities: Dict[str, str] = field(default_factory=dict)
    tool: Optional[str] = None
    contains_keywords: List[str] = field(default_factory=list)
    should_clarify: bool = False
    context_reference: Optional[str] = None
    validation_checks: List[str] = field(default_factory=list)  # What to validate
    

@dataclass
class ConversationTurn:
    """A single turn with rich context."""
    turn_number: int
    user_query: str
    expected: ExpectedResult
    description: str = ""
    simulated_response: Optional[str] = None  # For context propagation
    simulated_data: Optional[Dict[str, Any]] = None  # Job IDs, dataset IDs, etc.


@dataclass
class LifecycleConversation:
    """A complete lifecycle conversation test."""
    id: str
    name: str
    category: LifecycleCategory
    description: str
    turns: List[ConversationTurn]
    difficulty: str = "hard"
    expected_context_keys: List[str] = field(default_factory=list)


# =============================================================================
# LIFECYCLE SCENARIO 1: Full Workflow Lifecycle (8 turns)
# =============================================================================

WORKFLOW_LIFECYCLE_1 = LifecycleConversation(
    id="LC-WF-001",
    name="Complete RNA-seq Analysis Lifecycle",
    category=LifecycleCategory.WORKFLOW_LIFECYCLE,
    description="""
    Full lifecycle from data discovery to result interpretation:
    1. Search for data
    2. Check references
    3. Generate workflow
    4. Validate workflow
    5. Submit job
    6. Monitor progress
    7. Check results
    8. Interpret QC
    """,
    expected_context_keys=["search_results", "workflow_path", "job_id", "results_path"],
    turns=[
        ConversationTurn(
            turn_number=1,
            user_query="Search for human liver RNA-seq data from GEO",
            expected=ExpectedResult(
                intent="DATA_SEARCH",
                entities={"ORGANISM": "human", "TISSUE": "liver", "ASSAY_TYPE": "RNA-seq"},
                tool="search_databases",
            ),
            description="Initial data search",
            simulated_response="Found 15 datasets matching your query. Top result: GSE12345 - Human liver RNA-seq (6 samples)",
            simulated_data={"search_results": ["GSE12345", "GSE67890", "GSE11111"]},
        ),
        ConversationTurn(
            turn_number=2,
            user_query="Download the first one",
            expected=ExpectedResult(
                intent="DATA_DOWNLOAD",
                entities={"DATASET_ID": "GSE12345"},
                tool="download_dataset",
                context_reference="search_results",
            ),
            description="Download using context from previous search",
            simulated_response="Downloading GSE12345 to /data/raw/GSE12345/",
            simulated_data={"data_path": "/data/raw/GSE12345"},
        ),
        ConversationTurn(
            turn_number=3,
            user_query="Check if we have the human reference genome",
            expected=ExpectedResult(
                intent="REFERENCE_CHECK",
                entities={"ORGANISM": "human"},
                tool="check_references",
            ),
            description="Verify references before workflow creation",
            simulated_response="✅ Human reference genome (GRCh38) available at /data/references/human/GRCh38",
        ),
        ConversationTurn(
            turn_number=4,
            user_query="Create an RNA-seq workflow for this data with differential expression analysis",
            expected=ExpectedResult(
                intent="WORKFLOW_CREATE",
                entities={"ASSAY_TYPE": "RNA-seq"},
                tool="generate_workflow",
                context_reference="data_path",
                contains_keywords=["differential", "expression"],
            ),
            description="Generate workflow using context",
            simulated_response="Generated RNA-seq workflow at /workflows/rnaseq_GSE12345/main.nf",
            simulated_data={"workflow_path": "/workflows/rnaseq_GSE12345/main.nf"},
        ),
        ConversationTurn(
            turn_number=5,
            user_query="Validate the workflow before running",
            expected=ExpectedResult(
                intent="WORKFLOW_VALIDATE",  # New intent for validation
                tool="validate_workflow",
                context_reference="workflow_path",
                validation_checks=["syntax", "semantic", "references"],
            ),
            description="Validate generated workflow",
            simulated_response="✅ Workflow validation passed. Score: 95/100. No errors, 1 warning: Consider adding memory limits.",
        ),
        ConversationTurn(
            turn_number=6,
            user_query="Submit it to the cluster",
            expected=ExpectedResult(
                intent="JOB_SUBMIT",
                tool="submit_job",
                context_reference="workflow_path",
            ),
            description="Submit the validated workflow",
            simulated_response="Job submitted successfully. Job ID: 27548",
            simulated_data={"job_id": "27548"},
        ),
        ConversationTurn(
            turn_number=7,
            user_query="How is the job doing?",
            expected=ExpectedResult(
                intent="JOB_STATUS",
                entities={"JOB_ID": "27548"},
                tool="get_job_status",
                context_reference="job_id",
            ),
            description="Check job status using context",
            simulated_response="Job 27548 is RUNNING (45% complete). ETA: 2 hours.",
        ),
        ConversationTurn(
            turn_number=8,
            user_query="Once it's done, interpret the QC results",
            expected=ExpectedResult(
                intent="ANALYSIS_INTERPRET",
                tool="interpret_results",
                context_reference="job_id",
                contains_keywords=["QC", "quality"],
            ),
            description="Plan for result interpretation",
            simulated_response="I'll analyze the QC results when the job completes. Key metrics I'll check: alignment rate, duplication rate, gene detection.",
        ),
    ]
)


# =============================================================================
# LIFECYCLE SCENARIO 2: Error Recovery Flow (7 turns)
# =============================================================================

ERROR_RECOVERY_1 = LifecycleConversation(
    id="LC-ER-001",
    name="Job Failure Diagnosis and Recovery",
    category=LifecycleCategory.ERROR_RECOVERY,
    description="""
    Error recovery workflow:
    1. Check job status (fails)
    2. View error logs
    3. Diagnose the error
    4. Get fix suggestion
    5. Apply fix to workflow
    6. Resubmit job
    7. Monitor recovery
    """,
    expected_context_keys=["job_id", "error_type", "workflow_path"],
    turns=[
        ConversationTurn(
            turn_number=1,
            user_query="Check status of my ChIP-seq job",
            expected=ExpectedResult(
                intent="JOB_STATUS",
                tool="get_job_status",
            ),
            description="Initial status check reveals failure",
            simulated_response="Job 34521 FAILED after 2 hours. Exit code: 137 (OOM)",
            simulated_data={"job_id": "34521", "status": "FAILED", "error_code": 137},
        ),
        ConversationTurn(
            turn_number=2,
            user_query="Show me the error logs",
            expected=ExpectedResult(
                intent="JOB_LOGS",
                entities={"JOB_ID": "34521"},
                tool="get_logs",
                context_reference="job_id",
            ),
            description="View logs for failed job",
            simulated_response="Error in MACS2 peak calling: Java heap space error. Process killed (OOM).",
            simulated_data={"error_type": "OOM", "failing_process": "MACS2"},
        ),
        ConversationTurn(
            turn_number=3,
            user_query="What went wrong and how do I fix it?",
            expected=ExpectedResult(
                intent="DIAGNOSE_ERROR",
                tool="diagnose_error",
                context_reference="job_id",
            ),
            description="Diagnose the OOM error",
            simulated_response="""
            **Diagnosis**: Out of Memory (OOM) error in MACS2 peak calling.
            
            **Root cause**: MACS2 requested more memory than allocated (8GB requested, 4GB limit).
            
            **Recommended fixes**:
            1. Increase memory allocation for MACS2 process to 16GB
            2. Add memory directive: `memory '16 GB'`
            """,
            simulated_data={"fix_suggestion": "increase_memory", "process": "MACS2", "recommended_memory": "16GB"},
        ),
        ConversationTurn(
            turn_number=4,
            user_query="Apply that fix to the workflow",
            expected=ExpectedResult(
                intent="WORKFLOW_MODIFY",  # Modify existing workflow
                tool="modify_workflow",
                context_reference="fix_suggestion",
            ),
            description="Apply the memory fix",
            simulated_response="Updated MACS2 process with `memory '16 GB'`. Workflow saved.",
            simulated_data={"workflow_path": "/workflows/chipseq/main.nf", "modified": True},
        ),
        ConversationTurn(
            turn_number=5,
            user_query="Validate the changes",
            expected=ExpectedResult(
                intent="WORKFLOW_VALIDATE",
                tool="validate_workflow",
                context_reference="workflow_path",
            ),
            description="Validate modified workflow",
            simulated_response="✅ Workflow validation passed. Memory configuration looks correct.",
        ),
        ConversationTurn(
            turn_number=6,
            user_query="Resubmit the job",
            expected=ExpectedResult(
                intent="JOB_RESUBMIT",
                tool="submit_job",
                context_reference="job_id",
            ),
            description="Resubmit with fixes",
            simulated_response="Job resubmitted. New Job ID: 34599",
            simulated_data={"new_job_id": "34599"},
        ),
        ConversationTurn(
            turn_number=7,
            user_query="Monitor it and let me know if it fails again",
            expected=ExpectedResult(
                intent="JOB_WATCH",
                entities={"JOB_ID": "34599"},
                tool="monitor_job",
                context_reference="new_job_id",
            ),
            description="Set up monitoring for the retry",
            simulated_response="Monitoring job 34599. I'll notify you of any issues.",
        ),
    ]
)


# =============================================================================
# LIFECYCLE SCENARIO 3: Iterative Workflow Refinement (6 turns)
# =============================================================================

ITERATIVE_REFINEMENT_1 = LifecycleConversation(
    id="LC-IR-001",
    name="Iterative Workflow Tool Selection",
    category=LifecycleCategory.ITERATIVE_REFINEMENT,
    description="""
    User iteratively refines workflow tool choices:
    1. Create initial workflow
    2. Review tool selection
    3. Request tool change
    4. Add a step
    5. Validate final workflow
    6. Submit
    """,
    expected_context_keys=["workflow_path", "aligner"],  # Track tool changes via aligner key
    turns=[
        ConversationTurn(
            turn_number=1,
            user_query="Create an ATAC-seq workflow for mouse samples",
            expected=ExpectedResult(
                intent="WORKFLOW_CREATE",
                entities={"ASSAY_TYPE": "ATAC-seq", "ORGANISM": "mouse"},
                tool="generate_workflow",
            ),
            description="Initial workflow creation",
            simulated_response="""
            Generated ATAC-seq workflow with:
            - Alignment: bowtie2
            - Peak calling: MACS2
            - QC: FastQC, MultiQC
            
            Workflow saved to: /workflows/atacseq_mouse/main.nf
            """,
            simulated_data={"workflow_path": "/workflows/atacseq_mouse/main.nf", "aligner": "bowtie2", "peak_caller": "MACS2"},
        ),
        ConversationTurn(
            turn_number=2,
            user_query="What aligner did you use?",
            expected=ExpectedResult(
                intent="WORKFLOW_DESCRIBE",  # Ask about workflow details
                tool="describe_workflow",
                context_reference="workflow_path",
            ),
            description="Query workflow details",
            simulated_response="The workflow uses bowtie2 for alignment, which is the default for ATAC-seq.",
        ),
        ConversationTurn(
            turn_number=3,
            user_query="Use BWA-MEM2 instead, it's faster",
            expected=ExpectedResult(
                intent="WORKFLOW_MODIFY",
                entities={"TOOL": "bwa-mem2"},
                tool="modify_workflow",
                context_reference="workflow_path",
            ),
            description="User requests tool change",
            simulated_response="Updated aligner from bowtie2 to BWA-MEM2. Note: BWA-MEM2 requires a different index format.",
            simulated_data={"aligner": "bwa-mem2"},
        ),
        ConversationTurn(
            turn_number=4,
            user_query="Also add nucleosome positioning analysis",
            expected=ExpectedResult(
                intent="WORKFLOW_MODIFY",
                tool="modify_workflow",
                context_reference="workflow_path",
                contains_keywords=["nucleosome"],
            ),
            description="Add additional analysis step",
            simulated_response="Added NucleoATAC for nucleosome positioning analysis after peak calling.",
        ),
        ConversationTurn(
            turn_number=5,
            user_query="Validate the complete workflow",
            expected=ExpectedResult(
                intent="WORKFLOW_VALIDATE",
                tool="validate_workflow",
                context_reference="workflow_path",
                validation_checks=["syntax", "semantic", "tool_compatibility"],
            ),
            description="Final validation",
            simulated_response="""
            ✅ Workflow validation passed (Score: 92/100)
            
            Checks:
            - Syntax: ✅ Valid Nextflow DSL2
            - Semantic: ✅ No unused channels
            - Tools: ✅ All tools available
            - References: ✅ Mouse mm10 available
            
            Warning: Consider adding error handling for NucleoATAC.
            """,
        ),
        ConversationTurn(
            turn_number=6,
            user_query="Looks good, submit it",
            expected=ExpectedResult(
                intent="JOB_SUBMIT",
                tool="submit_job",
                context_reference="workflow_path",
            ),
            description="Submit refined workflow",
            simulated_response="Job submitted. Job ID: 45678",
            simulated_data={"job_id": "45678"},
        ),
    ]
)


# =============================================================================
# LIFECYCLE SCENARIO 4: Complex Data Discovery (6 turns)
# =============================================================================

DATA_DISCOVERY_FLOW_1 = LifecycleConversation(
    id="LC-DD-001",
    name="Complex Data Discovery and Comparison",
    category=LifecycleCategory.DATA_DISCOVERY_FLOW,
    description="""
    Multi-step data discovery with comparison:
    1. Search ENCODE
    2. Search GEO
    3. Compare results
    4. Filter by criteria
    5. Download selected
    6. Verify download
    """,
    expected_context_keys=["encode_results", "geo_results", "selected_datasets"],
    turns=[
        ConversationTurn(
            turn_number=1,
            user_query="Search ENCODE for human brain H3K27ac ChIP-seq",
            expected=ExpectedResult(
                intent="DATA_SEARCH",
                entities={"ORGANISM": "human", "TISSUE": "brain", "ASSAY_TYPE": "ChIP-seq", "DATABASE": "ENCODE"},
                tool="search_databases",
            ),
            description="Search ENCODE database",
            simulated_response="Found 8 datasets in ENCODE matching your criteria.",
            simulated_data={"encode_results": ["ENCSR001", "ENCSR002", "ENCSR003"]},
        ),
        ConversationTurn(
            turn_number=2,
            user_query="Also search GEO for the same",
            expected=ExpectedResult(
                intent="DATA_SEARCH",
                entities={"DATABASE": "GEO"},
                tool="search_databases",
                context_reference="previous_search_criteria",
            ),
            description="Search GEO with same criteria",
            simulated_response="Found 12 datasets in GEO matching human brain H3K27ac ChIP-seq.",
            simulated_data={"geo_results": ["GSE001", "GSE002", "GSE003", "GSE004"]},
        ),
        ConversationTurn(
            turn_number=3,
            user_query="Compare the quality of results from both databases",
            expected=ExpectedResult(
                intent="DATA_COMPARE",
                tool="compare_datasets",
                context_reference="encode_results,geo_results",
            ),
            description="Compare ENCODE vs GEO results",
            simulated_response="""
            **Comparison Summary:**
            
            ENCODE (8 datasets):
            - Average read depth: 45M
            - All have input controls
            - Standardized processing
            
            GEO (12 datasets):
            - Average read depth: 32M (varies widely)
            - 8/12 have input controls
            - Various processing pipelines
            
            Recommendation: ENCODE datasets are more consistent.
            """,
        ),
        ConversationTurn(
            turn_number=4,
            user_query="Filter to only datasets with more than 50 million reads",
            expected=ExpectedResult(
                intent="DATA_FILTER",
                tool="filter_datasets",
                context_reference="encode_results,geo_results",
            ),
            description="Apply quality filter",
            simulated_response="After filtering: 5 ENCODE datasets, 3 GEO datasets meet the criteria.",
            simulated_data={"filtered_results": ["ENCSR001", "ENCSR002", "GSE001"]},
        ),
        ConversationTurn(
            turn_number=5,
            user_query="Download all the filtered ENCODE ones",
            expected=ExpectedResult(
                intent="DATA_DOWNLOAD",
                tool="download_dataset",
                context_reference="filtered_results",
            ),
            description="Download filtered selection",
            simulated_response="Downloading 5 ENCODE datasets to /data/raw/encode_h3k27ac/",
            simulated_data={"download_path": "/data/raw/encode_h3k27ac"},
        ),
        ConversationTurn(
            turn_number=6,
            user_query="Verify the downloads completed successfully",
            expected=ExpectedResult(
                intent="DATA_VALIDATE",
                tool="validate_data",
                context_reference="download_path",
            ),
            description="Verify downloaded data",
            simulated_response="""
            ✅ Download verification complete:
            - 5/5 datasets downloaded
            - All checksums match
            - Total size: 125 GB
            - Files: 30 FASTQ files
            """,
        ),
    ]
)


# =============================================================================
# LIFECYCLE SCENARIO 5: Education to Action Flow (5 turns)
# =============================================================================

EDUCATION_TO_ACTION_1 = LifecycleConversation(
    id="LC-EA-001",
    name="Learn Methylation Then Analyze",
    category=LifecycleCategory.EDUCATION_TO_ACTION,
    description="""
    User learns about analysis type then performs it:
    1. Ask about methylation analysis
    2. Ask about tools
    3. Find relevant data
    4. Create workflow
    5. Submit analysis
    """,
    expected_context_keys=["learned_concept", "workflow_path"],
    turns=[
        ConversationTurn(
            turn_number=1,
            user_query="What is DNA methylation analysis and why would I do it?",
            expected=ExpectedResult(
                intent="EDUCATION_EXPLAIN",
                tool="explain_concept",
                contains_keywords=["methylation", "epigenetic"],
            ),
            description="Educational query about methylation",
            simulated_response="""
            **DNA Methylation Analysis**
            
            DNA methylation is an epigenetic modification where methyl groups are added to cytosine bases.
            
            **Why analyze it:**
            - Understand gene regulation
            - Study cancer epigenetics
            - Identify tissue-specific patterns
            
            **Common methods:**
            - Whole Genome Bisulfite Seq (WGBS)
            - Reduced Representation Bisulfite Seq (RRBS)
            - Methylation arrays (450K, EPIC)
            """,
            simulated_data={"learned_concept": "methylation"},
        ),
        ConversationTurn(
            turn_number=2,
            user_query="What tools do you recommend for WGBS analysis?",
            expected=ExpectedResult(
                intent="EDUCATION_EXPLAIN",
                tool="explain_concept",
                contains_keywords=["bismark", "bsmap", "methyldackel"],
            ),
            description="Ask about tools",
            simulated_response="""
            **Recommended WGBS Tools:**
            
            1. **Bismark** (most popular)
               - Alignment + methylation calling
               - Well documented
            
            2. **bwa-meth + MethylDackel**
               - Faster alignment
               - Flexible calling
            
            3. **BSMAP**
               - Good for large datasets
            """,
        ),
        ConversationTurn(
            turn_number=3,
            user_query="Find some human liver WGBS data I can practice with",
            expected=ExpectedResult(
                intent="DATA_SEARCH",
                entities={"ORGANISM": "human", "TISSUE": "liver", "ASSAY_TYPE": "WGBS"},
                tool="search_databases",
            ),
            description="Search for practice data",
            simulated_response="Found 6 WGBS datasets from human liver. Top: GSE89999 (normal liver, 3 samples).",
            simulated_data={"dataset_id": "GSE89999"},
        ),
        ConversationTurn(
            turn_number=4,
            user_query="Create a WGBS workflow using Bismark for this data",
            expected=ExpectedResult(
                intent="WORKFLOW_CREATE",
                entities={"ASSAY_TYPE": "WGBS", "TOOL": "bismark"},
                tool="generate_workflow",
                context_reference="dataset_id",
            ),
            description="Create workflow based on learned concepts",
            simulated_response="Generated WGBS workflow with Bismark at /workflows/wgbs_liver/main.nf",
            simulated_data={"workflow_path": "/workflows/wgbs_liver/main.nf"},
        ),
        ConversationTurn(
            turn_number=5,
            user_query="Great, let's run it!",
            expected=ExpectedResult(
                intent="JOB_SUBMIT",
                tool="submit_job",
                context_reference="workflow_path",
            ),
            description="Submit the learning-inspired workflow",
            simulated_response="Job submitted. Job ID: 56789. This WGBS analysis will take approximately 8 hours.",
            simulated_data={"job_id": "56789"},
        ),
    ]
)


# =============================================================================
# LIFECYCLE SCENARIO 6: Extended Job Monitoring (7 turns)
# =============================================================================

JOB_MONITORING_1 = LifecycleConversation(
    id="LC-JM-001",
    name="Extended Job Monitoring Session",
    category=LifecycleCategory.JOB_MONITORING,
    description="""
    Extended monitoring with multiple check-ins:
    1. Submit job
    2. Initial status check
    3. Check resource usage
    4. View intermediate logs
    5. Job completes
    6. View final results
    7. Generate report
    """,
    expected_context_keys=["job_id", "results_path"],
    turns=[
        ConversationTurn(
            turn_number=1,
            user_query="Submit the scRNA-seq workflow for my 10X data",
            expected=ExpectedResult(
                intent="JOB_SUBMIT",
                entities={"ASSAY_TYPE": "scRNA-seq"},
                tool="submit_job",
            ),
            description="Submit single-cell workflow",
            simulated_response="Submitted scRNA-seq workflow. Job ID: 67890. Estimated runtime: 6 hours.",
            simulated_data={"job_id": "67890"},
        ),
        ConversationTurn(
            turn_number=2,
            user_query="Is it running yet?",
            expected=ExpectedResult(
                intent="JOB_STATUS",
                tool="get_job_status",
                context_reference="job_id",
            ),
            description="Initial status check",
            simulated_response="Job 67890 is RUNNING. Currently executing: CellRanger count (step 2/7).",
        ),
        ConversationTurn(
            turn_number=3,
            user_query="How much memory is it using?",
            expected=ExpectedResult(
                intent="JOB_RESOURCES",
                tool="get_job_resources",
                context_reference="job_id",
            ),
            description="Check resource usage",
            simulated_response="""
            **Resource Usage for Job 67890:**
            - Memory: 45 GB / 64 GB (70%)
            - CPUs: 16/16 (100%)
            - Disk: 120 GB written
            - Runtime: 1h 23m
            """,
        ),
        ConversationTurn(
            turn_number=4,
            user_query="Show me the logs so far",
            expected=ExpectedResult(
                intent="JOB_LOGS",
                tool="get_logs",
                context_reference="job_id",
            ),
            description="View intermediate logs",
            simulated_response="""
            [Recent log output]
            2024-01-15 10:23:45 - CellRanger count started
            2024-01-15 10:24:12 - Processing sample_1: 45M reads
            2024-01-15 10:45:33 - Alignment complete: 92% mapped
            2024-01-15 11:02:18 - UMI counting in progress...
            """,
        ),
        ConversationTurn(
            turn_number=5,
            user_query="Check on it again",
            expected=ExpectedResult(
                intent="JOB_STATUS",
                tool="get_job_status",
                context_reference="job_id",
            ),
            description="Follow-up status check",
            simulated_response="Job 67890 COMPLETED successfully! Runtime: 5h 42m. Results in /results/scrnaseq_67890/",
            simulated_data={"status": "COMPLETED", "results_path": "/results/scrnaseq_67890"},
        ),
        ConversationTurn(
            turn_number=6,
            user_query="What are the key results?",
            expected=ExpectedResult(
                intent="ANALYSIS_INTERPRET",
                tool="interpret_results",
                context_reference="results_path",
            ),
            description="View result summary",
            simulated_response="""
            **scRNA-seq Results Summary:**
            
            - Total cells: 12,450
            - Median genes/cell: 2,847
            - Clusters identified: 15
            - Top marker genes: CD3E (T-cells), CD79A (B-cells)
            
            UMAP and cluster plots available in results folder.
            """,
        ),
        ConversationTurn(
            turn_number=7,
            user_query="Generate a QC report I can share",
            expected=ExpectedResult(
                intent="ANALYSIS_REPORT",
                tool="generate_report",
                context_reference="results_path",
            ),
            description="Generate shareable report",
            simulated_response="Generated HTML report at /results/scrnaseq_67890/qc_report.html",
        ),
    ]
)


# =============================================================================
# Get All Lifecycle Conversations
# =============================================================================

def get_lifecycle_conversations() -> List[LifecycleConversation]:
    """Return all lifecycle conversation tests."""
    return [
        WORKFLOW_LIFECYCLE_1,
        ERROR_RECOVERY_1,
        ITERATIVE_REFINEMENT_1,
        DATA_DISCOVERY_FLOW_1,
        EDUCATION_TO_ACTION_1,
        JOB_MONITORING_1,
    ]


def get_conversations_by_category(category: LifecycleCategory) -> List[LifecycleConversation]:
    """Get conversations for a specific category."""
    return [c for c in get_lifecycle_conversations() if c.category == category]


# =============================================================================
# Lifecycle Evaluator
# =============================================================================

class LifecycleEvaluator:
    """Evaluator for lifecycle conversations."""
    
    def __init__(self):
        self._parser = None
        self._context = {}
    
    @property
    def parser(self):
        """Lazy load parser."""
        if self._parser is None:
            try:
                from src.workflow_composer.agents.intent.parser import IntentParser
                self._parser = IntentParser(llm_client=None, use_semantic=False)
            except ImportError:
                from workflow_composer.agents.intent.parser import IntentParser
                self._parser = IntentParser(llm_client=None, use_semantic=False)
        return self._parser
    
    def evaluate_turn(
        self,
        turn: ConversationTurn,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single turn with context."""
        result = self.parser.parse(turn.user_query, context)
        
        actual_intent = result.primary_intent.name
        expected_intent = turn.expected.intent
        
        # Check intent match
        intent_correct = actual_intent == expected_intent
        
        # Check context reference was used
        context_used = False
        if turn.expected.context_reference:
            context_used = turn.expected.context_reference in context
        
        # Update context with simulated data
        if turn.simulated_data:
            context.update(turn.simulated_data)
        
        return {
            "turn_number": turn.turn_number,
            "query": turn.user_query,
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "intent_correct": intent_correct,
            "confidence": result.confidence,
            "context_reference": turn.expected.context_reference,
            "context_used": context_used,
            "context_keys": list(context.keys()),
        }
    
    def evaluate_conversation(
        self,
        conversation: LifecycleConversation
    ) -> Dict[str, Any]:
        """Evaluate a complete lifecycle conversation."""
        context = {}
        turn_results = []
        
        for turn in conversation.turns:
            result = self.evaluate_turn(turn, context)
            turn_results.append(result)
        
        # Calculate metrics
        total_turns = len(turn_results)
        correct_intents = sum(1 for t in turn_results if t["intent_correct"])
        intent_accuracy = correct_intents / total_turns if total_turns > 0 else 0
        
        # Check context retention
        final_context_keys = turn_results[-1]["context_keys"] if turn_results else []
        expected_keys = set(conversation.expected_context_keys)
        actual_keys = set(final_context_keys)
        context_retention = len(expected_keys & actual_keys) / len(expected_keys) if expected_keys else 1.0
        
        return {
            "conversation_id": conversation.id,
            "conversation_name": conversation.name,
            "category": conversation.category.value,
            "total_turns": total_turns,
            "correct_intents": correct_intents,
            "intent_accuracy": intent_accuracy,
            "context_retention": context_retention,
            "turn_results": turn_results,
            "passed": intent_accuracy >= 0.8 and context_retention >= 0.6,
        }
    
    def run_all(self) -> Dict[str, Any]:
        """Run all lifecycle conversation tests."""
        conversations = get_lifecycle_conversations()
        results = []
        
        for conv in conversations:
            result = self.evaluate_conversation(conv)
            results.append(result)
        
        # Aggregate metrics
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        avg_intent_accuracy = sum(r["intent_accuracy"] for r in results) / total if total > 0 else 0
        avg_context_retention = sum(r["context_retention"] for r in results) / total if total > 0 else 0
        
        return {
            "total_conversations": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_intent_accuracy": avg_intent_accuracy,
            "avg_context_retention": avg_context_retention,
            "results": results,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run lifecycle conversation tests."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Run lifecycle conversation tests")
    parser.add_argument("--scenario", type=str, help="Specific scenario to run")
    parser.add_argument("--category", type=str, help="Category to run")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    evaluator = LifecycleEvaluator()
    
    if args.scenario:
        # Find specific scenario
        conversations = get_lifecycle_conversations()
        conv = next((c for c in conversations if c.id == args.scenario), None)
        if conv:
            result = evaluator.evaluate_conversation(conv)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\n{'='*60}")
                print(f"Scenario: {conv.name} ({conv.id})")
                print(f"Category: {conv.category.value}")
                print(f"{'='*60}")
                for turn in result["turn_results"]:
                    status = "✅" if turn["intent_correct"] else "❌"
                    print(f"{status} Turn {turn['turn_number']}: \"{turn['query'][:50]}...\"")
                    print(f"   Expected: {turn['expected_intent']}, Got: {turn['actual_intent']}")
                print(f"\nIntent Accuracy: {result['intent_accuracy']*100:.1f}%")
                print(f"Context Retention: {result['context_retention']*100:.1f}%")
                print(f"Passed: {'✅' if result['passed'] else '❌'}")
        else:
            print(f"Scenario '{args.scenario}' not found")
    else:
        # Run all
        results = evaluator.run_all()
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\n{'='*60}")
            print("Lifecycle Conversation Test Results")
            print(f"{'='*60}")
            
            for r in results["results"]:
                status = "✅" if r["passed"] else "❌"
                print(f"{status} {r['conversation_name']}")
                print(f"   Turns: {r['total_turns']}, Intent Acc: {r['intent_accuracy']*100:.1f}%, Context: {r['context_retention']*100:.1f}%")
            
            print(f"\n{'='*60}")
            print(f"Summary: {results['passed']}/{results['total_conversations']} passed ({results['pass_rate']*100:.1f}%)")
            print(f"Avg Intent Accuracy: {results['avg_intent_accuracy']*100:.1f}%")
            print(f"Avg Context Retention: {results['avg_context_retention']*100:.1f}%")


if __name__ == "__main__":
    main()
