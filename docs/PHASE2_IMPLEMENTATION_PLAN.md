# Phase 2: Smart Chat Agent Enhancement - Implementation Plan

> **Version**: 2.0.0  
> **Created**: December 1, 2025  
> **Status**: Planning → Implementation

---

## Executive Summary

This document outlines the implementation plan for enhancing the BioPipelines Smart Chat Agentic System. The plan covers 6 major focus areas with 15+ specific improvements, estimated at 4-6 weeks of development effort.

**Current State Metrics:**
- 46,000+ lines of Python code
- 7 LLM providers (Gemini, Cerebras, Groq, OpenRouter, Lightning, GitHub Models, OpenAI)
- 87.4% accuracy on UnifiedIntentParser
- 10 production Nextflow pipelines

**Target State:**
- Real-time streaming responses
- Persistent session memory
- Auto-provisioning of references/containers
- Multi-agent coordination
- Comprehensive observability

---

## Phase 2.1: Streaming Responses (High UX Impact)

### 2.1.1 Problem Statement
Currently, users wait for the entire LLM response before seeing any output. This creates a poor UX, especially for longer responses like workflow explanations.

### 2.1.2 Current Implementation
```python
# src/workflow_composer/providers/base.py
async def generate(self, prompt: str, **kwargs) -> str:
    # Returns full response at once
    response = await self._call_api(prompt)
    return response.text
```

### 2.1.3 Target Implementation

#### A. Add Streaming to Base Provider
```python
# src/workflow_composer/providers/base.py
from typing import AsyncIterator

async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
    """Stream response tokens as they arrive."""
    async for chunk in self._stream_api(prompt):
        yield chunk.text
```

#### B. Implement for Each Provider

| Provider | Native Streaming | Implementation |
|----------|-----------------|----------------|
| Gemini | ✅ Yes | `generate_content_stream()` |
| Cerebras | ✅ Yes | `stream=True` parameter |
| Groq | ✅ Yes | `stream=True` parameter |
| OpenRouter | ✅ Yes | SSE streaming |
| Lightning | ⚠️ Limited | Polling fallback |
| OpenAI | ✅ Yes | `stream=True` parameter |
| Ollama | ✅ Yes | Native streaming |
| vLLM | ✅ Yes | AsyncEngine streaming |

#### C. Update Router for Streaming
```python
# src/workflow_composer/providers/router.py
async def route_stream(self, prompt: str) -> AsyncIterator[str]:
    """Route with streaming support."""
    provider = await self._select_provider()
    async for chunk in provider.generate_stream(prompt):
        yield chunk
```

#### D. Gradio Streaming Integration
```python
# src/workflow_composer/web/app.py
def chat_stream(message, history):
    """Streaming chat handler for Gradio."""
    response = ""
    for chunk in agent.stream_response(message):
        response += chunk
        yield response
```

### 2.1.4 Files to Modify
1. `src/workflow_composer/providers/base.py` - Add `generate_stream()` abstract method
2. `src/workflow_composer/providers/gemini.py` - Implement Gemini streaming
3. `src/workflow_composer/providers/cerebras.py` - Implement Cerebras streaming
4. `src/workflow_composer/providers/groq.py` - Implement Groq streaming
5. `src/workflow_composer/providers/openrouter.py` - Implement OpenRouter streaming
6. `src/workflow_composer/providers/router.py` - Add `route_stream()` method
7. `src/workflow_composer/agents/unified_agent.py` - Add `stream_response()` method
8. `src/workflow_composer/web/app.py` - Update Gradio to use streaming

### 2.1.5 Testing Strategy
```python
# tests/test_streaming.py
@pytest.mark.asyncio
async def test_gemini_streaming():
    provider = GeminiProvider()
    chunks = []
    async for chunk in provider.generate_stream("Hello"):
        chunks.append(chunk)
    assert len(chunks) > 1  # Multiple chunks received
    assert "".join(chunks)  # Valid response
```

### 2.1.6 Acceptance Criteria
- [ ] All cloud providers support streaming
- [ ] Gradio UI updates in real-time
- [ ] Fallback to non-streaming if provider doesn't support it
- [ ] No increase in latency for first token

---

## Phase 2.2: Session Memory & Learning (High UX Impact)

### 2.2.1 Problem Statement
The system doesn't remember user preferences across sessions. Users must re-specify organism, read type, tool preferences each time.

### 2.2.2 Current Implementation
```python
# src/workflow_composer/agents/memory.py
class ConversationMemory:
    """In-memory conversation storage - lost on restart."""
    def __init__(self):
        self.messages = []  # Ephemeral
```

### 2.2.3 Target Implementation

#### A. User Profile Schema
```python
# src/workflow_composer/agents/memory/user_profile.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json

@dataclass
class UserProfile:
    """Persistent user preferences and history."""
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    # Inferred preferences
    preferred_organism: Optional[str] = None  # "human", "mouse", etc.
    preferred_read_type: Optional[str] = None  # "paired", "single"
    preferred_aligner: Optional[str] = None  # "bwa", "bowtie2", etc.
    
    # Usage history
    query_count: int = 0
    successful_workflows: int = 0
    analysis_types: Dict[str, int] = field(default_factory=dict)  # {"rna-seq": 5, "chip-seq": 2}
    
    # Saved workflows
    workflow_templates: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "preferred_organism": self.preferred_organism,
            "preferred_read_type": self.preferred_read_type,
            "preferred_aligner": self.preferred_aligner,
            "query_count": self.query_count,
            "successful_workflows": self.successful_workflows,
            "analysis_types": self.analysis_types,
            "workflow_templates": self.workflow_templates
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_active"] = datetime.fromisoformat(data["last_active"])
        return cls(**data)
```

#### B. Persistent Memory Store
```python
# src/workflow_composer/agents/memory/persistent_store.py
from pathlib import Path
import sqlite3
import json

class PersistentMemoryStore:
    """SQLite-backed persistent memory."""
    
    def __init__(self, db_path: str = "~/.biopipelines/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS successful_workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    analysis_type TEXT,
                    workflow_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_profile(self, profile: UserProfile):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_profiles (user_id, profile_json) VALUES (?, ?)",
                (profile.user_id, json.dumps(profile.to_dict()))
            )
    
    def load_profile(self, user_id: str) -> Optional[UserProfile]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT profile_json FROM user_profiles WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            if row:
                return UserProfile.from_dict(json.loads(row[0]))
        return None
    
    def save_successful_workflow(self, user_id: str, query: str, 
                                  analysis_type: str, workflow: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO successful_workflows 
                   (user_id, query, analysis_type, workflow_json) VALUES (?, ?, ?, ?)""",
                (user_id, query, analysis_type, json.dumps(workflow))
            )
    
    def get_similar_workflows(self, analysis_type: str, limit: int = 5) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT query, workflow_json FROM successful_workflows 
                   WHERE analysis_type = ? ORDER BY created_at DESC LIMIT ?""",
                (analysis_type, limit)
            ).fetchall()
            return [{"query": r[0], "workflow": json.loads(r[1])} for r in rows]
```

#### C. Preference Learning
```python
# src/workflow_composer/agents/memory/preference_learner.py
class PreferenceLearner:
    """Learn user preferences from interactions."""
    
    def __init__(self, store: PersistentMemoryStore):
        self.store = store
    
    def update_from_query(self, user_id: str, parsed_intent: dict):
        """Update profile based on parsed query intent."""
        profile = self.store.load_profile(user_id) or UserProfile(user_id=user_id)
        
        # Update preferred organism
        if organism := parsed_intent.get("organism"):
            profile.preferred_organism = organism
        
        # Update preferred read type
        if read_type := parsed_intent.get("read_type"):
            profile.preferred_read_type = read_type
        
        # Track analysis type usage
        if analysis_type := parsed_intent.get("analysis_type"):
            profile.analysis_types[analysis_type] = \
                profile.analysis_types.get(analysis_type, 0) + 1
        
        profile.query_count += 1
        profile.last_active = datetime.now()
        self.store.save_profile(profile)
    
    def get_context_for_query(self, user_id: str) -> dict:
        """Get user context to enhance query parsing."""
        profile = self.store.load_profile(user_id)
        if not profile:
            return {}
        
        return {
            "default_organism": profile.preferred_organism,
            "default_read_type": profile.preferred_read_type,
            "preferred_aligner": profile.preferred_aligner,
            "most_common_analysis": max(profile.analysis_types, 
                                         key=profile.analysis_types.get, 
                                         default=None)
        }
```

### 2.2.4 Files to Create/Modify
1. `src/workflow_composer/agents/memory/user_profile.py` - NEW
2. `src/workflow_composer/agents/memory/persistent_store.py` - NEW
3. `src/workflow_composer/agents/memory/preference_learner.py` - NEW
4. `src/workflow_composer/agents/memory/__init__.py` - Update exports
5. `src/workflow_composer/agents/unified_agent.py` - Integrate memory
6. `src/workflow_composer/web/app.py` - Add session management

### 2.2.5 Testing Strategy
```python
# tests/test_persistent_memory.py
def test_profile_persistence():
    store = PersistentMemoryStore(":memory:")
    profile = UserProfile(user_id="test", preferred_organism="human")
    store.save_profile(profile)
    
    loaded = store.load_profile("test")
    assert loaded.preferred_organism == "human"

def test_preference_learning():
    store = PersistentMemoryStore(":memory:")
    learner = PreferenceLearner(store)
    
    learner.update_from_query("user1", {"organism": "mouse", "analysis_type": "rna-seq"})
    learner.update_from_query("user1", {"analysis_type": "rna-seq"})
    
    context = learner.get_context_for_query("user1")
    assert context["default_organism"] == "mouse"
    assert context["most_common_analysis"] == "rna-seq"
```

### 2.2.6 Acceptance Criteria
- [ ] User preferences persist across sessions
- [ ] System learns from successful workflows
- [ ] Similar past workflows suggested for new queries
- [ ] Graceful handling of new users (no profile)

---

## Phase 2.3: Auto-Provisioning of References & Containers (Reliability)

### 2.3.1 Problem Statement
Pipeline execution fails when reference genomes or containers are missing. Users must manually download large files.

### 2.3.2 Current Implementation
```python
# src/workflow_composer/core/preflight_validator.py
def _prepare_reference(self, reference: str) -> Optional[str]:
    # TODO: Implement - download from Ensembl/UCSC
    pass
```

### 2.3.3 Target Implementation

#### A. Reference Genome Manager
```python
# src/workflow_composer/provisioning/reference_manager.py
from pathlib import Path
import subprocess
from typing import Optional
from dataclasses import dataclass

@dataclass
class ReferenceGenome:
    """Reference genome metadata."""
    organism: str
    build: str
    source: str  # ensembl, ucsc, gencode
    fasta_url: str
    gtf_url: Optional[str] = None
    index_type: Optional[str] = None  # bwa, bowtie2, star, hisat2
    local_path: Optional[Path] = None

REFERENCE_CATALOG = {
    "human_GRCh38": ReferenceGenome(
        organism="human",
        build="GRCh38",
        source="ensembl",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz"
    ),
    "mouse_GRCm39": ReferenceGenome(
        organism="mouse",
        build="GRCm39",
        source="ensembl",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/mus_musculus/Mus_musculus.GRCm39.110.gtf.gz"
    ),
    # Add more organisms...
}

class ReferenceManager:
    """Manage reference genome downloads and indexing."""
    
    def __init__(self, base_path: str = "~/data/references"):
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_reference(self, organism: str, build: str = None) -> Optional[Path]:
        """Get reference path, downloading if needed."""
        key = self._resolve_key(organism, build)
        if key not in REFERENCE_CATALOG:
            return None
        
        ref = REFERENCE_CATALOG[key]
        local_fasta = self.base_path / ref.organism / ref.build / "genome.fa"
        
        if not local_fasta.exists():
            self._download_reference(ref)
        
        return local_fasta
    
    def ensure_index(self, organism: str, build: str, index_type: str) -> Path:
        """Ensure index exists, building if needed."""
        fasta = self.get_reference(organism, build)
        index_dir = fasta.parent / "indices" / index_type
        
        if not self._index_exists(index_dir, index_type):
            self._build_index(fasta, index_dir, index_type)
        
        return index_dir
    
    def _download_reference(self, ref: ReferenceGenome):
        """Download and decompress reference."""
        dest_dir = self.base_path / ref.organism / ref.build
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Download FASTA
        fasta_gz = dest_dir / "genome.fa.gz"
        subprocess.run(["wget", "-O", str(fasta_gz), ref.fasta_url], check=True)
        subprocess.run(["gunzip", str(fasta_gz)], check=True)
        
        # Download GTF if available
        if ref.gtf_url:
            gtf_gz = dest_dir / "genes.gtf.gz"
            subprocess.run(["wget", "-O", str(gtf_gz), ref.gtf_url], check=True)
            subprocess.run(["gunzip", str(gtf_gz)], check=True)
    
    def _build_index(self, fasta: Path, index_dir: Path, index_type: str):
        """Build aligner index."""
        index_dir.mkdir(parents=True, exist_ok=True)
        
        if index_type == "bwa":
            subprocess.run(["bwa", "index", "-p", str(index_dir / "genome"), str(fasta)])
        elif index_type == "bowtie2":
            subprocess.run(["bowtie2-build", str(fasta), str(index_dir / "genome")])
        elif index_type == "star":
            subprocess.run([
                "STAR", "--runMode", "genomeGenerate",
                "--genomeDir", str(index_dir),
                "--genomeFastaFiles", str(fasta),
                "--runThreadN", "8"
            ])
        elif index_type == "hisat2":
            subprocess.run(["hisat2-build", str(fasta), str(index_dir / "genome")])
```

#### B. Container Manager
```python
# src/workflow_composer/provisioning/container_manager.py
from pathlib import Path
import subprocess
from typing import Dict, Optional

CONTAINER_REGISTRY = {
    "base": "docker://ghcr.io/sdodlapati3/biopipelines-base:latest",
    "rna-seq": "docker://ghcr.io/sdodlapati3/biopipelines-rnaseq:latest",
    "chip-seq": "docker://ghcr.io/sdodlapati3/biopipelines-chipseq:latest",
    "dna-seq": "docker://ghcr.io/sdodlapati3/biopipelines-dnaseq:latest",
    # Map all pipeline types...
}

class ContainerManager:
    """Manage Singularity container provisioning."""
    
    def __init__(self, cache_dir: str = "~/.biopipelines/containers"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_container(self, pipeline_type: str) -> Optional[Path]:
        """Get container path, pulling if needed."""
        if pipeline_type not in CONTAINER_REGISTRY:
            return None
        
        sif_path = self.cache_dir / f"{pipeline_type}.sif"
        
        if not sif_path.exists():
            self._pull_container(pipeline_type, sif_path)
        
        return sif_path
    
    def verify_container(self, sif_path: Path) -> bool:
        """Verify container is valid and runnable."""
        try:
            result = subprocess.run(
                ["singularity", "exec", str(sif_path), "echo", "OK"],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _pull_container(self, pipeline_type: str, sif_path: Path):
        """Pull container from registry."""
        docker_uri = CONTAINER_REGISTRY[pipeline_type]
        subprocess.run([
            "singularity", "pull", "--force",
            str(sif_path), docker_uri
        ], check=True)
    
    def list_available(self) -> Dict[str, bool]:
        """List containers and their availability."""
        return {
            name: (self.cache_dir / f"{name}.sif").exists()
            for name in CONTAINER_REGISTRY
        }
```

#### C. Preflight Validator Integration
```python
# Update src/workflow_composer/core/preflight_validator.py
class PreflightValidator:
    def __init__(self):
        self.ref_manager = ReferenceManager()
        self.container_manager = ContainerManager()
    
    async def validate_and_provision(self, workflow_config: dict) -> ValidationResult:
        """Validate workflow requirements and auto-provision missing resources."""
        issues = []
        provisions = []
        
        # Check references
        organism = workflow_config.get("organism")
        if organism:
            ref_path = self.ref_manager.get_reference(organism)
            if ref_path:
                provisions.append(f"Reference genome: {ref_path}")
            else:
                issues.append(f"Unknown organism: {organism}")
        
        # Check indexes
        aligner = workflow_config.get("aligner", "bwa")
        if organism and aligner:
            index_path = self.ref_manager.ensure_index(organism, None, aligner)
            provisions.append(f"{aligner} index: {index_path}")
        
        # Check containers
        pipeline_type = workflow_config.get("pipeline_type")
        if pipeline_type:
            container = self.container_manager.get_container(pipeline_type)
            if container and self.container_manager.verify_container(container):
                provisions.append(f"Container: {container}")
            else:
                issues.append(f"Container unavailable: {pipeline_type}")
        
        return ValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            provisions=provisions
        )
```

### 2.3.4 Files to Create/Modify
1. `src/workflow_composer/provisioning/__init__.py` - NEW
2. `src/workflow_composer/provisioning/reference_manager.py` - NEW
3. `src/workflow_composer/provisioning/container_manager.py` - NEW
4. `src/workflow_composer/core/preflight_validator.py` - Update with provisioning
5. `config/references.yaml` - Reference catalog configuration
6. `config/containers.yaml` - Container registry configuration

### 2.3.5 Testing Strategy
```python
# tests/test_provisioning.py
def test_reference_catalog():
    manager = ReferenceManager("/tmp/test_refs")
    assert "human_GRCh38" in REFERENCE_CATALOG
    
def test_container_verification(tmp_path):
    manager = ContainerManager(str(tmp_path))
    # Mock container for testing
    sif = tmp_path / "test.sif"
    sif.touch()
    # Real test would need actual container
```

### 2.3.6 Acceptance Criteria
- [ ] Common references (human, mouse) auto-download
- [ ] Indexes auto-build for required aligners
- [ ] Containers auto-pull from registry
- [ ] Clear progress reporting during provisioning
- [ ] Graceful failure with actionable error messages

---

## Phase 2.4: Multi-Agent Coordination (Advanced Capability)

### 2.4.1 Problem Statement
Current agents work independently. Complex workflows need coordinated specialist agents for planning, code generation, validation, and documentation.

### 2.4.2 Current Implementation
```python
# src/workflow_composer/agents/orchestrator.py
class Orchestrator:
    """Basic task routing to single agent."""
    pass
```

### 2.4.3 Target Implementation

#### A. Agent Hierarchy
```
SupervisorAgent (coordinator)
├── PlannerAgent (workflow design)
├── CodeGenAgent (Nextflow generation)
├── ValidatorAgent (code review)
├── DocAgent (documentation)
└── QCAgent (output validation)
```

#### B. Supervisor Agent
```python
# src/workflow_composer/agents/supervisor.py
from typing import List, Dict, Any
from enum import Enum

class AgentRole(Enum):
    PLANNER = "planner"
    CODEGEN = "codegen"
    VALIDATOR = "validator"
    DOCS = "documentation"
    QC = "quality_control"

class SupervisorAgent:
    """Coordinates specialist agents for complex workflows."""
    
    def __init__(self, router: ProviderRouter):
        self.router = router
        self.agents = {
            AgentRole.PLANNER: PlannerAgent(router),
            AgentRole.CODEGEN: CodeGenAgent(router),
            AgentRole.VALIDATOR: ValidatorAgent(router),
            AgentRole.DOCS: DocAgent(router),
            AgentRole.QC: QCAgent(router),
        }
        self.execution_log = []
    
    async def execute_workflow_generation(self, user_query: str) -> WorkflowResult:
        """Coordinate agents to generate a complete workflow."""
        
        # Step 1: Planning
        self._log("Starting workflow planning...")
        plan = await self.agents[AgentRole.PLANNER].create_plan(user_query)
        
        # Step 2: Code Generation
        self._log("Generating Nextflow code...")
        code = await self.agents[AgentRole.CODEGEN].generate(plan)
        
        # Step 3: Validation Loop
        self._log("Validating generated code...")
        validation = await self.agents[AgentRole.VALIDATOR].validate(code)
        
        max_iterations = 3
        iteration = 0
        while not validation.is_valid and iteration < max_iterations:
            self._log(f"Fixing issues (iteration {iteration + 1})...")
            code = await self.agents[AgentRole.CODEGEN].fix_issues(
                code, validation.issues
            )
            validation = await self.agents[AgentRole.VALIDATOR].validate(code)
            iteration += 1
        
        # Step 4: Documentation
        self._log("Generating documentation...")
        docs = await self.agents[AgentRole.DOCS].generate(plan, code)
        
        return WorkflowResult(
            plan=plan,
            code=code,
            documentation=docs,
            validation=validation,
            execution_log=self.execution_log
        )
```

#### C. Specialist Agents

```python
# src/workflow_composer/agents/specialists/planner.py
class PlannerAgent:
    """Designs workflow architecture."""
    
    SYSTEM_PROMPT = """You are a bioinformatics workflow architect.
    Given a user query, create a detailed workflow plan including:
    1. Input requirements (file types, organism, read type)
    2. Processing steps in order
    3. Tools for each step with parameters
    4. Output files and formats
    5. Quality control checkpoints
    
    Output as structured JSON."""
    
    async def create_plan(self, query: str) -> WorkflowPlan:
        response = await self.router.route(
            f"{self.SYSTEM_PROMPT}\n\nQuery: {query}"
        )
        return WorkflowPlan.from_json(response)


# src/workflow_composer/agents/specialists/codegen.py
class CodeGenAgent:
    """Generates Nextflow DSL2 code."""
    
    SYSTEM_PROMPT = """You are a Nextflow DSL2 expert.
    Given a workflow plan, generate production-ready Nextflow code.
    
    Requirements:
    - Use DSL2 syntax with processes and workflows
    - Include proper input/output channels
    - Add resource directives (cpus, memory, time)
    - Use containers for reproducibility
    - Include error handling with errorStrategy
    
    Output complete main.nf file."""
    
    async def generate(self, plan: WorkflowPlan) -> str:
        response = await self.router.route(
            f"{self.SYSTEM_PROMPT}\n\nPlan:\n{plan.to_json()}"
        )
        return self._extract_code(response)


# src/workflow_composer/agents/specialists/validator.py
class ValidatorAgent:
    """Reviews and validates generated code."""
    
    SYSTEM_PROMPT = """You are a Nextflow code reviewer.
    Validate the following code for:
    1. Syntax correctness
    2. DSL2 best practices
    3. Resource specifications
    4. Error handling
    5. Channel connections
    6. Container availability
    
    Return JSON with {valid: bool, issues: [...], suggestions: [...]}"""
    
    async def validate(self, code: str) -> ValidationResult:
        # Static analysis first
        static_issues = self._static_analysis(code)
        
        # LLM review
        response = await self.router.route(
            f"{self.SYSTEM_PROMPT}\n\nCode:\n{code}"
        )
        llm_review = json.loads(response)
        
        return ValidationResult(
            is_valid=len(static_issues) == 0 and llm_review["valid"],
            issues=static_issues + llm_review.get("issues", []),
            suggestions=llm_review.get("suggestions", [])
        )
    
    def _static_analysis(self, code: str) -> List[str]:
        """Basic static checks."""
        issues = []
        if "process " not in code:
            issues.append("No processes defined")
        if "workflow " not in code:
            issues.append("No workflow block defined")
        if "container" not in code:
            issues.append("No container directives found")
        return issues
```

### 2.4.4 Files to Create/Modify
1. `src/workflow_composer/agents/supervisor.py` - NEW
2. `src/workflow_composer/agents/specialists/__init__.py` - NEW
3. `src/workflow_composer/agents/specialists/planner.py` - NEW
4. `src/workflow_composer/agents/specialists/codegen.py` - NEW
5. `src/workflow_composer/agents/specialists/validator.py` - NEW
6. `src/workflow_composer/agents/specialists/docs.py` - NEW
7. `src/workflow_composer/agents/specialists/qc.py` - NEW
8. `src/workflow_composer/agents/orchestrator.py` - Integrate supervisor

### 2.4.5 Testing Strategy
```python
# tests/test_multi_agent.py
@pytest.mark.asyncio
async def test_supervisor_workflow():
    supervisor = SupervisorAgent(mock_router)
    result = await supervisor.execute_workflow_generation(
        "RNA-seq differential expression for human"
    )
    assert result.plan is not None
    assert "process" in result.code
    assert result.documentation is not None
```

### 2.4.6 Acceptance Criteria
- [ ] Supervisor coordinates all specialist agents
- [ ] Validation loop catches and fixes issues
- [ ] Documentation auto-generated for workflows
- [ ] Execution log provides transparency
- [ ] Graceful degradation if agents fail

---

## Phase 2.5: Observability & Analytics (Operations)

### 2.5.1 Problem Statement
No visibility into provider usage, query patterns, or system health. Difficult to optimize costs and performance.

### 2.5.2 Target Implementation

#### A. Metrics Collection
```python
# src/workflow_composer/observability/metrics.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import json

@dataclass
class ProviderMetrics:
    """Metrics for a single provider."""
    provider_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0
        return self.total_latency_ms / self.successful_requests

class MetricsCollector:
    """Collect and aggregate system metrics."""
    
    def __init__(self, db_path: str = "~/.biopipelines/metrics.db"):
        self.db_path = Path(db_path).expanduser()
        self._init_db()
        self._provider_metrics: Dict[str, ProviderMetrics] = {}
    
    def record_request(self, provider_id: str, success: bool, 
                       latency_ms: float, tokens: int = 0, error: str = None):
        """Record a provider request."""
        if provider_id not in self._provider_metrics:
            self._provider_metrics[provider_id] = ProviderMetrics(provider_id)
        
        m = self._provider_metrics[provider_id]
        m.total_requests += 1
        if success:
            m.successful_requests += 1
            m.total_latency_ms += latency_ms
            m.total_tokens += tokens
        else:
            m.failed_requests += 1
            if error:
                m.errors_by_type[error] = m.errors_by_type.get(error, 0) + 1
        
        # Persist to DB
        self._save_request(provider_id, success, latency_ms, tokens, error)
    
    def get_dashboard_data(self) -> dict:
        """Get data for dashboard display."""
        return {
            "providers": {
                pid: {
                    "requests": m.total_requests,
                    "success_rate": f"{m.success_rate:.1%}",
                    "avg_latency": f"{m.avg_latency_ms:.0f}ms",
                    "tokens": m.total_tokens
                }
                for pid, m in self._provider_metrics.items()
            },
            "total_requests": sum(m.total_requests for m in self._provider_metrics.values()),
            "overall_success_rate": self._overall_success_rate()
        }
```

#### B. Query Analytics
```python
# src/workflow_composer/observability/query_analytics.py
class QueryAnalytics:
    """Analyze query patterns and success rates."""
    
    def __init__(self, db_path: str = "~/.biopipelines/analytics.db"):
        self.db_path = Path(db_path).expanduser()
        self._init_db()
    
    def record_query(self, query: str, analysis_type: str, 
                     organism: str, success: bool, duration_ms: float):
        """Record a user query."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO queries (query, analysis_type, organism, success, duration_ms)
                VALUES (?, ?, ?, ?, ?)
            """, (query, analysis_type, organism, success, duration_ms))
    
    def get_analysis_distribution(self, days: int = 30) -> Dict[str, int]:
        """Get distribution of analysis types."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT analysis_type, COUNT(*) as count
                FROM queries
                WHERE timestamp > datetime('now', ?)
                GROUP BY analysis_type
                ORDER BY count DESC
            """, (f"-{days} days",)).fetchall()
            return dict(rows)
    
    def get_success_by_type(self) -> Dict[str, float]:
        """Get success rate by analysis type."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT analysis_type, 
                       AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as rate
                FROM queries
                GROUP BY analysis_type
            """).fetchall()
            return {r[0]: r[1] for r in rows}
```

#### C. Health Endpoint
```python
# src/workflow_composer/api/routes/health.py
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@router.get("/health/providers")
async def provider_health():
    """Check health of all LLM providers."""
    from workflow_composer.providers import get_registry
    
    registry = get_registry()
    health = {}
    
    for provider_config in registry.list_providers():
        try:
            provider = registry.get_provider(provider_config.id)
            start = datetime.now()
            await provider.generate("Hello", max_tokens=5)
            latency = (datetime.now() - start).total_seconds() * 1000
            health[provider_config.id] = {
                "status": "healthy",
                "latency_ms": latency
            }
        except Exception as e:
            health[provider_config.id] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    return health

@router.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    from workflow_composer.observability import get_metrics_collector
    return get_metrics_collector().get_dashboard_data()
```

### 2.5.3 Files to Create/Modify
1. `src/workflow_composer/observability/__init__.py` - NEW
2. `src/workflow_composer/observability/metrics.py` - NEW
3. `src/workflow_composer/observability/query_analytics.py` - NEW
4. `src/workflow_composer/api/routes/health.py` - NEW/Update
5. `src/workflow_composer/providers/router.py` - Add metrics recording
6. `src/workflow_composer/web/app.py` - Add metrics dashboard tab

### 2.5.4 Acceptance Criteria
- [ ] Provider usage tracked with latency/tokens
- [ ] Query patterns analyzed
- [ ] `/health` endpoint for monitoring
- [ ] `/metrics` endpoint for dashboards
- [ ] Success rates by analysis type

---

## Phase 2.6: RAG Enhancement (Context Quality)

### 2.6.1 Problem Statement
RAG currently limited to tool catalog. Need to index nf-core modules, paper abstracts, and error patterns for better context.

### 2.6.2 Target Implementation

#### A. Expanded Knowledge Base
```python
# src/workflow_composer/agents/rag/knowledge_base.py
from enum import Enum

class KnowledgeSource(Enum):
    TOOL_CATALOG = "tool_catalog"
    NF_CORE_MODULES = "nf_core_modules"
    PAPER_ABSTRACTS = "paper_abstracts"
    ERROR_PATTERNS = "error_patterns"
    BEST_PRACTICES = "best_practices"

class KnowledgeBase:
    """Multi-source knowledge base for RAG."""
    
    def __init__(self, base_path: str = "~/.biopipelines/knowledge"):
        self.base_path = Path(base_path).expanduser()
        self.sources = {}
    
    async def index_nf_core(self):
        """Index nf-core modules from GitHub."""
        # Clone/update nf-core/modules
        modules_path = self.base_path / "nf-core-modules"
        if not modules_path.exists():
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/nf-core/modules.git",
                str(modules_path)
            ])
        
        # Index all module meta.yml files
        for meta_file in modules_path.glob("**/meta.yml"):
            module_info = yaml.safe_load(meta_file.read_text())
            self._add_to_index(
                KnowledgeSource.NF_CORE_MODULES,
                module_info
            )
    
    async def index_error_patterns(self):
        """Index common error patterns and solutions."""
        error_patterns = [
            {
                "pattern": "No space left on device",
                "cause": "Disk full during execution",
                "solution": "Increase scratch space or clean temp files"
            },
            {
                "pattern": "SLURM job exceeded memory limit",
                "cause": "Insufficient memory allocation",
                "solution": "Increase memory in process directive"
            },
            # Add more patterns...
        ]
        for pattern in error_patterns:
            self._add_to_index(KnowledgeSource.ERROR_PATTERNS, pattern)
```

### 2.6.3 Files to Create/Modify
1. `src/workflow_composer/agents/rag/knowledge_base.py` - NEW
2. `src/workflow_composer/agents/rag/nf_core_indexer.py` - NEW
3. `src/workflow_composer/agents/rag/error_patterns.py` - NEW
4. `config/error_patterns.yaml` - Error pattern catalog
5. `scripts/index_knowledge.py` - Indexing script

---

## Implementation Schedule

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | 2.1 Streaming | Streaming for all providers, Gradio integration |
| 2 | 2.2 Memory | User profiles, preference learning, persistence |
| 3 | 2.3 Provisioning | Reference manager, container manager, preflight |
| 4 | 2.4 Multi-Agent | Supervisor, specialist agents, validation loop |
| 5 | 2.5 Observability | Metrics, analytics, health endpoints |
| 6 | 2.6 RAG | Knowledge base expansion, nf-core indexing |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| API rate limits during testing | Use mock providers for unit tests |
| Large reference downloads | Implement resumable downloads, progress bars |
| Container registry unavailable | Cache containers locally, fallback to Docker Hub |
| Multi-agent latency | Parallel agent execution where possible |
| Database corruption | Regular backups, WAL mode for SQLite |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| First token latency | ~500ms | <200ms (streaming) |
| User preference accuracy | N/A | >90% correct defaults |
| Reference provisioning success | Manual | >95% auto |
| Workflow generation accuracy | 87.4% | >92% |
| System uptime | N/A | 99.5% |

---

## Appendix: Quick Wins (Can Implement Immediately)

### A. Add Gemini Flash Model
```python
# In src/workflow_composer/providers/gemini.py
MODELS = {
    "gemini-2.0-flash-exp": {"context": 1048576, "speed": "fast"},
    "gemini-1.5-pro": {"context": 2097152, "speed": "medium"},
    "gemini-1.5-flash": {"context": 1048576, "speed": "fastest"},  # ADD THIS
}
```

### B. Conversation Export
```python
# In src/workflow_composer/web/app.py
def export_conversation(history):
    """Export chat as markdown."""
    md = "# BioPipelines Conversation\n\n"
    for user, assistant in history:
        md += f"**User:** {user}\n\n**Assistant:** {assistant}\n\n---\n\n"
    return md
```

### C. Provider Status Endpoint
```python
# Quick health check
@app.get("/api/status")
async def status():
    from workflow_composer.providers import check_providers
    return check_providers()
```

---

*Document Version: 1.0 | Last Updated: December 1, 2025*
