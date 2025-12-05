# MCP Enhancement Plan for BioPipelines

> **Document Version**: 1.0  
> **Created**: December 2025  
> **Status**: Planning  
> **Scope**: 10-20 pipelines (current scale), future-proofed for growth

---

## Executive Summary

BioPipelines already has a **robust, production-ready MCP server** at `src/workflow_composer/mcp/server.py`. This document synthesizes recommendations from multiple sources (ChatGPT analysis, MCP specification updates, and internal code review) to provide a prioritized enhancement plan that is **appropriate for our current 10-20 pipeline scale** while remaining extensible.

### Key Principles

1. **Don't over-engineer** - Our current scale (10-20 pipelines) doesn't require the full registry abstraction ChatGPT suggests
2. **MCP as canonical API** - Valid long-term vision, but implement incrementally
3. **Human-in-the-loop** - Critical for HPC job submission (per MCP spec security guidelines)
4. **Fewer, richer tools** - Avoid tool explosion by using parameterized tools

---

## 1. Current State Assessment

### 1.1 What We Have (Excellent Foundation) ✅

| Component | Status | Lines | Notes |
|-----------|--------|-------|-------|
| `src/workflow_composer/mcp/server.py` | Production-ready | 1123 | BioPipelinesMCPServer class |
| `tests/test_mcp_server.py` | Comprehensive | 432 | 24 tests, all passing |
| `docs/MCP_INTEGRATION.md` | Complete | 304 | Configuration guide |
| `mcp-config.json` | Ready | 11 | Claude Code/Cursor config |

### 1.2 Registered Tools (12 total)

| Category | Tools | Status |
|----------|-------|--------|
| **Data Discovery** | `search_encode`, `search_geo` | ✅ Working |
| **Workflow** | `create_workflow`, `use_workflow_template` | ✅ Working |
| **Databases** | 6 tools (UniProt, STRING, KEGG, etc.) | ✅ Working |
| **Education** | `explain_concept` | ✅ Working |
| **Job Management** | `check_job_status` | ⚠️ Read-only |

### 1.3 Current Pipeline Inventory

From `config/analysis_definitions.yaml`:

```yaml
# Tier 1: Core Pipelines (7)
- rna_seq_basic
- rna_seq_differential_expression  
- chip_seq_peak_calling
- atac_seq
- wgs_variant_calling
- single_cell_rna_seq
- metagenomics_profiling

# Tier 2: Specialized (5)
- bisulfite_seq_methylation
- hic_chromatin_interaction
- long_read_assembly
- structural_variant_detection
- somatic_variant_calling

# Tier 3: Advanced/Multi-omics (5+)
- spatial_transcriptomics
- multi_omics_integration
- proteogenomics
- long_read_rna_seq
- multi_modal_scrna
```

**Current Count**: ~17 analysis types defined (matches our 10-20 target)

---

## 2. Enhancement Priorities

### Priority Matrix

| Priority | Enhancement | Effort | Impact | ChatGPT Suggested |
|----------|-------------|--------|--------|-------------------|
| 🔴 **P1** | Add SLURM job management tools | Medium | High | Yes |
| 🔴 **P1** | Add output schemas to tools | Low | High | No (MCP spec) |
| 🟡 **P2** | Add MCP Prompts for workflows | Medium | Medium | Implicit |
| 🟡 **P2** | Add strategy management tools | Low | Medium | No (our idea) |
| 🟡 **P2** | Dynamic pipeline resources | Medium | Medium | Yes |
| 🟢 **P3** | Resource templates (URI patterns) | Medium | Low | Yes |
| 🟢 **P3** | High-level agent tools | High | Low | Yes |
| ⚪ **Skip** | Full PipelineRegistry abstraction | High | Low | Yes |

### Why We Skip Full PipelineRegistry

ChatGPT's suggestion for a `PipelineRegistry` with `PipelineSpec` dataclasses makes sense at 50-100 pipelines, but at our current scale:

- We have only ~17 analysis types
- `config/analysis_definitions.yaml` already serves as our registry
- `config/workflow_templates/` already has structured templates
- Adding a registry layer adds maintenance overhead without proportional benefit

**Recommendation**: Revisit when pipeline count exceeds 30.

---

## 3. Phase 1: SLURM Job Management Tools (P1)

### 3.1 Missing Tools Identified

The MCP server exposes `check_job_status` but lacks:

```python
# Currently missing from MCP but exist in agents/tools:
- submit_job      # sbatch wrapper
- cancel_job      # scancel wrapper  
- list_jobs       # squeue for current user
- get_job_logs    # Tail SLURM logs
```

### 3.2 Proposed Implementation

Add to `BioPipelinesMCPServer._setup_tools()`:

```python
# Job Submission (with human confirmation requirement)
self._register_tool(
    name="submit_job",
    description="Submit a workflow job to the SLURM cluster. REQUIRES USER CONFIRMATION.",
    parameters={
        "type": "object",
        "properties": {
            "workflow_path": {
                "type": "string",
                "description": "Path to workflow file (Nextflow/Snakemake)"
            },
            "partition": {
                "type": "string",
                "enum": ["t4flex", "gpu", "standard"],
                "default": "t4flex",
                "description": "SLURM partition"
            },
            "time_limit": {
                "type": "string", 
                "default": "4:00:00",
                "description": "Job time limit (HH:MM:SS)"
            },
            "dry_run": {
                "type": "boolean",
                "default": True,
                "description": "If true, only validate and show sbatch command"
            }
        },
        "required": ["workflow_path"]
    },
    handler=self._handle_submit_job
)

# List Jobs
self._register_tool(
    name="list_jobs",
    description="List user's current SLURM jobs.",
    parameters={
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "enum": ["all", "running", "pending", "completed"],
                "default": "all"
            },
            "limit": {
                "type": "integer",
                "default": 20
            }
        }
    },
    handler=self._handle_list_jobs
)

# Cancel Job (with confirmation)
self._register_tool(
    name="cancel_job",
    description="Cancel a running SLURM job. REQUIRES USER CONFIRMATION.",
    parameters={
        "type": "object", 
        "properties": {
            "job_id": {
                "type": "string",
                "description": "SLURM job ID to cancel"
            }
        },
        "required": ["job_id"]
    },
    handler=self._handle_cancel_job
)

# Get Logs
self._register_tool(
    name="get_job_logs",
    description="Retrieve logs from a SLURM job.",
    parameters={
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "SLURM job ID"
            },
            "log_type": {
                "type": "string",
                "enum": ["stdout", "stderr", "nextflow", "all"],
                "default": "all"
            },
            "tail_lines": {
                "type": "integer",
                "default": 100,
                "description": "Number of lines from end"
            }
        },
        "required": ["job_id"]
    },
    handler=self._handle_get_job_logs
)
```

### 3.3 Security Considerations

Per MCP spec (2025-06-18):

> "For trust & safety and security, there SHOULD always be a human in the loop with the ability to deny tool invocations."

Implementation requirements:
1. **Dry-run by default** for `submit_job`
2. **Confirmation flags** in tool annotations
3. **Workspace sandboxing** (all paths under `/scratch/$USER/biopipelines/`)
4. **Job ID validation** before cancel operations

### 3.4 Tool Annotations (New MCP Feature)

MCP 2025-06-18 supports tool annotations. We should add:

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    annotations: Optional[Dict[str, Any]] = None  # NEW

# Usage:
self._register_tool(
    name="submit_job",
    ...,
    annotations={
        "requires_confirmation": True,
        "category": "job_management",
        "destructive": False
    }
)

self._register_tool(
    name="cancel_job",
    ...,
    annotations={
        "requires_confirmation": True,
        "category": "job_management", 
        "destructive": True
    }
)
```

---

## 4. Phase 1: Output Schemas (P1)

### 4.1 MCP Spec Enhancement

MCP 2025-06-18 added `outputSchema` support for tools. This helps LLMs understand and parse structured responses.

### 4.2 Implementation

Update existing tools with output schemas:

```python
self._register_tool(
    name="search_encode",
    description="Search ENCODE database for experiments.",
    parameters={...},  # existing
    outputSchema={
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "count": {"type": "integer"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "accession": {"type": "string"},
                        "assay": {"type": "string"},
                        "biosample": {"type": "string"},
                        "target": {"type": "string"}
                    }
                }
            },
            "error": {"type": "string"}
        },
        "required": ["success"]
    },
    handler=self._handle_search_encode
)
```

### 4.3 Handler Updates

Return both `content` (text) and `structuredContent` (JSON):

```python
async def _handle_search_encode(self, **kwargs) -> Dict[str, Any]:
    try:
        result = search_data(...)
        
        structured = {
            "success": True,
            "count": len(result.data),
            "results": [
                {
                    "accession": r.get("accession"),
                    "assay": r.get("assay_title"),
                    "biosample": r.get("biosample_summary"),
                    "target": r.get("target", {}).get("label")
                }
                for r in result.data[:10]
            ]
        }
        
        return {
            "success": True,
            "content": self._format_search_results(result),
            "structuredContent": structured
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "structuredContent": {"success": False, "error": str(e)}
        }
```

---

## 5. Phase 2: MCP Prompts (P2)

### 5.1 What Are MCP Prompts?

Prompts are **user-controlled prompt templates** that can be discovered and invoked. Unlike tools (model-controlled), prompts are explicitly selected by users.

Use cases:
- Common workflow patterns (RNA-seq DE, ChIP-seq peaks)
- Analysis templates with embedded resources
- Educational walkthroughs

### 5.2 Proposed Prompts

```python
def _setup_prompts(self):
    """Register MCP prompts for common workflows."""
    
    self._register_prompt(
        name="analyze_rnaseq",
        title="RNA-seq Analysis Wizard",
        description="Guide through RNA-seq differential expression analysis",
        arguments=[
            {"name": "data_path", "description": "Path to FASTQ files", "required": True},
            {"name": "organism", "description": "Organism (human, mouse)", "required": True}
        ],
        handler=self._handle_prompt_rnaseq
    )
    
    self._register_prompt(
        name="debug_workflow",
        title="Debug Failed Workflow",
        description="Analyze workflow errors and suggest fixes",
        arguments=[
            {"name": "job_id", "description": "Failed job ID", "required": True}
        ],
        handler=self._handle_prompt_debug
    )
    
    self._register_prompt(
        name="find_datasets",
        title="Find Public Datasets",
        description="Search ENCODE/GEO for datasets matching criteria",
        arguments=[
            {"name": "topic", "description": "Research topic or keywords", "required": True},
            {"name": "organism", "description": "Target organism", "required": False}
        ],
        handler=self._handle_prompt_find_datasets
    )
```

### 5.3 Prompt Handler Example

```python
async def _handle_prompt_rnaseq(self, data_path: str, organism: str) -> Dict:
    """Generate RNA-seq analysis prompt with embedded resources."""
    
    # Get relevant template documentation
    template_doc = await self.read_resource("biopipelines://templates")
    
    return {
        "description": "RNA-seq differential expression analysis setup",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"""I want to analyze RNA-seq data for differential expression.

**Data Location**: {data_path}
**Organism**: {organism}

Please help me:
1. Validate my input data
2. Select appropriate workflow template
3. Configure parameters for my organism
4. Submit the analysis job"""
                }
            },
            {
                "role": "assistant", 
                "content": {
                    "type": "resource",
                    "resource": {
                        "uri": "biopipelines://templates",
                        "mimeType": "text/markdown",
                        "text": template_doc[:2000]  # Truncate for context
                    }
                }
            }
        ]
    }
```

---

## 6. Phase 2: Strategy Management Tools (P2)

### 6.1 Integration with Dynamic Strategy System

Our new LLM strategy system (Phase 1-2 completed) should be exposed via MCP:

```python
# In _setup_tools():
self._register_tool(
    name="get_llm_strategy",
    description="Get current LLM routing strategy and metrics.",
    parameters={"type": "object", "properties": {}},
    handler=self._handle_get_strategy
)

self._register_tool(
    name="set_llm_strategy",
    description="Switch LLM routing strategy profile.",
    parameters={
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "enum": ["t4_hybrid", "t4_local_only", "cloud_only", "development"],
                "description": "Strategy profile to activate"
            }
        },
        "required": ["profile"]
    },
    handler=self._handle_set_strategy
)
```

### 6.2 Handler Implementation

```python
async def _handle_get_strategy(self, **kwargs) -> Dict[str, Any]:
    """Get current strategy and routing metrics."""
    try:
        from workflow_composer.llm import LLMOrchestrator
        from workflow_composer.llm.metrics import RoutingMetrics
        
        orchestrator = LLMOrchestrator()
        metrics = RoutingMetrics()
        
        return {
            "success": True,
            "content": f"""**Current LLM Strategy**

Profile: {orchestrator.current_strategy.profile or 'default'}
Strategy: {orchestrator.current_strategy.strategy.value}
Cloud Fallback: {orchestrator.current_strategy.enable_cloud_fallback}

**Recent Metrics**
{metrics.format_summary()}
"""
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## 7. Phase 2: Dynamic Pipeline Resources (P2)

### 7.1 Current vs Enhanced Resources

| Current | Enhanced |
|---------|----------|
| `biopipelines://skills` (static) | `biopipelines://pipelines` (dynamic from YAML) |
| `biopipelines://templates` (static) | `biopipelines://pipelines/{id}` (per-pipeline) |
| `biopipelines://databases` (static) | `biopipelines://jobs/{id}` (live job status) |

### 7.2 Implementation

```python
def _setup_resources(self):
    """Register MCP resources."""
    
    # Existing static resources...
    
    # NEW: Dynamic pipeline list
    self._register_resource(
        uri="biopipelines://pipelines",
        name="Available Pipelines",
        description="List of all BioPipelines workflows with descriptions",
        handler=self._handle_get_pipelines
    )

async def read_resource(self, uri: str) -> str:
    """Read a resource by URI with pattern matching."""
    
    # Handle static resources
    if uri in self.resources:
        return await self.resources[uri].handler()
    
    # Handle dynamic URI patterns
    if uri.startswith("biopipelines://pipelines/"):
        pipeline_id = uri.split("/")[-1]
        return await self._handle_get_pipeline_doc(pipeline_id)
    
    if uri.startswith("biopipelines://jobs/"):
        job_id = uri.split("/")[-1]
        return await self._handle_get_job_resource(job_id)
    
    return f"Unknown resource: {uri}"

async def _handle_get_pipelines(self) -> str:
    """Generate pipeline list from analysis_definitions.yaml."""
    import yaml
    
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "analysis_definitions.yaml"
    
    with open(config_path) as f:
        definitions = yaml.safe_load(f)
    
    text = "# Available BioPipelines Workflows\n\n"
    
    categories = {
        "RNA Analysis": ["rna_seq_basic", "rna_seq_differential_expression", "single_cell_rna_seq"],
        "DNA/Variant": ["wgs_variant_calling", "somatic_variant_calling", "structural_variant_detection"],
        "Epigenetics": ["chip_seq_peak_calling", "atac_seq", "bisulfite_seq_methylation"],
        "Other": []  # Catch-all
    }
    
    for cat_name, pipeline_ids in categories.items():
        text += f"## {cat_name}\n\n"
        for pid in pipeline_ids:
            if pid in definitions:
                defn = definitions[pid]
                required = ", ".join(defn.get("required", [])[:3])
                text += f"- **{pid}**: Requires {required}\n"
        text += "\n"
    
    return text
```

---

## 8. Phase 3: Resource Templates (P3)

### 8.1 MCP Resource Templates

MCP supports URI templates (RFC 6570) for parameterized resources:

```python
async def _handle_resources_templates_list(self) -> Dict:
    """Return resource templates for dynamic resources."""
    return {
        "resourceTemplates": [
            {
                "uriTemplate": "biopipelines://pipelines/{pipeline_id}",
                "name": "Pipeline Documentation",
                "title": "📋 Pipeline Details",
                "description": "Detailed documentation for a specific pipeline",
                "mimeType": "text/markdown"
            },
            {
                "uriTemplate": "biopipelines://jobs/{job_id}",
                "name": "Job Status",
                "title": "📊 Job Information", 
                "description": "Status and logs for a SLURM job",
                "mimeType": "text/markdown"
            },
            {
                "uriTemplate": "biopipelines://databases/{db_name}",
                "name": "Database Info",
                "title": "🗄️ Database Details",
                "description": "Connection info and schema for a database",
                "mimeType": "text/markdown"
            }
        ]
    }
```

---

## 9. Deferred: High-Level Agent Tools (P3)

ChatGPT suggests tools like:
- `design_pipeline_from_question`
- `refine_pipeline`
- `debug_failed_run`

### 9.1 Why Defer?

1. **Complexity**: These require multi-step agent orchestration
2. **LLM Dependency**: Already handled by the LLM layer above MCP
3. **Overlap**: Claude/ChatGPT already does this reasoning with our tools
4. **Maintenance**: High surface area to test and maintain

### 9.2 Alternative Approach

Instead of embedding agents in MCP, keep MCP as **primitive tools** and let the external LLM (Claude, ChatGPT) orchestrate them:

```
User → Claude Desktop → MCP Tools → BioPipelines
         (reasoning)      (execution)
```

This is cleaner than:

```
User → Claude Desktop → MCP Agent Tool → Internal LLM → BioPipelines
```

### 9.3 Exception: Debug Tool

One high-level tool that **does** make sense:

```python
self._register_tool(
    name="diagnose_job_failure",
    description="Analyze a failed job and suggest fixes.",
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "required": True},
            "include_logs": {"type": "boolean", "default": True}
        }
    },
    handler=self._handle_diagnose_failure
)
```

This doesn't require another LLM—it's pattern matching on known error types from our error catalog.

---

## 10. Deferred: Full PipelineRegistry (Skip)

### 10.1 ChatGPT's Suggestion

```python
@dataclass
class PipelineSpec:
    id: str
    display_name: str
    category: str
    engine: str
    tags: list[str]
    input_schema: dict
    output_schema: dict
    template_name: str
```

### 10.2 Why Skip Now

| Concern | Reality |
|---------|---------|
| "50-100 workflows" | We have ~17 |
| "Avoid editing MCP per workflow" | We're not doing that anyway |
| "Registry auto-discovery" | YAML files + templates already work |
| "Maintenance overhead" | Registry layer adds code to maintain |

### 10.3 When to Revisit

Implement `PipelineRegistry` when:
- Pipeline count exceeds 30
- Multiple teams are adding pipelines
- Need versioning/deprecation tracking
- Want API generation from specs

---

## 11. Implementation Roadmap

### Phase 1 (2-3 weeks)

| Task | Effort | Files |
|------|--------|-------|
| Add `submit_job` tool with dry-run | 1d | `server.py` |
| Add `list_jobs` tool | 0.5d | `server.py` |
| Add `cancel_job` tool | 0.5d | `server.py` |
| Add `get_job_logs` tool | 0.5d | `server.py` |
| Add tool annotations support | 0.5d | `server.py` |
| Add output schemas to 5 tools | 1d | `server.py` |
| Tests for new tools | 1d | `test_mcp_server.py` |
| Update documentation | 0.5d | `MCP_INTEGRATION.md` |

### Phase 2 (2-3 weeks)

| Task | Effort | Files |
|------|--------|-------|
| Add MCP Prompts support | 1d | `server.py` |
| Implement 3 prompts | 1d | `server.py` |
| Add strategy tools | 0.5d | `server.py` |
| Dynamic pipeline resources | 1d | `server.py` |
| Resource URI pattern matching | 0.5d | `server.py` |
| Tests and documentation | 1d | Various |

### Phase 3 (Optional, 1-2 weeks)

| Task | Effort | Files |
|------|--------|-------|
| Resource templates | 1d | `server.py` |
| `diagnose_job_failure` tool | 1d | `server.py` |
| Integration tests | 1d | `test_mcp_server.py` |

---

## 12. Security Checklist

Per MCP specification security requirements:

- [ ] Validate all tool inputs
- [ ] Implement workspace sandboxing (`/scratch/$USER/biopipelines/`)
- [ ] Rate limit tool invocations (future)
- [ ] Sanitize tool outputs
- [ ] Add confirmation requirements for destructive operations
- [ ] Log tool usage for audit (integrate with existing logging)
- [ ] Validate resource URIs before reading

---

## 13. Testing Strategy

### 13.1 Unit Tests (Existing + New)

```python
class TestMCPJobTools:
    """Tests for job management tools."""
    
    @pytest.mark.asyncio
    async def test_submit_job_dry_run(self, server):
        """Test submit with dry_run=True."""
        result = await server.call_tool("submit_job", {
            "workflow_path": "/path/to/workflow.nf",
            "dry_run": True
        })
        assert result["success"]
        assert "sbatch" in result["content"]
    
    @pytest.mark.asyncio  
    async def test_list_jobs(self, server):
        """Test listing user jobs."""
        result = await server.call_tool("list_jobs", {"state": "all"})
        assert result["success"]
```

### 13.2 Integration Tests

```python
class TestMCPEndToEnd:
    """End-to-end MCP tests."""
    
    @pytest.mark.integration
    async def test_workflow_creation_to_submission(self, server):
        """Test full workflow: create → submit → monitor."""
        # Create workflow
        create_result = await server.call_tool("create_workflow", {
            "analysis_type": "rnaseq",
            "output_dir": "/tmp/test_workflow"
        })
        assert create_result["success"]
        
        # Submit (dry run)
        submit_result = await server.call_tool("submit_job", {
            "workflow_path": "/tmp/test_workflow/main.nf",
            "dry_run": True
        })
        assert submit_result["success"]
```

---

## 14. Compatibility Notes

### 14.1 MCP Protocol Version

Current: `2024-11-05`  
Latest: `2025-06-18`

New features in 2025-06-18 we should adopt:
- ✅ Tool annotations
- ✅ Output schemas
- ✅ Resource annotations
- ⏳ Resource templates
- ⏳ Structured content

### 14.2 Client Compatibility

| Client | Status | Notes |
|--------|--------|-------|
| Claude Code | ✅ | Primary target |
| Claude Desktop | ✅ | stdio transport |
| Cursor | ✅ | Via mcp-config.json |
| ChatGPT Agents | 🔄 | MCP adoption in progress |
| GitHub Copilot | 🔄 | MCP registry support coming |

---

## 15. Summary

### What ChatGPT Got Right ✅

1. **MCP as canonical API** - Valid long-term architecture
2. **Avoid tool explosion** - Use parameterized tools
3. **Separate plan/submit/monitor** - Good separation of concerns
4. **Security considerations** - Human-in-the-loop critical for HPC
5. **Progressive discovery** - List → Describe → Run pattern

### Where We Diverge 🔀

1. **PipelineRegistry** - Overkill for 10-20 pipelines
2. **High-level agent tools** - Let external LLMs orchestrate
3. **Complexity timeline** - Start simple, scale later

### Immediate Actions (This Sprint)

1. **Add 4 job management tools** (submit, list, cancel, logs)
2. **Add tool annotations** for confirmation requirements
3. **Add output schemas** to top 5 tools
4. **Update tests** (target: 30+ tests)

### Future Considerations

- Monitor pipeline count; implement registry at 30+
- Add MCP Prompts when workflow patterns stabilize
- Evaluate resource subscriptions for live job monitoring
- Consider SSE/WebSocket for streaming (Phase 4+)

---

## Appendix A: Tool Comparison

### Current vs Proposed

| Tool | Current | Proposed |
|------|---------|----------|
| `search_encode` | ✅ | + outputSchema |
| `search_geo` | ✅ | + outputSchema |
| `create_workflow` | ✅ | + outputSchema |
| `use_workflow_template` | ✅ | (unchanged) |
| `search_uniprot` | ✅ | + outputSchema |
| `get_protein_interactions` | ✅ | + outputSchema |
| `get_functional_enrichment` | ✅ | (unchanged) |
| `search_kegg_pathways` | ✅ | (unchanged) |
| `search_pubmed` | ✅ | (unchanged) |
| `search_variants` | ✅ | (unchanged) |
| `explain_concept` | ✅ | (unchanged) |
| `check_job_status` | ✅ | (unchanged) |
| `submit_job` | ❌ | **NEW** |
| `list_jobs` | ❌ | **NEW** |
| `cancel_job` | ❌ | **NEW** |
| `get_job_logs` | ❌ | **NEW** |
| `get_llm_strategy` | ❌ | **NEW** (P2) |
| `set_llm_strategy` | ❌ | **NEW** (P2) |

### Proposed Total: 18 tools (from 12)

---

## Appendix B: References

1. [MCP Specification 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18/)
2. [MCP Tools Documentation](https://modelcontextprotocol.io/docs/concepts/tools)
3. [MCP Resources Documentation](https://modelcontextprotocol.io/docs/concepts/resources)
4. [MCP Prompts Documentation](https://modelcontextprotocol.io/docs/concepts/prompts)
5. [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)
6. ChatGPT Analysis (December 2025)
7. Internal: `src/workflow_composer/mcp/server.py`
8. Internal: `docs/MCP_INTEGRATION.md`
