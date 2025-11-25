# Dynamic Pipeline Requirements & Architecture Reset

**Date**: November 24, 2025  
**Purpose**: Define what's actually needed for AI-driven dynamic pipeline generation  
**Status**: Strategic Reset - Use What Works, Fix What's Broken

---

## User Requirements (What People Actually Want)

### Scenario 1: Graduate Student
**Request**: "I want to analyze my RNA-seq data with STAR and get gene counts"

**System Needs**:
1. Identify tools: STAR (alignment), featureCounts (quantification)
2. Find/create workflow using these tools
3. Run it with their data
4. Return results in <30 minutes

### Scenario 2: Postdoc Comparison Study
**Request**: "Compare STAR vs HISAT2 alignment, then quantify with both featureCounts and Salmon"

**System Needs**:
1. Compose workflow: (STAR → featureCounts) + (STAR → Salmon) + (HISAT2 → featureCounts) + (HISAT2 → Salmon)
2. Run 4 parallel branches
3. Generate comparison report
4. Allow iteration on parameters

### Scenario 3: PI with Custom Script
**Request**: "Use BWA, call variants with GATK, then run my custom R filtering script"

**System Needs**:
1. Standard tools: BWA, GATK (from containers)
2. Custom integration: User's R script
3. Automatic containerization of custom script
4. Reproducible execution

### Scenario 4: Method Developer
**Request**: "I'm developing a new peak caller - test it against MACS2 on 10 datasets"

**System Needs**:
1. User tool integration (not in any container)
2. Benchmark against standard (MACS2)
3. Statistical comparison
4. Visualization of differences

---

## What We Actually Have (Current Assets)

### ✅ Working Infrastructure

**1. Nextflow Platform** (Proven in Phase 1)
- 7 concurrent workflows tested successfully
- DSL2 module architecture established
- SLURM integration working
- Cloud-ready (GCP compatibility)

**2. Existing Containers** (Already Built, Already Working)
```
12 containers × 1.4-2.1 GB = ~20 GB total
Each contains 500-700 conda-installed tools

rna-seq_1.0.0.sif:
├── STAR 2.7.11a (alignment)
├── HISAT2 2.2.1 (alternative alignment)
├── Salmon 1.10.3 (quantification)
├── featureCounts 2.0.6 (quantification)
├── HTSeq (quantification)
├── DESeq2 (differential expression)
├── EdgeR (differential expression)
└── 636 other tools...

Similar comprehensive coverage for:
- DNA-seq (BWA, GATK, FreeBayes, bcftools, ...)
- ChIP-seq (Bowtie2, MACS2, HOMER, deepTools, ...)
- ATAC-seq, scRNA-seq, Hi-C, etc.
```

**3. Module Library Structure**
- `/nextflow-pipelines/modules/` - Process definitions
- `/nextflow-pipelines/workflows/` - Complete pipelines
- Configuration already points to existing containers

### ❌ What's Broken

**1. Pipeline Testing** (Fixed but not validated)
- Test script infrastructure works
- Haven't successfully run end-to-end test with existing containers
- **Need**: One successful RNA-seq run to prove the concept

**2. Dynamic Composition** (Not Built Yet)
- Can't programmatically generate workflows
- No AI agent to interpret user requests
- No tool discovery/selection system
- **Need**: This is the actual Phase 3 work

**3. Custom Tool Integration** (Not Addressed)
- No way to add user's own scripts
- No on-demand containerization
- **Need**: Tier 3 containers (but simpler than we designed)

---

## Revised Architecture: Pragmatic Approach

### Layer 1: Use Existing Containers (NOW)

**Strategy**: Stop trying to rebuild containers. Use what works.

```
Existing containers cover 95% of common tools:
✓ rna-seq_1.0.0.sif → RNA-seq workflows
✓ dna-seq_1.0.0.sif → Variant calling workflows  
✓ chipseq_1.0.0.sif → ChIP/ATAC workflows
✓ scrna-seq_1.0.0.sif → Single-cell workflows
... etc

Each is 1.4-2.1 GB (reasonable size)
Each has 500-700 tools (comprehensive)
Each is conda-based (reliable, tested)
All already built and deployed
```

**Action Items**:
1. ✅ Verify containers work (test one tool from each)
2. ✅ Run one complete RNA-seq workflow end-to-end
3. ✅ Document what tools are in each container
4. ✅ Create tool → container mapping

### Layer 2: Nextflow Module Library (EXPAND)

**Strategy**: Build library of reusable processes, one per tool

```nextflow
// Example: modules/alignment/star.nf
process STAR_ALIGN {
    container params.containers.rnaseq  // Uses existing rna-seq_1.0.0.sif
    
    input:
    tuple val(meta), path(reads)
    path(index)
    
    output:
    tuple val(meta), path('*.bam'), emit: bam
    
    script:
    """
    STAR --genomeDir ${index} \\
         --readFilesIn ${reads} \\
         --runThreadN ${task.cpus} \\
         --outSAMtype BAM SortedByCoordinate
    """
}

// Example: modules/quantification/salmon.nf
process SALMON_QUANT {
    container params.containers.rnaseq  // Same container, different tool
    
    input:
    tuple val(meta), path(reads)
    path(index)
    
    output:
    tuple val(meta), path('quant'), emit: results
    
    script:
    """
    salmon quant -i ${index} \\
                 -l A \\
                 -r ${reads} \\
                 -o quant
    """
}
```

**Benefits**:
- One module per tool (fine-grained)
- Multiple modules can use same container (efficient)
- AI can compose workflows at tool level
- Easy to add new tools (just new .nf files)

**Action Items**:
1. Create 20-30 core tool modules (alignment, QC, quantification)
2. Document each module's inputs/outputs
3. Test each module individually
4. Build example workflows by composition

### Layer 3: AI Workflow Composer (BUILD)

**Strategy**: AI translates user request → Nextflow workflow

```python
class WorkflowComposer:
    def __init__(self):
        self.tool_catalog = ToolCatalog()  # Maps tools to modules
        self.container_map = ContainerMap()  # Maps modules to containers
    
    def compose_from_request(self, user_request: str) -> str:
        """Generate Nextflow workflow from natural language"""
        
        # Step 1: Extract intent
        intent = self.parse_intent(user_request)
        # Example: {"task": "rnaseq", "tools": ["STAR", "featureCounts"], 
        #           "data": "/path/to/fastq"}
        
        # Step 2: Find modules
        modules = []
        for tool in intent['tools']:
            module = self.tool_catalog.find_module(tool)
            modules.append(module)
        
        # Step 3: Compose workflow
        workflow_nf = self.generate_nextflow(modules, intent)
        
        # Step 4: Return executable workflow
        return workflow_nf
    
    def generate_nextflow(self, modules, intent):
        """Generate Nextflow DSL2 workflow code"""
        template = """
        #!/usr/bin/env nextflow
        nextflow.enable.dsl=2
        
        {module_imports}
        
        workflow {{
            {workflow_logic}
        }}
        """
        
        imports = [f"include {{ {m.name} }} from '{m.path}'" for m in modules]
        logic = self.compose_workflow_logic(modules, intent)
        
        return template.format(
            module_imports="\n".join(imports),
            workflow_logic=logic
        )
```

**Example Conversation**:
```
User: "Analyze my RNA-seq with STAR and featureCounts"

AI: [Parses request]
    - Task: RNA-seq analysis
    - Tools: STAR (alignment), featureCounts (quantification)
    - Needs: FASTQ input, reference genome, GTF annotation

AI: [Generates workflow]
    ```nextflow
    include { STAR_ALIGN } from './modules/alignment/star'
    include { FEATURECOUNTS } from './modules/quantification/featurecounts'
    
    workflow {
        reads_ch = Channel.fromFilePairs(params.reads)
        STAR_ALIGN(reads_ch, params.star_index)
        FEATURECOUNTS(STAR_ALIGN.out.bam, params.gtf)
    }
    ```

AI: "I'll align your reads with STAR and quantify with featureCounts. 
     Please provide: FASTQ files, STAR index, and GTF file."

User: [Provides paths]

AI: [Executes workflow]
    "Running analysis... estimated time: 20 minutes"
```

**Action Items**:
1. Build tool catalog (tool name → module path → container)
2. Create intent parser (NLP or structured prompts)
3. Implement workflow generator (templates + composition)
4. Test with 5-10 common use cases

### Layer 4: Custom Tool Integration (OPTIONAL)

**Strategy**: Simple overlay approach for user scripts

```python
class CustomToolIntegrator:
    def add_user_script(self, script_path: str, dependencies: list):
        """Add user's custom script to a workflow"""
        
        # Option 1: Overlay on existing container (FAST - 30 seconds)
        if self.dependencies_available_in_base(dependencies):
            overlay = self.create_overlay(script_path, dependencies)
            return overlay  # Just mounts user script into container
        
        # Option 2: Extend existing container (MEDIUM - 2-5 minutes)
        elif self.can_pip_install(dependencies):
            extended_container = self.extend_with_pip(
                base_container='rna-seq_1.0.0.sif',
                script=script_path,
                pip_packages=dependencies
            )
            return extended_container
        
        # Option 3: Build custom container (SLOW - 10-30 minutes)
        else:
            custom_container = self.build_custom_container(
                script_path, 
                dependencies
            )
            return custom_container
```

**Action Items** (Future):
1. Implement overlay mounts (OverlayFS or --bind)
2. Create pip extension templates
3. Custom container builder (last resort)

---

## Immediate Action Plan (Next 4 Hours)

### Hour 1: Validate Existing Infrastructure

**Task 1.1: Test Containers** (15 min)
```bash
# Test each container has expected tools
for container in containers/images/*.sif; do
    name=$(basename $container .sif)
    echo "Testing $name..."
    singularity exec $container bash -c "conda list | wc -l"
done
```

**Task 1.2: Run RNA-seq Workflow** (30 min)
```bash
# Use existing RNA-seq workflow with existing container
cd nextflow-pipelines
nextflow run workflows/rnaseq_simple.nf \\
    --reads '/scratch/.../mut_rep1_R*.fastq.gz' \\
    --star_index '/scratch/.../star_index_hg38' \\
    --gtf '/scratch/.../genes_GRCh38.gtf'
```

**Task 1.3: Document Success** (15 min)
- Record runtime, outputs, any issues
- Prove: Nextflow + existing containers = WORKS

### Hour 2: Create Tool Catalog

**Task 2.1: Inventory Tools** (20 min)
```bash
# List all tools in each container
for container in containers/images/*-seq_*.sif; do
    name=$(basename $container .sif)
    singularity exec $container bash -c "ls /opt/conda/bin/" > tools_${name}.txt
done
```

**Task 2.2: Build Tool → Container Map** (30 min)
```json
{
    "STAR": {
        "container": "rna-seq_1.0.0.sif",
        "version": "2.7.11a",
        "module": "modules/alignment/star.nf",
        "category": "alignment"
    },
    "featureCounts": {
        "container": "rna-seq_1.0.0.sif",
        "version": "2.0.6",
        "module": "modules/quantification/featurecounts.nf",
        "category": "quantification"
    },
    ...
}
```

**Task 2.3: Document Workflows** (10 min)
- What tools each workflow uses
- What containers it requires
- Example use cases

### Hour 3: Expand Module Library

**Task 3.1: Create Core Modules** (40 min)
```
Priority modules (tools users request most):
✓ alignment/star.nf
✓ alignment/hisat2.nf
✓ alignment/bwa.nf
✓ quantification/featurecounts.nf
✓ quantification/salmon.nf
✓ qc/fastqc.nf
✓ qc/multiqc.nf
✓ peaks/macs2.nf
```

**Task 3.2: Test Each Module** (20 min)
```bash
# Test module individually
nextflow run test_module.nf \\
    --module alignment/star \\
    --test_data test/data/sample_R1.fq.gz
```

### Hour 4: Document & Plan AI Integration

**Task 4.1: Create User Guide** (20 min)
```markdown
# How to Request Custom Workflows

## Example 1: RNA-seq Analysis
"I want to align my RNA-seq reads with STAR and quantify with featureCounts"
→ System generates workflow using modules/alignment/star.nf + modules/quantification/featurecounts.nf

## Example 2: Method Comparison
"Compare STAR vs HISAT2 for alignment, then use Salmon for quantification"
→ System generates parallel workflow with both aligners + Salmon

## Example 3: Custom Script
"After BWA alignment, run my custom SNP filter (filter_snps.py requires pandas)"
→ System creates extended container with pandas, mounts your script
```

**Task 4.2: Design AI Composer** (30 min)
```python
# Pseudocode for AI workflow composer
class WorkflowComposer:
    def compose(self, user_request):
        intent = self.parse_intent(user_request)
        modules = self.select_modules(intent)
        workflow = self.generate_workflow(modules)
        return workflow
```

**Task 4.3: Roadmap** (10 min)
- Week 2: Complete tool catalog + 30 modules
- Week 3: Build AI composer prototype
- Week 4: User testing + iteration
- Week 5+: Custom tool integration

---

## Success Metrics (Realistic)

**Week 1 (This Week)**:
- ✓ 1 successful RNA-seq workflow with existing containers
- ✓ Tool catalog complete (500+ tools mapped)
- ✓ 10-15 core modules created and tested
- ✓ Documentation for manual workflow composition

**Week 2-3**:
- ✓ 30-50 modules covering common tools
- ✓ 5-10 example workflows by composition
- ✓ AI composer prototype (intent → workflow)
- ✓ 3-5 user test cases

**Week 4+**:
- ✓ AI-driven workflow generation working
- ✓ Custom script integration (overlays)
- ✓ 10+ users successfully creating custom workflows
- ✓ Documentation and tutorials

---

## Key Insights

### What We Learned

1. **Don't rebuild what works** - 12 existing containers with 500-700 tools each
2. **Granularity at module level, not container level** - Many modules can share one container
3. **Conda is the right choice** - Already used in existing containers, proven reliable
4. **Nextflow proven** - 7 concurrent workflows validated in Phase 1
5. **Focus on composition, not construction** - Building containers was a distraction

### What We're Discarding

- ❌ Tier 2 "domain modules" (over-engineered)
- ❌ Source compilation containers (unreliable)
- ❌ Building new containers from scratch (unnecessary)
- ❌ Complex multi-tier architecture before proof-of-concept

### What We're Keeping

- ✅ Nextflow DSL2 platform
- ✅ Existing 12 containers
- ✅ Module library concept (but simpler)
- ✅ AI workflow composition vision (Phase 3)

---

## Conclusion

**The path forward is clear**:

1. **Validate** existing infrastructure works (1 successful workflow)
2. **Document** what tools exist and where (tool catalog)
3. **Expand** module library for common tools (30-50 modules)
4. **Build** AI composer to translate requests → workflows
5. **Integrate** custom tools as overlays (future enhancement)

**Stop trying to build perfect containers**. Start enabling users to compose workflows dynamically using what already exists.

The goal isn't perfect architecture - it's **empowering users to run any analysis they need, without knowing Nextflow or bioinformatics tooling**.

Next step: Run one successful RNA-seq workflow. Everything builds from there.
