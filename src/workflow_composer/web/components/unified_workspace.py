"""
Unified Workspace Component
===========================

A unified workspace that combines:
- AI Chat for workflow generation
- Data discovery (as AI tools + sidebar)
- Job execution and monitoring (as AI tools + sidebar)

Layout:
┌─────────────────────────────────────────────────────────────────────────────┐
│  UNIFIED WORKSPACE                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────┐  ┌────────────────────────────┐  │
│  │  CHAT AREA (70%)                      │  │  CONTEXT SIDEBAR (30%)     │  │
│  │  - Stats dashboard                    │  │  - LLM Provider            │  │
│  │  - Chatbot                            │  │  - Data Manifest           │  │
│  │  - Message input                      │  │  - Active Jobs             │  │
│  │  - Example prompts                    │  │  - Quick Actions           │  │
│  └───────────────────────────────────────┘  └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
"""

import gradio as gr
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_stats() -> Tuple[str, str, str, str]:
    """Get stats for the dashboard."""
    try:
        from workflow_composer.config import Config
        config = Config()
        catalog = config.load_tool_catalog()
        tools = catalog.get("tools", [])
        modules = catalog.get("modules", [])
        containers = 10  # Approximate
        analyses = 15
        return (
            f"📊 **{len(tools):,}** Tools",
            f"📦 **{len(modules)}** Modules",
            f"🐳 **{containers}** Containers",
            f"🧬 **{analyses}** Analysis Types"
        )
    except Exception:
        return (
            "📊 **9,909** Tools",
            "📦 **71** Modules",
            "🐳 **10** Containers",
            "🧬 **15** Analysis Types"
        )


def get_provider_choices() -> List[str]:
    """Get available LLM provider choices."""
    providers = []
    
    try:
        import os
        if os.environ.get("OPENAI_API_KEY"):
            providers.extend([
                "🤖 OpenAI GPT-4o",
                "🤖 OpenAI GPT-4o-mini",
            ])
        if os.environ.get("ANTHROPIC_API_KEY"):
            providers.extend([
                "🧠 Anthropic Claude 3.5",
                "🧠 Anthropic Claude 3 Haiku",
            ])
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            providers.append("💎 Google Gemini Pro")
    except Exception:
        pass
    
    # Always add local options
    providers.extend([
        "🦙 Local Ollama (llama3)",
        "🏠 Local LM Studio",
    ])
    
    return providers if providers else ["🦙 Local Ollama (llama3)"]


def get_available_workflows() -> List[str]:
    """Get list of available workflows."""
    workflows_dir = Path.home() / "BioPipelines" / "generated_workflows"
    if not workflows_dir.exists():
        return []
    
    return sorted(
        [d.name for d in workflows_dir.iterdir() if d.is_dir()],
        reverse=True
    )[:10]


def get_example_prompts() -> List[str]:
    """Get example prompts for the chat."""
    return [
        "Analyze RNA-seq data for differential expression between treatment and control",
        "Create a ChIP-seq pipeline for human H3K27ac data",
        "Process ATAC-seq samples to find open chromatin regions",
        "Build a variant calling pipeline for whole genome sequencing",
        "Analyze single-cell RNA-seq with Seurat clustering",
        "Create a metagenomics profiling workflow with Kraken2",
    ]


def get_manifest_summary() -> Dict[str, str]:
    """Get current data manifest summary."""
    try:
        from workflow_composer.web.components.data_tab import _data_state
        if _data_state.manifest:
            manifest = _data_state.manifest
            return {
                "sample_count": f"**{len(manifest.samples)}** samples",
                "paired_count": f"**{sum(1 for s in manifest.samples if s.library_layout.value == 'paired')}** paired",
                "organisms": ", ".join(set(s.metadata.get('organism', 'unknown') for s in manifest.samples)) or "Not set",
                "reference": manifest.reference.genome_build if manifest.reference else "Not configured"
            }
    except Exception:
        pass
    
    return {
        "sample_count": "**0** samples",
        "paired_count": "**0** paired",
        "organisms": "Not set",
        "reference": "Not configured"
    }


def get_active_jobs_html() -> str:
    """Generate HTML for active jobs panel."""
    try:
        from workflow_composer.web.gradio_app import PipelineExecutor
        # This would integrate with the actual executor
        # For now, return a placeholder
        return """
        <div style="font-size: 0.9em; color: #666;">
            <em>No active jobs</em>
        </div>
        """
    except Exception:
        return """
        <div style="font-size: 0.9em; color: #666;">
            <em>No active jobs</em>
        </div>
        """


def create_unified_workspace() -> Dict[str, Any]:
    """
    Create the unified workspace with chat and context sidebar.
    
    Returns:
        Dict of component references for event wiring
    """
    components = {}
    stats = get_stats()
    
    with gr.Row():
        # ===== MAIN CHAT AREA (70%) =====
        with gr.Column(scale=7):
            # Stats dashboard
            with gr.Row():
                with gr.Column(scale=1):
                    components["tools_stat"] = gr.Markdown(stats[0])
                with gr.Column(scale=1):
                    components["modules_stat"] = gr.Markdown(stats[1])
                with gr.Column(scale=1):
                    components["containers_stat"] = gr.Markdown(stats[2])
                with gr.Column(scale=1):
                    components["analyses_stat"] = gr.Markdown(stats[3])
            
            # Chatbot
            components["chatbot"] = gr.Chatbot(
                label="BioPipelines AI Assistant",
                height=550,
                show_label=False,
                type="messages",
            )
            
            # Message input area
            with gr.Row():
                components["msg_input"] = gr.Textbox(
                    label="Your message",
                    placeholder="Describe your analysis, scan data, or ask questions... (e.g., 'Scan data in /data/project1' or 'Create RNA-seq pipeline')",
                    lines=2,
                    scale=5,
                    show_label=False,
                )
                components["send_btn"] = gr.Button(
                    "Send 🚀",
                    variant="primary",
                    scale=1,
                    size="lg"
                )
            
            # Example prompts
            with gr.Accordion("📝 Example Prompts", open=False):
                gr.Examples(
                    examples=get_example_prompts(),
                    inputs=components["msg_input"],
                    label="Click an example to use it:",
                )
        
        # ===== CONTEXT SIDEBAR (30%) =====
        with gr.Column(scale=3):
            # LLM Provider selector
            gr.Markdown("### 🤖 AI Model")
            components["provider_dropdown"] = gr.Dropdown(
                choices=get_provider_choices(),
                value=get_provider_choices()[0] if get_provider_choices() else None,
                label="Model",
                interactive=True,
                show_label=False,
            )
            
            gr.Markdown("---")
            
            # Data Manifest Panel
            with gr.Accordion("📁 Data Manifest", open=True):
                manifest_summary = get_manifest_summary()
                
                with gr.Row():
                    components["manifest_samples"] = gr.Markdown(manifest_summary["sample_count"])
                    components["manifest_paired"] = gr.Markdown(manifest_summary["paired_count"])
                
                components["manifest_organisms"] = gr.Markdown(f"🧬 {manifest_summary['organisms']}")
                components["manifest_reference"] = gr.Markdown(f"📚 {manifest_summary['reference']}")
                
                with gr.Row():
                    components["scan_data_btn"] = gr.Button("📂 Scan", size="sm", scale=1)
                    components["search_db_btn"] = gr.Button("🔍 Search", size="sm", scale=1)
                    components["refresh_manifest_btn"] = gr.Button("🔄", size="sm", scale=1)
            
            gr.Markdown("---")
            
            # Active Jobs Panel
            gr.Markdown("### 🚀 Active Jobs")
            components["jobs_html"] = gr.HTML(
                value=get_active_jobs_html()
            )
            
            with gr.Row():
                components["refresh_jobs_btn"] = gr.Button("🔄", size="sm", scale=1)
                components["cancel_job_btn"] = gr.Button("🛑", size="sm", scale=1)
                components["view_logs_btn"] = gr.Button("📄", size="sm", scale=1)
            
            components["job_selector"] = gr.Dropdown(
                choices=[],
                label="Select Job",
                interactive=True,
                show_label=False,
                visible=False,  # Hidden until there are jobs
            )
            
            # Auto-refresh timer for jobs (every 15 seconds)
            components["job_timer"] = gr.Timer(15, active=True)
            
            gr.Markdown("---")
            
            # Recent Workflows
            with gr.Accordion("📋 Recent Workflows", open=False):
                workflows = get_available_workflows()
                components["recent_workflows"] = gr.Markdown(
                    "\n".join([f"- `{w}`" for w in workflows[:5]])
                    if workflows else "*No workflows yet*"
                )
                
                components["workflow_dropdown"] = gr.Dropdown(
                    choices=workflows,
                    label="Select Workflow",
                    interactive=True,
                    show_label=False,
                )
                
                with gr.Row():
                    components["run_workflow_btn"] = gr.Button("▶️ Run", size="sm", variant="primary", scale=1)
                    components["view_workflow_btn"] = gr.Button("👁️ View", size="sm", scale=1)
            
            gr.Markdown("---")
            
            # Quick Actions
            gr.Markdown("### ⚡ Quick Actions")
            with gr.Row():
                components["clear_chat_btn"] = gr.Button("🗑️ Clear", size="sm", scale=1)
                components["goto_results_btn"] = gr.Button("📊 Results", size="sm", scale=1)
            
            # Tips
            gr.Markdown("""
---
### 💡 Tips
- Say **"scan data in /path"** to find files
- Say **"run it on SLURM"** after generating
- Say **"show logs"** to view output
- Say **"diagnose"** if something fails
""")
    
    return components


def create_scan_dialog() -> Tuple[Any, Dict[str, Any]]:
    """
    Create a modal dialog for data scanning.
    
    Returns:
        Tuple of (dialog component, component dict)
    """
    components = {}
    
    with gr.Column(visible=False) as dialog:
        gr.Markdown("### 📂 Scan Directory for Data")
        
        components["scan_path"] = gr.Textbox(
            label="Directory Path",
            placeholder="/path/to/your/data",
            value=str(Path.home() / "data"),
        )
        
        with gr.Row():
            components["recursive"] = gr.Checkbox(label="Recursive", value=True)
            components["pattern"] = gr.Textbox(
                label="Pattern",
                value="*.fastq.gz,*.fq.gz",
                scale=2
            )
        
        with gr.Row():
            components["scan_confirm_btn"] = gr.Button("Scan", variant="primary")
            components["scan_cancel_btn"] = gr.Button("Cancel")
        
        components["scan_status"] = gr.Markdown("")
    
    return dialog, components


def create_search_dialog() -> Tuple[Any, Dict[str, Any]]:
    """
    Create a modal dialog for database searching.
    
    Returns:
        Tuple of (dialog component, component dict)
    """
    components = {}
    
    with gr.Column(visible=False) as dialog:
        gr.Markdown("### 🔍 Search Remote Databases")
        
        components["search_query"] = gr.Textbox(
            label="Search Query",
            placeholder="e.g., human RNA-seq liver cancer",
        )
        
        with gr.Row():
            components["search_encode"] = gr.Checkbox(label="ENCODE", value=True)
            components["search_geo"] = gr.Checkbox(label="GEO", value=True)
            components["search_ensembl"] = gr.Checkbox(label="Ensembl", value=False)
        
        with gr.Row():
            components["search_confirm_btn"] = gr.Button("Search", variant="primary")
            components["search_cancel_btn"] = gr.Button("Cancel")
        
        components["search_status"] = gr.Markdown("")
        components["search_results"] = gr.Dataframe(
            headers=["Source", "ID", "Title"],
            visible=False
        )
    
    return dialog, components


# Helper functions for event handlers

def refresh_manifest_panel() -> Tuple[str, str, str, str]:
    """Refresh the manifest summary panel."""
    summary = get_manifest_summary()
    return (
        summary["sample_count"],
        summary["paired_count"],
        f"🧬 {summary['organisms']}",
        f"📚 {summary['reference']}"
    )


def refresh_jobs_panel() -> str:
    """Refresh the active jobs panel."""
    return get_active_jobs_html()


def refresh_workflows_list() -> Tuple[List[str], str]:
    """Refresh the workflows list."""
    workflows = get_available_workflows()
    markdown = "\n".join([f"- `{w}`" for w in workflows[:5]]) if workflows else "*No workflows yet*"
    return workflows, markdown


def insert_scan_message(current_input: str) -> str:
    """Insert a scan command into the chat input."""
    return "Scan data in /path/to/your/data"


def insert_search_message(current_input: str) -> str:
    """Insert a search command into the chat input."""
    return "Search for human RNA-seq data"
