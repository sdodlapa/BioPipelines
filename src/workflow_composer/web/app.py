"""
BioPipelines - Chat-First Web Interface (Gradio 6.x)
====================================================

A minimal, focused UI where chat is the primary interface.
All features are accessible through natural conversation.

This version uses:
- BioPipelines facade (the single entry point)
- ModelOrchestrator for LLM routing
- Session management built into the facade
- Job Status Panel for monitoring SLURM jobs
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Generator

import gradio as gr

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Configuration
# ============================================================================

from workflow_composer.web.utils import detect_vllm_endpoint, get_default_port, use_local_llm
from workflow_composer.web.components.job_panel import (
    get_user_jobs, format_jobs_table, get_recent_jobs, get_job_log, cancel_job
)

VLLM_URL = detect_vllm_endpoint()
USE_LOCAL_LLM = use_local_llm()
DEFAULT_PORT = get_default_port()

# ============================================================================
# Import BioPipelines Facade
# ============================================================================

bp = None
BP_AVAILABLE = False

try:
    from workflow_composer import BioPipelines
    bp = BioPipelines()
    BP_AVAILABLE = True
    print("✓ BioPipelines facade initialized")
except Exception as e:
    print(f"⚠ BioPipelines facade not available: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# Chat Response Generator
# ============================================================================

# Session ID per web client (simplified - in production, use cookies/auth)
_session_ids = {}


def get_or_create_session(request: gr.Request = None) -> str:
    """Get or create a session for the web client."""
    if not bp:
        return None
    
    # Use request client ID if available, otherwise default
    client_id = "web_default"
    if request and hasattr(request, 'client'):
        client_id = f"web_{request.client.host}_{request.client.port}"
    
    if client_id not in _session_ids:
        try:
            _session_ids[client_id] = bp.create_session(client_id)
        except Exception:
            return None
    
    return _session_ids.get(client_id)


def chat_response(message: str, history: List[Dict], request: gr.Request = None) -> Generator[List[Dict], None, None]:
    """
    Generate chat response using the BioPipelines facade with streaming.
    
    Supports session management for multi-turn conversations.
    
    Args:
        message: User's input message
        history: Chat history as list of dicts with "role" and "content"
        request: Gradio request for session identification
    
    Yields:
        Updated history with assistant response (progressively)
    """
    if not message.strip():
        yield history
        return
    
    if not bp:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ BioPipelines not available."}
        ]
        return
    
    # Get session for this client
    session_id = get_or_create_session(request)
    
    # Add user message to history first
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]
    
    # Try streaming if available
    try:
        if hasattr(bp, 'chat_stream'):
            # Use streaming chat with session support
            full_response = ""
            for chunk in bp.chat_stream(message, history=history, session_id=session_id):
                full_response += chunk
                new_history[-1]["content"] = full_response
                yield new_history
        else:
            # Fallback to non-streaming
            response = bp.chat(message, history=history, session_id=session_id)
            new_history[-1]["content"] = response.message
            yield new_history
    except Exception as e:
        new_history[-1]["content"] = f"❌ Error: {e}"
        yield new_history


# ============================================================================
# Status Functions
# ============================================================================

def get_status() -> str:
    """Get current status as HTML."""
    if bp:
        health = bp.health_check()
        status_parts = []
        if health.get("llm_available"):
            status_parts.append(f"🟢 LLM: {health.get('llm_provider', 'unknown')}")
        else:
            status_parts.append("🔴 No LLM")
        if health.get("tools_available"):
            status_parts.append(f"🛠️ {health.get('tool_count', 0)} tools")
        return " | ".join(status_parts) if status_parts else "🟢 Ready"
    return "🔴 BioPipelines not available"


# ============================================================================
# Example Messages
# ============================================================================

EXAMPLES = [
    {"text": "Help me get started with BioPipelines", "display_text": "🚀 Get Started"},
    {"text": "Scan my data in ~/data/fastq", "display_text": "📁 Scan Data"},
    {"text": "Create an RNA-seq differential expression workflow", "display_text": "🧬 RNA-seq"},
    {"text": "What workflows are available?", "display_text": "📋 List Workflows"},
    {"text": "Show me my running jobs", "display_text": "📊 Check Jobs"},
    {"text": "Search ENCODE for H3K27ac ChIP-seq in liver", "display_text": "🔬 Search ENCODE"},
    {"text": "Search TCGA for lung cancer RNA-seq data", "display_text": "🏥 Search TCGA"},
]


# ============================================================================
# Main UI
# ============================================================================

def create_app() -> gr.Blocks:
    """Create the Gradio app."""
    
    with gr.Blocks(title="🧬 BioPipelines") as app:
        
        # Header
        gr.Markdown("# 🧬 BioPipelines\n*AI-powered bioinformatics workflow composer*")
        
        # Main layout: Chat + Job Panel side by side
        with gr.Row():
            # Main Chat Column (75%)
            with gr.Column(scale=3):
                # Main Chat
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=450,
                    placeholder="Ask me to scan data, create workflows, or run analyses...",
                    examples=EXAMPLES,
                    buttons=["copy"],
                    avatar_images=(None, "🧬"),
                    layout="bubble",
                )
                
                # Input
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=9,
                        lines=1,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
            
            # Job Status Panel (25%)
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📊 Jobs")
                
                # Active jobs with auto-refresh
                with gr.Group():
                    jobs_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    active_jobs_html = gr.HTML(value="<p>Loading...</p>")
                
                # Recent jobs (collapsed)
                with gr.Accordion("📋 Recent (24h)", open=False):
                    recent_jobs_html = gr.HTML(value="<p>Click refresh...</p>")
                
                # Quick job actions
                with gr.Accordion("⚡ Quick Actions", open=False):
                    quick_job_id = gr.Textbox(
                        label="Job ID",
                        placeholder="Enter job ID...",
                        scale=1,
                    )
                    with gr.Row():
                        view_log_btn = gr.Button("📄 Log", size="sm")
                        cancel_job_btn = gr.Button("❌ Cancel", size="sm", variant="stop")
                    job_action_output = gr.Markdown("")
        
        # Feedback section (for intent corrections)
        with gr.Accordion("📝 Feedback & Learning", open=False):
            gr.Markdown("**Help improve the assistant by correcting intent classifications:**")
            with gr.Row():
                feedback_query = gr.Textbox(
                    label="Query",
                    placeholder="The query that was misclassified...",
                    scale=3,
                )
                feedback_intent = gr.Dropdown(
                    label="Correct Intent",
                    choices=[
                        "DATA_SEARCH", "DATA_DOWNLOAD", "DATA_SCAN", "DATA_VALIDATE",
                        "WORKFLOW_CREATE", "WORKFLOW_LIST", "WORKFLOW_CONFIGURE",
                        "JOB_SUBMIT", "JOB_STATUS", "JOB_CANCEL", "JOB_LOGS",
                        "ANALYSIS_QC", "ANALYSIS_RESULTS", "ANALYSIS_COMPARE",
                        "META_HELP", "META_EXPLAIN", "META_STATUS",
                    ],
                    scale=2,
                )
            feedback_text = gr.Textbox(
                label="Additional Feedback (optional)",
                placeholder="Any context that might help...",
                lines=1,
            )
            feedback_btn = gr.Button("Submit Feedback", variant="secondary")
            feedback_result = gr.Markdown("")
            
            # Learning stats
            with gr.Row():
                stats_btn = gr.Button("📊 Show Stats")
                stats_output = gr.JSON(label="Learning Statistics")
        
        # Settings (collapsed)
        with gr.Accordion("⚙️ Settings", open=False):
            settings_info = "**BioPipelines:** " + ("Available ✓" if BP_AVAILABLE else "Not available")
            if bp:
                health = bp.health_check()
                settings_info += f"\n**LLM:** {health.get('llm_provider', 'Not configured')}"
                settings_info += f"\n**Tools:** {health.get('tool_count', 0)}"
            gr.Markdown(settings_info)
            clear = gr.Button("🗑️ Clear Chat")
        
        # Event handlers
        def submit(message, history):
            if not message.strip():
                return history, ""
            for response in chat_response(message, history):
                yield response, ""
        
        def clear_chat():
            return [], ""
        
        def submit_feedback(query, intent, text):
            if not bp:
                return "❌ BioPipelines not available"
            if not query or not intent:
                return "⚠️ Please provide both query and correct intent"
            # Feedback through the agent if available
            try:
                if hasattr(bp, 'agent') and bp.agent:
                    bp.agent.submit_feedback(query, intent, text)
                    return "✅ Feedback recorded!"
                return "⚠️ Feedback system not available"
            except Exception as e:
                return f"❌ {e}"
        
        def get_learning_stats():
            if not bp:
                return {}
            try:
                if hasattr(bp, 'agent') and bp.agent and hasattr(bp.agent, 'get_learning_stats'):
                    return bp.agent.get_learning_stats()
            except Exception:
                pass
            return {"message": "Stats not available"}
        
        # Wire up events
        msg.submit(submit, [msg, chatbot], [chatbot, msg])
        send.click(submit, [msg, chatbot], [chatbot, msg])
        clear.click(clear_chat, outputs=[chatbot, msg])
        feedback_btn.click(submit_feedback, [feedback_query, feedback_intent, feedback_text], feedback_result)
        stats_btn.click(get_learning_stats, outputs=stats_output)
        
        # Job panel events
        def refresh_all_jobs():
            """Refresh both active and recent job panels."""
            return format_jobs_table(get_user_jobs()), format_jobs_table(get_recent_jobs())
        
        def view_job_log(job_id):
            if not job_id or not job_id.strip():
                return "Enter a job ID"
            return get_job_log(job_id.strip())
        
        def cancel_slurm_job(job_id):
            if not job_id or not job_id.strip():
                return "Enter a job ID"
            return cancel_job(job_id.strip())
        
        jobs_refresh_btn.click(refresh_all_jobs, outputs=[active_jobs_html, recent_jobs_html])
        view_log_btn.click(view_job_log, inputs=[quick_job_id], outputs=[job_action_output])
        cancel_job_btn.click(cancel_slurm_job, inputs=[quick_job_id], outputs=[job_action_output])
        
        # Auto-refresh both panels every 30 seconds
        job_timer = gr.Timer(value=30)
        job_timer.tick(refresh_all_jobs, outputs=[active_jobs_html, recent_jobs_html])
        
        # Load jobs on page open
        app.load(refresh_all_jobs, outputs=[active_jobs_html, recent_jobs_html])
    
    return app


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BioPipelines Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                    🧬 BioPipelines                         ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print(f"║  Server: http://localhost:{args.port:<5}                          ║")
    print(f"║  Facade: {'BioPipelines ✓' if BP_AVAILABLE else 'Not available ✗':<30}    ║")
    if bp:
        health = bp.health_check()
        llm_provider = health.get('llm_provider') or 'None'
        tool_count = health.get('tool_count') or 0
        print(f"║  LLM: {llm_provider:<10}                                   ║")
        print(f"║  Tools: {tool_count:<3} available                                   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
