"""
Result browser component for the Gradio UI.

This module provides the main result browsing interface including:
- Job selection dropdown
- Result summary display
- QC report embedding
- File tree navigation
- Download options
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import html

logger = logging.getLogger(__name__)


class ResultBrowserComponent:
    """
    Component for browsing pipeline results in Gradio UI.
    
    This component manages:
    - Scanning output directories
    - Displaying file summaries
    - Rendering file previews
    - Handling downloads
    """
    
    def __init__(self):
        """Initialize the result browser."""
        from workflow_composer.results import ResultCollector, ResultViewer, ResultArchiver
        
        self.collector = ResultCollector()
        self.viewer = ResultViewer()
        self.archiver = ResultArchiver()
        
        self._current_summary = None
        self._cache: Dict[str, Any] = {}
    
    def scan_job_results(
        self,
        output_dir: str,
        job_id: Optional[str] = None,
        pipeline_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scan a job's output directory and return summary.
        
        Args:
            output_dir: Path to output directory
            job_id: Optional job identifier
            pipeline_type: Optional pipeline type for specialized patterns
            
        Returns:
            Dictionary with summary information
        """
        from workflow_composer.results import ResultCollector
        
        collector = ResultCollector(pipeline_type=pipeline_type)
        summary = collector.scan(output_dir, job_id)
        
        self._current_summary = summary
        
        return {
            "success": True,
            "job_id": summary.job_id,
            "total_files": summary.total_files,
            "total_size": summary.size_human,
            "has_multiqc": summary.has_multiqc,
            "categories": summary.get_category_summary(),
            "downloadable_size": summary.downloadable_size_human,
        }
    
    def get_summary_html(self) -> str:
        """Get HTML summary of current results."""
        if not self._current_summary:
            return "<p>No results loaded. Select a completed job to view results.</p>"
        
        return format_result_summary_html(self._current_summary)
    
    def get_file_tree_html(self) -> str:
        """Get HTML file tree of current results."""
        if not self._current_summary or not self._current_summary.file_tree:
            return "<p>No file tree available.</p>"
        
        return format_file_tree_html(self._current_summary.file_tree)
    
    def get_qc_reports_html(self) -> str:
        """Get HTML list of QC reports."""
        if not self._current_summary:
            return "<p>No QC reports available.</p>"
        
        if not self._current_summary.qc_reports:
            return "<p>No QC reports found in this job's output.</p>"
        
        html_parts = ["<div class='qc-reports'>"]
        
        for report in self._current_summary.qc_reports:
            is_multiqc = 'multiqc' in report.name.lower()
            icon = "📊" if is_multiqc else "📋"
            primary = " (Primary)" if report.is_primary else ""
            
            html_parts.append(f"""
                <div class='qc-report-item'>
                    <span class='icon'>{icon}</span>
                    <span class='name'>{html.escape(report.name)}{primary}</span>
                    <span class='size'>{report.size_human}</span>
                </div>
            """)
        
        html_parts.append("</div>")
        
        return "\n".join(html_parts)
    
    def get_multiqc_content(self) -> Optional[str]:
        """Get the MultiQC report HTML content."""
        if not self._current_summary or not self._current_summary.multiqc_report:
            return None
        
        try:
            with open(self._current_summary.multiqc_report.path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading MultiQC report: {e}")
            return None
    
    def get_visualizations(self) -> List[str]:
        """Get list of visualization image paths."""
        if not self._current_summary:
            return []
        
        return [
            str(f.path) for f in self._current_summary.visualizations
            if f.file_type.value == "image"
        ]
    
    def render_file(self, file_path: str) -> Dict[str, Any]:
        """
        Render a specific file for preview.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with content type and content
        """
        if not self._current_summary:
            return {"error": "No results loaded"}
        
        # Find the file in our summary
        target_path = Path(file_path)
        result_file = None
        
        for f in self._current_summary.all_files:
            if f.path == target_path or str(f.path) == file_path:
                result_file = f
                break
        
        if not result_file:
            return {"error": f"File not found: {file_path}"}
        
        content = self.viewer.render(result_file)
        
        return {
            "content_type": content.content_type,
            "content": content.content if not content.is_error else None,
            "title": content.title,
            "error": content.error,
        }
    
    def create_download_archive(
        self,
        archive_type: str = "full",
    ) -> Optional[str]:
        """
        Create a download archive.
        
        Args:
            archive_type: "full", "qc_only", or "with_bam"
            
        Returns:
            Path to the created ZIP file, or None on error
        """
        if not self._current_summary:
            return None
        
        try:
            if archive_type == "qc_only":
                from workflow_composer.results import create_qc_archive
                return str(create_qc_archive(self._current_summary))
            elif archive_type == "with_bam":
                from workflow_composer.results import create_full_archive
                return str(create_full_archive(self._current_summary, include_bam=True))
            else:
                from workflow_composer.results import create_full_archive
                return str(create_full_archive(self._current_summary, include_bam=False))
        except Exception as e:
            logger.error(f"Error creating archive: {e}")
            return None


# =============================================================================
# HTML FORMATTING FUNCTIONS
# =============================================================================

def format_result_summary_html(summary) -> str:
    """
    Format a ResultSummary as HTML for display.
    
    Args:
        summary: ResultSummary object
        
    Returns:
        HTML string
    """
    categories = summary.get_category_summary()
    
    return f"""
<div class="result-summary">
    <h3>📁 Results Summary</h3>
    
    <div class="summary-stats">
        <div class="stat">
            <span class="label">Output Directory:</span>
            <code>{html.escape(str(summary.output_dir))}</code>
        </div>
        <div class="stat">
            <span class="label">Total Files:</span>
            <span class="value">{summary.total_files}</span>
        </div>
        <div class="stat">
            <span class="label">Total Size:</span>
            <span class="value">{summary.size_human}</span>
        </div>
        <div class="stat">
            <span class="label">Downloadable Size:</span>
            <span class="value">{summary.downloadable_size_human}</span>
            <span class="note">(excludes large BAM files)</span>
        </div>
    </div>
    
    <h4>📊 File Categories</h4>
    <div class="category-grid">
        <div class="category">
            <span class="icon">📊</span>
            <span class="name">QC Reports</span>
            <span class="count">{categories['qc_reports']}</span>
        </div>
        <div class="category">
            <span class="icon">📈</span>
            <span class="name">Visualizations</span>
            <span class="count">{categories['visualizations']}</span>
        </div>
        <div class="category">
            <span class="icon">📋</span>
            <span class="name">Data Files</span>
            <span class="count">{categories['data_files']}</span>
        </div>
        <div class="category">
            <span class="icon">💾</span>
            <span class="name">Alignments</span>
            <span class="count">{categories['alignments']}</span>
        </div>
        <div class="category">
            <span class="icon">📝</span>
            <span class="name">Logs</span>
            <span class="count">{categories['logs']}</span>
        </div>
    </div>
    
    {"<p class='multiqc-note'>✅ MultiQC report available</p>" if summary.has_multiqc else ""}
</div>

<style>
.result-summary {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 1rem;
}}
.summary-stats {{
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1rem;
}}
.summary-stats .stat {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.summary-stats .label {{
    font-weight: 600;
    min-width: 150px;
}}
.summary-stats code {{
    background: #f0f0f0;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.9em;
}}
.summary-stats .note {{
    color: #666;
    font-size: 0.85em;
}}
.category-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
}}
.category {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem;
    background: #f8f9fa;
    border-radius: 6px;
}}
.category .icon {{
    font-size: 1.2em;
}}
.category .count {{
    margin-left: auto;
    font-weight: 600;
    color: #0066cc;
}}
.multiqc-note {{
    margin-top: 1rem;
    color: #28a745;
    font-weight: 500;
}}
</style>
"""


def format_file_tree_html(tree_node, depth: int = 0) -> str:
    """
    Format a FileTreeNode as HTML.
    
    Args:
        tree_node: FileTreeNode object
        depth: Current depth for indentation
        
    Returns:
        HTML string
    """
    indent = "  " * depth
    
    if tree_node.is_dir:
        # Directory node
        icon = "📁" if depth > 0 else "📂"
        children_html = "\n".join(
            format_file_tree_html(child, depth + 1)
            for child in tree_node.children
        )
        
        return f"""
{indent}<div class="tree-dir" style="margin-left: {depth * 20}px;">
{indent}  <span class="tree-icon">{icon}</span>
{indent}  <span class="tree-name">{html.escape(tree_node.name)}</span>
{indent}  <span class="tree-meta">({tree_node.file_count} files)</span>
{indent}</div>
{children_html}
"""
    else:
        # File node
        if tree_node.result_file:
            file_type = tree_node.result_file.file_type.value
            size = tree_node.result_file.size_human
            viewable = "viewable" if tree_node.result_file.is_viewable else ""
            
            icon = _get_file_icon(file_type)
        else:
            file_type = "unknown"
            size = ""
            viewable = ""
            icon = "📄"
        
        return f"""
{indent}<div class="tree-file {viewable}" style="margin-left: {depth * 20}px;" data-path="{html.escape(str(tree_node.path))}">
{indent}  <span class="tree-icon">{icon}</span>
{indent}  <span class="tree-name">{html.escape(tree_node.name)}</span>
{indent}  <span class="tree-size">{size}</span>
{indent}</div>
"""


def _get_file_icon(file_type: str) -> str:
    """Get icon for file type."""
    icons = {
        "qc_report": "📊",
        "image": "🖼️",
        "pdf": "📕",
        "table": "📋",
        "alignment": "🧬",
        "variant": "🔬",
        "matrix": "🔢",
        "log": "📝",
        "config": "⚙️",
        "text": "📄",
        "json": "📦",
        "yaml": "📦",
        "archive": "📦",
    }
    return icons.get(file_type, "📄")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_browser_instance: Optional[ResultBrowserComponent] = None


def create_result_browser() -> ResultBrowserComponent:
    """
    Get or create the result browser component.
    
    Returns:
        ResultBrowserComponent instance
    """
    global _browser_instance
    
    if _browser_instance is None:
        _browser_instance = ResultBrowserComponent()
    
    return _browser_instance


def get_result_browser() -> Optional[ResultBrowserComponent]:
    """Get the current result browser instance."""
    return _browser_instance
