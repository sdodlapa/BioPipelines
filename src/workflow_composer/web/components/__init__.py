"""
Web components for the BioPipelines Gradio UI.

This package provides reusable components for:
- Result browsing and visualization
- File tree navigation
- Download management
"""

from .result_browser import (
    ResultBrowserComponent,
    create_result_browser,
    format_file_tree_html,
    format_result_summary_html,
)

__all__ = [
    "ResultBrowserComponent",
    "create_result_browser",
    "format_file_tree_html",
    "format_result_summary_html",
]
