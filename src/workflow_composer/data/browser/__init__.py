"""
Data Browser UI Components
==========================

Gradio components for browsing and downloading genomics data.

Usage:
    from workflow_composer.data.browser import create_reference_browser_tab
    
    with gr.Blocks() as app:
        with gr.Tabs():
            with gr.Tab("Reference Browser"):
                create_reference_browser_tab()
"""

from .reference_browser import (
    create_reference_browser_tab,
    search_datasets,
    get_dataset_details,
    download_dataset,
    scan_local_references,
)

__all__ = [
    "create_reference_browser_tab",
    "search_datasets",
    "get_dataset_details",
    "download_dataset",
    "scan_local_references",
]
