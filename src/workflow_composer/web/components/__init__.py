"""
Web components for the BioPipelines Gradio UI.

This package provides reusable components for:
- Result browsing and visualization
- File tree navigation
- Download management
- Data discovery and management
- Unified workspace with chat + sidebar
"""

from .result_browser import (
    ResultBrowserComponent,
    create_result_browser,
    format_file_tree_html,
    format_result_summary_html,
)

# Import data tab components (may fail if gradio not installed)
try:
    from .data_tab import (
        create_data_tab,
        create_local_scanner_ui,
        create_remote_search_ui,
        create_reference_manager_ui,
        create_data_summary_panel,
    )
    _DATA_TAB_AVAILABLE = True
except ImportError:
    _DATA_TAB_AVAILABLE = False
    create_data_tab = None
    create_local_scanner_ui = None
    create_remote_search_ui = None
    create_reference_manager_ui = None
    create_data_summary_panel = None

# Import unified workspace component
try:
    from .unified_workspace import (
        create_unified_workspace,
        refresh_manifest_panel,
        refresh_jobs_panel,
        refresh_workflows_list,
        get_provider_choices,
        get_available_workflows,
        get_example_prompts,
    )
    _UNIFIED_WORKSPACE_AVAILABLE = True
except ImportError:
    _UNIFIED_WORKSPACE_AVAILABLE = False
    create_unified_workspace = None
    refresh_manifest_panel = None
    refresh_jobs_panel = None
    refresh_workflows_list = None
    get_provider_choices = None
    get_available_workflows = None
    get_example_prompts = None

__all__ = [
    # Result browser
    "ResultBrowserComponent",
    "create_result_browser",
    "format_file_tree_html",
    "format_result_summary_html",
    # Data tab
    "create_data_tab",
    "create_local_scanner_ui",
    "create_remote_search_ui",
    "create_reference_manager_ui",
    "create_data_summary_panel",
    # Unified workspace
    "create_unified_workspace",
    "refresh_manifest_panel",
    "refresh_jobs_panel",
    "refresh_workflows_list",
    "get_provider_choices",
    "get_available_workflows",
    "get_example_prompts",
]
