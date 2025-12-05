"""
Tests for BioPipelines MCP Server
==================================

Tests for the Model Context Protocol server implementation.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workflow_composer.mcp.server import (
    BioPipelinesMCPServer,
    create_server,
    ToolDefinition,
    ResourceDefinition,
    PromptDefinition,
    PromptArgument,
    ToolAnnotations,
)


class TestMCPServerInitialization:
    """Tests for MCP server initialization."""
    
    def test_create_server(self):
        """Test server creation."""
        server = create_server()
        assert isinstance(server, BioPipelinesMCPServer)
    
    def test_server_has_tools(self):
        """Test that server has registered tools."""
        server = create_server()
        assert len(server.tools) > 0
    
    def test_server_has_resources(self):
        """Test that server has registered resources."""
        server = create_server()
        assert len(server.resources) > 0
    
    def test_tools_have_required_fields(self):
        """Test that all tools have required fields."""
        server = create_server()
        
        for name, tool in server.tools.items():
            assert tool.name == name
            assert tool.description
            assert "type" in tool.parameters
            assert callable(tool.handler)
    
    def test_resources_have_required_fields(self):
        """Test that all resources have required fields."""
        server = create_server()
        
        for uri, resource in server.resources.items():
            assert resource.uri == uri
            assert resource.name
            assert resource.description
            assert callable(resource.handler)


class TestMCPToolList:
    """Tests for tool listing functionality."""
    
    def test_get_tools_list_format(self):
        """Test tools list format."""
        server = create_server()
        tools = server.get_tools_list()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
    
    def test_expected_tools_present(self):
        """Test that expected tools are registered."""
        server = create_server()
        tool_names = [t["name"] for t in server.get_tools_list()]
        
        expected_tools = [
            "search_encode",
            "search_geo",
            "create_workflow",
            "search_uniprot",
            "get_protein_interactions",
            "get_functional_enrichment",
            "search_kegg_pathways",
            "search_pubmed",
            "search_variants",
            "explain_concept",
        ]
        
        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"


class TestMCPResourceList:
    """Tests for resource listing functionality."""
    
    def test_get_resources_list_format(self):
        """Test resources list format."""
        server = create_server()
        resources = server.get_resources_list()
        
        assert isinstance(resources, list)
        assert len(resources) > 0
        
        for resource in resources:
            assert "uri" in resource
            assert "name" in resource
            assert "description" in resource
    
    def test_expected_resources_present(self):
        """Test that expected resources are registered."""
        server = create_server()
        resource_uris = [r["uri"] for r in server.get_resources_list()]
        
        expected_resources = [
            "biopipelines://skills",
            "biopipelines://templates",
            "biopipelines://databases",
        ]
        
        for expected in expected_resources:
            assert expected in resource_uris, f"Missing resource: {expected}"


class TestMCPProtocol:
    """Tests for MCP protocol handling."""
    
    @pytest.fixture
    def server(self):
        """Create server fixture."""
        return create_server()
    
    @pytest.mark.asyncio
    async def test_initialize_request(self, server):
        """Test initialize protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "protocolVersion" in response["result"]
        assert "serverInfo" in response["result"]
        assert response["result"]["serverInfo"]["name"] == "biopipelines"
    
    @pytest.mark.asyncio
    async def test_tools_list_request(self, server):
        """Test tools/list protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0
    
    @pytest.mark.asyncio
    async def test_resources_list_request(self, server):
        """Test resources/list protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert "resources" in response["result"]
    
    @pytest.mark.asyncio
    async def test_unknown_method(self, server):
        """Test handling of unknown method."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown/method",
            "params": {}
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32601


class TestMCPToolCalls:
    """Tests for MCP tool call handling."""
    
    @pytest.fixture
    def server(self):
        """Create server fixture."""
        return create_server()
    
    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, server):
        """Test calling an unknown tool."""
        result = await server.call_tool("unknown_tool", {})
        
        assert not result["success"]
        assert "error" in result
        assert "Unknown tool" in result["error"]
    
    @pytest.mark.asyncio
    async def test_tool_call_request(self, server):
        """Test tools/call protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "explain_concept",
                "arguments": {"concept": "RNA-seq"}
            }
        }
        
        # Mock the handler to avoid actual API calls
        with patch.object(server, 'call_tool', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "success": True,
                "content": "RNA-seq is a technique..."
            }
            
            response = await server._handle_request(request)
            
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 5
            assert "result" in response
            assert "content" in response["result"]


class TestMCPResourceReads:
    """Tests for MCP resource read handling."""
    
    @pytest.fixture
    def server(self):
        """Create server fixture."""
        return create_server()
    
    @pytest.mark.asyncio
    async def test_read_unknown_resource(self, server):
        """Test reading an unknown resource."""
        result = await server.read_resource("unknown://resource")
        
        assert "Unknown resource" in result
    
    @pytest.mark.asyncio
    async def test_resource_read_request(self, server):
        """Test resources/read protocol method."""
        request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "resources/read",
            "params": {
                "uri": "biopipelines://databases"
            }
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 6
        assert "result" in response
        assert "contents" in response["result"]
        
        # Database resource should return markdown content
        content = response["result"]["contents"][0]
        assert content["uri"] == "biopipelines://databases"
        assert "UniProt" in content["text"]


class TestMCPFormatters:
    """Tests for result formatting functions."""
    
    @pytest.fixture
    def server(self):
        """Create server fixture."""
        return create_server()
    
    def test_format_search_results_success(self, server):
        """Test search results formatting with success."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.count = 2
        mock_result.data = [
            {"id": "EXP001", "title": "Test experiment"},
            {"id": "EXP002", "name": "Another experiment"}
        ]
        
        formatted = server._format_search_results(mock_result)
        
        assert "Found 2 results" in formatted
        assert "EXP001" in formatted
        assert "EXP002" in formatted
    
    def test_format_search_results_failure(self, server):
        """Test search results formatting with failure."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.message = "API error"
        
        formatted = server._format_search_results(mock_result)
        
        assert "Search failed" in formatted
        assert "API error" in formatted
    
    def test_format_protein_results_success(self, server):
        """Test protein results formatting."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.count = 1
        mock_result.data = [
            {
                "primaryAccession": "P12345",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Test Protein"}
                    }
                },
                "genes": [{"geneName": {"value": "TESTP"}}]
            }
        ]
        
        formatted = server._format_protein_results(mock_result)
        
        assert "Found 1 proteins" in formatted
        assert "P12345" in formatted
        assert "TESTP" in formatted
    
    def test_format_enrichment_results(self, server):
        """Test enrichment results formatting."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.count = 3
        mock_result.data = [
            {"category": "GO:BP", "description": "cell cycle", "p_value": 0.001},
            {"category": "GO:BP", "description": "apoptosis", "p_value": 0.01},
            {"category": "KEGG", "description": "cancer pathway", "p_value": 0.05}
        ]
        
        formatted = server._format_enrichment_results(mock_result)
        
        assert "Found 3 enriched terms" in formatted
        assert "GO:BP" in formatted
        assert "cell cycle" in formatted
    
    def test_format_pathway_results(self, server):
        """Test KEGG pathway results formatting."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.count = 2
        mock_result.data = [
            {"id": "hsa04110", "name": "Cell cycle"},
            {"id": "hsa04210", "name": "Apoptosis"}
        ]
        
        formatted = server._format_pathway_results(mock_result)
        
        assert "Found 2 pathways" in formatted
        assert "hsa04110" in formatted
        assert "Cell cycle" in formatted


class TestMCPServerUnit:
    """Unit tests for server methods."""
    
    def test_register_tool(self):
        """Test tool registration."""
        server = create_server()
        initial_count = len(server.tools)
        
        async def dummy_handler(**kwargs):
            return {"success": True}
        
        server._register_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=dummy_handler
        )
        
        assert len(server.tools) == initial_count + 1
        assert "test_tool" in server.tools
    
    def test_register_resource(self):
        """Test resource registration."""
        server = create_server()
        initial_count = len(server.resources)
        
        async def dummy_handler():
            return "test content"
        
        server._register_resource(
            uri="test://resource",
            name="Test Resource",
            description="A test resource",
            handler=dummy_handler
        )
        
        assert len(server.resources) == initial_count + 1
        assert "test://resource" in server.resources


class TestMCPPrompts:
    """Tests for MCP prompts functionality."""
    
    def test_server_has_prompts(self):
        """Test that server has registered prompts."""
        server = create_server()
        assert hasattr(server, 'prompts')
        assert len(server.prompts) > 0
    
    def test_get_prompts_list_format(self):
        """Test prompts list format."""
        server = create_server()
        prompts = server.get_prompts_list()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
        for prompt in prompts:
            assert "name" in prompt
            assert "description" in prompt
    
    def test_expected_prompts_present(self):
        """Test that expected prompts are registered."""
        server = create_server()
        prompt_names = [p["name"] for p in server.get_prompts_list()]
        
        expected_prompts = [
            "analyze_rnaseq",
            "debug_workflow",
            "find_datasets",
            "design_pipeline",
            "explain_results",
        ]
        
        for expected in expected_prompts:
            assert expected in prompt_names, f"Missing prompt: {expected}"
    
    def test_register_prompt(self):
        """Test prompt registration."""
        from workflow_composer.mcp.server import PromptArgument
        
        server = create_server()
        initial_count = len(server.prompts)
        
        async def dummy_handler(**kwargs):
            return {"description": "test", "messages": []}
        
        server._register_prompt(
            name="test_prompt",
            description="A test prompt",
            handler=dummy_handler,
            arguments=[PromptArgument(name="arg1", description="Arg 1")]
        )
        
        assert len(server.prompts) == initial_count + 1
        assert "test_prompt" in server.prompts
    
    @pytest.mark.asyncio
    async def test_get_prompt(self):
        """Test getting a prompt."""
        server = create_server()
        
        result = await server.get_prompt("analyze_rnaseq", {
            "data_path": "/test/path",
            "organism": "human"
        })
        
        assert "error" not in result
        assert "messages" in result
        assert len(result["messages"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_unknown_prompt(self):
        """Test getting an unknown prompt."""
        server = create_server()
        
        result = await server.get_prompt("nonexistent_prompt")
        
        assert "error" in result
        assert "Unknown prompt" in result["error"]


class TestMCPToolAnnotations:
    """Tests for MCP tool annotations (2025-06-18 spec)."""
    
    def test_tools_with_annotations(self):
        """Test that tools can have annotations."""
        from workflow_composer.mcp.server import ToolAnnotations
        
        server = create_server()
        tools = server.get_tools_list()
        
        # Check that some tools have annotations
        annotated_tools = [t for t in tools if "annotations" in t]
        
        # We should have tools with annotations now
        assert len(annotated_tools) > 0, "Expected at least some tools to have annotations"
    
    def test_annotation_structure(self):
        """Test that annotations have correct structure."""
        server = create_server()
        tools = server.get_tools_list()
        
        for tool in tools:
            if "annotations" in tool:
                ann = tool["annotations"]
                assert "readOnly" in ann
                assert "requiresConfirmation" in ann
                assert "destructive" in ann
                assert "idempotent" in ann
                assert "category" in ann
    
    def test_output_schemas(self):
        """Test that some tools have output schemas."""
        server = create_server()
        tools = server.get_tools_list()
        
        # Check for tools with output schemas
        tools_with_schemas = [t for t in tools if "outputSchema" in t]
        
        assert len(tools_with_schemas) > 0, "Expected at least some tools to have output schemas"
        
        # Validate schema structure
        for tool in tools_with_schemas:
            schema = tool["outputSchema"]
            assert "type" in schema


class TestMCPProtocolPrompts:
    """Tests for prompts-related MCP protocol handlers."""
    
    @pytest.mark.asyncio
    async def test_prompts_list_request(self):
        """Test prompts/list request."""
        server = create_server()
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/list"
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "prompts" in response["result"]
        assert len(response["result"]["prompts"]) > 0
    
    @pytest.mark.asyncio
    async def test_prompts_get_request(self):
        """Test prompts/get request."""
        server = create_server()
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {
                "name": "analyze_rnaseq",
                "arguments": {
                    "data_path": "/test/data",
                    "organism": "mouse"
                }
            }
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "messages" in response["result"]
    
    @pytest.mark.asyncio
    async def test_prompts_get_unknown_request(self):
        """Test prompts/get with unknown prompt."""
        server = create_server()
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {
                "name": "nonexistent"
            }
        }
        
        response = await server._handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "error" in response


class TestMCPJobManagement:
    """Tests for job management tools."""
    
    def test_job_tools_present(self):
        """Test that job management tools are registered."""
        server = create_server()
        tool_names = [t["name"] for t in server.get_tools_list()]
        
        job_tools = [
            "submit_job",
            "list_jobs",
            "cancel_job",
            "get_job_logs",
        ]
        
        for tool in job_tools:
            assert tool in tool_names, f"Missing job tool: {tool}"
    
    def test_submit_job_annotations(self):
        """Test that submit_job has correct annotations."""
        server = create_server()
        tools = server.get_tools_list()
        
        submit_job = next((t for t in tools if t["name"] == "submit_job"), None)
        assert submit_job is not None
        
        if "annotations" in submit_job:
            ann = submit_job["annotations"]
            # Submit is not read-only, not destructive
            assert ann["readOnly"] == False
            assert ann["destructive"] == False


class TestMCPStrategyTools:
    """Tests for LLM strategy tools."""
    
    def test_strategy_tools_present(self):
        """Test that strategy management tools are registered."""
        server = create_server()
        tool_names = [t["name"] for t in server.get_tools_list()]
        
        strategy_tools = [
            "get_llm_strategy",
            "set_llm_strategy",
        ]
        
        for tool in strategy_tools:
            assert tool in tool_names, f"Missing strategy tool: {tool}"


class TestMCPInitializeCapabilities:
    """Tests for enhanced initialize capabilities."""
    
    @pytest.mark.asyncio
    async def test_initialize_includes_prompts_capability(self):
        """Test that initialize returns prompts capability."""
        server = create_server()
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize"
        }
        
        response = await server._handle_request(request)
        
        assert "result" in response
        result = response["result"]
        
        assert "capabilities" in result
        capabilities = result["capabilities"]
        
        assert "prompts" in capabilities, "Initialize should expose prompts capability"
        assert "tools" in capabilities
        assert "resources" in capabilities
    
    @pytest.mark.asyncio
    async def test_server_version_updated(self):
        """Test that server version is 2.0.0."""
        server = create_server()
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize"
        }
        
        response = await server._handle_request(request)
        
        assert response["result"]["serverInfo"]["version"] == "2.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
