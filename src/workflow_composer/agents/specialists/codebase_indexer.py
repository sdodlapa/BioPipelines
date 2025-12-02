"""
Codebase Indexer
================

Indexes existing codebases for intelligent reference during code generation.

Capabilities:
- Parse and index Nextflow workflows
- Extract process definitions and channel flows
- Build semantic relationships between components
- Enable intelligent code retrieval
"""

import logging
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class CodeElementType(Enum):
    """Types of code elements that can be indexed."""
    PROCESS = "process"
    WORKFLOW = "workflow"
    FUNCTION = "function"
    CHANNEL = "channel"
    PARAMETER = "parameter"
    IMPORT = "import"
    CONFIG = "config"


@dataclass
class CodeElement:
    """An indexed code element."""
    element_type: CodeElementType
    name: str
    file_path: str
    line_number: int
    content: str
    description: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    container: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "element_type": self.element_type.value,
            "name": self.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "content": self.content,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "container": self.container,
            "tools": self.tools,
            "metadata": self.metadata,
        }


@dataclass
class CodebaseIndex:
    """Index of a codebase."""
    name: str
    root_path: str
    created_at: datetime = field(default_factory=datetime.now)
    elements: List[CodeElement] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    statistics: Dict[str, int] = field(default_factory=dict)
    
    def get_processes(self) -> List[CodeElement]:
        """Get all process definitions."""
        return [e for e in self.elements if e.element_type == CodeElementType.PROCESS]
    
    def get_workflows(self) -> List[CodeElement]:
        """Get all workflow definitions."""
        return [e for e in self.elements if e.element_type == CodeElementType.WORKFLOW]
    
    def get_by_name(self, name: str) -> Optional[CodeElement]:
        """Get element by name."""
        for e in self.elements:
            if e.name.lower() == name.lower():
                return e
        return None
    
    def get_by_tool(self, tool: str) -> List[CodeElement]:
        """Get elements using a specific tool."""
        tool_lower = tool.lower()
        return [e for e in self.elements if tool_lower in [t.lower() for t in e.tools]]
    
    def search(self, query: str) -> List[CodeElement]:
        """Search elements by keyword."""
        query_lower = query.lower()
        results = []
        for e in self.elements:
            score = 0
            if query_lower in e.name.lower():
                score += 3
            if query_lower in e.description.lower():
                score += 2
            if query_lower in e.content.lower():
                score += 1
            if any(query_lower in t.lower() for t in e.tools):
                score += 2
            if score > 0:
                results.append((score, e))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "root_path": self.root_path,
            "created_at": self.created_at.isoformat(),
            "elements": [e.to_dict() for e in self.elements],
            "relationships": self.relationships,
            "statistics": self.statistics,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_path: str):
        """Save index to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        logger.info(f"Saved index to {output_path}")
    
    @classmethod
    def load(cls, input_path: str) -> "CodebaseIndex":
        """Load index from file."""
        path = Path(input_path)
        data = json.loads(path.read_text())
        
        elements = []
        for e_data in data.get("elements", []):
            e_data["element_type"] = CodeElementType(e_data["element_type"])
            elements.append(CodeElement(**e_data))
        
        return cls(
            name=data["name"],
            root_path=data["root_path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            elements=elements,
            relationships=data.get("relationships", {}),
            statistics=data.get("statistics", {}),
        )


class CodebaseIndexer:
    """
    Indexes Nextflow codebases for intelligent reference.
    
    Extracts:
    - Process definitions with inputs/outputs
    - Workflow structure and channel flows
    - Container specifications
    - Tool usage patterns
    - Parameter definitions
    
    Inspired by DeepCode's Codebase Intelligence Agent.
    """
    
    # Regex patterns for Nextflow parsing
    PATTERNS = {
        "process": re.compile(
            r"process\s+(\w+)\s*\{(.*?)\n\}",
            re.DOTALL | re.MULTILINE
        ),
        "workflow": re.compile(
            r"workflow\s+(\w*)\s*\{(.*?)\n\}",
            re.DOTALL | re.MULTILINE
        ),
        "input_block": re.compile(
            r"input:\s*(.*?)(?=output:|script:|shell:|exec:|when:|$)",
            re.DOTALL
        ),
        "output_block": re.compile(
            r"output:\s*(.*?)(?=script:|shell:|exec:|when:|$)",
            re.DOTALL
        ),
        "container": re.compile(
            r"container\s+['\"]([^'\"]+)['\"]"
        ),
        "include": re.compile(
            r"include\s*\{\s*(\w+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]"
        ),
        "params": re.compile(
            r"params\.(\w+)\s*=\s*([^\n]+)"
        ),
        "channel_def": re.compile(
            r"(\w+)\s*=\s*Channel\.(from|of|value|fromPath|fromFilePairs)"
        ),
        "tool_in_script": re.compile(
            r"(?:^|\s)(fastqc|fastp|star|salmon|hisat2|bowtie2|bwa|samtools|"
            r"picard|gatk|bcftools|macs2|deeptools|multiqc|featureCounts|"
            r"kallisto|trimmomatic|cutadapt|bedtools|stringtie)(?:\s|$|\\)",
            re.IGNORECASE
        ),
    }
    
    def __init__(self, router=None):
        """
        Initialize codebase indexer.
        
        Args:
            router: Optional LLM router for semantic analysis
        """
        self.router = router
        self._indices: Dict[str, CodebaseIndex] = {}
    
    def index_directory(
        self,
        directory: str,
        name: str = None,
        recursive: bool = True,
        file_patterns: List[str] = None,
    ) -> CodebaseIndex:
        """
        Index all Nextflow files in a directory.
        
        Args:
            directory: Path to directory to index
            name: Name for the index (default: directory name)
            recursive: Whether to search recursively
            file_patterns: File patterns to match (default: *.nf, *.config)
            
        Returns:
            CodebaseIndex with all indexed elements
        """
        root_path = Path(directory)
        if not root_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        if name is None:
            name = root_path.name
        
        if file_patterns is None:
            file_patterns = ["*.nf", "*.config", "*.nextflow"]
        
        # Find all matching files
        files = []
        for pattern in file_patterns:
            if recursive:
                files.extend(root_path.rglob(pattern))
            else:
                files.extend(root_path.glob(pattern))
        
        logger.info(f"Found {len(files)} files to index in {directory}")
        
        # Index each file
        all_elements = []
        for file_path in files:
            try:
                elements = self._index_file(file_path)
                all_elements.extend(elements)
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")
        
        # Build relationships
        relationships = self._build_relationships(all_elements)
        
        # Calculate statistics
        statistics = self._calculate_statistics(all_elements)
        
        # Create index
        index = CodebaseIndex(
            name=name,
            root_path=str(root_path),
            elements=all_elements,
            relationships=relationships,
            statistics=statistics,
        )
        
        # Cache index
        self._indices[name] = index
        
        logger.info(
            f"Indexed {len(all_elements)} elements from {len(files)} files"
        )
        
        return index
    
    def index_file(self, file_path: str) -> List[CodeElement]:
        """
        Index a single Nextflow file.
        
        Args:
            file_path: Path to file to index
            
        Returns:
            List of indexed CodeElements
        """
        return self._index_file(Path(file_path))
    
    def _index_file(self, file_path: Path) -> List[CodeElement]:
        """Index a single file."""
        content = file_path.read_text()
        elements = []
        
        # Index processes
        for match in self.PATTERNS["process"].finditer(content):
            process_name = match.group(1)
            process_body = match.group(2)
            line_number = content[:match.start()].count("\n") + 1
            
            element = self._parse_process(
                process_name,
                process_body,
                str(file_path),
                line_number,
            )
            elements.append(element)
        
        # Index workflows
        for match in self.PATTERNS["workflow"].finditer(content):
            workflow_name = match.group(1) or "main"
            workflow_body = match.group(2)
            line_number = content[:match.start()].count("\n") + 1
            
            element = self._parse_workflow(
                workflow_name,
                workflow_body,
                str(file_path),
                line_number,
            )
            elements.append(element)
        
        # Index includes/imports
        for match in self.PATTERNS["include"].finditer(content):
            import_name = match.group(1)
            import_path = match.group(2)
            line_number = content[:match.start()].count("\n") + 1
            
            elements.append(CodeElement(
                element_type=CodeElementType.IMPORT,
                name=import_name,
                file_path=str(file_path),
                line_number=line_number,
                content=match.group(0),
                metadata={"import_path": import_path},
            ))
        
        # Index parameters
        for match in self.PATTERNS["params"].finditer(content):
            param_name = match.group(1)
            param_value = match.group(2).strip()
            line_number = content[:match.start()].count("\n") + 1
            
            elements.append(CodeElement(
                element_type=CodeElementType.PARAMETER,
                name=param_name,
                file_path=str(file_path),
                line_number=line_number,
                content=match.group(0),
                metadata={"default_value": param_value},
            ))
        
        # Index channel definitions
        for match in self.PATTERNS["channel_def"].finditer(content):
            channel_name = match.group(1)
            channel_type = match.group(2)
            line_number = content[:match.start()].count("\n") + 1
            
            elements.append(CodeElement(
                element_type=CodeElementType.CHANNEL,
                name=channel_name,
                file_path=str(file_path),
                line_number=line_number,
                content=match.group(0),
                metadata={"channel_type": channel_type},
            ))
        
        return elements
    
    def _parse_process(
        self,
        name: str,
        body: str,
        file_path: str,
        line_number: int,
    ) -> CodeElement:
        """Parse a process definition."""
        inputs = []
        outputs = []
        container = None
        tools = []
        
        # Extract input block
        input_match = self.PATTERNS["input_block"].search(body)
        if input_match:
            input_block = input_match.group(1)
            # Parse input declarations
            for line in input_block.split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    # Extract channel/variable names
                    if "tuple" in line or "path" in line or "val" in line:
                        inputs.append(line)
        
        # Extract output block
        output_match = self.PATTERNS["output_block"].search(body)
        if output_match:
            output_block = output_match.group(1)
            for line in output_block.split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    if "tuple" in line or "path" in line or "val" in line or "emit" in line:
                        outputs.append(line)
        
        # Extract container
        container_match = self.PATTERNS["container"].search(body)
        if container_match:
            container = container_match.group(1)
        
        # Detect tools used in script
        for tool_match in self.PATTERNS["tool_in_script"].finditer(body):
            tool = tool_match.group(1).lower()
            if tool not in tools:
                tools.append(tool)
        
        # Generate description
        description = f"Process {name}"
        if tools:
            description += f" using {', '.join(tools)}"
        
        return CodeElement(
            element_type=CodeElementType.PROCESS,
            name=name,
            file_path=file_path,
            line_number=line_number,
            content=f"process {name} {{{body}\n}}",
            description=description,
            inputs=inputs,
            outputs=outputs,
            container=container,
            tools=tools,
        )
    
    def _parse_workflow(
        self,
        name: str,
        body: str,
        file_path: str,
        line_number: int,
    ) -> CodeElement:
        """Parse a workflow definition."""
        # Extract process calls
        dependencies = []
        process_call_pattern = re.compile(r"(\w+)\s*\(")
        for match in process_call_pattern.finditer(body):
            call_name = match.group(1)
            # Filter out keywords and common functions
            if call_name not in ["if", "else", "for", "while", "Channel", "file", "tuple", "val", "path"]:
                if call_name not in dependencies:
                    dependencies.append(call_name)
        
        return CodeElement(
            element_type=CodeElementType.WORKFLOW,
            name=name,
            file_path=file_path,
            line_number=line_number,
            content=f"workflow {name} {{{body}\n}}",
            description=f"Workflow {name} calling {len(dependencies)} processes",
            dependencies=dependencies,
        )
    
    def _build_relationships(
        self,
        elements: List[CodeElement],
    ) -> Dict[str, List[str]]:
        """Build relationship graph between elements."""
        relationships = {}
        
        # Map process names to elements
        processes = {e.name: e for e in elements if e.element_type == CodeElementType.PROCESS}
        
        # Build workflow -> process relationships
        for e in elements:
            if e.element_type == CodeElementType.WORKFLOW:
                relationships[e.name] = []
                for dep in e.dependencies:
                    if dep in processes:
                        relationships[e.name].append(dep)
        
        # Build process -> tool relationships
        for e in elements:
            if e.element_type == CodeElementType.PROCESS:
                relationships[e.name] = e.tools
        
        return relationships
    
    def _calculate_statistics(
        self,
        elements: List[CodeElement],
    ) -> Dict[str, int]:
        """Calculate statistics for the index."""
        stats = {
            "total_elements": len(elements),
            "processes": 0,
            "workflows": 0,
            "imports": 0,
            "parameters": 0,
            "channels": 0,
            "unique_tools": 0,
            "unique_containers": 0,
        }
        
        tools: Set[str] = set()
        containers: Set[str] = set()
        
        for e in elements:
            if e.element_type == CodeElementType.PROCESS:
                stats["processes"] += 1
                tools.update(e.tools)
                if e.container:
                    containers.add(e.container)
            elif e.element_type == CodeElementType.WORKFLOW:
                stats["workflows"] += 1
            elif e.element_type == CodeElementType.IMPORT:
                stats["imports"] += 1
            elif e.element_type == CodeElementType.PARAMETER:
                stats["parameters"] += 1
            elif e.element_type == CodeElementType.CHANNEL:
                stats["channels"] += 1
        
        stats["unique_tools"] = len(tools)
        stats["unique_containers"] = len(containers)
        
        return stats
    
    def get_index(self, name: str) -> Optional[CodebaseIndex]:
        """Get a cached index by name."""
        return self._indices.get(name)
    
    def list_indices(self) -> List[str]:
        """List all cached indices."""
        return list(self._indices.keys())
    
    def format_for_prompt(
        self,
        index: CodebaseIndex,
        include_content: bool = False,
        max_elements: int = 10,
    ) -> str:
        """
        Format index for inclusion in LLM prompts.
        
        Args:
            index: Index to format
            include_content: Whether to include full content
            max_elements: Maximum elements to include
            
        Returns:
            Formatted string for prompt inclusion
        """
        lines = [
            f"## Codebase Index: {index.name}",
            f"Path: {index.root_path}",
            "",
            "### Statistics",
        ]
        
        for key, value in index.statistics.items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        lines.extend(["", "### Processes"])
        
        processes = index.get_processes()[:max_elements]
        for p in processes:
            lines.append(f"\n**{p.name}**")
            lines.append(f"- Tools: {', '.join(p.tools) if p.tools else 'unknown'}")
            lines.append(f"- Container: {p.container or 'not specified'}")
            
            if p.inputs:
                lines.append(f"- Inputs: {len(p.inputs)} declarations")
            if p.outputs:
                lines.append(f"- Outputs: {len(p.outputs)} declarations")
            
            if include_content:
                lines.append("```nextflow")
                lines.append(p.content[:500])
                if len(p.content) > 500:
                    lines.append("... (truncated)")
                lines.append("```")
        
        if len(index.get_processes()) > max_elements:
            lines.append(f"\n... and {len(index.get_processes()) - max_elements} more processes")
        
        return "\n".join(lines)
    
    def find_similar_processes(
        self,
        query: str,
        index: CodebaseIndex = None,
        limit: int = 5,
    ) -> List[CodeElement]:
        """
        Find processes similar to a query.
        
        Args:
            query: Search query (tool name, description, etc.)
            index: Index to search (or search all cached)
            limit: Maximum results
            
        Returns:
            List of similar processes
        """
        results = []
        
        indices_to_search = [index] if index else list(self._indices.values())
        
        for idx in indices_to_search:
            matches = idx.search(query)
            # Only include processes
            matches = [m for m in matches if m.element_type == CodeElementType.PROCESS]
            results.extend(matches)
        
        return results[:limit]
    
    def get_tool_usage_examples(
        self,
        tool: str,
        index: CodebaseIndex = None,
    ) -> List[CodeElement]:
        """
        Get examples of how a tool is used.
        
        Args:
            tool: Tool name to search for
            index: Index to search (or search all cached)
            
        Returns:
            List of processes using the tool
        """
        results = []
        
        indices_to_search = [index] if index else list(self._indices.values())
        
        for idx in indices_to_search:
            matches = idx.get_by_tool(tool)
            results.extend(matches)
        
        return results
