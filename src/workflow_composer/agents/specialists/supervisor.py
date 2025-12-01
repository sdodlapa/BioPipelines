"""
Supervisor Agent
================

Coordinates specialist agents to generate complete workflows.

Orchestrates:
1. Planner → Workflow design
2. CodeGen → Nextflow generation
3. Validator → Code review with fix loop
4. Docs → Documentation generation
5. QC → Output validation setup
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from .planner import PlannerAgent, WorkflowPlan
from .codegen import CodeGenAgent
from .validator import ValidatorAgent
from .docs import DocAgent
from .qc import QCAgent

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Role identifiers for agents."""
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    CODEGEN = "codegen"
    VALIDATOR = "validator"
    DOCS = "docs"
    QC = "qc"


class WorkflowState(Enum):
    """Workflow generation state."""
    IDLE = "idle"
    PLANNING = "planning"
    GENERATING = "generating"
    VALIDATING = "validating"
    FIXING = "fixing"
    DOCUMENTING = "documenting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class WorkflowResult:
    """Complete result from workflow generation."""
    success: bool
    plan: Optional[WorkflowPlan] = None
    code: Optional[str] = None
    config: Optional[str] = None
    documentation: Optional[str] = None
    validation_passed: bool = False
    validation_issues: List[str] = field(default_factory=list)
    output_files: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message between agents."""
    sender: AgentRole
    receiver: AgentRole
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GenerationContext:
    """Context for workflow generation."""
    query: str
    plan: Optional[WorkflowPlan] = None
    code: Optional[str] = None
    config: Optional[str] = None
    readme: Optional[str] = None
    validation_attempts: int = 0
    max_validation_attempts: int = 3
    messages: List[AgentMessage] = field(default_factory=list)
    state: WorkflowState = WorkflowState.IDLE


class SupervisorAgent:
    """
    Coordinates specialist agents for workflow generation.
    
    Flow:
    1. PlannerAgent designs the workflow
    2. CodeGenAgent generates Nextflow code
    3. ValidatorAgent reviews code (may loop back to CodeGen)
    4. DocAgent creates documentation
    5. QCAgent configures quality checks
    """
    
    def __init__(self, router=None, knowledge_base=None):
        """
        Initialize supervisor with specialist agents.
        
        Args:
            router: LLM provider router for agent operations
            knowledge_base: Optional KnowledgeBase for RAG enhancement
        """
        self.router = router
        self.knowledge_base = knowledge_base
        
        # Initialize specialist agents with knowledge base
        self.planner = PlannerAgent(router, knowledge_base=knowledge_base)
        self.codegen = CodeGenAgent(router)
        self.validator = ValidatorAgent(router, knowledge_base=knowledge_base)
        self.docs = DocAgent(router)
        self.qc = None  # Initialized per-workflow based on analysis type
        
        # State tracking
        self.current_context: Optional[GenerationContext] = None
    
    async def execute(self, query: str, output_dir: str = None) -> WorkflowResult:
        """
        Execute full workflow generation pipeline.
        
        Args:
            query: User query describing desired workflow
            output_dir: Optional directory to write output files
            
        Returns:
            WorkflowResult with generated artifacts
        """
        self.current_context = GenerationContext(query=query)
        
        try:
            # Phase 1: Planning
            self.current_context.state = WorkflowState.PLANNING
            logger.info("Phase 1: Planning workflow...")
            
            plan = await self.planner.create_plan(query)
            self.current_context.plan = plan
            self._log_message(AgentRole.PLANNER, AgentRole.SUPERVISOR, plan)
            
            # Initialize QC agent with analysis type
            self.qc = QCAgent(analysis_type=plan.analysis_type)
            
            # Phase 2: Code Generation
            self.current_context.state = WorkflowState.GENERATING
            logger.info("Phase 2: Generating Nextflow code...")
            
            code = await self.codegen.generate(plan)
            config = self.codegen.generate_config(plan)
            self.current_context.code = code
            self.current_context.config = config
            self._log_message(AgentRole.CODEGEN, AgentRole.SUPERVISOR, {"code": code[:500], "config": config[:200]})
            
            # Phase 3: Validation Loop
            self.current_context.state = WorkflowState.VALIDATING
            logger.info("Phase 3: Validating code...")
            
            code, validation = await self._validation_loop(code, plan)
            self.current_context.code = code
            
            if not validation.valid:
                logger.warning("Validation issues remain after max attempts")
            
            # Phase 4: Documentation
            self.current_context.state = WorkflowState.DOCUMENTING
            logger.info("Phase 4: Generating documentation...")
            
            readme = await self.docs.generate_readme(plan, code)
            self.current_context.readme = readme
            self._log_message(AgentRole.DOCS, AgentRole.SUPERVISOR, readme[:200])
            
            # Phase 5: Completion
            self.current_context.state = WorkflowState.COMPLETE
            logger.info("Workflow generation complete!")
            
            result = WorkflowResult(
                success=True,
                plan=plan,
                code=code,
                config=config,
                documentation=readme,
                validation_passed=validation.valid,
                validation_issues=validation.issues,
                output_files={}
            )
            
            # Write to disk if output_dir provided
            if output_dir:
                result.output_files = self._write_outputs(output_dir, result)
            
            return result
            
        except Exception as e:
            self.current_context.state = WorkflowState.FAILED
            logger.error(f"Workflow generation failed: {e}")
            
            return WorkflowResult(
                success=False,
                plan=self.current_context.plan,
                code=self.current_context.code,
                config=self.current_context.config,
                documentation=self.current_context.readme,
                validation_passed=False,
                validation_issues=[str(e)],
                output_files={}
            )
    
    def execute_sync(self, query: str, output_dir: str = None) -> WorkflowResult:
        """
        Synchronous execution using template-based generation.
        
        Args:
            query: User query describing desired workflow
            output_dir: Optional directory to write output files
            
        Returns:
            WorkflowResult with generated artifacts
        """
        self.current_context = GenerationContext(query=query)
        
        try:
            # Phase 1: Planning
            logger.info("Phase 1: Planning workflow...")
            plan = self.planner.create_plan_sync(query)
            self.current_context.plan = plan
            
            # Initialize QC agent
            self.qc = QCAgent(analysis_type=plan.analysis_type)
            
            # Phase 2: Code Generation
            logger.info("Phase 2: Generating Nextflow code...")
            code = self.codegen.generate_sync(plan)
            config = self.codegen.generate_config(plan)
            self.current_context.code = code
            self.current_context.config = config
            
            # Phase 3: Validation
            logger.info("Phase 3: Validating code...")
            validation = self.validator.validate_sync(code)
            
            # Phase 4: Documentation
            logger.info("Phase 4: Generating documentation...")
            readme = self.docs.generate_readme_sync(plan, code)
            self.current_context.readme = readme
            
            result = WorkflowResult(
                success=True,
                plan=plan,
                code=code,
                config=config,
                documentation=readme,
                validation_passed=validation.valid,
                validation_issues=validation.issues,
                output_files={}
            )
            
            if output_dir:
                result.output_files = self._write_outputs(output_dir, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow generation failed: {e}")
            return WorkflowResult(
                success=False,
                plan=self.current_context.plan,
                code=self.current_context.code,
                config=self.current_context.config,
                documentation=None,
                validation_passed=False,
                validation_issues=[str(e)],
                output_files={}
            )
    
    async def execute_streaming(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute with streaming progress updates.
        
        Args:
            query: User query
            
        Yields:
            Progress updates as dictionaries
        """
        self.current_context = GenerationContext(query=query)
        
        try:
            # Planning
            yield {"phase": "planning", "status": "started", "message": "Designing workflow..."}
            plan = await self.planner.create_plan(query)
            self.current_context.plan = plan
            yield {
                "phase": "planning",
                "status": "complete",
                "plan": {
                    "name": plan.name,
                    "steps": len(plan.steps),
                    "analysis_type": plan.analysis_type
                }
            }
            
            # Initialize QC
            self.qc = QCAgent(analysis_type=plan.analysis_type)
            
            # Code generation
            yield {"phase": "codegen", "status": "started", "message": "Generating Nextflow code..."}
            code = await self.codegen.generate(plan)
            config = self.codegen.generate_config(plan)
            self.current_context.code = code
            self.current_context.config = config
            yield {
                "phase": "codegen",
                "status": "complete",
                "lines": len(code.split('\n'))
            }
            
            # Validation
            yield {"phase": "validation", "status": "started", "message": "Validating code..."}
            code, validation = await self._validation_loop(code, plan)
            self.current_context.code = code
            yield {
                "phase": "validation",
                "status": "complete",
                "valid": validation.valid,
                "issues": len(validation.issues)
            }
            
            # Documentation
            yield {"phase": "documentation", "status": "started", "message": "Generating documentation..."}
            readme = await self.docs.generate_readme(plan, code)
            self.current_context.readme = readme
            yield {
                "phase": "documentation",
                "status": "complete"
            }
            
            # Final result
            yield {
                "phase": "complete",
                "status": "success",
                "result": {
                    "name": plan.name,
                    "code_lines": len(code.split('\n')),
                    "validation_passed": validation.valid
                }
            }
            
        except Exception as e:
            yield {
                "phase": "error",
                "status": "failed",
                "error": str(e)
            }
    
    async def _validation_loop(self, code: str, plan: WorkflowPlan):
        """
        Run validation with fix attempts.
        
        Args:
            code: Initial code
            plan: Workflow plan for context
            
        Returns:
            Tuple of (fixed_code, final_validation)
        """
        ctx = self.current_context
        
        for attempt in range(ctx.max_validation_attempts):
            ctx.validation_attempts = attempt + 1
            
            # Validate
            validation = await self.validator.validate(code)
            self._log_message(AgentRole.VALIDATOR, AgentRole.SUPERVISOR, validation)
            
            if validation.valid:
                return code, validation
            
            # Try to fix
            if attempt < ctx.max_validation_attempts - 1:
                ctx.state = WorkflowState.FIXING
                logger.info(f"Fix attempt {attempt + 1}: {len(validation.issues)} issues")
                
                code = await self.codegen.fix_issues(code, validation.issues)
                self._log_message(AgentRole.CODEGEN, AgentRole.VALIDATOR, "Fixed code")
                
                ctx.state = WorkflowState.VALIDATING
        
        return code, validation
    
    def _log_message(self, sender: AgentRole, receiver: AgentRole, content: Any):
        """Log inter-agent message."""
        if self.current_context:
            self.current_context.messages.append(
                AgentMessage(sender=sender, receiver=receiver, content=content)
            )
    
    def _write_outputs(self, output_dir: str, result: WorkflowResult) -> Dict[str, str]:
        """Write generated files to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Write main.nf
        if result.code:
            main_nf = output_path / "main.nf"
            main_nf.write_text(result.code)
            files["main.nf"] = str(main_nf)
        
        # Write nextflow.config
        if result.config:
            config_file = output_path / "nextflow.config"
            config_file.write_text(result.config)
            files["nextflow.config"] = str(config_file)
        
        # Write README.md
        if result.documentation:
            readme = output_path / "README.md"
            readme.write_text(result.documentation)
            files["README.md"] = str(readme)
        
        # Write plan as JSON
        if result.plan:
            import json
            plan_file = output_path / "workflow_plan.json"
            plan_file.write_text(result.plan.to_json())
            files["workflow_plan.json"] = str(plan_file)
        
        logger.info(f"Wrote {len(files)} files to {output_dir}")
        return files
    
    def get_state(self) -> Dict[str, Any]:
        """Get current generation state."""
        if not self.current_context:
            return {"state": "idle"}
        
        ctx = self.current_context
        return {
            "state": ctx.state.value,
            "query": ctx.query,
            "has_plan": ctx.plan is not None,
            "has_code": ctx.code is not None,
            "validation_attempts": ctx.validation_attempts,
            "messages_count": len(ctx.messages)
        }
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get inter-agent message history."""
        if not self.current_context:
            return []
        
        return [
            {
                "sender": msg.sender.value,
                "receiver": msg.receiver.value,
                "timestamp": msg.timestamp.isoformat(),
                "content_preview": str(msg.content)[:100] if msg.content else None
            }
            for msg in self.current_context.messages
        ]
