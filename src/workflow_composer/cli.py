#!/usr/bin/env python3
"""
BioPipelines Workflow Composer CLI
==================================

Command-line interface for the AI Workflow Composer.

Usage:
    biocomposer generate "RNA-seq differential expression, mouse"
    biocomposer chat --llm ollama
    biocomposer tools search star
    biocomposer modules list
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflow_composer import Composer, Config
from workflow_composer.llm import get_llm, list_providers, check_providers
from workflow_composer.llm import Strategy, get_orchestrator, load_profile


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def setup_strategy(strategy_arg: Optional[str] = None):
    """
    Configure LLM strategy based on CLI argument.
    
    Args:
        strategy_arg: Strategy name, profile name, or None for auto
        
    Returns:
        Configured ModelOrchestrator
    """
    if not strategy_arg:
        # Auto-detect
        return get_orchestrator(strategy=Strategy.AUTO)
    
    # Check if it's a Strategy enum value
    strategy_name = strategy_arg.upper().replace("-", "_")
    try:
        strategy = Strategy[strategy_name]
        return get_orchestrator(strategy=strategy)
    except KeyError:
        pass
    
    # Try as profile name
    try:
        config = load_profile(strategy_arg)
        return get_orchestrator(strategy=config.strategy)
    except FileNotFoundError:
        print(f"⚠ Unknown strategy/profile: {strategy_arg}")
        print("  Available strategies: LOCAL_ONLY, LOCAL_FIRST, CLOUD_ONLY, ENSEMBLE, AUTO")
        print("  Available profiles: t4_hybrid, t4_local_only, development, cloud_only")
        sys.exit(1)


def cmd_generate(args):
    """Generate a workflow from description."""
    setup_logging(args.verbose)
    
    # Initialize composer
    llm = get_llm(args.llm, args.model) if args.llm else None
    composer = Composer(llm=llm)
    
    # Check if using multi-agent system
    if args.agents:
        print("🤖 Using multi-agent generation system...")
        print("   Agents: Planner → CodeGen → Validator → Doc → QC\n")
        
        if args.streaming:
            # Streaming mode shows progress in real-time
            import asyncio
            asyncio.run(_generate_with_agents_streaming(composer, args))
        else:
            # Synchronous mode
            workflow = composer.generate_with_agents(
                args.description,
                output_dir=args.output
            )
            
            if args.output:
                print(f"\n✓ Workflow saved to: {args.output}")
            else:
                print(f"\n✓ Workflow generated: {workflow.name}")
                if args.show:
                    print("\n" + "="*60)
                    print("main.nf:")
                    print("="*60)
                    print(workflow.main_nf)
    else:
        # Standard generation
        workflow = composer.generate(
            args.description,
            output_dir=args.output,
            auto_create_modules=not args.no_auto_create
        )
        
        if args.output:
            print(f"\n✓ Workflow saved to: {args.output}")
        else:
            print(f"\n✓ Workflow generated: {workflow.name}")
            if args.show:
                print("\n" + "="*60)
                print("main.nf:")
                print("="*60)
                print(workflow.main_nf)


async def _generate_with_agents_streaming(composer, args):
    """Helper to run streaming multi-agent generation."""
    from .agents.specialists import WorkflowState
    
    state_emojis = {
        WorkflowState.IDLE: "⏳",
        WorkflowState.PLANNING: "📋",
        WorkflowState.GENERATING: "🔧",
        WorkflowState.VALIDATING: "🔍",
        WorkflowState.FIXING: "🔨",
        WorkflowState.DOCUMENTING: "📝",
        WorkflowState.COMPLETE: "✅",
        WorkflowState.FAILED: "❌"
    }
    
    workflow = None
    async for update in composer.generate_with_agents_streaming(
        args.description,
        output_dir=args.output
    ):
        state = update.get("state", WorkflowState.IDLE)
        emoji = state_emojis.get(state, "🔄")
        
        if "message" in update:
            print(f"{emoji} [{state.name}] {update['message']}")
        
        if state == WorkflowState.COMPLETE and "result" in update:
            workflow = update["result"]
    
    if workflow:
        if args.output:
            print(f"\n✓ Workflow saved to: {args.output}")
        else:
            print(f"\n✓ Workflow generated: {workflow.name}")
            if args.show:
                print("\n" + "="*60)
                print("main.nf:")
                print("="*60)
                print(workflow.main_nf)


def cmd_chat(args):
    """Interactive chat mode."""
    setup_logging(args.verbose)
    
    # Initialize composer
    llm = get_llm(args.llm, args.model) if args.llm else None
    composer = Composer(llm=llm)
    
    print("BioPipelines Workflow Composer - Interactive Mode")
    print(f"Using LLM: {composer.llm}")
    print("Type 'quit' to exit, 'help' for commands\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "help":
            print("""
Commands:
  generate <description>  - Generate a workflow
  parse <description>     - Parse intent only
  tools <name>           - Search for tools
  modules                - List available modules
  stats                  - Show system stats
  switch <provider>      - Switch LLM provider
  quit                   - Exit
""")
            continue
        
        if user_input.lower().startswith("switch "):
            provider = user_input[7:].strip()
            composer.switch_llm(provider)
            print(f"Switched to: {composer.llm}")
            continue
        
        if user_input.lower() == "stats":
            stats = composer.get_stats()
            print(json.dumps(stats, indent=2, default=str))
            continue
        
        if user_input.lower() == "modules":
            modules = composer.module_mapper.list_by_category()
            for cat, mods in modules.items():
                print(f"\n{cat}:")
                for m in mods:
                    print(f"  - {m}")
            continue
        
        if user_input.lower().startswith("tools "):
            query = user_input[6:].strip()
            matches = composer.tool_selector.fuzzy_search(query)
            print(f"\nTools matching '{query}':")
            for match in matches[:10]:
                print(f"  - {match.tool.name} ({match.tool.container}) - score: {match.score:.2f}")
            continue
        
        if user_input.lower().startswith("parse "):
            desc = user_input[6:].strip()
            intent = composer.parse_intent(desc)
            print(f"\nParsed Intent:")
            print(json.dumps(intent.to_dict(), indent=2))
            continue
        
        # Default: try to generate workflow
        print("\nAssistant: Analyzing your request...")
        
        try:
            # Check readiness first
            readiness = composer.check_readiness(user_input)
            
            if not readiness["ready"]:
                print("Issues detected:")
                for issue in readiness["issues"]:
                    print(f"  - {issue}")
                continue
            
            if readiness["warnings"]:
                print("Warnings:")
                for warning in readiness["warnings"]:
                    print(f"  - {warning}")
            
            print(f"\nTools found: {readiness['tools_found']}")
            print(f"Modules found: {readiness['modules_found']}")
            
            if readiness["modules_missing"]:
                print(f"Missing modules: {', '.join(readiness['modules_missing'])}")
            
            proceed = input("\nGenerate workflow? (y/n): ").strip().lower()
            if proceed == 'y':
                workflow = composer.generate(user_input)
                output_dir = input("Save to directory (or Enter to skip): ").strip()
                if output_dir:
                    workflow.save(output_dir)
                    print(f"✓ Saved to {output_dir}")
                else:
                    print("✓ Workflow generated (not saved)")
        
        except Exception as e:
            print(f"Error: {e}")


def cmd_tools(args):
    """Search or list tools."""
    setup_logging(args.verbose)
    
    composer = Composer()
    
    if args.search:
        matches = composer.tool_selector.fuzzy_search(args.search)
        print(f"Tools matching '{args.search}':")
        for match in matches[:20]:
            print(f"  {match.tool.name:20} ({match.tool.container:15}) score: {match.score:.2f}")
    
    elif args.container:
        tools = composer.tool_selector.get_tools_in_container(args.container)
        print(f"Tools in {args.container}:")
        for tool in sorted(tools, key=lambda t: t.name)[:50]:
            print(f"  {tool.name}")
        if len(tools) > 50:
            print(f"  ... and {len(tools) - 50} more")
    
    else:
        stats = composer.tool_selector.get_stats()
        print(f"Total tools: {stats['total_tools']}")
        print(f"Containers: {stats['containers']}")
        print("\nTools per container:")
        for name, count in stats['tools_per_container'].items():
            print(f"  {name}: {count}")


def cmd_modules(args):
    """List or search modules."""
    setup_logging(args.verbose)
    
    composer = Composer()
    
    if args.list:
        modules = composer.module_mapper.list_by_category()
        for category, mods in sorted(modules.items()):
            print(f"\n{category}:")
            for mod in sorted(mods):
                print(f"  - {mod}")
    
    elif args.find:
        module = composer.module_mapper.find_module(args.find)
        if module:
            print(f"Module: {module.name}")
            print(f"Path: {module.path}")
            print(f"Container: {module.container}")
            print(f"Processes: {', '.join(module.processes)}")
        else:
            print(f"Module not found: {args.find}")
    
    else:
        print(f"Total modules: {len(composer.module_mapper.modules)}")
        modules = composer.module_mapper.list_by_category()
        for category, mods in sorted(modules.items()):
            print(f"  {category}: {len(mods)}")


def cmd_agents(args):
    """Show or test multi-agent system."""
    setup_logging(args.verbose)
    
    from .agents.specialists import (
        SupervisorAgent, PlannerAgent, CodeGenAgent, 
        ValidatorAgent, DocAgent, QCAgent, WorkflowState
    )
    
    if args.status:
        # Show status of all agents
        print("\n🤖 Multi-Agent System Status\n")
        print("=" * 50)
        
        agents_info = [
            ("SupervisorAgent", "Coordinator", "Orchestrates all specialist agents"),
            ("PlannerAgent", "Planner", "Designs workflow from NL query"),
            ("CodeGenAgent", "Code Generator", "Generates Nextflow DSL2 code"),
            ("ValidatorAgent", "Validator", "Static analysis + LLM review"),
            ("DocAgent", "Documentation", "Creates README, DAGs, parameter docs"),
            ("QCAgent", "Quality Control", "Checks against analysis-type thresholds"),
        ]
        
        print(f"{'Agent':<18} {'Role':<16} {'Description'}")
        print("-" * 50)
        for name, role, desc in agents_info:
            print(f"{name:<18} {role:<16} {desc}")
        
        print("\n📋 Workflow States:")
        print("-" * 30)
        state_flow = ["IDLE", "PLANNING", "GENERATING", "VALIDATING", 
                      "FIXING", "DOCUMENTING", "COMPLETE/FAILED"]
        print(" → ".join(state_flow))
        
        print("\n✅ Multi-agent system is available")
    
    elif args.test:
        # Quick test with a simple query
        print("\n🧪 Testing Multi-Agent System...\n")
        
        test_query = args.test
        print(f"Query: {test_query}\n")
        
        # Test PlannerAgent
        print("Testing PlannerAgent...")
        planner = PlannerAgent()
        try:
            plan = planner.plan_sync(test_query)
            print(f"  ✓ Generated plan with {len(plan.steps)} steps")
            print(f"  Analysis type: {plan.analysis_type}")
            for step in plan.steps[:3]:
                print(f"    - {step.name}: {step.tools}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print("\n✅ Agent test complete")
    
    else:
        # Default: show help
        print("""
🤖 Multi-Agent Generation System
================================

The multi-agent system uses specialized AI agents to generate
high-quality, validated Nextflow workflows.

Agent Pipeline:
  1. PlannerAgent    - Analyzes query, designs workflow structure
  2. CodeGenAgent    - Generates Nextflow DSL2 code  
  3. ValidatorAgent  - Static analysis + LLM code review
  4. DocAgent        - Creates README and documentation
  5. QCAgent         - Quality control checks

Usage:
  biocomposer generate --agents "RNA-seq differential expression"
  biocomposer generate --agents --streaming "ChIP-seq analysis"
  biocomposer agents --status
  biocomposer agents --test "Simple alignment workflow"
""")


def cmd_providers(args):
    """List and check LLM providers."""
    setup_logging(args.verbose)
    
    print("Available LLM Providers:")
    providers = list_providers()
    for name, cls in providers.items():
        print(f"  {name}: {cls}")
    
    if args.check:
        print("\nAvailability check:")
        status = check_providers()
        for name, available in status.items():
            status_str = "✓" if available else "✗"
            print(f"  {status_str} {name}")


def cmd_ui(args):
    """Launch web UI."""
    setup_logging(args.verbose)
    
    ui_type = args.type or "gradio"
    
    print(f"\n🧬 Launching BioPipelines Web UI ({ui_type})...\n")
    
    if ui_type == "gradio":
        try:
            from workflow_composer.web.gradio_app import main as gradio_main
            # Override sys.argv for gradio
            import sys
            sys.argv = ["gradio_app"]
            if args.port:
                sys.argv.extend(["--port", str(args.port)])
            if args.host:
                sys.argv.extend(["--host", args.host])
            if args.share:
                sys.argv.append("--share")
            gradio_main()
        except ImportError as e:
            print(f"Error: Gradio not installed. Run: pip install gradio")
            print(f"Details: {e}")
            sys.exit(1)
    
    elif ui_type == "flask":
        try:
            from workflow_composer.web.app import main as flask_main
            import sys
            sys.argv = ["app"]
            if args.port:
                sys.argv.extend(["--port", str(args.port)])
            if args.host:
                sys.argv.extend(["--host", args.host])
            flask_main()
        except ImportError as e:
            print(f"Error: Flask not installed. Run: pip install flask")
            sys.exit(1)
    
    elif ui_type == "api":
        try:
            from workflow_composer.web.api import main as api_main
            import sys
            sys.argv = ["api"]
            if args.port:
                sys.argv.extend(["--port", str(args.port)])
            if args.host:
                sys.argv.extend(["--host", args.host])
            api_main()
        except ImportError as e:
            print(f"Error: FastAPI not installed. Run: pip install fastapi uvicorn")
            sys.exit(1)
    
    else:
        print(f"Unknown UI type: {ui_type}")
        print("Available: gradio, flask, api")
        sys.exit(1)


def cmd_strategy(args):
    """Manage LLM routing strategies."""
    setup_logging(args.verbose)
    
    if args.list:
        # List available strategies and profiles
        print("\n📋 LLM Routing Strategies\n")
        print("=" * 60)
        
        print("\n🔹 Strategy Modes:")
        strategies = [
            ("LOCAL_ONLY", "Use only local T4 vLLM servers"),
            ("LOCAL_FIRST", "Try local first, fallback to cloud"),
            ("CLOUD_ONLY", "Use only cloud APIs (OpenAI, Anthropic, etc.)"),
            ("ENSEMBLE", "Consult multiple models for consensus"),
            ("CASCADE", "Try providers in sequence until success"),
            ("AUTO", "Auto-detect best strategy from resources"),
        ]
        for name, desc in strategies:
            print(f"  {name:<15} - {desc}")
        
        print("\n🔹 Configuration Profiles:")
        from pathlib import Path
        profile_dir = Path(__file__).parent.parent.parent / "config" / "strategies"
        if profile_dir.exists():
            for profile_file in sorted(profile_dir.glob("*.yaml")):
                name = profile_file.stem
                print(f"  {name:<15} - config/strategies/{name}.yaml")
        else:
            print("  (No profiles found in config/strategies/)")
        
        print("\n💡 Usage:")
        print("  biocomposer generate --strategy LOCAL_FIRST 'RNA-seq analysis'")
        print("  biocomposer generate --strategy t4_hybrid 'ChIP-seq pipeline'")
        print("")
    
    elif args.check:
        # Check current resources and recommend strategy
        print("\n🔍 Checking LLM Resources...\n")
        
        from workflow_composer.llm import ResourceDetector
        
        detector = ResourceDetector()
        status = detector.detect()
        
        # Count healthy vLLM endpoints
        vllm_healthy = len([v for v in status.vllm_endpoints.values() if v])
        vllm_total = len(status.vllm_endpoints)
        
        print(f"Deployment Mode: {status.deployment_mode}")
        print(f"SLURM Available: {'✓' if status.slurm_available else '✗'}")
        print(f"vLLM Servers:    {vllm_healthy}/{vllm_total}")
        print(f"Cloud APIs:      {', '.join(status.available_cloud_apis) or 'None'}")
        
        recommended = detector.get_best_strategy()
        print(f"\n✨ Recommended Strategy: {recommended}")
        
        if status.available_models:
            print("\n🟢 Available local models:")
            for name in status.available_models:
                url = status.vllm_urls.get(name, "unknown")
                print(f"   - {name}: {url}")
    
    elif args.test:
        # Test strategy with a simple query
        print(f"\n🧪 Testing Strategy: {args.test}\n")
        
        import asyncio
        
        orch = setup_strategy(args.test)
        print(f"  Strategy: {orch.strategy.value}")
        print(f"  Profile:  {orch.get_current_profile()}")
        print(f"  Local:    {'available' if orch.local.is_available() else 'unavailable'}")
        print(f"  Cloud:    {'available' if orch.cloud.is_available() else 'unavailable'}")
        
        if args.query:
            print(f"\n  Running test query...")
            try:
                async def test():
                    response = await orch.complete(args.query)
                    print(f"\n  Provider: {response.provider}")
                    print(f"  Model:    {response.model}")
                    print(f"  Latency:  {response.latency_ms:.0f}ms")
                    print(f"  Response: {response.content[:200]}...")
                
                asyncio.run(test())
                print("\n  ✓ Test passed")
            except Exception as e:
                print(f"\n  ✗ Test failed: {e}")
    
    else:
        # Default: show help
        print("""
📋 LLM Routing Strategy Management
===================================

Strategies control how requests are routed between local T4 vLLM servers
and cloud APIs (OpenAI, Anthropic, etc.).

Commands:
  biocomposer strategy --list          List available strategies and profiles
  biocomposer strategy --check         Check resources and get recommendation
  biocomposer strategy --test LOCAL_FIRST --query "Hello"
                                       Test a strategy with optional query

Usage with other commands:
  biocomposer generate --strategy t4_hybrid "RNA-seq analysis"
  biocomposer chat --strategy LOCAL_ONLY
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="biocomposer",
        description="BioPipelines AI Workflow Composer"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a workflow")
    gen_parser.add_argument("description", help="Natural language description")
    gen_parser.add_argument("-o", "--output", help="Output directory")
    gen_parser.add_argument("-l", "--llm", help="LLM provider (ollama, openai, anthropic)")
    gen_parser.add_argument("-m", "--model", help="Model name")
    gen_parser.add_argument("-s", "--strategy", help="LLM strategy (LOCAL_ONLY, LOCAL_FIRST, t4_hybrid, etc.)")
    gen_parser.add_argument("--no-auto-create", action="store_true", help="Don't auto-create missing modules")
    gen_parser.add_argument("--show", action="store_true", help="Show generated code")
    gen_parser.add_argument("--agents", action="store_true", 
                           help="Use multi-agent system (Planner→CodeGen→Validator→Doc→QC)")
    gen_parser.add_argument("--streaming", action="store_true",
                           help="Show real-time progress (requires --agents)")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument("-l", "--llm", help="LLM provider")
    chat_parser.add_argument("-m", "--model", help="Model name")
    chat_parser.add_argument("-s", "--strategy", help="LLM strategy (LOCAL_ONLY, LOCAL_FIRST, t4_hybrid, etc.)")
    chat_parser.set_defaults(func=cmd_chat)
    
    # Tools command
    tools_parser = subparsers.add_parser("tools", help="Search tools")
    tools_parser.add_argument("-s", "--search", help="Search query")
    tools_parser.add_argument("-c", "--container", help="List tools in container")
    tools_parser.set_defaults(func=cmd_tools)
    
    # Modules command
    modules_parser = subparsers.add_parser("modules", help="List modules")
    modules_parser.add_argument("-l", "--list", action="store_true", help="List all modules")
    modules_parser.add_argument("-f", "--find", help="Find specific module")
    modules_parser.set_defaults(func=cmd_modules)
    
    # Providers command
    prov_parser = subparsers.add_parser("providers", help="List LLM providers")
    prov_parser.add_argument("-c", "--check", action="store_true", help="Check availability")
    prov_parser.set_defaults(func=cmd_providers)
    
    # Agents command
    agents_parser = subparsers.add_parser("agents", help="Multi-agent system info and testing")
    agents_parser.add_argument("--status", action="store_true", help="Show agent system status")
    agents_parser.add_argument("--test", metavar="QUERY", help="Test agents with a query")
    agents_parser.set_defaults(func=cmd_agents)
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch web UI")
    ui_parser.add_argument("-t", "--type", choices=["gradio", "flask", "api"], 
                          default="gradio", help="UI type (default: gradio)")
    ui_parser.add_argument("-p", "--port", type=int, help="Port number")
    ui_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    ui_parser.add_argument("--share", action="store_true", help="Create public link (Gradio only)")
    ui_parser.set_defaults(func=cmd_ui)
    
    # Strategy command
    strat_parser = subparsers.add_parser("strategy", help="Manage LLM routing strategies")
    strat_parser.add_argument("-l", "--list", action="store_true", help="List available strategies and profiles")
    strat_parser.add_argument("-c", "--check", action="store_true", help="Check resources and recommend strategy")
    strat_parser.add_argument("-t", "--test", metavar="STRATEGY", help="Test a strategy")
    strat_parser.add_argument("-q", "--query", metavar="TEXT", help="Query for testing (with --test)")
    strat_parser.set_defaults(func=cmd_strategy)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
