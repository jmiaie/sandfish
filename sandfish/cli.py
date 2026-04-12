"""
SandFish CLI - Command line interface for swarm simulations.

Usage:
    sandfish orchestrator --config config.yaml
    sandfish api --host 0.0.0.0 --port 8000
    sandfish security-audit
    sandfish --version
"""

import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional

from . import __version__
from .core.orchestrator import create_orchestrator, SimulationConfig
from .security.audit import run_security_audit, SecurityAuditor


def print_banner():
    """Print SandFish banner."""
    print("""
    🌵  SandFish - Swarm Intelligence System
    ========================================
    Clean-room multi-agent simulation platform
    Zero dependencies | Security-first | OMPA-native
    """)


def cmd_orchestrator(args):
    """Run simulation orchestrator."""
    print_banner()
    print(f"Starting orchestrator with vault: {args.vault}")
    
    # Create orchestrator
    orchestrator = create_orchestrator(args.vault)
    
    # Load config if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        # TODO: Load YAML config
        config = SimulationConfig(
            name="CLI Simulation",
            description="Started from CLI",
            max_rounds=args.rounds,
            num_agents=args.agents
        )
    else:
        config = SimulationConfig(
            name="CLI Simulation",
            description="Started from CLI",
            max_rounds=args.rounds,
            num_agents=args.agents
        )
    
    # Create and run simulation
    sim_id = orchestrator.create_simulation(config)
    print(f"Created simulation: {sim_id}")
    
    # Run simulation
    if args.dry_run:
        print("Dry run - not executing simulation")
        return
    
    print(f"Running simulation for {config.max_rounds} rounds...")
    result = asyncio.run(orchestrator.run_simulation(sim_id))
    
    print(f"\nSimulation complete!")
    print(f"Status: {result.status.value}")
    print(f"Rounds: {result.rounds_completed}")
    print(f"Metrics: {result.metrics}")


def cmd_api(args):
    """Run API server."""
    print_banner()
    print(f"Starting API server on {args.host}:{args.port}")
    
    try:
        from .api.main import configure_app
        import uvicorn
        
        app = configure_app(vault_path=args.vault, debug=args.debug)
        uvicorn.run(app, host=args.host, port=args.port)
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure FastAPI and Uvicorn are installed:")
        print("  pip install fastapi uvicorn")
        sys.exit(1)


def cmd_security_audit(args):
    """Run security audit."""
    print_banner()
    print("Running security audit...")
    
    auditor = SecurityAuditor(Path(args.path))
    findings = auditor.run_full_audit()
    
    print(f"\nFound {len(findings)} security issues:")
    
    critical = [f for f in findings if f.severity == "CRITICAL"]
    high = [f for f in findings if f.severity == "HIGH"]
    medium = [f for f in findings if f.severity == "MEDIUM"]
    low = [f for f in findings if f.severity == "LOW"]
    
    if critical:
        print(f"\n  🔴 CRITICAL: {len(critical)}")
        for f in critical:
            print(f"    - {f.description}")
    
    if high:
        print(f"\n  🟠 HIGH: {len(high)}")
        for f in high:
            print(f"    - {f.description}")
    
    if medium:
        print(f"\n  🟡 MEDIUM: {len(medium)}")
    
    if low:
        print(f"\n  🟢 LOW: {len(low)}")
    
    if not findings:
        print("\n  ✅ No security issues found!")
    
    # Generate report if requested
    if args.output:
        report = auditor.generate_report()
        Path(args.output).write_text(report)
        print(f"\nReport saved to: {args.output}")


def cmd_version(args):
    """Show version."""
    print(f"SandFish {__version__}")


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sandfish",
        description="SandFish - Multi-agent swarm intelligence system"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Orchestrator command
    orch_parser = subparsers.add_parser(
        "orchestrator",
        help="Run simulation orchestrator"
    )
    orch_parser.add_argument(
        "--vault",
        default="./sandfish_vault",
        help="Path to OMPA vault"
    )
    orch_parser.add_argument(
        "--config",
        help="Path to simulation config YAML"
    )
    orch_parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of simulation rounds"
    )
    orch_parser.add_argument(
        "--agents",
        type=int,
        default=10,
        help="Number of agents"
    )
    orch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create simulation but don't run"
    )
    orch_parser.set_defaults(func=cmd_orchestrator)
    
    # API command
    api_parser = subparsers.add_parser(
        "api",
        help="Run API server"
    )
    api_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    api_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    api_parser.add_argument(
        "--vault",
        default="./sandfish_vault",
        help="Path to OMPA vault"
    )
    api_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    api_parser.set_defaults(func=cmd_api)
    
    # Security audit command
    audit_parser = subparsers.add_parser(
        "security-audit",
        help="Run security audit"
    )
    audit_parser.add_argument(
        "--path",
        default=".",
        help="Path to audit"
    )
    audit_parser.add_argument(
        "--output",
        help="Output file for report"
    )
    audit_parser.set_defaults(func=cmd_security_audit)
    
    # Parse and execute
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
