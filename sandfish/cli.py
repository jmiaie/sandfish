"""
SandFish CLI.

Subcommands:
    sandfish orchestrator [--config CONFIG] [--vault PATH] [--rounds N] [--agents N] [--dry-run]
    sandfish api          [--host HOST] [--port PORT] [--vault PATH] [--debug]
    sandfish security-audit [--path PATH] [--output FILE]
    sandfish --version
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

from . import __version__
from .core.orchestrator import SimulationConfig, create_orchestrator
from .memory.ompa_adapter import default_vault_path
from .security.audit import SecurityAuditor

BANNER = """
    SandFish - Multi-agent swarm intelligence
    ==========================================
    Local-first simulation platform with OMPA-native memory.
"""


def print_banner() -> None:
    print(BANNER)


# ----- Config loading -----


def _load_config_file(path: Path) -> Dict[str, Any]:
    """Load a JSON or YAML config file. YAML requires PyYAML."""
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(raw)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with `pip install pyyaml`."
            ) from exc
        return yaml.safe_load(raw) or {}

    # Fall back: try JSON, then YAML if available.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(raw) or {}
        except ImportError:
            raise ValueError(
                f"Cannot parse {path}: not JSON and PyYAML is not installed."
            )


def _build_simulation_config(
    args: argparse.Namespace,
    config_file_data: Optional[Dict[str, Any]] = None,
) -> SimulationConfig:
    """Merge CLI args with optional config file. CLI flags win."""
    base: Dict[str, Any] = {
        "name": "CLI Simulation",
        "description": "Started from CLI",
        "max_rounds": args.rounds,
        "num_agents": args.agents,
    }
    if config_file_data:
        # Only carry over keys that SimulationConfig actually accepts.
        allowed = {f.name for f in fields(SimulationConfig)}
        for key, value in config_file_data.items():
            if key in allowed:
                base[key] = value

    # Explicit CLI flags should override the config file when the user
    # actually passed them. argparse puts values in the namespace either way,
    # so we use sentinels (defaults) below to detect "user passed it".
    if args.rounds_set:
        base["max_rounds"] = args.rounds
    if args.agents_set:
        base["num_agents"] = args.agents
    if getattr(args, "start_date", None):
        base["start_date"] = args.start_date
    if getattr(args, "agent_types", None):
        base["agent_types"] = [t.strip() for t in args.agent_types.split(",") if t.strip()]
    if getattr(args, "seed", None) is not None:
        base["seed"] = args.seed

    return SimulationConfig(**base)


# ----- Subcommands -----


def cmd_orchestrator(args: argparse.Namespace) -> int:
    print_banner()
    print(f"Vault: {args.vault}")

    config_data: Optional[Dict[str, Any]] = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}", file=sys.stderr)
            return 2
        print(f"Loading config from: {config_path}")
        config_data = _load_config_file(config_path)

    config = _build_simulation_config(args, config_data)
    llm_budget = None
    client = None
    if "llm" in config.agent_types:
        # Opt-in only: provider from --llm-provider or SANDFISH_LLM_PROVIDER.
        from .llm import client_from_env
        from .llm.agent import enable_llm_agents

        client = client_from_env(getattr(args, "llm_provider", None))
        llm_budget = enable_llm_agents(client, max_calls=args.llm_max_calls)
        label = f"{client.name}:{client.model}" if client is not None else "none (heuristic fallback)"
        print(f"LLM agents: {label}, max {args.llm_max_calls} calls per simulation")
    orchestrator = create_orchestrator(args.vault, checkpoint_dir=args.checkpoint_dir)
    try:
        sim_id = orchestrator.create_simulation(config)
    except (ValueError, RuntimeError) as exc:  # includes SimulationLimitError
        print(f"Invalid simulation config: {exc}", file=sys.stderr)
        return 2
    print(f"Created simulation: {sim_id}")

    if args.dry_run:
        print("Dry run: not executing simulation.")
        return 0

    print(f"Running {config.max_rounds} rounds across {config.num_agents} agents...")

    async def _run() -> Any:
        try:
            return await orchestrator.run_simulation(sim_id)
        finally:
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                await aclose()  # the provider's connection pool

    result = asyncio.run(_run())

    print()
    print("Simulation complete:")
    print(f"  Status:   {result.status.value}")
    print(f"  Rounds:   {result.rounds_completed}")
    print(f"  Metrics:  {result.metrics}")
    if llm_budget is not None:
        # Agents are released after the run; read the final-state snapshot.
        fallbacks = sum(
            (state.get("llm") or {}).get("fallbacks", 0)
            for state in result.final_state.values()
        )
        print(
            f"  LLM:      {llm_budget.calls} calls, "
            f"{llm_budget.input_tokens} in / {llm_budget.output_tokens} out tokens, "
            f"{fallbacks} heuristic fallbacks"
        )
    if result.error_message:
        print(f"  Error:    {result.error_message}")
        return 1
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    print_banner()
    try:
        import uvicorn

        from .api.main import check_bind_allowed, configure_app
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Install API extras with: pip install 'fastapi uvicorn[standard]'", file=sys.stderr)
        return 1

    try:
        check_bind_allowed(args.host)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Starting API on http://{args.host}:{args.port} (vault: {args.vault})")
    app = configure_app(vault_path=args.vault, debug=args.debug)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_security_audit(args: argparse.Namespace) -> int:
    print_banner()
    print(f"Running security audit against: {args.path}")

    auditor = SecurityAuditor(Path(args.path))
    findings = auditor.run_full_audit()

    by_sev = {sev: [f for f in findings if f.severity == sev] for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}

    print(f"\nFound {len(findings)} security issue(s):")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        bucket = by_sev[sev]
        if not bucket:
            continue
        print(f"\n  [{sev}] {len(bucket)} finding(s)")
        for f in bucket:
            location = ""
            if f.file_path:
                location = f" ({f.file_path}"
                if f.line_number:
                    location += f":{f.line_number}"
                location += ")"
            print(f"    - {f.description}{location}")

    if not findings:
        print("\n  No security issues found.")

    if args.output:
        Path(args.output).write_text(auditor.generate_report(), encoding="utf-8")
        print(f"\nReport saved to: {args.output}")

    # Non-zero exit code if anything CRITICAL or HIGH was found.
    if by_sev["CRITICAL"] or by_sev["HIGH"]:
        return 2
    return 0


# ----- Argument parser -----


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sandfish",
        description="SandFish - Multi-agent swarm intelligence system",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # orchestrator
    orch = subparsers.add_parser("orchestrator", help="Run a simulation")
    orch.add_argument(
        "--vault",
        default=default_vault_path(),
        help="Path to OMPA vault (default: $SANDFISH_VAULT_PATH, else ./sandfish_vault)",
    )
    orch.add_argument("--config", help="Path to JSON or YAML config file")
    orch.add_argument("--rounds", type=int, default=100, help="Number of simulation rounds")
    orch.add_argument("--agents", type=int, default=10, help="Number of agents")
    orch.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed; runs with the same seed and config are reproducible",
    )
    orch.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to write per-round checkpoints (default: none)",
    )
    orch.add_argument(
        "--start-date",
        help="Simulated date of round 0 (YYYY-MM-DD); each round advances one day",
    )
    orch.add_argument(
        "--agent-types",
        help="Comma-separated agent types, e.g. default,lurker,llm ('llm' is opt-in)",
    )
    orch.add_argument(
        "--llm-provider",
        choices=["anthropic", "gemini", "openrouter", "mock"],
        help="Provider for 'llm' agents (default: $SANDFISH_LLM_PROVIDER)",
    )
    orch.add_argument(
        "--llm-max-calls", type=int, default=200,
        help="Hard cap on LLM calls per simulation (default: 200)",
    )
    orch.add_argument("--dry-run", action="store_true", help="Create the simulation but don't run it")
    orch.set_defaults(func=cmd_orchestrator)

    # api
    api = subparsers.add_parser("api", help="Run the HTTP API server")
    api.add_argument(
        "--host", default="127.0.0.1",
        help="Bind host (default 127.0.0.1; non-loopback requires SANDFISH_API_KEY)",
    )
    api.add_argument("--port", type=int, default=8000, help="Bind port")
    api.add_argument(
        "--vault",
        default=default_vault_path(),
        help="Path to OMPA vault (default: $SANDFISH_VAULT_PATH, else ./sandfish_vault)",
    )
    api.add_argument("--debug", action="store_true", help="Enable verbose error responses")
    api.set_defaults(func=cmd_api)

    # security-audit
    audit = subparsers.add_parser("security-audit", help="Run the security auditor")
    audit.add_argument("--path", default=".", help="Project root to audit")
    audit.add_argument("--output", help="Write a markdown report to this path")
    audit.set_defaults(func=cmd_security_audit)

    return parser


def _annotate_overrides(parser: argparse.ArgumentParser, argv: list) -> None:
    """Track which orchestrator flags were explicitly passed (for config merging)."""
    # We can't easily detect this in argparse without sentinels; do a quick scan.
    # This only matters for orchestrator subcommand.
    pass


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)

    # For the orchestrator subcommand, record which flags were explicitly passed
    # so we can let CLI flags override config-file values.
    args.rounds_set = "--rounds" in raw_argv
    args.agents_set = "--agents" in raw_argv

    if not args.command:
        parser.print_help()
        return 1

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
