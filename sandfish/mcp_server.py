"""SandFish MCP server: run and inspect simulations from an MCP client.

Stdio transport, built on the official Python MCP SDK (the ``[mcp]``
extra: ``pip install -e '.[mcp]'``). Tool names carry an ``sf_`` prefix, like OMPA's ``ao_``
and CognitionOS's ``cos_``.

Usage::

    sandfish-mcp --vault ./sandfish_vault
    claude mcp add sandfish -- sandfish-mcp --vault /abs/path/to/vault

Tools: sf_create_simulation, sf_start, sf_pause, sf_resume, sf_stop,
sf_status, sf_list, sf_results, sf_report, sf_memory_search,
sf_memory_entity.

Resources: ``sandfish://events`` (recent orchestrator events, newest last)
and ``sandfish://events/{simulation_id}`` (one simulation's events).

Like other stdio MCP servers it has no authentication: it runs as the user
who launched it and only talks to that client over stdin/stdout. The same
resource caps as the HTTP API apply (``SANDFISH_MAX_*`` env vars).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The SandFish MCP server needs the optional 'mcp' extra: "
        "pip install -e '.[mcp]' (from a checkout)"
    ) from exc

from . import __version__
from .core.orchestrator import (
    SimulationConfig,
    SimulationLimitError,
    SimulationLimits,
    SimulationStatus,
    SwarmOrchestrator,
)
from .memory.ompa_adapter import OMPAMemoryAdapter

logger = logging.getLogger("sandfish.mcp")

EVENT_LOG_SIZE = 2000
MAX_EVENTS_PER_READ = 500
MAX_SEARCH_LIMIT = 50
MAX_WAIT_SECONDS = 300.0
MAX_REPORT_AGENTS = 50

INSTRUCTIONS = (
    "SandFish runs round-based multi-agent social simulations with OMPA memory. "
    "Typical flow: sf_create_simulation -> sf_start (optionally wait_seconds) -> "
    "sf_status / sf_results / sf_report; sf_memory_search and sf_memory_entity "
    "read what agents remembered. Agent ids look like '<simulation_id>_agent_<n>'. "
    "Read sandfish://events for the event log."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SandfishMCP:
    """Orchestrator, background run tasks and the event log behind the tools."""

    def __init__(
        self,
        vault_path: str,
        *,
        limits: Optional[SimulationLimits] = None,
        enable_semantic: Optional[bool] = None,
    ):
        kwargs: Dict[str, Any] = {}
        if enable_semantic is not None:
            kwargs["enable_semantic"] = enable_semantic
        self.vault_path = os.path.abspath(vault_path)
        self.memory = OMPAMemoryAdapter(vault_path, **kwargs)
        self.orchestrator = SwarmOrchestrator(
            self.memory, limits=limits or SimulationLimits.from_env()
        )
        self.orchestrator.on_event(self._on_orchestrator_event)
        self.tasks: Dict[str, "asyncio.Task[Any]"] = {}
        self.events: Deque[Dict[str, Any]] = deque(maxlen=EVENT_LOG_SIZE)
        self._seq = 0

    # ----- event log -----

    def log_event(self, event_type: str, simulation_id: Optional[str], **data: Any) -> None:
        self._seq += 1
        self.events.append(
            {
                "seq": self._seq,
                "timestamp": _now(),
                "type": event_type,
                "simulation_id": simulation_id,
                "data": data,
            }
        )

    def _on_orchestrator_event(self, event_type: str, data: Dict[str, Any]) -> None:
        payload = {k: v for k, v in data.items() if k != "simulation_id"}
        self.log_event(event_type, data.get("simulation_id"), **payload)

    def recent_events(self, simulation_id: Optional[str] = None, limit: int = MAX_EVENTS_PER_READ) -> List[Dict[str, Any]]:
        items = [e for e in self.events if simulation_id is None or e["simulation_id"] == simulation_id]
        return items[-limit:]

    # ----- helpers -----

    def sim(self, simulation_id: str) -> Dict[str, Any]:
        sim = self.orchestrator.simulations.get(simulation_id)
        if sim is None:
            raise ToolError(f"Unknown simulation_id {simulation_id!r} (see sf_list)")
        return sim

    def status(self, simulation_id: str) -> Dict[str, Any]:
        sim = self.sim(simulation_id)
        status = self.orchestrator.get_simulation_status(simulation_id)
        assert status is not None
        # num_agents counts live agents (0 before the first start).
        status["configured_agents"] = sim["config"].num_agents
        return status

    def relative_source(self, source: str) -> str:
        """Vault-relative note path (don't echo absolute local paths)."""
        try:
            path = os.path.abspath(source)
            if os.path.commonpath([path, self.vault_path]) == self.vault_path:
                return os.path.relpath(path, self.vault_path)
        except ValueError:
            pass
        return source

    def spawn(self, simulation_id: str) -> None:
        for sid in [sid for sid, t in self.tasks.items() if t.done()]:
            del self.tasks[sid]
        existing = self.tasks.get(simulation_id)
        if existing is not None and not existing.done():
            raise ToolError("Simulation is already running")
        try:
            self.orchestrator.reserve_start(simulation_id)
        except SimulationLimitError as exc:
            raise ToolError(str(exc)) from exc
        except RuntimeError as exc:
            raise ToolError(str(exc)) from exc
        task = asyncio.create_task(self.orchestrator.run_simulation(simulation_id))
        self.tasks[simulation_id] = task
        task.add_done_callback(lambda t: self._on_done(simulation_id, t))

    def _on_done(self, simulation_id: str, task: "asyncio.Task[Any]") -> None:
        if task.cancelled():
            self.log_event("run_cancelled", simulation_id)
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Simulation %s run failed: %s", simulation_id, exc)
            self.log_event("run_failed", simulation_id, error=str(exc))
            return
        result = task.result()
        self.log_event(
            "run_ended",
            simulation_id,
            status=result.status.value,
            rounds_completed=result.rounds_completed,
            error=result.error_message,
        )

    async def wait(self, simulation_id: str, seconds: float) -> None:
        task = self.tasks.get(simulation_id)
        if task is None or seconds <= 0:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=min(seconds, MAX_WAIT_SECONDS))
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass  # reported through status / the event log

    def results(self, simulation_id: str, top_agents: int = 10) -> Dict[str, Any]:
        sim = self.sim(simulation_id)
        orch = self.orchestrator
        n = max(0, min(int(top_agents), MAX_REPORT_AGENTS))
        return {
            **self.status(simulation_id),
            "metrics": orch._calculate_metrics(sim),
            **orch.results_summary(simulation_id, n),
        }

    def close(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        self.orchestrator.close()


def _report_markdown(r: Dict[str, Any]) -> str:
    m = r["metrics"]
    lines = [
        f"# Simulation {r['name']} ({r['id']})",
        "",
        f"- Status: **{r['status']}**, {r['rounds_completed']}/{r['total_rounds']} rounds"
        + (f", error: {r['error']}" if r.get("error") else ""),
        f"- Agents: {r['num_agents']} ({', '.join(f'{k}: {v}' for k, v in sorted(r['agent_types'].items())) or 'none'})",
        f"- Actions: {r['total_actions']} total, {r['memory_targeted_actions']} targeted from memory",
        f"- Completion: {m.get('completion_rate', 0.0):.0%}; checkpoints: {int(m.get('checkpoint_count', 0))}",
    ]
    if "llm_calls" in m:
        lines.append(
            f"- LLM: {int(m['llm_calls'])} calls, {int(m.get('llm_input_tokens', 0))} in / "
            f"{int(m.get('llm_output_tokens', 0))} out tokens"
        )
    if r["top_agents"]:
        lines += [
            "",
            "| Agent | Type | Reputation | Following | Posts | Actions |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for a in r["top_agents"]:
            lines.append(
                f"| {a['name']} (`{a['id']}`) | {a['type']} | {a['reputation']:.3f} | "
                f"{a['following']} | {a['posts']} | {a['actions']} |"
            )
    return "\n".join(lines) + "\n"


def build_server(
    vault_path: Optional[str] = None,
    *,
    limits: Optional[SimulationLimits] = None,
    enable_semantic: Optional[bool] = None,
) -> MCPServer:
    """Build the MCP server. ``server.sandfish`` exposes the state for tests."""
    vault = vault_path or os.environ.get("SANDFISH_VAULT_PATH") or os.environ.get(
        "OMPA_VAULT_PATH", "./sandfish_vault"
    )
    app = SandfishMCP(vault, limits=limits, enable_semantic=enable_semantic)
    server = MCPServer(
        name="sandfish",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    server.sandfish = app  # type: ignore[attr-defined]

    @server.tool(
        name="sf_create_simulation",
        description=(
            "Create a simulation (not started). agent_types cycles over the agents; "
            "known types: default, influencer, lurker (and llm when enabled). "
            "start_date (YYYY-MM-DD) dates memory writes in simulated time; "
            "seed makes runs reproducible."
        ),
    )
    async def sf_create_simulation(
        name: str,
        num_agents: int = 10,
        max_rounds: int = 50,
        agent_types: Optional[List[str]] = None,
        seed: Optional[int] = None,
        start_date: Optional[str] = None,
        days_per_round: int = 1,
        description: str = "",
        checkpoint_interval: int = 10,
    ) -> dict[str, Any]:
        if not name.strip() or len(name) > 100:
            raise ToolError("name must be 1-100 characters")
        config = SimulationConfig(
            name=name,
            description=description[:500],
            max_rounds=max_rounds,
            num_agents=num_agents,
            agent_types=list(agent_types) if agent_types is not None else ["default"],
            seed=seed,
            start_date=start_date,
            days_per_round=days_per_round,
            checkpoint_interval=checkpoint_interval,
        )
        try:
            sim_id = app.orchestrator.create_simulation(config)
        except (ValueError, SimulationLimitError) as exc:
            raise ToolError(str(exc)) from exc
        app.log_event("created", sim_id, name=name, num_agents=num_agents, max_rounds=max_rounds)
        return app.status(sim_id)

    @server.tool(
        name="sf_start",
        description=(
            "Start a pending simulation in the background. wait_seconds (max 300) "
            "blocks until it finishes or the time is up; returns the status."
        ),
    )
    async def sf_start(simulation_id: str, wait_seconds: float = 0.0) -> dict[str, Any]:
        app.sim(simulation_id)
        app.spawn(simulation_id)
        app.log_event("started", simulation_id)
        await app.wait(simulation_id, wait_seconds)
        return app.status(simulation_id)

    @server.tool(
        name="sf_pause",
        description="Request a pause. Status is 'pausing' until the in-flight round ends, then 'paused'.",
    )
    async def sf_pause(simulation_id: str) -> dict[str, Any]:
        app.sim(simulation_id)
        if not app.orchestrator.pause_simulation(simulation_id):
            raise ToolError("Simulation is not running")
        app.log_event("pause_requested", simulation_id)
        return app.status(simulation_id)

    @server.tool(
        name="sf_resume",
        description=(
            "Resume a paused simulation from its next round (no round is re-run). "
            "If it is still 'pausing', the pause is cancelled. wait_seconds as in sf_start."
        ),
    )
    async def sf_resume(simulation_id: str, wait_seconds: float = 0.0) -> dict[str, Any]:
        sim = app.sim(simulation_id)
        orch = app.orchestrator
        was_paused = sim["status"] == SimulationStatus.PAUSED
        if was_paused:
            cap = orch.limits.max_concurrent
            if cap is not None and orch.active_count(exclude=simulation_id) >= cap:
                raise ToolError(f"{cap} simulation(s) already running; the limit is {cap}")
        if not orch.resume_simulation(simulation_id):
            raise ToolError("Simulation is not paused")
        if sim["status"] == SimulationStatus.PENDING:
            try:
                app.spawn(simulation_id)
            except ToolError:
                if was_paused and sim["status"] == SimulationStatus.PENDING:
                    sim["status"] = SimulationStatus.PAUSED
                raise
        app.log_event("resumed", simulation_id)
        await app.wait(simulation_id, wait_seconds)
        return app.status(simulation_id)

    @server.tool(
        name="sf_stop",
        description="Stop a running or paused simulation (takes effect after the in-flight round).",
    )
    async def sf_stop(simulation_id: str) -> dict[str, Any]:
        app.sim(simulation_id)
        if not app.orchestrator.stop_simulation(simulation_id):
            raise ToolError("Simulation is not running or paused")
        app.log_event("stop_requested", simulation_id)
        return app.status(simulation_id)

    @server.tool(name="sf_status", description="Status of one simulation.")
    async def sf_status(simulation_id: str) -> dict[str, Any]:
        return app.status(simulation_id)

    @server.tool(name="sf_list", description="All known simulations with their status.")
    async def sf_list() -> dict[str, Any]:
        sims = [app.status(s["id"]) for s in app.orchestrator.list_simulations()]
        return {"simulations": sims, "count": len(sims)}

    @server.tool(
        name="sf_results",
        description=(
            "Results for a simulation (works mid-run too): status, rounds, metrics "
            "(including per-simulation LLM calls/tokens) and the top agents by reputation."
        ),
    )
    async def sf_results(simulation_id: str, top_agents: int = 10) -> dict[str, Any]:
        return app.results(simulation_id, top_agents)

    @server.tool(name="sf_report", description="A short Markdown report of a simulation's results.")
    async def sf_report(simulation_id: str, top_agents: int = 10) -> str:
        return _report_markdown(app.results(simulation_id, top_agents))

    @server.tool(
        name="sf_memory_search",
        description=(
            "Search the shared OMPA memory (dated event notes). reference_date "
            "(YYYY-MM-DD) anchors relative phrases like 'yesterday' in simulated time."
        ),
    )
    async def sf_memory_search(
        query: str, limit: int = 5, reference_date: Optional[str] = None
    ) -> dict[str, Any]:
        if not query.strip():
            raise ToolError("query must not be empty")
        n = max(1, min(int(limit), MAX_SEARCH_LIMIT))
        hits = await asyncio.to_thread(
            app.memory.search, query[:500], n, reference_date=reference_date
        )
        return {
            "query": query,
            "results": [
                {"source": app.relative_source(h.source), "score": round(float(h.score), 4), "content": h.content[:1000]}
                for h in hits
            ],
        }

    @server.tool(
        name="sf_memory_entity",
        description=(
            "Knowledge-graph view of one entity (e.g. an agent id): its attributes, "
            "facts, and incoming/outgoing interactions (follows, liked, commented_on, "
            "reposted), optionally as of a simulated date."
        ),
    )
    async def sf_memory_entity(
        name: str, as_of: Optional[str] = None, limit: int = 50
    ) -> dict[str, Any]:
        n = max(1, min(int(limit), 500))

        def _lookup() -> dict[str, Any]:
            entity = app.memory.get_entity(name)
            facts = app.memory.get_facts(name, as_of=as_of)
            incoming = app.memory.get_interactions(name, direction="incoming", as_of=as_of)
            outgoing = app.memory.get_interactions(name, direction="outgoing", as_of=as_of)
            return {
                "name": name,
                "found": entity is not None or bool(facts),
                "entity": asdict(entity) if entity is not None else None,
                "facts": [asdict(f) for f in facts[:n]],
                "incoming": [asdict(f) for f in incoming[:n]],
                "outgoing": [asdict(f) for f in outgoing[:n]],
                "counts": {
                    "facts": len(facts),
                    "incoming": len(incoming),
                    "outgoing": len(outgoing),
                },
            }

        return await asyncio.to_thread(_lookup)

    @server.resource(
        "sandfish://events",
        name="event-log",
        description=f"The most recent orchestrator events (up to {MAX_EVENTS_PER_READ}), oldest first.",
        mime_type="application/json",
    )
    def events_resource() -> str:
        return json.dumps(app.recent_events(), default=str)

    @server.resource(
        "sandfish://events/{simulation_id}",
        name="simulation-event-log",
        description="One simulation's recent events, oldest first.",
        mime_type="application/json",
    )
    def simulation_events_resource(simulation_id: str) -> str:
        return json.dumps(app.recent_events(simulation_id), default=str)

    return server


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sandfish-mcp", description="SandFish MCP server (stdio transport)"
    )
    parser.add_argument(
        "--vault",
        default=None,
        help="OMPA vault directory (default: $SANDFISH_VAULT_PATH, $OMPA_VAULT_PATH or ./sandfish_vault)",
    )
    parser.add_argument("--version", action="version", version=f"sandfish-mcp {__version__}")
    args = parser.parse_args(argv)
    # stdout carries the protocol; all logging goes to stderr.
    logging.basicConfig(
        level=os.environ.get("SANDFISH_LOG_LEVEL", "WARNING"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = build_server(args.vault)
    try:
        server.run("stdio")
    finally:
        server.sandfish.close()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
