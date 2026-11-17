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

Opt-in parameter search (``sandfish.search``): sf_search_evaluators,
sf_search_run, sf_search_ledger, sf_search_summary, sf_search_validate.
Evaluators are looked up by name only (registered in-process or installed
under the ``sandfish.evaluators`` entry-point group; ``toy`` is built in).
Ledgers are written to ``<vault>/search_ledgers/<search_id>.jsonl``.

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
import re
import sys
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
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
from .search import (
    TrialLedger,
    attach_search,
    available_evaluators,
    get_evaluator,
    grid_size,
    register_search_agents,
    run_grid,
    summarize,
    validate_holdout,
    validate_space,
)
from .search.ledger import LedgerIntegrityError

logger = logging.getLogger("sandfish.mcp")

EVENT_LOG_SIZE = 2000
MAX_EVENTS_PER_READ = 500
MAX_SEARCH_LIMIT = 50
MAX_WAIT_SECONDS = 300.0
MAX_REPORT_AGENTS = 50
MAX_LEDGER_ROWS_PER_READ = 500
MAX_LIVE_SEARCHES = 200
DEFAULT_MAX_SEARCH_EVALUATIONS = 10_000
_SEARCH_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

INSTRUCTIONS = (
    "SandFish runs round-based multi-agent social simulations with OMPA memory. "
    "Typical flow: sf_create_simulation -> sf_start (optionally wait_seconds) -> "
    "sf_status / sf_results / sf_report; sf_memory_search and sf_memory_entity "
    "read what agents remembered. Agent ids look like '<simulation_id>_agent_<n>'. "
    "Read sandfish://events for the event log. "
    "Opt-in parameter search: sf_search_evaluators -> sf_search_run (strategy "
    "'grid' when the space is small enough to enumerate, else 'swarm') -> "
    "sf_search_summary / sf_search_ledger -> sf_search_validate (held-out test "
    "split). SandFish only orchestrates and logs; scores come from the external "
    "evaluator. Use the ledger's unique-variant count as N for multiple-testing "
    "corrections."
)


def max_search_evaluations(env: Optional[Dict[str, str]] = None) -> int:
    env = os.environ if env is None else env
    raw = str(env.get("SANDFISH_MAX_SEARCH_EVALUATIONS", DEFAULT_MAX_SEARCH_EVALUATIONS)).strip()
    value = int(raw)
    return value if value > 0 else DEFAULT_MAX_SEARCH_EVALUATIONS


@dataclass
class SearchRun:
    search_id: str
    strategy: str
    ledger: TrialLedger
    evaluator: Any
    evaluator_name: str
    space: Dict[str, List[Any]]
    min_n: int = 0
    simulation_id: Optional[str] = None
    task: Optional["asyncio.Task[Any]"] = None  # grid searches
    stop: bool = False


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
        self.searches: Dict[str, SearchRun] = {}
        self.ledger_dir = Path(self.vault_path) / "search_ledgers"

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
        states = orch._aggregate_final_state(sim)
        agents = sorted(
            states.values(),
            key=lambda s: s.get("state", {}).get("reputation", 0.0),
            reverse=True,
        )
        type_counts: Dict[str, int] = {}
        for s in states.values():
            t = s.get("profile", {}).get("type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1
        n = max(0, min(int(top_agents), MAX_REPORT_AGENTS))
        return {
            **self.status(simulation_id),
            "metrics": orch._calculate_metrics(sim),
            "agent_types": type_counts,
            "total_actions": sum(int(s.get("action_count", 0)) for s in states.values()),
            "memory_targeted_actions": sum(
                int(s.get("memory_targeted_actions", 0)) for s in states.values()
            ),
            "top_agents": [
                {
                    "id": s["id"],
                    "name": s.get("profile", {}).get("name"),
                    "type": s.get("profile", {}).get("type"),
                    "reputation": round(s.get("state", {}).get("reputation", 0.0), 4),
                    "energy": round(s.get("state", {}).get("energy", 0.0), 2),
                    "mood": round(s.get("state", {}).get("mood", 0.0), 3),
                    "following": len(s.get("state", {}).get("connections", [])),
                    "posts": len(s.get("state", {}).get("posts_created", [])),
                    "actions": s.get("action_count", 0),
                }
                for s in agents[:n]
            ],
        }

    # ----- parameter search -----

    def add_search(self, run: SearchRun) -> None:
        """Track a live search; forget the oldest finished ones beyond the cap
        (their ledgers stay on disk and remain readable)."""
        self.searches[run.search_id] = run
        for sid in list(self.searches):
            if len(self.searches) <= MAX_LIVE_SEARCHES:
                break
            if sid != run.search_id and not self.search_running(sid):
                del self.searches[sid]

    def search_running(self, search_id: str) -> bool:
        run = self.searches.get(search_id)
        if run is None:
            return False
        if run.task is not None:
            return not run.task.done()
        if run.simulation_id is not None:
            sim = self.orchestrator.simulations.get(run.simulation_id)
            task = self.tasks.get(run.simulation_id)
            if task is not None and not task.done():
                return True
            return sim is not None and sim["status"] in (
                SimulationStatus.PENDING, SimulationStatus.RUNNING, SimulationStatus.PAUSING,
                SimulationStatus.PAUSED)
        return False

    def ledger_path(self, search_id: str) -> Path:
        if not isinstance(search_id, str) or not _SEARCH_ID_RE.fullmatch(search_id):
            raise ToolError("search_id must be 1-64 characters of [A-Za-z0-9_-]")
        return self.ledger_dir / f"{search_id}.jsonl"

    def search_ledger(self, search_id: str) -> TrialLedger:
        """A live search's ledger, or a ledger file left by an earlier server."""
        run = self.searches.get(search_id)
        if run is not None:
            return run.ledger
        path = self.ledger_path(search_id)
        if not path.exists():
            raise ToolError(f"Unknown search_id {search_id!r}")
        try:
            return TrialLedger(path)
        except LedgerIntegrityError as exc:
            raise ToolError(f"Ledger failed verification: {exc}") from exc

    def search_status(self, search_id: str) -> Dict[str, Any]:
        run = self.searches.get(search_id)
        if run is None:
            return {"search_id": search_id, "strategy": None, "status": "on_disk", "live": False}
        out: Dict[str, Any] = {
            "search_id": search_id,
            "strategy": run.strategy,
            "evaluator": run.evaluator_name,
            "space": run.space,
            "grid_size": grid_size(run.space),
            "min_n": run.min_n,
            "live": True,
            "ledger_file": os.path.relpath(self.ledger_path(search_id), self.vault_path),
        }
        if run.simulation_id is not None:
            out["simulation_id"] = run.simulation_id
            if run.simulation_id in self.orchestrator.simulations:
                st = self.status(run.simulation_id)
                out.update(status=st["status"], rounds_completed=st.get("rounds_completed"),
                           error=st.get("error"))
            else:
                out["status"] = "finished"  # evicted from the simulation registry
        elif run.task is not None and not run.task.done():
            out["status"] = "running"
        elif run.task is not None and not run.task.cancelled() and run.task.exception() is not None:
            out.update(status="failed", error=str(run.task.exception()))
        else:
            out["status"] = "stopped" if run.stop else "completed"
        return out

    def close(self) -> None:
        for run in self.searches.values():
            run.stop = True
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

    @server.tool(
        name="sf_search_evaluators",
        description=(
            "Opt-in parameter search: list the external evaluators that can be used by "
            "name (built-in 'toy' plus installed 'sandfish.evaluators' plugins)."
        ),
    )
    async def sf_search_evaluators() -> dict[str, Any]:
        names = available_evaluators()
        info = []
        for n in names:
            item: dict[str, Any] = {"name": n}
            if n == "toy":
                from .search import ToyQuadraticEvaluator

                item["default_space"] = ToyQuadraticEvaluator.DEFAULT_SPACE
                item["description"] = "Synthetic quadratic with split-specific noise (tests and docs)."
            info.append(item)
        return {"evaluators": info, "count": len(info),
                "max_evaluations": max_search_evaluations()}

    @server.tool(
        name="sf_search_run",
        description=(
            "Start a parameter search with a named external evaluator on the train split. "
            "strategy 'grid' evaluates every variant (use it whenever the space is small "
            "enough to enumerate); 'swarm' runs researcher agents for num_agents x max_rounds "
            "evaluations. Both run in the background; wait_seconds (max 300) waits for the "
            "result. space maps parameter -> "
            "list of values (default: the evaluator's default space). Every evaluation, "
            "repeats included, goes to an append-only hash-chained ledger."
        ),
    )
    async def sf_search_run(
        evaluator: str = "toy",
        strategy: str = "swarm",
        space: Optional[Dict[str, List[Any]]] = None,
        evaluator_options: Optional[Dict[str, Any]] = None,
        num_agents: int = 4,
        max_rounds: int = 10,
        seed: Optional[int] = None,
        min_n: int = 0,
        name: str = "search",
        wait_seconds: float = 0.0,
    ) -> dict[str, Any]:
        if strategy not in ("swarm", "grid"):
            raise ToolError("strategy must be 'swarm' or 'grid'")
        if not name.strip() or len(name) > 100:
            raise ToolError("name must be 1-100 characters")
        if min_n < 0:
            raise ToolError("min_n must be >= 0")
        try:
            ev = get_evaluator(evaluator, **(evaluator_options or {}))
        except KeyError as exc:
            raise ToolError(str(exc.args[0])) from exc
        except Exception as exc:  # noqa: BLE001 - plugin factories may raise anything
            raise ToolError(f"Could not build evaluator {evaluator!r}: {exc}") from exc
        if space is None:
            space = getattr(ev, "DEFAULT_SPACE", None)
            if space is None:
                raise ToolError("space is required for this evaluator")
        try:
            space = validate_space(space)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        cap = max_search_evaluations()

        if strategy == "grid":
            size = grid_size(space)
            if size > cap:
                raise ToolError(
                    f"grid has {size} variants; the limit is {cap} (SANDFISH_MAX_SEARCH_EVALUATIONS)"
                )
            search_id = f"grid-{uuid.uuid4().hex[:12]}"
            ledger = TrialLedger(app.ledger_path(search_id))
            run = SearchRun(search_id, "grid", ledger, ev, evaluator, space, min_n)
            run.task = asyncio.create_task(asyncio.to_thread(
                run_grid, space, ev, ledger, max_variants=cap, run_id=search_id,
                min_n=min_n, should_stop=lambda: run.stop,
            ))

            def _grid_done(t: "asyncio.Task[Any]") -> None:
                err = None if t.cancelled() else t.exception()
                app.log_event("search_ended", None, search_id=search_id, trials=len(ledger),
                              error=str(err) if err else None)

            run.task.add_done_callback(_grid_done)
            app.add_search(run)
            app.log_event("search_started", None, search_id=search_id, strategy="grid", evaluator=evaluator)
            if wait_seconds > 0:
                try:
                    await asyncio.wait_for(asyncio.shield(run.task),
                                           timeout=min(wait_seconds, MAX_WAIT_SECONDS))
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    pass  # reported through the status
            summary = await asyncio.to_thread(summarize, ledger, min_n=min_n)
            return {**app.search_status(search_id), **summary}

        if num_agents < 1 or max_rounds < 1:
            raise ToolError("num_agents and max_rounds must be >= 1")
        if num_agents * max_rounds > cap:
            raise ToolError(
                f"num_agents x max_rounds = {num_agents * max_rounds} exceeds the limit of "
                f"{cap} evaluations (SANDFISH_MAX_SEARCH_EVALUATIONS)"
            )
        register_search_agents()
        config = SimulationConfig(
            name=name, num_agents=num_agents, max_rounds=max_rounds, seed=seed,
            agent_types=["researcher_explorer", "researcher_exploiter"], checkpoint_interval=0,
        )
        try:
            sim_id = app.orchestrator.create_simulation(config)
        except (ValueError, SimulationLimitError) as exc:
            raise ToolError(str(exc)) from exc
        ledger = TrialLedger(app.ledger_path(sim_id))
        attach_search(app.orchestrator, sim_id, space=space, evaluator=ev, ledger=ledger, min_n=min_n)
        app.add_search(SearchRun(sim_id, "swarm", ledger, ev, evaluator, space, min_n, sim_id))
        app.spawn(sim_id)
        app.log_event("search_started", sim_id, search_id=sim_id, strategy="swarm", evaluator=evaluator)
        await app.wait(sim_id, wait_seconds)
        summary = await asyncio.to_thread(summarize, ledger, min_n=min_n)
        return {**app.search_status(sim_id), **summary}

    @server.tool(
        name="sf_search_ledger",
        description=(
            "Read rows of a search's trial ledger (oldest first; offset/limit, max 500; "
            "optional split 'train' or 'test'), with exact counts and the hash-chain check."
        ),
    )
    async def sf_search_ledger(
        search_id: str, offset: int = 0, limit: int = 50, split: Optional[str] = None
    ) -> dict[str, Any]:
        if split not in (None, "train", "test"):
            raise ToolError("split must be 'train', 'test' or omitted")
        n = max(1, min(int(limit), MAX_LEDGER_ROWS_PER_READ))
        start = max(0, int(offset))

        def _read() -> dict[str, Any]:
            ledger = app.search_ledger(search_id)
            rows = ledger.rows(start, n, split=split)
            return {
                "search_id": search_id,
                "offset": start,
                "returned": len(rows),
                "rows": rows,
                "counts": ledger.counts(),
                "integrity": ledger.verify(),
            }

        return await asyncio.to_thread(_read)

    @server.tool(
        name="sf_search_summary",
        description=(
            "Summarise a search: exact trial counts (every evaluation, repeats included; "
            "unique variants per split), ledger integrity, best and top train variants with "
            "any test scores, and the N to use for multiple-testing corrections."
        ),
    )
    async def sf_search_summary(search_id: str, top: int = 5) -> dict[str, Any]:
        run = app.searches.get(search_id)
        k = max(1, min(int(top), MAX_REPORT_AGENTS))
        summary = await asyncio.to_thread(
            lambda: summarize(app.search_ledger(search_id), top=k, min_n=run.min_n if run else 0)
        )
        return {**app.search_status(search_id), **summary}

    @server.tool(
        name="sf_search_validate",
        description=(
            "Score finalists on the held-out 'test' split: the top_k train variants (or the "
            "given variants). Test evaluations are logged in the same ledger. Choose finalists "
            "before calling; repeated peeking is visible in the ledger."
        ),
    )
    async def sf_search_validate(
        search_id: str, top_k: int = 3, variants: Optional[List[Dict[str, Any]]] = None
    ) -> dict[str, Any]:
        run = app.searches.get(search_id)
        if run is None:
            raise ToolError("Validation needs a search started by this server (its evaluator is in memory)")
        if app.search_running(search_id):
            raise ToolError("Search is still running; validate after it finishes")
        k = max(1, min(int(top_k), MAX_REPORT_AGENTS))
        if variants is not None:
            if not variants or len(variants) > MAX_REPORT_AGENTS:
                raise ToolError(f"variants must list 1-{MAX_REPORT_AGENTS} variants")
            try:
                for v in variants:
                    validate_space({key: [val] for key, val in v.items()})
            except (ValueError, AttributeError) as exc:
                raise ToolError(f"invalid variant: {exc}") from exc
        results = await asyncio.to_thread(
            validate_holdout, run.ledger, run.evaluator, top_k=k, min_n=run.min_n,
            variants=variants, run_id=search_id,
        )
        app.log_event("search_validated", run.simulation_id, search_id=search_id, finalists=len(results))
        return {"search_id": search_id, "results": results, "counts": run.ledger.counts()}

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
