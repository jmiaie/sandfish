"""
Swarm Orchestrator for SandFish.

Manages multi-agent simulations with OMPA-native memory.
"""

import asyncio
import inspect
import json
import logging
import random
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ..memory.ompa_adapter import OMPAMemoryAdapter


class SimulationStatus(Enum):
    """Simulation execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


# Statuses that the run loop should treat as "exit immediately".
_TERMINAL_RUN_STATUSES = {SimulationStatus.STOPPED, SimulationStatus.FAILED}

# Statuses after which a simulation will never run again, so its agents can be
# released. PAUSED is deliberately absent: a paused run resumes with the same
# agent objects.
_FINISHED_STATUSES = {
    SimulationStatus.COMPLETED,
    SimulationStatus.STOPPED,
    SimulationStatus.FAILED,
}


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    name: str
    description: str = ""
    max_rounds: int = 100
    num_agents: int = 10
    agent_types: List[str] = field(default_factory=lambda: ["default"])
    seed_data: Dict[str, Any] = field(default_factory=dict)
    enable_logging: bool = True
    checkpoint_interval: int = 10
    # RNG seed for reproducible runs. With a seed set, two runs of the same
    # config produce identical action sequences. Left as None, each run is
    # independently seeded from the OS. Distinct from `seed_data`, which
    # carries starting agent attributes rather than randomness.
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Results from a completed simulation."""
    simulation_id: str
    status: SimulationStatus
    rounds_completed: int
    final_state: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


EventCallback = Callable[[str, Dict[str, Any]], Union[None, Awaitable[None]]]


def _utcnow() -> datetime:
    """Timezone-aware UTC now (replaces the deprecated datetime.utcnow)."""
    return datetime.now(timezone.utc)


class SwarmOrchestrator:
    """
    Main orchestrator for multi-agent swarm simulations.

    Responsibilities:
    - Agent lifecycle (create, initialize, run, tear down)
    - Round-based execution with pause/resume/stop semantics
    - OMPA-native memory and event logging
    - Event broadcast to registered sync or async callbacks
    - Optional checkpoint persistence to disk
    """

    def __init__(
        self,
        memory_adapter: OMPAMemoryAdapter,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.memory = memory_adapter
        self.logger = logging.getLogger("sandfish.orchestrator")

        # Active simulations keyed by sim_id.
        self.simulations: Dict[str, Dict[str, Any]] = {}

        # Agent registry, namespaced by sim_id via the agent ID prefix.
        self.agents: Dict[str, Any] = {}

        # Event callbacks (sync or async).
        self.event_callbacks: List[EventCallback] = []

        # Per-simulation locks, protect concurrent run/start/stop calls.
        self._sim_locks: Dict[str, asyncio.Lock] = {}

        # Optional checkpoint persistence.
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ----- Simulation lifecycle -----

    def create_simulation(self, config: SimulationConfig) -> str:
        """Create a new simulation and return its ID."""
        sim_id = uuid.uuid4().hex[:12]

        self.simulations[sim_id] = {
            "id": sim_id,
            "config": config,
            "status": SimulationStatus.PENDING,
            "current_round": 0,
            "agents": [],
            "agents_initialized": False,
            "start_time": None,
            "end_time": None,
            "checkpoints": [],
            "error": None,
            # Snapshot taken when agents are released; see _release_agents.
            "final_state": None,
            "last_checkpoint_round": None,
        }
        self._sim_locks[sim_id] = asyncio.Lock()

        self.memory.record_event(
            event_type="SIMULATION_CREATED",
            description=f"Created simulation '{config.name}' with {config.num_agents} agents",
            metadata={"simulation_id": sim_id, "config": asdict(config)},
        )

        self.logger.info("Created simulation %s: %s", sim_id, config.name)
        return sim_id

    async def run_simulation(self, sim_id: str) -> SimulationResult:
        """
        Execute a simulation. Safe to call again after a pause to resume
        from the last completed round.
        """
        if sim_id not in self.simulations:
            raise ValueError(f"Simulation {sim_id} not found")

        lock = self._sim_locks.setdefault(sim_id, asyncio.Lock())
        if lock.locked():
            raise RuntimeError(
                f"Simulation {sim_id} is already running. Pause it first."
            )

        async with lock:
            sim = self.simulations[sim_id]
            config: SimulationConfig = sim["config"]

            # Decide where to resume from.
            resume_from = sim["current_round"] if sim["agents_initialized"] else 0
            sim["status"] = SimulationStatus.RUNNING
            if sim["start_time"] is None:
                sim["start_time"] = _utcnow()

            self.logger.info(
                "Starting simulation %s from round %d", sim_id, resume_from
            )

            try:
                if not sim["agents_initialized"]:
                    await self._initialize_agents(sim_id, config)
                    sim["agents_initialized"] = True

                for round_num in range(resume_from, config.max_rounds):
                    sim["current_round"] = round_num
                    await self._execute_round(sim_id, round_num)

                    # Count completed rounds, not the zero-based index: the old
                    # `round_num % interval` snapshotted round 0 (after a single
                    # round) and then drifted off the interval boundary.
                    if (
                        config.checkpoint_interval
                        and (round_num + 1) % config.checkpoint_interval == 0
                    ):
                        await self._create_checkpoint(sim_id, round_num + 1)

                    await self._emit_event(
                        "round_complete",
                        {
                            "simulation_id": sim_id,
                            "round": round_num,
                            "total_rounds": config.max_rounds,
                        },
                    )

                    # Cooperative cancellation: yield, then check status.
                    await asyncio.sleep(0)
                    status = sim["status"]
                    if status == SimulationStatus.PAUSED:
                        self.logger.info(
                            "Simulation %s paused at round %d", sim_id, round_num
                        )
                        return self._create_result(sim, partial=True)
                    if status in _TERMINAL_RUN_STATUSES:
                        self.logger.info(
                            "Simulation %s stopped at round %d", sim_id, round_num
                        )
                        sim["end_time"] = _utcnow()
                        await self._create_final_checkpoint(sim_id, partial=True)
                        result = self._create_result(sim, partial=True)
                        self._release_agents(sim_id)
                        return result

                # Loop completed naturally.
                sim["status"] = SimulationStatus.COMPLETED
                sim["current_round"] = config.max_rounds
                sim["end_time"] = _utcnow()
                await self._create_final_checkpoint(sim_id)

            except Exception as exc:
                sim["status"] = SimulationStatus.FAILED
                sim["error"] = str(exc)
                sim["end_time"] = _utcnow()
                self.logger.exception("Simulation %s failed", sim_id)

            result = self._create_result(sim)
            self._release_agents(sim_id)

            self.memory.record_event(
                event_type="SIMULATION_COMPLETED",
                description=(
                    f"Simulation '{config.name}' ended with status "
                    f"{result.status.value}"
                ),
                metadata={
                    "simulation_id": sim_id,
                    "rounds": result.rounds_completed,
                    "metrics": result.metrics,
                },
            )

            return result

    # ----- Internal execution helpers -----

    async def _initialize_agents(self, sim_id: str, config: SimulationConfig) -> None:
        """Create and initialize agents for a simulation."""
        from ..agents.base_agent import create_agent  # local import avoids cycle

        sim = self.simulations[sim_id]

        for i in range(config.num_agents):
            agent_type = config.agent_types[i % len(config.agent_types)]

            # Give each agent its own stream derived from the run seed, so a
            # seeded run is reproducible no matter how the agents' async
            # decide_action calls interleave. The string form is hashed with
            # SHA-512 by random.Random, so adjacent seeds do not yield
            # overlapping streams and results do not depend on PYTHONHASHSEED.
            rng = random.Random(f"{config.seed}:{i}") if config.seed is not None else None

            agent = create_agent(
                agent_type=agent_type,
                agent_id=f"{sim_id}_agent_{i}",
                memory_adapter=self.memory,
                rng=rng,
            )
            sim["agents"].append(agent.id)
            self.agents[agent.id] = agent
            agent.simulation_id = sim_id
            await agent.initialize(config.seed_data)

    async def _execute_round(self, sim_id: str, round_num: int) -> None:
        """Execute a single simulation round."""
        sim = self.simulations[sim_id]
        agents = [self.agents[aid] for aid in sim["agents"]]

        if not agents:
            return

        # Provide each agent a view of its peers so it can pick targets.
        peer_ids = [a.id for a in agents]
        for agent in agents:
            agent.set_peers(peer_ids)

        # Decide actions in parallel — each agent can think independently.
        actions = await asyncio.gather(*(agent.decide_action() for agent in agents))

        # Execute actions sequentially so shared state mutations are deterministic.
        for agent, action in zip(agents, actions):
            await agent.execute_action(action)

        await self._update_shared_state(sim_id, list(zip(peer_ids, actions)))

    async def _update_shared_state(self, sim_id: str, actions: List[tuple]) -> None:
        """Hook for subclasses to update simulation-wide state from a round."""
        # Default implementation is a no-op; useful tests live in subclasses.
        return None

    async def _create_final_checkpoint(self, sim_id: str, partial: bool = False) -> None:
        """Checkpoint the last executed round, unless it already was.

        Without this, a run whose round count is not a multiple of
        `checkpoint_interval` — or one stopped mid-interval — left its final
        rounds unsnapshotted, so the last checkpoint could trail the actual
        end state by up to `checkpoint_interval` rounds.
        """
        sim = self.simulations[sim_id]
        if not sim["config"].checkpoint_interval:
            return

        rounds_completed = self._rounds_completed(sim, partial)
        if rounds_completed <= 0:
            return
        if sim["last_checkpoint_round"] == rounds_completed:
            return

        await self._create_checkpoint(sim_id, rounds_completed)

    def _release_agents(self, sim_id: str) -> None:
        """Drop agent objects for a simulation that will not run again.

        `self.agents` is process-wide and the API server is long-lived, so
        every finished run used to leak its agents — each holding an action
        history of up to DEFAULT_HISTORY_LIMIT entries — for the life of the
        process. The final state is snapshotted first, so results and status
        queries are unaffected.
        """
        sim = self.simulations.get(sim_id)
        if sim is None or sim["status"] not in _FINISHED_STATUSES:
            return

        if sim["final_state"] is None:
            sim["final_state"] = {
                aid: self.agents[aid].get_state()
                for aid in sim["agents"]
                if aid in self.agents
            }

        for agent_id in sim["agents"]:
            self.agents.pop(agent_id, None)

    async def _create_checkpoint(self, sim_id: str, rounds_completed: int) -> None:
        """Snapshot agent state and persist to disk if configured.

        `rounds_completed` labels the checkpoint by how many rounds have
        finished, so a checkpoint at label 10 always means "state after 10
        rounds" regardless of where it was taken from.
        """
        sim = self.simulations[sim_id]

        checkpoint = {
            "round": rounds_completed,
            "timestamp": _utcnow().isoformat(),
            "agent_states": {
                aid: self.agents[aid].get_state()
                for aid in sim["agents"]
                if aid in self.agents
            },
        }
        sim["checkpoints"].append(checkpoint)
        sim["last_checkpoint_round"] = rounds_completed

        if self.checkpoint_dir is not None:
            path = self.checkpoint_dir / f"{sim_id}_round_{checkpoint['round']:06d}.json"
            try:
                path.write_text(json.dumps(checkpoint, default=str), encoding="utf-8")
            except OSError as exc:
                self.logger.warning("Failed to persist checkpoint %s: %s", path, exc)

        self.logger.debug(
            "Created checkpoint for %s at round %d", sim_id, checkpoint["round"]
        )

    # ----- Control plane -----

    def pause_simulation(self, sim_id: str) -> bool:
        """Request a pause; takes effect at the end of the current round."""
        sim = self.simulations.get(sim_id)
        if sim is None:
            return False
        if sim["status"] == SimulationStatus.RUNNING:
            sim["status"] = SimulationStatus.PAUSED
            self.logger.info("Paused simulation %s", sim_id)
            return True
        return False

    def resume_simulation(self, sim_id: str) -> bool:
        """Mark a paused simulation runnable again. Caller must invoke run_simulation."""
        sim = self.simulations.get(sim_id)
        if sim is None:
            return False
        if sim["status"] == SimulationStatus.PAUSED:
            sim["status"] = SimulationStatus.PENDING
            self.logger.info("Resumed simulation %s", sim_id)
            return True
        return False

    def stop_simulation(self, sim_id: str) -> bool:
        """Request a stop; takes effect at the end of the current round."""
        sim = self.simulations.get(sim_id)
        if sim is None:
            return False
        if sim["status"] in (SimulationStatus.RUNNING, SimulationStatus.PAUSED):
            was_paused = sim["status"] == SimulationStatus.PAUSED
            sim["status"] = SimulationStatus.STOPPED
            if was_paused:
                # A running sim is released by its own run loop; a paused one
                # has no loop to observe the stop, so free its agents here.
                sim["end_time"] = _utcnow()
                self._release_agents(sim_id)
            self.logger.info("Stopped simulation %s", sim_id)
            return True
        return False

    def get_simulation_status(self, sim_id: str) -> Optional[Dict[str, Any]]:
        """Return current status for a simulation, or None if not found."""
        sim = self.simulations.get(sim_id)
        if sim is None:
            return None

        return {
            "id": sim_id,
            "name": sim["config"].name,
            "status": sim["status"].value,
            "current_round": sim["current_round"],
            "total_rounds": sim["config"].max_rounds,
            "num_agents": len(sim["agents"]),
            "start_time": sim["start_time"].isoformat() if sim["start_time"] else None,
            "end_time": sim["end_time"].isoformat() if sim["end_time"] else None,
            "error": sim["error"],
        }

    def list_simulations(self) -> List[Dict[str, Any]]:
        """List all simulations as lightweight summaries."""
        return [
            {
                "id": sim_id,
                "name": sim["config"].name,
                "status": sim["status"].value,
                "round": sim["current_round"],
                "total_rounds": sim["config"].max_rounds,
            }
            for sim_id, sim in self.simulations.items()
        ]

    # ----- Events -----

    def on_event(self, callback: EventCallback) -> None:
        """Register a sync or async event callback."""
        self.event_callbacks.append(callback)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Invoke registered callbacks; awaits coroutine callbacks correctly."""
        for callback in list(self.event_callbacks):
            try:
                result = callback(event_type, data)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self.logger.error("Event callback %r failed: %s", callback, exc)

    # ----- Result construction -----

    def _rounds_completed(self, sim: Dict[str, Any], partial: bool = False) -> int:
        """Number of rounds fully executed.

        On natural completion the run loop already advances current_round to
        max_rounds, so rounds_completed == current_round. On a partial exit
        (pause/stop) the current_round is the index of the last fully executed
        round, so rounds_completed == current_round + 1. On a failure the
        current round threw before finishing, so it is not counted.
        """
        if not sim["agents_initialized"]:
            return 0
        if partial:
            return sim["current_round"] + 1
        return sim["current_round"]

    def _create_result(self, sim: Dict[str, Any], partial: bool = False) -> SimulationResult:
        """Build a SimulationResult for the current state of `sim`."""
        rounds_completed = self._rounds_completed(sim, partial)

        return SimulationResult(
            simulation_id=sim["id"],
            status=sim["status"],
            rounds_completed=rounds_completed,
            final_state=self._aggregate_final_state(sim),
            metrics=self._calculate_metrics(sim),
            start_time=sim["start_time"] or _utcnow(),
            end_time=sim["end_time"],
            error_message=sim["error"],
        )

    def _aggregate_final_state(self, sim: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate per-agent state for the result.

        Prefers the snapshot taken when the agents were released, so results
        stay complete after a finished run's agents are freed.
        """
        cached = sim["final_state"]
        if cached is not None:
            return cached

        return {
            aid: self.agents[aid].get_state()
            for aid in sim["agents"]
            if aid in self.agents
        }

    def _calculate_metrics(self, sim: Dict[str, Any]) -> Dict[str, float]:
        """Compute coarse simulation metrics."""
        config: SimulationConfig = sim["config"]
        max_rounds = max(1, config.max_rounds)
        return {
            "completion_rate": min(1.0, sim["current_round"] / max_rounds),
            "agent_count": float(len(sim["agents"])),
            "checkpoint_count": float(len(sim["checkpoints"])),
        }


def create_orchestrator(
    vault_path: str,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> SwarmOrchestrator:
    """Convenience factory: build an orchestrator with an OMPA-backed memory."""
    memory = OMPAMemoryAdapter(vault_path)
    return SwarmOrchestrator(memory, checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None)
