"""
Tests for run reproducibility, agent lifecycle cleanup, checkpoint coverage,
and lazy context gathering.
"""

import random

import pytest

from sandfish.agents.base_agent import (
    ActionType,
    BaseAgent,
    create_agent,
    register_agent_type,
)
from sandfish.core.orchestrator import (
    SimulationConfig,
    SimulationStatus,
    SwarmOrchestrator,
)
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


def _orchestrator(tmp_path, checkpoint_dir=None):
    memory = OMPAMemoryAdapter(str(tmp_path / "vault"))
    return SwarmOrchestrator(memory, checkpoint_dir=checkpoint_dir)


def _fingerprint(result):
    """A comparable summary of a run's outcome.

    Read from the public result rather than the live agent registry, which a
    finished run releases. Every field here is downstream of the action
    sequence, so two runs matching on all of them ran identically.
    """
    def strip_sim(agent_id):
        # Drop the per-run sim_id prefix: agent identity is deliberately unique
        # per run, only the dynamics need to reproduce.
        return agent_id.split("_", 1)[1]

    return sorted(
        (
            strip_sim(aid),
            state["state"]["energy"],
            state["state"]["mood"],
            state["state"]["reputation"],
            tuple(strip_sim(c) for c in state["state"]["connections"]),
            tuple(state["state"]["posts_created"]),
            state["action_count"],
        )
        for aid, state in result.final_state.items()
    )


# ----- Seeding / reproducibility -----


class TestReproducibility:
    @pytest.mark.asyncio
    async def test_same_seed_produces_identical_runs(self, tmp_path):
        runs = []
        for i in range(2):
            orch = _orchestrator(tmp_path / f"run{i}")
            sim_id = orch.create_simulation(
                SimulationConfig(
                    name="seeded", max_rounds=12, num_agents=6, seed=1234
                )
            )
            runs.append(_fingerprint(await orch.run_simulation(sim_id)))

        assert runs[0] == runs[1]
        assert runs[0], "expected a non-empty result"

    @pytest.mark.asyncio
    async def test_different_seeds_diverge(self, tmp_path):
        runs = []
        for i, seed in enumerate((1234, 5678)):
            orch = _orchestrator(tmp_path / f"run{i}")
            sim_id = orch.create_simulation(
                SimulationConfig(
                    name="seeded", max_rounds=12, num_agents=6, seed=seed
                )
            )
            runs.append(_fingerprint(await orch.run_simulation(sim_id)))

        assert runs[0] != runs[1]

    @pytest.mark.asyncio
    async def test_unseeded_runs_remain_independent(self, tmp_path):
        """No seed means OS-seeded, preserving pre-existing behaviour."""
        runs = []
        for i in range(2):
            orch = _orchestrator(tmp_path / f"run{i}")
            sim_id = orch.create_simulation(
                SimulationConfig(name="unseeded", max_rounds=25, num_agents=8)
            )
            runs.append(_fingerprint(await orch.run_simulation(sim_id)))

        assert runs[0] != runs[1]

    @pytest.mark.asyncio
    async def test_seeded_post_ids_are_reproducible(self, tmp_path):
        ids = []
        for i in range(2):
            orch = _orchestrator(tmp_path / f"run{i}")
            sim_id = orch.create_simulation(
                SimulationConfig(name="posts", max_rounds=15, num_agents=5, seed=99)
            )
            result = await orch.run_simulation(sim_id)
            ids.append(
                [
                    pid
                    for _, state in sorted(result.final_state.items())
                    for pid in state["state"]["posts_created"]
                ]
            )

        assert ids[0] == ids[1]
        assert ids[0], "expected posts to have been created"

    def test_agents_do_not_use_the_global_random_stream(self):
        """A seeded agent must be unaffected by global `random` consumption."""
        agent_a = create_agent("default", agent_id="a", rng=random.Random(7))
        random.seed(0)
        first = [agent_a._select_action({}).action_type for _ in range(20)]

        agent_b = create_agent("default", agent_id="a", rng=random.Random(7))
        random.seed(12345)
        [random.random() for _ in range(50)]  # perturb the global stream
        second = [agent_b._select_action({}).action_type for _ in range(20)]

        assert first == second

    def test_agent_defaults_to_independent_rng(self):
        agent = create_agent("default", agent_id="x")
        assert isinstance(agent.rng, random.Random)


# ----- Lazy context -----


class _RecordingMemory(OMPAMemoryAdapter):
    """Memory adapter that counts how often the expensive lookups run."""

    def __init__(self, vault_path):
        super().__init__(vault_path)
        self.search_calls = 0
        self.related_calls = 0

    def search(self, query, limit=10):
        self.search_calls += 1
        return super().search(query, limit=limit)

    def get_related_entities(self, entity_name, **kwargs):
        self.related_calls += 1
        return super().get_related_entities(entity_name, **kwargs)


class TestLazyContext:
    @pytest.mark.asyncio
    async def test_builtin_agents_do_not_touch_memory_lookups(self, tmp_path):
        memory = _RecordingMemory(str(tmp_path / "vault"))
        orch = SwarmOrchestrator(memory)
        sim_id = orch.create_simulation(
            SimulationConfig(
                name="lazy",
                max_rounds=20,
                num_agents=10,
                agent_types=["default", "influencer", "lurker"],
                seed=3,
            )
        )
        await orch.run_simulation(sim_id)

        # None of the shipped agents read context["memories"], so the searches
        # that used to run once per agent per round should not happen at all.
        assert memory.search_calls == 0
        assert memory.related_calls == 0

    @pytest.mark.asyncio
    async def test_context_still_resolves_when_read(self, tmp_path):
        memory = _RecordingMemory(str(tmp_path / "vault"))
        agent = create_agent("default", agent_id="reader", memory_adapter=memory)
        context = agent._gather_context()

        assert memory.search_calls == 0

        assert context["memories"] == []
        assert memory.search_calls == 1

        # Cached: a second read must not repeat the lookup.
        assert context["memories"] == []
        assert memory.search_calls == 1

        assert context["related_entities"] == []
        assert memory.related_calls == 1

    def test_context_exposes_eager_keys(self, tmp_path):
        agent = create_agent("default", agent_id="ctx")
        agent.set_peers(["p1", "p2"])
        context = agent._gather_context()

        assert context["round"] == 0
        assert context["peers"] == ["p1", "p2"]
        assert set(context) == {
            "current_state",
            "round",
            "peers",
            "memories",
            "related_entities",
        }
        assert len(context) == 5
        assert "memories" in context
        with pytest.raises(KeyError):
            context["nope"]

    def test_context_copy_resolves_everything(self, tmp_path):
        memory = _RecordingMemory(str(tmp_path / "vault"))
        agent = create_agent("default", agent_id="ctx", memory_adapter=memory)
        snapshot = agent._gather_context().copy()

        assert isinstance(snapshot, dict)
        assert snapshot["memories"] == []
        assert memory.search_calls == 1

    def test_failing_backend_does_not_break_decisions(self, tmp_path):
        class _BrokenMemory(OMPAMemoryAdapter):
            def search(self, query, limit=10):
                raise RuntimeError("backend down")

            def get_related_entities(self, entity_name, **kwargs):
                raise RuntimeError("backend down")

        agent = create_agent(
            "default",
            agent_id="broken",
            memory_adapter=_BrokenMemory(str(tmp_path / "vault")),
        )
        context = agent._gather_context()

        assert context["memories"] == []
        assert context["related_entities"] == []


class _ContextReadingAgent(BaseAgent):
    """Custom agent that consults memory, proving the lazy keys still work."""

    def _select_action(self, context):
        if context["memories"]:
            return __import__("sandfish.agents.base_agent", fromlist=["Action"]).Action(
                action_type=ActionType.SEARCH
            )
        return __import__("sandfish.agents.base_agent", fromlist=["Action"]).Action(
            action_type=ActionType.CREATE_POST
        )


class TestCustomAgentCompatibility:
    def test_custom_agent_reading_context_triggers_lookup(self, tmp_path):
        register_agent_type("context_reader", _ContextReadingAgent)
        memory = _RecordingMemory(str(tmp_path / "vault"))
        agent = create_agent(
            "context_reader", agent_id="cr", memory_adapter=memory
        )

        action = agent._select_action(agent._gather_context())

        assert memory.search_calls == 1
        assert action.action_type == ActionType.CREATE_POST

    def test_legacy_agent_without_rng_kwarg_still_constructs(self, tmp_path):
        """Agent classes predating `rng` must keep working when unseeded."""

        class _LegacyAgent(BaseAgent):
            def __init__(self, agent_id, profile, memory_adapter=None):
                super().__init__(agent_id, profile, memory_adapter)

            def _select_action(self, context):
                return __import__(
                    "sandfish.agents.base_agent", fromlist=["Action"]
                ).Action(action_type=ActionType.DO_NOTHING)

        register_agent_type("legacy", _LegacyAgent)
        agent = create_agent("legacy", agent_id="legacy_1")
        assert isinstance(agent.rng, random.Random)


# ----- Agent registry cleanup -----


class TestAgentRelease:
    @pytest.mark.asyncio
    async def test_completed_simulation_releases_agents(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(name="done", max_rounds=5, num_agents=4, seed=1)
        )
        result = await orch.run_simulation(sim_id)

        assert result.status == SimulationStatus.COMPLETED
        assert orch.agents == {}
        # The result must still carry every agent's final state.
        assert len(result.final_state) == 4

    @pytest.mark.asyncio
    async def test_repeated_runs_do_not_accumulate_agents(self, tmp_path):
        orch = _orchestrator(tmp_path)
        for i in range(10):
            sim_id = orch.create_simulation(
                SimulationConfig(name=f"s{i}", max_rounds=3, num_agents=5, seed=i)
            )
            await orch.run_simulation(sim_id)
            assert orch.agents == {}, f"agents leaked after run {i}"

    @pytest.mark.asyncio
    async def test_paused_simulation_keeps_agents(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(name="paused", max_rounds=50, num_agents=3, seed=2)
        )

        def pause_after_first(event_type, data):
            if event_type == "round_complete":
                orch.pause_simulation(sim_id)

        orch.on_event(pause_after_first)
        await orch.run_simulation(sim_id)

        assert orch.simulations[sim_id]["status"] == SimulationStatus.PAUSED
        assert len(orch.agents) == 3, "a paused run must keep its agents to resume"

    @pytest.mark.asyncio
    async def test_resume_after_pause_continues(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(name="resume", max_rounds=6, num_agents=3, seed=2)
        )

        calls = {"n": 0}

        def pause_once(event_type, data):
            if event_type == "round_complete" and calls["n"] == 0:
                calls["n"] += 1
                orch.pause_simulation(sim_id)

        orch.on_event(pause_once)
        await orch.run_simulation(sim_id)
        assert orch.resume_simulation(sim_id) is True

        result = await orch.run_simulation(sim_id)
        assert result.status == SimulationStatus.COMPLETED
        assert result.rounds_completed == 6
        assert orch.agents == {}

    @pytest.mark.asyncio
    async def test_stop_while_paused_releases_agents(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(name="stopped", max_rounds=50, num_agents=3, seed=2)
        )

        def pause_after_first(event_type, data):
            if event_type == "round_complete":
                orch.pause_simulation(sim_id)

        orch.on_event(pause_after_first)
        await orch.run_simulation(sim_id)
        assert len(orch.agents) == 3

        assert orch.stop_simulation(sim_id) is True
        assert orch.agents == {}

    @pytest.mark.asyncio
    async def test_failed_simulation_releases_agents(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(name="fails", max_rounds=5, num_agents=3, seed=1)
        )

        async def boom(sim_id_arg, round_num):
            raise RuntimeError("round exploded")

        orch._execute_round = boom
        result = await orch.run_simulation(sim_id)

        assert result.status == SimulationStatus.FAILED
        assert orch.agents == {}

    @pytest.mark.asyncio
    async def test_status_queries_survive_release(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(name="after", max_rounds=4, num_agents=3, seed=1)
        )
        await orch.run_simulation(sim_id)

        status = orch.get_simulation_status(sim_id)
        assert status is not None
        assert status["num_agents"] == 3
        assert status["status"] == "completed"
        assert len(orch.list_simulations()) == 1


# ----- Checkpoints -----


class TestCheckpoints:
    @pytest.mark.asyncio
    async def test_final_round_is_checkpointed(self, tmp_path):
        orch = _orchestrator(tmp_path)
        # 25 rounds at interval 10 leaves rounds 21-25 past the last boundary.
        sim_id = orch.create_simulation(
            SimulationConfig(
                name="cp", max_rounds=25, num_agents=2, checkpoint_interval=10, seed=1
            )
        )
        await orch.run_simulation(sim_id)

        rounds = [c["round"] for c in orch.simulations[sim_id]["checkpoints"]]
        assert rounds == [10, 20, 25]

    @pytest.mark.asyncio
    async def test_no_duplicate_checkpoint_on_exact_boundary(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(
                name="cp", max_rounds=20, num_agents=2, checkpoint_interval=10, seed=1
            )
        )
        await orch.run_simulation(sim_id)

        rounds = [c["round"] for c in orch.simulations[sim_id]["checkpoints"]]
        assert rounds == [10, 20]

    @pytest.mark.asyncio
    async def test_checkpoints_persist_to_disk(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        orch = _orchestrator(tmp_path, checkpoint_dir=checkpoint_dir)
        sim_id = orch.create_simulation(
            SimulationConfig(
                name="cp", max_rounds=15, num_agents=2, checkpoint_interval=10, seed=1
            )
        )
        await orch.run_simulation(sim_id)

        written = sorted(p.name for p in checkpoint_dir.glob("*.json"))
        assert len(written) == 2
        assert written[-1].endswith("_round_000015.json")

    @pytest.mark.asyncio
    async def test_checkpoint_interval_zero_disables_checkpoints(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(
                name="cp", max_rounds=10, num_agents=2, checkpoint_interval=0, seed=1
            )
        )
        await orch.run_simulation(sim_id)

        assert orch.simulations[sim_id]["checkpoints"] == []

    @pytest.mark.asyncio
    async def test_checkpoint_captures_agent_states(self, tmp_path):
        orch = _orchestrator(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(
                name="cp", max_rounds=5, num_agents=3, checkpoint_interval=5, seed=1
            )
        )
        await orch.run_simulation(sim_id)

        checkpoint = orch.simulations[sim_id]["checkpoints"][-1]
        assert len(checkpoint["agent_states"]) == 3
