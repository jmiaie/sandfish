"""
Tests for SwarmOrchestrator.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from sandfish.core.orchestrator import (
    SimulationConfig,
    SimulationStatus,
    SwarmOrchestrator,
    create_orchestrator,
)
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


@pytest.fixture
def mock_memory():
    """Mock OMPA memory adapter that returns no results."""
    memory = Mock(spec=OMPAMemoryAdapter)
    memory.record_event = Mock()
    memory.search = Mock(return_value=[])
    memory.add_entity = Mock()
    memory.get_related_entities = Mock(return_value=[])
    return memory


@pytest.fixture
def orchestrator(mock_memory):
    return SwarmOrchestrator(mock_memory)


@pytest.fixture
def sample_config():
    return SimulationConfig(
        name="Test Simulation",
        description="A test simulation",
        max_rounds=10,
        num_agents=5,
        agent_types=["default"],
        seed_data={"initial_energy": 100},
    )


class TestSimulationCreation:
    def test_create_simulation(self, orchestrator, sample_config):
        sim_id = orchestrator.create_simulation(sample_config)

        assert sim_id
        assert sim_id in orchestrator.simulations

        sim = orchestrator.simulations[sim_id]
        assert sim["config"] == sample_config
        assert sim["status"] == SimulationStatus.PENDING
        assert sim["current_round"] == 0
        assert sim["agents_initialized"] is False

    def test_create_multiple_simulations(self, orchestrator, sample_config):
        sim_ids = [orchestrator.create_simulation(sample_config) for _ in range(3)]

        assert len(set(sim_ids)) == 3
        assert len(orchestrator.simulations) == 3


class TestSimulationStatus:
    def test_get_status_existing(self, orchestrator, sample_config):
        sim_id = orchestrator.create_simulation(sample_config)
        status = orchestrator.get_simulation_status(sim_id)

        assert status is not None
        assert status["id"] == sim_id
        assert status["status"] == "pending"
        assert status["current_round"] == 0
        assert status["total_rounds"] == sample_config.max_rounds

    def test_get_status_nonexistent(self, orchestrator):
        assert orchestrator.get_simulation_status("invalid-id") is None

    def test_list_simulations(self, orchestrator, sample_config):
        sim_id = orchestrator.create_simulation(sample_config)
        sims = orchestrator.list_simulations()

        assert len(sims) == 1
        assert sims[0]["id"] == sim_id
        assert sims[0]["name"] == sample_config.name
        assert sims[0]["total_rounds"] == sample_config.max_rounds


class TestSimulationControl:
    def test_pause_simulation_only_when_running(self, orchestrator, sample_config):
        sim_id = orchestrator.create_simulation(sample_config)

        # PENDING → can't pause
        assert orchestrator.pause_simulation(sim_id) is False

        # Force RUNNING and pause
        orchestrator.simulations[sim_id]["status"] = SimulationStatus.RUNNING
        assert orchestrator.pause_simulation(sim_id) is True
        assert orchestrator.simulations[sim_id]["status"] == SimulationStatus.PAUSED

    def test_resume_simulation_only_when_paused(self, orchestrator, sample_config):
        sim_id = orchestrator.create_simulation(sample_config)
        orchestrator.simulations[sim_id]["status"] = SimulationStatus.PAUSED

        assert orchestrator.resume_simulation(sim_id) is True
        # Now it should be PENDING again, ready to run.
        assert orchestrator.simulations[sim_id]["status"] == SimulationStatus.PENDING

    def test_stop_simulation_uses_dedicated_status(self, orchestrator, sample_config):
        sim_id = orchestrator.create_simulation(sample_config)
        orchestrator.simulations[sim_id]["status"] = SimulationStatus.RUNNING

        assert orchestrator.stop_simulation(sim_id) is True
        assert orchestrator.simulations[sim_id]["status"] == SimulationStatus.STOPPED

    def test_control_returns_false_for_unknown_id(self, orchestrator):
        assert orchestrator.pause_simulation("nope") is False
        assert orchestrator.resume_simulation("nope") is False
        assert orchestrator.stop_simulation("nope") is False


class TestEventHandling:
    def test_event_callback_registration(self, orchestrator):
        callback = Mock()
        orchestrator.on_event(callback)
        assert callback in orchestrator.event_callbacks

    @pytest.mark.asyncio
    async def test_sync_callback_invoked(self, orchestrator):
        callback = Mock()
        orchestrator.on_event(callback)
        await orchestrator._emit_event("test_event", {"x": 1})
        callback.assert_called_once_with("test_event", {"x": 1})

    @pytest.mark.asyncio
    async def test_async_callback_awaited(self, orchestrator):
        seen = []

        async def cb(event_type, data):
            seen.append((event_type, data))

        orchestrator.on_event(cb)
        await orchestrator._emit_event("round_complete", {"r": 5})
        assert seen == [("round_complete", {"r": 5})]


class TestRunSimulation:
    @pytest.mark.asyncio
    async def test_run_to_completion(self, orchestrator):
        config = SimulationConfig(
            name="quick", description="", max_rounds=3, num_agents=2,
        )
        sim_id = orchestrator.create_simulation(config)
        result = await orchestrator.run_simulation(sim_id)

        assert result.status == SimulationStatus.COMPLETED
        assert result.rounds_completed == 3
        sim = orchestrator.simulations[sim_id]
        assert sim["agents_initialized"] is True
        assert len(sim["agents"]) == 2

    @pytest.mark.asyncio
    async def test_resume_from_pause_does_not_reinit(self, orchestrator):
        config = SimulationConfig(
            name="resume", description="", max_rounds=4, num_agents=2,
        )
        sim_id = orchestrator.create_simulation(config)

        # Pre-mark as paused at round 2 so the loop should resume from round 2.
        orchestrator.simulations[sim_id]["agents_initialized"] = True
        orchestrator.simulations[sim_id]["current_round"] = 2
        orchestrator.simulations[sim_id]["status"] = SimulationStatus.PAUSED

        # Pre-populate fake agents so _initialize_agents isn't called.
        from sandfish.agents.base_agent import create_agent
        from unittest.mock import Mock as _Mock

        for i in range(2):
            agent = create_agent("default", agent_id=f"{sim_id}_agent_{i}")
            await agent.initialize({})
            orchestrator.simulations[sim_id]["agents"].append(agent.id)
            orchestrator.agents[agent.id] = agent

        # Resume must transition to PENDING (per resume_simulation contract).
        orchestrator.simulations[sim_id]["status"] = SimulationStatus.PENDING
        result = await orchestrator.run_simulation(sim_id)

        # 4 total rounds, resumed from 2, so 2 more rounds executed.
        assert result.status == SimulationStatus.COMPLETED
        assert orchestrator.simulations[sim_id]["current_round"] == 4

    @pytest.mark.asyncio
    async def test_run_simulation_emits_round_events(self, orchestrator):
        config = SimulationConfig(name="emit", description="", max_rounds=2, num_agents=1)
        sim_id = orchestrator.create_simulation(config)

        events = []

        async def cb(event_type, data):
            events.append((event_type, data["round"]))

        orchestrator.on_event(cb)
        await orchestrator.run_simulation(sim_id)
        assert ("round_complete", 0) in events
        assert ("round_complete", 1) in events


class TestFactory:
    @patch("sandfish.core.orchestrator.OMPAMemoryAdapter")
    def test_create_orchestrator(self, mock_adapter_class, tmp_path):
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        orch = create_orchestrator(str(tmp_path / "vault"))

        assert isinstance(orch, SwarmOrchestrator)
        assert orch.memory == mock_adapter
        mock_adapter_class.assert_called_once_with(str(tmp_path / "vault"))
