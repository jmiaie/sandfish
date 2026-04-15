"""
Edge-case and stress tests for SandFish.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from sandfish.agents.base_agent import ActionType, create_agent
from sandfish.core.orchestrator import (
    SimulationConfig,
    SimulationStatus,
    SwarmOrchestrator,
    create_orchestrator,
)
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


@pytest.fixture
def mock_memory():
    memory = Mock(spec=OMPAMemoryAdapter)
    memory.record_event = Mock()
    memory.search = Mock(return_value=[])
    memory.add_entity = Mock()
    memory.get_related_entities = Mock(return_value=[])
    return memory


class TestEdgeCases:
    def test_create_simulation_with_zero_agents(self, mock_memory):
        orch = SwarmOrchestrator(mock_memory)
        config = SimulationConfig(
            name="Empty Sim", description="No agents", max_rounds=10, num_agents=0,
        )
        sim_id = orch.create_simulation(config)
        assert sim_id is not None
        assert orch.simulations[sim_id]["config"].num_agents == 0

    def test_create_simulation_with_max_rounds_zero(self, mock_memory):
        orch = SwarmOrchestrator(mock_memory)
        config = SimulationConfig(
            name="Zero Rounds", description="No rounds", max_rounds=0, num_agents=5,
        )
        sim_id = orch.create_simulation(config)
        assert sim_id is not None

    def test_simulation_with_special_characters_in_name(self, mock_memory):
        orch = SwarmOrchestrator(mock_memory)
        names = [
            "Test Sim with spaces",
            "Test-Sim-with-dashes",
            "Test_Sim_with_underscores",
            "Test.Sim.with.dots",
            "TestSim123",
            "TestSim!@#$%",
        ]
        for name in names:
            config = SimulationConfig(name=name, description="Test")
            assert orch.create_simulation(config) is not None

    def test_control_nonexistent_simulation_returns_false(self, mock_memory):
        orch = SwarmOrchestrator(mock_memory)
        assert orch.pause_simulation("nonexistent-id") is False
        assert orch.resume_simulation("nonexistent-id") is False
        assert orch.stop_simulation("nonexistent-id") is False

    def test_create_100_simulations(self, mock_memory):
        orch = SwarmOrchestrator(mock_memory)
        sim_ids = [
            orch.create_simulation(
                SimulationConfig(
                    name=f"Sim {i}", description=f"T{i}", max_rounds=10, num_agents=5,
                )
            )
            for i in range(100)
        ]
        assert len(sim_ids) == 100
        assert len(set(sim_ids)) == 100

    def test_simulation_id_collision_resistance(self, mock_memory):
        orch = SwarmOrchestrator(mock_memory)
        config = SimulationConfig(name="Test", description="Test")
        ids = [orch.create_simulation(config) for _ in range(50)]
        assert len(set(ids)) == 50


class TestAgentEdgeCases:
    @pytest.mark.asyncio
    async def test_agent_with_zero_energy(self, mock_memory):
        agent = create_agent("default", memory_adapter=mock_memory)
        await agent.initialize({})
        agent.state.energy = 0
        action = await agent.decide_action()
        assert action.action_type == ActionType.DO_NOTHING

    @pytest.mark.asyncio
    async def test_agent_with_negative_energy(self, mock_memory):
        agent = create_agent("default", memory_adapter=mock_memory)
        await agent.initialize({})
        agent.state.energy = -10
        action = await agent.decide_action()
        assert action.action_type == ActionType.DO_NOTHING

    @pytest.mark.asyncio
    async def test_agent_with_max_energy(self, mock_memory):
        agent = create_agent("default", memory_adapter=mock_memory)
        await agent.initialize({})
        agent.state.energy = 1000
        action = await agent.decide_action()
        assert isinstance(action.action_type, ActionType)

    @pytest.mark.asyncio
    async def test_agent_with_very_long_name(self, mock_memory):
        long_name = "A" * 1000
        agent = create_agent("default", memory_adapter=mock_memory, name=long_name)
        await agent.initialize({})
        assert agent.profile.name == long_name

    @pytest.mark.asyncio
    async def test_agent_with_empty_name(self, mock_memory):
        agent = create_agent("default", memory_adapter=mock_memory, name="")
        await agent.initialize({})
        assert agent.profile.name == ""

    @pytest.mark.asyncio
    async def test_agent_with_unicode_name(self, mock_memory):
        name = "测试代理 🌵 Émoji"
        agent = create_agent("default", memory_adapter=mock_memory, name=name)
        await agent.initialize({})
        assert agent.profile.name == name

    def test_create_1000_agents(self, mock_memory):
        agents = [
            create_agent("default", agent_id=f"agent_{i}", memory_adapter=mock_memory)
            for i in range(1000)
        ]
        assert len(agents) == 1000
        assert len({a.id for a in agents}) == 1000


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_simulation_lifecycle(self, tmp_path):
        orch = create_orchestrator(str(tmp_path / "integration-vault"))
        config = SimulationConfig(
            name="Integration Test",
            description="Full lifecycle test",
            max_rounds=5,
            num_agents=3,
        )
        sim_id = orch.create_simulation(config)
        assert sim_id in orch.simulations
        assert orch.simulations[sim_id]["status"] == SimulationStatus.PENDING

        result = await orch.run_simulation(sim_id)
        assert result.status == SimulationStatus.COMPLETED
        assert result.rounds_completed == 5
        assert result.simulation_id == sim_id

    @pytest.mark.asyncio
    async def test_multiple_simulations_parallel(self, tmp_path):
        orch = create_orchestrator(str(tmp_path / "parallel-vault"))
        sim_ids = [
            orch.create_simulation(
                SimulationConfig(
                    name=f"Parallel {i}",
                    description="Parallel test",
                    max_rounds=3,
                    num_agents=2,
                )
            )
            for i in range(3)
        ]
        results = await asyncio.gather(
            *[orch.run_simulation(sim_id) for sim_id in sim_ids]
        )
        assert len(results) == 3
        assert all(r.status == SimulationStatus.COMPLETED for r in results)
