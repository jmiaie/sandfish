"""
Edge case and stress tests for SandFish.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from sandfish.core.orchestrator import (
    SwarmOrchestrator,
    SimulationConfig,
    SimulationStatus,
    create_orchestrator
)
from sandfish.agents.base_agent import (
    create_agent,
    AgentProfile,
    ActionType,
    Action
)
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def mock_memory(self):
        """Mock OMPA memory adapter."""
        memory = Mock(spec=OMPAMemoryAdapter)
        memory.record_event = Mock()
        memory.search = Mock(return_value=[])
        memory.add_entity = Mock()
        return memory
    
    def test_create_simulation_with_zero_agents(self, mock_memory):
        """Test creating simulation with 0 agents."""
        orch = SwarmOrchestrator(mock_memory)
        config = SimulationConfig(
            name="Empty Sim",
            description="No agents",
            max_rounds=10,
            num_agents=0
        )
        
        sim_id = orch.create_simulation(config)
        assert sim_id is not None
        assert orch.simulations[sim_id]['config'].num_agents == 0
    
    def test_create_simulation_with_max_rounds_zero(self, mock_memory):
        """Test creating simulation with 0 max rounds."""
        orch = SwarmOrchestrator(mock_memory)
        config = SimulationConfig(
            name="Zero Rounds",
            description="No rounds",
            max_rounds=0,
            num_agents=5
        )
        
        sim_id = orch.create_simulation(config)
        assert sim_id is not None
    
    def test_simulation_with_special_characters_in_name(self, mock_memory):
        """Test simulation names with special characters."""
        orch = SwarmOrchestrator(mock_memory)
        
        special_names = [
            "Test Sim with spaces",
            "Test-Sim-with-dashes",
            "Test_Sim_with_underscores",
            "Test.Sim.with.dots",
            "TestSim123",
            "TestSim!@#$%",  # Special chars
        ]
        
        for name in special_names:
            config = SimulationConfig(name=name, description="Test")
            sim_id = orch.create_simulation(config)
            assert sim_id is not None
    
    def test_pause_nonexistent_simulation(self, mock_memory):
        """Test pausing a simulation that doesn't exist."""
        orch = SwarmOrchestrator(mock_memory)
        
        # Should not raise error
        orch.pause_simulation("nonexistent-id")
    
    def test_resume_nonexistent_simulation(self, mock_memory):
        """Test resuming a simulation that doesn't exist."""
        orch = SwarmOrchestrator(mock_memory)
        
        # Should not raise error
        orch.resume_simulation("nonexistent-id")
    
    def test_stop_nonexistent_simulation(self, mock_memory):
        """Test stopping a simulation that doesn't exist."""
        orch = SwarmOrchestrator(mock_memory)
        
        # Should not raise error
        orch.stop_simulation("nonexistent-id")
    
    def test_create_100_simulations(self, mock_memory):
        """Stress test: Create 100 simulations."""
        orch = SwarmOrchestrator(mock_memory)
        
        sim_ids = []
        for i in range(100):
            config = SimulationConfig(
                name=f"Sim {i}",
                description=f"Test simulation {i}",
                max_rounds=10,
                num_agents=5
            )
            sim_id = orch.create_simulation(config)
            sim_ids.append(sim_id)
        
        assert len(sim_ids) == 100
        assert len(set(sim_ids)) == 100  # All unique
    
    def test_simulation_id_collision_resistance(self, mock_memory):
        """Test that simulation IDs are unique."""
        orch = SwarmOrchestrator(mock_memory)
        config = SimulationConfig(name="Test", description="Test")
        
        ids = [orch.create_simulation(config) for _ in range(50)]
        assert len(set(ids)) == 50  # No collisions


class TestAgentEdgeCases:
    """Test agent edge cases."""
    
    @pytest.fixture
    def mock_memory(self):
        """Mock OMPA memory adapter."""
        memory = Mock(spec=OMPAMemoryAdapter)
        memory.record_event = Mock()
        memory.search = Mock(return_value=[])
        memory.add_entity = Mock()
        return memory
    
    @pytest.mark.asyncio
    async def test_agent_with_zero_energy(self, mock_memory):
        """Test agent behavior with zero energy."""
        agent = create_agent("default", memory_adapter=mock_memory)
        await agent.initialize({})
        
        agent.state.energy = 0
        action = await agent.decide_action()
        
        # Should choose DO_NOTHING when energy is 0
        assert action.action_type == ActionType.DO_NOTHING
    
    @pytest.mark.asyncio
    async def test_agent_with_negative_energy(self, mock_memory):
        """Test agent behavior with negative energy."""
        agent = create_agent("default", memory_adapter=mock_memory)
        await agent.initialize({})
        
        agent.state.energy = -10
        action = await agent.decide_action()
        
        # Should choose DO_NOTHING when energy is negative
        assert action.action_type == ActionType.DO_NOTHING
    
    @pytest.mark.asyncio
    async def test_agent_with_max_energy(self, mock_memory):
        """Test agent with maximum energy."""
        agent = create_agent("default", memory_adapter=mock_memory)
        await agent.initialize({})
        
        agent.state.energy = 1000  # Way above max
        action = await agent.decide_action()
        
        # Should still be able to act
        assert action.action_type in ActionType
    
    @pytest.mark.asyncio
    async def test_agent_with_very_long_name(self, mock_memory):
        """Test agent with extremely long name."""
        long_name = "A" * 1000
        agent = create_agent(
            "default",
            memory_adapter=mock_memory,
            name=long_name
        )
        await agent.initialize({})
        
        assert agent.profile.name == long_name
    
    @pytest.mark.asyncio
    async def test_agent_with_empty_name(self, mock_memory):
        """Test agent with empty name."""
        agent = create_agent(
            "default",
            memory_adapter=mock_memory,
            name=""
        )
        await agent.initialize({})
        
        assert agent.profile.name == ""
    
    @pytest.mark.asyncio
    async def test_agent_with_unicode_name(self, mock_memory):
        """Test agent with unicode characters in name."""
        unicode_name = "测试代理 🌵 Émoji"
        agent = create_agent(
            "default",
            memory_adapter=mock_memory,
            name=unicode_name
        )
        await agent.initialize({})
        
        assert agent.profile.name == unicode_name
    
    def test_create_1000_agents(self, mock_memory):
        """Stress test: Create 1000 agents."""
        agents = []
        for i in range(1000):
            agent = create_agent(
                "default",
                agent_id=f"agent_{i}",
                memory_adapter=mock_memory
            )
            agents.append(agent)
        
        assert len(agents) == 1000
        assert len(set(a.id for a in agents)) == 1000  # All unique


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_simulation_lifecycle(self):
        """Test complete simulation from creation to completion."""
        orch = create_orchestrator("/tmp/test-integration-vault")
        
        # Create
        config = SimulationConfig(
            name="Integration Test",
            description="Full lifecycle test",
            max_rounds=5,
            num_agents=3
        )
        sim_id = orch.create_simulation(config)
        
        # Verify created
        assert sim_id in orch.simulations
        assert orch.simulations[sim_id]['status'] == SimulationStatus.PENDING
        
        # Run
        result = await orch.run_simulation(sim_id)
        
        # Verify completed
        assert result.status == SimulationStatus.COMPLETED
        assert result.rounds_completed == 5
        assert result.simulation_id == sim_id
    
    @pytest.mark.asyncio
    async def test_multiple_simulations_parallel(self):
        """Test running multiple simulations."""
        orch = create_orchestrator("/tmp/test-parallel-vault")
        
        # Create 3 simulations
        sim_ids = []
        for i in range(3):
            config = SimulationConfig(
                name=f"Parallel {i}",
                description="Parallel test",
                max_rounds=3,
                num_agents=2
            )
            sim_id = orch.create_simulation(config)
            sim_ids.append(sim_id)
        
        # Run all
        results = await asyncio.gather(*[
            orch.run_simulation(sim_id)
            for sim_id in sim_ids
        ])
        
        # Verify all completed
        assert len(results) == 3
        assert all(r.status == SimulationStatus.COMPLETED for r in results)
