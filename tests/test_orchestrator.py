"""
Tests for SwarmOrchestrator.
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
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


@pytest.fixture
def mock_memory():
    """Mock OMPA memory adapter."""
    memory = Mock(spec=OMPAMemoryAdapter)
    memory.record_event = Mock()
    memory.search = Mock(return_value=[])
    memory.add_entity = Mock()
    return memory


@pytest.fixture
def orchestrator(mock_memory):
    """Create orchestrator with mock memory."""
    return SwarmOrchestrator(mock_memory)


@pytest.fixture
def sample_config():
    """Sample simulation configuration."""
    return SimulationConfig(
        name="Test Simulation",
        description="A test simulation",
        max_rounds=10,
        num_agents=5,
        agent_types=["default"],
        seed_data={"initial_energy": 100}
    )


class TestSimulationCreation:
    """Test simulation creation."""
    
    def test_create_simulation(self, orchestrator, sample_config):
        """Test creating a simulation."""
        sim_id = orchestrator.create_simulation(sample_config)
        
        assert sim_id is not None
        assert len(sim_id) == 8  # UUID truncated
        assert sim_id in orchestrator.simulations
        
        sim = orchestrator.simulations[sim_id]
        assert sim['config'] == sample_config
        assert sim['status'] == SimulationStatus.PENDING
        assert sim['current_round'] == 0
    
    def test_create_multiple_simulations(self, orchestrator, sample_config):
        """Test creating multiple simulations."""
        sim_ids = [
            orchestrator.create_simulation(sample_config)
            for _ in range(3)
        ]
        
        assert len(set(sim_ids)) == 3  # All unique
        assert len(orchestrator.simulations) == 3


class TestSimulationStatus:
    """Test simulation status management."""
    
    def test_get_status_existing(self, orchestrator, sample_config):
        """Test getting status of existing simulation."""
        sim_id = orchestrator.create_simulation(sample_config)
        status = orchestrator.get_simulation_status(sim_id)
        
        assert status is not None
        assert status['id'] == sim_id
        assert status['status'] == 'pending'
        assert status['current_round'] == 0
    
    def test_get_status_nonexistent(self, orchestrator):
        """Test getting status of non-existent simulation."""
        status = orchestrator.get_simulation_status("invalid-id")
        assert status is None
    
    def test_list_simulations(self, orchestrator, sample_config):
        """Test listing all simulations."""
        sim_id = orchestrator.create_simulation(sample_config)
        sims = orchestrator.list_simulations()
        
        assert len(sims) == 1
        assert sims[0]['id'] == sim_id
        assert sims[0]['name'] == sample_config.name


class TestSimulationControl:
    """Test simulation control operations."""
    
    def test_pause_simulation(self, orchestrator, sample_config):
        """Test pausing a simulation."""
        sim_id = orchestrator.create_simulation(sample_config)
        orchestrator.pause_simulation(sim_id)
        
        assert orchestrator.simulations[sim_id]['status'] == SimulationStatus.PAUSED
    
    def test_resume_simulation(self, orchestrator, sample_config):
        """Test resuming a simulation."""
        sim_id = orchestrator.create_simulation(sample_config)
        orchestrator.pause_simulation(sim_id)
        orchestrator.resume_simulation(sim_id)
        
        assert orchestrator.simulations[sim_id]['status'] == SimulationStatus.RUNNING
    
    def test_stop_simulation(self, orchestrator, sample_config):
        """Test stopping a simulation."""
        sim_id = orchestrator.create_simulation(sample_config)
        orchestrator.stop_simulation(sim_id)
        
        assert orchestrator.simulations[sim_id]['status'] == SimulationStatus.COMPLETED


class TestEventHandling:
    """Test event handling."""
    
    def test_event_callback_registration(self, orchestrator):
        """Test registering event callbacks."""
        callback = Mock()
        orchestrator.on_event(callback)
        
        assert callback in orchestrator.event_callbacks
    
    def test_event_emission(self, orchestrator, sample_config):
        """Test event emission to callbacks."""
        callback = Mock()
        orchestrator.on_event(callback)
        
        # Create simulation triggers event
        orchestrator.create_simulation(sample_config)
        
        # Callback should have been called
        assert callback.called


class TestFactory:
    """Test factory function."""
    
    @patch('sandfish.core.orchestrator.OMPAMemoryAdapter')
    def test_create_orchestrator(self, mock_adapter_class):
        """Test factory function creates orchestrator."""
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        
        orch = create_orchestrator("/tmp/test-vault")
        
        assert isinstance(orch, SwarmOrchestrator)
        assert orch.memory == mock_adapter
        mock_adapter_class.assert_called_once_with("/tmp/test-vault")
