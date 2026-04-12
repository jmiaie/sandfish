"""
Tests for agent definitions.
"""

import pytest
from unittest.mock import Mock
import asyncio

from sandfish.agents.base_agent import (
    BaseAgent,
    DefaultAgent,
    InfluencerAgent,
    LurkerAgent,
    create_agent,
    AgentProfile,
    AgentState,
    Action,
    ActionType
)
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


@pytest.fixture
def mock_memory():
    """Mock OMPA memory adapter."""
    memory = Mock(spec=OMPAMemoryAdapter)
    memory.record_event = Mock()
    memory.search = Mock(return_value=[])
    memory.add_entity = Mock()
    memory.get_related_entities = Mock(return_value=[])
    return memory


@pytest.fixture
def sample_profile():
    """Sample agent profile."""
    return AgentProfile(
        name="TestAgent",
        agent_type="default",
        traits={"creativity": 0.8, "sociability": 0.6},
        backstory="A test agent",
        goals=["test_goal"]
    )


class TestBaseAgent:
    """Test BaseAgent functionality."""
    
    @pytest.fixture
    def agent(self, mock_memory, sample_profile):
        """Create a test agent."""
        return DefaultAgent("test_id", sample_profile, mock_memory)
    
    def test_initialization(self, agent, sample_profile):
        """Test agent initialization."""
        assert agent.id == "test_id"
        assert agent.profile == sample_profile
        assert agent.status == AgentState.INITIALIZING
        assert agent.state.energy == 100.0
    
    @pytest.mark.asyncio
    async def test_initialize(self, agent, mock_memory):
        """Test agent initialize method."""
        await agent.initialize({"initial_energy": 80})
        
        assert agent.status == AgentState.IDLE
        assert agent.state.energy == 80
        mock_memory.add_entity.assert_called_once()
        mock_memory.record_event.assert_called()
    
    def test_get_state(self, agent):
        """Test getting agent state."""
        state = agent.get_state()
        
        assert state['id'] == "test_id"
        assert state['profile']['name'] == "TestAgent"
        assert 'energy' in state['state']
        assert 'mood' in state['state']
    
    @pytest.mark.asyncio
    async def test_decide_action_returns_action(self, agent):
        """Test that decide_action returns an Action."""
        await agent.initialize({})
        action = await agent.decide_action()
        
        assert isinstance(action, Action)
        assert isinstance(action.action_type, ActionType)
    
    @pytest.mark.asyncio
    async def test_execute_action(self, agent):
        """Test action execution."""
        await agent.initialize({})
        
        initial_energy = agent.state.energy
        action = Action(action_type=ActionType.DO_NOTHING)
        
        await agent.execute_action(action)
        
        assert len(agent.action_history) == 1
        assert agent.state.energy > initial_energy  # DO_NOTHING recovers energy


class TestAgentTypes:
    """Test different agent types."""
    
    @pytest.mark.asyncio
    async def test_default_agent_action_selection(self, mock_memory):
        """Test DefaultAgent action selection."""
        profile = AgentProfile(name="Default", agent_type="default")
        agent = DefaultAgent("id1", profile, mock_memory)
        await agent.initialize({})
        
        action = await agent.decide_action()
        assert action.action_type in ActionType
    
    @pytest.mark.asyncio
    async def test_influencer_agent_prefers_content(self, mock_memory):
        """Test InfluencerAgent prefers content creation."""
        profile = AgentProfile(name="Influencer", agent_type="influencer")
        agent = InfluencerAgent("id2", profile, mock_memory)
        await agent.initialize({})
        
        # Run multiple times to check distribution
        actions = [await agent.decide_action() for _ in range(100)]
        create_posts = sum(1 for a in actions if a.action_type == ActionType.CREATE_POST)
        
        # Should create posts more often than not
        assert create_posts > 20  # ~40% expected
    
    @pytest.mark.asyncio
    async def test_lurker_agent_prefers_passive(self, mock_memory):
        """Test LurkerAgent prefers passive actions."""
        profile = AgentProfile(name="Lurker", agent_type="lurker")
        agent = LurkerAgent("id3", profile, mock_memory)
        await agent.initialize({})
        
        actions = [await agent.decide_action() for _ in range(100)]
        do_nothings = sum(1 for a in actions if a.action_type == ActionType.DO_NOTHING)
        
        # Should do nothing ~50% of the time
        assert do_nothings > 40


class TestAgentFactory:
    """Test agent factory function."""
    
    def test_create_default_agent(self, mock_memory):
        """Test creating default agent."""
        agent = create_agent("default", memory_adapter=mock_memory)
        
        assert isinstance(agent, DefaultAgent)
        assert agent.profile.agent_type == "default"
    
    def test_create_influencer_agent(self, mock_memory):
        """Test creating influencer agent."""
        agent = create_agent("influencer", memory_adapter=mock_memory)
        
        assert isinstance(agent, InfluencerAgent)
    
    def test_create_lurker_agent(self, mock_memory):
        """Test creating lurker agent."""
        agent = create_agent("lurker", memory_adapter=mock_memory)
        
        assert isinstance(agent, LurkerAgent)
    
    def test_create_with_custom_id(self, mock_memory):
        """Test creating agent with custom ID."""
        agent = create_agent("default", agent_id="custom_123", memory_adapter=mock_memory)
        
        assert agent.id == "custom_123"
    
    def test_create_unknown_type_raises(self, mock_memory):
        """Test that unknown agent type raises error."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("unknown_type", memory_adapter=mock_memory)
    
    def test_create_with_profile_params(self, mock_memory):
        """Test creating agent with profile parameters."""
        agent = create_agent(
            "default",
            memory_adapter=mock_memory,
            name="CustomName",
            backstory="Custom backstory",
            goals=["goal1", "goal2"]
        )
        
        assert agent.profile.name == "CustomName"
        assert agent.profile.backstory == "Custom backstory"
        assert agent.profile.goals == ["goal1", "goal2"]


class TestActionTypes:
    """Test action types and energy costs."""
    
    @pytest.mark.asyncio
    async def test_energy_cost_calculation(self, mock_memory):
        """Test that actions cost energy."""
        profile = AgentProfile(name="Test", agent_type="default")
        agent = DefaultAgent("id", profile, mock_memory)
        await agent.initialize({})
        
        initial_energy = agent.state.energy
        
        # Execute a costly action
        action = Action(action_type=ActionType.CREATE_POST)
        await agent.execute_action(action)
        
        assert agent.state.energy < initial_energy
    
    @pytest.mark.asyncio
    async def test_do_nothing_recover_energy(self, mock_memory):
        """Test that DO_NOTHING recovers energy."""
        profile = AgentProfile(name="Test", agent_type="default")
        agent = DefaultAgent("id", profile, mock_memory)
        await agent.initialize({})
        
        # Deplete some energy first
        agent.state.energy = 50
        
        action = Action(action_type=ActionType.DO_NOTHING)
        await agent.execute_action(action)
        
        assert agent.state.energy > 50
    
    @pytest.mark.asyncio
    async def test_low_energy_forces_rest(self, mock_memory):
        """Test that low energy forces DO_NOTHING."""
        profile = AgentProfile(name="Test", agent_type="default")
        agent = DefaultAgent("id", profile, mock_memory)
        await agent.initialize({})
        
        agent.state.energy = 10  # Very low
        
        action = await agent.decide_action()
        assert action.action_type == ActionType.DO_NOTHING
