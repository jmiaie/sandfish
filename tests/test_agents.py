"""
Tests for agent definitions.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from sandfish.agents.base_agent import (
    Action,
    ActionType,
    AgentProfile,
    AgentState,
    DefaultAgent,
    InfluencerAgent,
    LurkerAgent,
    create_agent,
    register_agent_type,
    BaseAgent,
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


@pytest.fixture
def sample_profile():
    return AgentProfile(
        name="TestAgent",
        agent_type="default",
        traits={"creativity": 0.8, "sociability": 0.6},
        backstory="A test agent",
        goals=["test_goal"],
    )


class TestBaseAgent:
    @pytest.fixture
    def agent(self, mock_memory, sample_profile):
        return DefaultAgent("test_id", sample_profile, mock_memory)

    def test_initialization(self, agent, sample_profile):
        assert agent.id == "test_id"
        assert agent.profile == sample_profile
        assert agent.status == AgentState.INITIALIZING
        assert agent.state.energy == 100.0

    @pytest.mark.asyncio
    async def test_initialize(self, agent, mock_memory):
        await agent.initialize({"initial_energy": 80})

        assert agent.status == AgentState.IDLE
        assert agent.state.energy == 80
        mock_memory.add_entity.assert_called_once()
        mock_memory.record_event.assert_called()

    def test_get_state(self, agent):
        state = agent.get_state()
        assert state["id"] == "test_id"
        assert state["profile"]["name"] == "TestAgent"
        assert "energy" in state["state"]
        assert "mood" in state["state"]

    @pytest.mark.asyncio
    async def test_decide_action_returns_action(self, agent):
        await agent.initialize({})
        action = await agent.decide_action()
        assert isinstance(action, Action)
        assert isinstance(action.action_type, ActionType)

    @pytest.mark.asyncio
    async def test_execute_action_appends_history(self, agent):
        await agent.initialize({})
        agent.state.energy = 50
        action = Action(action_type=ActionType.DO_NOTHING)

        await agent.execute_action(action)
        assert len(agent.action_history) == 1
        # DO_NOTHING recovers energy.
        assert agent.state.energy > 50

    @pytest.mark.asyncio
    async def test_history_is_bounded(self, mock_memory, sample_profile):
        # Use a small bound to exercise the deque cap quickly.
        agent = DefaultAgent("a", sample_profile, mock_memory, history_limit=5)
        await agent.initialize({})
        for _ in range(20):
            await agent.execute_action(Action(action_type=ActionType.DO_NOTHING))
        assert len(agent.action_history) == 5

    @pytest.mark.asyncio
    async def test_targeted_action_picks_peer(self, agent):
        await agent.initialize({})
        agent.set_peers(["a", "b", "c"])
        action = await agent._enrich_action_for_test(
            Action(action_type=ActionType.FOLLOW)
        )
        assert action.target in {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_create_post_generates_content(self, agent):
        await agent.initialize({})
        action = await agent._enrich_action_for_test(
            Action(action_type=ActionType.CREATE_POST)
        )
        assert action.content
        assert "post_id" in action.metadata


# Helper used by the targeted test above to expose the (intentionally private)
# enrichment method without monkeypatching.
async def _enrich(self, action):
    return self._enrich_action(action)


BaseAgent._enrich_action_for_test = _enrich  # type: ignore[attr-defined]


class TestAgentTypes:
    @pytest.mark.asyncio
    async def test_default_agent_action_selection(self, mock_memory):
        profile = AgentProfile(name="Default", agent_type="default")
        agent = DefaultAgent("id1", profile, mock_memory)
        await agent.initialize({})
        action = await agent.decide_action()
        assert isinstance(action.action_type, ActionType)

    @pytest.mark.asyncio
    async def test_influencer_agent_prefers_content(self, mock_memory):
        profile = AgentProfile(name="Influencer", agent_type="influencer")
        agent = InfluencerAgent("id2", profile, mock_memory)
        await agent.initialize({})

        actions = [await agent.decide_action() for _ in range(200)]
        create_posts = sum(1 for a in actions if a.action_type == ActionType.CREATE_POST)
        # Probability ~0.4; expect at least 25% in 200 trials.
        assert create_posts > 50

    @pytest.mark.asyncio
    async def test_lurker_agent_prefers_passive(self, mock_memory):
        profile = AgentProfile(name="Lurker", agent_type="lurker")
        agent = LurkerAgent("id3", profile, mock_memory)
        await agent.initialize({})

        actions = [await agent.decide_action() for _ in range(200)]
        do_nothings = sum(1 for a in actions if a.action_type == ActionType.DO_NOTHING)
        assert do_nothings > 80


class TestAgentFactory:
    def test_create_default_agent(self, mock_memory):
        agent = create_agent("default", memory_adapter=mock_memory)
        assert isinstance(agent, DefaultAgent)
        assert agent.profile.agent_type == "default"

    def test_create_influencer_agent(self, mock_memory):
        agent = create_agent("influencer", memory_adapter=mock_memory)
        assert isinstance(agent, InfluencerAgent)

    def test_create_lurker_agent(self, mock_memory):
        agent = create_agent("lurker", memory_adapter=mock_memory)
        assert isinstance(agent, LurkerAgent)

    def test_create_with_custom_id(self, mock_memory):
        agent = create_agent("default", agent_id="custom_123", memory_adapter=mock_memory)
        assert agent.id == "custom_123"

    def test_create_unknown_type_raises(self, mock_memory):
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("unknown_type", memory_adapter=mock_memory)

    def test_create_with_profile_params(self, mock_memory):
        agent = create_agent(
            "default",
            memory_adapter=mock_memory,
            name="CustomName",
            backstory="Custom backstory",
            goals=["goal1", "goal2"],
        )
        assert agent.profile.name == "CustomName"
        assert agent.profile.backstory == "Custom backstory"
        assert agent.profile.goals == ["goal1", "goal2"]

    def test_register_custom_type(self, mock_memory):
        class MyAgent(DefaultAgent):
            pass

        register_agent_type("my_agent", MyAgent)
        agent = create_agent("my_agent", memory_adapter=mock_memory)
        assert isinstance(agent, MyAgent)


class TestActionTypes:
    @pytest.mark.asyncio
    async def test_energy_cost_calculation(self, mock_memory):
        profile = AgentProfile(name="Test", agent_type="default")
        agent = DefaultAgent("id", profile, mock_memory)
        await agent.initialize({})

        initial_energy = agent.state.energy
        await agent.execute_action(Action(action_type=ActionType.CREATE_POST))
        assert agent.state.energy < initial_energy

    @pytest.mark.asyncio
    async def test_do_nothing_recover_energy(self, mock_memory):
        profile = AgentProfile(name="Test", agent_type="default")
        agent = DefaultAgent("id", profile, mock_memory)
        await agent.initialize({})

        agent.state.energy = 50
        await agent.execute_action(Action(action_type=ActionType.DO_NOTHING))
        assert agent.state.energy > 50

    @pytest.mark.asyncio
    async def test_low_energy_forces_rest(self, mock_memory):
        profile = AgentProfile(name="Test", agent_type="default")
        agent = DefaultAgent("id", profile, mock_memory)
        await agent.initialize({})

        agent.state.energy = 5  # Below the threshold.
        action = await agent.decide_action()
        assert action.action_type == ActionType.DO_NOTHING

    @pytest.mark.asyncio
    async def test_lurker_low_energy_floor(self, mock_memory):
        profile = AgentProfile(name="Test", agent_type="lurker")
        agent = LurkerAgent("id", profile, mock_memory)
        await agent.initialize({})

        agent.state.energy = 5
        action = await agent.decide_action()
        assert action.action_type == ActionType.DO_NOTHING

    @pytest.mark.asyncio
    async def test_energy_cannot_go_negative(self, mock_memory):
        profile = AgentProfile(name="Test", agent_type="default")
        agent = DefaultAgent("id", profile, mock_memory)
        await agent.initialize({})

        agent.state.energy = 1.0
        await agent.execute_action(Action(action_type=ActionType.CREATE_POST))  # cost 10
        assert agent.state.energy == 0.0
