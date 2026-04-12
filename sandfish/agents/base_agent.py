"""
Base agent definitions for SandFish.

Provides agent types for swarm simulations with OMPA-native memory.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random

from ..memory.ompa_adapter import OMPAMemoryAdapter


class AgentState(Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COMMUNICATING = "communicating"
    TERMINATED = "terminated"


class ActionType(Enum):
    """Types of actions agents can take."""
    CREATE_POST = "create_post"
    LIKE_POST = "like_post"
    REPOST = "repost"
    FOLLOW = "follow"
    COMMENT = "comment"
    SEARCH = "search"
    DO_NOTHING = "do_nothing"
    CUSTOM = "custom"


@dataclass
class AgentProfile:
    """Static profile defining agent personality."""
    name: str
    agent_type: str
    traits: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    backstory: str = ""
    goals: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    """Dynamic state of an agent during simulation."""
    energy: float = 100.0
    mood: float = 50.0
    reputation: float = 0.0
    connections: List[str] = field(default_factory=list)
    memory_summary: str = ""
    recent_actions: List[Dict] = field(default_factory=list)


@dataclass
class Action:
    """An action taken by an agent."""
    action_type: ActionType
    target: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class BaseAgent(ABC):
    """
    Abstract base class for all SandFish agents.
    
    Features:
    - OMPA-native memory integration
    - Personality-driven decision making
    - Action execution framework
    - State management and persistence
    """
    
    def __init__(self, 
                 agent_id: str,
                 profile: AgentProfile,
                 memory_adapter: OMPAMemoryAdapter):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique agent identifier
            profile: Static personality profile
            memory_adapter: OMPA memory adapter
        """
        self.id = agent_id
        self.profile = profile
        self.memory = memory_adapter
        
        # Dynamic state
        self.state = AgentState()
        self.status = AgentState.INITIALIZING
        
        # Simulation context
        self.simulation_id: Optional[str] = None
        self.round_number: int = 0
        
        # Action history
        self.action_history: List[Action] = []
        
    async def initialize(self, seed_data: Dict[str, Any]) -> None:
        """
        Initialize agent with simulation seed data.
        
        Args:
            seed_data: Simulation initialization data
        """
        self.status = AgentState.INITIALIZING
        
        # Store agent entity in OMPA knowledge graph
        self.memory.add_entity(
            name=self.profile.name,
            entity_type=self.profile.agent_type,
            attributes={
                'agent_id': self.id,
                'traits': self.profile.traits,
                'backstory': self.profile.backstory,
                'goals': self.profile.goals
            }
        )
        
        # Initialize state from seed data
        if 'initial_energy' in seed_data:
            self.state.energy = seed_data['initial_energy']
        if 'initial_mood' in seed_data:
            self.state.mood = seed_data['initial_mood']
        
        self.status = AgentState.IDLE
        
        # Log initialization
        self._log_event("AGENT_INITIALIZED", {
            'agent_id': self.id,
            'profile': self.profile.name,
            'type': self.profile.agent_type
        })
    
    async def decide_action(self) -> Action:
        """
        Decide next action based on state and memory.
        
        Returns:
            Action to execute
        """
        self.status = AgentState.THINKING
        
        # Query memory for relevant context
        context = self._gather_context()
        
        # Use personality to decide
        action = self._select_action(context)
        
        self.status = AgentState.IDLE
        return action
    
    async def execute_action(self, action: Action) -> None:
        """
        Execute a decided action.
        
        Args:
            action: Action to execute
        """
        self.status = AgentState.ACTING
        
        # Execute based on type
        if action.action_type == ActionType.CREATE_POST:
            await self._action_create_post(action)
        elif action.action_type == ActionType.LIKE_POST:
            await self._action_like_post(action)
        elif action.action_type == ActionType.FOLLOW:
            await self._action_follow(action)
        elif action.action_type == ActionType.COMMENT:
            await self._action_comment(action)
        elif action.action_type == ActionType.SEARCH:
            await self._action_search(action)
        elif action.action_type == ActionType.DO_NOTHING:
            await self._action_do_nothing(action)
        else:
            await self._action_custom(action)
        
        # Record action
        self.action_history.append(action)
        self.state.recent_actions.append({
            'type': action.action_type.value,
            'timestamp': action.timestamp.isoformat()
        })
        
        # Update state
        self.state.energy -= self._calculate_energy_cost(action)
        self.round_number += 1
        
        self.status = AgentState.IDLE
        
        # Log action
        self._log_event("ACTION_EXECUTED", {
            'agent_id': self.id,
            'action_type': action.action_type.value,
            'target': action.target
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state for serialization."""
        return {
            'id': self.id,
            'profile': {
                'name': self.profile.name,
                'type': self.profile.agent_type
            },
            'state': {
                'energy': self.state.energy,
                'mood': self.state.mood,
                'reputation': self.state.reputation,
                'status': self.status.value
            },
            'round': self.round_number,
            'action_count': len(self.action_history)
        }
    
    def _gather_context(self) -> Dict[str, Any]:
        """Gather relevant context from memory."""
        # Search for relevant memories
        recent_memories = self.memory.search(
            query=f"agent {self.profile.name} recent actions",
            limit=5
        )
        
        # Get related entities
        related = self.memory.get_related_entities(self.profile.name)
        
        return {
            'memories': recent_memories,
            'related_entities': related,
            'current_state': self.state,
            'round': self.round_number
        }
    
    @abstractmethod
    def _select_action(self, context: Dict[str, Any]) -> Action:
        """
        Select action based on context and personality.
        
        Args:
            context: Gathered context
            
        Returns:
            Selected action
        """
        pass
    
    def _calculate_energy_cost(self, action: Action) -> float:
        """Calculate energy cost of an action."""
        base_costs = {
            ActionType.CREATE_POST: 10.0,
            ActionType.LIKE_POST: 2.0,
            ActionType.REPOST: 5.0,
            ActionType.FOLLOW: 3.0,
            ActionType.COMMENT: 8.0,
            ActionType.SEARCH: 4.0,
            ActionType.DO_NOTHING: 0.5,
            ActionType.CUSTOM: 5.0
        }
        return base_costs.get(action.action_type, 5.0)
    
    def _log_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        """Log event to OMPA."""
        self.memory.record_event(
            event_type=event_type,
            description=f"Agent {self.profile.name}: {event_type}",
            metadata=metadata
        )
    
    # Action implementations (to be overridden)
    
    async def _action_create_post(self, action: Action) -> None:
        """Create a post."""
        pass
    
    async def _action_like_post(self, action: Action) -> None:
        """Like a post."""
        pass
    
    async def _action_follow(self, action: Action) -> None:
        """Follow another agent."""
        if action.target:
            self.state.connections.append(action.target)
    
    async def _action_comment(self, action: Action) -> None:
        """Comment on content."""
        pass
    
    async def _action_search(self, action: Action) -> None:
        """Search for content."""
        pass
    
    async def _action_do_nothing(self, action: Action) -> None:
        """Rest and recover."""
        self.state.energy = min(100.0, self.state.energy + 5.0)
    
    async def _action_custom(self, action: Action) -> None:
        """Custom action."""
        pass


class DefaultAgent(BaseAgent):
    """Default agent with basic decision making."""
    
    def _select_action(self, context: Dict[str, Any]) -> Action:
        """Simple random action selection."""
        # If low energy, rest
        if self.state.energy < 20:
            return Action(action_type=ActionType.DO_NOTHING)
        
        # Otherwise random action
        actions = [
            ActionType.CREATE_POST,
            ActionType.LIKE_POST,
            ActionType.FOLLOW,
            ActionType.COMMENT,
            ActionType.SEARCH,
            ActionType.DO_NOTHING
        ]
        
        weights = [0.2, 0.3, 0.2, 0.15, 0.1, 0.05]
        selected = random.choices(actions, weights=weights)[0]
        
        return Action(action_type=selected)


class InfluencerAgent(BaseAgent):
    """Agent that prioritizes creating content and gaining followers."""
    
    def _select_action(self, context: Dict[str, Any]) -> Action:
        """Prefer content creation and social actions."""
        if self.state.energy < 15:
            return Action(action_type=ActionType.DO_NOTHING)
        
        actions = [
            ActionType.CREATE_POST,
            ActionType.FOLLOW,
            ActionType.LIKE_POST,
            ActionType.COMMENT,
            ActionType.DO_NOTHING
        ]
        
        weights = [0.4, 0.25, 0.2, 0.1, 0.05]
        selected = random.choices(actions, weights=weights)[0]
        
        return Action(action_type=selected)


class LurkerAgent(BaseAgent):
    """Agent that mostly observes with occasional interaction."""
    
    def _select_action(self, context: Dict[str, Any]) -> Action:
        """Prefer passive actions."""
        actions = [
            ActionType.DO_NOTHING,
            ActionType.SEARCH,
            ActionType.LIKE_POST,
            ActionType.COMMENT,
            ActionType.CREATE_POST
        ]
        
        weights = [0.5, 0.25, 0.15, 0.07, 0.03]
        selected = random.choices(actions, weights=weights)[0]
        
        return Action(action_type=selected)


# Agent factory
AGENT_TYPES = {
    'default': DefaultAgent,
    'influencer': InfluencerAgent,
    'lurker': LurkerAgent,
}


def create_agent(agent_type: str,
                 agent_id: Optional[str] = None,
                 memory_adapter: Optional[OMPAMemoryAdapter] = None,
                 **kwargs) -> BaseAgent:
    """
    Factory function to create agents.
    
    Args:
        agent_type: Type of agent to create
        agent_id: Optional ID (generated if not provided)
        memory_adapter: OMPA memory adapter
        **kwargs: Additional profile parameters
        
    Returns:
        Configured agent instance
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
    
    profile = AgentProfile(
        name=kwargs.get('name', f"Agent_{agent_id[:4]}"),
        agent_type=agent_type,
        traits=kwargs.get('traits', {}),
        backstory=kwargs.get('backstory', ''),
        goals=kwargs.get('goals', [])
    )
    
    agent_class = AGENT_TYPES[agent_type]
    return agent_class(agent_id, profile, memory_adapter)


def register_agent_type(name: str, agent_class: type) -> None:
    """Register a custom agent type."""
    if not issubclass(agent_class, BaseAgent):
        raise ValueError("Agent class must inherit from BaseAgent")
    AGENT_TYPES[name] = agent_class
