"""
Base agent definitions for SandFish.

Provides agent types for swarm simulations with OMPA-native memory.
"""

import random
import secrets
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from ..memory.ompa_adapter import OMPAMemoryAdapter


# Soft cap on retained action history; older entries are dropped.
DEFAULT_HISTORY_LIMIT = 1000


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


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


# Actions that should target another agent (or one of their posts).
_AGENT_TARGETED_ACTIONS = {
    ActionType.FOLLOW,
    ActionType.LIKE_POST,
    ActionType.COMMENT,
    ActionType.REPOST,
}


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
class AgentStateData:
    """Dynamic state of an agent during simulation."""
    energy: float = 100.0
    mood: float = 50.0
    reputation: float = 0.0
    connections: List[str] = field(default_factory=list)
    posts_created: List[str] = field(default_factory=list)
    memory_summary: str = ""


@dataclass
class Action:
    """An action taken by an agent."""
    action_type: ActionType
    target: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)


# Energy costs by action type.
_ENERGY_COSTS = {
    ActionType.CREATE_POST: 10.0,
    ActionType.LIKE_POST: 2.0,
    ActionType.REPOST: 5.0,
    ActionType.FOLLOW: 3.0,
    ActionType.COMMENT: 8.0,
    ActionType.SEARCH: 4.0,
    ActionType.DO_NOTHING: 0.5,
    ActionType.CUSTOM: 5.0,
}


class BaseAgent(ABC):
    """
    Abstract base class for all SandFish agents.

    Subclasses implement `_select_action` to pick the next action type given
    the gathered context. Helpers for choosing targets and synthesising
    content are provided so subclasses don't need to duplicate that logic.
    """

    # Subclasses can lower this to be more conservative with energy spending.
    LOW_ENERGY_THRESHOLD: float = 20.0

    def __init__(
        self,
        agent_id: str,
        profile: AgentProfile,
        memory_adapter: Optional[OMPAMemoryAdapter] = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ):
        self.id = agent_id
        self.profile = profile
        self.memory = memory_adapter

        self.state = AgentStateData()
        self.status = AgentState.INITIALIZING

        self.simulation_id: Optional[str] = None
        self.round_number: int = 0

        # Bounded action log. Use a deque so old entries drop in O(1).
        self.action_history: Deque[Action] = deque(maxlen=history_limit)

        # Visible peers, set by the orchestrator each round.
        self._peers: List[str] = []

    # ----- Lifecycle -----

    async def initialize(self, seed_data: Dict[str, Any]) -> None:
        """Initialize the agent for a simulation."""
        self.status = AgentState.INITIALIZING

        if self.memory is not None:
            self.memory.add_entity(
                name=self.profile.name,
                entity_type=self.profile.agent_type,
                attributes={
                    "agent_id": self.id,
                    "traits": self.profile.traits,
                    "backstory": self.profile.backstory,
                    "goals": self.profile.goals,
                },
            )

        if "initial_energy" in seed_data:
            self.state.energy = float(seed_data["initial_energy"])
        if "initial_mood" in seed_data:
            self.state.mood = float(seed_data["initial_mood"])

        self.status = AgentState.IDLE

        self._log_event(
            "AGENT_INITIALIZED",
            {
                "agent_id": self.id,
                "profile": self.profile.name,
                "type": self.profile.agent_type,
            },
        )

    def set_peers(self, peer_ids: List[str]) -> None:
        """Inform the agent of currently-visible peer IDs (set each round)."""
        self._peers = [pid for pid in peer_ids if pid != self.id]

    async def decide_action(self) -> Action:
        """Decide and return the next action."""
        self.status = AgentState.THINKING

        # Low-energy agents always rest, regardless of subclass policy.
        if self.state.energy < self.LOW_ENERGY_THRESHOLD:
            self.status = AgentState.IDLE
            return Action(action_type=ActionType.DO_NOTHING)

        context = self._gather_context()
        action = self._select_action(context)
        action = self._enrich_action(action)

        self.status = AgentState.IDLE
        return action

    async def execute_action(self, action: Action) -> None:
        """Execute an action and update agent state."""
        self.status = AgentState.ACTING

        handler = self._action_handler(action.action_type)
        await handler(action)

        self.action_history.append(action)
        self.state.energy = max(0.0, self.state.energy - _ENERGY_COSTS.get(action.action_type, 5.0))
        self.round_number += 1

        self.status = AgentState.IDLE

        self._log_event(
            "ACTION_EXECUTED",
            {
                "agent_id": self.id,
                "action_type": action.action_type.value,
                "target": action.target,
            },
        )

    def get_state(self) -> Dict[str, Any]:
        """Snapshot for serialization."""
        return {
            "id": self.id,
            "profile": {
                "name": self.profile.name,
                "type": self.profile.agent_type,
            },
            "state": {
                "energy": self.state.energy,
                "mood": self.state.mood,
                "reputation": self.state.reputation,
                "status": self.status.value,
                "connections": list(self.state.connections),
                "posts_created": list(self.state.posts_created),
            },
            "round": self.round_number,
            "action_count": len(self.action_history),
        }

    # ----- Internal helpers -----

    def _gather_context(self) -> Dict[str, Any]:
        """Gather context from memory and current state."""
        memories: List[Any] = []
        related: List[Any] = []
        if self.memory is not None:
            try:
                memories = self.memory.search(
                    query=f"agent {self.profile.name} recent actions",
                    limit=5,
                )
                related = self.memory.get_related_entities(self.profile.name)
            except Exception:
                # Memory backends are best-effort during decision-making.
                memories, related = [], []

        return {
            "memories": memories,
            "related_entities": related,
            "current_state": self.state,
            "round": self.round_number,
            "peers": list(self._peers),
        }

    def _enrich_action(self, action: Action) -> Action:
        """Fill in target/content for actions that need them."""
        if action.action_type in _AGENT_TARGETED_ACTIONS and not action.target:
            if self._peers:
                action.target = random.choice(self._peers)

        if action.action_type == ActionType.CREATE_POST and not action.content:
            content = self._generate_post_content()
            post_id = f"post_{secrets.token_hex(4)}"
            action.content = content
            action.metadata.setdefault("post_id", post_id)
            self.state.posts_created.append(post_id)

        return action

    def _generate_post_content(self) -> str:
        """Generate placeholder content for a post.

        Subclasses with LLM access can override this with real generation.
        """
        topic = random.choice(self.profile.goals) if self.profile.goals else "the swarm"
        return f"{self.profile.name}: thoughts on {topic}"

    def _action_handler(self, action_type: ActionType):
        """Return the coroutine handler for an action type."""
        handlers = {
            ActionType.CREATE_POST: self._action_create_post,
            ActionType.LIKE_POST: self._action_like_post,
            ActionType.REPOST: self._action_repost,
            ActionType.FOLLOW: self._action_follow,
            ActionType.COMMENT: self._action_comment,
            ActionType.SEARCH: self._action_search,
            ActionType.DO_NOTHING: self._action_do_nothing,
            ActionType.CUSTOM: self._action_custom,
        }
        return handlers.get(action_type, self._action_custom)

    @abstractmethod
    def _select_action(self, context: Dict[str, Any]) -> Action:
        """Select an action type given the gathered context."""

    def _log_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        if self.memory is None:
            return
        try:
            self.memory.record_event(
                event_type=event_type,
                description=f"Agent {self.profile.name}: {event_type}",
                metadata=metadata,
            )
        except Exception:
            # Logging must never break the simulation loop.
            pass

    # ----- Default action implementations -----

    async def _action_create_post(self, action: Action) -> None:
        # Content already populated in _enrich_action; reputation gains a tick.
        self.state.reputation += 0.5

    async def _action_like_post(self, action: Action) -> None:
        if action.target:
            self.state.reputation += 0.1

    async def _action_repost(self, action: Action) -> None:
        if action.target:
            self.state.reputation += 0.2

    async def _action_follow(self, action: Action) -> None:
        if action.target and action.target not in self.state.connections:
            self.state.connections.append(action.target)

    async def _action_comment(self, action: Action) -> None:
        if action.target:
            self.state.reputation += 0.15

    async def _action_search(self, action: Action) -> None:
        # Search slightly improves "informedness" (mood proxy).
        self.state.mood = min(100.0, self.state.mood + 1.0)

    async def _action_do_nothing(self, action: Action) -> None:
        # Rest recovers energy.
        self.state.energy = min(100.0, self.state.energy + 5.0)

    async def _action_custom(self, action: Action) -> None:
        # No-op by default; subclasses can override.
        return None


class DefaultAgent(BaseAgent):
    """Balanced agent with broad action distribution."""

    def _select_action(self, context: Dict[str, Any]) -> Action:
        actions = [
            ActionType.CREATE_POST,
            ActionType.LIKE_POST,
            ActionType.FOLLOW,
            ActionType.COMMENT,
            ActionType.SEARCH,
            ActionType.DO_NOTHING,
        ]
        weights = [0.2, 0.3, 0.2, 0.15, 0.1, 0.05]
        return Action(action_type=random.choices(actions, weights=weights)[0])


class InfluencerAgent(BaseAgent):
    """Prefers content creation and growing connections."""

    LOW_ENERGY_THRESHOLD = 15.0

    def _select_action(self, context: Dict[str, Any]) -> Action:
        actions = [
            ActionType.CREATE_POST,
            ActionType.FOLLOW,
            ActionType.LIKE_POST,
            ActionType.COMMENT,
            ActionType.DO_NOTHING,
        ]
        weights = [0.4, 0.25, 0.2, 0.1, 0.05]
        return Action(action_type=random.choices(actions, weights=weights)[0])


class LurkerAgent(BaseAgent):
    """Mostly observes; occasionally interacts."""

    LOW_ENERGY_THRESHOLD = 10.0

    def _select_action(self, context: Dict[str, Any]) -> Action:
        actions = [
            ActionType.DO_NOTHING,
            ActionType.SEARCH,
            ActionType.LIKE_POST,
            ActionType.COMMENT,
            ActionType.CREATE_POST,
        ]
        weights = [0.5, 0.25, 0.15, 0.07, 0.03]
        return Action(action_type=random.choices(actions, weights=weights)[0])


# Agent factory registry.
AGENT_TYPES: Dict[str, type] = {
    "default": DefaultAgent,
    "influencer": InfluencerAgent,
    "lurker": LurkerAgent,
}


def create_agent(
    agent_type: str,
    agent_id: Optional[str] = None,
    memory_adapter: Optional[OMPAMemoryAdapter] = None,
    **kwargs: Any,
) -> BaseAgent:
    """Factory: construct an agent of the requested type."""
    if agent_type not in AGENT_TYPES:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Known: {sorted(AGENT_TYPES)}"
        )

    agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"

    profile = AgentProfile(
        name=kwargs.get("name", f"Agent_{agent_id[:4]}"),
        agent_type=agent_type,
        traits=kwargs.get("traits", {}),
        preferences=kwargs.get("preferences", {}),
        backstory=kwargs.get("backstory", ""),
        goals=kwargs.get("goals", []),
    )

    agent_class = AGENT_TYPES[agent_type]
    return agent_class(agent_id, profile, memory_adapter)


def register_agent_type(name: str, agent_class: type) -> None:
    """Register a custom agent class under a name."""
    if not isinstance(agent_class, type) or not issubclass(agent_class, BaseAgent):
        raise ValueError("Agent class must inherit from BaseAgent")
    AGENT_TYPES[name] = agent_class
