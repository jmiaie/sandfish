"""
Base agent definitions for SandFish.

Provides agent types for swarm simulations with OMPA-native memory.
"""

import random
import uuid
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

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


# KG predicate recorded for each agent-targeted action.
_INTERACTION_PREDICATE = {
    ActionType.FOLLOW: "follows",
    ActionType.LIKE_POST: "liked",
    ActionType.COMMENT: "commented_on",
    ActionType.REPOST: "reposted",
}


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


class _LazyContext(Mapping):
    """A read-only mapping that defers expensive lookups until a key is read.

    Some context keys (`memories`, `related_entities`) cost a semantic search
    and a graph traversal against the memory backend. Agents that decide from
    their own state never read them, so computing every key up front spent one
    search plus one traversal per agent per round on values that were then
    discarded.

    Deferred keys are resolved on first access and cached, so the cost is paid
    only by agents that actually consult memory, and only once per round.
    """

    __slots__ = ("_values", "_deferred")

    def __init__(
        self,
        values: Dict[str, Any],
        deferred: Dict[str, Callable[[], Any]],
    ):
        self._values = dict(values)
        self._deferred = dict(deferred)

    def __getitem__(self, key: str) -> Any:
        if key in self._values:
            return self._values[key]
        if key in self._deferred:
            # pop first: a raising factory must not be retried on every access.
            factory = self._deferred.pop(key)
            value = factory()
            self._values[key] = value
            return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from self._values
        yield from self._deferred

    def __len__(self) -> int:
        return len(self._values) + len(self._deferred)

    def copy(self) -> Dict[str, Any]:
        """Resolve every key and return a plain dict.

        Provided because callers commonly treat context as a dict; this forces
        the deferred lookups, so prefer plain key access when you only need a
        subset.
        """
        return {key: self[key] for key in list(self)}


class BaseAgent(ABC):
    """
    Abstract base class for all SandFish agents.

    Subclasses implement `_select_action` to pick the next action type given
    the gathered context. Helpers for choosing targets and synthesising
    content are provided so subclasses don't need to duplicate that logic.
    """

    # Subclasses can lower this to be more conservative with energy spending.
    LOW_ENERGY_THRESHOLD: float = 20.0

    # Probability that a targeted action goes to a peer who previously
    # engaged with this agent (read from OMPA interaction facts) instead of a
    # uniformly random peer. 0 disables memory-driven targeting.
    RECIPROCITY: float = 0.5

    def __init__(
        self,
        agent_id: str,
        profile: AgentProfile,
        memory_adapter: Optional[OMPAMemoryAdapter] = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        rng: Optional[random.Random] = None,
    ):
        self.id = agent_id
        self.profile = profile
        self.memory = memory_adapter

        # Per-agent RNG. Agents must never touch the `random` module globals:
        # a shared global stream makes a run's outcome depend on how agent
        # calls interleave, so the same seed would not reproduce the same run.
        # Defaults to an independently seeded instance, preserving the
        # unseeded behaviour of earlier versions.
        self.rng = rng if rng is not None else random.Random()

        self.state = AgentStateData()
        self.status = AgentState.INITIALIZING

        self.simulation_id: Optional[str] = None
        # Shared by all agents of one simulation (set by the orchestrator);
        # holds per-simulation resources such as the LLM call budget.
        self.simulation_context: Dict[str, Any] = {}
        self.round_number: int = 0
        # Simulation calendar date (YYYY-MM-DD) set by the orchestrator each
        # round; used to date memory writes. None means "today".
        self.current_date: Optional[str] = None
        self.reciprocity: float = self.RECIPROCITY
        self._engaged_cache: Optional[Tuple[int, "Counter[str]"]] = None
        self.memory_targeted_actions: int = 0

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
                name=self.id,
                entity_type=self.profile.agent_type,
                valid_from=self.current_date,
                attributes={
                    "display_name": self.profile.name,
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

        predicate = _INTERACTION_PREDICATE.get(action.action_type)
        if predicate and action.target and self.memory is not None:
            try:
                self.memory.record_interaction(
                    self.id, predicate, action.target, date=self.current_date
                )
            except Exception:
                pass  # memory is best-effort; never break the round

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
            "memory_targeted_actions": self.memory_targeted_actions,
        }

    # ----- Internal helpers -----

    def _gather_context(self) -> Mapping[str, Any]:
        """Decision context. Memory lookups are lazy: they only run (and
        cost anything) if a policy actually reads ``memories``,
        ``related_entities`` or ``engaged_peers``."""
        return _LazyContext(
            {
                "current_state": self.state,
                "round": self.round_number,
                "date": self.current_date,
                "peers": list(self._peers),
            },
            {
                "memories": self._recall_memories,
                "related_entities": self._recall_related,
                "engaged_peers": lambda: dict(self._engaged_peers()),
            },
        )

    def _recall_memories(self) -> List[Any]:
        if self.memory is None:
            return []
        try:
            hits = self.memory.search(
                query=f"{self.id} {self.profile.name}", limit=5,
                reference_date=self.current_date,
            )
            return list(hits) if isinstance(hits, list) else []
        except Exception:
            return []

    def _recall_related(self) -> List[Any]:
        if self.memory is None:
            return []
        try:
            related = self.memory.get_related_entities(self.id)
            return list(related) if isinstance(related, list) else []
        except Exception:
            return []

    def _engaged_peers(self) -> "Counter[str]":
        """Peers who have engaged with this agent, with counts (cached per round)."""
        if self._engaged_cache is not None and self._engaged_cache[0] == self.round_number:
            return self._engaged_cache[1]
        counts: "Counter[str]" = Counter()
        if self.memory is not None:
            try:
                facts = self.memory.get_interactions(
                    self.id, direction="incoming", as_of=self.current_date
                )
                if isinstance(facts, list):
                    counts.update(f.subject for f in facts)
            except Exception:
                counts = Counter()
        self._engaged_cache = (self.round_number, counts)
        return counts

    def _choose_target(self) -> Optional[str]:
        """Pick a peer; with probability ``reciprocity`` prefer peers who
        engaged with us before (weighted by how often), else uniform."""
        if not self._peers:
            return None
        if self.memory is not None and self.reciprocity > 0 and self.rng.random() < self.reciprocity:
            engaged = self._engaged_peers()
            candidates = [p for p in self._peers if engaged.get(p)]
            if candidates:
                self.memory_targeted_actions += 1
                weights = [engaged[p] for p in candidates]
                return self.rng.choices(candidates, weights=weights)[0]
        return self.rng.choice(self._peers)

    def _enrich_action(self, action: Action) -> Action:
        """Fill in target/content for actions that need them."""
        if action.action_type in _AGENT_TARGETED_ACTIONS and not action.target:
            action.target = self._choose_target()

        if action.action_type == ActionType.CREATE_POST and not action.content:
            content = self._generate_post_content()
            # Drawn from the agent RNG rather than `secrets` so a seeded run
            # reproduces identical post IDs. These are simulation labels, not
            # security tokens, so unpredictability is not a requirement.
            post_id = f"post_{self.rng.getrandbits(32):08x}"
            action.content = content
            action.metadata.setdefault("post_id", post_id)
            self.state.posts_created.append(post_id)

        return action

    def _generate_post_content(self) -> str:
        """Generate placeholder content for a post.

        Subclasses with LLM access can override this with real generation.
        """
        topic = self.rng.choice(self.profile.goals) if self.profile.goals else "the swarm"
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
    def _select_action(self, context: Mapping[str, Any]) -> Action:
        """Select an action type given the gathered context.

        Draw any randomness from `self.rng`, never the `random` module, or
        seeded runs will not reproduce.
        """

    def _log_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        if self.memory is None:
            return
        try:
            self.memory.record_event(
                event_type=event_type,
                description=self._describe_event(event_type, metadata),
                metadata=metadata,
                date=self.current_date,
            )
        except Exception:
            # Logging must never break the simulation loop.
            pass

    def _describe_event(self, event_type: str, metadata: Dict[str, Any]) -> str:
        """Human-readable, search-friendly event text."""
        if event_type == "ACTION_EXECUTED":
            verb = metadata.get("action_type", "acted")
            target = metadata.get("target")
            return f"{self.id} ({self.profile.name}) {verb}" + (f" -> {target}" if target else "")
        return f"{self.id} ({self.profile.name}): {event_type}"

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

    def _select_action(self, context: Mapping[str, Any]) -> Action:
        actions = [
            ActionType.CREATE_POST,
            ActionType.LIKE_POST,
            ActionType.FOLLOW,
            ActionType.COMMENT,
            ActionType.SEARCH,
            ActionType.DO_NOTHING,
        ]
        weights = [0.2, 0.3, 0.2, 0.15, 0.1, 0.05]
        return Action(action_type=self.rng.choices(actions, weights=weights)[0])


class InfluencerAgent(BaseAgent):
    """Prefers content creation and growing connections."""

    LOW_ENERGY_THRESHOLD = 15.0
    RECIPROCITY = 0.3  # broadcasters engage widely

    def _select_action(self, context: Mapping[str, Any]) -> Action:
        actions = [
            ActionType.CREATE_POST,
            ActionType.FOLLOW,
            ActionType.LIKE_POST,
            ActionType.COMMENT,
            ActionType.DO_NOTHING,
        ]
        weights = [0.4, 0.25, 0.2, 0.1, 0.05]
        return Action(action_type=self.rng.choices(actions, weights=weights)[0])


class LurkerAgent(BaseAgent):
    """Mostly observes; occasionally interacts."""

    LOW_ENERGY_THRESHOLD = 10.0
    RECIPROCITY = 0.8  # lurkers mostly respond to people who engaged them

    def _select_action(self, context: Mapping[str, Any]) -> Action:
        actions = [
            ActionType.DO_NOTHING,
            ActionType.SEARCH,
            ActionType.LIKE_POST,
            ActionType.COMMENT,
            ActionType.CREATE_POST,
        ]
        weights = [0.5, 0.25, 0.15, 0.07, 0.03]
        return Action(action_type=self.rng.choices(actions, weights=weights)[0])


def _default_agent_name(agent_id: str) -> str:
    """Readable, per-simulation-unique default name.

    Orchestrator IDs look like ``<sim_id>_agent_<i>``; the old
    ``agent_id[:4]`` took the sim prefix, so every agent shared one name.
    """
    if "_agent_" in agent_id:
        return f"Agent_{agent_id.rsplit('_agent_', 1)[1]}"
    return f"Agent_{agent_id[:8]}"


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
    rng: Optional[random.Random] = None,
    **kwargs: Any,
) -> BaseAgent:
    """Factory: construct an agent of the requested type.

    Pass `rng` (a `random.Random`) to make the agent's choices reproducible.
    Omit it and the agent seeds itself from the OS, as before.
    """
    if agent_type not in AGENT_TYPES:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Known: {sorted(AGENT_TYPES)}"
        )

    agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"

    profile = AgentProfile(
        name=kwargs.get("name", _default_agent_name(agent_id)),
        agent_type=agent_type,
        traits=kwargs.get("traits", {}),
        preferences=kwargs.get("preferences", {}),
        backstory=kwargs.get("backstory", ""),
        goals=kwargs.get("goals", []),
    )

    agent_class = AGENT_TYPES[agent_type]
    if rng is None:
        # Omit the kwarg entirely so custom agent classes registered before
        # `rng` existed — which may override __init__ without it — keep working.
        return agent_class(agent_id, profile, memory_adapter)
    return agent_class(agent_id, profile, memory_adapter, rng=rng)


def register_agent_type(name: str, agent_class: type) -> None:
    """Register a custom agent class under a name."""
    if not isinstance(agent_class, type) or not issubclass(agent_class, BaseAgent):
        raise ValueError("Agent class must inherit from BaseAgent")
    AGENT_TYPES[name] = agent_class
