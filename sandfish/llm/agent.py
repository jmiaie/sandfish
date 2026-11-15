"""``LLMAgent``: an agent whose policy is a structured-output LLM call.

Off by default. Opt in with::

    from sandfish.llm import client_from_env
    from sandfish.llm.agent import enable_llm_agents

    enable_llm_agents(client_from_env("gemini"), max_calls=200)
    # then use agent_types=["llm"] (or mix: ["llm", "default", "lurker"])

Any failure (no client, budget exhausted, timeout, invalid output) falls
back to the heuristic ``DefaultAgent`` policy for that round, so a run never
fails because a provider did.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Optional

from ..agents.base_agent import (
    _AGENT_TARGETED_ACTIONS,
    AGENT_TYPES,
    Action,
    ActionType,
    AgentState,
    DefaultAgent,
    register_agent_type,
)
from .base import LLMBudget, LLMClient, action_schema

logger = logging.getLogger("sandfish.llm")

# Actions offered to the model (CUSTOM is for subclasses).
_LLM_ACTIONS = [a for a in ActionType if a is not ActionType.CUSTOM]
_MAX_VISIBLE_PEERS = 20
_MAX_CONTENT = 280


class LLMAgent(DefaultAgent):
    """Decides each round by asking an LLM for a schema-constrained action."""

    client: Optional[LLMClient] = None
    budget: Optional[LLMBudget] = None
    timeout_s: float = 30.0
    max_tokens: int = 200
    _warned_fallback: bool = False

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.llm_calls = 0
        self.llm_fallbacks = 0
        self.last_reason: Optional[str] = None

    def _system_prompt(self) -> str:
        p = self.profile
        traits = ", ".join(f"{k}={v}" for k, v in (p.traits or {}).items()) or "none"
        goals = "; ".join(p.goals) or "engage with the community"
        return (
            "You control one agent in a round-based social simulation. "
            "Choose exactly one action per round by calling the schema. "
            "Prefer actions consistent with the persona and memory. "
            "Use target 'none' for actions that do not target a peer.\n"
            f"Persona: {p.name} ({p.agent_type}). Traits: {traits}. Goals: {goals}. "
            f"Backstory: {p.backstory or 'n/a'}"
        )

    def _visible_peers(self, engaged: Dict[str, int]) -> List[str]:
        """Peers who engaged us first (most engaged first), then a fresh
        random sample of the rest so large populations aren't truncated
        to the same prefix every round."""
        hot = sorted((p for p in self._peers if engaged.get(p)), key=lambda p: -engaged[p])
        hot = hot[:_MAX_VISIBLE_PEERS]
        rest = [p for p in self._peers if not engaged.get(p)]
        room = _MAX_VISIBLE_PEERS - len(hot)
        if room > 0 and rest:
            hot += random.sample(rest, min(room, len(rest)))
        return hot

    def _user_prompt(self, context: Any, peers: List[str], engaged: Dict[str, int]) -> str:
        memories = []
        for m in (context["memories"] or [])[:5]:
            text = getattr(m, "content", str(m))
            memories.append(text[:200])
        payload = {
            "round": self.round_number,
            "date": self.current_date,
            "energy": round(self.state.energy, 1),
            "mood": round(self.state.mood, 1),
            "reputation": round(self.state.reputation, 2),
            "following": self.state.connections[-10:],
            "peers_who_engaged_you": {p: engaged[p] for p in peers if engaged.get(p)},
            "visible_peers": peers,
            "recent_memory": memories,
        }
        return json.dumps(payload, ensure_ascii=False)

    async def decide_action(self) -> Action:
        self.status = AgentState.THINKING
        if self.state.energy < self.LOW_ENERGY_THRESHOLD:
            self.status = AgentState.IDLE
            return Action(action_type=ActionType.DO_NOTHING)

        client = type(self).client
        budget = type(self).budget
        if client is None:
            return await self._fallback("no client")
        # HTTP clients enforce (and tally tokens on) the shared budget
        # themselves; for clients without one, the agent enforces it here.
        if budget is not None and getattr(client, "budget", None) is not budget:
            if not budget.try_acquire():
                return await self._fallback("budget exhausted")

        context = self._gather_context()
        engaged = context["engaged_peers"]
        peers = self._visible_peers(engaged)
        schema = action_schema([a.value for a in _LLM_ACTIONS], peers)
        try:
            raw = await asyncio.wait_for(
                client.complete_json(
                    system=self._system_prompt(),
                    prompt=self._user_prompt(context, peers, engaged),
                    schema=schema,
                    max_tokens=self.max_tokens,
                ),
                timeout=self.timeout_s,
            )
            self.llm_calls += 1
            action = self._to_action(raw, peers)
        except Exception as exc:  # provider errors are already key-redacted
            return await self._fallback(f"{type(exc).__name__}: {exc}")

        action = self._enrich_action(action)
        self.status = AgentState.IDLE
        return action

    def _to_action(self, raw: Dict[str, Any], peers: List[str]) -> Action:
        if not isinstance(raw, dict):
            raise ValueError("decision is not an object")
        action_type = ActionType(str(raw.get("action")))
        if action_type is ActionType.CUSTOM:
            raise ValueError("custom action not allowed")
        target = raw.get("target")
        target = target if isinstance(target, str) and target in peers else None
        if action_type not in _AGENT_TARGETED_ACTIONS:
            target = None
        content = raw.get("content")
        content = content.strip()[:_MAX_CONTENT] if isinstance(content, str) and content.strip() else None
        self.last_reason = str(raw.get("reason", ""))[:200] or None
        return Action(
            action_type=action_type,
            target=target,
            content=content,
            metadata={"policy": "llm", "reason": self.last_reason},
        )

    async def _fallback(self, why: str) -> Action:
        self.llm_fallbacks += 1
        cls = type(self)
        if not cls._warned_fallback:
            cls._warned_fallback = True
            logger.warning("LLM agents falling back to heuristics (first cause: %s)", why)
        else:
            logger.debug("LLMAgent %s fallback: %s", self.id, why)
        action = self._enrich_action(self._select_action(self._gather_context()))
        action.metadata.setdefault("policy", "heuristic_fallback")
        self.status = AgentState.IDLE
        return action


def enable_llm_agents(
    client: Optional[LLMClient],
    *,
    max_calls: Optional[int] = 200,
    timeout_s: float = 30.0,
    type_name: str = "llm",
) -> Optional[LLMBudget]:
    """Register ``LLMAgent`` under ``type_name`` with a shared call budget.

    Passing ``client=None`` registers the type in heuristic-fallback mode.
    Returns the budget so callers can report calls/tokens afterwards.
    """
    LLMAgent.client = client
    LLMAgent.timeout_s = timeout_s
    LLMAgent._warned_fallback = False
    LLMAgent.budget = LLMBudget(max_calls=max_calls) if max_calls is not None else None
    if client is not None and hasattr(client, "budget") and getattr(client, "budget") is None:
        client.budget = LLMAgent.budget  # enforce + record token usage per call
    if type_name not in AGENT_TYPES or AGENT_TYPES[type_name] is not LLMAgent:
        register_agent_type(type_name, LLMAgent)
    return LLMAgent.budget


def disable_llm_agents(type_name: str = "llm") -> None:
    """Unregister the LLM agent type and drop the client."""
    LLMAgent.client = None
    LLMAgent.budget = None
    if AGENT_TYPES.get(type_name) is LLMAgent:
        del AGENT_TYPES[type_name]
