"""``LLMAgent``: an agent whose policy is a structured-output LLM call.

Off by default. Opt in with::

    from sandfish.llm import client_from_env
    from sandfish.llm.agent import enable_llm_agents

    enable_llm_agents(client_from_env("gemini"), max_calls=200)  # per simulation
    # then use agent_types=["llm"] (or mix: ["llm", "default", "lurker"])

Any failure (no client, budget exhausted, timeout, invalid output) falls
back to the heuristic ``DefaultAgent`` policy for that round, so a run never
fails because a provider did.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional

from ..agents.base_agent import (
    _AGENT_TARGETED_ACTIONS,
    AGENT_TYPES,
    Action,
    ActionType,
    AgentState,
    DefaultAgent,
    register_agent_type,
)
from .base import LLMBudget, LLMBudgetExhausted, LLMClient, LLMError, action_schema, usage_sink

logger = logging.getLogger("sandfish.llm")

# Actions offered to the model (CUSTOM is for subclasses).
_LLM_ACTIONS = [a for a in ActionType if a is not ActionType.CUSTOM]
_MAX_VISIBLE_PEERS = 20
_MAX_CONTENT = 280
_UNLIMITED = 2**62


class LLMAgent(DefaultAgent):
    """Decides each round by asking an LLM for a schema-constrained action."""

    client: Optional[LLMClient] = None
    # Process-wide tally (and optional overall cap) across every simulation.
    budget: Optional[LLMBudget] = None
    # Call cap for each simulation; every simulation gets its own budget, so
    # one long run can't starve the next. None = no per-simulation cap.
    max_calls_per_simulation: Optional[int] = 200
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
            hot += self.rng.sample(rest, min(room, len(rest)))
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
        sim_budget = self.simulation_budget()
        if sim_budget is not None and not sim_budget.try_acquire():
            return await self._fallback("simulation budget exhausted")
        # HTTP clients enforce (and tally tokens on) the process-wide budget
        # themselves; for clients without one, the agent enforces it here.
        if budget is not None and getattr(client, "budget", None) is not budget:
            if not budget.try_acquire():
                if sim_budget is not None:
                    sim_budget.release()  # no call was made
                return await self._fallback("budget exhausted")

        context = self._gather_context()
        engaged = context["engaged_peers"]
        peers = self._visible_peers(engaged)
        schema = action_schema([a.value for a in _LLM_ACTIONS], peers)
        sink = usage_sink.set(sim_budget)  # per-simulation token tally
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
            if sim_budget is not None and isinstance(exc, LLMBudgetExhausted):
                sim_budget.release()  # the client refused before any request
            return await self._fallback(f"{type(exc).__name__}: {exc}")
        finally:
            usage_sink.reset(sink)

        action = self._enrich_action(action)
        self.status = AgentState.IDLE
        return action

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["llm"] = {"calls": self.llm_calls, "fallbacks": self.llm_fallbacks}
        return state

    def simulation_budget(self) -> Optional[LLMBudget]:
        """This simulation's call budget, created on first use.

        Lives in the orchestrator's per-simulation context, so it is shared
        by the simulation's agents and released with the simulation.
        """
        cap = type(self).max_calls_per_simulation
        if cap is None:
            return None
        ctx = self.simulation_context
        budget = ctx.get("llm_budget")
        if budget is None:
            budget = ctx["llm_budget"] = LLMBudget(max_calls=cap)
        return budget

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
    max_total_calls: Optional[int] = None,
) -> LLMBudget:
    """Register ``LLMAgent`` under ``type_name``.

    ``max_calls`` caps LLM calls *per simulation* (each simulation gets its
    own budget; ``None`` = uncapped). ``max_total_calls`` optionally caps
    calls across all simulations in this process.

    Passing ``client=None`` registers the type in heuristic-fallback mode.
    Returns the process-wide budget, which tallies calls and tokens across
    every simulation (per-simulation tallies are in the results' metrics).
    """
    LLMAgent.client = client
    LLMAgent.timeout_s = timeout_s
    LLMAgent._warned_fallback = False
    LLMAgent.max_calls_per_simulation = max_calls
    LLMAgent.budget = LLMBudget(
        max_calls=max_total_calls if max_total_calls is not None else _UNLIMITED
    )
    if client is not None and hasattr(client, "budget") and getattr(client, "budget") is None:
        client.budget = LLMAgent.budget  # enforce + record token usage per call
    if type_name not in AGENT_TYPES or AGENT_TYPES[type_name] is not LLMAgent:
        register_agent_type(type_name, LLMAgent)
    return LLMAgent.budget


SERVER_OPT_IN = "SANDFISH_SERVER_LLM_AGENTS"
DEFAULT_SERVER_CALLS_PER_SIMULATION = 200
DEFAULT_SERVER_TOTAL_CALLS = 2000


def _cap(env: Mapping[str, str], name: str, default: int) -> Optional[int]:
    raw = str(env.get(name, "") or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise LLMError(f"{name} must be a whole number, got {raw!r}") from None
    return None if value <= 0 else value


def enable_llm_agents_from_env(env: Optional[Mapping[str, str]] = None) -> Optional[LLMBudget]:
    """Let a server's callers run ``llm`` agents, if its operator asked for it.

    The HTTP API and the MCP server call this at start-up. Callers of a
    server choose the agent types, and every ``llm`` agent spends the
    operator's provider key, so nothing is enabled unless the operator sets
    ``SANDFISH_SERVER_LLM_AGENTS=1``. A provider configured for the command
    line (``SANDFISH_LLM_PROVIDER`` alone) does not turn it on.

    - ``SANDFISH_LLM_PROVIDER``: required with the opt-in.
    - ``SANDFISH_LLM_MAX_CALLS``: requests per simulation (default 200).
    - ``SANDFISH_LLM_MAX_TOTAL_CALLS``: requests for the life of the process
      (default 2,000). ``0`` lifts either cap.

    Returns the process-wide budget, or None when the opt-in is not set.
    Raises ``LLMError`` when it is set but the rest is missing or invalid, so
    a server does not start half-configured.
    """
    env = os.environ if env is None else env
    if str(env.get(SERVER_OPT_IN, "") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    provider = str(env.get("SANDFISH_LLM_PROVIDER", "") or "").strip()
    if not provider:
        raise LLMError(f"{SERVER_OPT_IN} is set but SANDFISH_LLM_PROVIDER is not")
    per_simulation = _cap(env, "SANDFISH_LLM_MAX_CALLS", DEFAULT_SERVER_CALLS_PER_SIMULATION)
    total = _cap(env, "SANDFISH_LLM_MAX_TOTAL_CALLS", DEFAULT_SERVER_TOTAL_CALLS)
    from .providers import client_from_env

    client = client_from_env(provider)
    budget = enable_llm_agents(client, max_calls=per_simulation, max_total_calls=total)
    logger.warning(
        "LLM agents are available to this server's callers: %s:%s, at most %s requests "
        "per simulation and %s in total.",
        client.name,
        client.model,
        per_simulation if per_simulation is not None else "unlimited",
        total if total is not None else "unlimited",
    )
    return budget


async def close_llm_client() -> None:
    """Close the registered client's connection pool, if it has one."""
    aclose = getattr(LLMAgent.client, "aclose", None)
    if aclose is not None:
        await aclose()


def disable_llm_agents(type_name: str = "llm") -> None:
    """Unregister the LLM agent type and drop the client."""
    LLMAgent.client = None
    LLMAgent.budget = None
    LLMAgent.max_calls_per_simulation = 200
    if AGENT_TYPES.get(type_name) is LLMAgent:
        del AGENT_TYPES[type_name]
