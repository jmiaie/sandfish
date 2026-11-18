"""Researcher agents: SandFish agents that propose and evaluate variants.

Each round every researcher proposes one variant from the search space:
with probability ``EPSILON`` a fresh uniform draw ("explore"), otherwise the
ledger's current best training variant with one parameter re-drawn
("exploit"). It then scores the variant on the **train** split through the
attached evaluator, reusing the ledger's earlier result for a repeat, and
records the evaluation in the trial ledger.

Agents decide in parallel from the ledger as it stood at the start of the
round and evaluate one after another in agent order, so a seeded run is
reproducible given a deterministic evaluator.

These agent types are only registered by :func:`register_search_agents`
(``run_search`` and the MCP search tools call it). The social-simulation
agents and defaults are untouched.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..agents.base_agent import Action, ActionType, BaseAgent, register_agent_type
from .evaluator import (
    TRAIN,
    EvaluationResult,
    Evaluator,
    as_evaluator,
    evaluator_name,
    run_evaluation,
)
from .ledger import TrialLedger, variant_key
from .space import validate_space

Repair = Callable[[Dict[str, Any]], Dict[str, Any]]

SEARCH_AGENT_TYPES = ("researcher", "researcher_explorer", "researcher_exploiter")


@dataclass
class SearchContext:
    """What :func:`attach_search` puts in a simulation's shared context."""

    space: Dict[str, List[Any]]
    evaluator: Evaluator
    ledger: TrialLedger
    repair: Optional[Repair] = None
    min_n: int = 0
    remember: bool = False
    run_id: Optional[str] = None
    evaluator_name: str = field(default="")

    def __post_init__(self) -> None:
        if not self.evaluator_name:
            self.evaluator_name = evaluator_name(self.evaluator)

    def metrics(self) -> Dict[str, float]:
        c = self.ledger.counts()
        train = c["by_split"][TRAIN]
        return {
            "search_trials": float(train["trials"]),
            "search_evaluator_calls": float(train["evaluator_calls"]),
            "search_unique_variants": float(train["unique_variants"]),
            "search_errors": float(train["errors"]),
        }


def evaluate_and_record(
    ctx: SearchContext,
    variant: Mapping[str, Any],
    *,
    source: str,
    split: str = TRAIN,
    round_number: Optional[int] = None,
) -> Dict[str, Any]:
    """Score ``variant`` (or reuse an earlier result) and append a ledger row.

    Evaluator exceptions are recorded as an ``error`` row, never raised: a
    failed evaluation is still a trial and must be counted.
    """
    try:
        json.dumps(dict(variant), allow_nan=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"variant is not JSON-serialisable (check repair=): {exc}") from exc
    cached = ctx.ledger.lookup(variant, split, ctx.evaluator_name)
    if cached is not None:
        return ctx.ledger.record(
            source=source, variant=variant, split=split, result=cached, cached=True,
            round_number=round_number, evaluator=ctx.evaluator_name, run_id=ctx.run_id,
        )
    try:
        result: Optional[EvaluationResult] = run_evaluation(ctx.evaluator, variant, split)
        error = None
    except Exception as exc:  # noqa: BLE001 - any evaluator failure is a recorded trial
        result, error = None, f"{type(exc).__name__}: {exc}"[:500]
    common = dict(source=source, variant=variant, split=split, round_number=round_number,
                  evaluator=ctx.evaluator_name, run_id=ctx.run_id)
    try:
        return ctx.ledger.record(result=result, error=error, **common)
    except (TypeError, ValueError) as exc:
        # The evaluator ran, so the trial must still be counted.
        if result is None:
            raise
        return ctx.ledger.record(error=f"unrecordable result: {exc}"[:500], **common)


class ResearcherAgent(BaseAgent):
    """Proposes and evaluates one variant per round (train split only)."""

    EPSILON: float = 0.5
    LOW_ENERGY_THRESHOLD: float = -1.0  # researchers never rest
    RECIPROCITY: float = 0.0

    def _search(self) -> SearchContext:
        ctx = self.simulation_context.get("search")
        if not isinstance(ctx, SearchContext):
            raise RuntimeError("attach_search() was not called for this simulation")
        return ctx

    def propose(self) -> Dict[str, Any]:
        ctx = self._search()
        best = ctx.ledger.best(TRAIN, min_n=ctx.min_n)
        keys = sorted(ctx.space)
        if best is None or self.rng.random() < self.EPSILON:
            variant = {k: self.rng.choice(ctx.space[k]) for k in keys}
            mode = "explore"
        else:
            variant = dict(best["variant"])
            k = self.rng.choice(keys)
            variant[k] = self.rng.choice(ctx.space[k])
            mode = "exploit"
        if ctx.repair is not None:
            variant = dict(ctx.repair(dict(variant)))
        return {"variant": variant, "mode": mode}

    def _select_action(self, context: Mapping[str, Any]) -> Action:
        return Action(action_type=ActionType.CUSTOM, metadata=self.propose())

    async def _action_custom(self, action: Action) -> None:
        ctx = self._search()
        variant = action.metadata["variant"]
        row = await asyncio.to_thread(
            evaluate_and_record, ctx, variant, source=self.id, round_number=self.round_number
        )
        action.metadata["ledger_seq"] = row["seq"]
        if ctx.remember and self.memory is not None and not row["cached"] and row["result"]:
            try:
                self.memory.add_entity(
                    name=f"variant:{variant_key(variant)}",
                    entity_type="variant",
                    valid_from=self.current_date,
                    attributes={"score": row["result"]["score"], "n": row["result"]["n"],
                                "split": TRAIN, "ledger_seq": row["seq"]},
                )
            except Exception:
                pass  # memory is best-effort
        self.state.energy = 100.0


class ExplorerAgent(ResearcherAgent):
    """Mostly random draws (EPSILON 0.9)."""

    EPSILON = 0.9


class ExploiterAgent(ResearcherAgent):
    """Mostly one-parameter mutations of the current best (EPSILON 0.2)."""

    EPSILON = 0.2


def register_search_agents() -> None:
    """Register ``researcher``, ``researcher_explorer``, ``researcher_exploiter``."""
    register_agent_type("researcher", ResearcherAgent)
    register_agent_type("researcher_explorer", ExplorerAgent)
    register_agent_type("researcher_exploiter", ExploiterAgent)


def attach_search(
    orchestrator: Any,
    simulation_id: str,
    *,
    space: Mapping[str, Any],
    evaluator: Any,
    ledger: TrialLedger,
    repair: Optional[Repair] = None,
    min_n: int = 0,
    remember: bool = False,
) -> SearchContext:
    """Attach a search to a created (not yet started) simulation.

    ``evaluator`` is an object with ``evaluate(variant, split)`` or a
    callable ``(variant, split)``. ``repair`` may map a proposed variant to a
    valid one (e.g. enforce ``low < high``). ``min_n`` ignores results with a
    smaller sample size when picking the best variant to exploit.
    """
    sim = orchestrator.simulations.get(simulation_id)
    if sim is None:
        raise KeyError(f"unknown simulation {simulation_id!r}")
    if min_n < 0:
        raise ValueError("min_n must be >= 0")
    ctx = SearchContext(
        space=validate_space(space),
        evaluator=as_evaluator(evaluator),
        ledger=ledger,
        repair=repair,
        min_n=int(min_n),
        remember=remember,
        run_id=simulation_id,
    )
    sim["context"]["search"] = ctx
    return ctx
