"""High-level helpers: run a swarm search or a plain grid, validate on the
held-out split, and summarise a ledger."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..core.orchestrator import SimulationConfig
from .agents import (
    Repair,
    SearchContext,
    attach_search,
    evaluate_and_record,
    register_search_agents,
)
from .evaluator import TEST, TRAIN, as_evaluator
from .ledger import TrialLedger, variant_key
from .space import grid_size, iter_grid, validate_space

DEFAULT_AGENT_TYPES = ("researcher_explorer", "researcher_exploiter")

MULTIPLE_TESTING_NOTE = (
    "Every variant in this ledger was a hypothesis tested on the train split. "
    "Use counts.by_split.train.unique_variants (plus any variants tried outside "
    "this ledger) as the number of trials N in a multiple-testing correction "
    "(e.g. Bonferroni, deflated Sharpe ratio). Do not use the number of "
    "finalists. Agent-picked variants must be validated on held-out data "
    "(validate_holdout / the test split) before they mean anything."
)


async def run_search(
    orchestrator: Any,
    *,
    space: Mapping[str, Any],
    evaluator: Any,
    ledger: TrialLedger,
    name: str = "search",
    num_agents: int = 4,
    max_rounds: int = 10,
    seed: Optional[int] = None,
    agent_types: Sequence[str] = DEFAULT_AGENT_TYPES,
    repair: Optional[Repair] = None,
    min_n: int = 0,
    remember: bool = False,
) -> Dict[str, Any]:
    """Create, attach and run a swarm search to completion; return its summary.

    ``num_agents * max_rounds`` evaluations are recorded (repeats included).
    """
    register_search_agents()
    sim_id = orchestrator.create_simulation(
        SimulationConfig(
            name=name,
            num_agents=num_agents,
            max_rounds=max_rounds,
            seed=seed,
            agent_types=list(agent_types),
            checkpoint_interval=0,
        )
    )
    attach_search(
        orchestrator, sim_id, space=space, evaluator=evaluator, ledger=ledger,
        repair=repair, min_n=min_n, remember=remember,
    )
    result = await orchestrator.run_simulation(sim_id)
    out = summarize(ledger, min_n=min_n)
    out.update(
        simulation_id=sim_id,
        status=result.status.value,
        rounds_completed=result.rounds_completed,
        error=result.error_message,
    )
    return out


def run_grid(
    space: Mapping[str, Any],
    evaluator: Any,
    ledger: TrialLedger,
    *,
    max_variants: int = 10_000,
    repair: Optional[Repair] = None,
    run_id: Optional[str] = None,
    min_n: int = 0,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Evaluate every variant of ``space`` on the train split.

    The right tool whenever the space is small enough to enumerate: exact
    trial count, no path dependence. Refuses grids above ``max_variants``.
    ``should_stop()`` is checked before each evaluation; ``stopped`` in the
    result says whether the grid was cut short.
    """
    space = validate_space(space)
    size = grid_size(space)
    if size > max_variants:
        raise ValueError(f"grid has {size} variants; max_variants is {max_variants}")
    ctx = SearchContext(
        space=space, evaluator=as_evaluator(evaluator), ledger=ledger,
        repair=repair, min_n=min_n, run_id=run_id,
    )
    stopped = False
    for variant in iter_grid(space):
        if should_stop is not None and should_stop():
            stopped = True
            break
        if repair is not None:
            variant = dict(repair(dict(variant)))
        evaluate_and_record(ctx, variant, source="grid")
    out = summarize(ledger, min_n=min_n)
    out["grid_size"] = size
    out["stopped"] = stopped
    return out


def validate_holdout(
    ledger: TrialLedger,
    evaluator: Any,
    *,
    top_k: int = 3,
    min_n: int = 0,
    variants: Optional[Sequence[Mapping[str, Any]]] = None,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Score finalists on the **test** split and record those rows too.

    Finalists are the ``top_k`` train variants (or ``variants`` if given).
    Pick them *before* calling this, and call it once: every test evaluation
    is logged, so re-selecting after peeking shows up in the ledger.
    """
    ctx = SearchContext(space={}, evaluator=as_evaluator(evaluator), ledger=ledger, run_id=run_id)
    if variants is None:
        finalists = [dict(r["variant"]) for r in ledger.top(top_k, split=TRAIN, min_n=min_n)]
    else:
        finalists = [dict(v) for v in variants]
    train = {r["key"]: r for r in ledger.top(10**9, split=TRAIN)}
    out = []
    for variant in finalists:
        row = evaluate_and_record(ctx, variant, source="holdout", split=TEST)
        tr = train.get(variant_key(variant))
        out.append({
            "variant": variant,
            "train_score": tr["result"]["score"] if tr else None,
            "train_n": tr["result"]["n"] if tr else None,
            "test_score": row["result"]["score"] if row["result"] else None,
            "test_n": row["result"]["n"] if row["result"] else None,
            "test_error": row["error"],
            "ledger_seq": row["seq"],
        })
    return out


def _brief(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    res = row["result"] or {}
    return {"variant": row["variant"], "score": res.get("score"), "n": res.get("n"),
            "metadata": res.get("metadata"), "seq": row["seq"], "source": row["source"]}


def summarize(ledger: TrialLedger, *, top: int = 5, min_n: int = 0) -> Dict[str, Any]:
    """Counts, chain verification, best/top train variants and test results."""
    test = {r["key"]: r for r in ledger.top(10**9, split=TEST)}
    top_train = []
    for r in ledger.top(top, split=TRAIN, min_n=min_n):
        item = _brief(r)
        t = test.get(r["key"])
        item["test_score"] = t["result"]["score"] if t else None  # type: ignore[index]
        top_train.append(item)
    return {
        "counts": ledger.counts(),
        "integrity": ledger.verify(),
        "head": ledger.head,
        "best_train": _brief(ledger.best(TRAIN, min_n=min_n)),
        "top_train": top_train,
        "min_n": min_n,
        "multiple_testing": MULTIPLE_TESTING_NOTE,
    }
