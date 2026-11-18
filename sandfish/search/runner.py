"""High-level helpers: run a swarm search or a plain grid, validate on the
held-out split, and summarise a ledger."""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

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
from .overfit import overfit_report
from .space import grid_size, iter_grid, validate_space

# Maps a train ledger row (``{"variant", "result": {"score", "n", "metadata"}, ...}``)
# to a hashable fingerprint; finalists with equal fingerprints are duplicates.
DedupeKey = Callable[[Mapping[str, Any]], Optional[Hashable]]

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
    lock_holdout: bool = False,
) -> Dict[str, Any]:
    """Create, attach and run a swarm search to completion; return its summary.

    ``num_agents * max_rounds`` evaluations are recorded (repeats included).
    ``lock_holdout=True`` locks the test split on the ledger first, so only
    :func:`validate_holdout` can score it (the lock is logged).
    """
    register_search_agents()
    if lock_holdout:
        ledger.lock_holdout(reason="run_search")
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
    valid: Optional[Callable[[Dict[str, Any]], bool]] = None,
    lock_holdout: bool = False,
) -> Dict[str, Any]:
    """Evaluate every variant of ``space`` on the train split.

    The right tool whenever the space is small enough to enumerate: exact
    trial count, no path dependence. Refuses grids above ``max_variants``
    (judged on the raw grid size, before ``valid`` filtering).
    ``should_stop()`` is checked before each evaluation; ``stopped`` in the
    result says whether the grid was cut short.

    ``valid(variant) -> bool`` skips combinations that make no sense (e.g.
    ``fast >= slow``). A skipped combination is never evaluated and never
    recorded, so the ledger's trial count N stays exact: it counts the
    hypotheses actually tested, not the raw grid. ``valid`` sees the raw
    combination, before ``repair``. The result reports ``grid_size`` (raw),
    ``valid_size`` and ``skipped_invalid``. ``lock_holdout=True`` locks the
    test split on the ledger first (see :meth:`TrialLedger.lock_holdout`).
    """
    space = validate_space(space)
    size = grid_size(space)
    if size > max_variants:
        raise ValueError(f"grid has {size} variants; max_variants is {max_variants}")
    ctx = SearchContext(
        space=space, evaluator=as_evaluator(evaluator), ledger=ledger,
        repair=repair, min_n=min_n, run_id=run_id,
    )
    if lock_holdout:
        ledger.lock_holdout(reason="run_grid")
    variants = list(iter_grid(space, valid))  # raw size is capped above
    stopped = False
    for variant in variants:
        if should_stop is not None and should_stop():
            stopped = True
            break
        if repair is not None:
            variant = dict(repair(dict(variant)))
        evaluate_and_record(ctx, variant, source="grid")
    out = summarize(ledger, min_n=min_n)
    out["grid_size"] = size
    out["valid_size"] = len(variants)
    out["skipped_invalid"] = size - len(variants)
    out["stopped"] = stopped
    return out


class HoldoutResults(list):
    """The list of finalist results ``validate_holdout`` returns, plus
    ``skipped``: finalists dropped as duplicates (``dedupe_key``), each with
    the finalist it duplicated. A plain list for every other purpose."""

    skipped: List[Dict[str, Any]]

    def __init__(self, items: Sequence[Dict[str, Any]] = (), skipped: Optional[List[Dict[str, Any]]] = None):
        super().__init__(items)
        self.skipped = list(skipped or [])


def dedupe_by_metadata(*fields: str) -> DedupeKey:
    """Fingerprint = the given metadata fields of the train result, e.g.
    ``dedupe_by_metadata("trade_set_hash")`` for evaluators that report a hash
    of the trades a variant produced. Rows lacking any field are never deduped."""
    if not fields:
        raise ValueError("name at least one metadata field")

    def key(row: Mapping[str, Any]) -> Optional[Hashable]:
        meta = (row.get("result") or {}).get("metadata") or {}
        if any(f not in meta for f in fields):
            return None
        return tuple(variant_key({f: meta[f]}) for f in fields)

    return key


def dedupe_by_score(params: Optional[Sequence[str]] = None, ndigits: int = 9) -> DedupeKey:
    """Fingerprint = (score rounded to ``ndigits``, sample size) and, if
    ``params`` is given, those parameters' values. With no ``params`` any two
    variants with the same score and ``n`` collapse; name the parameters that
    really change behaviour to dedupe only variants that agree on them."""

    def key(row: Mapping[str, Any]) -> Optional[Hashable]:
        res = row.get("result") or {}
        if res.get("score") is None:
            return None
        var = row.get("variant") or {}
        return (round(float(res["score"]), ndigits), res.get("n"),
                tuple(variant_key({p: var.get(p)}) for p in (params or ())))

    return key


def _select_finalists(
    ledger: TrialLedger, top_k: int, min_n: int, dedupe_key: Optional[DedupeKey]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if dedupe_key is None:
        return [dict(r["variant"]) for r in ledger.top(top_k, split=TRAIN, min_n=min_n)], []
    finalists: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    seen: Dict[Hashable, Dict[str, Any]] = {}
    for row in ledger.top(10**9, split=TRAIN, min_n=min_n):
        if len(finalists) >= max(0, top_k):
            break
        fp = dedupe_key(row)
        if fp is not None and fp in seen:
            skipped.append({"variant": dict(row["variant"]), "duplicate_of": seen[fp],
                            "train_score": row["result"]["score"]})
            continue
        variant = dict(row["variant"])
        if fp is not None:
            seen[fp] = variant
        finalists.append(variant)
    return finalists, skipped


def validate_holdout(
    ledger: TrialLedger,
    evaluator: Any,
    *,
    top_k: int = 3,
    min_n: int = 0,
    variants: Optional[Sequence[Mapping[str, Any]]] = None,
    run_id: Optional[str] = None,
    dedupe_key: Optional[DedupeKey] = None,
    allow_repeat: bool = False,
) -> HoldoutResults:
    """Score finalists on the **test** split and record those rows too.

    Finalists are the ``top_k`` train variants (or ``variants`` if given).
    Pick them *before* calling this, and call it once: every test evaluation
    is logged, so re-selecting after peeking shows up in the ledger.

    ``dedupe_key(train_row) -> fingerprint`` drops near-identical finalists
    (same fingerprint as a better-ranked one) so they do not fill ``top_k``;
    the next distinct variant takes the slot, and the dropped ones are listed
    in ``.skipped`` on the result. See :func:`dedupe_by_metadata` and
    :func:`dedupe_by_score`. Explicit ``variants`` are used as given.

    If the ledger's holdout guard is on (``ledger.lock_holdout()``), this is
    the only way to score the test split, and it works once: a second call
    raises ``HoldoutLockedError`` unless ``allow_repeat=True`` (logged).
    """
    ctx = SearchContext(space={}, evaluator=as_evaluator(evaluator), ledger=ledger, run_id=run_id)
    skipped: List[Dict[str, Any]] = []
    if variants is None:
        finalists, skipped = _select_finalists(ledger, top_k, min_n, dedupe_key)
    else:
        finalists = [dict(v) for v in variants]
    train = {r["key"]: r for r in ledger.top(10**9, split=TRAIN)}
    out = []
    with ledger.holdout_access(finalists, allow_repeat=allow_repeat):
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
    return HoldoutResults(out, skipped)


def _brief(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    res = row["result"] or {}
    return {"variant": row["variant"], "score": res.get("score"), "n": res.get("n"),
            "metadata": res.get("metadata"), "seq": row["seq"], "source": row["source"]}


def summarize(
    ledger: TrialLedger,
    *,
    top: int = 5,
    min_n: int = 0,
    overfit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Counts, chain verification, best/top train variants and test results,
    the holdout-guard status and the deterministic ``content_head``.

    ``overfit`` (a dict of :func:`~sandfish.search.overfit.overfit_report`
    options, ``{}`` for the defaults) adds an ``overfit`` section: holdout
    degradation, walk-forward runs and, with ``score_is_sharpe=True``, the
    Deflated Sharpe Ratio using the ledger's N."""
    test = {r["key"]: r for r in ledger.top(10**9, split=TEST)}
    top_train = []
    for r in ledger.top(top, split=TRAIN, min_n=min_n):
        item = _brief(r)
        t = test.get(r["key"])
        item["test_score"] = t["result"]["score"] if t else None  # type: ignore[index]
        top_train.append(item)
    out: Dict[str, Any] = {
        "counts": ledger.counts(),
        "integrity": ledger.verify(),
        "head": ledger.head,
        "content_head": ledger.content_head(),
        "holdout_guard": ledger.holdout_status(),
        "best_train": _brief(ledger.best(TRAIN, min_n=min_n)),
        "top_train": top_train,
        "min_n": min_n,
        "multiple_testing": MULTIPLE_TESTING_NOTE,
    }
    if overfit is not None:
        out["overfit"] = overfit_report(ledger, **{"min_n": min_n, **dict(overfit)})
    return out
