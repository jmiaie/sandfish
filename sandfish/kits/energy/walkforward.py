"""Walk-forward search for the kit 1 energy evaluator (stdlib only, holdout locked).

Quarterly test blocks, an embargo between the end of each training window and its test block,
an expanding training window, and selection on the training window only. The selection score is
the skill against the stronger baseline (the bias-corrected operator forecast), so a variant
must beat that to look good. The locked holdout (``holdout_start`` onward) is never read: input
series are cut before it, and any fold that would touch it raises.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sandfish.kits.energy.evaluator import EnergyLoadEvaluator
from sandfish.kits.energy.gate0 import iter_series, parse_eia_hour

LAG_SETS: Tuple[Tuple[int, ...], ...] = ((48,), (72,), (168,), (48, 168), (72, 168), (48, 72, 168))
TEMP_COEFS: Tuple[int, ...] = tuple(range(-400, 901, 100))
# Prior weights for the three city points (not fitted): the LA basin is the largest load.
POINT_WEIGHTS: Dict[str, float] = {"la": 0.5, "sf": 0.3, "sd": 0.2}
DEFAULT_HOLDOUT_START = date(2025, 1, 1)


class HoldoutError(ValueError):
    """A fold or series would touch the locked holdout."""


Fold = Tuple[Tuple[date, date], Tuple[date, date]]


def quarter_blocks(first: date, last: date) -> List[Tuple[date, date]]:
    """Calendar-quarter ``(start, end)`` blocks fully inside ``first..last``."""
    out: List[Tuple[date, date]] = []
    year, q = first.year, (first.month - 1) // 3
    while True:
        start = date(year, 3 * q + 1, 1)
        nxt = date(year + (q + 1) // 4, 3 * ((q + 1) % 4) + 1, 1)
        end = nxt - timedelta(days=1)
        if end > last:
            break
        if start >= first:
            out.append((start, end))
        year, q = nxt.year, (nxt.month - 1) // 3
    return out


def make_folds(
    search_start: date,
    first_test: date,
    search_end: date,
    *,
    holdout_start: date = DEFAULT_HOLDOUT_START,
    embargo_days: int = 14,
    min_train_days: int = 90,
) -> List[Fold]:
    """Expanding-window folds: train ``search_start..test_start-embargo-1``, test = a quarter."""
    if search_end >= holdout_start:
        raise HoldoutError(f"search_end {search_end} must be before holdout_start {holdout_start}")
    if isinstance(embargo_days, bool) or not isinstance(embargo_days, int) or embargo_days < 0:
        raise ValueError("embargo_days must be an int >= 0")
    folds: List[Fold] = []
    for start, end in quarter_blocks(first_test, search_end):
        train_end = start - timedelta(days=embargo_days + 1)
        if (train_end - search_start).days + 1 < min_train_days:
            continue
        folds.append(((search_start, train_end), (start, end)))
    if not folds:
        raise ValueError("no folds fit between search_start and search_end")
    return folds


def default_variants() -> List[Dict[str, Any]]:
    """The search space: seasonal naive, lag means, and lag means with a temperature term."""
    out: List[Dict[str, Any]] = [{"model": "seasonal_naive"}]
    out += [{"model": "lag_mean", "lags": list(lags)} for lags in LAG_SETS]
    out += [
        {"model": "lag_mean_temp", "lags": list(lags), "temp_coef": float(c)}
        for lags in LAG_SETS
        for c in TEMP_COEFS
    ]
    return out


def _before(series: Mapping[datetime, float], cut: date) -> Dict[datetime, float]:
    limit = datetime(cut.year, cut.month, cut.day, tzinfo=timezone.utc)
    return {t: v for t, v in series.items() if t < limit}


def _summ(result: Any) -> Dict[str, Any]:
    m = result.metadata
    return {
        "scorable": result.scorable,
        "n": result.n,
        "score_vs_operator": result.score if result.scorable else None,
        "skill_vs_bias_corrected_df": m.get("skill_vs_bias_corrected_df"),
        "skill_vs_seasonal_naive": m.get("skill_vs_seasonal_naive"),
        "bias_corrected_df_vs_operator": m.get("skill_bias_corrected_df_vs_operator"),
        "days_without_weather": m.get("days_without_weather"),
        "days_without_bias_baseline": m.get("days_without_bias_baseline"),
    }


def _day_errors(ev: EnergyLoadEvaluator, variant: Mapping[str, Any], days: Iterable[str]):
    """Per-day mean absolute errors (model, operator, seasonal naive, BCDF) in MW.

    Uses the evaluator's own day scorer so the numbers are exactly what it scored.
    """
    _, lags = ev._lags(variant)  # noqa: SLF001
    coef = ev._temp_coef(variant) if variant.get("model") == "lag_mean_temp" else None  # noqa: SLF001
    rows = []
    for d in days:
        day = date.fromisoformat(d)
        rows.append(ev._score_day(ev._day_hours(day), lags, coef))  # noqa: SLF001
    return rows


def _block_bootstrap_ci(
    diffs: Sequence[float], *, block: int = 7, reps: int = 2000, seed: int = 20261118
) -> Tuple[float, float]:
    """95% CI of the mean of ``diffs`` with a moving-block bootstrap (fixed seed)."""
    n = len(diffs)
    if n == 0:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    block = max(1, min(block, n))
    means = []
    for _ in range(reps):
        sample: List[float] = []
        while len(sample) < n:
            s = rng.randrange(0, n - block + 1)
            sample.extend(diffs[s : s + block])
        means.append(sum(sample[:n]) / n)
    means.sort()
    return (means[int(0.025 * reps)], means[int(0.975 * reps) - 1])


def run_search(
    demand: Mapping[datetime, float],
    operator_forecast: Mapping[datetime, float],
    weather: Mapping[datetime, float],
    *,
    search_start: date,
    first_test: date,
    search_end: date,
    holdout_start: date = DEFAULT_HOLDOUT_START,
    embargo_days: int = 14,
    min_days: int = 45,
    variants: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the walk-forward search and return a JSON-able report."""
    folds = make_folds(
        search_start, first_test, search_end, holdout_start=holdout_start, embargo_days=embargo_days
    )
    variants = variants if variants is not None else default_variants()
    dem = _before(demand, holdout_start)
    df = _before(operator_forecast, holdout_start)
    wx = _before(weather, holdout_start)
    report_folds: List[Dict[str, Any]] = []
    pooled_selected: List[Tuple[float, float, float, float]] = []
    for train, test in folds:
        ev = EnergyLoadEvaluator(
            dem, df, train, test, min_days=min_days, weather=wx, bias_baseline=True
        )
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        scorable_train = 0
        for v in variants:
            r = ev.evaluate(v, "train")
            sk = r.metadata.get("skill_vs_bias_corrected_df") if r.scorable else None
            if sk is None:
                continue
            scorable_train += 1
            if best is None or sk > best[0]:
                best = (sk, v)
        if best is None:
            report_folds.append({"train": [str(d) for d in train], "test": [str(d) for d in test],
                                 "selected": None})  # fmt: skip
            continue
        sel = best[1]
        train_res = ev.evaluate(sel, "train")
        test_res = ev.evaluate(sel, "test")
        entry: Dict[str, Any] = {
            "train": [str(d) for d in train],
            "test": [str(d) for d in test],
            "variants_scorable_on_train": scorable_train,
            "selected": sel,
            "train_result": _summ(train_res),
            "test_result": _summ(test_res),
        }
        if test_res.scorable:
            rows = _day_errors(ev, sel, test_res.metadata["days"])
            entry["test_mae_mw"] = {
                "model": sum(r[0] for r in rows) / len(rows),
                "operator": sum(r[1] for r in rows) / len(rows),
                "seasonal_naive": sum(r[2] for r in rows) / len(rows),
                "bias_corrected_df": sum(r[3] for r in rows) / len(rows),
            }
            pooled_selected += [(r[0], r[1], r[2], r[3]) for r in rows]
        # Diagnostic only (hindsight, not a valid result): best single variant on this test block.
        hind = None
        for v in variants:
            r = ev.evaluate(v, "test")
            sk = r.metadata.get("skill_vs_bias_corrected_df") if r.scorable else None
            if sk is not None and (hind is None or sk > hind[0]):
                hind = (sk, v)
        entry["hindsight_best_on_test"] = (
            {"variant": hind[1], "skill_vs_bias_corrected_df": hind[0]} if hind else None
        )
        report_folds.append(entry)
    out: Dict[str, Any] = {
        "search": [str(search_start), str(search_end)],
        "holdout_start": str(holdout_start),
        "holdout_touched": False,
        "embargo_days": embargo_days,
        "trials_per_fold": len(variants),
        "folds": report_folds,
    }
    if pooled_selected:
        n = len(pooled_selected)
        m = sum(r[0] for r in pooled_selected) / n
        d = sum(r[1] for r in pooled_selected) / n
        s = sum(r[2] for r in pooled_selected) / n
        b = sum(r[3] for r in pooled_selected) / n
        ci_b = _block_bootstrap_ci([r[3] - r[0] for r in pooled_selected])
        ci_d = _block_bootstrap_ci([r[1] - r[0] for r in pooled_selected])
        out["pooled_selected"] = {
            "days": n,
            "mae_mw": {"model": m, "operator": d, "seasonal_naive": s, "bias_corrected_df": b},
            "skill_vs_operator": 1 - m / d,
            "skill_vs_bias_corrected_df": 1 - m / b,
            "bias_corrected_df_vs_operator": 1 - b / d,
            "mean_daily_mae_gain_vs_bcdf_mw": {"mean": b - m, "ci95_block7": list(ci_b)},
            "mean_daily_mae_gain_vs_operator_mw": {"mean": d - m, "ci95_block7": list(ci_d)},
            "folds_beating_bcdf": sum(
                1
                for f in report_folds
                if f.get("selected") and (f["test_result"]["skill_vs_bias_corrected_df"] or -1) > 0
            ),
            "folds": len(report_folds),
        }
    return out


def load_series(path: str, series_id: str) -> Dict[datetime, float]:
    """Load one EIA-930 series from an EBA.txt-style NDJSON file as ``{UTC hour: value}``."""
    out: Dict[datetime, float] = {}
    with open(path, encoding="utf-8") as fh:
        for obj in iter_series(fh, {series_id}):
            for pair in obj.get("data", []):
                try:
                    hour = parse_eia_hour(pair[0])
                    value = pair[1]
                    if value is not None:
                        out[hour] = float(value)
                except (ValueError, IndexError, TypeError):
                    continue
    return out
