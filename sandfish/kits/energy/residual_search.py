"""Walk-forward search over the pre-registered residual corrections, with overfit checks.

Variants are scanned once over the whole search range; folds only aggregate the per-day scores.
Selection per fold uses the training range only. The locked holdout is never read: the input
series are cut before ``holdout_start`` before anything is built. Standard library only.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sandfish.kits.energy.residual import (
    ResidualEvaluator,
    default_residual_variants,
    validate_variant,
)
from sandfish.kits.energy.walkforward import (
    DEFAULT_HOLDOUT_START,
    _block_bootstrap_ci,
    make_folds,
)
from sandfish.search.overfit import deflated_sharpe_ratio, probability_of_backtest_overfitting

RC_ALPHA = 0.05
PBO_MAX = 0.5
WIN_SHARE = 2 / 3

DayRow = Tuple[float, float, float]
Scan = Dict[date, DayRow]

DSR_CAVEAT = (
    "Daily gains are autocorrelated, so the DSR is optimistic; the block-bootstrap Reality "
    "Check is the primary multiple-testing evidence."
)


def _key(variant: Mapping[str, Any]) -> str:
    return json.dumps(variant, sort_keys=True)


def _floatify(obj: Any, name: str = "") -> Any:
    if isinstance(obj, dict):
        return {k: _floatify(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_floatify(v, name) for v in obj]
    if name not in ("windows", "lag_days") and isinstance(obj, (int, float)):
        if not isinstance(obj, bool):
            return float(obj)
    return obj


def _canon(variant: Mapping[str, Any]) -> str:
    """Canonical key of a normalised variant: numerics other than windows/lag are floats."""
    return _key(_floatify(dict(variant)))


def _cut(series: Mapping[datetime, float], cut: date) -> Dict[datetime, float]:
    """Keep keys strictly before ``cut`` (midnight UTC); values at/after it are never fetched."""
    limit = datetime(cut.year, cut.month, cut.day, tzinfo=timezone.utc)
    return {k: series[k] for k in series if k < limit}


def _identity_scan(scan: Scan) -> Scan:
    """The identity variant's forecast IS BCDF, so its model MAE is set to the BCDF MAE exactly.

    (Avoids float summation-path noise between the two evaluator computations.)
    """
    return {d: (r[1], r[1], r[2]) for d, r in scan.items()}


def _days(first: date, last: date) -> List[date]:
    n = (last - first).days + 1
    return [first + timedelta(days=i) for i in range(max(n, 0))]


def _in_range(scan: Scan, rng: Tuple[date, date]) -> List[Tuple[date, DayRow]]:
    return [(d, scan[d]) for d in _days(*rng) if d in scan]


def _skill(rows: Sequence[DayRow], col: int) -> Optional[float]:
    """``1 - sum(model)/sum(other)``; None if there are no rows or the denominator is 0."""
    den = sum(r[col] for r in rows)
    if not rows or den == 0:
        return None
    return 1 - sum(r[0] for r in rows) / den


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _binom_tail(wins: int, n: int) -> float:
    """P(X >= wins) for X ~ Binomial(n, 1/2), exact."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def _finite_or_none(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return _finite_or_none(obj)


def _pooled_block(
    rows: List[Tuple[date, DayRow]],
    block_days: int,
    reps: int,
    seed: int,
    fold_skills: List[Optional[float]],
) -> Dict[str, Any]:
    vals = [r for _, r in rows]
    n = len(vals)
    folds_scored = len(fold_skills)

    def gain(diffs: List[float]) -> Dict[str, Any]:
        ci = [
            _finite_or_none(x)
            for x in _block_bootstrap_ci(diffs, block=block_days, reps=reps, seed=seed)
        ]
        return {"mean": sum(diffs) / n, "ci95_block": ci}

    sum_b, sum_d = sum(r[1] for r in vals), sum(r[2] for r in vals)
    wins = sum(1 for s in fold_skills if s is not None and s > 0)
    known = [s for s in fold_skills if s is not None]
    by_year: Dict[str, Any] = {}
    for year in sorted({d.year for d, _ in rows}):
        yr = [r for d, r in rows if d.year == year]
        by_year[str(year)] = {
            "days": len(yr),
            "gain_vs_bcdf_mw": sum(r[1] - r[0] for r in yr) / len(yr),
            "skill_vs_bcdf": _skill(yr, 1),
        }
    return {
        "folds_scored": folds_scored,
        "days": n,
        "mae_mw": {"model": sum(r[0] for r in vals) / n, "bcdf": sum_b / n, "df": sum_d / n},
        "skill_vs_bcdf": _skill(vals, 1),
        "skill_vs_df": _skill(vals, 2),
        "bcdf_skill_vs_df": (1 - sum_b / sum_d) if sum_d != 0 else None,
        "gain_vs_bcdf_mw": gain([r[1] - r[0] for r in vals]),
        "gain_vs_df_mw": gain([r[2] - r[0] for r in vals]),
        "fold_wins": wins,
        "sign_test_p": _binom_tail(wins, folds_scored),
        "fold_skill_min": min(known) if known else None,
        "fold_skill_max": max(known) if known else None,
        "fold_skill_mean": _mean(known),
        "by_year": by_year,
    }


def _reality_check(
    g: List[List[float]],
    block_days: int,
    reps: int,
    seed: int,
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n, nv = len(g), len(variants)
    cols = [[row[v] for row in g] for v in range(nv)]
    means = [sum(c) / n for c in cols]
    root = math.sqrt(n)
    stat = max(root * m for m in means)
    block = min(block_days, n)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(reps):
        starts: List[Tuple[int, int]] = []
        got = 0
        while got < n:
            s = rng.randrange(0, n - block + 1)
            ln = min(block, n - got)
            starts.append((s, ln))
            got += ln
        vstar = -math.inf
        for v in range(nv):
            c = cols[v]
            tot = 0.0
            for s, ln in starts:
                tot += sum(c[s : s + ln])
            vstar = max(vstar, root * (tot / n - means[v]))
        if vstar >= stat:
            exceed += 1
    best = max(range(nv), key=lambda v: (means[v], -v))
    return {
        "p_value": (1 + exceed) / (1 + reps),
        "statistic": stat,
        "reps": reps,
        "block_days": block_days,
        "n_days": n,
        "best_variant": variants[best],
    }


def _moments(xs: Sequence[float]) -> Tuple[float, float]:
    """Population (skew, non-excess kurtosis); (0.0, 3.0) if the stdev is 0."""
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    if var <= 0:
        return 0.0, 3.0
    sd = math.sqrt(var)
    skew = sum(((x - m) / sd) ** 3 for x in xs) / len(xs)
    kurt = sum(((x - m) / sd) ** 4 for x in xs) / len(xs)
    return skew, kurt


def _dsr(g: List[List[float]], n_trials: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    srs: List[float] = []
    best: Optional[Tuple[float, List[float]]] = None
    for v in range(len(g[0])):
        col = [row[v] for row in g]
        sd = statistics.stdev(col)
        if sd == 0:
            continue
        sr = statistics.fmean(col) / sd
        if not math.isfinite(sr):
            continue
        srs.append(sr)
        if best is None or sr > best[0]:
            best = (sr, col)
    if best is None or len(srs) < 2:
        return None, "fewer than 2 variants have a finite per-day Sharpe"
    try:
        skew, kurt = _moments(best[1])
        out = deflated_sharpe_ratio(
            best[0], len(g), n_trials, trial_srs=srs, skew=skew, kurtosis=kurt
        )
    except (ValueError, ArithmeticError) as exc:
        return None, f"deflated_sharpe_ratio failed: {exc}"
    res = dict(out)
    res["caveat"] = DSR_CAVEAT
    return res, None


def _overfit(
    variants: List[Dict[str, Any]],
    scans: List[Scan],
    folds: Sequence[Any],
    min_days: int,
    pbo_splits: int,
    rc_reps: int,
    block_days: int,
    seed: int,
) -> Dict[str, Any]:
    test_days = sorted({d for _, t in folds for d in _days(*t)})
    summaries = []
    for v, scan in zip(variants, scans):
        rows = [scan[d] for d in test_days if d in scan]
        summaries.append(
            {
                "variant": v,
                "days": len(rows),
                "skill_vs_bcdf": _skill(rows, 1),
                "mean_gain_mw": _mean([r[1] - r[0] for r in rows]),
            }
        )
    best_fixed = None
    for i, s in enumerate(summaries):
        sk = s["skill_vs_bcdf"]
        if (
            s["days"] >= min_days
            and sk is not None
            and (best_fixed is None or sk > best_fixed["skill_vs_bcdf"])
        ):
            best_fixed = {"index": i, "variant": s["variant"], "skill_vs_bcdf": sk}
    common = [d for d in test_days if all(d in s for s in scans)]
    out: Dict[str, Any] = {
        "n_trials": len(variants),
        "variant_summaries": summaries,
        "best_fixed": best_fixed,
        "common_days": len(common),
        "reality_check": None,
        "pbo": None,
        "dsr": None,
    }
    need = max(30, pbo_splits)
    if len(common) < need:
        out["reason"] = f"only {len(common)} common days; need at least {need}"
        return out
    g = [[s[d][1] - s[d][0] for s in scans] for d in common]
    out["reality_check"] = _reality_check(g, block_days, rc_reps, seed, variants)
    keep = [j for j in range(len(variants)) if len({row[j] for row in g}) > 1]
    dropped = len(variants) - len(keep)
    reasons: List[str] = []
    if len(keep) < 2:
        reasons.append("pbo: fewer than 2 non-constant columns")
    else:
        try:
            pbo = probability_of_backtest_overfitting(
                [[row[j] for j in keep] for row in g], n_splits=pbo_splits, seed=seed
            )
            out["pbo"] = dict(pbo, columns_used=len(keep), columns_dropped_constant=dropped)
        except (ValueError, ArithmeticError) as exc:
            reasons.append(f"pbo: {exc}")
    try:
        dsr, why = _dsr(g, len(variants))
    except (ValueError, ArithmeticError) as exc:
        dsr, why = None, f"computation failed: {exc}"
    out["dsr"] = dsr
    if why:
        reasons.append(f"dsr: {why}")
    if reasons:
        out["reason"] = "; ".join(reasons)
    return out


def _verdict(pooled: Optional[Dict[str, Any]], overfit: Dict[str, Any]) -> Dict[str, Any]:
    ci_ok = rc_ok = pbo_ok = wins_ok = False
    gain_mean = None
    if pooled is not None:
        lo = pooled["gain_vs_bcdf_mw"]["ci95_block"][0]
        ci_ok = lo is not None and lo > 0
        gain_mean = pooled["gain_vs_bcdf_mw"]["mean"]
        n = pooled["folds_scored"]
        wins_ok = n >= 1 and pooled["fold_wins"] >= WIN_SHARE * n
    if overfit["reality_check"] is not None:
        rc_ok = overfit["reality_check"]["p_value"] < RC_ALPHA
    if overfit["pbo"] is not None:
        pbo_ok = overfit["pbo"]["pbo"] < PBO_MAX
    flags = [ci_ok, rc_ok, pbo_ok, wins_ok]
    if all(flags):
        label = "ready_for_holdout"
    elif gain_mean is not None and gain_mean > 0 and sum(flags) >= 2:
        label = "promising"
    else:
        label = "no_evidence"
    names = [
        "ci_lower_gain_positive",
        "reality_check_significant",
        "pbo_acceptable",
        "fold_wins_enough",
    ]
    failed = [n for n, f in zip(names, flags) if not f]
    explanation = "all four checks passed" if not failed else "not passed: " + ", ".join(failed)
    return {
        "ci_lower_gain_positive": ci_ok,
        "reality_check_significant": rc_ok,
        "pbo_acceptable": pbo_ok,
        "fold_wins_enough": wins_ok,
        "label": label,
        "ready_for_holdout": label == "ready_for_holdout",
        "explanation": explanation,
    }


def _check_int(name: str, val: Any, lo: int) -> None:
    if isinstance(val, bool) or not isinstance(val, int) or val < lo:
        raise ValueError(f"{name} must be an int >= {lo}")


def run_residual_search(
    demand: Mapping[datetime, float],
    operator_forecast: Mapping[datetime, float],
    weather: Optional[Mapping[datetime, float]],
    *,
    search_start: date,
    first_test: date,
    search_end: date,
    holdout_start: date = DEFAULT_HOLDOUT_START,
    embargo_days: int = 14,
    min_days: int = 45,
    variants: Optional[List[Dict[str, Any]]] = None,
    bootstrap_reps: int = 2000,
    block_days: int = 7,
    rc_reps: int = 2000,
    pbo_splits: int = 12,
    seed: int = 20261118,
) -> Dict[str, Any]:
    """Walk-forward search over residual corrections; returns a JSON-serialisable report."""
    folds = make_folds(
        search_start, first_test, search_end, holdout_start=holdout_start, embargo_days=embargo_days
    )
    _check_int("min_days", min_days, 1)
    _check_int("bootstrap_reps", bootstrap_reps, 100)
    _check_int("rc_reps", rc_reps, 100)
    _check_int("block_days", block_days, 1)
    _check_int("pbo_splits", pbo_splits, 2)
    if pbo_splits % 2 or pbo_splits > 20:
        raise ValueError("pbo_splits must be an even int in 2..20")
    raw = default_residual_variants() if variants is None else list(variants)
    if not raw:
        raise ValueError("variants must not be empty")
    norm = [validate_variant(v) for v in raw]
    keys = [_canon(v) for v in norm]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate variants")
    identity = validate_variant({"windows": [7]})
    id_key = _canon(identity)

    dem = _cut(demand, holdout_start)
    opf = _cut(operator_forecast, holdout_start)
    wx = None if weather is None else _cut(weather, holdout_start)
    ev = ResidualEvaluator(dem, opf, wx)
    scans: List[Scan] = []
    for v, k in zip(raw, keys):
        sc = ev.scan(v, search_start, search_end)[0]
        scans.append(_identity_scan(sc) if k == id_key else sc)
    if id_key in keys:
        id_scan = scans[keys.index(id_key)]
    else:
        id_scan = _identity_scan(ev.scan({"windows": [7]}, search_start, search_end)[0])
    n_vars = len(norm)

    fold_out: List[Dict[str, Any]] = []
    scored_rows: List[Tuple[date, DayRow]] = []
    scored_skills: List[Optional[float]] = []
    for train, test in folds:
        train_rows = [_in_range(s, train) for s in scans]
        test_rows = [_in_range(s, test) for s in scans]
        best_i: Optional[int] = None
        best_sk = 0.0
        eligible = 0
        for i in range(n_vars):
            if len(train_rows[i]) < min_days:
                continue
            eligible += 1
            sk = _skill([r for _, r in train_rows[i]], 1)
            if sk is not None and sk > best_sk:
                best_i, best_sk = i, sk
        if best_i is None:
            fallback = True
            sel = identity
            tr = _in_range(id_scan, train)
            te = _in_range(id_scan, test)
            tr_skill = 0.0
        else:
            fallback = False
            sel, tr, te, tr_skill = norm[best_i], train_rows[best_i], test_rows[best_i], best_sk
        te_vals = [r for _, r in te]
        scored = len(te) >= min_days
        sk_b = _skill(te_vals, 1) if scored else None
        if scored and fallback:
            sk_b = 0.0  # plain BCDF vs itself, also when its error is zero
        others: List[float] = []
        hind = None
        for i in range(n_vars):
            if len(test_rows[i]) < min_days:
                continue
            sk = _skill([r for _, r in test_rows[i]], 1)
            if sk is None:
                continue
            others.append(sk)
            if hind is None or sk > hind["skill_vs_bcdf"]:
                hind = {"variant": norm[i], "skill_vs_bcdf": sk}
        rank = None
        if scored and sk_b is not None and others:
            rank = sum(1 for o in others if o <= sk_b) / len(others)
        mae = None
        if scored:
            mae = {
                "model": sum(r[0] for r in te_vals) / len(te_vals),
                "bcdf": sum(r[1] for r in te_vals) / len(te_vals),
                "df": sum(r[2] for r in te_vals) / len(te_vals),
            }
        fold_out.append(
            {
                "train": [train[0].isoformat(), train[1].isoformat()],
                "test": [test[0].isoformat(), test[1].isoformat()],
                "selected": sel,
                "fallback_to_bcdf": fallback,
                "train_skill_vs_bcdf": tr_skill,
                "train_days": len(tr),
                "variants_eligible": eligible,
                "test_days": len(te),
                "test_scored": scored,
                "test_mae_mw": mae,
                "test_skill_vs_bcdf": sk_b,
                "test_skill_vs_df": _skill(te_vals, 2) if scored else None,
                "hindsight_best_test": hind,
                "selected_test_rank_pct": rank,
            }
        )
        if scored:
            scored_rows.extend(te)
            scored_skills.append(sk_b)

    pooled = None
    if scored_skills:
        scored_rows.sort(key=lambda x: x[0])
        pooled = _pooled_block(scored_rows, block_days, bootstrap_reps, seed, scored_skills)
        counts: Dict[str, int] = {}
        for f in fold_out:
            if f["test_scored"]:
                k = _key(f["selected"])
                counts[k] = counts.get(k, 0) + 1
        pooled["selected_counts"] = counts
    overfit = _overfit(norm, scans, folds, min_days, pbo_splits, rc_reps, block_days, seed)
    result = {
        "search": [search_start.isoformat(), search_end.isoformat()],
        "holdout_start": holdout_start.isoformat(),
        "holdout_touched": False,
        "embargo_days": embargo_days,
        "n_trials": n_vars,
        "n_folds": len(folds),
        "folds": fold_out,
        "pooled": pooled,
        "overfit": overfit,
        "verdict": _verdict(pooled, overfit),
    }
    return _sanitize(result)
