"""Overfitting checks for parameter searches.

A search that tries N variants and keeps the best one has a selection-bias
problem: the best of N noisy scores is optimistic even when nothing works.
These helpers put numbers on that. They are plain statistics over numbers
you give them (or over a :class:`~sandfish.search.ledger.TrialLedger`); none
of them evaluates or backtests anything itself.

* :func:`probabilistic_sharpe_ratio`, :func:`expected_max_sharpe`,
  :func:`deflated_sharpe_ratio` and :func:`ledger_deflated_sharpe`: the
  Deflated Sharpe Ratio, which deflates the best Sharpe ratio by the number
  of trials N (taken from the ledger) and by non-normal returns.
* :func:`probability_of_backtest_overfitting`: Combinatorially Symmetric
  Cross-Validation (CSCV) over a matrix of per-period returns, one column per
  tried variant.
* :func:`make_walk_forward_folds` and :func:`walk_forward`: rolling or
  expanding window validation, logged in the ledger.
* :func:`holdout_report`: how train scores degrade on the held-out split.
* :func:`overfit_report`: all of the above that apply, for ``summarize``
  and the ``sf_search_overfit`` MCP tool.

The holdout lock itself lives on the ledger (``lock_holdout`` /
``holdout_access``) and is used by ``validate_holdout``.

References
----------
[1] D. H. Bailey and M. López de Prado (2014), "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
    Journal of Portfolio Management 40(5), 94-107.
[2] D. H. Bailey and M. López de Prado (2012), "The Sharpe Ratio Efficient
    Frontier", Journal of Risk 15(2), 3-44 (the Probabilistic Sharpe Ratio).
[3] D. H. Bailey, J. M. Borwein, M. López de Prado and Q. J. Zhu (2017),
    "The Probability of Backtest Overfitting", Journal of Computational
    Finance 20(4), 39-69.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .evaluator import TEST, TRAIN, EvaluationResult
from .ledger import TrialLedger, variant_key

EULER_GAMMA = 0.5772156649015328606065120900824024  # Euler-Mascheroni constant
_N = NormalDist()

__all__ = [
    "EULER_GAMMA", "Fold", "expected_max_sharpe", "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio", "ledger_deflated_sharpe", "sharpe_stats",
    "probability_of_backtest_overfitting", "make_walk_forward_folds", "walk_forward",
    "holdout_report", "overfit_report", "spearman",
]


# ----- Deflated Sharpe Ratio -----


def sharpe_stats(returns: Sequence[float]) -> Dict[str, float]:
    """Per-period Sharpe ratio (mean / sample std, ddof=1), the number of
    observations, skewness (gamma_3) and *non-excess* kurtosis (gamma_4;
    a normal has 3), from moments about the mean, as the formulas in [1] use."""
    x = [float(v) for v in returns]
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 observations")
    if not all(math.isfinite(v) for v in x):
        raise ValueError("returns must be finite")
    mean = math.fsum(x) / n
    m2 = math.fsum((v - mean) ** 2 for v in x) / n
    if m2 <= 0:
        raise ValueError("returns have zero variance")
    m3 = math.fsum((v - mean) ** 3 for v in x) / n
    m4 = math.fsum((v - mean) ** 4 for v in x) / n
    sd = math.sqrt(m2 * n / (n - 1))
    return {"sharpe": mean / sd, "n_obs": n, "skew": m3 / m2 ** 1.5, "kurtosis": m4 / m2 ** 2}


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` independent trials whose
    true Sharpe is zero (the "SR_0" benchmark of [1], eq. 6):

        SR_0 = sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))

    where V is the variance of the trials' Sharpe ratios, g the Euler-
    Mascheroni constant and Z^-1 the inverse standard normal CDF. With a
    single trial there is no selection, so the benchmark is 0.
    """
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1")
    if var_sr < 0 or not math.isfinite(var_sr):
        raise ValueError("var_sr must be finite and >= 0")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    return math.sqrt(var_sr) * (
        (1.0 - EULER_GAMMA) * _N.inv_cdf(1.0 - 1.0 / n)
        + EULER_GAMMA * _N.inv_cdf(1.0 - 1.0 / (n * math.e))
    )


def probabilistic_sharpe_ratio(
    sr: float, n_obs: int, benchmark: float = 0.0, *, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Probabilistic Sharpe Ratio ([2], used as eq. 2 in [1]): the probability
    that the true Sharpe ratio exceeds ``benchmark``, given the estimated
    per-period ``sr`` over ``n_obs`` observations and the returns' skewness and
    (non-excess) kurtosis:

        PSR = Z( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )

    All Sharpe ratios are per period (not annualised).
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if not denom > 0:
        raise ValueError("non-normality adjustment is not positive for these skew/kurtosis/sr")
    return _N.cdf((sr - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom))


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: int,
    *,
    var_sr: Optional[float] = None,
    trial_srs: Optional[Sequence[float]] = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> Dict[str, Any]:
    """Deflated Sharpe Ratio ([1], eq. 2 with SR* = SR_0 from eq. 6): the
    probability that the selected strategy's true Sharpe ratio is positive
    after accounting for ``n_trials`` trials and non-normal returns.

    ``sr`` is the best strategy's per-period Sharpe ratio over ``n_obs``
    observations. Give the variance of the trials' Sharpe ratios as ``var_sr``
    or the trials' Sharpe ratios as ``trial_srs`` (sample variance, ddof=1).
    ``n_trials`` is N: every distinct variant tried, including ones that were
    not scorable or failed (use the ledger's count, not the finalist count).

    A DSR near 1 means the result survives the deflation; a common bar is
    0.95. It is evidence, not proof: it assumes the trials are roughly
    independent and N is honest.
    """
    if (var_sr is None) == (trial_srs is None):
        raise ValueError("pass exactly one of var_sr or trial_srs")
    if trial_srs is not None:
        vals = [float(v) for v in trial_srs]
        if len(vals) < 2:
            raise ValueError("need at least 2 trial Sharpe ratios to estimate their variance")
        mean = math.fsum(vals) / len(vals)
        var_sr = math.fsum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    assert var_sr is not None
    sr0 = expected_max_sharpe(n_trials, var_sr)
    dsr = probabilistic_sharpe_ratio(sr, n_obs, sr0, skew=skew, kurtosis=kurtosis)
    return {
        "dsr": dsr, "sharpe": float(sr), "sr0": sr0, "var_sr": float(var_sr),
        "n_trials": int(n_trials), "n_obs": int(n_obs), "skew": float(skew),
        "kurtosis": float(kurtosis), "psr_vs_zero": probabilistic_sharpe_ratio(
            sr, n_obs, 0.0, skew=skew, kurtosis=kurtosis),
    }


def ledger_deflated_sharpe(
    ledger: TrialLedger,
    *,
    score_is_sharpe: bool,
    periods_per_year: float = 1.0,
    n_obs: Optional[int] = None,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
    min_n: int = 0,
    split: str = TRAIN,
    n_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """Deflated Sharpe Ratio of the ledger's best variant on ``split``.

    SandFish does not know what an evaluator's score means, so you must
    state that scores are Sharpe ratios (``score_is_sharpe=True``); if they are
    annualised, give ``periods_per_year`` (e.g. 252) and they are converted to
    per-period ratios. N is the number of unique variants tried on the split,
    not-scorable and failed ones included (override with ``n_trials`` when
    you also tried variants outside this ledger). The variance of Sharpe
    ratios is taken across the scorable variants. ``n_obs`` defaults to the
    best result's ``n``; ``skew`` and ``kurtosis`` default to the best
    result's metadata keys of those names, else a normal (0 and 3).
    """
    if not score_is_sharpe:
        raise ValueError(
            "the Deflated Sharpe Ratio needs scores that are Sharpe ratios: pass score_is_sharpe=True")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    scored = ledger.top(10**9, split=split, min_n=min_n)
    if len(scored) < 2:
        return {"available": False, "reason": "need at least 2 scorable variants", "n_scorable": len(scored)}
    best = scored[0]
    res, meta = best["result"], best["result"].get("metadata") or {}
    t = n_obs if n_obs is not None else res.get("n")
    if t is None:
        return {"available": False, "reason": "no observation count: pass n_obs", "n_scorable": len(scored)}
    scale = math.sqrt(periods_per_year)
    srs = [r["result"]["score"] / scale for r in scored]
    n = n_trials if n_trials is not None else ledger.counts()["by_split"][split]["unique_variants"]
    g3 = skew if skew is not None else _num(meta.get("skew"), 0.0)
    g4 = kurtosis if kurtosis is not None else _num(meta.get("kurtosis"), 3.0)
    out = deflated_sharpe_ratio(srs[0], int(t), int(n), trial_srs=srs, skew=g3, kurtosis=g4)
    out.update(available=True, variant=best["variant"], n_scorable=len(scored), split=split,
               periods_per_year=periods_per_year, significant_95=out["dsr"] >= 0.95)
    return out


def _num(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default


# ----- Probability of Backtest Overfitting (CSCV) -----


def _sharpe_from_sums(s: float, ss: float, n: int) -> float:
    mean = s / n
    var = max((ss - n * mean * mean) / (n - 1), 0.0)
    sd = math.sqrt(var)
    return mean / sd if sd > 1e-300 else 0.0


def probability_of_backtest_overfitting(
    performance: Sequence[Sequence[float]],
    n_splits: int = 16,
    *,
    max_combinations: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Probability of Backtest Overfitting by CSCV ([3], section 2).

    ``performance`` is a T x N matrix: T per-period returns (or P&L) for each
    of N tried variants, aligned in time. It is split into ``n_splits`` (S,
    even) contiguous blocks; for each of the C(S, S/2) ways to pick half the
    blocks as in-sample (IS), the variant with the best IS Sharpe ratio is
    ranked among all N on the remaining out-of-sample (OOS) half. With w =
    rank / (N + 1) (rank 1 = worst; ties share an average rank) and logit
    lambda = ln(w / (1 - w)), PBO is the share of combinations with lambda <= 0,
    i.e. where the IS winner landed at or below the OOS median.

    The first ``T mod S`` rows are dropped so the blocks are equal. With
    ``max_combinations`` set and fewer than C(S, S/2), that many combinations
    are drawn without replacement using ``seed``. Returns ``pbo``, the logit
    mean/std, ``prob_oos_loss`` (share where the IS winner's OOS Sharpe < 0),
    and ``degradation_slope`` (OLS slope of the winner's OOS on IS Sharpe;
    negative means the IS edge reverses).
    """
    rows = [[float(v) for v in r] for r in performance]
    if not rows:
        raise ValueError("performance is empty")
    n_strat = len(rows[0])
    if n_strat < 2:
        raise ValueError("need at least 2 variants (columns)")
    if any(len(r) != n_strat for r in rows):
        raise ValueError("performance must be a rectangular T x N matrix")
    if not all(math.isfinite(v) for r in rows for v in r):
        raise ValueError("performance must be finite")
    s = int(n_splits)
    if s < 2 or s % 2 or s > 20:
        raise ValueError("n_splits must be an even number between 2 and 20")
    t_total = len(rows)
    if t_total < 2 * s:
        raise ValueError(f"need at least {2 * s} rows for {s} splits (2 per block)")
    m = t_total // s
    rows = rows[t_total - m * s:]
    # Per block and variant: sum and sum of squares.
    bsum = [[math.fsum(rows[b * m + i][j] for i in range(m)) for j in range(n_strat)] for b in range(s)]
    bsq = [[math.fsum(rows[b * m + i][j] ** 2 for i in range(m)) for j in range(n_strat)] for b in range(s)]
    combos = list(itertools.combinations(range(s), s // 2))
    total = len(combos)
    if max_combinations is not None and 0 < max_combinations < total:
        combos = random.Random(seed).sample(combos, int(max_combinations))
    half = (s // 2) * m
    logits: List[float] = []
    is_best: List[float] = []
    oos_best: List[float] = []
    for comb in combos:
        in_blocks = set(comb)
        out_blocks = [b for b in range(s) if b not in in_blocks]
        is_sr, oos_sr = [], []
        for j in range(n_strat):
            a = math.fsum(bsum[b][j] for b in comb)
            aq = math.fsum(bsq[b][j] for b in comb)
            c = math.fsum(bsum[b][j] for b in out_blocks)
            cq = math.fsum(bsq[b][j] for b in out_blocks)
            is_sr.append(_sharpe_from_sums(a, aq, half))
            oos_sr.append(_sharpe_from_sums(c, cq, half))
        star = max(range(n_strat), key=lambda j: (is_sr[j], -j))
        target = oos_sr[star]
        less = sum(1 for v in oos_sr if v < target)
        equal = sum(1 for v in oos_sr if v == target)
        rank = less + (equal + 1) / 2.0
        w = rank / (n_strat + 1)
        logits.append(math.log(w / (1.0 - w)))
        is_best.append(is_sr[star])
        oos_best.append(target)
    k = len(logits)
    mean_l = math.fsum(logits) / k
    std_l = math.sqrt(math.fsum((v - mean_l) ** 2 for v in logits) / (k - 1)) if k > 1 else 0.0
    mx, my = math.fsum(is_best) / k, math.fsum(oos_best) / k
    sxx = math.fsum((v - mx) ** 2 for v in is_best)
    slope = (math.fsum((a - mx) * (b - my) for a, b in zip(is_best, oos_best)) / sxx) if sxx > 0 else None
    return {
        "pbo": sum(1 for v in logits if v <= 0) / k,
        "logit_mean": mean_l, "logit_std": std_l,
        "prob_oos_loss": sum(1 for v in oos_best if v < 0) / k,
        "degradation_slope": slope,
        "n_combinations": k, "n_combinations_total": total,
        "n_splits": s, "n_strategies": n_strat, "rows_used": m * s,
    }


# ----- walk-forward validation -----


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold over a train-split series of ``n_obs`` points.
    ``train`` and ``test`` are half-open ``(start, stop)`` index windows."""

    index: int
    train: Tuple[int, int]
    test: Tuple[int, int]

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "train": list(self.train), "test": list(self.test)}


def make_walk_forward_folds(
    n_obs: int,
    train_size: int,
    test_size: int,
    *,
    step: Optional[int] = None,
    anchored: bool = False,
    embargo: int = 0,
) -> List[Fold]:
    """Rolling (or, with ``anchored``, expanding) walk-forward windows.

    Fold i trains on ``[a, a + train_size)`` (``a`` = 0 if anchored, else
    ``i * step``; an anchored train window grows by ``step`` each fold), skips
    ``embargo`` points (to keep overlapping labels from leaking), then tests
    on the next ``test_size`` points. ``step`` defaults to ``test_size`` so
    test windows tile without overlap. ``n_obs`` is the length of the
    *train-split* data: folds never reach beyond it, and the held-out split is
    never involved. Raises if not even one fold fits.
    """
    step = test_size if step is None else step
    if min(train_size, test_size, step) < 1 or embargo < 0:
        raise ValueError("train_size, test_size and step must be >= 1 and embargo >= 0")
    folds: List[Fold] = []
    i = 0
    while True:
        start = 0 if anchored else i * step
        train_end = start + train_size + (i * step if anchored else 0)
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n_obs:
            break
        folds.append(Fold(i, (start, train_end), (test_start, test_end)))
        i += 1
    if not folds:
        raise ValueError("no walk-forward fold fits in n_obs; shrink train_size/test_size/embargo")
    return folds


def walk_forward(
    evaluator: Any,
    variant: Mapping[str, Any],
    folds: Sequence[Fold],
    *,
    ledger: Optional[TrialLedger] = None,
    source: str = "walk_forward",
) -> Dict[str, Any]:
    """Score one variant on each fold and summarise how stable it is.

    ``evaluator`` needs ``evaluate_fold(variant, fold)`` (fit on
    ``fold.train``, score ``fold.test``; stay inside the train-split data) or
    be a plain callable ``fn(variant, fold)``. Results may be
    :class:`EvaluationResult` values or mappings; a not-scorable fold or one
    that raises is counted and left out of the statistics. With a ``ledger``,
    the run is logged there as a ``walk_forward`` event (not a trial: it
    re-scores a variant the search already counted).
    """
    fn: Optional[Callable[[Mapping[str, Any], Fold], Any]] = getattr(evaluator, "evaluate_fold", None)
    if fn is None and callable(evaluator) and not callable(getattr(evaluator, "evaluate", None)):
        fn = evaluator
    if fn is None:
        raise TypeError("evaluator must have evaluate_fold(variant, fold) or be a callable(variant, fold)")
    if not folds:
        raise ValueError("no folds")
    per_fold: List[Dict[str, Any]] = []
    for fold in folds:
        entry: Dict[str, Any] = {**fold.as_dict(), "score": None, "n": None, "error": None}
        try:
            res = EvaluationResult.coerce(fn(dict(variant), fold))
            entry.update(score=res.score, n=res.n)
        except Exception as exc:  # noqa: BLE001 - a failed fold is data, not a crash
            entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
        per_fold.append(entry)
    scores = [e["score"] for e in per_fold if e["score"] is not None]
    k = len(scores)
    mean = math.fsum(scores) / k if k else None
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in scores) / (k - 1)) if k > 1 else None
    out: Dict[str, Any] = {
        "variant": dict(variant), "n_folds": len(per_fold), "n_scorable": k,
        "n_errors": sum(1 for e in per_fold if e["error"]),
        "mean": mean, "std": std,
        "min": min(scores) if k else None, "max": max(scores) if k else None,
        "positive_share": (sum(1 for v in scores if v > 0) / k) if k else None,
        "stability": (mean / std) if (std and mean is not None) else None,
        "folds": per_fold,
    }
    if ledger is not None:
        ledger.record_event("walk_forward", {
            "source": source, "key": variant_key(variant), "variant": dict(variant),
            "n_folds": out["n_folds"], "n_scorable": k, "mean": mean, "std": std,
            "scores": [e["score"] for e in per_fold],
        })
    return out


# ----- holdout degradation -----


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for idx in order[i:j + 1]:
            ranks[idx] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation (average ranks for ties); None if undefined."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    sxx = math.fsum((a - mx) ** 2 for a in rx)
    syy = math.fsum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def holdout_report(ledger: TrialLedger, *, min_n: int = 0) -> Dict[str, Any]:
    """Train versus held-out scores for the variants scored on both.

    ``mean_degradation`` is mean(train - test): positive means the held-out
    split scored lower. ``rank_correlation`` is Spearman's rho between train
    and test scores across those variants (None with fewer than 3 pairs).
    Few finalists make these noisy; treat them as a smoke test.
    """
    train = {r["key"]: r for r in ledger.top(10**9, split=TRAIN, min_n=min_n)}
    test = {r["key"]: r for r in ledger.top(10**9, split=TEST)}
    keys = [k for k in train if k in test]
    if not keys:
        return {"n_pairs": 0, "mean_train": None, "mean_test": None, "mean_degradation": None,
                "rank_correlation": None, "test_below_train_share": None, "pairs": []}
    tr = [train[k]["result"]["score"] for k in keys]
    te = [test[k]["result"]["score"] for k in keys]
    n = len(keys)
    return {
        "n_pairs": n,
        "mean_train": math.fsum(tr) / n, "mean_test": math.fsum(te) / n,
        "mean_degradation": math.fsum(a - b for a, b in zip(tr, te)) / n,
        "rank_correlation": spearman(tr, te),
        "test_below_train_share": sum(1 for a, b in zip(tr, te) if b < a) / n,
        "pairs": [{"variant": train[k]["variant"], "train": a, "test": b}
                  for k, a, b in list(zip(keys, tr, te))[:50]],
    }


# ----- combined report -----


def overfit_report(
    ledger: TrialLedger,
    *,
    score_is_sharpe: bool = False,
    periods_per_year: float = 1.0,
    n_obs: Optional[int] = None,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
    min_n: int = 0,
    n_trials: Optional[int] = None,
) -> Dict[str, Any]:
    """Everything that can be computed from a ledger alone: the holdout
    guard status, train-versus-test degradation, logged walk-forward runs
    and, if ``score_is_sharpe``, the Deflated Sharpe Ratio of the best train
    variant. (PBO needs a returns matrix: call
    :func:`probability_of_backtest_overfitting` directly.)"""
    counts = ledger.counts()
    out: Dict[str, Any] = {
        "n_trials_train": counts["by_split"][TRAIN]["unique_variants"],
        "n_not_scorable": counts["n_not_scorable"],
        "holdout_guard": ledger.holdout_status(),
        "holdout": holdout_report(ledger, min_n=min_n),
        "walk_forward": [
            {k: e["data"].get(k) for k in ("key", "variant", "n_folds", "n_scorable", "mean", "std")}
            for e in ledger.events("walk_forward")
        ],
    }
    if score_is_sharpe:
        out["deflated_sharpe"] = ledger_deflated_sharpe(
            ledger, score_is_sharpe=True, periods_per_year=periods_per_year, n_obs=n_obs,
            skew=skew, kurtosis=kurtosis, min_n=min_n, n_trials=n_trials)
    else:
        out["deflated_sharpe"] = {
            "available": False, "reason": "pass score_is_sharpe=True if scores are Sharpe ratios"}
    return out
