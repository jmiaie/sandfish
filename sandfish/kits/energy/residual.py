"""Residual corrections applied on top of the operator day-ahead forecast (DF).

The corrected forecast is ``F(t) = DF(t) + r(t)``; ``r(t)`` estimates the recent DF error
``e = D - DF`` from days at least ``lag_days`` old, optionally with a temperature term.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from sandfish.kits.energy.availability import min_lag_hours as _min_lag_hours
from sandfish.kits.energy.evaluator import EnergyLoadEvaluator, _clean, _get
from sandfish.kits.energy.weather import WeatherLeakError, assert_safe_lead, is_lead_usable

_ALLOWED = {"lag_days", "windows", "mix", "agg", "hour_weight", "temp"}
_COUNT_KEYS = ("scored", "not_24h", "missing_data", "no_weather", "no_correction", "no_bcdf")
_HOUR = timedelta(hours=1)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:
        return False


def validate_variant(variant: Any, min_lag_hours: int = 39) -> Dict[str, Any]:
    """Return a normalised copy of ``variant`` with defaults filled; raise ValueError if bad."""
    if not isinstance(variant, dict):
        raise ValueError(f"variant must be a dict, got {type(variant).__name__}")
    extra = set(variant) - _ALLOWED
    if extra:
        raise ValueError(f"unknown variant keys: {sorted(map(repr, extra))}")
    lag = variant.get("lag_days", 2)
    if not _is_int(lag) or not 2 <= lag <= 7:
        raise ValueError(f"lag_days must be an int in 2..7, got {lag!r}")
    if 24 * lag < min_lag_hours:
        raise ValueError(f"24 * lag_days ({24 * lag}) must be >= min_lag_hours ({min_lag_hours})")
    if "windows" not in variant:
        raise ValueError("windows is required")
    wins = variant["windows"]
    if not isinstance(wins, (list, tuple)) or len(wins) not in (1, 2):
        raise ValueError(f"windows must be a list of 1 or 2 ints, got {wins!r}")
    for w in wins:
        if not _is_int(w) or not 1 <= w <= 120:
            raise ValueError(f"each window must be an int in 1..120, got {w!r}")
    if len(wins) == 2 and not wins[0] < wins[1]:
        raise ValueError(f"windows must be strictly increasing, got {list(wins)!r}")
    mix = variant.get("mix")
    if len(wins) == 1:
        if "mix" in variant:
            raise ValueError("mix is allowed only with two windows")
    elif not _is_num(mix) or not -1.0 <= mix <= 2.0:
        raise ValueError(f"mix is required with two windows, in [-1, 2]; got {mix!r}")
    agg = variant.get("agg", "mean")
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    lam = variant.get("hour_weight", 1.0)
    if not _is_num(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError(f"hour_weight must be a number in [0, 1], got {lam!r}")
    temp = variant.get("temp")
    if temp is not None:
        if not isinstance(temp, dict) or len(temp) != 1:
            raise ValueError(f"temp must be None or a dict with exactly one key, got {temp!r}")
        ((key, val),) = temp.items()
        if key == "beta":
            if not _is_num(val):
                raise ValueError(f"temp beta must be a finite number, got {val!r}")
        elif key == "shrink":
            if not _is_num(val) or not 0 < val <= 1:
                raise ValueError(f"temp shrink must be in (0, 1], got {val!r}")
            if len(wins) != 1 or wins[0] < 14:
                raise ValueError("temp shrink requires a single window >= 14")
        else:
            raise ValueError(f"temp key must be 'beta' or 'shrink', got {key!r}")
        temp = dict(temp)
    return {
        "lag_days": lag,
        "windows": list(wins),
        "mix": mix,
        "agg": agg,
        "hour_weight": lam,
        "temp": temp,
    }


def _agg(values: List[float], how: str) -> float:
    if how == "median":
        return statistics.median(values)
    return sum(values) / len(values)


class ResidualEvaluator:
    """Computes DF corrections and scores them per local day against BCDF and DF."""

    def __init__(
        self,
        demand: Mapping[datetime, float],
        operator_forecast: Mapping[datetime, float],
        weather: Optional[Mapping[datetime, float]] = None,
        *,
        tz: str = "America/Los_Angeles",
        min_lag_hours: int = 39,
        weather_lead_days: int = 2,
    ) -> None:
        self.tz = tz
        self.min_lag_hours = int(min_lag_hours)
        self.weather_lead_days: Optional[int] = None
        wx_raw = None
        if weather is not None:
            self.weather_lead_days = assert_safe_lead(weather_lead_days, tz)
            wx_raw = _clean(weather, "weather")
        d_raw = _clean(demand, "demand")
        f_raw = _clean(operator_forecast, "operator_forecast")
        self._d = self._finite(d_raw)
        self._f = self._finite(f_raw)
        self._w: Optional[Dict[datetime, float]] = (
            None if wx_raw is None else self._finite(wx_raw)
        )
        self._e = {s: v - self._f[s] for s, v in self._d.items() if s in self._f}
        self._base = EnergyLoadEvaluator(
            d_raw,
            f_raw,
            (date(2000, 1, 1), date(2000, 1, 2)),
            (date(2000, 1, 3), date(2000, 1, 4)),
            tz=tz,
            bias_baseline=True,
        )

    @staticmethod
    def _finite(series: Mapping[datetime, Any]) -> Dict[datetime, float]:
        out: Dict[datetime, float] = {}
        for key in series:
            v = _get(series, key)
            if v is not None:
                out[key] = v
        return out

    # -- estimators -------------------------------------------------------------------
    def _hour_samples(self, t: datetime, lag: int, wmax: int) -> List[Optional[float]]:
        """Same-hour residuals for k = lag .. lag+wmax-1 (None where absent)."""
        return [self._e.get(t - 24 * k * _HOUR) for k in range(lag, lag + wmax)]

    def _level_samples(self, t: datetime, lag: int, wmax: int) -> List[Optional[float]]:
        """Trailing-24h block means for k = lag .. lag+wmax-1 (None if < 20 values)."""
        out: List[Optional[float]] = []
        for k in range(lag, lag + wmax):
            end = t - 24 * k * _HOUR
            vals = [x for x in (self._e.get(end - j * _HOUR) for j in range(24)) if x is not None]
            out.append(sum(vals) / len(vals) if len(vals) >= 20 else None)
        return out

    @staticmethod
    def _from_samples(samples: List[Optional[float]], wn: int, how: str) -> Optional[float]:
        vals = [x for x in samples[:wn] if x is not None]
        if len(vals) < max(1, wn - 2):
            return None
        return _agg(vals, how)

    def _core(self, v: Dict[str, Any], t: datetime) -> Optional[float]:
        wins = v["windows"]
        lag, how, lam = v["lag_days"], v["agg"], v["hour_weight"]
        wmax = wins[-1]
        hs = self._hour_samples(t, lag, wmax) if lam > 0 else None
        ls = self._level_samples(t, lag, wmax) if lam < 1 else None
        bs: List[float] = []
        for wn in wins:
            bh = self._from_samples(hs, wn, how) if hs is not None else None
            bl = self._from_samples(ls, wn, how) if ls is not None else None
            if lam == 1:
                b = bh
            elif lam == 0:
                b = bl
            else:
                b = None if bh is None or bl is None else lam * bh + (1 - lam) * bl
            if b is None:
                return None
            bs.append(b)
        if len(bs) == 1:
            return bs[0]
        return bs[1] + v["mix"] * (bs[0] - bs[1])

    def _temp_term(self, v: Dict[str, Any], t: datetime) -> Optional[float]:
        """Temperature adjustment, or None when missing or numerically unusable."""
        try:
            r = self._temp_term_raw(v, t)
        except OverflowError:
            return None
        return r if r is not None and math.isfinite(r) else None

    def _temp_term_raw(self, v: Dict[str, Any], t: datetime) -> Optional[float]:
        w = self._w or {}
        w_now = w.get(t)
        if w_now is None:
            return None
        lag = v["lag_days"]
        ((kind, param),) = v["temp"].items()
        wmax = v["windows"][-1]
        stamps = [t - 24 * k * _HOUR for k in range(lag, lag + wmax)]
        if kind == "beta":
            xs = [w[s] for s in stamps if s in w]
            if len(xs) < max(1, wmax - 2):
                return None
            return param * (w_now - sum(xs) / len(xs))
        xs, ys = [], []
        for s in stamps:
            if s in w and s in self._e:
                xs.append(w[s])
                ys.append(self._e[s])
        if len(xs) < max(5, wmax // 2):
            return None
        xbar = sum(xs) / len(xs)
        ybar = sum(ys) / len(ys)
        sxx = sum((x - xbar) ** 2 for x in xs)
        sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
        slope = sxy / sxx if sxx > 0 else 0.0
        return param * slope * (w_now - xbar)

    def _check_weather(self, v: Dict[str, Any]) -> None:
        if v["temp"] is not None and self._w is None:
            raise ValueError("variant has a temp term but no weather series was given")

    # -- public API -------------------------------------------------------------------
    def correction_at(self, variant: Mapping[str, Any], t: datetime) -> Optional[float]:
        """Return r(t) for aware hour ``t``, or None if unavailable."""
        v = validate_variant(variant, self.min_lag_hours)
        self._check_weather(v)
        if not isinstance(t, datetime) or t.tzinfo is None or t.utcoffset() is None:
            raise ValueError("t must be a timezone-aware datetime")
        t = t.astimezone(timezone.utc)
        try:
            r = self._core(v, t)
            if r is None or not math.isfinite(r):
                return None
            if v["temp"] is not None:
                term = self._temp_term(v, t)
                if term is None:
                    return None
                r += term
        except OverflowError:
            return None
        return r if math.isfinite(r) else None

    def scan(
        self, variant: Mapping[str, Any], first_day: date, last_day: date
    ) -> Tuple[Dict[date, Tuple[float, float, float]], Dict[str, int]]:
        """Score each local day: ``{day: (mae_model, mae_bcdf, mae_df)}`` plus bucket counts."""
        v = validate_variant(variant, self.min_lag_hours)
        self._check_weather(v)
        if first_day > last_day:
            raise ValueError("first_day must be <= last_day")
        counts = dict.fromkeys(_COUNT_KEYS, 0)
        out: Dict[date, Tuple[float, float, float]] = {}
        day = first_day
        while day <= last_day:
            res = self._score_day(v, day)
            if isinstance(res, str):
                counts[res] += 1
            else:
                counts["scored"] += 1
                out[day] = res
            day += timedelta(days=1)
        return out, counts

    def _score_day(self, v: Dict[str, Any], day: date) -> Any:
        hours = self._base._day_hours(day)
        if len(hours) != 24:
            return "not_24h"
        actual, op = [], []
        for t in hours:
            a, f = self._d.get(t), self._f.get(t)
            if a is None or f is None:
                return "missing_data"
            actual.append(a)
            op.append(f)
        terms: List[float] = []
        if v["temp"] is not None:
            for t in hours:
                term = self._temp_term(v, t)
                if term is None:
                    return "no_weather"
                terms.append(term)
        cores = []
        for t in hours:
            try:
                r = self._core(v, t)
                if r is not None and terms:
                    r = r + terms[len(cores)]
            except OverflowError:
                r = None
            if r is None or not math.isfinite(r):
                return "no_correction"
            cores.append(r)
        bcdf = []
        for t, f in zip(hours, op):
            try:
                b = self._base._bias_forecast(t, f)
            except OverflowError:
                b = None
            if b is None or not math.isfinite(b):
                return "no_bcdf"
            bcdf.append(b)
        lead = self.weather_lead_days or 0
        for t in hours:
            if v["temp"] is not None and not is_lead_usable(t, lead, self.tz):
                raise WeatherLeakError(
                    f"weather lead {lead}d is not usable at {t.isoformat()}"
                )
            if 24 * v["lag_days"] < _min_lag_hours(t, self.tz):
                raise ValueError(f"lag_days {v['lag_days']} would leak demand at {t.isoformat()}")
        m = 0.0
        for i, (a, f) in enumerate(zip(actual, op)):
            m += abs(a - (f + cores[i]))
        mb = sum(abs(a - b) for a, b in zip(actual, bcdf))
        md = sum(abs(a - f) for a, f in zip(actual, op))
        row = (m / 24, mb / 24, md / 24)
        return row if all(math.isfinite(x) for x in row) else "no_correction"


def default_residual_variants() -> List[Dict[str, Any]]:
    """The pre-registered 55-variant search space (freshly built on every call)."""
    out: List[Dict[str, Any]] = []
    out += [{"windows": [w]} for w in (3, 5, 7, 10, 14, 21, 28, 56)]
    out += [{"lag_days": 3, "windows": [7]}, {"lag_days": 3, "windows": [14]}]
    out += [{"agg": "median", "windows": [w]} for w in (7, 14, 28)]
    for pair in ((3, 28), (7, 28), (7, 56), (14, 56)):
        out += [{"windows": list(pair), "mix": m} for m in (0.25, 0.5, 0.75, 1.25, 1.5)]
    for lam in (0.0, 0.5):
        out += [{"hour_weight": lam, "windows": [w]} for w in (7, 14, 28)]
    for w in (7, 28):
        out += [{"windows": [w], "temp": {"beta": b}} for b in (-200, -100, 100, 200, 400)]
    for w in (28, 56, 90):
        out += [{"windows": [w], "temp": {"shrink": s}} for s in (0.5, 1.0)]
    return out
