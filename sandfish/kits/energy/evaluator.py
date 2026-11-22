"""Stage-1 evaluator: day-ahead hourly demand forecasts vs the operator's forecast."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from sandfish.kits.energy.availability import min_lag_hours
from sandfish.kits.energy.weather import WeatherLeakError, assert_safe_lead, is_lead_usable
from sandfish.search.evaluator import SPLITS, EvaluationResult

_HOUR = timedelta(hours=1)
_SEASONAL_LAG = 168
_MODELS = ("seasonal_naive", "lag_mean", "lag_mean_temp")
_NO_WEATHER = object()
_NO_BIAS = object()


def _clean(series: Mapping[datetime, float], label: str) -> Dict[datetime, float]:
    out: Dict[datetime, float] = {}
    for key, value in series.items():
        if not isinstance(key, datetime) or key.tzinfo is None or key.utcoffset() is None:
            raise ValueError(f"{label} keys must be timezone-aware datetimes")
        out[key.astimezone(timezone.utc)] = value
    return out


def _lagged(series: Mapping[datetime, float], t: datetime, lag: int) -> Optional[float]:
    try:
        return _get(series, t - timedelta(hours=lag))
    except OverflowError:
        return None


def _skill(num: float, den: float) -> Optional[float]:
    if den == 0 or not math.isfinite(num) or not math.isfinite(den):
        return None
    return 1 - num / den


def _get(series: Mapping[datetime, float], key: datetime) -> Optional[float]:
    value = series.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _check_range(name: str, rng: Tuple[date, date]) -> Tuple[date, date]:
    start, end = rng
    if start > end:
        raise ValueError(f"{name} start must be <= end")
    return start, end


class EnergyLoadEvaluator:
    """Scores lag-based demand forecasts as skill against the operator forecast."""

    name = "energy_load"

    def __init__(
        self,
        demand: Mapping[datetime, float],
        operator_forecast: Mapping[datetime, float],
        train_days: Tuple[date, date],
        test_days: Tuple[date, date],
        *,
        min_days: int = 60,
        min_lag_hours: int = 39,
        tz: str = "America/Los_Angeles",
        weather: Optional[Mapping[datetime, float]] = None,
        weather_lead_days: int = 2,
        bias_baseline: bool = False,
        bias_window_days: int = 7,
        bias_lag_days: int = 2,
    ) -> None:
        self._demand = _clean(demand, "demand")
        self._df = _clean(operator_forecast, "operator_forecast")
        self.train_days = _check_range("train_days", train_days)
        self.test_days = _check_range("test_days", test_days)
        if self.train_days[0] <= self.test_days[1] and self.test_days[0] <= self.train_days[1]:
            raise ValueError("train_days and test_days must not overlap")
        self.min_days = int(min_days)
        self.min_lag_hours = int(min_lag_hours)
        self.tz = tz
        self._zone = ZoneInfo(tz)
        if not isinstance(bias_baseline, bool):
            raise ValueError(f"bias_baseline must be a bool, got {bias_baseline!r}")
        if bias_baseline:
            for label, val, low in (
                ("bias_window_days", bias_window_days, 1),
                ("bias_lag_days", bias_lag_days, 2),
            ):
                if isinstance(val, bool) or not isinstance(val, int) or val < low:
                    raise ValueError(f"{label} must be an int >= {low}, got {val!r}")
            if bias_window_days > 365:
                raise ValueError(f"bias_window_days must be <= 365, got {bias_window_days!r}")
            if 24 * bias_lag_days < self.min_lag_hours:
                raise ValueError(
                    f"24 * bias_lag_days ({24 * bias_lag_days}) must be >= "
                    f"min_lag_hours ({self.min_lag_hours})"
                )
        self.bias_baseline = bias_baseline
        self.bias_window_days = bias_window_days
        self.bias_lag_days = bias_lag_days
        self.weather_lead_days: Optional[int] = None
        self._weather: Optional[Dict[datetime, float]] = None
        if weather is not None:
            # Unsafe lead is an error whatever models are evaluated later.
            self.weather_lead_days = assert_safe_lead(weather_lead_days, tz)
            # Naive weather keys raise here, like every other series in this evaluator
            # (_clean rejects naive keys); that is intended, not a weather-specific failure.
            self._weather = _clean(weather, "weather")

    def _day_hours(self, day: date) -> List[datetime]:
        """UTC hour starts of a local delivery day (23/24/25 of them)."""
        start = datetime(day.year, day.month, day.day, tzinfo=self._zone)
        nxt = day + timedelta(days=1)
        end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=self._zone)
        start_u = start.astimezone(timezone.utc)
        end_u = end.astimezone(timezone.utc)
        if end_u - start_u != 24 * _HOUR:
            return []
        first = start_u.replace(minute=0, second=0, microsecond=0)
        if first < start_u:
            first += _HOUR
        out = []
        t = first
        while t < end_u:
            out.append(t)
            t += _HOUR
        return out

    def _lags(self, variant: Mapping[str, Any]) -> Tuple[str, List[int]]:
        model = variant.get("model", "lag_mean")
        if model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}, not {model!r}")
        if model == "lag_mean_temp" and self._weather is None:
            raise ValueError("lag_mean_temp requires weather")
        if model == "seasonal_naive":
            lags = [_SEASONAL_LAG]
        else:
            raw = variant.get("lags")
            if raw is not None and not isinstance(raw, (list, tuple)):
                raise ValueError(f"lags must be a list of positive ints, got {raw!r}")
            if not raw:
                raise ValueError("lag_mean requires a non-empty 'lags' list")
            lags = []
            for lag in raw:
                if isinstance(lag, bool) or not isinstance(lag, int) or lag <= 0:
                    raise ValueError(f"lags must be positive ints, got {lag!r}")
                lags.append(lag)
        # Leak guard: demand lags must be >= min_lag_hours. A custom lag shorter than that
        # is rejected even with weather; weather never relaxes the demand guard.
        for lag in lags:
            if lag < self.min_lag_hours:
                raise ValueError(f"lag {lag}h would leak (min_lag_hours={self.min_lag_hours})")
        return model, lags

    def evaluate(self, variant: Mapping[str, Any], split: str) -> EvaluationResult:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        model, lags = self._lags(variant)
        coef = self._temp_coef(variant) if model == "lag_mean_temp" else None
        start, end = self.train_days if split == "train" else self.test_days

        days: List[str] = []
        mae_model: List[float] = []
        mae_df: List[float] = []
        mae_sn: List[float] = []
        total = 0
        mae_bc: List[float] = []
        no_wx = 0
        no_bias = 0
        day = start
        while day <= end:
            total += 1
            hours = self._day_hours(day)
            row = self._score_day(hours, lags, coef) if len(hours) == 24 else None
            if row is _NO_WEATHER:
                no_wx += 1
            elif row is _NO_BIAS:
                no_bias += 1
            elif row is not None:
                days.append(day.isoformat())
                mae_model.append(row[0])
                mae_df.append(row[1])
                mae_sn.append(row[2])
                if self.bias_baseline:
                    mae_bc.append(row[3])
            day += timedelta(days=1)

        n = len(days)
        s_model, s_df, s_sn = sum(mae_model), sum(mae_df), sum(mae_sn)
        meta: Dict[str, Any] = {
            "days": days,
            "loss_diff": [d - m for d, m in zip(mae_df, mae_model)],
            "skill_vs_seasonal_naive": _skill(s_model, s_sn),
            "skill_seasonal_naive_vs_operator": _skill(s_sn, s_df),
            "dropped_days": total - n,
            "split": split,
            "model": model,
        }
        if model == "lag_mean_temp":
            meta["days_without_weather"] = no_wx
        if self.bias_baseline:
            s_bc = sum(mae_bc)
            meta["skill_vs_bias_corrected_df"] = _skill(s_model, s_bc)
            meta["skill_bias_corrected_df_vs_operator"] = _skill(s_bc, s_df)
            meta["days_without_bias_baseline"] = no_bias
        if n < self.min_days:
            return EvaluationResult.not_scorable(
                n=n, reason=f"only {n} usable days (min_days={self.min_days})", **meta
            )
        if not all(math.isfinite(x) for x in (s_model, s_df, s_sn)) or any(
            not math.isfinite(x) for x in meta["loss_diff"]
        ):
            return EvaluationResult.not_scorable(n=n, reason="non-finite error totals", **meta)
        if s_df == 0:
            return EvaluationResult.not_scorable(
                n=n, reason="operator forecast has zero error", **meta
            )
        return EvaluationResult(score=1 - s_model / s_df, n=n, metadata=meta)

    @staticmethod
    def _temp_coef(variant: Mapping[str, Any]) -> float:
        c = variant.get("temp_coef", 0.0)
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
            raise ValueError(f"temp_coef must be a finite number, got {c!r}")
        return float(c)

    def _bias_forecast(self, t: datetime, op: float) -> Optional[float]:
        """BCDF(t): DF(t) plus the mean of D - DF at t - 24*k hours, k >= lag_days."""
        diffs: List[float] = []
        for k in range(self.bias_lag_days, self.bias_lag_days + self.bias_window_days):
            d = _lagged(self._demand, t, 24 * k)
            f = _lagged(self._df, t, 24 * k)
            if d is not None and f is not None:
                diffs.append(d - f)
        if len(diffs) < max(1, self.bias_window_days - 2):
            return None
        return op + sum(diffs) / len(diffs)

    def _score_day(
        self, hours: List[datetime], lags: List[int], coef: Optional[float] = None
    ) -> Any:
        if coef is not None:
            # Runtime guards, per scored target hour (the constructor scan covers only a
            # fixed date range, so it cannot vouch for every delivery date or tz).
            for t in hours:
                if not is_lead_usable(t, self.weather_lead_days or 0, self.tz):
                    raise WeatherLeakError(
                        f"weather lead {self.weather_lead_days}d is not usable at {t.isoformat()}"
                    )
                floor = min_lag_hours(t, self.tz)
                for lag in lags:
                    if lag < floor:
                        raise ValueError(
                            f"lag {lag}h would leak demand at {t.isoformat()} "
                            f"(needs >= {floor}h) for lag_mean_temp"
                        )
        err_m = err_d = err_s = err_b = 0.0
        bias_missing = False
        missing_wx = False
        wx = self._weather or {}
        for t in hours:
            actual = _get(self._demand, t)
            op = _get(self._df, t)
            sn = _lagged(self._demand, t, _SEASONAL_LAG)
            if actual is None or op is None or sn is None:
                return None
            vals: List[float] = []
            for lag in lags:
                v = _lagged(self._demand, t, lag)
                if v is None:
                    return None
                vals.append(v)
            pred = sum(v / len(vals) for v in vals)
            if coef is not None:
                # Lagged weather W[t-lag] is allowed: it is a lead-k forecast for an earlier
                # valid hour, issued even earlier than W[t] (< cutoff), so it leaks nothing.
                w_now = _get(wx, t)
                w_lags = [_lagged(wx, t, lag) for lag in lags]
                if w_now is None or any(w is None for w in w_lags):
                    missing_wx = True
                else:
                    w_mean = sum(w / len(w_lags) for w in w_lags if w is not None)
                    pred += coef * (w_now - w_mean)
            err_m += abs(actual / 24 - pred / 24)
            err_d += abs(actual / 24 - op / 24)
            err_s += abs(actual / 24 - sn / 24)
            if self.bias_baseline:
                bc = self._bias_forecast(t, op)
                if bc is None or not math.isfinite(bc):
                    bias_missing = True
                else:
                    err_b += abs(actual / 24 - bc / 24)
        if missing_wx:
            return _NO_WEATHER
        row = (err_m, err_d, err_s)
        if not all(math.isfinite(x) for x in row):
            return None
        if self.bias_baseline:
            return _NO_BIAS if bias_missing else (*row, err_b)
        return row


def split_summary(evaluator: EnergyLoadEvaluator) -> dict:
    """Return the evaluator's split ranges and guard settings."""
    out: Dict[str, Any] = {
        "train": [d.isoformat() for d in evaluator.train_days],
        "test": [d.isoformat() for d in evaluator.test_days],
        "min_days": evaluator.min_days,
        "min_lag_hours": evaluator.min_lag_hours,
    }
    if evaluator.weather_lead_days is not None:
        out["weather_lead_days"] = evaluator.weather_lead_days
    if evaluator.bias_baseline:
        out["bias_baseline"] = {
            "window_days": evaluator.bias_window_days,
            "lag_days": evaluator.bias_lag_days,
        }
    return out
