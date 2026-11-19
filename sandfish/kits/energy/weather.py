"""Stage-2 weather features: Open-Meteo Previous Runs parsing with leak checks."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Mapping, Set
from zoneinfo import ZoneInfo

TZ = "America/Los_Angeles"
CUTOFF_HOUR = 10
_UTC_NAMES = {"gmt", "utc", "etc/utc", "etc/gmt", "z"}


class WeatherLeakError(ValueError):
    """A weather lead is not usable for every hour of the delivery day."""


def _aware_utc(dt: Any, name: str) -> datetime:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return dt.astimezone(timezone.utc)


def _check_lead(lead_days: Any) -> int:
    if isinstance(lead_days, bool) or not isinstance(lead_days, int) or not 1 <= lead_days <= 7:
        raise ValueError(f"lead_days must be an int in 1..7, got {lead_days!r}")
    return lead_days


def forecast_issue_time(valid: datetime, lead_days: int) -> datetime:
    """Issue time of ``previous_day<lead_days>`` at ``valid``: exactly 24*k real hours before."""
    v = _aware_utc(valid, "valid")
    return v - timedelta(hours=24 * _check_lead(lead_days))


def is_lead_usable(valid: datetime, lead_days: int, tz: str = TZ) -> bool:
    """True iff the issue time is strictly before 10:00 local on the day before delivery."""
    issue = forecast_issue_time(valid, lead_days)
    zone = ZoneInfo(tz)
    day: date = valid.astimezone(zone).date() - timedelta(days=1)
    cutoff = datetime(day.year, day.month, day.day, CUTOFF_HOUR, tzinfo=zone)
    return issue < cutoff.astimezone(timezone.utc)


def min_usable_lead_days(valid: datetime, tz: str = TZ) -> int:
    """Smallest k in 1..7 whose forecast is usable at ``valid``."""
    for k in range(1, 8):
        if is_lead_usable(valid, k, tz):
            return k
    raise WeatherLeakError(f"no lead in 1..7 is usable at {valid.isoformat()}")


@lru_cache(maxsize=None)
def safe_lead_days(tz: str = TZ) -> int:
    """Smallest lead usable for every hour of every day (scans 2023-2025, incl. DST days)."""
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    worst = 1
    while t < end:
        worst = max(worst, min_usable_lead_days(t, tz))
        t += timedelta(hours=1)
    return worst


def assert_safe_lead(lead_days: int, tz: str = TZ) -> int:
    """Return ``lead_days`` if safe for all 24 hours, else raise ``WeatherLeakError``."""
    k = _check_lead(lead_days)
    if k < safe_lead_days(tz):
        raise WeatherLeakError(
            f"lead_days={k} is not usable for every hour (minimum safe lead is "
            f"{safe_lead_days(tz)})"
        )
    return k


def _proves_utc(payload: Mapping[str, Any]) -> bool:
    off = payload.get("utc_offset_seconds")
    if isinstance(off, bool) or not isinstance(off, (int, float)) or off != 0:
        return False
    zone = payload.get("timezone")
    return isinstance(zone, str) and zone.strip().lower() in _UTC_NAMES


def parse_previous_runs(
    payload: Mapping[str, Any], *, lead_days: int = 2, variable: str = "temperature_2m"
) -> Dict[datetime, float]:
    """Parse a Previous Runs response into ``{aware UTC valid hour: value}``."""
    assert_safe_lead(lead_days)
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping) or "time" not in hourly:
        raise ValueError("payload lacks hourly/time")
    if not _proves_utc(payload):
        raise ValueError("payload does not prove UTC (request with timezone=GMT)")
    column = f"{variable}_previous_day{lead_days}"
    if column not in hourly:
        raise ValueError(f"payload lacks column {column!r}")
    times, values = hourly["time"], hourly[column]
    if not isinstance(times, (list, tuple)) or not isinstance(values, (list, tuple)):
        raise ValueError("hourly time and value columns must be lists")
    if len(times) != len(values):
        raise ValueError(f"column {column!r} length differs from time")
    out: Dict[datetime, float] = {}
    seen: Set[datetime] = set()
    for raw, val in zip(times, values):
        try:
            t = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            raise ValueError(f"bad time {raw!r}") from None
        if t.tzinfo is not None or raw.strip().upper().endswith("Z"):
            raise ValueError(f"time {raw!r} must be a naive ISO string (no offset or Z)")
        if t.minute or t.second or t.microsecond:
            raise ValueError(f"time {raw!r} is not on a whole hour")
        t = t.replace(tzinfo=timezone.utc)
        if t in seen:
            raise ValueError(f"duplicate time {raw!r}")
        seen.add(t)
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"non-numeric value {val!r} at {raw!r}")
        if math.isfinite(val):
            out[t] = float(val)
    return out


def merge_weather(*maps: Mapping[datetime, float]) -> Dict[datetime, float]:
    """Merge parsed maps; an hour present twice must agree within 1e-9."""
    out: Dict[datetime, float] = {}
    for m in maps:
        for key, val in m.items():
            k = _aware_utc(key, "weather key")
            if k in out and not abs(out[k] - val) <= 1e-9:
                raise ValueError(f"conflicting values at {k.isoformat()}: {out[k]} vs {val}")
            out.setdefault(k, val)
    return out


def population_weighted(
    maps: Mapping[str, Mapping[datetime, float]], weights: Mapping[str, float]
) -> Dict[datetime, float]:
    """Weighted mean over hours present at every positively weighted point."""
    for name, w in weights.items():
        if name not in maps:
            raise ValueError(f"weight for unknown point {name!r}")
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ValueError(f"weight for {name!r} must be a number")
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"weight for {name!r} must be finite and >= 0")
    top = max((float(w) for w in weights.values()), default=0.0)
    if not top > 0:
        raise ValueError("weights must have a positive sum")
    # Normalise by the maximum weight so sums cannot overflow (every share is in (0, 1]).
    active = {n: float(w) / top for n, w in weights.items() if w > 0}
    total = sum(active.values())
    names = list(active)
    common = set(maps[names[0]])
    for n in names[1:]:
        common &= set(maps[n])
    return {
        t: sum((active[n] / total) * maps[n][t] for n in names) for t in sorted(common)
    }
