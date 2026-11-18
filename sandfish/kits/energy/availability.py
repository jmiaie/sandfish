"""Day-ahead availability rule: which demand hours are usable for a forecast."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

TZ = "America/Los_Angeles"
CUTOFF_HOUR = 10
# 39 is the largest minimum lag over all days: on the 25-hour fall-back day the
# last hour (23:00 local) is 39 real hours after 09:00 local on D-1. On ordinary
# 24-hour days 38 would be enough, but one uniform lag must be valid for every day.
UNIFORM_MIN_LAG_HOURS = 39


def _require_aware(dt: datetime, name: str = "datetime") -> None:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def last_usable_hour(delivery_day: date, tz: str = TZ) -> datetime:
    """Return 09:00 local on ``delivery_day - 1`` as an aware datetime in ``tz``."""
    prev = delivery_day - timedelta(days=1)
    return datetime(prev.year, prev.month, prev.day, CUTOFF_HOUR - 1, tzinfo=ZoneInfo(tz))


def _local_date(dt: datetime, tz: str) -> date:
    return dt.astimezone(ZoneInfo(tz)).date()


def min_lag_hours(target: datetime, tz: str = TZ) -> int:
    """Real elapsed whole hours from the last usable hour to ``target``."""
    _require_aware(target, "target")
    last = last_usable_hour(_local_date(target, tz), tz)
    delta = target.astimezone(timezone.utc) - last.astimezone(timezone.utc)
    return int(delta.total_seconds() // 3600)


def is_lag_valid(target: datetime, lag_hours: int, tz: str = TZ) -> bool:
    """True iff ``lag_hours`` is at least the minimum lag for ``target``."""
    if isinstance(lag_hours, bool) or not isinstance(lag_hours, int):
        raise TypeError("lag_hours must be an int")
    return lag_hours >= min_lag_hours(target, tz)


class LeakageError(ValueError):
    """A feature uses data not available at the forecast cutoff."""


def assert_no_leak(target: datetime, feature_times: Iterable[datetime], tz: str = TZ) -> None:
    """Raise ``LeakageError`` if any feature time is after the last usable hour."""
    _require_aware(target, "target")
    limit = last_usable_hour(_local_date(target, tz), tz)
    for ft in feature_times:
        _require_aware(ft, "feature time")
        if ft.astimezone(timezone.utc) > limit.astimezone(timezone.utc):
            raise LeakageError(f"feature timestamp {ft.isoformat()} is after {limit.isoformat()}")


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def valid_variant(variant: Mapping[str, Any]) -> bool:
    """Predicate for ``run_grid(valid=...)``; malformed input returns False."""
    try:
        lags = variant["lags"]
        if not isinstance(lags, (list, tuple)) or not lags:
            return False
        if not all(_is_int(x) and x >= UNIFORM_MIN_LAG_HOURS for x in lags):
            return False
        if "train_days" in variant:
            td = variant["train_days"]
            if not _is_int(td) or td * 24 < max(lags):
                return False
        return True
    except Exception:
        return False


def _require_whole_hour_day(d: date, tz: str) -> None:
    """Raise ``ValueError`` if local day ``d`` is not a whole number of hours long."""
    zone = ZoneInfo(tz)
    start = datetime(d.year, d.month, d.day, tzinfo=zone).astimezone(timezone.utc)
    nxt = d + timedelta(days=1)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=zone).astimezone(timezone.utc)
    seconds = (end - start).total_seconds()
    if seconds % 3600:
        raise ValueError(
            f"local day {d.isoformat()} in {tz} is {seconds / 3600:g} hours long, "
            "not a whole number of hours (half-hour DST shifts are unsupported)"
        )


def earliest_valid_lag(days: Iterable[date], tz: str = TZ) -> int:
    """Maximum ``min_lag_hours`` over every real hour of every given local day.

    Raises ``ValueError`` for a day that is not a whole number of hours long
    (an unsupported half-hour DST shift such as Australia/Lord_Howe).
    """
    zone = ZoneInfo(tz)
    worst: Optional[int] = None
    for d in days:
        _require_whole_hour_day(d, tz)
        # Scan a UTC window wide enough for any offset and keep every hour whose
        # local date is d; a rollback across midnight can revisit d later.
        t = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - timedelta(hours=30)
        for _ in range(84):
            if t.astimezone(zone).date() == d:
                lag = min_lag_hours(t, tz)
                if worst is None or lag > worst:
                    worst = lag
            t += timedelta(hours=1)
    if worst is None:
        raise ValueError("days must be non-empty")
    return worst
