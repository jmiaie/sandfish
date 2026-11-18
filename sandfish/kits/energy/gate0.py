"""Kit 1 Gate 0: offline check of EIA-930 bulk (EBA.txt) demand and forecast series."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Collection, Iterable, Iterator, Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

_PERIOD_RE = re.compile(r"[0-9]{8}T[0-9]{2}Z")
_HOUR = timedelta(hours=1)


def parse_eia_hour(period: str) -> datetime:
    """Parse ``YYYYMMDDTHHZ`` into an aware UTC datetime."""
    if not isinstance(period, str) or not _PERIOD_RE.fullmatch(period):
        raise ValueError(f"invalid EIA period: {period!r}")
    try:
        return datetime.strptime(period, "%Y%m%dT%HZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"invalid EIA period: {period!r}") from exc


def iter_series(
    lines: Iterable[str],
    series_ids: Optional[Collection[str]] = None,
    stats: Optional[dict] = None,
) -> Iterator[dict]:
    """Yield series dicts from NDJSON lines, optionally filtered by series id."""
    if stats is not None:
        for key in ("lines", "series", "skipped_malformed"):
            stats.setdefault(key, 0)
    for line in lines:
        if stats is not None:
            stats["lines"] += 1
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if not isinstance(obj, dict):
            if stats is not None:
                stats["skipped_malformed"] += 1
            continue
        sid = obj.get("series_id")
        if not isinstance(sid, str):
            continue
        if stats is not None:
            stats["series"] += 1
        if series_ids is None or sid in series_ids:
            yield obj


def _hour_map(data: Sequence[Sequence[Any]]) -> tuple[dict[datetime, Any], int]:
    """Map UTC hour -> value (last occurrence wins); also return the bad-period count."""
    hours: dict[datetime, Any] = {}
    bad = 0
    for pair in data:
        try:
            hour = parse_eia_hour(pair[0])
        except (ValueError, IndexError, TypeError, KeyError):
            bad += 1
            continue
        try:
            value = pair[1]
        except (IndexError, TypeError, KeyError):
            value = None
        hours[hour] = value
    return hours, bad


def _summary(hours: dict[datetime, Any], bad: int) -> dict:
    ordered = sorted(hours)
    if not ordered:
        return {
            "count": 0,
            "null_values": 0,
            "first": None,
            "last": None,
            "expected_hours": 0,
            "missing_hours": 0,
            "longest_gap_hours": 0,
            "bad_periods": bad,
        }
    expected = int((ordered[-1] - ordered[0]) / _HOUR) + 1
    longest = 0
    for prev, cur in zip(ordered, ordered[1:]):
        longest = max(longest, int((cur - prev) / _HOUR) - 1)
    return {
        "count": len(ordered),
        "null_values": sum(1 for h in ordered if hours[h] is None),
        "first": ordered[0].isoformat(),
        "last": ordered[-1].isoformat(),
        "expected_hours": expected,
        "missing_hours": expected - len(ordered),
        "longest_gap_hours": longest,
        "bad_periods": bad,
    }


def summarize_series(data: Sequence[Sequence[Any]]) -> dict:
    """Summarize coverage of ``[period, value]`` pairs over UTC hours."""
    hours, bad = _hour_map(data)
    return _summary(hours, bad)


def local_day_hours(day: date, tz: str = "America/Los_Angeles") -> int:
    """Return the number of elapsed hours (23, 24 or 25) in a local calendar day."""
    zone = ZoneInfo(tz)
    start = datetime.combine(day, time(0), tzinfo=zone).astimezone(timezone.utc)
    nxt = datetime.combine(day + timedelta(days=1), time(0), tzinfo=zone)
    return int((nxt.astimezone(timezone.utc) - start) / _HOUR)


def check_ba(lines: Iterable[str], ba: str = "CISO") -> dict:
    """Check that a balancing authority has demand and day-ahead forecast series."""
    ids = {"D": f"EBA.{ba}-ALL.D.H", "DF": f"EBA.{ba}-ALL.DF.H"}
    stats: dict = {}
    best: dict[str, tuple[dict[datetime, Any], dict]] = {}
    for obj in iter_series(lines, set(ids.values()), stats):
        key = "D" if obj["series_id"] == ids["D"] else "DF"
        data = obj.get("data")
        hours, bad = _hour_map(data if isinstance(data, list) else [])
        summary = _summary(hours, bad)
        if key not in best or summary["count"] > best[key][1]["count"]:
            best[key] = (hours, summary)
    overlap = 0
    if "D" in best and "DF" in best:
        d, df = best["D"][0], best["DF"][0]
        overlap = sum(1 for h, v in d.items() if v is not None and df.get(h) is not None)
    return {
        "ba": ba,
        "has_D": "D" in best,
        "has_DF": "DF" in best,
        "D": best["D"][1] if "D" in best else None,
        "DF": best["DF"][1] if "DF" in best else None,
        "overlap_hours": overlap,
        "parse_stats": stats,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns 0 when both D and DF series are present."""
    parser = argparse.ArgumentParser(description="EIA-930 Gate 0 data check")
    parser.add_argument("path", help="EBA.txt path, or - for stdin")
    parser.add_argument("--ba", default="CISO")
    args = parser.parse_args(argv)
    if args.path == "-":
        result = check_ba(sys.stdin, args.ba)
    else:
        with open(args.path, encoding="utf-8") as fh:
            result = check_ba(fh, args.ba)
    print(json.dumps(result, indent=2))
    return 0 if result["has_D"] and result["has_DF"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
