"""Open-Meteo Previous Runs client for stage-2 weather (stdlib only, cached, pinned model).

Licence and use (checked 2026-11-18, https://open-meteo.com/en/terms):
- The free API is for NON-COMMERCIAL use only (research, personal, education). Subscriptions,
  advertising or commercial products need a paid plan, and the Previous Runs API is only in the
  Professional and Enterprise plans (not Standard).
- Free limits: 600 calls/minute, 5,000/hour, 10,000/day, 300,000/month. Requests longer than
  14 days count as several calls (one month is about 2.1 calls), so this client fetches one
  month per request, pauses between requests and refuses large uncached plans.
- The data is CC BY 4.0: credit Open-Meteo (https://open-meteo.com) in anything published.
Raw responses are cached as JSON under a caller-chosen directory. Never commit that directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode

from sandfish.kits.energy.weather import merge_weather, parse_previous_runs

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DEFAULT_MODEL = "gfs_seamless"
USER_AGENT = "sandfish-research (non-commercial research)"
# Approximate city points for the California ISO load area (name -> (lat, lon)).
POINTS: Dict[str, Tuple[float, float]] = {
    "sf": (37.7749, -122.4194),
    "la": (34.0522, -118.2437),
    "sd": (32.7157, -117.1611),
}
Fetcher = Callable[[str], bytes]


def month_chunks(start: date, end: date) -> List[Tuple[date, date]]:
    """Split ``start..end`` (inclusive) into calendar-month ``(first, last)`` chunks."""
    if start > end:
        raise ValueError("start must be <= end")
    out: List[Tuple[date, date]] = []
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        out.append((cur, min(nxt - timedelta(days=1), end)))
        cur = nxt
    return out


def previous_runs_url(
    lat: float,
    lon: float,
    start: date,
    end: date,
    *,
    variable: str = "temperature_2m",
    lead_days: int = 2,
    model: str = DEFAULT_MODEL,
) -> str:
    """Build a Previous Runs URL: fixed lead column, pinned model, UTC times."""
    if isinstance(lead_days, bool) or not isinstance(lead_days, int) or not 1 <= lead_days <= 7:
        raise ValueError(f"lead_days must be an int in 1..7, got {lead_days!r}")
    if not model or not isinstance(model, str) or "," in model:
        raise ValueError("model must be one pinned model name (no best_match, no list)")
    if model == "best_match":
        raise ValueError("model must be pinned; best_match can change the underlying model")
    for name, value in (("lat", lat), ("lon", lon)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
    query = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": f"{variable}_previous_day{lead_days}",
        "models": model,
        "timezone": "GMT",
    }
    return f"{BASE_URL}?{urlencode(query)}"


def _http_fetch(url: str) -> bytes:
    """GET with a descriptive User-Agent, retries and a pause on HTTP 429."""
    last: Optional[Exception] = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https host)
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            time.sleep(65 if exc.code == 429 else 5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"fetch failed after retries: {last}")


def _cache_file(cache_dir: str, model: str, name: str, start: date) -> str:
    return os.path.join(cache_dir, model, f"{name}_{start:%Y-%m}.json")


def fetch_chunk(
    url: str, path: str, *, fetcher: Optional[Fetcher] = None, pause: float = 0.5
) -> Tuple[dict, bool]:
    """Return ``(payload, from_cache)``; write the raw response atomically on a miss."""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh), True
        except ValueError:
            pass  # corrupt cache file: refetch
    raw = (fetcher or _http_fetch)(url)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("error"):
        raise RuntimeError(f"Open-Meteo returned an error: {str(payload)[:200]}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    if fetcher is None and pause > 0:
        time.sleep(pause)
    return payload, False


def fetch_series(
    name: str,
    lat: float,
    lon: float,
    start: date,
    end: date,
    cache_dir: str,
    *,
    variable: str = "temperature_2m",
    lead_days: int = 2,
    model: str = DEFAULT_MODEL,
    fetcher: Optional[Fetcher] = None,
    pause: float = 0.5,
) -> Dict[datetime, float]:
    """Fetch (or load from cache) one point month by month and merge to ``{UTC hour: value}``."""
    maps = []
    for first, last in month_chunks(start, end):
        url = previous_runs_url(
            lat, lon, first, last, variable=variable, lead_days=lead_days, model=model
        )
        payload, _ = fetch_chunk(
            url, _cache_file(cache_dir, model, name, first), fetcher=fetcher, pause=pause
        )
        maps.append(parse_previous_runs(payload, lead_days=lead_days, variable=variable))
    return merge_weather(*maps)


def coverage_by_month(series: Mapping[datetime, float], start: date, end: date) -> Dict[str, float]:
    """Share of UTC hours in each calendar month of ``start..end`` that have a value."""
    out: Dict[str, float] = {}
    for first, last in month_chunks(start, end):
        t0 = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
        hours = int(((last - first).days + 1) * 24)
        have = sum(1 for k in range(hours) if (t0 + timedelta(hours=k)) in series)
        out[f"{first:%Y-%m}"] = round(have / hours, 4)
    return out


def planned_calls(names: Sequence[str], start: date, end: date, cache_dir: str, model: str) -> int:
    """Number of requests a fetch would make (uncached monthly chunks)."""
    return sum(
        1
        for name in names
        for first, _ in month_chunks(start, end)
        if not os.path.exists(_cache_file(cache_dir, model, name, first))
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: fetch cached monthly chunks for the named points and print a coverage report."""
    p = argparse.ArgumentParser(description="Open-Meteo Previous Runs fetch + coverage")
    p.add_argument("--cache", required=True, help="cache dir (outside git)")
    p.add_argument("--start", required=True, type=date.fromisoformat)
    p.add_argument("--end", required=True, type=date.fromisoformat)
    p.add_argument("--points", default="sf,la,sd")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lead-days", type=int, default=2)
    p.add_argument("--max-calls", type=int, default=300, help="refuse larger uncached plans")
    args = p.parse_args(argv)
    names = [n.strip() for n in args.points.split(",") if n.strip()]
    unknown = [n for n in names if n not in POINTS]
    if unknown:
        print(f"unknown points: {unknown}; known: {sorted(POINTS)}", file=sys.stderr)
        return 2
    calls = planned_calls(names, args.start, args.end, args.cache, args.model)
    if calls > args.max_calls:
        print(
            f"refusing {calls} uncached requests (> --max-calls {args.max_calls})", file=sys.stderr
        )
        return 2
    print(
        "Open-Meteo free API: non-commercial research use only; data CC BY 4.0 (credit "
        "Open-Meteo.com).",
        file=sys.stderr,
    )
    report: Dict[str, dict] = {}
    for name in names:
        lat, lon = POINTS[name]
        series = fetch_series(
            name, lat, lon, args.start, args.end, args.cache,
            lead_days=args.lead_days, model=args.model,
        )  # fmt: skip
        report[name] = {
            "hours": len(series),
            "coverage_by_month": coverage_by_month(series, args.start, args.end),
        }
    print(
        json.dumps(
            {"model": args.model, "lead_days": args.lead_days, "requests": calls, "points": report},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
