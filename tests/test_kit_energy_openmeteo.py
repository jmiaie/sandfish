import json
import os
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from sandfish.kits.energy.openmeteo import (
    DEFAULT_MODEL,
    POINTS,
    coverage_by_month,
    fetch_series,
    main,
    month_chunks,
    planned_calls,
    previous_runs_url,
)

UTC = timezone.utc


def _payload_for(url, null_days=()):
    q = parse_qs(urlparse(url).query)
    start = date.fromisoformat(q["start_date"][0])
    end = date.fromisoformat(q["end_date"][0])
    col = q["hourly"][0]
    times, vals = [], []
    d = start
    while d <= end:
        for h in range(24):
            times.append(f"{d.isoformat()}T{h:02d}:00")
            vals.append(None if d in null_days else 10.0 + h)
        d += timedelta(days=1)
    return {
        "latitude": 1.0,
        "longitude": 2.0,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "hourly": {"time": times, col: vals},
    }


class FakeFetcher:
    def __init__(self, null_days=()):
        self.urls = []
        self.null_days = null_days

    def __call__(self, url):
        self.urls.append(url)
        return json.dumps(_payload_for(url, self.null_days)).encode()


def test_month_chunks():
    assert month_chunks(date(2021, 4, 15), date(2021, 6, 2)) == [
        (date(2021, 4, 15), date(2021, 4, 30)),
        (date(2021, 5, 1), date(2021, 5, 31)),
        (date(2021, 6, 1), date(2021, 6, 2)),
    ]
    assert month_chunks(date(2023, 12, 1), date(2024, 1, 31))[-1] == (
        date(2024, 1, 1),
        date(2024, 1, 31),
    )
    with pytest.raises(ValueError):
        month_chunks(date(2024, 2, 2), date(2024, 2, 1))


def test_url_pins_model_lead_and_utc():
    url = previous_runs_url(37.7749, -122.4194, date(2024, 3, 1), date(2024, 3, 31))
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).netloc == "previous-runs-api.open-meteo.com"
    assert q["models"] == [DEFAULT_MODEL] == ["gfs_seamless"]
    assert q["hourly"] == ["temperature_2m_previous_day2"]
    assert q["timezone"] == ["GMT"]
    assert q["start_date"] == ["2024-03-01"] and q["end_date"] == ["2024-03-31"]
    assert "archive" not in url and "historical-forecast" not in url


@pytest.mark.parametrize(
    "kw",
    [
        {"model": "best_match"},
        {"model": ""},
        {"model": "gfs_seamless,ecmwf_ifs"},
        {"lead_days": 0},
        {"lead_days": 8},
        {"lead_days": True},
    ],
)
def test_url_rejects_unpinned_model_or_bad_lead(kw):
    with pytest.raises(ValueError):
        previous_runs_url(1.0, 2.0, date(2024, 3, 1), date(2024, 3, 2), **kw)


def test_fetch_series_caches_and_merges(tmp_path):
    f = FakeFetcher()
    cache = str(tmp_path / "cache")
    s1 = fetch_series("sf", 37.0, -122.0, date(2024, 2, 20), date(2024, 3, 5), cache, fetcher=f)
    assert len(f.urls) == 2  # Feb chunk and March chunk
    assert all(k.tzinfo is not None and k.utcoffset() == timedelta(0) for k in s1)
    assert s1[datetime(2024, 3, 1, 5, tzinfo=UTC)] == 15.0
    assert len(s1) == (10 + 5) * 24  # Feb 20..29 (leap year) and Mar 1..5
    s2 = fetch_series("sf", 37.0, -122.0, date(2024, 2, 20), date(2024, 3, 5), cache, fetcher=f)
    assert len(f.urls) == 2  # fully cached: no new request
    assert s1 == s2
    assert os.path.exists(os.path.join(cache, DEFAULT_MODEL, "sf_2024-03.json"))


def test_null_hours_are_absent_and_coverage_reports_them(tmp_path):
    f = FakeFetcher(null_days=(date(2024, 3, 3),))
    s = fetch_series(
        "la", 34.0, -118.0, date(2024, 3, 1), date(2024, 3, 31), str(tmp_path), fetcher=f
    )
    assert len(s) == 30 * 24
    cov = coverage_by_month(s, date(2024, 3, 1), date(2024, 3, 31))
    assert cov == {"2024-03": round(30 / 31, 4)}


def test_coverage_full_and_empty_months():
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    s = {t0 + timedelta(hours=k): 1.0 for k in range(24 * 31)}
    cov = coverage_by_month(s, date(2024, 1, 1), date(2024, 2, 29))
    assert cov["2024-01"] == 1.0 and cov["2024-02"] == 0.0


def test_non_gmt_payload_is_refused(tmp_path):
    def local(url):
        p = _payload_for(url)
        p["timezone"] = "America/Los_Angeles"
        p["utc_offset_seconds"] = -28800
        return json.dumps(p).encode()

    with pytest.raises(ValueError):
        fetch_series(
            "sf", 1.0, 2.0, date(2024, 3, 1), date(2024, 3, 2), str(tmp_path), fetcher=local
        )


def test_error_payload_is_not_cached(tmp_path):
    def err(url):
        return json.dumps({"error": True, "reason": "rate limited"}).encode()

    with pytest.raises(RuntimeError):
        fetch_series("sf", 1.0, 2.0, date(2024, 3, 1), date(2024, 3, 2), str(tmp_path), fetcher=err)
    assert not list(tmp_path.rglob("*.json"))


def test_corrupt_cache_file_is_refetched(tmp_path):
    f = FakeFetcher()
    cache = str(tmp_path)
    os.makedirs(os.path.join(cache, DEFAULT_MODEL))
    with open(os.path.join(cache, DEFAULT_MODEL, "sf_2024-03.json"), "w") as fh:
        fh.write("{not json")
    s = fetch_series("sf", 1.0, 2.0, date(2024, 3, 1), date(2024, 3, 2), cache, fetcher=f)
    assert len(f.urls) == 1 and len(s) == 48


def test_planned_calls_counts_only_uncached(tmp_path):
    cache = str(tmp_path)
    assert (
        planned_calls(["sf", "la"], date(2024, 2, 1), date(2024, 4, 30), cache, DEFAULT_MODEL) == 6
    )
    fetch_series("sf", 1.0, 2.0, date(2024, 2, 1), date(2024, 2, 29), cache, fetcher=FakeFetcher())
    assert (
        planned_calls(["sf", "la"], date(2024, 2, 1), date(2024, 4, 30), cache, DEFAULT_MODEL) == 5
    )


def test_cli_refuses_large_plan_and_unknown_point(tmp_path, capsys):
    args = ["--cache", str(tmp_path), "--start", "2021-04-01", "--end", "2024-12-31"]
    assert main(args + ["--max-calls", "10"]) == 2  # 45 months x 3 points
    assert "refusing" in capsys.readouterr().err
    assert main(args + ["--points", "nowhere"]) == 2
    assert set(POINTS) == {"sf", "la", "sd"}
