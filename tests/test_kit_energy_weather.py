"""Tests for kit 1 stage 2: weather features and the lag_mean_temp model."""

from datetime import date, datetime, timedelta, timezone

import pytest

from sandfish.kits.energy.evaluator import EnergyLoadEvaluator, split_summary
from sandfish.kits.energy.weather import (
    WeatherLeakError,
    forecast_issue_time,
    is_lead_usable,
    merge_weather,
    min_usable_lead_days,
    parse_previous_runs,
    population_weighted,
)

UTC = timezone.utc
START = datetime(2023, 12, 1, tzinfo=UTC)
TRAIN = (date(2024, 1, 10), date(2024, 3, 31))
TEST = (date(2024, 4, 1), date(2024, 5, 31))


def payload(times=None, values=None, **kw):
    times = times if times is not None else ["2024-03-01T00:00", "2024-03-01T01:00"]
    values = values if values is not None else [1.0, 2.0]
    p = {
        "latitude": 1.0,
        "longitude": 2.0,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "hourly": {"time": times, "temperature_2m_previous_day2": values},
    }
    p.update(kw)
    return p


def h(*args):
    return datetime(*args, tzinfo=UTC)


# ---- lead arithmetic -------------------------------------------------------------------


def test_issue_time_is_utc_arithmetic_across_dst():
    valid = datetime(2024, 3, 11, 12, tzinfo=UTC)
    assert forecast_issue_time(valid, 2) == datetime(2024, 3, 9, 12, tzinfo=UTC)
    with pytest.raises(ValueError):
        forecast_issue_time(datetime(2024, 3, 11, 12), 2)
    for bad in (0, 8, True, 2.0):
        with pytest.raises(ValueError):
            forecast_issue_time(valid, bad)


def test_lead_usable_spring_forward_day():
    # 2024-03-10: PST->PDT. Cutoff on 03-09 is 10:00 PST = 18:00 UTC.
    last = h(2024, 3, 11, 6)  # 23:00 PDT on 03-10 (last hour of the 23h day)
    first = h(2024, 3, 10, 8)  # 00:00 PST on 03-10
    assert is_lead_usable(first, 1)  # issue 03-09 08:00 UTC = 00:00 PST
    assert not is_lead_usable(last, 1)  # issue 03-10 06:00 UTC
    assert is_lead_usable(last, 2)
    assert min_usable_lead_days(last) == 2


def test_lead_usable_fall_back_day():
    # 2024-11-03: PDT->PST, 25 hours. Cutoff on 11-02 is 10:00 PDT = 17:00 UTC.
    last = h(2024, 11, 4, 7)  # 23:00 PST on 11-03
    assert forecast_issue_time(last, 1) == h(2024, 11, 3, 7)
    assert not is_lead_usable(last, 1)
    assert is_lead_usable(last, 2)
    # issue exactly at the cutoff is unusable (strictly before)
    edge = h(2024, 11, 3, 17)  # valid; lead 1 issue = 11-02 17:00 UTC = cutoff
    assert not is_lead_usable(edge, 1)
    assert is_lead_usable(edge - timedelta(hours=1), 1)


def test_lead_usable_is_local_not_utc():
    # Valid 00:00 local on a PDT day: lead 1 issue is 00:00 local D-1, before 10:00.
    assert min_usable_lead_days(h(2024, 7, 2, 7)) == 1
    assert min_usable_lead_days(h(2024, 7, 3, 6, 59 - 59)) == 2  # 23:00 PDT on 07-02


# ---- parse_previous_runs -----------------------------------------------------------------


def test_parse_ok_skips_missing():
    p = payload(
        ["2024-03-01T00:00", "2024-03-01T01:00", "2024-03-01T02:00", "2024-03-01T03:00"],
        [1.0, None, float("nan"), 3],
    )
    assert parse_previous_runs(p) == {h(2024, 3, 1, 0): 1.0, h(2024, 3, 1, 3): 3.0}


def test_parse_rejects_local_times():
    for zone in ("America/Los_Angeles", "Europe/Berlin"):
        with pytest.raises(ValueError):
            parse_previous_runs(payload(timezone=zone, utc_offset_seconds=-28800))
    with pytest.raises(ValueError):
        parse_previous_runs(payload(utc_offset_seconds=3600))  # GMT with offset


def test_parse_requires_utc_proof():
    p = payload()
    del p["timezone"]
    with pytest.raises(ValueError):
        parse_previous_runs(p)
    p = payload()
    del p["utc_offset_seconds"]
    with pytest.raises(ValueError):
        parse_previous_runs(p)
    for ok in ("utc", "Etc/UTC", "etc/gmt", "Z", "GMT"):
        assert parse_previous_runs(payload(timezone=ok))


def test_parse_plain_column_only_refused():
    for col in ("temperature_2m", "temperature_2m_previous_day0"):
        p = payload()
        p["hourly"] = {"time": ["2024-03-01T00:00"], col: [1.0]}
        with pytest.raises(ValueError):
            parse_previous_runs(p)


def test_parse_structure_errors():
    with pytest.raises(ValueError):
        parse_previous_runs({"utc_offset_seconds": 0, "timezone": "GMT"})
    with pytest.raises(ValueError):
        parse_previous_runs(payload(values=[1.0]))  # length mismatch
    with pytest.raises(ValueError):
        parse_previous_runs(payload(times=["2024-03-01T00:00"] * 2))  # duplicates


def test_parse_lead_one_is_a_leak():
    p = payload()
    p["hourly"]["temperature_2m_previous_day1"] = [1.0, 2.0]
    with pytest.raises(WeatherLeakError):
        parse_previous_runs(p, lead_days=1)
    assert issubclass(WeatherLeakError, ValueError)


# ---- merge / weighted ----------------------------------------------------------------------


def test_merge_identical_ok_and_conflict_raises():
    a = {h(2024, 3, 1, 0): 1.0, h(2024, 3, 1, 1): 2.0}
    b = {h(2024, 3, 1, 1): 2.0 + 1e-12, h(2024, 3, 1, 2): 3.0}
    assert merge_weather(a, b) == {h(2024, 3, 1, 0): 1.0, h(2024, 3, 1, 1): 2.0,
                                   h(2024, 3, 1, 2): 3.0}
    with pytest.raises(ValueError, match="2024-03-01T01:00"):
        merge_weather(a, {h(2024, 3, 1, 1): 2.5})


def test_population_weighted_drops_hour_missing_at_weighted_point():
    t0, t1 = h(2024, 3, 1, 0), h(2024, 3, 1, 1)
    maps = {"a": {t0: 10.0, t1: 20.0}, "b": {t0: 20.0}, "c": {}}
    out = population_weighted(maps, {"a": 1.0, "b": 3.0, "c": 0.0})
    assert out == {t0: pytest.approx(17.5)}


def test_population_weighted_validation():
    maps = {"a": {h(2024, 3, 1, 0): 1.0}}
    for w in ({"a": -1.0}, {"a": 0.0}, {"a": float("nan")}, {"a": 1.0, "zz": 1.0}, {}):
        with pytest.raises(ValueError):
            population_weighted(maps, w)


# ---- evaluator -------------------------------------------------------------------------------


def _pattern(i):
    return ((i % 168) * 37 % 101) / 101.0


def series(hours=24 * 200):
    demand, op, wx = {}, {}, {}
    for i in range(hours):
        t = START + timedelta(hours=i)
        demand[t] = 1000 + 50 * _pattern(i)
        op[t] = demand[t] + 10.0
        wx[t] = 15 + 10 * _pattern(i * 7) + (i // 24) % 11
    return demand, op, wx


def make_eval(weather=True, **kw):
    d, o, w = series()
    args = dict(min_days=10)
    if weather:
        args["weather"] = w
    args.update(kw)
    return EnergyLoadEvaluator(d, o, TRAIN, TEST, **args), w


def test_unsafe_lead_fails_construction():
    d, o, w = series(24 * 10)
    with pytest.raises(WeatherLeakError):
        EnergyLoadEvaluator(d, o, TRAIN, TEST, weather=w, weather_lead_days=1)
    with pytest.raises(ValueError):
        EnergyLoadEvaluator(d, o, TRAIN, TEST, weather=w, weather_lead_days=True)


def test_lag_mean_temp_requires_weather_and_valid_coef():
    ev, _ = make_eval(weather=False)
    with pytest.raises(ValueError):
        ev.evaluate({"model": "lag_mean_temp", "lags": [168]}, "test")
    ev, _ = make_eval()
    for bad in (True, "1", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ev.evaluate({"model": "lag_mean_temp", "lags": [168], "temp_coef": bad}, "test")
    with pytest.raises(ValueError):  # demand leak guard still applies with weather
        ev.evaluate({"model": "lag_mean_temp", "lags": [24], "temp_coef": 1.0}, "test")


def test_zero_coef_equals_lag_mean_and_day_sets_of_other_models_unchanged():
    ev, w = make_eval()
    plain, _ = make_eval(weather=False)
    v = {"model": "lag_mean", "lags": [48, 168]}
    a, b = ev.evaluate(v, "test"), plain.evaluate(v, "test")
    assert a.score == b.score and a.metadata == b.metadata
    assert "days_without_weather" not in a.metadata
    t = ev.evaluate({"model": "lag_mean_temp", "lags": [48, 168], "temp_coef": 0.0}, "test")
    assert t.score == pytest.approx(a.score)
    assert t.metadata["days_without_weather"] == 0
    assert t.metadata["days"] == a.metadata["days"]


def test_missing_weather_drops_days_only_for_temp_model():
    ev, w = make_eval()
    gap = datetime(2024, 4, 10, 15, tzinfo=UTC)  # inside local day 2024-04-10
    del w[gap]
    ev2 = EnergyLoadEvaluator(*_args(), weather=w, min_days=10)
    base = ev2.evaluate({"model": "lag_mean", "lags": [168]}, "test")
    sn = ev2.evaluate({"model": "seasonal_naive"}, "test")
    r = ev2.evaluate({"model": "lag_mean_temp", "lags": [168], "temp_coef": 0.5}, "test")
    # W[t] missing drops 04-10; W[t-168] missing drops 04-17 as well
    assert r.metadata["days_without_weather"] == 2
    assert r.n == base.n - 2
    assert "2024-04-10" in base.metadata["days"] and "2024-04-10" not in r.metadata["days"]
    assert r.metadata["dropped_days"] == base.metadata["dropped_days"] + 2
    assert "days_without_weather" not in sn.metadata
    assert len(r.metadata["loss_diff"]) == r.n


def _args():
    d, o, _ = series()
    return d, o, TRAIN, TEST


def test_different_lags_may_have_different_subsets():
    ev, w = make_eval()
    del w[datetime(2024, 4, 10, 15, tzinfo=UTC)]
    ev = EnergyLoadEvaluator(*_args(), weather=w, min_days=10)
    r1 = ev.evaluate({"model": "lag_mean_temp", "lags": [168]}, "test")
    r2 = ev.evaluate({"model": "lag_mean_temp", "lags": [48]}, "test")
    assert r1.metadata["days"] != r2.metadata["days"]


def test_temp_coef_changes_forecast_exactly():
    d, o, w = series()
    # weather perfectly explains demand deviation: demand = base + 2*W
    d = {t: 1000 + 2 * w[t] for t in d}
    o = {t: d[t] + 10 for t in d}
    ev = EnergyLoadEvaluator(d, o, TRAIN, TEST, min_days=10, weather=w)
    good = ev.evaluate({"model": "lag_mean_temp", "lags": [168], "temp_coef": 2.0}, "test")
    zero = ev.evaluate({"model": "lag_mean_temp", "lags": [168], "temp_coef": 0.0}, "test")
    assert good.score == pytest.approx(1.0)  # zero error vs biased operator
    assert good.score > zero.score


def test_split_summary_weather_key():
    plain, _ = make_eval(weather=False)
    assert "weather_lead_days" not in split_summary(plain)
    ev, _ = make_eval(weather_lead_days=3)
    assert split_summary(ev)["weather_lead_days"] == 3


# ---- review fixes ------------------------------------------------------------------------


def test_runtime_lead_guard_other_timezone():
    # Pacific/Kwajalein passes the 2023-2025 scan with lead 2 but leaks on 1969-10-01.
    d, o, _ = series(24)
    ev = EnergyLoadEvaluator(
        d,
        o,
        (date(1969, 9, 1), date(1969, 9, 20)),
        (date(1969, 10, 1), date(1969, 10, 2)),
        weather={},
        tz="Pacific/Kwajalein",
        min_days=1,
    )
    with pytest.raises(WeatherLeakError):
        ev.evaluate({"model": "lag_mean_temp", "lags": [168]}, "test")


def test_temp_model_rejects_lag_below_per_hour_floor():
    ev, _ = make_eval(min_lag_hours=24)
    with pytest.raises(ValueError):
        ev.evaluate({"model": "lag_mean_temp", "lags": [24]}, "test")
    # other models keep the configured floor behaviour
    assert ev.evaluate({"model": "lag_mean", "lags": [24]}, "test").n >= 0


def test_parse_rejects_offsets_z_and_non_whole_hours():
    for bad in ("2024-03-01T00:00-08:00", "2024-03-01T00:00+00:00", "2024-03-01T00:00Z"):
        with pytest.raises(ValueError):
            parse_previous_runs(payload(times=[bad, "2024-03-01T01:00"]))
    for bad in ("2024-03-01T00:30", "2024-03-01T00:00:30", "2024-03-01T00:00:00.5"):
        with pytest.raises(ValueError):
            parse_previous_runs(payload(times=[bad, "2024-03-01T01:00"]))


def test_population_weighted_no_overflow():
    t = h(2024, 3, 1)
    pts = {"a": {t: 10.0}, "b": {t: 20.0}}
    out = population_weighted(pts, {"a": 1e308, "b": 1e308})
    assert out[t] == pytest.approx(15.0)


def test_naive_weather_keys_raise_at_construction():
    d, o, _ = series(24 * 10)
    with pytest.raises(ValueError):
        EnergyLoadEvaluator(d, o, TRAIN, TEST, weather={datetime(2024, 4, 1): 15.0})
