"""Tests for the opt-in bias-corrected operator forecast baseline (BCDF)."""

import math
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from sandfish.kits.energy.evaluator import EnergyLoadEvaluator, split_summary

UTC = timezone.utc
LA = ZoneInfo("America/Los_Angeles")
LAG48 = {"model": "lag_mean", "lags": [48]}
FAR = (date(2030, 1, 1), date(2030, 1, 2))  # unused second split


def hours(start, end):
    out = []
    t = start
    while t < end:
        out.append(t)
        t += timedelta(hours=1)
    return out


def base(i):
    return (
        20000
        + 3000 * math.sin(2 * math.pi * (i % 24) / 24)
        + 400 * math.sin(i / 17.0)
        + 40 * ((i * 37) % 11)
    )


def noisy_df(i):
    return base(i) + 150 * math.sin(i * 1.3)


def make_series(start, end, bias=None):
    """Return (demand, df). bias=None gives a drifting, hour-varying bias."""
    demand, df = {}, {}
    for i, t in enumerate(hours(start, end)):
        f = noisy_df(i)
        df[t] = f
        demand[t] = f + bias if bias is not None else base(i) + 300 * math.sin(i / 5.0)
    return demand, df


S0 = datetime(2024, 1, 1, tzinfo=UTC)
S1 = datetime(2024, 7, 1, tzinfo=UTC)
TRAIN = (date(2024, 1, 20), date(2024, 2, 29))
TEST = (date(2024, 4, 1), date(2024, 5, 31))


def build(demand, df, train=TRAIN, test=TEST, **kw):
    kw.setdefault("min_days", 1)
    return EnergyLoadEvaluator(demand, df, train, test, **kw)


def local_day_hours(day):
    start = datetime(day.year, day.month, day.day, tzinfo=LA).astimezone(UTC)
    nxt = day + timedelta(days=1)
    end = datetime(nxt.year, nxt.month, nxt.day, tzinfo=LA).astimezone(UTC)
    return hours(start, end)


def ref_bcdf(demand, df, t, window=7, lag=2):
    diffs = []
    for k in range(lag, lag + window):
        tk = t - timedelta(hours=24 * k)
        d, f = demand.get(tk), df.get(tk)
        if d is not None and f is not None and math.isfinite(d) and math.isfinite(f):
            diffs.append(d - f)
    if len(diffs) < max(1, window - 2):
        return None
    return df[t] + sum(diffs) / len(diffs)


def test_constant_bias_is_corrected_exactly():
    demand, df = make_series(S0, S1, bias=500.0)
    res = build(demand, df, bias_baseline=True).evaluate(LAG48, "test")
    md = res.metadata
    assert res.n > 0
    assert md["skill_bias_corrected_df_vs_operator"] == pytest.approx(1.0, abs=1e-9)
    assert md["days_without_bias_baseline"] == 0
    # BCDF error is ~0 here, so the model-vs-BCDF skill is hugely negative or None.
    v = md["skill_vs_bias_corrected_df"]
    assert v is None or v < -1e6


def test_warmup_days_dropped_and_counted():
    demand, df = make_series(S0, S1, bias=500.0)
    ev = build(
        demand,
        df,
        train=(date(2024, 1, 1), date(2024, 2, 29)),
        bias_baseline=True,
        bias_window_days=14,
    )
    res = ev.evaluate(LAG48, "train")
    # Seasonal naive (168h) already drops Jan 1-7; the 14-day window needs k up to 15
    # for 12 observations: first fully available local day is Jan 14.
    assert res.metadata["days"][0] == "2024-01-14"
    assert res.metadata["days_without_bias_baseline"] == 6  # Jan 8..13
    plain = build(demand, df, train=(date(2024, 1, 1), date(2024, 2, 29)))
    base_res = plain.evaluate(LAG48, "train")
    assert base_res.metadata["days"][0] == "2024-01-08"
    assert res.metadata["dropped_days"] == base_res.metadata["dropped_days"] + 6


def test_hand_computed_formula_matches_all_scored_days():
    demand, df = make_series(S0, S1)
    res = build(demand, df, bias_baseline=True).evaluate(LAG48, "test")
    md = res.metadata
    s_m = s_d = s_s = s_b = 0.0
    expected_days = []
    day = TEST[0]
    while day <= TEST[1]:
        hs = local_day_hours(day)
        if len(hs) == 24:
            expected_days.append(day.isoformat())
            for t in hs:
                a = demand[t]
                bc = ref_bcdf(demand, df, t)
                assert bc is not None
                s_m += abs(a - demand[t - timedelta(hours=48)]) / 24
                s_d += abs(a - df[t]) / 24
                s_s += abs(a - demand[t - timedelta(hours=168)]) / 24
                s_b += abs(a - bc) / 24
        day += timedelta(days=1)
    assert md["days"] == expected_days
    assert md["skill_vs_bias_corrected_df"] == pytest.approx(1 - s_m / s_b)
    assert md["skill_bias_corrected_df_vs_operator"] == pytest.approx(1 - s_b / s_d)
    assert md["skill_vs_seasonal_naive"] == pytest.approx(1 - s_m / s_s)
    assert md["skill_seasonal_naive_vs_operator"] == pytest.approx(1 - s_s / s_d)
    assert res.score == pytest.approx(1 - s_m / s_d)


def test_no_data_after_split_end_is_used():
    demand, df = make_series(S0, S1)
    clean = build(demand, df, bias_baseline=True).evaluate(LAG48, "test")
    last = local_day_hours(TEST[1])[-1]
    d2, f2 = dict(demand), dict(df)
    for t in list(d2):
        if t > last:
            d2[t] = 1e12
            f2[t] = -1e12
    dirty = build(d2, f2, bias_baseline=True).evaluate(LAG48, "test")
    assert dirty.score == clean.score
    assert dirty.n == clean.n
    assert dirty.metadata == clean.metadata


def test_leak_pin_recent_days_do_not_matter():
    demand, df = make_series(S0, S1)
    clean = build(demand, df, bias_baseline=True).evaluate(LAG48, "test")
    # Pin the lag floor on one target hour: k=1 (t-24h) must never enter BCDF, while
    # k=2 (t-48h) must. Check the evaluator's day loss against a corrupted-k=1 copy.
    t = local_day_hours(TEST[0])[5]
    a = ref_bcdf(demand, df, t)
    d3 = dict(demand)
    d3[t - timedelta(hours=24)] = 1e9
    assert ref_bcdf(d3, df, t) == a
    d3[t - timedelta(hours=48)] = 1e9
    assert ref_bcdf(d3, df, t) != a
    assert clean.metadata["days_without_bias_baseline"] == 0


@pytest.mark.parametrize("removed_days,expect_available", [(2, True), (3, False)])
def test_missing_observation_threshold(removed_days, expect_available):
    demand, df = make_series(S0, S1, bias=300.0)
    for dd in range(18 - removed_days + 1, 19):
        for t in hours(datetime(2024, 1, dd, tzinfo=UTC), datetime(2024, 1, dd + 1, tzinfo=UTC)):
            del df[t]
    ev = build(demand, df, bias_baseline=True)
    res = ev.evaluate(LAG48, "train")
    scored = "2024-01-20" in res.metadata["days"]
    assert scored is expect_available
    if not expect_available:
        assert res.metadata["days_without_bias_baseline"] == 5  # Jan 20..24
        assert "2024-01-25" in res.metadata["days"]
    else:
        assert res.metadata["days_without_bias_baseline"] == 0


def test_window_one_requires_single_observation():
    demand, df = make_series(S0, S1, bias=300.0)
    ev = build(demand, df, bias_baseline=True, bias_window_days=1)
    res = ev.evaluate(LAG48, "train")
    assert res.metadata["days"][0] == "2024-01-20"
    del df[datetime(2024, 1, 22, 12, tzinfo=UTC)]  # k=2 pair for one target hour
    res2 = build(demand, df, bias_baseline=True, bias_window_days=1).evaluate(LAG48, "train")
    assert res2.metadata["days_without_bias_baseline"] == 1


def test_dst_days_dropped_and_neighbours_use_utc_arithmetic():
    demand, df = make_series(S0, S1)
    train = (date(2024, 3, 1), date(2024, 3, 20))
    res = build(demand, df, train=train, bias_baseline=True).evaluate(LAG48, "train")
    days = res.metadata["days"]
    assert "2024-03-10" not in days
    for d in ("2024-03-09", "2024-03-11", "2024-03-12", "2024-03-13"):
        assert d in days
    assert res.metadata["dropped_days"] == 1
    assert res.metadata["days_without_bias_baseline"] == 0
    # Hand check the day after spring-forward against UTC arithmetic.
    base_res = build(demand, df, train=train).evaluate(LAG48, "train")
    s_m = s_b = 0.0
    for t in local_day_hours(date(2024, 3, 11)):
        bc = ref_bcdf(demand, df, t)
        s_m += abs(demand[t] - demand[t - timedelta(hours=48)]) / 24
        s_b += abs(demand[t] - bc) / 24
    idx = days.index("2024-03-11")
    assert res.metadata["loss_diff"][idx] == base_res.metadata["loss_diff"][idx]
    assert s_b > 0 and s_m > 0


def test_fall_back_day_dropped():
    start = datetime(2024, 10, 1, tzinfo=UTC)
    demand, df = make_series(start, datetime(2024, 12, 1, tzinfo=UTC))
    train = (date(2024, 10, 20), date(2024, 11, 20))
    res = build(demand, df, train=train, test=FAR, bias_baseline=True).evaluate(LAG48, "train")
    days = res.metadata["days"]
    assert "2024-11-03" not in days
    assert {"2024-11-02", "2024-11-04", "2024-11-05"} <= set(days)
    assert res.metadata["dropped_days"] == 1


def test_flag_off_is_identical_to_default():
    demand, df = make_series(S0, S1)
    weather = {t: 10 + 5 * math.sin(i / 24) for i, t in enumerate(hours(S0, S1))}
    plain = build(demand, df, weather=weather)
    off = build(demand, df, weather=weather, bias_baseline=False, bias_window_days=3)
    variants = [
        {"model": "seasonal_naive"},
        {"model": "lag_mean", "lags": [48, 72]},
        {"model": "lag_mean_temp", "lags": [48, 72], "temp_coef": 20.0},
    ]
    for v in variants:
        for split in ("train", "test"):
            a, b = plain.evaluate(v, split), off.evaluate(v, split)
            assert a.score == b.score and a.n == b.n
            assert a.metadata == b.metadata
            assert list(a.metadata) == list(b.metadata)
            assert "days_without_bias_baseline" not in b.metadata
    assert split_summary(off) == split_summary(plain)


def test_validation_errors():
    demand, df = make_series(S0, S1)
    bad = [
        {"bias_baseline": 1},
        {"bias_baseline": "yes"},
        {"bias_baseline": None},
        {"bias_baseline": True, "bias_window_days": 0},
        {"bias_baseline": True, "bias_window_days": True},
        {"bias_baseline": True, "bias_window_days": 2.5},
        {"bias_baseline": True, "bias_window_days": 366},
        {"bias_baseline": True, "bias_window_days": 10**12},
        {"bias_baseline": True, "bias_lag_days": 1},
        {"bias_baseline": True, "bias_lag_days": True},
        {"bias_baseline": True, "bias_lag_days": 2.0},
        {"bias_baseline": True, "min_lag_hours": 60},
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            build(demand, df, **kw)
    build(demand, df, bias_baseline=True, min_lag_hours=48)
    build(demand, df, bias_baseline=True, bias_lag_days=3, min_lag_hours=60)
    build(demand, df, bias_baseline=False, bias_window_days=0)


def test_split_summary_both_ways():
    demand, df = make_series(S0, S1)
    off = split_summary(build(demand, df))
    assert "bias_baseline" not in off
    on = split_summary(build(demand, df, bias_baseline=True, bias_window_days=5, bias_lag_days=3))
    assert on["bias_baseline"] == {"window_days": 5, "lag_days": 3}
    assert {k: v for k, v in on.items() if k != "bias_baseline"} == off


def test_all_models_carry_bias_keys():
    demand, df = make_series(S0, S1)
    weather = {t: 10.0 + (i % 24) for i, t in enumerate(hours(S0, S1))}
    ev = build(demand, df, weather=weather, bias_baseline=True)
    for v in (
        {"model": "seasonal_naive"},
        {"model": "lag_mean", "lags": [48]},
        {"model": "lag_mean_temp", "lags": [48], "temp_coef": 1.0},
    ):
        md = ev.evaluate(v, "test").metadata
        for key in (
            "skill_vs_bias_corrected_df",
            "skill_bias_corrected_df_vs_operator",
            "days_without_bias_baseline",
        ):
            assert key in md


def test_skill_none_when_bcdf_is_exact():
    demand, df = make_series(S0, S1, bias=500.0)
    md = build(demand, df, bias_baseline=True).evaluate(
        {"model": "lag_mean", "lags": [24 * 7]}, "test"
    ).metadata
    # Model error is nonzero while BCDF error is ~0: skill is hugely negative or None.
    v = md["skill_vs_bias_corrected_df"]
    assert v is None or v < -1e6


def test_lag_mean_temp_with_bias_baseline_disjoint_counts():
    demand, df = make_series(S0, S1, bias=300.0)
    for dd in (16, 17, 18):
        for t in hours(datetime(2024, 1, dd, tzinfo=UTC), datetime(2024, 1, dd + 1, tzinfo=UTC)):
            del df[t]
    weather = {t: 5.0 + (i % 24) for i, t in enumerate(hours(S0, S1))}
    variant = {"model": "lag_mean_temp", "lags": [48, 72], "temp_coef": 10.0}
    full = build(demand, df, weather=weather, bias_baseline=True).evaluate(variant, "train")
    assert full.metadata["days_without_weather"] == 0
    assert full.metadata["days_without_bias_baseline"] == 5  # Jan 20..24
    holes = dict(weather)
    for t in local_day_hours(date(2024, 1, 20)):
        del holes[t]
    res = build(demand, df, weather=holes, bias_baseline=True).evaluate(variant, "train")
    md = res.metadata
    assert md["days_without_weather"] == 3  # Jan 20 plus the two days lagging onto it
    assert md["days_without_bias_baseline"] == 2  # Jan 21 and Jan 24 only
    assert "2024-01-20" not in md["days"]
    assert md["dropped_days"] >= md["days_without_weather"] + md["days_without_bias_baseline"]
    assert "skill_vs_bias_corrected_df" in md


def test_lag_mean_temp_like_for_like_on_bias_subset():
    demand, df = make_series(S0, S1)
    weather = {t: 5.0 + (i % 24) for i, t in enumerate(hours(S0, S1))}
    variant = {"model": "lag_mean_temp", "lags": [48], "temp_coef": 5.0}
    on = build(demand, df, weather=weather, bias_baseline=True, bias_window_days=14)
    off = build(demand, df, weather=weather)
    r_on = on.evaluate(variant, "train").metadata
    r_off = off.evaluate(variant, "train").metadata
    keep = [i for i, d in enumerate(r_off["days"]) if d in set(r_on["days"])]
    assert [r_off["days"][i] for i in keep] == r_on["days"]
    assert [r_off["loss_diff"][i] for i in keep] == r_on["loss_diff"]


def test_window_upper_bound():
    demand, df = make_series(S0, S1)
    build(demand, df, bias_baseline=True, bias_window_days=365)
    with pytest.raises(ValueError):
        build(demand, df, bias_baseline=True, bias_window_days=366)
    build(demand, df, bias_baseline=False, bias_window_days=10**12)


def local_clock_bcdf(demand, df, t, window=7, lag=2):
    """Wrong-on-purpose BCDF using local wall-clock day steps."""
    diffs = []
    for k in range(lag, lag + window):
        tk = (t.astimezone(LA).replace(tzinfo=None) - timedelta(days=k)).replace(tzinfo=LA)
        tk = tk.astimezone(UTC)
        if tk in demand and tk in df:
            diffs.append(demand[tk] - df[tk])
    return df[t] + sum(diffs) / len(diffs)


@pytest.mark.parametrize(
    "start,train,day_strs",
    [
        (
            datetime(2024, 2, 20, tzinfo=UTC),
            (date(2024, 3, 11), date(2024, 3, 13)),
            ["2024-03-11", "2024-03-12", "2024-03-13"],
        ),
        (
            datetime(2024, 10, 15, tzinfo=UTC),
            (date(2024, 11, 4), date(2024, 11, 6)),
            ["2024-11-04", "2024-11-05", "2024-11-06"],
        ),
    ],
)
def test_after_dst_transition_bcdf_uses_utc_arithmetic(start, train, day_strs):
    end = start + timedelta(days=60)
    demand, df = make_series(start, end)
    for i, t in enumerate(hours(start, end)):  # residual differs at every UTC hour
        demand[t] = df[t] + 100 + 37 * ((i * 7919) % 101)
    res = build(demand, df, train=train, test=FAR, bias_baseline=True).evaluate(LAG48, "train")
    md = res.metadata
    assert md["days"] == day_strs
    s_b = s_d = s_local = 0.0
    for ds in day_strs:
        for t in local_day_hours(date.fromisoformat(ds)):
            s_b += abs(demand[t] - ref_bcdf(demand, df, t)) / 24
            s_local += abs(demand[t] - local_clock_bcdf(demand, df, t)) / 24
            s_d += abs(demand[t] - df[t]) / 24
    assert s_b != pytest.approx(s_local)  # the test can tell the two apart
    assert md["skill_bias_corrected_df_vs_operator"] == pytest.approx(1 - s_b / s_d)
    assert md["skill_bias_corrected_df_vs_operator"] != pytest.approx(1 - s_local / s_d)


def test_weather_subset_drops_days_and_bias_keys_on_reduced_set():
    demand, df = make_series(S0, S1)
    for dd in (27, 28, 29):  # thins the BCDF window for the early February days
        for t in hours(datetime(2024, 1, dd, tzinfo=UTC), datetime(2024, 1, dd + 1, tzinfo=UTC)):
            del df[t]
    weather = {t: 5.0 + (i % 24) for i, t in enumerate(hours(S0, S1))}
    for t in local_day_hours(date(2024, 2, 8)):
        del weather[t]
    variant = {"model": "lag_mean_temp", "lags": [48], "temp_coef": 5.0}
    train = (date(2024, 2, 1), date(2024, 2, 12))
    md = build(demand, df, train=train, weather=weather, bias_baseline=True)
    md = md.evaluate(variant, "train").metadata
    off = build(demand, df, train=train, weather=weather).evaluate(variant, "train").metadata
    assert "2024-02-08" not in md["days"]
    assert md["days_without_weather"] >= 1
    assert md["days_without_weather"] == off["days_without_weather"]
    assert md["days_without_bias_baseline"] >= 1
    assert md["dropped_days"] == md["days_without_weather"] + md["days_without_bias_baseline"]
    keep = [i for i, d in enumerate(off["days"]) if d in set(md["days"])]
    assert len(keep) == len(md["days"]) < len(off["days"])
    assert [off["loss_diff"][i] for i in keep] == md["loss_diff"]
    s_b = s_d = 0.0
    for ds in md["days"]:
        for t in local_day_hours(date.fromisoformat(ds)):
            s_b += abs(demand[t] - ref_bcdf(demand, df, t)) / 24
            s_d += abs(demand[t] - df[t]) / 24
    assert md["skill_bias_corrected_df_vs_operator"] == pytest.approx(1 - s_b / s_d)
