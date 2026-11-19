"""Tests for sandfish.kits.energy.residual (synthetic, offline)."""

from __future__ import annotations

import math
import random
import statistics
from datetime import date, datetime, timedelta, timezone

import pytest

from sandfish.kits.energy import residual as res
from sandfish.kits.energy.evaluator import EnergyLoadEvaluator
from sandfish.kits.energy.residual import (
    ResidualEvaluator,
    default_residual_variants,
    validate_variant,
)
from sandfish.kits.energy.weather import WeatherLeakError

UTC = timezone.utc
H = timedelta(hours=1)


def make_series(start, end, seed=1, *, bias=None, temp_slope=None):
    """Hourly (D, DF, W) dicts on [start, end)."""
    rng = random.Random(seed)
    d, f, w = {}, {}, {}
    s = start
    day_level = 0.0
    while s < end:
        if s.hour == 0:
            day_level = rng.uniform(-6, 6)
        temp = 15 + day_level + 6 * math.sin(2 * math.pi * s.hour / 24)
        w[s] = temp
        base = 20000 + 3000 * math.sin(2 * math.pi * (s.hour - 6) / 24) + rng.uniform(-50, 50)
        if bias is not None:
            err = bias
        elif temp_slope is not None:
            err = temp_slope * temp + rng.uniform(-5, 5)
        else:
            err = 80 * math.sin(s.hour) + 40 * math.cos(s.toordinal()) + rng.uniform(-60, 60)
        d[s] = base
        f[s] = base - err
        s += H
    return d, f, w


START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 7, 1, tzinfo=UTC)
T = datetime(2024, 6, 10, 15, tzinfo=UTC)


@pytest.fixture(scope="module")
def data():
    return make_series(START, END)


@pytest.fixture(scope="module")
def ev(data):
    return ResidualEvaluator(data[0], data[1], data[2])


def hand_agg(vals, how):
    return statistics.median(vals) if how == "median" else sum(vals) / len(vals)


def hand_r(d, f, t, *, L=2, windows, mix=None, agg="mean", lam=1.0):
    e = {s: d[s] - f[s] for s in d if s in f}

    def hour(wn):
        v = [e[t - timedelta(hours=24 * k)] for k in range(L, L + wn)
             if t - timedelta(hours=24 * k) in e]
        return hand_agg(v, agg) if len(v) >= max(1, wn - 2) else None

    def level(wn):
        out = []
        for k in range(L, L + wn):
            v = [e[t - timedelta(hours=24 * k + j)] for j in range(24)
                 if t - timedelta(hours=24 * k + j) in e]
            if len(v) >= 20:
                out.append(sum(v) / len(v))
        return hand_agg(out, agg) if len(out) >= max(1, wn - 2) else None

    def b(wn):
        if lam == 1.0:
            return hour(wn)
        if lam == 0.0:
            return level(wn)
        bh, bl = hour(wn), level(wn)
        return None if bh is None or bl is None else lam * bh + (1 - lam) * bl

    if len(windows) == 1:
        return b(windows[0])
    b1, b2 = b(windows[0]), b(windows[1])
    return b2 + mix * (b1 - b2)


# ---- validation ----------------------------------------------------------------------
def test_validate_fills_defaults():
    v = validate_variant({"windows": (7,)})
    assert v == {
        "lag_days": 2, "windows": [7], "mix": None, "agg": "mean", "hour_weight": 1.0,
        "temp": None,
    }


def test_validate_returns_copy_and_does_not_mutate():
    raw = {"windows": [7], "temp": {"beta": 100}}
    snap = {"windows": [7], "temp": {"beta": 100}}
    out = validate_variant(raw)
    out["temp"]["beta"] = 5
    out["windows"].append(9)
    assert raw == snap


@pytest.mark.parametrize(
    "bad",
    [
        "x", None, [], {}, {"windows": [7], "foo": 1},
        {"windows": [7], "lag_days": 1}, {"windows": [7], "lag_days": 8},
        {"windows": [7], "lag_days": True}, {"windows": [7], "lag_days": 2.0},
        {"windows": []}, {"windows": [1, 2, 3]}, {"windows": [0]}, {"windows": [121]},
        {"windows": [True]}, {"windows": [7.0]}, {"windows": 7}, {"windows": [7, 7], "mix": 0.5},
        {"windows": [9, 7], "mix": 0.5}, {"windows": [7, 28]},
        {"windows": [7], "mix": 0.5}, {"windows": [7, 28], "mix": 2.5},
        {"windows": [7, 28], "mix": -1.5}, {"windows": [7, 28], "mix": True},
        {"windows": [7, 28], "mix": float("nan")},
        {"windows": [7], "agg": "max"}, {"windows": [7], "hour_weight": 1.1},
        {"windows": [7], "hour_weight": -0.1}, {"windows": [7], "hour_weight": True},
        {"windows": [7], "temp": {}}, {"windows": [7], "temp": {"beta": 1, "shrink": 1}},
        {"windows": [7], "temp": {"beta": True}}, {"windows": [7], "temp": {"beta": math.inf}},
        {"windows": [7], "temp": {"gamma": 1}}, {"windows": [7], "temp": 5},
        {"windows": [28], "temp": {"shrink": 0}}, {"windows": [28], "temp": {"shrink": 1.5}},
        {"windows": [13], "temp": {"shrink": 1.0}},
        {"windows": [14, 28], "mix": 0.5, "temp": {"shrink": 1.0}},
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        validate_variant(bad)


def test_validate_accepts_boundaries():
    validate_variant({"windows": [1, 120], "mix": -1.0, "lag_days": 7, "hour_weight": 0})
    validate_variant({"windows": [1, 2], "mix": 2})
    validate_variant({"windows": [14], "temp": {"shrink": 1}})
    validate_variant({"windows": [7], "temp": {"beta": -3}})
    validate_variant({"windows": [7], "temp": None})


def test_validate_lag_floor():
    with pytest.raises(ValueError):
        validate_variant({"windows": [7]}, min_lag_hours=49)
    validate_variant({"windows": [7]}, min_lag_hours=48)
    with pytest.raises(ValueError):
        ResidualEvaluator({}, {}, min_lag_hours=60).correction_at({"windows": [7]}, T)


# ---- search space ---------------------------------------------------------------------
def test_default_variants_shape():
    vs = default_residual_variants()
    assert len(vs) == 55
    assert vs[0] == {"windows": [3]}
    assert vs[2] == {"windows": [7]}
    assert vs[8] == {"lag_days": 3, "windows": [7]}
    assert vs[10] == {"agg": "median", "windows": [7]}
    assert vs[13] == {"windows": [3, 28], "mix": 0.25}
    assert vs[32] == {"windows": [14, 56], "mix": 1.5}
    assert vs[33] == {"hour_weight": 0.0, "windows": [7]}
    assert vs[39] == {"windows": [7], "temp": {"beta": -200}}
    assert vs[49] == {"windows": [28], "temp": {"shrink": 0.5}}
    assert vs[-1] == {"windows": [90], "temp": {"shrink": 1.0}}
    assert len({repr(sorted(v.items())) for v in vs}) == 55
    for v in vs:
        validate_variant(v)


def test_default_variants_fresh_each_call():
    a = default_residual_variants()
    a[0]["windows"].append(99)
    a[-1]["temp"]["shrink"] = 0.1
    a.pop()
    b = default_residual_variants()
    assert len(b) == 55 and b[0] == {"windows": [3]} and b[-1]["temp"] == {"shrink": 1.0}


# ---- BCDF reproduction ----------------------------------------------------------------
def test_window7_reproduces_bcdf_with_holes():
    d, f, _ = make_series(START, datetime(2024, 3, 1, tzinfo=UTC), seed=3)
    rng = random.Random(9)
    for s in list(d):
        if rng.random() < 0.12:
            del d[s]
        if rng.random() < 0.12:
            f.pop(s, None)
    r = ResidualEvaluator(d, f)
    base = EnergyLoadEvaluator(
        d, f, (date(2000, 1, 1), date(2000, 1, 2)), (date(2000, 1, 3), date(2000, 1, 4)),
        bias_baseline=True,
    )
    n_avail = n_none = 0
    for s in sorted(f):
        if s < START + timedelta(days=5):
            continue
        want = base._bias_forecast(s, f[s])
        got = r.correction_at({"windows": [7]}, s)
        if want is None:
            assert got is None
            n_none += 1
        else:
            assert got == pytest.approx(want - f[s], abs=1e-9)
            n_avail += 1
    assert n_avail > 100 and n_none > 0


def test_scan_baseline_matches_evaluator_day_errors(data, ev):
    d, f, _ = data
    base = EnergyLoadEvaluator(
        d, f, (date(2000, 1, 1), date(2000, 1, 2)), (date(2000, 1, 3), date(2000, 1, 4)),
        bias_baseline=True,
    )
    days, counts = ev.scan({"windows": [7]}, date(2024, 2, 1), date(2024, 2, 20))
    assert counts["scored"] == 20 and len(days) == 20
    for day, (m, b, o) in days.items():
        row = base._score_day(base._day_hours(day), [48])
        assert m == pytest.approx(row[3], rel=1e-9)
        assert b == pytest.approx(row[3], rel=1e-9)
        assert o == pytest.approx(row[1], rel=1e-9)


# ---- hand recomputation ---------------------------------------------------------------
HAND_CASES = [
    {"windows": [7]},
    {"windows": [14]},
    {"lag_days": 3, "windows": [7]},
    {"agg": "median", "windows": [10]},
    {"agg": "median", "windows": [28]},
    {"windows": [7, 28], "mix": 0.5},
    {"windows": [3, 28], "mix": 1.5},
    {"windows": [7, 56], "mix": -0.5},
    {"hour_weight": 0.0, "windows": [7]},
    {"hour_weight": 0.5, "windows": [14]},
    {"hour_weight": 0.25, "agg": "median", "windows": [7, 14], "mix": 0.3},
]


@pytest.mark.parametrize("var", HAND_CASES)
def test_core_matches_hand_computation(data, ev, var):
    d, f, _ = data
    kw = {
        "L": var.get("lag_days", 2), "windows": var["windows"], "mix": var.get("mix"),
        "agg": var.get("agg", "mean"), "lam": var.get("hour_weight", 1.0),
    }
    for t in (T, T + 7 * H, T + 30 * H):
        assert ev.correction_at(var, t) == pytest.approx(hand_r(d, f, t, **kw), abs=1e-9)


def test_median_even_count_averages_middle():
    t = T
    d = {t - timedelta(hours=24 * k): 100.0 + v for k, v in zip((2, 3, 4, 5), (1, 2, 10, 50))}
    f = {s: 100.0 for s in d}
    r = ResidualEvaluator(d, f)
    assert r.correction_at({"agg": "median", "windows": [4]}, t) == 6.0
    assert r.correction_at({"windows": [4]}, t) == 15.75


def test_beta_matches_hand(data, ev):
    d, f, w = data
    e = {s: d[s] - f[s] for s in d}
    var = {"windows": [7], "temp": {"beta": 120}}
    xs = [w[T - timedelta(hours=24 * k)] for k in range(2, 9)]
    want = hand_r(d, f, T, windows=[7]) + 120 * (w[T] - sum(xs) / 7)
    assert e and ev.correction_at(var, T) == pytest.approx(want)


def test_beta_uses_wmax_for_two_windows(data, ev):
    d, f, w = data
    var = {"windows": [7, 28], "mix": 0.5, "temp": {"beta": 50}}
    xs = [w[T - timedelta(hours=24 * k)] for k in range(2, 30)]
    want = hand_r(d, f, T, windows=[7, 28], mix=0.5) + 50 * (w[T] - sum(xs) / 28)
    assert ev.correction_at(var, T) == pytest.approx(want)


def test_shrink_matches_hand(data, ev):
    d, f, w = data
    pts = [(w[s], d[s] - f[s]) for s in (T - timedelta(hours=24 * k) for k in range(2, 30))]
    xb = sum(p[0] for p in pts) / len(pts)
    yb = sum(p[1] for p in pts) / len(pts)
    sxx = sum((x - xb) ** 2 for x, _ in pts)
    sxy = sum((x - xb) * (y - yb) for x, y in pts)
    want = hand_r(d, f, T, windows=[28]) + 0.7 * (sxy / sxx) * (w[T] - xb)
    got = ev.correction_at({"windows": [28], "temp": {"shrink": 0.7}}, T)
    assert got == pytest.approx(want)


def test_shrink_zero_variance_gives_zero_slope():
    d, f, w = make_series(START, datetime(2024, 3, 1, tzinfo=UTC), seed=4)
    t = datetime(2024, 2, 20, 5, tzinfo=UTC)
    for k in range(2, 16):
        w[t - timedelta(hours=24 * k)] = 10.0
    r = ResidualEvaluator(d, f, w)
    got = r.correction_at({"windows": [14], "temp": {"shrink": 1.0}}, t)
    assert got == pytest.approx(hand_r(d, f, t, windows=[14]))


# ---- leak tests -----------------------------------------------------------------------
LEAK_VARIANTS = [
    {"windows": [7]},
    {"lag_days": 3, "windows": [14]},
    {"windows": [7, 28], "mix": 1.25},
    {"hour_weight": 0.0, "windows": [28]},
    {"hour_weight": 0.5, "windows": [7]},
    {"agg": "median", "windows": [14]},
    {"windows": [7], "temp": {"beta": 200}},
    {"windows": [28], "temp": {"shrink": 1.0}},
    {"windows": [56], "temp": {"shrink": 0.5}},
]


@pytest.mark.parametrize("var", LEAK_VARIANTS)
def test_no_future_data_is_read(data, var):
    d, f, w = data
    lag = var.get("lag_days", 2)
    t = T
    base = ResidualEvaluator(d, f, w).correction_at(var, t)
    assert base is not None
    d2, f2, w2 = dict(d), dict(f), dict(w)
    cut = t - timedelta(hours=24 * lag)
    for s in d:
        if s > cut:
            d2[s], f2[s] = 1e9, -1e9
        if s > t:
            w2[s] = 1e6
    assert ResidualEvaluator(d2, f2, w2).correction_at(var, t) == base


# ---- thresholds -----------------------------------------------------------------------
def full(days=40):
    return make_series(START, START + timedelta(days=days), seed=5)


def drop_f(f, t, ks):
    return {s: v for s, v in f.items() if s not in {t - timedelta(hours=24 * k) for k in ks}}


def test_same_hour_threshold():
    d, f, _ = full()
    t = START + timedelta(days=30, hours=7)
    ok = ResidualEvaluator(d, drop_f(f, t, (2, 3)))
    assert ok.correction_at({"windows": [7]}, t) is not None  # 5 of 7
    bad = ResidualEvaluator(d, drop_f(f, t, (2, 3, 4)))
    assert bad.correction_at({"windows": [7]}, t) is None  # 4 of 7
    one = ResidualEvaluator(d, drop_f(f, t, (2, 3)))
    assert one.correction_at({"windows": [1]}, t) is None  # needs 1 of 1
    assert one.correction_at({"windows": [2]}, t) is None
    assert one.correction_at({"windows": [3]}, t) is not None  # 1 of 3 (needs max(1,1))


def test_two_windows_unavailable_if_either_is():
    d, f, _ = full()
    t = START + timedelta(days=30, hours=7)
    var = {"windows": [3, 7], "mix": 0.5}
    assert ResidualEvaluator(d, drop_f(f, t, (2, 3))).correction_at(var, t) is not None
    assert ResidualEvaluator(d, drop_f(f, t, (2, 3, 4))).correction_at(var, t) is None
    assert ResidualEvaluator(d, drop_f(f, t, (5, 6, 7))).correction_at(var, t) is None


def drop_block(f, t, k, n):
    gone = {t - timedelta(hours=24 * k + j) for j in range(n)}
    return {s: v for s, v in f.items() if s not in gone}


def test_level_block_threshold_20_of_24():
    d, f, _ = full()
    t = START + timedelta(days=30, hours=7)
    var = {"hour_weight": 0.0, "windows": [1]}
    assert ResidualEvaluator(d, drop_block(f, t, 2, 4)).correction_at(var, t) is not None
    assert ResidualEvaluator(d, drop_block(f, t, 2, 5)).correction_at(var, t) is None


def test_level_sample_count_threshold():
    d, f, _ = full()
    t = START + timedelta(days=30, hours=7)
    var = {"hour_weight": 0.0, "windows": [7]}
    g = f
    for k in (2, 3):
        g = drop_block(g, t, k, 5)
    assert ResidualEvaluator(d, g).correction_at(var, t) is not None  # 5 of 7
    g = drop_block(g, t, 4, 5)
    assert ResidualEvaluator(d, g).correction_at(var, t) is None  # 4 of 7


def test_mixed_hour_weight_needs_both():
    d, f, _ = full()
    t = START + timedelta(days=30, hours=7)
    var = {"hour_weight": 0.5, "windows": [1]}
    assert ResidualEvaluator(d, f).correction_at(var, t) is not None
    assert ResidualEvaluator(d, drop_block(f, t, 2, 5)).correction_at(var, t) is None
    # lambda 1 ignores the level part, lambda 0 ignores the hour part
    g = drop_block(f, t, 2, 5)
    assert ResidualEvaluator(d, g).correction_at({"windows": [3]}, t) is not None
    only_hour = {s: v for s, v in f.items() if s != t - 48 * H}
    one = ResidualEvaluator(d, only_hour)
    assert one.correction_at({"hour_weight": 0.0, "windows": [1]}, t) is not None
    assert one.correction_at({"windows": [1]}, t) is None


def drop_w(w, t, ks):
    return {s: v for s, v in w.items() if s not in {t - timedelta(hours=24 * k) for k in ks}}


def test_beta_weather_thresholds():
    d, f, w = full()
    t = START + timedelta(days=30, hours=7)
    var = {"windows": [7], "temp": {"beta": 10}}
    assert ResidualEvaluator(d, f, drop_w(w, t, (2, 3))).correction_at(var, t) is not None
    assert ResidualEvaluator(d, f, drop_w(w, t, (2, 3, 4))).correction_at(var, t) is None
    no_now = {s: v for s, v in w.items() if s != t}
    assert ResidualEvaluator(d, f, no_now).correction_at(var, t) is None


def test_shrink_pair_thresholds():
    d, f, w = full()
    t = START + timedelta(days=30, hours=7)
    var = {"windows": [14], "temp": {"shrink": 1.0}}  # needs max(5, 7) = 7 pairs
    assert ResidualEvaluator(d, f, drop_w(w, t, range(2, 9))).correction_at(var, t) is not None
    assert ResidualEvaluator(d, f, drop_w(w, t, range(2, 10))).correction_at(var, t) is None
    var10 = {"windows": [28], "temp": {"shrink": 1.0}}  # max(5, 14) = 14
    d2, f2, w2 = make_series(START, START + timedelta(days=60), seed=5)
    t2 = START + timedelta(days=55)
    assert ResidualEvaluator(d2, f2, drop_w(w2, t2, range(2, 16))).correction_at(var10, t2)
    assert ResidualEvaluator(d2, f2, drop_w(w2, t2, range(2, 17))).correction_at(var10, t2) is None


def test_shrink_pairs_need_both_e_and_w():
    d, f, w = full()
    t = START + timedelta(days=30, hours=7)
    var = {"windows": [14], "temp": {"shrink": 1.0}}
    g = drop_f(f, t, (2, 3))  # e missing at 2 stamps; core still has 12 of 14
    r = ResidualEvaluator(d, g, drop_w(w, t, range(4, 10)))
    assert r.correction_at(var, t) is None  # 6 pairs < 7
    r = ResidualEvaluator(d, g, drop_w(w, t, range(4, 9)))
    assert r.correction_at(var, t) is not None  # 7 pairs


# ---- scan buckets ---------------------------------------------------------------------
DAY = date(2024, 5, 10)


def day_hours(r, day=DAY):
    return r._base._day_hours(day)


def check_sum(counts, first, last):
    assert set(counts) == {
        "scored", "not_24h", "missing_data", "no_weather", "no_correction", "no_bcdf",
    }
    assert sum(counts.values()) == (last - first).days + 1


def test_scan_scored_and_values(data, ev):
    days, counts = ev.scan({"windows": [14]}, DAY, DAY + timedelta(days=2))
    check_sum(counts, DAY, DAY + timedelta(days=2))
    assert counts["scored"] == 3
    d, f, _ = data
    hrs = day_hours(ev)
    want = sum(
        abs(d[t] - (f[t] + hand_r(d, f, t, windows=[14]))) for t in hrs
    ) / 24
    assert days[DAY][0] == pytest.approx(want)
    assert days[DAY][2] == pytest.approx(sum(abs(d[t] - f[t]) for t in hrs) / 24)


def test_scan_missing_data(data):
    d, f, _ = data
    d = dict(d)
    r0 = ResidualEvaluator(d, f)
    del d[day_hours(r0)[5]]
    r = ResidualEvaluator(d, f)
    days, counts = r.scan({"windows": [7]}, DAY, DAY)
    assert counts["missing_data"] == 1 and not days
    check_sum(counts, DAY, DAY)
    f2 = dict(f)
    f2[day_hours(r0)[9]] = float("nan")
    _, counts = ResidualEvaluator(data[0], f2).scan({"windows": [7]}, DAY, DAY)
    assert counts["missing_data"] == 1


def test_scan_no_weather(data):
    d, f, w = data
    r0 = ResidualEvaluator(d, f, w)
    w2 = dict(w)
    w2[day_hours(r0)[3]] = None
    r = ResidualEvaluator(d, f, w2)
    days, counts = r.scan({"windows": [7], "temp": {"beta": 10}}, DAY, DAY)
    assert counts["no_weather"] == 1 and not days
    _, counts = r.scan({"windows": [7]}, DAY, DAY)
    assert counts["scored"] == 1  # weather ignored without a temp term


def test_scan_no_weather_takes_precedence_over_no_correction(data):
    d, f, w = data
    r0 = ResidualEvaluator(d, f, w)
    t = day_hours(r0)[3]
    w2 = {s: v for s, v in w.items() if s != t}
    g = drop_f(f, t, range(2, 12))
    g = {s: v for s, v in g.items()}
    _, counts = ResidualEvaluator(d, g, w2).scan(
        {"windows": [7], "temp": {"beta": 10}}, DAY, DAY
    )
    assert counts["no_weather"] == 1 and counts["no_correction"] == 0


def test_scan_no_correction_and_no_bcdf(data):
    d, f, _ = data
    r0 = ResidualEvaluator(d, f)
    t = day_hours(r0)[5]
    g = drop_f(f, t, range(2, 9))
    _, counts = ResidualEvaluator(d, g).scan({"windows": [7]}, DAY, DAY)
    assert counts["no_correction"] == 1 and counts["no_bcdf"] == 0
    # window 3 still has k=2, but BCDF (window 7, needs 5 of 7) does not
    g = drop_f(f, t, (3, 4, 5, 6))
    days, counts = ResidualEvaluator(d, g).scan({"windows": [3]}, DAY, DAY)
    assert counts["no_bcdf"] == 1 and counts["no_correction"] == 0 and not days
    check_sum(counts, DAY, DAY)


def test_scan_counts_sum_over_mixed_range(data, ev):
    first, last = date(2024, 1, 1), date(2024, 3, 31)
    days, counts = ev.scan({"windows": [28]}, first, last)
    check_sum(counts, first, last)
    assert counts["scored"] == len(days)
    assert counts["not_24h"] == 1  # 2024-03-10 only
    assert counts["no_correction"] > 0 and counts["missing_data"] >= 0


def test_scan_rejects_bad_range(ev):
    with pytest.raises(ValueError):
        ev.scan({"windows": [7]}, date(2024, 5, 2), date(2024, 5, 1))


# ---- DST ------------------------------------------------------------------------------
def test_dst_days_skipped_neighbours_scored():
    d, f, _ = make_series(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 11, 10, tzinfo=UTC))
    r = ResidualEvaluator(d, f)
    first, last = date(2024, 3, 8), date(2024, 3, 12)
    days, counts = r.scan({"windows": [7]}, first, last)
    assert date(2024, 3, 10) not in days
    assert counts["not_24h"] == 1 and counts["scored"] == 4
    assert date(2024, 3, 11) in days and date(2024, 3, 9) in days
    first, last = date(2024, 11, 1), date(2024, 11, 5)
    days, counts = r.scan({"windows": [7]}, first, last)
    assert date(2024, 11, 3) not in days
    assert counts["not_24h"] == 1 and counts["scored"] == 4
    check_sum(counts, first, last)


def test_day_after_dst_uses_utc_arithmetic():
    d, f, _ = make_series(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 3, 20, tzinfo=UTC))
    r = ResidualEvaluator(d, f)
    t = datetime(2024, 3, 12, 15, tzinfo=UTC)  # 2 days after the spring transition
    want = hand_r(d, f, t, windows=[7])
    assert r.correction_at({"windows": [7]}, t) == pytest.approx(want)
    days, _ = r.scan({"windows": [7]}, date(2024, 3, 11), date(2024, 3, 11))
    assert date(2024, 3, 11) in days


# ---- behaviour on synthetic structure -------------------------------------------------
def test_constant_bias_gives_near_zero_error():
    d, f, _ = make_series(START, datetime(2024, 5, 1, tzinfo=UTC), bias=100.0)
    r = ResidualEvaluator(d, f)
    day = date(2024, 4, 20)
    for v in default_residual_variants():
        if v.get("temp"):
            continue
        days, counts = r.scan(v, day, day)
        assert counts["scored"] == 1, v
        assert days[day][0] < 1e-6, v
        assert days[day][2] == pytest.approx(100.0)


def test_temperature_dependent_error_beta_and_shrink_beat_bcdf():
    d, f, w = make_series(START, datetime(2024, 5, 1, tzinfo=UTC), temp_slope=20.0)
    r = ResidualEvaluator(d, f, w)
    first, last = date(2024, 3, 15), date(2024, 4, 20)
    beta, cb = r.scan({"windows": [7], "temp": {"beta": 20}}, first, last)
    shr, cs = r.scan({"windows": [28], "temp": {"shrink": 1.0}}, first, last)
    assert cb["scored"] > 30 and cs["scored"] > 30
    common = sorted(set(beta) & set(shr))
    m_beta = sum(beta[x][0] for x in common) / len(common)
    m_shr = sum(shr[x][0] for x in common) / len(common)
    m_bcdf = sum(beta[x][1] for x in common) / len(common)
    assert m_beta < 0.5 * m_bcdf
    assert m_shr < 0.5 * m_bcdf
    assert m_shr > m_beta * 0.5  # sanity: the fit is not magically better than the truth


# ---- guards ---------------------------------------------------------------------------
def test_temp_variant_without_weather_raises(data):
    r = ResidualEvaluator(data[0], data[1])
    for v in ({"windows": [7], "temp": {"beta": 1}}, {"windows": [28], "temp": {"shrink": 1}}):
        with pytest.raises(ValueError):
            r.correction_at(v, T)
        with pytest.raises(ValueError):
            r.scan(v, DAY, DAY)


def test_unsafe_weather_lead_rejected_in_constructor(data):
    with pytest.raises(WeatherLeakError):
        ResidualEvaluator(data[0], data[1], data[2], weather_lead_days=1)


def test_runtime_weather_guard(data, monkeypatch):
    r = ResidualEvaluator(*data)
    monkeypatch.setattr(res, "is_lead_usable", lambda *a, **k: False)
    with pytest.raises(WeatherLeakError):
        r.scan({"windows": [7], "temp": {"beta": 10}}, DAY, DAY)
    days, _ = r.scan({"windows": [7]}, DAY, DAY)  # no temp term, no weather guard
    assert DAY in days


def test_runtime_lag_floor_guard(data, monkeypatch):
    r = ResidualEvaluator(*data)
    monkeypatch.setattr(res, "_min_lag_hours", lambda t, tz: 100)
    with pytest.raises(ValueError):
        r.scan({"windows": [7]}, DAY, DAY)


def test_naive_keys_rejected():
    with pytest.raises(ValueError):
        ResidualEvaluator({datetime(2024, 1, 1): 1.0}, {})


def test_non_finite_values_are_missing():
    d, f, _ = full()
    t = START + timedelta(days=30, hours=7)
    f2 = dict(f)
    for k in (2, 3, 4):
        f2[t - timedelta(hours=24 * k)] = float("inf") if k == 2 else "abc"
    assert ResidualEvaluator(d, f2).correction_at({"windows": [7]}, t) is None


def test_variant_not_mutated_by_use(data, ev):
    v = {"windows": [7, 28], "mix": 0.5, "temp": None}
    snap = {"windows": [7, 28], "mix": 0.5, "temp": None}
    ev.correction_at(v, T)
    ev.scan(v, DAY, DAY)
    assert v == snap
    v2 = {"windows": [28], "temp": {"shrink": 0.5}}
    ev.correction_at(v2, T)
    ev.scan(v2, DAY, DAY)
    assert v2 == {"windows": [28], "temp": {"shrink": 0.5}}


# ---- review fixes ----------------------------------------------------------------------
LA = "America/Los_Angeles"


def _la(*args):
    from zoneinfo import ZoneInfo

    return datetime(*args, tzinfo=ZoneInfo(LA))


def _hour_index_series():
    """e(s) equals the hour index of s, so any one-hour lag error changes the answer."""
    start = datetime(2024, 2, 20, tzinfo=UTC)
    d, f = {}, {}
    for i in range(24 * 40):
        s = start + i * H
        f[s] = 1000.0
        d[s] = 1000.0 + i
    return d, f


def test_local_target_across_dst_uses_utc_lags():
    d, f = _hour_index_series()
    ev2 = ResidualEvaluator(d, f)
    local = _la(2024, 3, 11, 0)  # 07:00 UTC, the day after spring-forward
    assert local.astimezone(UTC) == datetime(2024, 3, 11, 7, tzinfo=UTC)
    got = ev2.correction_at({"windows": [1]}, local)
    expected = d[datetime(2024, 3, 9, 7, tzinfo=UTC)] - 1000.0
    assert got == expected
    assert got == ev2.correction_at({"windows": [1]}, local.astimezone(UTC))
    lvl = {"windows": [2], "hour_weight": 0.0}
    assert ev2.correction_at(lvl, local) == ev2.correction_at(lvl, local.astimezone(UTC))


def test_local_target_leak_free():
    d, f = _hour_index_series()
    local = _la(2024, 3, 11, 0)
    t = local.astimezone(UTC)
    cut = t - timedelta(hours=48)
    var = {"windows": [3], "hour_weight": 0.5}
    base = ResidualEvaluator(d, f).correction_at(var, local)
    d2 = {s: (9e9 if s > cut else v) for s, v in d.items()}
    f2 = {s: (-9e9 if s > cut else v) for s, v in f.items()}
    assert ResidualEvaluator(d2, f2).correction_at(var, local) == base


def test_naive_target_rejected():
    with pytest.raises(ValueError):
        ResidualEvaluator({}, {}).correction_at({"windows": [1]}, datetime(2024, 3, 11))


def test_constructor_independent_of_lag_floor():
    d, f, _ = make_series(START, END)
    e49 = ResidualEvaluator(d, f, min_lag_hours=49)
    assert e49.correction_at({"lag_days": 3, "windows": [7]}, T) == pytest.approx(
        hand_r(d, f, T, L=3, windows=[7])
    )
    with pytest.raises(ValueError):
        e49.correction_at({"windows": [7]}, T)
    with pytest.raises(ValueError):
        validate_variant({"windows": [7]}, min_lag_hours=49)
    out, counts = e49.scan({"lag_days": 3, "windows": [7]}, date(2024, 6, 1), date(2024, 6, 3))
    assert counts["scored"] == 3 and len(out) == 3
    # BCDF is still the standard window 7 / lag 2 one
    base = ResidualEvaluator(d, f)
    out0, _ = base.scan({"windows": [7]}, date(2024, 6, 1), date(2024, 6, 3))
    for day, row in out.items():
        assert row[1] == out0[day][1]


def test_scan_survives_overflowing_fitted_slope():
    d, f, _ = make_series(START, END)
    w = {s: (1e155 if s.toordinal() % 2 else -1e155) for s in d}
    ev3 = ResidualEvaluator(d, f, w)
    var = {"windows": [14], "temp": {"shrink": 1}}
    assert ev3.correction_at(var, T) is None
    out, counts = ev3.scan(var, date(2024, 3, 1), date(2024, 3, 5))
    assert out == {}
    assert sum(counts.values()) == 5
    assert counts["no_weather"] == 5
    # a non-weather overflow is classified, never raised
    d2 = {s: (1.7e308 if s.toordinal() % 2 else -1.7e308) for s in d}
    ev4 = ResidualEvaluator(d2, f)
    out, counts = ev4.scan({"windows": [28]}, date(2024, 3, 1), date(2024, 3, 3))
    assert sum(counts.values()) == 3 and counts["scored"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"windows": [7, 28], "mix": 10**400},
        {"windows": [7], "hour_weight": 10**400},
        {"windows": [7], "temp": {"beta": 10**400}},
        {"windows": [14], "temp": {"shrink": 10**400}},
        {"windows": [7], "lag_days": 10**400},
        {"windows": [10**400]},
        {"windows": [7], 1: 0, "extra": 0},
        {"windows": [7], (1, 2): 0, None: 1},
    ],
)
def test_validate_never_leaks_other_exceptions(bad):
    with pytest.raises(ValueError):
        validate_variant(bad)


def test_scan_forecast_is_df_plus_r_bit_for_bit():
    day = date(2024, 6, 5)
    hours = ResidualEvaluator({}, {})._base._day_hours(day)
    cut = hours[0]
    d, f, w = {}, {}, {}
    s = datetime(2024, 5, 1, tzinfo=UTC)
    while s < cut + 24 * H:
        f[s] = 1e16
        d[s] = 1e16 + (4.0 if s.toordinal() % 3 == 0 else 2.0)
        w[s] = 0.75 if s >= cut else 0.0
        s += H
    ev5 = ResidualEvaluator(d, f, w)
    var = {"windows": [3], "temp": {"beta": 1}}
    total = 0.0
    differs = False
    for t in hours:
        r = ev5.correction_at(var, t)
        assert r is not None
        total += abs(d[t] - (f[t] + r))
        core = ev5.correction_at({"windows": [3]}, t)
        differs |= f[t] + core + (r - core) != f[t] + r
    assert differs  # the DF 1e16 case does distinguish the two association orders
    out, counts = ev5.scan(var, day, day)
    assert counts["scored"] == 1
    assert out[day][0] == total / 24


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_two_large_windows_match_hand_formula(data, ev, lam):
    d, f, _ = data
    var = {"windows": [100, 120], "mix": 0.5, "hour_weight": lam}
    t = datetime(2024, 6, 20, 9, tzinfo=UTC)
    exp = hand_r(d, f, t, windows=[100, 120], mix=0.5, lam=lam)
    assert ev.correction_at(var, t) == pytest.approx(exp, rel=1e-12, abs=1e-9)


def test_two_windows_share_samples_lookup_budget(data):
    d, f, _ = data
    ev6 = ResidualEvaluator(d, f)
    calls = 0

    class Counting(dict):
        def get(self, key, default=None):
            nonlocal calls
            calls += 1
            return dict.get(self, key, default)

    ev6._e = Counting(ev6._e)
    var = {"windows": [100, 120], "mix": 0.5, "hour_weight": 0.0}
    ev6.correction_at(var, datetime(2024, 6, 25, 9, tzinfo=UTC))
    assert calls <= 24 * 120
