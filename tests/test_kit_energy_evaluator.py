import copy
import math
from datetime import date, datetime, timedelta, timezone

import pytest

from sandfish.kits.energy.evaluator import EnergyLoadEvaluator, split_summary
from sandfish.search import TrialLedger, as_evaluator, run_grid

START = datetime(2023, 12, 1, tzinfo=timezone.utc)
HOURS = 24 * 200
BIAS = 10.0
TRAIN = (date(2024, 1, 10), date(2024, 3, 31))
TEST = (date(2024, 4, 1), date(2024, 5, 31))


def _pattern(i):
    return ((i % 168) * 37 % 101) / 101.0


def make_series(hours=HOURS):
    demand, op = {}, {}
    for i in range(hours):
        t = START + timedelta(hours=i)
        demand[t] = 1000 + 50 * _pattern(i)
        op[t] = demand[t] + BIAS
    return demand, op


def make_eval(demand=None, op=None, **kw):
    d, o = make_series()
    return EnergyLoadEvaluator(
        d if demand is None else demand,
        o if op is None else op,
        kw.pop("train", TRAIN),
        kw.pop("test", TEST),
        **kw,
    )


def test_perfect_periodic_seasonal_naive():
    res = make_eval().evaluate({"model": "seasonal_naive"}, "train")
    assert res.scorable
    assert res.metadata["skill_seasonal_naive_vs_operator"] == pytest.approx(1.0)
    assert res.score == pytest.approx(1.0)
    assert res.metadata["skill_vs_seasonal_naive"] is None
    assert res.metadata["model"] == "seasonal_naive"


def test_lag_mean_of_weekly_multiples():
    res = make_eval().evaluate({"lags": [168, 336]}, "test")
    assert res.score == pytest.approx(1.0)


def test_leak_guard_raises():
    with pytest.raises(ValueError, match="would leak"):
        make_eval().evaluate({"model": "lag_mean", "lags": [24, 168]}, "train")


def test_lags_missing_or_bad():
    ev = make_eval()
    for variant in ({"model": "lag_mean"}, {"lags": []}, {"lags": [-48]}):
        with pytest.raises(ValueError):
            ev.evaluate(variant, "train")


def test_bad_split_and_model():
    ev = make_eval()
    with pytest.raises(ValueError):
        ev.evaluate({"lags": [48]}, "valid")
    with pytest.raises(ValueError):
        ev.evaluate({"model": "nope", "lags": [48]}, "train")


def test_constructor_validation():
    d, o = make_series()
    with pytest.raises(ValueError):
        EnergyLoadEvaluator(d, o, (TRAIN[1], TRAIN[0]), TEST)
    with pytest.raises(ValueError):
        EnergyLoadEvaluator(d, o, (date(2024, 2, 1), date(2024, 4, 5)), TEST)
    naive = {datetime(2024, 1, 1): 1.0}
    with pytest.raises(ValueError):
        EnergyLoadEvaluator(naive, o, TRAIN, TEST)
    with pytest.raises(ValueError):
        EnergyLoadEvaluator(d, naive, TRAIN, TEST)


def test_dst_day_dropped():
    ev = make_eval(train=(date(2024, 3, 1), date(2024, 3, 31)), min_days=1)
    res = ev.evaluate({"lags": [168]}, "train")
    assert "2024-03-10" not in res.metadata["days"]
    assert res.metadata["dropped_days"] == 1
    assert res.n == 30


def test_missing_hour_drops_day():
    d, o = make_series()
    # local 2024-02-10 12:00 PST == 20:00 UTC
    del d[datetime(2024, 2, 10, 20, tzinfo=timezone.utc)]
    res = make_eval(d, o, min_days=1).evaluate({"lags": [168]}, "train")
    assert "2024-02-10" not in res.metadata["days"]
    # a week later the 168h lag is missing, so that day is dropped as well
    assert "2024-02-17" not in res.metadata["days"]


def test_nan_counts_as_absent():
    d, o = make_series()
    o[datetime(2024, 2, 10, 20, tzinfo=timezone.utc)] = float("nan")
    res = make_eval(d, o, min_days=1).evaluate({"lags": [168]}, "train")
    assert "2024-02-10" not in res.metadata["days"]
    assert res.metadata["dropped_days"] == 2  # the NaN day and the 23-hour DST day


def test_baseline_lag_missing_drops_day_for_every_variant():
    d, o = make_series()
    del d[datetime(2024, 2, 3, 20, tzinfo=timezone.utc)]  # 168h before 2024-02-10 20:00
    res = make_eval(d, o, min_days=1).evaluate({"lags": [48]}, "train")
    assert "2024-02-10" not in res.metadata["days"]


def test_not_scorable_too_few_days():
    res = make_eval(min_days=1000).evaluate({"lags": [168]}, "train")
    assert not res.scorable
    assert res.n == len(res.metadata["days"])
    assert "not_scorable_reason" in res.metadata


def test_not_scorable_zero_operator_error():
    d, _ = make_series()
    res = make_eval(d, dict(d)).evaluate({"lags": [168]}, "train")
    assert not res.scorable


def test_loss_diff_length_and_order():
    res = make_eval().evaluate({"lags": [48]}, "train")
    meta = res.metadata
    assert len(meta["loss_diff"]) == res.n == len(meta["days"])
    assert meta["days"] == sorted(meta["days"])
    assert all(math.isfinite(x) for x in meta["loss_diff"])


def test_deterministic_and_no_mutation():
    d, o = make_series()
    d0, o0 = copy.deepcopy(d), copy.deepcopy(o)
    ev = make_eval(d, o)
    variant = {"model": "lag_mean", "lags": [48, 168]}
    v0 = copy.deepcopy(variant)
    assert ev.evaluate(variant, "test") == ev.evaluate(variant, "test")
    assert d == d0 and o == o0 and variant == v0


def test_split_summary():
    assert split_summary(make_eval()) == {
        "train": ["2024-01-10", "2024-03-31"],
        "test": ["2024-04-01", "2024-05-31"],
        "min_days": 60,
        "min_lag_hours": 39,
    }


def test_run_grid_integration():
    ledger = TrialLedger()
    space = {"model": ["lag_mean"], "lag_set": ["48", "72", "168", "48,168"]}
    out = run_grid(
        space,
        as_evaluator(make_eval()),
        ledger,
        repair=lambda v: {"model": v["model"], "lags": [int(x) for x in v["lag_set"].split(",")]},
    )
    assert out is not None
    assert ledger.n_trials == 4
    assert ledger.best() is not None


def _flat(start, hours, value=1000.0, bias=10.0):
    d = {start + timedelta(hours=i): value for i in range(hours)}
    return d, {k: v + bias for k, v in d.items()}


def test_fractional_offset_zone_keeps_full_day():
    d, o = _flat(datetime(2024, 1, 1, tzinfo=timezone.utc), 24 * 40)
    ev = EnergyLoadEvaluator(
        d,
        o,
        (date(2024, 1, 20), date(2024, 1, 22)),
        (date(2024, 2, 1), date(2024, 2, 2)),
        min_days=1,
        tz="Asia/Kolkata",
    )
    res = ev.evaluate({"model": "seasonal_naive"}, "train")
    assert res.metadata["days"] == ["2024-01-20", "2024-01-21", "2024-01-22"]
    assert res.metadata["dropped_days"] == 0


def test_half_hour_dst_day_dropped():
    d, o = _flat(datetime(2024, 3, 1, tzinfo=timezone.utc), 24 * 60)
    ev = EnergyLoadEvaluator(
        d,
        o,
        (date(2024, 4, 5), date(2024, 4, 9)),
        (date(2024, 5, 1), date(2024, 5, 2)),
        min_days=1,
        tz="Australia/Lord_Howe",
    )
    res = ev.evaluate({"model": "seasonal_naive"}, "train")
    assert "2024-04-07" not in res.metadata["days"]
    assert res.metadata["dropped_days"] == 1


def test_large_finite_values_stay_finite():
    d, o = _flat(datetime(2024, 1, 1, tzinfo=timezone.utc), 24 * 40, 1e308, -1e306)
    ev = EnergyLoadEvaluator(
        d,
        o,
        (date(2024, 1, 20), date(2024, 1, 25)),
        (date(2024, 2, 1), date(2024, 2, 2)),
        min_days=1,
    )
    res = ev.evaluate({"lags": [48, 168]}, "train")
    assert res.scorable
    assert res.score == pytest.approx(1.0)


def test_huge_lag_drops_days_instead_of_crashing():
    res = make_eval(min_days=1).evaluate({"lags": [10**12]}, "train")
    assert not res.scorable
    assert res.n == 0


def test_lags_must_be_list():
    with pytest.raises(ValueError):
        make_eval().evaluate({"lags": {48: 72}}, "train")
    with pytest.raises(ValueError):
        make_eval().evaluate({"lags": 48}, "train")


def test_init_docstrings_empty():
    import sandfish.kits as kits
    import sandfish.kits.energy as energy

    assert (kits.__doc__ or "") == ""
    assert (energy.__doc__ or "") == ""
