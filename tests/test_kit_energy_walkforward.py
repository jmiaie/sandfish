"""Tests for the walk-forward search (folds, embargo, holdout lock, selection, pooling)."""

import json
import math
from datetime import date, datetime, timedelta, timezone

import pytest

from sandfish.kits.energy import walkforward as wf

UTC = timezone.utc
SEARCH0 = date(2021, 4, 1)


def hours(start, end):
    t = start
    while t < end:
        yield t
        t += timedelta(hours=1)


def synth(start, end):
    """Hourly demand/forecast/weather with a drifting forecast bias and a temperature link."""
    demand, df, wx = {}, {}, {}
    for i, t in enumerate(hours(start, end)):
        temp = 15 + 8 * math.sin(2 * math.pi * (i % 24) / 24) + 5 * math.sin(i / 400.0)
        base = 20000 + 3000 * math.sin(2 * math.pi * (i % 24) / 24) + 150 * temp
        demand[t] = base + 60 * math.sin(i / 7.0)
        df[t] = base - i * 0.05 + 120 * math.sin(i * 1.3)
        wx[t] = temp
    return demand, df, wx


SMALL = [
    {"model": "seasonal_naive"},
    {"model": "lag_mean", "lags": [48]},
    {"model": "lag_mean", "lags": [48, 168]},
    {"model": "lag_mean_temp", "lags": [48, 168], "temp_coef": 100.0},
]


def test_quarter_blocks_only_full_quarters():
    blocks = wf.quarter_blocks(date(2021, 4, 1), date(2022, 2, 10))
    assert blocks == [
        (date(2021, 4, 1), date(2021, 6, 30)),
        (date(2021, 7, 1), date(2021, 9, 30)),
        (date(2021, 10, 1), date(2021, 12, 31)),
    ]
    # a mid-quarter start skips the partial quarter
    assert wf.quarter_blocks(date(2021, 5, 15), date(2021, 12, 31))[0][0] == date(2021, 7, 1)


def test_make_folds_embargo_and_expanding_train():
    folds = wf.make_folds(SEARCH0, date(2022, 1, 1), date(2022, 12, 31), embargo_days=14)
    assert [f[1][0] for f in folds] == [
        date(2022, 1, 1),
        date(2022, 4, 1),
        date(2022, 7, 1),
        date(2022, 10, 1),
    ]
    for (tr0, tr1), (te0, te1) in folds:
        assert tr0 == SEARCH0
        assert (te0 - tr1).days == 15  # 14 embargo days strictly between train and test
        assert tr1 < te0 <= te1
    assert folds[1][0][1] > folds[0][0][1]  # expanding


def test_make_folds_zero_embargo_and_short_train_skipped():
    folds = wf.make_folds(
        SEARCH0, date(2021, 4, 1), date(2021, 12, 31), embargo_days=0, min_train_days=90
    )
    assert folds[0][1][0] == date(2021, 7, 1)  # the first quarter has no training data
    assert (folds[0][1][0] - folds[0][0][1]).days == 1


def test_make_folds_rejects_holdout_and_bad_args():
    with pytest.raises(wf.HoldoutError):
        wf.make_folds(SEARCH0, date(2022, 1, 1), date(2025, 1, 1))
    with pytest.raises(wf.HoldoutError):
        wf.make_folds(SEARCH0, date(2022, 1, 1), date(2025, 6, 30))
    with pytest.raises(ValueError):
        wf.make_folds(SEARCH0, date(2022, 1, 1), date(2022, 12, 31), embargo_days=-1)
    with pytest.raises(ValueError):
        wf.make_folds(SEARCH0, date(2022, 1, 1), date(2022, 12, 31), embargo_days=True)
    with pytest.raises(ValueError):
        wf.make_folds(SEARCH0, date(2021, 4, 1), date(2021, 5, 31))


def test_make_folds_custom_holdout():
    with pytest.raises(wf.HoldoutError):
        wf.make_folds(
            SEARCH0, date(2022, 1, 1), date(2022, 12, 31), holdout_start=date(2022, 12, 31)
        )


def test_default_variants_grid():
    v = wf.default_variants()
    assert len(v) == 1 + len(wf.LAG_SETS) + len(wf.LAG_SETS) * len(wf.TEMP_COEFS)
    assert len({json.dumps(x, sort_keys=True) for x in v}) == len(v)
    assert all(min(x["lags"]) >= 48 for x in v if "lags" in x)  # never inside the 39h floor


def test_before_cuts_at_midnight_utc():
    s = {datetime(2024, 12, 31, 23, tzinfo=UTC): 1.0, datetime(2025, 1, 1, 0, tzinfo=UTC): 2.0}
    assert list(wf._before(s, date(2025, 1, 1)).values()) == [1.0]


def test_block_bootstrap_deterministic_and_sane():
    diffs = [1.0, -2.0, 3.0, 0.5] * 25
    a = wf._block_bootstrap_ci(diffs, reps=300)
    assert a == wf._block_bootstrap_ci(diffs, reps=300)
    mean = sum(diffs) / len(diffs)
    assert a[0] <= mean <= a[1]
    assert wf._block_bootstrap_ci([2.0] * 30, reps=50) == (2.0, 2.0)
    assert all(math.isnan(x) for x in wf._block_bootstrap_ci([]))


def run(demand, df, wx, **kw):
    kw.setdefault("search_start", date(2021, 4, 1))
    kw.setdefault("first_test", date(2021, 10, 1))
    kw.setdefault("search_end", date(2022, 3, 31))
    kw.setdefault("variants", SMALL)
    kw.setdefault("min_days", 20)
    return wf.run_search(demand, df, wx, **kw)


@pytest.fixture(scope="module")
def data():
    return synth(datetime(2021, 3, 1, tzinfo=UTC), datetime(2022, 5, 1, tzinfo=UTC))


def test_run_search_reports_folds_and_pools(data):
    rep = run(*data)
    assert rep["holdout_touched"] is False
    assert rep["trials_per_fold"] == len(SMALL)
    assert len(rep["folds"]) == 2
    for f in rep["folds"]:
        assert f["selected"] in SMALL
        assert f["train"][1] < f["test"][0]
        # score vs operator from the evaluator equals 1 - model/operator from the day errors
        m = f["test_mae_mw"]
        assert f["test_result"]["score_vs_operator"] == pytest.approx(
            1 - m["model"] / m["operator"], abs=1e-9
        )
        assert f["test_result"]["skill_vs_bias_corrected_df"] == pytest.approx(
            1 - m["model"] / m["bias_corrected_df"], abs=1e-9
        )
    p = rep["pooled_selected"]
    n = [f["test_result"]["n"] for f in rep["folds"]]
    assert p["days"] == sum(n)
    expect = sum(f["test_mae_mw"]["model"] * f["test_result"]["n"] for f in rep["folds"]) / sum(n)
    assert p["mae_mw"]["model"] == pytest.approx(expect, abs=1e-6)
    assert p["skill_vs_bias_corrected_df"] == pytest.approx(
        1 - p["mae_mw"]["model"] / p["mae_mw"]["bias_corrected_df"]
    )
    assert p["folds"] == 2


def test_selection_uses_train_only(data):
    rep = run(*data)
    demand, df, wx = data
    from sandfish.kits.energy.evaluator import EnergyLoadEvaluator

    f = rep["folds"][0]
    tr, te = [tuple(date.fromisoformat(x) for x in k) for k in (f["train"], f["test"])]
    ev = EnergyLoadEvaluator(demand, df, tr, te, min_days=20, weather=wx, bias_baseline=True)
    skills = [ev.evaluate(v, "train").metadata["skill_vs_bias_corrected_df"] for v in SMALL]
    assert f["selected"] == SMALL[skills.index(max(skills))]
    assert f["train_result"]["skill_vs_bias_corrected_df"] == pytest.approx(max(skills))


def test_holdout_data_is_never_read(data):
    demand, df, wx = data
    base = run(demand, df, wx)
    # poisoned data after the holdout start (inside the search window's lookback it cannot be)
    demand2, df2, wx2 = dict(demand), dict(df), dict(wx)
    for t in hours(datetime(2022, 4, 1, tzinfo=UTC), datetime(2022, 5, 1, tzinfo=UTC)):
        demand2[t], df2[t], wx2[t] = 1e9, -1e9, 1e6
    hold = run(demand2, df2, wx2, holdout_start=date(2022, 4, 1))
    clean = run(demand, df, wx, holdout_start=date(2022, 4, 1))
    assert hold == clean
    assert base["folds"][0] == hold["folds"][0]


def test_search_end_into_holdout_raises(data):
    with pytest.raises(wf.HoldoutError):
        run(*data, search_end=date(2022, 3, 31), holdout_start=date(2022, 3, 31))


def test_load_series_parses_eia_lines(tmp_path):
    p = tmp_path / "e.ndjson"
    rows = [
        {
            "series_id": "EBA.X-ALL.D.H",
            "data": [["20240101T01", "5"], ["20240101T00", None], ["bad", "1"]],
        },
        {"series_id": "EBA.X-ALL.DF.H", "data": [["20240101T00", "9"]]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    got = wf.load_series(str(p), "EBA.X-ALL.D.H")
    assert got == {datetime(2024, 1, 1, 1, tzinfo=UTC): 5.0}
    assert wf.load_series(str(p), "EBA.Y.D.H") == {}
