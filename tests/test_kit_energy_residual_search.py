"""Tests for sandfish.kits.energy.residual_search (synthetic, offline)."""

from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, timedelta, timezone

import pytest

from sandfish.kits.energy import residual_search as rs
from sandfish.kits.energy.residual import validate_variant
from sandfish.kits.energy.walkforward import HoldoutError

UTC = timezone.utc
H = timedelta(hours=1)

ID = {"windows": [7]}
VA = {"windows": [3]}
VB = {"windows": [5]}
VC = {"windows": [14]}
NID = validate_variant(ID)
NA = validate_variant(VA)
NB = validate_variant(VB)

SMALL = {"min_days": 20, "bootstrap_reps": 100, "rc_reps": 100, "pbo_splits": 4}
# 12 quarterly folds 2013Q1..2015Q4
LONG = {
    "search_start": date(2012, 1, 1),
    "first_test": date(2013, 1, 1),
    "search_end": date(2015, 12, 31),
    "holdout_start": date(2016, 1, 1),
}
# 2 folds (2023Q3, 2023Q4)
SHORT = {
    "search_start": date(2023, 1, 1),
    "first_test": date(2023, 7, 1),
    "search_end": date(2023, 12, 31),
    "holdout_start": date(2024, 1, 1),
}


class FakeEv:
    """Stands in for ResidualEvaluator: per-day rows come from ``FakeEv.gain_fn``."""

    gain_fn = None
    seen = {"series": [], "ranges": []}

    def __init__(self, demand, operator_forecast, weather=None, **kw):
        FakeEv.seen["series"] = [demand, operator_forecast, weather]

    def scan(self, variant, first_day, last_day):
        FakeEv.seen["ranges"].append((first_day, last_day))
        out = {}
        d = first_day
        while d <= last_day:
            g = FakeEv.gain_fn(validate_variant(variant), d)
            if g is not None:
                out[d] = (100.0 - g, 100.0, 120.0)
            d += timedelta(days=1)
        return out, {}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(rs, "ResidualEvaluator", FakeEv)
    FakeEv.seen = {"series": [], "ranges": []}

    def setg(fn):
        def wrapped(v, d):
            if v == NID:
                return 0.0
            return fn(v, d)

        FakeEv.gain_fn = staticmethod(wrapped)

    return setg


def run(variants, cfg=SHORT, **kw):
    args = {**SMALL, **kw}
    return rs.run_residual_search({}, {}, {}, variants=variants, **cfg, **args)


def quarter(d):
    return (d.year, (d.month - 1) // 3 + 1)


# ---------------------------------------------------------------- real-evaluator data
def make_series(start, end, seed=1, temp_slope=None):
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
        if temp_slope is not None:
            err = temp_slope * temp + rng.uniform(-5, 5)
        else:
            err = 80 * math.sin(s.hour) + 40 * math.cos(s.toordinal()) + rng.uniform(-60, 60)
        d[s] = base
        f[s] = base - err
        s += H
    return d, f, w


S0 = datetime(2022, 12, 1, tzinfo=UTC)
S1 = datetime(2024, 1, 20, tzinfo=UTC)


@pytest.fixture(scope="module")
def real():
    return make_series(S0, S1)


@pytest.fixture(scope="module")
def planted():
    return make_series(S0, S1, temp_slope=100.0)


REAL_VARIANTS = [VA, ID, VC]


# ---------------------------------------------------------------- holdout
def test_holdout_error():
    with pytest.raises(HoldoutError):
        rs.run_residual_search(
            {}, {}, {}, variants=[VA], **{**SHORT, "search_end": date(2024, 1, 1)}, **SMALL
        )


def test_fake_never_scans_past_search_end(fake):
    fake(lambda v, d: 1.0)
    run([VA, ID])
    assert FakeEv.seen["ranges"]
    assert all(last <= SHORT["search_end"] for _, last in FakeEv.seen["ranges"])


def test_series_cut_before_holdout(fake):
    fake(lambda v, d: 1.0)
    hold = datetime(2024, 1, 1, tzinfo=UTC)
    series = {hold - H: 1.0, hold: 2.0, hold + H: 3.0}
    rs.run_residual_search(
        series, dict(series), dict(series), variants=[VA], **SHORT, **SMALL
    )
    for s in FakeEv.seen["series"]:
        assert list(s) == [hold - H]


def test_poisoned_post_holdout_data_changes_nothing(real):
    d, f, w = real
    hold = datetime(2024, 1, 1, tzinfo=UTC)
    pd_ = {t: (v * 1000 if t >= hold else v) for t, v in d.items()}
    pf = {t: (v * -50 if t >= hold else v) for t, v in f.items()}
    pw = {t: (1e6 if t >= hold else v) for t, v in w.items()}
    a = rs.run_residual_search(d, f, w, variants=REAL_VARIANTS, **SHORT, **SMALL)
    b = rs.run_residual_search(pd_, pf, pw, variants=REAL_VARIANTS, **SHORT, **SMALL)
    assert a == b
    assert a["holdout_touched"] is False


# ---------------------------------------------------------------- selection
def test_selection_uses_train_only(fake):
    early = date(2023, 7, 1)

    def fn(v, d):
        if v == NA:
            return 5.0 if d < early else -5.0
        return 3.0

    fake(fn)
    res = run([VA, VB, ID])
    f1, f2 = res["folds"]
    assert f1["selected"] == NA and not f1["fallback_to_bcdf"]
    assert f1["train_skill_vs_bcdf"] == pytest.approx(0.05)
    assert f1["test_mae_mw"]["model"] == pytest.approx(105.0)
    assert f1["test_skill_vs_bcdf"] == pytest.approx(-0.05)
    assert f1["test_skill_vs_df"] == pytest.approx(1 - 105 / 120)
    assert f1["hindsight_best_test"]["variant"] == NB
    assert f1["hindsight_best_test"]["skill_vs_bcdf"] == pytest.approx(0.03)
    assert f1["selected_test_rank_pct"] == pytest.approx(1 / 3)
    assert f1["variants_eligible"] == 3
    assert f1["train_days"] == 167
    # fold 2: A's train mean gain (181 days of +5, 78 of -5) is below B's 3
    assert f2["selected"] == NB


def test_tie_goes_to_earlier_index(fake):
    fake(lambda v, d: 2.0)
    r1 = run([VB, VA, ID])
    r2 = run([VA, VB, ID])
    assert all(f["selected"] == NB for f in r1["folds"])
    assert all(f["selected"] == NA for f in r2["folds"])


def test_ineligible_variant_not_selected(fake):
    def fn(v, d):
        if v == NB:
            return 9.0 if d >= date(2023, 4, 1) else None
        return 1.0

    fake(fn)
    res = run([VA, VB, ID], min_days=100)
    f1 = res["folds"][0]
    assert f1["variants_eligible"] == 2  # B has 77 train days only
    assert f1["selected"] == NA


# ---------------------------------------------------------------- fallback
def test_fallback_to_bcdf(fake):
    fake(lambda v, d: -2.0)
    res = run([VA, VB, ID])
    for f in res["folds"]:
        assert f["fallback_to_bcdf"] is True
        assert f["selected"] == NID
        assert f["train_skill_vs_bcdf"] == 0.0
        assert f["test_skill_vs_bcdf"] == 0.0
        assert f["selected_test_rank_pct"] == 1.0
        assert f["test_days"] > 0 and f["train_days"] > 0
    assert res["pooled"]["fold_wins"] == 0
    assert res["pooled"]["skill_vs_bcdf"] == 0.0
    assert res["verdict"]["label"] == "no_evidence"


def test_fallback_without_identity_in_list_is_extra_scan(fake):
    fake(lambda v, d: -2.0)
    res = run([VA, VB])
    assert res["n_trials"] == 2
    assert len(res["overfit"]["variant_summaries"]) == 2
    assert all(f["selected"] == NID for f in res["folds"])
    assert all(f["test_skill_vs_bcdf"] == 0.0 for f in res["folds"])
    # the extra identity scan happened (3 scans, one per variant plus identity)
    assert len(FakeEv.seen["ranges"]) == 3


def test_fallback_rank_pct_mixed(fake):
    def fn(v, d):
        if v == NA:
            return -2.0 if d < date(2023, 7, 1) else 3.0
        return -2.0

    fake(fn)
    res = run([VA, VB])
    f1 = res["folds"][0]
    assert f1["fallback_to_bcdf"]
    assert f1["selected_test_rank_pct"] == pytest.approx(0.5)  # only B's -2% is <= 0


# ---------------------------------------------------------------- pooled / folds
def test_pooled_matches_hand_aggregation(fake):
    def fn(v, d):
        return 4.0 + (d.toordinal() % 5)

    fake(fn)
    res = run([VA, ID])
    gains = []
    for f in res["folds"]:
        assert f["selected"] == NA
        d = date.fromisoformat(f["test"][0])
        end = date.fromisoformat(f["test"][1])
        while d <= end:
            gains.append(fn(NA, d))
            d += timedelta(days=1)
    p = res["pooled"]
    n = len(gains)
    assert p["days"] == n and p["folds_scored"] == 2
    assert p["mae_mw"]["model"] == pytest.approx(100 - sum(gains) / n)
    assert p["mae_mw"]["bcdf"] == 100.0 and p["mae_mw"]["df"] == 120.0
    assert p["skill_vs_bcdf"] == pytest.approx(sum(gains) / (100 * n))
    assert p["skill_vs_df"] == pytest.approx(1 - (100 - sum(gains) / n) / 120)
    assert p["bcdf_skill_vs_df"] == pytest.approx(1 - 100 / 120)
    assert p["gain_vs_bcdf_mw"]["mean"] == pytest.approx(sum(gains) / n)
    assert p["gain_vs_df_mw"]["mean"] == pytest.approx(sum(gains) / n + 20)
    lo, hi = p["gain_vs_bcdf_mw"]["ci95_block"]
    assert lo <= p["gain_vs_bcdf_mw"]["mean"] <= hi
    assert p["selected_counts"] == {json.dumps(NA, sort_keys=True): 2}
    assert p["by_year"]["2023"]["days"] == n
    assert p["by_year"]["2023"]["gain_vs_bcdf_mw"] == pytest.approx(sum(gains) / n)


def test_binomial_tail_exact():
    assert rs._binom_tail(8, 12) == pytest.approx(0.1938476562)
    assert rs._binom_tail(12, 12) == 1 / 4096
    assert rs._binom_tail(0, 5) == 1.0


def test_fold_wins_sign_test_and_by_year(fake):
    losing = {(2013, 2), (2014, 1), (2014, 3), (2015, 2)}

    def fn(v, d):
        return -3.0 if quarter(d) in losing else 3.0

    fake(fn)
    res = run([VA, ID], cfg=LONG)
    p = res["pooled"]
    assert res["n_folds"] == 12 and p["folds_scored"] == 12
    assert p["fold_wins"] == 8
    assert p["sign_test_p"] == pytest.approx(0.19384765625)
    assert p["fold_skill_max"] == pytest.approx(0.03)
    assert p["fold_skill_min"] == pytest.approx(-0.03)
    assert p["fold_skill_mean"] == pytest.approx((8 * 0.03 - 4 * 0.03) / 12)
    assert set(p["by_year"]) == {"2013", "2014", "2015"}
    assert p["by_year"]["2013"]["days"] == 365
    assert p["by_year"]["2013"]["gain_vs_bcdf_mw"] == pytest.approx(
        (3 * 273 - 3 * 91) / 365, rel=0.02
    )


def test_all_folds_win_sign_p(fake):
    fake(lambda v, d: 3.0)
    res = run([VA, ID], cfg=LONG)
    assert res["pooled"]["fold_wins"] == 12
    assert res["pooled"]["sign_test_p"] == 1 / 4096


def test_no_scored_fold_gives_pooled_none(fake):
    fake(lambda v, d: 1.0)
    res = run([VA, ID], min_days=1000)
    assert res["pooled"] is None
    assert all(not f["test_scored"] and f["test_mae_mw"] is None for f in res["folds"])
    assert res["verdict"]["label"] == "no_evidence"
    assert res["verdict"]["fold_wins_enough"] is False
    json.dumps(res, allow_nan=False)


# ---------------------------------------------------------------- reality check
def test_rc_hand_checkable():
    g = [[1.0, 0.0]] * 4
    out = rs._reality_check(g, 2, 100, 1, [NA, NB])
    assert out["statistic"] == 2.0
    assert out["p_value"] == pytest.approx(1 / 101)
    assert out["best_variant"] == NA and out["n_days"] == 4 and out["reps"] == 100
    zero = rs._reality_check([[0.0, 0.0]] * 4, 2, 100, 1, [NA, NB])
    assert zero["p_value"] == 1.0
    assert zero["best_variant"] == NA  # first on ties


def test_rc_planted_effect_is_significant(planted):
    d, f, w = planted
    variants = [
        {"windows": [7], "temp": {"beta": 100}},
        {"windows": [7], "temp": {"beta": -100}},
        ID,
    ]
    res = rs.run_residual_search(d, f, w, variants=variants, **SHORT, **SMALL)
    rc = res["overfit"]["reality_check"]
    assert rc["p_value"] < 0.05
    assert rc["best_variant"]["temp"] == {"beta": 100}
    assert res["pooled"]["skill_vs_bcdf"] > 0


def test_rc_pure_noise_is_not_significant(fake):
    def fn(v, d):
        return random.Random(f"{json.dumps(v, sort_keys=True)}{d}").gauss(0, 1)

    fake(fn)
    res = run([VA, VB, VC, ID])
    assert res["overfit"]["reality_check"]["p_value"] > 0.05


# ---------------------------------------------------------------- PBO / DSR
def test_pbo_and_dsr_present(fake):
    def fn(v, d):
        return random.Random(f"{json.dumps(v, sort_keys=True)}{d}").gauss(0.2, 1)

    fake(fn)
    ov = run([VA, VB, VC, ID])["overfit"]
    assert ov["common_days"] == 184
    assert 0.0 <= ov["pbo"]["pbo"] <= 1.0
    assert ov["pbo"]["columns_used"] == 3 and ov["pbo"]["columns_dropped_constant"] == 1
    assert ov["dsr"]["n_trials"] == 4 and ov["dsr"]["n_obs"] == 184
    assert "autocorrelated" in ov["dsr"]["caveat"]
    assert ov["best_fixed"]["variant"] in (NA, NB, validate_variant(VC))
    assert ov["variant_summaries"][3]["skill_vs_bcdf"] == 0.0
    assert ov["variant_summaries"][3]["mean_gain_mw"] == 0.0


def test_too_few_common_days(fake):
    def fn(v, d):
        if v == validate_variant(VC) and d.toordinal() % 10:
            return None
        return 1.0 + (d.toordinal() % 3)

    fake(fn)
    ov = run([VA, VC, ID], min_days=5)["overfit"]
    assert ov["common_days"] < 30
    assert ov["reality_check"] is None and ov["pbo"] is None and ov["dsr"] is None
    assert "common days" in ov["reason"]


def test_fewer_than_two_usable_columns(fake):
    fake(lambda v, d: 1.0 + (d.toordinal() % 3))
    ov = run([VA, ID])["overfit"]
    assert ov["reality_check"] is not None
    assert ov["pbo"] is None and ov["dsr"] is None
    assert "pbo" in ov["reason"] and "dsr" in ov["reason"]


# ---------------------------------------------------------------- verdict
def _pooled(lo, mean, wins, n=3):
    return {"gain_vs_bcdf_mw": {"mean": mean, "ci95_block": [lo, 9.0]},
            "folds_scored": n, "fold_wins": wins}  # fmt: skip


def _ov(p, pbo):
    return {"reality_check": None if p is None else {"p_value": p},
            "pbo": None if pbo is None else {"pbo": pbo}}  # fmt: skip


def test_verdict_labels():
    ready = rs._verdict(_pooled(1.0, 2.0, 3), _ov(0.01, 0.2))
    assert ready["label"] == "ready_for_holdout" and ready["ready_for_holdout"] is True
    promising = rs._verdict(_pooled(-1.0, 2.0, 3), _ov(0.01, 0.8))
    assert promising["label"] == "promising" and promising["ready_for_holdout"] is False
    assert promising["ci_lower_gain_positive"] is False
    assert promising["pbo_acceptable"] is False
    none = rs._verdict(_pooled(-1.0, 2.0, 1), _ov(0.5, 0.8))
    assert none["label"] == "no_evidence" and none["explanation"]
    neg = rs._verdict(_pooled(1.0, -2.0, 3), _ov(0.01, 0.2))
    assert neg["label"] == "ready_for_holdout"  # all four true overrides the mean test
    nulls = rs._verdict(None, _ov(None, None))
    assert nulls["label"] == "no_evidence"
    assert not any(nulls[k] for k in ("ci_lower_gain_positive", "reality_check_significant",
                                       "pbo_acceptable", "fold_wins_enough"))  # fmt: skip


def test_verdict_win_share_boundary():
    assert rs._verdict(_pooled(1.0, 1.0, 2), _ov(0.01, 0.2))["fold_wins_enough"] is True
    assert rs._verdict(_pooled(1.0, 1.0, 1), _ov(0.01, 0.2))["fold_wins_enough"] is False


def test_constants():
    assert (rs.RC_ALPHA, rs.PBO_MAX, rs.WIN_SHARE) == (0.05, 0.5, 2 / 3)


def test_planted_run_verdict_is_consistent(fake):
    def fn(v, d):
        return 5.0 + random.Random(f"{json.dumps(v, sort_keys=True)}{d}").gauss(0, 0.5)

    fake(fn)
    res = run([VA, ID])
    v = res["verdict"]
    assert v["ci_lower_gain_positive"] and v["fold_wins_enough"]
    assert v["label"] in ("ready_for_holdout", "promising")


# ---------------------------------------------------------------- determinism / json
def test_determinism_and_seed(fake):
    def fn(v, d):
        return random.Random(f"{json.dumps(v, sort_keys=True)}{d}").gauss(0.5, 1)

    fake(fn)
    a = run([VA, VB, ID], seed=1)
    assert a == run([VA, VB, ID], seed=1)
    b = run([VA, VB, ID], seed=2)
    assert a["folds"] == b["folds"]
    assert a["pooled"]["gain_vs_bcdf_mw"]["mean"] == b["pooled"]["gain_vs_bcdf_mw"]["mean"]
    assert a["pooled"]["gain_vs_bcdf_mw"]["ci95_block"] != b["pooled"]["gain_vs_bcdf_mw"]["ci95_block"]


def test_json_round_trip_real_data(real):
    d, f, w = real
    res = rs.run_residual_search(d, f, w, variants=REAL_VARIANTS, **SHORT, **SMALL)
    text = json.dumps(res, allow_nan=False)
    assert json.loads(text) == res
    summ = res["overfit"]["variant_summaries"][1]
    assert summ["variant"] == NID
    assert summ["skill_vs_bcdf"] == 0.0 and summ["mean_gain_mw"] == 0.0
    assert res["n_trials"] == 3 and res["n_folds"] == 2


def test_sanitize_non_finite():
    out = rs._sanitize({"a": math.nan, "b": [math.inf, 1.0], "c": (-math.inf,)})
    assert out == {"a": None, "b": [None, 1.0], "c": [None]}


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "kw",
    [
        {"min_days": 0},
        {"bootstrap_reps": 99},
        {"rc_reps": 99},
        {"block_days": 0},
        {"pbo_splits": 3},
        {"pbo_splits": 0},
        {"pbo_splits": 22},
        {"variants": [ID, {"windows": [7], "hour_weight": 1}]},
        {"variants": [{"windows": [3], "mix": None, "hour_weight": 1}, VA]},
        {"variants": [VA, {"windows": [3]}]},
        {"variants": [{"windows": [3], "agg": "mean"}, VA]},
        {"variants": [{"windows": [0]}]},
        {"variants": []},
    ],
)
def test_validation_errors(fake, kw):
    fake(lambda v, d: 1.0)
    args = {"variants": [VA, ID], **SMALL}
    args.update(kw)
    with pytest.raises(ValueError):
        rs.run_residual_search({}, {}, {}, **SHORT, **args)


# ---------------------------------------------------------------- missing weather
def test_missing_weather_days(real):
    d, f, w = real
    lo, hi = datetime(2023, 8, 10, tzinfo=UTC), datetime(2023, 8, 21, tzinfo=UTC)
    w2 = {t: v for t, v in w.items() if not lo <= t < hi}
    temp_v = {"windows": [7], "temp": {"beta": 100}}
    res = rs.run_residual_search(d, f, w2, variants=[temp_v, VA, ID], **SHORT, **SMALL)
    sums = res["overfit"]["variant_summaries"]
    assert sums[0]["days"] < sums[1]["days"]
    assert res["overfit"]["common_days"] == sums[0]["days"]
    assert res["overfit"]["common_days"] >= 30
    assert res["pooled"] is not None
    assert res["overfit"]["reality_check"] is not None
    json.dumps(res, allow_nan=False)


# ---------------------------------------------------------------- review fixes
class LazyMap(dict):
    """Mapping whose values at/after ``cut`` raise when fetched (keys are still listed)."""

    def __init__(self, data, cut):
        super().__init__(data)
        self.cut = cut

    def __getitem__(self, key):
        if key >= self.cut:
            raise AssertionError("holdout value fetched")
        return super().__getitem__(key)

    def items(self):
        raise AssertionError("items() must not be used")

    def values(self):
        raise AssertionError("values() must not be used")


def test_holdout_values_never_fetched(real):
    d, f, w = real
    cut = datetime(2024, 1, 1, tzinfo=UTC)
    a = rs.run_residual_search(d, f, w, variants=REAL_VARIANTS, **SHORT, **SMALL)
    b = rs.run_residual_search(
        LazyMap(d, cut), LazyMap(f, cut), LazyMap(w, cut),
        variants=REAL_VARIANTS, **SHORT, **SMALL,
    )  # fmt: skip
    assert a == b


def test_identity_gain_exactly_zero_despite_float_noise(monkeypatch):
    class Noisy(FakeEv):
        def scan(self, variant, first_day, last_day):
            out, meta = super().scan(variant, first_day, last_day)
            # model MAE differs from BCDF MAE by summation noise of 1.0 at 4e14 scale
            return {d: (r[0] + 1.0, 4e14, r[2]) for d, r in out.items()}, meta

    FakeEv.gain_fn = staticmethod(lambda v, d: 0.0)
    monkeypatch.setattr(rs, "ResidualEvaluator", Noisy)
    res = rs.run_residual_search(
        {}, {}, {}, variants=[{"windows": [7], "hour_weight": 1}, VA], **SHORT, **SMALL
    )
    summ = res["overfit"]["variant_summaries"][0]
    assert summ["mean_gain_mw"] == 0.0 and summ["skill_vs_bcdf"] == 0.0


def test_identity_gain_zero_real_data(real):
    d, f, w = real
    res = rs.run_residual_search(d, f, w, variants=REAL_VARIANTS, **SHORT, **SMALL)
    assert res["overfit"]["variant_summaries"][1]["mean_gain_mw"] == 0.0


def test_scored_fallback_with_zero_bcdf_error(monkeypatch):
    class Perfect(FakeEv):
        def scan(self, variant, first_day, last_day):
            out, meta = super().scan(variant, first_day, last_day)
            return {d: (1.0, 0.0, 5.0) for d in out}, meta

    FakeEv.gain_fn = staticmethod(lambda v, d: 0.0)
    monkeypatch.setattr(rs, "ResidualEvaluator", Perfect)
    res = rs.run_residual_search({}, {}, {}, variants=[VA, VB], **SHORT, **SMALL)
    for f in res["folds"]:
        assert f["fallback_to_bcdf"] and f["test_scored"]
        assert f["test_skill_vs_bcdf"] == 0.0
        assert f["test_skill_vs_df"] == pytest.approx(1.0)
        assert f["selected_test_rank_pct"] is None  # competitors have no defined skill


def test_duplicate_detection_is_numeric_canonical(fake):
    fake(lambda v, d: 1.0)
    with pytest.raises(ValueError):
        run([VA, {"windows": [3], "hour_weight": 1}])
    # explicit-int identity form counts as the identity: no extra scan
    res = run([{"windows": [7], "hour_weight": 1}, VA])
    assert res["n_trials"] == 2
    assert len(FakeEv.seen["ranges"]) == 2


def test_huge_gains_do_not_raise(fake):
    fake(lambda v, d: (1e160 if d.toordinal() % 2 else -1e160) * (1 + v["windows"][0] / 100))
    res = run([VA, VB, ID])
    json.dumps(res, allow_nan=False)
    ov = res["overfit"]
    assert ov["reality_check"] is not None
    assert ov["pbo"] is None or isinstance(ov["pbo"], dict)
    assert ov["dsr"] is None or isinstance(ov["dsr"], dict)


def test_pbo_helper_row_limit_gives_none_with_reason(fake):
    fake(lambda v, d: (1.0 + d.toordinal() % 3 + v["windows"][0]) if d >= date(2023, 12, 1) else None)
    res = run([VA, VB, ID], min_days=5, pbo_splits=16)
    ov = res["overfit"]
    assert ov["common_days"] == 31  # >= max(30, 16) but < 2 * 16 rows
    assert ov["reality_check"] is not None
    assert ov["pbo"] is None and "pbo" in ov["reason"]
