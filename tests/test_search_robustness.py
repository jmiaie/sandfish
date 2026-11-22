"""Search robustness: valid= grids, not-scorable results, content heads,
holdout dedupe and guard, and the overfitting statistics."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from sandfish.core.orchestrator import create_orchestrator
from sandfish.search import (
    TEST,
    TRAIN,
    EvaluationResult,
    HoldoutLockedError,
    LedgerIntegrityError,
    TrialLedger,
    as_evaluator,
    dedupe_by_metadata,
    dedupe_by_score,
    deflated_sharpe_ratio,
    evaluate_and_record,
    expected_max_sharpe,
    holdout_report,
    iter_grid,
    ledger_deflated_sharpe,
    make_walk_forward_folds,
    overfit_report,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    run_grid,
    run_search,
    sharpe_stats,
    summarize,
    validate_holdout,
    walk_forward,
)
from sandfish.search.agents import SearchContext
from sandfish.search.overfit import spearman

SPACE = {"a": [0, 1, 2, 3], "b": [0, 1]}


def _ctx(ev, led):
    return SearchContext(space={}, evaluator=as_evaluator(ev), ledger=led)


def _score(v, split):
    return {"score": v["a"] + v["b"], "n": 10}


def _close(a, b, rel=1e-12):
    assert a == pytest.approx(b, rel=rel, abs=1e-15), (a, b)


# ===== 1. run_grid valid= =====


def test_iter_grid_valid_filters_and_keeps_order():
    space = {"fast": [1, 2, 3], "slow": [2, 3]}
    got = list(iter_grid(space, valid=lambda v: v["fast"] < v["slow"]))
    assert got == [{"fast": 1, "slow": 2}, {"fast": 1, "slow": 3}, {"fast": 2, "slow": 3}]
    assert len(list(iter_grid(space))) == 6  # default unchanged


def test_run_grid_skips_invalid_without_recording_them():
    calls = []

    def ev(v, split):
        calls.append(v)
        return {"score": v["slow"] - v["fast"], "n": 5}

    led = TrialLedger()
    out = run_grid({"fast": [1, 2, 3], "slow": [2, 3]}, ev, led, valid=lambda v: v["fast"] < v["slow"])
    assert out["grid_size"] == 6 and out["valid_size"] == 3 and out["skipped_invalid"] == 3
    assert len(calls) == 3 and led.n_trials == 3 and led.n_unique == 3
    assert {r["key"] for r in led.rows()} == {
        json.dumps(v, sort_keys=True, separators=(",", ":")) for v in calls}
    assert out["counts"]["by_split"]["train"]["unique_variants"] == 3


def test_run_grid_without_valid_is_unchanged():
    led = TrialLedger()
    out = run_grid(SPACE, _score, led)
    assert led.n_trials == 8 and out["valid_size"] == 8 and out["skipped_invalid"] == 0


def test_valid_grid_with_exact_n_for_a_large_space():
    # 20 x 54 x 8 = 8,640 raw combinations; one of 8 values of c is invalid: 7,560 valid.
    space = {"a": list(range(20)), "b": list(range(54)), "c": list(range(8))}
    led = TrialLedger()
    out = run_grid(space, lambda v, s: {"score": v["a"] - v["b"] + v["c"], "n": 50}, led,
                   valid=lambda v: v["c"] != 0)
    assert out["grid_size"] == 8640 and out["valid_size"] == 7560
    assert led.n_trials == 7560 and led.n_unique == 7560
    assert led.verify()["ok"]


def test_valid_predicate_stop_and_cap_behaviour():
    led = TrialLedger()
    stop_after = {"n": 0}

    def stop():
        stop_after["n"] += 1
        return stop_after["n"] > 2

    out = run_grid(SPACE, _score, led, valid=lambda v: v["a"] > 0, should_stop=stop)
    assert out["stopped"] and led.n_trials == 2 and out["valid_size"] == 6
    with pytest.raises(ValueError, match="max_variants"):  # cap is on the raw grid
        run_grid(SPACE, _score, TrialLedger(), max_variants=7, valid=lambda v: v["a"] == 0)


# ===== 2. not scorable =====


def test_backward_compatible_construction_and_dict():
    r = EvaluationResult(1.5, 10, {"x": 1})
    assert r.scorable and r.to_dict() == {"score": 1.5, "n": 10, "metadata": {"x": 1}}
    assert EvaluationResult.coerce({"score": 2, "n": 3}).score == 2.0


def test_not_scorable_states_coerce():
    for value in (
        EvaluationResult.not_scorable(n=3, reason="too few trades"),
        {"score": None, "n": 3},
        {"scorable": False, "n": 3, "reason": "few"},
    ):
        r = EvaluationResult.coerce(value)
        assert r.score is None and not r.scorable and r.n == 3
    assert EvaluationResult.not_scorable(n=1, reason="x", trades=1).metadata == {
        "not_scorable_reason": "x", "trades": 1}
    with pytest.raises(ValueError):
        EvaluationResult.coerce({"n": 3})  # still needs a score or scorable=False
    with pytest.raises(ValueError):
        EvaluationResult.coerce({"score": "x"})
    with pytest.raises(ValueError):
        EvaluationResult.coerce({"score": float("nan")})


def _flaky(v, split):
    if v["a"] == 3:  # the would-be winner has too few observations
        return EvaluationResult.not_scorable(n=2, reason="too few trades")
    return {"score": v["a"] + v["b"], "n": 10}


def test_not_scorable_is_counted_but_never_ranked():
    led = TrialLedger()
    out = run_grid(SPACE, _flaky, led)
    assert led.n_trials == 8 and led.n_unique == 8  # counted in N
    c = led.counts()
    assert c["n_not_scorable"] == 2 and c["not_scorable_by_split"] == {"train": 2, "test": 0}
    assert c["by_split"]["train"]["errors"] == 0
    best = led.best()
    assert best["variant"] == {"a": 2, "b": 1} and best["result"]["score"] == 3.0
    assert all(r["result"]["score"] is not None for r in led.top(100))
    assert len(led.top(100)) == 6
    assert out["best_train"]["score"] == 3.0
    row = next(r for r in led.rows() if r["variant"] == {"a": 3, "b": 1})
    assert row["result"] == {"score": None, "n": 2, "metadata": {"not_scorable_reason": "too few trades"}}


def test_not_scorable_repeat_is_cached_and_survives_reload(tmp_path):
    p = tmp_path / "l.jsonl"
    led = TrialLedger(p)
    ctx = _ctx(_flaky, led)
    a = evaluate_and_record(ctx, {"a": 3, "b": 0}, source="t")
    b = evaluate_and_record(ctx, {"a": 3, "b": 0}, source="t")
    assert a["result"]["score"] is None and not a["cached"] and b["cached"] and b["result"]["score"] is None
    assert led.n_trials == 2 and led.n_evaluator_calls == 1
    again = TrialLedger(p)
    assert again.counts()["n_not_scorable"] == 2 and again.best() is None and again.verify()["ok"]


def test_not_scorable_does_not_steer_the_swarm_or_hold_out(tmp_path):
    o = create_orchestrator(str(tmp_path / "v"))
    led = TrialLedger()
    try:
        out = asyncio.run(run_search(o, space=SPACE, evaluator=_flaky, ledger=led, num_agents=4,
                                     max_rounds=6, seed=3))
    finally:
        o.close()
    assert out["status"] == "completed" and led.n_trials == 24
    assert out["best_train"]["variant"]["a"] != 3
    res = validate_holdout(led, _flaky, top_k=10)
    assert all(r["variant"]["a"] != 3 for r in res)  # unscorable variants are never finalists
    t = validate_holdout(TrialLedger(), _flaky, variants=[{"a": 3, "b": 0}])
    assert t[0]["test_score"] is None and t[0]["test_error"] is None


# ===== 3. content hash =====


ROW_KEYS = {"seq", "ts", "run_id", "source", "round", "split", "evaluator", "key", "variant",
            "result", "error", "cached", "prev", "hash"}


def test_default_rows_keep_their_shape_and_verify():
    led = TrialLedger()
    run_grid(SPACE, _score, led)
    assert all(set(r) == ROW_KEYS for r in led.rows())
    assert led.verify()["ok"] and len(led.verify().keys()) == 3


def test_identical_runs_have_identical_content_heads_but_different_heads(tmp_path):
    def build(path):
        led = TrialLedger(path)
        run_grid(SPACE, _score, led)
        validate_holdout(led, _score, top_k=2)
        return led

    a, b = build(tmp_path / "a.jsonl"), build(tmp_path / "b.jsonl")
    assert a.head != b.head  # timestamps differ, so the chain hashes do
    assert a.content_head() == b.content_head()
    assert TrialLedger(tmp_path / "a.jsonl").content_head() == a.content_head()  # survives reload
    c = TrialLedger()
    run_grid(SPACE, lambda v, s: {"score": v["a"] + v["b"] + (0.5 if v == {"a": 0, "b": 0} else 0), "n": 10}, c)
    validate_holdout(c, _score, top_k=2)
    assert c.content_head() != a.content_head()
    assert TrialLedger().content_head() == "0" * 64


def test_swarm_runs_with_the_same_seed_share_a_content_head(tmp_path):
    def swarm(tag):
        o = create_orchestrator(str(tmp_path / f"v_{tag}"))
        led = TrialLedger()
        try:
            asyncio.run(run_search(o, space=SPACE, evaluator=_score, ledger=led, num_agents=3,
                                   max_rounds=4, seed=11))
        finally:
            o.close()
        return led

    a, b = swarm("a"), swarm("b")
    assert a.rows()[0]["run_id"] != b.rows()[0]["run_id"]  # simulation ids differ
    assert a.content_head() == b.content_head() and a.head != b.head
    s = summarize(a)
    assert s["content_head"] == a.content_head() and s["head"] == a.head


# ===== 4. holdout dedupe =====


def _dup_ledger():
    # Variants a=3,2 produce the same trades (same fingerprint); b only changes the score.
    table = {
        (3, 0): (9.0, "T1"), (3, 1): (8.0, "T1"), (2, 1): (7.0, "T1"),
        (1, 1): (6.0, "T2"), (0, 1): (5.0, "T3"), (2, 0): (4.0, "T4"),
    }

    def ev(v, split):
        score, fp = table[(v["a"], v["b"])]
        return {"score": score + (0.1 if split == TEST else 0), "n": 20, "fingerprint": fp}

    led = TrialLedger()
    for a, b in table:
        evaluate_and_record(_ctx(ev, led), {"a": a, "b": b}, source="t")
    return led, ev


def test_validate_holdout_dedupes_by_metadata_fingerprint():
    led, ev = _dup_ledger()
    plain = validate_holdout(led, ev, top_k=3)
    assert [r["variant"] for r in plain] == [{"a": 3, "b": 0}, {"a": 3, "b": 1}, {"a": 2, "b": 1}]
    assert plain.skipped == []

    led2, ev = _dup_ledger()
    res = validate_holdout(led2, ev, top_k=3, dedupe_key=dedupe_by_metadata("fingerprint"))
    assert [r["variant"] for r in res] == [{"a": 3, "b": 0}, {"a": 1, "b": 1}, {"a": 0, "b": 1}]
    assert [s["variant"] for s in res.skipped] == [{"a": 3, "b": 1}, {"a": 2, "b": 1}]
    assert all(s["duplicate_of"] == {"a": 3, "b": 0} for s in res.skipped)
    assert led2.counts()["by_split"]["test"]["trials"] == 3  # skipped ones never touch the test split
    assert isinstance(res, list) and json.dumps(list(res))


def test_dedupe_by_score_and_params_signature():
    def ev(v, split):
        return {"score": 1.0 if v["a"] in (0, 1) else float(v["a"]), "n": 9}

    led = TrialLedger()
    ctx = _ctx(ev, led)
    for a in (0, 1, 2, 3):
        for b in (0, 1):
            evaluate_and_record(ctx, {"a": a, "b": b}, source="t")
    # a=0 and a=1 tie at 1.0 on every b: with params=["b"], same score AND same b collapse.
    res = validate_holdout(led, ev, top_k=8, dedupe_key=dedupe_by_score(params=["b"]))
    assert len(res) == 6 and len(res.skipped) == 2
    res2 = validate_holdout(TrialLedger(), ev)
    assert res2 == [] and res2.skipped == []
    key = dedupe_by_score()
    assert key({"result": {"score": None, "n": 1}, "variant": {}}) is None
    with pytest.raises(ValueError):
        dedupe_by_metadata()
    assert dedupe_by_metadata("missing")({"result": {"metadata": {}}}) is None


def test_explicit_variants_are_not_deduped():
    led, ev = _dup_ledger()
    vs = [{"a": 3, "b": 0}, {"a": 3, "b": 1}]
    res = validate_holdout(led, ev, variants=vs, dedupe_key=dedupe_by_metadata("fingerprint"))
    assert [r["variant"] for r in res] == vs and res.skipped == []


# ===== 5. holdout guard =====


class _Counting:
    name = "counting"

    def __init__(self):
        self.calls = []

    def evaluate(self, variant, split):
        self.calls.append((dict(variant), split))
        return {"score": variant["a"] + variant["b"], "n": 10}


def test_unguarded_ledger_is_unchanged_and_logs_nothing_extra():
    led = TrialLedger()
    ev = _Counting()
    run_grid(SPACE, ev, led)
    validate_holdout(led, ev, top_k=2)
    validate_holdout(led, ev, top_k=2)  # repeats stay allowed without a guard
    assert led.events() == [] and led.holdout_status()["state"] == "none"
    assert len(led.verify()) == 3 and led.verify()["rows"] == led.n_trials


def test_locked_holdout_blocks_the_search_before_the_evaluator_runs(tmp_path):
    led = TrialLedger(tmp_path / "g.jsonl")
    ev = _Counting()
    st = led.lock_holdout(reason="unit test")
    assert st["state"] == "locked" and st["clean"] and led.lock_holdout()["locked_seq"] == st["locked_seq"]
    assert [e["event"] for e in led.events()] == ["holdout_locked"]  # idempotent: one lock event
    run_grid(SPACE, ev, led)
    assert {s for _, s in ev.calls} == {TRAIN}
    ctx = _ctx(ev, led)
    n_calls = len(ev.calls)
    with pytest.raises(HoldoutLockedError):
        evaluate_and_record(ctx, {"a": 3, "b": 1}, source="sneaky", split=TEST)
    with pytest.raises(HoldoutLockedError):  # not even with the magic source name
        evaluate_and_record(ctx, {"a": 3, "b": 1}, source="holdout", split=TEST)
    assert len(ev.calls) == n_calls  # the evaluator never saw the test split
    with pytest.raises(HoldoutLockedError):
        led.record(source="x", variant={"a": 0, "b": 0}, split=TEST, error="boom")
    denied = led.events("holdout_denied")
    assert len(denied) == 2 and denied[0]["data"]["source"] == "sneaky"
    assert led.holdout_status()["denied_attempts"] == 2
    assert led.counts()["by_split"]["test"]["trials"] == 0
    assert led.verify()["ok"]


def test_validate_holdout_consumes_the_lock_once(tmp_path):
    p = tmp_path / "g.jsonl"
    led = TrialLedger(p)
    ev = _Counting()
    out = run_grid(SPACE, ev, led, lock_holdout=True)
    assert out["holdout_guard"]["state"] == "locked"
    res = validate_holdout(led, ev, top_k=2)
    assert len(res) == 2 and led.holdout_status()["state"] == "consumed"
    names = [e["event"] for e in led.events()]
    assert names == ["holdout_locked", "holdout_unlocked", "holdout_consumed"]
    assert led.events("holdout_unlocked")[0]["data"]["finalists"] == [
        json.dumps(r["variant"], sort_keys=True, separators=(",", ":")) for r in res]
    with pytest.raises(HoldoutLockedError, match="already consumed"):
        validate_holdout(led, ev, top_k=2)
    with pytest.raises(HoldoutLockedError):
        led.lock_holdout()
    again = validate_holdout(led, ev, top_k=1, allow_repeat=True)
    assert len(again) == 1
    assert [e["event"] for e in led.events()][-2:] == ["holdout_reopened", "holdout_consumed"]
    assert led.holdout_status()["validations"] == 2
    assert led.verify()["ok"]
    # State and log survive a reload; the test rows are in the ledger.
    re = TrialLedger(p)
    assert re.holdout_status()["state"] == "consumed" and re.holdout_status()["validations"] == 2
    assert re.counts()["by_split"]["test"]["trials"] == 3
    assert len(re) == len(led) and re.head == led.head


def test_lock_after_peeking_is_not_clean_and_interrupted_validation_stays_closed(tmp_path):
    led = TrialLedger()
    ev = _Counting()
    ctx = _ctx(ev, led)
    evaluate_and_record(ctx, {"a": 1, "b": 1}, source="peek", split=TEST)
    st = led.lock_holdout()
    assert st["state"] == "locked" and st["test_rows_before_lock"] == 1 and st["clean"] is False

    p = tmp_path / "crash.jsonl"
    led2 = TrialLedger(p)
    led2.lock_holdout()
    cm = led2.holdout_access([{"a": 1, "b": 1}])
    cm.__enter__()  # simulate a crash: never exits
    assert led2.holdout_status()["state"] == "open"
    with pytest.raises(HoldoutLockedError, match="already running"):
        with led2.holdout_access():
            pass
    assert TrialLedger(p).holdout_status()["state"] == "consumed"  # conservative on reload
    cm.__exit__(None, None, None)


def test_failed_validation_still_consumes_the_lock():
    led = TrialLedger()
    led.lock_holdout()

    def boom(v, split):
        raise RuntimeError("data gone")

    res = validate_holdout(led, boom, variants=[{"a": 0, "b": 0}])
    assert res[0]["test_error"] and led.holdout_status()["state"] == "consumed"


def test_swarm_cannot_reach_the_locked_holdout(tmp_path):
    o = create_orchestrator(str(tmp_path / "v"))
    led = TrialLedger()
    ev = _Counting()
    try:
        out = asyncio.run(run_search(o, space=SPACE, evaluator=ev, ledger=led, num_agents=3,
                                     max_rounds=3, seed=1, lock_holdout=True))
    finally:
        o.close()
    assert out["holdout_guard"]["state"] == "locked" and {s for _, s in ev.calls} == {TRAIN}
    assert led.n_trials == 9 and led.verify()["ok"] and led.verify()["rows"] == 10


def test_events_are_tamper_evident_and_not_trials(tmp_path):
    p = tmp_path / "e.jsonl"
    led = TrialLedger(p)
    run_grid({"a": [0, 1]}, lambda v, s: {"score": v["a"], "n": 5}, led, lock_holdout=True)
    assert len(led) == 2 and led.n_trials == 2 and len(led.rows()) == 2
    lines = p.read_text().splitlines()
    rows = [json.loads(x) for x in lines]
    assert "event" in rows[0] and rows[0]["seq"] == 0 and rows[1]["seq"] == 1
    rows[0]["data"]["test_rows_before_lock"] = 5  # rewrite history
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(p)
    with pytest.raises(ValueError):
        led.record_event("")


# ===== 6. statistics against independently computed references =====
# Reference values come from scipy.stats.norm.ppf / numpy / scipy.stats.rankdata
# implementations written separately from sandfish.search.overfit.


@pytest.mark.parametrize("n,var,ref", [
    (2, 0.01, 0.05197553442805939),
    (10, 0.04, 0.31491966026915),
    (100, 0.0025, 0.12653014466008425),
    (8640, 0.0009, 0.11474315521140549),
    (7560, 0.0009, 0.11375156216465994),
    (50, 1.0, 2.2763030934203483),
])
def test_expected_max_sharpe_matches_reference(n, var, ref):
    _close(expected_max_sharpe(n, var), ref, rel=1e-10)


def test_expected_max_sharpe_properties_and_errors():
    assert expected_max_sharpe(1, 0.5) == 0.0 and expected_max_sharpe(5, 0.0) == 0.0
    vals = [expected_max_sharpe(n, 0.01) for n in (2, 5, 20, 100, 1000)]
    assert vals == sorted(vals)  # more trials, higher bar
    for bad in ((0, 0.1), (3, -1.0), (3, float("nan"))):
        with pytest.raises(ValueError):
            expected_max_sharpe(*bad)


@pytest.mark.parametrize("args,ref", [
    ((0.1, 250, 0.0, 0.0, 3.0), 0.942261266719699),
    ((0.1, 250, 0.05, -0.5, 6.0), 0.7779920904212589),
    ((0.25, 60, 0.1, 0.3, 4.5), 0.8777991278563335),
    ((0.05, 500, 0.05, 0.0, 3.0), 0.5),
])
def test_probabilistic_sharpe_ratio_matches_reference(args, ref):
    sr, t, bench, skew, kurt = args
    _close(probabilistic_sharpe_ratio(sr, t, bench, skew=skew, kurtosis=kurt), ref, rel=1e-10)


def test_psr_validation():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, 1)
    with pytest.raises(ValueError):  # 1 - skew*sr + ... <= 0
        probabilistic_sharpe_ratio(2.0, 50, skew=10.0, kurtosis=1.0)


TRIAL_SRS = [0.02, 0.05, -0.01, 0.08, 0.03, 0.0, 0.06, 0.04, -0.03, 0.07]


def test_deflated_sharpe_ratio_matches_reference():
    out = deflated_sharpe_ratio(0.08, 250, 10, trial_srs=TRIAL_SRS, skew=-0.4, kurtosis=5.0)
    _close(out["var_sr"], 0.001298888888888889)
    _close(out["sr0"], 0.056748682007137376, rel=1e-10)
    _close(out["dsr"], 0.640595777785212, rel=1e-10)
    _close(out["psr_vs_zero"], 0.8922934484613194, rel=1e-10)
    assert out["n_trials"] == 10 and out["n_obs"] == 250
    same = deflated_sharpe_ratio(0.08, 250, 10, var_sr=0.001298888888888889, skew=-0.4, kurtosis=5.0)
    _close(same["dsr"], out["dsr"])


def test_dsr_deflation_grows_with_n_and_validates():
    d = [deflated_sharpe_ratio(0.1, 250, n, var_sr=0.002)["dsr"] for n in (1, 10, 100, 10000)]
    assert d == sorted(d, reverse=True) and d[0] == pytest.approx(
        probabilistic_sharpe_ratio(0.1, 250))
    for kw in ({}, {"var_sr": 0.1, "trial_srs": [0.1, 0.2]}):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(0.1, 250, 5, **kw)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.1, 250, 5, trial_srs=[0.1])


def test_sharpe_stats_matches_reference():
    x = [0.01 * math.sin(0.7 * i) + 0.004 * ((i * 5) % 7 - 3) + 0.001 for i in range(60)]
    s = sharpe_stats(x)
    _close(s["sharpe"], 0.10936713277291744, rel=1e-10)
    _close(s["skew"], -0.04576927799230433, rel=1e-9)
    _close(s["kurtosis"], 2.279931687579284, rel=1e-10)
    assert s["n_obs"] == 60
    for bad in ([1.0, 2.0], [1.0, 1.0, 1.0], [1.0, float("inf"), 2.0]):
        with pytest.raises(ValueError):
            sharpe_stats(bad)


def _matrix(t, n):
    return [[round(0.01 * math.sin(1.3 * i * (j + 1) + j) + 0.002 * (((i * 7 + j * 13) % 11) - 5) / 5
                   + 0.0005 * j, 8) for j in range(n)] for i in range(t)]


@pytest.mark.parametrize("t,n,s,ref", [
    (48, 6, 8, {"pbo": 0.37142857142857144, "lm": 0.0941900007960463, "ls": 1.3912791231034418,
                "loss": 0.2857142857142857, "slope": -1.2381832619052178, "k": 70}),
    (53, 5, 4, {"pbo": 0.16666666666666666, "lm": 0.9202434863103744, "ls": 1.2923479824907333,
                "loss": 0.16666666666666666, "slope": -0.583735453590431, "k": 6}),  # 53 -> 52 rows
    (64, 12, 16, {"pbo": 0.17536907536907537, "lm": 1.025581430570107, "ls": 0.8793838457338081,
                  "loss": 0.011888111888111888, "slope": -1.4234219873934602, "k": 12870}),
])
def test_pbo_matches_independent_cscv(t, n, s, ref):
    out = probability_of_backtest_overfitting(_matrix(t, n), n_splits=s)
    _close(out["pbo"], ref["pbo"], rel=1e-12)
    _close(out["logit_mean"], ref["lm"], rel=1e-9)
    _close(out["logit_std"], ref["ls"], rel=1e-9)
    _close(out["prob_oos_loss"], ref["loss"], rel=1e-12)
    _close(out["degradation_slope"], ref["slope"], rel=1e-8)
    assert out["n_combinations"] == ref["k"] == out["n_combinations_total"]
    assert out["n_strategies"] == n and out["n_splits"] == s and out["rows_used"] == (t // s) * s


def test_pbo_sampling_is_deterministic_and_validates():
    m = _matrix(64, 12)
    a = probability_of_backtest_overfitting(m, 16, max_combinations=500, seed=1)
    b = probability_of_backtest_overfitting(m, 16, max_combinations=500, seed=1)
    c = probability_of_backtest_overfitting(m, 16, max_combinations=500, seed=2)
    assert a == b and a["n_combinations"] == 500 and a["n_combinations_total"] == 12870
    assert abs(a["pbo"] - 0.17536907536907537) < 0.08 and c != a
    for kwargs in ({"n_splits": 3}, {"n_splits": 22}, {"n_splits": 0}):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(m, **kwargs)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0, 2.0]] * 5, 4)  # too few rows
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0]] * 40, 4)  # one variant
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0, 2.0], [1.0]] * 20, 4)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([], 4)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([[1.0, float("nan")]] * 40, 4)


def test_pbo_pure_noise_is_near_half_and_a_true_edge_is_low():
    import random
    rng = random.Random(5)
    noise = [[rng.gauss(0, 0.01) for _ in range(20)] for _ in range(320)]
    p_noise = probability_of_backtest_overfitting(noise, 8)["pbo"]
    edge = [[rng.gauss(0.004 if j == 0 else 0.0, 0.01) for j in range(20)] for _ in range(320)]
    p_edge = probability_of_backtest_overfitting(edge, 8)["pbo"]
    assert 0.25 < p_noise < 0.85 and p_edge < p_noise


def test_spearman_matches_scipy_with_ties():
    xs, ys = [1, 2, 2, 4, 5, 7, 7, 9], [3, 1, 4, 4, 8, 6, 9, 9]
    _close(spearman(xs, ys), 0.8902439024390245, rel=1e-12)
    assert spearman([1, 2], [1, 2]) is None and spearman([1, 1, 1], [1, 2, 3]) is None


# ===== 7. walk-forward =====


def test_walk_forward_fold_windows():
    rolling = make_walk_forward_folds(100, 40, 20)
    assert [(f.train, f.test) for f in rolling] == [
        ((0, 40), (40, 60)), ((20, 60), (60, 80)), ((40, 80), (80, 100))]
    anchored = make_walk_forward_folds(100, 40, 20, anchored=True)
    assert [(f.train, f.test) for f in anchored] == [
        ((0, 40), (40, 60)), ((0, 60), (60, 80)), ((0, 80), (80, 100))]
    emb = make_walk_forward_folds(100, 40, 20, embargo=5, step=10)
    assert emb[0].test == (45, 65) and emb[1].train == (10, 50) and emb[-1].test[1] <= 100
    assert all(f.train[1] + 5 <= f.test[0] for f in emb)
    assert all(f.test[1] <= 100 for f in rolling + anchored)
    assert rolling[0].as_dict() == {"index": 0, "train": [0, 40], "test": [40, 60]}
    with pytest.raises(ValueError):
        make_walk_forward_folds(50, 40, 20)
    with pytest.raises(ValueError):
        make_walk_forward_folds(100, 0, 20)


class _FoldEval:
    def evaluate_fold(self, variant, fold):
        if fold.index == 1:
            raise RuntimeError("missing data")
        if fold.index == 2:
            return EvaluationResult.not_scorable(n=1)
        return {"score": variant["a"] * (1 if fold.index == 0 else -1) + fold.index, "n": 20}


def test_walk_forward_summarises_and_logs_to_the_ledger():
    folds = make_walk_forward_folds(200, 60, 20)
    led = TrialLedger()
    out = walk_forward(_FoldEval(), {"a": 2}, folds, ledger=led)
    assert out["n_folds"] == len(folds) and out["n_errors"] == 1
    assert out["n_scorable"] == len(folds) - 2
    scores = [e["score"] for e in out["folds"] if e["score"] is not None]
    assert out["mean"] == pytest.approx(sum(scores) / len(scores))
    assert out["min"] == min(scores) and out["max"] == max(scores)
    assert 0.0 <= out["positive_share"] <= 1.0
    assert out["folds"][1]["error"].startswith("RuntimeError") and out["folds"][2]["score"] is None
    assert led.n_trials == 0 and [e["event"] for e in led.events()] == ["walk_forward"]  # not a trial
    assert led.verify()["ok"]
    rep = overfit_report(led)
    assert rep["walk_forward"][0]["n_folds"] == len(folds) and rep["walk_forward"][0]["variant"] == {"a": 2}


def test_walk_forward_accepts_callables_and_rejects_others():
    folds = make_walk_forward_folds(100, 40, 20)
    out = walk_forward(lambda v, f: {"score": float(f.index), "n": 5}, {"a": 1}, folds)
    assert out["mean"] == 1.0 and out["std"] == 1.0 and out["stability"] == 1.0
    assert out["positive_share"] == pytest.approx(2 / 3)
    with pytest.raises(TypeError):
        walk_forward(object(), {"a": 1}, folds)
    with pytest.raises(TypeError):
        walk_forward(_Counting(), {"a": 1}, folds)  # split-only evaluators have no folds
    with pytest.raises(ValueError):
        walk_forward(_FoldEval(), {"a": 1}, [])
    one = walk_forward(lambda v, f: {"score": 1.0}, {"a": 1}, folds[:1])
    assert one["std"] is None and one["stability"] is None
    none = walk_forward(lambda v, f: {"scorable": False}, {"a": 1}, folds)
    assert none["mean"] is None and none["n_scorable"] == 0


# ===== 8. ledger-level reports =====


def _sharpe_eval(v, split):
    base = {"a": 0.20, "b": 0.12, "c": 0.05, "d": -0.02}[v["k"]]
    shift = -0.08 if split == TEST else 0.0
    return {"score": base + shift, "n": 250, "skew": -0.3, "kurtosis": 4.0}


def test_ledger_deflated_sharpe_uses_the_ledger_n_including_unscorable():
    led = TrialLedger()
    run_grid({"k": ["a", "b", "c", "d", "e"]}, lambda v, s: (
        EvaluationResult.not_scorable(n=3) if v["k"] == "e" else _sharpe_eval(v, s)), led)
    out = ledger_deflated_sharpe(led, score_is_sharpe=True)
    assert out["available"] and out["n_trials"] == 5 and out["n_scorable"] == 4  # 'e' counts in N
    srs = [0.20, 0.12, 0.05, -0.02]
    ref = deflated_sharpe_ratio(0.20, 250, 5, trial_srs=srs, skew=-0.3, kurtosis=4.0)
    _close(out["dsr"], ref["dsr"])
    assert out["variant"] == {"k": "a"} and out["significant_95"] == (out["dsr"] >= 0.95)
    ann = ledger_deflated_sharpe(led, score_is_sharpe=True, periods_per_year=100, n_obs=500,
                                 skew=0.0, kurtosis=3.0, n_trials=50)
    ref2 = deflated_sharpe_ratio(0.02, 500, 50, trial_srs=[v / 10 for v in srs])
    _close(ann["dsr"], ref2["dsr"])
    assert ann["n_trials"] == 50 and ann["n_obs"] == 500
    with pytest.raises(ValueError, match="score_is_sharpe"):
        ledger_deflated_sharpe(led, score_is_sharpe=False)
    with pytest.raises(ValueError):
        ledger_deflated_sharpe(led, score_is_sharpe=True, periods_per_year=0)


def test_ledger_deflated_sharpe_degenerate_cases():
    led = TrialLedger()
    assert ledger_deflated_sharpe(led, score_is_sharpe=True)["available"] is False
    run_grid({"k": ["a", "b"]}, lambda v, s: {"score": 0.1}, led)  # no n -> needs n_obs
    out = ledger_deflated_sharpe(led, score_is_sharpe=True)
    assert out["available"] is False and "n_obs" in out["reason"]
    assert ledger_deflated_sharpe(led, score_is_sharpe=True, n_obs=100)["available"]


def test_holdout_report_and_summarize_overfit_section():
    led = TrialLedger()
    run_grid({"k": ["a", "b", "c", "d"]}, _sharpe_eval, led, lock_holdout=True)
    validate_holdout(led, _sharpe_eval, top_k=4)
    rep = holdout_report(led)
    assert rep["n_pairs"] == 4
    _close(rep["mean_degradation"], 0.08, rel=1e-9)
    assert rep["rank_correlation"] == pytest.approx(1.0) and rep["test_below_train_share"] == 1.0
    assert holdout_report(TrialLedger())["n_pairs"] == 0
    s = summarize(led, overfit={"score_is_sharpe": True})
    assert s["overfit"]["deflated_sharpe"]["available"] and s["overfit"]["holdout"]["n_pairs"] == 4
    assert s["overfit"]["holdout_guard"]["state"] == "consumed" and s["holdout_guard"]["clean"]
    assert "overfit" not in summarize(led)
    plain = summarize(led, overfit={})
    assert plain["overfit"]["deflated_sharpe"]["available"] is False
    assert summarize(led, overfit={"min_n": 1000, "score_is_sharpe": True})["overfit"][
        "deflated_sharpe"]["available"] is False  # explicit min_n overrides the default
