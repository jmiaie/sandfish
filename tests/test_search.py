"""Opt-in parameter search: evaluators, trial ledger, researcher agents."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from sandfish.core.orchestrator import SimulationConfig, SimulationStatus, create_orchestrator
from sandfish.search import (
    GENESIS,
    MULTIPLE_TESTING_NOTE,
    TEST,
    TRAIN,
    EvaluationResult,
    FunctionEvaluator,
    LedgerIntegrityError,
    ToyQuadraticEvaluator,
    TrialLedger,
    as_evaluator,
    attach_search,
    available_evaluators,
    evaluate_and_record,
    get_evaluator,
    grid_size,
    iter_grid,
    register_evaluator,
    register_search_agents,
    run_grid,
    run_search,
    summarize,
    unregister_evaluator,
    validate_holdout,
    validate_space,
    variant_key,
    verify_ledger_file,
)
from sandfish.search import evaluator as evaluator_mod
from sandfish.search.agents import SearchContext
from sandfish.search.evaluator import run_evaluation

SPACE = {"a": [0, 1, 2, 3], "b": [0, 1]}


def _score(v, split):
    return {"score": v["a"] + v["b"], "n": 10}


@pytest.fixture
def orch(tmp_path):
    o = create_orchestrator(str(tmp_path / "vault"))
    yield o
    o.close()


# ----- EvaluationResult / evaluator protocol -----


def test_coerce_mapping_moves_extras_into_metadata():
    r = EvaluationResult.coerce({"score": 2, "n": 5, "metadata": {"a": 1}, "hits": 3})
    assert r == EvaluationResult(score=2.0, n=5, metadata={"a": 1, "hits": 3})
    assert r.to_dict() == {"score": 2.0, "n": 5, "metadata": {"a": 1, "hits": 3}}
    same = EvaluationResult(score=1, n=None)
    assert EvaluationResult.coerce(same).score == 1.0


@pytest.mark.parametrize(
    "bad",
    [{"n": 1}, {"score": "1"}, {"score": True}, {"score": float("nan")},
     {"score": float("inf")}, {"score": 1, "n": 1.5}, {"score": 1, "n": True}, 3.0, None],
)
def test_coerce_rejects_bad_results(bad):
    with pytest.raises(ValueError):
        EvaluationResult.coerce(bad)


def test_as_evaluator_wraps_callables_and_rejects_others():
    ev = as_evaluator(_score)
    assert isinstance(ev, FunctionEvaluator) and ev.name == "_score"
    toy = ToyQuadraticEvaluator()
    assert as_evaluator(toy) is toy
    with pytest.raises(TypeError):
        as_evaluator(42)
    with pytest.raises(ValueError, match="split"):
        run_evaluation(ev, {"a": 0, "b": 0}, "validation")


def test_registry_register_replace_unregister():
    assert "toy" in available_evaluators()
    register_evaluator("unit_const", lambda value=1.0: (lambda v, s: {"score": value}))
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_evaluator("unit_const", lambda: _score)
        register_evaluator("unit_const", lambda value=2.0: (lambda v, s: {"score": value}), replace=True)
        ev = get_evaluator("unit_const", value=5.0)
        assert run_evaluation(ev, {}, TRAIN).score == 5.0
        with pytest.raises(ValueError):
            register_evaluator("", lambda: _score)
        with pytest.raises(TypeError):
            register_evaluator("unit_bad", 3)  # type: ignore[arg-type]
    finally:
        unregister_evaluator("unit_const")
    with pytest.raises(KeyError, match="unknown evaluator"):
        get_evaluator("unit_const")


def test_entry_point_plugins_are_loaded_by_name(monkeypatch):
    class _EP:
        name = "plugin_eval"

        def load(self):
            return lambda scale=1: FunctionEvaluator(lambda v, s: {"score": scale * v["a"]})

    monkeypatch.setattr(evaluator_mod, "_entry_points", lambda: {"plugin_eval": _EP()})
    assert "plugin_eval" in available_evaluators()
    ev = get_evaluator("plugin_eval", scale=3)
    assert run_evaluation(ev, {"a": 2}, TEST).score == 6.0


def test_entry_point_group_is_read_without_error():
    # Real importlib.metadata lookup; no plugins are installed in CI.
    assert isinstance(evaluator_mod._entry_points(), dict)


def test_toy_evaluator_is_deterministic_and_split_specific():
    toy = ToyQuadraticEvaluator(seed=1)
    v = {"x": 1, "y": -2}
    a, b = toy.evaluate(v, TRAIN), ToyQuadraticEvaluator(seed=1).evaluate(v, TRAIN)
    assert a == b
    assert toy.evaluate(v, TEST).score != a.score
    exact = ToyQuadraticEvaluator(noise=0).evaluate({"x": 0, "y": 0}, TRAIN)
    assert exact.score == -5.0 and exact.n == 100 and exact.metadata["distance2"] == 5.0
    with pytest.raises(ValueError, match="missing"):
        toy.evaluate({"x": 1}, TRAIN)
    with pytest.raises(ValueError):
        ToyQuadraticEvaluator(noise=-1)


# ----- search space -----


@pytest.mark.parametrize(
    "bad",
    [{}, [], {"a": []}, {"a": "abc"}, {"": [1]}, {"a": [[1]]}, {"a": [float("nan")]},
     {"a": list(range(1001))}, {f"p{i}": [1] for i in range(33)}, {1: [1]}],
)
def test_validate_space_rejects(bad):
    with pytest.raises(ValueError):
        validate_space(bad)


def test_grid_helpers():
    space = validate_space({"b": (1, 2), "a": [None, True, "x"]})
    assert grid_size(space) == 6
    grid = list(iter_grid(space))
    assert grid[0] == {"a": None, "b": 1} and grid[-1] == {"a": "x", "b": 2}
    assert len({variant_key(v) for v in grid}) == 6
    assert variant_key({"b": 1, "a": 2}) == variant_key({"a": 2, "b": 1})


# ----- ledger -----


def _ctx(ledger, fn=_score):
    return SearchContext(space=validate_space(SPACE), evaluator=as_evaluator(fn), ledger=ledger)


def test_ledger_counts_every_evaluation_including_duplicates(tmp_path):
    led = TrialLedger(tmp_path / "l.jsonl")
    calls = []

    def fn(v, s):
        calls.append((dict(v), s))
        return {"score": v["a"], "n": 3}

    ctx = _ctx(led, fn)
    for v in [{"a": 1, "b": 0}, {"a": 1, "b": 0}, {"b": 0, "a": 1}, {"a": 2, "b": 0}]:
        evaluate_and_record(ctx, v, source="t")
    evaluate_and_record(ctx, {"a": 1, "b": 0}, source="t", split=TEST)
    c = led.counts()
    assert c["n_trials"] == 5 and c["n_evaluator_calls"] == 3 and c["n_unique"] == 2
    assert c["by_split"][TRAIN] == {"trials": 4, "evaluator_calls": 2, "cached": 2,
                                    "errors": 0, "unique_variants": 2}
    assert c["by_split"][TEST]["trials"] == 1
    assert len(calls) == 3 and led.n_trials == 5 and led.n_unique == 2 and led.n_evaluator_calls == 3
    rows = led.rows()
    assert [r["seq"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["prev"] == GENESIS and rows[1]["prev"] == rows[0]["hash"]
    assert [r["cached"] for r in rows] == [False, True, True, False, False]
    assert led.verify()["ok"] and len(led) == 5 and led.head == rows[-1]["hash"]
    # On disk too, line for line.
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in lines] == rows


def test_ledger_records_evaluator_errors_as_trials():
    led = TrialLedger()

    def boom(v, s):
        raise RuntimeError("no data for " + str(v["a"]))

    ctx = _ctx(led, boom)
    evaluate_and_record(ctx, {"a": 1, "b": 0}, source="t")
    evaluate_and_record(ctx, {"a": 1, "b": 0}, source="t")  # errors are not cached
    ctx2 = _ctx(led, lambda v, s: {"no_score": 1})
    evaluate_and_record(ctx2, {"a": 2, "b": 0}, source="t")
    c = led.counts()
    assert c["n_trials"] == 3 and c["n_evaluator_calls"] == 3 and c["by_split"][TRAIN]["errors"] == 3
    assert led.rows()[0]["error"] == "RuntimeError: no data for 1"
    assert led.best() is None and led.lookup({"a": 1, "b": 0}) is None


def test_unrecordable_results_and_variants():
    led = TrialLedger()
    ctx = _ctx(led, lambda v, s: {"score": 1.0, "metadata": {"bad": float("nan")}})
    row = evaluate_and_record(ctx, {"a": 1, "b": 0}, source="t")
    assert row["result"] is None and "metadata" in row["error"]
    with pytest.raises(ValueError, match="repair"):
        evaluate_and_record(_ctx(led), {"a": float("nan")}, source="t")
    # Objects in metadata are stored as strings, not rejected.
    ok = evaluate_and_record(_ctx(led, lambda v, s: {"score": 2, "metadata": {"o": object}}),
                             {"a": 2, "b": 0}, source="t")
    assert ok["result"]["metadata"]["o"].startswith("<class")
    assert led.n_trials == 2


def test_record_failure_after_evaluation_is_still_counted(monkeypatch):
    led = TrialLedger()
    real = led.record
    calls = {"n": 0}

    def flaky_record(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("disk said no")
        return real(**kw)

    monkeypatch.setattr(led, "record", flaky_record)
    row = evaluate_and_record(_ctx(led), {"a": 1, "b": 1}, source="t")
    assert row["error"] == "unrecordable result: disk said no" and led.n_trials == 1


def test_cache_is_per_evaluator():
    led = TrialLedger()
    evaluate_and_record(_ctx(led, _score), {"a": 1, "b": 1}, source="t")
    other = SearchContext(space=validate_space(SPACE), evaluator=FunctionEvaluator(_score, "other"),
                          ledger=led)
    row = evaluate_and_record(other, {"a": 1, "b": 1}, source="t")
    assert row["cached"] is False and led.counts()["n_evaluator_calls"] == 2


def test_expected_head_catches_tail_truncation(tmp_path):
    p = tmp_path / "l.jsonl"
    led = TrialLedger(p)
    run_grid(SPACE, _score, led)
    head = led.head
    assert led.verify(expected_head=head)["ok"]
    assert verify_ledger_file(p, expected_head=head)["ok"]
    p.write_text("\n".join(p.read_text().splitlines()[:-2]) + "\n")
    assert verify_ledger_file(p)["ok"]  # undetectable from the file alone
    report = verify_ledger_file(p, expected_head=head)
    assert not report["ok"] and "not in the chain" in report["error"]
    assert not TrialLedger(p).verify(expected_head=head)["ok"]


def test_torn_last_line_is_reported(tmp_path):
    p = tmp_path / "l.jsonl"
    run_grid({"a": [1, 2]}, lambda v, s: {"score": v["a"]}, TrialLedger(p))
    p.write_text(p.read_text() + '{"seq": 2, "half')
    report = verify_ledger_file(p)
    assert not report["ok"] and "incomplete last line" in report["error"] and report["rows"] == 2


def test_ledger_record_validation():
    led = TrialLedger()
    with pytest.raises(ValueError, match="exactly one"):
        led.record(source="t", variant={})
    with pytest.raises(ValueError, match="exactly one"):
        led.record(source="t", variant={}, result=EvaluationResult(1.0), error="x")
    with pytest.raises(ValueError, match="split"):
        led.record(source="t", variant={}, result=EvaluationResult(1.0), split="dev")
    assert led.verify() == {"ok": True, "rows": 0, "error": None}


def test_ledger_reopen_appends_and_keeps_chain(tmp_path):
    p = tmp_path / "l.jsonl"
    a = TrialLedger(p)
    run_grid(SPACE, _score, a)
    b = TrialLedger(p)
    assert b.counts() == a.counts() and b.head == a.head
    assert b.lookup({"a": 3, "b": 1}, evaluator="_score").score == 4.0
    assert b.lookup({"a": 3, "b": 1}) is None  # cache is per evaluator name
    evaluate_and_record(_ctx(b), {"a": 3, "b": 1}, source="again")
    assert b.n_trials == 9 and b.rows()[-1]["cached"] is True
    assert verify_ledger_file(p) == {"ok": True, "rows": 9, "error": None, "head": b.head}


@pytest.mark.parametrize("tamper", ["edit", "delete", "swap", "garbage"])
def test_ledger_detects_tampering(tmp_path, tamper):
    p = tmp_path / "l.jsonl"
    run_grid(SPACE, _score, TrialLedger(p))
    lines = p.read_text().splitlines()
    if tamper == "edit":
        row = json.loads(lines[3])
        row["result"]["score"] = 99.0
        lines[3] = json.dumps(row)
    elif tamper == "delete":
        del lines[3]
    elif tamper == "swap":
        lines[2], lines[3] = lines[3], lines[2]
    else:
        lines[3] = "{not json"
    p.write_text("\n".join(lines) + "\n")
    report = verify_ledger_file(p)
    assert not report["ok"]
    assert report["error"].startswith(("row ", "line "))
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(p)


def test_ledger_verify_notices_file_diverging_from_memory(tmp_path):
    p = tmp_path / "l.jsonl"
    led = TrialLedger(p)
    run_grid({"a": [1, 2]}, lambda v, s: {"score": v["a"]}, led)
    p.write_text(p.read_text().splitlines()[0] + "\n")  # truncated behind our back
    report = led.verify()
    assert not report["ok"] and report["file"]["rows"] == 1
    assert verify_ledger_file(tmp_path / "missing.jsonl")["ok"]


def test_best_and_top_respect_min_n_and_split():
    led = TrialLedger()
    ctx = _ctx(led, lambda v, s: {"score": v["a"] * (1 if s == TRAIN else -1), "n": v["a"] * 10})
    for a in (1, 2, 3):
        evaluate_and_record(ctx, {"a": a, "b": 0}, source="t")
    evaluate_and_record(ctx, {"a": 3, "b": 0}, source="t", split=TEST)
    assert led.best()["variant"] == {"a": 3, "b": 0}
    assert led.best(min_n=35) is None
    assert [r["variant"]["a"] for r in led.top(2)] == [3, 2]
    assert led.best(TEST)["result"]["score"] == -3.0
    assert led.top(0) == []


# ----- researcher agents -----


def test_search_agents_are_opt_in_and_social_default_unchanged(tmp_path):
    # Importing SandFish (core, API) does not import or register search.
    code = (
        "import sys, sandfish, sandfish.core.orchestrator, sandfish.api.main\n"
        "from sandfish.agents.base_agent import AGENT_TYPES\n"
        "assert 'sandfish.search' not in sys.modules, 'search imported eagerly'\n"
        "assert sorted(AGENT_TYPES) == ['default', 'influencer', 'lurker'], sorted(AGENT_TYPES)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _social_state(tmp_path, tag):
    o = create_orchestrator(str(tmp_path / f"social_{tag}"))
    try:
        sim = o.create_simulation(SimulationConfig(name="s", num_agents=6, max_rounds=8, seed=11,
                                                   agent_types=["default", "influencer", "lurker"]))
        res = asyncio.run(o.run_simulation(sim))
        # Agent ids embed the random simulation id; normalise it away.
        state = json.loads(json.dumps(res.final_state, sort_keys=True, default=str).replace(sim, "SIM"))
        return state, sorted(res.metrics)
    finally:
        o.close()


def test_social_runs_identical_after_search_registration(tmp_path):
    before, metrics_before = _social_state(tmp_path, "a")
    register_search_agents()
    after, metrics_after = _social_state(tmp_path, "b")
    assert before == after
    assert metrics_before == metrics_after
    assert not any(m.startswith("search_") for m in metrics_after)


def _swarm(tmp_path, seed, tag, **kw):
    o = create_orchestrator(str(tmp_path / f"v_{tag}"))
    led = TrialLedger(tmp_path / f"{tag}.jsonl")
    try:
        out = asyncio.run(run_search(o, space=SPACE, evaluator=_score, ledger=led,
                                     num_agents=4, max_rounds=6, seed=seed, **kw))
        metrics = o._calculate_metrics(o.simulations[out["simulation_id"]])
    finally:
        o.close()
    return out, led, metrics


def test_swarm_logs_every_evaluation_and_is_reproducible(tmp_path):
    out_a, a, metrics = _swarm(tmp_path, 7, "a")
    _, b, _ = _swarm(tmp_path, 7, "b")
    assert out_a["status"] == "completed" and out_a["rounds_completed"] == 6
    assert a.n_trials == 24  # exactly agents x rounds, repeats included
    assert a.counts()["n_evaluator_calls"] == a.n_unique <= grid_size(SPACE)
    assert [r["key"] for r in a.rows()] == [r["key"] for r in b.rows()]
    assert all(r["variant"]["a"] in SPACE["a"] and r["variant"]["b"] in SPACE["b"] for r in a.rows())
    assert all(r["split"] == TRAIN and r["run_id"] == out_a["simulation_id"] for r in a.rows())
    assert {r["source"].rsplit("_agent_", 1)[1] for r in a.rows()} == {"0", "1", "2", "3"}
    assert [r["round"] for r in a.rows()[:4]] == [0, 0, 0, 0]
    assert metrics["search_trials"] == 24.0
    assert metrics["search_unique_variants"] == float(a.n_unique)
    assert out_a["integrity"]["ok"] and out_a["best_train"]["score"] <= 4


def test_swarm_repair_min_n_and_errors(tmp_path):
    def repair(v):
        v["b"] = 1 if v["a"] >= 2 else 0
        return v

    def flaky(v, s):
        if v["a"] == 0:
            raise ValueError("bad region")
        return {"score": v["a"], "n": v["a"]}

    o = create_orchestrator(str(tmp_path / "v"))
    led = TrialLedger()
    try:
        out = asyncio.run(run_search(o, space=SPACE, evaluator=flaky, ledger=led, num_agents=3,
                                     max_rounds=8, seed=1, repair=repair, min_n=2))
    finally:
        o.close()
    assert out["status"] == "completed" and led.n_trials == 24
    assert all(r["variant"]["b"] == (1 if r["variant"]["a"] >= 2 else 0) for r in led.rows())
    assert out["best_train"]["n"] >= 2 and out["min_n"] == 2


def test_researcher_without_attach_fails_the_run(orch):
    register_search_agents()
    sim = orch.create_simulation(SimulationConfig(name="x", num_agents=1, max_rounds=1,
                                                  agent_types=["researcher"], checkpoint_interval=0))
    res = asyncio.run(orch.run_simulation(sim))
    assert res.status == SimulationStatus.FAILED and "attach_search" in res.error_message


def test_attach_search_validation_and_remember(orch, tmp_path):
    register_search_agents()
    with pytest.raises(KeyError):
        attach_search(orch, "nope", space=SPACE, evaluator=_score, ledger=TrialLedger())
    sim = orch.create_simulation(SimulationConfig(name="m", num_agents=2, max_rounds=2, seed=3,
                                                  agent_types=["researcher"], checkpoint_interval=0))
    with pytest.raises(ValueError):
        attach_search(orch, sim, space=SPACE, evaluator=_score, ledger=TrialLedger(), min_n=-1)
    with pytest.raises(ValueError):
        attach_search(orch, sim, space={"a": []}, evaluator=_score, ledger=TrialLedger())
    led = TrialLedger()
    ctx = attach_search(orch, sim, space=SPACE, evaluator=_score, ledger=led, remember=True)
    assert ctx.evaluator_name == "_score" and ctx.run_id == sim
    asyncio.run(orch.run_simulation(sim))
    first = led.rows()[0]
    entity = orch.memory.get_entity(f"variant:{first['key']}")
    assert entity is not None


# ----- grid, holdout, summary -----


def test_run_grid_counts_and_cap(tmp_path):
    led = TrialLedger()
    out = run_grid(SPACE, _score, led)
    assert out["grid_size"] == 8 and led.n_trials == 8 and led.n_unique == 8
    assert out["best_train"]["variant"] == {"a": 3, "b": 1}
    with pytest.raises(ValueError, match="max_variants"):
        run_grid(SPACE, _score, TrialLedger(), max_variants=7)
    # A repair that collapses variants: repeats are logged as cached trials.
    led2 = TrialLedger()
    run_grid(SPACE, _score, led2, repair=lambda v: {**v, "b": 0})
    assert led2.n_trials == 8 and led2.n_unique == 4 and led2.counts()["n_evaluator_calls"] == 4


def test_validate_holdout_logs_test_rows():
    toy = ToyQuadraticEvaluator(seed=2)
    led = TrialLedger()
    run_grid(toy.DEFAULT_SPACE, toy, led)
    res = validate_holdout(led, toy, top_k=3)
    assert len(res) == 3 and all(r["test_score"] is not None for r in res)
    assert res[0]["train_score"] == led.best()["result"]["score"]
    assert led.counts()["by_split"][TEST] == {"trials": 3, "evaluator_calls": 3, "cached": 0,
                                              "errors": 0, "unique_variants": 3}
    again = validate_holdout(led, toy, variants=[res[0]["variant"], {"x": 9, "y": 9}])
    assert again[0]["test_score"] == res[0]["test_score"]
    assert again[1]["train_score"] is None
    assert led.counts()["by_split"][TEST]["trials"] == 5  # repeat peeks are counted
    s = summarize(led, top=2)
    assert s["top_train"][0]["test_score"] == res[0]["test_score"]
    assert s["counts"]["by_split"][TRAIN]["unique_variants"] == 49
    assert s["multiple_testing"] == MULTIPLE_TESTING_NOTE and s["integrity"]["ok"]


def test_summarize_empty_ledger():
    s = summarize(TrialLedger())
    assert s["best_train"] is None and s["top_train"] == [] and s["counts"]["n_trials"] == 0
