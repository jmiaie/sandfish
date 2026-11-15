"""Orchestrator hardening: resume semantics, pausing state, validation,
limits, eviction, per-simulation LLM budgets, and off-loop blocking I/O."""

import asyncio
import json
import threading

import pytest

from sandfish.core.orchestrator import (
    SimulationConfig,
    SimulationLimitError,
    SimulationLimits,
    SimulationStatus,
    SwarmOrchestrator,
)
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


def _orch(tmp_path, **kwargs):
    memory = OMPAMemoryAdapter(str(tmp_path / "vault"), enable_semantic=False)
    return SwarmOrchestrator(memory, **kwargs)


def _track_rounds(orch):
    """Record every executed round index."""
    executed = []
    original = orch._execute_round

    async def wrapped(sim_id, round_num):
        executed.append(round_num)
        await original(sim_id, round_num)

    orch._execute_round = wrapped
    return executed


# ----- resume / pausing -----


@pytest.mark.asyncio
async def test_resume_does_not_rerun_the_paused_round(tmp_path):
    orch = _orch(tmp_path)
    executed = _track_rounds(orch)
    sim_id = orch.create_simulation(SimulationConfig(name="r", max_rounds=6, num_agents=3, seed=1))

    def pause_at_2(event_type, data):
        if event_type == "round_complete" and data["round"] == 2 and executed.count(2) == 1:
            orch.pause_simulation(sim_id)

    orch.on_event(pause_at_2)
    paused = await orch.run_simulation(sim_id)
    assert paused.status == SimulationStatus.PAUSED
    assert paused.rounds_completed == 3
    assert orch.get_simulation_status(sim_id)["rounds_completed"] == 3

    assert orch.resume_simulation(sim_id) is True
    result = await orch.run_simulation(sim_id)
    assert result.status == SimulationStatus.COMPLETED
    assert executed == [0, 1, 2, 3, 4, 5], "each round must run exactly once"
    assert result.rounds_completed == 6


@pytest.mark.asyncio
async def test_pause_reports_pausing_until_the_loop_exits(tmp_path):
    orch = _orch(tmp_path)
    sim_id = orch.create_simulation(SimulationConfig(name="p", max_rounds=10, num_agents=2))
    seen = []

    def cb(event_type, data):
        if event_type == "round_complete" and data["round"] == 0:
            assert orch.pause_simulation(sim_id) is True
            seen.append(orch.simulations[sim_id]["status"])
            # A second pause request while pausing is a no-op.
            assert orch.pause_simulation(sim_id) is False

    orch.on_event(cb)
    await orch.run_simulation(sim_id)
    assert seen == [SimulationStatus.PAUSING]
    assert orch.simulations[sim_id]["status"] == SimulationStatus.PAUSED
    assert not orch._sim_locks[sim_id].locked()


@pytest.mark.asyncio
async def test_resume_while_pausing_cancels_the_pause(tmp_path):
    orch = _orch(tmp_path)
    executed = _track_rounds(orch)
    sim_id = orch.create_simulation(SimulationConfig(name="c", max_rounds=4, num_agents=2))

    def cb(event_type, data):
        if event_type == "round_complete" and data["round"] == 1:
            orch.pause_simulation(sim_id)
            assert orch.resume_simulation(sim_id) is True
            assert orch.simulations[sim_id]["status"] == SimulationStatus.RUNNING

    orch.on_event(cb)
    result = await orch.run_simulation(sim_id)
    assert result.status == SimulationStatus.COMPLETED
    assert executed == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_run_refuses_pausing_and_finished_simulations(tmp_path):
    orch = _orch(tmp_path)
    sim_id = orch.create_simulation(SimulationConfig(name="f", max_rounds=1, num_agents=1))
    await orch.run_simulation(sim_id)
    with pytest.raises(RuntimeError, match="completed"):
        await orch.run_simulation(sim_id)



@pytest.mark.asyncio
async def test_pause_or_stop_before_the_task_runs_is_honoured(tmp_path):
    """reserve_start() marks RUNNING; a pause/stop can land before the task
    starts. The run must settle the status instead of stranding it."""
    orch = _orch(tmp_path)
    a = orch.create_simulation(SimulationConfig(name="a", max_rounds=3, num_agents=1))
    orch.reserve_start(a)
    assert orch.pause_simulation(a) is True
    result = await orch.run_simulation(a)
    assert result.status == SimulationStatus.PAUSED and result.rounds_completed == 0
    assert orch.resume_simulation(a) is True
    assert (await orch.run_simulation(a)).rounds_completed == 3

    b = orch.create_simulation(SimulationConfig(name="b", max_rounds=3, num_agents=1))
    orch.reserve_start(b)
    assert orch.stop_simulation(b) is True
    result = await orch.run_simulation(b)
    assert result.status == SimulationStatus.STOPPED
    assert orch.simulations[b]["end_time"] is not None
    assert orch.active_count() == 0


@pytest.mark.asyncio
async def test_stop_while_pausing(tmp_path):
    orch = _orch(tmp_path)
    sim_id = orch.create_simulation(SimulationConfig(name="s", max_rounds=10, num_agents=2))

    def cb(event_type, data):
        if event_type == "round_complete" and data["round"] == 0:
            orch.pause_simulation(sim_id)
            assert orch.stop_simulation(sim_id) is True

    orch.on_event(cb)
    result = await orch.run_simulation(sim_id)
    assert result.status == SimulationStatus.STOPPED
    assert result.rounds_completed == 1
    assert orch.agents == {}


# ----- validation and limits -----


@pytest.mark.parametrize("types", [[], ["default", "nope"], ["Influencer"]])
def test_create_rejects_unknown_or_empty_agent_types(tmp_path, types):
    orch = _orch(tmp_path)
    with pytest.raises(ValueError, match="agent type"):
        orch.create_simulation(SimulationConfig(name="v", agent_types=types))
    assert orch.simulations == {}


def test_create_enforces_agent_and_round_caps(tmp_path):
    orch = _orch(tmp_path, limits=SimulationLimits(max_agents=5, max_rounds=10))
    with pytest.raises(SimulationLimitError, match="num_agents"):
        orch.create_simulation(SimulationConfig(name="a", num_agents=6, max_rounds=1))
    with pytest.raises(SimulationLimitError, match="max_rounds"):
        orch.create_simulation(SimulationConfig(name="b", num_agents=1, max_rounds=11))
    orch.create_simulation(SimulationConfig(name="ok", num_agents=5, max_rounds=10))


def test_limits_from_env():
    lim = SimulationLimits.from_env({"SANDFISH_MAX_AGENTS": "7", "SANDFISH_MAX_ROUNDS": "0"})
    assert lim.max_agents == 7
    assert lim.max_rounds is None  # 0 = unlimited
    assert lim.max_concurrent == 4 and lim.max_finished == 100


def test_reserve_start_enforces_concurrency(tmp_path):
    orch = _orch(tmp_path, limits=SimulationLimits(max_concurrent=1))
    a = orch.create_simulation(SimulationConfig(name="a", max_rounds=1, num_agents=1))
    b = orch.create_simulation(SimulationConfig(name="b", max_rounds=1, num_agents=1))
    orch.reserve_start(a)
    assert orch.simulations[a]["status"] == SimulationStatus.RUNNING
    with pytest.raises(SimulationLimitError):
        orch.reserve_start(b)
    assert orch.simulations[b]["status"] == SimulationStatus.PENDING
    with pytest.raises(RuntimeError, match="not pending"):
        orch.reserve_start(a)


@pytest.mark.asyncio
async def test_run_simulation_enforces_concurrency(tmp_path):
    orch = _orch(tmp_path, limits=SimulationLimits(max_concurrent=1))
    a = orch.create_simulation(SimulationConfig(name="a", max_rounds=1, num_agents=1))
    b = orch.create_simulation(SimulationConfig(name="b", max_rounds=1, num_agents=1))
    orch.simulations[a]["status"] = SimulationStatus.RUNNING
    with pytest.raises(SimulationLimitError):
        await orch.run_simulation(b)
    orch.simulations[a]["status"] = SimulationStatus.PENDING
    assert (await orch.run_simulation(b)).status == SimulationStatus.COMPLETED


@pytest.mark.asyncio
async def test_finished_simulations_are_evicted_oldest_first(tmp_path):
    orch = _orch(tmp_path, limits=SimulationLimits(max_finished=2))
    ids = []
    for i in range(4):
        sim_id = orch.create_simulation(SimulationConfig(name=f"s{i}", max_rounds=1, num_agents=2))
        ids.append(sim_id)
        await orch.run_simulation(sim_id)
    assert sorted(orch.simulations) == sorted(ids[2:])
    assert set(orch._sim_locks) == set(ids[2:])
    assert orch.agents == {}

    # Pending/paused simulations are never evicted.
    keep = orch.create_simulation(SimulationConfig(name="pending", max_rounds=1, num_agents=1))
    extra = orch.create_simulation(SimulationConfig(name="x", max_rounds=1, num_agents=1))
    await orch.run_simulation(extra)
    assert keep in orch.simulations and extra in orch.simulations
    assert len(orch.simulations) == 3


# ----- per-simulation LLM budget -----


@pytest.mark.asyncio
async def test_llm_budget_is_per_simulation(tmp_path):
    from sandfish.llm import MockLLMClient
    from sandfish.llm.agent import disable_llm_agents, enable_llm_agents

    client = MockLLMClient()
    total = enable_llm_agents(client, max_calls=3)
    try:
        orch = _orch(tmp_path)
        metrics = []
        for name in ("one", "two"):
            sim_id = orch.create_simulation(
                SimulationConfig(name=name, max_rounds=5, num_agents=2, agent_types=["llm"], seed=3)
            )
            result = await orch.run_simulation(sim_id)
            metrics.append(result.metrics)
        # The first run used its whole budget; the second still got its own.
        assert [m["llm_calls"] for m in metrics] == [3.0, 3.0]
        assert len(client.calls) == 6
        assert total.calls == 6
    finally:
        disable_llm_agents()


@pytest.mark.asyncio
async def test_llm_max_total_calls_caps_across_simulations(tmp_path):
    from sandfish.llm import MockLLMClient
    from sandfish.llm.agent import disable_llm_agents, enable_llm_agents

    client = MockLLMClient()
    enable_llm_agents(client, max_calls=3, max_total_calls=4)
    try:
        orch = _orch(tmp_path)
        for name in ("one", "two"):
            sim_id = orch.create_simulation(
                SimulationConfig(name=name, max_rounds=5, num_agents=2, agent_types=["llm"])
            )
            await orch.run_simulation(sim_id)
        assert len(client.calls) == 4
    finally:
        disable_llm_agents()


@pytest.mark.asyncio
async def test_http_tokens_are_tallied_per_simulation(tmp_path):
    httpx = pytest.importorskip("httpx")
    from sandfish.llm.agent import disable_llm_agents, enable_llm_agents
    from sandfish.llm.providers import OpenRouterClient

    decision = {"action": "create_post", "target": "none", "content": "hi", "reason": "r"}
    payload = {"choices": [{"message": {"content": json.dumps(decision)}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 4}}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = OpenRouterClient(api_key="sk-test-not-a-real-key", transport=transport)
    total = enable_llm_agents(client, max_calls=2)
    try:
        orch = _orch(tmp_path)
        sim_id = orch.create_simulation(
            SimulationConfig(name="tok", max_rounds=3, num_agents=1, agent_types=["llm"])
        )
        result = await orch.run_simulation(sim_id)
        assert result.metrics["llm_calls"] == 2.0
        assert result.metrics["llm_input_tokens"] == 20.0
        assert result.metrics["llm_output_tokens"] == 8.0
        assert total.input_tokens == 20 and total.output_tokens == 8
        state = next(iter(result.final_state.values()))
        assert state["llm"] == {"calls": 2, "fallbacks": 1}
    finally:
        disable_llm_agents()


# ----- blocking work runs off the event loop -----


@pytest.mark.asyncio
async def test_memory_flush_and_checkpoint_writes_run_in_worker_threads(tmp_path, monkeypatch):
    import sandfish.core.orchestrator as orch_mod

    orch = _orch(tmp_path, checkpoint_dir=tmp_path / "ckpt")
    loop_thread = threading.get_ident()
    flush_threads, write_threads = set(), set()

    real_flush = orch.memory.flush

    def flush():
        flush_threads.add(threading.get_ident())
        return real_flush()

    real_write = orch_mod._write_checkpoint

    def write(path, checkpoint):
        write_threads.add(threading.get_ident())
        real_write(path, checkpoint)

    monkeypatch.setattr(orch.memory, "flush", flush)
    monkeypatch.setattr(orch_mod, "_write_checkpoint", write)
    sim_id = orch.create_simulation(
        SimulationConfig(name="io", max_rounds=4, num_agents=2, checkpoint_interval=2)
    )
    await orch.run_simulation(sim_id)

    assert flush_threads and loop_thread not in flush_threads
    assert write_threads and loop_thread not in write_threads
    assert len(list((tmp_path / "ckpt").glob(f"{sim_id}_round_*.json"))) == 2


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_slow_flush(tmp_path, monkeypatch):
    import time

    orch = _orch(tmp_path)
    monkeypatch.setattr(orch.memory, "flush", lambda: time.sleep(0.05))
    sim_id = orch.create_simulation(SimulationConfig(name="slow", max_rounds=3, num_agents=1))

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    t = asyncio.create_task(ticker())
    await orch.run_simulation(sim_id)
    t.cancel()
    # ~4 flushes x 50 ms: a blocked loop would allow ~0 ticks.
    assert ticks >= 10


@pytest.mark.asyncio
async def test_close_releases_io_thread(tmp_path):
    orch = _orch(tmp_path)
    sim_id = orch.create_simulation(SimulationConfig(name="c", max_rounds=2, num_agents=1))
    await orch.run_simulation(sim_id)
    orch.close()
    assert orch._io_executor._shutdown


@pytest.mark.asyncio
async def test_global_budget_refusal_refunds_simulation_budget(tmp_path):
    from sandfish.agents.base_agent import create_agent
    from sandfish.llm import MockLLMClient
    from sandfish.llm.agent import disable_llm_agents, enable_llm_agents

    enable_llm_agents(MockLLMClient(), max_calls=5, max_total_calls=0)
    try:
        agent = create_agent("llm", agent_id="me")
        agent.set_peers(["me", "p1"])
        action = await agent.decide_action()
        assert action.metadata["policy"] == "heuristic_fallback"
        assert agent.simulation_budget().calls == 0
    finally:
        disable_llm_agents()


@pytest.mark.asyncio
async def test_client_side_budget_refusal_refunds_simulation_budget():
    httpx = pytest.importorskip("httpx")
    from sandfish.agents.base_agent import create_agent
    from sandfish.llm.agent import disable_llm_agents, enable_llm_agents
    from sandfish.llm.base import LLMBudget
    from sandfish.llm.providers import OpenRouterClient

    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    client = OpenRouterClient(api_key="sk-test-not-a-real-key", transport=transport,
                              budget=LLMBudget(max_calls=0))
    enable_llm_agents(client, max_calls=5)
    try:
        agent = create_agent("llm", agent_id="me")
        agent.set_peers(["me", "p1"])
        await agent.decide_action()
        assert agent.simulation_budget().calls == 0
    finally:
        disable_llm_agents()
