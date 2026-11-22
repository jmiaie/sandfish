"""Where the vault lives, how much a long run keeps in memory, and what an
LLM call budget actually limits."""

import asyncio
import json
import logging
import time

import pytest

import sandfish.memory.ompa_adapter as adapter
from sandfish import cli
from sandfish.core.orchestrator import SimulationConfig, SwarmOrchestrator
from sandfish.memory import DEFAULT_VAULT_PATH, default_vault_path
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter, _InMemoryBackend

# ----- vault location -----


@pytest.fixture
def fresh_warning(monkeypatch):
    monkeypatch.setattr(adapter, "_warned_ompa_vault_path", False)


def test_sandfish_has_its_own_vault_variable(fresh_warning):
    assert default_vault_path({}) == DEFAULT_VAULT_PATH == "./sandfish_vault"
    assert default_vault_path({"SANDFISH_VAULT_PATH": "/data/sf"}) == "/data/sf"
    assert default_vault_path({"SANDFISH_VAULT_PATH": "  "}) == DEFAULT_VAULT_PATH
    both = {"SANDFISH_VAULT_PATH": "/data/sf", "OMPA_VAULT_PATH": "/agent/memory"}
    assert default_vault_path(both) == "/data/sf"
    assert default_vault_path(both, legacy_fallback=True) == "/data/sf"


def test_the_ompa_variable_is_only_a_fallback_and_warns_once(fresh_warning, caplog):
    env = {"OMPA_VAULT_PATH": "/agent/memory"}
    # Not read at all unless a caller asks for the old behaviour.
    assert default_vault_path(env) == DEFAULT_VAULT_PATH
    with caplog.at_level(logging.WARNING, logger="sandfish.memory"):
        assert default_vault_path(env, legacy_fallback=True) == "/agent/memory"
        assert default_vault_path(env, legacy_fallback=True) == "/agent/memory"
    warnings = [r.getMessage() for r in caplog.records if "OMPA_VAULT_PATH" in r.getMessage()]
    assert len(warnings) == 1
    assert "SANDFISH_VAULT_PATH" in warnings[0] and "/agent/memory" in warnings[0]


def test_no_warning_when_sandfish_has_its_own_vault(fresh_warning, caplog):
    env = {"SANDFISH_VAULT_PATH": "/data/sf", "OMPA_VAULT_PATH": "/agent/memory"}
    with caplog.at_level(logging.WARNING, logger="sandfish.memory"):
        default_vault_path(env, legacy_fallback=True)
    assert not caplog.records


def test_cli_reads_the_sandfish_variable_and_never_the_ompa_one(fresh_warning, monkeypatch):
    """The CLI never read OMPA_VAULT_PATH, and must not start to."""
    monkeypatch.setenv("OMPA_VAULT_PATH", "/agent/memory")
    monkeypatch.delenv("SANDFISH_VAULT_PATH", raising=False)
    seen = []
    monkeypatch.setattr(cli, "cmd_orchestrator", lambda args: seen.append(args.vault) or 0)
    assert cli.main(["orchestrator", "--dry-run"]) == 0
    monkeypatch.setenv("SANDFISH_VAULT_PATH", "/data/sf")
    assert cli.main(["orchestrator", "--dry-run"]) == 0
    assert cli.main(["orchestrator", "--dry-run", "--vault", "/explicit"]) == 0
    assert seen == [DEFAULT_VAULT_PATH, "/data/sf", "/explicit"]


def test_api_vault_resolution(fresh_warning, monkeypatch, tmp_path):
    pytest.importorskip("httpx")
    import sandfish.api.main as api

    own, shared = str(tmp_path / "sf"), str(tmp_path / "agent")
    monkeypatch.setenv("OMPA_VAULT_PATH", shared)
    monkeypatch.delenv("SANDFISH_VAULT_PATH", raising=False)
    monkeypatch.setattr(api.app.state, "config", None, raising=False)
    # Started by importing the app (no configure_app): the old fallback.
    assert api._resolve_vault_path(api.app) == shared
    monkeypatch.setenv("SANDFISH_VAULT_PATH", own)
    assert api._resolve_vault_path(api.app) == own
    # configure_app never read OMPA_VAULT_PATH and still does not.
    monkeypatch.delenv("SANDFISH_VAULT_PATH")
    api.configure_app()
    assert api._resolve_vault_path(api.app) == DEFAULT_VAULT_PATH
    monkeypatch.setenv("SANDFISH_VAULT_PATH", own)
    api.configure_app()
    assert api._resolve_vault_path(api.app) == own
    api.configure_app(str(tmp_path / "explicit"))
    assert api._resolve_vault_path(api.app) == str(tmp_path / "explicit")


def test_mcp_server_vault_resolution(fresh_warning, monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    from sandfish.mcp_server import build_server

    own, shared = tmp_path / "sf", tmp_path / "agent"
    monkeypatch.setenv("OMPA_VAULT_PATH", str(shared))
    monkeypatch.setenv("SANDFISH_VAULT_PATH", str(own))
    assert build_server(enable_semantic=False).sandfish.vault_path == str(own)
    monkeypatch.delenv("SANDFISH_VAULT_PATH")
    assert build_server(enable_semantic=False).sandfish.vault_path == str(shared)
    explicit = tmp_path / "explicit"
    assert build_server(str(explicit), enable_semantic=False).sandfish.vault_path == str(explicit)


# ----- checkpoints kept in memory -----


def _orchestrator(tmp_path, **kwargs):
    return SwarmOrchestrator(OMPAMemoryAdapter(str(tmp_path / "vault")), **kwargs)


def _config(rounds, interval):
    return SimulationConfig(
        name="cp", max_rounds=rounds, num_agents=2, checkpoint_interval=interval, seed=1
    )


@pytest.mark.asyncio
async def test_only_the_newest_checkpoints_stay_in_memory(tmp_path):
    orch = _orchestrator(tmp_path, checkpoint_dir=tmp_path / "cp")
    sim_id = orch.create_simulation(_config(rounds=24, interval=2))
    result = await orch.run_simulation(sim_id)

    kept = [c["round"] for c in orch.simulations[sim_id]["checkpoints"]]
    assert kept == [16, 18, 20, 22, 24]  # the default keeps five
    # The count is of checkpoints taken, not of those still in memory...
    assert result.metrics["checkpoint_count"] == 12
    # ...and every one is still on disk.
    written = sorted(p.name for p in (tmp_path / "cp").glob("*.json"))
    assert len(written) == 12 and written[0].endswith("_round_000002.json")
    on_disk = json.loads((tmp_path / "cp" / written[-1]).read_text())
    assert on_disk["round"] == 24 and len(on_disk["agent_states"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("keep,expected", [(1, [24]), (3, [20, 22, 24]), (None, list(range(2, 25, 2)))])
async def test_keep_checkpoints_setting(tmp_path, keep, expected):
    orch = _orchestrator(tmp_path, keep_checkpoints=keep)
    sim_id = orch.create_simulation(_config(rounds=24, interval=2))
    result = await orch.run_simulation(sim_id)
    assert [c["round"] for c in orch.simulations[sim_id]["checkpoints"]] == expected
    assert result.metrics["checkpoint_count"] == 12


def test_keep_checkpoints_must_keep_something(tmp_path):
    with pytest.raises(ValueError, match="keep_checkpoints"):
        _orchestrator(tmp_path, keep_checkpoints=0)


# ----- LLM budgets count requests -----

httpx = pytest.importorskip("httpx")

from sandfish.agents.base_agent import create_agent  # noqa: E402
from sandfish.llm import LLMBudget, LLMError, OpenRouterClient, action_schema  # noqa: E402
from sandfish.llm.agent import disable_llm_agents, enable_llm_agents  # noqa: E402
from sandfish.llm.base import usage_sink  # noqa: E402

FAKE_KEY = "sk-test-NOT-A-REAL-KEY-123456"
DECISION = {"action": "follow", "target": "p1", "content": "", "reason": "test"}
SCHEMA = action_schema(["follow", "do_nothing"], ["p1"])
OK = {"choices": [{"message": {"content": json.dumps(DECISION)}}]}


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Retries wait 0.5 s then 1 s; the tests do not need to."""
    real_sleep = asyncio.sleep

    async def instant(_seconds):
        await real_sleep(0)

    monkeypatch.setattr("sandfish.llm.providers.asyncio.sleep", instant)
    yield
    disable_llm_agents()


def _scripted(statuses):
    """A transport answering with `statuses` in turn (an Exception is raised),
    then 200 for ever. Returns it with the list of requests it saw."""
    seen = []
    script = list(statuses)

    def handler(request):
        seen.append(request)
        step = script.pop(0) if script else 200
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, json=OK if step == 200 else {"error": "busy"})

    return httpx.MockTransport(handler), seen


async def _ask(client):
    return await client.complete_json(system="s", prompt="p", schema=SCHEMA)


@pytest.mark.asyncio
async def test_a_retry_is_charged_to_the_budget():
    transport, seen = _scripted([503, 503])
    budget = LLMBudget(max_calls=10)
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport, budget=budget)
    assert await _ask(client) == DECISION
    assert len(seen) == 3 and budget.calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 2, 3, 5])
async def test_requests_sent_never_exceed_the_budget(cap):
    """Each call used to take one unit and send up to three requests."""
    transport, seen = _scripted([503] * 50)
    budget = LLMBudget(max_calls=cap)
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport, budget=budget)
    for _ in range(4):
        with pytest.raises(LLMError):
            await _ask(client)
    assert len(seen) == cap == budget.calls


@pytest.mark.asyncio
async def test_out_of_budget_mid_retry_reports_the_failure_it_had():
    transport, seen = _scripted([503, 503])
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport, budget=LLMBudget(max_calls=1))
    with pytest.raises(LLMError, match="HTTP 503"):
        await _ask(client)
    assert len(seen) == 1

    transport, seen = _scripted([httpx.ConnectError(f"refused, key {FAKE_KEY}"), 200])
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport, budget=LLMBudget(max_calls=1))
    with pytest.raises(LLMError, match="request failed") as err:
        await _ask(client)
    assert len(seen) == 1 and FAKE_KEY not in str(err.value)


@pytest.mark.asyncio
async def test_a_failed_connection_is_retried_and_charged():
    transport, seen = _scripted([httpx.ConnectError("refused")])
    budget = LLMBudget(max_calls=10)
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport, budget=budget)
    assert await _ask(client) == DECISION
    assert len(seen) == 2 and budget.calls == 2


@pytest.mark.asyncio
async def test_retries_are_charged_to_the_simulation_too():
    """The per-simulation cap is the one `--llm-max-calls` sets."""
    transport, seen = _scripted([503, 503])
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport)
    simulation = LLMBudget(max_calls=2)
    assert simulation.try_acquire()  # the agent takes the first unit itself
    token = usage_sink.set(simulation)
    try:
        with pytest.raises(LLMError, match="HTTP 503"):
            await _ask(client)
    finally:
        usage_sink.reset(token)
    assert len(seen) == 2 and simulation.calls == 2


@pytest.mark.asyncio
async def test_a_refused_retry_takes_nothing_from_either_budget():
    transport, seen = _scripted([503, 503])
    own = LLMBudget(max_calls=10)
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport, budget=own)
    simulation = LLMBudget(max_calls=1)
    assert simulation.try_acquire()
    token = usage_sink.set(simulation)
    try:
        with pytest.raises(LLMError):
            await _ask(client)
    finally:
        usage_sink.reset(token)
    assert len(seen) == 1 and own.calls == 1 and simulation.calls == 1


@pytest.mark.asyncio
async def test_a_simulation_never_sends_more_requests_than_its_cap():
    transport, seen = _scripted([503] * 500)
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport)
    enable_llm_agents(client, max_calls=7)
    orch = SwarmOrchestrator(OMPAMemoryAdapter("unused", backend=_InMemoryBackend()))
    sim_id = orch.create_simulation(
        SimulationConfig(name="llm", max_rounds=4, num_agents=4, agent_types=["llm"], seed=3)
    )
    result = await orch.run_simulation(sim_id)
    assert len(seen) == 7
    assert result.metrics["llm_calls"] == 7


@pytest.mark.asyncio
async def test_one_connection_pool_for_the_life_of_a_client():
    transport, seen = _scripted([])
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport)
    for _ in range(3):
        await _ask(client)
    pool = client._client
    assert pool is not None and len(seen) == 3
    await _ask(client)
    assert client._client is pool
    await client.aclose()
    assert pool.is_closed and client._client is None
    await client.aclose()  # a second close is harmless
    await _ask(client)  # and the client opens a new pool when used again
    assert client._client is not pool
    await client.aclose()


def test_a_client_survives_a_change_of_event_loop():
    """`sandfish orchestrator` and tests call asyncio.run more than once."""
    transport, seen = _scripted([])
    client = OpenRouterClient(api_key=FAKE_KEY, transport=transport)
    assert asyncio.run(_ask(client)) == DECISION
    first = client._client
    assert asyncio.run(_ask(client)) == DECISION
    assert client._client is not first and len(seen) == 2


@pytest.mark.asyncio
async def test_llm_agent_still_falls_back_when_the_provider_fails():
    transport, seen = _scripted([500] * 20)
    enable_llm_agents(OpenRouterClient(api_key=FAKE_KEY, transport=transport), max_calls=50)
    agent = create_agent(
        "llm", agent_id="me", memory_adapter=OMPAMemoryAdapter("unused", backend=_InMemoryBackend())
    )
    action = await agent.decide_action()
    assert action.metadata.get("policy") == "heuristic_fallback"
    assert len(seen) == 3 and agent.simulation_budget().calls == 3


# ----- LLM agents on a server: the operator opts in, under caps -----

from sandfish.agents.base_agent import AGENT_TYPES  # noqa: E402
from sandfish.llm.agent import (  # noqa: E402
    DEFAULT_SERVER_CALLS_PER_SIMULATION,
    DEFAULT_SERVER_TOTAL_CALLS,
    LLMAgent,
    enable_llm_agents_from_env,
)

LLM_SIM = {"name": "llm", "num_agents": 3, "max_rounds": 4, "agent_types": ["llm"], "seed": 5}


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import sandfish.api.main as api

    monkeypatch.setenv("SANDFISH_OMPA_SEMANTIC", "0")
    monkeypatch.delenv("SANDFISH_API_KEY", raising=False)
    monkeypatch.setattr(api, "_limiter", api._SlidingWindowLimiter(10_000))
    for name in ("SANDFISH_SERVER_LLM_AGENTS", "SANDFISH_LLM_PROVIDER",
                 "SANDFISH_LLM_MAX_CALLS", "SANDFISH_LLM_MAX_TOTAL_CALLS"):
        monkeypatch.delenv(name, raising=False)

    def make(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, str(value))
        api.configure_app(str(tmp_path / "vault"))
        return TestClient(api.app)

    return make


def _run_to_completion(client, body):
    created = client.post("/api/simulations", json=body)
    assert created.status_code == 200, created.text
    sim_id = created.json()["id"]
    assert client.post(f"/api/simulations/{sim_id}/start").status_code == 200
    for _ in range(400):
        if client.get(f"/api/simulations/{sim_id}").json()["status"] == "completed":
            break
        time.sleep(0.01)
    results = client.get(f"/api/simulations/{sim_id}/results")
    assert results.status_code == 200, results.text
    return results.json()


def test_llm_agents_are_off_on_a_server_by_default(api_client):
    with api_client() as c:
        assert c.post("/api/simulations", json=LLM_SIM).status_code == 422
    assert "llm" not in AGENT_TYPES


def test_a_provider_set_for_the_command_line_does_not_turn_them_on(api_client):
    """SANDFISH_LLM_PROVIDER is what `sandfish orchestrator` reads; having it
    in the environment is not consent to spend on behalf of API callers."""
    with api_client(SANDFISH_LLM_PROVIDER="mock") as c:
        assert c.post("/api/simulations", json=LLM_SIM).status_code == 422
    assert enable_llm_agents_from_env({"SANDFISH_LLM_PROVIDER": "mock"}) is None
    assert "llm" not in AGENT_TYPES


def test_operator_opt_in_runs_llm_agents_over_http(api_client):
    with api_client(SANDFISH_SERVER_LLM_AGENTS="1", SANDFISH_LLM_PROVIDER="mock") as c:
        results = _run_to_completion(c, LLM_SIM)
        assert results["metrics"]["llm_calls"] == 12  # 3 agents x 4 rounds
        assert LLMAgent.max_calls_per_simulation == DEFAULT_SERVER_CALLS_PER_SIMULATION
        assert LLMAgent.budget.max_calls == DEFAULT_SERVER_TOTAL_CALLS
    # Shutting the server down takes the type away again.
    assert "llm" not in AGENT_TYPES and LLMAgent.client is None


def test_server_caps_apply_per_simulation_and_in_total(api_client):
    env = {"SANDFISH_SERVER_LLM_AGENTS": "true", "SANDFISH_LLM_PROVIDER": "mock",
           "SANDFISH_LLM_MAX_CALLS": 5, "SANDFISH_LLM_MAX_TOTAL_CALLS": 8}
    with api_client(**env) as c:
        first = _run_to_completion(c, LLM_SIM)
        second = _run_to_completion(c, dict(LLM_SIM, name="again"))
        assert first["metrics"]["llm_calls"] == 5  # its own cap
        assert LLMAgent.budget.calls == 8  # 5, then the 3 left in total
        assert second["metrics"]["completion_rate"] == 1.0  # heuristics took over


def test_a_half_configured_opt_in_refuses_to_start(api_client):
    with pytest.raises(LLMError, match="SANDFISH_LLM_PROVIDER"):
        with api_client(SANDFISH_SERVER_LLM_AGENTS="1"):
            pass
    with pytest.raises(LLMError, match="SANDFISH_LLM_MAX_CALLS"):
        enable_llm_agents_from_env({
            "SANDFISH_SERVER_LLM_AGENTS": "1", "SANDFISH_LLM_PROVIDER": "mock",
            "SANDFISH_LLM_MAX_CALLS": "lots",
        })
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        enable_llm_agents_from_env(
            {"SANDFISH_SERVER_LLM_AGENTS": "1", "SANDFISH_LLM_PROVIDER": "nope"}
        )
    assert "llm" not in AGENT_TYPES


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", " "])
def test_opt_in_values_that_mean_no(value):
    env = {"SANDFISH_SERVER_LLM_AGENTS": value, "SANDFISH_LLM_PROVIDER": "mock"}
    assert enable_llm_agents_from_env(env) is None
    assert "llm" not in AGENT_TYPES


def test_zero_lifts_a_cap():
    budget = enable_llm_agents_from_env({
        "SANDFISH_SERVER_LLM_AGENTS": "1", "SANDFISH_LLM_PROVIDER": "mock",
        "SANDFISH_LLM_MAX_CALLS": "0", "SANDFISH_LLM_MAX_TOTAL_CALLS": "0",
    })
    assert LLMAgent.max_calls_per_simulation is None
    assert budget.max_calls > 10**12


def test_mcp_server_start_up_follows_the_same_opt_in(monkeypatch, tmp_path, capsys):
    pytest.importorskip("mcp")
    import sandfish.mcp_server as mcp_server

    seen = []

    class _Server:
        class sandfish:  # noqa: N801 - stands in for the server's state object
            @staticmethod
            def close():
                seen.append("closed")

        @staticmethod
        def run(transport):
            seen.append(("llm" in AGENT_TYPES, transport))

    monkeypatch.setattr(mcp_server, "build_server", lambda vault: _Server)
    for name in ("SANDFISH_SERVER_LLM_AGENTS", "SANDFISH_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)

    assert mcp_server.main(["--vault", str(tmp_path)]) == 0
    monkeypatch.setenv("SANDFISH_SERVER_LLM_AGENTS", "1")
    assert mcp_server.main(["--vault", str(tmp_path)]) == 2  # no provider
    assert "SANDFISH_LLM_PROVIDER" in capsys.readouterr().err
    monkeypatch.setenv("SANDFISH_LLM_PROVIDER", "mock")
    assert mcp_server.main(["--vault", str(tmp_path)]) == 0
    assert seen == [(False, "stdio"), "closed", (True, "stdio"), "closed"]
