"""LLM backend tests: offline only (MockLLMClient + httpx.MockTransport)."""

import json

import pytest

httpx = pytest.importorskip("httpx")

from sandfish.agents.base_agent import AGENT_TYPES, ActionType, AgentProfile, create_agent  # noqa: E402
from sandfish.core.orchestrator import SimulationConfig, SwarmOrchestrator  # noqa: E402
from sandfish.llm import (  # noqa: E402
    AnthropicClient,
    GeminiClient,
    LLMBudget,
    LLMClient,
    LLMError,
    MockLLMClient,
    OpenRouterClient,
    action_schema,
    client_from_env,
)
from sandfish.llm.agent import LLMAgent, disable_llm_agents, enable_llm_agents  # noqa: E402
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter, _InMemoryBackend  # noqa: E402

FAKE_KEY = "sk-test-NOT-A-REAL-KEY-123456"
DECISION = {"action": "follow", "target": "p1", "content": "", "reason": "test"}


@pytest.fixture(autouse=True)
def _clean_llm_registry():
    yield
    disable_llm_agents()


def _mem():
    return OMPAMemoryAdapter("unused", backend=_InMemoryBackend())


def test_llm_type_not_registered_by_default():
    assert "llm" not in AGENT_TYPES
    with pytest.raises(ValueError):
        create_agent("llm")


def test_schema_is_strict_and_closed():
    schema = action_schema(["follow", "do_nothing"], ["p1", "p2"])
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"action", "target", "content", "reason"}
    assert schema["properties"]["target"]["enum"] == ["p1", "p2", "none"]


def test_mock_satisfies_protocol():
    assert isinstance(MockLLMClient(), LLMClient)


@pytest.mark.asyncio
async def test_llm_agent_uses_client_decision():
    client = MockLLMClient(responder=lambda s, p, sch: dict(DECISION))
    enable_llm_agents(client, max_calls=10)
    agent = create_agent("llm", agent_id="me", memory_adapter=_mem())
    agent.set_peers(["me", "p1", "p2"])
    action = await agent.decide_action()
    assert action.action_type is ActionType.FOLLOW and action.target == "p1"
    assert action.metadata["policy"] == "llm"
    assert len(client.calls) == 1
    prompt = json.loads(client.calls[0]["prompt"])
    assert set(prompt["visible_peers"]) == {"p1", "p2"}


@pytest.mark.asyncio
async def test_invalid_output_and_errors_fall_back():
    bad = MockLLMClient(responder=lambda s, p, sch: {"action": "hack_the_planet", "target": "x"})
    enable_llm_agents(bad, max_calls=10)
    agent = create_agent("llm", agent_id="me", memory_adapter=_mem())
    agent.set_peers(["me", "p1"])
    action = await agent.decide_action()
    assert action.metadata["policy"] == "heuristic_fallback"
    assert agent.llm_fallbacks == 1

    enable_llm_agents(MockLLMClient(fail_with=LLMError("boom")), max_calls=10)
    agent2 = create_agent("llm", agent_id="me2", memory_adapter=_mem())
    agent2.set_peers(["me2", "p1"])
    assert (await agent2.decide_action()).metadata["policy"] == "heuristic_fallback"


@pytest.mark.asyncio
async def test_invented_target_and_untargeted_action_are_sanitised():
    enable_llm_agents(MockLLMClient(responder=lambda s, p, sch: {
        "action": "create_post", "target": "p1", "content": "x" * 999, "reason": "r"}))
    agent = create_agent("llm", agent_id="me", memory_adapter=_mem())
    agent.set_peers(["me", "p1"])
    action = await agent.decide_action()
    assert action.action_type is ActionType.CREATE_POST
    assert action.target is None and len(action.content) == 280


@pytest.mark.asyncio
async def test_budget_caps_calls_across_a_run():
    client = MockLLMClient()
    budget = enable_llm_agents(client, max_calls=7)
    orch = SwarmOrchestrator(_mem())
    sid = orch.create_simulation(SimulationConfig(
        name="llm", max_rounds=5, num_agents=4, agent_types=["llm"], checkpoint_interval=0))
    result = await orch.run_simulation(sid)
    assert result.status.value == "completed"
    assert len(client.calls) == 7 and budget.calls == 7


def test_client_from_env(monkeypatch):
    monkeypatch.delenv("SANDFISH_LLM_PROVIDER", raising=False)
    assert client_from_env() is None
    assert isinstance(client_from_env("mock"), MockLLMClient)
    with pytest.raises(LLMError):
        client_from_env("nope")
    for var in ("ANTHROPIC_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMError):
        client_from_env("anthropic")


def _transport(capture, payload, status=200):
    def handler(request):
        capture["url"] = str(request.url)
        capture["headers"] = dict(request.headers)
        capture["body"] = json.loads(request.content)
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)


SCHEMA = action_schema(["follow", "do_nothing"], ["p1"])


@pytest.mark.asyncio
async def test_anthropic_forces_tool_and_parses_tool_use():
    cap = {}
    payload = {"content": [{"type": "tool_use", "name": "choose_action", "input": DECISION}],
               "usage": {"input_tokens": 11, "output_tokens": 5}}
    budget = LLMBudget(max_calls=5)
    c = AnthropicClient(api_key=FAKE_KEY, transport=_transport(cap, payload), budget=budget)
    out = await c.complete_json(system="s", prompt="p", schema=SCHEMA)
    assert out == DECISION
    assert cap["url"].endswith("/v1/messages")
    assert cap["body"]["tool_choice"] == {"type": "tool", "name": "choose_action"}
    assert cap["body"]["tools"][0]["input_schema"] == SCHEMA
    assert cap["body"]["model"] == "claude-sonnet-5-5"
    assert budget.input_tokens == 11 and budget.output_tokens == 5
    assert FAKE_KEY not in repr(c)


@pytest.mark.asyncio
async def test_gemini_uses_response_json_schema():
    cap = {}
    payload = {"candidates": [{"content": {"parts": [{"text": json.dumps(DECISION)}]}}]}
    c = GeminiClient(api_key=FAKE_KEY, model="gemini-test", transport=_transport(cap, payload))
    assert await c.complete_json(system="s", prompt="p", schema=SCHEMA) == DECISION
    assert ":generateContent" in cap["url"] and "gemini-test" in cap["url"]
    assert FAKE_KEY not in cap["url"]  # key travels in a header, not the URL
    gen = cap["body"]["generationConfig"]
    assert gen["responseMimeType"] == "application/json" and gen["responseJsonSchema"] == SCHEMA


@pytest.mark.asyncio
async def test_openrouter_strict_json_schema():
    cap = {}
    payload = {"choices": [{"message": {"content": json.dumps(DECISION)}}]}
    c = OpenRouterClient(api_key=FAKE_KEY, transport=_transport(cap, payload))
    assert await c.complete_json(system="s", prompt="p", schema=SCHEMA) == DECISION
    rf = cap["body"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_http_errors_never_leak_the_key():
    cap = {}
    c = OpenRouterClient(api_key=FAKE_KEY, transport=_transport(cap, {"error": f"bad key {FAKE_KEY}"}, 401))
    with pytest.raises(LLMError) as err:
        await c.complete_json(system="s", prompt="p", schema=SCHEMA)
    assert "401" in str(err.value) and FAKE_KEY not in str(err.value)


@pytest.mark.asyncio
async def test_client_budget_blocks_before_network():
    cap = {}
    c = GeminiClient(api_key=FAKE_KEY, budget=LLMBudget(max_calls=0), transport=_transport(cap, {}))
    with pytest.raises(LLMError, match="budget"):
        await c.complete_json(system="s", prompt="p", schema=SCHEMA)
    assert cap == {}


def test_llm_agent_is_default_agent_subclass():
    assert issubclass(LLMAgent, type(create_agent("default")))
    a = LLMAgent("x", AgentProfile(name="X", agent_type="llm"))
    assert a.client is None


@pytest.mark.asyncio
async def test_http_client_shares_budget_and_tallies_tokens():
    cap = {}
    payload = {"choices": [{"message": {"content": json.dumps(DECISION)}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 4}}
    client = OpenRouterClient(api_key=FAKE_KEY, transport=_transport(cap, payload))
    budget = enable_llm_agents(client, max_calls=2)
    assert client.budget is budget
    agent = create_agent("llm", agent_id="me", memory_adapter=_mem())
    agent.set_peers(["me", "p1"])
    policies = [(await agent.decide_action()).metadata["policy"] for _ in range(3)]
    assert policies == ["llm", "llm", "heuristic_fallback"]
    assert budget.calls == 2 and budget.input_tokens == 20 and budget.output_tokens == 8
