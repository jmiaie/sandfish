"""Event delivery must never hold up a simulation, WebSocket subscriptions
filter, and a finished simulation's results are still served over HTTP."""

import asyncio
import time

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import sandfish.api.main as api  # noqa: E402
from sandfish.core.orchestrator import SimulationConfig, SwarmOrchestrator  # noqa: E402
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDFISH_OMPA_SEMANTIC", "0")
    monkeypatch.delenv("SANDFISH_API_KEY", raising=False)
    monkeypatch.setattr(api, "_limiter", api._SlidingWindowLimiter(10_000))
    api.configure_app(str(tmp_path / "vault"))
    with TestClient(api.app) as c:
        yield c


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _run(c, **overrides):
    body = {"name": "t", "num_agents": 6, "max_rounds": 4, "seed": 7, **overrides}
    sim_id = c.post("/api/simulations", json=body).json()["id"]
    assert c.post(f"/api/simulations/{sim_id}/start").status_code == 200
    assert _wait(lambda: c.get(f"/api/simulations/{sim_id}").json()["status"] == "completed")
    return sim_id


# ----- a client that stops reading -----


class _StuckSocket:
    """A WebSocket whose peer never reads: every send waits for ever."""

    def __init__(self):
        self.close_codes = []

    async def send_json(self, _message):
        await asyncio.Event().wait()

    async def close(self, code=1000):
        self.close_codes.append(code)


class _RecordingSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)

    async def close(self, code=1000):
        pass


@pytest.fixture
def orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDFISH_OMPA_SEMANTIC", "0")
    orch = SwarmOrchestrator(OMPAMemoryAdapter(str(tmp_path / "vault")))
    orch.on_event(api._broadcast_event)
    yield orch
    api._websocket_clients.clear()
    orch.close()


async def _register(socket):
    subscriber = api._EventClient(socket)
    subscriber.start()
    api._websocket_clients.append(subscriber)
    return subscriber


async def test_stalled_client_does_not_stall_the_simulation(orchestrator):
    stuck = _StuckSocket()
    await _register(stuck)
    sim_id = orchestrator.create_simulation(
        SimulationConfig(name="s", num_agents=3, max_rounds=10, seed=1)
    )
    result = await asyncio.wait_for(orchestrator.run_simulation(sim_id), timeout=20)
    assert result.rounds_completed == 10


async def test_client_that_falls_behind_is_disconnected(orchestrator, monkeypatch):
    monkeypatch.setattr(api, "WS_OUTBOX_MAX", 3)
    stuck, healthy = _StuckSocket(), _RecordingSocket()
    stuck_client = await _register(stuck)
    await _register(healthy)

    sim_id = orchestrator.create_simulation(
        SimulationConfig(name="s", num_agents=3, max_rounds=12, seed=1)
    )
    result = await asyncio.wait_for(orchestrator.run_simulation(sim_id), timeout=20)
    assert result.rounds_completed == 12

    await asyncio.sleep(0.05)  # let the sender and closer tasks run
    assert stuck_client not in api._websocket_clients
    assert stuck.close_codes == [1013]
    # the other client was unaffected and saw every round, in order
    rounds = [m["data"]["round"] for m in healthy.messages if m["type"] == "round_complete"]
    assert rounds == list(range(12))


async def test_outbox_is_bounded(monkeypatch):
    monkeypatch.setattr(api, "WS_OUTBOX_MAX", 2)
    subscriber = api._EventClient(_StuckSocket())
    assert subscriber.offer({"n": 1}) and subscriber.offer({"n": 2})
    assert subscriber.offer({"n": 3}) is False


# ----- subscriptions -----


def _drain_until_pong(ws):
    """Events queued before the ping arrive before its pong (one ordered outbox)."""
    ws.send_json({"command": "ping"})
    seen = []
    while True:
        message = ws.receive_json()
        if message["type"] == "pong":
            return seen
        seen.append(message)


def test_default_stream_carries_every_simulation(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        a, b = _run(client), _run(client)
        events = [m for m in _drain_until_pong(ws) if m["type"] == "round_complete"]
    assert {m["data"]["simulation_id"] for m in events} == {a, b}
    assert len(events) == 8


def test_subscribe_filters_to_the_named_simulation(client):
    wanted = client.post(
        "/api/simulations", json={"name": "a", "num_agents": 4, "max_rounds": 3, "seed": 1}
    ).json()["id"]
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"command": "subscribe", "simulation_id": wanted})
        reply = ws.receive_json()
        assert reply["type"] == "subscribed" and reply["data"] == {"simulation_id": wanted}

        other = _run(client)  # a simulation this client did not ask for
        assert client.post(f"/api/simulations/{wanted}/start").status_code == 200
        assert _wait(
            lambda: client.get(f"/api/simulations/{wanted}").json()["status"] == "completed"
        )
        events = [m for m in _drain_until_pong(ws) if m["type"] == "round_complete"]

    assert events, "expected events for the subscribed simulation"
    assert {m["data"]["simulation_id"] for m in events} == {wanted}
    assert other not in {m["data"]["simulation_id"] for m in events}
    assert [m["data"]["round"] for m in events] == [0, 1, 2]


def test_unsubscribe_and_subscribe_all(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"command": "subscribe", "simulation_id": "not-a-real-id"})
        assert ws.receive_json()["type"] == "subscribed"
        _run(client)
        assert [m for m in _drain_until_pong(ws) if m["type"] == "round_complete"] == []

        ws.send_json({"command": "unsubscribe", "simulation_id": "not-a-real-id"})
        assert ws.receive_json()["type"] == "unsubscribed"
        _run(client)  # still filtered: an empty subscription set matches nothing
        assert [m for m in _drain_until_pong(ws) if m["type"] == "round_complete"] == []

        ws.send_json({"command": "subscribe"})  # no id: back to everything
        assert ws.receive_json()["data"] == {"simulation_id": None}
        sim_id = _run(client)
        events = [m for m in _drain_until_pong(ws) if m["type"] == "round_complete"]
        assert {m["data"]["simulation_id"] for m in events} == {sim_id}


def test_unknown_command_still_answers(client):
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"command": "dance"})
        reply = ws.receive_json()
        assert reply["type"] == "error" and "dance" in reply["data"]["message"]


# ----- results over HTTP after the run has finished -----


def test_agents_are_listed_after_completion(client):
    sim_id = _run(client, num_agents=6, max_rounds=8)
    agents = client.get(f"/api/simulations/{sim_id}/agents").json()
    assert len(agents) == 6
    assert {a["id"] for a in agents} == {f"{sim_id}_agent_{i}" for i in range(6)}
    assert sum(a["actions"] for a in agents) == 6 * 8
    assert all(a["type"] == "default" and 0 <= a["energy"] <= 100 for a in agents)


def test_results_describe_what_the_run_produced(client):
    sim_id = _run(
        client, num_agents=9, max_rounds=10, agent_types=["default", "influencer", "lurker"]
    )
    res = client.get(f"/api/simulations/{sim_id}/results").json()
    assert res["status"] == "completed" and res["rounds_completed"] == 10
    assert res["agent_types"] == {"default": 3, "influencer": 3, "lurker": 3}
    assert res["total_actions"] == 90
    assert len(res["top_agents"]) == 9
    reputations = [a["reputation"] for a in res["top_agents"]]
    assert reputations == sorted(reputations, reverse=True)
    assert {"id", "name", "type", "following", "posts", "actions"} <= set(res["top_agents"][0])


def test_results_top_agents_parameter(client):
    sim_id = _run(client, num_agents=6)
    assert len(client.get(f"/api/simulations/{sim_id}/results?top_agents=2").json()["top_agents"]) == 2
    assert client.get(f"/api/simulations/{sim_id}/results?top_agents=0").json()["top_agents"] == []
    assert client.get(f"/api/simulations/{sim_id}/results?top_agents=101").status_code == 422


def test_results_and_agents_match_the_mcp_summary(client):
    sim_id = _run(client, num_agents=5)
    orch = api.orchestrator
    summary = orch.results_summary(sim_id, 5)
    res = client.get(f"/api/simulations/{sim_id}/results?top_agents=5").json()
    assert res["top_agents"] == summary["top_agents"]
    assert res["total_actions"] == summary["total_actions"]


def test_agents_of_a_pending_simulation(client):
    sim_id = client.post(
        "/api/simulations", json={"name": "p", "num_agents": 3, "max_rounds": 2}
    ).json()["id"]
    assert client.get(f"/api/simulations/{sim_id}/agents").json() == []
    assert client.get(f"/api/simulations/{sim_id}/results").json()["top_agents"] == []
    assert client.get("/api/simulations/nope/agents").status_code == 404
