"""API hardening: bind safety, auth throttling, key encoding, WebSocket auth,
pause/resume over HTTP, limits, background failure logging, the audit
endpoint, and docs drift."""

import asyncio
import logging
import time
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

import sandfish.api.main as api  # noqa: E402

KEY = "fake" * 6  # placeholder, not a credential


def _fresh_limiters(monkeypatch, auth_failures=10, per_minute=1000):
    monkeypatch.setattr(api, "_limiter", api._SlidingWindowLimiter(per_minute))
    monkeypatch.setattr(api, "_auth_fail_limiter", api._SlidingWindowLimiter(auth_failures))


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDFISH_OMPA_SEMANTIC", "0")
    _fresh_limiters(monkeypatch)

    def _make(key=None, **env):
        if key:
            monkeypatch.setenv("SANDFISH_API_KEY", key)
        else:
            monkeypatch.delenv("SANDFISH_API_KEY", raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, str(value))
        api.configure_app(str(tmp_path / "vault"))
        return TestClient(api.app)

    return _make


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _slow_rounds(delay=0.02):
    """Patch the live orchestrator so each round takes `delay` seconds;
    returns the list of executed round indices."""
    orch = api.orchestrator
    executed = []
    original = orch._execute_round

    async def slow(sim_id, round_num):
        executed.append(round_num)
        await asyncio.sleep(delay)
        await original(sim_id, round_num)

    orch._execute_round = slow
    return executed


# ----- bind -----


@pytest.mark.parametrize("host,loop", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("[::1]", True), ("127.0.0.5", True),
    ("0.0.0.0", False), ("::", False), ("192.168.1.10", False), ("example.com", False),
])
def test_is_loopback_host(host, loop):
    assert api.is_loopback_host(host) is loop


def test_public_bind_requires_api_key(monkeypatch):
    monkeypatch.delenv("SANDFISH_API_KEY", raising=False)
    api.check_bind_allowed("127.0.0.1")
    with pytest.raises(RuntimeError, match="SANDFISH_API_KEY"):
        api.check_bind_allowed("0.0.0.0")
    monkeypatch.setenv("SANDFISH_API_KEY", KEY)
    api.check_bind_allowed("0.0.0.0")


def test_cli_api_defaults_to_loopback_and_refuses_public_bind(monkeypatch, capsys):
    import uvicorn

    from sandfish import cli

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.append(host))
    monkeypatch.delenv("SANDFISH_API_KEY", raising=False)

    assert cli.main(["api", "--host", "0.0.0.0"]) == 2
    assert calls == []
    assert "Refusing to bind" in capsys.readouterr().err

    assert cli.main(["api"]) == 0
    assert calls == ["127.0.0.1"]

    monkeypatch.setenv("SANDFISH_API_KEY", KEY)
    assert cli.main(["api", "--host", "0.0.0.0"]) == 0
    assert calls[-1] == "0.0.0.0"


# ----- auth -----


SHORT_KEY = "short-key"  # placeholder; under MIN_STRONG_KEY_CHARS on purpose


def test_failed_auth_is_rate_limited_per_ip(make_client, monkeypatch):
    _fresh_limiters(monkeypatch, auth_failures=3)
    with make_client(KEY) as c:
        codes = [c.get("/api/simulations", headers={"X-API-Key": f"wrong-{i}"}).status_code
                 for i in range(5)]
        assert codes == [401, 401, 401, 429, 429]
    # Successful requests never count as failures.
    _fresh_limiters(monkeypatch, auth_failures=3)
    with make_client(KEY) as c:
        for _ in range(5):
            assert c.get("/api/simulations", headers={"X-API-Key": KEY}).status_code == 200


def test_wrong_keys_do_not_lock_out_the_right_key(make_client, monkeypatch):
    """Behind a NAT or a reverse proxy every client has the same address, so
    one client's wrong keys must not shut out the others."""
    _fresh_limiters(monkeypatch, auth_failures=3)
    with make_client(KEY) as c:
        for i in range(6):
            c.get("/api/simulations", headers={"X-API-Key": f"wrong-{i}"})
        assert c.get("/api/simulations", headers={"X-API-Key": "wrong"}).status_code == 429
        for _ in range(3):
            assert c.get("/api/simulations", headers={"X-API-Key": KEY}).status_code == 200
        # The address is still over budget for wrong keys.
        assert c.get("/api/simulations", headers={"X-API-Key": "wrong"}).status_code == 429
        with c.websocket_connect("/ws", headers={"X-API-Key": KEY}) as ws:
            assert ws.receive_json()["type"] == "connected"


def test_a_short_key_still_locks_out_an_address_over_budget(make_client, monkeypatch, caplog):
    """A key short enough to guess is only protected by the limit on tries,
    and letting the right key through would remove that limit."""
    assert api.api_key_is_guessable(SHORT_KEY) and not api.api_key_is_guessable(KEY)
    assert not api.api_key_is_guessable("k" * api.MIN_STRONG_KEY_CHARS)
    _fresh_limiters(monkeypatch, auth_failures=3)
    with caplog.at_level(logging.WARNING, logger="sandfish.api"), make_client(SHORT_KEY) as c:
        assert c.get("/api/simulations", headers={"X-API-Key": SHORT_KEY}).status_code == 200
        codes = [c.get("/api/simulations", headers={"X-API-Key": f"wrong-{i}"}).status_code
                 for i in range(4)]
        assert codes == [401, 401, 401, 429]
        assert c.get("/api/simulations", headers={"X-API-Key": SHORT_KEY}).status_code == 429
    assert any("shorter than" in r.getMessage() for r in caplog.records)


def test_a_strong_key_starts_without_the_short_key_warning(make_client, caplog):
    with caplog.at_level(logging.WARNING, logger="sandfish.api"), make_client(KEY) as c:
        assert c.get("/health").status_code == 200
    assert not any("shorter than" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("bad", ["ключ".encode("utf-8"), "clé-ü".encode("latin-1"), b"\xff\xfe"])
def test_non_ascii_key_is_401_not_500(make_client, bad):
    with make_client(KEY) as c:
        r = c.get("/api/simulations", headers={"X-API-Key": bad})
        assert r.status_code == 401


def test_non_ascii_configured_key_works(make_client):
    secret = "pässwörd-ключ"
    with make_client(secret) as c:
        r = c.get("/api/simulations", headers={"X-API-Key": secret.encode("utf-8")})
        assert r.status_code == 200


# ----- WebSocket auth -----


def test_ws_header_auth(make_client):
    with make_client(KEY) as c:
        with c.websocket_connect("/ws", headers={"X-API-Key": KEY}) as ws:
            assert ws.receive_json()["type"] == "connected"


def test_ws_first_message_auth(make_client):
    with make_client(KEY) as c:
        with c.websocket_connect("/ws") as ws:
            ws.send_json({"command": "auth", "api_key": KEY})
            assert ws.receive_json()["type"] == "authenticated"
            assert ws.receive_json()["type"] == "connected"
            ws.send_json({"command": "ping"})
            assert ws.receive_json()["type"] == "pong"


def test_ws_query_param_still_accepted(make_client, caplog):
    with make_client(KEY) as c, caplog.at_level(logging.WARNING, logger="sandfish.api"):
        with c.websocket_connect(f"/ws?api_key={KEY}") as ws:
            assert ws.receive_json()["type"] == "connected"
    assert "deprecated" in caplog.text


@pytest.mark.parametrize("first", [
    {"command": "auth", "api_key": "wrong"},
    {"command": "ping"},
    {"command": "auth", "api_key": 123},
])
def test_ws_bad_first_message_is_rejected(make_client, first):
    with make_client(KEY) as c:
        with c.websocket_connect("/ws") as ws:
            ws.send_json(first)
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008


def test_ws_wrong_header_key_is_rejected(make_client):
    with make_client(KEY) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws", headers={"X-API-Key": "nope"}) as ws:
                ws.receive_json()


def test_ws_auth_times_out(make_client, monkeypatch):
    monkeypatch.setattr(api, "WS_AUTH_TIMEOUT_S", 0.05)
    with make_client(KEY) as c:
        with c.websocket_connect("/ws") as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
            assert exc.value.code == 1008


def test_ws_without_configured_key_is_open(make_client):
    with make_client() as c:
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "connected"


# ----- validation and limits -----


@pytest.mark.parametrize("types", [["nope"], [], ["default", "llm-typo"]])
def test_unknown_agent_types_rejected_at_create(make_client, types):
    with make_client() as c:
        r = c.post("/api/simulations", json={"name": "v", "agent_types": types})
        assert r.status_code == 422
        assert c.get("/api/simulations").json() == []


def test_agent_and_round_caps(make_client):
    with make_client(SANDFISH_MAX_AGENTS=5, SANDFISH_MAX_ROUNDS=10) as c:
        assert c.post("/api/simulations", json={"name": "a", "num_agents": 6}).status_code == 422
        assert c.post("/api/simulations", json={"name": "b", "max_rounds": 11}).status_code == 422
        r = c.post("/api/simulations", json={"name": "ok", "num_agents": 5, "max_rounds": 10})
        assert r.status_code == 200


def test_concurrent_simulation_cap(make_client):
    with make_client(SANDFISH_MAX_CONCURRENT_SIMULATIONS=1) as c:
        _slow_rounds()
        a = c.post("/api/simulations", json={"name": "a", "max_rounds": 500, "num_agents": 1}).json()["id"]
        b = c.post("/api/simulations", json={"name": "b", "max_rounds": 2, "num_agents": 1}).json()["id"]
        assert c.post(f"/api/simulations/{a}/start").status_code == 200
        r = c.post(f"/api/simulations/{b}/start")
        assert r.status_code == 429
        assert c.get(f"/api/simulations/{b}").json()["status"] == "pending"
        assert c.post(f"/api/simulations/{a}/stop").status_code == 200
        assert c.get(f"/api/simulations/{a}").json()["status"] == "stopped"
        # The stopped run still holds its slot until its round finishes.
        assert _wait(lambda: not api.orchestrator._loop_active(a))
        assert c.post(f"/api/simulations/{b}/start").status_code == 200


def test_finished_simulations_are_evicted(make_client):
    with make_client(SANDFISH_MAX_FINISHED_SIMULATIONS=2) as c:
        ids = []
        for i in range(3):
            sid = c.post("/api/simulations", json={"name": f"s{i}", "max_rounds": 1, "num_agents": 1}).json()["id"]
            ids.append(sid)
            assert c.post(f"/api/simulations/{sid}/start").status_code == 200
            assert _wait(lambda: c.get(f"/api/simulations/{sid}").json()["status"] == "completed")
        assert c.get(f"/api/simulations/{ids[0]}").status_code == 404
        assert {s["id"] for s in c.get("/api/simulations").json()} == set(ids[1:])


# ----- lifecycle over HTTP -----


def test_pause_resume_race_runs_each_round_once(make_client):
    with make_client() as c:
        executed = _slow_rounds()
        sid = c.post("/api/simulations", json={"name": "race", "max_rounds": 40, "num_agents": 2}).json()["id"]
        assert c.post(f"/api/simulations/{sid}/start").json()["status"] == "started"
        assert _wait(lambda: len(executed) >= 2)

        # Pause then resume immediately: the loop is still finishing its
        # round, so resume cancels the pause instead of racing a 2nd loop.
        assert c.post(f"/api/simulations/{sid}/pause").json()["status"] == "pausing"
        assert c.get(f"/api/simulations/{sid}").json()["status"] in ("pausing", "paused")
        r = c.post(f"/api/simulations/{sid}/resume")
        assert r.status_code == 200 and r.json()["status"] in ("running", "resumed")

        # Proper pause, observe it, then resume from the next round.
        assert c.post(f"/api/simulations/{sid}/pause").status_code == 200
        assert _wait(lambda: c.get(f"/api/simulations/{sid}").json()["status"] == "paused")
        paused = c.get(f"/api/simulations/{sid}").json()
        assert paused["rounds_completed"] == len(executed)
        assert c.post(f"/api/simulations/{sid}/start").status_code == 409  # use /resume
        assert c.post(f"/api/simulations/{sid}/resume").json()["status"] == "resumed"

        assert _wait(lambda: c.get(f"/api/simulations/{sid}").json()["status"] == "completed")
        assert executed == list(range(40)), "no round may run twice or be skipped"
        res = c.get(f"/api/simulations/{sid}/results").json()
        assert res["rounds_completed"] == 40
        assert c.post(f"/api/simulations/{sid}/start").status_code == 409


def test_resume_requires_pause(make_client):
    with make_client() as c:
        sid = c.post("/api/simulations", json={"name": "x", "max_rounds": 1}).json()["id"]
        assert c.post(f"/api/simulations/{sid}/resume").status_code == 409


def test_background_failures_are_logged(make_client, caplog):
    with make_client() as c, caplog.at_level(logging.ERROR, logger="sandfish.api"):
        sid = c.post("/api/simulations", json={"name": "boom", "max_rounds": 1}).json()["id"]

        async def explode(sim_id):
            raise RuntimeError("kaboom in background")

        api.orchestrator.run_simulation = explode
        assert c.post(f"/api/simulations/{sid}/start").status_code == 200
        assert _wait(lambda: "kaboom in background" in caplog.text)
    assert "background run failed" in caplog.text


def test_failed_simulation_is_logged(make_client, caplog):
    with make_client() as c, caplog.at_level(logging.ERROR, logger="sandfish.api"):
        orch = api.orchestrator

        async def bad_round(sim_id, round_num):
            raise ValueError("round broke")

        orch._execute_round = bad_round
        sid = c.post("/api/simulations", json={"name": "f", "max_rounds": 2}).json()["id"]
        c.post(f"/api/simulations/{sid}/start")
        assert _wait(lambda: c.get(f"/api/simulations/{sid}").json()["status"] == "failed")
        assert _wait(lambda: "round broke" in caplog.text)


# ----- audit endpoint -----


def test_audit_runs_off_loop_on_package_root(make_client, monkeypatch):
    seen = {}

    def fake_audit(root):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        seen["root"] = root
        return []

    monkeypatch.setattr(api, "run_security_audit", fake_audit)
    with make_client() as c:
        r = c.post("/api/security/audit")
        assert r.status_code == 200 and r.json()["findings_count"] == 0
    assert seen["on_loop"] is False
    assert Path(seen["root"]) == Path(api.__file__).resolve().parents[1]


# ----- docs -----


def test_readme_documents_ws_not_api_events():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "/api/events" not in readme
    assert "WS   /ws" in readme


# ----- review follow-ups -----


@pytest.mark.parametrize("origin,key,cors,ok", [
    (None, None, "", True),
    ("http://localhost:3000", None, "", True),
    ("http://127.0.0.1:8080", None, "", True),
    ("https://evil.example", None, "", False),
    ("https://evil.example", KEY, "", False),       # same CorsPolicy as REST
    ("https://app.example", None, "https://app.example", True),
    ("http://localhost:3000", None, "https://app.example", False),
])
def test_ws_origin_policy(make_client, monkeypatch, origin, key, cors, ok):
    monkeypatch.setenv("SANDFISH_CORS_ORIGINS", cors)
    headers = {"origin": origin} if origin else {}
    if key:
        headers["X-API-Key"] = key
    with make_client(key) as c:
        if ok:
            with c.websocket_connect("/ws", headers=headers) as ws:
                assert ws.receive_json()["type"] == "connected"
        else:
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect("/ws", headers=headers) as ws:
                    ws.receive_json()


def test_ws_pending_auth_cap(make_client, monkeypatch):
    monkeypatch.setattr(api, "WS_MAX_PENDING_AUTH", 0)
    with make_client(KEY) as c:
        with pytest.raises(WebSocketDisconnect) as exc:
            with c.websocket_connect("/ws") as ws:
                ws.receive_json()
        assert exc.value.code == 1013
        # Header auth doesn't wait, so it isn't subject to the cap.
        with c.websocket_connect("/ws", headers={"X-API-Key": KEY}) as ws:
            assert ws.receive_json()["type"] == "connected"


def test_ws_failed_first_messages_count_toward_lockout(make_client, monkeypatch):
    _fresh_limiters(monkeypatch, auth_failures=2)
    with make_client(KEY) as c:
        for _ in range(2):
            with c.websocket_connect("/ws") as ws:
                ws.send_json({"command": "ping"})
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()
        # Two failures used the budget: a wrong key is now refused with 429...
        assert c.get("/api/simulations", headers={"X-API-Key": "wrong"}).status_code == 429
        # ...and the right key still works.
        assert c.get("/api/simulations", headers={"X-API-Key": KEY}).status_code == 200


@pytest.mark.asyncio
async def test_limiter_key_map_stays_bounded():
    lim = api._SlidingWindowLimiter(5)
    lim.max_keys = 50
    for i in range(500):
        await lim.record(f"10.0.{i // 256}.{i % 256}")
    assert len(lim._hits) <= 50
