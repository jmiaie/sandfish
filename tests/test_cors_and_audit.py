"""CORS policy (SANDFISH_CORS_ORIGINS) and the pip-audit dependency check."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from sandfish.security.audit import SecurityAuditor

KEY = "fake" * 6  # placeholder, not a credential


# ----- CORS -----

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import sandfish.api.main as api  # noqa: E402


def test_policy_parsing():
    p = api.cors_policy({})
    assert p.loopback and not p.any_origin and p.origins == ()
    assert api.cors_policy({"SANDFISH_CORS_ORIGINS": "localhost"}) == p
    none = api.cors_policy({"SANDFISH_CORS_ORIGINS": "none"})
    assert not none.loopback and not none.allows("http://localhost:3000")
    listed = api.cors_policy({"SANDFISH_CORS_ORIGINS": "https://app.example/, localhost"})
    assert listed.origins == ("https://app.example",) and listed.loopback
    only = api.cors_policy({"SANDFISH_CORS_ORIGINS": "https://app.example"})
    assert not only.loopback and not only.allows("http://localhost:3000")
    assert only.allows(None)  # non-browser requests carry no Origin


@pytest.mark.parametrize("value", ["*", "https://a.example,*"])
def test_star_refused_without_auth(value):
    with pytest.raises(RuntimeError, match="auth is off"):
        api.cors_policy({"SANDFISH_CORS_ORIGINS": value})
    p = api.cors_policy({"SANDFISH_CORS_ORIGINS": value, "SANDFISH_API_KEY": KEY})
    assert p.any_origin and p.allows("https://anything.example")


@pytest.mark.parametrize("value", ["none,https://a.example", "app.example", "ftp://x"])
def test_invalid_values_rejected(value):
    with pytest.raises(RuntimeError):
        api.cors_policy({"SANDFISH_CORS_ORIGINS": value})


@pytest.mark.parametrize("origin,ok", [
    ("http://localhost:5173", True), ("http://127.0.0.1:8080", True),
    ("http://[::1]:3000", True), ("https://localhost", True),
    ("https://evil.example", False), ("http://localhost.evil.example", False),
    ("http://127.0.0.1.evil.example", False),
])
def test_loopback_matching(origin, ok):
    assert api.cors_policy({}).allows(origin) is ok
    import re
    assert bool(re.match(api._LOOPBACK_ORIGIN_RE, origin)) is ok


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDFISH_OMPA_SEMANTIC", "0")
    monkeypatch.delenv("SANDFISH_API_KEY", raising=False)
    monkeypatch.delenv("SANDFISH_CORS_ORIGINS", raising=False)
    monkeypatch.setattr(api, "_limiter", api._SlidingWindowLimiter(1000))
    monkeypatch.setattr(api, "_auth_fail_limiter", api._SlidingWindowLimiter(1000))
    api.configure_app(str(tmp_path / "vault"))

    def make(cors=None, key=None):
        if cors is not None:
            monkeypatch.setenv("SANDFISH_CORS_ORIGINS", cors)
        if key:
            monkeypatch.setenv("SANDFISH_API_KEY", key)
        return TestClient(api.app)

    return make


def _preflight(c, origin):
    return c.options("/api/simulations", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })


@pytest.mark.parametrize("cors,key,origin,allowed", [
    (None, None, "http://localhost:3000", True),
    (None, None, "https://evil.example", False),
    ("none", None, "http://localhost:3000", False),
    ("https://app.example", None, "https://app.example", True),
    ("https://app.example", None, "http://localhost:3000", False),
    ("https://app.example,localhost", None, "http://localhost:3000", True),
    ("*", KEY, "https://evil.example", True),
])
def test_preflight_and_simple_requests(client, cors, key, origin, allowed):
    with client(cors, key) as c:
        r = _preflight(c, origin)
        if allowed:
            assert r.status_code == 200
            assert r.headers["access-control-allow-origin"] in (origin, "*")
            assert "access-control-allow-credentials" not in r.headers
        else:
            assert r.status_code == 400
            assert "access-control-allow-origin" not in r.headers
        h = c.get("/health", headers={"Origin": origin})
        assert h.status_code == 200
        assert ("access-control-allow-origin" in h.headers) is allowed


def test_star_without_key_refuses_startup(client):
    with pytest.raises(RuntimeError, match="auth is off"):
        with client("*"):
            pass


def test_star_without_key_refuses_bind(monkeypatch):
    monkeypatch.delenv("SANDFISH_API_KEY", raising=False)
    monkeypatch.setenv("SANDFISH_CORS_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="auth is off"):
        api.check_bind_allowed("127.0.0.1")


def test_ws_follows_cors_policy(client):
    from starlette.websockets import WebSocketDisconnect

    with client("https://app.example") as c:
        with c.websocket_connect("/ws", headers={"origin": "https://app.example"}) as ws:
            assert ws.receive_json()["type"] == "connected"
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws", headers={"origin": "http://localhost:3000"}) as ws:
                ws.receive_json()


# ----- pip-audit -----


@pytest.fixture(autouse=True)
def _audit_enabled(monkeypatch):
    # CI disables the (networked) dependency audit globally; these tests
    # mock subprocess and need it on.
    monkeypatch.delenv("SANDFISH_AUDIT_DEPENDENCIES", raising=False)


def _auditor(tmp_path):
    return SecurityAuditor(tmp_path)


def _fake_run(stdout, returncode=0, stderr=""):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

    return run, calls


def test_pip_audit_clean(tmp_path, monkeypatch):
    run, calls = _fake_run(json.dumps({"dependencies": [{"name": "fastapi", "version": "1.0", "vulns": []}]}))
    monkeypatch.setattr(subprocess, "run", run)
    a = _auditor(tmp_path)
    a._audit_dependencies()
    assert a.findings == []
    joined = " ".join(calls[0])
    assert "pip_audit" in joined or "pip-audit" in joined
    assert "--format" in calls[0] and "json" in calls[0]


def test_pip_audit_vulnerabilities_become_findings(tmp_path, monkeypatch):
    payload = {"dependencies": [
        {"name": "ok", "version": "1", "vulns": []},
        {"name": "badpkg", "version": "1.0", "vulns": [
            {"id": "PYSEC-1", "fix_versions": ["1.2"]},
            {"id": "GHSA-x", "fix_versions": ["1.1", "1.2"]},
        ]},
        {"name": "nofix", "version": "2.0", "vulns": [{"id": "CVE-9", "fix_versions": []}]},
    ]}
    run, _ = _fake_run(json.dumps(payload), returncode=1)
    monkeypatch.setattr(subprocess, "run", run)
    a = _auditor(tmp_path)
    a._audit_dependencies()
    assert [f.severity for f in a.findings] == ["HIGH", "HIGH"]
    assert "badpkg 1.0" in a.findings[0].description and "PYSEC-1, GHSA-x" in a.findings[0].description
    assert a.findings[0].remediation == "Upgrade badpkg to 1.1 or 1.2."
    assert "No fixed release" in a.findings[1].remediation


@pytest.mark.parametrize("stdout,code", [("", 2), ("not json", 1), ('{"dependencies": []}', 2)])
def test_pip_audit_tool_failure_is_not_a_finding(tmp_path, monkeypatch, caplog, stdout, code):
    run, _ = _fake_run(stdout, returncode=code, stderr="boom: network down")
    monkeypatch.setattr(subprocess, "run", run)
    a = _auditor(tmp_path)
    a._audit_dependencies()
    assert a.findings == []
    assert "pip-audit` failed" in caplog.text


def test_pip_audit_timeout(tmp_path, monkeypatch, caplog):
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(subprocess, "run", run)
    a = _auditor(tmp_path)
    a._audit_dependencies()
    assert a.findings == [] and "invocation failed" in caplog.text


def test_pip_audit_missing_or_disabled(tmp_path, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="sandfish.security")
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a))
    a = _auditor(tmp_path)
    monkeypatch.setattr(a, "_pip_audit_command", lambda: None)
    a._audit_dependencies()
    assert "not installed" in caplog.text

    monkeypatch.setenv("SANDFISH_AUDIT_DEPENDENCIES", "0")
    b = _auditor(tmp_path)
    b._audit_dependencies()
    assert called == [] and "disabled" in caplog.text


def test_pip_audit_command_prefers_this_interpreter(tmp_path):
    import sys

    cmd = _auditor(tmp_path)._pip_audit_command()
    pytest.importorskip("pip_audit")
    assert cmd == [sys.executable, "-m", "pip_audit"]
