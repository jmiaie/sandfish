"""API wiring for the simulated calendar (real app, in-process)."""

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from sandfish.api.main import app, configure_app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("SANDFISH_API_KEY", raising=False)
    monkeypatch.setenv("SANDFISH_OMPA_SEMANTIC", "0")
    configure_app(str(tmp_path / "vault"))
    with TestClient(app) as c:
        yield c


def test_create_with_start_date(client):
    r = client.post("/api/simulations", json={"name": "cal", "start_date": "2026-04-01"})
    assert r.status_code == 200


@pytest.mark.parametrize("bad", ["2026/04/01", "2026-13-01", "yesterday"])
def test_invalid_start_date_is_422(client, bad):
    r = client.post("/api/simulations", json={"name": "cal", "start_date": bad})
    assert r.status_code == 422
