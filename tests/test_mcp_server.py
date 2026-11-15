"""MCP server tests through a real MCP client (in-process, plus one stdio run)."""

import asyncio
import json
import sys

import pytest

pytest.importorskip("mcp")
from mcp import Client, StdioServerParameters  # noqa: E402

from sandfish.core.orchestrator import SimulationLimits  # noqa: E402
from sandfish.mcp_server import EVENT_LOG_SIZE, build_server, main  # noqa: E402

EXPECTED_TOOLS = {
    "sf_create_simulation", "sf_start", "sf_pause", "sf_resume", "sf_stop",
    "sf_status", "sf_list", "sf_results", "sf_report",
    "sf_memory_search", "sf_memory_entity",
}


@pytest.fixture
def server(tmp_path):
    srv = build_server(str(tmp_path / "vault"), enable_semantic=False)
    yield srv
    srv.sandfish.close()


async def _call(client, tool, **args):
    result = await client.call_tool(tool, args)
    return result


async def _ok(client, tool, **args):
    result = await _call(client, tool, **args)
    assert not result.is_error, result.content[0].text
    return result.structured_content


async def _create(client, **overrides):
    args = {"name": "t", "num_agents": 4, "max_rounds": 5, "seed": 7,
            "start_date": "2026-03-01", "agent_types": ["default", "influencer", "lurker"]}
    args.update(overrides)
    return (await _ok(client, "sf_create_simulation", **args))["id"]


async def _wait_status(client, sim_id, want, timeout=10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        st = await _ok(client, "sf_status", simulation_id=sim_id)
        if st["status"] in want:
            return st
        await asyncio.sleep(0.01)
    raise AssertionError(f"status never reached {want}: {st}")


@pytest.mark.asyncio
async def test_lists_tools_and_resources(server):
    async with Client(server, cache=None) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert set(tools) == EXPECTED_TOOLS
        assert all(t.description for t in tools.values())
        schema = tools["sf_create_simulation"].input_schema
        assert schema["required"] == ["name"]
        assert {"num_agents", "max_rounds", "agent_types", "seed", "start_date"} <= set(schema["properties"])
        assert [str(r.uri) for r in (await c.list_resources()).resources] == ["sandfish://events"]
        templates = (await c.list_resource_templates()).resource_templates
        assert [t.uri_template for t in templates] == ["sandfish://events/{simulation_id}"]


@pytest.mark.asyncio
async def test_full_lifecycle_results_report_and_memory(server):
    async with Client(server, cache=None) as c:
        sim_id = await _create(c)
        created = await _ok(c, "sf_status", simulation_id=sim_id)
        assert created["status"] == "pending" and created["configured_agents"] == 4

        done = await _ok(c, "sf_start", simulation_id=sim_id, wait_seconds=20)
        assert done["status"] == "completed" and done["rounds_completed"] == 5

        listed = await _ok(c, "sf_list")
        assert listed["count"] == 1 and listed["simulations"][0]["id"] == sim_id

        res = await _ok(c, "sf_results", simulation_id=sim_id, top_agents=2)
        assert res["metrics"]["completion_rate"] == 1.0
        assert res["agent_types"] == {"default": 2, "influencer": 1, "lurker": 1}
        assert res["total_actions"] == 20
        assert len(res["top_agents"]) == 2
        assert res["top_agents"][0]["reputation"] >= res["top_agents"][1]["reputation"]

        report = await _call(c, "sf_report", simulation_id=sim_id)
        text = report.content[0].text
        assert text.startswith(f"# Simulation t ({sim_id})")
        assert "**completed**, 5/5 rounds" in text and "| Agent |" in text

        hits = await _ok(c, "sf_memory_search", query="follow", limit=3)
        assert 0 < len(hits["results"]) <= 3
        assert all(not r["source"].startswith("/") for r in hits["results"])

        entity = await _ok(c, "sf_memory_entity", name=f"{sim_id}_agent_0")
        assert entity["found"] is True
        assert entity["entity"]["attributes"]["display_name"] == "Agent_0"
        assert entity["counts"]["facts"] >= 1
        early = await _ok(c, "sf_memory_entity", name=f"{sim_id}_agent_0", as_of="2026-02-01")
        assert early["counts"]["incoming"] == 0 and early["counts"]["outgoing"] == 0

        missing = await _ok(c, "sf_memory_entity", name="nobody")
        assert missing["found"] is False


@pytest.mark.asyncio
async def test_seeded_runs_are_reproducible_over_mcp(tmp_path):
    outcomes = []
    for i in range(2):
        srv = build_server(str(tmp_path / f"v{i}"), enable_semantic=False)
        try:
            async with Client(srv, cache=None) as c:
                sim_id = await _create(c, max_rounds=8, seed=42)
                await _ok(c, "sf_start", simulation_id=sim_id, wait_seconds=20)
                res = await _ok(c, "sf_results", simulation_id=sim_id, top_agents=50)
                outcomes.append(sorted(
                    (a["id"].split("_agent_")[1], a["reputation"], a["following"], a["posts"])
                    for a in res["top_agents"]
                ))
        finally:
            srv.sandfish.close()
    assert outcomes[0] == outcomes[1]


@pytest.mark.asyncio
async def test_pause_resume_stop(server):
    app = server.sandfish
    executed = []
    original = app.orchestrator._execute_round

    async def slow(sim_id, round_num):
        executed.append(round_num)
        await asyncio.sleep(0.02)
        await original(sim_id, round_num)

    app.orchestrator._execute_round = slow
    async with Client(server, cache=None) as c:
        sim_id = await _create(c, max_rounds=30)
        await _ok(c, "sf_start", simulation_id=sim_id)
        while len(executed) < 2:
            await asyncio.sleep(0.01)
        paused = await _ok(c, "sf_pause", simulation_id=sim_id)
        assert paused["status"] in ("pausing", "paused")
        st = await _wait_status(c, sim_id, {"paused"})
        assert st["rounds_completed"] == len(executed)

        again = await _call(c, "sf_pause", simulation_id=sim_id)
        assert again.is_error and "not running" in again.content[0].text
        start = await _call(c, "sf_start", simulation_id=sim_id)
        assert start.is_error  # paused simulations use sf_resume

        await _ok(c, "sf_resume", simulation_id=sim_id)
        while len(executed) < st["rounds_completed"] + 2:
            await asyncio.sleep(0.01)
        stopped = await _ok(c, "sf_stop", simulation_id=sim_id)
        assert stopped["status"] == "stopped"
        await asyncio.sleep(0.1)
        assert executed == list(range(len(executed))), "no round re-run after resume"

        events = json.loads((await c.read_resource(f"sandfish://events/{sim_id}")).contents[0].text)
        types = [e["type"] for e in events]
        for expected in ("created", "started", "pause_requested", "resumed", "stop_requested", "round_complete", "run_ended"):
            assert expected in types, (expected, types)
        assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)


@pytest.mark.asyncio
async def test_errors_are_tool_errors(server):
    async with Client(server, cache=None) as c:
        bad = await _call(c, "sf_create_simulation", name="x", agent_types=["nope"])
        assert bad.is_error and "Unknown agent type" in bad.content[0].text
        empty = await _call(c, "sf_create_simulation", name="x", agent_types=[])
        assert empty.is_error
        bad_date = await _call(c, "sf_create_simulation", name="x", start_date="tomorrow")
        assert bad_date.is_error
        for tool in ("sf_status", "sf_start", "sf_pause", "sf_resume", "sf_stop", "sf_results", "sf_report"):
            r = await _call(c, tool, simulation_id="does-not-exist")
            assert r.is_error and "Unknown simulation_id" in r.content[0].text, tool
        assert (await _call(c, "sf_memory_search", query="  ")).is_error
        sim_id = await _create(c, max_rounds=1)
        assert (await _call(c, "sf_resume", simulation_id=sim_id)).is_error
        assert (await _call(c, "sf_stop", simulation_id=sim_id)).is_error


@pytest.mark.asyncio
async def test_limits_apply(tmp_path):
    srv = build_server(str(tmp_path / "v"), enable_semantic=False,
                       limits=SimulationLimits(max_agents=3, max_rounds=10, max_concurrent=1))
    app = srv.sandfish
    original = app.orchestrator._execute_round

    async def slow(sim_id, round_num):
        await asyncio.sleep(0.02)
        await original(sim_id, round_num)

    app.orchestrator._execute_round = slow
    try:
        async with Client(srv, cache=None) as c:
            assert (await _call(c, "sf_create_simulation", name="big", num_agents=4)).is_error
            assert (await _call(c, "sf_create_simulation", name="long", num_agents=1, max_rounds=11)).is_error
            a = await _create(c, num_agents=2, max_rounds=10, agent_types=["default"])
            b = await _create(c, num_agents=2, max_rounds=2, agent_types=["default"])
            await _ok(c, "sf_start", simulation_id=a)
            second = await _call(c, "sf_start", simulation_id=b)
            assert second.is_error and "limit" in second.content[0].text
            assert (await _ok(c, "sf_status", simulation_id=b))["status"] == "pending"
            await _ok(c, "sf_stop", simulation_id=a)
    finally:
        app.close()


@pytest.mark.asyncio
async def test_event_log_is_bounded(server):
    app = server.sandfish
    for i in range(EVENT_LOG_SIZE + 50):
        app.log_event("noise", "x", i=i)
    async with Client(server, cache=None) as c:
        events = json.loads((await c.read_resource("sandfish://events")).contents[0].text)
    assert len(events) == 500
    assert events[-1]["data"]["i"] == EVENT_LOG_SIZE + 49
    assert len(app.events) == EVENT_LOG_SIZE


@pytest.mark.asyncio
async def test_stdio_transport_end_to_end(tmp_path):
    """Launch the real `python -m sandfish.mcp_server` over stdio."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "sandfish.mcp_server", "--vault", str(tmp_path / "vault")],
        env={"SANDFISH_OMPA_SEMANTIC": "0", "PATH": "/usr/bin:/bin",
             "PYTHONPATH": ":".join(p for p in sys.path if p)},
    )
    async with Client(params, cache=None) as c:
        assert {t.name for t in (await c.list_tools()).tools} == EXPECTED_TOOLS
        sim_id = await _create(c, max_rounds=3)
        done = await _ok(c, "sf_start", simulation_id=sim_id, wait_seconds=30)
        assert done["status"] == "completed" and done["rounds_completed"] == 3
        events = json.loads((await c.read_resource("sandfish://events")).contents[0].text)
        assert any(e["type"] == "run_ended" for e in events)


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "sandfish-mcp" in capsys.readouterr().out
