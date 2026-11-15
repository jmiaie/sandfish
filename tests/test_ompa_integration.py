"""Integration tests against the real OMPA package (no mocks).

Semantic search is disabled so OMPA uses its offline BM25 index: no model
download and no network, so these run free in CI.
"""

from collections import Counter

import pytest

ompa = pytest.importorskip("ompa")

from sandfish.agents.base_agent import Action, ActionType, AgentProfile, DefaultAgent  # noqa: E402
from sandfish.core.orchestrator import SimulationConfig, SwarmOrchestrator  # noqa: E402
from sandfish.memory.ompa_adapter import (  # noqa: E402
    EVENT_NOTE_DIR,
    Entity,
    Fact,
    OMPAMemoryAdapter,
    SearchResult,
    _InMemoryBackend,
)


@pytest.fixture
def memory(tmp_path):
    return OMPAMemoryAdapter(str(tmp_path / "vault"), enable_semantic=False)


def test_uses_real_ompa_backend(memory):
    assert memory.backend_name == "ompa"
    assert isinstance(memory.backend, ompa.Ompa)
    info = memory.start_session()
    assert info["success"] is True and info["backend"] == "ompa"


def test_entity_roundtrip_handles_triple_objects(memory):
    memory.add_entity("Alice", "influencer", {"team": "Blue", "empty": ""})
    entity = memory.get_entity("Alice")
    assert isinstance(entity, Entity)
    assert entity.entity_type == "influencer"
    assert entity.attributes == {"team": "Blue"}
    assert memory.get_entity("Nobody") is None


def test_related_entities(memory):
    memory.add_entity("Bob", "default")
    memory.add_entity("Alice", "default", {"mentor": "Bob"})
    related = memory.get_related_entities("Alice")
    assert [e.name for e in related] == ["Bob"]
    assert memory.get_related_entities("Alice", relation="other") == []


def test_events_become_searchable(memory):
    memory.record_event("ACTION_EXECUTED", "Zephyr followed Quill", {"agent_id": "Zephyr"})
    memory.record_event("ACTION_EXECUTED", "Marlow liked a post", {"agent_id": "Marlow"})
    hits = memory.search("Zephyr Quill", limit=5)  # search() flushes first
    assert hits, "events must be searchable through OMPA"
    assert all(isinstance(h, SearchResult) for h in hits)
    assert "Zephyr followed Quill" in hits[0].content
    assert not hits[0].content.startswith("---")  # frontmatter stripped
    assert EVENT_NOTE_DIR in hits[0].source
    assert hits[0].metadata.get("match_type")


def test_events_batched_into_dated_notes(memory, tmp_path):
    for i in range(10):
        memory.record_event("TICK", f"tick {i}", date="2026-03-01")
    memory.record_event("TICK", "next day", date="2026-03-02")
    assert memory.flush() == 2  # one note per date
    notes = sorted((tmp_path / "vault" / EVENT_NOTE_DIR).rglob("*.md"))
    assert [n.parent.name for n in notes] == ["2026-03-01", "2026-03-02"]
    assert "date: '2026-03-01'" in notes[0].read_text()
    assert memory.flush() == 0


def test_invalid_event_date_rejected(memory):
    with pytest.raises(ValueError):
        memory.record_event("X", "bad", date="03/01/2026")


def test_temporal_ranking_prefers_referenced_day(memory):
    # Identical text, so only the note date can change the order.
    for day in ("2026-03-01", "2026-03-05", "2026-03-09"):
        memory.record_event("TRADE", "Corvid traded tokens", date=day)
    q = "what did Corvid trade yesterday"
    plain = memory.search(q, limit=3, temporal=False, reference_date="2026-03-10")
    dated = memory.search(q, limit=3, reference_date="2026-03-10")  # temporal on by default
    assert "/2026-03-09/" not in plain[0].source
    assert "/2026-03-09/" in dated[0].source


def test_interactions_are_dated_facts(memory):
    memory.record_interaction("a1", "follows", "a2", date="2026-01-01")
    memory.record_interaction("a3", "liked", "a2", date="2026-01-05")
    memory.record_interaction("a2", "commented_on", "a1", date="2026-01-03")
    incoming = memory.get_interactions("a2")
    assert {(f.subject, f.predicate) for f in incoming} == {("a1", "follows"), ("a3", "liked")}
    assert all(isinstance(f, Fact) for f in incoming)
    outgoing = memory.get_interactions("a2", direction="outgoing")
    assert [(f.predicate, f.object) for f in outgoing] == [("commented_on", "a1")]
    early = memory.get_interactions("a2", as_of="2026-01-02")
    assert [f.subject for f in early] == ["a1"]


def test_history_mirror_is_bounded(tmp_path):
    mem = OMPAMemoryAdapter(str(tmp_path / "v"), enable_semantic=False, history_limit=5)
    for i in range(12):
        mem.record_event("E", f"e{i}")
    hist = mem.get_simulation_history(100)
    assert [e["description"] for e in hist] == [f"e{i}" for i in range(7, 12)]


def test_in_memory_backend_has_same_shapes(tmp_path):
    mem = OMPAMemoryAdapter(str(tmp_path / "m"), backend=_InMemoryBackend())
    assert mem.backend_name == "in_memory"
    mem.add_entity("Alice", "lurker")
    assert mem.get_entity("Alice").entity_type == "lurker"
    mem.record_event("E", "Alice waved")
    assert "Alice waved" in mem.search("Alice")[0].content
    mem.record_interaction("b", "follows", "Alice")
    assert [f.subject for f in mem.get_interactions("Alice")] == ["b"]


@pytest.mark.asyncio
async def test_reciprocity_targets_peers_who_engaged(memory):
    memory.record_interaction("fan", "liked", "me", date="2026-01-01")
    agent = DefaultAgent("me", AgentProfile(name="Me", agent_type="default"), memory)
    agent.reciprocity = 1.0
    agent.set_peers(["me", "fan", "p1", "p2", "p3", "p4"])
    targets = Counter(
        agent._enrich_action(Action(action_type=ActionType.FOLLOW)).target for _ in range(50)
    )
    assert targets == Counter({"fan": 50})
    assert agent.memory_targeted_actions == 50


@pytest.mark.asyncio
async def test_simulation_end_to_end_with_real_ompa(memory):
    orch = SwarmOrchestrator(memory)
    sim_id = orch.create_simulation(SimulationConfig(
        name="e2e", max_rounds=12, num_agents=6, checkpoint_interval=0,
        agent_types=["default", "influencer", "lurker"], start_date="2026-02-01",
    ))
    result = await orch.run_simulation(sim_id)
    assert result.status.value == "completed"

    agent_ids = orch.simulations[sim_id]["agents"]
    final = result.final_state  # agents are released after a terminal run
    names = {final[a]["profile"]["name"] for a in agent_ids}
    assert len(names) == len(agent_ids)  # unique default names

    entity = memory.get_entity(agent_ids[0])
    assert entity is not None and entity.entity_type == "default"

    total_incoming = sum(len(memory.get_interactions(a)) for a in agent_ids)
    assert total_incoming > 0
    assert sum(final[a]["memory_targeted_actions"] for a in agent_ids) > 0

    hits = memory.search(f"{agent_ids[1]} follow", limit=5)
    assert hits and any(agent_ids[1].split("_")[0] in h.content for h in hits)

    # Simulated calendar: round 11 is 2026-02-12.
    dates = {e["date"] for e in memory.get_simulation_history(10_000)}
    assert "2026-02-01" in dates and "2026-02-12" in dates


def test_round_date_validation():
    cfg = SimulationConfig(name="d", start_date="2026-12-30", days_per_round=2)
    assert cfg.round_date(0) == "2026-12-30"
    assert cfg.round_date(2) == "2027-01-03"
    assert SimulationConfig(name="n").round_date(5) is None
    orch = SwarmOrchestrator(OMPAMemoryAdapter("unused", backend=_InMemoryBackend()))
    with pytest.raises(ValueError):
        orch.create_simulation(SimulationConfig(name="bad", start_date="2026-13-01"))


def test_future_dated_facts_are_visible(memory):
    # OMPA's kg_query defaults as_of to today; simulations may run in the future.
    for d in ("2030-01-01", "2030-01-02"):
        memory.record_interaction("a", "follows", "b", date=d)
    memory.record_interaction("c", "liked", "b", date="2030-02-01")
    assert {f.subject for f in memory.get_interactions("b")} == {"a", "c"}
    assert {f.subject for f in memory.get_interactions("b", as_of="2030-01-15")} == {"a"}


def test_out_of_range_calendar_rejected():
    orch = SwarmOrchestrator(OMPAMemoryAdapter("unused", backend=_InMemoryBackend()))
    with pytest.raises(ValueError):
        orch.create_simulation(SimulationConfig(
            name="far", start_date="9999-12-01", max_rounds=100, days_per_round=1))


def test_failed_note_write_is_retried(tmp_path):
    class Flaky(_InMemoryBackend):
        fail = True

        def write(self, *a, **k):
            if self.fail:
                raise OSError("disk full")
            return super().write(*a, **k)

    backend = Flaky()
    mem = OMPAMemoryAdapter(str(tmp_path), backend=backend)
    mem.record_event("E", "Kestrel survived")
    assert mem.flush() == 0
    backend.fail = False
    assert mem.flush() == 1
    assert "Kestrel survived" in mem.search("Kestrel")[0].content
