"""
OMPA integration layer for SandFish.

Targets ``ompa>=1.1.1,<2`` (1.1.1 added ``write(date=)`` and temporal ranking):

- **Events** are batched and written as dated vault notes with
  ``Ompa.write(..., date=)`` so ``Ompa.search`` (BM25, or hybrid BM25 +
  embeddings when ``ompa[semantic]`` is installed) actually returns them.
  Temporal ranking is on by default, so queries such as "yesterday" or
  "last week" resolve against the simulation's own calendar.
- **Interactions** between agents (follow / like / comment / repost) are
  stored as dated knowledge-graph facts (``kg_add(valid_from=)``), which
  makes "who has engaged with me?" a cheap indexed SQLite lookup and gives
  point-in-time replay via ``as_of``.
- OMPA 1.1 returns ``Triple`` and ``SearchResult`` dataclasses rather than
  dicts; everything that comes back from the backend goes through the
  normalisers below, so callers always get SandFish types.

If OMPA cannot be imported or initialised, an in-memory backend with the
same result shapes is used so tests and small runs still work.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger("sandfish.memory")


try:  # pragma: no cover - import guard
    from ompa import Ompa  # type: ignore

    HAS_OMPA = True
except Exception:  # pragma: no cover - exercised only without OMPA
    Ompa = None  # type: ignore
    HAS_OMPA = False


# Predicates used for agent-to-agent interaction facts.
INTERACTION_PREDICATES = ("follows", "liked", "commented_on", "reposted")

# Where event notes live inside the vault.
EVENT_NOTE_DIR = "sandfish/events"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# OMPA's kg_query defaults ``as_of`` to *today*, which hides facts dated in
# a simulated future. "All facts" queries pass this instead.
_END_OF_TIME = "9999-12-31"
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n?", re.DOTALL)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return _date.today().isoformat()


def _valid_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, _date):
        return value.isoformat()
    if isinstance(value, str) and _DATE_RE.match(value):
        try:
            _date.fromisoformat(value)
            return value
        except ValueError:
            pass
    raise ValueError(f"date must be 'YYYY-MM-DD', got {value!r}")


DEFAULT_VAULT_PATH = "./sandfish_vault"
_warned_ompa_vault_path = False


def default_vault_path(
    env: Optional[Mapping[str, str]] = None, *, legacy_fallback: bool = False
) -> str:
    """The vault Sandfish uses when no path is given.

    ``$SANDFISH_VAULT_PATH``, else ``./sandfish_vault``.

    ``legacy_fallback`` keeps the old behaviour of the API and MCP server,
    which read ``$OMPA_VAULT_PATH`` when nothing else was set. That is the
    variable OMPA's own MCP server reads to find an agent's real memory, so
    with it set for both programs, simulation notes are written into that
    memory. The fallback still works, so nobody's vault moves, and it warns
    once per process.
    """
    global _warned_ompa_vault_path
    env = os.environ if env is None else env
    own = (env.get("SANDFISH_VAULT_PATH") or "").strip()
    if own:
        return own
    shared = (env.get("OMPA_VAULT_PATH") or "").strip()
    if legacy_fallback and shared:
        if not _warned_ompa_vault_path:
            _warned_ompa_vault_path = True
            logger.warning(
                "Using OMPA_VAULT_PATH (%s) as the Sandfish vault. OMPA's MCP server "
                "reads the same variable, so simulation notes will be written into "
                "that vault. Set SANDFISH_VAULT_PATH to give Sandfish its own.",
                shared,
            )
        return shared
    return DEFAULT_VAULT_PATH


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    uuid: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class SearchResult:
    """A search hit, normalised from the backend."""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    """A knowledge-graph fact, normalised from OMPA's ``Triple``."""
    subject: str
    predicate: str
    object: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source: Optional[str] = None


# ---------------------------------------------------------------------------
# Normalisers (OMPA 1.1 dataclasses, legacy dicts, or the in-memory backend)
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_fact(obj: Any) -> Fact:
    return Fact(
        subject=str(_field(obj, "subject", "")),
        predicate=str(_field(obj, "predicate", "")),
        object=str(_field(obj, "object", "")),
        valid_from=_field(obj, "valid_from"),
        valid_to=_field(obj, "valid_to"),
        source=_field(obj, "source_file") or _field(obj, "source"),
    )


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text or "", count=1).strip()


def _to_search_result(obj: Any) -> SearchResult:
    content = _field(obj, "content_excerpt")
    if content is None:
        content = _field(obj, "content", "")
    path = _field(obj, "path") or _field(obj, "source") or "unknown"
    metadata: Dict[str, Any] = dict(_field(obj, "metadata", None) or {})
    for key in ("match_type", "chunk_index", "citation"):
        value = _field(obj, key)
        if value is not None:
            metadata.setdefault(key, value)
    metadata.setdefault("path", str(path))
    return SearchResult(
        content=_strip_frontmatter(str(content)),
        score=float(_field(obj, "score", 0.0) or 0.0),
        source=str(path),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# In-memory backend (fallback). Mirrors the OMPA 1.1 surface SandFish uses.
# ---------------------------------------------------------------------------


@dataclass
class _MemTriple:
    subject: str
    predicate: str
    object: str
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    source_file: Optional[str] = None


@dataclass
class _MemHit:
    path: str
    content_excerpt: str
    score: float
    match_type: str = "keyword"


class _InMemoryBackend:
    """Process-local stand-in for ``ompa.Ompa`` (same method names/shapes)."""

    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = vault_path
        self._facts: List[_MemTriple] = []
        self._notes: Dict[str, tuple] = {}  # path -> (date, content)
        self._lock = threading.Lock()

    def kg_add(self, subject: str, predicate: str, object: str,
               valid_from: Optional[str] = None, source: Optional[str] = None) -> None:
        with self._lock:
            self._facts.append(_MemTriple(subject, predicate, object, valid_from or _today(), None, source))

    def kg_query(self, entity: str, as_of: Optional[str] = None) -> List[_MemTriple]:
        with self._lock:
            return [
                t for t in self._facts
                if (t.subject == entity or t.object == entity)
                and (as_of is None or (t.valid_from or "") <= as_of)
            ]

    def write(self, content: str, file_path: Optional[str] = None,
              tags: Optional[List[str]] = None, vault: Optional[str] = None,
              date: Optional[str] = None) -> Dict[str, Any]:
        path = file_path or f"notes/{uuid.uuid4().hex}.md"
        with self._lock:
            self._notes[path] = (date or _today(), content)
        return {"vault": "shared", "path": path}

    def search(self, query: str, limit: int = 5, **_: Any) -> List[_MemHit]:
        tokens = {t.lower() for t in re.findall(r"\w+", query)}
        scored = []
        with self._lock:
            for path, (_d, text) in self._notes.items():
                low = text.lower()
                score = float(sum(1 for t in tokens if t in low))
                if score:
                    scored.append(_MemHit(path=path, content_excerpt=text[:400], score=score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]

    def session_start(self) -> Dict[str, Any]:
        return {"success": True, "tokens_hint": 0}

    def stop(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OMPAMemoryAdapter:
    """
    SandFish's memory facade over OMPA (or the in-memory fallback).

    Args:
        vault_path: OMPA vault directory.
        enable_semantic: use embeddings when ``ompa[semantic]`` is installed.
            Default from ``SANDFISH_OMPA_SEMANTIC`` (on). With it off, or the
            extra missing, OMPA uses offline BM25 keyword search — no model
            download, no network, which is what tests and CI use.
        embedding_backend: optional OMPA ``EmbeddingBackend`` (e.g. a hosted
            embedding API); passed straight through.
        temporal_ranking: date-aware ranking for relative-time queries.
        index_refresh_interval: seconds between OMPA index refresh checks.
            0 means every search sees every flushed event.
        event_batch_size: buffered events are written as one note when this
            many accumulate (and always on ``flush()`` / before ``search()``).
        history_limit: cap on the in-process event mirror.
        backend: inject a backend (tests); bypasses OMPA construction.
    """

    def __init__(
        self,
        vault_path: str,
        *,
        enable_semantic: Optional[bool] = None,
        embedding_backend: Any = None,
        temporal_ranking: bool = True,
        index_refresh_interval: float = 0.0,
        event_batch_size: int = 500,
        history_limit: int = 10_000,
        backend: Any = None,
    ):
        self.vault_path = Path(vault_path)
        self.session_active = False
        self.event_batch_size = max(1, int(event_batch_size))
        if enable_semantic is None:
            enable_semantic = _env_flag("SANDFISH_OMPA_SEMANTIC", True)

        if backend is not None:
            self.backend: Any = backend
            self._using_ompa = HAS_OMPA and Ompa is not None and isinstance(backend, Ompa)
        elif HAS_OMPA:
            try:
                self.backend = Ompa(
                    vault_path=str(self.vault_path),
                    agent_name="sandfish",
                    enable_semantic=enable_semantic,
                    embedding_backend=embedding_backend,
                    temporal_ranking=temporal_ranking,
                    index_refresh_interval=index_refresh_interval,
                )
                self._using_ompa = True
            except Exception as exc:
                logger.warning(
                    "OMPA installed but failed to initialize (%s); "
                    "falling back to in-memory backend.",
                    exc,
                )
                self.backend = _InMemoryBackend(self.vault_path)
                self._using_ompa = False
        else:
            self.backend = _InMemoryBackend(self.vault_path)
            self._using_ompa = False

        self._lock = threading.RLock()
        self._pending: List[Dict[str, Any]] = []
        self._note_seq = 0
        self._notes_written = 0
        self._events_total = 0
        # Bounded local mirror for get_simulation_history().
        self._event_mirror: Deque[Dict[str, Any]] = deque(maxlen=max(1, int(history_limit)))

    @property
    def backend_name(self) -> str:
        return "ompa" if self._using_ompa else "in_memory"

    # ----- Session -----

    def start_session(self) -> Dict[str, Any]:
        result = self.backend.session_start()
        self.session_active = True
        return {
            "success": bool(_field(result, "success", True)),
            "context_tokens": int(_field(result, "tokens_hint", 0) or 0),
            "vault_stats": self._get_vault_stats(),
            "backend": self.backend_name,
        }

    def end_session(self) -> None:
        try:
            self.flush()
            self.backend.stop()
        finally:
            self.session_active = False

    # ----- Search -----

    def search(
        self,
        query: str,
        limit: int = 10,
        *,
        temporal: Optional[bool] = None,
        reference_date: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search recorded events and any other vault notes."""
        self.flush()
        kwargs: Dict[str, Any] = {"limit": limit}
        try:
            if self._using_ompa:
                if temporal is not None:
                    kwargs["temporal"] = temporal
                if reference_date is not None:
                    kwargs["reference_date"] = _valid_date(reference_date)
            raw = self.backend.search(query, **kwargs) or []
            return [_to_search_result(r) for r in raw]
        except Exception as exc:
            logger.warning("search() failed: %s", exc)
            return []

    # ----- Knowledge graph -----

    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        *,
        valid_from: Optional[str] = None,
    ) -> Entity:
        attributes = attributes or {}
        when = _valid_date(valid_from)
        try:
            self.backend.kg_add(
                subject=name, predicate="is_a", object=entity_type,
                valid_from=when, source="sandfish/simulation",
            )
            for key, value in attributes.items():
                if value in (None, "", [], {}):
                    continue
                self.backend.kg_add(
                    subject=name, predicate=str(key), object=str(value),
                    valid_from=when, source="sandfish/simulation",
                )
        except Exception as exc:
            logger.warning("add_entity(%s) failed: %s", name, exc)
        return Entity(name=name, entity_type=entity_type, attributes=attributes)

    def get_facts(self, name: str, as_of: Optional[str] = None) -> List[Fact]:
        """Facts where ``name`` is subject or object.

        ``as_of`` limits to facts valid on that date; default is all facts,
        including ones dated in a simulated future.
        """
        try:
            when = _valid_date(as_of) or _END_OF_TIME
            raw = self.backend.kg_query(name, as_of=when) or []
            return [_to_fact(t) for t in raw]
        except Exception as exc:
            logger.warning("kg_query(%s) failed: %s", name, exc)
            return []

    def get_entity(self, name: str) -> Optional[Entity]:
        facts = [f for f in self.get_facts(name) if f.subject == name]
        if not facts:
            return None
        entity_type = "Unknown"
        attributes: Dict[str, Any] = {}
        for fact in facts:
            if fact.predicate == "is_a":
                entity_type = fact.object
            elif fact.predicate not in INTERACTION_PREDICATES:
                attributes[fact.predicate] = fact.object
        return Entity(name=name, entity_type=entity_type, attributes=attributes)

    def get_related_entities(
        self, name: str, relation: Optional[str] = None
    ) -> List[Entity]:
        """Entities linked to ``name`` by an outgoing fact (optionally one predicate)."""
        related: List[Entity] = []
        seen = set()
        for fact in self.get_facts(name):
            if fact.subject != name or fact.predicate == "is_a":
                continue
            if relation and fact.predicate != relation:
                continue
            if fact.object in seen or not self._looks_like_entity(fact.object):
                continue
            seen.add(fact.object)
            entity = self.get_entity(fact.object)
            if entity:
                related.append(entity)
        return related

    # ----- Interactions (dated KG facts) -----

    def record_interaction(
        self,
        actor: str,
        predicate: str,
        target: str,
        *,
        date: Optional[str] = None,
    ) -> None:
        """Store ``actor --predicate--> target`` valid from ``date``."""
        try:
            self.backend.kg_add(
                subject=actor, predicate=predicate, object=target,
                valid_from=_valid_date(date) or _today(), source="sandfish/simulation",
            )
        except Exception as exc:
            logger.warning("record_interaction failed: %s", exc)

    def get_interactions(
        self,
        entity: str,
        *,
        direction: str = "incoming",
        as_of: Optional[str] = None,
        predicates: Iterable[str] = INTERACTION_PREDICATES,
    ) -> List[Fact]:
        """Interaction facts touching ``entity``.

        ``direction``: "incoming" (others → entity), "outgoing", or "both".
        """
        wanted = set(predicates)
        out: List[Fact] = []
        for fact in self.get_facts(entity, as_of=as_of):
            if fact.predicate not in wanted:
                continue
            incoming = fact.object == entity and fact.subject != entity
            outgoing = fact.subject == entity and fact.object != entity
            if direction == "both" and (incoming or outgoing):
                out.append(fact)
            elif direction == "incoming" and incoming:
                out.append(fact)
            elif direction == "outgoing" and outgoing:
                out.append(fact)
        return out

    # ----- Events (dated vault notes) -----

    def record_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        date: Optional[str] = None,
    ) -> None:
        """Record an event. It becomes searchable after the next flush."""
        event = {
            "id": uuid.uuid4().hex,
            "type": event_type,
            "description": description,
            "metadata": metadata or {},
            "timestamp": _utcnow().isoformat(),
            "date": _valid_date(date) or _today(),
            "source": "sandfish/simulation",
        }
        with self._lock:
            self._event_mirror.append(event)
            self._events_total += 1
            self._pending.append(event)
            should_flush = len(self._pending) >= self.event_batch_size
        if should_flush:
            self.flush()

    def flush(self) -> int:
        """Write buffered events to the vault. Returns notes written."""
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return 0
        failed: List[Dict[str, Any]] = []
        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for ev in pending:
            by_date.setdefault(ev["date"], []).append(ev)
        written = 0
        for day, events in by_date.items():
            with self._lock:
                self._note_seq += 1
                seq = self._note_seq
            path = f"{EVENT_NOTE_DIR}/{day}/{seq:06d}-{uuid.uuid4().hex[:8]}.md"
            try:
                self.backend.write(
                    self._render_note(day, events),
                    file_path=path,
                    tags=["sandfish", "event"],
                    date=day,
                )
                written += 1
            except Exception as exc:
                logger.warning("event note write failed (%d events): %s", len(events), exc)
                failed.extend(events)
        with self._lock:
            self._notes_written += written
            if failed:  # keep for the next flush rather than dropping them
                self._pending[:0] = failed
        return written

    @staticmethod
    def _render_note(day: str, events: List[Dict[str, Any]]) -> str:
        sims = sorted({str(e["metadata"].get("simulation_id")) for e in events
                       if isinstance(e.get("metadata"), dict) and e["metadata"].get("simulation_id")})
        lines = [f"# SandFish events {day}"]
        if sims:
            lines.append(f"Simulations: {', '.join(sims)}")
        lines.append("")
        for e in events:
            meta = e.get("metadata") or {}
            extras = []
            for key in ("agent_id", "action_type", "target", "round", "simulation_id"):
                if isinstance(meta, dict) and meta.get(key) not in (None, ""):
                    extras.append(f"{key}={meta[key]}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"- {e['type']}: {e['description']}{suffix}")
        return "\n".join(lines) + "\n"

    def get_simulation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Most recent events recorded via record_event() (bounded mirror)."""
        if limit <= 0:
            return []
        with self._lock:
            items = list(self._event_mirror)
        return items[-limit:]

    # ----- Internal -----

    def _get_vault_stats(self) -> Dict[str, Any]:
        try:
            brain_files = list((self.vault_path / "brain").glob("*.md")) if self.vault_path.exists() else []
            return {
                "brain_notes": len(brain_files),
                "vault_path": str(self.vault_path),
                "events_recorded": self._events_total,
                "event_notes_written": self._notes_written,
            }
        except Exception as exc:
            return {"error": f"Could not read vault stats: {exc}"}

    @staticmethod
    def _looks_like_entity(text: str) -> bool:
        """Heuristic: short identifier-like strings are probably entity names."""
        if not text or len(text) > 80 or text.isnumeric():
            return False
        return text[:1].isupper() or "_agent_" in text


# Factory + utilities ---------------------------------------------------

def create_memory_adapter(vault_path: str, **kwargs: Any) -> OMPAMemoryAdapter:
    """Convenience factory."""
    return OMPAMemoryAdapter(vault_path, **kwargs)


def migrate_from_external(
    records: List[Dict[str, Any]], adapter: OMPAMemoryAdapter
) -> int:
    """
    Import entities from an external source into the OMPA-backed adapter.

    Each record should be a dict with optional `name`, `type`, `attributes` keys.
    Returns the number of entities written.
    """
    count = 0
    for item in records:
        name = item.get("name")
        if not name:
            continue
        adapter.add_entity(
            name=name,
            entity_type=item.get("type", "Entity"),
            attributes=item.get("attributes", {}),
        )
        count += 1
    return count
