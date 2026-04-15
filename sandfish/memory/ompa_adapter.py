"""
OMPA integration layer for SandFish.

Provides:
- Semantic search
- Knowledge graph (entities + relations)
- Event log with simulation history retrieval

If the optional `ompa` package is installed, queries are dispatched to it.
Otherwise an in-memory fallback backend is used so the rest of the system
remains usable for development, testing, and small simulations.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sandfish.memory")


# Try to import the real OMPA SDK; fall back to in-memory backend if missing.
try:  # pragma: no cover - exercised only when OMPA is installed
    from ompa import Ompa  # type: ignore

    HAS_OMPA = True
except Exception:
    Ompa = None  # type: ignore
    HAS_OMPA = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    """A semantic-search result."""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class _InMemoryBackend:
    """
    Drop-in OMPA backend that keeps everything in process memory plus an
    optional JSONL event log on disk.

    Suitable for tests and light deployments. Not durable across crashes
    unless `vault_path` is set.
    """

    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = vault_path
        self._facts: List[Tuple[str, str, str, str]] = []  # (subject, predicate, object, source)
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        if self.vault_path is not None:
            self.vault_path.mkdir(parents=True, exist_ok=True)
            self._event_log_path: Optional[Path] = self.vault_path / "events.jsonl"
        else:
            self._event_log_path = None

    # KG methods ---------------------------------------------------------
    def kg_add(self, subject: str, predicate: str, object: str, source: str = "") -> None:
        with self._lock:
            self._facts.append((subject, predicate, object, source))

    def kg_query(self, subject: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"subject": s, "predicate": p, "object": o, "source": src}
                for (s, p, o, src) in self._facts
                if s == subject
            ]

    # Search -------------------------------------------------------------
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Naive token-overlap search over recorded events."""
        tokens = {t.lower() for t in query.split() if t}
        scored: List[Tuple[float, Dict[str, Any]]] = []
        with self._lock:
            for ev in self._events:
                text = (ev.get("description", "") + " " + ev.get("type", "")).lower()
                overlap = sum(1 for t in tokens if t in text)
                if overlap:
                    scored.append((float(overlap), ev))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {
                "content": ev.get("description", ""),
                "score": score,
                "source": ev.get("source", "sandfish"),
                "metadata": ev.get("metadata", {}),
            }
            for score, ev in scored[:limit]
        ]

    # Events -------------------------------------------------------------
    def record_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            if self._event_log_path is not None:
                try:
                    with self._event_log_path.open("a", encoding="utf-8") as fp:
                        fp.write(json.dumps(event, default=str) + "\n")
                except OSError as exc:
                    logger.warning("Failed to persist event log: %s", exc)

    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events[-limit:])

    # Lifecycle ----------------------------------------------------------
    def session_start(self) -> Dict[str, Any]:
        return {"success": True, "tokens_hint": 0}

    def stop(self) -> None:
        return None


class OMPAMemoryAdapter:
    """
    Memory adapter backed by OMPA when available, in-memory otherwise.

    All public methods are safe to call even if OMPA is not installed.
    """

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.session_active = False

        if HAS_OMPA:
            try:
                # Real OMPA SDK; treat it as the backend.
                self.backend: Any = Ompa(vault_path=str(self.vault_path))
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

        # Local event mirror — used regardless of backend so
        # get_simulation_history is always functional.
        self._event_mirror: List[Dict[str, Any]] = []

    # ----- Session -----

    def start_session(self) -> Dict[str, Any]:
        result = self.backend.session_start()
        self.session_active = True
        return {
            "success": getattr(result, "success", result.get("success", True) if isinstance(result, dict) else True),
            "context_tokens": getattr(result, "tokens_hint", result.get("tokens_hint", 0) if isinstance(result, dict) else 0),
            "vault_stats": self._get_vault_stats(),
            "backend": "ompa" if self._using_ompa else "in_memory",
        }

    def end_session(self) -> None:
        try:
            self.backend.stop()
        finally:
            self.session_active = False

    # ----- Search -----

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        if not self.session_active:
            self.start_session()

        try:
            raw = self.backend.search(query, limit=limit) or []
        except Exception as exc:
            logger.warning("search() failed: %s", exc)
            raw = []

        return [
            SearchResult(
                content=r.get("content", ""),
                score=float(r.get("score", 0.0)),
                source=r.get("source", "unknown"),
                metadata=r.get("metadata", {}),
            )
            for r in raw
        ]

    # ----- Knowledge graph -----

    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        attributes = attributes or {}

        try:
            self.backend.kg_add(
                subject=name, predicate="is_a", object=entity_type, source="sandfish/simulation"
            )
            for key, value in attributes.items():
                self.backend.kg_add(
                    subject=name,
                    predicate=key,
                    object=str(value),
                    source="sandfish/simulation",
                )
        except Exception as exc:
            logger.warning("add_entity(%s) failed: %s", name, exc)

        return Entity(name=name, entity_type=entity_type, attributes=attributes)

    def get_entity(self, name: str) -> Optional[Entity]:
        try:
            facts = self.backend.kg_query(name) or []
        except Exception as exc:
            logger.warning("get_entity(%s) failed: %s", name, exc)
            return None

        if not facts:
            return None

        entity_type = "Unknown"
        attributes: Dict[str, Any] = {}
        for fact in facts:
            predicate = fact.get("predicate", "")
            object_val = fact.get("object", "")
            if predicate == "is_a":
                entity_type = object_val
            else:
                attributes[predicate] = object_val

        return Entity(name=name, entity_type=entity_type, attributes=attributes)

    def get_related_entities(
        self, name: str, relation: Optional[str] = None
    ) -> List[Entity]:
        try:
            facts = self.backend.kg_query(name) or []
        except Exception as exc:
            logger.warning("get_related_entities(%s) failed: %s", name, exc)
            return []

        related: List[Entity] = []
        for fact in facts:
            if relation and fact.get("predicate") != relation:
                continue
            obj = fact.get("object", "")
            if self._looks_like_entity(obj):
                entity = self.get_entity(obj)
                if entity:
                    related.append(entity)
        return related

    # ----- Events -----

    def record_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an event so it's queryable via search() and history()."""
        event = {
            "id": uuid.uuid4().hex,
            "type": event_type,
            "description": description,
            "metadata": metadata or {},
            "timestamp": _utcnow().isoformat(),
            "source": "sandfish/simulation",
        }

        # Always mirror locally — this is the source of truth for history().
        self._event_mirror.append(event)

        # Best-effort write to backend so searches can find it.
        try:
            recorder = getattr(self.backend, "record_event", None)
            if callable(recorder):
                recorder(event)
            else:
                # Fall back to writing as KG facts on backends without record_event.
                self.backend.kg_add(
                    subject=event["id"],
                    predicate="is_a",
                    object="Event",
                    source="sandfish/simulation",
                )
                self.backend.kg_add(
                    subject=event["id"],
                    predicate="event_type",
                    object=event_type,
                    source="sandfish/simulation",
                )
                self.backend.kg_add(
                    subject=event["id"],
                    predicate="description",
                    object=description,
                    source="sandfish/simulation",
                )
        except Exception as exc:
            logger.warning("record_event backend write failed: %s", exc)

    def get_simulation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent events recorded via record_event()."""
        if limit <= 0:
            return []
        return list(self._event_mirror[-limit:])

    # ----- Internal -----

    def _get_vault_stats(self) -> Dict[str, Any]:
        try:
            brain_files = list((self.vault_path / "brain").glob("*.md")) if self.vault_path.exists() else []
            return {
                "brain_notes": len(brain_files),
                "vault_path": str(self.vault_path),
                "events_recorded": len(self._event_mirror),
            }
        except Exception as exc:
            return {"error": f"Could not read vault stats: {exc}"}

    @staticmethod
    def _looks_like_entity(text: str) -> bool:
        """Heuristic: short, capitalized strings are probably entity names."""
        if not text or len(text) > 80:
            return False
        return text[:1].isupper() and not text.isnumeric()


# Factory + utilities ---------------------------------------------------

def create_memory_adapter(vault_path: str) -> OMPAMemoryAdapter:
    """Convenience factory."""
    return OMPAMemoryAdapter(vault_path)


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
