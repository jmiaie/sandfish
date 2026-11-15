"""OMPA-backed memory adapter."""

from .ompa_adapter import (
    INTERACTION_PREDICATES,
    Entity,
    Fact,
    OMPAMemoryAdapter,
    SearchResult,
    create_memory_adapter,
    migrate_from_external,
)

__all__ = [
    "INTERACTION_PREDICATES",
    "Entity",
    "Fact",
    "OMPAMemoryAdapter",
    "SearchResult",
    "create_memory_adapter",
    "migrate_from_external",
]
