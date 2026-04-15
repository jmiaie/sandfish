"""OMPA-backed memory adapter."""

from .ompa_adapter import (
    Entity,
    OMPAMemoryAdapter,
    SearchResult,
    create_memory_adapter,
    migrate_from_external,
)

__all__ = [
    "Entity",
    "OMPAMemoryAdapter",
    "SearchResult",
    "create_memory_adapter",
    "migrate_from_external",
]
