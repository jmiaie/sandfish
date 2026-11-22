"""OMPA-backed memory adapter."""

from .ompa_adapter import (
    DEFAULT_VAULT_PATH,
    INTERACTION_PREDICATES,
    Entity,
    Fact,
    OMPAMemoryAdapter,
    SearchResult,
    create_memory_adapter,
    default_vault_path,
    migrate_from_external,
)

__all__ = [
    "DEFAULT_VAULT_PATH",
    "INTERACTION_PREDICATES",
    "Entity",
    "Fact",
    "OMPAMemoryAdapter",
    "SearchResult",
    "create_memory_adapter",
    "default_vault_path",
    "migrate_from_external",
]
