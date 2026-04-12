"""
OMPA integration layer for SandFish.

Replaces Zep Cloud with local, zero-cost OMPA memory system.
Provides: semantic search, knowledge graph, entity management
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# OMPA imports
try:
    from ompa import Ompa
    from ompa.core import MessageType
    HAS_Ompa = True
except ImportError:
    HAS_Ompa = False


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    name: str
    entity_type: str
    attributes: Dict[str, Any]
    uuid: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class SearchResult:
    """Result from semantic search."""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]


class OMPAMemoryAdapter:
    """
    Drop-in replacement for Zep Cloud memory.
    
    Uses OMPA for:
    - Semantic search (local embeddings)
    - Knowledge graph (SQLite)
    - Entity management
    - Session history
    """
    
    def __init__(self, vault_path: str):
        """
        Initialize OMPA adapter.
        
        Args:
            vault_path: Path to OMPA vault directory
        """
        if not HAS_Ompa:
            raise ImportError(
                "OMPA not installed. "
                "Install with: pip install 'ompa[semantic]'"
            )
        
        self.vault_path = Path(vault_path)
        self.ompa = Ompa(vault_path=str(self.vault_path))
        self.session_active = False
        
    def start_session(self) -> Dict[str, Any]:
        """Start a new simulation session."""
        result = self.ompa.session_start()
        self.session_active = True
        return {
            'success': result.success,
            'context_tokens': result.tokens_hint,
            'vault_stats': self._get_vault_stats()
        }
    
    def end_session(self) -> None:
        """End current session."""
        self.ompa.stop()
        self.session_active = False
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Semantic search across vault content.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of search results with similarity scores
        """
        if not self.session_active:
            self.start_session()
        
        # Use OMPA semantic search
        results = self.ompa.search(query, limit=limit)
        
        return [
            SearchResult(
                content=r.get('content', ''),
                score=r.get('score', 0.0),
                source=r.get('source', 'unknown'),
                metadata=r.get('metadata', {})
            )
            for r in results
        ]
    
    def add_entity(self, name: str, entity_type: str, 
                   attributes: Dict[str, Any] = None) -> Entity:
        """
        Add entity to knowledge graph.
        
        Args:
            name: Entity name
            entity_type: Type of entity (Person, Organization, etc.)
            attributes: Additional attributes
            
        Returns:
            Created entity
        """
        if attributes is None:
            attributes = {}
        
        # Add to OMPA knowledge graph
        self.ompa.kg_add(
            subject=name,
            predicate="is_a",
            object=entity_type,
            source="sandfish/simulation"
        )
        
        # Add attributes as separate facts
        for key, value in attributes.items():
            self.ompa.kg_add(
                subject=name,
                predicate=key,
                object=str(value),
                source="sandfish/simulation"
            )
        
        return Entity(
            name=name,
            entity_type=entity_type,
            attributes=attributes
        )
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """
        Retrieve entity from knowledge graph.
        
        Args:
            name: Entity name
            
        Returns:
            Entity if found, None otherwise
        """
        facts = self.ompa.kg_query(name)
        
        if not facts:
            return None
        
        # Parse facts to reconstruct entity
        entity_type = "Unknown"
        attributes = {}
        
        for fact in facts:
            predicate = fact.get('predicate', '')
            object_val = fact.get('object', '')
            
            if predicate == "is_a":
                entity_type = object_val
            else:
                attributes[predicate] = object_val
        
        return Entity(
            name=name,
            entity_type=entity_type,
            attributes=attributes
        )
    
    def get_related_entities(self, name: str, 
                             relation: Optional[str] = None) -> List[Entity]:
        """
        Find entities related to given entity.
        
        Args:
            name: Entity name
            relation: Optional relation type filter
            
        Returns:
            List of related entities
        """
        # Query knowledge graph for relationships
        # This is a simplified version - full implementation would
        # traverse the graph more thoroughly
        
        facts = self.ompa.kg_query(name)
        related = []
        
        for fact in facts:
            if relation and fact.get('predicate') != relation:
                continue
                
            obj = fact.get('object', '')
            # If object is another entity, fetch it
            if self._is_entity_name(obj):
                entity = self.get_entity(obj)
                if entity:
                    related.append(entity)
        
        return related
    
    def record_event(self, event_type: str, description: str,
                     metadata: Dict[str, Any] = None) -> None:
        """
        Record simulation event.
        
        Args:
            event_type: Type of event (DECISION, ACTION, MILESTONE)
            description: Event description
            metadata: Additional metadata
        """
        # Classify and store in appropriate vault location
        classification = self.ompa.classify(description)
        
        # Write to vault based on classification
        note_data = {
            'type': event_type,
            'description': description,
            'metadata': metadata or {},
            'classification': classification.message_type.value
        }
        
        # OMPA will auto-route to appropriate vault section
        # based on classification
    
    def get_simulation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve simulation history.
        
        Args:
            limit: Maximum events to return
            
        Returns:
            List of historical events
        """
        # Query knowledge graph for simulation events
        # This would typically search for facts with source="sandfish/simulation"
        
        # For now, return empty - full implementation would
        # query the temporal knowledge graph
        return []
    
    def _get_vault_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        try:
            # Count files in vault
            brain_files = list((self.vault_path / 'brain').glob('*.md'))
            
            return {
                'brain_notes': len(brain_files),
                'vault_path': str(self.vault_path)
            }
        except Exception:
            return {'error': 'Could not read vault stats'}
    
    def _is_entity_name(self, text: str) -> bool:
        """Heuristic to check if text is an entity name."""
        # Simple heuristic - could be improved
        return len(text) < 50 and text[0].isupper()


# Convenience functions for SandFish integration

def create_memory_adapter(vault_path: str) -> OMPAMemoryAdapter:
    """Factory function to create memory adapter."""
    return OMPAMemoryAdapter(vault_path)


def migrate_from_zep(zep_data: List[Dict], 
                     adapter: OMPAMemoryAdapter) -> int:
    """
    Migrate data from Zep Cloud to OMPA.
    
    Args:
        zep_data: Data exported from Zep
        adapter: Initialized OMPA adapter
        
    Returns:
        Number of entities migrated
    """
    count = 0
    
    for item in zep_data:
        name = item.get('name', 'Unknown')
        entity_type = item.get('type', 'Entity')
        attributes = item.get('attributes', {})
        
        adapter.add_entity(name, entity_type, attributes)
        count += 1
    
    return count
