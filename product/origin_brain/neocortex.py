"""
Neocortical Learning Engine — Slow Statistical Knowledge Extraction

Implements the "neocortex" side of Complementary Learning Systems (CLS):
- Hippocampus learns FAST (one-shot episodic binding) → already built
- Neocortex learns SLOW (statistical regularities, schemas) → THIS MODULE

McClelland, McNaughton & O'Reilly (1995):
"Why There Are Complementary Learning Systems in the Hippocampus and Neocortex"

How it works:
1. During sleep consolidation, episodic memories are "replayed" to the neocortex
2. The neocortex extracts STATISTICAL PATTERNS across episodes
3. These patterns become SEMANTIC KNOWLEDGE (facts, schemas, concepts)
4. Eventually, the neocortex can answer queries WITHOUT the hippocampus

Brain analogue:
- Hippocampal replay during SWS → neocortical slow learning
- Schema formation in vmPFC
- Gradual extraction of gist from episodes

Example:
    After many episodes about "meetings with Bob at cafe discussing budgets",
    the neocortex should learn:
    - Bob is associated with meetings
    - Meetings often discuss budgets
    - Cafe is a common meeting location
    Without storing any specific episode.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NeocortexConfig(BaseModel):
    """Configuration for neocortical learning engine."""
    # Learning rate (how fast patterns are extracted)
    learning_rate: float = 0.05  # Slow — that's the point
    # Minimum co-occurrence count to form an association
    min_cooccurrence: int = 3
    # How much to decay association strengths over time
    decay_rate: float = 0.01
    # Maximum number of semantic concepts to track
    max_concepts: int = 10_000
    # Schema formation threshold (how many episodes before a schema forms)
    schema_threshold: int = 5
    # Interleaved training — mix old patterns with new (CLS key insight)
    interleave_ratio: float = 0.3  # 30% old, 70% new during replay


@dataclass
class SemanticConcept:
    """
    A learned concept in the neocortex.
    
    Unlike episodic memories (specific events), concepts are
    statistical abstractions extracted across many episodes.
    """
    name: str
    # Associations: concept_name → strength
    associations: Dict[str, float] = field(default_factory=dict)
    # How many episodes contributed to this concept
    episode_count: int = 0
    # Activation level (for spreading activation)
    activation: float = 0.0
    # Category/type tags
    categories: Set[str] = field(default_factory=set)
    # Prototype features (averaged across episodes)
    prototype: Dict[str, float] = field(default_factory=dict)


@dataclass
class Schema:
    """
    A knowledge schema — structured pattern extracted from episodes.
    
    Schemas accelerate consolidation (Tse et al. 2007): when new info
    fits an existing schema, hippocampal→neocortical transfer is FASTER.
    """
    name: str
    # Slots: slot_name → expected values/types
    slots: Dict[str, List[str]] = field(default_factory=dict)
    # How many episodes have been absorbed into this schema
    instance_count: int = 0
    # Confidence in this schema (0-1)
    confidence: float = 0.0
    # Default values for slots (most common)
    defaults: Dict[str, str] = field(default_factory=dict)


class NeocorticalEngine:
    """
    The slow-learning neocortex for statistical knowledge extraction.
    
    Usage:
        neocortex = NeocorticalEngine()
        
        # During sleep, replay episodic memories
        neocortex.learn_from_episode({
            "entities": ["Bob", "budget", "cafe"],
            "type": "meeting",
            "emotion": "neutral"
        })
        
        # After many replays, query semantic knowledge
        related = neocortex.get_associations("Bob")
        # → [("meeting", 0.8), ("budget", 0.7), ("cafe", 0.6)]
        
        # Check if a schema exists
        schema = neocortex.get_schema("meeting")
        # → Schema with slots: who, where, topic
    """
    
    def __init__(self, config: Optional[NeocortexConfig] = None):
        self.config = config or NeocortexConfig()
        
        # Semantic concept store
        self._concepts: Dict[str, SemanticConcept] = {}
        
        # Co-occurrence matrix (sparse): (concept_a, concept_b) → count
        self._cooccurrence: Counter = Counter()
        
        # Schemas
        self._schemas: Dict[str, Schema] = {}
        
        # Episode patterns for schema detection
        self._episode_patterns: List[Dict[str, Any]] = []
        
        # Statistics
        self._total_episodes_learned: int = 0
        self._total_queries: int = 0
        
        logger.info("NeocorticalEngine initialized (slow learning mode)")
    
    def _get_or_create_concept(self, name: str) -> SemanticConcept:
        """Get existing concept or create new one."""
        if name not in self._concepts:
            if len(self._concepts) >= self.config.max_concepts:
                # Evict weakest concept
                weakest = min(
                    self._concepts.values(),
                    key=lambda c: c.episode_count
                )
                del self._concepts[weakest.name]
            self._concepts[name] = SemanticConcept(name=name)
        return self._concepts[name]
    
    def learn_from_episode(
        self,
        episode: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Learn statistical patterns from a replayed episodic memory.
        
        This is called during sleep consolidation. The neocortex
        extracts entities, co-occurrences, and patterns.
        
        Args:
            episode: Dict with keys like "entities", "type", "emotion",
                     "content", etc.
        
        Returns:
            Dict with learning statistics
        """
        entities = episode.get("entities", [])
        episode_type = episode.get("type", "unknown")
        
        # 1. Update concepts for each entity
        for entity in entities:
            concept = self._get_or_create_concept(entity.lower())
            concept.episode_count += 1
            if episode_type:
                concept.categories.add(episode_type)
        
        # 2. Update co-occurrence matrix (all pairs)
        entities_lower = [e.lower() for e in entities]
        for i in range(len(entities_lower)):
            for j in range(i + 1, len(entities_lower)):
                pair = tuple(sorted([entities_lower[i], entities_lower[j]]))
                self._cooccurrence[pair] += 1
                
                # If co-occurrence exceeds threshold, form association
                if self._cooccurrence[pair] >= self.config.min_cooccurrence:
                    a, b = pair
                    strength = math.log1p(self._cooccurrence[pair]) * self.config.learning_rate
                    concept_a = self._get_or_create_concept(a)
                    concept_b = self._get_or_create_concept(b)
                    
                    # Slow Hebbian update
                    old_a = concept_a.associations.get(b, 0.0)
                    concept_a.associations[b] = old_a + (strength - old_a) * self.config.learning_rate
                    
                    old_b = concept_b.associations.get(a, 0.0)
                    concept_b.associations[a] = old_b + (strength - old_b) * self.config.learning_rate
        
        # 3. Track episode pattern for schema detection
        pattern = {
            "type": episode_type,
            "entity_count": len(entities),
            "entities": set(entities_lower),
        }
        self._episode_patterns.append(pattern)
        
        # 4. Try schema formation
        self._try_form_schema(episode_type, entities_lower)
        
        self._total_episodes_learned += 1
        
        return {
            "entities_updated": len(entities),
            "new_associations": sum(
                1 for pair, count in self._cooccurrence.items()
                if count == self.config.min_cooccurrence
            ),
            "total_concepts": len(self._concepts),
            "total_schemas": len(self._schemas),
        }
    
    def _try_form_schema(self, episode_type: str, entities: List[str]):
        """Try to form or update a schema from accumulated patterns."""
        if not episode_type or episode_type == "unknown":
            return
        
        # Count episodes of this type
        type_episodes = [
            p for p in self._episode_patterns
            if p["type"] == episode_type
        ]
        
        if len(type_episodes) < self.config.schema_threshold:
            return
        
        # Extract common entities across episodes of this type
        if episode_type not in self._schemas:
            self._schemas[episode_type] = Schema(
                name=episode_type,
                slots={},
                instance_count=0,
            )
        
        schema = self._schemas[episode_type]
        schema.instance_count = len(type_episodes)
        
        # Find entities that appear in >50% of episodes
        entity_counts: Counter = Counter()
        for ep in type_episodes:
            for e in ep["entities"]:
                entity_counts[e] += 1
        
        common_entities = {
            e: count / len(type_episodes)
            for e, count in entity_counts.items()
            if count / len(type_episodes) > 0.5
        }
        
        schema.slots["common_entities"] = list(common_entities.keys())
        schema.confidence = min(1.0, len(type_episodes) / (self.config.schema_threshold * 2))
    
    def get_associations(
        self,
        concept_name: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get associated concepts (spreading activation).
        
        Returns:
            List of (concept_name, strength) sorted by strength
        """
        self._total_queries += 1
        concept = self._concepts.get(concept_name.lower())
        if not concept:
            return []
        
        associations = sorted(
            concept.associations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return associations[:top_k]
    
    def get_schema(self, schema_type: str) -> Optional[Schema]:
        """Get a schema by type name."""
        return self._schemas.get(schema_type)
    
    def schema_consistent(self, episode: Dict[str, Any]) -> float:
        """
        Check how consistent an episode is with existing schemas.
        
        Returns 0.0-1.0 (higher = more consistent).
        Schema-consistent info consolidates FASTER (Tse et al. 2007).
        """
        episode_type = episode.get("type", "unknown")
        schema = self._schemas.get(episode_type)
        
        if not schema or not schema.slots.get("common_entities"):
            return 0.0
        
        entities = set(e.lower() for e in episode.get("entities", []))
        common = set(schema.slots.get("common_entities", []))
        
        if not common:
            return 0.0
        
        overlap = len(entities & common) / len(common)
        return min(1.0, overlap * schema.confidence)
    
    def decay(self):
        """Apply decay to all association strengths."""
        for concept in self._concepts.values():
            for assoc_name in list(concept.associations.keys()):
                concept.associations[assoc_name] *= (1.0 - self.config.decay_rate)
                if concept.associations[assoc_name] < 0.001:
                    del concept.associations[assoc_name]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get neocortex statistics."""
        return {
            "total_concepts": len(self._concepts),
            "total_schemas": len(self._schemas),
            "total_episodes_learned": self._total_episodes_learned,
            "total_associations": sum(
                len(c.associations) for c in self._concepts.values()
            ),
            "total_queries": self._total_queries,
            "schema_names": list(self._schemas.keys()),
        }
