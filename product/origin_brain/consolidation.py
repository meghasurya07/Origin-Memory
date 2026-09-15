"""
Origin Brain — Consolidation Daemon.

Analogous to sleep-based memory replay in the brain. Transforms
episodic memories into semantic knowledge through pattern extraction
and repetition-based consolidation.
"""
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
from uuid import uuid4

from .models import (
    BrainConfig,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    ConsolidationResult,
)
from .hippocampus import HippocampalEngine

logger = logging.getLogger(__name__)


class ConsolidationDaemon:
    """
    The Consolidation Daemon — analogous to sleep-based memory replay.

    Periodically processes episodic memories, extracts semantic patterns,
    and manages the semantic and procedural memory stores.
    """

    def __init__(self, hippocampus: HippocampalEngine, config: BrainConfig):
        self.hippocampus = hippocampus
        self.config = config
        self._semantic_store: Dict[str, SemanticMemory] = {}
        self._procedural_store: Dict[str, ProceduralMemory] = {}
        self._consolidation_history: List[ConsolidationResult] = []
        self._running: bool = False

    def run_cycle(self) -> ConsolidationResult:
        """
        Run one consolidation cycle:
        1. Select episodic memories ready for consolidation
        2. Extract semantic patterns
        3. Apply decay to all episodic memories
        4. Evict dead memories
        """
        start_time = time.time()
        logger.info("Starting consolidation cycle...")

        # Phase 1: Select episodes ready for consolidation
        episodes = self.hippocampus.select_for_consolidation(
            min_age_hours=self.config.consolidation_min_age_hours,
            min_access_count=1,
            min_salience=self.config.salience_threshold,
        )

        # Phase 2: Extract semantic patterns
        new_semantics = self._extract_patterns(episodes)
        for sem in new_semantics:
            self.store_semantic(sem)

        # Phase 3: Apply decay to all episodic memories
        all_memories = list(self.hippocampus._episodic_store.values())
        self.hippocampus.decay_engine.apply_decay(all_memories)

        # Phase 4: Evict dead memories
        evicted = self.hippocampus.evict(self.config.survival_threshold)

        duration = time.time() - start_time

        result = ConsolidationResult(
            consolidated_count=len(episodes),
            evicted_count=len(evicted),
            archived_count=0,
            new_semantic_count=len(new_semantics),
            duration_seconds=duration,
        )
        self._consolidation_history.append(result)
        logger.info(
            f"Consolidation complete in {duration:.2f}s: "
            f"{len(episodes)} consolidated, {len(new_semantics)} semantic, "
            f"{len(evicted)} evicted"
        )
        return result

    def _extract_patterns(
        self, episodes: List[EpisodicMemory]
    ) -> List[SemanticMemory]:
        """
        Extract semantic facts from episodic memories.

        Groups episodes by context topic, then extracts salient keywords
        as semantic facts. In a future version, this will use LLM-based
        extraction for much richer knowledge distillation.
        """
        groups: Dict[str, List[EpisodicMemory]] = {}
        for ep in episodes:
            topic = ep.context.get("topic", "general") if ep.context else "general"
            groups.setdefault(topic, []).append(ep)

        new_semantics: List[SemanticMemory] = []

        for topic, group_eps in groups.items():
            content = " ".join(ep.content for ep in group_eps)
            # Simple keyword extraction (placeholder for LLM-based extraction)
            words = content.split()
            keywords = {w.lower() for w in words if len(w) > 4}

            source_ids = [ep.id for ep in group_eps]

            for kw in keywords:
                # Check if this knowledge already exists
                existing = None
                for sm in self._semantic_store.values():
                    if sm.key == kw:
                        existing = sm
                        break

                if existing:
                    # Update confidence (reconsolidation)
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                    existing.source_episodes = list(
                        set(existing.source_episodes + source_ids)
                    )
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    new_mem = SemanticMemory(
                        key=kw,
                        value=f"Extracted fact about '{kw}' from {topic}",
                        confidence=0.5,
                        category=topic,
                        source_episodes=source_ids,
                    )
                    new_semantics.append(new_mem)

        return new_semantics

    def store_semantic(self, memory: SemanticMemory) -> None:
        """Add a semantic memory to the store."""
        self._semantic_store[memory.id] = memory

    def retrieve_semantic(self, query: str, top_k: int = 5) -> List[SemanticMemory]:
        """Retrieve semantic memories matching the query via keyword matching."""
        query_words = set(query.lower().split())
        scored: List[tuple] = []

        for mem in self._semantic_store.values():
            score = 0.0
            if mem.key.lower() in query_words:
                score += 1.0
            for word in query_words:
                if word in mem.value.lower():
                    score += 0.5
            if score > 0:
                scored.append((score * mem.confidence, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def store_procedural(self, memory: ProceduralMemory) -> None:
        """Add a procedural memory to the store."""
        self._procedural_store[memory.id] = memory

    def retrieve_procedural(self, trigger: str) -> List[ProceduralMemory]:
        """Retrieve procedural memories matching the trigger conditions."""
        matches = []
        trigger_lower = trigger.lower()
        for mem in self._procedural_store.values():
            if any(t.lower() in trigger_lower for t in mem.trigger_conditions):
                matches.append(mem)
        return matches

    def get_statistics(self) -> dict:
        """Get consolidation statistics."""
        return {
            "semantic_count": len(self._semantic_store),
            "procedural_count": len(self._procedural_store),
            "total_consolidation_cycles": len(self._consolidation_history),
            "last_consolidation": (
                self._consolidation_history[-1].timestamp.isoformat()
                if self._consolidation_history
                else None
            ),
        }
