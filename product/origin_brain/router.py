import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

from .models import (
    BrainConfig,
    MemoryType,
    MemoryTier,
    MemoryQuery,
    MemoryResult,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory
)

logger = logging.getLogger(__name__)

class MemoryRouter:
    """
    Memory Router — analogous to the Thalamus in the brain.
    It determines which memory system(s) to query for a given input and routes queries intelligently.
    """

    def __init__(self, config: BrainConfig):
        self.config = config
        logger.info("MemoryRouter initialized")

    def classify_query(self, query: str) -> List[MemoryType]:
        """
        Analyzes the query to determine which memory systems to search.
        """
        q_lower = query.lower()
        
        temporal_words = {'when', 'last time', 'recently', 'yesterday', 'ago', 'before', 'after', 'first time', 'history'}
        knowledge_words = {'what is', 'who is', 'define', 'explain', 'means', 'facts', 'know', 'preference', 'likes', 'dislikes', 'favorite'}
        procedural_words = {'how to', 'steps', 'process', 'procedure', 'workflow', 'do i', 'should i', 'instructions'}
        
        types = []
        
        if any(word in q_lower for word in temporal_words):
            types.append(MemoryType.EPISODIC)
            
        if any(word in q_lower for word in knowledge_words):
            types.append(MemoryType.SEMANTIC)
            
        if any(word in q_lower for word in procedural_words):
            types.append(MemoryType.PROCEDURAL)
            
        if not types:
            types = [MemoryType.EPISODIC, MemoryType.SEMANTIC]
            
        logger.debug(f"Classified query '{query}' -> {types}")
        return types

    def build_query(self, raw_query: str, context: Optional[dict] = None, top_k: int = 5) -> MemoryQuery:
        """
        Builds a structured MemoryQuery from a raw text query.
        """
        memory_types = self.classify_query(raw_query)
        temporal_range = self._extract_temporal_range(raw_query)
        
        query_obj = MemoryQuery(
            query=raw_query,
            memory_types=memory_types,
            temporal_range=temporal_range,
            top_k=top_k,
            context_filter=context
        )
        logger.debug(f"Built MemoryQuery: {query_obj}")
        return query_obj

    def rank_results(self, results: List[MemoryResult], query: str) -> List[MemoryResult]:
        """
        Re-ranks memory results using multi-signal scoring.
        """
        if not results:
            return []
            
        tier_weights = {
            MemoryTier.WORKING: 1.0,
            MemoryTier.EPISODIC: 0.8,
            MemoryTier.SEMANTIC: 0.6,
            MemoryTier.PROCEDURAL: 0.4,
            MemoryTier.ARCHIVAL: 0.2
        }
        
        diversity_scores = self._compute_diversity_penalty(results)
        
        ranked_results = []
        now = datetime.now(timezone.utc)
        
        for i, res in enumerate(results):
            relevance = res.relevance_score
            tier_prio = tier_weights.get(res.tier, 0.5)
            
            # Extract timestamp for recency bonus
            mem_time = None
            if isinstance(res.memory, EpisodicMemory):
                mem_time = res.memory.timestamp
            elif isinstance(res.memory, SemanticMemory):
                mem_time = res.memory.updated_at
            elif isinstance(res.memory, ProceduralMemory) and res.memory.last_executed:
                mem_time = res.memory.last_executed
                
            recency_bonus = 0.0
            if mem_time:
                # Ensure timezone aware
                if mem_time.tzinfo is None:
                    mem_time = mem_time.replace(tzinfo=timezone.utc)
                delta = now - mem_time
                days = max(0.0, delta.total_seconds() / 86400.0)
                # Simple decay over a year
                recency_bonus = max(0.0, 1.0 - (days / 365.0))
                
            diversity_bonus = diversity_scores[i]
            
            final_score = (
                0.5 * relevance +
                0.2 * tier_prio +
                0.15 * recency_bonus +
                0.15 * diversity_bonus
            )
            
            ranked_results.append((final_score, res))
            
        # Sort by final score descending
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in ranked_results]

    def _extract_temporal_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Parses temporal expressions from natural language.
        """
        q_lower = query.lower()
        now = datetime.now(timezone.utc)
        
        if 'today' in q_lower:
            start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return (start_of_today, now)
            
        if 'yesterday' in q_lower:
            start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_of_yesterday = start_of_today - timedelta(days=1)
            end_of_yesterday = start_of_today - timedelta(microseconds=1)
            return (start_of_yesterday, end_of_yesterday)
            
        if 'last week' in q_lower:
            seven_days_ago = now - timedelta(days=7)
            return (seven_days_ago, now)
            
        if 'last month' in q_lower:
            thirty_days_ago = now - timedelta(days=30)
            return (thirty_days_ago, now)
            
        if 'this year' in q_lower:
            start_of_year = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            return (start_of_year, now)
            
        return None

    def _compute_diversity_penalty(self, results: List[MemoryResult]) -> List[float]:
        """
        MMR-style (Maximal Marginal Relevance) diversity scoring.
        """
        def extract_text(mem) -> set:
            if isinstance(mem, EpisodicMemory):
                return set(mem.content.lower().split())
            elif isinstance(mem, SemanticMemory):
                return set((mem.key + " " + mem.value).lower().split())
            elif isinstance(mem, ProceduralMemory):
                return set((mem.name + " " + mem.description).lower().split())
            return set()
            
        def jaccard(set1: set, set2: set) -> float:
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0

        scores = []
        seen_texts = []
        
        for res in results:
            text_set = extract_text(res.memory)
            
            if not seen_texts:
                scores.append(1.0)
            else:
                sims = [jaccard(text_set, st) for st in seen_texts]
                max_sim = max(sims) if sims else 0.0
                scores.append(1.0 - max_sim)
                
            seen_texts.append(text_set)
            
        return scores
