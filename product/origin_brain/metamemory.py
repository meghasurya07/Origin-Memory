import logging
from enum import Enum
from typing import Any
from pydantic import BaseModel

from .models import EpisodicMemory, SemanticMemory, MemoryResult

logger = logging.getLogger(__name__)

class ConfidenceLevel(str, Enum):
    CERTAIN = 'certain'
    CONFIDENT = 'confident'
    UNCERTAIN = 'uncertain'
    GUESSING = 'guessing'
    NO_KNOWLEDGE = 'no_knowledge'

class MetamemoryAssessment(BaseModel):
    query: str
    confidence_level: ConfidenceLevel
    confidence_score: float
    reasoning: str
    has_relevant_memories: bool
    memory_count: int
    avg_retrievability: float
    avg_relevance: float
    source_diversity: int
    recommendation: str

class MetamemoryEngine:
    """
    Metamemory engine for assessing AI agent's own memory and confidence.
    """
    def __init__(self):
        pass
        
    def assess(self, query: str, results: list[MemoryResult]) -> MetamemoryAssessment:
        """Evaluate confidence in answering a query based on memory results."""
        memory_count = len(results)
        has_relevant_memories = memory_count > 0
        
        if not has_relevant_memories:
            return MetamemoryAssessment(
                query=query,
                confidence_level=ConfidenceLevel.NO_KNOWLEDGE,
                confidence_score=0.0,
                reasoning="No relevant memories retrieved.",
                has_relevant_memories=False,
                memory_count=0,
                avg_retrievability=0.0,
                avg_relevance=0.0,
                source_diversity=0,
                recommendation="Suggest looking up externally."
            )
            
        avg_relevance = sum(r.relevance_score for r in results) / memory_count
        
        retrievabilities = []
        source_types = set()
        
        for r in results:
            if hasattr(r, 'memory'):
                mem = r.memory
                retrievabilities.append(getattr(mem, 'retrievability', 1.0))
                source_types.add(type(mem).__name__)
            else:
                retrievabilities.append(1.0)
                source_types.add("Unknown")
                
        avg_retrievability = sum(retrievabilities) / len(retrievabilities)
        source_diversity = len(source_types)
        
        # Base score on relevance and retrievability
        base_score = (avg_relevance * 0.6) + (avg_retrievability * 0.4)
        
        # Boost for diversity
        if source_diversity > 1:
            base_score = min(1.0, base_score + 0.1)
            
        # Penalize if very few memories
        if memory_count == 1:
            base_score = base_score * 0.8
            
        confidence_score = max(0.0, min(1.0, base_score))
        
        if confidence_score > 0.9:
            confidence_level = ConfidenceLevel.CERTAIN
            recommendation = 'Answer with high confidence'
        elif confidence_score > 0.7:
            confidence_level = ConfidenceLevel.CONFIDENT
            recommendation = 'Answer with confidence'
        elif confidence_score > 0.4:
            confidence_level = ConfidenceLevel.UNCERTAIN
            recommendation = 'Mention uncertainty to user'
        elif confidence_score > 0.2:
            confidence_level = ConfidenceLevel.GUESSING
            recommendation = 'Highlight that this is a guess based on weak signals'
        else:
            confidence_level = ConfidenceLevel.NO_KNOWLEDGE
            recommendation = 'Suggest looking up externally'
            
        reasoning = f"Found {memory_count} memories with average relevance {avg_relevance:.2f} and diversity {source_diversity}."
        
        return MetamemoryAssessment(
            query=query,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            reasoning=reasoning,
            has_relevant_memories=has_relevant_memories,
            memory_count=memory_count,
            avg_retrievability=avg_retrievability,
            avg_relevance=avg_relevance,
            source_diversity=source_diversity,
            recommendation=recommendation
        )
        
    def feeling_of_knowing(self, query: str, results: list[MemoryResult]) -> float:
        """Estimate probability that answer is known even if not fully retrieved."""
        if not results:
            return 0.1 # Base rate
            
        avg_score = sum(r.relevance_score for r in results) / len(results)
        count_boost = min(1.0, len(results) / 10.0)
        fok_score = (avg_score * 0.5) + (count_boost * 0.5)
        
        return max(0.0, min(1.0, fok_score))
        
    def tip_of_tongue(self, query: str, partial_results: list[MemoryResult]) -> dict[str, Any]:
        """Simulate TOT state by returning partial hints."""
        hints = {
            'partial_content': [],
            'related_memories': [],
            'temporal_context': []
        }
        
        for r in partial_results[:5]:
            if hasattr(r, 'memory'):
                mem = r.memory
                if hasattr(mem, 'content'):
                    hints['partial_content'].append(str(mem.content)[:50] + "...")
                hints['related_memories'].append(str(type(mem).__name__))
                if hasattr(mem, 'created_at'):
                    hints['temporal_context'].append(str(mem.created_at))
                    
        return hints
        
    def calibration_score(self, predictions: list[tuple[float, bool]]) -> float:
        """Compute calibration using Brier score (1 is best)."""
        if not predictions:
            return 0.5
            
        squared_errors = []
        for predicted_conf, was_correct in predictions:
            actual = 1.0 if was_correct else 0.0
            squared_errors.append((predicted_conf - actual) ** 2)
            
        mean_squared_error = sum(squared_errors) / len(squared_errors)
        brier_score = 1.0 - mean_squared_error
        
        return max(0.0, min(1.0, brier_score))
