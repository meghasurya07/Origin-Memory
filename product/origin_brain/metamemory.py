import logging
import math
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

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
    overall_confidence: float = 0.0  # v0.9: alias for compatibility
    reasoning: str
    has_relevant_memories: bool
    memory_count: int
    avg_retrievability: float
    avg_relevance: float
    source_diversity: int
    recommendation: str
    # v0.9: Belief Entropy fields
    belief_entropy: float = 0.0  # Low = consistent memories, High = contradictory
    contradiction_detected: bool = False
    agreement_score: float = 1.0  # How much the top results agree with each other

class MetamemoryEngine:
    """
    Metamemory engine for assessing AI agent's own memory and confidence.
    
    v0.9 Upgrade: Belief Entropy (inspired by MMPO, Meta-Cognitive Memory
    Policy Optimization, 2026).
    
    Belief Entropy measures the epistemic uncertainty across candidate
    memories. When multiple memories contradict each other, entropy is
    high and the system should flag uncertainty or trigger external search.
    
    H(belief) = -sum(p_i * log(p_i)) over relevance-weighted memory clusters
    
    Low entropy (< 0.3): Memories agree -> answer with confidence
    Medium entropy (0.3-0.7): Some disagreement -> mention uncertainty
    High entropy (> 0.7): Contradictions -> flag as GUESSING or search externally
    """
    def __init__(self):
        self._calibration_history: list[tuple[float, bool]] = []
        
    def assess(self, query: str, results: list[MemoryResult]) -> MetamemoryAssessment:
        """Evaluate confidence in answering a query based on memory results."""
        memory_count = len(results)
        has_relevant_memories = memory_count > 0
        
        if not has_relevant_memories:
            return MetamemoryAssessment(
                query=query,
                confidence_level=ConfidenceLevel.NO_KNOWLEDGE,
                confidence_score=0.0,
                overall_confidence=0.0,
                reasoning="No relevant memories retrieved.",
                has_relevant_memories=False,
                memory_count=0,
                avg_retrievability=0.0,
                avg_relevance=0.0,
                source_diversity=0,
                recommendation="Suggest looking up externally.",
                belief_entropy=1.0,
                contradiction_detected=False,
                agreement_score=0.0,
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
        
        # v0.9: Compute Belief Entropy
        belief_entropy = self._compute_belief_entropy(results)
        agreement_score = self._compute_agreement(results)
        # v0.9: Contradiction requires BOTH high entropy AND low agreement.
        # High entropy alone just means many equally relevant memories (not bad).
        # Low agreement alone means diverse content (also not necessarily bad).
        # True contradiction = uncertain distribution + content disagrees.
        contradiction_detected = belief_entropy > 0.7 and agreement_score < 0.5
        
        # Base score on relevance, retrievability, and agreement
        base_score = (avg_relevance * 0.4) + (avg_retrievability * 0.3) + (agreement_score * 0.3)
        
        # Boost for diversity
        if source_diversity > 1:
            base_score = min(1.0, base_score + 0.1)
            
        # Penalize if very few memories
        if memory_count == 1:
            base_score = base_score * 0.8
        
        # v0.9: Penalize for high entropy ONLY when agreement is also low.
        # High entropy + high agreement = many corroborating sources (good!)
        # High entropy + low agreement = genuine contradiction (bad!)
        if belief_entropy > 0.5 and agreement_score < 0.7:
            base_score *= (1.0 - belief_entropy * 0.5)
            
        confidence_score = max(0.0, min(1.0, base_score))
        
        if contradiction_detected:
            confidence_level = ConfidenceLevel.GUESSING
            recommendation = 'Contradictory memories detected. Verify externally or use most recent.'
        elif confidence_score > 0.9:
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
            
        reasoning = (
            f"Found {memory_count} memories with avg relevance {avg_relevance:.2f}, "
            f"diversity {source_diversity}, belief entropy {belief_entropy:.2f}, "
            f"agreement {agreement_score:.2f}."
        )
        
        return MetamemoryAssessment(
            query=query,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            overall_confidence=confidence_score,
            reasoning=reasoning,
            has_relevant_memories=has_relevant_memories,
            memory_count=memory_count,
            avg_retrievability=avg_retrievability,
            avg_relevance=avg_relevance,
            source_diversity=source_diversity,
            recommendation=recommendation,
            belief_entropy=belief_entropy,
            contradiction_detected=contradiction_detected,
            agreement_score=agreement_score,
        )
    
    def _compute_belief_entropy(self, results: list[MemoryResult]) -> float:
        """
        Compute Belief Entropy over memory results.
        
        Uses relevance scores as a probability distribution over candidate
        memories. High entropy = spread across many equally relevant (possibly
        contradicting) memories. Low entropy = one dominant memory.
        
        H = -sum(p_i * log2(p_i)) / log2(n)  (normalized to [0, 1])
        """
        if len(results) <= 1:
            return 0.0
        
        # Create probability distribution from relevance scores
        scores = [max(r.relevance_score, 1e-10) for r in results]
        total = sum(scores)
        probs = [s / total for s in scores]
        
        # Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = math.log2(len(results))
        if max_entropy == 0:
            return 0.0
        
        return entropy / max_entropy
    
    def _compute_agreement(self, results: list[MemoryResult]) -> float:
        """
        Measure how much the top results agree with each other.
        
        Uses content overlap (Jaccard similarity) between top results.
        High agreement = memories reinforce each other.
        Low agreement = memories contradict or are about different things.
        """
        if len(results) <= 1:
            return 1.0
        
        top_results = results[:min(5, len(results))]
        contents = []
        for r in top_results:
            if hasattr(r, 'memory') and hasattr(r.memory, 'content'):
                contents.append(set(r.memory.content.lower().split()))
            else:
                contents.append(set())
        
        if len(contents) < 2:
            return 1.0
        
        # Average pairwise Jaccard similarity
        similarities = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                if contents[i] and contents[j]:
                    intersection = len(contents[i] & contents[j])
                    union = len(contents[i] | contents[j])
                    similarities.append(intersection / union if union > 0 else 0.0)
        
        return sum(similarities) / len(similarities) if similarities else 1.0
        
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
