"""
Pattern Completion Engine — CA3 Autoassociative Attractor Network

Implements the brain's most powerful memory retrieval mechanism:
a partial cue activates a stored pattern, and recurrent dynamics
"complete" the pattern to reconstruct the full memory.

This is HOW the brain retrieves from a 2-4% sparse engram trace.
It's the computational engine behind reconstruction.

Brain analogue: CA3 pyramidal neurons with recurrent collaterals
Neuroscience: Marr 1971, McNaughton & Morris 1987, Hopfield 1982

Mathematical foundation:
    Modern Hopfield Network (Ramsauer et al. 2020):
    h_new = softmax(β · X^T · h) · X
    
    This is mathematically equivalent to attention:
    Attention(Q, K, V) = softmax(Q K^T / √d) V
    
    The stored patterns ARE the keys/values.
    The cue IS the query.
    Pattern completion IS attention.
"""

from __future__ import annotations

import math
import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StoredPattern(BaseModel):
    """A pattern stored in the attractor network."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str           # Associated memory ID
    pattern: List[float]     # The full pattern vector
    label: str = ""          # Human-readable label
    access_count: int = 0
    strength: float = 1.0    # Synaptic strength (weakens with time/interference)


class CompletionResult(BaseModel):
    """Result of a pattern completion attempt."""
    completed_pattern: List[float]     # The completed (reconstructed) pattern
    matched_pattern_id: str            # Which stored pattern was matched
    matched_memory_id: str             # Associated memory ID
    similarity: float                  # Cosine similarity to best match
    completion_ratio: float            # How much of the pattern was "filled in"
    iterations: int                    # How many attractor iterations
    confidence: float                  # Confidence in the completion
    is_spurious: bool                  # Whether this might be a spurious attractor (false memory)


class PatternCompletionConfig(BaseModel):
    """Configuration for the Pattern Completion Engine."""
    beta: float = 8.0               # Inverse temperature (higher = sharper attention)
    max_iterations: int = 10        # Max attractor dynamics iterations
    convergence_threshold: float = 0.001  # Stop when change < this
    spurious_threshold: float = 0.5  # Below this similarity → likely spurious
    separation_threshold: float = 0.85  # Above this → patterns too similar, separate
    max_patterns: int = 10000       # Maximum stored patterns


class PatternCompletionEngine:
    """
    Implements CA3-like pattern completion via Modern Hopfield Networks.
    
    The key insight from Ramsauer et al. (2020): Modern Hopfield Networks
    with continuous states are MATHEMATICALLY EQUIVALENT to the attention
    mechanism in Transformers. This means:
    
    - Stored memories = Key/Value pairs
    - Retrieval cue = Query
    - Pattern completion = Attention computation
    - Attractor dynamics = Iterative attention refinement
    
    The network can reconstruct a FULL pattern from a PARTIAL cue,
    which is exactly what CA3 does with 20-30% of the original pattern.
    
    Usage:
        engine = PatternCompletionEngine()
        
        # Store patterns
        engine.store(pattern=[0.1, 0.2, ...], memory_id="mem-1")
        
        # Complete from partial cue (e.g., 30% of original)
        result = engine.complete(partial_cue=[0.1, 0.0, ...])
        # result.completed_pattern ≈ full original pattern
    """

    def __init__(self, config: Optional[PatternCompletionConfig] = None):
        self.config = config or PatternCompletionConfig()
        self._patterns: List[StoredPattern] = []
        self._pattern_dim: Optional[int] = None
        self._completion_count: int = 0
        self._spurious_count: int = 0
        
        logger.info(f"PatternCompletionEngine initialized (β={self.config.beta})")

    def _dot(self, a: List[float], b: List[float]) -> float:
        """Dot product."""
        return sum(x * y for x, y in zip(a, b))

    def _norm(self, v: List[float]) -> float:
        """L2 norm."""
        return math.sqrt(sum(x * x for x in v))

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        na, nb = self._norm(a), self._norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return self._dot(a, b) / (na * nb)

    def _softmax(self, logits: List[float]) -> List[float]:
        """Numerically stable softmax."""
        max_val = max(logits) if logits else 0.0
        exp_vals = [math.exp(x - max_val) for x in logits]
        total = sum(exp_vals)
        if total < 1e-10:
            return [1.0 / len(logits)] * len(logits)
        return [e / total for e in exp_vals]

    def store(
        self,
        pattern: List[float],
        memory_id: str,
        label: str = "",
        strength: float = 1.0
    ) -> StoredPattern:
        """
        Store a new pattern in the attractor network.
        
        In biological terms: form new synaptic connections via
        Hebbian learning in CA3 recurrent collaterals.
        
        Args:
            pattern: The full pattern to store
            memory_id: Associated memory ID
            label: Human-readable label
            strength: Initial synaptic strength
            
        Returns:
            The stored pattern object
        """
        if self._pattern_dim is None:
            self._pattern_dim = len(pattern)
        
        # Ensure consistent dimension
        p = list(pattern[:self._pattern_dim])
        while len(p) < self._pattern_dim:
            p.append(0.0)
        
        # Pattern separation check (DG function)
        # If too similar to existing pattern, don't store (avoid interference)
        for existing in self._patterns:
            sim = self._cosine_similarity(p, existing.pattern)
            if sim > self.config.separation_threshold:
                # Too similar — update existing instead of creating new
                existing.strength = max(existing.strength, strength)
                existing.access_count += 1
                logger.debug(f"Pattern too similar to {existing.id}, merged")
                return existing
        
        stored = StoredPattern(
            memory_id=memory_id,
            pattern=p,
            label=label,
            strength=strength
        )
        self._patterns.append(stored)
        
        # Evict oldest if at capacity
        if len(self._patterns) > self.config.max_patterns:
            # Remove pattern with lowest strength * access_count
            self._patterns.sort(key=lambda p: p.strength * (1 + p.access_count))
            self._patterns.pop(0)
        
        logger.debug(f"Stored pattern {stored.id} (dim={len(p)}, total={len(self._patterns)})")
        return stored

    def complete(
        self,
        cue: List[float],
        top_k: int = 1
    ) -> List[CompletionResult]:
        """
        Complete a partial cue into a full pattern using attractor dynamics.
        
        This is the core CA3 pattern completion:
        1. Present partial cue
        2. Compute attention (similarity) over all stored patterns
        3. Weighted combination produces initial completion
        4. Iterate until convergence (attractor dynamics)
        
        The Modern Hopfield update rule:
            h_new = softmax(β · X^T · h) · X
        
        where X = matrix of stored patterns, h = current state (starts as cue)
        
        Args:
            cue: Partial or noisy cue vector (can have zeros for unknown dimensions)
            top_k: Number of completion results to return
            
        Returns:
            List of CompletionResult, sorted by similarity
        """
        if not self._patterns:
            return []
        
        dim = self._pattern_dim or len(cue)
        
        # Pad/truncate cue
        h = list(cue[:dim])
        while len(h) < dim:
            h.append(0.0)
        
        # Track which dimensions were provided (non-zero in cue)
        provided_dims = sum(1 for x in h if abs(x) > 1e-10)
        total_dims = len(h)
        cue_fraction = provided_dims / max(total_dims, 1)
        
        # Attractor dynamics: iterate until convergence
        iterations = 0
        for iteration in range(self.config.max_iterations):
            iterations = iteration + 1
            
            # Step 1: Compute similarities (Q · K^T)
            logits = []
            for sp in self._patterns:
                sim = self._dot(h, sp.pattern) * self.config.beta * sp.strength
                logits.append(sim)
            
            # Step 2: Softmax attention weights
            weights = self._softmax(logits)
            
            # Step 3: Weighted combination (attention · V)
            h_new = [0.0] * dim
            for w, sp in zip(weights, self._patterns):
                for j in range(dim):
                    h_new[j] += w * sp.pattern[j]
            
            # Step 4: Check convergence
            change = sum((a - b) ** 2 for a, b in zip(h, h_new))
            h = h_new
            
            if change < self.config.convergence_threshold:
                break
        
        # Find top-k matching patterns
        results = []
        for sp in self._patterns:
            sim = self._cosine_similarity(h, sp.pattern)
            is_spurious = sim < self.config.spurious_threshold
            
            # Completion ratio: how much was "filled in" beyond the cue
            completion_ratio = 1.0 - cue_fraction
            
            # Confidence: based on similarity and cue fraction
            confidence = sim * (0.5 + 0.5 * cue_fraction)
            
            sp.access_count += 1
            
            results.append(CompletionResult(
                completed_pattern=h,
                matched_pattern_id=sp.id,
                matched_memory_id=sp.memory_id,
                similarity=sim,
                completion_ratio=completion_ratio,
                iterations=iterations,
                confidence=confidence,
                is_spurious=is_spurious
            ))
        
        results.sort(key=lambda r: r.similarity, reverse=True)
        
        self._completion_count += 1
        if results and results[0].is_spurious:
            self._spurious_count += 1
        
        return results[:top_k]

    def complete_by_memory_id(
        self,
        cue: List[float],
        memory_id: str
    ) -> Optional[CompletionResult]:
        """Complete a cue specifically toward a known memory's pattern."""
        target = None
        for sp in self._patterns:
            if sp.memory_id == memory_id:
                target = sp
                break
        
        if target is None:
            return None
        
        # Run completion
        results = self.complete(cue, top_k=len(self._patterns))
        
        # Find the result matching the target
        for r in results:
            if r.matched_memory_id == memory_id:
                return r
        
        return None

    def degrade_cue(
        self,
        full_pattern: List[float],
        fraction: float = 0.3
    ) -> List[float]:
        """
        Create a degraded (partial) cue from a full pattern.
        
        Useful for testing: simulate having only a partial memory.
        
        Args:
            full_pattern: The complete pattern
            fraction: What fraction of dimensions to KEEP (rest zeroed)
            
        Returns:
            Degraded cue with (1-fraction) dimensions zeroed out
        """
        import hashlib
        degraded = list(full_pattern)
        n_keep = max(1, int(len(degraded) * fraction))
        
        # Deterministic selection based on pattern content
        seed = hashlib.md5(str(full_pattern[:5]).encode()).digest()
        # Zero out dimensions that aren't kept
        indices = list(range(len(degraded)))
        # Simple deterministic shuffle
        for i in range(len(indices)):
            j = (seed[i % len(seed)] + i) % len(indices)
            indices[i], indices[j] = indices[j], indices[i]
        
        # Keep only the first n_keep indices
        keep_set = set(indices[:n_keep])
        for i in range(len(degraded)):
            if i not in keep_set:
                degraded[i] = 0.0
        
        return degraded

    @property
    def pattern_count(self) -> int:
        """Number of stored patterns."""
        return len(self._patterns)

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "pattern_count": len(self._patterns),
            "pattern_dimension": self._pattern_dim,
            "total_completions": self._completion_count,
            "spurious_completions": self._spurious_count,
            "spurious_rate": self._spurious_count / max(self._completion_count, 1),
            "avg_pattern_strength": (
                sum(p.strength for p in self._patterns) / max(len(self._patterns), 1)
            ),
        }
