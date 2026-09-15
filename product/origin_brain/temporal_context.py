"""
Temporal Context Engine — Howard & Kahana TCM (2002) / Polyn CMR (2009)

Implements the drifting temporal context vector that binds memories to their
temporal neighborhood. When you recall an event, the context at encoding time
is partially reinstated, allowing "mental time travel" — recalling temporally
adjacent events.

Brain analogue: Entorhinal Cortex (medial), hippocampal CA1 temporal coding
Neuroscience: Theta-gamma coupling, phase precession, contiguity effect

Mathematical foundation (Doc 25 §1):
    t_i = ρ_i * t_{i-1} + β * t^IN_i
    ρ_i = sqrt(1 + β²[(t_{i-1} · t^IN_i)² - 1]) - β(t_{i-1} · t^IN_i)
    
    M^FT: Feature → Context matrix (updates context when item presented)
    M^TF: Context → Feature matrix (retrieves items given context cue)
"""

from __future__ import annotations

import math
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContextSnapshot(BaseModel):
    """A snapshot of the temporal context at a specific moment."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    associated_memory_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemporalContextConfig(BaseModel):
    """Configuration for the Temporal Context Engine."""
    dimension: int = 128
    beta: float = 0.5           # Drift rate: how much new items shift context
    learning_rate: float = 0.1  # Hebbian learning rate for M^FT / M^TF
    max_snapshots: int = 10000  # Maximum context history retained
    contiguity_window: int = 5  # Number of adjacent items for contiguity bonus


class TemporalContextEngine:
    """
    Implements the Temporal Context Model (TCM) for binding memories
    to their temporal context.
    
    The context vector slowly drifts as new items are encoded, creating
    a continuous representation of "when" in the agent's experience.
    Items encoded close in time share similar contexts, explaining the
    contiguity effect in free recall.
    
    Usage:
        engine = TemporalContextEngine(config=TemporalContextConfig())
        
        # On encoding a new memory
        context = engine.update(feature_vector, memory_id="mem-123")
        
        # On recall — probe with a cue
        similar_contexts = engine.probe(cue_vector, top_k=5)
        
        # Reinstate a past context for "mental time travel"
        engine.reinstate(memory_id="mem-123", strength=0.7)
    """

    def __init__(self, config: Optional[TemporalContextConfig] = None):
        self.config = config or TemporalContextConfig()
        dim = self.config.dimension

        # The current context vector (normalized to unit length)
        self._context: List[float] = [0.0] * dim
        # Initialize with small random-ish values for non-degeneracy
        import hashlib
        seed = hashlib.md5(b"origin_brain_tcm").digest()
        for i in range(dim):
            self._context[i] = ((seed[i % len(seed)] - 128) / 256.0) * 0.01
        self._normalize_context()

        # Feature-to-Context matrix (M^FT): updates context when item is presented
        # Stored as list of rows, each row is a dim-length vector
        self._M_FT: List[List[float]] = [[0.0] * dim for _ in range(dim)]
        # Initialize with small identity-like values
        for i in range(dim):
            self._M_FT[i][i] = 0.01

        # Context-to-Feature matrix (M^TF): retrieves items given context
        self._M_TF: List[List[float]] = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            self._M_TF[i][i] = 0.01

        # History of context snapshots
        self._snapshots: List[ContextSnapshot] = []
        
        # Memory ID → snapshot index mapping
        self._memory_to_snapshot: Dict[str, int] = {}

        # Sequence counter for ordering
        self._sequence_counter: int = 0

        logger.info(f"TemporalContextEngine initialized (dim={dim}, β={self.config.beta})")

    def _normalize_context(self) -> None:
        """Normalize context vector to unit length."""
        norm = math.sqrt(sum(x * x for x in self._context))
        if norm > 1e-10:
            self._context = [x / norm for x in self._context]

    def _dot(self, a: List[float], b: List[float]) -> float:
        """Dot product of two vectors."""
        return sum(x * y for x, y in zip(a, b))

    def _mat_vec(self, matrix: List[List[float]], vec: List[float]) -> List[float]:
        """Matrix-vector multiplication."""
        return [self._dot(row, vec) for row in matrix]

    def _compute_rho(self, t_in: List[float]) -> float:
        """
        Compute the TCM normalization factor ρ.
        ρ = sqrt(1 + β²[(t · t_in)² - 1]) - β(t · t_in)
        """
        beta = self.config.beta
        dot = self._dot(self._context, t_in)
        inner = 1.0 + beta * beta * (dot * dot - 1.0)
        # Clamp for numerical stability
        inner = max(inner, 0.0)
        rho = math.sqrt(inner) - beta * dot
        return rho

    def update(
        self,
        feature_vector: List[float],
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextSnapshot:
        """
        Update the context vector when a new item is encoded.
        
        Implements: t_i = ρ_i * t_{i-1} + β * t^IN_i
        Then performs Hebbian binding between feature and context.
        
        Args:
            feature_vector: The feature representation of the item (will be
                           truncated/padded to match config.dimension)
            memory_id: Optional ID of the associated memory
            metadata: Optional metadata for the context snapshot
            
        Returns:
            ContextSnapshot of the new context state
        """
        dim = self.config.dimension
        
        # Ensure feature vector matches dimension
        fv = list(feature_vector[:dim])
        while len(fv) < dim:
            fv.append(0.0)

        # Step 1: Compute context input via M^FT
        t_in = self._mat_vec(self._M_FT, fv)
        # Normalize t_in
        t_in_norm = math.sqrt(sum(x * x for x in t_in))
        if t_in_norm > 1e-10:
            t_in = [x / t_in_norm for x in t_in]

        # Step 2: Compute ρ (normalization factor)
        rho = self._compute_rho(t_in)

        # Step 3: Update context: t_i = ρ * t_{i-1} + β * t_in
        beta = self.config.beta
        self._context = [
            rho * old + beta * new
            for old, new in zip(self._context, t_in)
        ]
        self._normalize_context()

        # Step 4: Hebbian learning — bind feature to context
        lr = self.config.learning_rate
        for i in range(dim):
            for j in range(dim):
                # M^TF: context → feature (for retrieval)
                self._M_TF[i][j] += lr * fv[i] * self._context[j]
                # M^FT: feature → context (for context update)
                self._M_FT[i][j] += lr * self._context[i] * fv[j]

        # Step 5: Save snapshot
        snapshot = ContextSnapshot(
            vector=list(self._context),
            associated_memory_id=memory_id,
            metadata=metadata or {}
        )
        self._snapshots.append(snapshot)
        
        if memory_id:
            self._memory_to_snapshot[memory_id] = len(self._snapshots) - 1

        self._sequence_counter += 1

        # Evict old snapshots if needed
        if len(self._snapshots) > self.config.max_snapshots:
            removed = self._snapshots.pop(0)
            if removed.associated_memory_id:
                self._memory_to_snapshot.pop(removed.associated_memory_id, None)
            # Reindex
            self._memory_to_snapshot = {
                mid: idx - 1
                for mid, idx in self._memory_to_snapshot.items()
                if idx > 0
            }

        logger.debug(
            f"Context updated (seq={self._sequence_counter}, "
            f"memory={memory_id}, snapshots={len(self._snapshots)})"
        )

        return snapshot

    def probe(
        self,
        cue_vector: List[float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Probe the context-to-feature matrix to find memories associated
        with the current or a given context.
        
        Uses M^TF to map context → feature activations, then compares
        against stored snapshots.
        
        Args:
            cue_vector: Feature vector to use as retrieval cue
            top_k: Number of results to return
            
        Returns:
            List of (memory_id, similarity_score) tuples, sorted by score
        """
        dim = self.config.dimension
        cue = list(cue_vector[:dim])
        while len(cue) < dim:
            cue.append(0.0)

        # Get the context that this cue would evoke
        evoked_context = self._mat_vec(self._M_FT, cue)
        ec_norm = math.sqrt(sum(x * x for x in evoked_context))
        if ec_norm > 1e-10:
            evoked_context = [x / ec_norm for x in evoked_context]

        # Compare against all stored snapshots
        results: List[Tuple[str, float]] = []
        for snapshot in self._snapshots:
            if not snapshot.associated_memory_id:
                continue
            sim = self._dot(evoked_context, snapshot.vector)
            results.append((snapshot.associated_memory_id, sim))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_temporal_neighbors(
        self,
        memory_id: str,
        window: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get temporally adjacent memories (the contiguity effect).
        
        Items encoded close in time share similar contexts, so recalling
        one should make adjacent items easier to recall.
        
        Args:
            memory_id: The memory to find neighbors for
            window: Number of neighbors on each side (default: config.contiguity_window)
            
        Returns:
            List of (memory_id, contiguity_score) tuples
        """
        if memory_id not in self._memory_to_snapshot:
            return []

        idx = self._memory_to_snapshot[memory_id]
        w = window or self.config.contiguity_window
        
        # Get the target context
        target_ctx = self._snapshots[idx].vector

        neighbors: List[Tuple[str, float]] = []
        for offset in range(-w, w + 1):
            if offset == 0:
                continue
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(self._snapshots):
                snap = self._snapshots[neighbor_idx]
                if snap.associated_memory_id:
                    # Context similarity = contiguity score
                    sim = self._dot(target_ctx, snap.vector)
                    # Apply distance decay (closer = stronger)
                    distance_decay = 1.0 / (1.0 + abs(offset) * 0.3)
                    score = sim * distance_decay
                    neighbors.append((snap.associated_memory_id, score))

        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def reinstate(
        self,
        memory_id: str,
        strength: float = 0.5
    ) -> bool:
        """
        Reinstate a past context — "mental time travel."
        
        When you recall a memory, the context at encoding time is
        partially reinstated, allowing you to recall temporally
        adjacent events. This is the key mechanism of the contiguity
        effect in free recall.
        
        Args:
            memory_id: The memory whose context to reinstate
            strength: How strongly to reinstate (0.0 = none, 1.0 = full)
            
        Returns:
            True if reinstated, False if memory not found
        """
        if memory_id not in self._memory_to_snapshot:
            return False

        idx = self._memory_to_snapshot[memory_id]
        past_context = self._snapshots[idx].vector
        
        # Blend past context with current context
        # t_current = (1 - strength) * t_current + strength * t_past
        self._context = [
            (1.0 - strength) * curr + strength * past
            for curr, past in zip(self._context, past_context)
        ]
        self._normalize_context()

        logger.debug(f"Context reinstated for memory {memory_id} (strength={strength})")
        return True

    def get_context_similarity(self, memory_id_a: str, memory_id_b: str) -> float:
        """
        Compute the context similarity between two memories.
        
        High similarity means the memories were encoded in similar temporal
        contexts (close in time or in similar cognitive states).
        """
        if memory_id_a not in self._memory_to_snapshot:
            return 0.0
        if memory_id_b not in self._memory_to_snapshot:
            return 0.0
        
        ctx_a = self._snapshots[self._memory_to_snapshot[memory_id_a]].vector
        ctx_b = self._snapshots[self._memory_to_snapshot[memory_id_b]].vector
        return self._dot(ctx_a, ctx_b)

    @property
    def current_context(self) -> List[float]:
        """Return a copy of the current context vector."""
        return list(self._context)

    @property
    def snapshot_count(self) -> int:
        """Number of stored context snapshots."""
        return len(self._snapshots)

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "dimension": self.config.dimension,
            "beta": self.config.beta,
            "snapshot_count": len(self._snapshots),
            "memory_bindings": len(self._memory_to_snapshot),
            "sequence_counter": self._sequence_counter,
            "context_norm": math.sqrt(sum(x * x for x in self._context)),
        }
