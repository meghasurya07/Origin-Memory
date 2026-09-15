"""
Cognitive Map Engine — Successor Representation for Memory Navigation

Implements a cognitive map of "memory space" using successor representations,
inspired by hippocampal place cells and entorhinal grid cells.

Instead of finding memories by content similarity (semantic space), this
engine finds memories by transition probability (cognitive map space) —
which memories tend to be recalled together or in sequence.

Key concepts:
1. **Successor Representation (SR)** — Expected future occupancy of memory states
2. **Transition recording** — Track which memories follow each other during recall
3. **Map navigation** — Find memories that are "nearby" in transition space
4. **Sleep update** — Recompute SR during consolidation (offline replay)

Neuroscience basis:
- O'Keefe & Nadel 1978: Cognitive map theory
- Stachenfeld, Botvinick & Gershman 2017: "The hippocampus as a predictive map"
- Moser & Moser 2008: Grid cells as metric for cognitive space
- Behrens et al. 2018: "What is a cognitive map?"

Brain analogue:
- Place cells in CA1 → memory location in cognitive space
- Grid cells in MEC → multi-scale distance metric
- Successor representation → predictive map of transitions
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CognitiveMapConfig(BaseModel):
    """Configuration for the Cognitive Map."""
    gamma: float = 0.85           # Discount factor (how far ahead to predict)
    learning_rate: float = 0.1     # TD learning rate for SR updates
    transition_decay: float = 0.99 # Decay for transition counts over time
    min_transitions: int = 2       # Min transitions to count as "nearby"


class CognitiveMapEngine:
    """
    Builds and maintains a cognitive map of memory space.
    
    The map is defined by TRANSITIONS between memories:
    - If memory A is often recalled before memory B, they're "close" in the map
    - The Successor Representation captures multi-step transition structure
    - Navigation finds memories that are experientially linked, not just semantically similar
    
    Usage:
        engine = CognitiveMapEngine()
        
        # Record transitions during recall
        engine.record_transition("mem_1", "mem_2")
        engine.record_transition("mem_2", "mem_3")
        engine.record_transition("mem_1", "mem_3")
        
        # Find nearby memories in cognitive map
        nearby = engine.find_nearby("mem_1", top_k=5)
        
        # Update SR during sleep
        engine.update_successor_representation()
        
        # Get cognitive distance between two memories
        dist = engine.distance("mem_1", "mem_3")
    """
    
    def __init__(self, config: Optional[CognitiveMapConfig] = None):
        self.config = config or CognitiveMapConfig()
        
        # Raw transition counts: (from, to) → count
        self._transitions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Successor representation: from → {to: SR_value}
        self._sr: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Track last recalled memory for transition recording
        self._last_recalled: Optional[str] = None
        
        # Statistics
        self._total_transitions: int = 0
        self._total_updates: int = 0
        
        logger.info("CognitiveMapEngine initialized")
    
    def record_transition(self, from_memory_id: str, to_memory_id: str):
        """
        Record a transition from one memory to another.
        
        Call this during recall when memories are retrieved in sequence.
        """
        self._transitions[from_memory_id][to_memory_id] += 1.0
        self._total_transitions += 1
        
        # Incremental SR update (TD learning)
        self._td_update(from_memory_id, to_memory_id)
    
    def record_recall_sequence(self, memory_ids: List[str]):
        """
        Record a sequence of recalled memories.
        
        Creates transitions between each consecutive pair.
        """
        for i in range(len(memory_ids) - 1):
            self.record_transition(memory_ids[i], memory_ids[i + 1])
    
    def on_recall(self, memory_id: str):
        """
        Notify the cognitive map that a memory was just recalled.
        
        Automatically records transition from the previous recall.
        """
        if self._last_recalled is not None and self._last_recalled != memory_id:
            self.record_transition(self._last_recalled, memory_id)
        self._last_recalled = memory_id
    
    def _td_update(self, from_id: str, to_id: str):
        """
        Temporal-difference update of the successor representation.
        
        SR(s, s') ← SR(s, s') + α * (I(s=s') + γ * SR(to, s') - SR(s, s'))
        
        This is the key learning rule — it propagates transition structure
        through multi-step paths.
        """
        lr = self.config.learning_rate
        gamma = self.config.gamma
        
        # Get all states we've seen
        all_states = set(self._transitions.keys())
        all_states.update(self._sr.keys())
        all_states.add(from_id)
        all_states.add(to_id)
        
        for s_prime in all_states:
            # I(from = s') — identity: 1 if from_id == s_prime, else 0
            indicator = 1.0 if from_id == s_prime else 0.0
            
            # Current SR values
            current_sr = self._sr[from_id][s_prime]
            successor_sr = self._sr[to_id][s_prime]
            
            # TD update
            target = indicator + gamma * successor_sr
            self._sr[from_id][s_prime] += lr * (target - current_sr)
        
        self._total_updates += 1
    
    def update_successor_representation(self):
        """
        Full recomputation of the successor representation from transition counts.
        
        Call during 'sleep' for a complete refresh.
        
        Computes: SR = (I - γT)^(-1) where T is the transition matrix
        Uses iterative power method for efficiency.
        """
        # Get all memory IDs
        all_ids = list(set(
            list(self._transitions.keys()) + 
            [to_id for frm in self._transitions.values() for to_id in frm]
        ))
        
        if not all_ids:
            return
        
        # Build normalized transition matrix
        n = len(all_ids)
        id_to_idx = {mid: i for i, mid in enumerate(all_ids)}
        
        # Normalize transitions to probabilities
        T = [[0.0] * n for _ in range(n)]
        for from_id, targets in self._transitions.items():
            total = sum(targets.values())
            if total > 0:
                i = id_to_idx[from_id]
                for to_id, count in targets.items():
                    j = id_to_idx[to_id]
                    T[i][j] = count / total
        
        # Compute SR iteratively: SR ≈ I + γT + γ²T² + ... (power series)
        gamma = self.config.gamma
        
        # Initialize SR = I (identity)
        SR = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        # Add γ^k * T^k for k = 1, 2, ..., max_iter
        T_power = [row[:] for row in T]  # T^1
        
        for k in range(1, 20):  # 20 iterations ≈ converged
            gamma_k = gamma ** k
            if gamma_k < 0.001:
                break
            
            for i in range(n):
                for j in range(n):
                    SR[i][j] += gamma_k * T_power[i][j]
            
            # T_power = T_power @ T (matrix multiply)
            new_T = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for l in range(n):
                        new_T[i][j] += T_power[i][l] * T[l][j]
            T_power = new_T
        
        # Write back to SR dict
        self._sr.clear()
        for i, from_id in enumerate(all_ids):
            for j, to_id in enumerate(all_ids):
                if SR[i][j] > 0.001:  # Threshold noise
                    self._sr[from_id][to_id] = SR[i][j]
    
    def find_nearby(self, memory_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find memories that are nearby in cognitive map space.
        
        Returns memories with highest SR values (most likely to be
        visited from the given memory).
        
        Args:
            memory_id: Starting memory
            top_k: Number of results
            
        Returns:
            List of (memory_id, sr_value) tuples, sorted by proximity
        """
        if memory_id not in self._sr:
            return []
        
        neighbors = [
            (to_id, sr_val)
            for to_id, sr_val in self._sr[memory_id].items()
            if to_id != memory_id and sr_val > 0.01
        ]
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]
    
    def distance(self, from_id: str, to_id: str) -> float:
        """
        Compute cognitive distance between two memories.
        
        Higher SR value = closer in cognitive map = lower distance.
        
        Returns:
            Distance (0 = same memory, higher = further apart)
        """
        sr_val = self._sr.get(from_id, {}).get(to_id, 0.0)
        if sr_val <= 0:
            return float('inf')
        return 1.0 / sr_val
    
    def get_transition_probability(self, from_id: str, to_id: str) -> float:
        """Get raw transition probability between two memories."""
        if from_id not in self._transitions:
            return 0.0
        total = sum(self._transitions[from_id].values())
        if total == 0:
            return 0.0
        return self._transitions[from_id].get(to_id, 0.0) / total
    
    def decay_transitions(self):
        """Decay transition counts (forgetting old patterns)."""
        decay = self.config.transition_decay
        for from_id in self._transitions:
            for to_id in list(self._transitions[from_id].keys()):
                self._transitions[from_id][to_id] *= decay
                if self._transitions[from_id][to_id] < 0.01:
                    del self._transitions[from_id][to_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cognitive map statistics."""
        n_memories = len(set(
            list(self._transitions.keys()) +
            [to for frm in self._transitions.values() for to in frm]
        ))
        
        n_edges = sum(len(targets) for targets in self._transitions.values())
        
        return {
            "memories_in_map": n_memories,
            "transition_edges": n_edges,
            "total_transitions_recorded": self._total_transitions,
            "total_sr_updates": self._total_updates,
            "sr_entries": sum(len(v) for v in self._sr.values()),
        }
