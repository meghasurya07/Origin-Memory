"""
World Model Engine — Predictive Transition Networks

The brain doesn't just STORE the past — it PREDICTS the future.
This module builds predictive transition models from neocortical
schemas and episodic replay.

Neuroscience basis:
- Hippocampal preplay: before navigating a route, rats' place cells
  fire in the sequence they'll follow (Dragoi & Tonegawa 2011)
- Default Mode Network: in rest, humans mentally simulate futures
- Prospection: Schacter & Addis (2007) — memory exists FOR prediction

How it works:
1. During sleep, the neocortex extracts TRANSITION PATTERNS:
   A → B → C (what follows what?)
2. Given a current state, the world model PREDICTS likely next states
3. The brain can then "mentally simulate" sequences before acting

Brain analogue:
- vmPFC world models
- Successor Representations in entorhinal cortex (already built in cognitive_map)
- Hippocampal preplay / replay during SWS

Example:
    # After many episodes: "user greets" → "asks question" → "gets answer"
    wm.predict_next("user greets")
    → [("asks question", 0.85), ("says goodbye", 0.1)]

    # Simulate full sequence
    wm.simulate("user greets", steps=3)
    → ["user greets", "asks question", "gets answer"]
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorldModelConfig(BaseModel):
    """Configuration for the world model engine."""
    # Minimum transition count to be considered a real pattern
    min_transition_count: int = 2
    # Maximum number of states to track
    max_states: int = 10_000
    # Temporal decay for old transitions
    decay_rate: float = 0.01
    # Simulation max depth (prevent infinite loops)
    max_simulation_depth: int = 20
    # Confidence threshold for predictions
    prediction_threshold: float = 0.05


@dataclass
class TransitionRecord:
    """A single observed state transition."""
    from_state: str
    to_state: str
    count: int = 0
    probability: float = 0.0
    last_observed_step: int = 0


@dataclass
class WorldState:
    """A state node in the world model."""
    name: str
    # Outgoing transitions: to_state → TransitionRecord
    transitions: Dict[str, TransitionRecord] = field(default_factory=dict)
    # How many times this state has been visited
    visit_count: int = 0
    # Is this a terminal/absorbing state?
    is_terminal: bool = False


class WorldModelEngine:
    """
    Predictive World Model — learns state transitions and simulates futures.
    
    Usage:
        wm = WorldModelEngine()
        
        # Observe sequences during episodic replay
        wm.observe_sequence(["greet", "ask_question", "get_answer", "thank"])
        wm.observe_sequence(["greet", "ask_question", "get_answer", "leave"])
        wm.observe_sequence(["greet", "small_talk", "leave"])
        
        # Predict next state
        wm.predict_next("greet")
        → [("ask_question", 0.67), ("small_talk", 0.33)]
        
        # Simulate forward
        wm.simulate("greet", steps=3)
        → ["greet", "ask_question", "get_answer"]
        
        # Get transition matrix
        wm.get_transition_matrix()
    """
    
    def __init__(self, config: Optional[WorldModelConfig] = None):
        self.config = config or WorldModelConfig()
        
        # State graph
        self._states: Dict[str, WorldState] = {}
        
        # Statistics
        self._total_sequences: int = 0
        self._total_transitions: int = 0
        self._global_step: int = 0
        
        logger.info("WorldModelEngine initialized")
    
    def _get_or_create_state(self, name: str) -> WorldState:
        """Get existing state or create new one."""
        if name not in self._states:
            if len(self._states) >= self.config.max_states:
                # Evict least-visited state
                least = min(
                    self._states.values(),
                    key=lambda s: s.visit_count
                )
                del self._states[least.name]
            self._states[name] = WorldState(name=name)
        return self._states[name]
    
    def observe_transition(self, from_state: str, to_state: str):
        """
        Record a single state transition.
        
        Call this during episodic replay to build the transition model.
        """
        self._global_step += 1
        
        state = self._get_or_create_state(from_state)
        self._get_or_create_state(to_state)  # Ensure target exists
        
        state.visit_count += 1
        
        if to_state not in state.transitions:
            state.transitions[to_state] = TransitionRecord(
                from_state=from_state,
                to_state=to_state,
            )
        
        record = state.transitions[to_state]
        record.count += 1
        record.last_observed_step = self._global_step
        self._total_transitions += 1
        
        # Update probabilities for this state
        self._update_probabilities(from_state)
    
    def observe_sequence(self, sequence: List[str]):
        """
        Record a full sequence of state transitions.
        
        Args:
            sequence: Ordered list of states, e.g. ["greet", "ask", "answer"]
        """
        if len(sequence) < 2:
            return
        
        for i in range(len(sequence) - 1):
            self.observe_transition(sequence[i], sequence[i + 1])
        
        self._total_sequences += 1
    
    def _update_probabilities(self, state_name: str):
        """Recompute transition probabilities for a state."""
        state = self._states.get(state_name)
        if not state:
            return
        
        total = sum(t.count for t in state.transitions.values())
        if total == 0:
            return
        
        for record in state.transitions.values():
            record.probability = record.count / total
    
    def predict_next(
        self,
        current_state: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely next states from current state.
        
        Returns:
            List of (next_state, probability) sorted by probability
        """
        state = self._states.get(current_state)
        if not state:
            return []
        
        predictions = [
            (t.to_state, t.probability)
            for t in state.transitions.values()
            if t.probability >= self.config.prediction_threshold
        ]
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]
    
    def simulate(
        self,
        start_state: str,
        steps: int = 5,
        strategy: str = "greedy",
    ) -> List[str]:
        """
        Simulate a sequence forward from start_state.
        
        Args:
            start_state: Where to start
            steps: Maximum number of steps
            strategy: "greedy" (always pick most likely) or 
                      "sample" (weighted random sampling)
        
        Returns:
            List of states in the simulated sequence
        """
        sequence = [start_state]
        current = start_state
        visited = {start_state}
        
        for _ in range(min(steps, self.config.max_simulation_depth)):
            predictions = self.predict_next(current)
            if not predictions:
                break
            
            if strategy == "greedy":
                next_state = predictions[0][0]
            else:
                # Weighted random sampling
                import random
                states, probs = zip(*predictions)
                next_state = random.choices(states, weights=probs, k=1)[0]
            
            # Prevent infinite loops
            if next_state in visited:
                break
            
            sequence.append(next_state)
            visited.add(next_state)
            current = next_state
        
        return sequence
    
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get the full transition probability matrix.
        
        Returns:
            Nested dict: from_state → to_state → probability
        """
        matrix = {}
        for name, state in self._states.items():
            matrix[name] = {
                t.to_state: t.probability
                for t in state.transitions.values()
            }
        return matrix
    
    def get_reachable_states(self, from_state: str, max_depth: int = 3) -> Set[str]:
        """Get all states reachable from a given state within max_depth steps."""
        reachable = set()
        frontier = {from_state}
        
        for _ in range(max_depth):
            new_frontier = set()
            for state_name in frontier:
                state = self._states.get(state_name)
                if state:
                    for t in state.transitions.values():
                        if t.to_state not in reachable:
                            new_frontier.add(t.to_state)
                            reachable.add(t.to_state)
            frontier = new_frontier
            if not frontier:
                break
        
        return reachable
    
    def surprise(self, from_state: str, observed_next: str) -> float:
        """
        Compute surprise (negative log probability) of an observed transition.
        
        High surprise = unexpected event → should trigger special encoding.
        
        Returns:
            Surprise value (0 = expected, higher = more surprising)
            Returns max surprise if transition never observed.
        """
        state = self._states.get(from_state)
        if not state:
            return 10.0  # Maximum surprise for unknown state
        
        record = state.transitions.get(observed_next)
        if not record or record.probability == 0:
            return 10.0  # Never observed this transition
        
        return -math.log(record.probability)
    
    def decay(self):
        """Apply decay to transition counts (forget old patterns)."""
        for state in self._states.values():
            for record in list(state.transitions.values()):
                record.count = int(record.count * (1.0 - self.config.decay_rate))
                if record.count <= 0:
                    del state.transitions[record.to_state]
            self._update_probabilities(state.name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get world model statistics."""
        return {
            "total_states": len(self._states),
            "total_transitions": self._total_transitions,
            "total_sequences": self._total_sequences,
            "avg_branching_factor": (
                sum(len(s.transitions) for s in self._states.values()) / max(1, len(self._states))
            ),
            "most_visited": sorted(
                [(s.name, s.visit_count) for s in self._states.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
        }
