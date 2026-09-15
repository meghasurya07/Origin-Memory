"""
Neurogenesis Engine — Dynamic Memory Allocation

Implements the brain's ability to create and destroy neurons daily.
The hippocampus generates ~700 new neurons/day (Spalding et al. 2013),
and aging destroys ~85,000/day.

In our system, this translates to:
- Dynamic creation of new memory "slots" based on novelty
- Pruning of unused/low-value memory slots
- Maintaining a population of active memory units

This is what the user specifically asked for:
"create and destroy number of them everyday"

Brain analogue: Adult hippocampal neurogenesis in the dentate gyrus
Key research: Spalding et al. 2013 (C-14 dating), Aimone et al. 2006
"""

from __future__ import annotations

import logging
import math
import random
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field

from .models import EpisodicMemory, MemoryStatus

logger = logging.getLogger(__name__)


class NeuronState(str, Enum):
    """Lifecycle states of a memory neuron."""
    IMMATURE = "immature"     # Just born, highly plastic, easily overwritten
    MATURING = "maturing"     # Developing, starting to stabilize
    MATURE = "mature"         # Fully integrated, stable
    DECLINING = "declining"   # Losing connections, about to die
    DEAD = "dead"             # Pruned / removed


class MemoryNeuron(BaseModel):
    """
    A single memory neuron — the computational unit of memory.
    
    Each neuron has a lifecycle:
    IMMATURE → MATURING → MATURE → DECLINING → DEAD
    
    New neurons are highly plastic (easy to overwrite/retrain),
    mature neurons are stable (hard to change but reliable).
    """
    id: str = Field(default_factory=lambda: f"neuron_{uuid.uuid4().hex[:12]}")
    state: NeuronState = NeuronState.IMMATURE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    activation_count: int = 0
    plasticity: float = 1.0       # 1.0 = highly plastic (new), 0.1 = rigid (mature)
    integration_score: float = 0.0  # How well integrated into the network
    memory_id: Optional[str] = None  # Linked episodic memory, if any
    
    def activate(self):
        """Activate this neuron (used during encoding/retrieval)."""
        self.activation_count += 1
        self.last_activated = datetime.now(timezone.utc)
        # Integration improves with use
        self.integration_score = min(1.0, self.integration_score + 0.1)
    
    def age_hours(self) -> float:
        """How old is this neuron in hours."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 3600.0
    
    def idle_hours(self) -> float:
        """Hours since last activation."""
        delta = datetime.now(timezone.utc) - self.last_activated
        return delta.total_seconds() / 3600.0


class NeurogenesisConfig(BaseModel):
    """Configuration for neurogenesis dynamics."""
    # Birth rate: new neurons per day
    # Human hippocampus: ~700/day (Spalding et al. 2013)
    birth_rate_per_day: int = 10       # Scaled down for computational purposes
    
    # Death rate: fraction of neurons that die per day
    # Human: ~85,000/day (cortical aging)
    death_rate_fraction: float = 0.02   # 2% daily turnover
    
    # Maturation timeline (in hours)
    immature_duration_hours: float = 24.0   # Stays immature for 24h
    maturing_duration_hours: float = 72.0   # Matures over 3 days
    
    # Survival criteria
    min_activations_to_survive: int = 2     # Must be used at least 2x
    min_integration_to_survive: float = 0.2  # Must integrate into network
    
    # Maximum population
    max_neurons: int = 1000


class NeurogenesisEvent(BaseModel):
    """Records a neurogenesis event (birth or death)."""
    event_type: str  # "birth" or "death" or "maturation"
    neuron_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""


class NeurogenesisEngine:
    """
    Manages the dynamic population of memory neurons.
    
    Implements:
    1. Neurogenesis — creating new neurons (memory slots)
    2. Apoptosis — programmed death of unused neurons
    3. Maturation — neurons becoming more stable over time
    4. Activity-dependent survival — used neurons survive, unused die
    
    Usage:
        engine = NeurogenesisEngine()
        
        # Create new neurons
        new_neurons = engine.generate_neurons(count=5)
        
        # Run daily lifecycle
        births, deaths, matured = engine.daily_cycle()
        
        # Get population stats
        stats = engine.get_statistics()
    """
    
    def __init__(self, config: Optional[NeurogenesisConfig] = None):
        self.config = config or NeurogenesisConfig()
        
        # Population
        self._neurons: Dict[str, MemoryNeuron] = {}
        
        # Event log
        self._events: List[NeurogenesisEvent] = []
        
        # Statistics
        self._total_born: int = 0
        self._total_died: int = 0
        self._total_matured: int = 0
        self._cycles_run: int = 0
        
        logger.info("NeurogenesisEngine initialized")
    
    def generate_neurons(self, count: Optional[int] = None) -> List[MemoryNeuron]:
        """
        Generate new immature neurons (memory slots).
        
        New neurons are highly plastic — they can easily be assigned to
        new memories. This is how the brain stays adaptable.
        
        Args:
            count: Number to generate (default: config.birth_rate_per_day)
            
        Returns:
            List of newly created neurons
        """
        count = count or self.config.birth_rate_per_day
        
        # Don't exceed max population
        available_slots = self.config.max_neurons - len(self._neurons)
        count = min(count, max(0, available_slots))
        
        if count <= 0:
            logger.info("Population at max capacity, no new neurons generated")
            return []
        
        new_neurons = []
        for _ in range(count):
            neuron = MemoryNeuron(
                plasticity=1.0,  # Highly plastic — ready to learn
                integration_score=0.0,
            )
            self._neurons[neuron.id] = neuron
            self._total_born += 1
            new_neurons.append(neuron)
            
            self._events.append(NeurogenesisEvent(
                event_type="birth",
                neuron_id=neuron.id,
                details=f"New immature neuron born (plasticity=1.0)"
            ))
        
        logger.info(f"Generated {count} new neurons (total population: {len(self._neurons)})")
        return new_neurons
    
    def update_maturation(self) -> List[MemoryNeuron]:
        """
        Update neuron lifecycle states based on age and activity.
        
        IMMATURE → MATURING (after immature_duration_hours)
        MATURING → MATURE (after maturing_duration_hours, IF sufficiently active)
        MATURE → DECLINING (if idle for too long)
        
        Returns:
            List of neurons that matured (changed state)
        """
        matured = []
        
        for neuron in list(self._neurons.values()):
            old_state = neuron.state
            age = neuron.age_hours()
            idle = neuron.idle_hours()
            
            if neuron.state == NeuronState.IMMATURE:
                if age >= self.config.immature_duration_hours:
                    neuron.state = NeuronState.MATURING
                    neuron.plasticity = 0.7  # Less plastic
                    
            elif neuron.state == NeuronState.MATURING:
                if age >= self.config.immature_duration_hours + self.config.maturing_duration_hours:
                    if (neuron.activation_count >= self.config.min_activations_to_survive and
                        neuron.integration_score >= self.config.min_integration_to_survive):
                        neuron.state = NeuronState.MATURE
                        neuron.plasticity = 0.2  # Stable
                        self._total_matured += 1
                    else:
                        neuron.state = NeuronState.DECLINING
                        neuron.plasticity = 0.0
                        
            elif neuron.state == NeuronState.MATURE:
                # Mature neurons decline if completely unused for a long time
                if idle > 168.0:  # 1 week of no activation
                    neuron.state = NeuronState.DECLINING
                    neuron.plasticity = 0.0
            
            if neuron.state != old_state:
                matured.append(neuron)
                self._events.append(NeurogenesisEvent(
                    event_type="maturation",
                    neuron_id=neuron.id,
                    details=f"State changed: {old_state.value} -> {neuron.state.value}"
                ))
        
        return matured
    
    def apoptosis(self) -> List[MemoryNeuron]:
        """
        Programmed cell death — remove declining and unfit neurons.
        
        Criteria for death:
        1. Declining neurons always die
        2. Immature/maturing neurons die if they haven't been activated enough
        3. Random fraction die (simulating natural turnover)
        
        Returns:
            List of neurons that died
        """
        dead = []
        
        for neuron in list(self._neurons.values()):
            should_die = False
            reason = ""
            
            # Declining neurons die
            if neuron.state == NeuronState.DECLINING:
                should_die = True
                reason = "declining state"
            
            # Immature neurons that are old but unused die
            elif neuron.state == NeuronState.IMMATURE:
                if (neuron.age_hours() > self.config.immature_duration_hours * 2 and
                    neuron.activation_count < 1):
                    should_die = True
                    reason = "old immature neuron, never activated"
            
            # Random turnover (very small fraction)
            elif random.random() < self.config.death_rate_fraction / 100:
                should_die = True
                reason = "random turnover (natural death)"
            
            if should_die:
                neuron.state = NeuronState.DEAD
                del self._neurons[neuron.id]
                self._total_died += 1
                dead.append(neuron)
                
                self._events.append(NeurogenesisEvent(
                    event_type="death",
                    neuron_id=neuron.id,
                    details=f"Died: {reason}"
                ))
        
        if dead:
            logger.info(f"Apoptosis: {len(dead)} neurons died (population: {len(self._neurons)})")
        
        return dead
    
    def daily_cycle(self) -> Tuple[List[MemoryNeuron], List[MemoryNeuron], List[MemoryNeuron]]:
        """
        Run a full daily neurogenesis cycle.
        
        This should be called once per day (or simulated day).
        
        Steps:
        1. Generate new neurons (birth)
        2. Update maturation states
        3. Run apoptosis (death)
        
        Returns:
            Tuple of (born, died, matured) neuron lists
        """
        self._cycles_run += 1
        
        # 1. Birth
        born = self.generate_neurons()
        
        # 2. Maturation
        matured = self.update_maturation()
        
        # 3. Death
        died = self.apoptosis()
        
        logger.info(
            f"Daily cycle #{self._cycles_run}: "
            f"+{len(born)} born, -{len(died)} died, "
            f"{len(matured)} matured, "
            f"population={len(self._neurons)}"
        )
        
        return born, died, matured
    
    def get_available_neurons(self, plasticity_min: float = 0.0) -> List[MemoryNeuron]:
        """Get neurons available for new memory assignment."""
        return [
            n for n in self._neurons.values()
            if n.memory_id is None and n.plasticity >= plasticity_min
            and n.state in (NeuronState.IMMATURE, NeuronState.MATURING)
        ]
    
    def assign_memory(self, neuron_id: str, memory_id: str) -> bool:
        """Assign an episodic memory to a neuron."""
        if neuron_id not in self._neurons:
            return False
        neuron = self._neurons[neuron_id]
        neuron.memory_id = memory_id
        neuron.activate()
        return True
    
    def get_population_by_state(self) -> Dict[str, int]:
        """Get population breakdown by state."""
        counts: Dict[str, int] = {}
        for state in NeuronState:
            if state == NeuronState.DEAD:
                continue
            counts[state.value] = sum(
                1 for n in self._neurons.values() if n.state == state
            )
        return counts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive neurogenesis statistics."""
        pop = self.get_population_by_state()
        avg_plasticity = 0.0
        avg_integration = 0.0
        if self._neurons:
            avg_plasticity = sum(n.plasticity for n in self._neurons.values()) / len(self._neurons)
            avg_integration = sum(n.integration_score for n in self._neurons.values()) / len(self._neurons)
        
        return {
            "population": len(self._neurons),
            "max_population": self.config.max_neurons,
            "by_state": pop,
            "total_born": self._total_born,
            "total_died": self._total_died,
            "total_matured": self._total_matured,
            "cycles_run": self._cycles_run,
            "average_plasticity": round(avg_plasticity, 3),
            "average_integration": round(avg_integration, 3),
            "turnover_rate": round(self._total_died / max(self._total_born, 1), 3),
        }
