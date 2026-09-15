"""
Engram Tracking Engine — Memory Trace Formation and Reactivation

Implements the neuroscience of ENGRAMS — the physical traces of memories
in the brain.

Key concepts:
1. **Engram formation** — When a memory is encoded, a specific ensemble
   of neurons becomes the "engram" for that memory
2. **Engram cells** — Sparse subset (~2-5%) of neurons allocated to each memory
3. **Excitability competition** — Neurons compete for inclusion based on
   intrinsic excitability (CREB levels)
4. **Engram overlap** — Memories sharing engram cells become linked
5. **Silent engrams** — Engrams exist but can't be retrieved (Alzheimer's model)
6. **Engram reactivation** — Retrieving a memory reactivates its engram ensemble

Neuroscience basis:
- Josselyn & Tonegawa 2020: "Memory Engrams: Recalling the Past and Imagining the Future"
- Tonegawa lab 2024: Systems consolidation of engram complexes
- Liu et al. 2012: Optogenetic reactivation of hippocampal engram cells

Brain analogue: c-Fos+ neurons in hippocampal CA1/CA3 and amygdala
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EngramState(str, Enum):
    """State of an engram trace."""
    ACTIVE = "active"        # Recently formed/reactivated, readily accessible
    STABLE = "stable"        # Consolidated, accessible with proper cue
    SILENT = "silent"        # Exists but not accessible (needs reactivation)
    DEGRADED = "degraded"    # Partially lost, retrieval produces errors
    DISSOLVED = "dissolved"  # Fully lost, cannot be recovered


class EngramCell(BaseModel):
    """
    A simulated 'neuron' allocated to an engram.
    
    In the brain, engram cells are neurons expressing c-Fos/Arc
    during encoding that become necessary for memory retrieval.
    """
    id: str = Field(default_factory=lambda: f"ec_{uuid.uuid4().hex[:8]}")
    excitability: float = 0.5    # Intrinsic excitability (CREB analog)
    allocation_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    activation_count: int = 1
    
    # Which engrams this cell belongs to (enables overlap detection)
    engram_ids: List[str] = Field(default_factory=list)


class Engram(BaseModel):
    """
    A memory engram — the ensemble of cells that encode a specific memory.
    
    Properties:
    - cell_ids: The neurons allocated to this engram
    - strength: How accessible the engram is (decays over time)
    - state: Current lifecycle state
    - overlap: Other engrams sharing cells with this one
    """
    id: str = Field(default_factory=lambda: f"eng_{uuid.uuid4().hex[:8]}")
    memory_id: str                # The memory this engram represents
    cell_ids: List[str] = Field(default_factory=list)
    strength: float = 1.0         # Accessibility (1.0 = just formed)
    state: EngramState = EngramState.ACTIVE
    formation_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_reactivated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reactivation_count: int = 1
    
    # Consolidation tracking
    consolidated: bool = False
    consolidation_time: Optional[datetime] = None


class EngramConfig(BaseModel):
    """Configuration for engram tracking."""
    # Allocation parameters
    cells_per_engram: int = 5         # Number of cells per engram (~2-5% sparsity)
    total_cell_pool: int = 200        # Total available cells
    excitability_boost: float = 0.2   # Boost from CREB/encoding
    excitability_decay: float = 0.01  # Decay rate per hour
    
    # State transitions
    active_to_stable_hours: float = 6.0    # Consolidation window
    stable_to_silent_hours: float = 720.0  # ~30 days without reactivation
    silent_to_degraded_hours: float = 2160.0  # ~90 days
    
    # Overlap effects
    max_overlap_fraction: float = 0.4  # Max fraction of cells shared
    overlap_linking_threshold: float = 0.2  # Min overlap for linking effect


class EngramEngine:
    """
    Tracks memory engrams — the neural ensembles that physically encode memories.
    
    This engine manages:
    - Which cells are allocated to each memory (excitability-based competition)
    - How engrams transition through states (active → stable → silent → degraded)
    - Engram overlap detection (linked memories)
    - Engram reactivation on retrieval
    
    Usage:
        engine = EngramEngine()
        
        # Allocate engram when encoding
        engram = engine.allocate_engram("memory_123")
        
        # Reactivate on retrieval
        engine.reactivate("memory_123")
        
        # Find linked memories via overlap
        linked = engine.find_linked_memories("memory_123")
        
        # Run maintenance (state transitions, excitability decay)
        engine.maintain()
    """
    
    def __init__(self, config: Optional[EngramConfig] = None):
        self.config = config or EngramConfig()
        
        # Cell pool
        self._cells: Dict[str, EngramCell] = {}
        self._initialize_cell_pool()
        
        # Engram registry
        self._engrams: Dict[str, Engram] = {}        # engram_id → Engram
        self._memory_to_engram: Dict[str, str] = {}   # memory_id → engram_id
        
        logger.info(
            f"EngramEngine initialized with {self.config.total_cell_pool} cells, "
            f"{self.config.cells_per_engram} cells/engram"
        )
    
    def _initialize_cell_pool(self):
        """Create the initial pool of cells with random excitability."""
        import random
        for _ in range(self.config.total_cell_pool):
            cell = EngramCell(
                excitability=0.3 + random.random() * 0.4  # Range 0.3-0.7
            )
            self._cells[cell.id] = cell
    
    def allocate_engram(self, memory_id: str) -> Engram:
        """
        Allocate an engram for a new memory.
        
        Uses excitability-based competition: the most excitable cells
        are allocated to the new engram (CREB model from Josselyn lab).
        
        Args:
            memory_id: ID of the memory being encoded
            
        Returns:
            The newly allocated Engram
        """
        # Check if engram already exists
        if memory_id in self._memory_to_engram:
            return self._engrams[self._memory_to_engram[memory_id]]
        
        # Sort cells by excitability (winner-take-all competition)
        available_cells = sorted(
            self._cells.values(),
            key=lambda c: c.excitability,
            reverse=True
        )
        
        # Allocate top-N most excitable cells
        allocated_ids = []
        for cell in available_cells[:self.config.cells_per_engram]:
            cell.excitability += self.config.excitability_boost
            cell.excitability = min(1.0, cell.excitability)
            cell.last_activated = datetime.now(timezone.utc)
            cell.activation_count += 1
            allocated_ids.append(cell.id)
        
        # Create engram
        engram = Engram(
            memory_id=memory_id,
            cell_ids=allocated_ids,
        )
        
        # Register cell membership
        for cell_id in allocated_ids:
            self._cells[cell_id].engram_ids.append(engram.id)
        
        self._engrams[engram.id] = engram
        self._memory_to_engram[memory_id] = engram.id
        
        return engram
    
    def reactivate(self, memory_id: str) -> Optional[Engram]:
        """
        Reactivate an engram on retrieval.
        
        Strengthens the engram and boosts cell excitability.
        Can reactivate SILENT engrams (like optogenetic reactivation).
        
        Returns:
            The reactivated Engram, or None if not found
        """
        engram_id = self._memory_to_engram.get(memory_id)
        if engram_id is None:
            return None
        
        engram = self._engrams[engram_id]
        
        # Can't reactivate dissolved engrams
        if engram.state == EngramState.DISSOLVED:
            return None
        
        now = datetime.now(timezone.utc)
        
        # Reactivate
        engram.strength = min(1.0, engram.strength + 0.2)
        engram.last_reactivated = now
        engram.reactivation_count += 1
        
        # If silent, restore to active (optogenetic-like reactivation)
        if engram.state in (EngramState.SILENT, EngramState.DEGRADED):
            engram.state = EngramState.ACTIVE
        
        # Boost cell excitability
        for cell_id in engram.cell_ids:
            cell = self._cells.get(cell_id)
            if cell:
                cell.excitability = min(1.0, cell.excitability + 0.1)
                cell.last_activated = now
                cell.activation_count += 1
        
        return engram
    
    def find_linked_memories(self, memory_id: str) -> List[Tuple[str, float]]:
        """
        Find memories linked to this one through engram cell overlap.
        
        When two memories share engram cells, they become associated.
        This is the physical basis for memory linking/chaining.
        
        Returns:
            List of (memory_id, overlap_fraction) tuples, sorted by overlap
        """
        engram_id = self._memory_to_engram.get(memory_id)
        if engram_id is None:
            return []
        
        engram = self._engrams[engram_id]
        my_cells = set(engram.cell_ids)
        
        linked = []
        for other_eng in self._engrams.values():
            if other_eng.id == engram_id:
                continue
            if other_eng.state == EngramState.DISSOLVED:
                continue
            
            other_cells = set(other_eng.cell_ids)
            overlap = len(my_cells & other_cells)
            
            if overlap > 0:
                overlap_fraction = overlap / len(my_cells)
                if overlap_fraction >= self.config.overlap_linking_threshold:
                    linked.append((other_eng.memory_id, overlap_fraction))
        
        linked.sort(key=lambda x: x[1], reverse=True)
        return linked
    
    def consolidate_engram(self, memory_id: str) -> bool:
        """Mark an engram as consolidated (systems consolidation)."""
        engram_id = self._memory_to_engram.get(memory_id)
        if engram_id is None:
            return False
        
        engram = self._engrams[engram_id]
        engram.consolidated = True
        engram.consolidation_time = datetime.now(timezone.utc)
        engram.state = EngramState.STABLE
        return True
    
    def maintain(self) -> Dict[str, int]:
        """
        Run maintenance cycle — decay excitability and transition engram states.
        
        Returns:
            Dict with counts of state transitions
        """
        now = datetime.now(timezone.utc)
        transitions = {"active_to_stable": 0, "stable_to_silent": 0, 
                       "silent_to_degraded": 0, "degraded_to_dissolved": 0}
        
        for engram in list(self._engrams.values()):
            hours_since_reactivation = (
                now - engram.last_reactivated
            ).total_seconds() / 3600
            
            if engram.state == EngramState.ACTIVE:
                if hours_since_reactivation > self.config.active_to_stable_hours:
                    engram.state = EngramState.STABLE
                    transitions["active_to_stable"] += 1
            
            elif engram.state == EngramState.STABLE:
                if hours_since_reactivation > self.config.stable_to_silent_hours:
                    engram.state = EngramState.SILENT
                    engram.strength *= 0.5
                    transitions["stable_to_silent"] += 1
            
            elif engram.state == EngramState.SILENT:
                if hours_since_reactivation > self.config.silent_to_degraded_hours:
                    engram.state = EngramState.DEGRADED
                    engram.strength *= 0.3
                    transitions["silent_to_degraded"] += 1
            
            elif engram.state == EngramState.DEGRADED:
                if engram.strength < 0.05:
                    engram.state = EngramState.DISSOLVED
                    transitions["degraded_to_dissolved"] += 1
        
        # Decay cell excitability
        for cell in self._cells.values():
            hours = (now - cell.last_activated).total_seconds() / 3600
            decay = self.config.excitability_decay * hours
            cell.excitability = max(0.1, cell.excitability - decay)
        
        return transitions
    
    def get_engram(self, memory_id: str) -> Optional[Engram]:
        """Get the engram for a memory."""
        engram_id = self._memory_to_engram.get(memory_id)
        if engram_id:
            return self._engrams.get(engram_id)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engram statistics."""
        state_counts = {}
        for state in EngramState:
            state_counts[state.value] = sum(
                1 for e in self._engrams.values() if e.state == state
            )
        
        avg_strength = 0.0
        if self._engrams:
            avg_strength = sum(e.strength for e in self._engrams.values()) / len(self._engrams)
        
        avg_excitability = 0.0
        if self._cells:
            avg_excitability = sum(c.excitability for c in self._cells.values()) / len(self._cells)
        
        return {
            "total_engrams": len(self._engrams),
            "total_cells": len(self._cells),
            "average_strength": round(avg_strength, 3),
            "average_excitability": round(avg_excitability, 3),
            "state_distribution": state_counts,
            "consolidated_count": sum(1 for e in self._engrams.values() if e.consolidated),
        }
