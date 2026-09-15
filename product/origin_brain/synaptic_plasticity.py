"""
Synaptic Plasticity Engine — STDP-Inspired Learning Rules

Implements the microscopic learning rules that govern HOW memory
connections strengthen and weaken:

1. **Hebbian Learning** — "Neurons that fire together wire together"
   - Co-activation strengthens connections
   - Basis: Hebb 1949

2. **STDP Abstraction** — Timing-dependent updates
   - Pre-before-post → strengthen (LTP analogue)
   - Post-before-pre → weaken (LTD analogue)
   - Basis: Bi & Poo 1998, Markram 1997

3. **Synaptic Tagging** — Two-phase consolidation
   - Weak stimulation → temporary tag (hours)
   - Strong stimulation → permanent change (requires protein synthesis analogue)
   - Basis: Frey & Morris 1997

4. **Metaplasticity** — The plasticity of plasticity
   - Prior activity history changes how plastic a synapse is
   - Basis: BCM theory (Bienenstock, Cooper, Munro 1982)

Brain analogue: Glutamatergic synapses in hippocampal CA1/CA3
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SynapticTag(str, Enum):
    """Tags on synapses marking them for potential consolidation."""
    NONE = "none"           # No tag
    EARLY_LTP = "early_ltp"  # Temporary strengthening (1-3 hours)
    LATE_LTP = "late_ltp"    # Permanent strengthening (protein synthesis)
    EARLY_LTD = "early_ltd"  # Temporary weakening
    LATE_LTD = "late_ltd"    # Permanent weakening


class Synapse(BaseModel):
    """
    A connection between two memory elements.
    
    Synapses have:
    - Weight: connection strength (0.0 to 1.0)
    - Tag: marking for consolidation
    - History: activation trace for STDP
    """
    id: str = Field(default_factory=lambda: f"syn_{uuid.uuid4().hex[:10]}")
    source_id: str          # Pre-synaptic memory/neuron
    target_id: str          # Post-synaptic memory/neuron
    weight: float = 0.5     # Connection strength
    tag: SynapticTag = SynapticTag.NONE
    tag_timestamp: Optional[datetime] = None
    creation_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    activation_count: int = 0
    
    # Metaplasticity: sliding threshold for LTP/LTD
    theta_m: float = 0.5   # BCM sliding threshold


class PlasticityConfig(BaseModel):
    """Configuration for synaptic plasticity rules."""
    # STDP parameters (abstracted to memory-level timing)
    a_plus: float = 0.05        # LTP magnitude
    a_minus: float = 0.06       # LTD magnitude (slightly larger → forgetting bias)
    tau_plus: float = 3600.0     # LTP time constant (seconds) — abstracted from 20ms
    tau_minus: float = 3600.0    # LTD time constant (seconds)
    
    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0
    
    # Synaptic tagging
    tag_duration_hours: float = 3.0   # Early tags last ~3 hours
    consolidation_threshold: float = 0.7  # Weight needed for late-LTP
    
    # Metaplasticity (BCM)
    bcm_learning_rate: float = 0.01
    bcm_target_rate: float = 0.3    # Target average activation rate


class PlasticityEvent(BaseModel):
    """Records a plasticity event."""
    synapse_id: str
    event_type: str  # "ltp", "ltd", "tag", "consolidate", "prune"
    old_weight: float
    new_weight: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SynapticPlasticityEngine:
    """
    Manages the learning rules for memory connections.
    
    This engine operates on the CONNECTIONS between memories,
    not the memories themselves. It determines:
    - Which associations get stronger (co-retrieval → LTP)
    - Which associations get weaker (interference → LTD)
    - Which connections are consolidated (tagged → permanent)
    
    Usage:
        engine = SynapticPlasticityEngine()
        
        # Create a connection
        syn = engine.create_synapse("mem_1", "mem_2")
        
        # Co-activation strengthens (Hebbian)
        engine.hebbian_update(syn.id, co_activation=0.8)
        
        # STDP-like update based on retrieval order
        engine.stdp_update(syn.id, dt_seconds=5.0)
        
        # Run tag maintenance
        engine.maintain_tags()
    """
    
    def __init__(self, config: Optional[PlasticityConfig] = None):
        self.config = config or PlasticityConfig()
        self._synapses: Dict[str, Synapse] = {}
        self._events: List[PlasticityEvent] = []
        self._total_ltp: int = 0
        self._total_ltd: int = 0
        
        logger.info("SynapticPlasticityEngine initialized")
    
    def create_synapse(
        self, source_id: str, target_id: str, 
        initial_weight: float = 0.5
    ) -> Synapse:
        """Create a new synapse between two memory elements."""
        synapse = Synapse(
            source_id=source_id,
            target_id=target_id,
            weight=initial_weight,
        )
        self._synapses[synapse.id] = synapse
        return synapse
    
    def get_synapse(self, synapse_id: str) -> Optional[Synapse]:
        """Get a synapse by ID."""
        return self._synapses.get(synapse_id)
    
    def find_synapses(
        self, source_id: Optional[str] = None, 
        target_id: Optional[str] = None
    ) -> List[Synapse]:
        """Find synapses by source and/or target."""
        results = []
        for syn in self._synapses.values():
            if source_id and syn.source_id != source_id:
                continue
            if target_id and syn.target_id != target_id:
                continue
            results.append(syn)
        return results
    
    def hebbian_update(self, synapse_id: str, co_activation: float) -> float:
        """
        Hebbian learning: strengthen connections between co-active elements.
        
        "Neurons that fire together wire together"
        
        Args:
            synapse_id: ID of synapse to update
            co_activation: How strongly both elements were active (0-1)
            
        Returns:
            New weight
        """
        syn = self._synapses.get(synapse_id)
        if syn is None:
            return 0.0
        
        old_weight = syn.weight
        
        # Hebbian: dw = eta * pre * post
        # Abstracted: co_activation represents pre*post
        dw = self.config.a_plus * co_activation
        
        # Apply BCM metaplasticity: above threshold → LTP, below → LTD
        if co_activation > syn.theta_m:
            # LTP
            syn.weight = min(self.config.w_max, syn.weight + dw)
            self._total_ltp += 1
        else:
            # LTD (weakly active → weaken)
            ltd_amount = self.config.a_minus * (syn.theta_m - co_activation)
            syn.weight = max(self.config.w_min, syn.weight - ltd_amount)
            self._total_ltd += 1
        
        # Update sliding threshold (BCM metaplasticity)
        syn.theta_m += self.config.bcm_learning_rate * (
            co_activation - self.config.bcm_target_rate
        )
        syn.theta_m = max(0.1, min(0.9, syn.theta_m))
        
        syn.activation_count += 1
        syn.last_activated = datetime.now(timezone.utc)
        
        self._events.append(PlasticityEvent(
            synapse_id=synapse_id,
            event_type="ltp" if syn.weight > old_weight else "ltd",
            old_weight=old_weight,
            new_weight=syn.weight,
        ))
        
        return syn.weight
    
    def stdp_update(self, synapse_id: str, dt_seconds: float) -> float:
        """
        STDP-inspired update based on temporal ordering.
        
        Abstracted from millisecond spike timing to second/minute
        retrieval ordering.
        
        Args:
            synapse_id: ID of synapse
            dt_seconds: Time difference (post - pre). 
                       Positive = source recalled before target → LTP
                       Negative = target recalled before source → LTD
                       
        Returns:
            New weight
        """
        syn = self._synapses.get(synapse_id)
        if syn is None:
            return 0.0
        
        old_weight = syn.weight
        
        if dt_seconds > 0:
            # Pre before post → LTP (causal, strengthen)
            dw = self.config.a_plus * math.exp(-dt_seconds / self.config.tau_plus)
            syn.weight = min(self.config.w_max, syn.weight + dw)
            self._total_ltp += 1
        elif dt_seconds < 0:
            # Post before pre → LTD (anti-causal, weaken)
            dw = self.config.a_minus * math.exp(dt_seconds / self.config.tau_minus)
            syn.weight = max(self.config.w_min, syn.weight - dw)
            self._total_ltd += 1
        
        syn.activation_count += 1
        syn.last_activated = datetime.now(timezone.utc)
        
        self._events.append(PlasticityEvent(
            synapse_id=synapse_id,
            event_type="ltp" if syn.weight > old_weight else "ltd",
            old_weight=old_weight,
            new_weight=syn.weight,
        ))
        
        return syn.weight
    
    def apply_tag(self, synapse_id: str) -> Optional[SynapticTag]:
        """
        Apply a synaptic tag based on current weight.
        
        Weak changes → early tag (temporary, decays in hours)
        Strong changes → late tag (permanent if consolidated)
        
        Returns:
            The tag applied, or None
        """
        syn = self._synapses.get(synapse_id)
        if syn is None:
            return None
        
        now = datetime.now(timezone.utc)
        
        if syn.weight >= self.config.consolidation_threshold:
            syn.tag = SynapticTag.LATE_LTP
        elif syn.weight > 0.5:
            syn.tag = SynapticTag.EARLY_LTP
        elif syn.weight < 0.3:
            syn.tag = SynapticTag.EARLY_LTD
        else:
            syn.tag = SynapticTag.NONE
        
        syn.tag_timestamp = now
        return syn.tag
    
    def maintain_tags(self) -> int:
        """
        Maintain synaptic tags — expire temporary ones.
        
        Early tags decay after tag_duration_hours if not consolidated.
        
        Returns:
            Number of tags that expired
        """
        now = datetime.now(timezone.utc)
        expired = 0
        
        for syn in self._synapses.values():
            if syn.tag in (SynapticTag.EARLY_LTP, SynapticTag.EARLY_LTD):
                if syn.tag_timestamp:
                    hours_elapsed = (now - syn.tag_timestamp).total_seconds() / 3600
                    if hours_elapsed > self.config.tag_duration_hours:
                        # Tag expired — revert toward baseline
                        if syn.tag == SynapticTag.EARLY_LTP:
                            syn.weight = max(0.5, syn.weight - 0.1)
                        elif syn.tag == SynapticTag.EARLY_LTD:
                            syn.weight = min(0.5, syn.weight + 0.1)
                        syn.tag = SynapticTag.NONE
                        syn.tag_timestamp = None
                        expired += 1
        
        return expired
    
    def prune_weak_synapses(self, threshold: float = 0.05) -> List[Synapse]:
        """Remove synapses with very low weights."""
        pruned = []
        for syn_id in list(self._synapses.keys()):
            syn = self._synapses[syn_id]
            if syn.weight <= threshold:
                pruned.append(syn)
                del self._synapses[syn_id]
        return pruned
    
    def get_connection_strength(self, source_id: str, target_id: str) -> float:
        """Get the total connection strength between two memory elements."""
        synapses = self.find_synapses(source_id=source_id, target_id=target_id)
        if not synapses:
            return 0.0
        return sum(s.weight for s in synapses) / len(synapses)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plasticity statistics."""
        avg_weight = 0.0
        if self._synapses:
            avg_weight = sum(s.weight for s in self._synapses.values()) / len(self._synapses)
        
        tag_counts = {}
        for tag in SynapticTag:
            tag_counts[tag.value] = sum(
                1 for s in self._synapses.values() if s.tag == tag
            )
        
        return {
            "total_synapses": len(self._synapses),
            "average_weight": round(avg_weight, 3),
            "total_ltp_events": self._total_ltp,
            "total_ltd_events": self._total_ltd,
            "ltp_ltd_ratio": round(self._total_ltp / max(self._total_ltd, 1), 2),
            "tags": tag_counts,
            "total_events": len(self._events),
        }
