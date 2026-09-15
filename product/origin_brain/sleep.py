"""
Sleep Engine — Multi-Phase Offline Consolidation

Implements biologically-grounded sleep consolidation with multiple phases
that mirror the brain's SO-spindle-ripple coupling:

    Phase 1 (SWS): Replay and transfer — high-importance episodic memories
                   are replayed and compressed into semantic knowledge
    Phase 2 (REM): Schema integration — new semantic knowledge integrated
                   with existing schemas, emotional charge decoupled
    Phase 3 (Pruning): Active forgetting — low-salience, rarely accessed
                       memories are decayed or evicted

Brain analogue: NREM Stage 3 (SWS), REM sleep, sleep spindles
Neuroscience: SO-spindle-ripple coupling (Doc 26 §2), prioritized replay

Mathematical foundation (Doc 25 §5):
    Information Bottleneck: min I(X;T) - β I(T;Y)
    Compression ratio determines how much detail is preserved
    
    Optimal forgetting (Doc 25 §7):
    P(need) ∝ t^(-d) — power-law decay scoring
"""

from __future__ import annotations

import math
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .models import (
    EpisodicMemory, SemanticMemory, MemoryStatus, MemoryType
)

logger = logging.getLogger(__name__)


class SleepPhaseReport(BaseModel):
    """Report from a single sleep phase."""
    phase: str
    duration_ms: float = 0.0
    items_processed: int = 0
    items_consolidated: int = 0
    items_evicted: int = 0
    details: Dict[str, Any] = Field(default_factory=dict)


class SleepReport(BaseModel):
    """Full report from a sleep cycle."""
    cycle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_duration_ms: float = 0.0
    phases: List[SleepPhaseReport] = Field(default_factory=list)
    total_consolidated: int = 0
    total_evicted: int = 0
    new_semantic_count: int = 0
    compression_ratio: float = 0.0


class SleepConfig(BaseModel):
    """Configuration for the Sleep Engine."""
    # Phase 1: SWS Replay
    replay_priority_by_salience: bool = True    # Prioritize high-salience for replay
    replay_priority_by_surprise: bool = True    # Prioritize high-surprise for replay
    replay_priority_by_recency: bool = True     # Prioritize recent for replay
    max_replay_per_cycle: int = 100             # Max memories to replay per cycle
    min_age_for_consolidation_hours: float = 1.0  # Minimum age before consolidation
    consolidation_salience_threshold: float = 0.4  # Min salience to consolidate
    
    # Phase 2: REM Integration
    schema_integration_enabled: bool = True
    emotional_decoupling_rate: float = 0.3      # How much to reduce emotional charge
    
    # Phase 3: Pruning
    decay_exponent: float = 0.5                 # d in P(need) ∝ t^(-d)
    eviction_threshold: float = 0.05            # Below this need score → evict
    max_evictions_per_cycle: int = 200           # Safety limit on evictions
    
    # General
    num_cycles: int = 3                         # Number of sleep cycles to run


class SleepEngine:
    """
    Multi-phase sleep consolidation engine.
    
    Mirrors the brain's three sleep functions:
    1. SWS (Slow-Wave Sleep): Replay important memories, compress to semantic gist
    2. REM: Integrate with schemas, decouple emotional charge
    3. Pruning: Active forgetting via power-law need scoring
    
    Usage:
        engine = SleepEngine(config=SleepConfig())
        
        # Run full sleep cycle
        report = engine.run_sleep_cycle(
            episodic_store=memories,
            semantic_store=facts,
            store_semantic_fn=store_fn,
            evict_fn=evict_fn
        )
    """

    def __init__(self, config: Optional[SleepConfig] = None):
        self.config = config or SleepConfig()
        self._cycle_count: int = 0
        self._total_consolidated: int = 0
        self._total_evicted: int = 0
        
        logger.info("SleepEngine initialized")

    def _compute_replay_priority(self, memory: EpisodicMemory, now: datetime) -> float:
        """
        Compute replay priority for a memory.
        
        The brain prioritizes replay of:
        1. High-salience events (emotional significance)
        2. High-surprise events (prediction error)
        3. Recent events (temporal proximity)
        
        This mirrors dopaminergic replay prioritization in the hippocampus.
        """
        priority = 0.0
        
        if self.config.replay_priority_by_salience:
            priority += memory.salience * 0.4
        
        if self.config.replay_priority_by_surprise:
            # Use emotional arousal as a proxy for surprise
            surprise = memory.metadata.get('emotional_arousal', 0.5)
            priority += surprise * 0.35
        
        if self.config.replay_priority_by_recency:
            if memory.timestamp.tzinfo is None:
                age_hours = (now.replace(tzinfo=None) - memory.timestamp).total_seconds() / 3600
            else:
                age_hours = (now - memory.timestamp).total_seconds() / 3600
            # Recency bonus: decays over days
            recency = 1.0 / (1.0 + age_hours / 24.0)
            priority += recency * 0.25
        
        return min(1.0, priority)

    def _compress_to_semantic(self, memory: EpisodicMemory) -> SemanticMemory:
        """
        Compress an episodic memory into semantic gist.
        
        This is the Information Bottleneck in action:
        - Preserve the MEANING (what happened, key facts)
        - Discard the DETAILS (exact wording, minor context)
        - Track provenance (which episodes contributed)
        
        In the brain: hippocampal trace → neocortical representation
        """
        # Extract key content (simplified — in production, would use LLM)
        content = memory.content
        
        # Generate a key from the content
        words = content.lower().split()
        # Use most significant words as key
        key_words = [w for w in words if len(w) > 3][:5]
        key = "_".join(key_words) if key_words else f"memory_{memory.id[:8]}"
        
        return SemanticMemory(
            key=key,
            value=content,
            confidence=min(1.0, memory.salience + 0.2),  # Consolidation boosts confidence
            source_episodes=[memory.id],
            category=memory.memory_type.value.lower(),
            metadata={
                "consolidated_from": memory.id,
                "original_timestamp": memory.timestamp.isoformat(),
                "consolidation_time": datetime.now(timezone.utc).isoformat(),
                "original_salience": memory.salience,
                "compression_type": "gist_extraction"
            }
        )

    def _compute_need_score(self, memory: EpisodicMemory, now: datetime) -> float:
        """
        Compute P(need) using Anderson & Schooler (1991) power-law model.
        
        P(need | history) = Σ (t_now - t_access)^(-d)
        
        Memories that were recently and frequently accessed have high need.
        Memories that haven't been accessed in a long time have low need.
        """
        d = self.config.decay_exponent
        
        # Time since last access
        if memory.last_accessed.tzinfo is None:
            t_since_access = max(1.0, (now.replace(tzinfo=None) - memory.last_accessed).total_seconds() / 3600)
        else:
            t_since_access = max(1.0, (now - memory.last_accessed).total_seconds() / 3600)
        
        # Base need from recency
        recency_need = t_since_access ** (-d)
        
        # Frequency bonus (more accesses = higher need)
        frequency_bonus = math.log(1.0 + memory.access_count) * 0.1
        
        # Salience bonus (important memories are always needed)
        salience_bonus = memory.salience * 0.3
        
        return recency_need + frequency_bonus + salience_bonus

    def _phase_sws(
        self,
        episodic_memories: List[EpisodicMemory],
        store_semantic_fn: Callable[[SemanticMemory], None],
        now: datetime
    ) -> SleepPhaseReport:
        """
        Phase 1: Slow-Wave Sleep — Replay and Consolidation
        
        Replays high-priority memories and compresses them into semantic gist.
        Mirrors the SO-spindle-ripple temporal coupling:
        - Cortex provides the timing frame (SO up-state)
        - Thalamus triggers plasticity (spindle)
        - Hippocampus delivers the memory (ripple)
        """
        import time
        start = time.time()
        
        # Filter eligible memories (must be old enough)
        min_age = timedelta(hours=self.config.min_age_for_consolidation_hours)
        eligible = [
            m for m in episodic_memories
            if m.status == MemoryStatus.ACTIVE
            and (now - (m.timestamp if m.timestamp.tzinfo else m.timestamp.replace(tzinfo=timezone.utc))) >= min_age
            and m.salience >= self.config.consolidation_salience_threshold
        ]
        
        # Compute replay priorities
        prioritized = [
            (self._compute_replay_priority(m, now), m)
            for m in eligible
        ]
        prioritized.sort(key=lambda x: x[0], reverse=True)
        
        # Replay top-priority memories
        consolidated = 0
        to_replay = prioritized[:self.config.max_replay_per_cycle]
        
        for priority, memory in to_replay:
            # Compress to semantic gist
            semantic = self._compress_to_semantic(memory)
            store_semantic_fn(semantic)
            
            # Mark the episodic memory as consolidating
            memory.status = MemoryStatus.CONSOLIDATING
            
            # Increase stability (consolidated memories are more robust)
            memory.stability *= 1.5
            
            consolidated += 1

        elapsed = (time.time() - start) * 1000
        
        return SleepPhaseReport(
            phase="SWS",
            duration_ms=elapsed,
            items_processed=len(to_replay),
            items_consolidated=consolidated,
            details={
                "eligible_count": len(eligible),
                "replayed_count": len(to_replay),
                "avg_priority": sum(p for p, _ in to_replay) / max(len(to_replay), 1)
            }
        )

    def _phase_rem(
        self,
        episodic_memories: List[EpisodicMemory],
        now: datetime
    ) -> SleepPhaseReport:
        """
        Phase 2: REM Sleep — Emotional Decoupling and Schema Integration
        
        During REM sleep, the brain:
        1. Consolidates the informational CORE of emotional memories
        2. STRIPS the visceral emotional charge (Walker's hypothesis)
        3. Integrates new knowledge into existing schemas
        
        Failure of this process → PTSD (emotion never decoupled from memory)
        """
        import time
        start = time.time()
        
        processed = 0
        
        for memory in episodic_memories:
            if memory.status not in (MemoryStatus.ACTIVE, MemoryStatus.CONSOLIDATING):
                continue
            
            # Emotional decoupling: reduce arousal while preserving content
            current_arousal = memory.metadata.get('emotional_arousal', 0.5)
            if current_arousal > 0.6:  # Only decouple highly aroused memories
                decoupled_arousal = current_arousal * (1.0 - self.config.emotional_decoupling_rate)
                memory.metadata['emotional_arousal'] = decoupled_arousal
                memory.metadata['rem_processed'] = True
                processed += 1
        
        elapsed = (time.time() - start) * 1000
        
        return SleepPhaseReport(
            phase="REM",
            duration_ms=elapsed,
            items_processed=processed,
            details={
                "emotional_decoupling_rate": self.config.emotional_decoupling_rate,
                "memories_decoupled": processed
            }
        )

    def _phase_pruning(
        self,
        episodic_memories: List[EpisodicMemory],
        evict_fn: Callable[[str], bool],
        now: datetime
    ) -> SleepPhaseReport:
        """
        Phase 3: Active Forgetting (Pruning)
        
        Implements Anderson & Schooler optimal forgetting:
        P(need) ∝ t^(-d) — power-law decay scoring
        
        Memories below the need threshold are evicted, freeing
        computational resources and preventing the pathological
        effects of infinite retention (hyperthymesia).
        """
        import time
        start = time.time()
        
        # Score all active memories
        scored = [
            (self._compute_need_score(m, now), m)
            for m in episodic_memories
            if m.status in (MemoryStatus.ACTIVE, MemoryStatus.CONSOLIDATING)
        ]
        scored.sort(key=lambda x: x[0])
        
        evicted = 0
        for need_score, memory in scored:
            if evicted >= self.config.max_evictions_per_cycle:
                break
            if need_score < self.config.eviction_threshold:
                if evict_fn(memory.id):
                    evicted += 1
        
        elapsed = (time.time() - start) * 1000
        
        return SleepPhaseReport(
            phase="PRUNING",
            duration_ms=elapsed,
            items_processed=len(scored),
            items_evicted=evicted,
            details={
                "eviction_threshold": self.config.eviction_threshold,
                "lowest_need_score": scored[0][0] if scored else 0.0,
                "highest_need_score": scored[-1][0] if scored else 0.0,
            }
        )

    def run_sleep_cycle(
        self,
        episodic_memories: List[EpisodicMemory],
        store_semantic_fn: Callable[[SemanticMemory], None],
        evict_fn: Callable[[str], bool],
        num_cycles: Optional[int] = None
    ) -> SleepReport:
        """
        Run a full multi-phase sleep cycle.
        
        Each cycle runs 3 phases in order:
        1. SWS: Replay and consolidate important memories
        2. REM: Decouple emotions, integrate schemas
        3. PRUNING: Evict low-need memories
        
        Multiple cycles can be run to deepen consolidation (mirrors
        the 4-5 sleep cycles in a full night of human sleep).
        
        Args:
            episodic_memories: List of current episodic memories
            store_semantic_fn: Function to store a new semantic memory
            evict_fn: Function to evict a memory by ID (returns success)
            num_cycles: Override number of cycles (default: config.num_cycles)
            
        Returns:
            SleepReport with details of all phases
        """
        import time
        overall_start = time.time()
        now = datetime.now(timezone.utc)
        
        cycles = num_cycles or self.config.num_cycles
        all_phases: List[SleepPhaseReport] = []
        total_consolidated = 0
        total_evicted = 0
        total_semantic = 0
        
        for cycle_num in range(cycles):
            # Phase 1: SWS
            sws_report = self._phase_sws(episodic_memories, store_semantic_fn, now)
            all_phases.append(sws_report)
            total_consolidated += sws_report.items_consolidated
            total_semantic += sws_report.items_consolidated
            
            # Phase 2: REM
            rem_report = self._phase_rem(episodic_memories, now)
            all_phases.append(rem_report)
            
            # Phase 3: Pruning
            prune_report = self._phase_pruning(episodic_memories, evict_fn, now)
            all_phases.append(prune_report)
            total_evicted += prune_report.items_evicted
            
            logger.info(
                f"Sleep cycle {cycle_num + 1}/{cycles}: "
                f"consolidated={sws_report.items_consolidated}, "
                f"rem_processed={rem_report.items_processed}, "
                f"evicted={prune_report.items_evicted}"
            )
        
        total_elapsed = (time.time() - overall_start) * 1000
        self._cycle_count += 1
        self._total_consolidated += total_consolidated
        self._total_evicted += total_evicted
        
        # Compression ratio: how much we reduced
        initial_count = len(episodic_memories)
        compression_ratio = total_evicted / max(initial_count, 1)
        
        return SleepReport(
            total_duration_ms=total_elapsed,
            phases=all_phases,
            total_consolidated=total_consolidated,
            total_evicted=total_evicted,
            new_semantic_count=total_semantic,
            compression_ratio=compression_ratio
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_cycles": self._cycle_count,
            "total_consolidated": self._total_consolidated,
            "total_evicted": self._total_evicted,
            "config": self.config.model_dump(),
        }
