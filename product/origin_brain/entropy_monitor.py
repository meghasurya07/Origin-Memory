"""
Attention Entropy Monitor — Intelligent Consolidation Triggers

Implements SleepGate-inspired attention entropy monitoring to detect when
the memory system needs consolidation. Instead of fixed-interval sleep
cycles, consolidation is triggered dynamically based on the information
theoretic state of the memory system.

Key insight: When many memories have similar relevance scores (high entropy),
the system is "confused" — it's time to consolidate, deduplicate, and
strengthen important memories while pruning noise.

Metrics monitored:
1. MEMORY ENTROPY: Shannon entropy over memory access patterns
   - High entropy = memories equally accessed = system lacks focus
   - Low entropy = clear priority hierarchy = system is organized

2. INTERFERENCE RATE: How often new memories conflict with existing ones
   - High rate = information overload = needs sleep consolidation

3. WORKING MEMORY CHURN: How fast items are displaced from working memory
   - High churn = too much information flowing in = needs consolidation

4. SCHEMA SATURATION: How well existing schemas cover incoming information
   - Low coverage = novel information = needs new schema formation

Triggers:
- When entropy exceeds threshold → trigger SWS consolidation
- When interference rate spikes → trigger conflict resolution
- When working memory churns too fast → trigger importance-based pruning

Brain analogue:
  - Entropy monitoring = locus coeruleus norepinephrine system
  - Consolidation triggers = adenosine accumulation → sleep pressure
  - SleepGate = hypothalamic sleep-wake switch (VLPO)

Research basis:
  - SleepGate (2026) — Attention-entropy triggered consolidation
  - Google SLEEP Framework — Wake-phase + Sleep-phase continual learning
  - LightMem — Online/offline memory tier management
  - Allostatic regulation in neuroscience — maintaining homeostasis
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EntropyState(BaseModel):
    """Current entropy state of the memory system."""
    memory_entropy: float = 0.0          # Shannon entropy over access patterns
    interference_rate: float = 0.0       # Conflicts per encode in recent window
    working_memory_churn: float = 0.0    # Displacement rate
    schema_coverage: float = 1.0         # How well schemas cover new info
    sleep_pressure: float = 0.0          # Accumulated consolidation pressure [0, 1]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def needs_consolidation(self) -> bool:
        """Whether consolidation should be triggered."""
        return self.sleep_pressure > 0.7

    @property
    def urgency(self) -> str:
        """How urgently consolidation is needed."""
        if self.sleep_pressure > 0.9:
            return "critical"
        elif self.sleep_pressure > 0.7:
            return "high"
        elif self.sleep_pressure > 0.4:
            return "moderate"
        else:
            return "low"


class ConsolidationTrigger(BaseModel):
    """A consolidation trigger event."""
    reason: str
    pressure_contribution: float
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EntropyConfig(BaseModel):
    """Configuration for the entropy monitor."""
    # Entropy thresholds
    entropy_trigger_threshold: float = 0.8   # Above this, high entropy = needs consolidation
    entropy_optimal_range: Tuple[float, float] = (0.3, 0.6)  # Healthy entropy range

    # Interference thresholds
    interference_rate_threshold: float = 0.3  # > 30% of encodes cause interference
    interference_window: int = 20             # Look at last N encodes

    # Working memory thresholds
    churn_rate_threshold: float = 0.5         # > 50% displacement rate per encode
    churn_window: int = 10                    # Look at last N working memory operations

    # Sleep pressure
    pressure_decay_rate: float = 0.1          # How fast pressure decays after consolidation
    pressure_accumulation_rate: float = 0.05  # Base pressure per encode
    max_pressure: float = 1.0

    # Trigger weights
    entropy_weight: float = 0.4              # How much entropy contributes to pressure
    interference_weight: float = 0.3          # How much interference contributes
    churn_weight: float = 0.2                 # How much churn contributes
    schema_weight: float = 0.1                # How much schema coverage contributes


class AttentionEntropyMonitor:
    """
    Monitors the information-theoretic state of the memory system
    and triggers consolidation when entropy exceeds healthy bounds.

    Usage:
        monitor = AttentionEntropyMonitor()

        # During encode:
        monitor.record_encode(had_interference=False, wm_displacement=True)

        # Check if consolidation needed:
        state = monitor.get_state()
        if state.needs_consolidation:
            brain.sleep()
            monitor.on_consolidation_complete()
    """

    def __init__(self, config: Optional[EntropyConfig] = None):
        self.config = config or EntropyConfig()

        # Tracking windows
        self._encode_count: int = 0
        self._interference_history: deque = deque(maxlen=self.config.interference_window)
        self._churn_history: deque = deque(maxlen=self.config.churn_window)
        self._access_counts: Dict[str, int] = {}  # memory_id -> access count
        self._schema_matches: deque = deque(maxlen=50)  # Recent schema match rates

        # Accumulated pressure
        self._sleep_pressure: float = 0.0
        self._last_consolidation: Optional[datetime] = None
        self._trigger_log: List[ConsolidationTrigger] = []

        logger.info("AttentionEntropyMonitor initialized")

    def record_encode(
        self,
        memory_id: str = "",
        had_interference: bool = False,
        wm_displacement: bool = False,
        schema_matched: bool = True,
    ) -> EntropyState:
        """
        Record an encode event and update entropy metrics.

        Returns the current entropy state.
        """
        self._encode_count += 1
        self._interference_history.append(1 if had_interference else 0)
        self._churn_history.append(1 if wm_displacement else 0)
        self._schema_matches.append(1 if schema_matched else 0)

        if memory_id:
            self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1

        # Accumulate base pressure
        self._sleep_pressure = min(
            self.config.max_pressure,
            self._sleep_pressure + self.config.pressure_accumulation_rate,
        )

        # Compute current state
        state = self._compute_state()

        # Check for trigger conditions
        self._check_triggers(state)

        return state

    def record_retrieval(self, memory_id: str) -> None:
        """Record a retrieval event to update access distribution."""
        self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1

    def on_consolidation_complete(self) -> None:
        """Reset pressure after consolidation."""
        self._sleep_pressure *= (1.0 - self.config.pressure_decay_rate * 5)
        self._sleep_pressure = max(0.0, self._sleep_pressure - 0.3)
        self._last_consolidation = datetime.now(timezone.utc)
        logger.info(f"Post-consolidation pressure: {self._sleep_pressure:.2f}")

    def get_state(self) -> EntropyState:
        """Get the current entropy state."""
        return self._compute_state()

    def get_trigger_log(self) -> List[ConsolidationTrigger]:
        """Get the log of all consolidation triggers."""
        return list(self._trigger_log)

    # ── Internal Computation ────────────────────────────────────────────

    def _compute_state(self) -> EntropyState:
        """Compute the current entropy state of the memory system."""
        memory_entropy = self._compute_memory_entropy()
        interference_rate = self._compute_interference_rate()
        churn_rate = self._compute_churn_rate()
        schema_coverage = self._compute_schema_coverage()

        return EntropyState(
            memory_entropy=memory_entropy,
            interference_rate=interference_rate,
            working_memory_churn=churn_rate,
            schema_coverage=schema_coverage,
            sleep_pressure=self._sleep_pressure,
        )

    def _compute_memory_entropy(self) -> float:
        """
        Compute Shannon entropy over memory access distribution.

        H = -sum(p_i * log2(p_i)) / log2(n)

        Normalized to [0, 1] where:
        - 0 = one memory dominates all access (perfect organization)
        - 1 = all memories accessed equally (maximum confusion)
        """
        if len(self._access_counts) <= 1:
            return 0.0

        total = sum(self._access_counts.values())
        if total == 0:
            return 0.0

        probs = [count / total for count in self._access_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        max_entropy = math.log2(len(self._access_counts))
        if max_entropy == 0:
            return 0.0

        return entropy / max_entropy

    def _compute_interference_rate(self) -> float:
        """Compute the rate of interference events in the recent window."""
        if not self._interference_history:
            return 0.0
        return sum(self._interference_history) / len(self._interference_history)

    def _compute_churn_rate(self) -> float:
        """Compute the working memory displacement rate."""
        if not self._churn_history:
            return 0.0
        return sum(self._churn_history) / len(self._churn_history)

    def _compute_schema_coverage(self) -> float:
        """Compute how well existing schemas cover incoming information."""
        if not self._schema_matches:
            return 1.0  # No data = assume good coverage
        return sum(self._schema_matches) / len(self._schema_matches)

    def _check_triggers(self, state: EntropyState) -> None:
        """Check if any metrics exceed trigger thresholds."""
        # Entropy trigger
        if state.memory_entropy > self.config.entropy_trigger_threshold:
            contribution = (
                (state.memory_entropy - self.config.entropy_trigger_threshold)
                * self.config.entropy_weight
            )
            self._sleep_pressure = min(
                self.config.max_pressure,
                self._sleep_pressure + contribution,
            )
            self._trigger_log.append(ConsolidationTrigger(
                reason="High memory entropy — memories accessed too uniformly",
                pressure_contribution=contribution,
                metric_name="memory_entropy",
                metric_value=state.memory_entropy,
                threshold=self.config.entropy_trigger_threshold,
            ))

        # Interference trigger
        if state.interference_rate > self.config.interference_rate_threshold:
            contribution = (
                (state.interference_rate - self.config.interference_rate_threshold)
                * self.config.interference_weight
            )
            self._sleep_pressure = min(
                self.config.max_pressure,
                self._sleep_pressure + contribution,
            )
            self._trigger_log.append(ConsolidationTrigger(
                reason="High interference rate — too many memory conflicts",
                pressure_contribution=contribution,
                metric_name="interference_rate",
                metric_value=state.interference_rate,
                threshold=self.config.interference_rate_threshold,
            ))

        # Churn trigger
        if state.working_memory_churn > self.config.churn_rate_threshold:
            contribution = (
                (state.working_memory_churn - self.config.churn_rate_threshold)
                * self.config.churn_weight
            )
            self._sleep_pressure = min(
                self.config.max_pressure,
                self._sleep_pressure + contribution,
            )
            self._trigger_log.append(ConsolidationTrigger(
                reason="High working memory churn — information overload",
                pressure_contribution=contribution,
                metric_name="working_memory_churn",
                metric_value=state.working_memory_churn,
                threshold=self.config.churn_rate_threshold,
            ))

        # Schema coverage trigger
        if state.schema_coverage < 0.5:
            contribution = (0.5 - state.schema_coverage) * self.config.schema_weight
            self._sleep_pressure = min(
                self.config.max_pressure,
                self._sleep_pressure + contribution,
            )
            self._trigger_log.append(ConsolidationTrigger(
                reason="Low schema coverage — novel information needs schema formation",
                pressure_contribution=contribution,
                metric_name="schema_coverage",
                metric_value=state.schema_coverage,
                threshold=0.5,
            ))

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        state = self._compute_state()
        return {
            "encode_count": self._encode_count,
            "unique_memories_tracked": len(self._access_counts),
            "memory_entropy": round(state.memory_entropy, 3),
            "interference_rate": round(state.interference_rate, 3),
            "working_memory_churn": round(state.working_memory_churn, 3),
            "schema_coverage": round(state.schema_coverage, 3),
            "sleep_pressure": round(self._sleep_pressure, 3),
            "urgency": state.urgency,
            "needs_consolidation": state.needs_consolidation,
            "trigger_count": len(self._trigger_log),
            "last_consolidation": self._last_consolidation.isoformat() if self._last_consolidation else None,
        }
