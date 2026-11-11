"""
RL-Gated Encoding Pipeline — Active Memory Decision Controller
══════════════════════════════════════════════════════════════

Wires the RL Memory Policy as an active gating mechanism in the Brain's
encode pipeline. Instead of always encoding everything that passes the
prediction surprise gate, the RL controller makes a secondary decision
about whether to ENCODE, UPDATE, SKIP, or trigger CONSOLIDATION.

This creates a two-stage gating system:
  1. Prediction Engine: Is this SURPRISING enough to consider? (biological)
  2. RL Controller: Should we STORE this? (learned from experience)

The deferred reward mechanism tracks which encoded memories are later
retrieved (and at what rank), feeding this back as reward signal to
improve future decisions.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Encode Pipeline                                     │
  │                                                       │
  │  Content → Prediction Gate → [PASS] → RL Gate         │
  │                                         │             │
  │                              ┌──────────┼──────────┐  │
  │                              │ ENCODE   │ UPDATE   │  │
  │                              │ SKIP     │ EVICT    │  │
  │                              │ CONSOLIDATE          │  │
  │                              └─────────────────────┘  │
  │                                         │             │
  │  Deferred Reward ←──── Recall Pipeline ──┘            │
  │  (was this memory retrieved? at what rank?)           │
  └─────────────────────────────────────────────────────┘

Brain analogue:
  - Two-stage gating mirrors:
    1. Locus coeruleus NE → novelty gate (prediction engine)
    2. VTA dopamine → value gate (RL controller)
  - Both gates must pass for memory to reach hippocampus
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .rl_policy import RLMemoryPolicy, MemoryAction, MemoryState, RewardSignal

logger = logging.getLogger(__name__)


class EncodingDecision(BaseModel):
    """Record of an RL encoding decision for deferred reward."""
    memory_id: Optional[str] = None    # ID of the encoded memory (if encoded)
    content_hash: int = 0              # Hash of the content for tracking
    action: MemoryAction = MemoryAction.ENCODE
    state: Optional[MemoryState] = None
    similarity_to_best: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reward_delivered: bool = False


class RLGateConfig(BaseModel):
    """Configuration for RL-gated encoding."""
    enabled: bool = True                    # Whether RL gating is active
    use_heuristic_until: int = 100          # Use heuristic for first N encodes
    track_decisions: int = 5000             # How many decisions to track for deferred reward
    retrieval_reward_decay: float = 0.95    # Discount for older decisions


class RLEncodingGate:
    """
    Active RL gating mechanism for the encode pipeline.

    Tracks encoding decisions and delivers deferred rewards when
    encoded memories are later retrieved.

    Usage in Brain.encode():
        decision = self.rl_gate.pre_encode(content, similarity, salience, entropy_state)
        if decision.action == MemoryAction.SKIP:
            return EncodeResult(action='skipped_by_rl')
        elif decision.action == MemoryAction.CONSOLIDATE:
            self.sleep()
            return EncodeResult(action='triggered_consolidation')
        # else: proceed with normal encoding

    Usage in Brain.recall():
        self.rl_gate.on_retrieval(memory_id, retrieval_rank)
    """

    def __init__(
        self,
        policy: RLMemoryPolicy,
        config: Optional[RLGateConfig] = None,
    ):
        self.policy = policy
        self.config = config or RLGateConfig()

        # Decision tracking for deferred reward
        self._decisions: deque[EncodingDecision] = deque(maxlen=self.config.track_decisions)
        self._memory_to_decision: Dict[str, EncodingDecision] = {}
        self._content_hash_to_decision: Dict[int, EncodingDecision] = {}

        # Statistics
        self._total_encodes: int = 0
        self._total_skips: int = 0
        self._total_updates: int = 0
        self._total_consolidations: int = 0
        self._total_evictions: int = 0
        self._rewards_delivered: int = 0

        logger.info("RLEncodingGate initialized")

    def pre_encode(
        self,
        content: str,
        similarity_to_best: float = 0.0,
        recency_of_similar: float = 0.0,
        salience: float = 0.3,
        entropy_state: Optional[Any] = None,
        memory_count: int = 0,
        wm_full: bool = False,
    ) -> EncodingDecision:
        """
        Make a pre-encoding decision using the RL policy.

        Called before hippocampal encoding to decide whether to proceed.
        """
        # Build observation
        state = self.policy.observe(
            content=content,
            entropy_state=entropy_state,
            similarity_to_best=similarity_to_best,
            recency_of_similar=recency_of_similar,
            salience=salience,
            memory_count=memory_count,
            wm_full=wm_full,
        )

        # Choose action
        if not self.config.enabled:
            action = MemoryAction.ENCODE
        else:
            action = self.policy.decide_adaptive(state)

        # Create decision record
        decision = EncodingDecision(
            content_hash=hash(content),
            action=action,
            state=state,
            similarity_to_best=similarity_to_best,
        )

        # Track by content hash for skip decisions
        self._content_hash_to_decision[decision.content_hash] = decision
        self._decisions.append(decision)

        # Update counters
        if action == MemoryAction.ENCODE:
            self._total_encodes += 1
        elif action == MemoryAction.SKIP:
            self._total_skips += 1
        elif action == MemoryAction.UPDATE:
            self._total_updates += 1
        elif action == MemoryAction.CONSOLIDATE:
            self._total_consolidations += 1
        elif action == MemoryAction.EVICT:
            self._total_evictions += 1

        return decision

    def post_encode(self, decision: EncodingDecision, memory_id: str) -> None:
        """
        Record the memory ID after successful encoding.

        Called after hippocampal encoding to link the decision to the memory.
        """
        decision.memory_id = memory_id
        self._memory_to_decision[memory_id] = decision

    def on_retrieval(
        self,
        memory_id: str,
        retrieval_rank: int = 0,
        confidence: float = 0.5,
    ) -> Optional[float]:
        """
        Deliver deferred reward when an encoded memory is retrieved.

        If this memory was encoded by the RL controller, we feed back the
        retrieval quality as a reward signal.

        Returns the TD error (or None if no decision was tracked).
        """
        decision = self._memory_to_decision.get(memory_id)
        if not decision or decision.reward_delivered:
            return None

        # Compute reward
        reward = self.policy.compute_reward(
            action=decision.action,
            was_retrieved=True,
            retrieval_rank=retrieval_rank,
            confidence_delta=confidence - 0.5,  # Improvement over baseline
        )

        # Build next state (current system state at retrieval time)
        next_state = decision.state or MemoryState()

        # Learn
        td_error = self.policy.learn_from_pending(reward, next_state)
        if td_error is None and decision.state:
            td_error = self.policy.learn(decision.state, decision.action, reward, next_state)

        decision.reward_delivered = True
        self._rewards_delivered += 1

        logger.debug(
            f"RL deferred reward: memory={memory_id} rank={retrieval_rank} "
            f"reward={reward.total:.3f} td_error={td_error}"
        )
        return td_error

    def on_skip_verified(self, content_hash: int, was_needed: bool) -> Optional[float]:
        """
        Verify a SKIP decision: was the skipped content later needed?

        If the content was later queried and not found, the SKIP was wrong.
        If it was never needed, the SKIP was correct.
        """
        decision = self._content_hash_to_decision.get(content_hash)
        if not decision or decision.action != MemoryAction.SKIP or decision.reward_delivered:
            return None

        reward = self.policy.compute_reward(
            action=MemoryAction.SKIP,
            was_retrieved=was_needed,  # If needed, skip was wrong
        )

        next_state = decision.state or MemoryState()
        td_error = None
        if decision.state:
            td_error = self.policy.learn(decision.state, MemoryAction.SKIP, reward, next_state)

        decision.reward_delivered = True
        self._rewards_delivered += 1
        return td_error

    def get_statistics(self) -> Dict[str, Any]:
        """Get gate statistics."""
        total_decisions = (
            self._total_encodes + self._total_skips + self._total_updates
            + self._total_consolidations + self._total_evictions
        )
        return {
            "enabled": self.config.enabled,
            "total_decisions": total_decisions,
            "encodes": self._total_encodes,
            "skips": self._total_skips,
            "updates": self._total_updates,
            "consolidations": self._total_consolidations,
            "evictions": self._total_evictions,
            "skip_rate": round(
                self._total_skips / max(1, total_decisions), 3
            ),
            "rewards_delivered": self._rewards_delivered,
            "tracked_decisions": len(self._decisions),
            "policy_stats": self.policy.get_statistics(),
        }
