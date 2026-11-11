"""
RL Memory Policy — Learned Memory Management Decisions
═══════════════════════════════════════════════════════

Replaces heuristic encode/update/forget decisions with a learned policy.
Instead of hand-tuned thresholds for when to store, update, or evict memories,
the system learns optimal memory management from downstream task performance.

Key insight from MemAgent (2025) and Mem-α (2026):
  An RL agent that learns WHEN to read/write memory massively outperforms
  heuristic approaches. MemAgent extrapolates 8K → 3.5M+ tokens.
  Mem-α extrapolates 30K → 400K+ tokens (13x improvement).

Our approach:
  We implement a lightweight RL controller that sits on top of Origin Brain's
  existing pipeline. At each encode/recall event, the controller observes the
  memory state and decides the best action. The reward signal is downstream
  memory quality (recall accuracy, metamemory confidence).

Architecture:
  ┌─────────────────────────────────────────────┐
  │  RL Memory Controller                        │
  │                                               │
  │  State = [entropy, interference, churn,       │
  │           pressure, novelty, recency,         │
  │           similarity_to_existing, salience,   │
  │           memory_count, wm_load]              │
  │                                               │
  │  Actions = ENCODE | UPDATE | SKIP | EVICT     │
  │                                               │
  │  Policy = ε-greedy Q-learning (tabular)       │
  │  → upgradable to DQN when scale warrants      │
  │                                               │
  │  Reward = recall_accuracy + confidence_bonus  │
  │           - memory_cost - interference_penalty │
  └─────────────────────────────────────────────┘

Brain analogue:
  - Dopaminergic reward prediction error (VTA → striatum)
  - Basal ganglia action selection (Go/NoGo pathways)
  - Prefrontal cortex executive control of memory encoding

Research basis:
  - MemAgent (NeurIPS 2025): RL read/write for 3.5M+ token contexts
  - Mem-α (ICLR 2026): RL memory policy, 30K → 400K extrapolation
  - Anderson ACT-R: Activation-based memory management with learned parameters
  - Dopaminergic gating theory: DA signals gate hippocampal encoding
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict, deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Action Space ──────────────────────────────────────────────────────

class MemoryAction(str, Enum):
    """Actions the RL controller can take on incoming information."""
    ENCODE = "encode"           # Store as new episodic memory
    UPDATE = "update"           # Update existing similar memory (reconsolidate)
    SKIP = "skip"               # Don't store — low value, redundant, or noise
    EVICT = "evict"             # Store new but evict a low-value existing memory
    CONSOLIDATE = "consolidate" # Trigger consolidation (compress to semantic)


# ── State Representation ─────────────────────────────────────────────

class MemoryState(BaseModel):
    """
    Feature vector representing the current memory system state.

    Each feature is normalized to [0, 1] for stable Q-learning.
    """
    # Content features
    novelty: float = 0.5          # How novel is this content? (1 - max_similarity)
    salience: float = 0.3         # Emotional/prediction-based importance
    content_length: float = 0.5   # Normalized length (short=0, long=1)

    # Memory system state
    memory_load: float = 0.0      # episodic_count / max_capacity
    entropy: float = 0.0          # Current memory entropy (from entropy monitor)
    interference_rate: float = 0.0 # Recent interference rate
    sleep_pressure: float = 0.0   # Current consolidation pressure
    wm_load: float = 0.0          # Working memory fullness

    # Retrieval features
    similarity_to_best: float = 0.0  # Max similarity to existing memories
    recency_of_similar: float = 0.0  # How recent is the most similar memory?

    def to_discrete(self, bins: int = 5) -> Tuple[int, ...]:
        """
        Discretize continuous state into bins for tabular Q-learning.

        Each feature is mapped to [0, bins-1], creating a discrete state tuple
        that serves as a dictionary key in the Q-table.
        """
        def discretize(val: float) -> int:
            return min(bins - 1, max(0, int(val * bins)))

        return (
            discretize(self.novelty),
            discretize(self.salience),
            discretize(self.content_length),
            discretize(self.memory_load),
            discretize(self.entropy),
            discretize(self.interference_rate),
            discretize(self.sleep_pressure),
            discretize(self.wm_load),
            discretize(self.similarity_to_best),
            discretize(self.recency_of_similar),
        )


# ── Reward Signal ─────────────────────────────────────────────────────

class RewardSignal(BaseModel):
    """
    Reward signal components for RL training.

    Total reward = recall_accuracy + confidence_bonus - memory_cost - interference_penalty
    """
    recall_accuracy: float = 0.0        # Was the stored memory useful later?
    confidence_bonus: float = 0.0       # Metamemory confidence improvement
    memory_cost: float = 0.0            # Penalty for storing low-value memories
    interference_penalty: float = 0.0   # Penalty for causing interference

    @property
    def total(self) -> float:
        return (
            self.recall_accuracy
            + self.confidence_bonus
            - self.memory_cost
            - self.interference_penalty
        )


# ── Experience Replay Buffer ──────────────────────────────────────────

class Experience(BaseModel):
    """Single experience tuple for replay learning."""
    state: Tuple[int, ...] = ()
    action: MemoryAction = MemoryAction.ENCODE
    reward: float = 0.0
    next_state: Tuple[int, ...] = ()
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Q-Learning Policy ────────────────────────────────────────────────

class RLPolicyConfig(BaseModel):
    """Configuration for the RL memory policy."""
    # Q-learning hyperparameters
    learning_rate: float = 0.1        # α — step size for Q-updates
    discount_factor: float = 0.95     # γ — future reward discount
    epsilon: float = 0.2              # ε — exploration rate
    epsilon_decay: float = 0.999      # ε decay per episode
    epsilon_min: float = 0.01         # Minimum exploration rate

    # State discretization
    state_bins: int = 5               # Number of bins per feature dimension

    # Experience replay
    replay_buffer_size: int = 10000   # Max experiences stored
    batch_size: int = 32              # Mini-batch size for replay updates
    min_experiences: int = 50         # Min experiences before replay learning

    # Reward shaping
    encode_cost: float = 0.01        # Small penalty for every encode (memory budget)
    eviction_cost: float = 0.005     # Small penalty for evictions
    skip_bonus: float = 0.005        # Tiny bonus for correctly skipping noise
    consolidation_bonus: float = 0.02 # Bonus for timely consolidation

    # Memory capacity for load computation
    max_memory_capacity: int = 10000  # Reference capacity for normalization

    # Persistence
    save_path: Optional[str] = None   # Path to save/load Q-table


class RLMemoryPolicy:
    """
    Reinforcement Learning controller for memory management decisions.

    Uses ε-greedy Q-learning to learn when to encode, update, skip, or evict
    memories based on the current state of the memory system.

    Usage:
        policy = RLMemoryPolicy()

        # At each encode event:
        state = policy.observe(content, brain)
        action = policy.decide(state)

        # Execute action...

        # After downstream use (recall), provide reward:
        reward = policy.compute_reward(recall_result, action)
        policy.learn(state, action, reward, next_state)
    """

    def __init__(self, config: Optional[RLPolicyConfig] = None):
        self.config = config or RLPolicyConfig()

        # Q-table: state_tuple → {action: q_value}
        self._q_table: Dict[Tuple[int, ...], Dict[MemoryAction, float]] = defaultdict(
            lambda: {a: 0.0 for a in MemoryAction}
        )

        # Experience replay buffer
        self._replay_buffer: deque[Experience] = deque(maxlen=self.config.replay_buffer_size)

        # Training statistics
        self._episode_count: int = 0
        self._total_reward: float = 0.0
        self._action_counts: Dict[MemoryAction, int] = {a: 0 for a in MemoryAction}
        self._reward_history: deque[float] = deque(maxlen=1000)

        # Current epsilon
        self._epsilon: float = self.config.epsilon

        # Pending experience (state, action waiting for reward)
        self._pending_state: Optional[Tuple[int, ...]] = None
        self._pending_action: Optional[MemoryAction] = None

        # Load saved Q-table if available
        if self.config.save_path:
            self._load_q_table()

        logger.info("RLMemoryPolicy initialized (ε-greedy Q-learning)")

    # ── Observation ────────────────────────────────────────────────────

    def observe(
        self,
        content: str,
        entropy_state: Optional[Any] = None,
        similarity_to_best: float = 0.0,
        recency_of_similar: float = 0.0,
        salience: float = 0.3,
        memory_count: int = 0,
        wm_full: bool = False,
    ) -> MemoryState:
        """
        Build the state observation from current context.

        Can be called with raw values (standalone) or with a Brain instance.
        """
        # Compute novelty from similarity
        novelty = max(0.0, 1.0 - similarity_to_best)

        # Normalize content length (assume ~500 chars is "long")
        content_length = min(1.0, len(content) / 500.0)

        # Memory load
        memory_load = min(1.0, memory_count / max(1, self.config.max_memory_capacity))

        # Entropy features (from entropy monitor if available)
        entropy = 0.0
        interference_rate = 0.0
        sleep_pressure = 0.0
        if entropy_state is not None:
            entropy = getattr(entropy_state, 'memory_entropy', 0.0)
            interference_rate = getattr(entropy_state, 'interference_rate', 0.0)
            sleep_pressure = getattr(entropy_state, 'sleep_pressure', 0.0)

        return MemoryState(
            novelty=novelty,
            salience=salience,
            content_length=content_length,
            memory_load=memory_load,
            entropy=entropy,
            interference_rate=interference_rate,
            sleep_pressure=sleep_pressure,
            wm_load=1.0 if wm_full else 0.0,
            similarity_to_best=similarity_to_best,
            recency_of_similar=recency_of_similar,
        )

    # ── Decision ──────────────────────────────────────────────────────

    def decide(self, state: MemoryState) -> MemoryAction:
        """
        Choose an action using ε-greedy policy.

        With probability ε: explore (random action)
        With probability 1-ε: exploit (best Q-value action)
        """
        discrete_state = state.to_discrete(bins=self.config.state_bins)

        if random.random() < self._epsilon:
            action = random.choice(list(MemoryAction))
        else:
            q_values = self._q_table[discrete_state]
            action = max(q_values, key=q_values.get)

        # Track pending for deferred reward
        self._pending_state = discrete_state
        self._pending_action = action

        self._action_counts[action] += 1
        return action

    def decide_greedy(self, state: MemoryState) -> MemoryAction:
        """
        Choose the best action without exploration (for inference).
        """
        discrete_state = state.to_discrete(bins=self.config.state_bins)
        q_values = self._q_table[discrete_state]
        return max(q_values, key=q_values.get)

    # ── Reward Computation ────────────────────────────────────────────

    def compute_reward(
        self,
        action: MemoryAction,
        was_retrieved: bool = False,
        retrieval_rank: int = -1,
        confidence_delta: float = 0.0,
        caused_interference: bool = False,
    ) -> RewardSignal:
        """
        Compute the reward signal for a memory action.

        Called after downstream events reveal the quality of the decision.
        """
        reward = RewardSignal()

        if action == MemoryAction.ENCODE:
            # Reward for encoding: positive if it was later retrieved
            if was_retrieved:
                # Higher reward for top-ranked retrievals
                rank_bonus = max(0.0, 1.0 - (retrieval_rank / 10.0)) if retrieval_rank >= 0 else 0.5
                reward.recall_accuracy = rank_bonus
            # Cost for encoding (memory budget)
            reward.memory_cost = self.config.encode_cost

        elif action == MemoryAction.UPDATE:
            if was_retrieved:
                reward.recall_accuracy = 0.8  # Updates that get retrieved are valuable
            reward.memory_cost = self.config.encode_cost * 0.5  # Cheaper than new encode

        elif action == MemoryAction.SKIP:
            if not was_retrieved:
                # Correctly skipped something that wasn't needed
                reward.recall_accuracy = self.config.skip_bonus
            else:
                # Skipped something that was later needed — penalty
                reward.recall_accuracy = -0.5

        elif action == MemoryAction.EVICT:
            if was_retrieved:
                reward.recall_accuracy = 0.6
            reward.memory_cost = self.config.eviction_cost

        elif action == MemoryAction.CONSOLIDATE:
            reward.confidence_bonus = self.config.consolidation_bonus

        # Interference penalty
        if caused_interference:
            reward.interference_penalty = 0.1

        # Confidence change reward
        reward.confidence_bonus += max(0.0, confidence_delta) * 0.3

        return reward

    # ── Learning ──────────────────────────────────────────────────────

    def learn(
        self,
        state: MemoryState,
        action: MemoryAction,
        reward: RewardSignal,
        next_state: MemoryState,
    ) -> float:
        """
        Update Q-table with new experience using Q-learning update rule.

        Q(s, a) ← Q(s, a) + α[r + γ·max_a' Q(s', a') - Q(s, a)]

        Returns the TD error (useful for monitoring convergence).
        """
        s = state.to_discrete(bins=self.config.state_bins)
        s_next = next_state.to_discrete(bins=self.config.state_bins)
        r = reward.total

        # Current Q-value
        current_q = self._q_table[s][action]

        # Best future Q-value
        future_q = max(self._q_table[s_next].values())

        # TD target
        td_target = r + self.config.discount_factor * future_q

        # TD error
        td_error = td_target - current_q

        # Q-update
        self._q_table[s][action] = current_q + self.config.learning_rate * td_error

        # Store experience for replay
        self._replay_buffer.append(Experience(
            state=s,
            action=action,
            reward=r,
            next_state=s_next,
        ))

        # Update statistics
        self._episode_count += 1
        self._total_reward += r
        self._reward_history.append(r)

        # Decay epsilon
        self._epsilon = max(
            self.config.epsilon_min,
            self._epsilon * self.config.epsilon_decay,
        )

        # Experience replay learning
        if len(self._replay_buffer) >= self.config.min_experiences:
            self._replay_update()

        return td_error

    def learn_from_pending(self, reward: RewardSignal, next_state: MemoryState) -> Optional[float]:
        """
        Learn from the pending (state, action) pair with deferred reward.

        Useful when reward is only available after downstream recall.
        """
        if self._pending_state is None or self._pending_action is None:
            return None

        s = self._pending_state
        s_next = next_state.to_discrete(bins=self.config.state_bins)
        r = reward.total
        action = self._pending_action

        # Q-update
        current_q = self._q_table[s][action]
        future_q = max(self._q_table[s_next].values())
        td_target = r + self.config.discount_factor * future_q
        td_error = td_target - current_q
        self._q_table[s][action] = current_q + self.config.learning_rate * td_error

        # Store experience
        self._replay_buffer.append(Experience(
            state=s, action=action, reward=r, next_state=s_next,
        ))

        # Update stats
        self._episode_count += 1
        self._total_reward += r
        self._reward_history.append(r)
        self._epsilon = max(self.config.epsilon_min, self._epsilon * self.config.epsilon_decay)

        # Clear pending
        self._pending_state = None
        self._pending_action = None

        # Replay
        if len(self._replay_buffer) >= self.config.min_experiences:
            self._replay_update()

        return td_error

    def _replay_update(self) -> None:
        """
        Experience replay: sample random mini-batch and update Q-values.

        Breaks temporal correlations and improves sample efficiency.
        """
        batch_size = min(self.config.batch_size, len(self._replay_buffer))
        batch = random.sample(list(self._replay_buffer), batch_size)

        for exp in batch:
            current_q = self._q_table[exp.state][exp.action]
            future_q = max(self._q_table[exp.next_state].values())
            td_target = exp.reward + self.config.discount_factor * future_q
            td_error = td_target - current_q
            self._q_table[exp.state][exp.action] = (
                current_q + self.config.learning_rate * td_error
            )

    # ── Heuristic Fallback ────────────────────────────────────────────

    def heuristic_decide(self, state: MemoryState) -> MemoryAction:
        """
        Rule-based fallback for before the policy has learned enough.

        This encodes the hand-tuned heuristics that RL should eventually improve upon.
        """
        # Very high sleep pressure → consolidate
        if state.sleep_pressure > 0.8:
            return MemoryAction.CONSOLIDATE

        # Very novel content → always encode
        if state.novelty > 0.8 and state.salience > 0.4:
            return MemoryAction.ENCODE

        # Very similar to existing → update instead
        if state.similarity_to_best > 0.8:
            return MemoryAction.UPDATE

        # Low novelty + low salience → skip
        if state.novelty < 0.2 and state.salience < 0.2:
            return MemoryAction.SKIP

        # Memory overloaded → evict something to make room
        if state.memory_load > 0.9 and state.novelty > 0.5:
            return MemoryAction.EVICT

        # Default: encode
        return MemoryAction.ENCODE

    def should_use_heuristic(self) -> bool:
        """Whether to fall back to heuristic (not enough training data)."""
        return self._episode_count < self.config.min_experiences

    def decide_adaptive(self, state: MemoryState) -> MemoryAction:
        """
        Adaptive decision: use heuristic until enough training data, then RL.
        """
        if self.should_use_heuristic():
            return self.heuristic_decide(state)
        return self.decide(state)

    # ── Persistence ───────────────────────────────────────────────────

    def save_q_table(self, path: Optional[str] = None) -> str:
        """Save Q-table and training stats to JSON."""
        save_path = path or self.config.save_path
        if not save_path:
            save_path = "rl_memory_policy.json"

        # Convert defaultdict and tuples to serializable format
        q_data = {}
        for state_tuple, action_values in self._q_table.items():
            key = json.dumps(list(state_tuple))
            q_data[key] = {a.value: v for a, v in action_values.items()}

        data = {
            "q_table": q_data,
            "episode_count": self._episode_count,
            "total_reward": self._total_reward,
            "epsilon": self._epsilon,
            "action_counts": {a.value: c for a, c in self._action_counts.items()},
            "config": self.config.model_dump(mode="json"),
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved RL policy to {save_path} ({len(q_data)} states)")
        return save_path

    def _load_q_table(self) -> None:
        """Load Q-table from JSON."""
        if not self.config.save_path or not Path(self.config.save_path).exists():
            return

        try:
            with open(self.config.save_path, "r") as f:
                data = json.load(f)

            for key, action_values in data.get("q_table", {}).items():
                state_tuple = tuple(json.loads(key))
                self._q_table[state_tuple] = {
                    MemoryAction(a): v for a, v in action_values.items()
                }

            self._episode_count = data.get("episode_count", 0)
            self._total_reward = data.get("total_reward", 0.0)
            self._epsilon = data.get("epsilon", self.config.epsilon)

            for a_str, count in data.get("action_counts", {}).items():
                self._action_counts[MemoryAction(a_str)] = count

            logger.info(
                f"Loaded RL policy: {len(self._q_table)} states, "
                f"{self._episode_count} episodes, ε={self._epsilon:.3f}"
            )
        except Exception as e:
            logger.warning(f"Failed to load RL policy: {e}")

    # ── Statistics ────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """Get training and policy statistics."""
        avg_reward = (
            sum(self._reward_history) / len(self._reward_history)
            if self._reward_history else 0.0
        )
        recent_reward = (
            sum(list(self._reward_history)[-100:]) / min(100, len(self._reward_history))
            if self._reward_history else 0.0
        )

        return {
            "episode_count": self._episode_count,
            "total_reward": round(self._total_reward, 3),
            "average_reward": round(avg_reward, 4),
            "recent_avg_reward": round(recent_reward, 4),
            "epsilon": round(self._epsilon, 4),
            "q_table_size": len(self._q_table),
            "replay_buffer_size": len(self._replay_buffer),
            "action_distribution": {
                a.value: c for a, c in self._action_counts.items()
            },
            "using_heuristic": self.should_use_heuristic(),
            "mode": "heuristic" if self.should_use_heuristic() else "rl",
        }

    def get_policy_summary(self) -> Dict[str, Any]:
        """
        Summarize what the learned policy does in different situations.

        Returns the preferred action for key state archetypes.
        """
        archetypes = {
            "novel_important": MemoryState(novelty=0.9, salience=0.8),
            "novel_unimportant": MemoryState(novelty=0.9, salience=0.1),
            "redundant": MemoryState(novelty=0.1, similarity_to_best=0.9),
            "overloaded": MemoryState(memory_load=0.95, novelty=0.5, salience=0.5),
            "high_pressure": MemoryState(sleep_pressure=0.9, entropy=0.8),
            "normal": MemoryState(novelty=0.5, salience=0.3),
        }

        summary = {}
        for name, state in archetypes.items():
            if self.should_use_heuristic():
                action = self.heuristic_decide(state)
            else:
                action = self.decide_greedy(state)
            discrete = state.to_discrete(bins=self.config.state_bins)
            q_vals = {a.value: round(v, 3) for a, v in self._q_table[discrete].items()}
            summary[name] = {
                "preferred_action": action.value,
                "q_values": q_vals,
            }

        return summary
