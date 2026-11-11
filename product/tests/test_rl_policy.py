"""
Tests for RL Memory Policy — Learned Memory Management Decisions.

Tests verify:
- State representation and discretization
- Action space completeness
- ε-greedy exploration/exploitation
- Q-learning updates and convergence
- Reward signal computation
- Experience replay
- Heuristic fallback behavior
- Adaptive policy switching
- Q-table persistence (save/load)
- Policy statistics and summaries
- Integration with Brain pipeline
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from origin_brain.rl_policy import (
    RLMemoryPolicy,
    RLPolicyConfig,
    MemoryAction,
    MemoryState,
    RewardSignal,
    Experience,
)


class TestMemoryAction:
    """Test the action space."""

    def test_all_actions_exist(self):
        actions = list(MemoryAction)
        assert MemoryAction.ENCODE in actions
        assert MemoryAction.UPDATE in actions
        assert MemoryAction.SKIP in actions
        assert MemoryAction.EVICT in actions
        assert MemoryAction.CONSOLIDATE in actions
        assert len(actions) == 5

    def test_action_values(self):
        assert MemoryAction.ENCODE.value == "encode"
        assert MemoryAction.SKIP.value == "skip"


class TestMemoryState:
    """Test state representation and discretization."""

    def test_default_state(self):
        state = MemoryState()
        assert state.novelty == 0.5
        assert state.salience == 0.3
        assert state.memory_load == 0.0

    def test_custom_state(self):
        state = MemoryState(novelty=0.9, salience=0.8, entropy=0.7)
        assert state.novelty == 0.9
        assert state.salience == 0.8
        assert state.entropy == 0.7

    def test_discretization(self):
        state = MemoryState(novelty=0.5, salience=0.5)
        discrete = state.to_discrete(bins=5)
        assert isinstance(discrete, tuple)
        assert len(discrete) == 10  # 10 features
        # All values should be in [0, 4] for 5 bins
        for v in discrete:
            assert 0 <= v <= 4

    def test_discretization_extremes(self):
        # All zeros
        low = MemoryState(
            novelty=0.0, salience=0.0, content_length=0.0,
            memory_load=0.0, entropy=0.0, interference_rate=0.0,
            sleep_pressure=0.0, wm_load=0.0, similarity_to_best=0.0,
            recency_of_similar=0.0,
        )
        assert all(v == 0 for v in low.to_discrete(bins=5))

        # All ones
        high = MemoryState(
            novelty=1.0, salience=1.0, content_length=1.0,
            memory_load=1.0, entropy=1.0, interference_rate=1.0,
            sleep_pressure=1.0, wm_load=1.0, similarity_to_best=1.0,
            recency_of_similar=1.0,
        )
        assert all(v == 4 for v in high.to_discrete(bins=5))

    def test_different_states_different_discrete(self):
        s1 = MemoryState(novelty=0.1, salience=0.1)
        s2 = MemoryState(novelty=0.9, salience=0.9)
        assert s1.to_discrete() != s2.to_discrete()


class TestRewardSignal:
    """Test reward computation."""

    def test_default_reward(self):
        reward = RewardSignal()
        assert reward.total == 0.0

    def test_positive_reward(self):
        reward = RewardSignal(recall_accuracy=1.0, confidence_bonus=0.2)
        assert reward.total == pytest.approx(1.2)

    def test_negative_components(self):
        reward = RewardSignal(
            recall_accuracy=0.5,
            memory_cost=0.1,
            interference_penalty=0.2,
        )
        assert reward.total == pytest.approx(0.2)

    def test_fully_negative(self):
        reward = RewardSignal(memory_cost=0.5, interference_penalty=0.5)
        assert reward.total == -1.0


class TestRLPolicyConfig:
    """Test configuration."""

    def test_defaults(self):
        cfg = RLPolicyConfig()
        assert cfg.learning_rate == 0.1
        assert cfg.discount_factor == 0.95
        assert cfg.epsilon == 0.2
        assert cfg.state_bins == 5
        assert cfg.replay_buffer_size == 10000

    def test_custom(self):
        cfg = RLPolicyConfig(learning_rate=0.01, epsilon=0.5, state_bins=10)
        assert cfg.learning_rate == 0.01
        assert cfg.epsilon == 0.5
        assert cfg.state_bins == 10


class TestObservation:
    """Test state observation construction."""

    def test_basic_observation(self):
        policy = RLMemoryPolicy()
        state = policy.observe(
            content="Alice met Bob at the park",
            similarity_to_best=0.3,
            salience=0.6,
            memory_count=50,
        )
        assert isinstance(state, MemoryState)
        assert state.novelty == pytest.approx(0.7)  # 1 - 0.3
        assert state.salience == 0.6

    def test_observation_with_entropy_state(self):
        from origin_brain.entropy_monitor import EntropyState

        entropy = EntropyState(
            memory_entropy=0.7,
            interference_rate=0.4,
            sleep_pressure=0.6,
        )
        policy = RLMemoryPolicy()
        state = policy.observe(
            content="test content",
            entropy_state=entropy,
        )
        assert state.entropy == 0.7
        assert state.interference_rate == 0.4
        assert state.sleep_pressure == 0.6

    def test_content_length_normalization(self):
        policy = RLMemoryPolicy()
        short = policy.observe(content="hi", similarity_to_best=0.0)
        long = policy.observe(content="x " * 300, similarity_to_best=0.0)
        assert short.content_length < long.content_length
        assert long.content_length <= 1.0


class TestDecision:
    """Test action selection."""

    def test_decide_returns_action(self):
        policy = RLMemoryPolicy()
        state = MemoryState()
        action = policy.decide(state)
        assert isinstance(action, MemoryAction)

    def test_decide_greedy(self):
        policy = RLMemoryPolicy()
        state = MemoryState()
        # Set a clear best action
        discrete = state.to_discrete(bins=policy.config.state_bins)
        policy._q_table[discrete][MemoryAction.ENCODE] = 10.0
        policy._q_table[discrete][MemoryAction.SKIP] = -5.0

        action = policy.decide_greedy(state)
        assert action == MemoryAction.ENCODE

    def test_exploration_randomness(self):
        """With high epsilon, should see variety in actions."""
        policy = RLMemoryPolicy(config=RLPolicyConfig(epsilon=1.0))
        state = MemoryState()
        actions_seen = set()
        for _ in range(100):
            action = policy.decide(state)
            actions_seen.add(action)
        # With ε=1.0, should explore all actions over 100 tries
        assert len(actions_seen) >= 3  # At least 3 of 5 actions

    def test_exploitation(self):
        """With epsilon=0, should always pick best action."""
        policy = RLMemoryPolicy(config=RLPolicyConfig(epsilon=0.0))
        state = MemoryState(novelty=0.5)
        discrete = state.to_discrete(bins=policy.config.state_bins)
        policy._q_table[discrete][MemoryAction.UPDATE] = 100.0

        actions = [policy.decide(state) for _ in range(20)]
        assert all(a == MemoryAction.UPDATE for a in actions)

    def test_action_count_tracking(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(epsilon=0.0))
        state = MemoryState()
        discrete = state.to_discrete(bins=policy.config.state_bins)
        policy._q_table[discrete][MemoryAction.ENCODE] = 10.0

        for _ in range(5):
            policy.decide(state)

        assert policy._action_counts[MemoryAction.ENCODE] == 5


class TestQLearning:
    """Test Q-learning updates."""

    def test_basic_learn(self):
        policy = RLMemoryPolicy()
        state = MemoryState(novelty=0.9)
        next_state = MemoryState(novelty=0.5)
        reward = RewardSignal(recall_accuracy=1.0)

        td_error = policy.learn(state, MemoryAction.ENCODE, reward, next_state)
        assert isinstance(td_error, float)

    def test_q_value_increases_with_positive_reward(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(
            min_experiences=9999,  # Disable replay for this test
        ))
        state = MemoryState(novelty=0.9)
        next_state = MemoryState(novelty=0.5)

        discrete = state.to_discrete(bins=policy.config.state_bins)
        q_before = policy._q_table[discrete][MemoryAction.ENCODE]

        reward = RewardSignal(recall_accuracy=1.0)
        policy.learn(state, MemoryAction.ENCODE, reward, next_state)

        q_after = policy._q_table[discrete][MemoryAction.ENCODE]
        assert q_after > q_before

    def test_q_value_decreases_with_negative_reward(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(
            min_experiences=9999,
        ))
        state = MemoryState(novelty=0.1)
        next_state = MemoryState(novelty=0.1)

        # First give it a positive value to decrease from
        discrete = state.to_discrete(bins=policy.config.state_bins)
        policy._q_table[discrete][MemoryAction.SKIP] = 1.0

        reward = RewardSignal(memory_cost=0.5, interference_penalty=0.5)
        policy.learn(state, MemoryAction.SKIP, reward, next_state)

        q_after = policy._q_table[discrete][MemoryAction.SKIP]
        assert q_after < 1.0

    def test_epsilon_decay(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(
            epsilon=0.5, epsilon_decay=0.99, min_experiences=9999,
        ))
        initial_eps = policy._epsilon

        state = MemoryState()
        reward = RewardSignal()
        for _ in range(10):
            policy.learn(state, MemoryAction.ENCODE, reward, state)

        assert policy._epsilon < initial_eps

    def test_episode_count_increments(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=9999))
        state = MemoryState()
        reward = RewardSignal()

        assert policy._episode_count == 0
        policy.learn(state, MemoryAction.ENCODE, reward, state)
        assert policy._episode_count == 1
        policy.learn(state, MemoryAction.SKIP, reward, state)
        assert policy._episode_count == 2


class TestExperienceReplay:
    """Test experience replay buffer and learning."""

    def test_experiences_stored(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=9999))
        state = MemoryState()
        reward = RewardSignal(recall_accuracy=0.5)

        policy.learn(state, MemoryAction.ENCODE, reward, state)
        assert len(policy._replay_buffer) == 1

    def test_replay_buffer_bounded(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(
            replay_buffer_size=10, min_experiences=9999,
        ))
        state = MemoryState()
        reward = RewardSignal()

        for _ in range(20):
            policy.learn(state, MemoryAction.ENCODE, reward, state)

        assert len(policy._replay_buffer) <= 10

    def test_replay_learning_triggers(self):
        """After min_experiences, replay should update Q-values."""
        policy = RLMemoryPolicy(config=RLPolicyConfig(
            min_experiences=5, batch_size=3,
        ))
        state = MemoryState(novelty=0.9)
        reward = RewardSignal(recall_accuracy=1.0)

        # Feed enough experiences to trigger replay
        for _ in range(10):
            policy.learn(state, MemoryAction.ENCODE, reward, state)

        # Q-value should be significantly positive after replay
        discrete = state.to_discrete(bins=policy.config.state_bins)
        assert policy._q_table[discrete][MemoryAction.ENCODE] > 0


class TestRewardComputation:
    """Test reward signal computation for different scenarios."""

    def test_encode_retrieved(self):
        policy = RLMemoryPolicy()
        reward = policy.compute_reward(
            MemoryAction.ENCODE, was_retrieved=True, retrieval_rank=0,
        )
        assert reward.recall_accuracy > 0
        assert reward.total > 0

    def test_encode_not_retrieved(self):
        policy = RLMemoryPolicy()
        reward = policy.compute_reward(
            MemoryAction.ENCODE, was_retrieved=False,
        )
        # Should still have a memory cost
        assert reward.memory_cost > 0

    def test_skip_correct(self):
        policy = RLMemoryPolicy()
        reward = policy.compute_reward(
            MemoryAction.SKIP, was_retrieved=False,
        )
        assert reward.total >= 0  # Small positive for correct skip

    def test_skip_wrong(self):
        policy = RLMemoryPolicy()
        reward = policy.compute_reward(
            MemoryAction.SKIP, was_retrieved=True,
        )
        assert reward.total < 0  # Negative for skipping needed content

    def test_interference_penalty(self):
        policy = RLMemoryPolicy()
        reward_clean = policy.compute_reward(
            MemoryAction.ENCODE, was_retrieved=True,
            retrieval_rank=0, caused_interference=False,
        )
        reward_dirty = policy.compute_reward(
            MemoryAction.ENCODE, was_retrieved=True,
            retrieval_rank=0, caused_interference=True,
        )
        assert reward_dirty.total < reward_clean.total

    def test_update_action_reward(self):
        policy = RLMemoryPolicy()
        reward = policy.compute_reward(
            MemoryAction.UPDATE, was_retrieved=True,
        )
        assert reward.recall_accuracy > 0
        # Update cost should be less than encode cost
        assert reward.memory_cost < policy.config.encode_cost


class TestHeuristicFallback:
    """Test the rule-based heuristic policy."""

    def test_high_pressure_consolidates(self):
        policy = RLMemoryPolicy()
        state = MemoryState(sleep_pressure=0.9, entropy=0.8)
        assert policy.heuristic_decide(state) == MemoryAction.CONSOLIDATE

    def test_novel_important_encodes(self):
        policy = RLMemoryPolicy()
        state = MemoryState(novelty=0.9, salience=0.8)
        assert policy.heuristic_decide(state) == MemoryAction.ENCODE

    def test_similar_updates(self):
        policy = RLMemoryPolicy()
        state = MemoryState(similarity_to_best=0.9, novelty=0.1)
        assert policy.heuristic_decide(state) == MemoryAction.UPDATE

    def test_low_value_skips(self):
        policy = RLMemoryPolicy()
        state = MemoryState(novelty=0.1, salience=0.1)
        assert policy.heuristic_decide(state) == MemoryAction.SKIP

    def test_overloaded_evicts(self):
        policy = RLMemoryPolicy()
        state = MemoryState(memory_load=0.95, novelty=0.6, salience=0.3)
        assert policy.heuristic_decide(state) == MemoryAction.EVICT

    def test_default_encodes(self):
        policy = RLMemoryPolicy()
        state = MemoryState(novelty=0.5, salience=0.3)
        assert policy.heuristic_decide(state) == MemoryAction.ENCODE


class TestAdaptivePolicy:
    """Test heuristic → RL transition."""

    def test_starts_with_heuristic(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=50))
        assert policy.should_use_heuristic()

    def test_switches_to_rl_after_training(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=5))
        state = MemoryState()
        reward = RewardSignal()

        for _ in range(10):
            policy.learn(state, MemoryAction.ENCODE, reward, state)

        assert not policy.should_use_heuristic()

    def test_adaptive_decide_uses_heuristic_initially(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=100))
        state = MemoryState(sleep_pressure=0.9, entropy=0.8)
        action = policy.decide_adaptive(state)
        assert action == MemoryAction.CONSOLIDATE  # Heuristic behavior


class TestPendingLearning:
    """Test deferred reward learning."""

    def test_learn_from_pending(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=9999))
        state = MemoryState(novelty=0.9)

        # Decide sets pending state
        action = policy.decide(state)
        assert policy._pending_state is not None
        assert policy._pending_action is not None

        # Learn from pending with deferred reward
        reward = RewardSignal(recall_accuracy=1.0)
        next_state = MemoryState(novelty=0.5)
        td_error = policy.learn_from_pending(reward, next_state)
        assert td_error is not None

        # Pending should be cleared
        assert policy._pending_state is None
        assert policy._pending_action is None

    def test_learn_from_pending_no_pending(self):
        policy = RLMemoryPolicy()
        reward = RewardSignal()
        result = policy.learn_from_pending(reward, MemoryState())
        assert result is None


class TestPersistence:
    """Test Q-table save/load."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "policy.json")

            # Train a policy
            policy1 = RLMemoryPolicy(config=RLPolicyConfig(
                save_path=save_path, min_experiences=9999,
            ))
            state = MemoryState(novelty=0.9)
            reward = RewardSignal(recall_accuracy=1.0)
            for _ in range(5):
                policy1.learn(state, MemoryAction.ENCODE, reward, state)

            # Save
            policy1.save_q_table()
            assert Path(save_path).exists()

            # Load into new policy
            policy2 = RLMemoryPolicy(config=RLPolicyConfig(save_path=save_path))

            # Q-values should match
            discrete = state.to_discrete(bins=policy1.config.state_bins)
            assert (
                policy2._q_table[discrete][MemoryAction.ENCODE]
                == pytest.approx(policy1._q_table[discrete][MemoryAction.ENCODE], abs=0.001)
            )
            assert policy2._episode_count == policy1._episode_count

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "subdir", "policy.json")
            policy = RLMemoryPolicy()
            result = policy.save_q_table(save_path)
            assert Path(result).exists()

    def test_load_nonexistent_file(self):
        """Loading from nonexistent file should silently succeed."""
        policy = RLMemoryPolicy(config=RLPolicyConfig(
            save_path="/nonexistent/path/policy.json"
        ))
        assert policy._episode_count == 0


class TestStatistics:
    """Test statistics and policy summaries."""

    def test_empty_statistics(self):
        policy = RLMemoryPolicy()
        stats = policy.get_statistics()
        assert stats["episode_count"] == 0
        assert stats["q_table_size"] == 0
        assert stats["using_heuristic"] is True
        assert stats["mode"] == "heuristic"

    def test_populated_statistics(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=3))
        state = MemoryState()
        reward = RewardSignal(recall_accuracy=0.5)

        for _ in range(5):
            policy.learn(state, MemoryAction.ENCODE, reward, state)

        stats = policy.get_statistics()
        assert stats["episode_count"] == 5
        assert stats["total_reward"] > 0
        assert stats["mode"] == "rl"
        assert stats["replay_buffer_size"] == 5

    def test_policy_summary(self):
        policy = RLMemoryPolicy()
        summary = policy.get_policy_summary()
        assert "novel_important" in summary
        assert "redundant" in summary
        assert "high_pressure" in summary
        assert summary["novel_important"]["preferred_action"] == "encode"
        assert summary["high_pressure"]["preferred_action"] == "consolidate"

    def test_reward_history_bounded(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=9999))
        state = MemoryState()
        reward = RewardSignal()
        for _ in range(1500):
            policy.learn(state, MemoryAction.ENCODE, reward, state)
        assert len(policy._reward_history) <= 1000


class TestConvergence:
    """Test that Q-learning converges on simple scenarios."""

    def test_learns_to_encode_novel(self):
        """Policy should learn that encoding novel content is good."""
        policy = RLMemoryPolicy(config=RLPolicyConfig(
            epsilon=0.3, min_experiences=10, batch_size=5,
            learning_rate=0.2,
        ))

        novel_state = MemoryState(novelty=0.9, salience=0.8)
        boring_state = MemoryState(novelty=0.1, salience=0.1)

        # Train: encoding novel content → positive reward
        for _ in range(100):
            policy.learn(
                novel_state, MemoryAction.ENCODE,
                RewardSignal(recall_accuracy=0.8), novel_state,
            )
            # Skipping novel content → negative reward
            policy.learn(
                novel_state, MemoryAction.SKIP,
                RewardSignal(recall_accuracy=-0.5), novel_state,
            )

        # After training, ENCODE should have higher Q than SKIP for novel state
        discrete = novel_state.to_discrete(bins=policy.config.state_bins)
        assert (
            policy._q_table[discrete][MemoryAction.ENCODE]
            > policy._q_table[discrete][MemoryAction.SKIP]
        )

    def test_learns_to_skip_redundant(self):
        """Policy should learn that skipping redundant content is good."""
        policy = RLMemoryPolicy(config=RLPolicyConfig(
            epsilon=0.3, min_experiences=10, batch_size=5,
            learning_rate=0.2,
        ))

        redundant_state = MemoryState(novelty=0.1, similarity_to_best=0.9)

        for _ in range(100):
            policy.learn(
                redundant_state, MemoryAction.SKIP,
                RewardSignal(recall_accuracy=0.1), redundant_state,
            )
            policy.learn(
                redundant_state, MemoryAction.ENCODE,
                RewardSignal(memory_cost=0.1, interference_penalty=0.2), redundant_state,
            )

        discrete = redundant_state.to_discrete(bins=policy.config.state_bins)
        assert (
            policy._q_table[discrete][MemoryAction.SKIP]
            > policy._q_table[discrete][MemoryAction.ENCODE]
        )
