"""
Tests for RLEncodingGate — Active RL-gated encode pipeline.

Tests verify:
- Pre-encode decision making
- Post-encode memory ID linking
- Deferred reward on retrieval
- Skip verification
- Statistics tracking
- Disabled gate passthrough
- Integration with Brain
"""

import pytest
from origin_brain.rl_policy import RLMemoryPolicy, RLPolicyConfig, MemoryAction, MemoryState, RewardSignal
from origin_brain.rl_gate import RLEncodingGate, RLGateConfig, EncodingDecision
from origin_brain.entropy_monitor import EntropyState


class TestEncodingDecision:
    """Test decision model."""

    def test_default_decision(self):
        d = EncodingDecision()
        assert d.action == MemoryAction.ENCODE
        assert d.memory_id is None
        assert not d.reward_delivered


class TestPreEncode:
    """Test pre-encode decision making."""

    def test_returns_decision(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy)
        decision = gate.pre_encode("Some new content", similarity_to_best=0.1, salience=0.5)
        assert isinstance(decision, EncodingDecision)
        assert decision.action in list(MemoryAction)

    def test_with_entropy_state(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy)
        entropy = EntropyState(memory_entropy=0.7, sleep_pressure=0.5)
        decision = gate.pre_encode(
            "test", entropy_state=entropy, salience=0.3,
        )
        assert decision.state is not None
        assert decision.state.entropy == 0.7

    def test_disabled_gate_always_encodes(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy, config=RLGateConfig(enabled=False))
        for _ in range(10):
            d = gate.pre_encode("content", similarity_to_best=0.9, salience=0.1)
            assert d.action == MemoryAction.ENCODE

    def test_tracks_decisions(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy)
        gate.pre_encode("first content")
        gate.pre_encode("second content")
        assert len(gate._decisions) == 2


class TestPostEncode:
    """Test post-encode memory linking."""

    def test_links_memory_id(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy)
        decision = gate.pre_encode("content")
        gate.post_encode(decision, memory_id="mem-001")
        assert decision.memory_id == "mem-001"
        assert "mem-001" in gate._memory_to_decision


class TestDeferredReward:
    """Test deferred reward delivery on retrieval."""

    def test_reward_on_retrieval(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=9999))
        gate = RLEncodingGate(policy)

        # Encode
        decision = gate.pre_encode("important fact", salience=0.8)
        decision.action = MemoryAction.ENCODE  # Force encode
        gate.post_encode(decision, memory_id="mem-001")

        # Retrieve
        td_error = gate.on_retrieval("mem-001", retrieval_rank=0, confidence=0.9)
        # Should have delivered reward
        assert decision.reward_delivered
        assert gate._rewards_delivered == 1

    def test_no_reward_for_unknown_memory(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy)
        result = gate.on_retrieval("unknown-mem")
        assert result is None

    def test_no_double_reward(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=9999))
        gate = RLEncodingGate(policy)

        decision = gate.pre_encode("content")
        decision.action = MemoryAction.ENCODE
        gate.post_encode(decision, memory_id="mem-002")

        gate.on_retrieval("mem-002", retrieval_rank=0)
        result2 = gate.on_retrieval("mem-002", retrieval_rank=1)
        assert result2 is None  # Already rewarded


class TestSkipVerification:
    """Test skip decision verification."""

    def test_correct_skip(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=9999))
        gate = RLEncodingGate(policy)

        decision = gate.pre_encode("noise content")
        decision.action = MemoryAction.SKIP
        content_hash = decision.content_hash

        td_error = gate.on_skip_verified(content_hash, was_needed=False)
        assert decision.reward_delivered

    def test_wrong_skip(self):
        policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=9999))
        gate = RLEncodingGate(policy)

        decision = gate.pre_encode("important content")
        decision.action = MemoryAction.SKIP
        content_hash = decision.content_hash

        td_error = gate.on_skip_verified(content_hash, was_needed=True)
        assert decision.reward_delivered

    def test_skip_verify_unknown(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy)
        result = gate.on_skip_verified(99999, was_needed=True)
        assert result is None


class TestStatistics:
    """Test statistics reporting."""

    def test_empty_stats(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy)
        stats = gate.get_statistics()
        assert stats["total_decisions"] == 0
        assert stats["skip_rate"] == 0.0
        assert stats["enabled"] is True

    def test_populated_stats(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy, config=RLGateConfig(enabled=False))

        for _ in range(10):
            gate.pre_encode("content")

        stats = gate.get_statistics()
        assert stats["total_decisions"] == 10
        assert stats["encodes"] == 10  # All encode when disabled
        assert "policy_stats" in stats

    def test_skip_rate(self):
        policy = RLMemoryPolicy()
        gate = RLEncodingGate(policy)

        # Manually create mixed decisions
        for _ in range(7):
            d = gate.pre_encode("novel content")
            d.action = MemoryAction.ENCODE
        gate._total_encodes = 7
        gate._total_skips = 3

        stats = gate.get_statistics()
        assert stats["skip_rate"] == pytest.approx(0.3, abs=0.01)


class TestBrainIntegration:
    """Test RL gate works with Brain."""

    def test_brain_has_rl_gate(self):
        from origin_brain.brain import Brain
        from origin_brain.models import BrainConfig
        brain = Brain(config=BrainConfig(agent_id="gate-test"))
        assert hasattr(brain, 'rl_gate')
        assert isinstance(brain.rl_gate, RLEncodingGate)

    def test_encode_through_gate(self):
        from origin_brain.brain import Brain
        from origin_brain.models import BrainConfig
        brain = Brain(config=BrainConfig(agent_id="gate-encode"))

        result = brain.encode("Alice discovered a new quantum algorithm")
        # Should still encode (gate in early heuristic mode encodes novel content)
        assert result.action in ('encoded', 'schema_matched', 'reconsolidated', 'duplicate')

    def test_gate_stats_in_brain(self):
        from origin_brain.brain import Brain
        from origin_brain.models import BrainConfig
        brain = Brain(config=BrainConfig(agent_id="gate-stats"))
        brain.encode("Test content for gate statistics")
        stats = brain.get_statistics()
        assert "rl_gate" in stats
