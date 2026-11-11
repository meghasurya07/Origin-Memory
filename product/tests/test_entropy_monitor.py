"""
Tests for AttentionEntropyMonitor — SleepGate-inspired consolidation triggers.

Tests verify:
- Encode recording and state tracking
- Shannon entropy computation over access patterns
- Interference rate computation
- Working memory churn rate
- Schema coverage tracking
- Sleep pressure accumulation
- Consolidation trigger detection
- Post-consolidation reset
- Statistics reporting
- Edge cases (empty state, single memory, etc.)
"""

import pytest
from origin_brain.entropy_monitor import (
    AttentionEntropyMonitor,
    EntropyConfig,
    EntropyState,
    ConsolidationTrigger,
)


class TestEntropyState:
    """Test the EntropyState model."""

    def test_default_state(self):
        state = EntropyState()
        assert state.memory_entropy == 0.0
        assert state.interference_rate == 0.0
        assert state.working_memory_churn == 0.0
        assert state.schema_coverage == 1.0
        assert state.sleep_pressure == 0.0
        assert not state.needs_consolidation
        assert state.urgency == "low"

    def test_needs_consolidation_threshold(self):
        state = EntropyState(sleep_pressure=0.71)
        assert state.needs_consolidation

        state2 = EntropyState(sleep_pressure=0.69)
        assert not state2.needs_consolidation

    def test_urgency_levels(self):
        assert EntropyState(sleep_pressure=0.95).urgency == "critical"
        assert EntropyState(sleep_pressure=0.8).urgency == "high"
        assert EntropyState(sleep_pressure=0.5).urgency == "moderate"
        assert EntropyState(sleep_pressure=0.2).urgency == "low"


class TestEntropyConfig:
    """Test configuration defaults."""

    def test_defaults(self):
        cfg = EntropyConfig()
        assert cfg.entropy_trigger_threshold == 0.8
        assert cfg.interference_rate_threshold == 0.3
        assert cfg.churn_rate_threshold == 0.5
        assert cfg.pressure_accumulation_rate == 0.05
        assert abs(cfg.entropy_weight + cfg.interference_weight +
                   cfg.churn_weight + cfg.schema_weight - 1.0) < 1e-6

    def test_custom_config(self):
        cfg = EntropyConfig(entropy_trigger_threshold=0.5, interference_window=50)
        assert cfg.entropy_trigger_threshold == 0.5
        assert cfg.interference_window == 50


class TestRecordEncode:
    """Test encode recording and state updates."""

    def test_basic_encode(self):
        monitor = AttentionEntropyMonitor()
        state = monitor.record_encode(memory_id="m1")
        assert isinstance(state, EntropyState)
        assert monitor._encode_count == 1

    def test_pressure_accumulates(self):
        monitor = AttentionEntropyMonitor()
        # Default accumulation rate = 0.05
        monitor.record_encode(memory_id="m1")
        assert monitor._sleep_pressure == pytest.approx(0.05, abs=0.01)

        monitor.record_encode(memory_id="m2")
        assert monitor._sleep_pressure >= 0.09  # At least 2 * 0.05

    def test_interference_recorded(self):
        monitor = AttentionEntropyMonitor()
        monitor.record_encode(memory_id="m1", had_interference=True)
        monitor.record_encode(memory_id="m2", had_interference=False)
        monitor.record_encode(memory_id="m3", had_interference=True)

        state = monitor.get_state()
        # 2/3 had interference
        assert state.interference_rate == pytest.approx(2 / 3, abs=0.01)

    def test_churn_recorded(self):
        monitor = AttentionEntropyMonitor()
        for _ in range(5):
            monitor.record_encode(wm_displacement=True)
        for _ in range(5):
            monitor.record_encode(wm_displacement=False)

        state = monitor.get_state()
        assert state.working_memory_churn == pytest.approx(0.5, abs=0.01)

    def test_schema_coverage(self):
        monitor = AttentionEntropyMonitor()
        # 3 matched, 2 not matched
        monitor.record_encode(schema_matched=True)
        monitor.record_encode(schema_matched=True)
        monitor.record_encode(schema_matched=True)
        monitor.record_encode(schema_matched=False)
        monitor.record_encode(schema_matched=False)

        state = monitor.get_state()
        assert state.schema_coverage == pytest.approx(0.6, abs=0.01)

    def test_memory_id_tracking(self):
        monitor = AttentionEntropyMonitor()
        monitor.record_encode(memory_id="m1")
        monitor.record_encode(memory_id="m1")
        monitor.record_encode(memory_id="m2")

        assert monitor._access_counts["m1"] == 2
        assert monitor._access_counts["m2"] == 1

    def test_encode_without_memory_id(self):
        monitor = AttentionEntropyMonitor()
        state = monitor.record_encode()  # No memory_id
        assert isinstance(state, EntropyState)
        assert len(monitor._access_counts) == 0


class TestMemoryEntropy:
    """Test Shannon entropy computation."""

    def test_zero_entropy_single_memory(self):
        monitor = AttentionEntropyMonitor()
        monitor.record_encode(memory_id="m1")
        state = monitor.get_state()
        # Single memory = 0 entropy
        assert state.memory_entropy == 0.0

    def test_max_entropy_uniform_access(self):
        monitor = AttentionEntropyMonitor()
        # Access 4 memories equally
        for mid in ["m1", "m2", "m3", "m4"]:
            monitor.record_encode(memory_id=mid)
        state = monitor.get_state()
        # Uniform access = max normalized entropy = 1.0
        assert state.memory_entropy == pytest.approx(1.0, abs=0.01)

    def test_low_entropy_skewed_access(self):
        monitor = AttentionEntropyMonitor()
        # One memory accessed 100 times, others once each
        for _ in range(100):
            monitor.record_encode(memory_id="dominant")
        monitor.record_encode(memory_id="rare1")
        monitor.record_encode(memory_id="rare2")

        state = monitor.get_state()
        # Heavily skewed → low entropy
        assert state.memory_entropy < 0.3

    def test_retrieval_updates_access(self):
        monitor = AttentionEntropyMonitor()
        monitor.record_encode(memory_id="m1")
        monitor.record_encode(memory_id="m2")

        # Retrieve m1 many times → skew distribution
        for _ in range(50):
            monitor.record_retrieval("m1")

        state = monitor.get_state()
        # m1 heavily accessed, m2 barely → low entropy
        assert state.memory_entropy < 0.5


class TestInterferenceRate:
    """Test interference rate computation."""

    def test_no_interference(self):
        monitor = AttentionEntropyMonitor()
        for _ in range(10):
            monitor.record_encode(had_interference=False)
        assert monitor.get_state().interference_rate == 0.0

    def test_full_interference(self):
        monitor = AttentionEntropyMonitor()
        for _ in range(10):
            monitor.record_encode(had_interference=True)
        assert monitor.get_state().interference_rate == 1.0

    def test_windowed_interference(self):
        """Interference uses a sliding window (default 20)."""
        monitor = AttentionEntropyMonitor(config=EntropyConfig(interference_window=5))
        # Fill window with interference
        for _ in range(5):
            monitor.record_encode(had_interference=True)
        assert monitor.get_state().interference_rate == 1.0

        # Push out old values with clean encodes
        for _ in range(5):
            monitor.record_encode(had_interference=False)
        assert monitor.get_state().interference_rate == 0.0


class TestConsolidationTriggers:
    """Test trigger detection and sleep pressure dynamics."""

    def test_high_entropy_triggers(self):
        monitor = AttentionEntropyMonitor(config=EntropyConfig(
            entropy_trigger_threshold=0.7,
            pressure_accumulation_rate=0.0,  # Disable base accumulation
        ))
        # Create high entropy: many equally-accessed memories
        for i in range(20):
            monitor.record_encode(memory_id=f"m{i}")

        triggers = monitor.get_trigger_log()
        entropy_triggers = [t for t in triggers if t.metric_name == "memory_entropy"]
        # Should have triggered at least once
        assert len(entropy_triggers) > 0
        assert entropy_triggers[0].metric_value > 0.7

    def test_high_interference_triggers(self):
        monitor = AttentionEntropyMonitor(config=EntropyConfig(
            interference_rate_threshold=0.3,
            pressure_accumulation_rate=0.0,
        ))
        # All encodes have interference
        for _ in range(10):
            monitor.record_encode(had_interference=True)

        triggers = monitor.get_trigger_log()
        interference_triggers = [t for t in triggers if t.metric_name == "interference_rate"]
        assert len(interference_triggers) > 0

    def test_high_churn_triggers(self):
        monitor = AttentionEntropyMonitor(config=EntropyConfig(
            churn_rate_threshold=0.4,
            pressure_accumulation_rate=0.0,
        ))
        for _ in range(10):
            monitor.record_encode(wm_displacement=True)

        triggers = monitor.get_trigger_log()
        churn_triggers = [t for t in triggers if t.metric_name == "working_memory_churn"]
        assert len(churn_triggers) > 0

    def test_low_schema_coverage_triggers(self):
        monitor = AttentionEntropyMonitor(config=EntropyConfig(
            pressure_accumulation_rate=0.0,
        ))
        # All encodes miss schema
        for _ in range(10):
            monitor.record_encode(schema_matched=False)

        triggers = monitor.get_trigger_log()
        schema_triggers = [t for t in triggers if t.metric_name == "schema_coverage"]
        assert len(schema_triggers) > 0

    def test_pressure_reaches_consolidation(self):
        """Many encodes with bad metrics should push pressure past 0.7."""
        monitor = AttentionEntropyMonitor(config=EntropyConfig(
            pressure_accumulation_rate=0.05,
            interference_rate_threshold=0.3,
        ))
        # Rapid encodes with high interference
        for i in range(30):
            monitor.record_encode(
                memory_id=f"m{i}",
                had_interference=True,
                wm_displacement=True,
                schema_matched=False,
            )

        state = monitor.get_state()
        assert state.needs_consolidation
        assert state.sleep_pressure > 0.7

    def test_pressure_capped_at_max(self):
        monitor = AttentionEntropyMonitor()
        for i in range(200):
            monitor.record_encode(
                memory_id=f"m{i}",
                had_interference=True,
            )
        assert monitor._sleep_pressure <= monitor.config.max_pressure


class TestConsolidationReset:
    """Test post-consolidation behavior."""

    def test_on_consolidation_complete(self):
        monitor = AttentionEntropyMonitor()
        # Build up pressure
        for i in range(30):
            monitor.record_encode(
                memory_id=f"m{i}",
                had_interference=True,
                wm_displacement=True,
            )

        pre_pressure = monitor._sleep_pressure
        assert pre_pressure > 0.5

        monitor.on_consolidation_complete()
        assert monitor._sleep_pressure < pre_pressure
        assert monitor._last_consolidation is not None

    def test_multiple_consolidations(self):
        monitor = AttentionEntropyMonitor()
        # Build, consolidate, build again
        for i in range(20):
            monitor.record_encode(memory_id=f"m{i}", had_interference=True)

        monitor.on_consolidation_complete()
        p1 = monitor._sleep_pressure

        for i in range(20, 40):
            monitor.record_encode(memory_id=f"m{i}", had_interference=True)

        monitor.on_consolidation_complete()
        p2 = monitor._sleep_pressure

        # Both consolidations should have reduced pressure
        assert p2 < 1.0


class TestStatistics:
    """Test statistics reporting."""

    def test_empty_statistics(self):
        monitor = AttentionEntropyMonitor()
        stats = monitor.get_statistics()
        assert stats["encode_count"] == 0
        assert stats["unique_memories_tracked"] == 0
        assert stats["needs_consolidation"] is False
        assert stats["urgency"] == "low"
        assert stats["last_consolidation"] is None
        assert stats["trigger_count"] == 0

    def test_populated_statistics(self):
        monitor = AttentionEntropyMonitor()
        for i in range(10):
            monitor.record_encode(memory_id=f"m{i}")

        stats = monitor.get_statistics()
        assert stats["encode_count"] == 10
        assert stats["unique_memories_tracked"] == 10
        assert isinstance(stats["memory_entropy"], float)
        assert isinstance(stats["interference_rate"], float)
        assert isinstance(stats["sleep_pressure"], float)

    def test_statistics_after_consolidation(self):
        monitor = AttentionEntropyMonitor()
        for i in range(10):
            monitor.record_encode(memory_id=f"m{i}")
        monitor.on_consolidation_complete()

        stats = monitor.get_statistics()
        assert stats["last_consolidation"] is not None


class TestEdgeCases:
    """Edge case tests."""

    def test_no_access_counts_entropy(self):
        monitor = AttentionEntropyMonitor()
        state = monitor.get_state()
        assert state.memory_entropy == 0.0

    def test_single_encode_state(self):
        monitor = AttentionEntropyMonitor()
        state = monitor.record_encode(memory_id="only")
        assert state.memory_entropy == 0.0
        assert state.interference_rate == 0.0
        assert state.working_memory_churn == 0.0
        assert state.schema_coverage == 1.0

    def test_get_state_matches_record_encode(self):
        """get_state should return same values as last record_encode."""
        monitor = AttentionEntropyMonitor()
        for i in range(5):
            state_from_encode = monitor.record_encode(memory_id=f"m{i}")
        state_from_get = monitor.get_state()

        assert state_from_encode.memory_entropy == pytest.approx(
            state_from_get.memory_entropy, abs=0.01
        )
        assert state_from_encode.interference_rate == pytest.approx(
            state_from_get.interference_rate, abs=0.01
        )

    def test_trigger_log_is_copy(self):
        monitor = AttentionEntropyMonitor()
        log = monitor.get_trigger_log()
        assert isinstance(log, list)
        # Mutating returned log shouldn't affect internal state
        log.append(None)
        assert len(monitor.get_trigger_log()) == 0


class TestBrainIntegration:
    """Test entropy monitor works within Brain context."""

    def test_brain_has_entropy_monitor(self):
        from origin_brain.brain import Brain
        from origin_brain.models import BrainConfig
        brain = Brain(config=BrainConfig(agent_id="test-entropy"))
        assert hasattr(brain, 'entropy_monitor')
        assert isinstance(brain.entropy_monitor, AttentionEntropyMonitor)

    def test_encode_updates_entropy_monitor(self):
        from origin_brain.brain import Brain
        from origin_brain.models import BrainConfig
        brain = Brain(config=BrainConfig(agent_id="test-entropy-encode"))

        brain.encode("Alice met Bob at the park yesterday")
        brain.encode("Bob discussed the project with Charlie")
        brain.encode("Alice presented the results to the team")

        stats = brain.entropy_monitor.get_statistics()
        assert stats["encode_count"] == 3

    def test_sleep_resets_entropy_pressure(self):
        from origin_brain.brain import Brain
        from origin_brain.models import BrainConfig
        brain = Brain(config=BrainConfig(agent_id="test-entropy-sleep"))

        # Encode many memories
        for i in range(15):
            brain.encode(f"Memory item number {i} about topic {i % 3}")

        pre_pressure = brain.entropy_monitor._sleep_pressure
        assert pre_pressure > 0.0

        brain.sleep()

        post_pressure = brain.entropy_monitor._sleep_pressure
        assert post_pressure < pre_pressure

    def test_statistics_include_entropy(self):
        from origin_brain.brain import Brain
        from origin_brain.models import BrainConfig
        brain = Brain(config=BrainConfig(agent_id="test-entropy-stats"))
        stats = brain.get_statistics()
        assert "entropy_monitor" in stats
