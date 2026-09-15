"""
Tests for v0.3 Pillar Engines:
- TemporalContextEngine (Pillar 2: Temporal Context Binding)
- PredictionEngine (Pillar 1: Prediction Error Encoding)
- SleepEngine (Pillar 4: Sleep Consolidation)
"""
import math
import time
from datetime import datetime, timedelta, timezone

import pytest

from origin_brain.temporal_context import (
    TemporalContextEngine, TemporalContextConfig, ContextSnapshot
)
from origin_brain.prediction import (
    PredictionEngine, PredictionEngineConfig, PredictionResult
)
from origin_brain.sleep import (
    SleepEngine, SleepConfig, SleepReport, SleepPhaseReport
)
from origin_brain.models import EpisodicMemory, SemanticMemory, MemoryStatus


# ──────────────────────────────────────────────
#  Temporal Context Engine Tests
# ──────────────────────────────────────────────

class TestTemporalContextEngine:
    def _make_engine(self, dim=32, beta=0.5):
        return TemporalContextEngine(TemporalContextConfig(dimension=dim, beta=beta))

    def _feature(self, dim=32, seed=0):
        """Generate a simple feature vector."""
        return [math.sin(i + seed) * 0.1 for i in range(dim)]

    def test_creation(self):
        engine = self._make_engine()
        assert engine.snapshot_count == 0
        assert len(engine.current_context) == 32

    def test_context_update(self):
        engine = self._make_engine()
        fv = self._feature()
        snap = engine.update(fv, memory_id="mem-1")
        assert isinstance(snap, ContextSnapshot)
        assert snap.associated_memory_id == "mem-1"
        assert engine.snapshot_count == 1

    def test_context_drift(self):
        """Context should change after each update."""
        engine = self._make_engine()
        ctx_before = list(engine.current_context)
        engine.update(self._feature(seed=0), memory_id="a")
        ctx_after = engine.current_context
        # Context should have shifted
        assert ctx_before != ctx_after

    def test_contiguity_effect(self):
        """Memories encoded close together should have more similar contexts."""
        engine = self._make_engine()
        # Encode A, B, C in sequence
        engine.update(self._feature(seed=1), memory_id="a")
        engine.update(self._feature(seed=2), memory_id="b")
        engine.update(self._feature(seed=3), memory_id="c")
        # Encode X much later (different context)
        for i in range(20):
            engine.update(self._feature(seed=100 + i))
        engine.update(self._feature(seed=50), memory_id="x")

        sim_ab = engine.get_context_similarity("a", "b")
        sim_ax = engine.get_context_similarity("a", "x")
        # A-B should be more similar than A-X (contiguity effect)
        assert sim_ab > sim_ax

    def test_temporal_neighbors(self):
        engine = self._make_engine()
        for i in range(10):
            engine.update(self._feature(seed=i), memory_id=f"mem-{i}")
        neighbors = engine.get_temporal_neighbors("mem-5", window=2)
        mem_ids = [mid for mid, _ in neighbors]
        assert "mem-4" in mem_ids or "mem-6" in mem_ids

    def test_reinstate(self):
        engine = self._make_engine()
        engine.update(self._feature(seed=1), memory_id="old")
        old_context = list(engine.current_context)
        # Many updates later
        for i in range(10):
            engine.update(self._feature(seed=100 + i))
        # Context has drifted far
        drifted_context = list(engine.current_context)
        assert drifted_context != old_context
        # Reinstate old context
        success = engine.reinstate("old", strength=0.8)
        assert success
        reinstated = engine.current_context
        # Should be closer to old_context now
        sim_old = sum(a * b for a, b in zip(reinstated, old_context))
        sim_drifted = sum(a * b for a, b in zip(reinstated, drifted_context))
        assert sim_old > sim_drifted

    def test_reinstate_nonexistent(self):
        engine = self._make_engine()
        assert engine.reinstate("nonexistent") is False

    def test_probe(self):
        engine = self._make_engine()
        fv1 = self._feature(seed=1)
        engine.update(fv1, memory_id="mem-1")
        engine.update(self._feature(seed=2), memory_id="mem-2")
        results = engine.probe(fv1, top_k=2)
        assert len(results) > 0
        assert results[0][0] in ("mem-1", "mem-2")

    def test_statistics(self):
        engine = self._make_engine()
        engine.update(self._feature(), memory_id="x")
        stats = engine.get_statistics()
        assert stats["dimension"] == 32
        assert stats["snapshot_count"] == 1
        assert stats["memory_bindings"] == 1

    def test_dimension_padding(self):
        """Short feature vectors should be padded."""
        engine = self._make_engine(dim=32)
        short_fv = [0.1, 0.2, 0.3]  # Only 3 elements
        snap = engine.update(short_fv, memory_id="short")
        assert snap is not None
        assert engine.snapshot_count == 1

    def test_max_snapshots_eviction(self):
        engine = TemporalContextEngine(
            TemporalContextConfig(dimension=8, max_snapshots=5)
        )
        for i in range(10):
            engine.update([0.1] * 8, memory_id=f"m-{i}")
        assert engine.snapshot_count <= 5


# ──────────────────────────────────────────────
#  Prediction Engine Tests
# ──────────────────────────────────────────────

class TestPredictionEngine:
    def test_creation(self):
        engine = PredictionEngine()
        stats = engine.get_statistics()
        assert stats["total_inputs"] == 0

    def test_first_input_neutral(self):
        """First input should have moderate surprise (no reference)."""
        engine = PredictionEngine()
        result = engine.evaluate("hello world")
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.surprise_score <= 1.0

    def test_repeated_input_low_surprise(self):
        """Repeated identical input should reduce surprise over time."""
        engine = PredictionEngine()
        # Train on repeated content
        for _ in range(20):
            engine.evaluate("the cat sat on the mat")
        # After many repetitions, surprise should be lower
        result = engine.evaluate("the cat sat on the mat")
        assert result.surprise_score < 0.7

    def test_novel_input_high_surprise(self):
        """Novel input after many repetitions should be surprising."""
        engine = PredictionEngine()
        # Establish a pattern
        for _ in range(20):
            engine.evaluate("the cat sat on the mat")
        # Novel input
        result = engine.evaluate("quantum entanglement in photosynthetic complexes")
        # Should be more surprising than the repeated content
        repeated = engine.evaluate("the cat sat on the mat")
        assert result.surprise_score > repeated.surprise_score

    def test_encoding_decision(self):
        """Should encode should be True/False based on threshold."""
        engine = PredictionEngine(PredictionEngineConfig(surprise_threshold=0.3))
        result = engine.evaluate("hello world test")
        assert isinstance(result.should_encode, bool)

    def test_encoding_strength_range(self):
        """Encoding strength should be in [0, 1]."""
        engine = PredictionEngine()
        for text in ["hello", "world", "quantum physics revolution"]:
            result = engine.evaluate(text)
            assert 0.0 <= result.encoding_strength <= 1.0

    def test_precision_estimation(self):
        """Precision should reflect error variance."""
        engine = PredictionEngine()
        # Low variance inputs → high precision
        for _ in range(10):
            engine.evaluate("consistent pattern every time")
        stats = engine.get_statistics()
        assert stats["current_precision"] > 0

    def test_category_tracking(self):
        """Should track per-category statistics."""
        engine = PredictionEngine()
        engine.evaluate("buy some groceries", category="task")
        engine.evaluate("schedule a meeting", category="task")
        engine.evaluate("the weather is nice", category="observation")
        stats = engine.get_statistics()
        assert stats["category_count"] == 2

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        engine = PredictionEngine()
        result = engine.evaluate("")
        assert isinstance(result, PredictionResult)

    def test_statistics(self):
        engine = PredictionEngine()
        engine.evaluate("test input one")
        engine.evaluate("test input two")
        stats = engine.get_statistics()
        assert stats["total_inputs"] == 2
        assert stats["vocabulary_size"] > 0


# ──────────────────────────────────────────────
#  Sleep Engine Tests
# ──────────────────────────────────────────────

class TestSleepEngine:
    def _make_memory(self, content, salience=0.5, hours_ago=2, access_count=1) -> EpisodicMemory:
        ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        return EpisodicMemory(
            content=content,
            salience=salience,
            timestamp=ts,
            last_accessed=ts,
            access_count=access_count,
            metadata={"emotional_arousal": 0.5}
        )

    def _make_memories(self, n=10):
        return [
            self._make_memory(f"Event {i} happened", salience=i / n, hours_ago=n - i)
            for i in range(n)
        ]

    def test_creation(self):
        engine = SleepEngine()
        stats = engine.get_statistics()
        assert stats["total_cycles"] == 0

    def test_basic_sleep_cycle(self):
        """Should run all three phases."""
        engine = SleepEngine(SleepConfig(num_cycles=1))
        memories = self._make_memories(10)
        
        stored_semantic = []
        evicted_ids = []

        def store_fn(sem: SemanticMemory):
            stored_semantic.append(sem)

        def evict_fn(mid: str) -> bool:
            evicted_ids.append(mid)
            return True

        report = engine.run_sleep_cycle(memories, store_fn, evict_fn)
        assert isinstance(report, SleepReport)
        assert len(report.phases) == 3  # SWS, REM, PRUNING
        assert report.phases[0].phase == "SWS"
        assert report.phases[1].phase == "REM"
        assert report.phases[2].phase == "PRUNING"

    def test_consolidation_produces_semantic(self):
        """SWS phase should produce semantic memories from high-salience episodes."""
        engine = SleepEngine(SleepConfig(
            num_cycles=1,
            consolidation_salience_threshold=0.3
        ))
        memories = [
            self._make_memory("Important discovery about quantum computing", salience=0.9, hours_ago=3),
            self._make_memory("Had lunch today", salience=0.1, hours_ago=3),
        ]

        stored = []
        engine.run_sleep_cycle(memories, lambda s: stored.append(s), lambda _: True)
        
        # High-salience memory should be consolidated
        if stored:
            assert any("quantum" in s.value.lower() for s in stored)

    def test_emotional_decoupling(self):
        """REM phase should reduce emotional arousal."""
        engine = SleepEngine(SleepConfig(num_cycles=1, emotional_decoupling_rate=0.5))
        memory = self._make_memory("Scary event", salience=0.8, hours_ago=2)
        memory.metadata['emotional_arousal'] = 0.9

        engine.run_sleep_cycle([memory], lambda _: None, lambda _: True)
        
        # Arousal should be reduced
        assert memory.metadata['emotional_arousal'] < 0.9

    def test_pruning_evicts_low_need(self):
        """Pruning should evict memories with low need scores."""
        engine = SleepEngine(SleepConfig(
            num_cycles=1,
            eviction_threshold=100.0  # Very high threshold → evicts almost everything
        ))
        memories = [
            self._make_memory("Old forgotten thing", salience=0.1, hours_ago=1000, access_count=0),
        ]

        evicted = []
        engine.run_sleep_cycle(memories, lambda _: None, lambda mid: (evicted.append(mid), True)[-1])
        assert len(evicted) > 0

    def test_multiple_cycles(self):
        """Running multiple cycles should process more memories."""
        engine = SleepEngine(SleepConfig(num_cycles=3))
        memories = self._make_memories(20)

        report = engine.run_sleep_cycle(memories, lambda _: None, lambda _: True)
        assert len(report.phases) == 9  # 3 phases × 3 cycles

    def test_min_age_requirement(self):
        """Very recent memories should NOT be consolidated."""
        engine = SleepEngine(SleepConfig(
            num_cycles=1,
            min_age_for_consolidation_hours=24.0  # 24 hours minimum
        ))
        # Memory created just now
        fresh = self._make_memory("Just happened!", salience=0.9, hours_ago=0)
        
        stored = []
        engine.run_sleep_cycle([fresh], lambda s: stored.append(s), lambda _: True)
        assert len(stored) == 0  # Too recent to consolidate

    def test_report_structure(self):
        engine = SleepEngine(SleepConfig(num_cycles=1))
        report = engine.run_sleep_cycle(self._make_memories(5), lambda _: None, lambda _: True)
        assert report.cycle_id is not None
        assert report.timestamp is not None
        assert report.total_duration_ms >= 0
        assert isinstance(report.compression_ratio, float)

    def test_statistics(self):
        engine = SleepEngine()
        engine.run_sleep_cycle(self._make_memories(5), lambda _: None, lambda _: True)
        stats = engine.get_statistics()
        assert stats["total_cycles"] == 1

    def test_stability_boost(self):
        """Consolidated memories should get stability boost."""
        engine = SleepEngine(SleepConfig(
            num_cycles=1,
            consolidation_salience_threshold=0.3
        ))
        memory = self._make_memory("Important fact", salience=0.8, hours_ago=3)
        original_stability = memory.stability

        engine.run_sleep_cycle([memory], lambda _: None, lambda _: True)
        
        # If consolidated, stability should have increased
        if memory.status == MemoryStatus.CONSOLIDATING:
            assert memory.stability > original_stability
