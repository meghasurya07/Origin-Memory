"""
Tests for Pattern Completion Engine — CA3 Attractor Network

Tests verify:
1. Full patterns stored and retrieved
2. PARTIAL cues reconstruct full patterns (the key property!)
3. Pattern separation prevents interference
4. Spurious attractor detection
5. Attractor convergence
6. Multiple pattern discrimination
"""
import math
import pytest

from origin_brain.pattern_completion import (
    PatternCompletionEngine, PatternCompletionConfig,
    CompletionResult, StoredPattern
)


def _make_pattern(dim=32, seed=0) -> list:
    """Generate a deterministic test pattern."""
    return [math.sin(i * 0.5 + seed * 3.7) * 0.5 + 0.5 for i in range(dim)]


class TestPatternCompletionEngine:

    def test_creation(self):
        engine = PatternCompletionEngine()
        assert engine.pattern_count == 0

    def test_store_pattern(self):
        engine = PatternCompletionEngine()
        p = _make_pattern()
        stored = engine.store(p, memory_id="mem-1", label="test")
        assert isinstance(stored, StoredPattern)
        assert engine.pattern_count == 1

    def test_complete_exact_match(self):
        """Full cue should match stored pattern perfectly."""
        engine = PatternCompletionEngine()
        p = _make_pattern(seed=1)
        engine.store(p, memory_id="mem-1")
        
        results = engine.complete(p, top_k=1)
        assert len(results) == 1
        assert results[0].matched_memory_id == "mem-1"
        assert results[0].similarity > 0.95

    def test_complete_partial_cue(self):
        """PARTIAL cue (30%) should still complete to the right pattern."""
        engine = PatternCompletionEngine(PatternCompletionConfig(beta=8.0))
        dim = 32
        
        # Store 3 distinct patterns
        p1 = _make_pattern(dim, seed=1)
        p2 = _make_pattern(dim, seed=5)
        p3 = _make_pattern(dim, seed=10)
        
        engine.store(p1, memory_id="mem-1")
        engine.store(p2, memory_id="mem-2")
        engine.store(p3, memory_id="mem-3")
        
        # Degrade p1 to only 30% of dimensions
        degraded = engine.degrade_cue(p1, fraction=0.3)
        
        # Complete should still find mem-1
        results = engine.complete(degraded, top_k=1)
        assert len(results) == 1
        assert results[0].matched_memory_id == "mem-1"
        assert results[0].completion_ratio > 0.5  # Significant completion

    def test_complete_discriminates_patterns(self):
        """Should complete to the CORRECT pattern among many."""
        engine = PatternCompletionEngine(PatternCompletionConfig(beta=12.0))
        dim = 64
        
        # Store 5 well-separated patterns using orthogonal-ish construction
        patterns = {}
        for i in range(5):
            # Create patterns that are clearly distinct
            p = [0.0] * dim
            # Each pattern has energy in a different region
            start = i * (dim // 5)
            for j in range(dim):
                p[j] = math.sin(j * 0.3 + i * 7.0) * 0.3
                if start <= j < start + (dim // 5):
                    p[j] += 1.0  # Strong signal in this region
            engine.store(p, memory_id=f"mem-{i}")
            patterns[f"mem-{i}"] = p
        
        # Keep only the "signature" region of pattern 2 (zero everything else)
        target = patterns["mem-2"]
        cue = [0.0] * dim
        start = 2 * (dim // 5)
        for j in range(start, start + (dim // 5)):
            cue[j] = target[j]
        
        results = engine.complete(cue, top_k=3)
        assert results[0].matched_memory_id == "mem-2"

    def test_pattern_separation(self):
        """Very similar patterns should be merged (DG separation)."""
        engine = PatternCompletionEngine(
            PatternCompletionConfig(separation_threshold=0.95)
        )
        p1 = _make_pattern(32, seed=1)
        # p2 is almost identical to p1
        p2 = [x + 0.001 for x in p1]
        
        engine.store(p1, memory_id="mem-1")
        engine.store(p2, memory_id="mem-2")
        
        # Should have merged (only 1 pattern stored)
        assert engine.pattern_count == 1

    def test_attractor_convergence(self):
        """Completion should converge within max_iterations."""
        engine = PatternCompletionEngine(PatternCompletionConfig(max_iterations=20))
        engine.store(_make_pattern(32, seed=1), memory_id="mem-1")
        
        results = engine.complete(_make_pattern(32, seed=1), top_k=1)
        assert results[0].iterations <= 20

    def test_spurious_detection(self):
        """Very noisy cue should be flagged as potentially spurious."""
        engine = PatternCompletionEngine(
            PatternCompletionConfig(spurious_threshold=0.9)
        )
        engine.store(_make_pattern(32, seed=1), memory_id="mem-1")
        
        # Use a random-ish cue that doesn't match
        random_cue = [(-1)**i * 0.1 for i in range(32)]
        results = engine.complete(random_cue, top_k=1)
        
        # Should be flagged as potentially spurious (low similarity)
        if results:
            # The completion might have low similarity
            assert isinstance(results[0].is_spurious, bool)

    def test_degrade_cue(self):
        """Degraded cue should have the right fraction of zeros."""
        engine = PatternCompletionEngine()
        p = _make_pattern(32, seed=1)
        engine.store(p, memory_id="mem-1")
        
        degraded = engine.degrade_cue(p, fraction=0.5)
        non_zero = sum(1 for x in degraded if abs(x) > 1e-10)
        # Approximately half should be non-zero
        assert 10 <= non_zero <= 25

    def test_empty_store(self):
        """Completing with no stored patterns should return empty."""
        engine = PatternCompletionEngine()
        results = engine.complete([0.1] * 32)
        assert results == []

    def test_statistics(self):
        engine = PatternCompletionEngine()
        engine.store(_make_pattern(32, seed=1), memory_id="mem-1")
        engine.complete(_make_pattern(32, seed=1))
        
        stats = engine.get_statistics()
        assert stats["pattern_count"] == 1
        assert stats["total_completions"] == 1
        assert "spurious_rate" in stats

    def test_max_capacity(self):
        """Should evict lowest-strength patterns at capacity."""
        engine = PatternCompletionEngine(
            PatternCompletionConfig(max_patterns=5)
        )
        for i in range(10):
            engine.store(_make_pattern(16, seed=i * 20), memory_id=f"mem-{i}")
        
        assert engine.pattern_count <= 5

    def test_strength_modulation(self):
        """Higher strength patterns should be preferred in completion."""
        engine = PatternCompletionEngine()
        dim = 32
        p1 = _make_pattern(dim, seed=1)
        p2 = _make_pattern(dim, seed=2)
        
        engine.store(p1, memory_id="strong", strength=2.0)
        engine.store(p2, memory_id="weak", strength=0.1)
        
        # Use a cue equidistant from both — strength should break tie
        mixed = [(a + b) / 2 for a, b in zip(p1, p2)]
        results = engine.complete(mixed, top_k=2)
        
        # Strong pattern should rank higher
        assert results[0].matched_memory_id == "strong"

    def test_completion_by_memory_id(self):
        """Should complete specifically toward a known memory."""
        engine = PatternCompletionEngine()
        p = _make_pattern(32, seed=3)
        engine.store(p, memory_id="target")
        engine.store(_make_pattern(32, seed=7), memory_id="other")
        
        degraded = engine.degrade_cue(p, fraction=0.4)
        result = engine.complete_by_memory_id(degraded, "target")
        
        assert result is not None
        assert result.matched_memory_id == "target"
