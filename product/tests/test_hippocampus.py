"""Tests for Origin Brain hippocampal engine."""
from datetime import datetime, timedelta, timezone

from origin_brain.models import BrainConfig, EpisodicMemory, MemoryType
from origin_brain.decay import DecayEngine
from origin_brain.hippocampus import HippocampalEngine


def _make_engine() -> HippocampalEngine:
    config = BrainConfig(agent_id="test-agent", embedding_dimension=64)
    decay = DecayEngine(model_type="ebbinghaus_weighted")
    return HippocampalEngine(config=config, decay_engine=decay)


def test_encode_basic():
    engine = _make_engine()
    mem = engine.encode("User prefers dark mode")
    assert mem is not None
    assert mem.content == "User prefers dark mode"
    assert len(engine) == 1


def test_encode_duplicate_rejected():
    engine = _make_engine()
    mem1 = engine.encode("User prefers dark mode")
    mem2 = engine.encode("User prefers dark mode")  # Exact duplicate
    assert mem1 is not None
    # Second should be rejected as not novel
    assert mem2 is None
    assert len(engine) == 1


def test_encode_different_accepted():
    engine = _make_engine()
    mem1 = engine.encode("User prefers dark mode")
    mem2 = engine.encode("Meeting scheduled for Friday at 3pm")
    assert mem1 is not None
    assert mem2 is not None
    assert len(engine) == 2


def test_retrieve_basic():
    engine = _make_engine()
    engine.encode("User loves Python programming")
    engine.encode("Meeting on Friday at 3pm")
    engine.encode("Weather is sunny today")

    results = engine.retrieve("Python")
    assert len(results) > 0
    # The Python-related memory should be in the results
    contents = [r.memory.content for r in results]
    assert any("Python" in c for c in contents)


def test_retrieve_updates_access_count():
    engine = _make_engine()
    mem = engine.encode("Important fact to remember")
    assert mem is not None
    assert mem.access_count == 0

    results = engine.retrieve("Important fact")
    # After retrieval, access count should increase
    stored_mem = engine._episodic_store[mem.id]
    assert stored_mem.access_count >= 1


def test_salience_scoring():
    engine = _make_engine()
    score_low = engine._score_salience("hello")
    score_high = engine._score_salience("This is very important! Remember this always!")
    assert score_high > score_low


def test_pattern_separation():
    engine = _make_engine()
    emb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sparse = engine._pattern_separate(emb, sparsity=0.3)

    # Sparse embedding should have many zeros
    zero_count = sum(1 for x in sparse if x == 0.0)
    assert zero_count > 0
    # But not all zeros
    assert zero_count < len(sparse)


def test_recency_score():
    engine = _make_engine()
    now = datetime.now(timezone.utc)
    score_now = engine._recency_score(now)
    score_old = engine._recency_score(now - timedelta(days=30))

    assert score_now > score_old
    assert score_now > 0.5
    assert score_old < 0.5


def test_eviction():
    engine = _make_engine()
    # Encode some memories
    for i in range(5):
        engine.encode(f"Memory content number {i} with some unique text {i*100}")

    # Manually age some memories
    old_time = datetime.now(timezone.utc) - timedelta(days=365)
    for mem_id, mem in list(engine._episodic_store.items())[:3]:
        mem.last_accessed = old_time
        mem.stability = 0.001  # Very unstable

    evicted = engine.evict(threshold=0.05)
    assert len(evicted) >= 1  # At least some should be evicted


def test_select_for_consolidation():
    engine = _make_engine()
    old_time = datetime.now(timezone.utc) - timedelta(hours=48)
    for i in range(5):
        mem = engine.encode(f"Experience {i} with detailed content to make it unique enough {i*999}")
        if mem:
            mem.timestamp = old_time  # Make it old enough
            mem.last_accessed = old_time
            mem.access_count = 3
            mem.salience = 0.6

    selected = engine.select_for_consolidation(
        min_age_hours=24, min_access_count=2, min_salience=0.4
    )
    assert len(selected) > 0


def test_statistics():
    engine = _make_engine()
    engine.encode("Memory one with unique content alpha")
    engine.encode("Memory two with unique content beta")

    stats = engine.get_statistics()
    assert stats["total_memories"] >= 2
    assert "avg_retrievability" in stats or "avg_salience" in stats


def test_contains():
    engine = _make_engine()
    mem = engine.encode("Test containment check")
    assert mem is not None
    assert mem.id in engine
    assert "nonexistent-id" not in engine
