"""Tests for Origin Brain decay engine."""
import math
from datetime import datetime, timedelta, timezone

from origin_brain.models import EpisodicMemory, BrainConfig
from origin_brain.decay import (
    DecayEngine,
    EbbinghausDecay,
    WeightedEbbinghausDecay,
    PowerLawDecay,
)


def test_ebbinghaus_decay_fresh():
    model = EbbinghausDecay()
    # At t=0, retrievability should be ~1.0
    r = model.compute_retrievability(stability=1.0, time_elapsed_seconds=0.0, salience=0.5)
    assert abs(r - 1.0) < 0.001


def test_ebbinghaus_decay_over_time():
    model = EbbinghausDecay()
    # After 1 day (86400s) with stability=86400 (1 day), R should be e^(-1) ≈ 0.368
    r = model.compute_retrievability(stability=86400.0, time_elapsed_seconds=86400.0, salience=0.5)
    assert abs(r - math.exp(-1)) < 0.01


def test_weighted_ebbinghaus_high_salience():
    model = WeightedEbbinghausDecay()
    t = 3600.0  # 1 hour
    s = 3600.0  # stability = 1 hour

    r_low = model.compute_retrievability(stability=s, time_elapsed_seconds=t, salience=0.1)
    r_high = model.compute_retrievability(stability=s, time_elapsed_seconds=t, salience=0.9)

    # Higher salience should give higher retrievability
    assert r_high > r_low


def test_power_law_decay():
    model = PowerLawDecay(a=1.0, b=0.5)
    # R(t) = a * t^(-b) = 1.0 * 100^(-0.5) = 0.1
    r = model.compute_retrievability(stability=1.0, time_elapsed_seconds=100.0, salience=0.5)
    assert abs(r - 0.1) < 0.01


def test_stability_increases_on_retrieval():
    model = EbbinghausDecay(alpha=0.3, w=-0.01)
    s_old = 1.0
    s_new = model.update_stability_on_retrieval(current_stability=s_old, access_count=1)
    assert s_new > s_old


def test_decay_engine_creation():
    engine = DecayEngine(model_type="ebbinghaus")
    assert engine.model is not None
    assert isinstance(engine.model, EbbinghausDecay)


def test_decay_engine_apply_batch():
    engine = DecayEngine(model_type="ebbinghaus_weighted")
    now = datetime.now(timezone.utc)

    memories = []
    # 5 fresh memories (should survive)
    for i in range(5):
        memories.append(EpisodicMemory(
            content=f"Fresh memory {i}",
            last_accessed=now,
            stability=86400.0,  # 1 day stability
        ))
    # 5 very old memories (should be evicted)
    for i in range(5):
        memories.append(EpisodicMemory(
            content=f"Old memory {i}",
            last_accessed=now - timedelta(days=365),
            stability=1.0,  # Very low stability
        ))

    surviving, evicted = engine.apply_decay(memories)
    assert len(surviving) == 5
    assert len(evicted) == 5


def test_reinforce_memory():
    engine = DecayEngine(model_type="ebbinghaus")
    mem = EpisodicMemory(content="Test", stability=1.0, access_count=0)
    old_stability = mem.stability

    engine.reinforce(mem)

    assert mem.stability > old_stability
    assert mem.access_count == 1


def test_batch_statistics():
    engine = DecayEngine(model_type="ebbinghaus")
    memories = [
        EpisodicMemory(content=f"Mem {i}", stability=86400.0)
        for i in range(10)
    ]

    stats = engine.batch_statistics(memories)
    assert stats["total"] == 10
    assert "alive" in stats
    assert "decayed" in stats
    assert "avg_retrievability" in stats
    assert "avg_stability" in stats
