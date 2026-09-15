"""Tests for Origin Brain data models."""
import math
from datetime import datetime, timedelta, timezone

from origin_brain.models import (
    BrainConfig,
    EpisodicMemory,
    MemoryStatus,
    MemoryTier,
    MemoryType,
    SemanticMemory,
    ProceduralMemory,
    ConsolidationResult,
)


def test_episodic_memory_creation():
    mem = EpisodicMemory(content="User said hello")
    assert mem.content == "User said hello"
    assert mem.salience == 0.5
    assert mem.stability == 1.0
    assert mem.retrievability == 1.0
    assert mem.access_count == 0
    assert mem.memory_type == MemoryType.EPISODIC
    assert mem.status == MemoryStatus.ACTIVE
    assert isinstance(mem.id, str)
    assert len(mem.id) > 0


def test_compute_retrievability_fresh():
    mem = EpisodicMemory(content="Fresh memory")
    r = mem.compute_retrievability()
    # Just created, should be close to 1.0 (times importance_boost)
    assert r > 0.9


def test_compute_retrievability_decayed():
    old_time = datetime.now(timezone.utc) - timedelta(days=7)
    mem = EpisodicMemory(
        content="Old memory",
        last_accessed=old_time,
        stability=1.0,  # stability in days
    )
    r = mem.compute_retrievability()
    # 7 days with stability=1.0 => e^(-7/1) * 1.5 ≈ 0.00137
    assert r < 0.5


def test_on_retrieval_increases_stability():
    mem = EpisodicMemory(content="Test memory", stability=1.0)
    old_stability = mem.stability
    old_access_count = mem.access_count

    mem.on_retrieval()

    assert mem.stability > old_stability
    assert mem.access_count == old_access_count + 1


def test_is_alive():
    # Fresh memory should be alive
    fresh = EpisodicMemory(content="Fresh")
    assert fresh.is_alive()

    # Very old memory should be dead
    very_old = EpisodicMemory(
        content="Ancient",
        last_accessed=datetime.now(timezone.utc) - timedelta(days=365),
        stability=1.0,
    )
    assert not very_old.is_alive()


def test_semantic_memory_creation():
    mem = SemanticMemory(
        key="user_name",
        value="Alice",
        confidence=0.95,
        category="preference",
    )
    assert mem.key == "user_name"
    assert mem.value == "Alice"
    assert mem.confidence == 0.95
    assert mem.category == "preference"
    assert mem.access_count == 0
    assert isinstance(mem.id, str)


def test_procedural_memory_creation():
    mem = ProceduralMemory(
        name="code_review",
        description="How to review code",
        steps=["Check tests", "Review logic"],
        trigger_conditions=["review code"],
    )
    assert mem.name == "code_review"
    assert len(mem.steps) == 2
    assert mem.success_rate == 0.0


def test_brain_config_defaults():
    config = BrainConfig(agent_id="test_agent")
    assert config.agent_id == "test_agent"
    assert config.working_memory_tokens == 128000
    assert config.decay_model == "ebbinghaus_weighted"
    assert config.initial_stability == 1.0
    assert config.survival_threshold == 0.05
    assert config.episodic_capacity == 1_000_000
    assert config.embedding_dimension == 1536


def test_episodic_to_semantic():
    ep = EpisodicMemory(content="User likes Python", salience=0.8)
    sem = ep.to_semantic()
    assert sem.value == "User likes Python"
    assert sem.confidence == 0.8
    assert ep.id in sem.source_episodes


def test_consolidation_result():
    result = ConsolidationResult(
        consolidated_count=10,
        evicted_count=3,
        archived_count=0,
        new_semantic_count=5,
        duration_seconds=1.2,
    )
    assert result.consolidated_count == 10
    assert result.evicted_count == 3
    assert result.new_semantic_count == 5
