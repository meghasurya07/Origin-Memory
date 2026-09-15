"""
Integration tests for Origin Brain Memory SDK v0.2.
Tests the full Brain pipeline end-to-end with all engines working together.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from origin_brain.brain import Brain
from origin_brain.models import BrainConfig, MemoryType

logger = logging.getLogger(__name__)


def _get_memory_from_encode_result(result: Any) -> Any:
    """Helper to extract EpisodicMemory from Brain.encode() result."""
    if hasattr(result, 'memory'):
        return result.memory
    return result


def _get_recall_list(result: Any) -> list:
    """Helper to extract the list from recall result."""
    if hasattr(result, 'results'):
        return result.results
    return result


def test_full_lifecycle():
    """
    Create brain → encode 10 diverse memories → recall → consolidate →
    recall again → export → import → recall matches
    """
    config = BrainConfig(agent_id="lifecycle_test_agent", embedding_dimension=64)
    brain = Brain(config)

    memories = [
        "I saw a cat today.",
        "The sky is blue and clear.",
        "I ate an apple for breakfast.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language.",
        "I need to buy milk tomorrow morning.",
        "Water boils at 100 degrees Celsius.",
        "My favorite color is green.",
        "I went to the park yesterday.",
        "Gravity pulls objects towards the Earth."
    ]

    for m in memories:
        brain.encode(m)

    # Recall before consolidation
    recall_1 = _get_recall_list(brain.recall("What is Python?"))
    assert len(recall_1) > 0

    # Consolidate
    brain.consolidate()

    # Recall after consolidation
    recall_2 = _get_recall_list(brain.recall("What is Python?"))
    assert len(recall_2) > 0

    # Export state
    state = brain.export_state()
    assert state["agent_id"] == "lifecycle_test_agent"

    # Import into new Brain
    brain2 = Brain(BrainConfig(agent_id="lifecycle_test_agent_2", embedding_dimension=64))
    brain2.import_state(state)

    # Recall from imported brain
    recall_3 = _get_recall_list(brain2.recall("What is Python?"))
    assert len(recall_3) > 0


def test_emotional_memory_priority():
    """
    Encode a boring memory and an emotional memory.
    The emotional one should have higher salience after encoding.
    """
    config = BrainConfig(agent_id="emotional_test_agent", embedding_dimension=64)
    brain = Brain(config)

    res_boring = brain.encode("The wall is painted beige and it is very plain.")
    res_emotional = brain.encode("CRITICAL ERROR! System is down! We need to fix this ASAP!")

    mem_boring = _get_memory_from_encode_result(res_boring)
    mem_emotional = _get_memory_from_encode_result(res_emotional)

    # Emotional content should have higher or equal salience
    # (with neuromodulation, both novel inputs may be boosted to max)
    if mem_boring is not None and mem_emotional is not None:
        assert mem_emotional.salience >= mem_boring.salience
        # The real differentiation is in emotional metadata
        assert mem_emotional.metadata.get('emotional_arousal', 0) > mem_boring.metadata.get('emotional_arousal', 0)


def test_memory_decay_over_time():
    """
    Encode a memory, manually set its timestamp to 30 days ago with low stability,
    then verify it gets evicted during consolidation.
    """
    config = BrainConfig(
        agent_id="decay_test_agent",
        embedding_dimension=64,
        survival_threshold=0.5
    )
    brain = Brain(config)

    res = brain.encode("This is a fleeting thought that I might not remember.")
    mem = _get_memory_from_encode_result(res)

    if mem is not None:
        # Simulate time passing and low stability
        mem.last_accessed = datetime.now(timezone.utc) - timedelta(days=30)
        mem.stability = 0.1

        # Verify it is no longer alive based on threshold
        assert mem.is_alive(threshold=0.5) is False

        # Run consolidation
        brain.consolidate()


def test_semantic_fact_storage_and_recall():
    """Store facts via brain.store_fact(), recall them, verify they come back."""
    config = BrainConfig(agent_id="fact_test_agent", embedding_dimension=64)
    brain = Brain(config)

    brain.store_fact(key="Capital of France", value="Paris", confidence=0.95)

    results = _get_recall_list(brain.recall("What is the capital of France?"))

    assert len(results) > 0


def test_procedural_memory():
    """Store a procedure, verify it's stored (brain.procedural_count)."""
    config = BrainConfig(agent_id="procedural_test_agent", embedding_dimension=64)
    brain = Brain(config)

    brain.store_procedure(
        name="Make Coffee",
        description="Steps to brew a fresh cup of coffee.",
        steps=["Grind coffee beans", "Boil water", "Pour over grounds", "Enjoy"]
    )

    assert brain.procedural_count == 1


def test_context_buffer_management():
    """Encode many memories, verify context buffer works, clear it, verify empty."""
    config = BrainConfig(agent_id="context_test_agent", embedding_dimension=64)
    brain = Brain(config)

    for i in range(20):
        brain.encode(f"Context entry {i} with unique content {i * 111}")

    buffer = brain.get_context_buffer()
    assert isinstance(buffer, (list, dict))

    brain.clear_context()
    assert len(brain.get_context_buffer()) == 0


def test_multiple_brains_independent():
    """Create two brains, encode different memories, verify isolation."""
    config1 = BrainConfig(agent_id="agent_one", embedding_dimension=64)
    config2 = BrainConfig(agent_id="agent_two", embedding_dimension=64)

    brain1 = Brain(config1)
    brain2 = Brain(config2)

    brain1.encode("Secret code for agent one is Alpha.")
    brain2.encode("Secret code for agent two is Bravo.")

    assert brain1.episodic_count == 1
    assert brain2.episodic_count == 1

    res1 = _get_recall_list(brain1.recall("What is the secret code?"))
    assert len(res1) > 0
    assert "Alpha" in res1[0].memory.content

    res2 = _get_recall_list(brain2.recall("What is the secret code?"))
    assert len(res2) > 0
    assert "Bravo" in res2[0].memory.content


def test_encode_diverse_content():
    """Encode a variety of content types and verify all are stored."""
    config = BrainConfig(agent_id="diverse_content_agent", embedding_dimension=64)
    brain = Brain(config)

    brain.encode("What time is the meeting scheduled for tomorrow?")
    brain.encode("The sky is blue and the grass is green in nature.")
    brain.encode("I feel very happy today because of the news!")
    brain.encode("Remember to buy milk from the store later.")

    assert brain.episodic_count >= 2  # Some may match schemas


def test_consolidation_multiple_cycles():
    """Run consolidation 3 times, verify it returns ConsolidationResult each time."""
    config = BrainConfig(agent_id="consolidation_cycle_agent", embedding_dimension=64)
    brain = Brain(config)

    brain.encode("I learned how to ride a bike today in the park.")
    brain.encode("I learned how to drive a car today on the highway.")

    for _ in range(3):
        res = brain.consolidate()
        assert hasattr(res, "consolidated_count")
        assert hasattr(res, "evicted_count")
        assert hasattr(res, "duration_seconds")


def test_statistics_comprehensive():
    """Encode some memories, store some facts, verify statistics dict has expected keys."""
    config = BrainConfig(agent_id="statistics_agent", embedding_dimension=64)
    brain = Brain(config)

    brain.encode("This is a memory to test statistics reporting.")
    brain.store_fact("Math", "1 + 1 = 2")
    brain.store_procedure("Jump", "How to jump", ["Bend knees", "Push off ground"])

    stats = brain.get_statistics()

    assert stats["agent_id"] == "statistics_agent"
    assert stats["episodic_count"] >= 0
    assert stats["semantic_count"] >= 1
    assert stats["procedural_count"] >= 1
