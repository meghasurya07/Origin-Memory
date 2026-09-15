"""Tests for Origin Brain top-level Brain API (v0.2)."""
from origin_brain.brain import Brain
from origin_brain.models import BrainConfig, MemoryType


def _make_brain() -> Brain:
    return Brain(config={"agent_id": "test-brain", "embedding_dimension": 64})


def _recall_results(brain, query, **kwargs):
    """Helper: extract the list from RecallResult or return as-is."""
    res = brain.recall(query, **kwargs)
    if hasattr(res, 'results'):
        return res.results
    return res


def test_brain_creation():
    brain = _make_brain()
    assert brain.agent_id == "test-brain"
    assert brain.episodic_count == 0


def test_brain_encode_and_recall():
    brain = _make_brain()
    result = brain.encode("User prefers Python for backend development", salience=0.8)
    assert result is not None

    results = _recall_results(brain, "Python backend")
    assert len(results) > 0


def test_brain_store_fact():
    brain = _make_brain()
    sem = brain.store_fact(
        key="user_language",
        value="Python",
        confidence=0.9,
        category="preference",
    )
    assert sem.key == "user_language"
    assert sem.value == "Python"
    assert brain.semantic_count == 1


def test_brain_store_procedure():
    brain = _make_brain()
    proc = brain.store_procedure(
        name="deploy",
        description="Deploy to production",
        steps=["Run tests", "Build", "Deploy"],
        trigger_conditions=["deploy to prod"],
    )
    assert proc.name == "deploy"
    assert brain.procedural_count == 1


def test_brain_consolidation():
    brain = _make_brain()
    for i in range(5):
        brain.encode(f"Experience {i} with unique content {i * 777}")

    result = brain.consolidate()
    assert result is not None
    assert hasattr(result, "consolidated_count")
    assert hasattr(result, "evicted_count")


def test_brain_statistics():
    brain = _make_brain()
    brain.encode("Test memory for statistics")
    stats = brain.get_statistics()
    assert "agent_id" in stats or "episodic_count" in stats


def test_brain_export_import():
    brain = _make_brain()
    brain.encode("Memory to export")
    brain.store_fact("key1", "value1", 0.9, "fact")

    state = brain.export_state()
    assert state is not None

    new_brain = _make_brain()
    new_brain.import_state(state)
    assert new_brain.episodic_count == brain.episodic_count


def test_brain_context_buffer():
    brain = _make_brain()
    brain.encode("Turn 1 content")
    brain.encode("Turn 2 content with different stuff")

    buffer = brain.get_context_buffer()
    assert isinstance(buffer, (list, dict))

    brain.clear_context()
    assert len(brain.get_context_buffer()) == 0


def test_brain_with_config_object():
    config = BrainConfig(agent_id="config-brain", embedding_dimension=64)
    brain = Brain(config=config)
    assert brain.agent_id == "config-brain"
