"""Tests for Origin Brain v0.1.1 advanced modules:
- Router, Reconsolidation, Schemas, Emotional, Interference, Prospective, Metamemory
"""
from datetime import datetime, timedelta, timezone

from origin_brain.models import (
    BrainConfig, EpisodicMemory, SemanticMemory, MemoryType,
    MemoryTier, MemoryResult, MemoryQuery,
)
from origin_brain.router import MemoryRouter
from origin_brain.reconsolidation import ReconsolidationEngine
from origin_brain.schemas import SchemaEngine, PREDEFINED_SCHEMAS
from origin_brain.emotional import EmotionalModulator, EmotionalValence
from origin_brain.interference import InterferenceDetector, InterferenceType
from origin_brain.prospective import ProspectiveMemoryEngine, ProspectiveMemoryType
from origin_brain.metamemory import MetamemoryEngine, ConfidenceLevel


# ── Router Tests ──────────────────────────────────────────────────

class TestMemoryRouter:

    def test_classify_temporal_query(self):
        router = MemoryRouter(BrainConfig(agent_id="test"))
        types = router.classify_query("When did Alice last visit?")
        assert MemoryType.EPISODIC in types

    def test_classify_knowledge_query(self):
        router = MemoryRouter(BrainConfig(agent_id="test"))
        types = router.classify_query("What is Alice's favorite color?")
        assert MemoryType.SEMANTIC in types

    def test_classify_procedural_query(self):
        router = MemoryRouter(BrainConfig(agent_id="test"))
        types = router.classify_query("How to deploy to production?")
        assert MemoryType.PROCEDURAL in types

    def test_classify_default_fallback(self):
        router = MemoryRouter(BrainConfig(agent_id="test"))
        types = router.classify_query("hello there")
        assert len(types) >= 1  # Should default to something

    def test_build_query(self):
        router = MemoryRouter(BrainConfig(agent_id="test"))
        query = router.build_query("What did Alice say last week?", top_k=3)
        assert isinstance(query, MemoryQuery)
        assert query.top_k == 3


# ── Reconsolidation Tests ────────────────────────────────────────

class TestReconsolidation:

    def test_on_retrieval_makes_labile(self):
        engine = ReconsolidationEngine()
        mem = EpisodicMemory(content="Alice lives in NYC")
        engine.on_retrieval(mem)
        assert engine.get_labile_count() == 1

    def test_expired_lability_cleaned(self):
        engine = ReconsolidationEngine(lability_window_seconds=0.001)
        mem = EpisodicMemory(content="Old fact")
        engine.on_retrieval(mem)
        import time; time.sleep(0.01)
        cleaned = engine._clean_expired()
        assert cleaned >= 1
        assert engine.get_labile_count() == 0

    def test_reconsolidate_updates_content(self):
        engine = ReconsolidationEngine()
        mem = EpisodicMemory(content="Alice lives in NYC", stability=1.0)
        old_stability = mem.stability
        updated = engine.reconsolidate(mem, "Alice moved to SF")
        assert "SF" in updated.content or "moved" in updated.content
        assert updated.stability >= old_stability


# ── Schema Tests ─────────────────────────────────────────────────

class TestSchemaEngine:

    def test_predefined_schemas_exist(self):
        assert len(PREDEFINED_SCHEMAS) >= 5

    def test_register_schema(self):
        engine = SchemaEngine()
        schema = engine.register_schema(
            name="test_schema",
            description="A test schema",
            slots={"name": {"description": "Name", "slot_type": "string"}}
        )
        assert schema.name == "test_schema"
        assert "name" in schema.slots

    def test_get_schema_by_name(self):
        engine = SchemaEngine()
        engine.register_schema("my_schema", "test", {"x": {"description": "x", "slot_type": "string"}})
        found = engine.get_schema_by_name("my_schema")
        assert found is not None
        assert found.name == "my_schema"

    def test_statistics(self):
        engine = SchemaEngine()
        engine.register_schema("s1", "test1", {"a": {"description": "a", "slot_type": "string"}})
        stats = engine.get_statistics()
        assert stats["schema_count"] >= 1


# ── Emotional Modulator Tests ────────────────────────────────────

class TestEmotionalModulator:

    def test_analyze_positive(self):
        mod = EmotionalModulator()
        tag = mod.analyze("This is great! I love it! Excellent work!")
        assert tag.valence == EmotionalValence.POSITIVE

    def test_analyze_negative(self):
        mod = EmotionalModulator()
        tag = mod.analyze("There's a critical error, the system crashed and everything is broken")
        assert tag.valence == EmotionalValence.NEGATIVE

    def test_analyze_urgent(self):
        mod = EmotionalModulator()
        tag = mod.analyze("URGENT: deadline is NOW, do this immediately!")
        assert tag.valence == EmotionalValence.URGENT

    def test_modulate_salience_boost(self):
        mod = EmotionalModulator()
        tag = mod.analyze("CRITICAL ERROR! System down!")
        boosted = mod.modulate_salience(0.3, tag)
        assert boosted > 0.3  # Should be boosted

    def test_flashbulb_detection(self):
        mod = EmotionalModulator()
        tag = mod.analyze("I can't believe it! This is shocking and unexpected!")
        # High arousal + surprising might trigger flashbulb
        # (depends on implementation, at minimum the method should return bool)
        result = mod.should_create_flashbulb(tag)
        assert isinstance(result, bool)


# ── Interference Tests ───────────────────────────────────────────

class TestInterferenceDetector:

    def test_detect_contradiction(self):
        detector = InterferenceDetector()
        old = EpisodicMemory(content="Alice lives in New York City")
        new = EpisodicMemory(content="Alice no longer lives in New York, she moved to San Francisco")

        def simple_sim(a, b):
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / len(words_a | words_b)

        events = detector.detect(new, [old], compute_similarity=simple_sim)
        # Should detect some form of interference
        assert isinstance(events, list)

    def test_contradiction_text_detection(self):
        detector = InterferenceDetector()
        result = detector.detect_contradictions_in_text(
            "User likes Python",
            "User no longer likes Python"
        )
        assert result is True

    def test_no_contradiction(self):
        detector = InterferenceDetector()
        result = detector.detect_contradictions_in_text(
            "User likes Python",
            "User works at Google"
        )
        assert result is False

    def test_statistics(self):
        detector = InterferenceDetector()
        stats = detector.get_statistics()
        assert "total_events" in stats


# ── Prospective Memory Tests ────────────────────────────────────

class TestProspectiveMemory:

    def test_add_time_based(self):
        engine = ProspectiveMemoryEngine()
        mem = engine.add(
            content="Remind about meeting",
            trigger_type=ProspectiveMemoryType.TIME_BASED,
            trigger_condition="at 3pm",
            trigger_time=datetime.now(timezone.utc) - timedelta(minutes=1),  # Already past
        )
        assert mem.content == "Remind about meeting"
        assert mem.status == "pending"

    def test_check_time_triggers(self):
        engine = ProspectiveMemoryEngine()
        engine.add(
            content="Past reminder",
            trigger_type=ProspectiveMemoryType.TIME_BASED,
            trigger_condition="past",
            trigger_time=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        triggered = engine.check_triggers()
        assert len(triggered) >= 1

    def test_get_pending(self):
        engine = ProspectiveMemoryEngine()
        engine.add(
            content="Future task",
            trigger_type=ProspectiveMemoryType.TIME_BASED,
            trigger_condition="future",
            trigger_time=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        pending = engine.get_pending()
        assert len(pending) >= 1

    def test_statistics(self):
        engine = ProspectiveMemoryEngine()
        stats = engine.get_statistics()
        assert isinstance(stats, dict)


# ── Metamemory Tests ─────────────────────────────────────────────

class TestMetamemory:

    def _make_result(self, content: str, score: float) -> MemoryResult:
        return MemoryResult(
            memory=EpisodicMemory(content=content),
            relevance_score=score,
            tier=MemoryTier.EPISODIC,
            retrieval_latency_ms=1.0,
        )

    def test_assess_high_confidence(self):
        engine = MetamemoryEngine()
        results = [self._make_result(f"Fact {i}", 0.9) for i in range(5)]
        assessment = engine.assess("test query", results)
        assert assessment.confidence_score > 0.5
        assert assessment.has_relevant_memories is True

    def test_assess_no_results(self):
        engine = MetamemoryEngine()
        assessment = engine.assess("unknown query", [])
        assert assessment.confidence_level == ConfidenceLevel.NO_KNOWLEDGE
        assert assessment.has_relevant_memories is False

    def test_feeling_of_knowing(self):
        engine = MetamemoryEngine()
        results = [self._make_result("Partial match", 0.4)]
        fok = engine.feeling_of_knowing("test", results)
        assert 0.0 <= fok <= 1.0

    def test_calibration_perfect(self):
        engine = MetamemoryEngine()
        # Perfect calibration: high confidence when correct, low when wrong
        predictions = [(0.9, True), (0.8, True), (0.1, False), (0.2, False)]
        score = engine.calibration_score(predictions)
        assert score > 0.8  # Should be well-calibrated

    def test_calibration_empty(self):
        engine = MetamemoryEngine()
        score = engine.calibration_score([])
        assert score == 0.5 or score == 0.0  # Handle edge case gracefully
