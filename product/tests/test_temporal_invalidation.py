"""Tests for temporal invalidation engine — bi-temporal fact management."""
import pytest
from datetime import datetime, timezone, timedelta
from origin_brain.temporal_invalidation import (
    TemporalInvalidationEngine,
    TemporalFact,
    FactStatus,
    ConflictResolution,
)


class TestTemporalFactExtraction:
    """Test entity-relation extraction from natural language."""
    
    def test_extract_works_at(self):
        engine = TemporalInvalidationEngine()
        facts = engine.extract_facts("Alice works at Google")
        assert len(facts) >= 1
        f = facts[0]
        assert f.subject.lower() == "alice"
        assert f.predicate == "works_at"
        assert "google" in f.object.lower()
    
    def test_extract_job_offer(self):
        engine = TemporalInvalidationEngine()
        facts = engine.extract_facts("Bob got a new job offer from Meta")
        assert len(facts) >= 1
        assert any("meta" in f.object.lower() for f in facts)
    
    def test_extract_allergy(self):
        engine = TemporalInvalidationEngine()
        facts = engine.extract_facts("Alice is allergic to shellfish")
        assert len(facts) >= 1
        f = facts[0]
        assert f.predicate == "allergic_to"
        assert "shellfish" in f.object.lower()
    
    def test_extract_pet(self):
        engine = TemporalInvalidationEngine()
        facts = engine.extract_facts("Alice adopted a golden retriever named Luna")
        assert len(facts) >= 1
        assert any("luna" in f.object.lower() for f in facts)
    
    def test_extract_no_match(self):
        engine = TemporalInvalidationEngine()
        facts = engine.extract_facts("The weather is nice today")
        assert len(facts) == 0
    
    def test_extract_promotion(self):
        engine = TemporalInvalidationEngine()
        facts = engine.extract_facts("Alice got promoted to Senior Engineer")
        assert len(facts) >= 1
        assert any(f.predicate == "role_is" for f in facts)


class TestConflictDetection:
    """Test automatic conflict detection and resolution."""
    
    def test_supersession_temporal(self):
        """Newer fact should supersede older one."""
        engine = TemporalInvalidationEngine(resolution_strategy=ConflictResolution.TEMPORAL)
        
        # Bob works at Google (old fact)
        old_fact = TemporalFact(
            subject="Bob", predicate="works_at", object="Google",
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        engine.register_fact(old_fact)
        
        # Bob works at Meta (new fact — should supersede)
        new_fact = TemporalFact(
            subject="Bob", predicate="works_at", object="Meta",
            valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        was_conflict, invalidated_id = engine.register_fact(new_fact)
        
        assert was_conflict is True
        assert invalidated_id == old_fact.id
        assert old_fact.status == FactStatus.SUPERSEDED
        assert old_fact.valid_to is not None
        assert new_fact.status == FactStatus.ACTIVE
    
    def test_no_conflict_same_value(self):
        """Same subject+predicate+object should not conflict."""
        engine = TemporalInvalidationEngine()
        
        f1 = TemporalFact(subject="Alice", predicate="works_at", object="Google")
        f2 = TemporalFact(subject="Alice", predicate="works_at", object="Google")
        
        engine.register_fact(f1)
        was_conflict, _ = engine.register_fact(f2)
        
        assert was_conflict is False
        assert engine.active_facts == 1
    
    def test_different_predicates_no_conflict(self):
        """Different predicates should not conflict."""
        engine = TemporalInvalidationEngine()
        
        f1 = TemporalFact(subject="Alice", predicate="works_at", object="Google")
        f2 = TemporalFact(subject="Alice", predicate="lives_in", object="New York")
        
        engine.register_fact(f1)
        was_conflict, _ = engine.register_fact(f2)
        
        assert was_conflict is False
        assert engine.active_facts == 2
    
    def test_confidence_resolution(self):
        """Higher confidence should win in CONFIDENCE mode."""
        engine = TemporalInvalidationEngine(resolution_strategy=ConflictResolution.CONFIDENCE)
        
        high_conf = TemporalFact(
            subject="Bob", predicate="works_at", object="Google", confidence=0.95)
        low_conf = TemporalFact(
            subject="Bob", predicate="works_at", object="Meta", confidence=0.3)
        
        engine.register_fact(high_conf)
        was_conflict, _ = engine.register_fact(low_conf)
        
        assert was_conflict is True
        # High confidence should survive
        assert high_conf.status == FactStatus.ACTIVE
        assert engine.active_facts == 1
    
    def test_conflict_log(self):
        """Conflicts should be logged."""
        engine = TemporalInvalidationEngine()
        
        f1 = TemporalFact(subject="Bob", predicate="works_at", object="Google")
        f2 = TemporalFact(subject="Bob", predicate="works_at", object="Meta")
        
        engine.register_fact(f1)
        engine.register_fact(f2)
        
        conflicts = engine.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["old_value"] == "Google"
        assert conflicts[0]["new_value"] == "Meta"


class TestTemporalQuery:
    """Test point-in-time querying."""
    
    def test_query_current(self):
        engine = TemporalInvalidationEngine()
        
        f1 = TemporalFact(
            subject="Bob", predicate="works_at", object="Google",
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc))
        f2 = TemporalFact(
            subject="Bob", predicate="works_at", object="Meta",
            valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc))
        
        engine.register_fact(f1)
        engine.register_fact(f2)
        
        current = engine.query("Bob", "works_at")
        assert current is not None
        assert current.object == "Meta"
    
    def test_query_at_past_time(self):
        engine = TemporalInvalidationEngine()
        
        f1 = TemporalFact(
            subject="Bob", predicate="works_at", object="Google",
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc))
        f2 = TemporalFact(
            subject="Bob", predicate="works_at", object="Meta",
            valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc))
        
        engine.register_fact(f1)
        engine.register_fact(f2)
        
        # Query at March 2026 — should return Google
        past = engine.query("Bob", "works_at", 
                          at_time=datetime(2026, 3, 1, tzinfo=timezone.utc))
        assert past is not None
        assert past.object == "Google"
    
    def test_fact_history(self):
        engine = TemporalInvalidationEngine()
        
        f1 = TemporalFact(
            subject="Bob", predicate="works_at", object="Google",
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc))
        f2 = TemporalFact(
            subject="Bob", predicate="works_at", object="Meta",
            valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc))
        f3 = TemporalFact(
            subject="Bob", predicate="works_at", object="OpenAI",
            valid_from=datetime(2026, 11, 1, tzinfo=timezone.utc))
        
        engine.register_fact(f1)
        engine.register_fact(f2)
        engine.register_fact(f3)
        
        history = engine.get_fact_history("Bob", "works_at")
        assert len(history) == 3
        assert history[0].object == "Google"
        assert history[1].object == "Meta"
        assert history[2].object == "OpenAI"
        assert history[0].status == FactStatus.SUPERSEDED
        assert history[1].status == FactStatus.SUPERSEDED
        assert history[2].status == FactStatus.ACTIVE


class TestGetCurrentFacts:
    """Test fact retrieval and filtering."""
    
    def test_get_all_current(self):
        engine = TemporalInvalidationEngine()
        
        engine.register_fact(TemporalFact(subject="Alice", predicate="works_at", object="Google"))
        engine.register_fact(TemporalFact(subject="Alice", predicate="lives_in", object="SF"))
        engine.register_fact(TemporalFact(subject="Bob", predicate="works_at", object="Meta"))
        
        all_facts = engine.get_current_facts()
        assert len(all_facts) == 3
    
    def test_get_by_subject(self):
        engine = TemporalInvalidationEngine()
        
        engine.register_fact(TemporalFact(subject="Alice", predicate="works_at", object="Google"))
        engine.register_fact(TemporalFact(subject="Alice", predicate="lives_in", object="SF"))
        engine.register_fact(TemporalFact(subject="Bob", predicate="works_at", object="Meta"))
        
        alice_facts = engine.get_current_facts(subject="Alice")
        assert len(alice_facts) == 2
    
    def test_superseded_not_in_current(self):
        engine = TemporalInvalidationEngine()
        
        engine.register_fact(TemporalFact(subject="Bob", predicate="works_at", object="Google"))
        engine.register_fact(TemporalFact(subject="Bob", predicate="works_at", object="Meta"))
        
        current = engine.get_current_facts(subject="Bob")
        assert len(current) == 1
        assert current[0].object == "Meta"


class TestStats:
    """Test statistics tracking."""
    
    def test_stats(self):
        engine = TemporalInvalidationEngine()
        
        engine.register_fact(TemporalFact(subject="Alice", predicate="works_at", object="Google"))
        engine.register_fact(TemporalFact(subject="Alice", predicate="works_at", object="Meta"))
        engine.register_fact(TemporalFact(subject="Bob", predicate="lives_in", object="NYC"))
        
        stats = engine.stats()
        assert stats["total_facts"] == 3
        assert stats["active_facts"] == 2
        assert stats["superseded_facts"] == 1
        assert stats["conflicts_detected"] == 1
    
    def test_category_inference(self):
        engine = TemporalInvalidationEngine()
        facts = engine.extract_facts("Alice works at Google")
        if facts:
            assert facts[0].category == "career"
        
        facts = engine.extract_facts("Alice is allergic to peanuts")
        if facts:
            assert facts[0].category == "health"
