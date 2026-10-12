"""Tests for Neocortical Learning Engine (CLS slow learning)."""
import pytest

from origin_brain.neocortex import (
    NeocorticalEngine, NeocortexConfig, SemanticConcept, Schema,
)


class TestNeocorticalEngine:
    @pytest.fixture
    def engine(self):
        return NeocorticalEngine(NeocortexConfig(
            min_cooccurrence=2,
            schema_threshold=3,
        ))

    def test_learn_from_episode_creates_concepts(self, engine):
        result = engine.learn_from_episode({
            "entities": ["Bob", "Budget", "Cafe"],
            "type": "meeting",
        })
        assert result["entities_updated"] == 3
        assert "bob" in engine._concepts
        assert "budget" in engine._concepts
        assert "cafe" in engine._concepts

    def test_cooccurrence_tracking(self, engine):
        engine.learn_from_episode({"entities": ["Bob", "Budget"], "type": "meeting"})
        engine.learn_from_episode({"entities": ["Bob", "Budget"], "type": "meeting"})
        # After 2 episodes, co-occurrence should be 2
        pair = ("bob", "budget")
        assert engine._cooccurrence[pair] == 2

    def test_associations_form_after_threshold(self, engine):
        # First episode — no association yet (min_cooccurrence=2)
        engine.learn_from_episode({"entities": ["Alice", "Report"], "type": "work"})
        assocs = engine.get_associations("Alice")
        assert len(assocs) == 0

        # Second episode — association should form
        engine.learn_from_episode({"entities": ["Alice", "Report"], "type": "work"})
        assocs = engine.get_associations("Alice")
        assert len(assocs) > 0
        assert assocs[0][0] == "report"

    def test_schema_formation(self, engine):
        # Feed 3 similar episodes (schema_threshold=3)
        for i in range(3):
            engine.learn_from_episode({
                "entities": ["Bob", "Budget"],
                "type": "meeting",
            })
        schema = engine.get_schema("meeting")
        assert schema is not None
        assert schema.instance_count == 3

    def test_schema_consistent(self, engine):
        # Build a schema
        for _ in range(4):
            engine.learn_from_episode({
                "entities": ["Bob", "Budget", "Office"],
                "type": "meeting",
            })
        
        # Consistent episode
        score = engine.schema_consistent({
            "entities": ["Bob", "Budget"],
            "type": "meeting",
        })
        assert score > 0.0

        # Inconsistent episode
        score = engine.schema_consistent({
            "entities": ["Cat", "Dog"],
            "type": "pet_care",
        })
        assert score == 0.0

    def test_decay(self, engine):
        engine.learn_from_episode({"entities": ["X", "Y"], "type": "test"})
        engine.learn_from_episode({"entities": ["X", "Y"], "type": "test"})
        
        assocs_before = engine.get_associations("X")
        assert len(assocs_before) > 0
        strength_before = assocs_before[0][1]
        
        engine.decay()
        
        assocs_after = engine.get_associations("X")
        if assocs_after:
            assert assocs_after[0][1] < strength_before

    def test_categories_tracked(self, engine):
        engine.learn_from_episode({"entities": ["Python"], "type": "coding"})
        engine.learn_from_episode({"entities": ["Python"], "type": "biology"})
        
        concept = engine._concepts["python"]
        assert "coding" in concept.categories
        assert "biology" in concept.categories

    def test_statistics(self, engine):
        engine.learn_from_episode({"entities": ["A", "B"], "type": "test"})
        stats = engine.get_statistics()
        assert stats["total_concepts"] == 2
        assert stats["total_episodes_learned"] == 1

    def test_max_concepts_eviction(self):
        engine = NeocorticalEngine(NeocortexConfig(max_concepts=3))
        engine.learn_from_episode({"entities": ["A"], "type": "t"})
        engine.learn_from_episode({"entities": ["B"], "type": "t"})
        engine.learn_from_episode({"entities": ["C"], "type": "t"})
        # Adding 4th should evict weakest
        engine.learn_from_episode({"entities": ["D"], "type": "t"})
        assert len(engine._concepts) <= 3

    def test_no_schema_for_unknown_type(self, engine):
        engine.learn_from_episode({"entities": ["X"], "type": "unknown"})
        assert engine.get_schema("unknown") is None

    def test_spreading_activation_multiple_hops(self, engine):
        """Test that associations form a semantic network."""
        # Bob knows Alice, Alice knows Carol
        for _ in range(3):
            engine.learn_from_episode({"entities": ["Bob", "Alice"], "type": "social"})
            engine.learn_from_episode({"entities": ["Alice", "Carol"], "type": "social"})
        
        bob_assocs = engine.get_associations("Bob")
        alice_assocs = engine.get_associations("Alice")
        
        # Bob should know Alice
        bob_names = [a[0] for a in bob_assocs]
        assert "alice" in bob_names
        
        # Alice should know both
        alice_names = [a[0] for a in alice_assocs]
        assert "bob" in alice_names
        assert "carol" in alice_names

    def test_empty_entities(self, engine):
        result = engine.learn_from_episode({"entities": [], "type": "empty"})
        assert result["entities_updated"] == 0

    def test_interleaved_learning(self, engine):
        """Multiple episode types should form separate schemas."""
        for _ in range(4):
            engine.learn_from_episode({"entities": ["Meeting", "Bob"], "type": "work"})
            engine.learn_from_episode({"entities": ["Gym", "Yoga"], "type": "exercise"})
        
        work_schema = engine.get_schema("work")
        exercise_schema = engine.get_schema("exercise")
        assert work_schema is not None
        assert exercise_schema is not None
