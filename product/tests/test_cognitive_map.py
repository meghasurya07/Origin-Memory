"""Tests for Cognitive Map Engine — Successor Representation."""
from origin_brain.cognitive_map import (
    CognitiveMapEngine, CognitiveMapConfig,
)


class TestTransitions:
    def test_record_transition(self):
        engine = CognitiveMapEngine()
        engine.record_transition("a", "b")
        assert engine._total_transitions == 1

    def test_record_sequence(self):
        engine = CognitiveMapEngine()
        engine.record_recall_sequence(["a", "b", "c", "d"])
        assert engine._total_transitions == 3  # a→b, b→c, c→d

    def test_on_recall_auto_transition(self):
        engine = CognitiveMapEngine()
        engine.on_recall("a")
        engine.on_recall("b")
        engine.on_recall("c")
        assert engine._total_transitions == 2  # a→b, b→c

    def test_transition_probability(self):
        engine = CognitiveMapEngine()
        engine.record_transition("a", "b")
        engine.record_transition("a", "b")
        engine.record_transition("a", "c")
        
        # 2/3 transitions from a go to b
        prob = engine.get_transition_probability("a", "b")
        assert abs(prob - 2/3) < 0.01


class TestSuccessorRepresentation:
    def test_sr_direct_neighbor(self):
        engine = CognitiveMapEngine()
        engine.record_transition("a", "b")
        engine.record_transition("a", "b")
        
        # Full SR update to get accurate values
        engine.update_successor_representation()
        
        # After update, SR(a, b) should be positive
        sr_val = engine._sr.get("a", {}).get("b", 0.0)
        assert sr_val > 0

    def test_sr_multi_step(self):
        """SR should capture multi-step relationships."""
        engine = CognitiveMapEngine()
        # Chain: a → b → c
        for _ in range(5):
            engine.record_transition("a", "b")
            engine.record_transition("b", "c")
        
        engine.update_successor_representation()
        
        # SR(a, c) should be positive (multi-step reachability)
        sr_ac = engine._sr.get("a", {}).get("c", 0.0)
        assert sr_ac > 0

    def test_full_sr_update(self):
        engine = CognitiveMapEngine()
        engine.record_transition("x", "y")
        engine.record_transition("y", "z")
        engine.record_transition("x", "z")
        
        engine.update_successor_representation()
        
        # After full update, SR should have entries
        assert len(engine._sr) > 0

    def test_full_sr_closer_neighbors_have_higher_sr(self):
        engine = CognitiveMapEngine()
        # Direct: a → b (strong), a → c (weak via b only)
        for _ in range(10):
            engine.record_transition("a", "b")
            engine.record_transition("b", "c")
        
        engine.update_successor_representation()
        
        sr_ab = engine._sr.get("a", {}).get("b", 0)
        sr_ac = engine._sr.get("a", {}).get("c", 0)
        
        # Direct neighbor should have higher SR
        assert sr_ab > sr_ac


class TestNavigation:
    def test_find_nearby(self):
        engine = CognitiveMapEngine()
        for _ in range(5):
            engine.record_transition("a", "b")
            engine.record_transition("a", "c")
        
        engine.update_successor_representation()
        nearby = engine.find_nearby("a", top_k=5)
        assert len(nearby) > 0
        # b and c should be nearby a
        nearby_ids = [n[0] for n in nearby]
        assert "b" in nearby_ids or "c" in nearby_ids

    def test_find_nearby_empty(self):
        engine = CognitiveMapEngine()
        nearby = engine.find_nearby("nonexistent")
        assert len(nearby) == 0

    def test_distance(self):
        engine = CognitiveMapEngine()
        for _ in range(10):
            engine.record_transition("a", "b")
        
        engine.update_successor_representation()
        dist_ab = engine.distance("a", "b")
        dist_unknown = engine.distance("a", "unknown")
        
        assert dist_ab < float('inf')
        assert dist_unknown == float('inf')


class TestDecay:
    def test_decay_reduces_transitions(self):
        engine = CognitiveMapEngine(config=CognitiveMapConfig(transition_decay=0.5))
        engine.record_transition("a", "b")
        
        initial = engine._transitions["a"]["b"]
        engine.decay_transitions()
        decayed = engine._transitions["a"]["b"]
        
        assert decayed < initial


class TestStatistics:
    def test_statistics(self):
        engine = CognitiveMapEngine()
        engine.record_transition("a", "b")
        engine.record_transition("b", "c")
        
        stats = engine.get_statistics()
        assert stats["memories_in_map"] == 3  # a, b, c
        assert stats["transition_edges"] == 2
        assert stats["total_transitions_recorded"] == 2
