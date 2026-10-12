"""Tests for World Model Engine — Predictive Transitions."""
import pytest

from origin_brain.world_model import (
    WorldModelEngine, WorldModelConfig, WorldState, TransitionRecord,
)


class TestWorldModelEngine:
    @pytest.fixture
    def engine(self):
        return WorldModelEngine(WorldModelConfig(min_transition_count=1))

    def test_observe_transition(self, engine):
        engine.observe_transition("A", "B")
        assert "A" in engine._states
        assert "B" in engine._states
        assert engine._states["A"].transitions["B"].count == 1

    def test_observe_sequence(self, engine):
        engine.observe_sequence(["greet", "ask", "answer", "thank"])
        assert engine._total_sequences == 1
        assert engine._total_transitions == 3
        # greet→ask, ask→answer, answer→thank

    def test_predict_next_deterministic(self, engine):
        for _ in range(5):
            engine.observe_transition("A", "B")
        engine.observe_transition("A", "C")
        
        predictions = engine.predict_next("A")
        assert len(predictions) >= 2
        # B should be more likely (5/6 vs 1/6)
        assert predictions[0][0] == "B"
        assert predictions[0][1] > predictions[1][1]

    def test_predict_unknown_state(self, engine):
        predictions = engine.predict_next("unknown")
        assert predictions == []

    def test_simulate_greedy(self, engine):
        engine.observe_sequence(["A", "B", "C", "D"])
        engine.observe_sequence(["A", "B", "C", "D"])
        
        path = engine.simulate("A", steps=5, strategy="greedy")
        assert path[0] == "A"
        assert len(path) >= 2
        # Should follow the only observed chain: A→B→C→D

    def test_simulate_stops_at_terminal(self, engine):
        engine.observe_transition("start", "end")
        # "end" has no outgoing transitions
        path = engine.simulate("start", steps=10)
        assert path == ["start", "end"]

    def test_simulate_no_loops(self, engine):
        engine.observe_transition("A", "B")
        engine.observe_transition("B", "A")
        # Should NOT loop forever
        path = engine.simulate("A", steps=100)
        assert len(path) <= 3

    def test_transition_matrix(self, engine):
        engine.observe_transition("A", "B")
        engine.observe_transition("A", "C")
        
        matrix = engine.get_transition_matrix()
        assert "A" in matrix
        assert abs(matrix["A"]["B"] - 0.5) < 0.01
        assert abs(matrix["A"]["C"] - 0.5) < 0.01

    def test_reachable_states(self, engine):
        engine.observe_sequence(["A", "B", "C", "D", "E"])
        
        reachable_1 = engine.get_reachable_states("A", max_depth=1)
        assert "B" in reachable_1
        assert "C" not in reachable_1
        
        reachable_3 = engine.get_reachable_states("A", max_depth=3)
        assert "B" in reachable_3
        assert "C" in reachable_3
        assert "D" in reachable_3

    def test_surprise_expected(self, engine):
        for _ in range(10):
            engine.observe_transition("A", "B")
        # B is highly expected after A → low surprise
        surprise = engine.surprise("A", "B")
        assert surprise < 1.0

    def test_surprise_unexpected(self, engine):
        for _ in range(10):
            engine.observe_transition("A", "B")
        # C was never observed after A → max surprise
        surprise = engine.surprise("A", "C")
        assert surprise == 10.0

    def test_surprise_unknown_state(self, engine):
        surprise = engine.surprise("unknown", "whatever")
        assert surprise == 10.0

    def test_decay(self, engine):
        engine.observe_transition("A", "B")
        engine.observe_transition("A", "B")
        assert engine._states["A"].transitions["B"].count == 2
        
        engine.decay()
        # Count should decrease
        assert engine._states["A"].transitions["B"].count < 2

    def test_statistics(self, engine):
        engine.observe_sequence(["A", "B", "C"])
        stats = engine.get_statistics()
        assert stats["total_states"] == 3
        assert stats["total_sequences"] == 1
        assert stats["total_transitions"] == 2

    def test_max_states_eviction(self):
        engine = WorldModelEngine(WorldModelConfig(max_states=3))
        engine.observe_transition("A", "B")
        engine.observe_transition("C", "D")
        # 4 states but max is 3 → eviction should happen
        assert len(engine._states) <= 3

    def test_probabilities_sum_to_one(self, engine):
        engine.observe_transition("A", "B")
        engine.observe_transition("A", "C")
        engine.observe_transition("A", "D")
        
        total_prob = sum(
            t.probability for t in engine._states["A"].transitions.values()
        )
        assert abs(total_prob - 1.0) < 0.001

    def test_empty_sequence_no_error(self, engine):
        engine.observe_sequence([])
        engine.observe_sequence(["single"])
        assert engine._total_sequences == 0  # Neither counted

    def test_branching_paths(self, engine):
        """Multiple possible futures from same state."""
        for _ in range(3):
            engine.observe_sequence(["start", "option_a", "end_a"])
        for _ in range(2):
            engine.observe_sequence(["start", "option_b", "end_b"])
        for _ in range(1):
            engine.observe_sequence(["start", "option_c", "end_c"])
        
        predictions = engine.predict_next("start")
        assert len(predictions) == 3
        # option_a should be most likely (3/6)
        assert predictions[0][0] == "option_a"
        assert abs(predictions[0][1] - 0.5) < 0.01
