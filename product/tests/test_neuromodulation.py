"""
Tests for NeuromodulationEngine — Chemical Mode Switching

Tests verify:
1. Neuromodulator state management
2. Novelty → ACh increase → encoding mode
3. Surprise → DA increase → consolidation boost
4. Arousal → NE increase → alert mode / flashbulb
5. Reward → DA increase
6. Decay to baseline
7. Encoding/retrieval/consolidation modifiers
8. Mode transitions
"""
import pytest

from origin_brain.neuromodulation import (
    NeuromodulationEngine, NeuromodulationConfig,
    NeuromodulatorState, BrainMode
)


class TestNeuromodulatorState:

    def test_default_state(self):
        state = NeuromodulatorState()
        assert state.acetylcholine == 0.5
        assert state.dopamine == 0.3
        assert state.norepinephrine == 0.2

    def test_encoding_drive(self):
        state = NeuromodulatorState(acetylcholine=1.0, norepinephrine=1.0)
        # encoding_drive = ACh * 0.7 + NE * 0.3 = 0.7 + 0.3 = 1.0
        assert state.encoding_drive == 1.0

    def test_retrieval_drive(self):
        state = NeuromodulatorState(acetylcholine=0.0, dopamine=1.0)
        # retrieval_drive = (1 - ACh) * 0.8 + DA * 0.2 = 0.8 + 0.2 = 1.0
        assert state.retrieval_drive == 1.0

    def test_encoding_mode(self):
        state = NeuromodulatorState(acetylcholine=0.9, dopamine=0.2, norepinephrine=0.3)
        assert state.current_mode == BrainMode.ENCODING

    def test_retrieval_mode(self):
        state = NeuromodulatorState(acetylcholine=0.1, dopamine=0.2, norepinephrine=0.1)
        assert state.current_mode == BrainMode.RETRIEVAL

    def test_alert_mode(self):
        state = NeuromodulatorState(acetylcholine=0.5, dopamine=0.3, norepinephrine=0.9)
        assert state.current_mode == BrainMode.ALERT


class TestNeuromodulationEngine:

    def test_initialization(self):
        engine = NeuromodulationEngine()
        assert engine.state.acetylcholine == 0.4  # baseline
        assert engine.state.dopamine == 0.2

    def test_on_novelty_boosts_ach(self):
        engine = NeuromodulationEngine()
        initial_ach = engine.state.acetylcholine
        engine.on_novelty(0.8)
        assert engine.state.acetylcholine > initial_ach

    def test_on_surprise_boosts_da(self):
        engine = NeuromodulationEngine()
        initial_da = engine.state.dopamine
        engine.on_surprise(0.9)
        assert engine.state.dopamine > initial_da

    def test_on_arousal_boosts_ne(self):
        engine = NeuromodulationEngine()
        initial_ne = engine.state.norepinephrine
        engine.on_arousal(0.8)
        assert engine.state.norepinephrine > initial_ne

    def test_on_reward_boosts_da(self):
        engine = NeuromodulationEngine()
        initial_da = engine.state.dopamine
        engine.on_reward(0.7)
        assert engine.state.dopamine > initial_da

    def test_high_arousal_triggers_alert_mode(self):
        engine = NeuromodulationEngine()
        # Need NE > 0.8. Baseline=0.15, each arousal(1.0) adds 0.5
        engine.on_arousal(1.0)
        engine.on_arousal(1.0)  # NE = min(1.0, 0.15 + 0.5 + 0.5) = 1.0
        assert engine.state.current_mode == BrainMode.ALERT

    def test_high_novelty_triggers_encoding_mode(self):
        engine = NeuromodulationEngine()
        engine.on_novelty(1.0)
        assert engine.state.current_mode == BrainMode.ENCODING

    def test_decay_toward_baseline(self):
        engine = NeuromodulationEngine()
        engine.on_novelty(1.0)  # Boost ACh high
        high_ach = engine.state.acetylcholine
        
        # Decay several times
        for _ in range(50):
            engine.decay_to_baseline()
        
        # Should be closer to baseline (0.4) than the boosted value
        assert engine.state.acetylcholine < high_ach
        assert abs(engine.state.acetylcholine - 0.4) < 0.1

    def test_encoding_modifier(self):
        engine = NeuromodulationEngine()
        
        # Low novelty → low modifier
        low_mod = engine.get_encoding_modifier()
        
        # High novelty → high modifier
        engine.on_novelty(1.0)
        engine.on_arousal(0.8)
        high_mod = engine.get_encoding_modifier()
        
        assert high_mod > low_mod

    def test_consolidation_modifier(self):
        engine = NeuromodulationEngine()
        
        low_mod = engine.get_consolidation_modifier()
        
        engine.on_surprise(1.0)
        high_mod = engine.get_consolidation_modifier()
        
        assert high_mod > low_mod

    def test_retrieval_modifier(self):
        engine = NeuromodulationEngine()
        
        # Low ACh → high retrieval
        engine.state.acetylcholine = 0.1
        high_ret = engine.get_retrieval_modifier()
        
        # High ACh → low retrieval
        engine.state.acetylcholine = 0.9
        low_ret = engine.get_retrieval_modifier()
        
        assert high_ret > low_ret

    def test_values_clamped(self):
        engine = NeuromodulationEngine()
        
        # Boost multiple times — should not exceed 1.0
        for _ in range(20):
            engine.on_novelty(1.0)
            engine.on_surprise(1.0)
            engine.on_arousal(1.0)
        
        assert engine.state.acetylcholine <= 1.0
        assert engine.state.dopamine <= 1.0
        assert engine.state.norepinephrine <= 1.0

    def test_statistics(self):
        engine = NeuromodulationEngine()
        engine.on_novelty(0.5)
        engine.on_surprise(0.3)
        
        stats = engine.get_statistics()
        assert stats["total_signals"] == 2
        assert "current_mode" in stats
        assert "encoding_modifier" in stats

    def test_log_mode(self):
        engine = NeuromodulationEngine()
        engine.log_mode()
        assert len(engine._mode_history) == 1
        assert "mode" in engine._mode_history[0]

    def test_signal_count(self):
        engine = NeuromodulationEngine()
        engine.on_novelty(0.5)
        engine.on_surprise(0.3)
        engine.on_arousal(0.2)
        engine.on_reward(0.7)
        assert engine._total_signals == 4
