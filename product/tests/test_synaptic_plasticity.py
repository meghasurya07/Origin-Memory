"""Tests for Synaptic Plasticity Engine — STDP, Hebbian, Tagging."""
import math
from datetime import datetime, timedelta, timezone

from origin_brain.synaptic_plasticity import (
    SynapticPlasticityEngine, PlasticityConfig,
    Synapse, SynapticTag,
)


class TestSynapseCreation:
    def test_create_synapse(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("mem_a", "mem_b")
        assert syn.source_id == "mem_a"
        assert syn.target_id == "mem_b"
        assert syn.weight == 0.5

    def test_create_with_custom_weight(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.8)
        assert syn.weight == 0.8

    def test_find_synapses_by_source(self):
        engine = SynapticPlasticityEngine()
        engine.create_synapse("a", "b")
        engine.create_synapse("a", "c")
        engine.create_synapse("b", "c")
        
        found = engine.find_synapses(source_id="a")
        assert len(found) == 2

    def test_find_synapses_by_target(self):
        engine = SynapticPlasticityEngine()
        engine.create_synapse("a", "c")
        engine.create_synapse("b", "c")
        engine.create_synapse("a", "d")
        
        found = engine.find_synapses(target_id="c")
        assert len(found) == 2


class TestHebbianLearning:
    def test_high_coactivation_strengthens(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.5)
        new_weight = engine.hebbian_update(syn.id, co_activation=0.9)
        assert new_weight > 0.5

    def test_low_coactivation_weakens(self):
        """Below the BCM threshold → LTD."""
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.5)
        new_weight = engine.hebbian_update(syn.id, co_activation=0.1)
        assert new_weight < 0.5

    def test_repeated_coactivation_keeps_strengthening(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.3)
        for _ in range(10):
            engine.hebbian_update(syn.id, co_activation=0.9)
        assert engine.get_synapse(syn.id).weight > 0.5

    def test_weight_bounded_at_max(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.95)
        for _ in range(100):
            engine.hebbian_update(syn.id, co_activation=1.0)
        assert engine.get_synapse(syn.id).weight <= 1.0

    def test_weight_bounded_at_min(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.1)
        for _ in range(100):
            engine.hebbian_update(syn.id, co_activation=0.0)
        assert engine.get_synapse(syn.id).weight >= 0.0


class TestSTDP:
    def test_pre_before_post_strengthens(self):
        """Positive dt → LTP."""
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.5)
        new_w = engine.stdp_update(syn.id, dt_seconds=5.0)
        assert new_w > 0.5

    def test_post_before_pre_weakens(self):
        """Negative dt → LTD."""
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.5)
        new_w = engine.stdp_update(syn.id, dt_seconds=-5.0)
        assert new_w < 0.5

    def test_larger_dt_smaller_effect(self):
        """STDP effect decays exponentially with time difference."""
        engine = SynapticPlasticityEngine()
        syn1 = engine.create_synapse("a", "b", initial_weight=0.5)
        syn2 = engine.create_synapse("c", "d", initial_weight=0.5)
        
        w1 = engine.stdp_update(syn1.id, dt_seconds=1.0)   # Close timing
        w2 = engine.stdp_update(syn2.id, dt_seconds=100.0)  # Far timing
        
        # Closer timing should have bigger effect
        assert (w1 - 0.5) > (w2 - 0.5)

    def test_zero_dt_no_change(self):
        """Simultaneous → no change."""
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.5)
        new_w = engine.stdp_update(syn.id, dt_seconds=0.0)
        assert new_w == 0.5


class TestSynapticTagging:
    def test_strong_weight_gets_late_ltp(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.8)
        tag = engine.apply_tag(syn.id)
        assert tag == SynapticTag.LATE_LTP

    def test_moderate_weight_gets_early_ltp(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.6)
        tag = engine.apply_tag(syn.id)
        assert tag == SynapticTag.EARLY_LTP

    def test_weak_weight_gets_early_ltd(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.2)
        tag = engine.apply_tag(syn.id)
        assert tag == SynapticTag.EARLY_LTD

    def test_middle_weight_no_tag(self):
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b", initial_weight=0.4)
        tag = engine.apply_tag(syn.id)
        assert tag == SynapticTag.NONE


class TestPruning:
    def test_prune_weak_synapses(self):
        engine = SynapticPlasticityEngine()
        engine.create_synapse("a", "b", initial_weight=0.01)
        engine.create_synapse("c", "d", initial_weight=0.8)
        engine.create_synapse("e", "f", initial_weight=0.03)
        
        pruned = engine.prune_weak_synapses(threshold=0.05)
        assert len(pruned) == 2
        assert len(engine._synapses) == 1


class TestStatistics:
    def test_stats_after_operations(self):
        engine = SynapticPlasticityEngine()
        s1 = engine.create_synapse("a", "b", initial_weight=0.5)
        s2 = engine.create_synapse("c", "d", initial_weight=0.5)
        
        engine.hebbian_update(s1.id, co_activation=0.9)  # LTP
        engine.hebbian_update(s2.id, co_activation=0.1)  # LTD
        
        stats = engine.get_statistics()
        assert stats["total_synapses"] == 2
        assert stats["total_ltp_events"] >= 1
        assert stats["total_ltd_events"] >= 1
        assert stats["total_events"] == 2

    def test_connection_strength(self):
        engine = SynapticPlasticityEngine()
        engine.create_synapse("a", "b", initial_weight=0.6)
        engine.create_synapse("a", "b", initial_weight=0.8)
        
        strength = engine.get_connection_strength("a", "b")
        assert abs(strength - 0.7) < 0.01


class TestMetaplasticity:
    def test_bcm_threshold_slides_up_with_high_activity(self):
        """High activity → harder to potentiate (sliding threshold rises)."""
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b")
        initial_theta = syn.theta_m
        
        for _ in range(20):
            engine.hebbian_update(syn.id, co_activation=0.9)
        
        assert engine.get_synapse(syn.id).theta_m > initial_theta

    def test_bcm_threshold_slides_down_with_low_activity(self):
        """Low activity → easier to potentiate (sliding threshold drops)."""
        engine = SynapticPlasticityEngine()
        syn = engine.create_synapse("a", "b")
        initial_theta = syn.theta_m
        
        for _ in range(20):
            engine.hebbian_update(syn.id, co_activation=0.1)
        
        assert engine.get_synapse(syn.id).theta_m < initial_theta
