"""
Tests for NeurogenesisEngine — Dynamic Memory Allocation

Tests verify:
1. Neuron birth (generation)
2. Maturation lifecycle
3. Apoptosis (programmed death)
4. Activity-dependent survival
5. Daily cycle
6. Population limits
7. Memory assignment
"""
import pytest
import time
from datetime import datetime, timezone, timedelta

from origin_brain.neurogenesis import (
    NeurogenesisEngine, NeurogenesisConfig,
    MemoryNeuron, NeuronState
)


class TestMemoryNeuron:

    def test_creation(self):
        neuron = MemoryNeuron()
        assert neuron.state == NeuronState.IMMATURE
        assert neuron.plasticity == 1.0
        assert neuron.activation_count == 0

    def test_activation(self):
        neuron = MemoryNeuron()
        neuron.activate()
        assert neuron.activation_count == 1
        assert neuron.integration_score == 0.1
        
        neuron.activate()
        assert neuron.activation_count == 2
        assert neuron.integration_score == 0.2

    def test_age(self):
        neuron = MemoryNeuron()
        assert neuron.age_hours() >= 0
        assert neuron.age_hours() < 0.01  # Just created


class TestNeurogenesisEngine:

    def _make_engine(self, **kwargs) -> NeurogenesisEngine:
        config = NeurogenesisConfig(
            birth_rate_per_day=5,
            max_neurons=50,
            immature_duration_hours=0.001,   # Very short for testing
            maturing_duration_hours=0.001,
            min_activations_to_survive=1,
            min_integration_to_survive=0.05,
            **kwargs
        )
        return NeurogenesisEngine(config=config)

    def test_generate_neurons(self):
        engine = self._make_engine()
        neurons = engine.generate_neurons(count=5)
        assert len(neurons) == 5
        assert all(n.state == NeuronState.IMMATURE for n in neurons)
        assert engine._total_born == 5

    def test_population_limit(self):
        engine = self._make_engine()
        engine.generate_neurons(count=50)
        assert len(engine._neurons) == 50
        
        # Should not exceed max
        extra = engine.generate_neurons(count=10)
        assert len(extra) == 0
        assert len(engine._neurons) == 50

    def test_maturation_immature_to_maturing(self):
        engine = self._make_engine()
        neurons = engine.generate_neurons(count=3)
        
        # Manually age the neurons past immature_duration
        from datetime import timedelta
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        for n in engine._neurons.values():
            n.created_at = past
        
        matured = engine.update_maturation()
        maturing = [n for n in engine._neurons.values() if n.state == NeuronState.MATURING]
        assert len(maturing) >= 1, f"Expected some to mature, got states: {[n.state.value for n in engine._neurons.values()]}"

    def test_maturation_to_mature_with_activity(self):
        engine = self._make_engine()
        neurons = engine.generate_neurons(count=3)
        
        # Activate neurons so they survive
        for n in neurons:
            n.activate()
            n.activate()
        
        # Age past immature + maturing duration
        from datetime import timedelta
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        for n in engine._neurons.values():
            n.created_at = past
        
        # Run maturation twice to go IMMATURE -> MATURING -> MATURE
        engine.update_maturation()
        engine.update_maturation()
        
        mature = [n for n in engine._neurons.values() if n.state == NeuronState.MATURE]
        assert len(mature) >= 1, f"Expected some mature neurons, got states: {[n.state.value for n in engine._neurons.values()]}"

    def test_maturation_to_declining_without_activity(self):
        engine = self._make_engine()
        neurons = engine.generate_neurons(count=3)
        
        # DON'T activate — age them past immature + maturing
        from datetime import timedelta
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        for n in engine._neurons.values():
            n.created_at = past
        
        engine.update_maturation()
        engine.update_maturation()
        
        declining = [n for n in engine._neurons.values() if n.state == NeuronState.DECLINING]
        assert len(declining) >= 1, f"Inactive neurons should decline, got: {[n.state.value for n in engine._neurons.values()]}"

    def test_apoptosis_kills_declining(self):
        engine = self._make_engine()
        neurons = engine.generate_neurons(count=5)
        
        # Force some to declining
        for n in list(engine._neurons.values())[:3]:
            n.state = NeuronState.DECLINING
        
        dead = engine.apoptosis()
        assert len(dead) == 3
        assert len(engine._neurons) == 2
        assert engine._total_died == 3

    def test_daily_cycle(self):
        engine = self._make_engine()
        
        # Seed with some neurons
        engine.generate_neurons(count=10)
        
        # Activate some
        for i, n in enumerate(engine._neurons.values()):
            if i < 5:
                n.activate()
                n.activate()
        
        time.sleep(0.01)
        born, died, matured = engine.daily_cycle()
        
        assert engine._cycles_run == 1
        assert isinstance(born, list)
        assert isinstance(died, list)
        assert isinstance(matured, list)

    def test_assign_memory(self):
        engine = self._make_engine()
        neurons = engine.generate_neurons(count=1)
        neuron_id = neurons[0].id
        
        success = engine.assign_memory(neuron_id, "mem_123")
        assert success is True
        assert engine._neurons[neuron_id].memory_id == "mem_123"
        assert engine._neurons[neuron_id].activation_count == 1

    def test_get_available_neurons(self):
        engine = self._make_engine()
        neurons = engine.generate_neurons(count=5)
        
        # Assign 2 to memories
        engine.assign_memory(neurons[0].id, "mem_1")
        engine.assign_memory(neurons[1].id, "mem_2")
        
        available = engine.get_available_neurons()
        assert len(available) == 3

    def test_statistics(self):
        engine = self._make_engine()
        engine.generate_neurons(count=10)
        
        stats = engine.get_statistics()
        assert stats["population"] == 10
        assert stats["total_born"] == 10
        assert stats["total_died"] == 0
        assert stats["average_plasticity"] == 1.0

    def test_population_by_state(self):
        engine = self._make_engine()
        engine.generate_neurons(count=5)
        
        pop = engine.get_population_by_state()
        assert pop["immature"] == 5
        assert pop["mature"] == 0

    def test_event_log(self):
        engine = self._make_engine()
        engine.generate_neurons(count=3)
        
        assert len(engine._events) == 3
        assert all(e.event_type == "birth" for e in engine._events)
