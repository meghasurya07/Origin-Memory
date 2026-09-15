"""
Tests for Working Memory Engine — Capacity-Limited Attention Buffer

Tests verify:
1. Capacity limit is enforced
2. Priority-based displacement
3. Rehearsal prevents displacement
4. Decay cycle evicts inactive items
5. Duplicate detection
6. Statistics tracking
"""
import pytest

from origin_brain.working_memory import (
    WorkingMemoryEngine, WorkingMemoryConfig, WorkingMemoryItem, DisplacementEvent
)


class TestWorkingMemoryEngine:

    def test_creation(self):
        wm = WorkingMemoryEngine()
        assert wm.current_load == 0
        assert not wm.is_full

    def test_attend_single_item(self):
        wm = WorkingMemoryEngine()
        item, displacement = wm.attend("The database is down", salience=0.9)
        assert isinstance(item, WorkingMemoryItem)
        assert displacement is None
        assert wm.current_load == 1

    def test_capacity_limit(self):
        """Should not exceed capacity."""
        wm = WorkingMemoryEngine(WorkingMemoryConfig(capacity=3))
        for i in range(5):
            wm.attend(f"Item {i}", salience=0.5)
        assert wm.current_load == 3

    def test_displacement_lowest_priority(self):
        """When full, should displace lowest-priority item."""
        wm = WorkingMemoryEngine(WorkingMemoryConfig(capacity=3))
        wm.attend("Low priority", salience=0.1)
        wm.attend("Medium priority", salience=0.5)
        wm.attend("High priority", salience=0.9)
        
        # Now add another high-priority item
        _, displacement = wm.attend("Very high priority", salience=0.95)
        
        assert displacement is not None
        assert "Low priority" in displacement.displaced_content
        assert wm.current_load == 3

    def test_high_salience_displaces_low(self):
        """High salience item should displace low salience."""
        wm = WorkingMemoryEngine(WorkingMemoryConfig(capacity=2))
        wm.attend("Boring fact", salience=0.1)
        wm.attend("Another boring fact", salience=0.2)
        
        item, disp = wm.attend("CRITICAL ALERT", salience=0.99)
        assert disp is not None
        assert "Boring fact" in disp.displaced_content

    def test_duplicate_refreshes(self):
        """Adding same content should refresh, not duplicate."""
        wm = WorkingMemoryEngine(WorkingMemoryConfig(capacity=3))
        wm.attend("Important fact", salience=0.7)
        wm.attend("Other thing", salience=0.5)
        
        # Attend to same content again
        item, _ = wm.attend("Important fact", salience=0.7)
        assert wm.current_load == 2  # Not 3
        assert item.activation == 1.0  # Refreshed

    def test_rehearsal(self):
        """Rehearsal should boost activation."""
        wm = WorkingMemoryEngine()
        wm.attend("Remember this", salience=0.5)
        
        # Decay a few times
        wm.decay_cycle()
        wm.decay_cycle()
        
        # Rehearse
        item = wm.rehearse("Remember this")
        assert item is not None
        assert item.activation > 0.5  # Boosted

    def test_decay_evicts(self):
        """Multiple decay cycles should evict items below threshold."""
        wm = WorkingMemoryEngine(WorkingMemoryConfig(
            capacity=3, recency_decay=0.5, activation_threshold=0.2
        ))
        wm.attend("Fading item", salience=0.3)
        
        # Decay many times
        for _ in range(10):
            wm.decay_cycle()
        
        assert wm.current_load == 0  # Should be evicted

    def test_get_active_items_sorted(self):
        """Should return items sorted by priority (highest first)."""
        wm = WorkingMemoryEngine()
        wm.attend("Low", salience=0.2)
        wm.attend("High", salience=0.9)
        wm.attend("Medium", salience=0.5)
        
        active = wm.get_active_items()
        assert active[0].content == "High"
        assert active[-1].content == "Low"

    def test_context_dict(self):
        """Should export as context dictionary."""
        wm = WorkingMemoryEngine()
        wm.attend("Fact A", salience=0.5, category="work")
        wm.attend("Fact B", salience=0.7)
        
        ctx = wm.get_context_dict()
        assert "wm_item_count" in ctx
        assert ctx["wm_item_count"] == 2

    def test_contains(self):
        wm = WorkingMemoryEngine()
        wm.attend("The sky is blue", salience=0.5)
        assert wm.contains("sky is blue")
        assert not wm.contains("grass is green")

    def test_clear(self):
        wm = WorkingMemoryEngine()
        wm.attend("Item 1", salience=0.5)
        wm.attend("Item 2", salience=0.5)
        wm.clear()
        assert wm.current_load == 0

    def test_statistics(self):
        wm = WorkingMemoryEngine(WorkingMemoryConfig(capacity=2))
        wm.attend("A", salience=0.5)
        wm.attend("B", salience=0.7)
        wm.attend("C", salience=0.9)  # Displaces A
        
        stats = wm.get_statistics()
        assert stats["capacity"] == 2
        assert stats["current_load"] == 2
        assert stats["total_attended"] == 3
        assert stats["total_displaced"] == 1
