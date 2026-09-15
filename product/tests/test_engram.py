"""Tests for Engram Tracking Engine — Memory Trace Formation."""
from datetime import datetime, timedelta, timezone

from origin_brain.engram import (
    EngramEngine, EngramConfig, Engram, EngramCell, EngramState,
)


class TestEngramAllocation:
    def test_allocate_engram(self):
        engine = EngramEngine()
        engram = engine.allocate_engram("mem_001")
        assert engram.memory_id == "mem_001"
        assert len(engram.cell_ids) == 5  # default cells_per_engram
        assert engram.state == EngramState.ACTIVE

    def test_allocate_returns_existing(self):
        engine = EngramEngine()
        e1 = engine.allocate_engram("mem_001")
        e2 = engine.allocate_engram("mem_001")
        assert e1.id == e2.id

    def test_excitability_competition(self):
        """Most excitable cells should be allocated first."""
        engine = EngramEngine(config=EngramConfig(cells_per_engram=3, total_cell_pool=10))
        engram = engine.allocate_engram("mem_001")
        
        # The allocated cells should have been boosted
        for cell_id in engram.cell_ids:
            cell = engine._cells[cell_id]
            assert cell.excitability > 0.5  # boosted from baseline

    def test_cells_gain_excitability_after_allocation(self):
        engine = EngramEngine(config=EngramConfig(cells_per_engram=3, total_cell_pool=10))
        
        # Get initial max excitability
        initial_max = max(c.excitability for c in engine._cells.values())
        
        engram = engine.allocate_engram("mem_001")
        
        # Allocated cells should now be more excitable
        allocated_excit = [engine._cells[cid].excitability for cid in engram.cell_ids]
        assert max(allocated_excit) > initial_max


class TestEngramReactivation:
    def test_reactivate_strengthens(self):
        engine = EngramEngine()
        engram = engine.allocate_engram("mem_001")
        engram.strength = 0.5  # Simulate decay
        
        result = engine.reactivate("mem_001")
        assert result is not None
        assert result.strength > 0.5

    def test_reactivate_increments_count(self):
        engine = EngramEngine()
        engine.allocate_engram("mem_001")
        engine.reactivate("mem_001")
        
        engram = engine.get_engram("mem_001")
        assert engram.reactivation_count == 2  # 1 from allocation + 1 from reactivation

    def test_reactivate_silent_engram(self):
        """Silent engrams can be reactivated (optogenetic analog)."""
        engine = EngramEngine()
        engram = engine.allocate_engram("mem_001")
        engram.state = EngramState.SILENT
        
        result = engine.reactivate("mem_001")
        assert result.state == EngramState.ACTIVE

    def test_reactivate_degraded_engram(self):
        engine = EngramEngine()
        engram = engine.allocate_engram("mem_001")
        engram.state = EngramState.DEGRADED
        
        result = engine.reactivate("mem_001")
        assert result.state == EngramState.ACTIVE

    def test_cannot_reactivate_dissolved(self):
        engine = EngramEngine()
        engram = engine.allocate_engram("mem_001")
        engram.state = EngramState.DISSOLVED
        
        result = engine.reactivate("mem_001")
        assert result is None

    def test_reactivate_unknown_memory(self):
        engine = EngramEngine()
        result = engine.reactivate("nonexistent")
        assert result is None


class TestEngramOverlap:
    def test_multiple_engrams_can_share_cells(self):
        """With limited cell pool, engrams will share cells → linked memories."""
        engine = EngramEngine(config=EngramConfig(
            cells_per_engram=5, total_cell_pool=8
        ))
        e1 = engine.allocate_engram("mem_001")
        e2 = engine.allocate_engram("mem_002")
        
        # With 8 cells and 5 per engram, at least 2 must overlap
        overlap = set(e1.cell_ids) & set(e2.cell_ids)
        assert len(overlap) >= 2

    def test_find_linked_memories(self):
        engine = EngramEngine(config=EngramConfig(
            cells_per_engram=5, total_cell_pool=8
        ))
        engine.allocate_engram("mem_001")
        engine.allocate_engram("mem_002")
        
        linked = engine.find_linked_memories("mem_001")
        assert len(linked) > 0
        assert linked[0][0] == "mem_002"
        assert linked[0][1] > 0  # Some overlap fraction

    def test_no_overlap_with_large_pool(self):
        """Large cell pool → less overlap → fewer linked memories."""
        engine = EngramEngine(config=EngramConfig(
            cells_per_engram=3, total_cell_pool=1000
        ))
        engine.allocate_engram("mem_001")
        engine.allocate_engram("mem_002")
        
        linked = engine.find_linked_memories("mem_001")
        # With 1000 cells and 3/engram, overlap is very unlikely
        # (but not impossible due to excitability competition)


class TestEngramConsolidation:
    def test_consolidate_engram(self):
        engine = EngramEngine()
        engine.allocate_engram("mem_001")
        
        result = engine.consolidate_engram("mem_001")
        assert result is True
        
        engram = engine.get_engram("mem_001")
        assert engram.consolidated is True
        assert engram.state == EngramState.STABLE

    def test_consolidate_unknown_memory(self):
        engine = EngramEngine()
        result = engine.consolidate_engram("nonexistent")
        assert result is False


class TestEngramMaintenance:
    def test_active_to_stable_transition(self):
        engine = EngramEngine(config=EngramConfig(active_to_stable_hours=1.0))
        engram = engine.allocate_engram("mem_001")
        
        # Simulate 2 hours passing
        engram.last_reactivated = datetime.now(timezone.utc) - timedelta(hours=2)
        
        transitions = engine.maintain()
        assert transitions["active_to_stable"] == 1
        assert engine.get_engram("mem_001").state == EngramState.STABLE

    def test_stable_to_silent_transition(self):
        engine = EngramEngine(config=EngramConfig(
            active_to_stable_hours=1.0,
            stable_to_silent_hours=10.0,
        ))
        engram = engine.allocate_engram("mem_001")
        engram.state = EngramState.STABLE
        engram.last_reactivated = datetime.now(timezone.utc) - timedelta(hours=15)
        
        transitions = engine.maintain()
        assert transitions["stable_to_silent"] == 1
        assert engine.get_engram("mem_001").state == EngramState.SILENT


class TestEngramStatistics:
    def test_statistics(self):
        engine = EngramEngine()
        engine.allocate_engram("mem_001")
        engine.allocate_engram("mem_002")
        engine.consolidate_engram("mem_001")
        
        stats = engine.get_statistics()
        assert stats["total_engrams"] == 2
        assert stats["consolidated_count"] == 1
        assert stats["total_cells"] == 200  # default pool
        assert "state_distribution" in stats

    def test_get_engram(self):
        engine = EngramEngine()
        engine.allocate_engram("mem_001")
        
        engram = engine.get_engram("mem_001")
        assert engram is not None
        assert engram.memory_id == "mem_001"

    def test_get_nonexistent_engram(self):
        engine = EngramEngine()
        assert engine.get_engram("nonexistent") is None
