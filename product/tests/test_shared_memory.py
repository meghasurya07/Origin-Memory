"""Tests for Multi-Agent Shared Memory Pool."""
import pytest

from origin_brain.shared_memory import (
    SharedMemoryPool, SharedMemoryConfig, SharedMemoryEntry,
    MemoryScope, ConflictStrategy,
)


class TestSharedMemoryPool:
    @pytest.fixture
    def pool(self):
        return SharedMemoryPool()

    def test_publish_global(self, pool):
        entry = pool.publish("agent_1", "CEO", "Alice", scope=MemoryScope.GLOBAL)
        assert entry.key == "CEO"
        assert entry.value == "Alice"
        assert entry.scope == MemoryScope.GLOBAL
        assert entry.author_agent_id == "agent_1"

    def test_publish_group(self, pool):
        pool.join_group("agent_1", "team_a")
        entry = pool.publish("agent_1", "status", "done",
                            scope=MemoryScope.GROUP, group_id="team_a")
        assert entry.scope == MemoryScope.GROUP
        assert entry.group_id == "team_a"

    def test_query_global(self, pool):
        pool.publish("agent_1", "CEO", "Alice", scope=MemoryScope.GLOBAL)
        results = pool.query("agent_2", key="CEO")
        assert len(results) == 1
        assert results[0].value == "Alice"

    def test_query_scoped_access(self, pool):
        """Agents should only see their own group memories."""
        pool.join_group("agent_1", "team_a")
        pool.join_group("agent_2", "team_b")
        
        pool.publish("agent_1", "secret", "team_a_data",
                    scope=MemoryScope.GROUP, group_id="team_a")
        pool.publish("agent_2", "secret", "team_b_data",
                    scope=MemoryScope.GROUP, group_id="team_b")
        
        # agent_1 should see team_a but NOT team_b
        results_1 = pool.query("agent_1", key="secret", include_global=False)
        assert len(results_1) == 1
        assert results_1[0].value == "team_a_data"
        
        # agent_2 should see team_b but NOT team_a
        results_2 = pool.query("agent_2", key="secret", include_global=False)
        assert len(results_2) == 1
        assert results_2[0].value == "team_b_data"

    def test_conflict_latest_wins(self, pool):
        """Default: latest write should supersede earlier."""
        pool.publish("agent_1", "CEO", "Alice", scope=MemoryScope.GLOBAL)
        pool.publish("agent_2", "CEO", "Bob", scope=MemoryScope.GLOBAL)
        
        results = pool.query("agent_3", key="CEO")
        assert len(results) == 1
        assert results[0].value == "Bob"  # Latest wins
        assert pool._total_conflicts == 1

    def test_conflict_higher_confidence(self):
        pool = SharedMemoryPool(SharedMemoryConfig(
            conflict_strategy=ConflictStrategy.HIGHER_CONFIDENCE
        ))
        pool.publish("agent_1", "CEO", "Alice", scope=MemoryScope.GLOBAL, confidence=0.9)
        pool.publish("agent_2", "CEO", "Bob", scope=MemoryScope.GLOBAL, confidence=0.5)
        
        results = pool.query("agent_3", key="CEO")
        assert len(results) == 1
        assert results[0].value == "Alice"  # Higher confidence wins

    def test_conflict_keep_both(self):
        pool = SharedMemoryPool(SharedMemoryConfig(
            conflict_strategy=ConflictStrategy.KEEP_BOTH
        ))
        pool.publish("agent_1", "CEO", "Alice", scope=MemoryScope.GLOBAL)
        pool.publish("agent_2", "CEO", "Bob", scope=MemoryScope.GLOBAL)
        
        # The latest entry should be flagged as conflicted
        results = pool.query("agent_3", key="CEO")
        assert any(r.is_conflicted for r in results)

    def test_conflict_merge(self):
        pool = SharedMemoryPool(SharedMemoryConfig(
            conflict_strategy=ConflictStrategy.MERGE
        ))
        pool.publish("agent_1", "findings", "A is true", scope=MemoryScope.GLOBAL)
        pool.publish("agent_2", "findings", "B is also true", scope=MemoryScope.GLOBAL)
        
        results = pool.query("agent_3", key="findings")
        assert len(results) == 1
        assert "A is true" in results[0].value
        assert "B is also true" in results[0].value

    def test_provenance_tracking(self, pool):
        pool.publish("agent_1", "fact_1", "v1", scope=MemoryScope.GLOBAL)
        pool.publish("agent_1", "fact_2", "v2", scope=MemoryScope.GLOBAL)
        pool.publish("agent_2", "fact_3", "v3", scope=MemoryScope.GLOBAL)
        
        agent_1_contributions = pool.get_agent_contributions("agent_1")
        agent_2_contributions = pool.get_agent_contributions("agent_2")
        assert len(agent_1_contributions) == 2
        assert len(agent_2_contributions) == 1

    def test_join_leave_group(self, pool):
        pool.join_group("agent_1", "team_a")
        assert "team_a" in pool._agent_groups["agent_1"]
        assert "agent_1" in pool._group_agents["team_a"]
        
        pool.leave_group("agent_1", "team_a")
        assert "team_a" not in pool._agent_groups["agent_1"]

    def test_broadcast(self, pool):
        pool.join_group("agent_1", "team_a")
        pool.join_group("agent_1", "team_b")
        
        entries = pool.broadcast("agent_1", "announcement", "important news")
        # Should publish to global + team_a + team_b = 3
        assert len(entries) == 3

    def test_category_filter(self, pool):
        pool.publish("agent_1", "fact_1", "v1", scope=MemoryScope.GLOBAL, category="science")
        pool.publish("agent_1", "fact_2", "v2", scope=MemoryScope.GLOBAL, category="business")
        
        results = pool.query("agent_2", category="science")
        assert len(results) == 1
        assert results[0].key == "fact_1"

    def test_statistics(self, pool):
        pool.publish("agent_1", "k", "v", scope=MemoryScope.GLOBAL)
        pool.query("agent_2")
        
        stats = pool.get_statistics()
        assert stats["global_memories"] == 1
        assert stats["total_publishes"] == 1
        assert stats["total_queries"] == 1

    def test_group_members(self, pool):
        pool.join_group("agent_1", "team")
        pool.join_group("agent_2", "team")
        pool.join_group("agent_3", "team")
        
        members = pool.get_group_members("team")
        assert len(members) == 3

    def test_conflict_log(self, pool):
        pool.publish("agent_1", "fact", "version_1", scope=MemoryScope.GLOBAL)
        pool.publish("agent_2", "fact", "version_2", scope=MemoryScope.GLOBAL)
        
        conflicts = pool.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0]["existing_value"] == "version_1"
        assert conflicts[0]["new_value"] == "version_2"

    def test_group_scope_requires_group_id(self, pool):
        with pytest.raises(ValueError, match="group_id required"):
            pool.publish("agent_1", "key", "val", scope=MemoryScope.GROUP)

    def test_multi_group_agent(self, pool):
        """Agent in multiple groups should see all group memories."""
        pool.join_group("agent_1", "team_a")
        pool.join_group("agent_1", "team_b")
        
        pool.publish("agent_2", "a_fact", "val_a",
                    scope=MemoryScope.GROUP, group_id="team_a")
        pool.publish("agent_3", "b_fact", "val_b",
                    scope=MemoryScope.GROUP, group_id="team_b")
        
        results = pool.query("agent_1", include_global=False)
        assert len(results) == 2

    def test_no_superseded_in_results(self, pool):
        """Superseded memories should not appear in query results."""
        pool.publish("agent_1", "fact", "old", scope=MemoryScope.GLOBAL)
        pool.publish("agent_2", "fact", "new", scope=MemoryScope.GLOBAL)
        
        results = pool.query("agent_3", key="fact")
        assert len(results) == 1
        assert results[0].value == "new"

    def test_access_count_increments(self, pool):
        pool.publish("agent_1", "fact", "value", scope=MemoryScope.GLOBAL)
        
        pool.query("agent_2", key="fact")
        pool.query("agent_3", key="fact")
        pool.query("agent_4", key="fact")
        
        results = pool.query("agent_5", key="fact")
        assert results[0].access_count == 4  # 3 previous + this query
