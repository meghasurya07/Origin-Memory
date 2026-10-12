"""
Experiment 06: Multi-Agent Shared Memory
=========================================

Tests that the shared memory pool correctly manages:
1. Scoped access (agents only see their groups)
2. Conflict resolution (temporal supersession)
3. Provenance tracking (who wrote what)
4. Multi-agent collaboration via shared knowledge
5. Brain + SharedPool integration (agents publish during consolidation)
"""

import sys
sys.path.insert(0, '.')

from origin_brain.brain import Brain
from origin_brain.models import BrainConfig
from origin_brain.shared_memory import (
    SharedMemoryPool, SharedMemoryConfig, MemoryScope, ConflictStrategy,
)


def test_01_basic_publish_query():
    """Test: Multiple agents can publish and query shared knowledge."""
    print("\n" + "="*70)
    print("TEST 01: Multi-Agent Publish & Query")
    print("="*70)
    
    pool = SharedMemoryPool()
    
    # Agent 1 discovers a fact
    pool.publish("researcher_1", "protein_X_function", "kinase activity", scope=MemoryScope.GLOBAL)
    
    # Agent 2 discovers related info
    pool.publish("researcher_2", "protein_X_structure", "alpha helix domain", scope=MemoryScope.GLOBAL)
    
    # Agent 3 queries — should see both
    results = pool.query("researcher_3")
    
    print(f"  Researcher 1 published: protein_X_function")
    print(f"  Researcher 2 published: protein_X_structure")
    print(f"  Researcher 3 query results: {len(results)}")
    
    for r in results:
        print(f"    {r.key} = {r.value} (by {r.author_agent_id})")
    
    passed = len(results) == 2
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_02_scoped_access():
    """Test: Group-scoped memories enforce access control."""
    print("\n" + "="*70)
    print("TEST 02: Scoped Access Control")
    print("="*70)
    
    pool = SharedMemoryPool()
    
    # Create two teams
    pool.join_group("alice", "bio_team")
    pool.join_group("bob", "bio_team")
    pool.join_group("charlie", "eng_team")
    pool.join_group("diana", "eng_team")
    
    # Bio team shares internal findings
    pool.publish("alice", "experiment_result", "positive",
                scope=MemoryScope.GROUP, group_id="bio_team")
    
    # Eng team shares their findings
    pool.publish("charlie", "system_status", "nominal",
                scope=MemoryScope.GROUP, group_id="eng_team")
    
    # Bob (bio_team) should see bio findings, NOT eng findings
    bob_results = pool.query("bob", include_global=False)
    bob_keys = [r.key for r in bob_results]
    
    # Diana (eng_team) should see eng findings, NOT bio findings
    diana_results = pool.query("diana", include_global=False)
    diana_keys = [r.key for r in diana_results]
    
    print(f"  Bob (bio_team) sees: {bob_keys}")
    print(f"  Diana (eng_team) sees: {diana_keys}")
    
    bob_ok = "experiment_result" in bob_keys and "system_status" not in bob_keys
    diana_ok = "system_status" in diana_keys and "experiment_result" not in diana_keys
    
    passed = bob_ok and diana_ok
    print(f"\n  Bob correctly scoped: {bob_ok}")
    print(f"  Diana correctly scoped: {diana_ok}")
    print(f"  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_03_conflict_resolution():
    """Test: Conflicting facts are resolved via temporal supersession."""
    print("\n" + "="*70)
    print("TEST 03: Conflict Resolution (Latest Wins)")
    print("="*70)
    
    pool = SharedMemoryPool()
    
    # Two agents disagree on a fact
    pool.publish("agent_1", "CEO_name", "Alice Johnson", scope=MemoryScope.GLOBAL, confidence=0.8)
    pool.publish("agent_2", "CEO_name", "Robert Chen", scope=MemoryScope.GLOBAL, confidence=0.9)
    
    # Query — latest should win
    results = pool.query("agent_3", key="CEO_name")
    
    print(f"  Agent 1 says CEO = Alice Johnson (conf=0.8)")
    print(f"  Agent 2 says CEO = Robert Chen (conf=0.9)")
    print(f"  Resolved value: {results[0].value if results else 'NONE'}")
    print(f"  Conflicts detected: {pool._total_conflicts}")
    
    passed = (len(results) == 1 and 
              results[0].value == "Robert Chen" and
              pool._total_conflicts == 1)
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_04_provenance_tracking():
    """Test: We can trace which agent asserted each fact."""
    print("\n" + "="*70)
    print("TEST 04: Provenance Tracking")
    print("="*70)
    
    pool = SharedMemoryPool()
    
    pool.publish("researcher_1", "finding_A", "positive", scope=MemoryScope.GLOBAL)
    pool.publish("researcher_1", "finding_B", "negative", scope=MemoryScope.GLOBAL)
    pool.publish("researcher_2", "finding_C", "inconclusive", scope=MemoryScope.GLOBAL)
    
    r1_contributions = pool.get_agent_contributions("researcher_1")
    r2_contributions = pool.get_agent_contributions("researcher_2")
    
    print(f"  Researcher 1 contributed: {len(r1_contributions)} facts")
    print(f"  Researcher 2 contributed: {len(r2_contributions)} fact(s)")
    
    passed = len(r1_contributions) == 2 and len(r2_contributions) == 1
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_05_brain_shared_integration():
    """Test: Brain instances can share knowledge through the pool."""
    print("\n" + "="*70)
    print("TEST 05: Brain + SharedPool Integration")
    print("="*70)
    
    pool = SharedMemoryPool()
    
    # Create two brain instances (two agents)
    brain_1 = Brain(config=BrainConfig(agent_id="agent_alpha"))
    brain_2 = Brain(config=BrainConfig(agent_id="agent_beta"))
    
    # Agent 1 learns something
    brain_1.encode("The project deadline is December 15th")
    brain_1.encode("Budget approved for $500K")
    
    # Agent 1 publishes key facts to shared pool
    pool.publish("agent_alpha", "deadline", "December 15th", scope=MemoryScope.GLOBAL)
    pool.publish("agent_alpha", "budget", "$500K", scope=MemoryScope.GLOBAL)
    
    # Agent 2 queries shared pool for project info
    results = pool.query("agent_beta", key="deadline")
    
    print(f"  Agent Alpha encoded 2 memories and published to pool")
    print(f"  Agent Beta queries pool for deadline: {results[0].value if results else 'NOT FOUND'}")
    
    # Agent 2 can now encode shared knowledge into its own brain
    if results:
        brain_2.encode(f"Shared: deadline is {results[0].value}")
    
    # Verify agent_2 can recall it
    recall = brain_2.recall("project deadline")
    recall_content = recall.results[0].memory.content if recall.results else "NONE"
    
    print(f"  Agent Beta recalls: {recall_content}")
    
    passed = len(results) >= 1 and "December" in recall_content
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_06_broadcast():
    """Test: Agent can broadcast discoveries to all groups."""
    print("\n" + "="*70)
    print("TEST 06: Broadcast to All Groups")
    print("="*70)
    
    pool = SharedMemoryPool()
    
    pool.join_group("discoverer", "team_a")
    pool.join_group("discoverer", "team_b")
    pool.join_group("listener_a", "team_a")
    pool.join_group("listener_b", "team_b")
    
    # Broadcast a discovery
    entries = pool.broadcast("discoverer", "breakthrough", "cold fusion achieved", confidence=0.95)
    
    print(f"  Broadcast created {len(entries)} entries (global + 2 groups)")
    
    # Both listeners should see it
    results_a = pool.query("listener_a", key="breakthrough")
    results_b = pool.query("listener_b", key="breakthrough")
    
    print(f"  Listener A sees: {len(results_a)} result(s)")
    print(f"  Listener B sees: {len(results_b)} result(s)")
    
    stats = pool.get_statistics()
    print(f"  Pool stats: {stats}")
    
    passed = len(results_a) >= 1 and len(results_b) >= 1
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 06: Multi-Agent Shared Memory")
    print("Testing the Origin Brain Platform Layer")
    print("=" * 70)
    
    results = {
        "Publish & Query": test_01_basic_publish_query(),
        "Scoped Access": test_02_scoped_access(),
        "Conflict Resolution": test_03_conflict_resolution(),
        "Provenance Tracking": test_04_provenance_tracking(),
        "Brain + SharedPool": test_05_brain_shared_integration(),
        "Broadcast": test_06_broadcast(),
    }
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 06 SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS ✅" if result else "FAIL ❌"
        print(f"  {name:30s} {status}")
    
    print(f"\n  Score: {passed}/{total}")
    
    if passed == total:
        print("  🎉 ALL TESTS PASS — Multi-agent memory is working!")
    else:
        print(f"  ⚠️  {total - passed} test(s) failed")
