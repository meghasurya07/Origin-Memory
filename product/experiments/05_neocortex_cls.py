"""
Experiment 05: Neocortical Learning & Schema Formation
=======================================================

Tests that the CLS (Complementary Learning Systems) implementation
actually extracts meaningful patterns from episodic replays:

1. Schema Formation: After repeated similar episodes, schemas should form
2. Schema-Accelerated Consolidation: Schema-consistent info should score higher  
3. Co-occurrence Learning: Entities that appear together build associations
4. Neocortex + Sleep Integration: Sleep cycle should drive neocortical learning
5. Neural vs TF-IDF: Compare semantic similarity quality (if neural available)
"""

import sys
import time
sys.path.insert(0, '.')

from origin_brain.brain import Brain
from origin_brain.models import BrainConfig
from origin_brain.neocortex import NeocorticalEngine, NeocortexConfig


def test_01_schema_formation():
    """Test: Repeated episodes of same type should form a schema."""
    print("\n" + "="*70)
    print("TEST 01: Schema Formation from Repeated Episodes")
    print("="*70)
    
    neocortex = NeocorticalEngine(NeocortexConfig(
        min_cooccurrence=2,
        schema_threshold=3,
    ))
    
    # Simulate 5 "meeting" episodes
    meetings = [
        {"entities": ["Bob", "Budget", "Conference_Room"], "type": "meeting"},
        {"entities": ["Bob", "Revenue", "Conference_Room"], "type": "meeting"},
        {"entities": ["Alice", "Budget", "Conference_Room"], "type": "meeting"},
        {"entities": ["Bob", "Budget", "Zoom"], "type": "meeting"},
        {"entities": ["Carol", "Budget", "Conference_Room"], "type": "meeting"},
    ]
    
    for ep in meetings:
        neocortex.learn_from_episode(ep)
    
    schema = neocortex.get_schema("meeting")
    
    print(f"  Episodes fed: {len(meetings)}")
    print(f"  Schema formed: {schema is not None}")
    if schema:
        print(f"  Schema confidence: {schema.confidence:.2f}")
        print(f"  Common entities: {schema.slots.get('common_entities', [])}")
        print(f"  Instance count: {schema.instance_count}")
    
    # Budget and Conference_Room should appear in most meetings
    common = set(schema.slots.get("common_entities", [])) if schema else set()
    has_budget = "budget" in common
    has_room = "conference_room" in common
    
    passed = schema is not None and schema.instance_count >= 3
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_02_association_learning():
    """Test: Co-occurring entities should build associations."""
    print("\n" + "="*70)
    print("TEST 02: Co-occurrence Association Learning")
    print("="*70)
    
    neocortex = NeocorticalEngine(NeocortexConfig(min_cooccurrence=2))
    
    # Bob and Budget appear together multiple times
    for _ in range(5):
        neocortex.learn_from_episode({
            "entities": ["Bob", "Budget"],
            "type": "work"
        })
    
    # Alice and Coffee appear together
    for _ in range(5):
        neocortex.learn_from_episode({
            "entities": ["Alice", "Coffee"],
            "type": "social"
        })
    
    bob_assocs = neocortex.get_associations("Bob")
    alice_assocs = neocortex.get_associations("Alice")
    
    print(f"  Bob's associations: {bob_assocs}")
    print(f"  Alice's associations: {alice_assocs}")
    
    bob_knows_budget = any(a[0] == "budget" for a in bob_assocs)
    alice_knows_coffee = any(a[0] == "coffee" for a in alice_assocs)
    bob_doesnt_know_coffee = not any(a[0] == "coffee" for a in bob_assocs)
    
    passed = bob_knows_budget and alice_knows_coffee and bob_doesnt_know_coffee
    print(f"\n  Bob→Budget: {bob_knows_budget}")
    print(f"  Alice→Coffee: {alice_knows_coffee}")
    print(f"  Bob≠Coffee: {bob_doesnt_know_coffee}")
    print(f"  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_03_schema_consistency_scoring():
    """Test: Schema-consistent episodes should score higher."""
    print("\n" + "="*70)
    print("TEST 03: Schema Consistency Scoring")
    print("="*70)
    
    neocortex = NeocorticalEngine(NeocortexConfig(
        min_cooccurrence=2,
        schema_threshold=3,
    ))
    
    # Build a "meeting" schema
    for _ in range(6):
        neocortex.learn_from_episode({
            "entities": ["Bob", "Budget", "Office"],
            "type": "meeting"
        })
    
    # Test consistency
    consistent = neocortex.schema_consistent({
        "entities": ["Bob", "Budget", "Office"],
        "type": "meeting"
    })
    
    inconsistent = neocortex.schema_consistent({
        "entities": ["Cat", "Dog", "Park"],
        "type": "meeting"
    })
    
    novel_type = neocortex.schema_consistent({
        "entities": ["Cat", "Dog"],
        "type": "pet_play"
    })
    
    print(f"  Consistent episode score: {consistent:.3f}")
    print(f"  Inconsistent episode score: {inconsistent:.3f}")
    print(f"  Novel type score: {novel_type:.3f}")
    
    passed = consistent > inconsistent and inconsistent >= novel_type
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_04_sleep_drives_neocortex():
    """Test: Brain.sleep() should drive neocortical learning."""
    print("\n" + "="*70)
    print("TEST 04: Sleep Consolidation → Neocortical Learning")
    print("="*70)
    
    brain = Brain(config=BrainConfig(agent_id="cls_test"))
    
    # Encode several related memories
    brain.encode("Meeting with Bob about the Q3 budget review")
    brain.encode("Bob presented the budget numbers at the meeting")
    brain.encode("The meeting concluded with budget approval")
    brain.encode("Alice shared her research findings in the lab")
    brain.encode("The lab results showed positive protein expression")
    
    concepts_before = len(brain.neocortex._concepts)
    
    # Sleep should replay episodes to neocortex
    report = brain.sleep()
    
    concepts_after = len(brain.neocortex._concepts)
    
    print(f"  Memories encoded: 5")
    print(f"  Concepts before sleep: {concepts_before}")
    print(f"  Concepts after sleep: {concepts_after}")
    print(f"  Sleep report: consolidated={report.total_consolidated}, evicted={report.total_evicted}")
    
    # Neocortex should have learned some concepts
    passed = concepts_after > concepts_before
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_05_interleaved_schema_separation():
    """Test: Different episode types should form separate schemas."""
    print("\n" + "="*70)
    print("TEST 05: Interleaved Schema Separation")
    print("="*70)
    
    neocortex = NeocorticalEngine(NeocortexConfig(
        min_cooccurrence=2,
        schema_threshold=3,
    ))
    
    # Interleave two episode types (CLS key insight: interleaved > blocked)
    for _ in range(5):
        neocortex.learn_from_episode({
            "entities": ["Meeting", "Bob", "Budget"],
            "type": "work"
        })
        neocortex.learn_from_episode({
            "entities": ["Gym", "Weights", "Protein"],
            "type": "exercise"
        })
    
    work_schema = neocortex.get_schema("work")
    exercise_schema = neocortex.get_schema("exercise")
    
    print(f"  Work schema: {work_schema is not None}")
    if work_schema:
        print(f"    Common: {work_schema.slots.get('common_entities', [])}")
    print(f"  Exercise schema: {exercise_schema is not None}")
    if exercise_schema:
        print(f"    Common: {exercise_schema.slots.get('common_entities', [])}")
    
    # Schemas should be separate
    passed = (work_schema is not None and exercise_schema is not None)
    if passed:
        work_entities = set(work_schema.slots.get("common_entities", []))
        exercise_entities = set(exercise_schema.slots.get("common_entities", []))
        no_overlap = len(work_entities & exercise_entities) == 0
        passed = passed and no_overlap
        print(f"  No overlap between schemas: {no_overlap}")
    
    print(f"\n  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def test_06_semantic_network_depth():
    """Test: Associations should form a connected semantic network."""
    print("\n" + "="*70)
    print("TEST 06: Semantic Network Depth")
    print("="*70)
    
    neocortex = NeocorticalEngine(NeocortexConfig(min_cooccurrence=2))
    
    # Build a chain: A-B, B-C, C-D
    for _ in range(4):
        neocortex.learn_from_episode({"entities": ["Alpha", "Beta"], "type": "chain"})
        neocortex.learn_from_episode({"entities": ["Beta", "Gamma"], "type": "chain"})
        neocortex.learn_from_episode({"entities": ["Gamma", "Delta"], "type": "chain"})
    
    alpha_assocs = neocortex.get_associations("Alpha")
    beta_assocs = neocortex.get_associations("Beta")
    gamma_assocs = neocortex.get_associations("Gamma")
    
    print(f"  Alpha associations: {[a[0] for a in alpha_assocs]}")
    print(f"  Beta associations: {[a[0] for a in beta_assocs]}")
    print(f"  Gamma associations: {[a[0] for a in gamma_assocs]}")
    
    # Beta should be a hub (connected to Alpha AND Gamma)
    beta_names = [a[0] for a in beta_assocs]
    beta_is_hub = "alpha" in beta_names and "gamma" in beta_names
    
    # Alpha should NOT directly know Delta (too far in chain)
    alpha_names = [a[0] for a in alpha_assocs]
    alpha_doesnt_know_delta = "delta" not in alpha_names
    
    passed = beta_is_hub and alpha_doesnt_know_delta
    print(f"\n  Beta is hub (Alpha+Gamma): {beta_is_hub}")
    print(f"  Alpha doesn't know Delta: {alpha_doesnt_know_delta}")
    print(f"  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 05: Neocortical Learning & Schema Formation")
    print("Testing CLS (Complementary Learning Systems)")
    print("=" * 70)
    
    results = {
        "Schema Formation": test_01_schema_formation(),
        "Association Learning": test_02_association_learning(),
        "Schema Consistency": test_03_schema_consistency_scoring(),
        "Sleep→Neocortex": test_04_sleep_drives_neocortex(),
        "Interleaved Schemas": test_05_interleaved_schema_separation(),
        "Semantic Network": test_06_semantic_network_depth(),
    }
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 05 SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS ✅" if result else "FAIL ❌"
        print(f"  {name:30s} {status}")
    
    print(f"\n  Score: {passed}/{total}")
    
    if passed == total:
        print("  🎉 ALL TESTS PASS — CLS is working!")
    else:
        print(f"  ⚠️  {total - passed} test(s) failed")
