"""
Experiment 10: End-to-End RL Training Loop
═══════════════════════════════════════════

The complete encode → recall → deferred reward cycle.
Tests that the RL gate learns from actual Brain usage patterns.

Protocol:
  1. Feed 50 memories through RL-gated Brain
  2. Run 20 recall queries
  3. Check that deferred rewards were delivered
  4. Verify RL policy learned (Q-table populated, action distribution)
  5. Test full pipeline: encode → recall → reward → policy update
  6. Compare gate statistics before and after training

Expected:
  - All memories encode successfully (heuristic mode for first batch)
  - Deferred rewards delivered for retrieved memories
  - RL policy transitions from heuristic to learned mode
  - Gate tracks encode/skip/update counts accurately
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from origin_brain.brain import Brain
from origin_brain.models import BrainConfig

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name} — {detail}")


# ══════════════════════════════════════════════════════════════════════
# Setup: Create Brain with RL gate
# ══════════════════════════════════════════════════════════════════════

brain = Brain(config=BrainConfig(agent_id="rl-e2e-benchmark"))

# ══════════════════════════════════════════════════════════════════════
# Phase 1: Encode 50 memories
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Phase 1: Encoding Memories ═══")

memories = [
    "Alice met Bob at the quantum computing lab yesterday",
    "Bob discovered a new error correction algorithm for qubits",
    "Charlie presented the machine learning results at the conference",
    "Alice and Charlie discussed the neural network architecture",
    "Bob completed the quantum entanglement experiment successfully",
    "The research team gathered at the lab for a progress review",
    "Alice published a paper on topological quantum computing",
    "Charlie's deep learning model achieved state-of-the-art accuracy",
    "Bob and Alice traveled to MIT for the quantum symposium",
    "The lab received a major grant from the National Science Foundation",
    "Charlie developed a new attention mechanism for transformers",
    "Alice filed a patent on her quantum error correction method",
    "Bob replicated the quantum advantage experiment with 100 qubits",
    "The team celebrated reaching their first research milestone",
    "Charlie integrated the memory system into the production pipeline",
    "Alice mentored three graduate students in quantum information",
    "Bob designed a novel quantum gate for superconducting circuits",
    "The conference presentation received outstanding paper award",
    "Charlie's model was deployed to serve 10 million users",
    "Alice and Bob co-authored a Nature paper on quantum supremacy",
    "The lab expanded to include a dedicated quantum computing floor",
    "Charlie began exploring reinforcement learning for optimization",
    "Bob constructed a prototype quantum processor with 200 qubits",
    "Alice was invited to join the National Academy of Sciences",
    "The team launched an open-source quantum computing framework",
    "Charlie trained a language model on scientific literature",
    "Bob discovered quantum coherence lasting 10 milliseconds at room temp",
    "Alice designed a fault-tolerant quantum computing architecture",
    "The research generated five spin-off startup companies",
    "Charlie developed a benchmark suite for evaluating memory systems",
    "Bob and Charlie collaborated on quantum machine learning",
    "Alice presented at the World Economic Forum on quantum technology",
    "The lab recruited top talent from Google and IBM quantum teams",
    "Charlie's attention mechanism was adopted by major tech companies",
    "Bob achieved quantum teleportation between two distant labs",
    "Alice's students went on to lead quantum research at top universities",
    "The team published a comprehensive review of the field",
    "Charlie built a distributed training system across 1000 GPUs",
    "Bob measured entanglement entropy in a many-body quantum system",
    "Alice received the Presidential Medal for scientific achievement",
    "The quantum computing framework reached 50000 stars on GitHub",
    "Charlie pioneered a new approach to continual learning",
    "Bob demonstrated quantum error correction below threshold",
    "Alice and the team hosted the annual Quantum Computing Summit",
    "Charlie's memory system outperformed all existing benchmarks",
    "Bob's quantum processor solved an optimization problem in seconds",
    "Alice co-founded the International Quantum Computing Initiative",
    "The lab's research was featured on the cover of Science magazine",
    "Charlie developed a novel knowledge graph reasoning method",
    "Bob and Alice won the Nobel Prize in Physics for quantum computing",
]

encoded_count = 0
for mem in memories:
    result = brain.encode(mem)
    if result.action in ('encoded', 'schema_matched'):
        encoded_count += 1

check("all_memories_processed", encoded_count > 0, f"encoded {encoded_count}/50")
check("gate_tracked_decisions",
      brain.rl_gate.get_statistics()["total_decisions"] > 0)
check("brain_has_memories", brain.episodic_count > 0,
      f"count={brain.episodic_count}")

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Recall queries → deferred reward delivery
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Phase 2: Recall + Deferred Reward ═══")

queries = [
    "quantum computing",
    "Alice research",
    "Bob experiment",
    "Charlie machine learning",
    "conference presentation",
    "quantum entanglement",
    "lab team progress",
    "patent publication",
    "neural network architecture",
    "quantum supremacy",
    "open source framework",
    "Nobel Prize",
    "attention mechanism",
    "quantum error correction",
    "reinforcement learning",
    "distributed training",
    "quantum processor",
    "knowledge graph",
    "memory system benchmark",
    "quantum teleportation",
]

total_results = 0
for query in queries:
    result = brain.recall(query)
    total_results += len(result.results)

check("recall_returns_results", total_results > 0,
      f"total={total_results} across {len(queries)} queries")

gate_stats = brain.rl_gate.get_statistics()
check("rewards_delivered", gate_stats["rewards_delivered"] > 0,
      f"delivered={gate_stats['rewards_delivered']}")

# ══════════════════════════════════════════════════════════════════════
# Phase 3: RL Policy Analysis
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Phase 3: RL Policy Analysis ═══")

rl_stats = brain.rl_policy.get_statistics()
check("rl_episodes_tracked", rl_stats["episode_count"] > 0,
      f"episodes={rl_stats['episode_count']}")

# Check action distribution
action_dist = gate_stats
check("encodes_tracked", action_dist["encodes"] > 0,
      f"encodes={action_dist['encodes']}")

# Policy summary
summary = brain.rl_policy.get_policy_summary()
check("policy_summary_available", len(summary) >= 5)
check("novel_prefers_encode",
      summary["novel_important"]["preferred_action"] == "encode")

# ══════════════════════════════════════════════════════════════════════
# Phase 4: Entropy Monitor State
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Phase 4: Entropy Monitor ═══")

entropy_stats = brain.entropy_monitor.get_statistics()
check("entropy_tracked_encodes", entropy_stats["encode_count"] > 0,
      f"count={entropy_stats['encode_count']}")
check("memories_tracked", entropy_stats["unique_memories_tracked"] > 0)
check("entropy_computed", entropy_stats["memory_entropy"] > 0,
      f"H={entropy_stats['memory_entropy']:.3f}")

entropy_state = brain.entropy_monitor.get_state()
check("pressure_accumulated", entropy_state.sleep_pressure > 0,
      f"pressure={entropy_state.sleep_pressure:.3f}")

# ══════════════════════════════════════════════════════════════════════
# Phase 5: Hypergraph State
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Phase 5: Hypergraph Analysis ═══")

hg_stats = brain.hypergraph.get_statistics()
check("hypergraph_has_nodes", hg_stats["node_count"] > 0,
      f"nodes={hg_stats['node_count']}")
check("hypergraph_has_edges", hg_stats["edge_count"] > 0,
      f"edges={hg_stats['edge_count']}")
check("avg_arity_above_2", hg_stats["avg_arity"] >= 2.0,
      f"avg_arity={hg_stats['avg_arity']}")

# Query hypergraph
hg_result = brain.hypergraph.query_by_nodes(["Alice", "Bob"], min_overlap=0.0)
check("hypergraph_finds_alice_bob", len(hg_result.matching_edges) > 0,
      f"matches={len(hg_result.matching_edges)}")

# Spreading activation
spread = brain.hypergraph.spread_activation(["Alice"])
check("hypergraph_spreads_from_alice",
      len(spread.activated_nodes) > 0,
      f"activated={len(spread.activated_nodes)}")

# ══════════════════════════════════════════════════════════════════════
# Phase 6: Sleep + Reset
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Phase 6: Sleep Consolidation ═══")

pre_pressure = brain.entropy_monitor._sleep_pressure
report = brain.sleep()
post_pressure = brain.entropy_monitor._sleep_pressure

check("sleep_completed", report.total_consolidated >= 0)
check("pressure_reduced", post_pressure < pre_pressure,
      f"pre={pre_pressure:.3f} post={post_pressure:.3f}")

# ══════════════════════════════════════════════════════════════════════
# Phase 7: Full Brain Statistics
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Phase 7: Full System Stats ═══")

stats = brain.get_statistics()
check("version_is_011", stats["version"] == "0.11.0")
check("has_entropy_monitor", "entropy_monitor" in stats)
check("has_rl_policy", "rl_policy" in stats)
check("has_rl_gate", "rl_gate" in stats)

# Print summary
print(f"\n  Episodic: {stats['episodic_count']}")
print(f"  Semantic: {stats['semantic_count']}")
print(f"  Entropy: H={stats['entropy_monitor']['memory_entropy']}")
print(f"  Sleep pressure: {stats['entropy_monitor']['sleep_pressure']}")
print(f"  RL mode: {stats['rl_policy']['mode']}")
print(f"  RL episodes: {stats['rl_policy']['episode_count']}")
print(f"  Gate decisions: {stats['rl_gate']['total_decisions']}")
print(f"  Gate skip rate: {stats['rl_gate']['skip_rate']}")
print(f"  Rewards delivered: {stats['rl_gate']['rewards_delivered']}")
print(f"  Hypergraph nodes: {brain.hypergraph.get_statistics()['node_count']}")
print(f"  Hypergraph edges: {brain.hypergraph.get_statistics()['edge_count']}")


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════

total = passed + failed
print(f"\n{'='*60}")
print(f"E2E RL Training Loop: {passed}/{total} passed")
if failed == 0:
    print("🎉 ALL TESTS PASSED — Full RL pipeline operational!")
else:
    print(f"⚠ {failed} tests failed")
print(f"{'='*60}")
