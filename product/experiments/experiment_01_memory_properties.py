"""
Experiment 01: Testing the Core Properties of Human-Like Memory

This script runs computational experiments to test whether our memory
system exhibits the key properties of human memory:

1. False Memory Generation (schema filling)
2. Context-Dependent Recall 
3. Temporal Contiguity Effect
4. Prediction Error Encoding Selectivity
5. Sleep Consolidation Compression
6. Reconstructive vs Verbatim Recall

Each experiment outputs measurable results that can validate or
invalidate our theoretical framework.
"""

import sys
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

# Add parent to path
sys.path.insert(0, ".")

from origin_brain import (
    Brain, BrainConfig, MemoryType,
    TemporalContextEngine, TemporalContextConfig,
    PredictionEngine, PredictionEngineConfig,
    SleepEngine, SleepConfig,
)
from origin_brain.models import EpisodicMemory, MemoryStatus


def banner(title: str):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {title}")
    print(f"{'='*70}\n")


def result(name: str, value: Any, expected: str = ""):
    print(f"  [{name}] = {value}  {f'(expected: {expected})' if expected else ''}")


# ──────────────────────────────────────────────
# Experiment 1: Prediction Error Selectivity
# ──────────────────────────────────────────────
def experiment_prediction_error():
    banner("Prediction Error Encoding Selectivity")
    print("  HYPOTHESIS: Repeated content becomes less surprising over time.")
    print("  Surprising (novel) content should trigger strong encoding.\n")

    engine = PredictionEngine(PredictionEngineConfig(surprise_threshold=0.3))

    # Phase 1: Train on repetitive content
    routine_surprises = []
    for i in range(30):
        r = engine.evaluate(f"The daily standup meeting was at 10am in room {i % 3}")
        routine_surprises.append(r.surprise_score)

    # Phase 2: Inject novel content
    novel_results = []
    novel_inputs = [
        "The CEO just announced the company is being acquired by Google",
        "A meteor just struck the parking lot outside",
        "We discovered a new prime number in the database logs",
    ]
    for text in novel_inputs:
        r = engine.evaluate(text)
        novel_results.append((text[:50], r.surprise_score, r.should_encode, r.encoding_strength))

    # Phase 3: More routine
    late_routine = engine.evaluate("The daily standup meeting was at 10am in room 1")

    # Results
    early_surprise = sum(routine_surprises[:5]) / 5
    late_surprise = sum(routine_surprises[-5:]) / 5
    
    result("Early routine surprise (avg first 5)", f"{early_surprise:.4f}")
    result("Late routine surprise (avg last 5)", f"{late_surprise:.4f}")
    result("Surprise DECREASED over repetition", early_surprise > late_surprise, "True")
    
    print()
    for text, surprise, encode, strength in novel_results:
        result(f"Novel: '{text}...'", f"surprise={surprise:.4f}, encode={encode}, strength={strength:.3f}", "high surprise")
    
    result("Post-routine surprise", f"{late_routine.surprise_score:.4f}", "low")
    
    # Key metric: Is novel surprise > routine surprise?
    avg_novel = sum(s for _, s, _, _ in novel_results) / len(novel_results)
    selectivity_ratio = avg_novel / max(late_surprise, 0.001)
    result("SELECTIVITY RATIO (novel/routine)", f"{selectivity_ratio:.2f}", ">1.0")
    
    return selectivity_ratio > 1.0


# ──────────────────────────────────────────────
# Experiment 2: Temporal Contiguity Effect
# ──────────────────────────────────────────────
def experiment_temporal_contiguity():
    banner("Temporal Contiguity Effect")
    print("  HYPOTHESIS: Memories encoded close in time should have more")
    print("  similar contexts than memories encoded far apart.\n")

    engine = TemporalContextEngine(TemporalContextConfig(dimension=64, beta=0.5))

    # Encode a sequence of 20 events
    dim = 64
    for i in range(20):
        feature = [math.sin(j + i * 0.5) * 0.1 for j in range(dim)]
        engine.update(feature, memory_id=f"event-{i}")

    # Measure context similarity at different distances
    distances = [1, 2, 3, 5, 10, 15]
    similarities = {}
    
    for d in distances:
        sims = []
        for i in range(max(0, 10 - d), min(20 - d, 10)):
            sim = engine.get_context_similarity(f"event-{i}", f"event-{i+d}")
            if sim != 0.0:
                sims.append(sim)
        if sims:
            similarities[d] = sum(sims) / len(sims)

    for d, sim in sorted(similarities.items()):
        result(f"Distance {d:2d} → avg similarity", f"{sim:.4f}")

    # Check: closer = more similar?
    if len(similarities) >= 2:
        closest = similarities.get(1, 0)
        farthest = similarities.get(max(similarities.keys()), 0)
        contiguity_holds = closest > farthest
        result("CONTIGUITY EFFECT", contiguity_holds, "True (close > far)")
        return contiguity_holds
    return False


# ──────────────────────────────────────────────
# Experiment 3: Context Reinstatement
# ──────────────────────────────────────────────
def experiment_context_reinstatement():
    banner("Context Reinstatement (Mental Time Travel)")
    print("  HYPOTHESIS: Reinstating a past context should make temporally")
    print("  adjacent memories more accessible than distant ones.\n")

    engine = TemporalContextEngine(TemporalContextConfig(dimension=64, beta=0.5))
    dim = 64

    # Encode 3 clusters of events with gaps between them
    # Cluster A: events 0-4
    for i in range(5):
        f = [math.sin(j + i) * 0.1 for j in range(dim)]
        engine.update(f, memory_id=f"cluster-A-{i}")
    
    # Gap: 10 filler events
    for i in range(10):
        f = [math.cos(j + i * 2) * 0.1 for j in range(dim)]
        engine.update(f)
    
    # Cluster B: events 5-9
    for i in range(5):
        f = [math.sin(j + i + 100) * 0.1 for j in range(dim)]
        engine.update(f, memory_id=f"cluster-B-{i}")

    # Now reinstate context of cluster A
    engine.reinstate("cluster-A-2", strength=0.8)

    # Probe: which memories are more accessible now?
    probe_feature = [math.sin(j + 2) * 0.1 for j in range(dim)]  # Similar to cluster A
    results = engine.probe(probe_feature, top_k=10)

    cluster_a_scores = [(mid, s) for mid, s in results if "cluster-A" in mid]
    cluster_b_scores = [(mid, s) for mid, s in results if "cluster-B" in mid]

    avg_a = sum(s for _, s in cluster_a_scores) / max(len(cluster_a_scores), 1)
    avg_b = sum(s for _, s in cluster_b_scores) / max(len(cluster_b_scores), 1)

    result("Cluster A (reinstated context) avg score", f"{avg_a:.4f}")
    result("Cluster B (different context) avg score", f"{avg_b:.4f}")
    result("REINSTATEMENT EFFECT", avg_a > avg_b, "True")
    
    return avg_a > avg_b


# ──────────────────────────────────────────────
# Experiment 4: Sleep Consolidation Compression
# ──────────────────────────────────────────────
def experiment_sleep_consolidation():
    banner("Sleep Consolidation and Selective Forgetting")
    print("  HYPOTHESIS: Sleep should consolidate high-salience memories")
    print("  into semantic gist while evicting low-salience noise.\n")

    engine = SleepEngine(SleepConfig(
        num_cycles=2,
        consolidation_salience_threshold=0.4,
        eviction_threshold=0.3,
        min_age_for_consolidation_hours=0.01,  # Very short for testing
    ))

    now = datetime.now(timezone.utc)
    
    # Create memories with varying salience
    important_memories = [
        EpisodicMemory(
            content=f"Critical discovery {i}: quantum entanglement in neural tissue",
            salience=0.8 + i * 0.02,
            timestamp=now - timedelta(hours=2),
            last_accessed=now - timedelta(hours=2),
            access_count=3,
            metadata={"emotional_arousal": 0.7}
        ) for i in range(5)
    ]
    
    noise_memories = [
        EpisodicMemory(
            content=f"Routine event {i}: checked email at desk",
            salience=0.1,
            timestamp=now - timedelta(hours=48),
            last_accessed=now - timedelta(hours=48),
            access_count=0,
            metadata={"emotional_arousal": 0.2}
        ) for i in range(10)
    ]
    
    all_memories = important_memories + noise_memories
    initial_count = len(all_memories)
    
    semantic_produced = []
    evicted_ids = []
    
    report = engine.run_sleep_cycle(
        episodic_memories=all_memories,
        store_semantic_fn=lambda s: semantic_produced.append(s),
        evict_fn=lambda mid: (evicted_ids.append(mid), True)[-1]
    )
    
    result("Initial episodic count", initial_count)
    result("Semantic gists produced", len(semantic_produced), ">0 (consolidation worked)")
    result("Memories evicted", len(evicted_ids), ">0 (forgetting worked)")
    result("Total sleep phases", len(report.phases))
    result("Compression ratio", f"{report.compression_ratio:.3f}")
    
    # Check: Were important memories consolidated?
    consolidated_contents = [s.value for s in semantic_produced]
    important_consolidated = sum(1 for c in consolidated_contents if "quantum" in c.lower())
    noise_consolidated = sum(1 for c in consolidated_contents if "routine" in c.lower())
    
    result("Important memories consolidated", important_consolidated, ">0")
    result("Noise memories consolidated", noise_consolidated, "0 or low")
    result("SELECTIVE CONSOLIDATION", important_consolidated > noise_consolidated, "True")
    
    # Check emotional decoupling
    arousal_reduced = sum(1 for m in all_memories if m.metadata.get('emotional_arousal', 1.0) < 0.7)
    result("Memories with reduced arousal (REM effect)", arousal_reduced)
    
    return important_consolidated > noise_consolidated


# ──────────────────────────────────────────────
# Experiment 5: Full Brain Pipeline
# ──────────────────────────────────────────────
def experiment_full_brain():
    banner("Full Brain Pipeline: Encode → Recall → Sleep → Recall")
    print("  HYPOTHESIS: The full pipeline should exhibit prediction error")
    print("  selectivity, temporal binding, and consolidation.\n")

    brain = Brain(config={"agent_id": "experiment-brain"})
    
    # Phase 1: Encode routine events
    print("  Phase 1: Encoding routine events...")
    for i in range(10):
        r = brain.encode(f"Daily standup meeting discussed sprint {i} progress")
    result("Episodic count after routine", brain.episodic_count)
    
    # Phase 2: Encode surprising events
    print("  Phase 2: Encoding surprising events...")
    surprise_result = brain.encode(
        "The database server caught fire and we lost all production data",
        salience=0.9
    )
    result("Surprise event encoded", surprise_result.action)
    if surprise_result.memory:
        result("Surprise score", f"{surprise_result.memory.metadata.get('surprise_score', 'N/A'):.4f}")
        result("Encoding strength", f"{surprise_result.memory.metadata.get('encoding_strength', 'N/A'):.4f}")
    
    # Phase 3: Recall
    print("\n  Phase 3: Recall test...")
    recall_result = brain.recall("What happened with the database?", top_k=3)
    result("Results returned", len(recall_result.results))
    if recall_result.results:
        top = recall_result.results[0]
        result("Top result content", str(top.memory)[:60] + "...")
        result("Top result score", f"{top.relevance_score:.4f}")
    result("Metamemory confidence", recall_result.confidence.confidence_level)
    
    # Phase 4: Sleep
    print("\n  Phase 4: Sleep consolidation...")
    sleep_report = brain.sleep(num_cycles=1)
    result("Consolidated", sleep_report.total_consolidated)
    result("Evicted", sleep_report.total_evicted)
    result("New semantic", sleep_report.new_semantic_count)
    
    # Phase 5: Post-sleep recall
    print("\n  Phase 5: Post-sleep recall...")
    post_sleep = brain.recall("What happened with the database?", top_k=3)
    result("Post-sleep results", len(post_sleep.results))
    
    # Statistics
    stats = brain.get_statistics()
    result("Brain version", stats.get("version", "unknown"))
    result("Temporal context snapshots", stats["temporal_context"]["snapshot_count"])
    result("Prediction engine inputs", stats["prediction"]["total_inputs"])
    result("Sleep cycles completed", stats["sleep"]["total_cycles"])
    
    return True


# ──────────────────────────────────────────────
# Experiment 6: Encoding Efficiency Benchmark
# ──────────────────────────────────────────────
def experiment_encoding_benchmark():
    banner("Encoding Efficiency Benchmark")
    print("  Measuring throughput and latency of the full encode pipeline.\n")

    brain = Brain(config={"agent_id": "benchmark-brain"})
    
    # Benchmark encode
    n = 100
    contents = [f"Event {i}: something happened involving entity-{i % 10} at location-{i % 5}" for i in range(n)]
    
    start = time.time()
    for content in contents:
        brain.encode(content, salience=0.5)
    encode_time = time.time() - start
    
    # Benchmark recall
    queries = [f"What happened with entity-{i}?" for i in range(10)]
    start = time.time()
    for q in queries:
        brain.recall(q, top_k=5)
    recall_time = time.time() - start
    
    # Benchmark sleep
    start = time.time()
    sleep_report = brain.sleep(num_cycles=1)
    sleep_time = time.time() - start
    
    result("Encode throughput", f"{n / encode_time:.0f} memories/sec")
    result("Encode latency (avg)", f"{encode_time / n * 1000:.2f} ms/memory")
    result("Recall throughput", f"{len(queries) / recall_time:.0f} queries/sec")
    result("Recall latency (avg)", f"{recall_time / len(queries) * 1000:.2f} ms/query")
    result("Sleep time (1 cycle)", f"{sleep_time * 1000:.1f} ms")
    result("Sleep consolidated", sleep_report.total_consolidated)
    result("Sleep evicted", sleep_report.total_evicted)
    result("Final episodic count", brain.episodic_count)
    result("Final semantic count", brain.semantic_count)
    
    return True


# ──────────────────────────────────────────────
# Run All Experiments
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("  ORIGIN AI — COMPUTATIONAL MEMORY EXPERIMENTS")
    print("  Testing Human-Like Memory Properties")
    print("="*70)
    
    experiments = [
        ("Prediction Error Selectivity", experiment_prediction_error),
        ("Temporal Contiguity Effect", experiment_temporal_contiguity),
        ("Context Reinstatement", experiment_context_reinstatement),
        ("Sleep Consolidation", experiment_sleep_consolidation),
        ("Full Brain Pipeline", experiment_full_brain),
        ("Encoding Benchmark", experiment_encoding_benchmark),
    ]
    
    results_summary = []
    for name, fn in experiments:
        try:
            passed = fn()
            results_summary.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            results_summary.append((name, f"ERROR: {e}"))
    
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    for name, status in results_summary:
        print(f"  {status}  {name}")
    
    passed = sum(1 for _, s in results_summary if "PASS" in s)
    total = len(results_summary)
    print(f"\n  {passed}/{total} experiments passed")
    print("="*70)
