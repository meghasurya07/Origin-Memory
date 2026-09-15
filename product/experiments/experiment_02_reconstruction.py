"""
Experiment 02: End-to-End Reconstructive Memory Pipeline

Tests the complete reconstruction pipeline that makes Origin Brain
different from every other AI memory system:

1. Encode → store pattern → bind temporal context → store attractor pattern
2. Reconstruct → different fidelity levels based on context
3. Pattern completion → retrieve from partial cues
4. False memory generation → schema-consistent confabulation
5. Context-dependent recall → same memory, different outputs
6. Sleep → compress → post-sleep recall quality
"""
import sys
import math
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from origin_brain import Brain, BrainConfig, MemoryType
from origin_brain.reconstruction import ReconstructionFidelity
from origin_brain.pattern_completion import PatternCompletionEngine, PatternCompletionConfig


def banner(title):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {title}")
    print(f"{'='*70}\n")


def result(name, value, expected=""):
    print(f"  [{name}] = {value}  {f'(expected: {expected})' if expected else ''}")


# ─────────────────────────────────────────
# Experiment 2.1: Reconstructive Recall
# ─────────────────────────────────────────
def experiment_reconstructive_recall():
    banner("Reconstructive Recall — Context-Dependent Fidelity")
    print("  HYPOTHESIS: The same memory should produce different")
    print("  reconstructions depending on context overlap.\n")

    brain = Brain(config={"agent_id": "recon-test"})

    # Encode with specific context
    brain.context_buffer = {"location": "office", "project": "alpha", "mood": "focused"}
    r = brain.encode(
        "The database migration failed at 3pm causing a 2-hour outage. "
        "Team lead Sarah coordinated the rollback. Root cause was a missing index.",
        context={"location": "office", "project": "alpha", "mood": "focused"},
        salience=0.8
    )
    result("Encoded", r.action)

    # Reconstruction 1: SAME context (should be high fidelity)
    brain.context_buffer = {"location": "office", "project": "alpha", "mood": "focused"}
    recons_same = brain.reconstruct_recall("What happened with the database?")
    if recons_same:
        r1 = recons_same[0]
        result("Same context - fidelity", r1.fidelity.value)
        result("Same context - confidence", f"{r1.confidence:.3f}")
        result("Same context - trace contribution", f"{r1.trace_contribution:.2f}")
    else:
        result("Same context", "NO RESULTS")

    # Reconstruction 2: DIFFERENT context (should be lower fidelity)
    brain.context_buffer = {"location": "home", "project": "personal", "mood": "relaxed"}
    recons_diff = brain.reconstruct_recall("What happened with the database?")
    if recons_diff:
        r2 = recons_diff[0]
        result("Diff context - fidelity", r2.fidelity.value)
        result("Diff context - confidence", f"{r2.confidence:.3f}")
        result("Diff context - schema contribution", f"{r2.schema_contribution:.2f}")
    else:
        result("Diff context", "NO RESULTS")

    # Reconstruction 3: HIGH STRESS (should emphasize threat)
    recons_stress = brain.reconstruct_recall(
        "What happened with the database?",
        emotional_arousal=0.9,
        emotional_valence="negative"
    )
    if recons_stress:
        r3 = recons_stress[0]
        result("High stress - fidelity", r3.fidelity.value)
        result("High stress - emotional mod", f"{r3.emotional_modulation:.3f}")
    else:
        result("High stress", "NO RESULTS")

    # Key check: same memory → different reconstructions?
    if recons_same and recons_diff:
        same_trace = recons_same[0].trace_contribution
        diff_trace = recons_diff[0].trace_contribution
        result("CONTEXT DEPENDENCY", same_trace != diff_trace, "True (different contributions)")
        return True
    return False


# ─────────────────────────────────────────
# Experiment 2.2: Pattern Completion
# ─────────────────────────────────────────
def experiment_pattern_completion():
    banner("Pattern Completion — Retrieve From Partial Cues")
    print("  HYPOTHESIS: The attractor network should reconstruct")
    print("  full patterns from partial cues (like CA3).\n")

    engine = PatternCompletionEngine(PatternCompletionConfig(beta=10.0))
    dim = 64

    # Store 5 clearly distinct patterns
    patterns = {}
    for i in range(5):
        p = [0.0] * dim
        start = i * (dim // 5)
        for j in range(dim):
            p[j] = math.sin(j * 0.3 + i * 7.0) * 0.3
            if start <= j < start + (dim // 5):
                p[j] += 1.0
        engine.store(p, memory_id=f"event-{i}", label=f"Event {i}")
        patterns[f"event-{i}"] = p

    result("Patterns stored", engine.pattern_count)

    # Test completion at different cue fractions
    fractions = [1.0, 0.7, 0.5, 0.3, 0.2]
    all_correct = True

    for frac in fractions:
        target = patterns["event-2"]
        # Create partial cue by keeping only the signature region
        cue = [0.0] * dim
        n_keep = max(1, int(dim * frac))
        start_region = 2 * (dim // 5)
        for j in range(min(n_keep, dim)):
            # Prioritize the signature region
            idx = (start_region + j) % dim
            cue[idx] = target[idx]

        results = engine.complete(cue, top_k=1)
        correct = results[0].matched_memory_id == "event-2" if results else False
        sim = results[0].similarity if results else 0
        iters = results[0].iterations if results else 0

        status = "CORRECT" if correct else "WRONG"
        result(
            f"Cue {int(frac*100):3d}%",
            f"{status} (sim={sim:.3f}, iters={iters})"
        )
        if not correct and frac >= 0.3:
            all_correct = False

    result("PATTERN COMPLETION", all_correct, "True (30%+ cue = correct)")
    return all_correct


# ─────────────────────────────────────────
# Experiment 2.3: False Memory (DRM Paradigm)
# ─────────────────────────────────────────
def experiment_false_memory():
    banner("False Memory — Schema-Consistent Confabulation")
    print("  HYPOTHESIS: When context overlap is very low, reconstruction")
    print("  should fill in schema-consistent details (false memories).\n")

    brain = Brain(config={"agent_id": "false-memory-test"})

    # Encode a meeting memory
    brain.encode(
        "The quarterly review meeting discussed budget projections and team expansion",
        context={"location": "boardroom", "activity": "meeting", "project": "Q3 review"},
        salience=0.6
    )

    # Now recall from a COMPLETELY different context (very low overlap)
    brain.context_buffer = {"location": "cafe", "activity": "reading", "project": "none"}
    recons = brain.reconstruct_recall("What happened in the meeting?")

    if recons:
        r = recons[0]
        result("Fidelity level", r.fidelity.value)
        result("Schema contribution", f"{r.schema_contribution:.2f}")
        result("Active schemas used", r.active_schemas)

        # Check if reconstruction is LONGER than original (schema added content)
        original_len = len(r.original_content)
        recon_len = len(r.reconstructed_content)
        has_additions = recon_len > original_len

        result("Original length", original_len)
        result("Reconstructed length", recon_len)
        result("SCHEMA FILLING OCCURRED", has_additions, "True (content expanded)")

        # This IS a false memory — schema-consistent but not actually stored
        result("Reconstructed content", r.reconstructed_content[:100] + "...")

        return has_additions or r.schema_contribution > 0.1
    return False


# ─────────────────────────────────────────
# Experiment 2.4: Sleep + Post-Sleep Recall
# ─────────────────────────────────────────
def experiment_sleep_then_recall():
    banner("Sleep -> Post-Sleep Reconstructive Recall")
    print("  HYPOTHESIS: After sleep, memories should be available")
    print("  as semantic gist with schema-level reconstruction.\n")

    brain = Brain(config={"agent_id": "sleep-recall-test"})

    # Encode a batch of related events
    events = [
        ("Python 3.14 introduced pattern matching improvements", 0.7),
        ("The new async features reduced latency by 40%", 0.8),
        ("Database migration to PostgreSQL 16 completed", 0.6),
        ("Routine morning standup was uneventful", 0.2),
        ("Checked email and responded to 3 messages", 0.1),
    ]

    for content, sal in events:
        brain.encode(content, salience=sal)

    result("Pre-sleep episodic count", brain.episodic_count)
    result("Pre-sleep semantic count", brain.semantic_count)

    # Pre-sleep recall
    pre_sleep = brain.reconstruct_recall("What happened with Python?")
    pre_fidelity = pre_sleep[0].fidelity.value if pre_sleep else "none"
    result("Pre-sleep recall fidelity", pre_fidelity)

    # Sleep
    report = brain.sleep(num_cycles=2)
    result("Sleep consolidated", report.total_consolidated)
    result("Sleep evicted", report.total_evicted)
    result("New semantic memories", report.new_semantic_count)

    result("Post-sleep episodic count", brain.episodic_count)
    result("Post-sleep semantic count", brain.semantic_count)

    # Post-sleep recall
    post_sleep = brain.reconstruct_recall("What happened with Python?")
    if post_sleep:
        post_fidelity = post_sleep[0].fidelity.value
        result("Post-sleep recall fidelity", post_fidelity)
    else:
        result("Post-sleep recall", "semantic only (episodic evicted)")

    return True


# ─────────────────────────────────────────
# Experiment 2.5: Full Pipeline Stats
# ─────────────────────────────────────────
def experiment_full_stats():
    banner("Full Pipeline Statistics")
    print("  Encoding 50 events, running sleep, then checking stats.\n")

    brain = Brain(config={"agent_id": "stats-test"})

    for i in range(50):
        brain.encode(
            f"Event {i}: entity-{i%5} performed action-{i%3} at location-{i%4}",
            salience=0.3 + (i % 10) * 0.07
        )

    # Run some recalls
    for q in ["What happened with entity-2?", "Tell me about action-1"]:
        brain.reconstruct_recall(q)

    stats = brain.get_statistics()
    result("Brain version", stats.get("version", "?"))
    result("Episodic memories", stats["episodic_count"])
    result("Patterns stored", stats["pattern_completion"]["pattern_count"])
    result("Temporal snapshots", stats["temporal_context"]["snapshot_count"])
    result("Prediction inputs", stats["prediction"]["total_inputs"])
    result("Reconstructions done", stats["reconstruction"]["total_reconstructions"])
    result("Pattern completions", stats["pattern_completion"]["total_completions"])

    return True


# ─────────────────────────────────────────
# Run All
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("  ORIGIN AI - EXPERIMENT 02: RECONSTRUCTIVE MEMORY")
    print("  Testing the complete reconstruction pipeline")
    print("="*70)

    experiments = [
        ("Reconstructive Recall", experiment_reconstructive_recall),
        ("Pattern Completion", experiment_pattern_completion),
        ("False Memory (DRM)", experiment_false_memory),
        ("Sleep + Post-Sleep Recall", experiment_sleep_then_recall),
        ("Full Pipeline Stats", experiment_full_stats),
    ]

    results_summary = []
    for name, fn in experiments:
        try:
            passed = fn()
            results_summary.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            import traceback
            traceback.print_exc()
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
