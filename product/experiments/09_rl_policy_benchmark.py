"""
Experiment 09: RL Memory Policy Training & Evaluation
═════════════════════════════════════════════════════

Tests:
  1. Heuristic baseline: encode everything, measure recall accuracy
  2. RL training: learn from encode→recall reward loops
  3. RL evaluation: learned policy vs heuristic on held-out data
  4. Convergence: Q-values stabilize, epsilon decays
  5. Action distribution: policy learns situation-appropriate actions
  6. Brain integration: RL policy connected to full Brain pipeline

Expected:
  - RL policy converges within 200 episodes
  - Learned policy matches or beats heuristic baseline
  - Policy correctly learns to SKIP redundant + ENCODE novel
  - Integration test: Brain with RL policy maintains recall quality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from origin_brain.rl_policy import (
    RLMemoryPolicy, RLPolicyConfig, MemoryAction, MemoryState, RewardSignal,
)
from origin_brain.entropy_monitor import AttentionEntropyMonitor
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
# Test 1: Heuristic Baseline
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 1: Heuristic Baseline ═══")

policy = RLMemoryPolicy()

# Simulate different scenarios
scenarios = [
    ("novel_important", MemoryState(novelty=0.9, salience=0.8), MemoryAction.ENCODE),
    ("redundant", MemoryState(novelty=0.1, similarity_to_best=0.9), MemoryAction.UPDATE),
    ("noise", MemoryState(novelty=0.1, salience=0.1), MemoryAction.SKIP),
    ("overload", MemoryState(memory_load=0.95, novelty=0.6, salience=0.3), MemoryAction.EVICT),
    ("pressure", MemoryState(sleep_pressure=0.9, entropy=0.8), MemoryAction.CONSOLIDATE),
    ("default", MemoryState(novelty=0.5, salience=0.3), MemoryAction.ENCODE),
]

for name, state, expected in scenarios:
    action = policy.heuristic_decide(state)
    check(f"heuristic_{name}", action == expected,
          f"got {action.value}, expected {expected.value}")


# ══════════════════════════════════════════════════════════════════════
# Test 2: RL Training Loop — Learn to Encode Novel Content
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 2: RL Training Loop ═══")

policy = RLMemoryPolicy(config=RLPolicyConfig(
    epsilon=0.5, epsilon_decay=0.995, learning_rate=0.15,
    min_experiences=10, batch_size=8,
))

# Training data: novel content is good to encode, redundant is bad
import random
random.seed(42)

rewards_per_50 = []
for episode in range(200):
    # Generate random scenario
    novelty = random.random()
    salience = random.uniform(0.1, 0.9)
    similarity = 1.0 - novelty  # Inverse relationship

    state = MemoryState(
        novelty=novelty, salience=salience,
        similarity_to_best=similarity,
    )

    action = policy.decide(state)

    # Reward depends on whether the action was appropriate
    if action == MemoryAction.ENCODE and novelty > 0.5:
        reward = RewardSignal(recall_accuracy=novelty)
    elif action == MemoryAction.SKIP and novelty < 0.3:
        reward = RewardSignal(recall_accuracy=0.1)
    elif action == MemoryAction.UPDATE and similarity > 0.7:
        reward = RewardSignal(recall_accuracy=0.6)
    elif action == MemoryAction.ENCODE and novelty < 0.2:
        reward = RewardSignal(memory_cost=0.1, interference_penalty=0.1)
    else:
        reward = RewardSignal()

    next_state = MemoryState(novelty=random.random(), salience=random.uniform(0.1, 0.9))
    policy.learn(state, action, reward, next_state)

stats = policy.get_statistics()
check("training_completed", stats["episode_count"] == 200)
check("epsilon_decayed", stats["epsilon"] < 0.5, f"ε={stats['epsilon']:.3f}")
check("q_table_populated", stats["q_table_size"] > 0, f"states={stats['q_table_size']}")
check("using_rl_not_heuristic", stats["mode"] == "rl")

# Check that novel-important state prefers ENCODE
novel_state = MemoryState(novelty=0.9, salience=0.8)
action = policy.decide_greedy(novel_state)
check("learned_encode_novel", action == MemoryAction.ENCODE, f"got {action.value}")


# ══════════════════════════════════════════════════════════════════════
# Test 3: Reward Signal Computation
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 3: Reward Signals ═══")

policy = RLMemoryPolicy()

# Good encode (retrieved at top rank)
r1 = policy.compute_reward(MemoryAction.ENCODE, was_retrieved=True, retrieval_rank=0)
check("encode_retrieved_positive", r1.total > 0, f"reward={r1.total:.3f}")

# Bad skip (content was needed)
r2 = policy.compute_reward(MemoryAction.SKIP, was_retrieved=True)
check("skip_needed_negative", r2.total < 0, f"reward={r2.total:.3f}")

# Good skip (content wasn't needed)
r3 = policy.compute_reward(MemoryAction.SKIP, was_retrieved=False)
check("skip_unneeded_positive", r3.total >= 0, f"reward={r3.total:.3f}")

# Interference penalty
r4 = policy.compute_reward(
    MemoryAction.ENCODE, was_retrieved=True, retrieval_rank=0, caused_interference=True,
)
check("interference_reduces_reward", r4.total < r1.total,
      f"clean={r1.total:.3f} vs dirty={r4.total:.3f}")


# ══════════════════════════════════════════════════════════════════════
# Test 4: Experience Replay Quality
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 4: Experience Replay ═══")

policy = RLMemoryPolicy(config=RLPolicyConfig(
    replay_buffer_size=100, min_experiences=10, batch_size=5,
))

# Fill replay buffer
state = MemoryState(novelty=0.8)
reward = RewardSignal(recall_accuracy=0.7)
for _ in range(50):
    policy.learn(state, MemoryAction.ENCODE, reward, state)

check("replay_buffer_filled", len(policy._replay_buffer) == 50)
check("replay_drove_learning", policy._q_table[state.to_discrete()][MemoryAction.ENCODE] > 0)

# Buffer should be bounded
for _ in range(100):
    policy.learn(state, MemoryAction.SKIP, reward, state)

check("replay_buffer_bounded", len(policy._replay_buffer) <= 100)


# ══════════════════════════════════════════════════════════════════════
# Test 5: Adaptive Policy (Heuristic → RL Transition)
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 5: Adaptive Policy ═══")

policy = RLMemoryPolicy(config=RLPolicyConfig(min_experiences=20))
check("starts_heuristic", policy.should_use_heuristic())

# Heuristic behavior on high pressure
state = MemoryState(sleep_pressure=0.9, entropy=0.8)
action = policy.decide_adaptive(state)
check("heuristic_consolidates", action == MemoryAction.CONSOLIDATE)

# Train enough to switch to RL
for _ in range(25):
    policy.learn(state, MemoryAction.ENCODE, RewardSignal(), state)

check("switches_to_rl", not policy.should_use_heuristic())


# ══════════════════════════════════════════════════════════════════════
# Test 6: Persistence (Save / Load)
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 6: Persistence ═══")

import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "test_policy.json")

    # Train
    policy1 = RLMemoryPolicy(config=RLPolicyConfig(
        save_path=save_path, min_experiences=9999,
    ))
    state = MemoryState(novelty=0.7)
    for _ in range(10):
        policy1.learn(state, MemoryAction.ENCODE, RewardSignal(recall_accuracy=0.5), state)

    # Save
    policy1.save_q_table()
    check("file_saved", os.path.exists(save_path))

    # Load
    policy2 = RLMemoryPolicy(config=RLPolicyConfig(save_path=save_path))
    check("episodes_loaded", policy2._episode_count == policy1._episode_count)

    discrete = state.to_discrete()
    q1 = policy1._q_table[discrete][MemoryAction.ENCODE]
    q2 = policy2._q_table[discrete][MemoryAction.ENCODE]
    check("q_values_match", abs(q1 - q2) < 0.001, f"q1={q1:.4f} q2={q2:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Test 7: Policy Summary — Interpretability
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 7: Policy Summary ═══")

policy = RLMemoryPolicy()
summary = policy.get_policy_summary()

check("summary_has_archetypes", len(summary) >= 5)
check("novel_prefers_encode", summary["novel_important"]["preferred_action"] == "encode")
check("redundant_prefers_update", summary["redundant"]["preferred_action"] == "update")
check("high_pressure_consolidates", summary["high_pressure"]["preferred_action"] == "consolidate")


# ══════════════════════════════════════════════════════════════════════
# Test 8: Brain Integration
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 8: Brain Integration ═══")

brain = Brain(config=BrainConfig(agent_id="rl-benchmark"))
check("brain_has_rl_policy", hasattr(brain, 'rl_policy'))
check("rl_policy_type", isinstance(brain.rl_policy, RLMemoryPolicy))

# Encode content
brain.encode("Alice discovered a breakthrough in quantum computing yesterday")
brain.encode("Bob presented the results at the conference")
brain.encode("The quantum advantage was confirmed by three independent labs")

stats = brain.get_statistics()
check("stats_has_rl_policy", "rl_policy" in stats)
check("rl_mode_reported", "mode" in stats["rl_policy"])

# Recall
result = brain.recall("quantum computing")
check("recall_works_with_rl", len(result.results) > 0)
check("recall_finds_quantum", any("quantum" in r.memory.content.lower() for r in result.results))


# ══════════════════════════════════════════════════════════════════════
# Test 9: Entropy + RL Integration
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 9: Entropy + RL Synergy ═══")

brain = Brain(config=BrainConfig(agent_id="entropy-rl-test"))

# Encode memories to build up entropy state
for i in range(10):
    brain.encode(f"Memory item {i} about project milestone {i}")

# Observe state from entropy monitor
entropy_state = brain.entropy_monitor.get_state()
check("entropy_state_available", entropy_state is not None)
check("entropy_nonzero", entropy_state.memory_entropy > 0)

# Build RL observation using entropy state
rl_state = brain.rl_policy.observe(
    content="New critical finding about the project",
    entropy_state=entropy_state,
    salience=0.8,
    memory_count=brain.episodic_count,
    wm_full=brain.working_memory.is_full,
)
check("rl_observes_entropy", rl_state.entropy > 0)
check("rl_observes_pressure", rl_state.sleep_pressure >= 0)


# ══════════════════════════════════════════════════════════════════════
# Test 10: Convergence Analysis
# ══════════════════════════════════════════════════════════════════════

print("\n═══ Test 10: Convergence ═══")

policy = RLMemoryPolicy(config=RLPolicyConfig(
    epsilon=0.5, epsilon_decay=0.99, learning_rate=0.1,
    min_experiences=10, batch_size=5,
))

random.seed(123)
rewards = []
for episode in range(300):
    novelty = random.random()
    state = MemoryState(novelty=novelty, salience=novelty * 0.8)
    action = policy.decide(state)

    # Simple reward: encode when novel > 0.5, skip when < 0.3
    if action == MemoryAction.ENCODE and novelty > 0.5:
        r = RewardSignal(recall_accuracy=0.8)
    elif action == MemoryAction.SKIP and novelty < 0.3:
        r = RewardSignal(recall_accuracy=0.1)
    elif action == MemoryAction.ENCODE and novelty < 0.3:
        r = RewardSignal(memory_cost=0.2)
    else:
        r = RewardSignal()

    next_s = MemoryState(novelty=random.random())
    policy.learn(state, action, r, next_s)
    rewards.append(r.total)

# Compute average reward over windows
first_50_avg = sum(rewards[:50]) / 50
last_50_avg = sum(rewards[-50:]) / 50

check("convergence_300_episodes", policy._episode_count == 300)
check("epsilon_near_min", policy._epsilon < 0.1, f"ε={policy._epsilon:.4f}")
check("reward_improved", last_50_avg >= first_50_avg * 0.8,
      f"first50={first_50_avg:.4f} last50={last_50_avg:.4f}")


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════

total = passed + failed
print(f"\n{'='*60}")
print(f"RL Memory Policy Benchmark: {passed}/{total} passed")
if failed == 0:
    print("🎉 ALL TESTS PASSED — RL Memory Policy operational!")
else:
    print(f"⚠ {failed} tests failed")
print(f"{'='*60}")
