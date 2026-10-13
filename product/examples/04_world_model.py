"""
Origin Brain — World Model & Prediction Example

Shows how the brain learns patterns from experience and predicts futures.
"""
from origin_brain import Brain, BrainConfig

brain = Brain(config=BrainConfig(agent_id="prediction_agent"))

print("=== World Model & Prediction ===\n")

# === Encode a series of episodes (daily patterns) ===
daily_patterns = [
    "User logged in and checked notifications",
    "User reviewed pending pull requests",
    "User started coding on feature branch",
    "User pushed code and created pull request",
    "User ran CI tests and merged to main",
] * 3  # Repeat 3 times to build patterns

print("1. Teaching daily workflow patterns (3 repetitions):")
for episode in daily_patterns:
    brain.encode(episode)
print(f"  Encoded {len(daily_patterns)} episodes")

# Run sleep to consolidate patterns into world model + neocortex
brain.sleep()
print("  Sleep consolidation complete")

# === Query the world model for predictions ===
print("\n2. World Model Predictions:")
wm_stats = brain.world_model.get_statistics()
print(f"  States learned: {wm_stats['total_states']}")
print(f"  Transitions observed: {wm_stats['total_transitions']}")
print(f"  Sequences replayed: {wm_stats['total_sequences']}")

# Check what the world model learned
print("\n3. Most visited states:")
for state, count in wm_stats["most_visited"]:
    print(f"  {state}: {count} visits")

# === Direct world model usage for custom patterns ===
print("\n4. Teaching custom state machine:")
brain.world_model.observe_sequence(["login", "dashboard", "search", "result", "action"])
brain.world_model.observe_sequence(["login", "dashboard", "search", "result", "save"])
brain.world_model.observe_sequence(["login", "dashboard", "settings", "logout"])
brain.world_model.observe_sequence(["login", "dashboard", "search", "result", "action"])

predictions = brain.predict_next("dashboard")
print("  After 'dashboard', user will likely:")
for state, prob in predictions:
    print(f"    → {state} ({prob:.0%})")

# Simulate full sequence
print("\n5. Simulated future from 'login':")
future = brain.simulate_future("login", steps=5)
print(f"  {' → '.join(future)}")

# Surprise computation
surprise_expected = brain.compute_surprise("login", "dashboard")
surprise_unexpected = brain.compute_surprise("login", "logout")
print(f"\n6. Surprise levels:")
print(f"  login → dashboard: {surprise_expected:.2f} (expected)")
print(f"  login → logout:    {surprise_unexpected:.2f} (unexpected)")
