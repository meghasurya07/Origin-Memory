"""
Origin Brain — Multi-Agent Shared Memory Example

Shows how multiple agents can share knowledge through a centralized pool
with scoped access control and conflict resolution.
"""
from origin_brain import (
    Brain, BrainConfig, SharedMemoryPool, MemoryScope, ConflictStrategy,
)

# === Setup: Create agents and shared pool ===
researcher = Brain(config=BrainConfig(agent_id="researcher"))
analyst = Brain(config=BrainConfig(agent_id="analyst"))
writer = Brain(config=BrainConfig(agent_id="writer"))

pool = SharedMemoryPool()

# Create teams
pool.join_group("researcher", "science_team")
pool.join_group("analyst", "science_team")
pool.join_group("analyst", "content_team")  # Analyst is in both teams
pool.join_group("writer", "content_team")

print("=== Multi-Agent Shared Memory ===\n")

# === Researcher discovers facts ===
print("1. Researcher discovers facts:")
facts = [
    ("protein_x_function", "kinase involved in apoptosis"),
    ("clinical_trial_result", "Phase 2 showed 40% tumor reduction"),
    ("mechanism", "inhibits mTOR pathway via allosteric binding"),
]
for key, value in facts:
    pool.publish("researcher", key, value,
                scope=MemoryScope.GROUP, group_id="science_team")
    researcher.encode(f"Discovery: {key} = {value}")
    print(f"  Published: {key} → {value}")

# === Analyst can see science team findings ===
print("\n2. Analyst queries science team knowledge:")
results = pool.query("analyst", include_global=False)
for r in results:
    print(f"  [{r.scope.value}] {r.key} = {r.value} (by {r.author_agent_id})")
    # Analyst also stores in their own brain for personal recall
    analyst.encode(f"From {r.author_agent_id}: {r.key} = {r.value}")

# === Analyst publishes summary to content team ===
print("\n3. Analyst publishes summary to content team:")
pool.publish("analyst", "article_draft",
            "Protein X inhibits mTOR, Phase 2 shows 40% reduction",
            scope=MemoryScope.GROUP, group_id="content_team")
print("  Published article_draft to content_team")

# === Writer queries — sees content team, NOT science team ===
print("\n4. Writer queries (should see content_team only):")
writer_results = pool.query("writer", include_global=False)
for r in writer_results:
    print(f"  [{r.scope.value}] {r.key} = {r.value}")

# Can writer see science team secrets?
science_visible = any(r.key == "mechanism" for r in writer_results)
print(f"\n  Writer can see science secrets? {'YES ⚠️' if science_visible else 'NO ✅ (correct)'}")

# === Broadcast important discovery ===
print("\n5. Researcher broadcasts breakthrough:")
pool.broadcast("researcher", "breakthrough",
              "Protein X is also effective against Alzheimer's plaques",
              confidence=0.95)
print("  Broadcast to all groups + global")

# All agents should see it
for agent_id in ["researcher", "analyst", "writer"]:
    r = pool.query(agent_id, key="breakthrough")
    print(f"  {agent_id} sees breakthrough: {'YES ✅' if r else 'NO ❌'}")

# === Stats ===
print("\n6. Pool statistics:")
stats = pool.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")
