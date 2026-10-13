"""
Origin Brain — Basic Usage Example

Shows the core encode → recall → sleep cycle.
"""
from origin_brain import Brain, BrainConfig

# 1. Create a brain
brain = Brain(config=BrainConfig(agent_id="demo_agent"))

# 2. Encode memories (the prediction gate automatically filters redundant info)
memories = [
    "User's name is Alice and she works at Acme Corp",
    "Alice prefers Python over JavaScript for backend work",
    "Alice is building a RAG pipeline for healthcare compliance documents",
    "The project deadline is December 15th, 2026",
    "Alice had a meeting with Bob about the Q3 budget — approved for $500K",
    "Alice mentioned she's allergic to peanuts",
    "The team uses PostgreSQL with pgvector for embeddings",
]

print("=== Encoding Memories ===")
for memory in memories:
    result = brain.encode(memory)
    action = result.action
    salience = result.memory.salience if result.memory else 0
    print(f"  [{action:>12}] (salience={salience:.2f}) {memory[:60]}")

# 3. Recall
print("\n=== Recall: 'What is Alice working on?' ===")
result = brain.recall("What is Alice working on?", top_k=3)
for r in result.results:
    print(f"  [{r.relevance_score:.3f}] {r.memory.content}")

print("\n=== Recall: 'project deadline' ===")
result = brain.recall("project deadline", top_k=2)
for r in result.results:
    print(f"  [{r.relevance_score:.3f}] {r.memory.content}")

# 4. Reconstructive recall — context-dependent output
print("\n=== Reconstructive Recall (high stress) ===")
reconstructed = brain.reconstruct_recall(
    "What happened in the meeting?",
    emotional_arousal=0.9,
    emotional_valence="negative",
    top_k=2,
)
for r in reconstructed:
    print(f"  [fidelity={r.fidelity.value}] {r.reconstructed_content[:80]}")

# 5. Sleep consolidation
print("\n=== Sleep Consolidation ===")
report = brain.sleep()
print(f"  Consolidated: {report.total_consolidated}")
print(f"  Pruned: {report.total_evicted}")
print(f"  New semantic memories: {report.new_semantic_count}")

# 6. After sleep — query semantic network (learned associations)
print("\n=== Semantic Network (after sleep) ===")
associations = brain.query_semantic_network("alice")
for concept, strength in associations[:5]:
    print(f"  alice → {concept} (strength={strength:.3f})")

# 7. Statistics
print("\n=== Brain Statistics ===")
stats = brain.get_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")
