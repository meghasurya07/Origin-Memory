"""
Origin Brain — Persistent Memory with SQLite Example

Shows how to persist memories across program restarts.
"""
from origin_brain import Brain, BrainConfig
import os
import tempfile

DB_PATH = os.path.join(tempfile.gettempdir(), "origin_brain_demo.db")

print("=== Persistent Memory Demo ===\n")

# === Session 1: Encode memories ===
print("Session 1: Encoding memories...")
brain = Brain(config=BrainConfig(
    agent_id="persistent_agent",
    storage_path=DB_PATH,
))

brain.encode("User prefers dark mode and Vim keybindings")
brain.encode("User's timezone is IST (UTC+5:30)")
brain.encode("User is building an AI agent for customer support")

count = brain.save()
print(f"  Saved {count} memories to {DB_PATH}")
del brain  # Destroy the brain object

# === Session 2: Reload and recall ===
print("\nSession 2: Reloading from disk...")
brain2 = Brain(config=BrainConfig(
    agent_id="persistent_agent",
    storage_path=DB_PATH,
))

result = brain2.recall("user preferences", top_k=3)
print(f"  Recalled {len(result.results)} memories from disk:")
for r in result.results:
    print(f"    [{r.relevance_score:.3f}] {r.memory.content}")

# Cleanup
del brain2
try:
    os.remove(DB_PATH)
    print(f"\n  Cleaned up {DB_PATH}")
except OSError:
    print(f"\n  Note: {DB_PATH} will be cleaned up on next OS restart")
