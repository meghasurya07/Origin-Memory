# 12 — API Reference & Usage Guide

## Overview

This document is the developer guide for the **Origin Brain SDK** — how to install, configure, and use the Human-Like Brain/Memory system for AI agents.

---

## 1. Installation

```bash
# From source (development)
git clone https://github.com/originai/origin-brain.git
cd origin-brain
pip install -e .

# With test dependencies
pip install -e ".[test]"

# Future: from PyPI
pip install origin-brain
```

---

## 2. Quick Start

```python
from origin_brain import Brain

# Create a brain for your agent
brain = Brain(config={
    "agent_id": "my-agent-001",
    "decay_model": "ebbinghaus_weighted",
    "working_memory_tokens": 128000,
})

# Encode experiences (after each interaction)
brain.encode(
    content="User prefers dark mode and uses Python primarily",
    context={"user_id": "alice", "session_id": "s-42"},
    salience=0.8  # This is important
)

brain.encode(
    content="User asked about flight prices to Tokyo for next week",
    context={"user_id": "alice", "session_id": "s-42"}
)

brain.encode(
    content="User mentioned they have a meeting on Friday at 3pm",
    context={"user_id": "alice", "session_id": "s-43"}
)

# Recall relevant memories
results = brain.recall(
    query="What does Alice prefer?",
    top_k=3
)

for result in results:
    print(f"Memory: {result.memory.content}")
    print(f"Relevance: {result.relevance_score:.2f}")
    print(f"Tier: {result.tier}")
    print()

# Store explicit facts
brain.store_fact(
    key="user_timezone",
    value="Asia/Kolkata (IST, UTC+5:30)",
    confidence=0.95,
    category="user_preference"
)

# Run consolidation (converts episodes → semantic knowledge)
result = brain.consolidate()
print(f"Consolidated: {result.consolidated_count} episodes")
print(f"New semantic memories: {result.new_semantic_count}")
print(f"Evicted (forgotten): {result.evicted_count}")

# Check brain statistics
stats = brain.get_statistics()
print(f"Episodic memories: {stats['episodic_count']}")
print(f"Semantic memories: {stats['semantic_count']}")
print(f"Avg retrievability: {stats['avg_retrievability']:.2f}")
```

---

## 3. Core Concepts

### 3.1 Memory Types

```python
from origin_brain.models import MemoryType, MemoryTier

# What happened (experiences, conversations)
MemoryType.EPISODIC

# What is known (facts, rules, preferences)  
MemoryType.SEMANTIC
MemoryType.FACT
MemoryType.PREFERENCE
MemoryType.RELATIONSHIP

# How to do things (procedures, workflows)
MemoryType.PROCEDURAL

# What's in the conversation right now (volatile)
# Managed automatically via context buffer
```

### 3.2 Memory Lifecycle

```
1. EXPERIENCE → 2. ENCODE → 3. STORE → 4. DECAY → 5. CONSOLIDATE → 6. ARCHIVE
                    │            │           │            │               │
               Novelty &    Episodic     Forgetting   Episodic →      DNA/Long-term
               Salience     Store        Curve        Semantic         Storage
               Filtering                              Transform
```

### 3.3 The Decay Model

Every episodic memory has a **retrievability** score that decays over time:

```python
# Ebbinghaus exponential decay
R(t) = e^(-t/S)

# Where:
#   R = retrievability (0.0 = forgotten, 1.0 = perfect recall)
#   t = time since last access (seconds)
#   S = stability (grows with each retrieval)
```

Each time a memory is successfully retrieved, its **stability increases**:

```python
# Spaced repetition stability growth
S_new = S_old * (1 + α * e^(w * S_old))

# Where:
#   α = 0.3 (learning efficiency)
#   w = -0.01 (decay modifier)
```

Memories with retrievability below the **survival threshold** (default: 0.05) are evicted during consolidation.

---

## 4. Configuration

```python
from origin_brain.models import BrainConfig

config = BrainConfig(
    # Required
    agent_id="my-agent-001",
    
    # Working memory
    working_memory_tokens=128000,       # Context window size
    
    # Episodic memory
    episodic_capacity=1_000_000,        # Max episodic memories
    
    # Decay model
    decay_model="ebbinghaus_weighted",  # Options: ebbinghaus, ebbinghaus_weighted, power_law
    initial_stability=1.0,              # Starting stability for new memories (seconds)
    survival_threshold=0.05,            # Below this = evicted
    
    # Encoding
    salience_threshold=0.3,             # Min salience to encode
    novelty_threshold=0.3,              # Min novelty to encode (1 - max_similarity)
    
    # Spaced repetition
    spaced_repetition_alpha=0.3,        # Learning efficiency
    spaced_repetition_w=-0.01,          # Decay modifier
    
    # Consolidation
    consolidation_interval_seconds=3600,  # Run every hour
    consolidation_min_age_hours=24,       # Min age to consolidate
    
    # Archival
    archival_min_age_days=90,           # Min age to archive to DNA
    archival_backend="silicon",         # Options: silicon, dna
    
    # Embeddings
    embedding_model="text-embedding-3-small",
    embedding_dimension=1536,
)
```

---

## 5. Advanced Usage

### 5.1 Custom Embedding Function

```python
import openai

def embed_with_openai(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

brain = Brain(
    config={"agent_id": "my-agent"},
    embedding_fn=embed_with_openai
)
```

### 5.2 Manual Consolidation Control

```python
# Run a single consolidation cycle
result = brain.consolidate()

# Access consolidation history
stats = brain.consolidation.get_statistics()
print(f"Total cycles: {stats['total_consolidation_cycles']}")
print(f"Last run: {stats['last_consolidation']}")
```

### 5.3 State Persistence (Export/Import)

```python
import json

# Export entire brain state
state = brain.export_state()

# Save to disk
with open("brain_state.json", "w") as f:
    json.dump(state, f, default=str)

# Later: restore brain from saved state
with open("brain_state.json", "r") as f:
    state = json.load(f)

new_brain = Brain(config={"agent_id": "my-agent"})
new_brain.import_state(state)
```

### 5.4 Procedural Memory

```python
# Store a learned procedure
brain.store_procedure(
    name="code_review",
    description="Steps to perform a thorough code review",
    steps=[
        "Check for security vulnerabilities",
        "Verify test coverage",
        "Review naming conventions",
        "Check error handling",
        "Verify documentation"
    ],
    triggers=["review this code", "code review", "PR review"]
)

# Retrieve procedures
results = brain.recall(
    query="How should I review this PR?",
    memory_types=[MemoryType.PROCEDURAL]
)
```

### 5.5 Direct Hippocampus Access

```python
# Access the hippocampal engine directly for advanced use
hippo = brain.hippocampus

# Get detailed statistics
stats = hippo.get_statistics()

# Select memories ready for consolidation
ready = hippo.select_for_consolidation(
    min_age_hours=48,
    min_access_count=3,
    min_salience=0.5
)
```

---

## 6. Integration Examples

### 6.1 With LangChain (Future v0.5)

```python
from langchain.chat_models import ChatOpenAI
from origin_brain.integrations.langchain import OriginBrainMemory

llm = ChatOpenAI(model="gpt-4o")
memory = OriginBrainMemory(agent_id="langchain-agent-001")

# Memory automatically encodes and retrieves
chain = ConversationChain(llm=llm, memory=memory)
chain.predict(input="My name is Alice and I work at Origin AI")
chain.predict(input="What company do I work at?")  # Retrieves from memory
```

### 6.2 As an Agent Tool

```python
# The brain can be exposed as a tool for function-calling agents
tools = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store an important fact or experience in long-term memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember"},
                    "importance": {"type": "number", "description": "0.0-1.0 importance score"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search long-term memory for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"]
            }
        }
    }
]
```

---

## 7. Architecture Deep Dive

### 7.1 How Encoding Works

```
Input text
    │
    ▼
┌──────────────┐
│ Compute       │ → Generate embedding vector
│ Embedding     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Novelty       │ → Compare with recent memories (cosine similarity)
│ Detection     │ → If too similar → REJECT (not novel enough)
│ (CA1)         │
└──────┬───────┘
       │ Novel enough
       ▼
┌──────────────┐
│ Salience      │ → Score importance (keywords, length, user signals)
│ Scoring       │ → If below threshold → REJECT (not important enough)
│ (Amygdala)    │
└──────┬───────┘
       │ Important enough
       ▼
┌──────────────┐
│ Pattern       │ → Apply k-winners-take-all sparsification
│ Separation    │ → Creates distinct representation (prevents interference)
│ (Dentate      │
│  Gyrus)       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Store in      │ → Save to episodic store with timestamp, context, salience
│ Episodic      │
│ Memory        │
└──────────────┘
```

### 7.2 How Retrieval Works

```
Query text
    │
    ▼
┌──────────────┐
│ Compute       │ → Generate query embedding
│ Query         │
│ Embedding     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Similarity    │ → Cosine similarity against all stored embeddings
│ Search        │ → Get candidate set (3x top_k)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Multi-signal  │ → Score = 0.6 * similarity + 0.2 * recency + 0.2 * salience
│ Ranking       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Reinforce     │ → Call on_retrieval() → stability increases (spaced repetition)
│ Retrieved     │ → Access count increments
│ Memories      │
└──────┬───────┘
       │
       ▼
Return top_k MemoryResult objects
```

---

## 8. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_brain.py -v

# Run with coverage
pytest tests/ --cov=origin_brain --cov-report=html
```

---

*This is the developer's guide to building with Origin Brain. For neuroscience background, see [01_neuroscience_foundations.md](01_neuroscience_foundations.md). For architecture details, see [08_architecture_proposal.md](08_architecture_proposal.md).*
