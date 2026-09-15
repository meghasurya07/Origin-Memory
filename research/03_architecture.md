# Origin Brain — Architecture & Design

## 1. The Problem (why AI memory is broken)

The single greatest architectural failure in modern AI is the **absence of persistent, structured memory**. In 2026, despite advances in reasoning and multimodal understanding, AI agents still cannot reliably maintain a coherent identity or remember past events over long horizons.

**The core problem: Context Window ≠ Memory.** 
The context window is working memory—volatile, expensive, and unstructured. Treating the context window as long-term storage is fundamentally flawed:
1. **Cost scales linearly or worse:** Processing 1M tokens is prohibitively expensive per inference.
2. **"Lost in the Middle" phenomenon:** Models struggle to attend to middle context, losing important details.
3. **No persistence or temporal awareness:** Context vanishes after the session ends.
4. **No organization or forgetting:** Information isn't structured or indexed, and there's no intelligent eviction (everything has equal weight).

Current solutions fall short:
- **RAG:** Built for document retrieval, not episodic agent memory. Semantic similarity ≠ relevance. Lacks temporal and causal links, forgetting, and consolidation.
- **Sliding Window / Summarization:** Lossy and degrades over time (summary of a summary).
- **Key-Value Stores:** Brittle, atomistic, and misses nuanced relationships.
- **Parametric Memory / Continual Learning:** Updating weights is expensive and prone to catastrophic forgetting unless strictly regularized.

Origin AI addresses this through a biologically grounded, 6-pillar cognitive architecture that models working, episodic, semantic, and procedural memory—replicating prediction error, sleep consolidation, and reconstructive recall.

## 2. System Architecture

The Origin Brain architecture mirrors the human brain's cognitive hierarchy, transitioning from static databases to a dynamic "Agentic Memory OS." It implements an event-driven loop (Observe → Reason → Act → Store) and a hybrid substrate (Silicon for hot/warm data, DNA for cold archival).

### The 6-Pillar Model
1. **Prediction Error:** Only encode the unexpected ($\text{PE} = \|\epsilon\| \cdot \pi$).
2. **Temporal Context:** Bind memories to time ($c_t = \rho c_{t-1} + \beta f(x_t)$).
3. **Forgetting:** Compress via information bottleneck.
4. **Sleep Consolidation:** Offline replay + compression (SWS + REM + Pruning).
5. **Reconstruction:** Generative recall, not rigid retrieval ($p(\hat{x} | z, c, s, e)$).
6. **Pattern Completion:** Attractor dynamics for partial cues.

### Module Diagram
```
┌─────────────────────────────────────────────────────────────────────┐
│                    ORIGIN AI ARTIFICIAL BRAIN                       │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐     │
│  │   SALIENCE  │    │   MEMORY     │    │    CONSOLIDATION    │     │
│  │   SCORER    │    │   ROUTER     │    │    DAEMON           │     │
│  │  (Amygdala) │    │  (Thalamus)  │    │   (Sleep/Replay)    │     │
│  └──────┬──────┘    └──────┬───────┘    └─────────┬───────────┘     │
│         │                  │                      │                 │
│  ┌──────▼──────────────────▼──────────────────────▼───────────────┐ │
│  │                    CONTEXT MANAGER (PFC)                       │ │
│  └─────────────────────────┬──────────────────────────────────────┘ │
│                            │                                        │
│  ┌─────────────────────────▼──────────────────────────────────────┐ │
│  │                   HIPPOCAMPAL ENGINE                           │ │
│  │  [ CA1 (Novelty) | CA3 (Completion) | DG (Separation) ]        │ │
│  │  EPISODIC STORE (Vector DB + Temporal Index)                   │ │
│  └─────────────────────────┬──────────────────────────────────────┘ │
│                            │ Consolidation                          │
│  ┌─────────────────────────▼──────────────────────────────────────┐ │
│  │                   NEOCORTICAL STORE                            │ │
│  │  SEMANTIC KNOWLEDGE GRAPH (Neo4j) + PROCEDURAL REGISTRY        │ │
│  └─────────────────────────┬──────────────────────────────────────┘ │
│                            │ Archival                               │
│  ┌─────────────────────────▼──────────────────────────────────────┐ │
│  │                   DNA ARCHIVE MODULE 🧬                        │ │
│  │  BIOCOMPUTE CHIP (Cold Storage) + SILICON INDEX                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Pipelines

**Encode Pipeline**
1. Compute Input Embedding
2. CA1 Novelty Detection (reject if too similar to recent memory)
3. Amygdala Salience Scoring (reject if unimportant)
4. Prediction Error Gating
5. DG Pattern Separation (k-winners-take-all sparsification to prevent interference)
6. Bind Temporal Context
7. Store in Episodic Memory with metadata

**Recall Pipeline**
1. Compute Query Embedding
2. CA3 Pattern Completion (multi-head similarity search + temporal weighting)
3. Filter via Salience & Temporal Context
4. Reconstruct / Confabulate (schema-filling generative recall)
5. Reinforce Retrieved Memories (update stability / retrievability via spaced repetition)
6. Inject Top-K into Context Manager

## 3. Module Reference

| Module / Structure | Brain Analogue | Purpose | API / Core Methods |
|:---|:---|:---|:---|
| **ContextManager** | Prefrontal Cortex (PFC) | Manages volatile working memory, executive function. | `compose_context(query)` |
| **HippocampalEngine** | Hippocampus (CA1, CA3, DG) | Fast episodic encoding, pattern completion, novelty detection, pattern separation. | `encode(exp)`, `retrieve(cue)`, `pattern_separate()` |
| **ConsolidationDaemon** | Sleep / Replay | Converts episodic events into semantic facts during downtime, prunes memories. | `run_consolidation_cycle()`, `archive_to_dna()` |
| **SalienceScorer** | Amygdala | Scores importance/emotion to weight memory encoding and retrieval. | `score(experience)` |
| **MemoryRouter** | Thalamus | Routes queries and experiences to the correct storage tiers. | `route(query)` |
| **DecayEngine** | Ebbinghaus / RIF | Forgetting curves and retrieval-induced forgetting. | `compute_retrievability(memory)`, `update_stability_on_retrieval()` |
| **Schema/Reconstruction**| vmPFC | Schema-guided, context-dependent generative recall. | `reconstruct_memory()` |
| **DNA Archive Module** | Molecular Storage | Unbounded archival storage for aged, highly stable semantic knowledge. | `write()`, `read()` |

## 4. API Reference

### Origin Brain Python SDK

```python
from origin_brain import Brain
from origin_brain.models import BrainConfig, MemoryType

# 1. Initialization
brain = Brain(
    config=BrainConfig(
        agent_id="agent-007",
        working_memory_tokens=128000,
        episodic_capacity=1_000_000,
        decay_model="ebbinghaus_weighted",
        consolidation_interval_seconds=3600
    )
)

# 2. Encoding (Prediction Error Gated)
encode_result = brain.encode(
    content="User prefers dark mode.",
    context={"user_id": "u1", "session_id": "s-42"},
    salience=0.8
)
# encode_result.was_encoded = True (gated by novelty/salience)

# 3. Recall
recall_results = brain.recall(
    query="What does the user prefer?",
    memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
    top_k=3
)
for res in recall_results:
    print(res.memory.content, res.relevance_score, res.tier)

# 4. Explicit Fact & Procedure Storage
brain.store_fact(key="timezone", value="UTC+5", confidence=0.9)
brain.store_procedure(name="review", steps=["..."], triggers=["..."])

# 5. Consolidation (Sleep)
sleep_report = brain.sleep(duration_cycles=3)
# Or: brain.consolidate()
```

## 5. Data Models

### Core Entities

- **BrainConfig**: Configuration parameters (decay models, embedding dims, capacity, thresholds).
- **EpisodicMemory**: An individual experience node.
  - Fields: `content` (str), `embedding` (vector), `timestamp` (datetime), `context` (dict), `salience` (float), `stability` (float), `access_count` (int), `last_accessed` (datetime).
- **MemoryType**: Enums (`EPISODIC`, `SEMANTIC`, `FACT`, `PROCEDURAL`).
- **EncodeResult**: Output of encoding.
  - Fields: `was_encoded` (bool), `surprise_score` (float), `memory_id` (str).
- **RecallResult**: Output of retrieval.
  - Fields: `memory` (EpisodicMemory/Semantic), `relevance_score` (float), `tier` (str).
- **ConsolidationReport**: Output of sleep phase.
  - Fields: `consolidated_count` (int), `new_semantic_count` (int), `evicted_count` (int).

## 6. Integration Patterns

- **Streaming / Real-Time (Ingestion):** Interactions flow through the agent into the Prediction Error Gate. If novel/surprising, they are written to the Episodic Store and immediately updated in working memory.
- **Batch / Offline (Consolidation):** A background daemon (SleepEngine) runs during idle periods. It replays prioritized recent memories, extracts semantic gists via Information Bottleneck, updates the Neo4j knowledge graph, and evicts decayed memories.
- **Tool-Calling Agents:** The Brain can be exposed to function-calling LLMs via `remember()` and `recall()` tool schemas, or natively integrated as an MCP server.

## 7. Design Decisions

- **Why Not Pure RAG?**
  RAG lacks temporal dynamics, causal relationships, active forgetting, and statefulness. The Brain prioritizes reconstructive recall and associative context over raw semantic similarity.
- **Why TF-IDF (Initial) vs Neural Embeddings?**
  TF-IDF was used for baseline tests to prove algorithmic scaffolding without neural overhead. Moving to production requires neural embeddings (e.g., `text-embedding-3-small` with MRL reduction to 256d) for rich semantic understanding.
- **Why SQLite / pgvector?**
  SQLite/in-memory was used for MVP testing. Production relies on **PostgreSQL + pgvector** (with TimescaleDB) to combine vector similarity and robust relational/temporal filtering, transitioning to Qdrant/Milvus at massive scale.
- **Why Pydantic?**
  Strict schema validation ensures the structured memory entities (Episodic, Semantic facts) remain consistent when integrating with diverse LLM outputs and database stores.
- **Weight Choices & Thresholds:**
  Parameters like Spaced Repetition efficiency ($\alpha = 0.3$), decay modifier ($w = -0.01$), and survival threshold ($0.05$) were tuned via 11 computational experiments to achieve a Human Cognitive Similarity Index (HCSI) of 0.80, perfectly modeling Ebbinghaus forgetting curves and retrieval-induced inhibition.
