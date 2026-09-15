# 14 — AI Memory Systems: State of the Art (2024-2026)

## Overview

This document catalogues the latest advances in memory-augmented AI systems — what's beyond simple RAG and vector databases. Understanding the cutting edge is essential for Origin AI to leapfrog, not follow.

---

## 1. Transformer Memory Mechanisms

### 1.1 Infini-attention (Google, 2024)

**The Problem**: Standard attention scales quadratically with sequence length. KV caches grow linearly and eventually OOM.

**The Solution**: Splits attention into two streams:
- **Local Masked Attention**: High-resolution attention over the recent window
- **Global Linear Attention**: A compressive memory matrix that summarizes all past context

**Key Insight**: The compressive memory acts like a learned summary of everything the model has ever seen — constant memory cost regardless of sequence length.

**Relevance to Origin AI**: Infini-attention is the *model-level* solution to long context. Origin AI's approach is the *system-level* solution — external memory that persists across sessions, models, and deployments. These are **complementary**, not competing.

### 1.2 Test-Time Training (TTT-E2E, 2025-2026)

**The Innovation**: Instead of using a fixed KV cache, TTT dynamically updates model weights during inference using self-supervised learning on the input sequence.

**Key Insight**: The model literally learns from each conversation in real-time, updating its weights to better handle the current context.

**Relevance**: This is closer to what Origin AI does at the memory layer — but TTT does it at the weight level. TTT is expensive (requires backpropagation during inference) while Origin AI's approach is efficient (database operations).

### 1.3 Hybrid Architectures (Jamba/Mamba)

**The Innovation**: Interleave Transformer attention blocks with Mamba-style state-space model (SSM) blocks. Attention handles precise recall; SSM handles compressed long-range context.

**Relevance**: The attention/SSM split mirrors Origin AI's episodic/semantic split — fast precise recall + slow compressed knowledge.

---

## 2. Graph-Based Memory for Agents

### 2.1 Microsoft GraphRAG (2024)

**The Innovation**: Uses LLMs to extract a structured knowledge graph from unstructured text, then applies hierarchical community detection (Leiden algorithm) to create summaries at different granularity levels.

**Architecture**:
```
Documents → LLM Entity Extraction → Knowledge Graph → Community Detection → Multi-Level Summaries
```

**Key Insight**: Graph structure enables multi-hop reasoning that flat vector similarity cannot. "Who does Alice work with, and what projects are they on?" requires traversing relationships.

**Relevance to Origin AI**: GraphRAG's entity extraction pipeline is directly applicable to Origin AI's schema discovery and semantic knowledge graph. We should build on this approach.

### 2.2 Zep / Graphiti — Bi-Temporal Knowledge Graphs

**The Innovation**: Tracks facts by TWO timestamps:
- **Event time**: When the fact was true in reality
- **Observation time**: When the system learned the fact

**Contradiction Resolution**: When a new fact contradicts an old one (e.g., "Alice now lives in SF" vs. stored "Alice lives in NYC"), Graphiti **invalidates the old edge** rather than creating a parallel truth.

**Key Insight**: This is a form of **reconsolidation** — updating existing knowledge when new information arrives, rather than just accumulating contradictions.

**Relevance**: Origin AI's reconsolidation engine does this at the memory level. Zep does it at the graph level. We need both.

### 2.3 Cognee — Extract-Cognify-Load

**The Innovation**: Validates extracted knowledge against RDF/OWL ontologies before storage, ensuring semantic consistency.

**Relevance**: This is similar to Origin AI's schema engine — validating that new information fits known structures before integration.

---

## 3. Continual Learning Breakthroughs

### 3.1 Surprise-Prioritized Replay (SuRe)

**The Innovation**: Instead of randomly sampling from a replay buffer, SuRe prioritizes experiences with **high prediction error** (surprise). This is directly analogous to the brain's prediction-error-driven encoding (see doc 13, section 4).

**Formula**:

$$P(\text{replay}_i) \propto \mathcal{L}(x_i) - \mathbb{E}[\mathcal{L}]$$

Where $\mathcal{L}(x_i)$ is the loss on example $i$. Higher loss = more surprising = more likely to be replayed.

### 3.2 Low-Rank Circuit Projection (LRCP)

**The Innovation**: Identifies the specific neural circuits responsible for each task, then constrains gradient updates to **orthogonal subspaces** — preventing new learning from interfering with old knowledge.

**Key Insight**: This is the computational equivalent of the brain's **pattern separation** — keeping representations distinct to prevent interference.

**Relevance**: Origin AI's DG-inspired sparse coding in `hippocampus.py` does this at the memory encoding level. LRCP does it at the model weight level.

### 3.3 Sleep Consolidation Frameworks

**The Innovation**: Async offline processing that mimics biological sleep. During "downtime," the system replays experiences, consolidates knowledge, and prunes redundant information.

**Relevance**: This IS what Origin AI's `ConsolidationDaemon` does. We're already on the right track — but need to make consolidation much smarter (LLM-based fact extraction, schema matching).

---

## 4. Multi-Agent Memory

### 4.1 Scoped Sharing Architecture

The emerging consensus for multi-agent memory uses three tiers:

| Scope | Access | Content |
|:------|:-------|:--------|
| **Global** | All agents | Shared knowledge base, company facts |
| **Role-based** | Agents with same role | Task-specific procedures, domain knowledge |
| **Private** | Single agent | Personal conversations, user preferences |

### 4.2 CRDT-Inspired Consistency

**The Innovation**: Using Conflict-Free Replicated Data Types (CRDTs) from distributed systems to handle concurrent memory updates without centralized coordination.

**Key Insight**: When two agents simultaneously learn conflicting facts about a user, CRDTs provide deterministic conflict resolution without a central "memory server."

### 4.3 Governed Shared Memory (2026)

**The Innovation**: Every memory write includes:
- **Provenance**: Which agent wrote it, from what source
- **Confidence**: How certain the agent is
- **Audit trail**: Complete history of modifications
- **Policy gates**: Who can read/write based on access rules

**Relevance**: Origin AI needs this for enterprise deployments where multiple agents serve the same user/organization.

---

## 5. Neuro-Symbolic Memory

### 5.1 The Hybrid Retrieval Stack

The frontier approach combines:
1. **Neural retrieval**: Embedding similarity for fuzzy matching
2. **Symbolic retrieval**: Logic-based querying for precise facts
3. **Graph traversal**: Relationship following for multi-hop reasoning

```
Query → Neural (embedding search) → candidates
      → Symbolic (exact match)     → candidates  } → Merge + Re-rank → Answer
      → Graph (traversal)          → candidates
```

### 5.2 Multi-Checker Pipelines

**The Innovation**: For verified knowledge, use:
1. Neural component handles language understanding and generalization
2. Symbolic component handles logical deduction and constraint checking
3. Results are cross-validated before being committed to memory

**Relevance**: Origin AI's metamemory engine should integrate this — cross-validating memory retrieval against logical constraints before presenting answers.

---

## 6. Evaluation Advances (2025-2026)

### New Benchmarks

| Benchmark | Year | Tests | Key Innovation |
|:----------|:-----|:------|:-------------|
| **LoCoMo** | 2024 | Multi-session long-term memory | De facto standard, 4 query categories |
| **LongMemEval** | 2024-2025 | Operational memory functions | Tests contradiction handling and updates |
| **MemoryAgentBench** | 2025-2026 | Dynamic knowledge updates | Tests contextual reliability |
| **TOFU** | 2025 | Machine unlearning | Measures targeted fact deletion without collateral damage |
| **RWKU** | 2025-2026 | Real-world knowledge unlearning | More realistic unlearning scenarios |
| **AuthMem-Bench** | 2026 | Consolidation quality | Tests if transient → structured knowledge preserves authority/source |
| **RecMem** | 2026 | Reconsolidation quality | Tests memory update vs new memory creation |

### Evaluation Dimensions We Need

| Dimension | What It Tests | Existing Coverage |
|:----------|:-------------|:-----------------|
| Recall accuracy | Can it retrieve the right fact? | ✅ LoCoMo |
| Temporal awareness | Does it know WHEN things happened? | ✅ LoCoMo |
| Contradiction handling | Does it resolve conflicting info? | ⚠️ LongMemEval (limited) |
| Forgetting quality | Does it forget the RIGHT things? | ⚠️ TOFU (limited) |
| Consolidation quality | Does episodic → semantic work? | ⚠️ AuthMem-Bench (new) |
| Schema learning | Can it learn patterns from examples? | ❌ No benchmark |
| Emotional modulation | Are emotional events remembered better? | ❌ No benchmark |
| Prospective memory | Can it remember future intentions? | ❌ No benchmark |
| Metamemory | Does it know what it knows? | ❌ No benchmark |
| Interference resistance | Does new learning corrupt old memories? | ❌ No benchmark |

> [!IMPORTANT]
> **Origin AI should create benchmarks for the ❌ categories above.** Nobody is testing these dimensions. Publishing OriginBench with these categories would establish us as the authority on brain-inspired memory evaluation.

---

## 7. Where Origin AI Sits in the Landscape

```
                    Neuroscience Depth
                         ▲
                         │
                    ●    │          ● Origin AI
              Numenta    │          (neuroscience + DNA + product)
                         │
                         │     ● Metacognition/Tex
                         │     (neuroscience + math)
                         │
         ────────────────┼─────────────────────────▶ Product Readiness
                         │
                    ●    │   ● Letta/MemGPT
              Academic   │   (engineering)
              Labs       │
                         │ ● Mem0   ● Zep
                         │ (API)    (graphs)
                         │
```

**Origin AI's unique position**: Deepest neuroscience grounding + DNA substrate + full product infrastructure. No one else combines all three.

---

*Previous: [← Advanced Neuroscience](13_advanced_neuroscience.md) | Back to [README](README.md)*
