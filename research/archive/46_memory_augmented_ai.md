# 46 — Memory-Augmented AI: Titans, Competitors, and SNNs

**Date**: 2026-09-14
**Stream**: Wave 9 — State of the Art
**Type**: RESEARCH + COMPETITIVE ANALYSIS

---

## 1. Google Titans — The Closest Thing to Our Approach

### Architecture
Titans (Google, late 2024/2025) split processing into:
- **Core Attention Module** — short-term (standard self-attention)
- **Neural Long-Term Memory Module** — TEST-TIME weight updates via gradient descent with momentum

### The Surprise Mechanism
Titans incorporate a **surprise mechanism** where novel/unexpected tokens trigger LARGER memory updates. This is EXACTLY our `PredictionEngine` surprise gate.

### Key Connection to Origin AI

| Titans Feature | Our Feature | Status |
|:-------------|:-----------|:-------|
| Surprise-based memory update | PredictionEngine surprise gate | ✅ BUILT |
| Test-time weight updates | on_retrieval() stability update | ✅ BUILT |
| Short-term + long-term split | Working memory + episodic | ✅ BUILT |
| Memory as Context (MAC) | Context buffer | ✅ BUILT |

**Titans validates our approach.** But Titans lacks:
- Reconstructive recall (they retrieve, we reconstruct)
- Sleep consolidation (they have no offline processing)
- Forgetting curves (they don't deliberately forget)
- Schema-based false memories (they don't confabulate)

---

## 2. RAG vs True Memory — The Critical Distinction

### Why RAG Is NOT Memory

RAG (Retrieval-Augmented Generation) is:
- **Stateless** — no persistent state between queries
- **Read-only** — doesn't learn from retrieval
- **Temporally blind** — can't distinguish old vs new facts
- **Context-polluting** — dumps loosely related chunks

True memory (what we build) is:
- **Stateful** — maintains persistent, evolving state
- **Read-write** — retrieval strengthens traces (testing effect)
- **Temporally aware** — tracks when things happened
- **Context-sensitive** — retrieval depends on current context

### The 2026 Consensus
SOTA architectures treat:
- RAG = static grounding layer (documentation)
- Memory = dynamic user-state persistence (learning, adapting)

Origin AI builds the MEMORY layer, not the RAG layer.

---

## 3. Competitor Deep Dive

| System | Architecture | Memory Model | Limitations |
|:-------|:-----------|:------------|:-----------|
| **Mem0** | Extraction → dual store (vectors + graph) | Extract facts → store | No temporal reasoning, no forgetting, no reconstruction |
| **Letta/MemGPT** | LLM as OS with tiered memory | Working → archival paging | High token overhead, relies on LLM routing reliability |
| **Zep/Graphiti** | Temporal Knowledge Graph | Bi-temporal (event time + ingestion time) | Rigid schema, no fluid episodic recall |
| **LangMem** | LangGraph dual-path | Hot-path real-time + background async | Heavy ecosystem lock-in to LangChain |
| **Origin AI** | **Reconstructive brain model** | **6-pillar architecture** | **No neural embeddings yet (TF-IDF), no LLM reconstruction** |

### What NONE of them do (and we DO):
1. ❌ Forgetting curves (they never forget)
2. ❌ False memories (they only recall what was stored)
3. ❌ Context-dependent fidelity (they return same content regardless)
4. ❌ Sleep consolidation (no offline processing)
5. ❌ Pattern completion from partial cues
6. ❌ Working memory capacity limits

---

## 4. Spiking Neural Networks — The Future

### SNNs for Hippocampal-Like Memory

Modern SNNs successfully replicate:
- **Dentate Gyrus** → pattern separation
- **CA3** → autoassociative pattern completion
- **CA1** → mismatch detection / prediction error

### STDP (Spike-Timing Dependent Plasticity)
- Learns associations from precise temporal firing
- 2025: "Plasticity-Driven Learning" — network learns plasticity rules dynamically
- **Event-driven** → orders of magnitude less energy than ANNs

### Implication for Origin AI v2.0
Long-term, our architecture could be implemented on neuromorphic hardware (SpiNNaker, Loihi) using SNNs. The computational principles are the same — only the substrate changes.

---

## 5. Cognitive Architecture Renaissance

ACT-R and SOAR are being fused with LLMs:
- **ACT-R** → psychological model with distinct memory modules (declarative, procedural, episodic)
- **SOAR** → general problem solving with System 2 reasoning
- **Modern integration** → SOAR's deliberate reasoning + LLM's pattern recognition

### Common Model of Cognition (CMC)
A unified framework with discrete cognitive cycles: Observe → Decide → Act

### Takeaway for Origin AI
We should structure our Brain around the CMC:
1. **Observe** → Encode (prediction error, emotional modulation)
2. **Decide** → Recall (reconstruction, pattern completion)
3. **Act** → Update (reconsolidation, working memory)

This maps cleanly to our existing pipeline.

---

*Sources: Google Titans (2025), Mem0 docs, Letta/MemGPT paper, Zep/Graphiti docs, LangMem docs, SNN STDP literature, ACT-R/SOAR/CMC frameworks*
