# 30 — Latest 2026 AI Memory, Multi-Modal, Causal Reasoning, Privacy

**Date**: 2026-09-14
**Stream**: Wave 5 — Bleeding Edge 2026

---

## 1. 2026 Memory Papers — The Bleeding Edge

### ZenBrain (arXiv:2604.23878)
The most comprehensive neuroscience-inspired AI memory architecture published to date:
- **7-layer architecture**: Working, Short-Term, Episodic, Semantic, Procedural, Core, Cross-Context
- **15 neuroscience algorithms** including:
  - vmPFC-coupled FSRS (spaced repetition)
  - Simulation-Selection sleep (hippocampal replay)
  - Global Workspace Theory coordinator
- **Our assessment**: Most complete competitor. We must study their implementation and identify where we go deeper (prediction error encoding, TCM temporal context).

### Key ICML/NeurIPS 2026 Papers
- **SimpleMem**: Streamlined memory for efficiency
- **EAM** (Efficient Agentic Memory): Low-overhead memory layer
- **MRAgent**: Multi-session reasoning agent
- **RoboMME**: Vision-language-action robotic memory benchmark
- **MemoryArena / LongMemEval-500**: Long-horizon evaluation

### The 2026 Paradigm Shift
Memory is now treated as **active processes** of "writing, consolidating, and active forgetting" guided by compression heuristics — NOT passive storage. This validates our core thesis.

---

## 2. Multi-Modal Memory

### Cross-Modal Retention (2026)
Frontier models (GPT-5.5, Gemini 3.5 Flash, Claude 4.7) now jointly process text, video, and audio:
- Match frustration in audio tone to facial expressions in video over time
- Persistent physical context tracking for robotics

### MEMLENS Benchmark (mid-2026)
Measures how well models maintain long-term memory across diverse multi-modal data streams.

### Spatial/3D Awareness
2D inputs mapped into spatial and temporal memory layers → persistent physical context tracking for agentic workflows.

### AI Implication for Origin Brain
Our initial focus is text-only memory, but architecture must be EXTENSIBLE to multi-modal. The binding index approach (doc 24 §8) naturally supports this — store modality-specific features in specialized modules, bind via convergence zone index.

---

## 3. Memory in Foundation Models (Internal Mechanisms)

### Memory Taxonomy in LLMs
| Type | What | Where | Persistence |
|:-----|:-----|:------|:-----------|
| Parametric | Facts in weights | Transformer layers | Permanent until retrained |
| Contextual | Working memory | Attention/KV cache | Single session |
| External | RAG/retrieval | Vector DB | Persistent but external |
| Episodic | Agent state | Memory infrastructure | Persistent and managed |

### Mechanistic Interpretability Breakthroughs
- "Circuit discovery" locates exact pathways for factual recall
- Enables precise knowledge editing WITHOUT full retraining
- Specific attention heads and MLP neurons identified as "memory circuits"

### "Thinking to Recall"
Chain-of-Thought acts as a **latent computational buffer**, unlocking parametric knowledge that models fail to access during zero-shot querying. CoT = working memory expansion.

### In-Memory Computing (IMC)
Trend toward bringing compute directly to where parametric weights reside — breaking the von Neumann bottleneck at hardware level.

---

## 4. Causal Memory and Reasoning

### Pearl's Causal Hierarchy in Memory

| Level | Capability | Memory Requirement |
|:------|:-----------|:------------------|
| **Association** | "What if I see X?" | Semantic memory (correlations) |
| **Intervention** | "What if I do X?" | Procedural + episodic memory (action outcomes) |
| **Counterfactual** | "What if I had done X instead?" | Full episodic replay + causal graph |

### Causal Memory Intervention (CMI)
Instead of retrieving by semantic similarity, agents evaluate which memories **causally improve outcomes**, filtering out noisy correlations.

### Hybrid Vector-Graph Architecture
- Vectors handle semantics (similarity)
- Knowledge Graphs handle deterministic, relational, and counterfactual reasoning
- **Causality dossiers**: Auditable causal chains of decisions over time

### AI Implication
Origin Brain needs a causal reasoning layer eventually. The temporal knowledge graph (doc 27 §4) is the foundation — add causal edges (A caused B) alongside temporal edges (A before B).

---

## 5. Privacy and Ethics of AI Memory

### The GDPR Problem
**Article 17 (Right to be Forgotten)** vs AI parametric memory:
- Once personal data is baked into weights → surgical removal nearly impossible
- Machine unlearning (2026) still lacks universally scalable guarantees

### Privacy by Design for Memory
| Approach | How | Status |
|:---------|:----|:-------|
| PII in external, deletable tiers | Separate PII into scoped RAG databases | Production-ready |
| Pre-training data redaction | Anonymize before training | Enforced by EU AI Act |
| Selective forgetting | Delete specific memories on request | Our Pillar 3 naturally supports this |
| Differential privacy | Add noise to memory embeddings | Research stage |

### Origin Brain Advantage
Our architecture of separate, explicit memory stores (episodic, semantic, procedural) with metadata and provenance makes GDPR compliance MUCH easier than parametric memory:
- Can delete specific memories by ID
- Can filter by user_id / tenant_id
- Can prove what was forgotten
- Pillar 3 (forgetting as regularization) naturally supports "right to be forgotten"

---

*Key Sources: ZenBrain (arXiv:2604.23878), MEMLENS 2026, Pearl 2009 (Causality), EU AI Act 2025, RoboMME 2026*

*Previous: [← Cognitive Maps](29_cognitive_maps_benchmarks.md)*
