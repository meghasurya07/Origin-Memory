# 29 — Cognitive Maps, Grid Cells, and Memory Benchmarks

**Date**: 2026-09-14
**Stream**: Wave 5 — Cognitive Maps & Benchmarks

---

## 1. Grid Cells for Abstract Reasoning

The brain's spatial navigation machinery is **repurposed for abstract thought**.

### Key Papers

**Bellmund et al. (2018) "Navigating cognition" (*Science*)**:
- The brain exapts spatial navigation mechanisms (grid/place cells) to map abstract, high-dimensional conceptual spaces
- Abstract concepts placed on spatial coordinate systems → brain can "navigate" relationships

**Behrens et al. (2018) "What is a cognitive map?" (*Nature Neuroscience*)**:
- Medial temporal lobe employs a general-purpose clustering and factorization algorithm
- Hexagonal grid-like firing patterns emerge naturally when applied to any continuous, high-dimensional domain
- This is a **universal mechanism** for structuring knowledge — not just spatial

**Whittington et al. (2020) "The Tolman-Eichenbaum Machine" (*Cell*)**:
- Unifies spatial representations and relational memory
- Separates STRUCTURAL knowledge (rules/geometry) from SENSORY input
- Mechanisms parallel Transformers: self-attention ≈ memory retrieval, path integration ≈ positional encodings

### 2024-2025 Developments
- Non-spatial grid-like codes mature between ages 8-25, predicting improvements in reasoning and intelligence
- Grid cells map "social cognitive maps" (warmth vs competence axes)
- Grid cells provide the METRIC for any continuous representational space

### AI Implementation
Grid cell analogues = continuous vector spaces + structural constraints (transition vectors) enabling algebraic navigation through semantic relationships. Decouple STRUCTURE from CONTENT.

---

## 2. Successor Representation (SR)

### Stachenfeld et al. (2017) "The hippocampus as a predictive map"

The hippocampus doesn't just record WHERE you are — it encodes WHERE YOU'RE LIKELY TO GO:

$$M(s, s') = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \mathbb{I}(s_t = s') \mid s_0 = s\right]$$

Where:
- $\gamma$ = discount factor (0 < γ < 1)
- $\mathbb{I}(s_t = s')$ = indicator: 1 if at state $s'$ at time $t$

### Properties
- Place fields SKEW backward along familiar trajectories (predictive coding)
- Grid cells = eigenvectors of the SR matrix
- When rewards change, SR allows immediate value recomputation WITHOUT relearning structure

### AI Implementation Strategy
Separate memory into:
1. **Transition matrix**: temporal/structural probabilities between semantic concepts
2. **Reward vector**: current goals/relevance
3. **Retrieval**: fast matrix multiplication of transition history × goal vector

---

## 3. OriginBench — Memory Benchmark Design

### Existing Benchmarks (2025-2026)

| Benchmark | Focus | Limitation |
|:----------|:------|:-----------|
| MemoryAgentBench (ICLR 2026) | Multi-turn retrieval, test-time learning, conflict resolution | Doesn't test forgetting quality |
| BEAM | Million-token scale, multi-session reasoning | Context-stuffing can brute-force pass |
| LoCoMo | Cross-session long conversations | Partially saturated |

### OriginBench Design — Tests Our 4 Pillars

| Pillar | Test | Metric |
|:-------|:-----|:-------|
| **1. Prediction Error** | Does system selectively encode surprising info? | Surprise-Retention Correlation |
| **2. Temporal Context** | Can system order events chronologically? Recall by temporal cue? | Temporal Accuracy (Kendall's τ) |
| **3. Forgetting Quality** | Does system decay obsolete info while keeping valid facts? | Forgetting F1 (decay precision × retention recall) |
| **4. Consolidation** | Does system generalize episodes into semantic rules? | Compression Ratio × Generalization Accuracy |

### Additional OriginBench Tests
- **Interference resolution**: AB-AC paradigm — learn A-B then A-C, test A-B recall
- **Reconsolidation**: Present new info about existing memory, test if updated
- **Prospective memory**: Set future intention, test if triggered
- **Emotional priority**: High-salience events recalled better than neutral ones

---

## 4. Developmental Memory and Curriculum Learning

### Human Development Stages

| Age | Memory Stage | Neural Mechanism |
|:----|:------------|:----------------|
| 0-6 months | Recognition memory, sensorimotor schemas | Implicit/procedural only |
| 6-8 months | Object permanence | Working memory emergence |
| 2-3 years | Semantic memory bootstrapping | Language-concept binding |
| 3-4 years | Episodic memory, autonoetic consciousness | Hippocampal maturation |
| 8-25 years | Grid-like codes for abstract reasoning | Entorhinal cortex maturation |

### AI Curriculum Learning
- Start with constrained, foundational data → progressively add complexity
- "Time-limited plasticity" (simulated critical periods) → better generalization
- Mimicking sensory-motor constraints before semantic tasks → robust abstraction

---

## 5. Neuromorphic Computing for Memory

### Intel Loihi / Loihi 2
- Asynchronous, fully digital architecture
- Memory and processing CO-LOCATED within neurocores (no von Neumann bottleneck)
- Native STDP (Spike-Timing-Dependent Plasticity)

### SpiNNaker
- Many-core supercomputer for real-time biological brain simulation
- Packet-based communication for spiking neurons

### Why SNNs for Memory?
- EVENT-DRIVEN: only consume power when spike occurs
- Directly mimics sparse hippocampal activation
- Localized plasticity (STDP) = hardware analogue for associative memory
- **Orders of magnitude more energy efficient** than GPU/CPU for lifelong learning

---

*Key Papers: Bellmund et al. 2018, Behrens et al. 2018, Whittington et al. 2020, Stachenfeld et al. 2017, Indiveri et al. 2011*

*Previous: [← Complete Synthesis](28_complete_synthesis.md) | Next: [→ Latest 2026 AI Memory](30_latest_2026_ai.md)*
