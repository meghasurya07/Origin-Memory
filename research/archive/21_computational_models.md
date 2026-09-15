# 21 — Existing Computational Models of Memory

**Date**: 2026-09-14
**Stream**: Wave 2 — Computational Models Research

---

## 1. ACT-R (Adaptive Control of Thought—Rational)

**Author**: John R. Anderson, 2007

### Architecture
Separates procedural memory (production rules) from declarative memory (chunks).

### The Activation Equation

Total activation of a memory chunk $i$:

$$A_i = B_i + \sum_j W_j S_{ji}$$

Where:
- **$B_i$ (Base-level activation)** — history of usage (recency + frequency):

$$B_i = \ln\left(\sum_{k=1}^n t_k^{-d}\right)$$

  - $n$ = number of times chunk has been practiced
  - $t_k$ = time since $k$-th practice
  - $d$ = decay parameter (typically ~0.5)

- **$W_j$** — attentional weight from working memory element $j$
- **$S_{ji}$** — spreading activation strength from context $j$ to chunk $i$

### Retrieval
- Chunk retrieved only if $A_i > \tau$ (threshold)
- Retrieval latency: $T = F \cdot e^{-A_i}$ (higher activation = faster retrieval)

### AI Relevance
ACT-R's base-level activation IS the mathematical model of "what you recently used and frequently used is most accessible." This directly maps to our decay/salience engine.

---

## 2. SOAR Cognitive Architecture

**Authors**: Laird, Newell, Rosenbloom

### Memory Systems
- **Procedural**: Production rules (if-then)
- **Semantic**: Long-term declarative facts
- **Episodic**: Stream of experience (working memory snapshots)

### Chunking (Learning)
When SOAR lacks knowledge → "impasse" → creates substate → resolves → compiles solution into permanent production rule ("chunk"). This transforms deliberate problem-solving into automated knowledge.

### AI Relevance
Chunking = the consolidation mechanism. Effortful first-time processing becomes automatic recall.

---

## 3. MINERVA 2 (Hintzman, 1984)

**Core Idea**: Instance-based / multiple-trace model. Every experience stored as a unique trace. No abstraction during storage.

### Retrieval Mechanism

When a probe is presented, it contacts ALL stored traces simultaneously:

- **Echo Intensity** ($I$): Sum of activations of all traces. Signals FAMILIARITY.
- **Echo Content** ($C$): Weighted average of all traces (weighted by activation). Produces RECOLLECTION and naturally extracts prototypes/gist from instances.

### Why This Matters
MINERVA 2 produces BOTH episodic and semantic memory from a SINGLE storage system — it doesn't need separate stores. Semantic knowledge emerges from the statistical overlap of many episode traces.

### AI Relevance
This challenges our multi-store architecture. Maybe semantic memory doesn't need a separate store — it can EMERGE from episodic traces via echo content. This is computationally elegant.

---

## 4. SAM (Search of Associative Memory)

**Authors**: Raaijmakers & Shiffrin, 1980

### Two-Stage Retrieval
1. **Sampling**: Probabilistically select a trace. Probability depends on associative strength to cues relative to ALL other items (competition → retrieval interference).
2. **Recovery**: Attempt to recover full information from sampled trace.

### AI Relevance
Retrieval is COMPETITIVE — items compete for activation. This explains why adding more memories can make specific retrieval harder (interference). Our system needs to model this.

---

## 5. CMR (Context Maintenance and Retrieval)

**Authors**: Polyn, Norman, Kahana

### Evolution of TCM

CMR extends the Temporal Context Model with source and semantic context:

$$\mathbf{c}_t = \rho \mathbf{c}_{t-1} + \beta \mathbf{c}^{IN}_t$$

Where:
- $\mathbf{c}_t$ = current context vector (slowly drifting)
- $\rho$ = persistence (how much old context preserved)
- $\beta$ = update rate (how much new input changes context)
- $\mathbf{c}^{IN}_t$ = context retrieved/evoked by current item

### Key Features
- **Contiguity effect**: Items studied close in time share similar contexts → easier to recall together
- **Source clustering**: Items from same source/category cluster in recall
- **Temporal jumping**: Retrieving an item reinstates its study context → cues temporally adjacent items

### AI Relevance
CMR is the most mature computational model of how context drives memory. Its mathematical framework directly maps to our Temporal Context Engine (Pillar 2).

---

## 6. Global Matching Models (TODAM, CHARM)

### Core Idea
Memory = single high-dimensional composite vector. All items stored in ONE representation via:
- **Convolution** (⊛): Binding operation to compress associated items
- **Correlation** (⊗): Inverse of convolution for retrieval

### Recognition
Test item is compared against the ENTIRE composite memory via dot product. High match = "old", low match = "new".

### AI Relevance
Holographic representations are computationally efficient and map directly to Modern Hopfield Networks. The convolution/correlation operations are differentiable.

---

## 7. Predictive Coding Models of Memory (2024-2025)

### Key Insight
Memory is "fictive prediction error":
- When sensory input is absent, internal cues drive the generative network to predict missing information
- Recall = minimizing prediction error internally (pattern completion)

### Modern Implementations
- **Cognitive Predictive Processing (CPP)** agents use recurrent predictive coding networks
- Local learning rules (not backprop) mimic hippocampal-neocortical loops
- Precision weighting gates memory formation — filtering noise from significant stimuli

### AI Relevance
This validates our Pillar 1 (Prediction Error Encoding). It's not just theory — there are working implementations.

---

## 8. Successor Representation

**Author**: Peter Dayan, 1993

### Core Idea
Factors value function into reward prediction + predictive map:

$$V(s) = \sum_{s'} M(s, s') \cdot R(s')$$

Where $M(s, s')$ is the discounted probability of visiting state $s'$ from state $s$.

### Hippocampus as Predictive Map (Stachenfeld et al., 2017)
Place cells don't just encode current location — they encode the SUCCESSOR REPRESENTATION (likelihood of visiting nearby locations in the future).

### AI Relevance
The hippocampus is fundamentally a PREDICTION ENGINE. Memory = predicting where you'll go / what you'll need next. This directly supports our predictive memory thesis.

---

## 9. Neural Network Models

### CLS Network (O'Reilly, 2002)
Implements complementary learning systems:
- Hippocampal network: sparse, fast learning, pattern separation
- Cortical network: dense, slow learning, overlapping representations
- Experience replay bridges the two

### DeepMind MERLIN / DNC
- External differentiable memory matrix (hippocampus analog)
- Neural network controller (neocortex analog)
- Attention-based read/write (vs. sequential addressing in NTM)

---

## 10. Memory Benchmarks for AI (2024-2026)

| Benchmark | Tests | Key Finding |
|:----------|:------|:------------|
| **MemoryAgentBench** | Retrieval, test-time learning, long-range, conflict resolution | Agents fail at test-time learning and conflicts |
| **LoCoMo** | Cross-session memory over 100s of turns | Partially "saturated" — long-context brute force passes |
| **TOFU** | Unlearning (can agent truly forget?) | Most agents can't selectively forget |
| **BEAM** (2026) | Million-token scale efficiency | Differentiates real memory from context stuffing |
| **AgentMemBench** | Latency, cost, quality across strategies | Graph episodic memory outperforms flat retrieval |

### Where Current Systems Fail
1. **Test-Time Learning**: Updating beliefs mid-session based on new facts
2. **Conflict Resolution**: Handling contradictory information
3. **Temporal Reasoning**: Understanding when facts were true
4. **Selective Forgetting**: Can't unlearn without degrading

---

*Key Papers: Anderson 2007 (ACT-R), Hintzman 1984 (MINERVA 2), Raaijmakers & Shiffrin 1980 (SAM), Polyn et al. 2009 (CMR), Dayan 1993 (SR), Stachenfeld et al. 2017 (hippocampal SR), O'Reilly 2002 (CLS network)*

*Previous: [← Neural Circuits](20_neural_circuits.md) | Next: [→ Memory Disorders](22_memory_disorders.md)*
