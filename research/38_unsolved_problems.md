# 38 — Unsolved Problems in Memory Neuroscience

**Date**: 2026-09-14
**Stream**: Wave 7 — Deep Unsolved Problems
**Type**: RESEARCH (foundational neuroscience)

---

## 1. Complementary Learning Systems (CLS) — The Two-Speed Architecture

### The Core Problem: How Do Fast and Slow Learning Coordinate?

The brain has TWO learning systems that MUST work together:

| System | Speed | Capacity | Interference | Brain Region |
|:-------|:------|:---------|:-------------|:-------------|
| **Fast** (Hippocampus) | One-shot | Limited | High (new overwrites old) | Hippocampus |
| **Slow** (Neocortex) | Statistical (1000s of exposures) | Massive | Low (gradual integration) | Entire cortex |

### The Transfer Mechanism

**Hippocampus-driven replay** during offline states:
1. During sleep SWR (sharp-wave ripples), the hippocampus "teaches" the neocortex
2. Each replay is a single training example for the neocortex
3. Neocortical synapses update SLIGHTLY per replay (slow learning rate)
4. Interleaving old + new prevents catastrophic interference

### Recent Refinements (2020-2024)
- **Schema acceleration**: When new info matches existing cortical schemas, neocortical integration is FAST (hours not weeks)
- **Bidirectional**: Cortex ALSO triggers specific hippocampal replays (not just passive receiving)
- **Dynamic learning rate**: Neocortical learning rate scales with schema-consistency

### Implication for Origin AI
Our architecture ALREADY implements CLS:
- **Hippocampus** (fast) = `HippocampalEngine` (one-shot encoding)
- **Neocortex** (slow) = `ConsolidationDaemon` + `SemanticMemory` (statistical)
- **Replay** (transfer) = `SleepEngine` Phase 1 (SWS replay)
- **Schema acceleration** = `SchemaEngine` + fast consolidation

BUT we're missing: **dynamic learning rate** based on schema consistency. When new info matches a schema, consolidation should be 10x faster.

---

## 2. Memory Allocation: CREB and Neuronal Competition

### The Core Problem: HOW Does the Brain Choose Which Neurons Encode a Memory?

It's NOT random. Neurons COMPETE.

**The CREB mechanism**:
1. Transcription factor **CREB** (cAMP Response Element-Binding protein) fluctuates in neurons
2. Neurons with HIGH CREB are more excitable (lower firing threshold)
3. When an event occurs, high-CREB neurons WIN the competition
4. These winners become the **engram cells** for that memory

**Molecular details**:
- CREB downregulates BK channels → reduces afterhyperpolarization → easier to fire again
- CREB promotes dendritic spine density → more synaptic connections
- CREB enables "synaptic tagging and capture" → ensures these cells undergo LTP

### Time-linking Effect
Memories encoded CLOSE IN TIME share overlapping engrams (because the same neurons are still in a high-CREB state). This automatically creates temporal associations — and it's EXACTLY what our TCM engine does at the computational level.

### Implication for Origin AI
We should implement a **neuronal allocation** mechanism:
- Each "memory slot" has an **excitability score** (analogous to CREB)
- Excitability fluctuates over time
- During encoding, only high-excitability slots are used → enforces sparsity
- Recently active slots have HIGHER excitability → time-linking
- This would make our memory allocation biologically realistic

---

## 3. Grid Cells and Abstract Concept Spaces

### The Core Problem: The Brain Maps CONCEPTS, Not Just Space

Grid cells in the entorhinal cortex create hexagonal firing patterns for physical navigation. But recent discoveries show they ALSO fire for:
- Social hierarchies (status gradients)
- Value spaces (reward magnitudes)
- Semantic feature spaces (word meanings)
- Time sequences (temporal navigation)

### The Continuous Conceptual Geometry

The hippocampus-entorhinal system creates a **universal coordinate system** for ANY continuous dimension. Memories are locations in this multi-dimensional concept space.

**Manifold representations**: Instead of chaotic remapping between tasks, the system uses stable neural manifolds. It extracts continuous latent variables and projects them onto these manifolds.

### Implication for Origin AI
Don't store concepts in a flat database. Embed them in a **navigable concept space**:
- Each memory has coordinates in multiple dimensions (time, topic, emotion, importance)
- "Recalling" = navigating to a location in concept space
- "Similar memories" = nearby locations
- "Mental time travel" = moving along the temporal dimension
- An AI "grid cell" module could generate structural coordinate vectors

This connects directly to our TCM (temporal dimension) and could extend to semantic/emotional dimensions.

---

## 4. Systems Consolidation: Trace Transformation Theory

### The Debate: What Happens to Memories Over Months/Years?

Three competing theories:

| Theory | Hippocampus Role | Remote Memory Quality | Status |
|:-------|:----------------|:---------------------|:-------|
| **Standard Model** | Temporary store, then discard | Identical to recent | Outdated |
| **Multiple Trace Theory** | New trace each retrieval | More traces = stronger | Partial |
| **Trace Transformation** | Keeps vivid detail forever | TRANSFORMS: gist → cortex, detail → hippo | Current consensus |

### Trace Transformation Theory (Current Consensus)
- The hippocampus **indefinitely retains** vivid, context-specific episodic details
- Systems consolidation **extracts overlapping features (gist)** and stores schematic representation in neocortex (vmPFC)
- Over time: detail stays in hippocampus, generalized rules go to cortex
- This is NOT a simple copy — it's an active dimensionality reduction

### Implication for Origin AI
Our consolidation should implement **active dimensionality reduction**:
- Don't just "move" memories from episodic to semantic
- Extract generalized rules and statistical regularities
- Keep high-predictive-value episodic traces
- This is what our SleepEngine Phase 1 does — but it could be much more sophisticated

---

## 5. Working Memory: Theta-Gamma Coupling and the ~4 Item Limit

### The Core Problem: WHY Is Working Memory Limited to ~4 Items?

**The theta-gamma coupling mechanism**:
1. Theta oscillations (4-8 Hz) create temporal "windows" for working memory
2. Gamma bursts (~40-80 Hz) encode individual items within each theta cycle
3. You can fit ~4 distinct gamma bursts into one theta cycle
4. More items → gamma bursts overlap → interference → capacity limit

### The Mathematical Limit

$$\text{Capacity} \approx \frac{f_{gamma}}{f_{theta}} = \frac{40\text{Hz}}{8\text{Hz}} \approx 5 \text{ items}$$

This is Lisman & Idiart's (1995) theta-gamma model. The capacity isn't arbitrary — it emerges from the PHYSICS of neural oscillations.

### Implication for Origin AI
Our working memory (`context_buffer`) has NO capacity limit — it's just a dict. We should:
1. Implement a **capacity-limited attention buffer** (~4-7 active items)
2. Items compete for slots based on salience and recency
3. When buffer is full, lowest-priority item is displaced
4. This naturally creates the "chunking" behavior humans use

---

*Sources: McClelland et al. 1995, O'Reilly & Norman 2002, Kumaran et al. 2016, Han et al. 2007, Josselyn et al. 2015, Moser et al. 2014, Bellmund et al. 2018, Winocur & Moscovitch 2011, Lisman & Idiart 1995, Cowan 2010*
