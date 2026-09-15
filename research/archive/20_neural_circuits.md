# 20 — Neural Oscillations, Circuits, and Spatial Memory

**Date**: 2026-09-14
**Stream**: Wave 2 — Neural Circuits Research

---

## 1. Theta-Gamma Coupling: The Brain's Memory Clock

### The Frequencies
- **Theta rhythm (4-8 Hz)**: Slow, high-amplitude oscillation dominant in hippocampus during active exploration, REM sleep, active learning. The brain's **temporal scaffold / clock**.
- **Gamma oscillations (30-100 Hz)**: Fast, low-amplitude. Associated with localized cortical processing and representation of specific items.

### The Theta-Gamma Code (Lisman & Jensen, 2013, *Neuron*)

Each gamma cycle nested within a single theta cycle represents ONE distinct memory item:

```
One Theta Cycle (~200ms, 5 Hz):
┌──────────────────────────────────────────────┐
│  γ₁   γ₂   γ₃   γ₄   γ₅   γ₆   γ₇         │
│ Item1 Item2 Item3 Item4 Item5 Item6 Item7    │
│ (~30ms each gamma cycle at ~40Hz)             │
└──────────────────────────────────────────────┘
```

### Working Memory Capacity Explained

- One theta cycle = ~125-250 ms
- Each gamma cycle = ~10-30 ms
- Maximum gamma cycles per theta cycle = ~4-8
- **This explains Miller's "Magic Number 7±2"** — working memory limit is a PHYSICAL limit of how many gamma cycles fit inside theta

### AI Implication
A dual-oscillator architecture: slow temporal scaffold (theta) dictates sequence/order, fast localized processing (gamma) handles content within each slot.

---

## 2. Sharp Wave Ripples (SWRs) and Memory Replay

### What SWRs Are
- **Frequency**: 150-250 Hz transient oscillatory events in hippocampus (CA3 → CA1)
- **When**: During slow-wave sleep (NREM) and quiet wakefulness
- **Duration**: ~50-100 ms each

### Time Compression
During SWR, place cell sequences that took seconds during awake experience are replayed in ~50ms — approximately **20x faster** than real-time.

### Forward vs Reverse Replay

| Type | Direction | When | Purpose |
|:-----|:----------|:-----|:--------|
| Forward | A → B → C | During sleep/rest | Memory consolidation, trajectory planning |
| Reverse | C → B → A | Immediately after reward | Credit assignment, value learning |

### Causal Evidence: Girardeau et al. (2009)
Selectively suppressing SWRs during post-training sleep severely impaired spatial memory without affecting sleep architecture. **SWRs are necessary for consolidation.**

### AI Implication
Background "replay" process that compresses temporal sequences and interleaves them with older memories — this is the computational mechanism that transfers hippocampal memories to neocortex.

---

## 3. Place Cells, Grid Cells, and Cognitive Maps

### The Discovery Timeline
- **Tolman (1948)**: Proposed "cognitive maps" — internal spatial representations
- **O'Keefe & Nadel (1978)**: Discovered **place cells** in hippocampus — fire at specific locations
- **Moser & Moser (2005)**: Discovered **grid cells** in entorhinal cortex — fire at hexagonal lattice of locations

### Place Cells (CA1/CA3)
Each place cell has a "place field" — fires only when the animal is at a specific location.

### Grid Cells (Medial Entorhinal Cortex)
A single grid cell fires at MULTIPLE locations forming a perfect hexagonal lattice. Provides the universal **metric/coordinate system** for space.

### From Spatial to Abstract Memory
**Critical insight**: The spatial machinery is REPURPOSED for abstract memory:
- Hippocampus/entorhinal cortex map **conceptual spaces** (social hierarchies, timelines, feature dimensions)
- Navigating a "map" of semantic concepts uses the same circuitry as navigating physical space
- Memory IS navigation through a cognitive map

### AI Implication
Abstract reasoning = navigating a continuous coordinate space. Grid cell analogues could provide the metric for organizing semantic knowledge.

---

## 4. Hippocampal Circuit Architecture

### The Trisynaptic Pathway

```
Entorhinal Cortex (EC) Layer II
        ↓ Perforant Path
Dentate Gyrus (DG) ← PATTERN SEPARATION
        ↓ Mossy Fibers
CA3 ← PATTERN COMPLETION (autoassociative recurrent network)
        ↓ Schaffer Collaterals
CA1 ← MISMATCH DETECTION (comparator)
        ↓
EC Layer V → Neocortex (long-term storage)
```

### Component Functions

| Region | Function | Mechanism | AI Equivalent |
|:-------|:---------|:----------|:-------------|
| **DG** | Pattern Separation | Massive neuron count + sparse firing → orthogonalizes similar inputs | Hashing / locality-sensitive hashing |
| **CA3** | Pattern Completion | Recurrent collaterals → complete memory from partial cue | Hopfield network / autoassociative memory |
| **CA1** | Novelty/Mismatch | Compares CA3 output (memory) with EC input (reality) | Prediction error detection |

### The Monosynaptic Pathway
Direct: EC → CA1. Provides raw sensory ground truth, bypassing DG/CA3 loop. CA1 compares "what I expect" (from CA3) vs "what I see" (from EC).

### AI Implication
The hippocampus IS a prediction error machine:
- DG separates (prevents interference)
- CA3 completes (retrieves from partial cue)
- CA1 compares (detects surprise)

---

## 5. Prefrontal Cortex and Working Memory

### Persistent Firing (Goldman-Rakic)
During delay periods (stimulus gone but must be remembered), PFC neurons fire continuously:
- **Mechanism**: Recurrent excitatory connections between layer III pyramidal neurons with similar tuning
- Balanced by GABAergic lateral inhibition
- Creates attractor states that keep information "online"

### PFC-Hippocampus Interaction
- PFC provides TOP-DOWN control over hippocampus
- Dictates what to encode, what to retrieve, what to suppress
- Working memory (PFC) acts as the "director" of long-term memory (hippocampus)

### AI Implication
Working memory isn't just a buffer — it's an active control system that governs the encoding/retrieval process.

---

## 6. Phase Precession and Temporal Coding

### O'Keefe & Recce (1993): Theta Phase Precession

As an animal moves through a place cell's field:
- **Entering**: Cell fires LATE in theta cycle
- **Center**: Cell fires at MIDDLE of theta cycle
- **Exiting**: Cell fires EARLY in theta cycle

```
Theta Phase:  360° ←────────────────── 0°
              Late                    Early
Position:     Enter    Center    Exit
```

### Why This Matters: Temporal Compression + STDP

Multiple cells precessing simultaneously compress a seconds-long behavioral sequence into a single ~150ms theta cycle:
- A fires early, B fires middle, C fires late
- This millisecond-scale ordering aligns with **Spike-Timing-Dependent Plasticity (STDP)**: synapses strengthen ONLY when presynaptic fires milliseconds before postsynaptic
- Phase precession FORCES slow sequences into STDP timing windows → wiring sequences into memory

### AI Implication
Phase coding could encode temporal ORDER within the phase of an activation function, rather than requiring explicit sequential processing or positional embeddings.

---

*Key Papers: Lisman & Jensen 2013 (theta-gamma code), Girardeau et al. 2009 (SWR disruption), O'Keefe & Nadel 1978 (place cells), Moser & Moser 2005 (grid cells), Goldman-Rakic 1995 (PFC working memory), O'Keefe & Recce 1993 (phase precession)*

*Previous: [← Molecular Biophysics](19_molecular_biophysics.md) | Next: [→ Computational Models](21_computational_models.md)*
