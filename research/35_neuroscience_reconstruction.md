# 35 — Deep Neuroscience: The Reconstruction Mechanism

**Date**: 2026-09-14
**Stream**: Wave 6 — Deep Neuroscience of Reconstruction
**Type**: RESEARCH (neuroscience mechanisms)

---

## 1. Hippocampal Pattern Completion — How CA3 Reconstructs

### The Autoassociative Network (Marr 1971, McNaughton & Morris 1987)

CA3 operates as an **attractor network**: densely interconnected pyramidal neurons with recurrent collateral synapses.

**Encoding**: NMDA-dependent LTP strengthens connections between co-active neurons (Hebbian: "neurons that fire together wire together").

**Retrieval**: A PARTIAL cue activates a SUBSET of the original ensemble. Recurrent collaterals spread activation until the network settles into a stable attractor state → the COMPLETE original pattern is recreated.

```
Partial cue (20% of pattern) → CA3 recurrent activation → 
  → Attractor dynamics → Stable state = FULL ORIGINAL PATTERN
```

**Critical question**: How much of the original needs to be present?
- Studies show CA3 can complete from as little as **20-30% of the original pattern**
- This means 70-80% of the "memory" is RECONSTRUCTED, not retrieved

**Pattern Separation (DG)**: Before reaching CA3, the Dentate Gyrus orthogonalizes similar inputs via ultra-sparse coding (only 2-4% of DG neurons active). This prevents catastrophic interference — similar experiences get mapped to distinct patterns.

### Failure Modes
When completion DOMINATES separation:
- Network falls into WRONG attractor → **false memories**
- Structurally similar but incorrect pattern retrieved → **confabulation**
- This is not a bug — it reveals that memory IS reconstruction

---

## 2. The Binding Problem — How Distributed Fragments Become Unified

### Convergence-Divergence Zones (Damasio 1989)

Memory components are distributed across cortex:
- Visual features → visual cortex
- Sounds → auditory cortex
- Emotions → amygdala
- Spatial layout → parietal cortex

The hippocampus stores an **INDEX** (pointer/binding code), NOT the data itself.

On recall: index → reactivates cortical patterns → binding into unified experience.

### Cohen & Eichenbaum: Relational Binding

The hippocampus binds RELATIONSHIPS between independent cortical representations. It doesn't store "the apple was on the table" — it stores the RELATION between apple-representation and table-representation.

### Oscillatory Binding Mechanism

**Theta oscillations (4-8 Hz)** orchestrate dynamic binding:
- Phase-locking between hippocampus and distributed cortical regions
- Synchronizes disparate sensory fragments into unified conscious experience
- Sharp-wave ripples coordinate the actual content transfer

### When Binding Fails
- **Source amnesia**: remember a fact but not WHERE learned (index partially degraded)
- **Fragmented recall**: remember a face but bind to WRONG context (theta desynchronization)
- **DRM false recognition**: schema fills in details that binding can't verify

---

## 3. Memory IS a Generative Process

### Constructive Episodic Simulation Hypothesis (Schacter & Addis 2007)

**The brain evolved memory NOT to remember the past, but to SIMULATE THE FUTURE.**

Key evidence:
- Episodic recall and future imagination use the **EXACT SAME neural substrate**
- Both recruit the Default Mode Network (DMN): hippocampus + mPFC + posterior cingulate
- Amnesic patients (hippocampal damage) can't remember the past AND can't imagine the future
- This proves memory and imagination are the SAME PROCESS — generative simulation

### Implications for Origin AI
Memory retrieval = running a generative simulation conditioned on past traces.
This is NOT lookup. This IS generation.

The hippocampus extracts and **flexibly recombines** past episodic details to generate:
1. Past recall (biased toward what actually happened)
2. Future simulation (biased toward what might happen)
3. Counterfactual reasoning (what COULD have happened)

All three are the SAME computational mechanism with different constraints.

---

## 4. Sparse Coding — How Few Neurons Encode a Memory

### Engram Cells (Josselyn & Tonegawa 2020)

**A memory is encoded by ~2-4% of neurons in a region.**

Evidence:
- Tonegawa's lab used optogenetics to TAG neurons active during encoding
- Artificially reactivating ONLY this sparse 2% ensemble → full behavioral recall
- The downstream CA3 pattern completion fills in the rest
- This is proof that memory traces are **extremely sparse**

### Key Insight for AI
A memory doesn't need to be stored in full. You need:
1. A **sparse binding code** (the 2-4% index)
2. A **completion mechanism** (CA3 autoassociative network / attractor dynamics)
3. **Distributed representations** to complete against (cortical knowledge)

Our TCM + hippocampus already do #1 and #3. We need #2 — a proper pattern completion mechanism.

---

## 5. Reconsolidation — Memories CHANGE Every Time You Recall

### Nader (2000): The Lability Discovery

Every time a memory is recalled, it becomes **LABILE** (unstable):
- Requires *de novo* protein synthesis to re-stabilize
- During the labile window, the memory can be MODIFIED, STRENGTHENED, or ERASED
- Protein synthesis inhibitors (anisomycin) during recall → memory LOST

### Molecular Mechanism
- **PKMζ** (Protein Kinase M zeta): maintains synaptic weights of the engram
- **ZIP** (Zeta Inhibitory Peptide): inhibits PKMζ → reverses late-phase LTP → unravels trace
- Memory reconsolidation = molecular RE-WRITING of the trace

### Evolutionary Purpose
Reconsolidation exists to **UPDATE memories with new information**:
1. Recall activates the trace
2. Prediction errors (new info) are incorporated
3. The trace is re-stored with the updates
4. The memory is now DIFFERENT from before recall

### Implications for Origin AI
Our `reconsolidation.py` module already handles this! But it operates on the stored content level. True reconsolidation should modify the BINDING INDEX, not just the surface content.

---

## 6. Schemas Accelerate Consolidation (Tse et al. 2007, *Science*)

### The Schema Acceleration Effect

Traditional view: hippocampal → cortical transfer takes WEEKS.
Tse et al. showed: if a compatible SCHEMA exists, consolidation happens in **48 HOURS**.

### vmPFC as Schema Hub
The ventromedial PFC:
1. Detects when new info is CONGRUENT with existing schemas
2. Orchestrates rapid assimilation into cortical networks
3. During recall, CONSTRAINS reconstruction to be schema-consistent

### Implication: Schema-Guided Reconstruction
During recall, vmPFC uses schemas to:
- GUIDE what elements are reconstructed
- FILL IN missing details with schema-consistent predictions
- CONSTRAIN plausibility of the reconstructed memory

This is why you "remember" birthday cake at your party even if you're not sure — the birthday schema predicts it.

---

## Unified Model: The Neural Reconstruction Pipeline

```
RECALL CUE arrives
    ↓
Entorhinal Cortex → sends partial pattern to CA3
    ↓
CA3 PATTERN COMPLETION → attractor dynamics fill in sparse trace
    ↓
Hippocampal INDEX reactivates distributed cortical patterns
    ↓
THETA OSCILLATIONS bind fragments across cortex
    ↓
vmPFC SCHEMAS constrain and fill gaps
    ↓
AMYGDALA tags emotional significance
    ↓
PFC WORKING MEMORY integrates into coherent narrative
    ↓
RECONSTRUCTED MEMORY emerges into consciousness
```

**Key: 70-80% of what you "remember" was GENERATED, not retrieved.**

---

*Sources: Marr 1971, McNaughton & Morris 1987, Damasio 1989, Cohen & Eichenbaum 1993, Nader 2000, Schacter & Addis 2007, Tse et al. 2007, Josselyn & Tonegawa 2020*
