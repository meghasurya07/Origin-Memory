# 41 — One-Shot Learning vs Catastrophic Forgetting: The Central Paradox

**Date**: 2026-09-14
**Stream**: Wave 8 — The Central Paradox of Memory
**Type**: RESEARCH (deep mechanisms)

---

## The Paradox

Humans learn in ONE exposure (episodic memory is one-shot). But new learning doesn't destroy old memories. Neural networks suffer catastrophic forgetting. How does the brain solve this?

---

## 1. Elastic Weight Consolidation (EWC) — Synaptic Protection

### The Mechanism
EWC (Kirkpatrick et al. 2017) protects important weights from being overwritten by new learning.

### The Equation

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{i,\text{old}})^2$$

Where:
- $\theta_i$ = current parameter value
- $\theta_{i,\text{old}}$ = parameter value after previous task
- $F_i$ = diagonal of Fisher Information Matrix (importance of parameter $i$)
- $\lambda$ = regularization strength

### How It Works
- Fisher Information $F_i$ measures how sensitive the loss is to changes in $\theta_i$
- High $F_i$ = this parameter is CRITICAL for past tasks → penalize changes heavily
- Low $F_i$ = this parameter is free to change → update normally

### Biological Analogue: Synaptic Metaplasticity
- Synapses crucial for acquired skills become "stiffer" (spine enlargement, receptor density changes)
- Less prone to modification by subsequent learning
- Same principle: protect important connections, free up unimportant ones

### Implication for Origin AI
We should implement **importance-weighted memory protection**:
- Each memory has a "stability" score (analogous to Fisher Information)
- High-stability memories resist modification during reconsolidation
- Low-stability memories can be freely updated
- Our `ReconsolidationEngine` already partially does this — but should use a proper importance metric

---

## 2. Sparse Distributed Memory (SDM / Kanerva Machine)

### Why Sparsity Prevents Interference

SDM operates in high-dimensional binary space (1000+ bits):
- Content-addressable: retrieve by similarity, not exact address
- SPARSE representations: fewer active neurons per pattern
- Low overlap between different memories → massive interference-resistant capacity

### The CA3 Connection

| SDM Component | CA3 Component | Function |
|:-------------|:-------------|:---------|
| Hard addresses | CA3 neurons | Storage locations |
| Address decoder | DG pattern separation | Create sparse codes |
| Content retrieval | CA3 recurrent collaterals | Pattern completion |
| Best-match retrieval | Attractor dynamics | Content-addressable recall |

DG creates sparse representation → CA3 stores and completes → This IS Kanerva's SDM.

### Implication for Origin AI
Our `PatternCompletionEngine` already implements the CA3 side. But we're missing the DG **pattern separation** step — creating maximally orthogonal sparse codes before storage. Adding a proper separation step would dramatically reduce interference.

---

## 3. Neuromodulatory Gating — The Chemical Mode Switches

### Three Modulators, Three Functions

| Modulator | Source | Function | AI Analogue |
|:----------|:-------|:---------|:-----------|
| **Acetylcholine (ACh)** | Basal forebrain | Encode ↔ Retrieve mode switch | `MemoryRouter` mode |
| **Dopamine (DA)** | VTA/SNc | Novelty/reward signal → gate encoding | `PredictionEngine` surprise |
| **Norepinephrine (NE)** | Locus Coeruleus | Arousal/urgency → consolidation priority | `EmotionalModulator` arousal |

### Acetylcholine: The Critical Mode Switch

**High ACh (waking/encoding)**:
- Suppresses internal hippocampal feedback
- Prevents old memories from interfering with new encoding
- Biases toward incoming sensory data

**Low ACh (sleep/consolidation)**:
- Removes suppression
- Enables internal feedback + hippo-cortex information flow
- Initiates consolidation and replay

### Implication for Origin AI
We need a **mode switch** in our Brain:
- **Encoding mode**: Incoming data prioritized, old memories suppressed
- **Retrieval mode**: Old memories activated, interference managed
- **Consolidation mode**: Offline replay, hippo→cortex transfer

Currently our Brain doesn't distinguish modes. Adding ACh-like mode switching would prevent encoding-retrieval interference.

---

## 4. CLS + Generative Replay (Latest 2024-2026)

### The Key Innovation: Hippocampus as GENERATIVE Model

Modern CLS views the hippocampus not as a raw data buffer but as a **generative model**:
- Replay is GENERATION of pseudo-samples from learned distributions
- NOT replaying stored data — generating new synthetic examples
- These pseudo-samples are interleaved with new data to train the neocortex
- This prevents catastrophic forgetting WITHOUT storing massive raw datasets

### Connection to Our SleepEngine
Our SleepEngine Phase 1 (SWS replay) already replays stored memories. But we should evolve it to:
1. Generate SYNTHETIC replay examples (not just replay stored data)
2. Interleave old synthetic + new real during consolidation
3. Use the `ReconstructionEngine` to generate varied replays (different contexts, different fidelity)

This is the key insight: **replay IS reconstruction**. The same mechanism that reconstructs memories during recall can generate training data during sleep.

---

## 5. Reconsolidation: The Update Mechanism

### When Does It Trigger?
**Only when there's a Prediction Error** — if the environment matches the memory perfectly, no destabilization occurs. The PE is the trigger boundary.

### The Lability Window
After destabilization:
1. Memory requires *de novo* protein synthesis to restabilize
2. Time-limited window: typically hours
3. During this window, the memory can be modified, strengthened, or even erased
4. If protein synthesis is blocked during window → memory is disrupted

### Implications for Our ReconsolidationEngine
Our engine already tracks labile memories. But it should:
1. Only trigger reconsolidation when PE > threshold (already partially implemented)
2. Allow memory content modification during lability window
3. Restabilize after a configurable time period
4. Support memory strengthening (not just modification) during reconsolidation

---

*Sources: Kirkpatrick et al. 2017, Kanerva 1988, Hasselmo 1999, Shin et al. 2017, van de Ven et al. 2018, Nader et al. 2000*
