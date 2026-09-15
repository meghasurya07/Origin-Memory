# 51 — Synaptic Plasticity & Complementary Learning Systems: The Path to Exact Brain

**Date**: 2026-09-15
**Type**: DEEP NEUROSCIENCE + ARCHITECTURE IMPLICATIONS
**Status**: Active research informing v0.6 design

---

## The Three Levels of Brain Replication

To build an "exact human-like brain" (user's stated goal), we need to understand memory at three levels:

### Level 1: Systems Level (DONE in v0.5)
What brain regions do: hippocampus encodes, neocortex stores, prefrontal controls.
- **Our implementation**: Brain orchestrator, HippocampalEngine, SleepEngine
- **Status**: ✅ Complete

### Level 2: Circuit Level (IN PROGRESS)
How neurons communicate: synaptic connections, oscillations, neuromodulation.
- **Our implementation**: NeurogenesisEngine, NeuromodulationEngine
- **Status**: 🔄 Partial — need STDP and CLS

### Level 3: Molecular Level (NEXT)
What happens at the synapse: NMDA receptors, CaMKII cascades, protein synthesis.
- **Our implementation**: Not yet
- **Status**: ❌ Not started

---

## Spike-Timing Dependent Plasticity (STDP)

### The Biology
STDP is HOW the brain decides which synapses to strengthen and which to weaken:

**If pre fires before post → strengthen (LTP)**
**If pre fires after post → weaken (LTD)**

The mathematical rule:

$$\Delta w = \begin{cases} A_+ \exp(-\Delta t / \tau_+) & \text{if } \Delta t > 0 \text{ (LTP)} \\ -A_- \exp(\Delta t / \tau_-) & \text{if } \Delta t < 0 \text{ (LTD)} \end{cases}$$

Where:
- $\Delta t = t_{post} - t_{pre}$ (timing difference)
- $A_+ \approx 0.01$ (LTP magnitude)
- $A_- \approx 0.012$ (LTD magnitude, slightly larger → net weakening bias)
- $\tau_+ \approx 20\text{ms}$, $\tau_- \approx 20\text{ms}$ (time constants)

### What This Means for Us

STDP is the **microscopic mechanism** behind our macroscopic rules:
- **Testing effect** (retrieval strengthens): Successful retrieval = pre fires before post → LTP
- **Interference** (similar memories compete): Competing memories have bad timing → LTD
- **Consolidation** (sleep replay): Replay events create coordinated timing → selective LTP

### Implementation Path

```python
class SynapticPlasticity:
    def stdp_update(self, pre_time, post_time, current_weight):
        dt = post_time - pre_time
        if dt > 0:  # LTP
            dw = A_plus * math.exp(-dt / tau_plus)
        else:  # LTD
            dw = -A_minus * math.exp(dt / tau_minus)
        return max(0, min(1, current_weight + dw))
```

### Frameworks Available
- **BindsNET**: PyTorch-based SNN simulator with STDP
- **snnTorch**: GPU-accelerated SNN framework
- **SpykeTorch**: STDP + R-STDP focused
- **Auryn**: High-performance C++ SNN simulator

---

## Complementary Learning Systems (CLS) Theory

### The Core Insight (McClelland et al. 1995, updated 2024-2025)

The brain has TWO learning systems that solve a fundamental dilemma:

| Property | Hippocampus | Neocortex |
|:---------|:-----------|:----------|
| **Learning rate** | Fast (one-shot) | Slow (gradual) |
| **Representations** | Sparse, separated | Distributed, overlapping |
| **Function** | Episodic encoding | Semantic extraction |
| **Forgetting** | Rapid | Slow |
| **Interference** | Low (pattern separation) | High (catastrophic) |

### 2025 Updates

#### C-HORSE Model (Intra-Hippocampal CLS)
The hippocampus ITSELF has complementary subsystems:
- **Trisynaptic pathway (TSP)**: EC → DG → CA3 → CA1 = rapid episodic encoding
- **Monosynaptic pathway (MSP)**: EC → CA1 directly = structure/regularity extraction

**Our mapping**: 
- TSP = our HippocampalEngine (fast encoding, pattern separation)
- MSP = our SchemaEngine (structure extraction, schema matching)

#### VAE + Modern Hopfield Networks (Arxiv 2025)
Researchers combined:
- VAE as neocortex analog (slow, generative)
- Modern Hopfield Network as hippocampus analog (fast, content-addressable)
- Result: solved catastrophic forgetting in continual learning

**Our mapping**:
- We already have Modern Hopfield (PatternCompletionEngine with beta=8.0!)
- We need the VAE/generative component for neocortical consolidation
- This validates our architecture choices

#### Meta-Learning of Plasticity (biorxiv 2025)
The brain's hierarchical organization (sensory → association → hippocampus) may emerge from **meta-learning** that optimizes plasticity parameters.

**Implication**: Our fixed plasticity parameters (decay rates, learning rates) should eventually be LEARNED, not hand-tuned.

---

## Sleep-Dependent Consolidation (2025 Update)

New PNAS models show NREM and REM serve complementary roles:

| Phase | What Happens | Our Implementation |
|:------|:------------|:------------------|
| **NREM/SWS** | Sharp-wave ripples replay memories | SleepEngine phase 1: consolidation |
| **REM** | Generative replay, emotional processing | SleepEngine phase 2: emotional integration |
| **Pruning** | Remove weak/redundant memories | SleepEngine phase 3: decay + prune |

The key insight: consolidation is "offline reinforcement learning" — the hippocampus generates simulations evaluated by DA-like mechanisms (Dyna architecture).

**This maps directly to our Prediction + Sleep architecture!**

---

## Architecture Gap Analysis for Exact Brain

| Brain Feature | Status | What's Missing |
|:-------------|:-------|:--------------|
| Episodic encoding | ✅ | — |
| Semantic extraction | ✅ | Needs generative model |
| Forgetting curves | ✅ | — |
| Sleep consolidation | ✅ | Needs DA evaluation |
| Pattern completion | ✅ | — |
| Working memory | ✅ | — |
| Neurogenesis | ✅ | — |
| Neuromodulation | ✅ | — |
| Reconstruction | ✅ | — |
| **STDP plasticity** | ❌ | Need spike timing rules |
| **CLS dual learning** | 🔄 | Have fast system, need slow |
| **Synaptic tagging** | ❌ | Need tag-and-capture |
| **Engram formation** | ❌ | Need memory trace cells |
| **Grid/place cells** | ❌ | Need spatial memory |

### Priority for v0.6
1. **Synaptic plasticity engine** — STDP-inspired weight updates
2. **CLS integration** — Slow neocortical learning system
3. **Engram tracking** — Which "neurons" form each memory trace

---

## Own Thinking: The Abstraction Principle

**Key insight from this research session:**

We don't need to simulate individual spikes. We need to capture the *computational principles* that emerge from billions of spikes.

The relationship between levels:

```
Spikes (ms)     →  abstract to →  Activation patterns (s)
STDP (ms)       →  abstract to →  Learning rules (min-hours)
Oscillations    →  abstract to →  Mode switching (s-min)
LTP/LTD (min)   →  abstract to →  Stability changes (hours-days)
Neurogenesis    →  abstract to →  Capacity dynamics (days-weeks)
Consolidation   →  abstract to →  Memory transformation (days-months)
```

Each level can be faithfully represented at a higher abstraction without simulating the lower level. This is our competitive advantage:
- We capture the **functional behavior** (forgetting, reconstruction, consolidation)
- Without the **computational cost** (10^15 synapses × millisecond resolution)

The STDP equation $\Delta w = A \exp(-|\Delta t|/\tau)$ abstracts to our stability update:
$\text{stability} \times= (1 + \alpha \times \text{retrieval\_success})$

Same principle. Different scale. Same behavior.

---

*Research document for Origin AI. See neurogenesis.py, neuromodulation.py for current implementations.
Next: Synaptic plasticity engine for v0.6.*
