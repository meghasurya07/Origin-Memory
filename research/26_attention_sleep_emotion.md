# 26 — Attention, Sleep Architecture, and Emotion-Memory Coupling

**Date**: 2026-09-14
**Stream**: Wave 4 — Attention/Sleep/Emotion Research

---

## 1. Attention as Memory Gate

### Biased Competition Model (Desimone & Duncan, 1995)

When multiple stimuli compete for neural representation:
- **Top-down signals** from PFC/parietal cortex BIAS competition toward task-relevant stimuli
- Attended stimuli get robust firing → fed to hippocampus
- Ignored stimuli are actively SUPPRESSED → never encoded

### Inhibition of Return (IoR)

Attention is biased AGAINST returning to previously attended locations:
- Prevents redundant re-encoding of the same information
- Forces attention to explore NOVEL stimuli
- Maximizes breadth of encoded environmental features

### Feature-Based vs Spatial Attention

| Type | What it does | Memory effect |
|:-----|:-------------|:-------------|
| Spatial ("where") | Enhances processing at specific locations | Remember WHERE things were |
| Feature-based ("what") | Enhances specific attributes across entire field | Remember WHAT things looked like |

**AI Implication**: Memory encoding should be gated by an attention mechanism that selects task-relevant features while actively suppressing irrelevant ones. Inhibition of return prevents storing duplicates.

---

## 2. Sleep Architecture and Memory Consolidation

### Sleep Stages and Memory Types

| Stage | Frequency | Memory Type | Mechanism |
|:------|:----------|:-----------|:----------|
| NREM Stage 2 | Spindles (12-15 Hz) | Declarative | Thalamocortical plasticity gating |
| NREM Stage 3 (SWS) | Slow oscillations (0.5-4 Hz) | Declarative | Hippocampal-neocortical dialogue |
| REM | Theta (4-8 Hz) | Emotional + Procedural | Schema integration, emotional uncoupling |

### The SO-Spindle-Ripple Coupling — The EXACT Consolidation Sequence

This is the precise hierarchical timing mechanism:

```
Step 1: Neocortical Slow Oscillation (SO, 0.5-4 Hz)
        └── Depolarizing "up-state" provides global timing frame
            ↓
Step 2: Thalamocortical Sleep Spindle (12-15 Hz)
        └── SO up-state triggers spindle in thalamus
            └── Spindle induces Ca²⁺ influx in cortical dendrites → STDP
                ↓
Step 3: Hippocampal Sharp-Wave Ripple (150-250 Hz)
        └── Spindle "nests" a ripple within its trough
            └── Hippocampus replays memory (20x compressed)
                ↓
Result: Cortex is in MAXIMALLY PLASTIC state (spindle + SO up-state)
        when hippocampus delivers the memory (ripple)
        → Memory transferred from hippocampus to neocortex
```

### REM Sleep: Emotional Memory Processing

Matthew Walker's "Sleep to Forget, Sleep to Remember":
- REM consolidates the INFORMATIONAL CORE of emotional memories
- While actively STRIPPING the visceral emotional charge
- Result: You remember WHAT happened, but stop FEELING it so intensely
- Failure of this process → PTSD (emotion never uncoupled)

### Duration Requirements

| Duration | What Happens |
|:---------|:-------------|
| 90 min (1 cycle) | Baseline synaptic downscaling + localized consolidation |
| 4-5 hours | Multiple SWS cycles for hippocampal-neocortical transfer |
| 7-8 hours | Full REM density in late morning for associative integration, creativity |

### AI Implication

The "sleep" process needs MULTIPLE PHASES:
1. **SWS-equivalent**: Replay memories at high compression, transfer to semantic store
2. **REM-equivalent**: Integrate into schemas, decouple emotional salience
3. **Spindle-equivalent**: Gate when cortical parameters are updated (not random times)

---

## 3. Emotion-Memory Coupling at Circuit Level

### McGaugh's Modulation Hypothesis

The amygdala is NOT the storage site — it's the MODULATOR. It enhances consolidation in other regions (hippocampus, cortex) during hours after learning.

### The BLA → Hippocampus Pathway

```
Stressful event
    ↓
Sympathetic nervous system activates
    ↓
Adrenaline released (can't cross blood-brain barrier)
    ↓
Vagus nerve → triggers central noradrenaline release
    ↓
Basolateral Amygdala (BLA) activated
    ↓
BLA → ventral hippocampus projection
    ↓
Enhanced LTP, lowered encoding threshold
    ↓
STRONGER MEMORY FORMED
```

### The Yerkes-Dodson Inverted-U

| Arousal Level | Memory Effect | Mechanism |
|:-------------|:-------------|:----------|
| Low | Poor memory | Insufficient noradrenaline, low LTP |
| **Moderate** | **BEST memory** | Optimal MR activation, robust LTP |
| High (extreme stress) | Impaired memory | GR saturation, LTP suppressed, LTD induced, shift to habit memory (striatum) |

### Cortisol's Dual Role

- **Moderate cortisol**: Crosses blood-brain barrier → binds hippocampal receptors → enhances gene expression for synaptic plasticity
- **Excessive cortisol**: Saturates receptors → suppresses LTP → induces LTD → hippocampal atrophy (chronic stress)

### AI Implication

Emotional modulation should follow the inverted-U:
- Low salience → minimal encoding
- Moderate salience → STRONGEST encoding
- Extreme salience → encode but MUST contextualize (prevent PTSD-like feedback loops)

---

## 4. Interference Theory — Deep Dive

### Proactive vs Retroactive Interference

| Type | Direction | Example | Mechanism |
|:-----|:----------|:--------|:----------|
| **Proactive** | Old → New | Ex's name blocks new partner's name | Prior associations compete during new encoding |
| **Retroactive** | New → Old | New password blocks old password | New associations overwrite overlapping traces |

### AB-AC Paradigm
- Learn Apple-Tree (A-B)
- Then learn Apple-Phone (A-C)
- A-C retroactively interferes with A-B recall
- A-B proactively interferes with A-C learning

### Brain's Solution: Pattern Separation (DG)
Dentate Gyrus takes similar, overlapping inputs → maps to distinct, non-overlapping neuron populations → prevents interference.

### Retrieval-Induced Forgetting (Anderson & Bjork, 1994)

**Critical finding**: Retrieving a memory ACTIVELY SUPPRESSES competitors.
- Trying to recall "Fruit - Orange"
- "Fruit - Banana" gets activated as competitor
- PFC applies INHIBITORY CONTROL to suppress Banana
- Result: Banana is HARDER to recall later than if never tested

### AI Implication
1. Pattern separation during encoding (orthogonalize similar memories)
2. Retrieval should model competitive inhibition (suppress competitors)
3. Retrieval-induced forgetting is a FEATURE — it sharpens future retrieval

---

## 5. Memory Replay Types

### Classification

| Type | Direction | When | Purpose |
|:-----|:----------|:-----|:--------|
| **Forward** | A→B→C | Sleep | Systems consolidation |
| **Reverse** | C→B→A | Awake (at reward) | Credit assignment, value learning |
| **Preplay** | Novel sequences | Pre-task | Planning, simulation |
| **Awake replay** | Various | Brief pauses | Decision-making, WM updating |
| **Sleep replay** | Forward dominant | SWR during SWS | Neocortical transfer |

### Prioritized Replay

NOT everything replayed equally. Priority determined by:
1. **High reward** events
2. **High fear** events
3. **High prediction error** (surprise)
4. Dopaminergic inputs flag salient events for prioritized offline replay

### Generative Replay (2024-2025 Focus)

**Replay is NOT just VCR playback.** The hippocampus engages in GENERATIVE replay:
- Stitches elements from DIFFERENT experiences
- Creates novel sequences that NEVER actually occurred
- Foundation for: inference, generalization, cognitive flexibility
- Allows solving novel problems without catastrophic forgetting

### AI Implication

Our consolidation engine should implement:
1. **Prioritized replay** — replay surprising/important events more
2. **Generative replay** — create novel combinations from existing memories
3. **Multi-type replay** — forward for consolidation, reverse for credit assignment
4. **Awake micro-replay** — brief replays during pauses for immediate decision support

---

*Key Papers: Desimone & Duncan 1995 (biased competition), Diekelmann & Born 2010 (sleep consolidation), Staresina et al. 2015 (SO-spindle-ripple), Walker 2017 (REM emotional processing), McGaugh 2000 (memory modulation), Anderson & Bjork 1994 (RIF), Ólafsdóttir et al. 2018 (replay)*

*Previous: [← Math Bridge](25_math_to_code_bridge.md) | Next: [→ Product Architecture](27_product_architecture.md)*
