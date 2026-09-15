# 50 — Neuromodulation & Neurogenesis: Building the Chemical Brain

**Date**: 2026-09-15
**Type**: IMPLEMENTATION + NEUROSCIENCE
**Status**: Both engines built, tested, integrated into Brain v0.5

---

## What We Built

### Neuromodulation Engine (neuromodulation.py)
Implements three key neuromodulators that control memory:

#### 1. Acetylcholine (ACh) — Encoding/Retrieval Mode Switch
**Source**: Hasselmo 1999, SPEAR Model (Hasselmo 2025)

- **High ACh → Encoding mode**: New inputs prioritized, recurrent connections suppressed
- **Low ACh → Retrieval mode**: Pattern completion enabled, old memories accessible
- **Implementation**: `on_novelty(score)` boosts ACh, `encoding_drive = ACh * 0.7 + NE * 0.3`

The SPEAR model (Separate Phases of Encoding and Retrieval, 2025) shows encoding and retrieval alternate within theta cycles (~125ms). Our implementation uses a continuous ACh level with decay, which approximates the same switching behavior at a higher abstraction level.

#### 2. Dopamine (DA) — Reward/Surprise Signal
**Source**: Schultz 1997, TD Learning, RPE Framework (Schultz 2024)

- **Unexpected outcomes → DA burst → stronger consolidation**
- **Prediction errors drive DA release** (maps to our PredictionEngine surprise_score)
- **Implementation**: `on_surprise(magnitude)` boosts DA, `consolidation_modifier = 0.5 + DA * 0.6 + (1-NE) * 0.4`

2024 update: Schultz published framework showing DA RPE signals as causal mechanism for reward maximization. Memory consolidation modeled as "offline RL" — hippocampus generates simulations evaluated by DA-like mechanisms (Dyna architecture).

#### 3. Norepinephrine (NE) — Arousal/Urgency
**Source**: McGaugh 2000, flashbulb memory research

- **High arousal → NE surge → flashbulb memory formation**
- **Fight-or-flight → enhanced encoding of current moment**
- **Implementation**: `on_arousal(level)` boosts NE, NE > 0.8 triggers ALERT mode

### Integration into Brain.encode()

The neuromodulation engine is now integrated into the 11-step encode pipeline:

```
Step 1:  Context buffer update
Step 2:  Prediction error evaluation (Pillar 1)
Step 3:  Emotional analysis
Step 3b: NEUROMODULATION SIGNALS ← NEW
         - Novel input → ACh boost (encoding mode)
         - High surprise → DA boost (consolidation)
         - High arousal → NE boost (flashbulb)
Step 4:  Salience modulation × neuromodulation modifier ← ENHANCED
Step 5:  Reconsolidation check
...
Step 11: Neuromodulation decay toward baseline ← NEW
```

---

### Neurogenesis Engine (neurogenesis.py)
**Source**: Spalding et al. 2013 (C-14 dating), Aimone et al. 2006

Implements dynamic neuron creation and destruction:

#### Neuron Lifecycle
```
IMMATURE (high plasticity=1.0)
    → MATURING (plasticity=0.7, after 24h)
        → MATURE (plasticity=0.2, if activated enough)
        → DECLINING (plasticity=0.0, if unused)
            → DEAD (removed from population)
```

#### Key Parameters
| Parameter | Human Brain | Our System |
|:----------|:-----------|:-----------|
| Birth rate | ~700 neurons/day | 10/day (scaled) |
| Death rate | ~85,000/day | 2% daily turnover |
| Maturation | 4-6 weeks | 24h immature + 72h maturing |
| Survival criterion | Synaptic integration | min 2 activations + 0.2 integration |
| Max population | ~86 billion | 1,000 (configurable) |

#### Activity-Dependent Survival
- New neurons that get activated survive → mature → stable
- New neurons that are ignored → decline → die
- This mirrors the brain: "use it or lose it"

---

## Why This Matters for the Product

### Before v0.5
- Brain encoded everything with equal strength
- No mode switching — always in same state
- No dynamic resource allocation

### After v0.5
- **Novel inputs get stronger encoding** (ACh boosts encoding modifier 1.5x-3.0x)
- **Surprising inputs strengthen consolidation** (DA boosts consolidation modifier)
- **Emotional/urgent inputs create flashbulb memories** (NE triggers ALERT mode)
- **Neuron population is dynamic** — new slots created, unused ones pruned

### Competitive Advantage
No competitor has neuromodulation:
- **Mem0**: No mode switching, no arousal-based encoding
- **supermemory.ai**: Basic decay, no chemical state
- **Zep**: No dynamic resource allocation
- **Letta**: No biological mode switching

---

## Test Results

| Module | Tests | Status |
|:-------|:------|:-------|
| Neuromodulation | 21 | ✅ ALL PASSING |
| Neurogenesis | 15 | ✅ ALL PASSING |
| Full suite | 214 | ✅ ALL PASSING |
| OriginBench | HCSI=0.80 | 4/5 ✅ |
| Competitive | 4-0-1 | Origin wins ✅ |

---

## Project Stats (v0.5)

| Metric | Value |
|:-------|:------|
| Source modules | 25 |
| Source lines | ~7,100 |
| Test files | 15 |
| Tests passing | **214** |
| Research docs | **50** |
| Experiments | 20+ (17 passing) |
| HCSI Score | **0.80** |
| Version | **v0.5** |

---

*Both engines represent a significant step toward the user's goal of "exact human-like brain that creates and destroys neurons every day."*
