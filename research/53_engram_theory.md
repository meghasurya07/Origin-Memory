# 53 — Engram Theory: Memory Traces in Origin Brain

**Date**: 2026-09-15
**Type**: NEUROSCIENCE + IMPLEMENTATION
**Status**: Implemented in v0.5.1

---

## What is an Engram?

An **engram** is the physical trace of a memory in the brain — the specific
ensemble of neurons that were active during encoding and become necessary
and sufficient for retrieval.

Key researchers:
- **Richard Semon (1904)**: Coined the term "engram"
- **Karl Lashley (1950)**: Failed to find engrams (mass action principle)
- **Sheena Josselyn (2014-2024)**: CREB-based allocation, excitability competition
- **Susumu Tonegawa (2012-2024)**: Optogenetic reactivation, engram complexes

---

## The Excitability Competition Model

### How the Brain Allocates Engrams (Josselyn Lab)

When a new memory forms, neurons **compete** for inclusion:

1. Each neuron has an intrinsic excitability level (set by CREB)
2. The most excitable neurons "win" the competition
3. These neurons become the engram (typically 2-5% of available neurons)
4. After allocation, the winning neurons become MORE excitable
5. This creates a bias for overlapping engrams → linked memories

$$P(\text{allocation}) \propto \exp(\text{excitability}_i / \tau)$$

### Our Implementation

```python
# Sort cells by excitability → winner-take-all
available_cells = sorted(cells, key=lambda c: c.excitability, reverse=True)
allocated = available_cells[:cells_per_engram]

# Boost winners' excitability (CREB positive feedback)
for cell in allocated:
    cell.excitability += excitability_boost
```

---

## Engram Lifecycle

| State | Duration | What It Means | Can Retrieve? |
|:------|:---------|:-------------|:-------------|
| **ACTIVE** | 0-6 hours | Just formed/reactivated | ✅ Easily |
| **STABLE** | 6h - 30 days | Consolidated | ✅ With cue |
| **SILENT** | 30-90 days | Exists but inaccessible | ❌ (without intervention) |
| **DEGRADED** | 90+ days | Partially lost | ❌ (errors likely) |
| **DISSOLVED** | Eventually | Fully lost | ❌ Permanently |

### Silent Engrams: The Alzheimer's Insight

Tonegawa's lab showed that in early Alzheimer's, engrams still EXIST
but can't be naturally reactivated. Optogenetic stimulation can
restore the memory.

**Our implementation**: Silent engrams can be reactivated through
`engine.reactivate(memory_id)`, restoring them to ACTIVE state.

---

## Engram Overlap: The Physical Basis of Association

When the cell pool is limited, engrams SHARE cells:

```
Memory A: cells {1, 2, 3, 4, 5}
Memory B: cells {3, 4, 5, 6, 7}   ← Overlap: {3, 4, 5}
```

**Overlap fraction**: 3/5 = 0.60

**Consequence**: Recalling Memory A partially activates Memory B's engram,
creating an associative link. This is why you think of "coffee" when you
remember the meeting at the coffee shop.

**Our implementation**: `find_linked_memories()` detects overlapping engrams
and returns (memory_id, overlap_fraction) pairs.

---

## Integration with Brain Pipeline

### Encoding (Step 13)
```
Content → Hippocampus → Prediction → Neuromodulation → ... → Engram Allocation
```
Every encoded memory gets an engram trace. The engram tracks WHICH cells
were allocated, enabling overlap detection and state tracking.

### Retrieval (Step 5d)
```
Query → Retrieve → Rank → RIF → Engram Reactivation → Plasticity → Return
```
Retrieved memories get their engrams reactivated, boosting strength and
cell excitability. This implements the testing effect at the engram level.

---

## Connection to Other Systems

| System | Engram Connection |
|:-------|:-----------------|
| **Neurogenesis** | New neurons can be allocated to engrams (integration) |
| **Neuromodulation** | DA/NE boost excitability → bias allocation |
| **Synaptic Plasticity** | STDP operates ON connections between engram cells |
| **Sleep Consolidation** | Consolidation transitions engrams ACTIVE → STABLE |
| **Pattern Completion** | Partial cue activates full engram → pattern completion |

---

## Statistics

v0.5.1 Engram Engine:
- **~340 lines** of Python
- **20 tests** all passing
- Excitability-based allocation (CREB model)
- 5 lifecycle states
- Overlap detection for associative linking
- Integrated into Brain encode/recall pipeline

---

*"Memory is not a passive recording. It is an active construction by
competing neural ensembles." — Josselyn & Tonegawa, 2020*
