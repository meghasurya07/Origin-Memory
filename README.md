# 🧠 Origin Brain — Human-Like Memory for AI Agents

> **The biggest problem in AI is Memory.** AI agents are stateless — they forget everything between sessions. Origin Brain is a cognitive memory system that thinks, forgets, consolidates, and reconstructs memories the way a human brain does.

[![Tests](https://img.shields.io/badge/tests-237%20passing-brightgreen)]()
[![HCSI](https://img.shields.io/badge/HCSI-1.00-gold)]()
[![Version](https://img.shields.io/badge/version-0.5.1-orange)]()
[![License](https://img.shields.io/badge/license-proprietary-red)]()

---

## What Makes This Different

Every AI memory system today (Mem0, supermemory.ai, Zep, Letta) is fundamentally a **database** — they store and retrieve. 

Origin Brain is a **cognitive architecture** — it thinks about memories the way a human does:

| Capability | Vector DBs (Mem0, etc.) | Origin Brain |
|:-----------|:----------------------|:-------------|
| **Forgetting curves** | ❌ Never forgets | ✅ Ebbinghaus-matched (HSI=0.806) |
| **False memories** | ❌ Impossible | ✅ Schema-consistent confabulation |
| **Context-dependent recall** | ❌ Same output always | ✅ VERBATIM in same context, GIST in different |
| **Pattern completion** | ❌ Needs exact query | ✅ 12.5% cue → correct memory |
| **Working memory limits** | ❌ Infinite capacity | ✅ 5-7 items (Cowan 2010) |
| **Testing effect** | ❌ Static storage | ✅ Retrieval strengthens memories |
| **Sleep consolidation** | ❌ None | ✅ 3-phase SWS/REM/Pruning |
| **Neurogenesis** | ❌ None | ✅ Dynamic neuron creation/destruction |
| **Neuromodulation** | ❌ None | ✅ ACh/DA/NE mode switching |

**Competitive Benchmark: Origin Brain wins 4-0-1 against vector database memory systems.**

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Origin Brain                      │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Prediction│  │ Temporal │  │ Neuromodulation  │  │
│  │ Engine   │  │ Context  │  │ (ACh/DA/NE)      │  │
│  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │              │                  │            │
│  ┌────▼──────────────▼──────────────────▼────────┐  │
│  │              Hippocampal Engine                │  │
│  │  (Encode → Pattern Separate → Store → Index)  │  │
│  └────┬──────────────┬──────────────────┬────────┘  │
│       │              │                  │            │
│  ┌────▼─────┐  ┌─────▼─────┐  ┌────────▼────────┐  │
│  │ Decay    │  │ Sleep     │  │ Reconstruction   │  │
│  │ Engine   │  │ Engine    │  │ Engine           │  │
│  │(forgetting)│ │(consolidation)│ │(generative recall)│  │
│  └──────────┘  └───────────┘  └──────────────────┘  │
│                                                      │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ Working  │  │ Pattern   │  │ Neurogenesis     │  │
│  │ Memory   │  │ Completion│  │ (birth/death)    │  │
│  └──────────┘  └───────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### The 6 Pillars

1. **Prediction Error Encoding** — Only store surprises (Pillar 1)
2. **Temporal Context Binding** — TCM drifting context (Pillar 2)
3. **Forgetting as Regularization** — Power-law decay + pruning (Pillar 3)
4. **Sleep Consolidation** — 3-phase SWS/REM/Pruning (Pillar 4)
5. **Reconstruction** — Generative context-dependent recall (Pillar 5)
6. **Pattern Completion** — Modern Hopfield attractor = Attention (Pillar 6)

---

## Quick Start

```python
from origin_brain import Brain, BrainConfig

# Create a brain for your agent
brain = Brain(config=BrainConfig(agent_id="my_agent"))

# Encode memories
brain.encode("Met Sarah at the coffee shop, she mentioned her new startup")
brain.encode("The quarterly budget meeting discussed a 15% increase")

# Recall with context-dependent reconstruction  
results = brain.recall("What happened at the coffee shop?")
for result in results.results:
    print(f"[{result.score:.2f}] {result.memory.content}")

# Reconstructive recall (different fidelity based on context)
reconstructed = brain.reconstruct_recall("budget meeting")
for r in reconstructed:
    print(f"[{r.fidelity.value}] {r.reconstructed_content}")

# Sleep consolidation (run daily)
sleep_report = brain.sleep()
print(f"Consolidated: {sleep_report.memories_consolidated}")
print(f"Pruned: {sleep_report.memories_pruned}")
```

---

## Project Stats

| Metric | Value |
|:-------|:------|
| Source modules | 25 |
| Source lines | 7,500+ |
| Test files | 16 |
| Tests passing | **237** |
| Research documents | **52** |
| Experiments | 20+ (all passing) |
| HCSI Score | **1.00** (5/5 benchmarks) |
| Version | **v0.5.1** |

---

## Benchmarks

### OriginBench — Human Cognitive Similarity Index

| Test | Score | Status |
|:-----|:------|:-------|
| Ebbinghaus Forgetting Curve | HSI = 0.806 | ✅ PASS |
| DRM False Memory | Schema-filled reconstructions | ✅ PASS |
| Testing Effect | stability 12.57 vs 3.64 | ✅ PASS |
| Working Memory Capacity | 5-7 items, priority displacement | ✅ PASS |
| Serial Position Effect | Recency 0.720 > Middle 0.706 | ✅ PASS |
| **HCSI** | **1.00** | **5/5** |

### Competitive Benchmark (vs VectorDB/Mem0-like)

| Test | Origin Brain | Competitor | Winner |
|:-----|:-----------|:-----------|:-------|
| Forgetting Curve Match | HSI=0.638 | HSI=0.362 | **ORIGIN** |
| Context Sensitivity | VERBATIM→GIST | static | **ORIGIN** |
| Working Memory Limits | 7 items | 20 items | **ORIGIN** |
| Pattern Completion | ✅ | ✅ | TIE |
| Testing Effect | strengthened | static | **ORIGIN** |
| **Overall** | | | **4-0-1** |

---

## Research

51 research documents (~440 KB) covering neuroscience, mathematics, AI, competitive analysis, and original thinking. See [`research/README.md`](research/README.md).

---

## Built by [Origin AI](https://originai.in)

*Focused on frontier intelligence and products.*
