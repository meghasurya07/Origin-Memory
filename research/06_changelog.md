# Origin Brain — Version History & Changelog

> All build reports consolidated into one living document.

---

## v0.6.0 (2026-09-15) — Engram + Cognitive Map + SDK

**283 tests | HCSI 1.00 | 30 modules | ~9,500 lines**

### New Modules
| Module | Lines | Tests | What |
|:-------|:------|:------|:-----|
| `engram.py` | ~340 | 20 | Memory trace tracking (CREB excitability, lifecycle states, overlap) |
| `cognitive_map.py` | ~300 | 13 | Successor representations for memory navigation |
| `client.py` | ~260 | 13 | Python SDK (`OriginBrainClient`) |

### Changes
- **Engram Engine**: Excitability-based cell allocation (Josselyn model), 5 lifecycle states (active→dissolved), overlap detection for associative linking, integrated into Brain encode/recall
- **Cognitive Map**: Successor representation (Stachenfeld 2017), transition recording during recall, full SR update during sleep, distance/navigation queries
- **Python SDK**: `client.remember()`, `client.recall()`, `client.recall_with_reconstruction()`, `client.sleep()`, `client.stats()`, `client.save()`, context management
- **REST API v2**: Versioned `/v1/` endpoints, per-agent SQLite persistence, reconstruct/sleep/save endpoints
- **Version bumped to 0.6.0**

---

## v0.5.1 (2026-09-15) — HCSI = 1.00 🎉

**250 tests | HCSI 1.00 (5/5) | 26 modules | ~7,800 lines**

### The HCSI 1.00 Milestone
- **Serial Position Effect: NOW PASSING** (was the last failing benchmark)
- Fix: Rebalanced retrieval weights to 0.45 semantic + 0.30 recency + 0.15 salience + 0.10 primacy
- Added primacy rehearsal bonus for first 3 items
- Encoding-order recency increased to 60% of blend

### New
- Synaptic plasticity integrated into Brain recall (co-recalled memories strengthen)
- GitHub Actions CI/CD configured (Python 3.11/3.12/3.13)
- Tag v0.5.0 and v0.5.1 created

---

## v0.5.0 (2026-09-15) — GitHub + Neuromodulation + Persistence

**237 tests | HCSI 0.80 | 25 modules | ~7,200 lines**

### New Modules
| Module | What |
|:-------|:-----|
| `synaptic_plasticity.py` | STDP + Hebbian + BCM metaplasticity |
| `neuromodulation.py` | ACh/DA/NE mode switching (ENCODING/ALERT/CONSOLIDATION) |
| `neurogenesis.py` | Dynamic neuron creation/destruction lifecycle |
| `storage.py` | SQLite persistence backend |

### Changes
- Neuromodulation integrated into Brain encode pipeline (novelty → ACh, surprise → DA, arousal → NE)
- SQLite persistence: auto-persist on encode, auto-load on init
- GitHub repo created: `meghasurya07/Origin-Memory`
- 4 commits pushed, CI/CD configured

---

## v0.4.1 (2026-09-14) — OriginBench

**214 tests | HCSI 0.80 | 20 modules | ~6,000 lines**

### New
- OriginBench v0.1: 5 cognitive benchmarks (4/5 passing)
- Competitive benchmark: Origin Brain vs VectorDB (4-0-1 WIN)
- TF-IDF Embedding Engine (replaced hash-based embeddings)
- Interference module with RIF (Retrieval-Induced Forgetting)

---

## v0.4.0 (2026-09-14) — Consolidation + Metacognition

**~180 tests | 16 modules | ~5,000 lines**

### New Modules
- Sleep consolidation engine (SWS + REM phases)
- Prospective memory engine (future intentions)
- Reconsolidation engine (memory updating)
- Metamemory engine (confidence assessment)

---

## v0.3.0 (2026-09-13) — Reconstruction Engine

**~120 tests | 12 modules | ~3,500 lines**

### Breakthrough
- Pattern Completion Engine (CA3 pattern completion from partial cues)
- Reconstructive Recall (generative, context-dependent memory)
- Schema Engine (knowledge structures influence reconstruction)
- Emotional Tagger (valence/arousal influence on encoding)

---

## v0.2.0 (2026-09-13) — Core Memory

**~60 tests | 8 modules | ~2,000 lines**

### Foundation
- Brain class with encode/recall pipeline
- Hippocampus with temporal context model
- Working Memory with capacity limits (4-7 items)
- Forgetting Engine (Ebbinghaus curve)
- Memory Router (type classification)

---

## v0.1.0 (2026-09-12) — Initial Research

**Models + basic architecture defined**

### Foundation
- Memory models (EpisodicMemory, SemanticMemory, ProceduralMemory)
- BrainConfig, MemoryTier, MemoryType
- Initial research documents (1-11)
- Project structure established
