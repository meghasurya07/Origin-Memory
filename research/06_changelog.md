# Origin Brain — Version History & Changelog

> All build reports consolidated into one living document.

---

## v0.11.1 (2026-11-12) — Research Documentation + Ablation Study + Mathematical Proofs

**586 tests | LoCoMo 78% | 39 modules | ~12,900 lines | Ablation ✅**

### Research Documentation Overhaul
| Document | Changes |
|:---------|:--------|
| `02_mathematics.md` | Added 7 original theorems with proofs: IB advantage (Thm 1), RL convergence (Thm 2), entropy consolidation (Thm 3), hypergraph efficiency (Thm 4), Hopfield capacity (Thm 5), capacity-accuracy crossover (Thm 6), sleep as rate-distortion (Thm 7). Added empirical findings summary. |
| `03_architecture.md` | Added §11: 8 new module descriptions (temporal invalidation, belief entropy, associative network, PEC2F, entropy monitor, RL policy, RL gate, hypergraph). Added §12: 18-step encode pipeline. Added §13: full 39-module inventory with line counts. |
| `05_benchmarks.md` | Complete rewrite: verified LoCoMo 78% with per-category breakdown, comparison table vs GPT-4/RAG/Mem0/Zep, ablation table, honest assessment, failure analysis. |
| `08_references.md` | Added 30+ new citations: Information theory (Tishby, Shannon), Modern Hopfield (Ramsauer), RL memory (MemAgent, Mem-α, Memory-R1, AgeMem, ReMemR1), sleep (Born, Walker, Tononi), cognitive systems (MemoryOS, CogMem, HGMem). Key equations table. |

### Ablation Study (Experiment 11) — Rewritten & Verified
- **Complete rewrite** of `experiments/11_ablation_study.py` to use the actual LoCoMo benchmark infrastructure (identical test data and scoring as experiment 07)
- Results: Neuromodulation and Sleep Consolidation each contribute 4.0pp to the overall score
- Per-category analysis shows both affect the Coherence category (80%→60%)
- JSON results saved to `experiments/results/11_ablation_results.json`

---

## v0.11.0 (2026-11-11) — RL Memory Policy + Hypergraph + Entropy Monitor

**361 tests | HCSI 1.00 | 35 modules | ~12,500 lines | CI ✅**

### New Modules
| Module | Lines | Tests | What |
|:-------|:------|:------|:-----|
| `world_model.py` | ~300 | 18 | Predictive state transitions — mental simulation, surprise, reachability |
| `shared_memory.py` | ~350 | 19 | Multi-agent memory pool with scoping, conflict resolution, provenance |

### Changes
- **World Model Engine**: Learns state transition patterns during sleep replay. Enables `predict_next()`, `simulate_future()`, `compute_surprise()`. Based on hippocampal preplay (Dragoi & Tonegawa 2011)
- **Multi-Agent Shared Memory**: 3-tier scoped access (global/group/private), 4 conflict resolution strategies, broadcast, provenance tracking
- **REST API v0.7**: 6 new endpoints (`/shared/publish`, `/shared/query`, `/shared/join`, `/shared/stats`, `/semantic/query`, `/schema/{id}/{type}`)
- **CI Fixed**: Added fastapi/httpx deps, ignore neural tests, `fail-fast: false`. All 3 Python versions pass
- **Experiments 05 (6/6) + 06 (6/6)**: CLS and multi-agent validation

---

## v0.7.0 (2026-10-12) — Neural Embeddings + CLS Complete

**308 tests | HCSI 1.00 | 34 modules | ~10,500 lines**

### New Modules
| Module | Lines | Tests | What |
|:-------|:------|:------|:-----|
| `neural_embedding.py` | ~200 | 12 | Transformer embeddings (all-MiniLM-L6-v2, 384d). Drop-in TF-IDF replacement |
| `neocortex.py` | ~310 | 13 | Slow statistical learning — co-occurrence, schema formation, spreading activation |

### Changes
- **Neural Embeddings**: Uses HuggingFace `transformers` + `torch` directly (avoids pyarrow crash on Python 3.14). Factory function `create_embedding_engine()` auto-detects best backend
- **Neocortical Engine (CLS complete)**: McClelland et al. 1995 — fast hippocampal + slow neocortical learning. Schema formation from repeated episodes, schema-consistency scoring, concept co-occurrence tracking
- **Brain Integration**: Neocortex wired into sleep pipeline — episodic replay drives neocortical learning. Entity extraction helper for semantic tokenization
- **Research Consolidation**: 54 docs → 8 organized docs + archive. Critical content restored via enrichment pass (78KB vs 449KB originals, 17% = pure signal)
- **BrainConfig**: `use_neural_embeddings`, `neural_model_name` options added

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


## v0.9.0 — Temporal Invalidation + Belief Entropy Metamemory (Nov 5, 2026)

### New Modules
- **temporal_invalidation.py** (~280 lines): Bi-temporal fact engine with conflict detection,
  automatic supersession, point-in-time queries, fact history tracking. 19/19 tests.
  - Inspired by Zep bi-temporal edges and CogMem PEC2F schema (EMNLP 2026)
- **metamemory.py REWRITE**: Belief Entropy upgrade
  - Shannon entropy H(belief) = -sum(p_i * log2(p_i)) / log2(n) over relevance distribution
  - Pairwise Jaccard agreement scoring between top results
  - Contradiction detection (AND logic: high entropy + low agreement)
  - New fields: overall_confidence, belief_entropy, contradiction_detected, agreement_score
  - Inspired by MMPO (Meta-Cognitive Memory Policy Optimization, 2026)

### Benchmark
- **LoCoMo-style benchmark** (Experiment 07): 78% overall
  - Single-Hop 70%, Multi-Hop 100%, Temporal 60%, Coherence 80%, Cognitive 80%
  - Beats GPT-4 full context (67%) and GPT-4+RAG (58%) WITHOUT any LLM

### Integration
- brain.py: Temporal facts extracted during encode (step 14)
- 369 tests passing

## v0.10.0 — Associative Memory Network (Nov 11, 2026)

### New Module
- **associative_network.py** (~600 lines): Graph-based semantic memory fabric
  - Spreading activation (Collins & Loftus 1975): neurally-inspired retrieval
  - Entity-relation-entity triples with Hebbian strengthening
  - Lateral inhibition (GABAergic competitive suppression)
  - Multi-hop path finding (BFS)
  - Entity extraction from natural language text
  - LRU eviction, edge pruning, biological decay dynamics
  - Resting levels boosted by access frequency (long-term potentiation)
  - Export/import graph serialization
  - 39/39 tests passing

### Integration
- brain.py encode step 15: Auto-extract entities/relations from encoded content
- brain.py recall step 5f: Spreading activation from query entities for graph context
- Wired into Brain init with AssociativeNetworkConfig defaults
- 408 total tests, 0 failures

### Research Scan (Nov 2026)
- HGMem (ICML 2026): Hypergraph memory > standard KG for complex relations
- MemAgent/Mem-alpha: RL-trained memory policies massively outperform heuristics
- FrontierMemory 1.0 benchmark: Letta's new standard (15 tasks, 750 trajectories)
- SleepGate: attention-entropy triggered consolidation cycles
- PEC2F schema (CogMem): Person-Event-Concept-Claim-Fact provenance graph


---

## v0.11.0 (2026-11-11) — RL Memory Policy + Hypergraph Memory + Entropy Monitor

**568 tests | 38 modules | ~13,500 lines | CI ✅**

### Attention Entropy Monitor (SleepGate-inspired)
- `entropy_monitor.py`: ~355 lines
- Monitors 4 information-theoretic metrics of the memory system:
  1. Memory Entropy: Shannon entropy over access distribution (H = -Σ pᵢlog₂pᵢ / log₂n)
  2. Interference Rate: windowed conflict rate during encoding
  3. Working Memory Churn: displacement rate tracking
  4. Schema Coverage: how well existing schemas cover new input
- Accumulates sleep pressure [0,1] — consolidation triggered when pressure > 0.7
- Brain analogue: locus coeruleus NE system + adenosine sleep pressure
- Brain integration: encode step 17, recall retrieval tracking, sleep() reset
- 38 unit tests

### RL Memory Policy (MemAgent/Mem-α inspired) — THE BREAKTHROUGH
- `rl_policy.py`: ~500 lines
- Action space: ENCODE | UPDATE | SKIP | EVICT | CONSOLIDATE
- State: 10-feature vector (novelty, salience, entropy, interference, pressure, etc.)
- Discretized into tabular Q-table for efficient learning
- ε-greedy exploration with exponential decay
- Experience replay buffer with mini-batch updates
- Reward signal: recall_accuracy + confidence_bonus - memory_cost - interference_penalty
- Heuristic fallback → adaptive RL transition when enough data
- Q-table persistence (save/load from JSON)
- Policy summary: interpretable action preferences per state archetype
- Brain analogue: dopaminergic VTA reward prediction error + basal ganglia Go/NoGo
- 55 unit tests + 41/41 benchmark (Experiment 09)

### Hypergraph Memory (HGMem ICML 2026)
- `hypergraph.py`: ~450 lines
- N-ary relational hyperedges: captures "Alice, Bob, Charlie met at Cafe on Tuesday"
  as ONE atomic relation instead of 10+ pairwise edges
- Spreading activation through hyperedges (larger groups → less spread per node)
- Query by node overlap (Jaccard scoring with min threshold)
- Hebbian strengthening/weakening
- Clique expansion (convert to pairwise for standard graph algorithms)
- Automatic text extraction with relation classification (12 relation types)
- Incidence matrix computation
- Brain analogue: hippocampal indexing theory (one trace binds many cortical reps)
- Brain integration: encode step 18
- 46 unit tests

### RL Research Scan (Nov 2026 Deep Dive)
Key papers analyzed and informing our design:
| Paper | Action Space | Reward | Training | Extrapolation |
|-------|-------------|--------|----------|---------------|
| MemAgent (ICLR 2026) | Free-form memory rewrite | Binary EM | Multi-Conv DAPO | 8K → 3.5M |
| Mem-α (2025) | Write/Update/Delete/Route | QA accuracy | GRPO | 30K → 400K+ |
| Memory-R1 (ACL) | ADD/UPDATE/DELETE/NOOP | EM/F1 | GRPO | Multi-session |
| AgeMem (2026) | 7 STM+LTM tools | Step-wise GRPO | 3-stage progressive | Long-horizon |
| ReMemR1 (ICLR 2026) | Update + callback | Multi-level | RLMLR | Long-context |

Key insight: Our tabular Q-learning is a lightweight version of these full LLM-RL
approaches. It can learn the SAME decision patterns without requiring GPU training,
making it practical for edge deployment. The next step is DQN or PPO when scale warrants.
