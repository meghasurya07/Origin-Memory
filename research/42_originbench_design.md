# 42 — OriginBench: The Human-Like Memory Benchmark

**Date**: 2026-09-14
**Stream**: Wave 8 — Benchmark Design
**Type**: RESEARCH + BENCHMARK DESIGN (original)

---

## The Problem with Existing AI Memory Benchmarks

All existing benchmarks (LongMemEval, LoCoMo, MemoryArena, MemBench) treat memory as a **perfect database**. They test:
- Can you find the needle in the haystack? (retrieval accuracy)
- Can you update stored facts? (test-time learning)
- Can you handle contradictions? (conflict resolution)

**What they COMPLETELY miss**: Human-like retrieval dynamics — associative interference, confidence calibration, predictable decay, false memories, serial position effects, and the testing effect.

A system that scores 100% on existing benchmarks is a **perfect database**, not a brain.

---

## OriginBench: 7 Sub-Tasks

### Task 1: The Ebbinghaus Forgetting Curve Test

**Paradigm**: Inject facts over 30 simulated days, test recall at delays.

**Human benchmark**:
- ~42% forgotten after 20 minutes
- ~67-70% forgotten after 24 hours
- ~79% forgotten after 31 days

**Metric**: HUMAN SIMILARITY INDEX — how closely does the AI's forgetting curve match Ebbinghaus?

$$HSI_{Ebbinghaus} = 1 - \frac{1}{N}\sum_{t}\left|R_{AI}(t) - R_{human}(t)\right|$$

**Expected competitor scores**:
| System | Score | Why |
|:-------|:------|:----|
| Mem0 | 0.0 | Never forgets (100% recall forever) |
| Letta | 0.1 | Page-based eviction, not power-law |
| Random | 0.2 | Uniform random loss |
| **Origin Brain** | **0.8+** | **Power-law decay matches Ebbinghaus** |
| Human | 1.0 | Reference |

---

### Task 2: The DRM False Memory Test

**Paradigm**: Present themed word lists (e.g., bed, rest, awake, tired, dream, doze...). Test whether the system "remembers" the critical lure ("sleep").

**Human benchmark**: ~40-55% false recognition rate for critical lures, with HIGH confidence.

**Metric**: Does the AI generate schema-consistent false memories?

$$HSI_{DRM} = \begin{cases} 1.0 & \text{if false recall rate matches human} \\ 0.0 & \text{if zero false recalls (perfect database)} \end{cases}$$

**Expected scores**:
| System | Score | Why |
|:-------|:------|:----|
| Mem0 | 0.0 | Perfect retrieval, no false memories |
| Letta | 0.0 | Perfect retrieval |
| **Origin Brain** | **0.7+** | **Schema filling + reconstruction** |

---

### Task 3: The Serial Position Curve (Murdock 1962)

**Paradigm**: Stream 10-40 items. Test free recall. Measure recall probability by serial position.

**Human benchmark**: U-shaped curve — primacy (first ~3 items) + recency (last ~3 items) recalled best. Middle items worst.

**Metric**: Shape similarity to Murdock's canonical U-curve.

$$HSI_{Serial} = 1 - D_{KL}(P_{AI} \| P_{human})$$

---

### Task 4: The Anderson Fan Effect

**Paradigm**: Teach 1 fact about Person A, 50 facts about Person B.

**Human benchmark**: Retrieval latency for Person B facts is logarithmically slower.

**Metric**: Does retrieval cost scale with association count?

$$HSI_{Fan} = \text{corr}(\log(\text{associations}), \text{retrieval\_latency})$$

---

### Task 5: The Testing Effect

**Paradigm**: Fact X is passively restudied in context. Fact Y is actively queried.

**Human benchmark**: Fact Y (tested) has ~20-40% better retention at delay.

**Metric**: Does active retrieval strengthen memory more than passive exposure?

---

### Task 6: Consolidation Test (Sleep Effects)

**Paradigm**: Encode memories. Run sleep. Test before/after.

**Human benchmark**: Sleep improves recall of important items, forgetting of trivial ones.

**Metric**: Selective consolidation accuracy.

---

### Task 7: Reconstruction Fidelity Test

**Paradigm**: Encode a detailed event. Recall in same vs different context.

**Human benchmark**: Same context → vivid recall. Different context → gist + schema filling.

**Metric**: Context-dependent fidelity gradient.

---

## The Human Cognitive Similarity Index (HCSI)

$$HCSI = \frac{1}{7}\sum_{k=1}^{7} HSI_k$$

A score of 1.0 = perfect human-like behavior.
A score of 0.0 = perfect database (no human-like properties).

**This is Origin AI's competitive moat**: While competitors optimize for RETRIEVAL ACCURACY (which any database can do), we optimize for HUMAN COGNITIVE SIMILARITY.

---

## Implementation Priority

| Task | Difficulty | Our Current Status |
|:-----|:----------|:------------------|
| Ebbinghaus | MEDIUM | DecayEngine already implements power-law |
| DRM | MEDIUM | ReconstructionEngine + SchemaEngine — **ALREADY WORKS** (Exp 2.3) |
| Serial Position | MEDIUM | TCM + working memory capacity |
| Fan Effect | HARD | Need interference scaling |
| Testing Effect | EASY | Track access_count, test retrieval-based strengthening |
| Consolidation | DONE | SleepEngine — **ALREADY PASSING** (Exp 1.4, 2.4) |
| Reconstruction | DONE | ReconstructionEngine — **ALREADY PASSING** (Exp 2.1) |

**3 of 7 already working. 4 to build.**

---

*Sources: Ebbinghaus 1885, Murdock 1962, Roediger & McDermott 1995, Anderson 1974, Roediger & Karpicke 2006*
