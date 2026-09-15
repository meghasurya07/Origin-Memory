# Origin Brain — Benchmarks & Experiments

## 1. OriginBench v0.1 (design, 5 benchmarks, HCSI = 1.00)

**The Problem with Existing Benchmarks:** All existing benchmarks (LongMemEval, LoCoMo, BEAM) treat memory as a perfect database, focusing on retrieval accuracy, test-time learning, and conflict resolution. They completely miss human-like retrieval dynamics such as associative interference, predictable decay, false memories, and the testing effect. 

**OriginBench Design (Human Cognitive Similarity Index - HCSI):**
OriginBench tests the brain-inspired dimensions of memory. A score of HCSI = 1.00 represents perfect human-like behavior, while HCSI = 0.0 represents a perfect database.

Our core sub-tasks include:
1. **The Ebbinghaus Forgetting Curve Test**: Matches AI forgetting rate to human power-law decay.
2. **The DRM False Memory Test**: Tests schema-consistent false memory generation.
3. **The Serial Position Curve Test**: Measures U-shaped recall curve (primacy and recency effects).
4. **The Anderson Fan Effect**: Tests retrieval latency scaling with association count.
5. **The Testing Effect**: Evaluates if active retrieval strengthens memory more than passive exposure.
6. **Consolidation Test**: Tests selective sleep consolidation (retaining important, pruning trivial).
7. **Reconstruction Fidelity Test**: Measures context-dependent fidelity gradient.

*Our competitive moat is optimizing for Human Cognitive Similarity over pure retrieval accuracy.*

## 2. Experiment Results (01-04 detailed results)

Across our testing suites, Origin Brain achieves a 93.75% pass rate (15/16 tests passing).

**Experiment 01: Memory Properties (6/6 Passed)**
- **Prediction Error Selectivity**: Novel content triggers maximum surprise (1.0) and encoding, with a selectivity ratio of 1.38. (Needs structural novelty calibration).
- **Temporal Contiguity**: Strong contiguity confirmed. Adjacent memories have context similarity of 0.94, dropping to 0.41 at distance 3, matching human free recall studies.
- **Context Reinstatement**: Reinstating context makes related memories 73% more accessible (Mental Time Travel effect confirmed).
- **Sleep Consolidation**: Perfect selective consolidation (5/5 important memories consolidated to gist; 0/10 noise consolidated).
- **Full Brain Pipeline**: End-to-end functionality confirmed (prediction error → temporal context binding → sleep consolidation).
- **Performance**: Recall is fast (19ms, 52/sec), but encode is currently slow (455ms, 2/sec) due to O(dim²) Hebbian matrix updates.

**Experiment 02: Reconstruction (5/5 Passed)**
- Reconstruction and pattern completion function as designed.

**Experiment 03: OriginBench (4/5 Passed)**
- Implementing a TF-IDF embedding upgrade was a major breakthrough, increasing HCSI from 0.60 to 0.80.
- **Testing Effect**: Passed. TF-IDF fixes discrimination issues, allowing proper measurement of access and stability.
- **Serial Position**: Failed. Due to pure content-based TF-IDF similarity, temporal recency weight (0.2) is insufficient. Needs blended temporal recency and semantic similarity.

## 3. Competitive Benchmark (Origin Brain vs VectorDB, 4-0-1)

**Experiment 04: Competitive Matchup**

| Benchmark | Origin Brain | Standard VectorDB | Winner |
|:----------|:-----------|:-----------|:-------|
| Forgetting Curve Match | HSI = 0.638 | HSI = 0.362 | **ORIGIN** |
| Context-Dependent Recall | VERBATIM to GIST | Exact to Exact | **ORIGIN** |
| Working Memory Capacity | 7 items (human-like) | 20 items (stores all) | **ORIGIN** |
| Pattern Completion | 12.5% cue works | Keyword match | TIE |
| Testing Effect | Stability strengthened | No change | **ORIGIN** |
| **OVERALL** | | | **ORIGIN 4-0-1** |

## 4. Honest Assessment (what works, what doesn't, known limitations)

**What Works Well:**
- **Reconstruction Engine**: Produces different outputs depending on context. No competitor does this.
- **Pattern Completion**: Modern Hopfield networks correctly retrieve from 20% partial cues.
- **Forgetting Curve**: Achieves HSI = 0.806 against human Ebbinghaus data.
- **Working Memory & Sleep Consolidation**: Biologically grounded priority displacement and selective pruning.

**What Doesn't Work Yet / Bottlenecks:**
1. **Similarity Search**: Current TF-IDF and Jaccard similarities cannot distinguish semantically different but lexically similar memories. We need true neural embeddings (e.g., BGE-M3, ada-002).
2. **Serial Position Effect**: Temporal context doesn't differentiate enough between early/late memories in short sequences.
3. **True Generative Reconstruction**: Current reconstructions concatenate templates. We need an LLM to produce natural language narratives and true generative recall.
4. **Scale**: All experiments are on <100 memories. Performance at 10K+ memories is unknown.

**The Missing Pieces:** 
1. Neural Embeddings for Semantic Understanding.
2. LLM-Powered Reconstruction.
3. Concept Space Navigation (embedding memories in a multi-dimensional metric space).

## 5. Industry Benchmarks (LoCoMo, LongMemEval — status and plan)

To achieve competitive parity, we are pursuing a two-track strategy: defining our own game (OriginBench) and playing theirs (Industry Standards).

**1. LoCoMo (Long Conversation Memory)**
- The de facto standard testing memory retention across multi-session dialogues.
- Categories: Single-hop recall, Multi-hop reasoning, Temporal awareness, Open-domain synthesis.
- *Status:* Mem0 scores ~92.5%. We need LLM integration and multi-session support to run this.

**2. LongMemEval**
- Focuses on operational functions: fact updates, conflict resolution, and abstention logic.
- *Status:* Mem0 scores ~94.4%. We require LLM integration and abstention logic.

**3. BEAM & LoCoMo-Plus**
- BEAM tests holistic multi-agent memory at high scale. 
- LoCoMo-Plus (Emerging) focuses on cognitive memory and non-semantic cue retrieval.

*Strategy:* We expect an initial score of ~60-70% on LoCoMo without an LLM. After neural embeddings and LLM integration, we target 80-90%. 

## 6. Future Benchmark Plans

1. **Open-Source OriginBench**: Establish it as the standard for brain-inspired memory evaluation by publishing the dataset and evaluation code.
2. **Implement Neural Embeddings (v0.5 Roadmap)**: Replace TF-IDF with actual vector embeddings to fix the similarity search bottleneck and enable true semantic navigation.
3. **LLM Reconstruction Engine**: Deploy an LLM for genuine narrative reconstruction to achieve Level 3 Reconstructive Memory.
4. **Address Scale**: Optimize TCM Hebbian matrix updates (using NumPy/Torch and reduced dimensions) to increase encoding throughput past 2/sec and prepare for 1M+ memory scaling tests.
5. **Paper Publication**: Publish "Reconstruction Is All You Need" demonstrating human memory generative dynamics vs existing AI database memory.
