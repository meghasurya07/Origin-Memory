# 09 — Benchmarks & Evaluation

## Overview

To build a credible AI memory system, Origin AI must rigorously evaluate its performance against established benchmarks and clearly demonstrate superiority over existing approaches. This document covers the key benchmarks, evaluation methodology, and proposed metrics.

---

## 1. Industry-Standard Benchmarks

### 1.1 LoCoMo (Long Conversation Memory)

The **de facto standard** for evaluating long-term conversational memory in AI agents.

| Attribute | Details |
|:----------|:--------|
| **Paper** | "Evaluating Very Long-Term Conversational Memory of LLM Agents" (Maharana et al., ACL 2024) |
| **What it tests** | Memory retention across multi-session, multi-turn conversations |
| **Scale** | Dozens of sessions, hundreds of conversation turns |
| **Categories** | Single-hop recall, Multi-hop reasoning, Temporal awareness, Open-domain synthesis |

#### Test Categories

| Category | Description | Example |
|:---------|:-----------|:--------|
| **Single-hop Recall** | Retrieve a specific fact stated at a single point | "What restaurant did the user mention in session 5?" |
| **Multi-hop Reasoning** | Connect facts across different sessions | "Given X said in session 3 and Y said in session 7, what follows?" |
| **Temporal Awareness** | Understand sequence and recency | "Which preference was stated more recently?" |
| **Open-domain Synthesis** | Summarize or reason across broad context | "What are the user's main interests based on all conversations?" |

#### Evaluation Method

- **LLM-as-a-Judge**: GPT-4/Claude scores responses on relevance, accuracy, and completeness
- **Human evaluation**: Gold-standard human ratings for a subset
- Scores: 0-1 scale per category, with aggregate F1 and accuracy

### 1.2 LongMemEval

| Attribute | Details |
|:----------|:--------|
| **Focus** | Evaluating operational memory functions: retrieval, update, conflict resolution |
| **Key feature** | Tests how well agents handle **contradictory information** and memory updates |
| **Categories** | Fact retrieval, fact update, temporal reasoning, conflict resolution |

### 1.3 BEAM (Benchmark for Evaluating Agent Memory)

| Attribute | Details |
|:----------|:--------|
| **Focus** | Holistic evaluation of agent memory across different interaction modalities |
| **Key feature** | Tests memory across multiple agents and shared memory scenarios |

### 1.4 LoCoMo-Plus (Emerging, 2025-2026)

| Attribute | Details |
|:----------|:--------|
| **Focus** | "Cognitive memory" — queries that are **not semantically similar** to the original cue |
| **Key feature** | Tests latent or inferred memory, not just keyword-based retrieval |
| **Why it matters** | This is where brain-inspired systems should excel — human memory is associative, not keyword-based |

---

## 2. Evaluation Dimensions for Origin AI

Beyond standard benchmarks, Origin AI's memory system should be evaluated on dimensions unique to its brain-inspired architecture:

### 2.1 Standard Memory Metrics

| Metric | Description | Target |
|:-------|:-----------|:-------|
| **Recall@K** | Fraction of relevant memories retrieved in top-K results | > 0.90 |
| **Precision@K** | Fraction of retrieved memories that are relevant | > 0.85 |
| **F1 Score** | Harmonic mean of precision and recall | > 0.87 |
| **Temporal Accuracy** | Correct ordering of events in memory | > 0.95 |
| **Contradiction Detection** | Ability to identify and resolve conflicting memories | > 0.90 |

### 2.2 Brain-Inspired Metrics (Novel)

| Metric | Description | Brain Analogue | Target |
|:-------|:-----------|:--------------|:-------|
| **Consolidation Fidelity** | How accurately episodic memories are transformed into semantic knowledge | Systems consolidation quality | > 0.85 |
| **Forgetting Precision** | Does the system forget the right things? (Low-importance → forgotten; High-importance → retained) | Selective forgetting | > 0.90 |
| **Pattern Completion** | Can the system reconstruct a full memory from a partial or noisy cue? | CA3 pattern completion | > 0.80 |
| **Pattern Separation** | Can the system distinguish between similar but distinct memories? | DG pattern separation | > 0.90 |
| **Interference Resistance** | Does new learning degrade old memories? | Catastrophic forgetting resistance | < 5% degradation |
| **Reconsolidation Accuracy** | When a memory is retrieved and updated, is the update correct? | Memory reconsolidation | > 0.90 |
| **Spaced Repetition Effect** | Do repeatedly accessed memories become more stable? | Stability growth with retrieval | Monotonic increase |
| **Semantic Drift Accuracy** | Do episodic memories correctly evolve toward semantic representations? | Episodic → semantic transformation | Gist preserved, noise removed |

### 2.3 Operational Metrics

| Metric | Description | Target |
|:-------|:-----------|:-------|
| **Encoding Latency** | Time to store a new memory | < 50ms (p99) |
| **Retrieval Latency** | Time to retrieve relevant memories | < 100ms (p99) |
| **Consolidation Throughput** | Memories consolidated per minute | > 1000/min |
| **Storage Efficiency** | Bytes per memory (after consolidation) | < 1KB average |
| **Token Savings** | Reduction in context tokens via memory injection vs. raw history | > 70% reduction |
| **Cost per Memory** | Infrastructure cost per stored memory | < \$0.0001 |

---

## 3. Evaluation Methodology

### 3.1 Test Corpus Construction

```
1. Generate synthetic multi-session conversations
   - 50+ sessions per conversation
   - 500+ total turns
   - Include: fact statements, preference changes, temporal events,
     contradictions, emotional moments, routine interactions

2. Create ground-truth annotations
   - Mark which facts should be remembered
   - Mark which facts should be forgotten (trivial)
   - Mark temporal ordering
   - Mark contradictions and their resolutions
   - Rate importance (1-5 scale)

3. Design test queries across all categories
   - Single-hop recall (200 queries)
   - Multi-hop reasoning (100 queries)
   - Temporal awareness (100 queries)
   - Open-domain synthesis (50 queries)
   - Pattern completion (100 queries — novel category)
   - Forgetting verification (100 queries — novel category)
```

### 3.2 Baselines for Comparison

| Baseline | Description |
|:---------|:-----------|
| **No Memory** | Raw LLM with no memory (context window only) |
| **Full History** | Entire conversation history stuffed into context |
| **RAG-Only** | Standard vector database retrieval |
| **Mem0** | Industry-leading memory API |
| **Letta/MemGPT** | Agent-managed memory hierarchy |
| **Origin AI** | Our brain-inspired system |

### 3.3 Evaluation Pipeline

```python
def evaluate_memory_system(system, test_corpus, test_queries):
    results = {}
    
    # Phase 1: Feed conversation history
    for session in test_corpus.sessions:
        for turn in session.turns:
            system.process_turn(turn)
        
        # Allow consolidation between sessions
        system.consolidate()
    
    # Phase 2: Run test queries
    for query in test_queries:
        response = system.answer(query.text)
        
        # Score against ground truth
        score = evaluate_response(
            response=response,
            ground_truth=query.expected_answer,
            method="llm_as_judge",  # + human eval for subset
            dimensions=["accuracy", "completeness", "temporal_correctness"]
        )
        
        results[query.id] = score
    
    # Phase 3: Evaluate forgetting
    for forgotten_item in test_corpus.should_forget:
        is_forgotten = not system.can_recall(forgotten_item)
        results[f"forget_{forgotten_item.id}"] = is_forgotten
    
    # Phase 4: Evaluate consolidation quality
    for pattern in test_corpus.expected_semantic_patterns:
        has_learned = system.semantic_store.contains(pattern)
        results[f"consolidation_{pattern.id}"] = has_learned
    
    return aggregate_results(results)
```

---

## 4. Proposed Benchmark: OriginBench

Origin AI should create its own benchmark that tests the dimensions existing benchmarks miss:

### 4.1 OriginBench Categories

| Category | Tests | Why Existing Benchmarks Miss This |
|:---------|:------|:----------------------------------|
| **Selective Forgetting** | Does the system forget trivial info while retaining important info? | No benchmark tests forgetting quality |
| **Consolidation Quality** | Are episodic memories correctly distilled into semantic knowledge? | No benchmark tests the transformation process |
| **Pattern Completion** | Can partial cues retrieve full memories? | Existing benchmarks use exact or similar queries |
| **Interference Resistance** | Does new contradictory info correctly update (not destroy) old memories? | Limited testing in existing benchmarks |
| **Cross-Session Identity** | Does the agent maintain a consistent identity across 100+ sessions? | Existing benchmarks test 10-50 sessions |
| **Cognitive Load** | How does performance degrade as memory grows (100K+ memories)? | Existing benchmarks use relatively small corpora |
| **Associative Retrieval** | Can the system retrieve memories via indirect associations? | LoCoMo-Plus is emerging but not comprehensive |

### 4.2 Publishing Strategy

- Open-source the benchmark dataset and evaluation code
- Publish results comparing Origin AI vs. baselines
- Submit to ACL / NeurIPS / ICML workshops on agent memory
- Establish OriginBench as the **standard for brain-inspired memory evaluation**

---

## 5. Key Research Papers for Benchmarking

| Paper | Venue | Key Contribution |
|:------|:------|:----------------|
| Maharana et al., "Evaluating Very Long-Term Conversational Memory" | ACL 2024 | LoCoMo benchmark |
| "A Survey on Memory Mechanisms for LLM Agents" | arXiv:2512.13564 (2025) | Comprehensive memory taxonomy |
| "LoCoMo-Plus: Cognitive Memory Evaluation" | arXiv 2025-2026 | Tests non-similar cue retrieval |
| Jaiswal et al., "Differential Memory Decay" | Conference paper, 2025 | Mathematical framework for decay |
| Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks" | PNAS 2017 | EWC — baseline for interference resistance |

---

## 6. Success Criteria

For Origin AI's memory system to be considered state-of-the-art:

| Benchmark | Metric | Target | Current Best |
|:----------|:-------|:-------|:-------------|
| LoCoMo (Single-hop) | F1 | > 0.92 | ~0.85 (Mem0/Letta) |
| LoCoMo (Multi-hop) | F1 | > 0.80 | ~0.65 |
| LoCoMo (Temporal) | Accuracy | > 0.95 | ~0.75 |
| LoCoMo (Synthesis) | ROUGE-L | > 0.70 | ~0.55 |
| OriginBench (Forgetting) | Precision | > 0.90 | N/A (new) |
| OriginBench (Consolidation) | Fidelity | > 0.85 | N/A (new) |
| OriginBench (Pattern Completion) | Recall@5 | > 0.80 | N/A (new) |
| Latency (Retrieval) | p99 | < 100ms | ~200ms |
| Scale | Max memories | 100M+ | ~1M |

---

*Previous: [← Architecture Proposal](08_architecture_proposal.md) | Next: [References →](10_references_bibliography.md)*
