# 18 — Deep Research Findings: Raw Reports from 4 Research Streams

**Date**: 2026-09-14
**Purpose**: Raw research outputs documenting all findings, evidence, and citations

---

## Stream 1: Computational Neuroscience Mechanisms

### 1.1 Complementary Learning Systems (CLS)

**Source**: McClelland et al. 1995; Kumaran et al. 2016

**The EXACT Computational Mechanism:**
CLS resolves the stability-plasticity dilemma using dual interacting systems:
- **Hippocampus** = fast-learning episodic buffer using sparse, orthogonalized representations (Pattern Separation) to encode instantly without interference
- **Neocortex** = slow-learning semantic system using dense, distributed representations
- **Bridge mechanism**: Interleaved Replay — hippocampus reinstates stored patterns during rest/sleep, iteratively teaching the neocortex through interleaved gradient updates

**Why it hasn't succeeded in production AI:**
- LLMs use monolithic architecture with slow gradient descent (pre-training) + transient in-context learning (prompt window)
- There is NO differentiable bridge that takes transient in-context memory and gradually factorizes it into permanent model weights online
- DQN's "Experience Replay" is inspired by CLS but is a weak implementation

**The Gap**: No production architecture has a continuous, differentiable mechanism for hippocampal→neocortical transfer without full retraining.

### 1.2 Synaptic Tagging and Capture (STC)

**Source**: Frey & Morris 1997; Redondo & Morris 2011

**How the brain decides WHICH memories to keep:**
- Weak stimulus → local transient changes (Early-LTP) + leaves a "Tag"
- Strong salient stimulus → triggers global synthesis of Plasticity-Related Proteins (PRPs)
- Any synapse with an active Tag can "Capture" the PRPs → consolidation into Late-LTP
- Result: You remember breakfast ONLY if something important happened near it

**Mathematical Formulation:**
$$\frac{d}{dt} W_i = \alpha \cdot Tag_i(t) \cdot PRP(t)$$

Where:
- $Tag_i(t)$ = local state variable for synapse $i$, decays over time
- $PRP(t)$ = global cellular state triggered by reward/salience threshold
- Consolidation happens ONLY when both are active simultaneously

**Algorithm Feasibility**: An STC-based optimizer would leave decaying "eligibility traces" (tags) on specific weights and only consolidate them into permanent updates if a global loss spike (PRP) occurred within a temporal window.

### 1.3 Memory Engram Cells

**Source**: Josselyn & Tonegawa 2020

**Competitive Allocation Hypothesis:**
- Neurons with temporarily high excitability (driven by CREB protein) "win" the competition to encode a memory
- Winners suppress neighbors via lateral inhibition (winner-take-all)
- Creates a highly sparse, orthogonal sub-network (the engram) for each memory

**Computational Model:**
$$Excitability_i(t) \propto CREB_i(t)$$

- If stimulus arrives: softmax-like activation where only top $k$ most excitable neurons update weights
- Forces sparse, orthogonal sub-networks per memory
- Co-allocation of temporally close memories (explains associative linking)

### 1.4 CSCG and Thousand Brains

**CSCG (Clone-Structured Causal Graph)**: Advanced HMM with emission matrix mapping clones to observations. Elegant but rigid/discrete.

**Thousand Brains (Hawkins/Numenta)**: Cortical columns + grid-cell reference frames + voting. Highly conceptual but lacks scalable differentiable formulation.

**Why neither has taken off**: Dense, continuous matrix multiplications (Transformers) scale infinitely better than rigid graphical models on GPUs. The bitter lesson wins.

---

## Stream 2: Free Energy Principle & Predictive Processing

### 2.1 Friston's FEP and Memory

**Core Equation:**
$$F = D_{KL}[q(s|o) \| p(s)] - \ln p(o|s) = \text{Complexity} - \text{Accuracy}$$

**What is stored**: NOT raw experiences. The brain stores the structural/synaptic adjustments needed to optimize its generative model.

**Trigger for encoding**: Surprise (prediction error). When outcome deviates from expectation, prediction error destabilizes synapses and triggers learning.

### 2.2 Predictive Coding (Rao & Ballard 1999)

- Brain cascades predictions DOWN cortical hierarchy
- Only prediction ERRORS travel UP
- Not all errors are equal: precision-weighted errors determine encoding strength
- High-precision errors (surprising in predictable context) → strong encoding
- Random noise in chaotic environments → ignored

### 2.3 Active Inference as Memory

Under active inference, memory IS the generative model:
- To "remember" = run generative model in reverse, reconstructing past state from current synaptic weights + context
- No filing cabinet. Only a continuous simulation engine.
- Traditional "store and retrieve" is replaced by "predict and reconstruct"

### 2.4 Recent Validation: Google Titans (2024)

"Titans: Learning to Memorize at Test Time" uses:
- **Surprise metric** (prediction error) to filter what gets committed to long-term memory weights
- Only updates persistent memory when incoming tokens are sufficiently unexpected
- Mirrors biological predictive processing

**This validates our direction.** Google is moving toward prediction-error-based memory.

---

## Stream 3: Mathematical Foundations

### 3.1 Modern Hopfield Networks

**Energy Function:**
$$E = -\beta^{-1} \log \sum_{i=1}^N \exp(\beta x^T \xi_i) + \frac{1}{2} x^T x$$

**Capacity**: Exponential $O(c^d)$ vs classical linear $O(N)$

**Key insight**: The update rule $x^{new} = \Xi \, \text{softmax}(\beta \Xi^T x)$ is EXACTLY equivalent to transformer attention with K, V = stored memories, Q = query. **Transformer attention IS associative memory retrieval.**

### 3.2 Sparse Distributed Memory (Kanerva)

- Memory in high-dimensional binary space ($\{0,1\}^{10000}$)
- Stored across multiple "hard locations" within Hamming distance
- Retrieval = sum of active locations, thresholded
- **Graceful degradation** — loss of individual bits only causes minor accuracy loss
- Modern revival: CALM (Continual Associative Learning Model) uses SDM to combat catastrophic forgetting

### 3.3 Neural ODEs for Memory

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

- $O(1)$ memory cost via adjoint sensitivity method
- Liquid Time-Constant (LTC) Networks adapt time constant to input:
$$\frac{dx(t)}{dt} = -(\tau^{-1} + f(x, I, \theta)) x(t) + A \cdot f(x, I, \theta)$$
- Dynamically alters memory "speed" based on stimuli importance

### 3.4 Test-Time Training (TTT)

**Sun et al. 2024**: Replace KV cache with a model that trains itself during inference:
$$W_t = W_{t-1} - \eta \nabla_W \mathcal{L}(M_{W_{t-1}}(x_t))$$

- Memory as fast weights updated at test time
- Fixes $O(N^2)$ bottleneck of transformers
- Essentially: memory IS learning, not storage

### 3.5 State Space Models (Mamba)

$$h_t = \bar{A}(x_t) h_{t-1} + \bar{B}(x_t) x_t$$

- Input-dependent $\bar{A}$ matrix = gating (when to remember/forget)
- But: exponential information decay over ultra-long contexts
- Struggles with "needle-in-haystack" exact recall
- 2025: Hybrid models (Jamba) combine SSM compression + Attention exact recall

### 3.6 Why DNC/NTM Failed

1. Optimization instability (NaN losses, vanishing gradients through addressing)
2. Sequential read/write bottleneck (can't parallelize on GPU)
3. Transformers achieved the same goal via attention but parallelizable

### 3.7 Most Promising Framework

**Continuous Associative Fast Weights** = Modern Hopfield + TTT:
- Memory states as fast weights (TTT) → decouple capacity from sequence length
- Structured as Modern Hopfield energy landscape → exponential capacity + stable one-step retrieval

---

## Stream 4: The Breakthrough Insight

### 4.1 Why RAG Fails

Three fundamental flaws:
1. **No temporal awareness**: All embeddings treated as equally valid regardless of time
2. **Similarity bottleneck**: Can't do multi-hop reasoning or connect logically-linked but textually-different concepts
3. **No forgetting/updating**: Accumulates orphan chunks and noise that degrade accuracy at scale

### 4.2 Forgetting as Algorithm

- Anderson & Schooler (1991): Memory decay mirrors statistical probability of needing info
- Richards & Frankland (2017): Forgetting = regularization against overfitting
- Key: system should forget to GENERALIZE, not just to save space

### 4.3 Memory as Compression

- Information Bottleneck principle (Tishby 2015)
- Memory = lossy compression optimized for future utility
- Brain doesn't store recordings; stores compressed representations
- Recall = reconstruction from compressed code

### 4.4 Sleep Replay

- NOT just video playback — active generative model doing self-supervised learning
- Takes raw episodic experiences and interleaves them
- Prunes noise, discovers reusable primitives
- Essentially: offline reinforcement learning + knowledge graph construction
- 2026: "Daydreaming algorithms" that mimic sleep-cycle consolidation

### 4.5 Temporal Context Model

- Howard & Kahana (2002): memories bound to slowly drifting temporal context
- Memory IS context — not content stored, but temporal state that enables reconstruction
- Exact opposite of vector DB (which retrieves isolated chunks without temporal state)

### 4.6 Competitive Landscape

- **Evermind (EverOS)**, **Letta/MemGPT**, **Mem0**, **Supermemory** — all building better databases
- None implement prediction error encoding, temporal context, sleep, or unified forgetting
- The field is shifting from "bigger context windows" to "selective compact and update"
- **Google Titans** validates prediction-error direction but doesn't have temporal binding or sleep

---

*This document preserves the complete raw research output from all four research streams.*

*See [17_breakthrough_thesis.md](17_breakthrough_thesis.md) for the synthesized thesis.*
