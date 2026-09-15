# 02 — Mathematics of Memory

## Overview

The neuroscience of memory can be formalized into precise mathematical models. These equations form the computational backbone for designing an artificial brain memory system. This document covers the key mathematical frameworks: Hebbian learning, forgetting dynamics, plasticity models, attractor networks, and energy-based memory.

---

## 1. Hebbian Learning: "Neurons That Fire Together, Wire Together"

### 1.1 The Basic Hebbian Rule

The foundational equation for synaptic learning:

$$\Delta w_{ij} = \eta \cdot x_j \cdot y_i$$

Where:
- $\Delta w_{ij}$ = change in synaptic weight from neuron $j$ to neuron $i$
- $\eta$ = learning rate
- $x_j$ = presynaptic activity (input neuron $j$)
- $y_i$ = postsynaptic activity (output neuron $i$)

**Problem**: Unbounded growth. Weights grow without limit if two neurons are repeatedly co-active.

### 1.2 The Oja Rule (Normalized Hebbian Learning)

Oja (1982) introduced a self-normalizing variant:

$$\Delta w_{ij} = \eta \cdot y_i \cdot (x_j - y_i \cdot w_{ij})$$

This ensures that the weight vector converges to the **first principal component** of the input distribution, providing stability.

### 1.3 The BCM Rule (Bienenstock, Cooper, Munro)

A biologically realistic rule that implements both LTP and LTD with a sliding threshold:

$$\Delta w_{ij} = \eta \cdot x_j \cdot y_i \cdot (y_i - \theta_M)$$

Where:
- $\theta_M$ = modification threshold (sliding)
- When $y_i > \theta_M$: **LTP** (weight increases)
- When $y_i < \theta_M$: **LTD** (weight decreases)
- $\theta_M$ itself adapts: $\theta_M = \mathbb{E}[y_i^2]$ (average squared postsynaptic activity)

**Key property**: The sliding threshold prevents runaway excitation and implements competitive learning.

---

## 2. Spike-Timing-Dependent Plasticity (STDP)

The most biologically accurate learning rule, based on **causal timing**:

$$\Delta w = \begin{cases}
A_+ \cdot e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \text{ (pre before post → LTP)} \\
-A_- \cdot e^{\Delta t / \tau_-} & \text{if } \Delta t < 0 \text{ (post before pre → LTD)}
\end{cases}$$

Where:
- $\Delta t = t_{\text{post}} - t_{\text{pre}}$ = timing difference between post- and presynaptic spikes
- $A_+, A_-$ = amplitudes of potentiation and depression
- $\tau_+, \tau_-$ = time constants (typically 10-20 ms)

**Significance**: STDP naturally implements **causal learning**—synapses strengthen only when the presynaptic neuron plausibly *caused* the postsynaptic firing.

---

## 3. The Ebbinghaus Forgetting Curve & Memory Decay

### 3.1 Exponential Decay Model

$$R(t) = e^{-t/S}$$

Where:
- $R(t)$ = retrievability at time $t$
- $S$ = stability constant (memory strength)

### 3.2 Ebbinghaus Original Formula (1885)

$$b = \frac{100k}{(\log t)^c + k}$$

Where:
- $b$ = percentage retained
- $t$ = time elapsed
- $k \approx 1.84$, $c \approx 1.25$ (empirical constants)

### 3.3 Power Law of Forgetting

Wixted & Ebbesen (1991) showed that forgetting often follows a **power law**:

$$R(t) = a \cdot t^{-b}$$

Where $a$ and $b$ are constants. The power law predicts slower forgetting than exponential decay at long time scales—matching human data better.

### 3.4 Spaced Repetition: The Stability Equation

Each successful retrieval **increases** stability:

$$S_{n+1} = S_n \cdot (1 + \alpha \cdot e^{w \cdot S_n})$$

Where:
- $S_n$ = stability after $n$-th review
- $\alpha$ = learning efficiency parameter
- $w$ = decay modifier

This is the mathematical basis of **spaced repetition systems** (Anki, SuperMemo) and should be a core component of any artificial memory system.

---

## 4. Hopfield Networks: Attractor-Based Memory

### 4.1 Energy Function

John Hopfield (1982, Nobel Prize 2024) showed that a recurrent neural network with symmetric weights has an energy function:

$$E = -\frac{1}{2} \sum_{i \neq j} w_{ij} \cdot s_i \cdot s_j - \sum_i b_i \cdot s_i$$

Where:
- $s_i \in \{-1, +1\}$ = state of neuron $i$
- $w_{ij}$ = connection weight between neurons $i$ and $j$
- $b_i$ = bias

### 4.2 Storage Rule

To store $p$ memory patterns $\{\xi^1, \xi^2, ..., \xi^p\}$:

$$w_{ij} = \frac{1}{N} \sum_{\mu=1}^{p} \xi_i^\mu \cdot \xi_j^\mu$$

### 4.3 Capacity Limit

The maximum number of patterns that can be reliably stored:

$$p_{\max} \approx \frac{N}{2 \ln N}$$

Where $N$ = number of neurons. For $N = 1000$, this gives $p_{\max} \approx 72$ patterns.

### 4.4 Modern Hopfield Networks (2020+)

Ramsauer et al. (2020) showed that **modern Hopfield networks** with a continuous energy function can store **exponentially many** patterns:

$$E = -\text{lse}(\beta \cdot \Xi^T \cdot \mathbf{s}) + \frac{1}{2} \mathbf{s}^T \mathbf{s} + \text{const}$$

Where $\text{lse}$ is the log-sum-exp function and $\beta$ is an inverse temperature parameter. Capacity scales as:

$$p_{\max} \sim e^{\alpha \cdot d}$$

Where $d$ is the dimension of the pattern space. This connects directly to the **attention mechanism** in Transformers.

> [!NOTE]
> **Key insight**: The attention mechanism in Transformers is mathematically equivalent to one step of pattern retrieval in a Modern Hopfield Network. This means that Transformers already perform memory retrieval within each forward pass—but only over the context window.

---

## 5. Complementary Learning Systems: Mathematical Framework

### 5.1 The Catastrophic Forgetting Problem

When training a neural network with parameters $\theta$ on task $A$ (producing $\theta_A^*$) and then on task $B$:

$$\theta_B^* = \theta_A^* + \Delta\theta_B$$

The update $\Delta\theta_B$ can **destroy** the representation for task $A$. The loss on task $A$ increases dramatically:

$$\mathcal{L}_A(\theta_B^*) \gg \mathcal{L}_A(\theta_A^*)$$

### 5.2 Elastic Weight Consolidation (EWC)

DeepMind's solution (Kirkpatrick et al., 2017): protect important parameters using the Fisher Information Matrix:

$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i \cdot (\theta_i - \theta_{A,i}^*)^2$$

Where:
- $F_i$ = diagonal of the Fisher Information Matrix (importance of parameter $i$ for task $A$)
- $\lambda$ = regularization strength
- $\theta_{A,i}^*$ = optimal parameter value for task $A$

**Intuition**: Parameters that are important for previous tasks are penalized for changing, simulating the neocortex's slow, stability-preserving learning.

### 5.3 Experience Replay

Store episodic traces $\mathcal{B} = \{(x_1, y_1), ..., (x_n, y_n)\}$ and interleave them with new data during training:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{new}}(\theta) + \gamma \cdot \mathbb{E}_{(x,y) \sim \mathcal{B}}[\mathcal{L}(f_\theta(x), y)]$$

This mimics hippocampal replay during sleep consolidation.

---

## 6. Differential Memory Decay (Metacognition/SST Research)

The Scaler School of Technology team (Metacognition) proposed a **continuous-time model of forgetting and semantic drift**:

### 6.1 Memory State Evolution

A memory $m$ evolves over time according to:

$$\frac{dm}{dt} = -\frac{m(t)}{S(t)} + I(t)$$

Where:
- $m(t)$ = memory strength at time $t$
- $S(t)$ = time-varying stability (influenced by reinforcement)
- $I(t)$ = input/retrieval signal

### 6.2 Semantic Drift

Over time, episodic memories drift toward semantic attractors:

$$\frac{d\mathbf{e}}{dt} = -\nabla_\mathbf{e} V(\mathbf{e}, \mathbf{S}) + \sigma \cdot \eta(t)$$

Where:
- $\mathbf{e}$ = episodic memory vector
- $V(\mathbf{e}, \mathbf{S})$ = potential landscape shaped by semantic knowledge $\mathbf{S}$
- $\sigma \cdot \eta(t)$ = stochastic noise term (modeling lossy consolidation)
- $\nabla_\mathbf{e} V$ = gradient of the semantic potential (pulls episodes toward schema-consistent representations)

This elegantly captures how specific episodic details blur over time while the gist is preserved—a hallmark of human memory.

---

## 7. Information-Theoretic View of Memory

### 7.1 Memory as Lossy Compression

Memory consolidation can be viewed as **rate-distortion optimization**:

$$\min_{q(\hat{m}|m)} I(M; \hat{M}) \quad \text{subject to} \quad \mathbb{E}[d(M, \hat{M})] \leq D$$

Where:
- $M$ = original memory
- $\hat{M}$ = consolidated (compressed) memory
- $I(M; \hat{M})$ = mutual information (storage cost)
- $d(M, \hat{M})$ = distortion (information loss)
- $D$ = acceptable distortion threshold

The brain optimizes this tradeoff: it compresses memories to minimize storage while keeping distortion below a functional threshold.

### 7.2 Channel Capacity of Working Memory

Working memory capacity can be modeled as an **information channel** with capacity $C$:

$$C = B \cdot \log_2(1 + \text{SNR})$$

Where $B$ is the bandwidth (number of slots) and SNR is the signal-to-noise ratio. Miller's "7 ± 2" and Cowan's "4 ± 1" are empirical estimates of this capacity.

---

## 8. Energy-Based Models for Memory Landscapes

### 8.1 Memory as Energy Minimization

Define a memory system with state $\mathbf{x}$ and stored patterns $\{\xi^\mu\}$. The energy landscape:

$$E(\mathbf{x}) = -\sum_\mu \phi(\mathbf{x}^T \xi^\mu)$$

Where $\phi$ is an activation function. Memories correspond to **local minima** (attractors) in this landscape.

### 8.2 Memory Retrieval as Gradient Descent

Retrieval is the process of iterating toward the nearest attractor:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \nabla E(\mathbf{x}_t)$$

Starting from a noisy or partial cue $\mathbf{x}_0$, the system converges to the closest stored pattern.

### 8.3 Forgetting as Attractor Erosion

When a memory is not reinforced, its attractor basin **flattens**:

$$E_\mu(t) = E_\mu(0) \cdot e^{-\lambda t}$$

The memory becomes harder to retrieve as its basin becomes shallower relative to neighboring attractors.

---

## 9. Summary: Mathematical Toolkit for Artificial Memory

| Model | What It Captures | Application |
|:------|:----------------|:------------|
| Hebbian/STDP | Associative learning from co-occurrence/causation | Encoding new associations |
| BCM Rule | Competitive learning with sliding threshold | Preventing runaway excitation |
| Forgetting curves | Exponential/power-law decay of unused memories | Memory eviction scheduling |
| Spaced repetition | Stability growth through timed reinforcement | Memory maintenance optimization |
| Hopfield networks | Associative retrieval from partial cues | Pattern completion engine |
| EWC | Protecting important parameters from overwriting | Catastrophic forgetting prevention |
| Experience replay | Interleaved retraining from stored episodes | Offline consolidation |
| Differential decay | Continuous-time forgetting with semantic drift | Episodic → semantic transformation |
| Rate-distortion | Optimal compression of memory traces | Memory consolidation as compression |
| Energy landscapes | Memory as attractor dynamics | Retrieval, forgetting, interference |

---

*Previous: [← Neuroscience Foundations](01_neuroscience_foundations.md) | Next: [The AI Memory Problem →](03_ai_memory_problem.md)*
