# 15 — Mathematical Models: Advanced Brain Mechanisms

## Overview

This document extends [02_mathematics_of_memory.md](02_mathematics_of_memory.md) with the mathematical formulations for the 7 new brain mechanisms implemented in Origin Brain v0.1.1.

---

## 1. Reconsolidation Dynamics

### Memory Lability Model

When a memory $M$ is retrieved, it enters a labile state for window $\tau_{lab}$ seconds:

$$L(t) = \begin{cases} 1 & \text{if } t - t_{retrieval} < \tau_{lab} \\ 0 & \text{otherwise} \end{cases}$$

Where $L(t) = 1$ means the memory is modifiable.

### Update Rule

When labile memory $M_{old}$ encounters new related content $M_{new}$:

$$M_{updated} = \alpha(S) \cdot M_{old} + (1 - \alpha(S)) \cdot M_{new}$$

Where $\alpha(S)$ is the consolidation strength of the old memory:

$$\alpha(S) = \frac{S_{old}}{S_{old} + S_0}$$

$S_0$ is a half-saturation constant. Strongly consolidated memories ($S_{old} \gg S_0$) resist modification ($\alpha \to 1$). Weakly consolidated memories are easily overwritten ($\alpha \to 0$).

### Stability After Reconsolidation

$$S_{new} = S_{old} \cdot (1 + \beta \cdot \Delta_{info})$$

Where $\beta$ is the reconsolidation strength factor and $\Delta_{info}$ is the information gain:

$$\Delta_{info} = 1 - \text{sim}(M_{old}, M_{new})$$

If the update provides substantial new information ($\Delta_{info}$ high), stability increases more.

---

## 2. Emotional Modulation Mathematics

### Salience Modulation (Amygdala Model)

The amygdala boosts encoding strength based on emotional arousal:

$$S_{emotional} = S_{base} \cdot (1 + \gamma \cdot A \cdot V_{weight})$$

Where:
- $S_{base}$: base salience from content analysis
- $\gamma$: arousal boost factor (default 1.5)
- $A$: arousal level (0.0 to 1.0)
- $V_{weight}$: valence weight (negative = 1.2, urgent = 1.5, positive = 1.0, neutral = 0.5)

### Negativity Bias

The negativity bias is a well-documented human psychological phenomenon — negative events are remembered ~20% better than positive events of equal intensity:

$$S_{negative} = S_{emotional} \cdot \beta_{neg}$$

Where $\beta_{neg} = 1.2$ (from Baumeister et al., 2001).

### Flashbulb Memory Criterion

A flashbulb memory is created when:

$$\text{Flashbulb} = \begin{cases} \text{True} & \text{if } A > 0.8 \text{ AND } V \in \{\text{SURPRISING}, \text{NEGATIVE}, \text{URGENT}\} \\ \text{False} & \text{otherwise} \end{cases}$$

Flashbulb memories receive maximum stability ($S = S_{max}$) and are protected from decay.

### Stability Modulation

$$S_{stable} = S_{base} \cdot (1 + 0.5 \cdot A) \cdot \begin{cases} 1.2 & \text{if } V \in \{\text{NEGATIVE}, \text{URGENT}\} \\ 1.0 & \text{otherwise} \end{cases}$$

---

## 3. Schema-Based Learning Mathematics

### Schema Matching Score

For content $C$ and schema $\mathcal{S}$ with slots $\{s_1, s_2, \ldots, s_n\}$:

$$\text{match}(C, \mathcal{S}) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\text{slot}_i \text{ can be filled from } C] \cdot w_i$$

Where $w_i$ is the slot importance weight (required slots have $w_i = 2.0$, optional have $w_i = 1.0$).

### One-Shot vs Gradual Learning

The learning speed depends on schema match quality:

$$t_{learn} = \begin{cases} 1 & \text{if } \text{match}(C, \mathcal{S}) > \theta_{instant} \quad \text{(one-shot, bypasses hippocampus)} \\ \lceil 1 / \text{match}(C, \mathcal{S}) \rceil & \text{otherwise} \quad \text{(gradual, hippocampus-dependent)} \end{cases}$$

Where $\theta_{instant} = 0.8$ and $t_{learn}$ is the number of exposures needed.

### Schema Confidence Growth

Schema confidence grows with usage (more instances = more reliable schema):

$$\text{confidence}(\mathcal{S}) = 1 - e^{-k \cdot n_{instances}}$$

Where $k$ is the learning rate and $n_{instances}$ is the number of times the schema has been successfully filled.

### Schema Discovery (Unsupervised)

From a collection of episodic memories, discover schemas by:

1. Extract context key sets: $K_i = \text{keys}(\text{context}_i)$ for each memory $i$
2. Find frequent itemsets: $\{K \mid |\{i : K \subseteq K_i\}| \geq \theta_{min}\}$
3. Each frequent itemset becomes a candidate schema

---

## 4. Interference Mathematics

### Proactive Interference Model

The retrieval probability of a new memory $M_{new}$ is reduced by similar old memories:

$$P(\text{retrieve } M_{new}) = \frac{S_{new}}{S_{new} + \sum_{j \in \text{similar}} S_j \cdot \text{sim}(M_{new}, M_j)}$$

This is the **fan effect** — more similar memories = harder to retrieve any specific one.

### Retroactive Interference Model

When new memory $M_{new}$ is stored, old similar memories decay faster:

$$S_{old}' = S_{old} \cdot (1 - \delta \cdot \text{sim}(M_{old}, M_{new}))$$

Where $\delta$ is the interference strength parameter. Higher similarity = more interference.

### Retrieval-Induced Forgetting (RIF)

When memory $M_i$ is successfully retrieved, competing memories $M_j$ are inhibited:

$$S_j' = S_j \cdot (1 - \rho \cdot \text{sim}(M_i, M_j) \cdot \mathbb{1}[M_j \neq M_i])$$

Where $\rho$ is the inhibition strength. This causes competitors to fade, reducing memory clutter.

### Contradiction Detection

Two memories contradict if:

$$\text{contradiction}(M_a, M_b) = \text{sim}(M_a, M_b) > \theta_{sim} \quad \text{AND} \quad \text{negation}(M_a, M_b) = \text{True}$$

Where $\text{negation}$ detects patterns like:
- "X is Y" vs "X is not Y"
- "X likes Y" vs "X dislikes Y"
- "X in A" vs "X in B" (where A ≠ B for the same attribute)

---

## 5. Prospective Memory Mathematics

### Time-Based Trigger

$$\text{trigger}(M_p, t) = \mathbb{1}[t \geq t_{scheduled}] \cdot \mathbb{1}[\text{status} = \text{pending}]$$

### Event-Based Trigger (Similarity Model)

$$\text{trigger}(M_p, C_{current}) = \mathbb{1}[\text{sim}(\text{cue}(M_p), C_{current}) > \theta_{trigger}]$$

Where $\text{cue}(M_p)$ is the trigger condition embedding and $C_{current}$ is the current context embedding.

### Monitoring Cost Model

Strategic monitoring (continuously scanning for cues) has a cognitive cost:

$$\text{cost}_{monitor} = \sum_{p \in \text{pending}} \frac{\text{priority}_p}{\text{specificity}_p}$$

Vague triggers (low specificity) require more monitoring resources. The system should switch to **spontaneous retrieval** when monitoring cost exceeds a threshold.

---

## 6. Metamemory Mathematics

### Feeling of Knowing (FOK)

$$\text{FOK}(q) = \sigma\left(\frac{1}{K} \sum_{k=1}^{K} \text{sim}(q, M_k) \cdot R(M_k)\right)$$

Where:
- $q$ is the query
- $M_k$ are the top-K retrieved memories
- $R(M_k)$ is the retrievability of memory $k$
- $\sigma$ is the sigmoid function (maps to 0-1)

High FOK with failed retrieval = **tip-of-the-tongue** state.

### Confidence Calibration (Brier Score)

$$\text{Brier} = 1 - \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2$$

Where $f_i$ is the forecast confidence and $o_i$ is the actual outcome (1 = correct, 0 = incorrect).

- Brier = 1.0: perfect calibration
- Brier = 0.75: random guessing
- Brier < 0.75: worse than random (overconfident or underconfident)

### Confidence Level Mapping

$$\text{Level} = \begin{cases} \text{CERTAIN} & \text{if } c > 0.9 \\ \text{CONFIDENT} & \text{if } 0.7 < c \leq 0.9 \\ \text{UNCERTAIN} & \text{if } 0.4 < c \leq 0.7 \\ \text{GUESSING} & \text{if } 0.2 < c \leq 0.4 \\ \text{NO\_KNOWLEDGE} & \text{if } c \leq 0.2 \end{cases}$$

---

## 7. Memory Router Mathematics (Thalamic Gating)

### Multi-Signal Ranking

For a set of memory results $\{R_1, R_2, \ldots, R_n\}$, the final score:

$$\text{score}(R_i) = w_r \cdot \text{relevance}_i + w_t \cdot \text{tier}_i + w_{rec} \cdot \text{recency}_i + w_d \cdot \text{diversity}_i$$

Where the default weights are:
- $w_r = 0.50$ (relevance from retrieval)
- $w_t = 0.20$ (tier priority: WORKING > EPISODIC > SEMANTIC > PROCEDURAL > ARCHIVAL)
- $w_{rec} = 0.15$ (recency bonus)
- $w_d = 0.15$ (diversity via MMR)

### Maximal Marginal Relevance (MMR) for Diversity

$$\text{MMR}(R_i) = \lambda \cdot \text{sim}(q, R_i) - (1 - \lambda) \cdot \max_{R_j \in \text{selected}} \text{sim}(R_i, R_j)$$

This penalizes results that are too similar to already-selected results, ensuring diverse recall.

---

## 8. Complete System: Energy Function

The complete Origin Brain can be described as an energy minimization system (inspired by Hopfield networks and free energy principle):

$$E_{total} = E_{encoding} + E_{decay} + E_{interference} + E_{consolidation}$$

Where:

$$E_{encoding} = -\sum_{i} \text{salience}_i \cdot \text{surprise}_i \cdot \text{emotional\_boost}_i$$

$$E_{decay} = \sum_{i} (1 - R_i(t)) \cdot S_i$$

$$E_{interference} = \sum_{i \neq j} \text{sim}(M_i, M_j) \cdot S_i \cdot S_j \cdot \mathbb{1}[\text{contradiction}(i,j)]$$

$$E_{consolidation} = -\sum_{i} \text{schema\_match}(M_i) \cdot \text{access\_count}_i$$

The brain minimizes $E_{total}$: encoding surprising/emotional events, decaying unimportant memories, resolving contradictions, and consolidating frequently-accessed pattern-matching memories.

---

*Previous: [← AI Memory State of Art](14_ai_memory_state_of_art.md) | Next: see [Development Roadmap](11_development_roadmap.md)*
