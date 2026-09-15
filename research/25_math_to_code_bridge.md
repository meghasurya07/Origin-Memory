# 25 — Mathematics-to-Code Bridge: Implementable Formulations

**Date**: 2026-09-14
**Stream**: Wave 4 — Math Bridge Research
**Purpose**: Exact mathematical formulations from neuroscience that can be directly implemented in code

---

## 1. Temporal Context Model (TCM/CMR) — Full Derivation

**Source**: Howard & Kahana 2002; Polyn et al. 2009

### Context Evolution

$$\mathbf{t}_i = \rho_i \mathbf{t}_{i-1} + \beta \mathbf{t}^{IN}_i$$

Where:
- $\mathbf{t}_{i-1}$: Previous context vector
- $\mathbf{t}^{IN}_i$: Input representation of current item
- $\beta$: Drift rate (how much new item shifts context)
- $\rho_i$: Normalization factor to keep $\|\mathbf{t}_i\| = 1$:

$$\rho_i = \sqrt{1 + \beta^2[(\mathbf{t}_{i-1} \cdot \mathbf{t}^{IN}_i)^2 - 1]} - \beta(\mathbf{t}_{i-1} \cdot \mathbf{t}^{IN}_i)$$

### Associative Matrices

| Matrix | Function | Learning Rule |
|:-------|:---------|:-------------|
| $M^{FT}$ | Feature → Context | $\mathbf{t}^{IN}_i = M^{FT} \mathbf{f}_i$ |
| $M^{TF}$ | Context → Feature (retrieval) | $\mathbf{a}_i = M^{TF} \mathbf{t}_i$ |

Both updated via Hebbian learning: $M_i = M_{i-1} + \Delta M$

### Implementation

```python
class TemporalContextEngine:
    def __init__(self, dim: int, beta: float = 0.5):
        self.context = np.zeros(dim)  # t_i
        self.M_FT = np.eye(dim) * 0.01  # Feature → Context
        self.M_TF = np.eye(dim) * 0.01  # Context → Feature
        self.beta = beta
    
    def update_context(self, feature: np.ndarray):
        """TCM context drift on new input"""
        t_in = self.M_FT @ feature
        dot = np.dot(self.context, t_in)
        rho = np.sqrt(1 + self.beta**2 * (dot**2 - 1)) - self.beta * dot
        self.context = rho * self.context + self.beta * t_in
        self.context /= np.linalg.norm(self.context) + 1e-8
        # Hebbian binding
        self.M_TF += np.outer(feature, self.context)
        self.M_FT += np.outer(self.context, feature)
    
    def retrieve(self) -> np.ndarray:
        """Probe memory with current context"""
        return self.M_TF @ self.context
```

**Implementability**: ✅ HIGHLY IMPLEMENTABLE. Context vector = hidden state. Matrices = linear layers updatable via Hebbian or backprop.

---

## 2. Prediction Error Memory Gate

**Source**: Friston 2005; Kingma & Welling 2013 (VAE)

### Precision-Weighted Update

$$\Delta\theta \propto \pi \cdot \epsilon \cdot \frac{\partial \hat{x}}{\partial \theta}$$

Where:
- $\epsilon = x - \hat{x}$ (prediction error)
- $\pi = 1/\sigma^2$ (precision = inverse variance)
- High precision × high error = STRONG encoding
- Low precision × any error = WEAK encoding (noisy environment)

### ELBO Connection (VAE)

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x) \| p(z))$$

The Gaussian log-likelihood term naturally scales reconstruction error by estimated variance (inverse precision).

### Memory Gating Algorithm

```python
def should_encode(x: np.ndarray, x_hat: np.ndarray, 
                  precision: float, threshold: float) -> str:
    """Prediction-error-based encoding gate"""
    error = np.linalg.norm(x - x_hat)
    weighted_error = precision * error
    
    if weighted_error > threshold * 2:
        return "STRONG_ENCODE"  # Very surprising in reliable context
    elif weighted_error > threshold:
        return "WEAK_ENCODE"    # Moderately surprising
    else:
        return "SKIP"           # Expected — no update needed
```

**Implementability**: ✅ HIGHLY IMPLEMENTABLE. Directly translates to VAEs with learned variances. Use error magnitude × precision as dynamic learning rate.

---

## 3. ACT-R Base-Level Activation

**Source**: Anderson et al. 2004

### The Equation

$$B_i = \ln\left(\sum_{k=1}^{n} (t - t_k)^{-d}\right)$$

Where:
- $n$ = number of accesses
- $t_k$ = timestamp of $k$-th access
- $d \approx 0.5$ = decay rate

### Retrieval

- Retrieved only if $A_i > \tau$ (threshold)
- Latency: $T = F \cdot e^{-A_i}$ (higher activation = faster)

### Efficient Implementation

For millions of memories, exact computation is expensive. Approximation strategies:

```python
def base_level_activation(access_times: list[float], 
                          now: float, d: float = 0.5) -> float:
    """ACT-R base-level activation"""
    total = sum((now - t)**(-d) for t in access_times if now > t)
    return math.log(total) if total > 0 else float('-inf')

# For scale: use exponential moving average (EMA) approximation
def ema_activation(prev_activation: float, time_delta: float, 
                   d: float = 0.5, alpha: float = 0.1) -> float:
    """Differentiable approximation for neural networks"""
    decay = time_delta ** (-d)
    return alpha * math.log(decay) + (1 - alpha) * prev_activation
```

**Implementability**: ⚠️ PARTIALLY IMPLEMENTABLE. Exact equation is discrete/non-differentiable. Use EMA or temporal attention decay masks as differentiable proxies.

---

## 4. Modern Hopfield Network as Memory Store

**Source**: Ramsauer et al. 2020

### Energy Function

$$E = -\beta^{-1} \ln\left(\sum_{i=1}^N \exp(\beta \mathbf{x}^T \boldsymbol{\xi}_i)\right) + \frac{1}{2}\mathbf{x}^T\mathbf{x}$$

### Retrieval Update (= Attention!)

$$\mathbf{x}^{new} = \mathbf{V} \cdot \text{softmax}(\beta \mathbf{K}^T \mathbf{q})$$

This IS transformer attention with:
- Stored patterns $\boldsymbol{\xi}_i$ = Keys ($K$) and Values ($V$)
- Query state $\mathbf{x}$ = Query ($Q$)
- Capacity: exponential $O(c^d)$

### Implementation

```python
import torch
import torch.nn.functional as F

def hopfield_retrieve(query: torch.Tensor, keys: torch.Tensor, 
                      values: torch.Tensor, beta: float = 1.0):
    """Modern Hopfield Network retrieval = attention"""
    scores = beta * (query @ keys.T)
    weights = F.softmax(scores, dim=-1)
    return weights @ values
```

**Implementability**: ✅ HIGHLY IMPLEMENTABLE. Already exists as standard attention. PyTorch packages available (`hopfield-layers`).

---

## 5. Information Bottleneck for Consolidation

**Source**: Tishby et al. 1999; Alemi et al. 2016 (VIB)

### The Objective

$$\min_{p(t|x)} I(X;T) - \beta I(T;Y)$$

Where:
- $I(X;T)$ = mutual info between input and compressed representation (compression cost)
- $I(T;Y)$ = mutual info between representation and target (utility)
- $\beta$ = compression-utility tradeoff

### Variational Approximation (VIB)

In practice, add KL penalty between encoded representation and prior:

$$\mathcal{L} = \mathcal{L}_{task} + \gamma D_{KL}(q(t|x) \| p(t))$$

### Use for Consolidation

```python
def consolidate_memory(episodic_memory: EpisodicMemory, 
                       compression_ratio: float = 0.3):
    """Information Bottleneck consolidation: episode → semantic gist"""
    # Encode episodic content through bottleneck
    encoded = encoder(episodic_memory.embedding)  # High-dim → low-dim
    kl_loss = kl_divergence(encoded, prior)        # Compression penalty
    gist = decoder(encoded)                         # Reconstruct gist
    
    # Only the "gist" survives — details that don't help predict are lost
    return SemanticMemory(
        key=extract_key(gist),
        value=extract_value(gist),
        confidence=1.0 - compression_ratio * kl_loss,
        source_episodes=[episodic_memory.id]
    )
```

**Implementability**: ✅ HIGHLY IMPLEMENTABLE via VIB. Perfect for the consolidation "sleep" phase.

---

## 6. Bayesian Reconsolidation

**Source**: Gershman et al. 2014

### The Update Rule

$$p(\theta | y) = \frac{p(y|\theta) \cdot p(\theta)}{\int p(y|\theta') p(\theta') d\theta'}$$

Where:
- Prior $p(\theta)$ = existing memory
- Likelihood $p(y|\theta)$ = new evidence
- Posterior $p(\theta|y)$ = updated memory

### Boundary Condition: Update vs New Trace

Compute marginal likelihood $p(y)$ under existing memory:
- If $p(y) > \text{threshold}$ → RECONSOLIDATE (Bayesian update)
- If $p(y) < \text{threshold}$ → CREATE NEW TRACE (surprise too high)

**Implementability**: ⚠️ MODERATELY IMPLEMENTABLE. Full Bayesian is expensive. Use variational inference or particle filters. Excellent theoretical basis for memory branching decisions.

---

## 7. Optimal Forgetting

**Source**: Anderson & Schooler 1991

### Need Probability

$$P(\text{need } A \mid t) \propto t^{-d}$$

### Implementation: Prioritized Eviction

```python
def compute_need_probability(memory: EpisodicMemory, 
                              now: datetime, d: float = 0.5) -> float:
    """Anderson-Schooler optimal forgetting"""
    accesses = memory.access_times  # list of timestamps
    need_score = sum(
        (now - t).total_seconds() ** (-d) 
        for t in accesses
    )
    return need_score

def evict_unneeded(memories: list, threshold: float):
    """Remove memories below need threshold"""
    return [m for m in memories 
            if compute_need_probability(m) > threshold]
```

**Implementability**: ✅ HIGHLY IMPLEMENTABLE. Maps directly to prioritized experience replay buffers and cache eviction policies.

---

## Summary: Implementation Readiness

| Mechanism | Math Ready | Code Ready | Notes |
|:----------|:-----------|:-----------|:------|
| TCM Context Drift | ✅ | ✅ | Hidden state + Hebbian matrices |
| Prediction Error Gate | ✅ | ✅ | VAE with learned variance |
| ACT-R Activation | ✅ | ⚠️ | Needs EMA approximation for differentiability |
| Modern Hopfield | ✅ | ✅ | IS transformer attention |
| Information Bottleneck | ✅ | ✅ | VIB — perfect for consolidation |
| Bayesian Reconsolidation | ✅ | ⚠️ | Variational approximation needed |
| Optimal Forgetting | ✅ | ✅ | Power-law decay scoring |

**All 7 mechanisms are mathematically formalized and at least partially implementable with current tools.**

---

*Key Papers: Howard & Kahana 2002, Friston 2005, Anderson et al. 2004, Ramsauer et al. 2020, Tishby et al. 1999, Alemi et al. 2016, Gershman et al. 2014, Anderson & Schooler 1991*

*Previous: [← Cognitive Science](24_cognitive_science.md) | Next: [→ Attention/Sleep/Emotion](26_attention_sleep_emotion.md)*
