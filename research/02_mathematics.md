# Origin Brain — Mathematical Foundations

## 1. Forgetting Curves (Ebbinghaus, power law, exponential)

### Exponential Decay Model
R(t) = e^{-t/S}
Where (t)$ is retrievability at time $ and $ is stability constant.

### Ebbinghaus Original Formula (1885)
b = \frac{100k}{(\log t)^c + k}
Where $ = percentage retained, $ = time elapsed,  \approx 1.84$,  \approx 1.25$.

### Power Law of Forgetting
Wixted & Ebbesen (1991) showed that forgetting often follows a power law:
R(t) = a \cdot t^{-b}
This predicts slower forgetting than exponential decay at long time scales, matching human data better.

### Spaced Repetition: The Stability Equation
Each successful retrieval increases stability:
S_{n+1} = S_n \cdot (1 + \alpha \cdot e^{w \cdot S_n})
Where $ = stability after $-th review, $\alpha$ = learning efficiency parameter, $ = decay modifier.

### ACT-R Base-Level Activation (Anderson et al. 2004)
B_i = \ln\left(\sum_{k=1}^{n} (t - t_k)^{-d}\right)
Where $ = number of accesses, $ = timestamp of $-th access,  \approx 0.5$ = decay rate.

### Optimal Forgetting (Anderson & Schooler 1991)
Need Probability: (\text{need } A \mid t) \propto t^{-d}$.

## 2. Bayesian Memory (prediction error, surprise, encoding)

### Prediction Error Memory Gate (Friston 2005)
Precision-Weighted Update:
\Delta\theta \propto \pi \cdot \epsilon \cdot \frac{\partial \hat{x}}{\partial \theta}
Where $\epsilon = x - \hat{x}$ (prediction error), $\pi = 1/\sigma^2$ (precision).
High precision × high error = STRONG encoding.

### Bayesian Reconsolidation (Gershman et al. 2014)
The Update Rule:
p(\theta | y) = \frac{p(y|\theta) \cdot p(\theta)}{\int p(y|\theta') p(\theta') d\theta'}
Compute marginal likelihood (y)$ under existing memory:
- If (y) > \text{threshold}$ → RECONSOLIDATE
- If (y) < \text{threshold}$ → CREATE NEW TRACE

### Memory Lability Model
L(t) = 1 \text{ if } t - t_{retrieval} < \tau_{lab} \text{ else } 0
Update Rule: {updated} = \alpha(S) \cdot M_{old} + (1 - \alpha(S)) \cdot M_{new}$
Where $\alpha(S) = \frac{S_{old}}{S_{old} + S_0}$

## 3. Synaptic Plasticity Math (STDP, BCM, Hebbian)

### The Basic Hebbian Rule
\Delta w_{ij} = \eta \cdot x_j \cdot y_i
Where $\Delta w_{ij}$ is the change in synaptic weight, $\eta$ is learning rate, $ is presynaptic activity, $ is postsynaptic activity.

### The Oja Rule (Normalized Hebbian Learning)
\Delta w_{ij} = \eta \cdot y_i \cdot (x_j - y_i \cdot w_{ij})
Ensures weight vector converges to the first principal component of input distribution.

### The BCM Rule (Bienenstock, Cooper, Munro)
\Delta w_{ij} = \eta \cdot x_j \cdot y_i \cdot (y_i - \theta_M)
Where $\theta_M = \mathbb{E}[y_i^2]$ is the sliding modification threshold.

### Spike-Timing-Dependent Plasticity (STDP)
\Delta w = \begin{cases} A_+ \cdot e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \\ -A_- \cdot e^{\Delta t / \tau_-} & \text{if } \Delta t < 0 \end{cases}
Where $\Delta t = t_{post} - t_{pre}$.

## 4. Information Theory (entropy, mutual information)

### Memory as Lossy Compression
Memory consolidation can be viewed as rate-distortion optimization:
\min_{q(\hat{m}|m)} I(M; \hat{M}) \quad \text{subject to} \quad \mathbb{E}[d(M, \hat{M})] \leq D

### Channel Capacity of Working Memory
C = B \cdot \log_2(1 + \text{SNR})
Where $ is bandwidth and SNR is signal-to-noise ratio.

### Information Bottleneck for Consolidation (Tishby et al. 1999)
\min_{p(t|x)} I(X;T) - \beta I(T;Y)
Variational Approximation (VIB): $\mathcal{L} = \mathcal{L}_{task} + \gamma D_{KL}(q(t|x) \| p(t))$.

## 5. Attractor Networks & Hopfield (pattern completion, energy landscape)

### Classical Hopfield Networks
Energy Function:  = -\frac{1}{2} \sum_{i \neq j} w_{ij} \cdot s_i \cdot s_j - \sum_i b_i \cdot s_i$
Capacity Limit: {\max} \approx \frac{N}{2 \ln N}$

### Modern Hopfield Networks (Ramsauer et al. 2020)
Energy Function:
E = -\beta^{-1} \ln\left(\sum_{i=1}^N \exp(\beta \mathbf{x}^T \boldsymbol{\xi}_i)\right) + \frac{1}{2}\mathbf{x}^T\mathbf{x}
Capacity: {\max} \sim c^{d-1}$ (Exponential in dimension).
Retrieval Update (= Attention!):
\mathbf{x}^{new} = \mathbf{V} \cdot \text{softmax}(\beta \mathbf{K}^T \mathbf{q})

### Forgetting as Attractor Erosion
E_\mu(t) = E_\mu(0) \cdot e^{-\lambda t}

## 6. Successor Representations (cognitive map math)

### Successor Representation (Dayan, 1993)
Factors value function into reward prediction + predictive map:
V(s) = \sum_{s'} M(s, s') \cdot R(s')
Where (s, s')$ is the discounted probability of visiting state '$ from state $.
Hippocampus place cells encode the SUCCESSOR REPRESENTATION, making memory a prediction engine.

## 7. Key Equations Reference

### Temporal Context Model (Howard & Kahana 2002)
\mathbf{t}_i = \rho_i \mathbf{t}_{i-1} + \beta \mathbf{t}^{IN}_i
\rho_i = \sqrt{1 + \beta^2[(\mathbf{t}_{i-1} \cdot \mathbf{t}^{IN}_i)^2 - 1]} - \beta(\mathbf{t}_{i-1} \cdot \mathbf{t}^{IN}_i)

### Complete System Energy Function
E_{total} = E_{encoding} + E_{decay} + E_{interference} + E_{consolidation}
E_{encoding} = -\sum_{i} \text{salience}_i \cdot \text{surprise}_i \cdot \text{emotional\_boost}_i
E_{interference} = \sum_{i \neq j} \text{sim}(M_i, M_j) \cdot S_i \cdot S_j \cdot \mathbb{1}[\text{contradiction}(i,j)]

## 8. Free Energy Principle

The Free Energy Principle (Friston 2005, 2010) establishes that self-organizing biological systems minimize variational free energy $F$ to maintain their structural and cognitive integrity. Memory is the updating of the internal generative model:

$$F = D_{KL}[q(s|o) \parallel p(s)] - \ln p(o|s) = \underbrace{\text{Complexity}}_{\text{cost of updating model}} - \underbrace{\text{Accuracy}}_{\text{how well model predicts}}$$

Where:
- $o$ represents sensory observations / incoming agent inputs.
- $s$ represents hidden environmental states.
- $q(s|o)$ is the recognition density (the agent's inferred belief state).
- $p(s)$ is the prior generative density over environmental states.
- $p(o|s)$ is the likelihood mapping from states to expected observations.

### Implications for Cognitive Memory
1. **Memory as Generative Parameters**: Rather than storing raw sensory inputs, memory stores the parameters of a predictive generative model.
2. **Selective Updating**: When $F$ is minimal, observation $o$ is already explained by existing schemas (zero learning required). Memory updates are triggered strictly to suppress prediction divergence.

---

## 9. Precision-Weighted Encoding Gate

Grounded in predictive processing, encoding is gated by precision-weighted prediction errors rather than logging raw streams.

### Mathematical Formulation
Given sensory input $x_t$, context prediction $\hat{x}_t$, and context precision (inverse variance / confidence) $\pi_t = 1/\sigma_t^2$:

$$\text{encode}(x_t) = \begin{cases} 
\text{STRONG encode} & \text{if } \pi_t \cdot \|x_t - \hat{x}_t\|^2 > \theta_{\text{encode}} \\ 
\text{WEAK encode} & \text{if } \|x_t - \hat{x}_t\|^2 > \theta_{\text{weak}} \\ 
\text{SKIP} & \text{otherwise} 
\end{cases}$$

Where:
- $\hat{x}_t$: The internal model's expectation of input $x_t$.
- $\pi_t$: Precision metric denoting contextual reliability; prevents random environmental noise or stochastic chatter from triggering memory allocation.
- $\theta_{\text{encode}}$: High threshold requiring both high prediction error and high contextual reliability, allocating new episodic engrams with elevated stability.
- $\theta_{\text{weak}}$: Intermediate threshold capturing moderate novelty without full precision reinforcement.
- $\text{SKIP}$: Predictable or noisy inputs below threshold are discarded, preventing memory bloat and preserving cognitive bandwidth.

---

## 10. Hopfield ≡ Attention Identity

A foundational insight uniting modern deep learning and associative memory (Ramsauer et al. 2020) is that Transformer self-attention is mathematically identical to the continuous state update in a Modern Hopfield Network.

### The Formal Identity
$$\text{Memory Retrieval} \equiv \text{Attention} \equiv \text{Hopfield Update}$$

$$\boldsymbol{\xi}^{t+1} = \mathbf{X} \cdot \text{softmax}\left(\beta \mathbf{X}^T \boldsymbol{\xi}^t\right) \equiv \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### The Softmax Equation Mapping
In a Modern Hopfield Network with continuous state $\boldsymbol{\xi}$, stored memory patterns $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)$, inverse temperature $\beta$, and energy function $E = -\beta^{-1}\ln\left(\sum_{i=1}^N \exp(\beta \boldsymbol{\xi}^T \mathbf{x}_i)\right) + \frac{1}{2}\boldsymbol{\xi}^T\boldsymbol{\xi}$:

| Hopfield Attractor Variable | Transformer Attention Analog | Cognitive Memory Analog |
|:---|:---|:---|
| State vector $\boldsymbol{\xi}^t$ | Query vector $\mathbf{Q}$ | Retrieval cue (partial episodic prompt) |
| Memory patterns $\mathbf{X}^T$ | Key matrix $\mathbf{K}^T$ | Stored memory representations |
| Normalization factor $\beta$ | Inverse scale factor $1/\sqrt{d_k}$ | Attractor sharpness / retrieval temperature |
| State update $\boldsymbol{\xi}^{t+1}$ | Attention output $\mathbf{A}$ | Reconstructed / retrieved memory pattern |
| Memory values $\mathbf{X}$ | Value matrix $\mathbf{V}$ | Content-addressable memory engrams |

This equivalence demonstrates that multi-head attention is a bank of parallel continuous-attractor associative memory networks operating over episodic representations.

---

## 11. Unified Energy Function

The global operation of the Origin Brain memory system minimizes a unified energy function $E_{\text{total}}$, balancing predictive error, temporal binding, storage complexity, and semantic consolidation:

$$E_{\text{total}} = E_{\text{prediction}} + E_{\text{context}} + E_{\text{complexity}} + E_{\text{consolidation}}$$

### Component Energy Terms

1. **Prediction Energy ($E_{\text{prediction}}$)**:
   $$E_{\text{prediction}} = \sum_t \pi_t \|x_t - \hat{x}_t\|^2$$
   Measures cumulative precision-weighted prediction error. Minimizing $E_{\text{prediction}}$ drives selective encoding of surprising observations.

2. **Context Binding Energy ($E_{\text{context}}$)**:
   $$E_{\text{context}} = -\sum_{i,j} \text{sim}(\mathbf{c}_i, \mathbf{c}_j) \cdot \text{Association}(m_i, m_j)$$
   Encodes temporal and associative contiguity (TCM). Minimizing $E_{\text{context}}$ pulls temporally contiguous memories into coherent, sequential episodic trajectories.

3. **Complexity / Retention Energy ($E_{\text{complexity}}$)**:
   $$E_{\text{complexity}} = \sum_i S_i \cdot (1 - R_i(t))$$
   Represents the thermodynamic and computational cost of maintaining low-retrievability traces. Minimizing $E_{\text{complexity}}$ drives active forgetting and engram decay of unused information.

4. **Consolidation Pressure Energy ($E_{\text{consolidation}}$)**:
   $$E_{\text{consolidation}} = -\sum_i \text{schema\_match}(m_i) \cdot f(\text{access\_count}_i)$$
   Reflects the structural alignment of memories into generalized cognitive schemas. Minimizing $E_{\text{consolidation}}$ compresses recurring episodic traces into durable semantic knowledge graphs during sleep/consolidation phases.
