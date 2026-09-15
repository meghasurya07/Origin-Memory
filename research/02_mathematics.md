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
