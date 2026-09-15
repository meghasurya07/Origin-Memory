# 36 — Generative Models for Reconstructive Memory

**Date**: 2026-09-14
**Stream**: Wave 6 — Generative Models Research
**Type**: RESEARCH + MATHEMATICAL FOUNDATIONS

---

## 1. VAEs as Memory Models

### The ELBO as Memory Equation

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \| p(z))$$

| Term | Memory Analogue | Function |
|:-----|:----------------|:---------|
| $q_\phi(z|x)$ | **Encoder** (hippocampus) | Compress experience → latent trace |
| $p_\theta(x|z)$ | **Decoder** (cortical reconstruction) | Reconstruct memory from trace |
| $\log p_\theta(x|z)$ | **Reconstruction fidelity** | How well the memory matches reality |
| $D_{KL}$ | **Forgetting / compression** | Force traces to be sparse and structured |
| $\beta$ | **Forgetting rate** | Higher = more compression = more forgetting |

### β-VAE for Controllable Forgetting
- $\beta > 1$: Strong compression → lose details, keep gist (semantic memory)
- $\beta < 1$: Weak compression → preserve details (fresh episodic memory)
- $\beta$ increases over time → **gradual forgetting** (Ebbinghaus decay in latent space!)

### Conditional VAE (CVAE) for Context-Dependent Recall

$$p_\theta(x|z, c) = \text{Decoder}(z, c; \theta)$$

Where $c$ = conditioning vector that combines:
- Temporal context (TCM vector)
- Emotional state (valence, arousal)
- Active schema (schema embedding)
- Current goals (PFC state)

**Same trace $z$, different context $c$ → different reconstruction $x$**

This IS context-dependent memory. This IS how the same event is "remembered" differently depending on when and how you recall it.

---

## 2. Generative Replay = Hippocampal Sleep Replay

### Shin et al. (2017): Deep Generative Replay

Architecture:
```
Generator (hippocampus):  produces pseudo-data from past tasks
Solver (cortex):           learns from pseudo-data + new data
```

When learning Task B:
1. Generator creates "replayed" examples of Task A
2. Solver trains on both Task A replays + real Task B data
3. Generator is then updated to also produce Task B examples

**This IS sleep replay.** The hippocampus generates compressed versions of past experiences (during SWS sharp-wave ripples) and the cortex learns from them.

### van de Ven & Tolias (2018): Brain-Inspired Replay (BI-R)

Improvements:
- **Distillation**: Train on soft probabilistic targets (not hard labels) — mirrors how cortex receives modulated, not verbatim, signals from hippocampus
- **Replay-through-Feedback**: Generator integrated INTO the main model via feedback connections — more biologically plausible

### Implication for Origin AI
Our Sleep Engine Phase 1 (SWS replay + compression) already does a simple version. But true generative replay would:
1. Use the compressed semantic gist as a GENERATOR input
2. Produce multiple synthetic "replays" with variations
3. Train the retrieval/prediction models on these replays
4. Each replay slightly different — mirrors how hippocampal replay adds noise

---

## 3. World Models as Implicit Memory

### Dreamer (Hafner): RSSM Architecture

$$z_t \sim p(z_t | z_{t-1}, a_{t-1})$$

The agent's "memory" IS the learned transition dynamics. By predicting future latent states, the agent "dreams" — simulates trajectories entirely in latent space.

### JEPA (LeCun): Joint Embedding Predictive Architecture

Predicts REPRESENTATIONS of future states (not raw pixels). Relies on semantic, abstracted memory of the world to maintain object permanence.

### Key Insight
Memory as compressed world model: you don't need to store every experience if your model of the world can PREDICT what would have happened.

This connects to our prediction error encoding (Pillar 1): if your world model is good, only PREDICTION ERRORS need to be stored. Everything else is reconstructable from the model.

---

## 4. Sparse Distributed Memory (Kanerva 1988)

### The Kanerva Machine (Wu et al. 2018)

A VAE-style model where:
- Memory addresses are SPARSE and HIGH-DIMENSIONAL (matching hippocampal sparse coding)
- Storage and retrieval use Hamming distance in binary space
- Retrieval naturally does PATTERN COMPLETION — a noisy address retrieves the nearest stored pattern

$$p(\text{read} | \text{address}) = \sum_i w_i \cdot \text{stored}_i, \quad w_i = \text{softmax}(\text{sim}(\text{address}, \text{address}_i))$$

This is EXACTLY the CA3 autoassociative network in mathematical form:
- The "addresses" are the sparse binding codes (engram cells)
- The "stored patterns" are the distributed cortical representations
- The "softmax over similarities" is the attractor dynamics

### SDM Properties That Match Biology
1. **Sparse addressing** — only ~2-4% of address space is used per memory
2. **Content-addressable** — retrieve by partial content, not by location
3. **Graceful degradation** — noisy cues still retrieve correct memories
4. **Capacity scaling** — storage capacity grows exponentially with dimension

---

## 5. Context-Conditional Generation Architecture

### Multi-Conditional Reconstruction

$$\hat{x} = G(z; c_{temporal}, c_{emotion}, c_{schema}, c_{goal})$$

Implementation patterns:

**Option A: Concatenation**
```python
combined = torch.cat([z, c_temporal, c_emotion, c_schema, c_goal], dim=-1)
x_hat = decoder(combined)
```

**Option B: FiLM conditioning (Feature-wise Linear Modulation)**
```python
# Each context produces scale and shift parameters
gamma, beta = film_generator(c_temporal, c_emotion, c_schema)
# Applied at each layer of the decoder
h = gamma * decoder_layer(z) + beta
```

**Option C: Cross-attention conditioning**
```python
# Context vectors as keys/values, latent z as query
x_hat = cross_attention(query=z, key=contexts, value=contexts)
```

FiLM conditioning is most biologically plausible — neuromodulators (dopamine, serotonin) literally modulate neural responses via gain control, which IS feature-wise linear modulation.

---

## 6. Benchmarking Reconstructive Memory

### Metrics for Reconstruction Quality

| Metric | What It Measures | Biological Equivalent |
|:-------|:----------------|:---------------------|
| BERTScore | Semantic preservation | Gist accuracy |
| BLEU/ROUGE | Surface-level overlap | Verbatim fidelity |
| Lure detection | Schema-consistent confabulation | "Appropriate" false memories |
| Context sensitivity | Different reconstructions from same trace | Context-dependent recall |
| Compression ratio | How much was compressed | Forgetting efficiency |

### Lure Dataset Method
Create items conceptually similar to stored memories but NOT actually stored. If the system "recognizes" them using the correct schema → it demonstrates true schema-based reconstruction (not just verbatim retrieval).

This is the DRM (Deese-Roediger-McDermott) paradigm for AI:
1. Encode: "bed, rest, wake, tired, dream, snooze, nap" 
2. Probe with lure: "sleep" (never presented)
3. If system "recalls" sleep → schema-based reconstruction WORKS

---

## Mathematical Unification: The Origin Brain Equation

Combining all insights, the complete memory system can be expressed as:

$$\mathcal{L}_{\text{memory}} = \underbrace{\mathbb{E}[\log p(x|z, c)]}_{\text{reconstruction quality}} - \underbrace{\beta(t) \cdot D_{KL}(q(z|x) \| p(z))}_{\text{time-dependent forgetting}} + \underbrace{\lambda \cdot \text{PE}(x)}_{\text{prediction error gate}}$$

Where:
- $\beta(t)$ increases over time → gradual forgetting
- $\text{PE}(x) = \|\epsilon\| \cdot \pi$ = precision-weighted prediction error
- $c = [c_{tcm}, c_{emotion}, c_{schema}]$ = multi-modal context

This single equation captures:
- Pillar 1 (prediction error): $\lambda \cdot \text{PE}(x)$ gates encoding
- Pillar 2 (temporal context): $c_{tcm}$ conditions reconstruction
- Pillar 3 (forgetting): $\beta(t) \cdot D_{KL}$ provides compression
- Pillar 4 (consolidation): $\beta$ steps up during "sleep" cycles
- **Reconstruction**: The decoder $p(x|z, c)$ GENERATES memory

---

*Sources: Kingma & Welling 2014 (VAE), Shin et al. 2017 (Generative Replay), Hafner et al. 2019 (Dreamer), Wu et al. 2018 (Kanerva Machine), LeCun 2022 (JEPA), Higgins et al. 2017 (β-VAE)*
