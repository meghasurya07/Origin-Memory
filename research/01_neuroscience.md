# Origin Brain — Neuroscience Foundations

## 1. Memory Systems Overview
The human brain's memory system is a pipeline of three interdependent stages:
- **Encoding**: Transformation of sensory experience into neural representation. Selective process, choosing engram cells based on excitability.
- **Consolidation**: Stabilizing memories. Occurs at two levels: Synaptic (LTP/LTD) and Systems (dialogue between hippocampus and neocortex during sleep). 
- **Retrieval**: Reactivation of traces. Retrieval is reconstructive, not reproductive.

**The Memory Taxonomy:**
- **Declarative (Explicit)**: 
  - **Episodic**: Personal experiences bound to time/place (Hippocampus).
  - **Semantic**: Facts and concepts (Neocortex).
- **Non-Declarative (Implicit)**:
  - **Procedural**: Motor skills/habits (Basal ganglia/Cerebellum).
- **Working Memory**: Temporary workspace (Prefrontal Cortex), limited capacity.

**Forgetting:**
A feature, not a bug, allowing generalization and cognitive efficiency. Ebbinghaus Forgetting Curve:
$$R(t) = e^{-t/S}$$
Where $R$ is retrievability, $t$ is time, and $S$ is memory stability.

## 2. Brain Regions
- **Hippocampus**: The memory coordinator and indexer.
  - **Dentate Gyrus (DG)**: Pattern separation (sparse coding to prevent interference).
  - **CA3**: Pattern completion (autoassociative network for retrieval from partial cues).
  - **CA1**: Novelty mismatch detection.
- **Entorhinal Cortex (EC)**: Gateway providing spatial/temporal coordinates.
- **Prefrontal Cortex (PFC)**: Working memory and executive control. Provides top-down control over what to encode/retrieve.
- **Amygdala**: Emotional modulator, enhances encoding/consolidation through stress hormones.

## 3. Cellular Mechanisms
Memories are physically stored as structural modifications:
- **LTP/LTD**: Long-Term Potentiation (strengthening) and Depression (weakening). NMDA receptors act as biological AND gates (Hebbian learning).
- **Spike-Timing Dependent Plasticity (STDP)**:
$$\Delta w = \begin{cases} A_+ \exp(-\Delta t / \tau_+) & \text{if } \Delta t > 0 \text{ (LTP)} \\ -A_- \exp(\Delta t / \tau_-) & \text{if } \Delta t < 0 \text{ (LTD)} \end{cases}$$
- **BCM Theory**: Sliding threshold $\theta_M$ for Ca2+ concentration determines LTP vs LTD:
$$\Delta w = \phi(c, \theta_M) \cdot x$$
- **Neurogenesis**: ~700 new neurons per day per hippocampus. New neurons integrate via activity-dependent survival.
- **Protein Synthesis**: CREB acts as the master switch for permanent memory (L-LTP), and Arc stabilizes structural changes. Epigenetic marks (DNA methylation, Histone acetylation) provide long-term stability.

## 4. Neurotransmitters & Neuromodulation
Neuromodulators act as a global tuning system:
- **Acetylcholine (ACh)**: High ACh = encoding mode, Low ACh = retrieval/consolidation. Switches hippocampal states.
- **Dopamine (DA)**: Reward prediction error. Strengthens consolidation for unexpected outcomes (surprise).
- **Norepinephrine (NE)**: Arousal/urgency. High NE triggers flashbulb memories.
- **Serotonin**: Mood-plasticity coupling, modulates LTP/LTD threshold.
- **GABA**: Active forgetting and inhibitory control.

## 5. Engrams — Memory Traces
An **engram** is the physical trace of a memory—the specific ensemble of neurons representing it.
- **Excitability Competition**: Neurons compete based on CREB-driven intrinsic excitability to form an engram.
$$P(\text{allocation}) \propto \exp(\text{excitability}_i / \tau)$$
- **Lifecycle**: Active (0-6h) -> Stable (6h-30d) -> Silent (30-90d, exists but inaccessible without intervention) -> Degraded -> Dissolved.
- **Overlap**: Engrams sharing neurons create associative links between memories.

## 6. Cognitive Maps
The hippocampus and entorhinal cortex map both physical space and abstract concepts (social, temporal, conceptual spaces).
- **Place cells**: Fire at specific abstract or physical locations.
- **Grid cells**: Provide the periodic, universal metric system.
- **Successor Representations**: The brain models future states.
$$M(s, s') = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \cdot \mathbb{1}[s_t = s'] \mid s_0 = s\right]$$
Context-dependent recall means retrieving memories based on transition space, not just static semantic similarity.

## 7. Sleep & Consolidation
Sleep is the primary vehicle for Systems Consolidation, with multi-stage processing:
- **Theta-Gamma Code**: Theta (4-8 Hz) provides temporal sequencing, while nested Gamma (30-100 Hz) processes specific items.
- **SO-Spindle-Ripple Coupling**:
  - Neocortical Slow Oscillation (SO) triggers Thalamocortical Spindles.
  - Spindles nest Hippocampal Sharp-Wave Ripples (SWRs).
  - Hippocampus replays memories at 20x compression to transfer to the maximally plastic cortex.
- **REM**: Emotional integration and uncoupling of visceral emotion from informational memory. Generative replay creates novel combinations.

## 8. Emotion & Memory
Emotions dictate encoding priority via amygdala modulation:
- **Yerkes-Dodson Inverted-U**: Moderate arousal yields optimal memory (robust LTP), while extreme stress impairs memory, saturating receptors and suppressing LTP.
- **Cortisol**: Moderate levels enhance gene expression; extreme levels suppress LTP and cause atrophy.
- **Retrieval-Induced Forgetting**: Retrieving a memory actively suppresses competitors through inhibitory control.

## 9. Complementary Learning Systems
The brain solves the stability-plasticity dilemma with two systems:
- **Hippocampus**: Fast, sparse, one-shot episodic encoding. Pattern separation prevents catastrophic interference.
- **Neocortex**: Slow, distributed, overlapping semantic extraction.
- **C-HORSE & Schemas**: Schemas (vmPFC) accelerate consolidation from weeks to 48 hours for congruent information. 

## 10. Memory Disorders & Insights
- **H.M. & Amnesia**: Proves declarative and procedural memories are distinct systems.
- **Korsakoff's**: Demonstrates memory is reconstructive (confabulation fills gaps).
- **Alzheimer's**: Pathological decay of episodic first, followed by semantic. Old memories are more resilient (Ribot's Law). Highlights silent engrams.
- **PTSD**: Extreme emotion without proper temporal/contextual binding creates non-terminating feedback loops.
- **Infantile Amnesia**: Extreme neurogenesis overwrites early traces (plasticity vs. stability tradeoff).
- **Hyperthymesia**: Infinite, perfect memory is a cognitive failure mode. Active forgetting is essential.

## 11. Key Principles for AI Implementation
Building an exact biological replica (10^22 FLOP/s, 2.7 GW power) is impossible. We must build **functional equivalence** using the brain's computational principles.
- **Reconstruction > Retrieval**: 70-80% of recalled memory is generated using a sparse index and schemas.
- **Prediction Error (Surprise)**: Drives encoding.
$$\text{Surprise} = D_{KL}[P(\text{observed}) \| P(\text{predicted})]$$
- **Prospective Memory**: Intentions triggered by time/events, not just historical facts.
- **Metamemory**: AI must gauge Feeling-of-Knowing (FOK) and Tip-of-Tongue (TOT) confidence before deep retrieval.
- **Interference Resolution**: Proactive/Retroactive interference detectors must handle conflicting knowledge.
- **Storage Tiering**: Hybrid architecture with Working Memory (GPU RAM) -> Episodic (SSD) -> Semantic (Vector DB) -> Archival/Genetic (DNA storage, 215 PB/gram for ultra-long-term, infrequent access).

## 12. Brain by the Numbers

Verified biophysical scale and cellular turnover rates of the human brain:

### Scale & Architecture
- **Total Neurons**: **86 billion** (Herculano-Houzel 2009, isotropic fractionator)
  - **Cerebellum**: ~69 billion (~80% of all neurons, sensorimotor coordination)
  - **Cerebral Cortex**: ~16 billion (site of high-level cognition, working memory, and executive reasoning)
- **Synapses**: **100 trillion – 1 quadrillion** ($10^{14} - 10^{15}$), with ~7,000 synaptic connections per cortical neuron
- **Transcriptomic Subtypes**: **3,000+ distinct neuron types** (BRAIN Initiative Cell Census Network / BICCN)
- **Glial Cells**: **~85 billion** (astrocytes, oligodendrocytes, microglia providing metabolic support, insulation, and synaptic pruning)
- **Synaptic Events**: **~10 quadrillion ($10^{16}$) events/sec** of continuous electrochemical computation
- **Power Consumption**: **~20 Watts** (metabolic efficiency surpassing digital silicon by orders of magnitude)

### Neurogenesis & Synaptic Turnover
- **Adult Neurogenesis**: **~700 new neurons/day per hippocampus** (Spalding et al. 2013, measured via genomic $^{14}\text{C}$ dating)
- **Annual Turnover**: **~1.75% annual turnover** of the renewing fraction within Dentate Gyrus (DG) granule cells
- **Synaptogenesis Peak**: ~40,000 new synapses/second during early childhood development
- **Synaptic Pruning**: Massive elimination (~50% net loss) from age 11 to 25 shaping adult cortical circuit efficiency

---

## 13. Exact Brain Simulation Feasibility

Can we simulate an exact human brain? Computational scaling and empirical project outcomes reveal physical and energetic bounds.

### Four Levels of Simulation

| Level | Model Type | Full Brain Compute Required | Current Feasibility |
|:---|:---|:---|:---|
| **Level 1: Behavioral** | Artificial Neural Networks / LLMs | Current GPU clusters | **Achieved today** |
| **Level 2: Spiking** | Leaky Integrate-and-Fire (LIF) | **~1 ExaFLOP/s** ($10^{18}$ FLOP/s) | **Frontier supercomputers (barely)** |
| **Level 3: Biophysical** | Hodgkin-Huxley Multi-Compartment | **$10^{22} - 10^{25}$ FLOP/s** | **Physically impossible today** (10,000× gap) |
| **Level 4: Molecular** | Stochastic protein & ion channel dynamics | **$10^{25}+$ FLOP/s** | **Fundamentally intractable** |

### The 135 Million× Energy Gap
- **Human Brain Power**: **20 Watts** (equivalent to a dim household light bulb).
- **Equivalent Digital Simulation**: **~2.7 Gigawatts** (the continuous output of a nuclear power plant).
- **Energetic Discrepancy**: A **135,000,000× gap** in energy efficiency, indicating a fundamental architectural divergence between Von Neumann physics and non-equilibrium biological computation.

### Outcomes of Connectome & Brain Simulation Projects
- **C. elegans (OpenWorm)**: 302 neurons, ~7,000 synapses. Full wiring diagram mapped in 1986. After **38+ years**, simulation still cannot fully replicate emergent worm behavior; connectome structure alone lacks neuromodulatory state, chemical fields, and embodiment dynamics.
- **Blue Brain Project (EPFL)**: 31,000 neurons (one neocortical microcolumn). Concluded in 2024 after 10+ years; proved that hyper-detailed structural reconstruction does not spontaneously generate behavioral cognition.
- **Human Brain Project (EU)**: €1 billion initiative concluded in 2023. Failed to achieve full-brain simulation; pivoted to building EBRAINS research infrastructure.
- **Drosophila Connectome**: 140,000 neurons, 15 million synapses mapped in 2024. Behavior is not replicable from the static synaptic map alone.
- **Intel Hala Point**: 1.15 billion neuromorphic neurons (1,152 Loihi 2 chips), representing **~1.3% of the human brain**, consuming kilowatts without cognitive agency.

### Conclusion: The Strategic Viability of Path C
- **Path A (Exact Biological Replica)**: Atom-for-atom mapping and biophysical simulation is physically and energetically impossible.
- **Path B (Architectural Replica on Silicon)**: Simulating 86B spiking neurons requires billions of dollars and decades of Moore's Law scaling.
- **Path C (Functional Equivalence)**: **The only viable path.** Just as a Boeing 747 achieves flight using aerodynamics rather than flapping feathers, an artificial cognitive architecture must implement the brain's computational principles (Bayesian surprise, attractor completion, temporal context binding, sleep consolidation, and active forgetting) rather than simulating individual ion channels.

---

## 14. Reconsolidation & Lability

Memories are not write-once archives; retrieving a memory reactivates structural malleability.

### The Science & Mechanism (Nader et al. 2000)
When a consolidated memory is recalled under mismatched conditions or prediction error, it enters a transient **labile (destabilized) state**:
- **Lability Window**: A critical window lasting **~5 minutes** (up to a few hours in humans) during which the engram is sensitive to modification, disruption, or reinforcement.
- **Protein Synthesis Requirement**: Re-stabilizing (reconsolidating) the modified memory trace requires de novo protein synthesis (blocked experimentally by anisomycin). If protein synthesis is inhibited during the labile phase, the memory is permanently weakened or erased.

```
Memory Retrieval (Triggered by Mismatch / Cue)
      │
      ▼
Labile State (Unstable, ~5-minute window)
      │
      ├── Prediction Error / Novel Info Present ──► Engram UPDATED & Reconsolidated
      ├── Congruent Context / Pure Reinforcement ──► Engram RE-STABILIZED (Increased Stability S)
      └── Interruption / Inhibition ───────────────► Engram DEGRADED or Suppressed
```

### Reconsolidation Trigger Conditions
1. **Retrieval alone is insufficient**: Reconsolidation is triggered only when there is a **boundary condition of prediction error** (an unexpected cue or mismatch during recall).
2. **Mathematical Update Formulation**:
   $$M_{\text{updated}} = \alpha \cdot M_{\text{old}} + (1 - \alpha) \cdot M_{\text{new}}$$
   Where $\alpha \in [0, 1]$ depends on initial consolidation stability $S_{\text{old}}$: deeply consolidated memories exhibit high resistance to overwrite.

---

## 15. Metamemory Phenomena

Metamemory denotes the cognitive system's capacity to monitor, evaluate, and calibrate its own internal memory states (Koriat 1993, Fleming 2026).

| Phenomenon | Cognitive Definition | AI & Engineering Equivalent |
|:---|:---|:---|
| **Feeling-of-Knowing (FOK)** | The subjective assessment that an item is stored and knowable, even when immediate free recall fails. | Partial retrieval scoring; assessing graph neighbourhood density or high unthresholded embedding similarity prior to full generative search. |
| **Tip-of-the-Tongue (TOT)** | Intense FOK accompanied by acute recall failure, often with access to partial features (phonological length, initial letter, category). | High attractor convergence energy or partial slot matching where specific entity keys fail extraction thresholds. |
| **Judgment of Learning (JOL)** | Metacognitive prediction made during or immediately after encoding regarding future retrievability. | Predictive encoding quality score evaluating salience $\times$ prediction error $\times$ signal-to-noise ratio to predict retention probability. |
| **Confidence Calibration** | The statistical alignment between subjective retrieval confidence and objective factual accuracy. | Second-order probability calibration: preventing agent hallucinations by distinguishing genuine knowledge from low-margin nearest-neighbour approximations. |
