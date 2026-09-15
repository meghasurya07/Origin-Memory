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
