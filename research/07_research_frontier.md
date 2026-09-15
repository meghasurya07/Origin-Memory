# Origin Brain — Research Frontier & Open Questions

## 1. The Breakthrough Thesis (reconstruction is all you need)
Memory is not a database. Memory is not storage. Memory is not retrieval. Memory is a generative reconstruction engine.
- **Predictive Memory Is All You Need**: Encode selectively based on surprise, bind to temporal context, forget strategically to generalize, and consolidate during sleep.
- **Reconstruction Is All You Need**: Recall is GENERATED from sparse traces conditioned on current context, schemas, and emotional state. A conditional generative model: (\hat{x} | m, c, s, e) = \text{Decoder}(z; \theta)$.
- The memory is not the stored data. The memory is the RECIPE for reconstructing the experience.

## 2. DNA Storage & Biocompute (feasibility, path forward)
DNA data storage offers extreme density (215-455 PB / gram) and longevity (100-1,000+ years), with zero energy for maintenance.
- **BioCompute (Anagha Rajesh)**: Building end-to-end integrated microfluidics chips for on-chip DNA synthesis, storage, and sequencing.
- **Origin AI's Tier 3**: Archival Memory using DNA storage for historical model weights, consolidated semantic knowledge, agent identity blueprints, and cross-generational knowledge. Write once, read rarely, store forever.

## 3. Unsolved Problems (catastrophic forgetting, schema learning, etc.)
- **Complementary Learning Systems (CLS)**: Fast (Hippocampus) and slow (Neocortex) learning coordination. Missing in AI: dynamic learning rate based on schema consistency.
- **Memory Allocation (CREB)**: Neurons compete based on CREB excitability, causing time-linking effects.
- **Grid Cells and Concept Spaces**: Brain maps continuous conceptual geometry (time, topic, emotion), moving beyond flat databases.
- **Systems Consolidation**: Trace Transformation Theory suggests hippocampus retains detail while neocortex extracts gist.
- **Working Memory Capacity**: Theta-gamma coupling dictates the ~4 item limit.

## 4. Generative Models of Memory (CVAE, Hopfield, autoencoders)
- **VAEs as Memory Models**: ELBO as Memory Equation. {KL}$ acts as forgetting/compression. Conditional VAEs for context-dependent recall.
- **Generative Replay**: Hippocampus generates pseudo-data from past tasks to train cortex (sleep replay).
- **World Models**: Predictive models like Dreamer and JEPA show that memory can be an implicit compressed world model.
- **Sparse Distributed Memory (SDM)**: Kanerva Machine acts as a VAE with sparse, content-addressable memory, mirroring CA3 autoassociative network.
- **Unified Memory Equation**: $\mathcal{L} = \mathbb{E}[\log p(x|z, c)] - \beta(t) \cdot D_{KL}(q(z|x) \| p(z)) + \lambda \cdot \text{PE}(x)$.

## 5. Computational Models (CLS, C-HORSE, SPEAR)
- **ACT-R (Anderson)**: Base-level activation tracking usage history.
- **SOAR**: Chunking mechanism for procedural learning.
- **MINERVA 2**: Multiple-trace model where semantic knowledge emerges from episodic echo content.
- **SAM**: Search of Associative Memory (competitive retrieval).
- **CMR (Polyn et al.)**: Context Maintenance and Retrieval extends TCM for contiguity and source clustering.
- **Global Matching Models (TODAM, CHARM)**: Memory as a single high-dimensional composite vector (convolution/correlation).

## 6. One-Shot vs Forgetting (tradeoffs)
- **Elastic Weight Consolidation (EWC)**: Synaptic metaplasticity to protect important weights, analogous to Fisher Information.
- **Sparse Distributed Memory (SDM)**: Sparsity prevents interference (DG pattern separation step before CA3 storage).
- **Neuromodulatory Gating**: Acetylcholine (ACh) mode switch between encoding (high ACh) and consolidation/retrieval (low ACh). Dopamine for surprise, Norepinephrine for arousal.
- **Reconsolidation**: Only triggered by prediction errors. Destabilized memories have a lability window where they can be modified or erased.

## 7. Future Research Directions
- Implement neuronal allocation with excitability scores.
- True generative replay (generate synthetic replay examples).
- ACh-like mode switching to prevent encoding-retrieval interference.
- LLM-powered reconstructive decoder.
- Neuromorphic and DNA storage integration.
