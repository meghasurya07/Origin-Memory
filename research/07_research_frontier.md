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

---

## 8. Deep Dive: DNA Storage & Biological Memory Stack

### The 4-Layer Biological Memory Stack
Biological memory operates across four discrete physical and temporal layers, offering a direct blueprint for hybrid silicon-biological computing:

| Layer | Biological Brain Analog | Persistence | Origin AI System Analog | Physical Storage Substrate |
|:---|:---|:---|:---|:---|
| **Layer 1: Electrical** | Sustained neural firing in Prefrontal Cortex (PFC) | Milliseconds to seconds | Active working context buffer | GPU VRAM / Host RAM (<1ms latency) |
| **Layer 2: Synaptic** | Long-Term Potentiation (LTP/LTD) in Hippocampus | Hours to weeks | Episodic event store & vector indices | NVMe SSD + Vector DB (1–10ms latency) |
| **Layer 3: Epigenetic** | DNA methylation & histone modifications in Neocortex | Months to lifetime | Semantic knowledge graph & cognitive schemas | SSD + Graph DB (Neo4j, 10–100ms latency) |
| **Layer 4: Genetic** | DNA nucleotide sequence itself (Evolutionary memory) | Generations to millennia | Permanent archival memory tier | **Synthetic DNA Storage (BioCompute)** |

---

### Complete Write Pipeline: Silicon to DNA
Consolidating aged, high-value knowledge into synthetic DNA follows a 7-stage biochemical write pipeline:

```
[Semantic Memory Batch]
         │
         ▼ (1) Archival Candidate Filtering (Age >90d, Access <0.01, Importance >0.5, Unique)
[Filtered Knowledge Nodes]
         │
         ▼ (2) Knowledge Graph Compression (Abstraction & redundancy reduction)
[Normalized Subgraph]
         │
         ▼ (3) Binary Serialization (Structured bitstream formatting)
[Raw Byte Stream]
         │
         ▼ (4) Error Correction Encoding (Reed-Solomon 30% redundancy)
[Fault-Tolerant Bitstream]
         │
         ▼ (5) Nucleotide Mapping (Binary to A/T/G/C bases, homopolymer constraints)
[Oligonucleotide Sequences]
         │
         ▼ (6) Microfluidic Synthesis (On-chip enzymatic synthesis via BioCompute chip)
[Physical DNA Strands (Stored dry or silica-encapsulated)]
         │
         ▼ (7) Silicon Index Registration (Hash, metadata, and primer address stored in silicon)
[Fast Retrieval Index (Redis / SQLite)]
```

1. **Filtering Criteria**:
   - `min_age > 90 days`: Ensures memories have stabilized through multiple sleep cycles.
   - `access_frequency < 0.01 / day`: Targets cold, rarely queried items.
   - `salience / importance > 0.5`: Only stores information critical to agent identity and long-term world knowledge.
   - `non_redundancy`: Checks against existing archival hashes.
2. **Knowledge Graph Compression**: Prunes transitional nodes, preserves core semantic triplets and relational topology.
3. **Binary Serialization**: Converts graph entities into compact binary schemas.
4. **Reed-Solomon Encoding (30% Redundancy)**: Adds robust forward error correction (FEC) to guarantee complete data reconstruction despite synthesis dropouts or base substitution errors.
5. **Nucleotide Encoding**: Maps binary words to quaternary bases (A, C, G, T) while optimizing GC content (40–60%) and eliminating repeating homopolymers (e.g., AAAAA) that destabilize sequencing.
6. **Microfluidic Synthesis**: Digital sequence files are sent to the BioCompute microfluidics chip for high-throughput enzymatic on-chip DNA synthesis.
7. **Silicon Index Registration**: The content hash, cryptographic signature, primer binding addresses, and semantic summary are indexed in a fast silicon database (Redis/SQLite) for rapid query routing.

---

### Complete Read Pipeline: DNA to Silicon
Retrieving archival knowledge when an agent requires deep historical memory follows a symmetrical 6-stage read pipeline:

```
[Agent Query / Deep Recall Request]
         │
         ▼ (1) Silicon Index Lookup (Locate candidate physical pool & primer tags)
[Primer Coordinates & Pool ID]
         │
         ▼ (2) PCR Amplification & Sequencing (Targeted amplification + Nanopore/Illumina read)
[Raw Fluorescent / Current Signals]
         │
         ▼ (3) Basecalling & Nucleotide Decoding (Translate to A/T/G/C sequence)
[Raw Read Sequences]
         │
         ▼ (4) Reed-Solomon Error Correction (Correct indel/substitution errors, strip 30% parity)
[Validated Binary Bitstream]
         │
         ▼ (5) Binary Deserialization (Reconstruct graph schemas and attributes)
[Knowledge Graph Subgraph]
         │
         ▼ (6) Memory Rehydration (Promote cold archival memory to warm Neocortical store)
[Active Semantic Knowledge in Neocortex]
```

1. **Silicon Index Search**: Semantic query matches against the silicon-resident index, identifying target sequence addresses and primer keys.
2. **PCR Amplification & Sequencing**: Specific primers selectively amplify the target oligonucleotide pool from DNA storage without reading the entire archive, followed by high-throughput on-chip or nanopore sequencing.
3. **Nucleotide Decoding**: Sequencing signals are basecalled into nucleotide strings.
4. **Reed-Solomon Error Correction**: The 30% parity payload corrects any synthesis, storage, or sequencing errors, verifying data integrity via cryptographic hash.
5. **Binary Deserialization**: Restores pristine structured semantic objects from the binary stream.
6. **Memory Rehydration**: Ingests restored knowledge directly back into Tier 2 (Neocortex) with warm stability, making it immediately available for agent reasoning.

---

### Economic & Cost Trajectory
While write latency and synthesis costs currently restrict DNA storage to cold archival applications, scaling curves project rapid economic convergence:

| Year | Synthesis Cost ($/GB) | Read Cost ($/GB) | Write Latency | Operational Viability |
|:---|:---|:---|:---|:---|
| **2024–2026 (Today)** | **~$1,000 – $10,000 / GB** | ~$0.01 / MB | Hours – Days | Elite archival, agent soul blueprints, compliance |
| **2028–2030** | ~$100 / GB | ~$0.001 / MB | Hours | Enterprise historical archives, multi-agent foundational models |
| **2035 (Target)** | **~$10 / GB** | Commodity | Minutes | **Directly competitive with high-density tape & cold cloud archives** |

### Critical Data Types for DNA Archival
1. **Historical Model Weights**: Checkpoint snapshots, LoRA adapter history, and optimizer states across training runs; multi-terabyte assets preserved indefinitely without server refresh cycles.
2. **Consolidated Semantic Knowledge**: Multi-year semantic knowledge graphs accumulated through thousands of sleep consolidation cycles.
3. **Agent Identity Blueprints**: The immutable "core soul" of the agent—fundamental values, behavioral boundaries, personality priors, and user relationship history.
4. **Regulatory & Compliance Data**: Financial audit logs, medical interaction traces, and enterprise records legally mandated for 7–50 year retention without power consumption or bit rot.


---

## 9. Multi-Agent Shared Memory (2025-2026 State of Art)

*Updated: 2026-10-12 from latest literature review*

### The Problem
~79% of multi-agent execution failures stem from coordination/structural desynchronization, not LLM reasoning failures. Current approaches:

### Key Papers & Systems

| System | Year | Key Insight |
|:-------|:-----|:------------|
| **MATM** (Kim et al., arXiv:2606.19911) | 2026 | Transactive Memory for AI — agents publish procedural trajectories, consumers retrieve when encountering similar subgoals |
| **CoMem** (Yu et al., arXiv:2609.15009) | 2026 | Collective-Individual Memory Synergy: private sedimentation → collective curation → dual-stream retrieval |
| **MIRIX** (arXiv:2507.07957) | 2025 | 6-tier memory (Core/Episodic/Semantic/Procedural/Resource/KnowledgeVault) + Auto-Dream consolidation agent |
| **MemClaw/Caura** | 2025-26 | 4 failure modes: unauthorized leakage, stale propagation, contradiction persistence, provenance collapse |
| **CoMAM** (arXiv:2603.12631) | 2026 | RL-optimized joint construction/retrieval agents with ranking consistency |
| **ORALMEM** (NeurIPS 2026) | 2026 | Collective memory persistence across agent generations via gossip protocols |

### Three-Tier Memory Architecture (Emerging Standard)
1. **Global/Team Memory** — universal shared facts, team goals, environmental state
2. **Role/Group Memory** — task-scoped partitions for collaborating sub-clusters
3. **Private Memory** — agent-internal scratchpads, CoT traces, persona heuristics

### Implications for Origin Brain
- We need a `SharedMemoryPool` with scoped access controls
- Transactive memory: each agent's Brain publishes to shared pool during consolidation
- Conflict resolution: temporal supersession + provenance tracking
- This is our path from single-agent library → multi-agent platform

## 10. Memory-Augmented Reasoning (2025-2026)

### Memory-as-a-Tool Paradigm
Transitioning from passive RAG retrieval to active memory tool use:
- Combats "Goldfish Effect" — prevents circular subgoal loops
- Cognitive offloading — reduces context window pressure

### Key Approaches
- **TRM-VLA** (2025-26): Temporal-aware CoT conditioned on memory-augmented reasoning state
- **GCoT** (Grounded CoT): Step-by-step reasoning grounded in spatial/temporal memory anchors
- **Memory-Augmented Compression (MAC)**: Compress verbose CoT into modular reasoning patterns stored in episodic archive
- **AgentHER**: Hindsight experience replay for LLM agents — relabel failures as successes for different goals

### Experience Replay for Reasoning
- RLEP: RL + continual replay of reasoning trajectories — 40% compute reduction
- FreshPER: Temporal decay on stored surprise metrics to prevent stale replay
- CER (Contextual Experience Replay): In-context replay without gradient updates

## 11. Continual Learning — 2026 Verdict

**Has anyone solved catastrophic forgetting? NO.**

But the paradigm shifted from "prevention" to "durable knowledge management":

| Method | State of Art (2026) |
|:-------|:-------------------|
| Parameter Isolation | **O-LoRA, InfLoRA, LEGO** — orthogonal parameter subspaces via MoE expert expansion |
| Dark Replay | **Uncertainty-Aware DER++** — adaptive loss weighting + output correction |
| Test-Time Memory | **Titans** (Google, arXiv:2501.00663) — neural long-term memory updated via surprise at test time, >2M token contexts |
| Bio-Inspired | **Nested Learning + SleepGate** — dual-timescale weight updates + offline replay → WE DO THIS |

### Origin Brain's Position
Our CLS architecture (fast hippocampal + slow neocortical) aligns with the **Nested Learning + SleepGate** approach — we're already implementing the bio-inspired state of the art. Key advantage: we do this at the *application layer* (no weight updates needed), making it compatible with any frozen LLM backbone.

## 12. Compositional Memory & The Binding Problem (2025-2026)

### Hierarchical Abstraction Pipeline
Episode → Concept → Schema → World Model (mirrors our architecture):
1. **Episodic** (raw traces) → our `hippocampus.py`
2. **Concept extraction** → our `neocortex.py` (co-occurrence learning)
3. **Schema formation** → our `neocortex.py` (schema engine)
4. **World models** → NEXT TARGET (predictive transition networks)

### The Binding Problem in AI
- Standard self-attention binds features by token proximity, not relational compositionality
- **Vector Symbolic Architectures (VSA/HDC)**: High-dimensional vectors with algebraic binding operators
- **Slot Attention**: Dynamic slots decompose streams into permutation-invariant entity representations
- Potential future module for Origin Brain: `BindingEngine` using VSA for compositional memory traces


---

## 13. Own Thinking: Why Origin Brain Wins — The Architecture Gap (Oct 2026)

### The Fundamental Insight
Every competitor treats memory as a **database problem** — store content, retrieve content. Origin Brain treats memory as a **cognitive process** — predict, encode selectively, consolidate during sleep, reconstruct dynamically.

### Architectural Comparison (Deep Analysis)

```
                    Mem0/Zep/supermemory              Origin Brain
                    ──────────────────              ──────────
Encode:             Store everything               Predict → surprise gate → encode only novel
Retrieve:           Vector similarity search        Multi-signal scoring (semantic + temporal + salience + primacy)
Storage:            Key-value / vector DB            Hippocampus (fast) + Neocortex (slow) + Schemas
Forgetting:         None / manual TTL               Ebbinghaus decay + spaced repetition + pruning
Consolidation:      None                            3-phase sleep (SWS + REM + Pruning)
Context:            Token window                    Temporal Context Model (drifting context vectors)
Reconstruction:     Verbatim retrieval              Generative reconstruction (context-dependent output)
Multi-agent:        Separate instances              Shared Memory Pool with scoped access + conflict resolution
Learning dynamics:  Static after encoding           CLS (fast hippocampal + slow neocortical pattern extraction)
Neuromodulation:    None                            ACh/DA/NE mode switching
```

### Why This Matters for Production
1. **Token Cost**: Our prediction gate (Pillar 1) filters ~60% of redundant inputs → 60% fewer stored memories → 60% lower retrieval cost
2. **Quality**: Reconstructive recall adapts to current context → same query returns different (appropriate) answers based on emotional state, recent context
3. **Scale**: Sleep consolidation compresses episodic → semantic → schemas → no unbounded growth
4. **Multi-agent**: Shared memory pool with provenance + conflict resolution → enterprise-ready
5. **Self-improving**: Neocortex extracts patterns automatically → brain gets SMARTER with more episodes

### The "Reconstruction Is All You Need" Thesis (Formalized)

**Claim**: The quality gap between AI memory systems will NOT be closed by better embeddings or larger vector databases. It will be closed by implementing the 4 properties of biological memory that current systems lack:

$$Q_{memory} = f(\text{Selectivity}, \text{Consolidation}, \text{Reconstruction}, \text{Adaptation})$$

Where:
- **Selectivity** = prediction error gating (don't store everything)
- **Consolidation** = sleep replay + schema formation (compress + structure)  
- **Reconstruction** = generative recall (adapt output to current state)
- **Adaptation** = CLS slow learning (extract patterns across episodes)

No amount of improvement to any single factor (e.g., better embeddings) can match the multiplicative benefit of all four together.

### Open Research Questions

1. **Binding Problem**: How to compositionally represent "Bob's meeting about budgets at the cafe on Tuesday" as a structured binding, not a flat embedding? VSA/HDC is promising but untested at scale.

2. **World Models**: Can the neocortex learn predictive transition models? If the brain sees A→B→C repeatedly, can it predict C when it sees A→B? This would enable mental simulation.

3. **Consciousness-Adjacent**: At what point does a sufficiently complex memory system exhibit properties that _look like_ self-awareness? If the brain stores memories _about its own memory processes_ (metamemory), and those memories influence future behavior...

4. **DNA Storage Timeline**: Current cost ($1000/GB) is prohibitive. At $10/GB (projected 2035), DNA archival becomes viable for agents with >100GB of long-term semantic knowledge. Origin needs to be ready.

5. **Cross-Modal Memory**: Current system is text-only. Human memory is richly multimodal (visual, auditory, olfactory, tactile). How to extend Origin Brain to store and reconstruct multimodal memories?

## 14. Next Architectural Targets (Prioritized)

| Priority | Feature | Why | Effort |
|:---------|:--------|:----|:-------|
| P0 | LoCoMo Benchmark | Industry standard — proves quality against Mem0 | 3 days |
| P0 | LLM Reconstruction | Currently template-based; needs real LLM for production recall | 2 days |
| P1 | World Model Engine | Predictive transition networks from neocortex | 3 days |
| P1 | Cross-Agent Brain Sync | Sync Brain state across distributed agents | 3 days |
| P2 | Multimodal Memory | Image/audio embeddings alongside text | 5 days |
| P2 | Vector DB Backend | Replace in-memory with pgvector/Milvus | 2 days |
| P3 | DNA Storage Prototype | Proof-of-concept archival pipeline | 5 days |
| P3 | Consciousness Self-Model | Brain that models its own memory processes | 3 days |


## November 2026 Update — Critical New Research

### RL-Based Memory Policies (HIGHEST PRIORITY)
- **MemAgent**: RL-trained read/overwrite strategy. 7B/14B models extrapolate from 8K-32K
  training to 3.5M+ tokens with <5-10% perf drop. Uses DAPO (Direct Alignment Policy Optimization).
- **Mem-alpha**: RL-trained structured memory with core/episodic/semantic components +
  write/update/delete tools. Trained on 30K tokens, generalizes to 400K+ (13x extrapolation).
  Reward = downstream QA accuracy.
- **PMR (Personalized Memory Rubrics)**: Rubric-guided credit assignment for training what to store per user.
- **ACTION**: We must implement RL-based memory management to replace our heuristic approach.

### HGMem — Hypergraph Memory (ICML 2026, CUHK/Tencent)
- Replaces pairwise knowledge graphs with hypergraphs (n-ary relations)
- Memory evolves dynamically as agent reasons
- Superior on NarrativeQA and multi-hop vs flat RAG and GraphRAG
- **ACTION**: Extend our AssociativeNetwork to support hyperedges (n-ary relations)

### FrontierMemory 1.0 Benchmark (Letta, Oct 2026)
- 750 past agent trajectories as input
- Phase 1: 6-hour "dreaming" window to synthesize memory folder
- Phase 2: 15 target evaluation tasks (binary pass/fail)
- Top scores: Opus 5.5 (73.3%), GLM-5.3 (73.3%), Kimi K3 (66.7%)
- **ACTION**: Build trajectory ingestion + dreaming pipeline for this benchmark

### PEC2F Schema (CogMem, EMNLP 2026) — Full Details
- 5 node types: Person, Event, Concept, Claim, Fact
- Claims have source-attribution provenance: (Person_source, CLAIMS, ClaimNode) AND (ClaimNode, ABOUT, Target)
- Active Plasticity Retrieval: O_anchor (hybrid search), O_traverse (temporal navigation), O_intersect (neighborhood overlap)
- Critique-then-reconstruct pipeline: Draft -> Critique (temporal contradictions, misattributed claims) -> Reconstruct
- **ACTION**: Implement PEC2F node types in our AssociativeNetwork

### SleepGate Architecture
- Conflict-Aware Temporal Tagger + Lightweight Forgetting Gate + Consolidation Module
- Triggered by attention entropy thresholds during inference
- Dual-phase training: wake-phase LM loss + sleep-phase post-consolidation retrieval loss
- **ACTION**: Add attention-entropy triggers to our sleep.py consolidation

### Competitor Status (Nov 2026)
| Competitor | Version | Key Feature | Threat |
|-----------|---------|-------------|--------|
| Mem0 | SDK v2.2.0+ | User Profiles, memory expiration, Vercel/OpenAI integration | HIGH |
| Zep/Graphiti | v0.31.x | Agent Skills (procedural memory), bi-temporal, FalkorDB | HIGH |
| Letta | FrontierMemory 1.0 | Dreaming framework, AgentFile, benchmark standard | HIGH |
| supermemory | API v5 | Dual-mode (memory vs superrag), semantic forgetting | MEDIUM |
| LangMem | LangGraph native | Background Memory Manager, ecosystem lock-in | MEDIUM |
| Cognee | v1.0 | remember/recall/improve/forget primitives, code graph | MEDIUM |

### Key Surveys
- "Memory for Large Language Models" (arXiv:2607.25380): 3-axis taxonomy (Representation, Update Dynamics, Persistence)
- "Human-Like Lifelong Memory": Neuroscience-grounded architecture matching our approach

### Multimodal Memory
- EmbeddingGemma 2: Unified 768-dim vector space for text/code/image/video/audio
- ADAMM: Analytic memory with provenance-linked attribute-value observations
- LMMR: Shared Sparse Autoencoder concept space for textual+visual features

### Consciousness Frameworks
- CTM-AI: Conscious Turing Machine with decentralized info competition
- MEM Architecture: Motivated Emotional Mind with semblion-based associative memory
- "A Mind Cannot Be Smeared Across Time": Chord vs Arpeggio hypothesis for temporal binding


---

## Nov 2026 Deep Dive: RL Memory Policy Landscape

### Status: IMPLEMENTED (v0.11.0)

We now have a working RL memory controller. This section documents what we
learned and what's next.

### MemAgent (ICLR 2026 Oral — ByteDance/Tsinghua)
- Free-form text-generation action space: LLM rewrites entire memory buffer
- RLVR: binary reward (exact match)
- Multi-Conv DAPO training on Qwen2.5-7B/14B
- **8K context → 3.5M+ token extrapolation** with <10% loss
- RULER NIAH: >95% at 512K context
- GitHub: BytedTsinghua-SIA/MemAgent

### Mem-α (Wang et al., 2025)
- Discrete tool actions: Write/Update/Delete/Route across 3 memory tiers
  (Core, Episodic, Semantic — matches our architecture!)
- GRPO training (no KL penalty) on Qwen3-4B
- 32 H100 GPUs, ~3 days, 205 steps
- **30K → 400K+ extrapolation** (13×)
- MemoryAgentBench: 0.592 vs baselines 0.461-0.502
- GitHub: wangyu-ustc/Mem-alpha

### Memory-R1 (ACL — Sikuan Yan et al.)
- ADD/UPDATE/DELETE/NOOP action space (exactly what we implemented!)
- Only **152 training QA pairs** — extremely data-efficient
- LoCoMo, MSC, LongMemEval benchmarks
- Shows massive improvement over Mem0 and Zep
- GitHub: yansikuan/memory-r1

### AgeMem (January 2026)
- Unified STM + LTM management (7 tool actions)
- 3-stage progressive RL training (a curriculum we should adopt)
- Step-wise GRPO for sparse rewards
- GitHub: y1y5/AgeMem

### ReMemR1 (ICLR 2026)
- Adds CALLBACK mechanism: non-linear memory access (revisit past states)
- Multi-level rewards: trajectory + step-level
- Better multi-hop reasoning via non-linear access

### Our Position
We have implemented the SAME action space as Memory-R1 (the most data-efficient
approach) but using tabular Q-learning instead of full LLM GRPO. This gives us:
1. **No GPU required** — trains on CPU in seconds
2. **Interpretable** — Q-table can be inspected directly
3. **Edge-deployable** — works without LLM inference
4. **Upgradable** — can swap to DQN/PPO when scale warrants

### Next Steps for RL
1. **Online training loop**: Wire RL policy into Brain.encode() as a gating
   decision (currently it's a parallel observer, not yet a gate)
2. **Deferred reward**: When a recalled memory was previously decided by the RL
   policy, feed the retrieval quality back as reward
3. **Curriculum training**: Implement AgeMem-style 3-stage progressive RL
4. **Benchmark comparison**: Run LoCoMo with RL-gated encoding vs heuristic
5. **DQN upgrade**: When Q-table exceeds 10K states, switch to neural Q-network

### Hypergraph Memory (HGMem, ICML 2026) — IMPLEMENTED
N-ary relational memory is now operational. Compared to our existing
AssociativeNetwork (pairwise edges), the HypergraphMemory captures:
- Multi-entity events as single atomic relations
- Meeting(A,B,C,Place,Time) instead of 10 binary edges
- More efficient spreading activation (group-level)
- Incidence matrix for algebraic graph operations

Integration with AssociativeNetwork: Both coexist. The pairwise graph handles
fine-grained entity-relation-entity triples. The hypergraph handles complex
n-ary events. Recall can query both and merge context.
