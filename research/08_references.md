# Origin Brain — References & Bibliography

> Key papers, researchers, and resources cited throughout the project.

---

## Core Memory Neuroscience

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:----------------|
| The organization of behavior | D.O. Hebb | 1949 | Hebbian learning: "neurons that fire together wire together" |
| Levels of processing | Craik & Lockhart | 1972 | Encoding depth determines retention |
| Forgetting curves | H. Ebbinghaus | 1885 | Exponential decay of memory over time |
| The magical number seven | G.A. Miller | 1956 | Working memory capacity limits |
| Working memory capacity | N. Cowan | 2010 | Revised to 4±1 items |
| Retrieval-induced forgetting | M.C. Anderson | 1994 | Retrieving one memory inhibits competitors |
| Free recall serial position | B.B. Murdock | 1962 | U-shaped primacy/recency curve |

## Engrams & Memory Traces

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:----------------|
| Memory engrams: past and future | Josselyn & Tonegawa | 2020 | Comprehensive engram review |
| Optogenetic recall of engrams | Liu, Ramirez et al. | 2012 | First engram reactivation (Tonegawa lab) |
| CREB and memory allocation | Josselyn et al. | 2014 | Excitability competition for engram cells |
| Systems consolidation of engrams | Tonegawa lab | 2024 | Engram complexes across brain regions |

## Complementary Learning Systems (CLS)

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:----------------|
| Why there are CLS in hippocampus | McClelland, McNaughton, O'Reilly | 1995 | Fast hippocampal + slow neocortical learning |
| C-HORSE model | (CLS 2025) | 2025 | Hippocampus has its own CLS: TSP + MSP |
| VAE + Modern Hopfield | (2025) | 2025 | Solves catastrophic forgetting via energy-based memory |

## Cognitive Maps & Spatial Memory

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:----------------|
| The hippocampus as a cognitive map | O'Keefe & Nadel | 1978 | Place cells and spatial memory |
| Grid cells in entorhinal cortex | Moser & Moser | 2005 | Multi-scale spatial metric |
| Hippocampus as predictive map | Stachenfeld, Botvinick, Gershman | 2017 | Successor representations in hippocampus |
| What is a cognitive map? | Behrens et al. | 2018 | Abstract cognitive maps beyond space |

## Neuromodulation

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:----------------|
| SPEAR model | Hasselmo | 2025 | ACh modulates encoding/retrieval within theta |
| Dopamine RPE | Schultz | 2024 | Consolidation as offline RL with DA evaluation |
| Norepinephrine and memory | McGaugh | 2000 | Arousal enhances encoding via NE/amygdala |

## Synaptic Plasticity

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:----------------|
| STDP | Bi & Poo | 1998 | Spike-timing dependent plasticity |
| BCM theory | Bienenstock, Cooper, Munro | 1982 | Sliding threshold for LTP/LTD |
| Synaptic tagging | Frey & Morris | 1997 | Early/late LTP, protein synthesis requirement |

## Memory Reconstruction

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:----------------|
| DRM false memory paradigm | Roediger & McDermott | 1995 | Schema-consistent false memories |
| Constructive memory | Schacter, Addis, Buckner | 2007 | Memory as constructive, not reproductive |
| Pattern completion in CA3 | Rolls | 2013 | Autoassociative network for pattern completion |

## AI Memory Systems

| System | Organization | Year | Key Contribution |
|:-------|:------------|:-----|:----------------|
| Mem0 | Mem0.ai | 2024 | Graph-based AI memory, $24M Series A |
| supermemory.ai | SuperMemory | 2024 | Context infrastructure for AI |
| Zep | Zep.ai | 2024 | Long-term memory for AI assistants |
| LoCoMo benchmark | ACL 2024 | 2024 | Long-context conversation memory evaluation |

## Benchmarks

| Benchmark | Source | What It Tests |
|:----------|:-------|:-------------|
| OriginBench v0.1 | Origin AI | 5 cognitive benchmarks (HCSI) |
| LoCoMo | ACL 2024 | 1540-1986 questions, 5 QA types |
| LongMemEval | 2024 | Long-term memory evaluation |

---

## Key Researchers to Follow

- **Sheena Josselyn** (SickKids, Toronto) — Engrams, CREB
- **Susumu Tonegawa** (MIT) — Engram reactivation, systems consolidation
- **Michael Hasselmo** (BU) — ACh, theta oscillations, SPEAR model
- **Wolfram Schultz** (Cambridge) — Dopamine, reward prediction error
- **Eleanor Maguire** (UCL) — Hippocampal navigation, London taxi drivers
- **Howard Eichenbaum** (BU, deceased) — Time cells, relational memory
- **May-Britt & Edvard Moser** (NTNU) — Grid cells, cognitive maps


## Information Theory & Memory

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:-----------------|
| The Information Bottleneck Method | Tishby, Pereira, Bialek | 1999 | Optimal lossy compression preserving task-relevant information |
| Rate-distortion theory | Shannon | 1959 | Fundamental limits of lossy compression |
| Deep Variational Information Bottleneck | Alemi et al. | 2017 | Neural IB for representation learning |
| Lost in the Middle | Liu et al. | 2023 | LLMs fail to retrieve from middle of long contexts |
| Attention Is All You Need | Vaswani et al. | 2017 | Transformer attention (= Hopfield update) |

## Modern Hopfield Networks

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:-----------------|
| Hopfield Networks Is All You Need | Ramsauer, Schäfl et al. | 2020 | Exponential capacity Modern Hopfield, Attention ≡ Hopfield identity |
| Dense Associative Memories | Krotov & Hopfield | 2016 | Polynomial energy Hopfield with higher capacity |
| Classical Hopfield Networks | Hopfield | 1982 | Original associative memory network, capacity N/(2 ln N) |

## RL for Memory Management

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:-----------------|
| MemAgent: Leveraging LLM-Agent RL | (ICLR 2026) | 2026 | RL agent learns memory operations via GRPO |
| Mem-α (Agentic Memory) | (NeurIPS 2025) | 2025 | Memory as learned behavior, α-transfer across tasks |
| Memory-R1 | (ACL 2025) | 2025 | RL-trained memory with reasoning traces |
| AgeMem (Age of Memory) | (2026) | 2026 | Temporal-aware memory management with age-based RL |
| ReMemR1 | (ICLR 2026) | 2026 | Retrieval-enhanced memory with RL training |
| Route to Remember | (2025) | 2025 | Routing-based memory architecture with learned policies |
| MIRA | (2025) | 2025 | Memory-Integrated Reasoning Agent |
| Q-learning convergence | Watkins & Dayan | 1992 | Tabular Q-learning converges to Q* for finite MDPs |
| PAC-MDP sample complexity | Strehl, Li, Littman | 2006 | Bounds for RL sample efficiency |

## Sleep & Consolidation

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:-----------------|
| Sleep and memory consolidation | Born & Wilhelm | 2012 | Two-stage consolidation model (SWS replay + REM integration) |
| Sleep's role in memory | Walker & Stickgold | 2010 | Active systems consolidation during sleep |
| Synaptic homeostasis hypothesis | Tononi & Cirelli | 2006 | SWS downscales synapses, enhancing signal-to-noise |
| Predictive Processing (Free Energy) | Friston | 2005, 2010 | Memory minimizes variational free energy |

## Cognitive Architecture & Memory Systems (2025–2026)

| Paper | Authors | Year | Key Contribution |
|:------|:--------|:-----|:-----------------|
| MemoryOS | (2025) | 2025 | OS-inspired hierarchical memory for LLM agents |
| CogMem | (2025) | 2025 | Cognitive memory with PEC2F node types |
| HGMem | (2025) | 2025 | Hypergraph-based relational memory |
| SleepGate | (2025) | 2025 | Entropy-triggered consolidation |
| FrontierMemory 1.0 | Letta | 2026 | Agentic memory benchmark (trajectory + dreaming) |

## Key Equations Sources

| Equation | Source | Used In |
|:---------|:-------|:--------|
| TCM update: $\mathbf{t}_i = \rho_i \mathbf{t}_{i-1} + \beta \mathbf{t}^{IN}_i$ | Howard & Kahana (2002) | `temporal_context.py` |
| Spaced repetition: $S_{n+1} = S_n \cdot (1 + \alpha \cdot e^{wS_n})$ | Wozniak (SuperMemo) | `decay.py` |
| ACT-R activation: $B_i = \ln(\sum_k (t-t_k)^{-d})$ | Anderson et al. (2004) | `decay.py` |
| Spreading activation: $a_j = \sum_i w_{ij} a_i d^{-\text{dist}}$ | Collins & Loftus (1975) | `associative_network.py` |
| Modern Hopfield energy: $E = -\beta^{-1}\ln\sum \exp(\beta x^T\xi_i) + \frac{1}{2}\|x\|^2$ | Ramsauer et al. (2020) | `pattern_completion.py` |
| Free energy: $F = D_{KL}[q(s|o) \| p(s)] - \ln p(o|s)$ | Friston (2005) | `prediction.py` |
| Q-learning: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ | Watkins (1992) | `rl_policy.py` |
| Shannon entropy: $H(\mathbf{p}) = -\sum p_i \log_2 p_i$ | Shannon (1948) | `entropy_monitor.py` |


### Information Retrieval — BM25 and Probabilistic Models (v0.12.0)

- Robertson, S.E. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 3(4), 333–389. — **The definitive BM25 reference.**
- Robertson, S.E. & Walker, S. (1994). "Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted Retrieval." *SIGIR '94*. — Derives BM25's length normalisation from 2-Poisson model.
- Robertson, S.E. & Spärck Jones, K. (1976). "Relevance Weighting of Search Terms." *Journal of the American Society for Information Science*, 27(3), 129–146. — The RSJ IDF weighting used in BM25.
- Harter, S.P. (1975). "A Probabilistic Approach to Automatic Keyword Indexing." *JASIS*, 26(4), 197–206. — The 2-Poisson term frequency distribution underlying BM25.
- Hunt, R.R. & Worthen, J.B. (2006). *Distinctiveness and Memory*. Oxford University Press. — Distinctiveness principle in human memory retrieval.

### Morphological Processing — Neurolinguistic Basis

- Bozic, M., Marslen-Wilson, W.D., et al. (2010). "Differentiating Morphology, Form, and Meaning: Neural Correlates of Morphological Complexity." *Journal of Cognitive Neuroscience*, 22(7), 1481–1497. — Left inferior frontal gyrus morphological decomposition.
- Marslen-Wilson, W.D., Bozic, M., & Randall, B. (2008). "Early Decomposition in Visual Word Recognition: Dissociating Morphology, Form, and Meaning." *Language and Cognitive Processes*, 23(3), 394–421. — Automatic morphological stripping during lexical access.

### Key Equations (v0.12.0 additions)

| Equation | Source | Used In |
|:---------|:-------|:--------|
| BM25(Q,D) = Σ IDF(t) · f(t,D)(k₁+1) / [f(t,D) + k₁(1-b+b·\|D\|/avgdl)] | Robertson & Zaragoza 2009 | `embedding.py:_bm25_score()` |
| IDF(t) = ln((N-df+0.5)/(df+0.5)+1) | Robertson & Spärck Jones 1976 | `embedding.py:_bm25_idf()` |
| σ(s) = 2/(1+e^{-0.8s}) - 1 | Custom (calibrated on corpus) | `embedding.py:_bm25_score()` normalisation |
