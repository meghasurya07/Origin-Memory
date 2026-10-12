# Origin Brain — Product Strategy

> Competitive positioning, gap analysis, and go-to-market for Origin Brain by Origin AI.

---

## The Market Problem

**AI agents are stateless.** Every conversation starts from scratch. Current solutions (RAG, vector databases) provide retrieval but not intelligence — they don't forget, consolidate, reconstruct, or learn from experience.

---

## Competitive Landscape

### Direct Competitors

| Feature | **Origin Brain** | **Mem0** | **supermemory.ai** | **Zep** |
|:--------|:----------------|:---------|:-------------------|:--------|
| Forgetting curves | ✅ Ebbinghaus | ❌ | ❌ | ❌ |
| False memories | ✅ Schema-based | ❌ | ❌ | ❌ |
| Context-dependent recall | ✅ Reconstructive | ❌ | ❌ | ❌ |
| Sleep consolidation | ✅ SWS+REM | ❌ | ❌ | ❌ |
| Working memory limits | ✅ 4-7 items | ❌ | ❌ | ❌ |
| Neuromodulation | ✅ ACh/DA/NE | ❌ | ❌ | ❌ |
| Engram tracking | ✅ CREB model | ❌ | ❌ | ❌ |
| Cognitive map | ✅ Successor rep | ❌ | ❌ | ❌ |
| Synaptic plasticity | ✅ STDP+BCM | ❌ | ❌ | ❌ |
| Neurogenesis | ✅ Dynamic neurons | ❌ | ❌ | ❌ |
| Neural embeddings | ❌ (TF-IDF) | ✅ | ✅ | ✅ |
| LLM integration | ❌ (planned) | ✅ | ✅ | ✅ |
| Production SDKs | 🟡 Python only | ✅ Multi | ✅ TS/Py/Go | ✅ |
| LoCoMo benchmark | ❌ (planned) | ~92.5% | N/A | N/A |

### Competitor Details

**Mem0** (YC W24, $24M Series A)
- Graph-based memory with entity extraction
- ~92.5% LoCoMo, ~94.4% LongMemEval
- p95 latency ~1.4s
- **Gap**: No cognitive processes — it's a smart database, not a brain

**supermemory.ai**
- Context infrastructure, sub-300ms retrieval
- \$0-399/mo pricing tiers
- SDKs: TypeScript, Python, Go
- **Gap**: No memory science — just fast RAG

---

## Our Unique Position

**Origin Brain is the ONLY system that models memory as a cognitive process.**

No competitor has:
- Forgetting (strategic, not a bug)
- Reconstruction (memories change based on context)
- Consolidation (memories reorganize during "sleep")
- Capacity limits (working memory overflow)
- Neuromodulation (attention/surprise/arousal effects)
- Engram lifecycle (memories can go silent and be reactivated)

This isn't an incremental improvement — it's a **category-defining difference**.

---

## Go-To-Market Strategy

### Phase 1: Open-Source Core (NOW → Q1 2027)
- `pip install origin-brain`
- GitHub with good docs, examples, benchmarks
- Build community, get feedback
- Target: AI agent developers, researchers

### Phase 2: Cloud Service (Q1 2027 → Q3 2027)
- Hosted Origin Brain API
- Per-agent billing
- Dashboard for memory visualization
- Target: Startups building AI agents

### Phase 3: Enterprise + Partnerships (Q3 2027+)
- SOC2, HIPAA compliance
- On-premises deployment
- Partner with frontier AI companies
- Target: Enterprise AI teams

### Pricing Model (Proposed)
| Tier | Price | What |
|:-----|:------|:-----|
| Free | \$0 | 1 agent, 10K memories, community support |
| Pro | \$49/mo | 10 agents, 100K memories, REST API |
| Team | \$199/mo | 50 agents, 1M memories, priority support |
| Enterprise | Custom | Unlimited, on-prem, SLA |

---

## Partnership Readiness

**Can we partner with frontier AI companies?** YES.

Our architecture is designed for this:
1. **Clean SDK** — `OriginBrainClient` with simple API
2. **REST API** — Language-agnostic integration
3. **Modular** — Companies can use specific engines (just forgetting, just consolidation)
4. **Benchmarked** — HCSI 1.00, competitive data available
5. **Research-backed** — 50+ citations, neuroscience foundations

---

## Critical Gaps to Close

| Gap | Priority | Effort | Impact |
|:----|:---------|:-------|:-------|
| Neural embeddings (sentence-transformers) | 🔴 HIGH | 1 week | 10x recall quality |
| LLM integration | 🔴 HIGH | 2 weeks | Required for LoCoMo |
| LoCoMo benchmark | 🔴 HIGH | 2 weeks | Industry credibility |
| TypeScript SDK | 🟡 MEDIUM | 1 week | Web developer adoption |
| Cloud hosting | 🟡 MEDIUM | 2 weeks | SaaS revenue |
| Documentation site | 🟡 MEDIUM | 1 week | Developer experience |

---

## Competitor Deep Dive: Metacognition (SST)

### Team & Background
- **Founding Team**: Krish Jaiswal, Priyam Ghosh, Sauhard Gupta from the Scaler School of Technology (SST) in Bengaluru, India.
- **Origins**: Started from a student research initiative ("Memora") exploring research paper consumption, evolving after domain expert feedback into an academic and commercial effort targeting AI memory loss.
- **Core Product**: **"Tex"**, their proprietary memory system claiming state-of-the-art performance on memory benchmarks across AI agent evaluations.

### Research Output & Core Papers
The SST team is the only other known group approaching agent memory from a rigorous neuroscience-first, mathematical perspective:
1. **"Differential Memory Decay: A Continuous-Time Model of Forgetting and Semantic Drift"**: Proposes continuous-time differential equations modeling how memories decay according to Ebbinghaus-style curves and undergo semantic drift into broader attractors.
2. **"The Dynamics of Memory Replay, Diffusion and Attractor Formation"**: An energy-based formulation modeling how episodic memories evolve into semantic attractors through replay, stochastic diffusion, and attractor basins.

### Academic Recognition & Conference Acceptances
- **International Conference on the Mathematics of Neuroscience and AI**: Research framework accepted.
- **IEEE ICACI** (International Conference on Advanced Computational Intelligence, China): Paper presentation.
- **Rome Neuroscience Conference**: Presentation of memory dynamics framework.
- **Advisory & Academic Ties**: Guidance from advisors connected to **UC Berkeley** and the **MIT Neuroscience Lab**.

### Strategic Differentiation: Origin AI vs. Metacognition
While Metacognition brings strong algorithmic rigor, Origin AI holds distinct structural advantages:

| Dimension | Metacognition (Tex) | Origin AI (Origin Brain) |
|:---|:---|:---|
| **Architectural Scope** | Software-only memory attractor layer | **Full-Stack Brain Architecture** (PFC working memory, Hippocampal CA1/CA3/DG circuits, Neocortical schemas, Amygdala salience) |
| **Physical Substrate** | Conventional silicon only | **Hybrid Silicon + DNA Storage Substrate** (BioCompute molecular tier for infinite-lifespan archival memory) |
| **Cognitive Mechanisms** | Attractor decay & diffusion equations | **Reconstructive Generative Recall**, neuromodulation (ACh/DA/NE gating), and dream/sleep consolidation |
| **Biological Fidelity** | Algorithmic approximation of decay | End-to-end biological functional equivalence across episodic, semantic, procedural, and genetic tiers |

**Summary**: Metacognition is developing a software-only mathematical attractor model for memory decay. Origin AI builds a full-stack cognitive brain—coupling biological memory dynamics (reconstruction, neuromodulatory gating, and sleep cycles) with a breakthrough DNA storage hardware substrate.
