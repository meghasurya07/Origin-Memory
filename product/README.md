<p align="center">
  <h1 align="center">🧠 Origin Brain</h1>
  <p align="center"><strong>Human-Like Memory Infrastructure for AI Agents</strong></p>
  <p align="center">
    <em>Not a database. A brain.</em>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/origin-brain/"><img src="https://img.shields.io/pypi/v/origin-brain?color=blue" alt="PyPI"></a>
  <a href="https://github.com/meghasurya07/Origin-Memory/actions"><img src="https://github.com/meghasurya07/Origin-Memory/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/meghasurya07/Origin-Memory"><img src="https://img.shields.io/github/stars/meghasurya07/Origin-Memory?style=social" alt="Stars"></a>
  <a href="https://github.com/meghasurya07/Origin-Memory/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"></a>
</p>

---

AI agents have **digital amnesia** — they forget everything between sessions. Current "memory" solutions are just databases with vector search.

Origin Brain is different. It's the first memory system built on **computational neuroscience** — with prediction-gated encoding, sleep consolidation, forgetting curves, reconstructive recall, and schema learning. It works like an actual human brain.

## ⚡ Quick Start (2 minutes)

```bash
pip install origin-brain
```

```python
from origin_brain import Brain, BrainConfig

# Create a brain for your agent
brain = Brain(config=BrainConfig(agent_id="my_agent"))

# Encode memories (prediction gate filters redundant info automatically)
brain.encode("User prefers dark mode and Python")
brain.encode("User is building a RAG pipeline for healthcare data")
brain.encode("Meeting with Alice about Q3 budget was moved to Friday")

# Recall with context-dependent reconstruction
result = brain.recall("What does the user prefer?")
for r in result.results:
    print(f"  {r.memory.content} (relevance: {r.relevance_score:.2f})")

# Sleep consolidation — compress, strengthen, forget
report = brain.sleep()
print(f"Consolidated: {report.total_consolidated}, Pruned: {report.total_evicted}")

# Query learned associations from the semantic network
associations = brain.query_semantic_network("budget")
# → [("meeting", 0.8), ("alice", 0.7), ("friday", 0.5)]

# Predict future states from learned patterns
predictions = brain.predict_next("meeting")
# → [("discussion", 0.6), ("decision", 0.3)]
```

## 🧬 What Makes Origin Brain Different

Every competitor treats memory as a **database problem** — store content, retrieve content. We treat it as a **cognitive process**.

| Capability | Mem0 | Zep | Letta | **Origin Brain** |
|:-----------|:----:|:---:|:-----:|:---------------:|
| Prediction-gated encoding | ❌ | ❌ | ❌ | ✅ |
| Sleep consolidation (SWS + REM) | ❌ | ❌ | ❌ | ✅ |
| Forgetting curves (Ebbinghaus) | ❌ | ❌ | ❌ | ✅ |
| Reconstructive recall | ❌ | ❌ | ❌ | ✅ |
| Complementary Learning Systems | ❌ | ❌ | ❌ | ✅ |
| World model (state prediction) | ❌ | ❌ | ❌ | ✅ |
| Neuromodulation (ACh/DA/NE) | ❌ | ❌ | ❌ | ✅ |
| Schema formation | ❌ | ❌ | ❌ | ✅ |
| Multi-agent shared memory | ❌ | ❌ | ❌ | ✅ |
| Synaptic plasticity (STDP) | ❌ | ❌ | ❌ | ✅ |
| Cognitive maps | ❌ | ❌ | ❌ | ✅ |
| Working memory (attention) | ❌ | ❌ | Partial | ✅ |

**Our thesis**: *"Reconstruction Is All You Need"* — the quality gap between AI memory systems won't be closed by better embeddings. It requires implementing the 4 properties of biological memory: **Selectivity**, **Consolidation**, **Reconstruction**, and **Adaptation**.

## 🏗️ Architecture

Origin Brain implements a full computational neuroscience stack:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORIGIN BRAIN v0.8                        │
├─────────────────────────────────────────────────────────────┤
│  Working Memory    │  Attention-gated buffer (capacity: 7±2)│
├────────────────────┼────────────────────────────────────────┤
│  Prediction Engine │  Surprise = |actual - predicted|       │
│                    │  Only encode what's NOVEL              │
├────────────────────┼────────────────────────────────────────┤
│  Hippocampus       │  Fast episodic encoding (one-shot)     │
│                    │  Pattern separation + completion       │
├────────────────────┼────────────────────────────────────────┤
│  Neocortex         │  Slow statistical learning (CLS)       │
│                    │  Schema extraction from episodes       │
├────────────────────┼────────────────────────────────────────┤
│  World Model       │  Predictive state transitions          │
│                    │  Mental simulation + surprise           │
├────────────────────┼────────────────────────────────────────┤
│  Sleep Engine      │  SWS replay → REM integration → Prune │
│                    │  Hippocampus → Neocortex transfer      │
├────────────────────┼────────────────────────────────────────┤
│  Reconstruction    │  Context-dependent generative recall   │
│                    │  Same query → different output          │
├────────────────────┼────────────────────────────────────────┤
│  Shared Memory     │  Multi-agent pool with scoped access   │
│                    │  Conflict resolution + provenance       │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 Integrations

### With LangChain

```python
from origin_brain import Brain, BrainConfig

brain = Brain(config=BrainConfig(agent_id="langchain_agent"))

# Use as a memory backend in your LangChain agent
class OriginMemory:
    def __init__(self, brain):
        self.brain = brain

    def save_context(self, inputs, outputs):
        self.brain.encode(f"User: {inputs['input']}\nAI: {outputs['output']}")

    def load_memory_variables(self, inputs):
        result = self.brain.recall(inputs.get("input", ""), top_k=3)
        history = "\n".join(r.memory.content for r in result.results)
        return {"history": history}
```

### With CrewAI

```python
from origin_brain import Brain, BrainConfig, SharedMemoryPool, MemoryScope

# Each agent gets its own brain
researcher_brain = Brain(config=BrainConfig(agent_id="researcher"))
writer_brain = Brain(config=BrainConfig(agent_id="writer"))

# Shared pool for cross-agent knowledge
pool = SharedMemoryPool()
pool.join_group("researcher", "content_team")
pool.join_group("writer", "content_team")

# Researcher discovers something
pool.publish("researcher", "key_finding", "AI memory market is $7.8B",
             scope=MemoryScope.GROUP, group_id="content_team")

# Writer queries shared knowledge
results = pool.query("writer", key="key_finding")
```

### REST API

```bash
# Start the server
uvicorn origin_brain.server:app --host 0.0.0.0 --port 8000

# Encode
curl -X POST http://localhost:8000/v1/encode \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent_1", "content": "User likes Python and dark mode"}'

# Recall
curl -X POST http://localhost:8000/v1/recall \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent_1", "query": "user preferences", "top_k": 5}'

# Sleep consolidation
curl -X POST http://localhost:8000/v1/sleep \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent_1"}'
```

## 📊 Benchmarks

**OriginBench HCSI (Human-Cognitive Similarity Index): 1.00**

| Property | What We Test | Result |
|:---------|:-------------|:-------|
| Selective Encoding | Prediction gate filters redundant input | ✅ PASS |
| Temporal Decay | Memories weaken over time (Ebbinghaus) | ✅ PASS |
| Reconstructive Recall | Same query, different context → different output | ✅ PASS |
| Sleep Consolidation | Episodic → semantic compression during sleep | ✅ PASS |
| Interference (RIF) | Recalling X weakens competing memory Y | ✅ PASS |

## 🧪 Development

```bash
# Clone
git clone https://github.com/meghasurya07/Origin-Memory.git
cd Origin-Memory/product

# Install (editable)
pip install -e ".[test,server]"

# Run tests (349 core tests, <5 seconds)
python -m pytest tests/ --ignore=tests/test_neural_embedding.py -q

# Run with neural embeddings (requires torch)
pip install -e ".[neural]"
python -m pytest tests/test_neural_embedding.py -v

# Run experiments
python experiments/experiment_03_originbench.py    # HCSI benchmark
python experiments/experiment_04_competitive.py    # vs competitors
python experiments/05_neocortex_cls.py             # CLS validation
python experiments/06_shared_memory.py             # Multi-agent
```

## 📦 Module Reference

| Module | Brain Region | What It Does |
|:-------|:------------|:-------------|
| `brain.py` | Orchestrator | 13-step encode pipeline, recall, sleep |
| `hippocampus.py` | Hippocampus | Fast one-shot episodic encoding |
| `neocortex.py` | Neocortex | Slow statistical learning, schemas |
| `world_model.py` | vmPFC | Predictive state transitions |
| `prediction.py` | Predictive coding | Surprise-gated encoding |
| `reconstruction.py` | Reconstructive | Context-dependent generative recall |
| `sleep.py` | Sleep cycles | SWS replay + REM + pruning |
| `consolidation.py` | Memory transfer | Hippocampus → neocortex |
| `decay.py` | Forgetting | Ebbinghaus exponential curves |
| `working_memory.py` | Prefrontal cortex | Attention-gated buffer (7±2) |
| `emotional.py` | Amygdala | Emotional modulation of encoding |
| `temporal_context.py` | Entorhinal cortex | Drifting temporal context vectors |
| `interference.py` | RIF | Retrieval-induced forgetting |
| `pattern_completion.py` | CA3 | Partial cue → full memory |
| `engram.py` | Engram cells | Memory trace allocation |
| `cognitive_map.py` | Grid cells | Successor representations |
| `shared_memory.py` | Social cognition | Multi-agent memory pool |
| `neural_embedding.py` | Deep semantics | Transformer embeddings |
| `synaptic_plasticity.py` | Synapses | STDP + Hebbian learning |
| `neuromodulation.py` | Brainstem | ACh/DA/NE mode switching |
| `neurogenesis.py` | Dentate gyrus | New neuron generation |
| `prospective.py` | Future thinking | Intention triggers |
| `reconsolidation.py` | Memory updating | Post-retrieval modification |
| `metamemory.py` | Metacognition | Self-assessment of recall quality |

## 📈 Project Stats

- **Version**: 0.8.0
- **Tests**: 361 (349 core + 12 neural)
- **Modules**: 32 source + 24 test files
- **Lines**: 12,600+ (source + tests)
- **CI**: Python 3.11, 3.12, 3.13 ✅
- **Dependencies**: Just `pydantic` (core), optional `fastapi`, `torch`

## 🗺️ Roadmap

- [x] v0.1–0.4: Core memory (hippocampus, prediction, temporal context, sleep)
- [x] v0.5: Reconstruction, working memory, pattern completion, storage
- [x] v0.6: Engram tracking, cognitive maps
- [x] v0.7: Neural embeddings, CLS (neocortex), multi-agent shared memory
- [x] v0.8: World model, REST API v2, CI green
- [ ] v0.9: LLM-powered reconstruction, LoCoMo benchmark
- [ ] v1.0: Origin Cloud (managed API), dashboard, production hardening

## 📄 License

Apache 2.0 — free for commercial use.

## 🔗 Links

- **Website**: [originai.in](https://originai.in)
- **GitHub**: [github.com/meghasurya07/Origin-Memory](https://github.com/meghasurya07/Origin-Memory)
- **Research**: See [`research/`](research/) for neuroscience foundations, mathematics, and architecture docs

---

<p align="center">
  <strong>Built by <a href="https://originai.in">Origin AI</a></strong> — Frontier Intelligence & Products
</p>
