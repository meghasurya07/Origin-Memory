# 27 — Product Architecture: Theory-to-Infrastructure Bridge

**Date**: 2026-09-14
**Stream**: Wave 4 — Product Architecture Research
**Purpose**: How to turn neuroscience mechanisms into a production-grade, deployable infrastructure

---

## 1. Architecture Patterns for Memory Infrastructure (2026)

### The "Agentic Memory OS" Pattern

The industry has moved from RAG-only to tiered cognitive architectures:

```
┌─────────────────────────────────────────────┐
│               AI Agent / LLM                │
├─────────────────────────────────────────────┤
│           Memory Orchestration Layer         │
│  (write-manage-read consolidation loop)     │
├──────────┬──────────┬──────────┬────────────┤
│ Working  │ Episodic │ Semantic │ Procedural │
│ Memory   │ Store    │ Store    │ Store      │
│ (context)│ (events) │ (facts)  │ (skills)   │
├──────────┴──────────┴──────────┴────────────┤
│           Storage Layer                      │
│  (Vector DB + Graph DB + Relational DB)     │
└─────────────────────────────────────────────┘
```

### Core Loop: Write → Manage → Read
Following Event Sourcing & CQRS principles:
- **Write**: Encode new experiences with prediction error gating
- **Manage**: Offline consolidation (compress, resolve conflicts, decay)
- **Read**: Context-aware retrieval with temporal weighting

### Architecture Style
**Microservices** favored for the memory layer:
- Coordination service, extraction service, storage service
- Supports multi-tenant scale
- Asynchronous processing decoupled from agent's LLM reasoning loop

---

## 2. Embedding Models for Memory

### Comparison (2025-2026)

| Model | Type | Dims | Best For | Cost |
|:------|:-----|:-----|:---------|:-----|
| **BGE-M3** (BAAI) | Open, Hybrid | 1024 | Dense+Sparse+Multi-vector | Free |
| **text-embedding-3-large** (OpenAI) | API | 3072 (truncatable) | General purpose, MRL | \$0.13/1M tokens |
| **text-embedding-3-small** (OpenAI) | API | 1536 (truncatable) | Cost-efficient | \$0.02/1M tokens |
| **Jina-embeddings-v3** | Open/API | 1024 | Long context, multi-lingual | Free/API |
| **ColBERT v2** | Late interaction | Per-token | Maximum precision | High storage |

### Recommendation for Origin Brain

**Start with**: OpenAI `text-embedding-3-small` (1536d, truncatable to 256d via MRL)
- Best balance of quality, cost, and simplicity
- Matryoshka Representation Learning (MRL) allows dimension reduction without retraining

**Future**: BGE-M3 for hybrid dense+sparse retrieval (better for memory-specific workloads)

### Dense vs Sparse vs Hybrid

| Approach | Strengths | Weaknesses |
|:---------|:----------|:-----------|
| Dense (embeddings) | Semantic understanding | Misses exact keywords |
| Sparse (BM25/SPLADE) | Exact keyword matching | No semantic understanding |
| **Hybrid** | **Both** | More complex indexing |

---

## 3. Vector Database Selection

### Comparison for Memory Workloads

| Database | Best For | Hybrid Search | Temporal Filter | Scale |
|:---------|:---------|:-------------|:---------------|:------|
| **pgvector** | Starting out, PostgreSQL stack | Via SQL | Native timestamps | ~100M vectors |
| **Qdrant** | Self-hosted, performance | Sparse vectors | Metadata filtering | 100M+ |
| **Milvus** | Billion-scale distributed | Built-in | Metadata filtering | Billions |
| **Pinecone** | Zero-ops managed | Built-in | Metadata filtering | Managed |
| **Weaviate** | AI-native features | Built-in reranking | Metadata filtering | 100M+ |

### Recommendation for Origin Brain

**Phase 1 (MVP)**: pgvector (simple, integrated with PostgreSQL for relational data)
**Phase 2 (Scale)**: Qdrant (Rust-based, excellent throughput, sparse vector support)
**Phase 3 (Enterprise)**: Milvus (distributed, billion-scale)

### Critical Requirement: Pre-filtering

Memory MUST be filtered by `agent_id` / `user_id` BEFORE vector search. Post-filtering destroys performance at scale. Use **partition keys** in the vector DB.

---

## 4. Graph Database for Memory Relations

### The Context Graph Pattern

Combine vector DB (semantic recall) + knowledge graph (relational reasoning):

```
Vector DB: "What is semantically similar?"
    ↕ linked via entity IDs
Graph DB: "How are things related? When were facts true?"
```

### Temporal Knowledge Graphs (TKGs)

Essential for solving the "amnesia" problem:
- Store time-stamped edges: `(User, lives_in, NYC, 2025-01) → (User, lives_in, LA, 2026-03)`
- Prevents hallucinating outdated information
- Zep's Graphiti is the leader here

### Database Options

| Database | Type | Best For |
|:---------|:-----|:---------|
| Neo4j | Property Graph | Mature, ecosystem, Cypher query language |
| Memgraph | Property Graph | Real-time, in-memory, low latency |
| ArangoDB | Multi-model | Graph + Document + Key-Value in one |

### Ontology

Start simple with **POLE+O** (Person, Object, Location, Event, Organization), evolve as data collisions occur. Don't over-engineer the schema upfront.

---

## 5. Real-Time vs Batch Processing

### Streaming Long-Term Memory (SLTM) Architecture

```
┌─── Real-Time (Ingestion) ──────────────────┐
│ Agent interaction → Stream processor        │
│ (Kafka/Redis Streams)                       │
│ → Prediction error gate → Encode if novel   │
│ → Working memory update (immediate)         │
└─────────────────────────────────────────────┘
                    ↕
┌─── Batch/Offline (Consolidation) ──────────┐
│ "Sleep Cycle" — runs during idle periods    │
│ → Replay recent memories (prioritized)      │
│ → Compress via Information Bottleneck       │
│ → Extract semantic gist                     │
│ → Update knowledge graph                   │
│ → Evict decayed memories                   │
│ → Resolve contradictions                    │
└─────────────────────────────────────────────┘
```

---

## 6. SDK Design Patterns (What Developers Expect)

### Competitor SDK Analysis

| Product | Pattern | API Style |
|:--------|:--------|:----------|
| Mem0 | Drop-in layer alongside LangChain | `mem.add()`, `mem.search()` |
| Letta | Autonomous agent manages own memory | Agent-as-OS, persistent entities |
| Zep | Temporal knowledge extraction | Entity/fact extraction APIs |

### Origin Brain SDK Design

```python
from origin_brain import Brain

# Initialize with agent identity
brain = Brain(agent_id="agent-007", config=BrainConfig(...))

# Encode (with prediction error gating)
result = brain.encode("User prefers dark mode", context={"user_id": "u1"})
# result.was_encoded = True/False (prediction error gate decides)
# result.surprise_score = 0.87

# Recall (with temporal context)
results = brain.recall("What does the user prefer?", context={"user_id": "u1"})
# results.memories = [...]
# results.confidence = 0.92
# results.temporal_context = {...}

# Sleep (consolidation)
report = brain.sleep(duration_cycles=3)
# report.consolidated_count = 15
# report.evicted_count = 8
# report.new_semantic_count = 5

# Export/Import (state management)
state = brain.export_state()
brain.import_state(state)
```

### API Style
- **REST** for core CRUD operations (encode, recall, stats)
- **gRPC** for high-throughput internal services
- **WebSocket** for real-time streaming memory updates

---

## 7. Scaling Considerations

### Memory Volume Estimates

| Tier | Per Agent/Day | Per Agent/Year | 1000 Agents |
|:-----|:-------------|:--------------|:------------|
| Episodic | ~100-500 memories | ~50-200K | ~50-200M |
| Semantic | ~10-50 facts | ~5-20K | ~5-20M |
| Procedural | ~1-5 skills | ~100-500 | ~100-500K |

### Cost Optimization
- **MRL dimension reduction**: 1536d → 256d saves 83% storage with <5% quality loss
- **Product Quantization (PQ)**: Further 4-8x compression in vector DB
- **Tiered storage**: Hot (in-memory) → Warm (SSD) → Cold (S3/archival)

### Multi-Tenant Architecture

```
┌─ Tenant A ──────────┐  ┌─ Tenant B ──────────┐
│ Agent 1  Agent 2    │  │ Agent 3  Agent 4    │
│ ↓        ↓          │  │ ↓        ↓          │
│ Partition: tenant_a │  │ Partition: tenant_b │
└─────────────────────┘  └─────────────────────┘
         ↓                        ↓
    ┌──────────────────────────────┐
    │     Shared Storage Layer     │
    │  (Vector DB + Graph DB)     │
    └──────────────────────────────┘
```

---

## 8. Deployment and Pricing Models

### Competitor Pricing

| Product | Self-Hosted | Cloud | Pricing Model |
|:--------|:-----------|:------|:-------------|
| Mem0 | Free (OSS) | \$99-\$299/mo | Operations + storage |
| Letta | Free (OSS) | \$20 base + per agent | Subscription + usage |
| Zep | Limited | Enterprise | Credit-based |

### Origin Brain Pricing Strategy

**Open-source core** (MIT/Apache) + **Managed cloud** (premium):

| Tier | Price | Includes |
|:-----|:------|:---------|
| Community | Free | Self-hosted, all core features |
| Developer | \$49/mo | Managed cloud, 10 agents, 100K memories |
| Team | \$199/mo | 100 agents, 1M memories, priority support |
| Enterprise | Custom | Unlimited, SLA, dedicated infra |

### What Developers Want (Market Research)
1. **Open-source core** — for local dev, privacy, vendor lock-in avoidance
2. **Simple SDK** — `pip install origin-brain`, 3-line setup
3. **Managed cloud** — for production scaling
4. **BYOK** — bring your own LLM API key (decouple from LLM costs)

---

*Previous: [← Attention/Sleep/Emotion](26_attention_sleep_emotion.md)*
