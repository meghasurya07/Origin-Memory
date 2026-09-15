# 08 — Origin AI Architecture Proposal

## Overview

This document proposes the technical architecture for Origin AI's **Artificial Brain Memory System** — a biologically-grounded, multi-tiered memory infrastructure for AI agents that combines neuroscience-inspired algorithms with a DNA-based archival storage substrate.

---

## 1. Design Philosophy

### 1.1 Core Principles

1. **Brain-first, not database-first**: Architecture mirrors the human brain's memory hierarchy, not a CRUD database
2. **Forgetting is essential**: Intelligent eviction is as important as storage
3. **Consolidation is continuous**: Memories must transform over time (episodic → semantic)
4. **Retrieval is cue-based**: Not keyword search, but associative, context-sensitive recall
5. **Dual-speed learning**: Fast episodic binding (hippocampus) + slow knowledge integration (neocortex)
6. **Hybrid substrate**: Silicon for hot/warm storage, DNA for cold/archival storage

### 1.2 Neuroscience ↔ Architecture Mapping

| Brain Structure | Function | Architecture Component |
|:---------------|:---------|:----------------------|
| Prefrontal Cortex | Working memory, executive control | **Context Manager** |
| Hippocampus | Episodic encoding, pattern completion, replay | **Hippocampal Engine** |
| Neocortex | Semantic knowledge, gradual learning | **Neocortical Store** |
| Basal Ganglia | Procedural memory, habits | **Procedure Registry** |
| Amygdala | Emotional tagging, importance weighting | **Salience Scorer** |
| Thalamus | Gating, routing information flow | **Memory Router** |
| Sleep/Replay | Consolidation, reorganization | **Consolidation Daemon** |
| DNA/Molecular | Permanent storage | **DNA Archive Module** |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ORIGIN AI ARTIFICIAL BRAIN                        │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐     │
│  │   SALIENCE   │    │   MEMORY     │    │    CONSOLIDATION    │     │
│  │   SCORER     │    │   ROUTER     │    │    DAEMON           │     │
│  │  (Amygdala)  │    │  (Thalamus)  │    │   (Sleep/Replay)    │     │
│  └──────┬───────┘    └──────┬───────┘    └─────────┬───────────┘     │
│         │                   │                      │                  │
│  ┌──────▼───────────────────▼──────────────────────▼───────────────┐ │
│  │                    CONTEXT MANAGER (PFC)                         │ │
│  │  ┌────────────────────────────────────────────────────────────┐ │ │
│  │  │  Active Working Memory: ~128K tokens                       │ │ │
│  │  │  Current task context + retrieved memories + system prompt  │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────┬──────────────────────────────────────┘ │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────────┐ │
│  │                   HIPPOCAMPAL ENGINE                             │ │
│  │                                                                  │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐     │ │
│  │  │  DENTATE     │  │     CA3      │  │       CA1          │     │ │
│  │  │  GYRUS       │  │  Auto-       │  │    Novelty         │     │ │
│  │  │  Pattern     │  │  Associative │  │    Detection       │     │ │
│  │  │  Separation  │  │  Completion  │  │    & Comparison    │     │ │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘     │ │
│  │         │                 │                    │                  │ │
│  │  ┌──────▼─────────────────▼────────────────────▼───────────┐    │ │
│  │  │  EPISODIC STORE: Vector DB + Temporal Index              │    │ │
│  │  │  [NVMe SSD / Cloud — Latency: 1-10ms]                   │    │ │
│  │  └─────────────────────────┬────────────────────────────────┘    │ │
│  └────────────────────────────┼─────────────────────────────────────┘ │
│                               │ Consolidation                        │
│  ┌────────────────────────────▼─────────────────────────────────────┐ │
│  │                   NEOCORTICAL STORE                               │ │
│  │                                                                   │ │
│  │  ┌────────────────────────────────────────────────────────────┐  │ │
│  │  │  SEMANTIC KNOWLEDGE GRAPH: Neo4j / Custom Graph DB         │  │ │
│  │  │  Facts, rules, user preferences, learned patterns          │  │ │
│  │  │  [SSD / Cloud — Latency: 10-100ms]                        │  │ │
│  │  └────────────────────────────────────────────────────────────┘  │ │
│  │  ┌────────────────────────────────────────────────────────────┐  │ │
│  │  │  PROCEDURE REGISTRY: Workflow DB                           │  │ │
│  │  │  Behavioral patterns, tool strategies, decision heuristics │  │ │
│  │  └────────────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────┬────────────────────────────────────┘ │
│                                 │ Archival                            │
│  ┌──────────────────────────────▼────────────────────────────────────┐ │
│  │                   DNA ARCHIVE MODULE 🧬                           │ │
│  │                                                                    │ │
│  │  ┌────────────────────────────────────────────────────────────┐   │ │
│  │  │  DNA STORAGE: BioCompute Microfluidics Chip                │   │ │
│  │  │  Permanent archive, model weights, cross-gen knowledge     │   │ │
│  │  │  [Molecular — Latency: Minutes-Hours — Capacity: ∞]       │   │ │
│  │  └────────────────────────────────────────────────────────────┘   │ │
│  │  ┌────────────────────────────────────────────────────────────┐   │ │
│  │  │  SILICON INDEX: Retrieval metadata for DNA-stored data     │   │ │
│  │  │  [SSD — Latency: <1ms — enables fast lookup into DNA]     │   │ │
│  │  └────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Context Manager (PFC)

```python
class ContextManager:
    """
    Manages the agent's working memory (context window).
    Analogous to the Prefrontal Cortex.
    """
    
    def __init__(self, max_tokens=128000):
        self.max_tokens = max_tokens
        self.system_prompt = None      # Fixed instructions
        self.active_memories = []      # Retrieved from long-term stores
        self.current_conversation = [] # Ongoing dialogue
        self.scratchpad = {}           # Internal reasoning workspace
    
    def compose_context(self, query):
        """
        Assemble the optimal context window for the current query.
        This is the 'executive function' — deciding what to attend to.
        """
        # 1. Always include system prompt
        context = [self.system_prompt]
        
        # 2. Retrieve relevant memories (via Memory Router)
        relevant_memories = self.memory_router.retrieve(
            query=query,
            sources=["episodic", "semantic", "procedural"],
            max_tokens=self.max_tokens * 0.3  # 30% budget for memories
        )
        context.extend(relevant_memories)
        
        # 3. Include recent conversation (recency bias)
        recent = self.current_conversation[-20:]  # Last 20 turns
        context.extend(recent)
        
        # 4. Reserve space for reasoning
        # Remaining tokens available for model output
        
        return self.truncate_to_budget(context)
```

### 3.2 Hippocampal Engine

```python
class HippocampalEngine:
    """
    Fast encoding and retrieval of episodic memories.
    Implements pattern separation (DG), pattern completion (CA3),
    and novelty detection (CA1).
    """
    
    def __init__(self):
        self.episodic_store = VectorDB()  # Embeddings for similarity
        self.temporal_index = TimeSeriesDB()  # Temporal ordering
        self.salience_scorer = SalienceScorer()
    
    def encode(self, experience):
        """Encode a new experience into episodic memory."""
        
        # CA1: Novelty detection — is this worth remembering?
        novelty_score = self.detect_novelty(experience)
        salience = self.salience_scorer.score(experience)
        
        if novelty_score < NOVELTY_THRESHOLD and salience < SALIENCE_THRESHOLD:
            return None  # Not worth encoding
        
        # DG: Pattern separation — create a unique, sparse representation
        sparse_embedding = self.pattern_separate(experience)
        
        # Store with temporal and contextual metadata
        memory = EpisodicMemory(
            content=experience,
            embedding=sparse_embedding,
            timestamp=now(),
            context=self.get_current_context(),
            salience=salience,
            stability=INITIAL_STABILITY,
            access_count=0
        )
        
        self.episodic_store.insert(memory)
        self.temporal_index.insert(memory)
        
        return memory
    
    def retrieve(self, cue, top_k=5):
        """
        CA3: Pattern completion — retrieve full memories from partial cues.
        """
        # Semantic similarity search
        candidates = self.episodic_store.search(
            query_embedding=self.embed(cue),
            top_k=top_k * 3  # Over-retrieve, then filter
        )
        
        # Temporal relevance weighting
        candidates = self.apply_temporal_weighting(candidates)
        
        # Salience-based filtering
        candidates = self.apply_salience_filter(candidates)
        
        # Return top-k after all filtering
        return candidates[:top_k]
    
    def pattern_separate(self, experience):
        """
        DG-inspired sparse coding: ensure similar experiences
        get distinct representations to prevent interference.
        """
        dense_embedding = self.embed(experience)
        # Apply competitive inhibition to create sparse code
        sparse = apply_k_winners_take_all(dense_embedding, k=0.05)
        return sparse
```

### 3.3 Consolidation Daemon

```python
class ConsolidationDaemon:
    """
    Runs during agent "idle" periods (the equivalent of sleep).
    Implements hippocampal replay to consolidate episodic → semantic.
    """
    
    def __init__(self, hippocampal_engine, neocortical_store, dna_archive):
        self.hippocampus = hippocampal_engine
        self.neocortex = neocortical_store
        self.dna = dna_archive
    
    async def run_consolidation_cycle(self):
        """
        Run one consolidation cycle. Called periodically (e.g., hourly)
        or during idle periods.
        """
        # Phase 1: Replay — select episodic memories for consolidation
        episodes = self.hippocampus.select_for_consolidation(
            criteria={
                "min_age": timedelta(hours=24),
                "min_access_count": 2,
                "min_salience": 0.3
            }
        )
        
        # Phase 2: Extract semantic knowledge from episodes
        for episode_batch in chunk(episodes, batch_size=50):
            # Identify patterns across episodes
            patterns = self.extract_patterns(episode_batch)
            
            for pattern in patterns:
                # Check if this knowledge already exists
                existing = self.neocortex.search(pattern.key)
                
                if existing:
                    # Update existing knowledge (reconsolidation)
                    self.neocortex.update(existing, pattern, strategy="ewc")
                else:
                    # Store new semantic knowledge
                    self.neocortex.insert(pattern)
        
        # Phase 3: Decay — weaken memories that haven't been accessed
        self.hippocampus.apply_decay(
            decay_function=lambda m: exponential_decay(
                m.stability,
                time_since_access=now() - m.last_accessed,
                importance=m.salience
            )
        )
        
        # Phase 4: Evict — remove memories below the survival threshold
        evicted = self.hippocampus.evict(threshold=SURVIVAL_THRESHOLD)
        
        # Phase 5: Archive to DNA — periodically archive old semantic knowledge
        if self.should_archive():
            await self.archive_to_dna()
    
    async def archive_to_dna(self):
        """Archive aged, stable semantic knowledge to DNA storage."""
        candidates = self.neocortex.select_for_archival(
            min_age=timedelta(days=90),
            min_stability=0.8,
            max_access_frequency=0.01
        )
        
        if candidates:
            await self.dna.write(candidates)
            logger.info(f"Archived {len(candidates)} knowledge units to DNA")
```

### 3.4 Memory Decay Engine

```python
class MemoryDecayEngine:
    """
    Implements biologically-inspired forgetting based on the
    Ebbinghaus forgetting curve and importance-weighted decay.
    """
    
    @staticmethod
    def compute_retrievability(memory) -> float:
        """
        R(t) = e^(-t/S) * importance_boost
        
        Where:
        - t = time since last access
        - S = stability (grows with each successful retrieval)
        """
        t = (now() - memory.last_accessed).total_seconds()
        S = memory.stability
        
        base_retrievability = math.exp(-t / S)
        
        # Importance boost: salient memories decay slower
        importance_boost = 1.0 + (memory.salience * 0.5)
        
        return min(1.0, base_retrievability * importance_boost)
    
    @staticmethod
    def update_stability_on_retrieval(memory):
        """
        Each successful retrieval increases stability:
        S_{n+1} = S_n * (1 + alpha * e^(w * S_n))
        
        This models the spaced repetition effect.
        """
        alpha = 0.3  # Learning efficiency
        w = -0.01    # Decay modifier
        
        memory.stability *= (1 + alpha * math.exp(w * memory.stability))
        memory.access_count += 1
        memory.last_accessed = now()
```

---

## 4. Data Flow Diagram

```
User Query
    │
    ▼
┌──────────────────┐
│  Context Manager  │─── Assembles working memory
│  (PFC)           │
└────────┬─────────┘
         │ What memories are relevant?
         ▼
┌──────────────────┐
│  Memory Router   │─── Routes query to appropriate store(s)
│  (Thalamus)      │
└────────┬─────────┘
         │
    ┌────┼────────────────┐
    │    │                │
    ▼    ▼                ▼
Episodic  Semantic    Procedural
Store     Store       Registry
    │    │                │
    └────┼────────────────┘
         │ Aggregate results
         ▼
┌──────────────────┐
│  Salience Scorer │─── Rank by importance & relevance
│  (Amygdala)      │
└────────┬─────────┘
         │ Top-K memories
         ▼
┌──────────────────┐
│  Context Manager  │─── Inject into working memory
│  (PFC)           │
└────────┬─────────┘
         │ Complete context
         ▼
┌──────────────────┐
│  LLM Inference   │─── Generate response
└────────┬─────────┘
         │ Response + new experience
         ▼
┌──────────────────┐
│  Hippocampal     │─── Encode new experience
│  Engine          │
└────────┬─────────┘
         │ Later (idle/sleep)
         ▼
┌──────────────────┐
│  Consolidation   │─── Episodic → Semantic → DNA Archive
│  Daemon          │
└──────────────────┘
```

---

## 5. API Design (Draft)

```python
# Origin AI Brain SDK — Draft API

from origin_brain import Brain

# Initialize brain for an agent
brain = Brain(
    agent_id="agent-001",
    config={
        "working_memory_tokens": 128000,
        "episodic_backend": "pgvector",
        "semantic_backend": "neo4j",
        "dna_provider": "biocompute",  # or "silicon_fallback"
        "consolidation_interval": "1h",
        "decay_model": "ebbinghaus_weighted"
    }
)

# Encoding (automatic — called after each interaction)
brain.encode(experience={
    "user_message": "Book me a flight to Tokyo next week",
    "agent_response": "I've found 3 flights...",
    "context": {"user_id": "alice", "session": "s-42"},
    "outcome": "successful_booking"
})

# Retrieval (automatic — called before each inference)
memories = brain.recall(
    cue="What flights did Alice book recently?",
    memory_types=["episodic", "semantic"],
    temporal_range=("2026-01-01", "now"),
    top_k=5
)

# Manual consolidation trigger
brain.consolidate()

# Archival to DNA
brain.archive_to_dna(
    criteria={"min_age_days": 90, "min_stability": 0.8}
)

# Restore agent from DNA archive (after model upgrade, etc.)
brain.restore_from_dna(agent_id="agent-001")
```

---

## 6. Technology Stack

| Component | Technology | Rationale |
|:----------|:----------|:----------|
| Working Memory | LLM context window | Standard |
| Episodic Store | PostgreSQL + pgvector | Battle-tested, vector + relational |
| Temporal Index | TimescaleDB (PostgreSQL extension) | Time-series queries |
| Semantic Store | Neo4j / Apache AGE | Knowledge graph with relationship reasoning |
| Procedure Registry | PostgreSQL + JSON | Structured behavioral patterns |
| Embedding Model | BERT / Nomic / Custom | Dense vector representations |
| Decay Engine | Custom Python / Rust | Mathematical decay models |
| Consolidation | Async workers (Celery / custom) | Background processing |
| DNA Interface | BioCompute API | DNA synthesis + sequencing |
| DNA Index | Redis / SQLite | Fast lookup into DNA archive |
| API Layer | FastAPI | Developer-facing SDK |
| Orchestration | Kubernetes | Scaling |

---

## 7. Performance Targets

| Metric | Target | Brain Equivalent |
|:-------|:-------|:----------------|
| Encoding latency | < 50ms | Hippocampal encoding |
| Episodic retrieval | < 10ms (p99) | Hippocampal recall |
| Semantic retrieval | < 100ms (p99) | Neocortical retrieval |
| Consolidation cycle | < 5 min / 1000 episodes | Sleep cycle |
| DNA write | < 1 hour per batch | Molecular consolidation |
| DNA read | < 30 min per retrieval | Deep memory access |
| Memory capacity | 100M+ episodes per agent | Lifetime of experiences |
| Forgetting accuracy | > 90% agreement with human-rated importance | Selective forgetting |

---

*Previous: [← DNA × Brain Convergence](07_dna_brain_convergence.md) | Next: [Benchmarks & Evaluation →](09_benchmarks_evaluation.md)*
