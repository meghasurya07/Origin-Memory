# 54 — Cognitive Maps: From Spatial Navigation to Abstract Memory Spaces

**Date**: 2026-09-15
**Type**: DEEP RESEARCH + ARCHITECTURE DESIGN
**Status**: Research for v0.6+

---

## The Cognitive Map Revolution

**Key insight (O'Keefe & Nadel 1978, updated 2025):**
The hippocampus doesn't just store memories — it builds MAPS.
And these maps work for abstract concepts, not just physical space.

### The Biological Components

| Cell Type | Location | What It Encodes | Discovered |
|:----------|:---------|:---------------|:-----------|
| **Place cells** | Hippocampus CA1/CA3 | Specific location | O'Keefe 1971 |
| **Grid cells** | Entorhinal cortex | Periodic spatial metric | Moser & Moser 2005 |
| **Head direction** | Various regions | Orientation | Taube 1990 |
| **Border cells** | Entorhinal cortex | Environmental boundaries | Solstad 2008 |
| **Time cells** | Hippocampus | Temporal position | MacDonald 2011 |

### 2025 Updates: Beyond Physical Space

**The hippocampus uses the SAME circuitry for abstract spaces:**

1. **Social spaces**: Grid-like patterns when navigating social hierarchies
2. **Concept spaces**: Place cells fire for positions in "concept space"
3. **Task spaces**: Grid cells represent abstract task structure
4. **Narrative spaces**: Time cells track position in stories/sequences

**Key paper**: "Abstract representations emerge naturally in neural networks
trained to navigate in continuous environments" (arxiv 2025)

---

## Successor Representations: The Mathematical Foundation

The brain doesn't just represent WHERE you are — it represents
WHERE you COULD GO next. This is formalized as the **Successor Representation**:

$$M(s, s') = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \cdot \mathbb{1}[s_t = s'] \mid s_0 = s\right]$$

Where:
- $M(s, s')$ = expected future occupancy of state $s'$ starting from $s$
- $\gamma$ = discount factor (how far ahead to look)
- This tells you the "reachability" of every state from your current position

**Key insight**: Grid cell patterns emerge naturally when you compute the
eigenvectors of the successor representation matrix!

### Application to Memory

Replace "spatial states" with "memory states":
- $s$ = current memory being recalled
- $s'$ = memory that could be recalled next
- $M(s, s')$ = probability of transitioning from one memory to another

**This IS context-dependent recall!** The successor representation tells you
which memories are "close" to the current one — not in embedding space,
but in TRANSITION space (how often they follow each other).

---

## How This Maps to Origin Brain

### Current Implementation (Embedding-Space Proximity)
```
Query → TF-IDF → Cosine similarity → Nearest memories
```
This finds memories that are SEMANTICALLY similar.

### Cognitive Map Enhancement (Transition-Space Proximity)
```
Query → Successor Representation → Transitionally related memories
```
This finds memories that are EXPERIENTIALLY linked — memories that
tend to be recalled together or that follow each other temporally.

### Implementation Design for v0.6

```python
class CognitiveMap:
    """
    Builds a map of memory space using successor representations.
    
    Instead of static embedding similarity, this tracks which memories
    tend to be accessed together (co-occurrence = adjacency in map).
    """
    
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.transition_counts = {}  # (mem_a, mem_b) → count
        self.successor_matrix = {}   # mem_a → {mem_b: SR_value}
    
    def record_transition(self, from_memory_id, to_memory_id):
        """Record that one memory was recalled after another."""
        ...
    
    def update_successor_representation(self):
        """Recompute SR from transitions (can be done during 'sleep')."""
        ...
    
    def find_related(self, memory_id, top_k=5):
        """Find memories that are close in cognitive map space."""
        ...
```

### Integration Points

| Component | Connection |
|:----------|:----------|
| **Recall pipeline** | Record transitions between sequential recalls |
| **Sleep consolidation** | Update SR matrix during offline processing |
| **Engram overlap** | Overlap = proximity in cognitive map |
| **Pattern completion** | Navigate from partial cue to full memory |
| **Working memory** | Current "location" in cognitive map |

---

## Energy-Efficient AI (2025 Breakthrough)

A 2025 paper showed that AI systems using cognitive maps:
- Solve complex planning tasks with **50x less computation**
- Adapt to novel environments without retraining
- Achieve human-like generalization

**Why**: Instead of brute-force search through all memories, the cognitive
map provides a STRUCTURE for navigation. You don't search — you NAVIGATE.

This is directly applicable to our recall system. Currently we do:
```
All memories → Score each one → Sort → Return top-K
```

With a cognitive map:
```
Current context → Position in map → Navigate to related region → Return nearby memories
```

O(N) → O(K) for recall. This is how we scale to millions of memories.

---

## Own Thinking: The Three Memory Spaces

I propose Origin Brain should operate in THREE spaces simultaneously:

1. **Semantic Space** (current TF-IDF/embedding)
   - Finds memories with similar CONTENT
   - "What is conceptually related?"

2. **Temporal Space** (current temporal context)
   - Finds memories encoded at similar TIMES
   - "What happened around the same time?"

3. **Cognitive Map Space** (NEW — successor representation)
   - Finds memories that are experientially LINKED
   - "What do I usually think of next?"

The final retrieval score should blend all three:

$$\text{score} = \alpha \cdot \text{semantic} + \beta \cdot \text{temporal} + \gamma \cdot \text{cognitive\_map}$$

This multi-space retrieval is closer to how the actual brain works:
- Semantic = neocortical (slow, conceptual)
- Temporal = hippocampal time cells
- Cognitive map = entorhinal grid cells

---

## Priority for Implementation

1. **Record memory transitions** during recall (low effort, high value)
2. **Build successor representation** during sleep (medium effort)
3. **Multi-space retrieval blending** (medium effort, high impact)
4. **Grid-like dimensionality reduction** for visualization (optional, cool)

**This is the most promising path to HCSI 2.0** — a second generation
of benchmarks testing relational memory, inference, and navigation.

---

*Research document for Origin AI. Next: Implement CognitiveMap engine for v0.6.*
