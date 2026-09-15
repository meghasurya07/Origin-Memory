# 24 — Philosophy & Cognitive Science of Memory

**Date**: 2026-09-14
**Stream**: Wave 3 — Theoretical Foundations

---

> Before we can build memory, we must understand what memory IS.

---

## 1. What IS a Memory?

### Reproductive vs Constructive

| View | Metaphor | Status |
|:-----|:---------|:-------|
| **Reproductive** | Memory = camera/tape recorder | Outdated. Disproven by false memory research |
| **Constructive** | Memory = Wikipedia page edited on the fly | Consensus. Supported by overwhelming evidence |

### Bartlett (1932): The Pioneer
Sir Frederic Bartlett showed that people don't replay memories — they RECONSTRUCT them using schemas, current emotions, and post-event information. His "War of the Ghosts" experiment showed systematic distortions in retelling.

### AI Implication
A human-like memory is NOT a read-only database. It's flexible, associative, meaning-prioritized, and prone to creative reconstruction. If we want human-like memory, we must accept that "accurate" and "human-like" are sometimes in tension.

---

## 2. Multiple Memory Systems Theory

### Squire's Taxonomy
```
Memory
├── Declarative (Explicit) — medial temporal lobe
│   ├── Episodic — personal experiences, mental time travel
│   └── Semantic — general facts, knowledge
└── Non-Declarative (Implicit) — various brain regions
    ├── Procedural (skills) — basal ganglia
    ├── Priming — neocortex
    ├── Classical conditioning — cerebellum, amygdala
    └── Non-associative learning — reflex pathways
```

### The Debate: Unitary vs Multiple Systems
- **Multiple systems**: Different neural substrates, different computational principles, independently impaired (H.M. evidence)
- **Unitary theorists**: One system, different outputs. MINERVA 2 shows semantic can emerge from episodic.
- **Current consensus**: Multiple interacting systems with some emergence

### AI Implication
We need differentiated storage for "knowing how," "knowing that," and "remembering when" — but they must interact.

---

## 3. Encoding Specificity (Tulving & Thomson, 1973)

### The Principle
Retrieval is most successful when cues at recall MATCH the context present during encoding. Context is bound INTO the trace.

### Evidence
- **State-dependent memory**: Learn underwater → recall better underwater
- **Mood-congruent memory**: Happy state → easier to recall happy memories
- **Environmental context**: Studying in the exam room improves exam performance

### AI Implication
Memory retrieval shouldn't rely solely on semantic queries. It must weight the "context" (system state, user mood, task environment) present during encoding. This directly supports Pillar 2 (Temporal Context Binding).

---

## 4. Levels of Processing (Craik & Lockhart, 1972)

### The Framework
Memory durability depends on DEPTH of processing during encoding:

| Level | Processing | Example | Memory Strength |
|:------|:-----------|:--------|:---------------|
| Shallow | Physical/surface features | "Is it in uppercase?" | Weak |
| Intermediate | Phonological | "Does it rhyme with train?" | Moderate |
| Deep | Semantic meaning | "Does it mean the same as happy?" | Strong |
| Deepest | Self-reference | "Does it describe you?" | Strongest |

### The Self-Reference Effect
Information evaluated in relation to the SELF is remembered vastly better than any other processing.

### AI Implication
Not all inputs should be encoded equally. Semantic elaboration and evaluation against the agent's "identity" or the user's core profile should create stronger, more durable memory traces.

---

## 5. Spreading Activation (Collins & Loftus, 1975)

### The Model
Memory is a web of connected concept nodes:

```
          Hospital
         /        \
    Doctor ─── Nurse
     |              |
  Stethoscope    Medicine
     |
   Patient
```

### Mechanism
- Activating "doctor" spreads activation to connected nodes
- Activation decays with distance
- Explains priming effects (seeing "doctor" makes "nurse" faster to process)

### AI Implication
Replace flat vector search with dynamic graph-based activation where context primes the network, anticipating related concepts.

---

## 6. Schema Theory

### What Are Schemas?
Cognitive frameworks / mental templates for organizing knowledge (Bartlett, Piaget).

### Schema-Congruent vs Schema-Incongruent

| Type | Processing | Memory | Risk |
|:-----|:-----------|:-------|:-----|
| Schema-congruent | Efficient, automatic | Good but generic | Bias, distortion |
| Schema-incongruent | Effortful, surprising | Often BETTER (deeper processing) | None |

**Paradox**: Surprising (schema-violating) information is often remembered more vividly because it forces deeper cognitive processing to resolve the conflict.

### AI Implication
This connects directly to prediction error encoding. Schema-matching events get compressed efficiently. Schema-violating events trigger deep encoding via prediction error. Both paths are necessary.

---

## 7. Consolidation Theories

### Three Competing Theories

| Theory | Claim | Hippocampus Role | Time Course |
|:-------|:------|:----------------|:------------|
| **Standard Model** (Squire) | Memories fully transfer to neocortex | Temporary scaffold | Weeks-years |
| **Multiple Trace Theory** (Moscovitch/Nadel) | Each recall creates new trace. Episodic always needs hippocampus | Permanent for episodic | Ongoing |
| **Trace Transformation** | Memories TRANSFORM: vivid episode → generic gist | Permanent but role changes | Ongoing |

### AI Implication
Trace Transformation is most aligned with our architecture:
- Recent: rich episodic traces (hippocampal store)
- Over time: transform to abstract semantic knowledge (neocortical/semantic store)
- Original episodic pointer maintained for deep retrieval

---

## 8. The Binding Problem

### The Question
How does the brain bind separate features (color, shape, location, time, emotion) processed in different brain areas into a unified memory?

### Damasio's Convergence Zones
Specific neural hubs (like hippocampus) store a "binding record" — a RECIPE for reconstructing the memory. When cued, the hub sends divergent signals to sensory cortices to reconstruct the full experience.

### AI Implication
Don't store complete multi-modal memories in one location. Store pointers/binding indices that can reconstruct across modalities. The "memory" IS the index, not the data.

---

## 9. Metamemory and Feeling-of-Knowing

### How We Know What We Know
- **Judgments of Learning (JOL)**: During encoding, we estimate how well we'll remember
- **Feeling of Knowing (FOK)**: We sense we know something even before retrieving it
- **Tip-of-Tongue (TOT)**: Strong FOK but retrieval failure

### The Accessibility Heuristic
FOK ≠ actual memory strength. FOK = how easily RELATED information is accessed. More related context → stronger FOK (even if target is unretrievable).

### AI Implication
AI needs rapid heuristic checks before deep retrieval. "Do I have enough related context to answer?" before launching expensive search.

---

## 10. Consciousness and Memory

### Global Workspace Theory (Baars)
Consciousness = "brightly lit stage" where information is globally broadcast to all brain networks.

- **Implicit memory** operates "backstage" (without consciousness)
- **Explicit memory** requires entering the "spotlight" for encoding/retrieval

### AI Implication
For explicit, human-like episodic recall: AI needs a "global workspace" — a central attention mechanism where retrieved memories are broadcast to all reasoning sub-agents simultaneously.

---

## 11. Embodied and Extended Memory

### Extended Mind Thesis (Clark & Chalmers)
The mind doesn't stop at the skull. Tools reliably used for cognition are literally part of the memory system.

### Transactive Memory (Wegner)
In groups, we don't remember everything — we remember WHO KNOWS WHAT.

### AI Implication
AI memory isn't just internal databases. It must:
1. Integrate external tools (user files, internet) as extensions of its own mind
2. Build "transactive" relationships — knowing what the user remembers vs what the AI is responsible for

---

## Summary: What Memory IS

Based on all cognitive science evidence:

1. **Memory is RECONSTRUCTIVE**, not reproductive (Bartlett)
2. **Memory is MULTIPLE SYSTEMS**, not one store (Squire, Tulving)
3. **Memory is CONTEXT-DEPENDENT** (encoding specificity)
4. **Memory depth depends on PROCESSING QUALITY** (levels of processing)
5. **Memory is an ASSOCIATIVE NETWORK** (spreading activation)
6. **Memory uses SCHEMAS** for efficient encoding and prediction error for exceptions
7. **Memory TRANSFORMS over time** — from vivid to abstract (trace transformation)
8. **Memory is INDEXED**, not stored (binding problem / convergence zones)
9. **Memory is SELF-MONITORING** (metamemory)
10. **Memory EXTENDS beyond the individual** (extended mind, transactive memory)

---

*Key Papers: Bartlett 1932, Squire 2004, Tulving & Thomson 1973, Craik & Lockhart 1972, Collins & Loftus 1975, Moscovitch & Nadel 1998, Damasio 1989, Clark & Chalmers 1998, Baars 1988*

*Previous: [← Cutting Edge AI](23_cutting_edge_ai.md) | Next: [→ Research continues...]*
