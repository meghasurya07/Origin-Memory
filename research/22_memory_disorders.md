# 22 — Memory Disorders: Clinical Evidence for Architecture Design

**Date**: 2026-09-14
**Stream**: Wave 2 — Memory Disorders Research

---

> Understanding how memory BREAKS tells us how memory WORKS.

---

## 1. Patient H.M. (Henry Molaison, 1926-2008)

### What Happened
Bilateral medial temporal lobectomy — removed anterior 2/3 of hippocampi, amygdala, and entorhinal cortex.

### What Was Lost
- **Anterograde amnesia**: Could NOT form new episodic/semantic memories
- Lived in a "permanent present" — forgot conversations within minutes

### What Was Preserved
- **Procedural memory INTACT**: Could learn mirror-tracing, improve over days, with ZERO conscious memory of ever having practiced
- Short-term/working memory: Could hold information briefly
- Pre-surgery long-term memories partially intact

### Design Principle
**Memory is NOT one system.** Declarative (episodic/semantic) requires hippocampus. Procedural operates independently (basal ganglia, cerebellum). AI memory MUST have separate systems for these.

---

## 2. Korsakoff's Syndrome

### What Happens
Thiamine (B1) deficiency (usually chronic alcoholism) → damage to mammillary bodies and thalamic nuclei.

### Key Symptom: Confabulation
Patients unconsciously fabricate memories to fill gaps. They don't lie — they genuinely believe their fabrications.

### Design Principle
**Memory retrieval is RECONSTRUCTIVE, not reproductive.** When retrieval pathways fail, the brain constructs plausible narratives. AI must distinguish between "retrieved" and "reconstructed" information — or it will confabulate like Korsakoff patients.

---

## 3. Alzheimer's Disease

### Progression Pattern
Pathology (amyloid plaques, tau tangles) starts in medial temporal lobe → spreads outward.

Memory loss follows anatomical spread:
1. **Episodic memory fails FIRST** (recent events) — hippocampus damaged early
2. **Semantic memory fails NEXT** (facts, language) — temporal cortex
3. **Procedural memory fails LAST** (muscle memory) — basal ganglia/cerebellum spared longest

### Ribot's Law
**Old memories preserved longer than new ones.** Old memories have been deeply consolidated into distributed neocortical networks → resilient to hippocampal decay.

### Design Principle
**Consolidation creates stability.** Memories that have been replayed, consolidated, and distributed across multiple representations survive damage. AI must implement progressive consolidation that makes critical knowledge increasingly resilient.

---

## 4. PTSD and Traumatic Memory

### Mechanism
Extreme stress disrupts normal consolidation:
- **Amygdala becomes hyperactive** → encodes memory with extreme emotional intensity
- **Hippocampus fails to contextualize** → memory not properly "time-stamped" as past event
- Memory stored as fragmented sensory triggers, not coherent narrative

### Flashbacks
The memory was never consolidated as "past event" → when triggered, brain experiences it as CURRENT threat → non-terminating feedback loop.

### Design Principle
**Emotional modulation without contextualization is dangerous.** High-salience encoding must ALWAYS be accompanied by temporal/contextual binding. Without context: disruptive feedback loops (the AI equivalent of flashbacks — obsessing over high-priority but uncontextualized information).

---

## 5. Infantile Amnesia

### Why We Can't Remember Before Age 3
Extreme rates of hippocampal neurogenesis (rapid birth of new neurons) in developing brains. New neurons integrate by REMODELING existing circuits — overwriting early memories.

### The Plasticity-Stability Tradeoff
- **High plasticity** → learn rapidly → but early memories get overwritten
- **Low plasticity** → retain memories → but slow to learn new things

### Design Principle
**Plasticity and stability are in tension.** A developing AI agent might need a high-plasticity "learning phase" where early data is naturally pruned, before entering a stable retention mode.

---

## 6. Retrograde Amnesia and Ribot's Law

### The Temporal Gradient
Brain injury → recent memories lost, remote memories preserved.

### Two Competing Theories

| Theory | Claim | Implication for AI |
|:-------|:------|:-------------------|
| **Standard Model** (Squire) | Memories physically transfer from hippocampus to neocortex over time | Implement progressive migration from fast to slow store |
| **Multiple Trace Theory** (Moscovitch/Nadel) | Every recall creates a new hippocampal trace. Old = more traces = more resilient. Memories TRANSFORM from episodic → semantic | Implement trace redundancy + semantic extraction on repeated access |

### Design Principle
**Repeatedly accessed information should become faster, more resilient, and more abstract.** Episodic logs that are frequently retrieved should be distilled into robust semantic facts.

---

## 7. Prosopagnosia (Face Blindness)

### What Happens
Can't recognize faces, but general memory and object recognition fully intact.

### Design Principle
**The brain has category-specific processing modules.** Not all data should be treated uniformly. High-priority, specialized categories (faces, voices, spatial layouts) benefit from dedicated processing pipelines and distinct memory indexes.

---

## 8. False Memories

### Elizabeth Loftus: The Misinformation Effect
Post-event information can overwrite or blend with original memory traces (source misattribution).

### DRM Paradigm
Present: bed, rest, dream, tired, night → Subject "remembers" hearing "SLEEP" (which was never presented). Associative activation creates false memories.

### Design Principle
**Memory is associative and reconstructive.** Powerful for generalization, vulnerable to interference. AI MUST maintain source tracking and provenance to avoid confusing inferences with raw episodic data.

---

## 9. Hyperthymesia (Superior Autobiographical Memory)

### Jill Price / AJ Case
Can remember virtually every day of her life in perfect detail.

### The COST of Not Forgetting
- Emotional paralysis — can't let go of negative experiences
- Cognitive exhaustion — overwhelmed by irrelevant detail
- Obsessive-compulsive traits

### Design Principle
**Infinite, perfect memory is a FAILURE MODE, not a feature.** Active forgetting, pruning, and decay are ESSENTIAL for cognitive flexibility, focus, and computational efficiency. This is the strongest clinical evidence for Pillar 3 (Forgetting as Regularization).

---

## 10. Unified Design Principles from Clinical Evidence

| Disorder | Lesson | Design Principle |
|:---------|:-------|:----------------|
| H.M. | Memory is multi-system | Separate episodic, semantic, procedural stores |
| Korsakoff's | Retrieval is reconstructive | Track provenance; distinguish retrieved vs reconstructed |
| Alzheimer's | Consolidation creates resilience | Progressive consolidation → distributed storage |
| PTSD | Emotion without context is dangerous | Always bind salience with temporal context |
| Infantile Amnesia | Plasticity vs stability tradeoff | Learning phase → stable phase transition |
| Retrograde Amnesia | Repeated access = resilience | Strengthen on retrieval; semantic extraction |
| Prosopagnosia | Category-specific processing | Specialized pipelines for distinct data types |
| False Memories | Associative reconstruction creates errors | Source tracking and provenance metadata |
| Hyperthymesia | Never forgetting is pathological | Active decay is essential, not optional |

---

*Key Sources: Scoville & Milner 1957 (H.M.), Ribot 1882 (temporal gradient), Loftus & Palmer 1974 (misinformation), Roediger & McDermott 1995 (DRM), Price & Davis 2008 (HSAM)*

*Previous: [← Computational Models](21_computational_models.md) | Next: [→ Cutting Edge AI](23_cutting_edge_ai.md)*
