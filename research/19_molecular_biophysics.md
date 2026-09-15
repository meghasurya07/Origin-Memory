# 19 — Molecular Biophysics of Memory

**Date**: 2026-09-14
**Stream**: Wave 2 — Molecular Memory Research

---

## 1. LTP/LTD Molecular Cascades

Long-Term Potentiation (LTP) and Long-Term Depression (LTD) are the bidirectional mechanisms of synaptic plasticity — the physical substrate of learning and memory.

### The Molecular Cascade (Early-LTP)

```
Step 1: Glutamate Release (presynaptic)
    ↓
Step 2: AMPA Activation → Na+ influx → depolarization
    ↓
Step 3: Mg²⁺ plug expelled from NMDA receptor (requires strong depolarization)
    ↓
Step 4: Ca²⁺ floods through NMDA receptor
    ↓
Step 5: Ca²⁺ binds calmodulin → activates CaMKII
    ↓
Step 6: CaMKII autophosphorylates, translocates to PSD
    ↓
Step 7: CaMKII phosphorylates existing AMPA receptors (↑ conductance)
         + drives insertion of NEW AMPA receptors into membrane
    ↓
Result: Higher AMPA receptor density = stronger synapse = MEMORY
```

### Early-LTP vs Late-LTP

| Property | Early-LTP (E-LTP) | Late-LTP (L-LTP) |
|:---------|:-------------------|:------------------|
| Duration | Minutes to hours | Hours to lifetime |
| Mechanism | Modification of EXISTING proteins | DE NOVO protein synthesis |
| Requires | CaMKII activation only | CREB transcription + Arc protein |
| Structural change | Temporary | Permanent (spine growth) |
| Blocked by | CaMKII inhibitors | Protein synthesis inhibitors (anisomycin) |

**AI Analog**: E-LTP = working memory / context buffer. L-LTP = long-term storage after consolidation.

---

## 2. NMDA Receptor as Coincidence Detector

The NMDA receptor is the biological implementation of Hebb's Rule: "neurons that fire together, wire together."

**Why NMDA is special**: It requires TWO simultaneous conditions to open:
1. **Presynaptic**: Glutamate must bind to the receptor
2. **Postsynaptic**: The cell must be depolarized enough to expel the Mg²⁺ block

This is a biological **AND gate** — it only strengthens connections when there is correlated pre- and post-synaptic activity.

**Computational Implication**: This is associative learning at the hardware level. Only correlations (meaningful patterns) get stored. Random noise doesn't trigger NMDA → doesn't trigger Ca²⁺ → no learning.

---

## 3. Calcium Signaling and the BCM Theory

### The BCM (Bienenstock-Cooper-Munro) Theory

A sliding threshold $\theta_M$ determines whether a synapse strengthens or weakens:

$$\Delta w = \phi(c, \theta_M) \cdot x$$

Where $c$ is postsynaptic activity (calcium concentration):
- **Low Ca²⁺** (low-frequency stimulation) → activates phosphatases (Calcineurin) → AMPA receptor internalization → **LTD** (weakening)
- **High Ca²⁺** (high-frequency stimulation) → activates kinases (CaMKII) → AMPA receptor insertion → **LTP** (strengthening)

**The sliding threshold**: $\theta_M$ itself slides based on recent history of the cell's activity. High recent activity raises the threshold (making LTP harder), implementing **homeostatic regulation**.

### Calcium Oscillations

The temporal dynamics of Ca²⁺ (amplitude, frequency, duration) encode specific information:
- Different oscillation patterns activate different downstream transcription factors
- Fast oscillations → CaMKII pathway
- Slow oscillations → Calcineurin pathway

**AI Analog**: This maps to precision-weighted prediction errors. The "amplitude" of surprise determines whether to strengthen (encode) or weaken (forget).

---

## 4. Protein Synthesis and Long-Term Memory

### CREB — The Master Switch

CREB (cAMP response element-binding protein) is the master transcription factor for permanent memory:
- Activated by Ca²⁺ and cAMP pathways
- Binds to DNA → initiates transcription of plasticity-related genes
- Neurons with high CREB levels are preferentially allocated to engrams

### Arc — The Structural Stabilizer

Arc (Activity-regulated cytoskeleton-associated protein):
- Immediate-early gene transcribed via CREB
- Arc RNA is rapidly trafficked to ACTIVATED synapses (not random ones)
- Arc protein regulates actin cytoskeleton → anchors new AMPA receptors → expands dendritic spines
- Without Arc: spines shrink back, memory is lost

### Why Blocking Protein Synthesis Erases New But Not Old Memories

- Old memories: structural changes already stabilized (spine permanently grown, AMPA permanently anchored)
- New memories: still in E-LTP phase, need L-LTP protein synthesis to become permanent
- Blocking protein synthesis after learning → L-LTP never happens → memory decays in hours

---

## 5. Structural Plasticity

Memories are physically stored as structural modifications:

| Time | Change | Mechanism |
|:-----|:-------|:----------|
| Seconds | AMPA receptor phosphorylation | CaMKII |
| Minutes | AMPA receptor trafficking + spine head expansion | CaMKII + actin |
| Hours | L-LTP protein synthesis | CREB → Arc |
| Days-Weeks | New synapse formation / old synapse pruning | Structural remodeling |
| Months-Years | Stable spine morphology | Epigenetic marks |

---

## 6. Neuromodulators — The Global Tuning System

| Neuromodulator | Source | Role in Memory | AI Equivalent |
|:---------------|:-------|:---------------|:-------------|
| **Dopamine** | VTA → hippocampus | Reward prediction error. Strengthens synapses tied to unexpected rewards | Surprise/salience signal |
| **Acetylcholine** | Basal forebrain | HIGH = encoding mode. LOW = retrieval/consolidation mode | Read/write mode toggle |
| **Norepinephrine** | Locus coeruleus | Arousal + emotional salience. Enhances consolidation of novel events | Priority/urgency signal |
| **Serotonin** | Raphe nuclei | Mood-plasticity coupling. Modulates LTP/LTD threshold | Emotional state modulation |

**Critical insight**: Acetylcholine literally SWITCHES the hippocampus between encoding and retrieval modes. This is a hardware-level mode switch, not a software toggle.

---

## 7. Epigenetic Memory

Beyond synapses, memories are stored at the chromatin level:

- **DNA Methylation**: DNMTs add methyl groups (usually repressing genes). Learning causes targeted demethylation of plasticity genes.
- **Histone Acetylation**: HATs relax chromatin → plasticity genes transcribed. HDACs act as memory suppressors.
- **Self-Perpetuating**: Epigenetic marks are highly stable and self-maintaining → memories persist for decades despite protein turnover.

**AI Analog**: This is "architectural" memory — not the data stored, but the configuration of the storage system itself.

---

*Key Papers: Bliss & Lomo 1973 (LTP discovery), Malenka & Bear 2004 (LTP/LTD review), Bienenstock et al. 1982 (BCM theory), Josselyn & Tonegawa 2020 (engrams), Day & Bhatt 2015 (epigenetics)*

*Previous: [← Deep Research Findings](18_deep_research_findings.md) | Next: [→ Neural Circuits](20_neural_circuits.md)*
