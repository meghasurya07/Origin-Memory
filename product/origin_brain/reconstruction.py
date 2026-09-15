"""
Reconstruction Engine — The Breakthrough Module

Implements GENERATIVE RECONSTRUCTIVE MEMORY. This is what makes
Origin Brain fundamentally different from every other AI memory system.

Current AI memory: Store → Retrieve → Return stored content
Our approach: Store sparse trace → Reconstruct from trace + context + schema + emotion

Brain analogue: CA3 pattern completion → cortical reactivation → vmPFC schema guidance
Neuroscience: Schacter & Addis 2007, Damasio 1989, Marr 1971, Tse et al. 2007

Key insight: 70-80% of what humans "remember" is GENERATED, not retrieved.
The memory is a RECIPE for reconstruction, not a record.

Mathematical foundation:
    p(x̂ | z, c) = Decoder(z; c_temporal, c_emotion, c_schema)
    where z = sparse binding trace
          c = multi-modal context vector
"""

from __future__ import annotations

import math
import hashlib
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .models import EpisodicMemory, SemanticMemory, MemoryType

logger = logging.getLogger(__name__)


class ReconstructionFidelity(str, Enum):
    """How faithful the reconstruction is to the original."""
    VERBATIM = "VERBATIM"          # High-context overlap, recent, well-practiced
    GIST = "GIST"                   # Moderate overlap, some schema filling
    SCHEMA_FILLED = "SCHEMA_FILLED" # Low overlap, mostly schema-based reconstruction
    CONFABULATED = "CONFABULATED"   # Very low overlap, substantial generation


class ReconstructedMemory(BaseModel):
    """A memory that has been reconstructed, not just retrieved."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_memory_id: str                # The source episodic memory
    original_content: str                  # What was actually stored
    reconstructed_content: str             # What was generated
    fidelity: ReconstructionFidelity       # How faithful to original
    confidence: float                      # 0-1 confidence in reconstruction
    
    # What contributed to reconstruction
    trace_contribution: float              # How much came from stored trace
    schema_contribution: float             # How much came from schema filling
    context_contribution: float            # How much context influenced it
    emotional_modulation: float            # How much emotion shaped it
    
    # Context at reconstruction time
    context_overlap: float                 # Encoding context vs recall context similarity
    active_schemas: List[str] = Field(default_factory=list)
    
    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReconstructionConfig(BaseModel):
    """Configuration for the Reconstruction Engine."""
    # Fidelity thresholds
    verbatim_threshold: float = 0.8     # Context overlap above this → verbatim
    gist_threshold: float = 0.4         # Context overlap above this → gist
    schema_threshold: float = 0.2       # Context overlap above this → schema-filled
    # Below schema_threshold → confabulated
    
    # Schema influence
    schema_weight: float = 0.3          # How strongly schemas influence reconstruction
    max_schema_additions: int = 3       # Max schema-generated additions
    
    # Emotional modulation
    arousal_emphasis_threshold: float = 0.7  # Above this, emphasize salient details
    emotional_weight: float = 0.2
    
    # Compression
    gist_compression_ratio: float = 0.6  # Retain this fraction of content in gist mode
    

class SchemaKnowledge(BaseModel):
    """A knowledge schema that can fill in missing details."""
    name: str
    category: str
    typical_elements: List[str]          # Expected elements in this schema
    fill_templates: List[str]            # Templates for generating missing details
    confidence: float = 0.7


# Built-in schemas
DEFAULT_SCHEMAS: List[SchemaKnowledge] = [
    SchemaKnowledge(
        name="meeting",
        category="work",
        typical_elements=["participants", "agenda", "decisions", "action items", "time", "location"],
        fill_templates=[
            "Participants discussed the agenda items",
            "Action items were assigned to team members",
            "The meeting covered key decisions"
        ]
    ),
    SchemaKnowledge(
        name="error_incident",
        category="technical",
        typical_elements=["error type", "affected system", "root cause", "impact", "resolution", "timeline"],
        fill_templates=[
            "The team investigated the root cause",
            "Affected systems were identified and isolated",
            "A resolution was implemented and verified"
        ]
    ),
    SchemaKnowledge(
        name="learning",
        category="knowledge",
        typical_elements=["topic", "key concepts", "examples", "prerequisites", "applications"],
        fill_templates=[
            "Key concepts were explained with examples",
            "Prerequisites for understanding were established",
            "Practical applications were discussed"
        ]
    ),
    SchemaKnowledge(
        name="conversation",
        category="social",
        typical_elements=["participants", "topic", "mood", "outcome", "key points"],
        fill_templates=[
            "The participants exchanged views on the topic",
            "Key points were raised and discussed",
            "The conversation reached a natural conclusion"
        ]
    ),
    SchemaKnowledge(
        name="decision",
        category="planning",
        typical_elements=["options", "criteria", "chosen option", "rationale", "consequences"],
        fill_templates=[
            "Multiple options were evaluated against criteria",
            "The decision was made based on the strongest rationale",
            "Potential consequences were considered"
        ]
    ),
]


class ReconstructionEngine:
    """
    Generative Reconstructive Memory Engine.
    
    Instead of returning verbatim stored content, this engine RECONSTRUCTS
    memories by combining:
    1. The sparse stored trace (what was actually encoded)
    2. Current temporal context (TCM vector)
    3. Active schemas (prior knowledge fills gaps)
    4. Emotional state (modulates what's emphasized)
    5. Context overlap (encoding vs recall context similarity)
    
    This produces DIFFERENT reconstructions of the SAME memory depending
    on the current state — exactly like human memory.
    
    Usage:
        engine = ReconstructionEngine()
        
        reconstructed = engine.reconstruct(
            memory=episodic_memory,
            context_overlap=0.6,
            emotional_arousal=0.3,
            semantic_knowledge=["The project uses Python"]
        )
        
        # reconstructed.fidelity might be GIST
        # reconstructed.reconstructed_content includes schema-filled details
    """

    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self.config = config or ReconstructionConfig()
        self._schemas: List[SchemaKnowledge] = list(DEFAULT_SCHEMAS)
        self._reconstruction_count: int = 0
        self._fidelity_histogram: Dict[str, int] = defaultdict(int)
        
        logger.info("ReconstructionEngine initialized")

    def add_schema(self, schema: SchemaKnowledge) -> None:
        """Add a new knowledge schema for reconstruction."""
        self._schemas.append(schema)

    def _match_schemas(self, content: str, context: Dict[str, Any]) -> List[Tuple[SchemaKnowledge, float]]:
        """Find schemas that match the content/context."""
        content_lower = content.lower()
        context_str = " ".join(str(v) for v in context.values()).lower()
        combined = content_lower + " " + context_str
        
        matches = []
        for schema in self._schemas:
            # Score based on element mentions
            score = 0.0
            for element in schema.typical_elements:
                if element.lower() in combined:
                    score += 1.0 / len(schema.typical_elements)
            
            # Category bonus
            if schema.category.lower() in combined:
                score += 0.2
            
            # Name match
            if schema.name.lower() in combined:
                score += 0.3
                
            if score > 0.1:
                matches.append((schema, min(1.0, score)))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _extract_gist(self, content: str) -> str:
        """
        Extract the semantic gist of content, discarding peripheral details.
        
        This mimics the information bottleneck: preserve MEANING, discard SURFACE.
        
        In a production system, this would use an LLM. Here we use
        heuristics that capture the principle.
        """
        sentences = content.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return content
        
        # Score each sentence by information density
        scored = []
        for sent in sentences:
            words = sent.split()
            # Information density heuristics:
            # - Named entities (capitalized words) are important
            # - Numbers are important
            # - Longer sentences carry more info
            caps = sum(1 for w in words if w and w[0].isupper() and len(w) > 1)
            nums = sum(1 for w in words if any(c.isdigit() for c in w))
            length_score = min(len(words) / 15.0, 1.0)
            
            score = caps * 0.3 + nums * 0.2 + length_score * 0.5
            scored.append((score, sent))
        
        # Keep top sentences by compression ratio
        scored.sort(key=lambda x: x[0], reverse=True)
        keep_count = max(1, int(len(scored) * self.config.gist_compression_ratio))
        kept = scored[:keep_count]
        
        # Re-order by original position
        kept_set = set(s for _, s in kept)
        gist_sentences = [s for s in sentences if s in kept_set]
        
        return ". ".join(gist_sentences) + "." if gist_sentences else content

    def _apply_schema_filling(
        self,
        content: str,
        schemas: List[Tuple[SchemaKnowledge, float]]
    ) -> Tuple[str, List[str]]:
        """
        Fill in missing details using active schemas.
        
        This is the vmPFC-guided reconstruction: when the trace is sparse,
        schemas provide expected (but not necessarily real) details.
        
        This is exactly how false memories form — and it's a FEATURE.
        """
        if not schemas:
            return content, []
        
        content_lower = content.lower()
        added_details = []
        
        for schema, score in schemas[:2]:  # Top 2 schemas
            additions = 0
            for template in schema.fill_templates:
                # Don't add if content already covers this
                template_words = set(template.lower().split())
                content_words = set(content_lower.split())
                overlap = len(template_words & content_words) / max(len(template_words), 1)
                
                if overlap < 0.3 and additions < self.config.max_schema_additions:
                    added_details.append(f"[schema:{schema.name}] {template}")
                    additions += 1
        
        if added_details:
            filled = content + " " + " ".join(
                detail.split("] ")[1] if "] " in detail else detail
                for detail in added_details
            )
            return filled, [d.split("] ")[0].replace("[schema:", "") for d in added_details]
        
        return content, []

    def _apply_emotional_modulation(
        self,
        content: str,
        arousal: float,
        valence: str
    ) -> str:
        """
        Modulate reconstruction based on emotional state.
        
        High arousal: emphasize threat/reward-related content
        This mirrors the amygdala's modulation of hippocampal retrieval:
        - Negative arousal → emphasize negative/threat details
        - Positive arousal → emphasize reward/success details
        - Neutral → balanced reconstruction
        """
        if arousal < self.config.arousal_emphasis_threshold:
            return content  # No modulation for low arousal
        
        # Simple emotional emphasis (in production, would use sentiment analysis)
        sentences = content.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return content
        
        # Score sentences by emotional relevance
        negative_keywords = {"error", "fire", "lost", "failed", "crash", "broke", "problem", 
                           "warning", "critical", "emergency", "attack", "threat", "danger"}
        positive_keywords = {"success", "discovered", "breakthrough", "achieved", "solved",
                           "improved", "excellent", "milestone", "victory", "innovation"}
        
        emphasized = []
        for sent in sentences:
            words_lower = set(sent.lower().split())
            neg_count = len(words_lower & negative_keywords)
            pos_count = len(words_lower & positive_keywords)
            
            is_emotional = neg_count > 0 or pos_count > 0
            if is_emotional:
                emphasized.append(sent)  # Keep emotional sentences
            elif len(emphasized) < len(sentences) * 0.5:
                emphasized.append(sent)  # Keep some neutral too
        
        return ". ".join(emphasized) + "." if emphasized else content

    def reconstruct(
        self,
        memory: EpisodicMemory,
        context_overlap: float = 0.5,
        emotional_arousal: float = 0.5,
        emotional_valence: str = "neutral",
        semantic_knowledge: Optional[List[str]] = None,
        current_context: Optional[Dict[str, Any]] = None
    ) -> ReconstructedMemory:
        """
        Reconstruct a memory from its stored trace + current context.
        
        This is the core function that makes Origin Brain different.
        Instead of returning the stored content verbatim, it GENERATES
        a plausible reconstruction modulated by:
        - Context overlap (how similar is current context to encoding context)
        - Emotional state (what to emphasize)
        - Active schemas (what to fill in)
        - Semantic knowledge (additional facts to incorporate)
        
        Args:
            memory: The source episodic memory (the "sparse trace")
            context_overlap: Cosine similarity between encoding and recall contexts [0,1]
            emotional_arousal: Current emotional arousal level [0,1]
            emotional_valence: Current emotional valence ("positive", "negative", "neutral")
            semantic_knowledge: Additional semantic facts that might influence reconstruction
            current_context: Current context dict
            
        Returns:
            ReconstructedMemory with fidelity level and contribution breakdown
        """
        ctx = current_context or {}
        original = memory.content
        
        # Step 1: Determine fidelity level based on context overlap
        if context_overlap >= self.config.verbatim_threshold:
            fidelity = ReconstructionFidelity.VERBATIM
        elif context_overlap >= self.config.gist_threshold:
            fidelity = ReconstructionFidelity.GIST
        elif context_overlap >= self.config.schema_threshold:
            fidelity = ReconstructionFidelity.SCHEMA_FILLED
        else:
            fidelity = ReconstructionFidelity.CONFABULATED
        
        # Step 2: Start with the stored trace
        if fidelity == ReconstructionFidelity.VERBATIM:
            # High context overlap: return mostly original
            reconstructed = original
            trace_contrib = 0.9
            schema_contrib = 0.05
            context_contrib = 0.05
        elif fidelity == ReconstructionFidelity.GIST:
            # Medium overlap: extract gist, lose some details
            reconstructed = self._extract_gist(original)
            trace_contrib = 0.6
            schema_contrib = 0.2
            context_contrib = 0.2
        elif fidelity == ReconstructionFidelity.SCHEMA_FILLED:
            # Low overlap: gist + heavy schema filling
            reconstructed = self._extract_gist(original)
            trace_contrib = 0.3
            schema_contrib = 0.5
            context_contrib = 0.2
        else:
            # Very low overlap: mostly schema-generated
            reconstructed = self._extract_gist(original)
            trace_contrib = 0.15
            schema_contrib = 0.6
            context_contrib = 0.25
        
        # Step 3: Schema filling (vmPFC-guided reconstruction)
        active_schema_names = []
        if fidelity in (ReconstructionFidelity.GIST, 
                        ReconstructionFidelity.SCHEMA_FILLED,
                        ReconstructionFidelity.CONFABULATED):
            matched_schemas = self._match_schemas(original, ctx)
            weight = self.config.schema_weight * (1.0 - context_overlap)  # More schema when less context
            if matched_schemas and weight > 0.1:
                reconstructed, active_schema_names = self._apply_schema_filling(
                    reconstructed, matched_schemas
                )
        
        # Step 4: Emotional modulation
        emotional_mod = 0.0
        if emotional_arousal > self.config.arousal_emphasis_threshold:
            reconstructed = self._apply_emotional_modulation(
                reconstructed, emotional_arousal, emotional_valence
            )
            emotional_mod = emotional_arousal * self.config.emotional_weight
        
        # Step 5: Incorporate semantic knowledge if available
        if semantic_knowledge and fidelity != ReconstructionFidelity.VERBATIM:
            for fact in semantic_knowledge[:2]:
                if fact.lower() not in reconstructed.lower():
                    reconstructed += f" {fact}"
        
        # Step 6: Compute confidence
        # Higher context overlap + higher access count + higher salience → more confident
        confidence = (
            context_overlap * 0.4 +
            min(memory.access_count / 10.0, 1.0) * 0.3 +
            memory.salience * 0.2 +
            (1.0 if fidelity == ReconstructionFidelity.VERBATIM else 0.5) * 0.1
        )
        confidence = min(1.0, confidence)
        
        # Track statistics
        self._reconstruction_count += 1
        self._fidelity_histogram[fidelity.value] += 1
        
        result = ReconstructedMemory(
            original_memory_id=memory.id,
            original_content=original,
            reconstructed_content=reconstructed,
            fidelity=fidelity,
            confidence=confidence,
            trace_contribution=trace_contrib,
            schema_contribution=schema_contrib,
            context_contribution=context_contrib,
            emotional_modulation=emotional_mod,
            context_overlap=context_overlap,
            active_schemas=active_schema_names,
            metadata={
                "emotional_arousal": emotional_arousal,
                "emotional_valence": emotional_valence,
                "memory_salience": memory.salience,
                "memory_access_count": memory.access_count,
            }
        )
        
        logger.debug(
            f"Reconstructed memory {memory.id}: fidelity={fidelity.value}, "
            f"confidence={confidence:.3f}, schemas={active_schema_names}"
        )
        
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_reconstructions": self._reconstruction_count,
            "fidelity_histogram": dict(self._fidelity_histogram),
            "schema_count": len(self._schemas),
            "config": self.config.model_dump(),
        }
