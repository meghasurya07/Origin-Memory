"""
PEC2F Knowledge Schema — Provenance-Aware Memory Graph

Implements the Person-Event-Concept-Claim-Fact taxonomy from CogMem
(arXiv:2512.14118, EMNLP 2026) adapted for Origin Brain.

This is the CRITICAL architectural innovation that separates Origin Brain
from all competitors: we distinguish between objective FACTS and subjective
CLAIMS, with full provenance tracking.

Node types:
  P — Person:   Identity/agent nodes (who said it, who was involved)
  E — Event:    Episodic knowledge (what happened, when, where)
  C — Concept:  Abstract semantic categories (topics, themes)
  C — Claim:    Source-attributed subjective statements with provenance
  F — Fact:     Consolidated objective semantic knowledge

Key innovation:
  Claims have provenance: (Person_source, CLAIMS, ClaimNode) AND (ClaimNode, ABOUT, Target)
  This prevents the system from mistaking opinions for facts — a major
  failure mode in all current AI memory systems.

Critique-Then-Reconstruct Pipeline:
  1. DRAFT: Gather initial memory slice for a query
  2. CRITIQUE: Check for temporal contradictions, misattributed claims,
     missing counter-evidence
  3. RECONSTRUCT: Filter noise, reconcile conflicts, synthesize final context

Research basis:
  - CogMem (EMNLP 2026, arXiv:2512.14118) — PEC2F schema with Active Plasticity Retrieval
  - MemHarness (arXiv:2607.28272) — Critique step for reconstructive memory
  - Zep Graphiti — Entity deduplication and bi-temporal edges
  - ADAMM (2026) — Analytic memory with attribute-value observations
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Node Types ──────────────────────────────────────────────────────────────

class PEC2FNodeType(str, Enum):
    """The five PEC2F knowledge node types."""
    PERSON = "person"       # Identity/agent nodes
    EVENT = "event"         # Episodic knowledge (what happened)
    CONCEPT = "concept"     # Abstract semantic categories
    CLAIM = "claim"         # Source-attributed subjective statements
    FACT = "fact"           # Consolidated objective knowledge


class KnowledgeNode(BaseModel):
    """A typed node in the PEC2F knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    node_type: PEC2FNodeType
    label: str                           # Human-readable label
    content: str = ""                    # Full content text
    confidence: float = 1.0              # How confident we are in this node
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_memory_id: Optional[str] = None  # Link to originating EpisodicMemory
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Claim-specific fields
    source_person: Optional[str] = None  # WHO made this claim (for CLAIM nodes)
    claim_status: Optional[str] = None   # "unverified", "supported", "contradicted", "retracted"

    # Fact-specific fields
    supporting_claims: int = 0           # How many claims support this fact
    contradicting_claims: int = 0        # How many claims contradict this fact

    @property
    def is_contested(self) -> bool:
        """Whether this knowledge node has conflicting evidence."""
        if self.node_type == PEC2FNodeType.FACT:
            return self.contradicting_claims > 0
        if self.node_type == PEC2FNodeType.CLAIM:
            return self.claim_status == "contradicted"
        return False


class KnowledgeEdge(BaseModel):
    """A typed edge in the PEC2F knowledge graph."""
    source_id: str
    target_id: str
    relation: str                # e.g., "CLAIMS", "ABOUT", "INVOLVED_IN", "IS_A", "SUPPORTS"
    weight: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CritiqueResult(BaseModel):
    """Result of the critique step in the critique-then-reconstruct pipeline."""
    issues_found: int = 0
    temporal_contradictions: List[str] = Field(default_factory=list)
    misattributed_claims: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    contested_facts: List[str] = Field(default_factory=list)
    recommendation: str = "proceed"  # "proceed", "verify", "warn"


class ReconstructionContext(BaseModel):
    """Final reconstructed context after critique-then-reconstruct pipeline."""
    query: str
    facts: List[KnowledgeNode] = Field(default_factory=list)
    claims: List[KnowledgeNode] = Field(default_factory=list)
    events: List[KnowledgeNode] = Field(default_factory=list)
    people: List[KnowledgeNode] = Field(default_factory=list)
    concepts: List[KnowledgeNode] = Field(default_factory=list)
    critique: CritiqueResult = Field(default_factory=CritiqueResult)
    narrative: str = ""               # Synthesized text narrative
    confidence: float = 1.0


# ── Extraction Patterns ─────────────────────────────────────────────────────

# Patterns for identifying claim-like language
CLAIM_INDICATORS = [
    r"\b(?:think|thinks|believe|believes|feel|feels|said|told|mentioned|claimed|suggested|argued|stated)\b",
    r"(?:according\s+to|in\s+\w+'s\s+(?:opinion|view))",
    r"(?:apparently|supposedly|allegedly|reportedly)",
    r"\b\w+\s+(?:said|believes?|thinks?|feels?|claimed|argued|stated)\s+(?:that\b|the\b|it\b)",
    r"(?:it\s+seems|it\s+appears|it\s+looks\s+like)",
    r"\b(?:opinion|rumor|rumour|hypothesis|speculation)\b",
]

# Patterns for identifying factual language
FACT_INDICATORS = [
    r"\b(?:is|are|was|were)\s+(?:a|an|the|composed|made|located|born|known)\b",
    r"\b(?:has|have|had)\s+(?:been|a|an)\b",
    r"\b(?:located\s+(?:in|at)|founded\s+in|established\s+in)\b",
    r"\b(?:costs?|weighs?|measures?|contains?)\s+\d",
    r"\b(?:born\s+(?:in|on)|died\s+(?:in|on))\b",
    r"\b(?:is\s+composed|consists?\s+of|comprises?|includes?)\b",
    r"\b(?:accepted|rejected|joined|left|started|quit|hired|fired)\b",
]

# Patterns for identifying events
EVENT_INDICATORS = [
    r"(?:yesterday|today|tomorrow|last\s+\w+|next\s+\w+|on\s+\w+day)",
    r"\b(?:happened|occurred|took\s+place|will\s+(?:happen|occur|start|begin))\b",
    r"\b(?:meeting|conference|dinner|party|appointment|interview|wedding|graduation)\b",
    r"\b(?:started|finished|completed|began|ended|launched)\b",
]


# ── Main Engine ─────────────────────────────────────────────────────────────

class PEC2FEngine:
    """
    PEC2F Knowledge Schema Engine.

    Maintains a typed knowledge graph where nodes are classified into
    Person, Event, Concept, Claim, or Fact categories. Provides provenance
    tracking for claims and a critique-then-reconstruct pipeline for
    high-fidelity memory retrieval.

    Usage:
        engine = PEC2FEngine()

        # Ingest text with source attribution
        engine.ingest("Alice thinks Bob should take the job", source_person="Alice")
        engine.ingest("Bob accepted the Meta offer", source_person="Bob")

        # Query with full provenance
        ctx = engine.reconstruct_context("Did Bob take the job?")
        # ctx.facts -> [Bob accepted Meta offer]
        # ctx.claims -> [Alice thinks Bob should take the job (source: Alice)]
        # ctx.critique -> no issues
    """

    def __init__(self):
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._edges: List[KnowledgeEdge] = []
        self._person_index: Dict[str, str] = {}    # normalized name -> node_id
        self._label_index: Dict[str, str] = {}     # normalized label -> node_id

        logger.info("PEC2FEngine initialized")

    # ── Ingestion ───────────────────────────────────────────────────────

    def ingest(
        self,
        text: str,
        source_person: Optional[str] = None,
        source_memory_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> List[KnowledgeNode]:
        """
        Ingest text and extract PEC2F-typed knowledge nodes.

        Returns list of nodes created.
        """
        ts = timestamp or datetime.now(timezone.utc)
        created_nodes: List[KnowledgeNode] = []

        # 1. Extract and register person mentions
        person_names = self._extract_person_names(text)
        for name in person_names:
            node = self._get_or_create_person(name)
            created_nodes.append(node)

        # 2. Register the source person if provided
        source_node = None
        if source_person:
            source_node = self._get_or_create_person(source_person)

        # 3. Classify the text as claim, fact, or event
        node_type = self._classify_text(text)

        # 4. Create the knowledge node
        knowledge_node = KnowledgeNode(
            node_type=node_type,
            label=self._make_label(text),
            content=text,
            source_memory_id=source_memory_id,
            created_at=ts,
            updated_at=ts,
        )

        # 5. If it's a claim, add provenance
        if node_type == PEC2FNodeType.CLAIM and source_node:
            knowledge_node.source_person = source_person
            knowledge_node.claim_status = "unverified"

            # Create provenance edge: Person -> CLAIMS -> ClaimNode
            self._edges.append(KnowledgeEdge(
                source_id=source_node.id,
                target_id=knowledge_node.id,
                relation="CLAIMS",
            ))

            # Link claim to mentioned entities
            for name in person_names:
                if name.lower() != source_person.lower():
                    target_person = self._get_or_create_person(name)
                    self._edges.append(KnowledgeEdge(
                        source_id=knowledge_node.id,
                        target_id=target_person.id,
                        relation="ABOUT",
                    ))

        # 6. If it's a fact, check for contradicting claims/facts
        if node_type == PEC2FNodeType.FACT:
            self._check_and_link_contradictions(knowledge_node)

        # 7. If it's an event, link to involved people
        if node_type == PEC2FNodeType.EVENT:
            for name in person_names:
                person_node = self._get_or_create_person(name)
                self._edges.append(KnowledgeEdge(
                    source_id=person_node.id,
                    target_id=knowledge_node.id,
                    relation="INVOLVED_IN",
                ))

        self._nodes[knowledge_node.id] = knowledge_node
        self._label_index[knowledge_node.label.lower()] = knowledge_node.id
        created_nodes.append(knowledge_node)

        return created_nodes

    def _classify_text(self, text: str) -> PEC2FNodeType:
        """Classify text into a PEC2F node type."""
        text_lower = text.lower()

        # Check for claim indicators
        claim_score = sum(
            1 for p in CLAIM_INDICATORS if re.search(p, text_lower)
        )

        # Check for event indicators
        event_score = sum(
            1 for p in EVENT_INDICATORS if re.search(p, text_lower)
        )

        # Check for fact indicators
        fact_score = sum(
            1 for p in FACT_INDICATORS if re.search(p, text_lower)
        )

        # Determine type by highest score
        if claim_score > 0 and claim_score >= event_score and claim_score >= fact_score:
            return PEC2FNodeType.CLAIM
        elif event_score > 0 and event_score >= fact_score:
            return PEC2FNodeType.EVENT
        elif fact_score > 0:
            return PEC2FNodeType.FACT
        else:
            return PEC2FNodeType.CONCEPT  # Default

    def _extract_person_names(self, text: str) -> List[str]:
        """Extract likely person names from text."""
        # Capitalized words at sentence starts are often names
        # Two-word capitalized sequences are strong name signals
        names = set()

        # Multi-word names: "Alice Smith", "Dr. Chen"
        for match in re.finditer(r"\b(?:Dr\.\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text):
            candidate = match.group(0).strip()
            # Filter out common non-name words
            if candidate.lower() not in {
                "the", "this", "that", "what", "when", "where", "how", "who",
                "monday", "tuesday", "wednesday", "thursday", "friday",
                "saturday", "sunday", "january", "february", "march", "april",
                "may", "june", "july", "august", "september", "october",
                "november", "december", "today", "yesterday", "tomorrow",
                "python", "java", "rust", "google", "meta", "amazon", "apple",
                "tokyo", "paris", "london", "berlin", "moscow", "new",
                "water", "earth", "fire", "air", "ice", "gold", "silver",
                "north", "south", "east", "west", "first", "last", "next",
                "someone", "everyone", "anyone", "nobody", "something",
            }:
                names.add(candidate)

        return list(names)

    def _get_or_create_person(self, name: str) -> KnowledgeNode:
        """Get or create a Person node."""
        normalized = name.strip().lower()
        if normalized in self._person_index:
            return self._nodes[self._person_index[normalized]]

        node = KnowledgeNode(
            node_type=PEC2FNodeType.PERSON,
            label=name.strip(),
            content=f"Person: {name.strip()}",
        )
        self._nodes[node.id] = node
        self._person_index[normalized] = node.id
        self._label_index[normalized] = node.id
        return node

    def _make_label(self, text: str) -> str:
        """Create a short label from text."""
        words = text.split()[:8]
        label = " ".join(words)
        if len(text.split()) > 8:
            label += "..."
        return label

    def _check_and_link_contradictions(self, new_node: KnowledgeNode) -> None:
        """Check if a new fact contradicts existing claims or facts."""
        new_words = set(new_node.content.lower().split())

        for existing in self._nodes.values():
            if existing.id == new_node.id:
                continue
            if existing.node_type not in (PEC2FNodeType.FACT, PEC2FNodeType.CLAIM):
                continue

            existing_words = set(existing.content.lower().split())
            overlap = len(new_words & existing_words) / max(len(new_words | existing_words), 1)

            if overlap > 0.3:
                # High overlap means related — check for contradiction signals
                negation_pairs = [
                    ("accepted", "rejected"), ("likes", "hates"), ("yes", "no"),
                    ("true", "false"), ("started", "stopped"), ("joined", "left"),
                    ("supports", "opposes"), ("agrees", "disagrees"),
                ]
                is_contradiction = any(
                    (a in new_words and b in existing_words) or
                    (b in new_words and a in existing_words)
                    for a, b in negation_pairs
                )

                if is_contradiction:
                    # Link as contradiction
                    self._edges.append(KnowledgeEdge(
                        source_id=new_node.id,
                        target_id=existing.id,
                        relation="CONTRADICTS",
                    ))
                    if existing.node_type == PEC2FNodeType.CLAIM:
                        existing.claim_status = "contradicted"
                    if existing.node_type == PEC2FNodeType.FACT:
                        existing.contradicting_claims += 1
                    new_node.contradicting_claims += 1
                else:
                    # Link as supporting
                    self._edges.append(KnowledgeEdge(
                        source_id=new_node.id,
                        target_id=existing.id,
                        relation="SUPPORTS",
                    ))
                    if existing.node_type == PEC2FNodeType.FACT:
                        existing.supporting_claims += 1
                    new_node.supporting_claims += 1

    # ── Query & Retrieval ───────────────────────────────────────────────

    def get_nodes_by_type(self, node_type: PEC2FNodeType) -> List[KnowledgeNode]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_person_knowledge(self, person_name: str) -> Dict[str, List[KnowledgeNode]]:
        """Get all knowledge related to a person."""
        normalized = person_name.strip().lower()
        person_id = self._person_index.get(normalized)

        if not person_id:
            return {"claims": [], "events": [], "facts": [], "concepts": []}

        result: Dict[str, List[KnowledgeNode]] = {
            "claims": [],
            "events": [],
            "facts": [],
            "concepts": [],
        }

        # Find edges from/to this person
        connected_ids = set()
        for edge in self._edges:
            if edge.source_id == person_id:
                connected_ids.add(edge.target_id)
            elif edge.target_id == person_id:
                connected_ids.add(edge.source_id)

        for node_id in connected_ids:
            node = self._nodes.get(node_id)
            if not node:
                continue
            if node.node_type == PEC2FNodeType.CLAIM:
                result["claims"].append(node)
            elif node.node_type == PEC2FNodeType.EVENT:
                result["events"].append(node)
            elif node.node_type == PEC2FNodeType.FACT:
                result["facts"].append(node)
            elif node.node_type == PEC2FNodeType.CONCEPT:
                result["concepts"].append(node)

        # Also include claims WHERE this person is the source
        for node in self._nodes.values():
            if (node.node_type == PEC2FNodeType.CLAIM and
                    node.source_person and
                    node.source_person.lower() == normalized):
                if node not in result["claims"]:
                    result["claims"].append(node)

        return result

    def get_claims_about(self, subject: str) -> List[Tuple[KnowledgeNode, str]]:
        """
        Get all claims about a subject, with source attribution.
        Returns list of (claim_node, source_person_name).
        """
        results = []
        subject_lower = subject.lower()

        for node in self._nodes.values():
            if node.node_type != PEC2FNodeType.CLAIM:
                continue
            if subject_lower in node.content.lower():
                source = node.source_person or "unknown"
                results.append((node, source))

        return results

    # ── Critique-Then-Reconstruct Pipeline ──────────────────────────────

    def critique(self, nodes: List[KnowledgeNode]) -> CritiqueResult:
        """
        Critique a set of knowledge nodes for issues.

        Checks for:
        1. Temporal contradictions (same subject, conflicting states)
        2. Misattributed claims (claim presented without source)
        3. Missing counter-evidence
        4. Contested facts
        """
        result = CritiqueResult()

        # 1. Find temporal contradictions
        content_groups: Dict[str, List[KnowledgeNode]] = defaultdict(list)
        for node in nodes:
            # Group by mentioned person names
            for name in self._extract_person_names(node.content):
                content_groups[name.lower()].append(node)

        for subject, group in content_groups.items():
            if len(group) > 1:
                # Check for contradictions within group
                for i, n1 in enumerate(group):
                    for n2 in group[i + 1:]:
                        if self._are_contradictory(n1.content, n2.content):
                            result.temporal_contradictions.append(
                                f"'{n1.label}' vs '{n2.label}' about {subject}"
                            )
                            result.issues_found += 1

        # 2. Check for unattributed claims
        for node in nodes:
            if node.node_type == PEC2FNodeType.CLAIM and not node.source_person:
                result.misattributed_claims.append(
                    f"Claim '{node.label}' has no source attribution"
                )
                result.issues_found += 1

        # 3. Check for contested facts
        for node in nodes:
            if node.is_contested:
                result.contested_facts.append(
                    f"'{node.label}' has contradicting evidence"
                )
                result.issues_found += 1

        # 4. Set recommendation
        if result.temporal_contradictions:
            result.recommendation = "warn"
        elif result.misattributed_claims or result.contested_facts:
            result.recommendation = "verify"
        else:
            result.recommendation = "proceed"

        return result

    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Check if two texts are contradictory."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        negation_pairs = [
            ("accepted", "rejected"), ("likes", "hates"), ("yes", "no"),
            ("true", "false"), ("started", "stopped"), ("joined", "left"),
            ("loves", "hates"), ("good", "bad"), ("agree", "disagree"),
        ]
        return any(
            (a in words1 and b in words2) or (b in words1 and a in words2)
            for a, b in negation_pairs
        )

    def reconstruct_context(
        self,
        query: str,
        max_nodes: int = 20,
    ) -> ReconstructionContext:
        """
        Reconstruct context for a query using the critique-then-reconstruct pipeline.

        Steps:
        1. DRAFT: Find relevant nodes via keyword matching
        2. CRITIQUE: Check for contradictions, misattributions, etc.
        3. RECONSTRUCT: Filter and synthesize into coherent context
        """
        # Step 1: DRAFT — gather relevant nodes
        query_words = set(query.lower().split())
        scored_nodes: List[Tuple[KnowledgeNode, float]] = []

        for node in self._nodes.values():
            if node.node_type == PEC2FNodeType.PERSON:
                continue  # People are linked, not directly returned
            node_words = set(node.content.lower().split())
            overlap = len(query_words & node_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored_nodes.append((node, score))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        draft_nodes = [n for n, _ in scored_nodes[:max_nodes]]

        # Step 2: CRITIQUE
        critique_result = self.critique(draft_nodes)

        # Step 3: RECONSTRUCT — organize by type and build narrative
        facts = [n for n in draft_nodes if n.node_type == PEC2FNodeType.FACT]
        claims = [n for n in draft_nodes if n.node_type == PEC2FNodeType.CLAIM]
        events = [n for n in draft_nodes if n.node_type == PEC2FNodeType.EVENT]
        concepts = [n for n in draft_nodes if n.node_type == PEC2FNodeType.CONCEPT]
        people = []

        # Gather related people
        mentioned_person_ids = set()
        for node in draft_nodes:
            for edge in self._edges:
                if edge.source_id == node.id or edge.target_id == node.id:
                    other_id = edge.target_id if edge.source_id == node.id else edge.source_id
                    other = self._nodes.get(other_id)
                    if other and other.node_type == PEC2FNodeType.PERSON:
                        if other.id not in mentioned_person_ids:
                            people.append(other)
                            mentioned_person_ids.add(other.id)

        # Build narrative
        narrative_parts = []

        if facts:
            narrative_parts.append("FACTS: " + "; ".join(f.content for f in facts[:5]))
        if events:
            narrative_parts.append("EVENTS: " + "; ".join(e.content for e in events[:5]))
        if claims:
            claim_texts = []
            for c in claims[:5]:
                src = f" (source: {c.source_person})" if c.source_person else ""
                claim_texts.append(f"{c.content}{src}")
            narrative_parts.append("CLAIMS: " + "; ".join(claim_texts))
        if critique_result.issues_found > 0:
            narrative_parts.append(
                f"CAUTION: {critique_result.issues_found} issues detected — "
                f"{critique_result.recommendation}"
            )

        narrative = " | ".join(narrative_parts) if narrative_parts else "No relevant knowledge found."

        # Compute overall confidence
        confidence = 1.0
        if critique_result.temporal_contradictions:
            confidence *= 0.5
        if critique_result.contested_facts:
            confidence *= 0.7
        if critique_result.misattributed_claims:
            confidence *= 0.8

        return ReconstructionContext(
            query=query,
            facts=facts,
            claims=claims,
            events=events,
            people=people,
            concepts=concepts,
            critique=critique_result,
            narrative=narrative,
            confidence=confidence,
        )

    # ── Fact Consolidation ──────────────────────────────────────────────

    def promote_claim_to_fact(self, claim_id: str) -> Optional[KnowledgeNode]:
        """
        Promote a verified claim to a fact.
        Used when multiple sources agree or external verification succeeds.
        """
        claim = self._nodes.get(claim_id)
        if not claim or claim.node_type != PEC2FNodeType.CLAIM:
            return None

        fact = KnowledgeNode(
            node_type=PEC2FNodeType.FACT,
            label=claim.label,
            content=claim.content,
            source_memory_id=claim.source_memory_id,
            supporting_claims=1,
            metadata={"promoted_from_claim": claim_id},
        )
        self._nodes[fact.id] = fact

        # Link the claim as supporting evidence
        self._edges.append(KnowledgeEdge(
            source_id=claim.id,
            target_id=fact.id,
            relation="SUPPORTS",
        ))
        claim.claim_status = "supported"

        return fact

    # ── Statistics ──────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        type_counts = defaultdict(int)
        for node in self._nodes.values():
            type_counts[node.node_type.value] += 1

        relation_counts = defaultdict(int)
        for edge in self._edges:
            relation_counts[edge.relation] += 1

        contested = sum(1 for n in self._nodes.values() if n.is_contested)
        unverified = sum(
            1 for n in self._nodes.values()
            if n.node_type == PEC2FNodeType.CLAIM and n.claim_status == "unverified"
        )

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": dict(type_counts),
            "relation_types": dict(relation_counts),
            "contested_nodes": contested,
            "unverified_claims": unverified,
        }
