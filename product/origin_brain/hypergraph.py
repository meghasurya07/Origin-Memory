"""
Hypergraph Memory — N-ary Relational Memory Fabric
═══════════════════════════════════════════════════

Extends the associative network from binary edges (subject-relation-object)
to HYPEREDGES that connect N nodes simultaneously. This captures complex
relational structures that pairwise graphs cannot represent.

Example that requires a hyperedge:
  "Alice, Bob, and Charlie met at the cafe on Tuesday to discuss Project X"
  → This is ONE event involving 5 entities simultaneously.
  → A standard graph needs 10+ binary edges to approximate this.
  → A hyperedge captures it as a single atomic relation.

Key insight from HGMem (ICML 2026):
  Hypergraph memory outperforms standard KG for complex relational modeling
  because real-world knowledge is inherently n-ary. Meetings involve multiple
  people + place + time + topic. Chemical reactions involve multiple reactants.
  Legal contracts bind multiple parties.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │  HypergraphMemory                                │
  │                                                   │
  │  Nodes: entities (people, places, concepts, etc.) │
  │                                                   │
  │  Hyperedges: n-ary relations                      │
  │    e.g., MEETING(Alice, Bob, Charlie, Cafe, Tue)  │
  │    Each hyperedge has:                            │
  │      - Set of participant nodes                   │
  │      - Relation type (MEETING, DISCUSSION, etc.)  │
  │      - Weight / strength                          │
  │      - Temporal metadata (when)                   │
  │      - Source memory ID                           │
  │                                                   │
  │  Operations:                                      │
  │    - Add hyperedge                                │
  │    - Query by node overlap (find hyperedges       │
  │      containing a subset of nodes)                │
  │    - Spreading activation over hypergraph         │
  │    - Clique expansion (convert to pairwise)       │
  │    - Incidence matrix operations                  │
  │    - Extract text from passage into hyperedges    │
  └─────────────────────────────────────────────────┘

Brain analogue:
  - Hippocampal index: A single hippocampal memory trace binds
    MULTIPLE cortical representations (place + person + emotion + time)
    into one coherent episodic index — this IS a hyperedge.
  - Pattern separation in DG: Orthogonalizes overlapping hyperedges
  - Pattern completion in CA3: Retrieves full hyperedge from partial cue

Research basis:
  - HGMem (ICML 2026): Hypergraph memory for complex relational modeling
  - Associative memory theory (Kohonen, 1972): Content-addressable storage
  - Hippocampal indexing theory (Teyler & DiScenna, 1986)
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────

class HyperNode(BaseModel):
    """A node in the hypergraph (entity)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str
    node_type: str = "entity"  # entity, concept, event, location, time
    attributes: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Activation level for spreading activation
    activation: float = 0.0
    resting_level: float = 0.0

    model_config = ConfigDict(frozen=False)


class HyperEdge(BaseModel):
    """
    A hyperedge connecting N nodes simultaneously.

    Unlike binary edges, a hyperedge captures an n-ary relation:
    MEETING(Alice, Bob, Charlie, Cafe, Tuesday) is ONE hyperedge.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_ids: List[str]          # Participating node IDs (ordered)
    relation_type: str = "associated"  # MEETING, DISCUSSION, CAUSES, etc.
    weight: float = 1.0
    source_memory_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def arity(self) -> int:
        """Number of nodes in this hyperedge."""
        return len(self.node_ids)

    @property
    def node_set(self) -> FrozenSet[str]:
        """Frozen set of node IDs for overlap computation."""
        return frozenset(self.node_ids)


class HypergraphConfig(BaseModel):
    """Configuration for the hypergraph memory."""
    max_nodes: int = 10000
    max_edges: int = 50000
    activation_decay: float = 0.85       # Per-hop decay in spreading activation
    min_activation: float = 0.01         # Threshold to stop spreading
    max_spread_depth: int = 3            # Max hops for spreading activation
    hebbian_rate: float = 0.1            # Learning rate for co-activation
    min_overlap_ratio: float = 0.3       # Min Jaccard for "related" hyperedges


class HypergraphQueryResult(BaseModel):
    """Result of a hypergraph query."""
    matching_edges: List[HyperEdge] = Field(default_factory=list)
    activated_nodes: List[Tuple[str, float]] = Field(default_factory=list)
    overlap_scores: List[float] = Field(default_factory=list)


# ── Hypergraph Engine ─────────────────────────────────────────────────

class HypergraphMemory:
    """
    N-ary relational memory fabric using hypergraph structure.

    Extends standard knowledge graphs to capture complex multi-entity
    relations as first-class objects (hyperedges).

    Usage:
        hg = HypergraphMemory()

        # Add a complex event as a hyperedge
        edge_id = hg.add_hyperedge(
            node_labels=["Alice", "Bob", "Charlie", "Cafe", "Tuesday"],
            relation_type="MEETING",
        )

        # Query: what events involved Alice AND Bob?
        results = hg.query_by_nodes(["Alice", "Bob"])

        # Spreading activation from Alice
        activated = hg.spread_activation(["Alice"])
    """

    def __init__(self, config: Optional[HypergraphConfig] = None):
        self.config = config or HypergraphConfig()

        self._nodes: Dict[str, HyperNode] = {}           # id → node
        self._label_to_id: Dict[str, str] = {}            # label → node_id
        self._edges: Dict[str, HyperEdge] = {}            # edge_id → edge
        self._node_to_edges: Dict[str, Set[str]] = defaultdict(set)  # node_id → edge_ids

        logger.info("HypergraphMemory initialized")

    # ── Node Management ───────────────────────────────────────────────

    def get_or_create_node(
        self, label: str, node_type: str = "entity", **attrs
    ) -> HyperNode:
        """Get existing node by label or create a new one."""
        label_lower = label.lower()
        if label_lower in self._label_to_id:
            return self._nodes[self._label_to_id[label_lower]]

        node = HyperNode(label=label, node_type=node_type, attributes=attrs)
        self._nodes[node.id] = node
        self._label_to_id[label_lower] = node.id
        return node

    def get_node(self, label: str) -> Optional[HyperNode]:
        """Get a node by label."""
        label_lower = label.lower()
        if label_lower in self._label_to_id:
            return self._nodes[self._label_to_id[label_lower]]
        return None

    # ── Hyperedge Management ──────────────────────────────────────────

    def add_hyperedge(
        self,
        node_labels: List[str],
        relation_type: str = "associated",
        weight: float = 1.0,
        node_types: Optional[List[str]] = None,
        source_memory_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a hyperedge connecting multiple nodes.

        Creates nodes if they don't exist.
        Returns the hyperedge ID.
        """
        if len(node_labels) < 2:
            raise ValueError("Hyperedge requires at least 2 nodes")

        # Create/get nodes
        types = node_types or ["entity"] * len(node_labels)
        node_ids = []
        for i, label in enumerate(node_labels):
            ntype = types[i] if i < len(types) else "entity"
            node = self.get_or_create_node(label, node_type=ntype)
            node_ids.append(node.id)

        # Create hyperedge
        edge = HyperEdge(
            node_ids=node_ids,
            relation_type=relation_type,
            weight=weight,
            source_memory_id=source_memory_id,
            context=context or {},
        )
        self._edges[edge.id] = edge

        # Update incidence structure
        for nid in node_ids:
            self._node_to_edges[nid].add(edge.id)

        logger.debug(
            f"Hyperedge {edge.id}: {relation_type}({', '.join(node_labels)}) "
            f"arity={edge.arity}"
        )
        return edge.id

    def get_edges_for_node(self, label: str) -> List[HyperEdge]:
        """Get all hyperedges containing a given node."""
        node = self.get_node(label)
        if not node:
            return []
        edge_ids = self._node_to_edges.get(node.id, set())
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    # ── Querying ──────────────────────────────────────────────────────

    def query_by_nodes(
        self,
        node_labels: List[str],
        min_overlap: Optional[float] = None,
        relation_type: Optional[str] = None,
    ) -> HypergraphQueryResult:
        """
        Find hyperedges that contain a subset of the given nodes.

        Args:
            node_labels: Labels of nodes to search for
            min_overlap: Minimum Jaccard overlap ratio (default: config value)
            relation_type: Filter by relation type

        Returns:
            HypergraphQueryResult with matching edges and overlap scores
        """
        min_ov = min_overlap if min_overlap is not None else self.config.min_overlap_ratio

        # Resolve labels to IDs
        query_ids = set()
        for label in node_labels:
            node = self.get_node(label)
            if node:
                query_ids.add(node.id)

        if not query_ids:
            return HypergraphQueryResult()

        # Find candidate edges (any node overlap)
        candidate_edge_ids: Set[str] = set()
        for nid in query_ids:
            candidate_edge_ids.update(self._node_to_edges.get(nid, set()))

        # Score by overlap
        matching = []
        scores = []
        for eid in candidate_edge_ids:
            edge = self._edges.get(eid)
            if not edge:
                continue
            if relation_type and edge.relation_type != relation_type:
                continue

            edge_node_set = set(edge.node_ids)
            intersection = query_ids & edge_node_set
            union = query_ids | edge_node_set
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard >= min_ov:
                matching.append(edge)
                scores.append(jaccard)

        # Sort by overlap score descending
        paired = sorted(zip(matching, scores), key=lambda x: -x[1])
        if paired:
            matching, scores = zip(*paired)
            matching = list(matching)
            scores = list(scores)
        else:
            matching = []
            scores = []

        return HypergraphQueryResult(
            matching_edges=matching,
            overlap_scores=scores,
        )

    def query_by_relation(self, relation_type: str) -> List[HyperEdge]:
        """Get all hyperedges of a given relation type."""
        return [e for e in self._edges.values() if e.relation_type == relation_type]

    # ── Spreading Activation ──────────────────────────────────────────

    def spread_activation(
        self,
        seed_labels: List[str],
        initial_activation: float = 1.0,
        max_depth: Optional[int] = None,
    ) -> HypergraphQueryResult:
        """
        Spread activation through the hypergraph from seed nodes.

        Unlike pairwise spreading, activation flows through HYPEREDGES:
        activating one node in a hyperedge partially activates ALL other
        nodes in that same hyperedge. This captures group-level associations.

        Returns activated nodes sorted by activation level.
        """
        depth = max_depth or self.config.max_spread_depth

        # Reset all activations
        for node in self._nodes.values():
            node.activation = node.resting_level

        # Seed activation
        active_ids: Set[str] = set()
        for label in seed_labels:
            node = self.get_node(label)
            if node:
                node.activation = initial_activation
                active_ids.add(node.id)

        # Spread through hyperedges
        for d in range(depth):
            decay = self.config.activation_decay ** (d + 1)
            new_active: Set[str] = set()

            for nid in active_ids:
                source_act = self._nodes[nid].activation
                if source_act < self.config.min_activation:
                    continue

                # Spread through all hyperedges containing this node
                for eid in self._node_to_edges.get(nid, set()):
                    edge = self._edges.get(eid)
                    if not edge:
                        continue

                    # Activation spread = source × decay × edge_weight / arity
                    # Divided by arity: larger groups spread less per node
                    spread_amount = source_act * decay * edge.weight / edge.arity

                    for target_id in edge.node_ids:
                        if target_id == nid:
                            continue
                        target = self._nodes.get(target_id)
                        if target:
                            target.activation = min(
                                1.0,
                                target.activation + spread_amount,
                            )
                            if target.activation >= self.config.min_activation:
                                new_active.add(target_id)

            active_ids = new_active
            if not active_ids:
                break

        # Collect all activated nodes
        activated = [
            (nid, node.activation)
            for nid, node in self._nodes.items()
            if node.activation >= self.config.min_activation
        ]
        activated.sort(key=lambda x: -x[1])

        return HypergraphQueryResult(activated_nodes=activated)

    # ── Hebbian Learning ──────────────────────────────────────────────

    def strengthen_edge(self, edge_id: str, amount: Optional[float] = None) -> None:
        """Strengthen a hyperedge (Hebbian: co-activation strengthens bonds)."""
        edge = self._edges.get(edge_id)
        if edge:
            delta = amount or self.config.hebbian_rate
            edge.weight = min(5.0, edge.weight + delta)

    def weaken_edge(self, edge_id: str, amount: float = 0.05) -> None:
        """Weaken a hyperedge through decay or inhibition."""
        edge = self._edges.get(edge_id)
        if edge:
            edge.weight = max(0.01, edge.weight - amount)

    # ── Clique Expansion ──────────────────────────────────────────────

    def to_pairwise(self) -> List[Tuple[str, str, str, float]]:
        """
        Convert hypergraph to pairwise edges (clique expansion).

        Each hyperedge {A, B, C} with relation R produces:
          (A, R, B, weight), (A, R, C, weight), (B, R, C, weight)

        Useful for compatibility with standard graph algorithms.
        """
        pairs = []
        for edge in self._edges.values():
            node_ids = edge.node_ids
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    n1 = self._nodes.get(node_ids[i])
                    n2 = self._nodes.get(node_ids[j])
                    if n1 and n2:
                        pairs.append((
                            n1.label,
                            edge.relation_type,
                            n2.label,
                            edge.weight / edge.arity,  # Distribute weight
                        ))
        return pairs

    # ── Text Extraction ───────────────────────────────────────────────

    def ingest_text(
        self, text: str, source_memory_id: Optional[str] = None
    ) -> List[str]:
        """
        Extract hyperedges from natural language text.

        Identifies multi-entity events and creates n-ary hyperedges.

        Returns list of created hyperedge IDs.
        """
        edge_ids = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Extract entities (capitalized words, excluding common words)
            EXCLUDE = {
                'The', 'This', 'That', 'These', 'Those', 'There', 'Their',
                'When', 'Where', 'What', 'Which', 'While', 'With', 'Would',
                'Could', 'Should', 'After', 'Before', 'During', 'About',
                'Under', 'Until', 'Without', 'Within', 'Between', 'Through',
                'Into', 'From', 'Then', 'Than', 'Also', 'Just', 'Even',
                'Some', 'Many', 'Much', 'Most', 'Each', 'Every', 'Both',
                'However', 'Moreover', 'Furthermore', 'Although', 'Because',
                'Since', 'Hence', 'Therefore', 'Meanwhile', 'Indeed',
                'Several', 'Various', 'According', 'Recently', 'Yesterday',
                'Today', 'Tomorrow', 'Here', 'Now', 'Well', 'First',
                'Second', 'Third', 'Finally', 'Next', 'Last', 'New', 'Old',
            }
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
            entities = [w for w in words if w not in EXCLUDE and len(w) > 1]

            if len(entities) < 2:
                continue

            # Deduplicate while preserving order
            seen = set()
            unique_entities = []
            for e in entities:
                if e.lower() not in seen:
                    seen.add(e.lower())
                    unique_entities.append(e)

            if len(unique_entities) < 2:
                continue

            # Determine relation type from sentence context
            relation_type = self._classify_relation(sentence)

            # Determine node types
            node_types = self._classify_node_types(unique_entities, sentence)

            # Create hyperedge
            eid = self.add_hyperedge(
                node_labels=unique_entities,
                relation_type=relation_type,
                node_types=node_types,
                source_memory_id=source_memory_id,
            )
            edge_ids.append(eid)

        return edge_ids

    def _classify_relation(self, sentence: str) -> str:
        """Classify the relation type from sentence context."""
        lower = sentence.lower()

        patterns = [
            (r'\b(?:met|meeting|gathered|assembled|convened)\b', "MEETING"),
            (r'\b(?:discussed|debated|talked|conversation)\b', "DISCUSSION"),
            (r'\b(?:collaborated|worked together|partnered)\b', "COLLABORATION"),
            (r'\b(?:caused|led to|resulted in|triggered)\b', "CAUSATION"),
            (r'\b(?:discovered|found|revealed|uncovered)\b', "DISCOVERY"),
            (r'\b(?:created|built|developed|designed|invented)\b', "CREATION"),
            (r'\b(?:agreed|consensus|accepted|approved)\b', "AGREEMENT"),
            (r'\b(?:disagreed|opposed|rejected|disputed)\b', "DISAGREEMENT"),
            (r'\b(?:taught|learned|studied|trained|mentored)\b', "EDUCATION"),
            (r'\b(?:traveled|visited|went to|arrived)\b', "TRAVEL"),
            (r'\b(?:presented|showed|demonstrated|exhibited)\b', "PRESENTATION"),
            (r'\b(?:competed|won|lost|played|matched)\b', "COMPETITION"),
        ]

        for pattern, rtype in patterns:
            if re.search(pattern, lower):
                return rtype

        return "ASSOCIATED"

    def _classify_node_types(self, entities: List[str], sentence: str) -> List[str]:
        """Classify node types based on context."""
        lower = sentence.lower()
        types = []

        # Simple heuristics for type classification
        location_indicators = {
            'park', 'cafe', 'office', 'lab', 'university', 'school',
            'hospital', 'airport', 'station', 'building', 'room', 'hall',
            'center', 'city', 'town', 'street', 'avenue', 'plaza',
            'library', 'museum', 'church', 'restaurant', 'hotel',
        }
        time_indicators = {
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'january', 'february', 'march',
            'april', 'may', 'june', 'july', 'august', 'september',
            'october', 'november', 'december', 'morning', 'evening',
            'afternoon', 'night', 'noon', 'midnight',
        }

        for entity in entities:
            e_lower = entity.lower()
            if e_lower in location_indicators:
                types.append("location")
            elif e_lower in time_indicators:
                types.append("time")
            elif any(ind in lower for ind in [f'project {e_lower}', f'{e_lower} project']):
                types.append("concept")
            else:
                types.append("entity")

        return types

    # ── Incidence Matrix ──────────────────────────────────────────────

    def get_incidence_info(self) -> Dict[str, Any]:
        """
        Get incidence matrix information.

        The incidence matrix B has shape (|V| × |E|) where:
        B[i,j] = 1 if node i is in hyperedge j, 0 otherwise.

        Returns dict with node/edge labels and the matrix as nested lists.
        """
        node_ids = sorted(self._nodes.keys())
        edge_ids = sorted(self._edges.keys())

        node_labels = [self._nodes[nid].label for nid in node_ids]
        edge_labels = [
            f"{self._edges[eid].relation_type}({self._edges[eid].arity})"
            for eid in edge_ids
        ]

        matrix = []
        for nid in node_ids:
            row = []
            for eid in edge_ids:
                edge = self._edges[eid]
                row.append(1 if nid in edge.node_ids else 0)
            matrix.append(row)

        return {
            "node_labels": node_labels,
            "edge_labels": edge_labels,
            "matrix": matrix,
            "shape": (len(node_ids), len(edge_ids)),
        }

    # ── Statistics ────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """Get hypergraph statistics."""
        arities = [e.arity for e in self._edges.values()]
        avg_arity = sum(arities) / len(arities) if arities else 0.0
        max_arity = max(arities) if arities else 0

        relation_counts: Dict[str, int] = defaultdict(int)
        for edge in self._edges.values():
            relation_counts[edge.relation_type] += 1

        type_counts: Dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            type_counts[node.node_type] += 1

        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "avg_arity": round(avg_arity, 2),
            "max_arity": max_arity,
            "relation_types": dict(relation_counts),
            "node_types": dict(type_counts),
            "pairwise_equivalent_edges": sum(
                e.arity * (e.arity - 1) // 2 for e in self._edges.values()
            ),
        }

    # ── Serialization ─────────────────────────────────────────────────

    def export_state(self) -> Dict[str, Any]:
        """Export hypergraph state for persistence."""
        return {
            "nodes": {nid: n.model_dump(mode="json") for nid, n in self._nodes.items()},
            "edges": {eid: e.model_dump(mode="json") for eid, e in self._edges.items()},
            "label_to_id": dict(self._label_to_id),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import hypergraph state from exported dict."""
        for nid, ndata in state.get("nodes", {}).items():
            node = HyperNode(**ndata)
            self._nodes[nid] = node

        self._label_to_id = state.get("label_to_id", {})

        for eid, edata in state.get("edges", {}).items():
            edge = HyperEdge(**edata)
            self._edges[eid] = edge
            for nid in edge.node_ids:
                self._node_to_edges[nid].add(eid)
