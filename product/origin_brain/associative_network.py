"""
Associative Memory Network — Graph-Based Semantic Memory Fabric

Implements a neuroscience-inspired associative network where memories are
connected by weighted edges. Enables:

1. SPREADING ACTIVATION (Collins & Loftus 1975):
   When a node activates, activation spreads to neighbors, decaying with
   distance. This produces priming effects, semantic similarity retrieval,
   and multi-hop reasoning.

2. ENTITY-RELATION-ENTITY TRIPLES:
   Structured knowledge representation as (subject, predicate, object) triples
   with confidence scores and temporal metadata. Similar to Zep's Graphiti
   but with biological activation dynamics.

3. HEBBIAN LEARNING:
   "Neurons that fire together wire together" — co-activated nodes strengthen
   their connections. This is how the brain builds semantic networks.

4. COMPETITIVE INHIBITION:
   Strongly activated nodes suppress weakly activated neighbors (lateral
   inhibition), sharpening retrieval and reducing interference.

Brain analogue:
  - Semantic network = temporal-parietal cortical areas
  - Spreading activation = neural cascade through axon collaterals
  - Hebbian learning = LTP at synapses
  - Inhibition = GABAergic interneurons

Research basis:
  - Collins & Loftus 1975 — Spreading Activation Theory of Semantic Processing
  - Anderson 1983 — ACT* (Adaptive Control of Thought)
  - Zep Graphiti 2026 — Entity-relationship extraction for AI memory
  - CogMem (EMNLP 2026) — Knowledge graph for reconstructive retrieval
  - Origin Brain thesis: graph structure + spreading activation + biological
    decay = human-like associative memory without LLM at inference time
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Models ──────────────────────────────────────────────────────────────────

class AssociativeNode(BaseModel):
    """A node in the associative network — represents a concept or entity."""
    id: str
    label: str                           # Human-readable label
    category: str = "concept"            # entity, concept, event, attribute
    activation: float = 0.0              # Current activation level [0, 1]
    resting_level: float = 0.0           # Baseline activation (frequency-based)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def activate(self, amount: float) -> None:
        """Increase activation, capped at 1.0."""
        self.activation = min(1.0, self.activation + amount)
        self.last_activated = datetime.now(timezone.utc)
        self.access_count += 1

    def decay(self, factor: float = 0.1) -> None:
        """Decay activation toward resting level."""
        self.activation = self.resting_level + (
            (self.activation - self.resting_level) * math.exp(-factor)
        )


class AssociativeEdge(BaseModel):
    """A weighted, typed edge between two nodes."""
    source_id: str
    target_id: str
    relation: str = "associated_with"    # predicate / relation type
    weight: float = 0.5                  # Strength of association [0, 1]
    bidirectional: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_strengthened: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    co_activation_count: int = 1         # Hebbian: how many times co-activated
    confidence: float = 1.0              # How confident we are in this relation
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def edge_key(self) -> str:
        return f"{self.source_id}|{self.relation}|{self.target_id}"

    def strengthen(self, amount: float = 0.05) -> None:
        """Hebbian strengthening — co-activation increases weight."""
        self.weight = min(1.0, self.weight + amount)
        self.co_activation_count += 1
        self.last_strengthened = datetime.now(timezone.utc)

    def weaken(self, amount: float = 0.01) -> None:
        """Decay — unused connections weaken over time."""
        self.weight = max(0.01, self.weight - amount)


class SpreadingActivationResult(BaseModel):
    """Result of spreading activation from a seed set."""
    activated_nodes: List[Tuple[str, float]]   # (node_id, activation_level)
    paths: Dict[str, List[str]] = Field(default_factory=dict)  # node_id -> path from seed
    hops: int = 0
    total_activation: float = 0.0


class AssociativeNetworkConfig(BaseModel):
    """Configuration for the associative network."""
    max_nodes: int = 50000
    max_edges_per_node: int = 200
    activation_decay_rate: float = 0.3          # Per-hop decay for spreading activation
    inhibition_threshold: float = 0.1           # Below this, node is inhibited
    spreading_depth: int = 3                    # Max hops for spreading activation
    hebbian_learning_rate: float = 0.05         # How fast co-activation strengthens edges
    resting_level_boost: float = 0.01           # How much each access boosts resting level
    edge_decay_rate: float = 0.001              # How fast unused edges weaken
    min_activation_to_spread: float = 0.05      # Min activation to propagate further
    lateral_inhibition: bool = True             # Enable competitive inhibition
    inhibition_radius: int = 1                  # How far inhibition spreads


# ── Entity Extraction ───────────────────────────────────────────────────────

# Relation patterns for extracting structured triples from text
RELATION_PATTERNS = [
    # is-a / type
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an)\s+(.+?)(?:\.|,|$)",
     "is_a"),
    # has / possesses
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+(?:a|an)?\s*(.+?)(?:\.|,|$)",
     "has"),
    # works at / employed
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works?\s+(?:at|for)|(?:is\s+)?employed\s+(?:at|by))\s+(.+?)(?:\.|,|$)",
     "works_at"),
    # lives in / located
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:lives?\s+in|resides?\s+in|(?:is\s+)?located\s+in)\s+(.+?)(?:\.|,|$)",
     "located_in"),
    # likes / prefers / enjoys
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:likes?|prefers?|enjoys?|loves?)\s+(.+?)(?:\.|,|$)",
     "prefers"),
    # married to / related to
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:(?:is\s+)?married\s+to|(?:is\s+)?(?:the\s+)?(?:sister|brother|mother|father|wife|husband|partner)\s+of)\s+(.+?)(?:\.|,|$)",
     "related_to"),
    # created / built / made
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:created?|built|made|developed|designed)\s+(.+?)(?:\.|,|$)",
     "created"),
    # studies / learning
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:studies|(?:is\s+)?(?:learning|studying))\s+(.+?)(?:\.|,|$)",
     "studies"),
    # general verb pattern (fallback)
    (r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+((?:will\s+)?(?:start|join|move|leave|visit|attend|buy|sell)(?:s|ed|ing)?)\s+(.+?)(?:\.|,|$)",
     "_verb_"),
]


def _make_node_id(label: str) -> str:
    """Create a stable ID for a node label."""
    normalized = label.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def extract_entities_and_relations(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract (subject, predicate, object) triples from natural language text.
    
    Returns list of (subject, relation, object) tuples.
    """
    triples = []
    
    for pattern, relation in RELATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            if relation == "_verb_":
                # General verb pattern: group(1)=subject, group(2)=verb, group(3)=object
                if len(groups) >= 3:
                    subject = groups[0].strip()
                    verb = groups[1].strip().lower()
                    obj = groups[2].strip()
                    triples.append((subject, verb, obj))
            else:
                subject = groups[0].strip()
                obj = groups[1].strip() if len(groups) > 1 else ""
                if subject and obj:
                    triples.append((subject, relation, obj))
    
    return triples


# ── Main Engine ─────────────────────────────────────────────────────────────

class AssociativeNetwork:
    """
    Graph-based associative memory network with spreading activation.
    
    Usage:
        net = AssociativeNetwork()
        
        # Add knowledge
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Alice", "prefers", "Python")
        net.add_triple("Bob", "works_at", "Google")
        
        # Query via spreading activation
        result = net.activate_and_spread(["Alice"])
        # -> Bob gets activated (shared employer), Python gets activated (preference)
        
        # Extract from text
        net.ingest_text("Alice works at Google and likes Python.")
        
        # Multi-hop query
        hops = net.find_path("Alice", "Bob")
        # -> Alice -> Google -> Bob
    """
    
    def __init__(self, config: Optional[AssociativeNetworkConfig] = None):
        self.config = config or AssociativeNetworkConfig()
        self._nodes: Dict[str, AssociativeNode] = {}
        self._edges: Dict[str, AssociativeEdge] = {}  # edge_key -> edge
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # node_id -> set of edge_keys
        self._label_to_id: Dict[str, str] = {}  # normalized label -> node_id
        
        logger.info("AssociativeNetwork initialized")
    
    # ── Node Management ─────────────────────────────────────────────────
    
    def get_or_create_node(
        self,
        label: str,
        category: str = "concept",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssociativeNode:
        """Get existing node or create new one."""
        normalized = label.strip().lower()
        
        if normalized in self._label_to_id:
            node_id = self._label_to_id[normalized]
            return self._nodes[node_id]
        
        # Evict if at capacity (LRU by last_activated)
        if len(self._nodes) >= self.config.max_nodes:
            self._evict_oldest_node()
        
        node_id = _make_node_id(label)
        node = AssociativeNode(
            id=node_id,
            label=label.strip(),
            category=category,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        self._label_to_id[normalized] = node_id
        return node
    
    def get_node_by_label(self, label: str) -> Optional[AssociativeNode]:
        """Look up a node by its label."""
        normalized = label.strip().lower()
        node_id = self._label_to_id.get(normalized)
        if node_id:
            return self._nodes.get(node_id)
        return None
    
    def _evict_oldest_node(self) -> None:
        """Evict the least recently activated node."""
        if not self._nodes:
            return
        oldest = min(self._nodes.values(), key=lambda n: n.last_activated)
        self._remove_node(oldest.id)
    
    def _remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return
        node = self._nodes[node_id]
        
        # Remove edges
        edge_keys = list(self._adjacency.get(node_id, set()))
        for ek in edge_keys:
            if ek in self._edges:
                edge = self._edges[ek]
                # Remove from other node's adjacency
                other_id = edge.target_id if edge.source_id == node_id else edge.source_id
                self._adjacency[other_id].discard(ek)
                del self._edges[ek]
        
        # Remove from indices
        self._adjacency.pop(node_id, None)
        self._label_to_id.pop(node.label.strip().lower(), None)
        del self._nodes[node_id]
    
    # ── Edge Management ─────────────────────────────────────────────────
    
    def add_edge(
        self,
        source_label: str,
        target_label: str,
        relation: str = "associated_with",
        weight: float = 0.5,
        bidirectional: bool = True,
        confidence: float = 1.0,
        source_category: str = "concept",
        target_category: str = "concept",
    ) -> AssociativeEdge:
        """Add or strengthen an edge between two nodes."""
        source = self.get_or_create_node(source_label, category=source_category)
        target = self.get_or_create_node(target_label, category=target_category)
        
        edge_key = f"{source.id}|{relation}|{target.id}"
        
        if edge_key in self._edges:
            # Hebbian strengthening — existing connection gets stronger
            existing = self._edges[edge_key]
            existing.strengthen(self.config.hebbian_learning_rate)
            return existing
        
        # Check edge capacity per node
        if len(self._adjacency[source.id]) >= self.config.max_edges_per_node:
            self._prune_weakest_edges(source.id)
        
        edge = AssociativeEdge(
            source_id=source.id,
            target_id=target.id,
            relation=relation,
            weight=weight,
            bidirectional=bidirectional,
            confidence=confidence,
        )
        self._edges[edge_key] = edge
        self._adjacency[source.id].add(edge_key)
        self._adjacency[target.id].add(edge_key)
        
        return edge
    
    def _prune_weakest_edges(self, node_id: str) -> None:
        """Remove the weakest edge from a node."""
        edge_keys = self._adjacency.get(node_id, set())
        if not edge_keys:
            return
        weakest_key = min(
            edge_keys,
            key=lambda ek: self._edges[ek].weight if ek in self._edges else 0
        )
        if weakest_key in self._edges:
            edge = self._edges[weakest_key]
            other_id = edge.target_id if edge.source_id == node_id else edge.source_id
            self._adjacency[other_id].discard(weakest_key)
            self._adjacency[node_id].discard(weakest_key)
            del self._edges[weakest_key]
    
    # ── High-Level API ──────────────────────────────────────────────────
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        weight: float = 0.6,
        confidence: float = 1.0,
    ) -> AssociativeEdge:
        """Add a (subject, predicate, object) triple to the network."""
        return self.add_edge(
            source_label=subject,
            target_label=obj,
            relation=predicate,
            weight=weight,
            confidence=confidence,
            source_category="entity",
            target_category="entity",
        )
    
    def ingest_text(self, text: str) -> List[AssociativeEdge]:
        """
        Extract entities and relations from text and add them to the network.
        
        Returns list of edges created/strengthened.
        """
        triples = extract_entities_and_relations(text)
        edges = []
        for subject, predicate, obj in triples:
            edge = self.add_triple(subject, predicate, obj)
            edges.append(edge)
        return edges
    
    # ── Spreading Activation ────────────────────────────────────────────
    
    def activate_and_spread(
        self,
        seed_labels: List[str],
        initial_activation: float = 1.0,
        max_depth: Optional[int] = None,
        min_activation: Optional[float] = None,
    ) -> SpreadingActivationResult:
        """
        Activate seed nodes and spread activation through the network.
        
        This is the core retrieval mechanism — replaces keyword matching
        with neurally-inspired activation spreading.
        
        Algorithm:
        1. Activate seed nodes to initial_activation
        2. For each active node, spread activation to neighbors:
             neighbor.activation += edge.weight * node.activation * decay
        3. Repeat for max_depth hops
        4. Apply lateral inhibition (if enabled)
        5. Return all nodes sorted by activation level
        
        Args:
            seed_labels: Labels of nodes to activate initially
            initial_activation: Starting activation level for seeds
            max_depth: Max hops (default: config.spreading_depth)
            min_activation: Min activation to include in results
            
        Returns:
            SpreadingActivationResult with activated nodes and paths
        """
        depth = max_depth or self.config.spreading_depth
        min_act = min_activation or self.config.min_activation_to_spread
        decay = self.config.activation_decay_rate
        
        # Reset all activations
        for node in self._nodes.values():
            node.activation = node.resting_level
        
        # Step 1: Activate seeds
        active_set: Dict[str, float] = {}
        paths: Dict[str, List[str]] = {}
        
        for label in seed_labels:
            node = self.get_node_by_label(label)
            if node:
                node.activate(initial_activation)
                active_set[node.id] = node.activation
                paths[node.id] = [node.label]
        
        if not active_set:
            return SpreadingActivationResult(
                activated_nodes=[], hops=0, total_activation=0.0
            )
        
        # Step 2-3: Spread activation
        frontier = set(active_set.keys())
        
        for hop in range(depth):
            next_frontier: Set[str] = set()
            
            for node_id in frontier:
                node = self._nodes.get(node_id)
                if not node or node.activation < min_act:
                    continue
                
                # Get all edges for this node
                for edge_key in self._adjacency.get(node_id, set()):
                    edge = self._edges.get(edge_key)
                    if not edge:
                        continue
                    
                    # Determine neighbor
                    if edge.source_id == node_id:
                        neighbor_id = edge.target_id
                    elif edge.bidirectional and edge.target_id == node_id:
                        neighbor_id = edge.source_id
                    else:
                        continue
                    
                    neighbor = self._nodes.get(neighbor_id)
                    if not neighbor:
                        continue
                    
                    # Compute activation to spread
                    spread_amount = (
                        node.activation * edge.weight * math.exp(-decay * (hop + 1))
                    )
                    
                    if spread_amount < min_act:
                        continue
                    
                    # Activate neighbor
                    old_activation = neighbor.activation
                    neighbor.activate(spread_amount)
                    
                    # Track if newly activated above threshold
                    if neighbor.id not in active_set or neighbor.activation > active_set.get(neighbor.id, 0):
                        active_set[neighbor.id] = neighbor.activation
                        next_frontier.add(neighbor.id)
                        
                        # Track path
                        if neighbor.id not in paths:
                            parent_path = paths.get(node_id, [node.label])
                            paths[neighbor.id] = parent_path + [neighbor.label]
                    
                    # Hebbian: co-activation strengthens edge
                    if old_activation > min_act and spread_amount > min_act:
                        edge.strengthen(self.config.hebbian_learning_rate * 0.1)
            
            frontier = next_frontier
            if not frontier:
                break
        
        # Step 4: Lateral inhibition
        if self.config.lateral_inhibition:
            self._apply_lateral_inhibition(active_set)
        
        # Step 5: Collect and sort results
        activated = [
            (node_id, self._nodes[node_id].activation)
            for node_id in active_set
            if node_id in self._nodes and self._nodes[node_id].activation > min_act
        ]
        activated.sort(key=lambda x: x[1], reverse=True)
        
        total = sum(act for _, act in activated)
        
        return SpreadingActivationResult(
            activated_nodes=activated,
            paths=paths,
            hops=depth,
            total_activation=total,
        )
    
    def _apply_lateral_inhibition(self, active_set: Dict[str, float]) -> None:
        """
        Apply lateral inhibition — strongly activated nodes suppress
        weakly activated neighbors.
        
        This sharpens retrieval by emphasizing the most relevant nodes
        and suppressing noise. Mimics GABAergic interneuron function.
        """
        inhibition_targets: Dict[str, float] = {}
        
        for node_id, activation in active_set.items():
            if activation < 0.3:  # Only strong nodes inhibit
                continue
            
            # Find neighbors within inhibition radius
            for edge_key in self._adjacency.get(node_id, set()):
                edge = self._edges.get(edge_key)
                if not edge:
                    continue
                
                neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
                neighbor = self._nodes.get(neighbor_id)
                
                if neighbor and neighbor.activation < activation * 0.5:
                    # Inhibit weakly activated neighbor
                    inhibition = activation * 0.1  # 10% of strong node's activation
                    inhibition_targets[neighbor_id] = inhibition_targets.get(
                        neighbor_id, 0
                    ) + inhibition
        
        # Apply inhibition
        for node_id, inhibition in inhibition_targets.items():
            node = self._nodes.get(node_id)
            if node:
                node.activation = max(0.0, node.activation - inhibition)
    
    # ── Query Methods ───────────────────────────────────────────────────
    
    def find_path(
        self,
        source_label: str,
        target_label: str,
        max_depth: int = 5,
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes via BFS.
        Returns list of node labels, or None if no path exists.
        """
        source = self.get_node_by_label(source_label)
        target = self.get_node_by_label(target_label)
        
        if not source or not target:
            return None
        
        if source.id == target.id:
            return [source.label]
        
        # BFS
        visited = {source.id}
        queue = [(source.id, [source.label])]
        
        for _ in range(max_depth):
            next_queue = []
            for node_id, path in queue:
                for edge_key in self._adjacency.get(node_id, set()):
                    edge = self._edges.get(edge_key)
                    if not edge:
                        continue
                    
                    # Get neighbor
                    if edge.source_id == node_id:
                        neighbor_id = edge.target_id
                    elif edge.bidirectional:
                        neighbor_id = edge.source_id
                    else:
                        continue
                    
                    if neighbor_id in visited:
                        continue
                    
                    neighbor = self._nodes.get(neighbor_id)
                    if not neighbor:
                        continue
                    
                    new_path = path + [neighbor.label]
                    
                    if neighbor_id == target.id:
                        return new_path
                    
                    visited.add(neighbor_id)
                    next_queue.append((neighbor_id, new_path))
            
            queue = next_queue
            if not queue:
                break
        
        return None
    
    def get_neighbors(
        self,
        label: str,
        relation: Optional[str] = None,
        min_weight: float = 0.0,
    ) -> List[Tuple[str, str, float]]:
        """
        Get all neighbors of a node.
        Returns list of (neighbor_label, relation, weight).
        """
        node = self.get_node_by_label(label)
        if not node:
            return []
        
        results = []
        for edge_key in self._adjacency.get(node.id, set()):
            edge = self._edges.get(edge_key)
            if not edge or edge.weight < min_weight:
                continue
            
            if relation and edge.relation != relation:
                continue
            
            # Determine neighbor
            if edge.source_id == node.id:
                neighbor_id = edge.target_id
            elif edge.bidirectional:
                neighbor_id = edge.source_id
            else:
                continue
            
            neighbor = self._nodes.get(neighbor_id)
            if neighbor:
                results.append((neighbor.label, edge.relation, edge.weight))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Query triples with optional wildcard matching.
        Returns list of (subject, predicate, object, weight).
        
        Examples:
            query_triples(subject="Alice")  # All facts about Alice
            query_triples(predicate="works_at")  # Who works where
            query_triples(subject="Alice", predicate="prefers")  # Alice's preferences
        """
        results = []
        
        for edge in self._edges.values():
            source_node = self._nodes.get(edge.source_id)
            target_node = self._nodes.get(edge.target_id)
            
            if not source_node or not target_node:
                continue
            
            # Check subject filter
            if subject and source_node.label.lower() != subject.lower():
                continue
            
            # Check predicate filter
            if predicate and edge.relation != predicate:
                continue
            
            # Check object filter
            if obj and target_node.label.lower() != obj.lower():
                continue
            
            results.append((
                source_node.label,
                edge.relation,
                target_node.label,
                edge.weight,
            ))
        
        results.sort(key=lambda x: x[3], reverse=True)
        return results
    
    def get_entity_summary(self, label: str) -> Dict[str, Any]:
        """
        Get a complete summary of an entity — all its connections grouped by relation.
        Like a knowledge card.
        """
        node = self.get_node_by_label(label)
        if not node:
            return {"entity": label, "found": False}
        
        relations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for edge_key in self._adjacency.get(node.id, set()):
            edge = self._edges.get(edge_key)
            if not edge:
                continue
            
            if edge.source_id == node.id:
                neighbor = self._nodes.get(edge.target_id)
                direction = "outgoing"
            elif edge.bidirectional:
                neighbor = self._nodes.get(edge.source_id)
                direction = "incoming"
            else:
                continue
            
            if neighbor:
                relations[edge.relation].append({
                    "target": neighbor.label,
                    "weight": round(edge.weight, 3),
                    "direction": direction,
                    "co_activations": edge.co_activation_count,
                })
        
        return {
            "entity": node.label,
            "found": True,
            "category": node.category,
            "activation": round(node.activation, 3),
            "resting_level": round(node.resting_level, 3),
            "access_count": node.access_count,
            "relations": dict(relations),
            "total_connections": sum(len(v) for v in relations.values()),
        }
    
    # ── Maintenance ─────────────────────────────────────────────────────
    
    def decay_all(self, factor: Optional[float] = None) -> int:
        """Decay all activations and edge weights. Returns count of decayed."""
        decay_factor = factor or self.config.activation_decay_rate
        
        count = 0
        for node in self._nodes.values():
            if node.activation > node.resting_level:
                node.decay(decay_factor)
                count += 1
        
        # Decay edges too
        for edge in self._edges.values():
            edge.weaken(self.config.edge_decay_rate)
        
        return count
    
    def boost_resting_levels(self) -> None:
        """Boost resting levels based on access frequency (long-term potentiation)."""
        if not self._nodes:
            return
        max_access = max(n.access_count for n in self._nodes.values())
        if max_access == 0:
            return
        
        for node in self._nodes.values():
            # Resting level proportional to log(access_count)
            if node.access_count > 0:
                node.resting_level = min(
                    0.3,
                    self.config.resting_level_boost * math.log(1 + node.access_count)
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        if not self._nodes:
            return {
                "node_count": 0,
                "edge_count": 0,
                "avg_degree": 0,
                "avg_weight": 0,
                "avg_activation": 0,
                "categories": {},
            }
        
        degrees = [len(self._adjacency.get(n.id, set())) for n in self._nodes.values()]
        weights = [e.weight for e in self._edges.values()]
        activations = [n.activation for n in self._nodes.values()]
        categories = defaultdict(int)
        for n in self._nodes.values():
            categories[n.category] += 1
        
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "avg_weight": sum(weights) / len(weights) if weights else 0,
            "avg_activation": sum(activations) / len(activations) if activations else 0,
            "categories": dict(categories),
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the graph as a serializable dict."""
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "edges": [e.model_dump() for e in self._edges.values()],
        }
    
    def import_graph(self, data: Dict[str, Any]) -> None:
        """Import a graph from exported data."""
        for node_data in data.get("nodes", []):
            node = AssociativeNode(**node_data)
            self._nodes[node.id] = node
            self._label_to_id[node.label.strip().lower()] = node.id
        
        for edge_data in data.get("edges", []):
            edge = AssociativeEdge(**edge_data)
            self._edges[edge.edge_key] = edge
            self._adjacency[edge.source_id].add(edge.edge_key)
            self._adjacency[edge.target_id].add(edge.edge_key)
