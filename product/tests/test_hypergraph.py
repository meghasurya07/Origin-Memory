"""
Tests for HypergraphMemory — N-ary Relational Memory Fabric.

Tests verify:
- Node creation and deduplication
- Hyperedge creation with multiple nodes
- Query by node overlap (Jaccard scoring)
- Query by relation type
- Spreading activation through hyperedges
- Hebbian strengthening/weakening
- Clique expansion to pairwise
- Text extraction into hyperedges
- Incidence matrix computation
- Serialization (export/import)
- Statistics reporting
- Edge cases
"""

import pytest
from origin_brain.hypergraph import (
    HypergraphMemory,
    HypergraphConfig,
    HyperNode,
    HyperEdge,
    HypergraphQueryResult,
)


class TestNodeManagement:
    """Test node creation and lookup."""

    def test_create_node(self):
        hg = HypergraphMemory()
        node = hg.get_or_create_node("Alice")
        assert node.label == "Alice"
        assert node.node_type == "entity"
        assert node.id in hg._nodes

    def test_deduplication(self):
        hg = HypergraphMemory()
        n1 = hg.get_or_create_node("Alice")
        n2 = hg.get_or_create_node("alice")  # Case insensitive
        assert n1.id == n2.id

    def test_get_node_exists(self):
        hg = HypergraphMemory()
        hg.get_or_create_node("Bob")
        assert hg.get_node("bob") is not None
        assert hg.get_node("Bob") is not None

    def test_get_node_missing(self):
        hg = HypergraphMemory()
        assert hg.get_node("Nobody") is None

    def test_node_types(self):
        hg = HypergraphMemory()
        n = hg.get_or_create_node("Lab", node_type="location")
        assert n.node_type == "location"


class TestHyperedgeCreation:
    """Test hyperedge creation."""

    def test_basic_hyperedge(self):
        hg = HypergraphMemory()
        eid = hg.add_hyperedge(["Alice", "Bob", "Charlie"], relation_type="MEETING")
        assert eid in hg._edges
        edge = hg._edges[eid]
        assert edge.arity == 3
        assert edge.relation_type == "MEETING"

    def test_binary_edge(self):
        hg = HypergraphMemory()
        eid = hg.add_hyperedge(["Alice", "Bob"])
        assert hg._edges[eid].arity == 2

    def test_large_hyperedge(self):
        hg = HypergraphMemory()
        nodes = [f"Person{i}" for i in range(10)]
        eid = hg.add_hyperedge(nodes, relation_type="CONFERENCE")
        assert hg._edges[eid].arity == 10

    def test_single_node_raises(self):
        hg = HypergraphMemory()
        with pytest.raises(ValueError):
            hg.add_hyperedge(["Alone"])

    def test_creates_nodes_automatically(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["X", "Y", "Z"])
        assert len(hg._nodes) == 3

    def test_node_types_on_edge(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(
            ["Alice", "Lab", "Tuesday"],
            node_types=["entity", "location", "time"],
        )
        assert hg.get_node("Lab").node_type == "location"
        assert hg.get_node("Tuesday").node_type == "time"

    def test_incidence_updated(self):
        hg = HypergraphMemory()
        eid = hg.add_hyperedge(["A", "B", "C"])
        a_node = hg.get_node("A")
        assert eid in hg._node_to_edges[a_node.id]

    def test_source_memory_id(self):
        hg = HypergraphMemory()
        eid = hg.add_hyperedge(["A", "B"], source_memory_id="mem-123")
        assert hg._edges[eid].source_memory_id == "mem-123"


class TestQueryByNodes:
    """Test node-overlap queries."""

    def test_exact_match(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["Alice", "Bob", "Charlie"], relation_type="MEETING")
        result = hg.query_by_nodes(["Alice", "Bob", "Charlie"], min_overlap=1.0)
        assert len(result.matching_edges) == 1

    def test_partial_match(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["Alice", "Bob", "Charlie"], relation_type="MEETING")
        result = hg.query_by_nodes(["Alice", "Bob"], min_overlap=0.3)
        assert len(result.matching_edges) == 1
        assert result.overlap_scores[0] > 0.3

    def test_no_match(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["Alice", "Bob"], relation_type="MEETING")
        result = hg.query_by_nodes(["Charlie", "Dave"])
        assert len(result.matching_edges) == 0

    def test_unknown_nodes(self):
        hg = HypergraphMemory()
        result = hg.query_by_nodes(["Nobody"])
        assert len(result.matching_edges) == 0

    def test_filter_by_relation(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["Alice", "Bob"], relation_type="MEETING")
        hg.add_hyperedge(["Alice", "Bob"], relation_type="DISCUSSION")

        r1 = hg.query_by_nodes(["Alice"], relation_type="MEETING", min_overlap=0.0)
        assert len(r1.matching_edges) == 1
        assert r1.matching_edges[0].relation_type == "MEETING"

    def test_multiple_matches_sorted(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["Alice", "Bob", "Charlie"])
        hg.add_hyperedge(["Alice", "Dave"])

        result = hg.query_by_nodes(["Alice", "Bob"], min_overlap=0.0)
        assert len(result.matching_edges) == 2
        # First match should have higher overlap (2/3 vs 1/3)
        assert result.overlap_scores[0] >= result.overlap_scores[1]


class TestQueryByRelation:
    """Test relation-type queries."""

    def test_query_by_relation(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["A", "B"], relation_type="MEETING")
        hg.add_hyperedge(["C", "D"], relation_type="MEETING")
        hg.add_hyperedge(["E", "F"], relation_type="DISCUSSION")

        meetings = hg.query_by_relation("MEETING")
        assert len(meetings) == 2

    def test_empty_relation(self):
        hg = HypergraphMemory()
        assert len(hg.query_by_relation("NONEXISTENT")) == 0


class TestSpreadingActivation:
    """Test activation spreading through hyperedges."""

    def test_basic_spread(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["Alice", "Bob", "Charlie"], relation_type="MEETING")

        result = hg.spread_activation(["Alice"])
        activated_ids = {nid for nid, _ in result.activated_nodes}

        bob = hg.get_node("Bob")
        charlie = hg.get_node("Charlie")
        assert bob.id in activated_ids
        assert charlie.id in activated_ids

    def test_activation_decay(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["Alice", "Bob"])
        hg.add_hyperedge(["Bob", "Charlie"])

        result = hg.spread_activation(["Alice"])
        activations = {nid: act for nid, act in result.activated_nodes}

        alice = hg.get_node("Alice")
        bob = hg.get_node("Bob")
        charlie = hg.get_node("Charlie")

        # Alice has highest (seed), Bob intermediate, Charlie weakest
        assert activations[alice.id] >= activations.get(bob.id, 0)

    def test_higher_arity_less_spread(self):
        """Larger hyperedges spread less activation per node."""
        # Small hyperedge (2 nodes)
        hg1 = HypergraphMemory()
        hg1.add_hyperedge(["Alice", "Bob"])
        r1 = hg1.spread_activation(["Alice"])
        bob1_act = dict(r1.activated_nodes).get(hg1.get_node("Bob").id, 0)

        # Large hyperedge (5 nodes)
        hg2 = HypergraphMemory()
        hg2.add_hyperedge(["Alice", "Bob", "C", "D", "E"])
        r2 = hg2.spread_activation(["Alice"])
        bob2_act = dict(r2.activated_nodes).get(hg2.get_node("Bob").id, 0)

        # Bob should receive less activation in larger group
        assert bob2_act < bob1_act

    def test_no_spread_from_unknown(self):
        hg = HypergraphMemory()
        result = hg.spread_activation(["Nobody"])
        assert len(result.activated_nodes) == 0


class TestHebbianLearning:
    """Test edge strengthening/weakening."""

    def test_strengthen(self):
        hg = HypergraphMemory()
        eid = hg.add_hyperedge(["A", "B"])
        w_before = hg._edges[eid].weight

        hg.strengthen_edge(eid)
        assert hg._edges[eid].weight > w_before

    def test_weaken(self):
        hg = HypergraphMemory()
        eid = hg.add_hyperedge(["A", "B"])
        hg.weaken_edge(eid)
        assert hg._edges[eid].weight < 1.0

    def test_weight_bounded(self):
        hg = HypergraphMemory()
        eid = hg.add_hyperedge(["A", "B"])
        for _ in range(100):
            hg.strengthen_edge(eid)
        assert hg._edges[eid].weight <= 5.0

    def test_weaken_bounded(self):
        hg = HypergraphMemory()
        eid = hg.add_hyperedge(["A", "B"])
        for _ in range(100):
            hg.weaken_edge(eid)
        assert hg._edges[eid].weight >= 0.01


class TestCliqueExpansion:
    """Test conversion to pairwise edges."""

    def test_binary_edge(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["A", "B"], relation_type="R")
        pairs = hg.to_pairwise()
        assert len(pairs) == 1
        assert pairs[0][0] in ("A", "B")
        assert pairs[0][2] in ("A", "B")

    def test_ternary_edge(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["A", "B", "C"], relation_type="R")
        pairs = hg.to_pairwise()
        # C(3,2) = 3 pairs
        assert len(pairs) == 3

    def test_quad_edge(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["A", "B", "C", "D"], relation_type="R")
        pairs = hg.to_pairwise()
        # C(4,2) = 6 pairs
        assert len(pairs) == 6


class TestTextIngestion:
    """Test extracting hyperedges from text."""

    def test_meeting_extraction(self):
        hg = HypergraphMemory()
        text = "Alice met Bob and Charlie at the Park on Tuesday."
        eids = hg.ingest_text(text)
        assert len(eids) >= 1
        # Should have at least Alice, Bob, Charlie as entities
        edge = hg._edges[eids[0]]
        assert edge.arity >= 3

    def test_multiple_sentences(self):
        hg = HypergraphMemory()
        text = (
            "Alice and Bob discussed the project. "
            "Charlie and Dave traveled to London."
        )
        eids = hg.ingest_text(text)
        assert len(eids) >= 2

    def test_relation_classification(self):
        hg = HypergraphMemory()
        text = "Alice met Bob at the Park."
        eids = hg.ingest_text(text)
        if eids:
            assert hg._edges[eids[0]].relation_type == "MEETING"

    def test_no_entities_skipped(self):
        hg = HypergraphMemory()
        text = "the weather is nice today."
        eids = hg.ingest_text(text)
        assert len(eids) == 0

    def test_source_memory_id(self):
        hg = HypergraphMemory()
        text = "Alice and Bob collaborated."
        eids = hg.ingest_text(text, source_memory_id="mem-456")
        if eids:
            assert hg._edges[eids[0]].source_memory_id == "mem-456"


class TestGetEdgesForNode:
    """Test getting all edges containing a node."""

    def test_edges_for_node(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["Alice", "Bob"])
        hg.add_hyperedge(["Alice", "Charlie"])
        hg.add_hyperedge(["Bob", "Charlie"])

        alice_edges = hg.get_edges_for_node("Alice")
        assert len(alice_edges) == 2

    def test_edges_for_unknown(self):
        hg = HypergraphMemory()
        assert len(hg.get_edges_for_node("Nobody")) == 0


class TestIncidenceMatrix:
    """Test incidence matrix computation."""

    def test_basic_incidence(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["A", "B", "C"])
        info = hg.get_incidence_info()
        assert info["shape"] == (3, 1)
        # All nodes should be in the one edge
        assert all(row[0] == 1 for row in info["matrix"])

    def test_multiple_edges(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["A", "B"])
        hg.add_hyperedge(["B", "C"])
        info = hg.get_incidence_info()
        assert info["shape"][0] == 3  # 3 nodes
        assert info["shape"][1] == 2  # 2 edges


class TestSerialization:
    """Test export/import."""

    def test_roundtrip(self):
        hg1 = HypergraphMemory()
        hg1.add_hyperedge(["Alice", "Bob", "Charlie"], relation_type="MEETING")
        hg1.add_hyperedge(["Bob", "Dave"], relation_type="DISCUSSION")

        state = hg1.export_state()

        hg2 = HypergraphMemory()
        hg2.import_state(state)

        assert len(hg2._nodes) == len(hg1._nodes)
        assert len(hg2._edges) == len(hg1._edges)
        assert hg2.get_node("Alice") is not None


class TestStatistics:
    """Test statistics reporting."""

    def test_empty_stats(self):
        hg = HypergraphMemory()
        stats = hg.get_statistics()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["avg_arity"] == 0.0

    def test_populated_stats(self):
        hg = HypergraphMemory()
        hg.add_hyperedge(["A", "B", "C"], relation_type="MEETING")
        hg.add_hyperedge(["D", "E"], relation_type="DISCUSSION")

        stats = hg.get_statistics()
        assert stats["node_count"] == 5
        assert stats["edge_count"] == 2
        assert stats["avg_arity"] == 2.5
        assert stats["max_arity"] == 3
        assert "MEETING" in stats["relation_types"]
        assert "DISCUSSION" in stats["relation_types"]
        # Pairwise equiv: C(3,2) + C(2,2) = 3 + 1 = 4
        assert stats["pairwise_equivalent_edges"] == 4


class TestEdgeCases:
    """Edge case tests."""

    def test_duplicate_hyperedges_allowed(self):
        """Same nodes can have multiple hyperedges (different events)."""
        hg = HypergraphMemory()
        e1 = hg.add_hyperedge(["A", "B"], relation_type="MEETING")
        e2 = hg.add_hyperedge(["A", "B"], relation_type="DISCUSSION")
        assert e1 != e2
        assert len(hg._edges) == 2

    def test_node_in_many_edges(self):
        hg = HypergraphMemory()
        for i in range(20):
            hg.add_hyperedge(["Hub", f"Spoke{i}"])
        edges = hg.get_edges_for_node("Hub")
        assert len(edges) == 20
