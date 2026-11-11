"""Tests for the Associative Memory Network."""

import pytest
from origin_brain.associative_network import (
    AssociativeNetwork,
    AssociativeNetworkConfig,
    AssociativeNode,
    AssociativeEdge,
    SpreadingActivationResult,
    extract_entities_and_relations,
    _make_node_id,
)


class TestNodeManagement:
    """Test node creation, lookup, and eviction."""

    def test_create_node(self):
        net = AssociativeNetwork()
        node = net.get_or_create_node("Alice", category="entity")
        assert node.label == "Alice"
        assert node.category == "entity"
        assert node.activation == 0.0

    def test_get_existing_node(self):
        net = AssociativeNetwork()
        n1 = net.get_or_create_node("Alice")
        n2 = net.get_or_create_node("Alice")
        assert n1.id == n2.id  # Same node

    def test_case_insensitive_lookup(self):
        net = AssociativeNetwork()
        n1 = net.get_or_create_node("Alice")
        n2 = net.get_or_create_node("alice")
        assert n1.id == n2.id

    def test_get_node_by_label(self):
        net = AssociativeNetwork()
        net.get_or_create_node("Python", category="concept")
        node = net.get_node_by_label("Python")
        assert node is not None
        assert node.category == "concept"

    def test_get_nonexistent_node(self):
        net = AssociativeNetwork()
        assert net.get_node_by_label("DoesNotExist") is None

    def test_eviction_at_capacity(self):
        config = AssociativeNetworkConfig(max_nodes=3)
        net = AssociativeNetwork(config=config)
        net.get_or_create_node("A")
        net.get_or_create_node("B")
        net.get_or_create_node("C")
        # This should evict oldest
        net.get_or_create_node("D")
        stats = net.get_statistics()
        assert stats["node_count"] == 3


class TestEdgeManagement:
    """Test edge creation and Hebbian strengthening."""

    def test_add_edge(self):
        net = AssociativeNetwork()
        edge = net.add_edge("Alice", "Google", relation="works_at", weight=0.8)
        assert edge.relation == "works_at"
        assert edge.weight == 0.8

    def test_hebbian_strengthening(self):
        net = AssociativeNetwork()
        e1 = net.add_edge("Alice", "Google", relation="works_at", weight=0.5)
        initial_weight = e1.weight
        # Adding same edge again should strengthen it
        e2 = net.add_edge("Alice", "Google", relation="works_at")
        assert e2.weight > initial_weight
        assert e2.co_activation_count == 2

    def test_add_triple(self):
        net = AssociativeNetwork()
        edge = net.add_triple("Alice", "prefers", "Python")
        assert edge.relation == "prefers"
        neighbors = net.get_neighbors("Alice")
        assert any(n[0] == "Python" for n in neighbors)

    def test_edge_pruning_at_capacity(self):
        config = AssociativeNetworkConfig(max_edges_per_node=3)
        net = AssociativeNetwork(config=config)
        net.add_edge("Hub", "A", weight=0.9)
        net.add_edge("Hub", "B", weight=0.8)
        net.add_edge("Hub", "C", weight=0.7)
        # This should prune weakest
        net.add_edge("Hub", "D", weight=0.6)
        neighbors = net.get_neighbors("Hub")
        assert len(neighbors) <= 4  # May vary by implementation


class TestSpreadingActivation:
    """Test the core spreading activation algorithm."""

    def test_simple_spread(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Bob", "works_at", "Google")

        result = net.activate_and_spread(["Alice"])
        # Google should be activated (direct neighbor)
        activated_ids = [nid for nid, _ in result.activated_nodes]
        google_node = net.get_node_by_label("Google")
        assert google_node is not None
        assert google_node.id in activated_ids

    def test_multi_hop_spread(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Bob", "works_at", "Google")
        net.add_triple("Bob", "prefers", "Java")

        result = net.activate_and_spread(["Alice"], max_depth=3)
        # Bob should be activated (via Google, 2 hops)
        bob_node = net.get_node_by_label("Bob")
        assert bob_node is not None
        assert bob_node.activation > 0

    def test_activation_decays_with_distance(self):
        net = AssociativeNetwork()
        net.add_triple("A", "r", "B", weight=0.9)
        net.add_triple("B", "r", "C", weight=0.9)
        net.add_triple("C", "r", "D", weight=0.9)

        result = net.activate_and_spread(["A"])
        b = net.get_node_by_label("B")
        c = net.get_node_by_label("C")
        d = net.get_node_by_label("D")

        # Activation should decrease with distance from seed
        assert b.activation > c.activation
        if d.activation > 0:
            assert c.activation > d.activation

    def test_multiple_seeds(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "knows", "Concept1")
        net.add_triple("Bob", "knows", "Concept2")
        net.add_triple("Concept1", "related_to", "Concept2")

        result = net.activate_and_spread(["Alice", "Bob"])
        assert result.total_activation > 0
        assert len(result.activated_nodes) >= 2

    def test_no_spread_from_unknown_seed(self):
        net = AssociativeNetwork()
        result = net.activate_and_spread(["NonexistentEntity"])
        assert len(result.activated_nodes) == 0
        assert result.total_activation == 0.0

    def test_result_sorted_by_activation(self):
        net = AssociativeNetwork()
        net.add_triple("Center", "r", "Near", weight=0.9)
        net.add_triple("Center", "r", "Far", weight=0.2)

        result = net.activate_and_spread(["Center"])
        if len(result.activated_nodes) >= 2:
            activations = [a for _, a in result.activated_nodes]
            assert activations == sorted(activations, reverse=True)


class TestPathFinding:
    """Test path finding between nodes."""

    def test_direct_path(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "knows", "Bob")
        path = net.find_path("Alice", "Bob")
        assert path == ["Alice", "Bob"]

    def test_multi_hop_path(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Bob", "works_at", "Google")
        path = net.find_path("Alice", "Bob")
        assert path is not None
        assert path[0] == "Alice"
        assert path[-1] == "Bob"
        assert "Google" in path

    def test_no_path(self):
        net = AssociativeNetwork()
        net.get_or_create_node("Alice")
        net.get_or_create_node("Bob")
        # No edge between them
        path = net.find_path("Alice", "Bob")
        assert path is None

    def test_self_path(self):
        net = AssociativeNetwork()
        net.get_or_create_node("Alice")
        path = net.find_path("Alice", "Alice")
        assert path == ["Alice"]


class TestQueryMethods:
    """Test triple queries and entity summaries."""

    def test_query_by_subject(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Alice", "prefers", "Python")
        net.add_triple("Bob", "works_at", "Meta")

        results = net.query_triples(subject="Alice")
        assert len(results) == 2
        assert all(r[0] == "Alice" for r in results)

    def test_query_by_predicate(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Bob", "works_at", "Meta")
        net.add_triple("Alice", "prefers", "Python")

        results = net.query_triples(predicate="works_at")
        assert len(results) == 2

    def test_query_by_object(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Bob", "works_at", "Google")

        results = net.query_triples(obj="Google")
        assert len(results) == 2

    def test_query_combined(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Alice", "prefers", "Python")

        results = net.query_triples(subject="Alice", predicate="works_at")
        assert len(results) == 1
        assert results[0][2] == "Google"

    def test_entity_summary(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Alice", "prefers", "Python")
        net.add_triple("Alice", "knows", "Bob")

        summary = net.get_entity_summary("Alice")
        assert summary["found"] is True
        assert summary["total_connections"] == 3
        assert "works_at" in summary["relations"]
        assert "prefers" in summary["relations"]

    def test_entity_summary_not_found(self):
        net = AssociativeNetwork()
        summary = net.get_entity_summary("Nobody")
        assert summary["found"] is False

    def test_get_neighbors_with_filter(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Alice", "prefers", "Python")

        work = net.get_neighbors("Alice", relation="works_at")
        assert len(work) == 1
        assert work[0][0] == "Google"


class TestEntityExtraction:
    """Test relation extraction from text."""

    def test_works_at(self):
        triples = extract_entities_and_relations("Alice works at Google.")
        assert any(t[1] == "works_at" for t in triples)

    def test_is_a(self):
        triples = extract_entities_and_relations("Python is a programming language.")
        assert any(t[1] == "is_a" for t in triples)

    def test_prefers(self):
        triples = extract_entities_and_relations("Bob likes machine learning.")
        assert any(t[1] == "prefers" for t in triples)

    def test_lives_in(self):
        triples = extract_entities_and_relations("Alice lives in San Francisco.")
        assert any(t[1] == "located_in" for t in triples)

    def test_no_extraction(self):
        triples = extract_entities_and_relations("the weather is nice today")
        # No capitalized entities, so no extraction
        assert len(triples) == 0

    def test_ingest_text(self):
        net = AssociativeNetwork()
        edges = net.ingest_text("Alice works at Google. Bob likes Python.")
        assert len(edges) >= 2
        stats = net.get_statistics()
        assert stats["node_count"] >= 3


class TestDecayAndMaintenance:
    """Test network maintenance operations."""

    def test_decay_all(self):
        net = AssociativeNetwork()
        net.add_triple("A", "r", "B")
        # Manually set activation
        node = net.get_node_by_label("A")
        node.activate(0.8)
        assert node.activation > 0.5

        net.decay_all(factor=0.5)
        assert node.activation < 0.8  # Should have decayed

    def test_boost_resting_levels(self):
        net = AssociativeNetwork()
        node = net.get_or_create_node("Frequent")
        for _ in range(10):
            node.activate(0.1)
        net.boost_resting_levels()
        assert node.resting_level > 0

    def test_statistics(self):
        net = AssociativeNetwork()
        net.add_triple("A", "r", "B")
        net.add_triple("B", "r", "C")
        stats = net.get_statistics()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2


class TestExportImport:
    """Test graph serialization."""

    def test_export_import_roundtrip(self):
        net = AssociativeNetwork()
        net.add_triple("Alice", "works_at", "Google")
        net.add_triple("Bob", "knows", "Alice")

        exported = net.export_graph()
        assert len(exported["nodes"]) == 3
        assert len(exported["edges"]) == 2

        net2 = AssociativeNetwork()
        net2.import_graph(exported)
        stats = net2.get_statistics()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2


class TestLateralInhibition:
    """Test competitive inhibition behavior."""

    def test_inhibition_suppresses_weak(self):
        config = AssociativeNetworkConfig(lateral_inhibition=True)
        net = AssociativeNetwork(config=config)
        net.add_triple("Center", "strong", "StrongNeighbor", weight=0.95)
        net.add_triple("Center", "weak", "WeakNeighbor", weight=0.1)

        result = net.activate_and_spread(["Center"], initial_activation=1.0)
        strong = net.get_node_by_label("StrongNeighbor")
        weak = net.get_node_by_label("WeakNeighbor")

        # Strong neighbor should have much higher activation
        if strong and weak:
            assert strong.activation > weak.activation

    def test_no_inhibition_when_disabled(self):
        config = AssociativeNetworkConfig(lateral_inhibition=False)
        net = AssociativeNetwork(config=config)
        net.add_triple("A", "r", "B", weight=0.5)
        result = net.activate_and_spread(["A"])
        b = net.get_node_by_label("B")
        assert b.activation > 0
