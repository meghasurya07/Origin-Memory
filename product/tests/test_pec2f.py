"""Tests for the PEC2F Knowledge Schema Engine."""

import pytest
from origin_brain.pec2f import (
    PEC2FEngine,
    PEC2FNodeType,
    KnowledgeNode,
    CritiqueResult,
    ReconstructionContext,
)


class TestIngestion:
    """Test text ingestion and node type classification."""

    def test_ingest_claim(self):
        engine = PEC2FEngine()
        nodes = engine.ingest(
            "Alice thinks Bob should take the job offer",
            source_person="Alice"
        )
        # Should create a CLAIM node and PERSON nodes
        claim_nodes = [n for n in nodes if n.node_type == PEC2FNodeType.CLAIM]
        assert len(claim_nodes) == 1
        assert claim_nodes[0].source_person == "Alice"
        assert claim_nodes[0].claim_status == "unverified"

    def test_ingest_event(self):
        engine = PEC2FEngine()
        nodes = engine.ingest("The meeting happened yesterday at the office")
        event_nodes = [n for n in nodes if n.node_type == PEC2FNodeType.EVENT]
        assert len(event_nodes) == 1

    def test_ingest_fact(self):
        engine = PEC2FEngine()
        nodes = engine.ingest("Python is a programming language")
        # Should be classified as FACT (is-a pattern)
        non_person = [n for n in nodes if n.node_type != PEC2FNodeType.PERSON]
        assert any(n.node_type == PEC2FNodeType.FACT for n in non_person)

    def test_ingest_concept_default(self):
        engine = PEC2FEngine()
        nodes = engine.ingest("memory architecture research")
        non_person = [n for n in nodes if n.node_type != PEC2FNodeType.PERSON]
        assert any(n.node_type == PEC2FNodeType.CONCEPT for n in non_person)

    def test_person_extraction(self):
        engine = PEC2FEngine()
        engine.ingest("Alice and Bob discussed the project")
        stats = engine.get_statistics()
        assert stats["node_types"].get("person", 0) >= 2

    def test_provenance_edge(self):
        engine = PEC2FEngine()
        engine.ingest(
            "Alice believes the project will succeed",
            source_person="Alice"
        )
        stats = engine.get_statistics()
        assert stats["relation_types"].get("CLAIMS", 0) >= 1


class TestClaimVsFact:
    """Test the critical claim vs fact distinction."""

    def test_claim_has_source(self):
        engine = PEC2FEngine()
        engine.ingest("Bob said that the deadline is Friday", source_person="Bob")
        claims = engine.get_nodes_by_type(PEC2FNodeType.CLAIM)
        assert len(claims) >= 1
        assert claims[0].source_person == "Bob"

    def test_fact_no_source_needed(self):
        engine = PEC2FEngine()
        engine.ingest("Water is composed of hydrogen and oxygen")
        facts = engine.get_nodes_by_type(PEC2FNodeType.FACT)
        assert len(facts) >= 1
        assert facts[0].source_person is None

    def test_promote_claim_to_fact(self):
        engine = PEC2FEngine()
        nodes = engine.ingest("Alice believes the sky is blue", source_person="Alice")
        claims = engine.get_nodes_by_type(PEC2FNodeType.CLAIM)
        assert len(claims) >= 1

        fact = engine.promote_claim_to_fact(claims[0].id)
        assert fact is not None
        assert fact.node_type == PEC2FNodeType.FACT
        assert fact.supporting_claims == 1
        # Original claim should be marked as supported
        assert claims[0].claim_status == "supported"

    def test_promote_nonexistent_claim(self):
        engine = PEC2FEngine()
        result = engine.promote_claim_to_fact("nonexistent")
        assert result is None


class TestContradictionDetection:
    """Test contradiction detection between claims/facts."""

    def test_detect_contradiction(self):
        engine = PEC2FEngine()
        engine.ingest("Bob accepted the job offer")
        engine.ingest("Bob rejected the job offer")
        stats = engine.get_statistics()
        assert stats["relation_types"].get("CONTRADICTS", 0) >= 1

    def test_supporting_evidence(self):
        engine = PEC2FEngine()
        engine.ingest("Alice joined the team last month")
        engine.ingest("Alice started working with the team recently")
        # Similar content should create SUPPORTS edges
        stats = engine.get_statistics()
        assert stats["total_edges"] >= 1


class TestPersonKnowledge:
    """Test person-centric knowledge retrieval."""

    def test_get_person_knowledge(self):
        engine = PEC2FEngine()
        engine.ingest("Alice started a new project yesterday")
        engine.ingest("Bob believes Alice is very talented", source_person="Bob")

        knowledge = engine.get_person_knowledge("Alice")
        # Should have events and/or claims involving Alice
        total = (len(knowledge["claims"]) + len(knowledge["events"]) +
                 len(knowledge["facts"]) + len(knowledge["concepts"]))
        assert total >= 1

    def test_get_claims_about(self):
        engine = PEC2FEngine()
        engine.ingest("Bob thinks Alice is brilliant", source_person="Bob")
        engine.ingest("Charlie believes Alice should lead", source_person="Charlie")

        claims = engine.get_claims_about("Alice")
        assert len(claims) >= 2
        sources = [src for _, src in claims]
        assert "Bob" in sources
        assert "Charlie" in sources


class TestCritiquePipeline:
    """Test the critique-then-reconstruct pipeline."""

    def test_clean_context(self):
        engine = PEC2FEngine()
        engine.ingest("Alice works at Google")
        engine.ingest("Bob likes Python")

        all_nodes = list(engine._nodes.values())
        non_person = [n for n in all_nodes if n.node_type != PEC2FNodeType.PERSON]
        critique = engine.critique(non_person)
        assert critique.recommendation == "proceed"
        assert critique.issues_found == 0

    def test_critique_finds_contradictions(self):
        engine = PEC2FEngine()
        engine.ingest("Bob accepted the new job")
        engine.ingest("Bob rejected the new job")

        all_nodes = list(engine._nodes.values())
        non_person = [n for n in all_nodes if n.node_type != PEC2FNodeType.PERSON]
        critique = engine.critique(non_person)
        # Should detect contradiction OR contested fact
        assert critique.issues_found >= 1

    def test_critique_unattributed_claim(self):
        engine = PEC2FEngine()
        # Ingest claim-like text without source
        engine.ingest("Someone reportedly said the deal fell through")

        claims = engine.get_nodes_by_type(PEC2FNodeType.CLAIM)
        if claims:
            critique = engine.critique(claims)
            assert len(critique.misattributed_claims) >= 1

    def test_reconstruct_context(self):
        engine = PEC2FEngine()
        engine.ingest("Alice works at Google as a senior engineer")
        engine.ingest("Bob said Alice is the best engineer", source_person="Bob")
        engine.ingest("Alice started a new project yesterday")

        ctx = engine.reconstruct_context("What does Alice do?")
        assert ctx.query == "What does Alice do?"
        assert ctx.confidence > 0
        assert len(ctx.narrative) > 0

    def test_reconstruct_with_contradictions(self):
        engine = PEC2FEngine()
        engine.ingest("Bob accepted the Meta offer")
        engine.ingest("Bob rejected the Meta offer")

        ctx = engine.reconstruct_context("Did Bob take the job?")
        # Confidence should be lowered due to contradictions
        assert ctx.confidence < 1.0 or ctx.critique.issues_found > 0


class TestStatistics:
    """Test statistics and introspection."""

    def test_empty_stats(self):
        engine = PEC2FEngine()
        stats = engine.get_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_stats_after_ingestion(self):
        engine = PEC2FEngine()
        engine.ingest("Alice works at Google")
        engine.ingest("Bob believes Alice should get promoted", source_person="Bob")

        stats = engine.get_statistics()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        assert "person" in stats["node_types"]
