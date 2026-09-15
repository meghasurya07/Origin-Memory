"""
Tests for the Reconstruction Engine — the breakthrough module.

Tests verify that:
1. High context overlap → verbatim fidelity
2. Low context overlap → schema-filled reconstruction
3. Emotional modulation changes output
4. Schema filling adds plausible details
5. Confidence correlates with context overlap
6. Different reconstructions from same memory (context-dependent)
"""
from datetime import datetime, timezone

import pytest

from origin_brain.reconstruction import (
    ReconstructionEngine, ReconstructionConfig, ReconstructionFidelity,
    ReconstructedMemory, SchemaKnowledge
)
from origin_brain.models import EpisodicMemory


def _make_memory(content, salience=0.5, access_count=3) -> EpisodicMemory:
    return EpisodicMemory(
        content=content,
        salience=salience,
        access_count=access_count,
        context={"location": "office", "activity": "working"},
        metadata={"emotional_arousal": 0.5}
    )


class TestReconstructionEngine:

    def test_creation(self):
        engine = ReconstructionEngine()
        stats = engine.get_statistics()
        assert stats["total_reconstructions"] == 0
        assert stats["schema_count"] > 0

    def test_verbatim_high_context(self):
        """High context overlap should produce verbatim reconstruction."""
        engine = ReconstructionEngine()
        memory = _make_memory("The database server failed at 3:00 PM causing a major outage")
        result = engine.reconstruct(memory, context_overlap=0.9)
        
        assert result.fidelity == ReconstructionFidelity.VERBATIM
        assert result.reconstructed_content == memory.content
        assert result.trace_contribution > 0.8
        assert result.confidence > 0.5

    def test_gist_medium_context(self):
        """Medium context overlap should produce gist reconstruction."""
        engine = ReconstructionEngine()
        memory = _make_memory(
            "The database server failed at 3:00 PM. "
            "It caused a major outage affecting all users. "
            "The team investigated and found the root cause. "
            "A fix was deployed by 5:00 PM."
        )
        result = engine.reconstruct(memory, context_overlap=0.5)
        
        assert result.fidelity == ReconstructionFidelity.GIST
        assert result.trace_contribution < 0.9
        assert result.schema_contribution > 0.0

    def test_schema_filled_low_context(self):
        """Low context overlap should trigger schema filling."""
        engine = ReconstructionEngine()
        memory = _make_memory(
            "The error caused a crash in the production system"
        )
        result = engine.reconstruct(
            memory, 
            context_overlap=0.25,
            current_context={"activity": "debugging", "error": True}
        )
        
        assert result.fidelity == ReconstructionFidelity.SCHEMA_FILLED
        assert result.schema_contribution > 0.3

    def test_confabulated_very_low_context(self):
        """Very low context overlap should produce confabulated reconstruction."""
        engine = ReconstructionEngine()
        memory = _make_memory("Something happened at the meeting about the project")
        result = engine.reconstruct(memory, context_overlap=0.1)
        
        assert result.fidelity == ReconstructionFidelity.CONFABULATED
        assert result.trace_contribution < 0.3
        assert result.schema_contribution > 0.4

    def test_emotional_modulation(self):
        """High arousal should modulate reconstruction."""
        engine = ReconstructionEngine()
        memory = _make_memory(
            "The server crashed and we lost data. "
            "Everyone was calm and continued working. "
            "The backup was running fine."
        )
        
        # Low arousal: balanced
        calm_result = engine.reconstruct(memory, context_overlap=0.5, emotional_arousal=0.3)
        # High arousal: emphasized
        stressed_result = engine.reconstruct(memory, context_overlap=0.5, emotional_arousal=0.9)
        
        assert stressed_result.emotional_modulation > calm_result.emotional_modulation

    def test_same_memory_different_contexts(self):
        """Same memory reconstructed in different contexts should differ."""
        engine = ReconstructionEngine()
        memory = _make_memory("The project meeting discussed budget and timeline issues")
        
        # Reconstruction 1: high context overlap (recent, similar context)
        r1 = engine.reconstruct(memory, context_overlap=0.9)
        # Reconstruction 2: low context overlap (different context entirely)
        r2 = engine.reconstruct(memory, context_overlap=0.2)
        
        # They should have different fidelity levels
        assert r1.fidelity != r2.fidelity
        # And potentially different content
        assert r1.trace_contribution != r2.trace_contribution

    def test_confidence_correlates_with_overlap(self):
        """Higher context overlap should produce higher confidence."""
        engine = ReconstructionEngine()
        memory = _make_memory("Important fact about the system architecture", salience=0.8, access_count=5)
        
        high_conf = engine.reconstruct(memory, context_overlap=0.9)
        low_conf = engine.reconstruct(memory, context_overlap=0.1)
        
        assert high_conf.confidence > low_conf.confidence

    def test_schema_knowledge_addition(self):
        """Adding semantic knowledge should influence reconstruction."""
        engine = ReconstructionEngine()
        memory = _make_memory("The system was updated with new features")
        
        result = engine.reconstruct(
            memory,
            context_overlap=0.4,
            semantic_knowledge=["The system uses Python 3.11", "Deployed on AWS"]
        )
        
        # Semantic knowledge should be incorporated
        assert "Python" in result.reconstructed_content or "AWS" in result.reconstructed_content

    def test_custom_schema(self):
        """Custom schemas should be usable in reconstruction."""
        engine = ReconstructionEngine()
        engine.add_schema(SchemaKnowledge(
            name="deployment",
            category="technical",
            typical_elements=["version", "environment", "rollback plan", "monitoring"],
            fill_templates=["Monitoring was set up to track the deployment", "A rollback plan was prepared"]
        ))
        
        memory = _make_memory("We deployed version 2.0 to production environment")
        result = engine.reconstruct(
            memory,
            context_overlap=0.3,
            current_context={"activity": "deployment"}
        )
        
        assert result.fidelity in (ReconstructionFidelity.SCHEMA_FILLED, ReconstructionFidelity.GIST)

    def test_statistics_tracking(self):
        """Should track reconstruction statistics."""
        engine = ReconstructionEngine()
        memory = _make_memory("Test content")
        
        engine.reconstruct(memory, context_overlap=0.9)  # VERBATIM
        engine.reconstruct(memory, context_overlap=0.5)  # GIST
        engine.reconstruct(memory, context_overlap=0.1)  # CONFABULATED
        
        stats = engine.get_statistics()
        assert stats["total_reconstructions"] == 3
        assert sum(stats["fidelity_histogram"].values()) == 3

    def test_reconstruction_result_structure(self):
        """Reconstructed memory should have all required fields."""
        engine = ReconstructionEngine()
        memory = _make_memory("The experiment yielded surprising results")
        result = engine.reconstruct(memory, context_overlap=0.6)
        
        assert isinstance(result, ReconstructedMemory)
        assert result.original_memory_id == memory.id
        assert result.original_content == memory.content
        assert len(result.reconstructed_content) > 0
        assert 0.0 <= result.confidence <= 1.0
        assert abs(result.trace_contribution + result.schema_contribution + result.context_contribution - 1.0) < 0.01

    def test_empty_content_handling(self):
        """Should handle empty or minimal content gracefully."""
        engine = ReconstructionEngine()
        memory = _make_memory("")
        result = engine.reconstruct(memory, context_overlap=0.5)
        assert isinstance(result, ReconstructedMemory)
