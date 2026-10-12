"""Tests for Neural Embedding Engine."""
import pytest

from origin_brain.neural_embedding import (
    NeuralEmbeddingEngine, NeuralEmbeddingConfig, create_embedding_engine,
)


# Check if transformers + torch are available
try:
    import torch
    import transformers
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

skip_no_neural = pytest.mark.skipif(not HAS_NEURAL, reason="torch/transformers not installed")


class TestNeuralEmbeddingEngine:
    """Tests using the real model (requires transformers + torch)."""

    @pytest.fixture(scope="class")
    def engine(self):
        """Shared engine instance (model load is expensive)."""
        if not HAS_NEURAL:
            pytest.skip("torch/transformers not installed")
        return NeuralEmbeddingEngine()

    @skip_no_neural
    def test_add_document(self, engine):
        vec = engine.add_document("doc1", "The Krebs cycle produces ATP")
        assert len(vec) == 384  # MiniLM dimension
        assert engine._total_docs >= 1

    @skip_no_neural
    def test_get_embedding(self, engine):
        vec = engine.get_embedding("Hello world")
        assert len(vec) == 384
        assert isinstance(vec[0], float)

    @skip_no_neural
    def test_semantic_similarity_high(self, engine):
        """Semantically similar sentences should have HIGH similarity."""
        engine.add_document("car_doc", "The car drove down the highway")
        sim = engine.similarity("An automobile traveled on the road", "car_doc")
        assert sim > 0.4  # Neural catches synonym relation

    @skip_no_neural
    def test_semantic_similarity_low(self, engine):
        """Unrelated sentences should have LOW similarity."""
        engine.add_document("car_doc2", "The car drove down the highway")
        sim = engine.similarity("Photosynthesis converts sunlight to energy", "car_doc2")
        assert sim < 0.3

    @skip_no_neural
    def test_neural_beats_tfidf_on_synonyms(self, engine):
        """Neural embeddings should handle synonyms that TF-IDF misses."""
        engine.add_document("happy_doc", "The happy child played in the garden")
        sim = engine.similarity("A joyful kid was playing outside", "happy_doc")
        assert sim > 0.3  # Neural catches synonym semantics

    @skip_no_neural
    def test_search(self, engine):
        engine.add_document("bio", "Mitochondria are the powerhouse of the cell")
        engine.add_document("lit", "Shakespeare wrote Romeo and Juliet")
        engine.add_document("bio2", "ATP is produced by cellular respiration")

        results = engine.search("cell biology energy production", top_k=3)
        assert len(results) == 3
        result_ids = [r[0] for r in results]
        # Bio docs should rank higher than literature
        assert result_ids[0] in ("bio", "bio2")

    @skip_no_neural
    def test_remove_document(self, engine):
        engine.add_document("temp_doc", "Test doc for removal")
        engine.remove_document("temp_doc")
        sim = engine.similarity("Test", "temp_doc")
        assert sim == 0.0

    @skip_no_neural
    def test_cosine_similarity(self, engine):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert engine.cosine_similarity(vec1, vec2) == pytest.approx(1.0)

        vec3 = [0.0, 1.0, 0.0]
        assert engine.cosine_similarity(vec1, vec3) == pytest.approx(0.0)

    @skip_no_neural
    def test_statistics(self, engine):
        stats = engine.get_statistics()
        assert stats["mode"] == "neural"
        assert stats["document_count"] >= 1
        assert stats["embedding_dimension"] == 384


class TestFactory:
    def test_create_tfidf_engine(self):
        engine = create_embedding_engine(mode="tfidf")
        stats = engine.get_statistics()
        assert stats["mode"] == "tfidf"

    def test_create_auto_engine(self):
        engine = create_embedding_engine(mode="auto")
        stats = engine.get_statistics()
        assert stats["mode"] in ("neural", "tfidf")

    @skip_no_neural
    def test_create_neural_engine(self):
        engine = create_embedding_engine(mode="neural")
        stats = engine.get_statistics()
        assert stats["mode"] == "neural"
