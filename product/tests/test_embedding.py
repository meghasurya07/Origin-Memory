"""
Tests for EmbeddingEngine — TF-IDF Semantic Similarity

Tests verify:
1. Tokenization with stop word removal
2. TF-IDF vector computation
3. Cosine similarity correctness
4. Semantic search ranking
5. Document add/remove
6. The critical test: Krebs vs Shakespeare discrimination
"""
import pytest

from origin_brain.embedding import EmbeddingEngine, EmbeddingConfig


class TestEmbeddingEngine:

    def test_creation(self):
        engine = EmbeddingEngine()
        assert engine._total_docs == 0
        assert engine._vocab_size == 0

    def test_tokenize_removes_stop_words(self):
        engine = EmbeddingEngine()
        tokens = engine.tokenize("The quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "over" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_tokenize_includes_bigrams(self):
        engine = EmbeddingEngine(EmbeddingConfig(use_bigrams=True))
        tokens = engine.tokenize("Krebs cycle produces ATP")
        assert "krebs_cycle" in tokens
        assert "cycle_produces" in tokens

    def test_add_document(self):
        engine = EmbeddingEngine()
        vec = engine.add_document("d1", "The Krebs cycle produces ATP")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert engine._total_docs == 1

    def test_similar_content_high_similarity(self):
        """Documents about the same topic should have HIGH similarity."""
        engine = EmbeddingEngine()
        engine.add_document("krebs", "The Krebs cycle produces ATP molecules in mitochondria")
        engine.add_document("other", "Shakespeare wrote Hamlet in sixteen hundred")
        
        sim = engine.similarity("Krebs cycle ATP mitochondria energy", "krebs")
        assert sim > 0.3, f"Similar content should have high sim, got {sim}"

    def test_different_content_low_similarity(self):
        """Documents about different topics should have LOW similarity."""
        engine = EmbeddingEngine()
        engine.add_document("krebs", "The Krebs cycle produces ATP molecules in mitochondria")
        engine.add_document("shakespeare", "Shakespeare wrote Hamlet in sixteen hundred")
        
        sim_krebs = engine.similarity("Krebs cycle ATP mitochondria", "krebs")
        sim_shakespeare = engine.similarity("Krebs cycle ATP mitochondria", "shakespeare")
        
        assert sim_krebs > sim_shakespeare, (
            f"Krebs query should match Krebs doc more than Shakespeare. "
            f"Krebs sim={sim_krebs:.3f}, Shakespeare sim={sim_shakespeare:.3f}"
        )

    def test_search_ranking(self):
        """Search should rank semantically similar docs first."""
        engine = EmbeddingEngine()
        engine.add_document("bio1", "The Krebs cycle produces ATP in mitochondria")
        engine.add_document("lit1", "Shakespeare wrote Hamlet and Romeo and Juliet")
        engine.add_document("bio2", "Mitochondria are the powerhouse of the cell")
        engine.add_document("lit2", "Charles Dickens wrote Oliver Twist and Great Expectations")
        
        results = engine.search("ATP energy mitochondria biology", top_k=4)
        
        # Biology docs should rank before literature docs
        top_2_ids = [r[0] for r in results[:2]]
        assert "bio1" in top_2_ids or "bio2" in top_2_ids, (
            f"Biology doc should be in top 2, got {top_2_ids}"
        )

    def test_remove_document(self):
        engine = EmbeddingEngine()
        engine.add_document("d1", "Test document one")
        engine.add_document("d2", "Test document two")
        assert engine._total_docs == 2
        
        engine.remove_document("d1")
        assert engine._total_docs == 1
        assert "d1" not in engine._doc_vectors

    def test_get_embedding(self):
        """get_embedding should NOT add to corpus."""
        engine = EmbeddingEngine()
        engine.add_document("d1", "Background document for IDF")
        
        vec = engine.get_embedding("Query text that is not stored")
        assert isinstance(vec, list)
        assert engine._total_docs == 1  # Not added

    def test_cosine_identical(self):
        """Same text should have similarity ~1.0."""
        engine = EmbeddingEngine()
        engine.add_document("d1", "Identical text for testing")
        sim = engine.similarity("Identical text for testing", "d1")
        assert sim > 0.9, f"Self-similarity should be ~1.0, got {sim}"

    def test_cosine_orthogonal(self):
        """Completely unrelated text should have low similarity."""
        engine = EmbeddingEngine()
        engine.add_document("d1", "quantum physics electron wave function")
        engine.add_document("d2", "chocolate cake recipe flour sugar eggs")
        
        sim = engine.similarity("quantum electron physics", "d2")
        assert sim < 0.15, f"Unrelated texts should have low sim, got {sim}"

    def test_statistics(self):
        engine = EmbeddingEngine()
        engine.add_document("d1", "First document")
        engine.add_document("d2", "Second document")
        engine.search("query")
        
        stats = engine.get_statistics()
        assert stats["document_count"] == 2
        assert stats["total_queries"] == 1
        assert stats["mode"] == "tfidf"

    def test_many_documents(self):
        """Should handle many documents efficiently."""
        engine = EmbeddingEngine()
        for i in range(100):
            engine.add_document(f"d{i}", f"Document {i} about topic {i % 10} with content {i*7}")
        
        assert engine._total_docs == 100
        results = engine.search("topic 5", top_k=5)
        assert len(results) == 5

    def test_the_critical_discrimination(self):
        """
        THE KEY TEST: Can TF-IDF distinguish semantically different content
        that Jaccard could not?
        
        Jaccard fails because both texts share common words (the, in, etc.)
        TF-IDF should succeed because rare topic words carry more weight.
        """
        engine = EmbeddingEngine()
        engine.add_document("krebs", 
            "The Krebs cycle produces ATP molecules through oxidative phosphorylation in mitochondria")
        engine.add_document("shakespeare",
            "Shakespeare wrote Hamlet in the year sixteen hundred and one at Stratford")
        
        query = "Krebs cycle ATP oxidative mitochondria"
        results = engine.search(query, top_k=2)
        
        # Krebs doc MUST be first
        assert results[0][0] == "krebs", (
            f"CRITICAL: Krebs query should match Krebs doc first! "
            f"Got: {results[0][0]} with sim={results[0][1]:.3f}, "
            f"vs {results[1][0]} with sim={results[1][1]:.3f}"
        )
        
        # And the gap should be significant
        gap = results[0][1] - results[1][1]
        assert gap > 0.1, f"Gap between Krebs and Shakespeare should be >0.1, got {gap:.3f}"
