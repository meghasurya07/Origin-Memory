"""
Embedding Engine — Semantic Similarity via TF-IDF Vectors

Replaces crude Jaccard similarity with proper TF-IDF cosine similarity.
This is a pure-Python implementation (no torch/transformers dependency)
that can be swapped for neural embeddings (BGE-M3, nomic-embed) later.

Why TF-IDF over Jaccard:
- Jaccard: "car" vs "automobile" = 0.0 (no shared tokens)
- TF-IDF: weights RARE words higher than common ones
- Cosine similarity on TF-IDF vectors gives proper semantic matching
- "The Krebs cycle produces ATP" vs "Krebs ATP mitochondria" = HIGH
- "The Krebs cycle produces ATP" vs "Shakespeare wrote Hamlet" = LOW

Brain analogue: Entorhinal cortex pattern separation before hippocampal storage
Neuroscience: Sparse distributed representations (Kanerva SDM)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Common English stop words to filter out
STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'just', 'don', 'now', 'and', 'but', 'or', 'if', 'while', 'that',
    'this', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
    'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what',
    'which', 'who', 'whom', 'about', 'up', 'also',
})


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding engine."""
    dimension: int = 128           # TF-IDF vector dimension (vocab limit)
    use_bigrams: bool = True       # Include word bigrams for phrase matching
    min_word_length: int = 2       # Minimum word length to consider
    sublinear_tf: bool = True      # Use log(1 + tf) instead of raw tf


class EmbeddingEngine:
    """
    TF-IDF based embedding engine for semantic similarity.
    
    Maintains a vocabulary learned from encoded content and computes
    TF-IDF vectors for cosine similarity search.
    
    This is a stepping stone — designed to be replaced by neural embeddings
    (BGE-M3, nomic-embed) when we add ML dependencies.
    
    Usage:
        engine = EmbeddingEngine()
        
        # Build vocabulary from corpus
        engine.add_document("doc1", "The Krebs cycle produces ATP")
        engine.add_document("doc2", "Shakespeare wrote Hamlet")
        
        # Get similarity
        sim = engine.similarity("Krebs ATP", "doc1")  # HIGH
        sim = engine.similarity("Krebs ATP", "doc2")  # LOW
        
        # Search
        results = engine.search("mitochondria energy", top_k=5)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        
        # Vocabulary: token → index
        self._vocab: Dict[str, int] = {}
        self._vocab_size: int = 0
        
        # Document store: doc_id → (tokens, tf_vector)
        self._documents: Dict[str, List[str]] = {}
        self._doc_vectors: Dict[str, Dict[int, float]] = {}  # sparse TF vectors
        
        # IDF: token_index → idf_score
        self._doc_freq: Counter = Counter()  # How many docs contain each token
        self._total_docs: int = 0
        self._idf_cache: Dict[int, float] = {}
        self._idf_dirty: bool = True
        
        # Statistics
        self._total_queries: int = 0
        
        logger.info("EmbeddingEngine initialized (TF-IDF mode)")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms (unigrams + optional bigrams).
        
        Includes basic normalization: lowercase, remove punctuation,
        filter stop words, minimum length filtering.
        """
        # Normalize
        text = text.lower()
        # Split on non-alphanumeric
        words = re.findall(r'[a-z0-9]+', text)
        # Filter
        words = [
            w for w in words
            if len(w) >= self.config.min_word_length and w not in STOP_WORDS
        ]
        
        tokens = list(words)
        
        # Add bigrams for phrase matching
        if self.config.use_bigrams and len(words) >= 2:
            for i in range(len(words) - 1):
                tokens.append(f"{words[i]}_{words[i+1]}")
        
        return tokens

    def _get_or_create_index(self, token: str) -> int:
        """Get vocab index for token, creating if new."""
        if token not in self._vocab:
            idx = self._vocab_size
            self._vocab[token] = idx
            self._vocab_size += 1
            return idx
        return self._vocab[token]

    def _compute_tf(self, tokens: List[str]) -> Dict[int, float]:
        """Compute term frequency vector (sparse)."""
        counts = Counter(tokens)
        tf = {}
        for token, count in counts.items():
            idx = self._get_or_create_index(token)
            if self.config.sublinear_tf:
                tf[idx] = 1.0 + math.log(count)
            else:
                tf[idx] = count
        return tf

    def _recompute_idf(self):
        """Recompute IDF scores for all terms."""
        if not self._idf_dirty or self._total_docs == 0:
            return
        
        self._idf_cache.clear()
        for idx in range(self._vocab_size):
            df = self._doc_freq.get(idx, 0)
            if df > 0:
                self._idf_cache[idx] = math.log(self._total_docs / df) + 1.0
            else:
                self._idf_cache[idx] = 1.0
        
        self._idf_dirty = False

    def add_document(self, doc_id: str, text: str) -> List[float]:
        """
        Add a document to the corpus and return its embedding vector.
        
        Args:
            doc_id: Unique identifier for this document
            text: The text content
            
        Returns:
            Dense embedding vector (list of floats)
        """
        tokens = self.tokenize(text)
        tf = self._compute_tf(tokens)
        
        # Update document frequency
        seen_indices = set(tf.keys())
        for idx in seen_indices:
            self._doc_freq[idx] += 1
        
        self._documents[doc_id] = tokens
        self._doc_vectors[doc_id] = tf
        self._total_docs += 1
        self._idf_dirty = True
        
        # Return dense vector
        return self._to_dense_tfidf(tf)

    def remove_document(self, doc_id: str):
        """Remove a document from the corpus."""
        if doc_id not in self._doc_vectors:
            return
        
        tf = self._doc_vectors[doc_id]
        for idx in tf:
            self._doc_freq[idx] = max(0, self._doc_freq.get(idx, 1) - 1)
        
        del self._documents[doc_id]
        del self._doc_vectors[doc_id]
        self._total_docs = max(0, self._total_docs - 1)
        self._idf_dirty = True

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for arbitrary text (query or new content).
        
        Does NOT add the text to the corpus.
        """
        tokens = self.tokenize(text)
        tf = self._compute_tf(tokens)
        return self._to_dense_tfidf(tf)

    def _to_dense_tfidf(self, tf: Dict[int, float]) -> List[float]:
        """Convert sparse TF to dense TF-IDF vector."""
        self._recompute_idf()
        
        dim = min(self._vocab_size, self.config.dimension)
        vec = [0.0] * max(dim, 1)
        
        for idx, tf_val in tf.items():
            if idx < dim:
                idf = self._idf_cache.get(idx, 1.0)
                vec[idx] = tf_val * idf
        
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        
        return vec

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        min_len = min(len(vec1), len(vec2))
        dot = sum(vec1[i] * vec2[i] for i in range(min_len))
        
        norm1 = math.sqrt(sum(v * v for v in vec1[:min_len]))
        norm2 = math.sqrt(sum(v * v for v in vec2[:min_len]))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)

    def similarity(self, query: str, doc_id: str) -> float:
        """Compute similarity between a query and a stored document."""
        if doc_id not in self._doc_vectors:
            return 0.0
        
        self._total_queries += 1
        query_vec = self.get_embedding(query)
        doc_vec = self._to_dense_tfidf(self._doc_vectors[doc_id])
        return self.cosine_similarity(query_vec, doc_vec)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for most similar documents to query.
        
        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score desc
        """
        self._total_queries += 1
        query_vec = self.get_embedding(query)
        
        scores = []
        for doc_id, tf in self._doc_vectors.items():
            doc_vec = self._to_dense_tfidf(tf)
            sim = self.cosine_similarity(query_vec, doc_vec)
            scores.append((doc_id, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "vocabulary_size": self._vocab_size,
            "document_count": self._total_docs,
            "total_queries": self._total_queries,
            "embedding_dimension": self.config.dimension,
            "mode": "tfidf",  # Will change to "neural" when upgraded
        }
