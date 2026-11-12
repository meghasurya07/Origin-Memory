"""
Embedding Engine — Semantic Similarity via BM25 + TF-IDF Vectors

Implements two mathematically grounded retrieval models:

1. **BM25 (Okapi BM25)** — The gold-standard probabilistic retrieval model
   (Robertson & Zaragoza, 2009). Addresses TF-IDF limitations through:
   - Term frequency saturation: tf(t,d) → (k₁+1)·tf / (k₁·(1 - b + b·|d|/avgdl) + tf)
   - Document length normalisation: penalises long documents proportionally
   - Optimal IDF: log((N - df + 0.5) / (df + 0.5) + 1)  (Robertson-Spärck Jones)
   
   Mathematical foundation:
     BM25(Q,D) = Σ_{t∈Q} IDF(t) · [f(t,D)·(k₁+1)] / [f(t,D) + k₁·(1-b+b·|D|/avgdl)]
     where k₁=1.2 controls saturation, b=0.75 controls length normalisation
   
   Proven optimal under the Binary Independence Model (BIM) for the 2-Poisson
   term frequency distribution (Harter, 1975; Robertson & Walker, 1994).

2. **TF-IDF cosine** — Used for dense embedding vectors and compatibility
   with existing pattern separation / dense similarity pipelines.

Pure Python — no torch/transformers dependency. Designed to be swapped
for neural embeddings (BGE-M3, nomic-embed) when ML dependencies are added.

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
    # BM25 parameters (Robertson & Zaragoza, 2009)
    use_bm25: bool = True          # Use BM25 scoring instead of TF-IDF cosine
    bm25_k1: float = 1.2           # Term frequency saturation parameter
    bm25_b: float = 0.75           # Document length normalisation parameter


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
        
        # BM25 state: document lengths for length normalisation
        self._doc_lengths: Dict[str, int] = {}  # doc_id → token count
        self._total_token_count: int = 0          # sum of all doc lengths
        self._doc_token_counts: Dict[str, Dict[int, int]] = {}  # doc_id → {token_idx: raw_count}
        
        # Statistics
        self._total_queries: int = 0
        
        logger.info("EmbeddingEngine initialized (BM25+TF-IDF mode)")

    # ── Morphological normalisation ─────────────────────────────────────
    # A lightweight lemmatizer that reduces inflected forms to a common
    # base WITHOUT the aggressive truncation of Porter/Snowball stemmers.
    # 
    # Design: We use suffix rules with minimum stem length guards to
    # prevent over-stemming (e.g. "ordering" → "order" but NOT "ring" → "r").
    # Rules are applied greedily (longest match first, one pass only).
    #
    # This bridges the morphological gap that causes BM25 failures like:
    #   query "allergy" vs stored "allergic" → both normalise to "allerg"
    #   query "ordered" vs stored "order" → both normalise to "order"
    #
    # Brain analogue: Morphological decomposition in left inferior frontal
    # gyrus (LIFG) — Bozic et al., 2010; Marslen-Wilson et al., 2008.

    _NORMALISE_RULES: List[Tuple[str, str, int]] = [
        # (suffix, replacement, min_remaining_stem_length)
        # Longer suffixes first to prevent partial matches
        ("isation", "ise", 3),
        ("ization", "ize", 3),
        ("ically", "ic", 3),
        ("iness", "y", 3),
        ("ation", "", 4),
        ("ement", "", 4),
        ("ment", "", 4),
        ("ness", "", 3),
        ("ible", "", 3),
        ("able", "", 3),
        ("ical", "ic", 3),
        ("ally", "al", 3),
        ("ious", "", 3),
        ("eous", "", 3),
        ("ting", "te", 3),
        ("ing", "", 4),
        ("ies", "y", 3),
        ("ied", "y", 3),
        ("ous", "", 3),
        ("ive", "", 3),
        ("ful", "", 3),
        ("ler", "le", 3),
        ("ier", "y", 3),
        ("ers", "er", 3),
        ("est", "", 4),
        ("ity", "", 3),
        ("ant", "", 4),
        ("ent", "", 4),
        ("ory", "", 4),
        ("ary", "", 4),
        ("ic", "", 4),
        ("ed", "", 4),
        ("er", "", 4),
        ("ly", "", 4),
        ("al", "", 4),
        ("es", "", 4),
        ("ts", "", 3),
        ("ss", "", 3),  # keep "less" → "less", "boss" → "boss"
        ("s", "", 4),
        ("y", "", 5),   # allergy → allerg (matches allergic → allerg)
    ]

    def _normalise(self, word: str) -> str:
        """
        Apply lightweight morphological normalisation to a single token.
        
        Returns the normalised form, or the original if no rule applies.
        Minimum stem length guards prevent over-reduction.
        """
        for suffix, replacement, min_stem in self._NORMALISE_RULES:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if len(stem) >= min_stem:
                    return stem + replacement
        return word

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms (unigrams + optional bigrams).
        
        Pipeline: lowercase → split → filter stops → morphological normalise → bigrams.
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
        
        # Morphological normalisation
        words = [self._normalise(w) for w in words]
        
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
        
        # BM25: store raw token counts and document length
        raw_counts: Dict[int, int] = {}
        counts = Counter(tokens)
        for token, count in counts.items():
            idx = self._get_or_create_index(token)
            raw_counts[idx] = count
        self._doc_token_counts[doc_id] = raw_counts
        self._doc_lengths[doc_id] = len(tokens)
        self._total_token_count += len(tokens)
        
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
        
        # Clean up BM25 state
        if doc_id in self._doc_lengths:
            self._total_token_count -= self._doc_lengths[doc_id]
            del self._doc_lengths[doc_id]
        self._doc_token_counts.pop(doc_id, None)
        
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

    def _bm25_idf(self, token_idx: int) -> float:
        """
        Robertson-Spärck Jones IDF for BM25.
        
        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        
        This is the probabilistic IDF derived from the Binary Independence
        Model (Robertson & Spärck Jones, 1976). The +1 inside the log
        prevents negative IDF for terms appearing in >50% of documents.
        """
        df = self._doc_freq.get(token_idx, 0)
        if df == 0:
            return 0.0
        n = self._total_docs
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_score(self, query: str, doc_id: str) -> float:
        """
        Compute BM25 score between a query and a stored document.
        
        BM25(Q, D) = Σ_{t ∈ Q} IDF(t) · [f(t,D) · (k₁ + 1)] / 
                                          [f(t,D) + k₁ · (1 - b + b · |D|/avgdl)]
        
        Where:
            f(t,D)  = raw term frequency of term t in document D
            |D|     = document length (token count)
            avgdl   = average document length across corpus
            k₁      = 1.2 (saturation parameter; controls tf diminishing returns)
            b       = 0.75 (length normalisation; 0 = no normalisation, 1 = full)
        
        Mathematical properties:
            - As tf → ∞, score → IDF(t) · (k₁ + 1)  (bounded, unlike raw TF-IDF)
            - Short documents with concentrated term usage score higher
            - Rare terms (low df) contribute much more than common terms
        
        Returns:
            BM25 relevance score (unnormalised, typically 0–10+ range).
            We normalise to [0, 1] using sigmoid for compatibility with
            the retrieval scoring pipeline.
        """
        if doc_id not in self._doc_token_counts:
            return 0.0
        
        k1 = self.config.bm25_k1
        b = self.config.bm25_b
        
        doc_len = self._doc_lengths.get(doc_id, 1)
        avgdl = (self._total_token_count / self._total_docs) if self._total_docs > 0 else 1.0
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        query_token_indices = {}
        for token in query_tokens:
            if token in self._vocab:
                idx = self._vocab[token]
                query_token_indices[idx] = query_token_indices.get(idx, 0) + 1
        
        doc_counts = self._doc_token_counts[doc_id]
        
        raw_score = 0.0
        for q_idx, q_tf in query_token_indices.items():
            idf = self._bm25_idf(q_idx)
            f_td = doc_counts.get(q_idx, 0)
            if f_td == 0:
                continue
            
            # BM25 term score
            numerator = f_td * (k1 + 1.0)
            denominator = f_td + k1 * (1.0 - b + b * (doc_len / avgdl))
            raw_score += idf * (numerator / denominator)
        
        # Normalise to [0, 1] via sigmoid: σ(score - μ) where μ centres typical scores
        # For short documents (10-15 tokens), typical BM25 scores are 0-5
        # We use a shifted sigmoid: 2·σ(score) - 1, clamped to [0,1]
        # This maps score=0 → 0.0, score≈2.5 → 0.5, score≈5 → ~0.95
        if raw_score <= 0:
            return 0.0
        normalised = 2.0 / (1.0 + math.exp(-0.8 * raw_score)) - 1.0
        return max(0.0, min(1.0, normalised))

    def similarity(self, query: str, doc_id: str) -> float:
        """
        Compute similarity between a query and a stored document.
        
        Uses BM25 when enabled (default), falls back to TF-IDF cosine.
        """
        if doc_id not in self._doc_vectors:
            return 0.0
        
        self._total_queries += 1
        
        if self.config.use_bm25:
            return self._bm25_score(query, doc_id)
        
        query_vec = self.get_embedding(query)
        doc_vec = self._to_dense_tfidf(self._doc_vectors[doc_id])
        return self.cosine_similarity(query_vec, doc_vec)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for most similar documents to query.
        
        Uses BM25 when enabled (default), falls back to TF-IDF cosine.
        
        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score desc
        """
        self._total_queries += 1
        
        scores = []
        if self.config.use_bm25:
            for doc_id in self._doc_vectors:
                sim = self._bm25_score(query, doc_id)
                scores.append((doc_id, sim))
        else:
            query_vec = self.get_embedding(query)
            for doc_id, tf in self._doc_vectors.items():
                doc_vec = self._to_dense_tfidf(tf)
                sim = self.cosine_similarity(query_vec, doc_vec)
                scores.append((doc_id, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        avgdl = (self._total_token_count / self._total_docs) if self._total_docs > 0 else 0.0
        return {
            "vocabulary_size": self._vocab_size,
            "document_count": self._total_docs,
            "total_queries": self._total_queries,
            "embedding_dimension": self.config.dimension,
            "mode": "bm25" if self.config.use_bm25 else "tfidf",
            "avg_doc_length": round(avgdl, 1),
            "bm25_k1": self.config.bm25_k1,
            "bm25_b": self.config.bm25_b,
        }
