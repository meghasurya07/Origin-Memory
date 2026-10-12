"""
Neural Embedding Engine — Semantic Similarity via Transformer Models

Uses HuggingFace transformers + torch directly (NOT sentence-transformers)
to avoid the pyarrow/datasets crash on Python 3.14.

Drop-in replacement for EmbeddingEngine with the same API.

Why neural embeddings over TF-IDF:
- TF-IDF: "car" vs "automobile" = 0.0 (no shared tokens)
- Neural: "car" vs "automobile" = 0.85+ (semantic understanding)

Brain analogue: Entorhinal cortex → deep semantic encoding
Default model: all-MiniLM-L6-v2 (384d, 80MB, fast)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NeuralEmbeddingConfig(BaseModel):
    """Configuration for neural embedding engine."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384  # MiniLM output dim
    device: str = "cpu"
    normalize: bool = True
    max_length: int = 256  # Max token length


class NeuralEmbeddingEngine:
    """
    Neural embedding engine using transformers + torch directly.
    
    Drop-in replacement for EmbeddingEngine with the same interface:
    - add_document(doc_id, text) → embedding
    - get_embedding(text) → embedding
    - similarity(query, doc_id) → float
    - search(query, top_k) → [(doc_id, score)]
    - cosine_similarity(vec1, vec2) → float
    """
    
    def __init__(self, config: Optional[NeuralEmbeddingConfig] = None):
        self.config = config or NeuralEmbeddingConfig()
        self._model = None
        self._tokenizer = None
        self._embeddings: Dict[str, List[float]] = {}
        self._total_docs: int = 0
        self._total_queries: int = 0
        
        logger.info(f"NeuralEmbeddingEngine initialized (model={self.config.model_name})")
    
    def _load_model(self):
        """Lazy-load the transformer model."""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModel.from_pretrained(self.config.model_name)
            self._model.eval()
            
            if self.config.device != "cpu":
                self._model = self._model.to(self.config.device)
            
            logger.info(f"Loaded model: {self.config.model_name}")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for neural embeddings. "
                "Install with: pip install transformers torch"
            )
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling — average token embeddings weighted by attention mask."""
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def _encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        import torch
        self._load_model()
        
        encoded = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        )
        
        if self.config.device != "cpu":
            encoded = {k: v.to(self.config.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            output = self._model(**encoded)
        
        embedding = self._mean_pooling(output, encoded['attention_mask'])
        
        if self.config.normalize:
            import torch.nn.functional as F
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding[0].cpu().tolist()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text (compatibility with TF-IDF engine interface)."""
        import re
        text = text.lower()
        return re.findall(r'[a-z0-9]+', text)
    
    def add_document(self, doc_id: str, text: str) -> List[float]:
        """Add a document and return its embedding."""
        embedding = self._encode(text)
        self._embeddings[doc_id] = embedding
        self._total_docs += 1
        return embedding
    
    def remove_document(self, doc_id: str):
        """Remove a document from the store."""
        if doc_id in self._embeddings:
            del self._embeddings[doc_id]
            self._total_docs = max(0, self._total_docs - 1)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text (does NOT add to corpus)."""
        return self._encode(text)
    
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
        """Compute similarity between query and stored document."""
        if doc_id not in self._embeddings:
            return 0.0
        self._total_queries += 1
        query_vec = self._encode(query)
        return self.cosine_similarity(query_vec, self._embeddings[doc_id])
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for most similar documents."""
        self._total_queries += 1
        query_vec = self._encode(query)
        scores = []
        for doc_id, doc_vec in self._embeddings.items():
            sim = self.cosine_similarity(query_vec, doc_vec)
            scores.append((doc_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "vocabulary_size": 0,
            "document_count": self._total_docs,
            "total_queries": self._total_queries,
            "embedding_dimension": self.config.dimension,
            "mode": "neural",
            "model": self.config.model_name,
        }


def create_embedding_engine(mode: str = "auto", **kwargs) -> Any:
    """
    Factory function to create the best available embedding engine.
    
    Args:
        mode: "neural", "tfidf", or "auto" (neural if available, else TF-IDF)
    """
    if mode == "neural":
        return NeuralEmbeddingEngine(NeuralEmbeddingConfig(**kwargs))
    
    if mode == "tfidf":
        from .embedding import EmbeddingEngine, EmbeddingConfig
        return EmbeddingEngine(EmbeddingConfig(**kwargs))
    
    # Auto: try neural, fallback to TF-IDF
    try:
        import torch
        import transformers
        logger.info("torch + transformers available → using neural embeddings")
        return NeuralEmbeddingEngine(NeuralEmbeddingConfig(**kwargs))
    except ImportError:
        logger.info("torch/transformers not available → using TF-IDF fallback")
        from .embedding import EmbeddingEngine, EmbeddingConfig
        return EmbeddingEngine(EmbeddingConfig(**kwargs))
