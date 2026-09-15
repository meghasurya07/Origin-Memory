import math
import hashlib
import random
import time
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import BrainConfig, EpisodicMemory, MemoryType, MemoryResult
from .decay import DecayEngine
from .embedding import EmbeddingEngine

logger = logging.getLogger(__name__)

class HippocampalEngine:
    """
    Core encoding and retrieval engine, inspired by the hippocampus brain structure.
    Handles episodic memory storage, pattern separation, and temporal-semantic retrieval.
    """

    def __init__(
        self,
        config: BrainConfig,
        decay_engine: DecayEngine,
        embedding_fn: Optional[Callable[[str], List[float]]] = None
    ):
        self.config = config
        self.decay_engine = decay_engine
        self.embedding_fn = embedding_fn
        
        # Internal state
        self._episodic_store: Dict[str, EpisodicMemory] = {}
        self._embeddings: Dict[str, List[float]] = {}       # Sparse (pattern-separated)
        self._dense_embeddings: Dict[str, List[float]] = {} # Dense (for novelty detection)
        # sorted list of (timestamp, memory_id)
        self._temporal_index: List[Tuple[datetime, str]] = []
        
        # TF-IDF semantic search engine (upgrades Jaccard-like hash similarity)
        self._semantic_engine = EmbeddingEngine()

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Computes standard cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)

    def _compute_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the text."""
        if self.embedding_fn is not None:
            return self.embedding_fn(text)
            
        # Deterministic pseudo-embedding
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        seed = int(h, 16)
        rng = random.Random(seed)
        
        # Generate embedding_dimension floats
        dim = self.config.embedding_dimension if hasattr(self.config, 'embedding_dimension') else 1536
        return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

    def _pattern_separate(self, embedding: List[float], sparsity: float = 0.05) -> List[float]:
        """
        DG-inspired k-winners-take-all: keep only the top k% of values, zero out the rest.
        """
        if not embedding:
            return []
            
        k = max(1, int(len(embedding) * sparsity))
        
        # Find the threshold value for top k
        sorted_vals = sorted(embedding, reverse=True)
        threshold = sorted_vals[k-1] if k <= len(sorted_vals) else sorted_vals[-1]
        
        # Zero out values below threshold
        return [x if x >= threshold else 0.0 for x in embedding]

    def _score_salience(self, content: str) -> float:
        """Heuristic salience scoring based on content features."""
        score = 0.3 # Base
        
        if '?' in content:
            score += 0.1
        if '!' in content:
            score += 0.1
            
        content_lower = content.lower()
        keywords = ['important', 'critical', 'remember', 'always', 'never', 'urgent']
        if any(kw in content_lower for kw in keywords):
            score += 0.15
            
        # Count sentences roughly by splitting on punctuation
        sentences = content.count('.') + content.count('?') + content.count('!')
        # If no punctuation, it's at least 1 sentence if there's text
        if sentences == 0 and content.strip():
            sentences = 1
            
        sentence_bonus = min(0.2, sentences * 0.05)
        score += sentence_bonus
        
        return max(0.0, min(1.0, score))

    def _recency_score(self, timestamp: datetime) -> float:
        """Returns a 0-1 score where 1.0 = now, 0.0 = very old."""
        now = datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        delta = now - timestamp
        hours_ago = delta.total_seconds() / 3600.0
        
        if hours_ago < 0:
            return 1.0
            
        # Exponential decay: e^(-hours_ago / 168) (168 hours = 1 week half-life)
        return math.exp(-hours_ago / 168.0)

    def encode(self, content: str, context: Optional[Dict[str, Any]] = None, salience: Optional[float] = None, memory_type: MemoryType = MemoryType.EPISODIC) -> Optional[EpisodicMemory]:
        """Encodes new content into episodic memory."""
        # 1. Compute dense embedding
        dense_embedding = self._compute_embedding(content)
        
        # 2. Novelty detection — compare dense-to-dense for accuracy
        max_similarity = 0.0
        recent_memories = [m for t, m in self._temporal_index[-100:]]
        for mem_id in recent_memories:
            existing_dense = self._dense_embeddings.get(mem_id)
            if existing_dense:
                sim = self._cosine_similarity(dense_embedding, existing_dense)
                if sim > max_similarity:
                    max_similarity = sim
                    
        novelty_threshold = self.config.novelty_threshold
        if max_similarity > (1.0 - novelty_threshold):
            logger.debug(f"Content not novel enough (sim={max_similarity:.2f})")
            return None
            
        # 3. Salience scoring
        if salience is None:
            salience = self._score_salience(content)
            
        # 4. Pattern separation (sparse representation for storage)
        sparse_embedding = self._pattern_separate(dense_embedding)
        
        # 5. Create memory object
        import uuid
        mem_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        memory = EpisodicMemory(
            id=mem_id,
            content=content,
            timestamp=now,
            memory_type=memory_type,
            context=context or {},
            salience=salience
        )
        
        # 6. Store
        self._episodic_store[mem_id] = memory
        self._embeddings[mem_id] = sparse_embedding
        self._dense_embeddings[mem_id] = dense_embedding
        self._temporal_index.append((now, mem_id))
        self._temporal_index.sort(key=lambda x: x[0])
        
        # 7. Index in semantic engine (TF-IDF)
        self._semantic_engine.add_document(mem_id, content)
        
        logger.info(f"Encoded new episodic memory: {mem_id} (salience={salience:.2f})")
        return memory

    def retrieve(self, query: str, top_k: int = 5, temporal_range: Optional[Tuple[datetime, datetime]] = None, min_salience: float = 0.0) -> List[MemoryResult]:
        """Retrieves episodic memories based on a query."""
        start_time = time.perf_counter()
        
        query_dense = self._compute_embedding(query)
        # Apply pattern separation to query for comparison against sparse store?
        # Typically we compare query (maybe sparse) to sparse embeddings. We'll use pattern separation here too.
        query_embedding = self._pattern_separate(query_dense)
        
        results: List[Tuple[EpisodicMemory, float]] = []
        
        for mem_id, memory in self._episodic_store.items():
            # Apply filters
            if memory.salience < min_salience:
                continue
                
            if temporal_range:
                start, end = temporal_range
                if start.tzinfo is None: start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None: end = end.replace(tzinfo=timezone.utc)
                mem_time = memory.timestamp
                if mem_time.tzinfo is None: mem_time = mem_time.replace(tzinfo=timezone.utc)
                
                if not (start <= mem_time <= end):
                    continue
            
            # Compute scores — use TF-IDF semantic similarity (much better than hash-based)
            if self._semantic_engine._total_docs > 0:
                semantic_similarity = self._semantic_engine.similarity(query, mem_id)
            else:
                # Fallback to hash-based cosine if semantic engine empty
                mem_embedding = self._embeddings.get(mem_id, [])
                semantic_similarity = self._cosine_similarity(query_embedding, mem_embedding)
            
            # Recency: blend wall-clock recency with encoding-order recency
            # Wall-clock handles long time spans, encoding-order handles same-session items
            wall_recency = self._recency_score(memory.timestamp)
            
            # Encoding-order recency: position in temporal index (later = higher)
            order_recency = 0.5
            total_items = max(len(self._temporal_index), 1)
            for idx, (ts, tid) in enumerate(self._temporal_index):
                if tid == mem_id:
                    order_recency = (idx + 1) / total_items
                    break
            
            # Blend: 40% wall-clock + 60% encoding order (order matters more for same-session)
            recency = 0.4 * wall_recency + 0.6 * order_recency
            salience_score = memory.salience
            
            # Primacy bonus: first few items get rehearsal advantage
            # (simulates working memory rehearsal of early list items)
            primacy_bonus = 0.0
            for idx, (ts, tid) in enumerate(self._temporal_index):
                if tid == mem_id:
                    if idx < 3:  # First 3 items get primacy boost
                        primacy_bonus = 0.3 * (1.0 - idx / 3.0)
                    break
            
            # Weighted score: 0.45 semantic + 0.30 recency + 0.15 salience + 0.10 primacy
            final_score = (
                0.45 * semantic_similarity 
                + 0.30 * recency 
                + 0.15 * salience_score 
                + 0.10 * primacy_bonus
            )
            
            results.append((memory, final_score))
            
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        # Reinforce retrieved memories and build MemoryResult
        memory_results = []
        for mem, score in top_results:
            mem.on_retrieval()
            
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            from .models import MemoryTier
            res = MemoryResult(
                memory=mem,
                relevance_score=score,
                tier=MemoryTier.EPISODIC,
                retrieval_latency_ms=latency_ms,
            )
            
            memory_results.append(res)
            
        return memory_results

    def select_for_consolidation(self, min_age_hours: int = 24, min_access_count: int = 1, min_salience: float = 0.3) -> List[EpisodicMemory]:
        """Returns memories that qualify for consolidation to semantic memory."""
        now = datetime.now(timezone.utc)
        qualifying = []
        
        for memory in self._episodic_store.values():
            mem_time = memory.timestamp
            if mem_time.tzinfo is None:
                mem_time = mem_time.replace(tzinfo=timezone.utc)
                
            age_hours = (now - mem_time).total_seconds() / 3600.0
            
            if age_hours >= min_age_hours and memory.access_count >= min_access_count and memory.salience >= min_salience:
                qualifying.append(memory)
                
        return qualifying

    def evict(self, threshold: float = 0.05) -> List[EpisodicMemory]:
        """Evicts memories whose retrievability falls below the threshold."""
        evicted = []
        to_remove = []
        
        for mem_id, memory in self._episodic_store.items():
            retrievability = self.decay_engine.apply_decay_single(memory)
            if retrievability < threshold:
                to_remove.append(mem_id)
                evicted.append(memory)
                
        for mem_id in to_remove:
            del self._episodic_store[mem_id]
            self._embeddings.pop(mem_id, None)
            self._dense_embeddings.pop(mem_id, None)
            self._temporal_index = [item for item in self._temporal_index if item[1] != mem_id]
            
        if evicted:
            logger.info(f"Evicted {len(evicted)} memories below threshold {threshold}")
            
        return evicted

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics about the hippocampal engine."""
        total_memories = len(self._episodic_store)
        
        if total_memories == 0:
            return {
                "total_memories": 0,
                "active_memories": 0,
                "avg_retrievability": 0.0,
                "avg_salience": 0.0,
                "oldest_memory": None,
                "newest_memory": None
            }
            
        total_retrievability = sum(self.decay_engine.apply_decay_single(m) for m in self._episodic_store.values())
        total_salience = sum(m.salience for m in self._episodic_store.values())
        
        # Sort to find oldest/newest
        sorted_memories = sorted(self._episodic_store.values(), key=lambda m: m.timestamp)
        oldest = sorted_memories[0].timestamp
        newest = sorted_memories[-1].timestamp
        
        return {
            "total_memories": total_memories,
            "active_memories": total_memories, # assuming all in store are active
            "avg_retrievability": total_retrievability / total_memories,
            "avg_salience": total_salience / total_memories,
            "oldest_memory": oldest.isoformat() if oldest else None,
            "newest_memory": newest.isoformat() if newest else None
        }

    def __len__(self) -> int:
        return len(self._episodic_store)

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self._episodic_store
