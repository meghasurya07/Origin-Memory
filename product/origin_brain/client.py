"""
Origin Brain SDK — Simple, Pythonic API for Human-Like AI Memory

Usage:
    from origin_brain.client import OriginBrainClient

    # Create a persistent brain
    client = OriginBrainClient(
        agent_id="my_agent",
        storage_path="./memory.db"  # SQLite persistence
    )

    # Add memories
    client.remember("Met Sarah at the coffee shop")
    client.remember("Budget meeting discussed 15% increase", importance=0.9)

    # Recall
    results = client.recall("coffee shop")
    for r in results:
        print(f"[{r['score']:.2f}] {r['content']}")

    # Sleep consolidation (run daily)
    report = client.sleep()

    # Stats
    stats = client.stats()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .brain import Brain
from .models import BrainConfig, MemoryType

logger = logging.getLogger(__name__)


class MemoryItem:
    """A simplified memory result for the SDK."""
    
    def __init__(self, content: str, score: float, memory_id: str, 
                 metadata: Optional[Dict[str, Any]] = None, fidelity: str = "verbatim"):
        self.content = content
        self.score = score
        self.memory_id = memory_id
        self.metadata = metadata or {}
        self.fidelity = fidelity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "memory_id": self.memory_id,
            "metadata": self.metadata,
            "fidelity": self.fidelity,
        }
    
    def __repr__(self):
        return f"MemoryItem(score={self.score:.2f}, content='{self.content[:50]}...')"


class OriginBrainClient:
    """
    High-level SDK for Origin Brain.
    
    Provides a simple, Pythonic interface for:
    - Storing memories with context and importance
    - Recalling memories with semantic search
    - Reconstructive recall (context-dependent)
    - Sleep consolidation
    - Working memory management
    - Brain statistics and health
    
    Example:
        client = OriginBrainClient("my_agent", storage_path="./brain.db")
        client.remember("I love chocolate ice cream", importance=0.8)
        results = client.recall("favorite ice cream")
        print(results[0].content)  # "I love chocolate ice cream"
    """
    
    def __init__(
        self,
        agent_id: str,
        storage_path: Optional[str] = None,
        embedding_dimension: int = 64,
        working_memory_capacity: int = 7,
        **kwargs
    ):
        """
        Initialize an Origin Brain client.
        
        Args:
            agent_id: Unique identifier for this agent's brain
            storage_path: Path to SQLite file for persistence (None = in-memory)
            embedding_dimension: Size of internal embeddings (default 64)
            working_memory_capacity: Max working memory items (default 7)
            **kwargs: Additional BrainConfig parameters
        """
        config = BrainConfig(
            agent_id=agent_id,
            embedding_dimension=embedding_dimension,
            storage_path=storage_path,
            **kwargs
        )
        self._brain = Brain(config=config)
        self._agent_id = agent_id
        
        logger.info(f"OriginBrainClient initialized for agent: {agent_id}")
    
    def remember(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        memory_type: str = "episodic"
    ) -> Dict[str, Any]:
        """
        Store a new memory.
        
        Args:
            content: The information to remember
            context: Optional context (location, time, people, etc.)
            importance: How important this memory is (0.0-1.0, None=auto)
            memory_type: Type of memory ("episodic", "semantic", "procedural")
            
        Returns:
            Dict with memory_id, action, and metadata
        """
        mem_type = MemoryType(memory_type.upper()) if memory_type != "episodic" else MemoryType.EPISODIC
        
        result = self._brain.encode(
            content=content,
            context=context,
            salience=importance,
            memory_type=mem_type
        )
        
        return {
            "memory_id": result.memory.id if result.memory else None,
            "action": result.action,
            "salience": result.memory.salience if result.memory else 0,
            "surprise_score": result.memory.metadata.get("surprise_score", 0) if result.memory else 0,
        }
    
    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[MemoryItem]:
        """
        Recall memories matching a query.
        
        Args:
            query: What to search for
            top_k: Maximum number of results
            min_score: Minimum relevance score filter
            
        Returns:
            List of MemoryItem results, sorted by relevance
        """
        result = self._brain.recall(query, top_k=top_k)
        
        items = []
        for r in result.results:
            if r.relevance_score >= min_score:
                items.append(MemoryItem(
                    content=r.memory.content if hasattr(r.memory, 'content') else str(r.memory),
                    score=r.relevance_score,
                    memory_id=r.memory.id if hasattr(r.memory, 'id') else "",
                    metadata=r.memory.metadata if hasattr(r.memory, 'metadata') else {},
                ))
        
        return items
    
    def recall_with_reconstruction(
        self,
        query: str,
        top_k: int = 5,
        emotional_state: str = "neutral",
        arousal: float = 0.5,
    ) -> List[MemoryItem]:
        """
        Reconstructive recall — memory content changes based on current state.
        
        This is the breakthrough: same query, different context = different memory.
        
        Args:
            query: What to search for
            top_k: Max results
            emotional_state: Current emotional valence
            arousal: Current arousal level (0-1)
            
        Returns:
            List of reconstructed MemoryItems with fidelity info
        """
        results = self._brain.reconstruct_recall(
            query=query,
            top_k=top_k,
            emotional_arousal=arousal,
            emotional_valence=emotional_state
        )
        
        items = []
        for r in results:
            items.append(MemoryItem(
                content=r.reconstructed_content,
                score=r.confidence,
                memory_id=r.original_memory_id,
                fidelity=r.fidelity.value if hasattr(r.fidelity, 'value') else str(r.fidelity),
            ))
        
        return items
    
    def think(self, content: str, salience: float = 0.5) -> None:
        """Add something to working memory (short-term attention)."""
        self._brain.working_memory.attend(content, salience=salience)
    
    def working_memory(self) -> List[str]:
        """Get current working memory contents."""
        return [item.content for item in self._brain.working_memory.get_active_items()]
    
    def sleep(self) -> Dict[str, Any]:
        """
        Run sleep consolidation — processes and reorganizes memories.
        Call this periodically (daily recommended).
        
        Returns:
            Sleep report with consolidation stats
        """
        report = self._brain.sleep()
        return {
            "memories_consolidated": report.total_consolidated,
            "memories_pruned": report.total_evicted,
        }
    
    def save(self) -> int:
        """Explicitly save all memories to storage. Returns count saved."""
        return self._brain.save()
    
    def stats(self) -> Dict[str, Any]:
        """Get brain statistics."""
        brain_stats = self._brain.get_statistics()
        brain_stats["agent_id"] = self._agent_id
        brain_stats["plasticity"] = self._brain.plasticity.get_statistics()
        return brain_stats
    
    def set_context(self, **kwargs) -> None:
        """Set context that influences recall (location, time, mood, etc.)."""
        self._brain.context_buffer.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear the context buffer."""
        self._brain.context_buffer.clear()
    
    @property
    def memory_count(self) -> int:
        """Number of episodic memories stored."""
        return self._brain.episodic_count
    
    @property 
    def brain(self) -> Brain:
        """Access the underlying Brain for advanced operations."""
        return self._brain
