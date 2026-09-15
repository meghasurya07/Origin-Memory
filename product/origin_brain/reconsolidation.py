"""
Reconsolidation module for the Origin Brain SDK.

Implements Memory Reconsolidation - a critical brain mechanism where retrieved
memories become temporarily labile and can be updated before being re-stored.
This is how the brain updates existing memories instead of always creating new ones.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Literal, Optional, Tuple

from .models import EpisodicMemory, SemanticMemory, MemoryStatus
from .decay import DecayEngine

logger = logging.getLogger(__name__)

@dataclass
class ReconsolidationResult:
    """Result of a reconsolidation attempt."""
    action: Literal['reconsolidated', 'new_memory', 'no_match']
    memory_id: Optional[str]
    similarity: float
    labile_count: int


class ReconsolidationEngine:
    """
    Engine for managing memory reconsolidation.
    
    When memories are retrieved, they enter a 'labile' state where they can be
    updated with new information instead of always creating new memories.
    """
    
    def __init__(
        self,
        lability_window_seconds: float = 300.0,
        update_threshold: float = 0.3
    ):
        """
        Initialize the reconsolidation engine.
        
        Args:
            lability_window_seconds: Time window (in seconds) after retrieval during
                which memory can be updated. Defaults to 300.0 (5 minutes).
            update_threshold: Minimum similarity between old and new content to trigger
                an update versus creating a new memory. Defaults to 0.3.
        """
        self.lability_window_seconds = lability_window_seconds
        self.update_threshold = update_threshold
        # Mapping of memory_id to a tuple of (memory_object, retrieval_time)
        self._labile_memories: Dict[str, Tuple[EpisodicMemory, datetime]] = {}
        
    def on_retrieval(self, memory: EpisodicMemory) -> None:
        """
        Mark a memory as labile after it has been retrieved.
        
        Args:
            memory: The memory that was retrieved.
        """
        now = datetime.now(timezone.utc)
        self._labile_memories[memory.id] = (memory, now)
        logger.info(f"Memory {memory.id} entered labile state for {self.lability_window_seconds}s.")
        
    def process_new_input(
        self,
        content: str,
        context: Optional[dict],
        compute_similarity_fn: Callable[[str, str], float]
    ) -> ReconsolidationResult:
        """
        Process new input and determine if an existing labile memory should be updated.
        
        Args:
            content: The new memory content to process.
            context: Optional context associated with the new memory.
            compute_similarity_fn: A function to compute similarity between two strings.
            
        Returns:
            ReconsolidationResult indicating the action taken.
        """
        self._clean_expired()
        labile_count = self.get_labile_count()
        
        if labile_count == 0:
            return ReconsolidationResult(
                action='no_match',
                memory_id=None,
                similarity=0.0,
                labile_count=0
            )
            
        best_match_id = None
        best_similarity = -1.0
        best_memory = None
        
        for memory_id, (memory, _) in self._labile_memories.items():
            similarity = compute_similarity_fn(content, memory.content)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = memory_id
                best_memory = memory
                
        if best_similarity > self.update_threshold and best_memory is not None:
            # Reconsolidate the memory since similarity exceeds threshold
            self.reconsolidate(best_memory, content, context)
            return ReconsolidationResult(
                action='reconsolidated',
                memory_id=best_match_id,
                similarity=best_similarity,
                labile_count=labile_count
            )
        else:
            # Did not meet update threshold
            return ReconsolidationResult(
                action='new_memory',
                memory_id=None,
                similarity=best_similarity,
                labile_count=labile_count
            )
            
    def reconsolidate(
        self,
        memory: EpisodicMemory,
        new_content: str,
        new_context: Optional[dict] = None
    ) -> EpisodicMemory:
        """
        Reconsolidate a memory by updating it with new content and context.
        
        Args:
            memory: The memory to reconsolidate.
            new_content: The new content to append or replace.
            new_context: Optional new context to merge into the memory's context.
            
        Returns:
            The updated EpisodicMemory.
        """
        # Append content
        if memory.content and new_content not in memory.content:
            memory.content = f"{memory.content}\n{new_content}"
        else:
            memory.content = new_content
            
        # Strengthen memory stability due to successful reconsolidation
        if hasattr(memory, 'stability'):
            memory.stability = min(1.0, memory.stability + 0.1)
            
        # Update timestamp to now
        memory.timestamp = datetime.now(timezone.utc)
        
        # Merge new context if provided
        if new_context:
            if getattr(memory, 'context', None) is None:
                memory.context = {}
            memory.context.update(new_context)
            
        # Remove from labile state as it has been successfully reconsolidated
        if memory.id in self._labile_memories:
            del self._labile_memories[memory.id]
            
        logger.info(f"Memory {memory.id} successfully reconsolidated.")
        return memory
        
    def _clean_expired(self) -> int:
        """
        Remove all labile memories whose lability window has expired.
        
        Returns:
            Number of cleaned entries.
        """
        now = datetime.now(timezone.utc)
        expired_ids = []
        
        for memory_id, (memory, retrieval_time) in self._labile_memories.items():
            elapsed = (now - retrieval_time).total_seconds()
            if elapsed > self.lability_window_seconds:
                expired_ids.append(memory_id)
                
        for memory_id in expired_ids:
            del self._labile_memories[memory_id]
            logger.debug(f"Memory {memory_id} lability window expired and was removed.")
            
        return len(expired_ids)
        
    def get_labile_count(self) -> int:
        """
        Get the number of currently labile memories.
        
        Returns:
            The count of labile memories.
        """
        return len(self._labile_memories)
