import uuid
import logging
from datetime import datetime
from enum import Enum
from typing import Callable, Literal, Optional, List, Set, Tuple, Dict, Any

from pydantic import BaseModel, Field

# Assuming these are available from the local models module
from .models import EpisodicMemory, SemanticMemory

logger = logging.getLogger(__name__)

class InterferenceType(str, Enum):
    """Types of memory interference."""
    PROACTIVE = 'proactive'
    RETROACTIVE = 'retroactive'
    RETRIEVAL_INDUCED = 'retrieval_induced'
    CONTRADICTION = 'contradiction'


class InterferenceEvent(BaseModel):
    """Represents a detected interference event between two memories."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interference_type: InterferenceType
    memory_a_id: str
    memory_b_id: str
    similarity: float
    description: str
    resolution: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InterferenceDetector:
    """
    Detects and resolves interference between memories, handling conflicting information.
    """
    def __init__(
        self,
        contradiction_threshold: float = 0.7,
        similarity_fn: Optional[Callable[[str, str], float]] = None
    ):
        self.contradiction_threshold = contradiction_threshold
        self.similarity_fn = similarity_fn
        self._events: List[InterferenceEvent] = []
        self._contradiction_pairs: Set[Tuple[str, str]] = set()

    def detect_contradictions_in_text(self, text_a: str, text_b: str) -> bool:
        """
        Check if two texts contain contradictory information.
        
        Looks for:
        - Opposite sentiment
        - Negation
        - Different values for same attribute
        - Temporal supersession
        """
        # Basic heuristic implementation for contradiction detection
        negation_terms = {'not', 'no longer', 'instead', 'changed', 'actually', 'correction', 'never', 'dislikes', 'hate'}
        temporal_terms = {'now', 'updated', 'new', 'switched to', 'moved to', 'used to'}
        
        lower_a = text_a.lower()
        lower_b = text_b.lower()
        
        a_has_neg = any(term in lower_a for term in negation_terms)
        b_has_neg = any(term in lower_b for term in negation_terms)
        
        a_has_temp = any(term in lower_a for term in temporal_terms)
        b_has_temp = any(term in lower_b for term in temporal_terms)
        
        # Simplistic contradiction heuristic based on asymmetric negation or temporal markers
        if (a_has_neg and not b_has_neg) or (b_has_neg and not a_has_neg):
            return True
            
        if (a_has_temp and not b_has_temp) or (b_has_temp and not a_has_temp):
            return True

        return False

    def detect(
        self, 
        new_memory: EpisodicMemory, 
        existing_memories: List[EpisodicMemory], 
        compute_similarity: Optional[Callable[[str, str], float]] = None
    ) -> List[InterferenceEvent]:
        """
        Compare new memory against all existing memories to detect potential interference.
        """
        sim_fn = compute_similarity or self.similarity_fn
        if not sim_fn:
            logger.warning("No similarity function provided, cannot detect interference.")
            return []

        detected_events = []
        
        # Fallback attribute extraction based on expected EpisodicMemory schema
        new_text = getattr(new_memory, 'content', getattr(new_memory, 'text', str(new_memory)))
        
        for existing in existing_memories:
            existing_text = getattr(existing, 'content', getattr(existing, 'text', str(existing)))
            
            similarity = sim_fn(existing_text, new_text)
            
            if similarity > self.contradiction_threshold:
                is_contradiction = self.detect_contradictions_in_text(existing_text, new_text)
                
                if is_contradiction:
                    event = InterferenceEvent(
                        interference_type=InterferenceType.CONTRADICTION,
                        memory_a_id=existing.id,
                        memory_b_id=new_memory.id,
                        similarity=similarity,
                        description=f"Contradiction detected between {existing.id} and {new_memory.id}"
                    )
                    self._contradiction_pairs.add((existing.id, new_memory.id))
                else:
                    event = InterferenceEvent(
                        interference_type=InterferenceType.PROACTIVE,
                        memory_a_id=existing.id,
                        memory_b_id=new_memory.id,
                        similarity=similarity,
                        description=f"Proactive interference detected between {existing.id} and {new_memory.id}"
                    )
                
                self._events.append(event)
                detected_events.append(event)
                
        return detected_events

    def resolve(self, event: InterferenceEvent, strategy: Literal['keep_newer', 'keep_both', 'keep_stronger', 'merge']) -> str:
        """
        Resolves an interference event using the specified strategy.
        """
        if strategy == 'keep_newer':
            resolution_desc = f"Marked older memory ({event.memory_a_id}) as superseded by newer memory ({event.memory_b_id})."
        elif strategy == 'keep_both':
            resolution_desc = f"Kept both memories and added an association link between {event.memory_a_id} and {event.memory_b_id}."
        elif strategy == 'keep_stronger':
            resolution_desc = f"Resolved interference by keeping the memory with higher salience/stability between {event.memory_a_id} and {event.memory_b_id}."
        elif strategy == 'merge':
            resolution_desc = f"Merged content and context from {event.memory_a_id} and {event.memory_b_id}."
        else:
            raise ValueError(f"Unknown resolution strategy: {strategy}")
            
        event.resolution = resolution_desc
        return resolution_desc

    def get_contradiction_pairs(self) -> List[Tuple[str, str]]:
        """Returns all known contradiction pairs."""
        return list(self._contradiction_pairs)

    def get_event_log(self) -> List[InterferenceEvent]:
        """Returns all interference events detected over time."""
        return self._events

    # ─── Retrieval-Induced Forgetting (RIF) ───────────────────────
    # Anderson et al. 1994: When you retrieve a memory, COMPETING
    # memories in the same category become LESS accessible.
    # 
    # Example: Recall "fruits: orange" → "fruits: banana" becomes 
    # HARDER to recall (inhibited), while "tools: hammer" is unaffected.
    #
    # This is ACTIVE INHIBITION, not passive decay.
    # ──────────────────────────────────────────────────────────────
    
    def apply_retrieval_induced_forgetting(
        self,
        retrieved_memory: EpisodicMemory,
        all_memories: List[EpisodicMemory],
        embedding_engine: Optional[Any] = None,
        inhibition_rate: float = 0.05,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Apply RIF: when a memory is retrieved, suppress competitors.
        
        Competitors = memories that share semantic category with the 
        retrieved memory but are NOT the retrieved memory itself.
        
        Args:
            retrieved_memory: The memory that was just successfully recalled
            all_memories: All episodic memories to check for competition
            embedding_engine: Optional EmbeddingEngine for semantic similarity
            inhibition_rate: How much to reduce competitor salience (0.0-1.0)
            similarity_threshold: Min similarity to count as a competitor
            
        Returns:
            List of inhibited memory records [{id, old_salience, new_salience, similarity}]
        """
        inhibited = []
        
        for mem in all_memories:
            if mem.id == retrieved_memory.id:
                continue
            
            # Compute similarity to determine if competitor
            similarity = 0.0
            if embedding_engine is not None:
                similarity = embedding_engine.similarity(
                    retrieved_memory.content, mem.id
                )
            else:
                # Fallback: simple word overlap
                words_a = set(retrieved_memory.content.lower().split())
                words_b = set(mem.content.lower().split())
                if words_a | words_b:
                    similarity = len(words_a & words_b) / len(words_a | words_b)
            
            if similarity >= similarity_threshold:
                # This is a competitor — INHIBIT it
                old_salience = mem.salience
                # Reduce salience (but never below 0.05)
                mem.salience = max(0.05, mem.salience - inhibition_rate)
                
                inhibited.append({
                    "memory_id": mem.id,
                    "old_salience": old_salience,
                    "new_salience": mem.salience,
                    "similarity": similarity,
                })
                
                # Log the RIF event
                event = InterferenceEvent(
                    interference_type=InterferenceType.RETRIEVAL_INDUCED,
                    memory_a_id=retrieved_memory.id,
                    memory_b_id=mem.id,
                    similarity=similarity,
                    description=(
                        f"RIF: retrieving {retrieved_memory.id} "
                        f"inhibited competitor {mem.id} "
                        f"(salience {old_salience:.3f}→{mem.salience:.3f})"
                    ),
                    resolution="automatic_inhibition"
                )
                self._events.append(event)
        
        if inhibited:
            logger.info(
                f"RIF applied: {len(inhibited)} competitors suppressed "
                f"after retrieving {retrieved_memory.id}"
            )
        
        return inhibited

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics on interference detection and resolution."""
        total_events = len(self._events)
        by_type = {t: 0 for t in InterferenceType}
        unresolved_count = 0
        
        for e in self._events:
            by_type[e.interference_type] += 1
            if e.resolution is None:
                unresolved_count += 1
                
        resolution_rate = 0.0
        if total_events > 0:
            resolution_rate = (total_events - unresolved_count) / total_events
            
        return {
            'total_events': total_events,
            'by_type': {k.value: v for k, v in by_type.items()},
            'unresolved_count': unresolved_count,
            'resolution_rate': resolution_rate,
            'rif_events': by_type.get(InterferenceType.RETRIEVAL_INDUCED, 0),
        }
