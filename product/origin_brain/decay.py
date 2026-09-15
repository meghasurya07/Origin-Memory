import math
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from datetime import datetime, timezone

from .models import EpisodicMemory

logger = logging.getLogger(__name__)

class DecayModel(ABC):
    """Abstract base class for memory decay models."""
    
    @abstractmethod
    def compute_retrievability(self, stability: float, time_elapsed_seconds: float, salience: float) -> float:
        """Compute current retrievability based on time elapsed and stability."""
        pass

    @abstractmethod
    def update_stability_on_retrieval(self, current_stability: float, access_count: int) -> float:
        """Calculate new stability after memory is retrieved/accessed."""
        pass

    def should_evict(self, retrievability: float, threshold: float = 0.05) -> bool:
        """Determine if memory should be evicted based on retrievability."""
        return retrievability < threshold


class EbbinghausDecay(DecayModel):
    """Classic exponential decay based on Ebbinghaus forgetting curve."""
    
    def __init__(self, alpha: float = 0.3, w: float = -0.01):
        self.alpha = alpha
        self.w = w

    def compute_retrievability(self, stability: float, time_elapsed_seconds: float, salience: float) -> float:
        if stability <= 0:
            return 0.0
        # R(t) = e^(-t/S)
        return math.exp(-time_elapsed_seconds / stability)

    def update_stability_on_retrieval(self, current_stability: float, access_count: int) -> float:
        # S_{n+1} = S_n * (1 + alpha * e^(w * S_n))
        return current_stability * (1.0 + self.alpha * math.exp(self.w * current_stability))


class WeightedEbbinghausDecay(DecayModel):
    """Extended Ebbinghaus decay with salience weighting."""
    
    def __init__(self, alpha: float = 0.3, w: float = -0.01):
        self.alpha = alpha
        self.w = w

    def compute_retrievability(self, stability: float, time_elapsed_seconds: float, salience: float) -> float:
        if stability <= 0:
            return 0.0
        # R(t) = e^(-t / (S * salience_boost)) where salience_boost = 1.0 + salience * 0.5
        salience_boost = 1.0 + salience * 0.5
        return math.exp(-time_elapsed_seconds / (stability * salience_boost))

    def update_stability_on_retrieval(self, current_stability: float, access_count: int) -> float:
        return current_stability * (1.0 + self.alpha * math.exp(self.w * current_stability))


class PowerLawDecay(DecayModel):
    """Power law forgetting (Wixted & Ebbesen 1991)."""
    
    def __init__(self, a: float = 1.0, b: float = 0.5, alpha: float = 0.1):
        self.a = a
        self.b = b
        self.alpha = alpha

    def compute_retrievability(self, stability: float, time_elapsed_seconds: float, salience: float) -> float:
        # R(t) = a * t^(-b)
        # Here, stability is treated as 'a' after learning.
        if time_elapsed_seconds <= 0:
            return stability
        
        # Avoid complex numbers/math domain errors
        time_val = max(1.0, time_elapsed_seconds)
        return stability * math.pow(time_val, -self.b)

    def update_stability_on_retrieval(self, current_stability: float, access_count: int) -> float:
        # Scales 'a' (represented by stability) by (1 + alpha)
        return current_stability * (1.0 + self.alpha)


class DecayEngine:
    """Orchestrates decay across a collection of episodic memories."""
    
    def __init__(self, model_type: str = 'ebbinghaus', **kwargs):
        self.model_type = model_type.lower()
        self.model = self._create_model(**kwargs)
        logger.info(f"Initialized DecayEngine with model: {self.model_type}")

    def _create_model(self, **kwargs) -> DecayModel:
        if self.model_type == 'ebbinghaus':
            return EbbinghausDecay(**kwargs)
        elif self.model_type == 'ebbinghaus_weighted':
            return WeightedEbbinghausDecay(**kwargs)
        elif self.model_type == 'power_law':
            return PowerLawDecay(**kwargs)
        else:
            raise ValueError(f"Unknown decay model type: {self.model_type}")

    def _get_time_elapsed(self, memory: EpisodicMemory) -> float:
        """Calculate time elapsed since memory was last accessed."""
        now = datetime.now(timezone.utc)
        last = memory.last_accessed
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        elapsed = (now - last).total_seconds()
        return max(0.0, elapsed)

    def apply_decay_single(self, memory: EpisodicMemory) -> float:
        """Applies decay to a single memory and returns the new retrievability."""
        time_elapsed = self._get_time_elapsed(memory)
        salience = getattr(memory, 'salience', 0.0)
        stability = getattr(memory, 'stability', 1.0)
        
        retrievability = self.model.compute_retrievability(stability, time_elapsed, salience)
        memory.retrievability = retrievability
        return retrievability

    def apply_decay(self, memories: List[EpisodicMemory]) -> Tuple[List[EpisodicMemory], List[EpisodicMemory]]:
        """Computes retrievability for a list of memories. Returns (surviving, evicted)."""
        surviving = []
        evicted = []
        
        for memory in memories:
            retrievability = self.apply_decay_single(memory)
            if self.model.should_evict(retrievability):
                evicted.append(memory)
            else:
                surviving.append(memory)
                
        logger.debug(f"Decay applied: {len(surviving)} survived, {len(evicted)} evicted.")
        return surviving, evicted

    def reinforce(self, memory: EpisodicMemory) -> EpisodicMemory:
        """Updates stability on retrieval using spaced repetition math."""
        current_stability = getattr(memory, 'stability', 1.0)
        access_count = getattr(memory, 'access_count', 0)
        
        new_stability = self.model.update_stability_on_retrieval(current_stability, access_count)
        memory.stability = new_stability
        
        if hasattr(memory, 'access_count'):
            memory.access_count += 1
            
        memory.last_accessed = datetime.now(timezone.utc)
        
        logger.debug(f"Memory reinforced. New stability: {new_stability:.4f}")
        return memory

    def batch_statistics(self, memories: List[EpisodicMemory]) -> Dict[str, Any]:
        """Returns statistical overview of a batch of memories."""
        total = len(memories)
        if total == 0:
            return {
                'total': 0,
                'alive': 0,
                'decayed': 0,
                'avg_retrievability': 0.0,
                'avg_stability': 0.0
            }
            
        alive_count = 0
        decayed_count = 0
        total_retrievability = 0.0
        total_stability = 0.0
        
        for mem in memories:
            r = getattr(mem, 'retrievability', 0.0)
            s = getattr(mem, 'stability', 0.0)
            total_retrievability += r
            total_stability += s
            
            if self.model.should_evict(r):
                decayed_count += 1
            else:
                alive_count += 1
                
        return {
            'total': total,
            'alive': alive_count,
            'decayed': decayed_count,
            'avg_retrievability': total_retrievability / total,
            'avg_stability': total_stability / total
        }
