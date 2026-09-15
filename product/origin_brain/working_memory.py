"""
Working Memory Module — Capacity-Limited Attention Buffer

Implements the brain's working memory system with biologically-grounded
capacity limits based on the theta-gamma coupling mechanism.

The brain's working memory capacity (~4 items) emerges from physics:
    Capacity ≈ f_gamma / f_theta ≈ 40Hz / 8Hz ≈ 5 items

Items compete for slots based on salience, recency, and goal-relevance.
When the buffer is full, the least important item is displaced.

Brain analogue: Prefrontal cortex + theta-gamma phase-amplitude coupling
Neuroscience: Lisman & Idiart 1995, Cowan 2010, Baddeley 2000

This module replaces the simple context_buffer dict with a proper
capacity-limited, priority-based working memory system.
"""

from __future__ import annotations

import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkingMemoryItem(BaseModel):
    """An item held in working memory."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str                         # What's being held
    source_memory_id: Optional[str] = None  # Where it came from (if from LTM)
    priority: float = 0.5               # Current priority (0-1)
    salience: float = 0.5               # Intrinsic importance
    recency: float = 1.0                # How recently activated (decays)
    activation: float = 1.0             # Current activation level
    category: str = "general"           # Semantic category
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_refreshed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkingMemoryConfig(BaseModel):
    """Configuration for working memory."""
    capacity: int = 5                   # Max items (theta-gamma: ~4-7)
    recency_decay: float = 0.85         # Per-cycle recency decay
    activation_threshold: float = 0.2   # Below this, item is evicted
    rehearsal_boost: float = 0.3        # Boost from active rehearsal
    displacement_strategy: str = "lowest_priority"  # How to choose what to evict


class DisplacementEvent(BaseModel):
    """Record of an item being displaced from working memory."""
    displaced_item_id: str
    displaced_content: str
    displaced_by_id: str
    displaced_by_content: str
    reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkingMemoryEngine:
    """
    Capacity-limited working memory with priority-based displacement.
    
    Models the PFC's working memory system:
    - Limited capacity (~4-7 items, based on theta-gamma coupling)
    - Items compete for slots based on priority = f(salience, recency, goal-relevance)
    - Least important items displaced when buffer is full
    - Active rehearsal maintains items (prevents displacement)
    - Recency decays over time (unrehearsed items fade)
    
    Usage:
        wm = WorkingMemoryEngine(WorkingMemoryConfig(capacity=5))
        
        # Add items
        wm.attend("The database is down", salience=0.9)
        wm.attend("Meeting at 3pm", salience=0.5)
        
        # Check what's active
        active = wm.get_active_items()
        
        # Rehearse to maintain
        wm.rehearse("The database is down")
        
        # Decay unrehearsed items
        wm.decay_cycle()
    """

    def __init__(self, config: Optional[WorkingMemoryConfig] = None):
        self.config = config or WorkingMemoryConfig()
        self._items: OrderedDict[str, WorkingMemoryItem] = OrderedDict()
        self._displacement_log: List[DisplacementEvent] = []
        self._total_attended: int = 0
        self._total_displaced: int = 0
        
        logger.info(f"WorkingMemoryEngine initialized (capacity={self.config.capacity})")

    def attend(
        self,
        content: str,
        salience: float = 0.5,
        source_memory_id: Optional[str] = None,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[WorkingMemoryItem, Optional[DisplacementEvent]]:
        """
        Attend to new information — bring it into working memory.
        
        If working memory is full, the lowest-priority item is displaced.
        
        Args:
            content: What to hold in working memory
            salience: How important this item is (0-1)
            source_memory_id: If retrieved from LTM, the source memory ID
            category: Semantic category for grouping
            metadata: Additional context
            
        Returns:
            Tuple of (the new item, displacement event if something was evicted)
        """
        self._total_attended += 1
        
        # Check if this content is already in WM (update instead of duplicate)
        for item_id, item in self._items.items():
            if item.content == content:
                # Refresh existing item
                item.activation = 1.0
                item.recency = 1.0
                item.last_refreshed = datetime.now(timezone.utc)
                item.priority = self._compute_priority(item)
                # Move to end (most recent)
                self._items.move_to_end(item_id)
                return item, None
        
        # Create new item
        new_item = WorkingMemoryItem(
            content=content,
            source_memory_id=source_memory_id,
            priority=salience,
            salience=salience,
            recency=1.0,
            activation=1.0,
            category=category,
            metadata=metadata or {}
        )
        new_item.priority = self._compute_priority(new_item)
        
        displacement = None
        
        # If at capacity, displace lowest-priority item
        if len(self._items) >= self.config.capacity:
            displaced_id, displaced_item = self._find_displacement_target()
            if displaced_item is not None:
                displacement = DisplacementEvent(
                    displaced_item_id=displaced_id,
                    displaced_content=displaced_item.content,
                    displaced_by_id=new_item.id,
                    displaced_by_content=content,
                    reason=f"capacity_limit (priority {displaced_item.priority:.3f} < {new_item.priority:.3f})"
                )
                del self._items[displaced_id]
                self._displacement_log.append(displacement)
                self._total_displaced += 1
                
                logger.debug(
                    f"Displaced '{displaced_item.content[:30]}...' "
                    f"(priority={displaced_item.priority:.3f}) for "
                    f"'{content[:30]}...' (priority={new_item.priority:.3f})"
                )
        
        self._items[new_item.id] = new_item
        return new_item, displacement

    def rehearse(self, content: str) -> Optional[WorkingMemoryItem]:
        """
        Actively rehearse an item to maintain it in working memory.
        
        Rehearsal boosts activation and recency, preventing decay.
        This is the brain's "inner voice" repeating information.
        """
        for item in self._items.values():
            if item.content == content or content.lower() in item.content.lower():
                item.activation = min(1.0, item.activation + self.config.rehearsal_boost)
                item.recency = 1.0
                item.last_refreshed = datetime.now(timezone.utc)
                item.priority = self._compute_priority(item)
                return item
        return None

    def decay_cycle(self):
        """
        Run a decay cycle — unrehearsed items lose activation.
        
        This models the natural fading of items not being actively
        maintained. Items below activation_threshold are evicted.
        """
        to_evict = []
        for item_id, item in self._items.items():
            item.recency *= self.config.recency_decay
            item.activation *= self.config.recency_decay
            item.priority = self._compute_priority(item)
            
            if item.activation < self.config.activation_threshold:
                to_evict.append(item_id)
        
        for item_id in to_evict:
            del self._items[item_id]
            self._total_displaced += 1

    def _compute_priority(self, item: WorkingMemoryItem) -> float:
        """
        Compute priority as weighted combination of salience and recency.
        
        Priority = 0.6 * salience + 0.4 * recency
        """
        return 0.6 * item.salience + 0.4 * item.recency

    def _find_displacement_target(self) -> Tuple[str, Optional[WorkingMemoryItem]]:
        """Find the lowest-priority item to displace."""
        if not self._items:
            return "", None
        
        min_id = min(self._items, key=lambda k: self._items[k].priority)
        return min_id, self._items[min_id]

    def get_active_items(self) -> List[WorkingMemoryItem]:
        """Get all currently active items, sorted by priority (highest first)."""
        return sorted(self._items.values(), key=lambda i: i.priority, reverse=True)

    def get_context_dict(self) -> Dict[str, Any]:
        """
        Export working memory as a context dictionary.
        
        This replaces the old context_buffer dict.
        """
        result = {}
        for i, item in enumerate(self.get_active_items()):
            result[f"wm_slot_{i}"] = item.content
            if item.category != "general":
                result[f"wm_category_{i}"] = item.category
        result["wm_item_count"] = len(self._items)
        return result

    def contains(self, content: str) -> bool:
        """Check if content is currently in working memory."""
        content_lower = content.lower()
        return any(
            content_lower in item.content.lower()
            for item in self._items.values()
        )

    def clear(self):
        """Clear all items from working memory."""
        self._items.clear()

    @property
    def current_load(self) -> int:
        """Current number of items in working memory."""
        return len(self._items)

    @property
    def is_full(self) -> bool:
        """Whether working memory is at capacity."""
        return len(self._items) >= self.config.capacity

    def get_statistics(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        items = list(self._items.values())
        return {
            "capacity": self.config.capacity,
            "current_load": len(items),
            "utilization": len(items) / max(self.config.capacity, 1),
            "total_attended": self._total_attended,
            "total_displaced": self._total_displaced,
            "displacement_rate": self._total_displaced / max(self._total_attended, 1),
            "avg_priority": sum(i.priority for i in items) / max(len(items), 1),
            "avg_activation": sum(i.activation for i in items) / max(len(items), 1),
            "categories": list(set(i.category for i in items)),
        }
