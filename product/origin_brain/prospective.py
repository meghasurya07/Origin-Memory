import uuid
import logging
from enum import Enum
from typing import Any, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from .models import EpisodicMemory

logger = logging.getLogger(__name__)

class ProspectiveMemoryType(str, Enum):
    TIME_BASED = 'time_based'
    EVENT_BASED = 'event_based'
    ACTIVITY_BASED = 'activity_based'

class ProspectiveMemory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    trigger_type: ProspectiveMemoryType
    trigger_condition: str
    trigger_time: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal['pending', 'triggered', 'completed', 'expired'] = 'pending'
    priority: float = 0.5
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

class ProspectiveMemoryEngine:
    """
    Engine to manage prospective memories: memories for future intentions.
    """
    def __init__(self):
        self._memories: dict[str, ProspectiveMemory] = {}
        self._triggered: list[ProspectiveMemory] = []
        
    def add(
        self,
        content: str,
        trigger_type: ProspectiveMemoryType,
        trigger_condition: str,
        trigger_time: datetime | None = None,
        priority: float = 0.5,
        context: dict[str, Any] | None = None
    ) -> ProspectiveMemory:
        """Add a new prospective memory."""
        memory = ProspectiveMemory(
            content=content,
            trigger_type=trigger_type,
            trigger_condition=trigger_condition,
            trigger_time=trigger_time,
            priority=priority,
            context=context or {}
        )
        self._memories[memory.id] = memory
        logger.debug(f"Added prospective memory: {memory.id} - {content}")
        return memory
        
    def check_triggers(self, current_context: dict[str, Any] | None = None) -> list[ProspectiveMemory]:
        """Check all pending memories against time and context to trigger them."""
        current_context = current_context or {}
        newly_triggered = []
        now = datetime.now(timezone.utc)
        
        for mem_id, memory in self._memories.items():
            if memory.status != 'pending':
                continue
                
            triggered = False
            if memory.trigger_type == ProspectiveMemoryType.TIME_BASED:
                if memory.trigger_time and memory.trigger_time <= now:
                    triggered = True
            elif memory.trigger_type == ProspectiveMemoryType.EVENT_BASED:
                for k, v in current_context.items():
                    if memory.trigger_condition in str(k) or memory.trigger_condition in str(v):
                        triggered = True
                        break
            elif memory.trigger_type == ProspectiveMemoryType.ACTIVITY_BASED:
                activity = current_context.get('activity')
                if activity and str(activity) == memory.trigger_condition:
                    triggered = True
                    
            if triggered:
                memory.status = 'triggered'
                self._triggered.append(memory)
                newly_triggered.append(memory)
                logger.info(f"Triggered prospective memory: {memory.id}")
                
        return newly_triggered
        
    def complete(self, memory_id: str) -> bool:
        """Mark a triggered prospective memory as completed."""
        if memory_id in self._memories:
            memory = self._memories[memory_id]
            if memory.status in ('triggered', 'pending'):
                memory.status = 'completed'
                logger.debug(f"Completed prospective memory: {memory_id}")
                return True
        return False
        
    def get_pending(self) -> list[ProspectiveMemory]:
        """Returns all pending (not yet triggered) prospective memories."""
        return [m for m in self._memories.values() if m.status == 'pending']
        
    def get_statistics(self) -> dict[str, int]:
        """Get statistics of prospective memories."""
        stats = {
            'total': len(self._memories),
            'pending': len(self.get_pending()),
            'triggered': sum(1 for m in self._memories.values() if m.status == 'triggered'),
            'completed': sum(1 for m in self._memories.values() if m.status == 'completed'),
            'expired': sum(1 for m in self._memories.values() if m.status == 'expired')
        }
        return stats
