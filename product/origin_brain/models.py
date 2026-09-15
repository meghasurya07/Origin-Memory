import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

class MemoryTier(str, Enum):
    """Tiers of memory based on access frequency and persistence."""
    WORKING = "WORKING"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"
    ARCHIVAL = "ARCHIVAL"

class MemoryType(str, Enum):
    """Categorization of memory types."""
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"
    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    EVENT = "EVENT"
    RELATIONSHIP = "RELATIONSHIP"

class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""
    ACTIVE = "ACTIVE"
    CONSOLIDATING = "CONSOLIDATING"
    ARCHIVED = "ARCHIVED"
    DECAYED = "DECAYED"
    EVICTED = "EVICTED"

class SemanticMemory(BaseModel):
    """Semantic memory for general facts, concepts, and relationships."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: str
    value: str
    confidence: float
    source_episodes: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    category: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EpisodicMemory(BaseModel):
    """Episodic memory for specific events or experiences tied to a context."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = Field(default_factory=dict)
    salience: float = 0.5
    stability: float = 1.0
    retrievability: float = 1.0
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    memory_type: MemoryType = MemoryType.EPISODIC
    status: MemoryStatus = MemoryStatus.ACTIVE
    associations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def compute_retrievability(self) -> float:
        """
        Compute retrievability based on the Ebbinghaus forgetting curve.
        R(t) = e^(-t/S) * importance_boost
        where t is the time since last access in days, S is stability.
        """
        now = datetime.now(timezone.utc)
        # Handle naive datetimes if necessary
        if self.last_accessed.tzinfo is None:
            now = now.replace(tzinfo=None)
        
        delta = now - self.last_accessed
        t_days = max(0.0, delta.total_seconds() / 86400.0)
        importance_boost = 1.0 + self.salience
        
        self.retrievability = math.exp(-t_days / self.stability) * importance_boost
        return self.retrievability

    def on_retrieval(self, alpha: float = 0.3, w: float = -0.01) -> None:
        """
        Update stability using spaced repetition formula:
        S_{n+1} = S_n * (1 + alpha * e^(w * S_n))
        Increments access_count and updates last_accessed.
        """
        self.stability = self.stability * (1.0 + alpha * math.exp(w * self.stability))
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def is_alive(self, threshold: float = 0.05) -> bool:
        """Check if memory is still alive based on its retrievability."""
        return self.compute_retrievability() > threshold

    def to_semantic(self) -> SemanticMemory:
        """Converts episodic memory to a semantic representation."""
        return SemanticMemory(
            key="semantic_from_episode",
            value=self.content,
            confidence=self.salience,
            source_episodes=[self.id],
            category="fact",
            metadata=self.metadata
        )

class ProceduralMemory(BaseModel):
    """Procedural memory for step-by-step processes or actions."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    steps: List[str]
    trigger_conditions: List[str]
    success_rate: float = 0.0
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryQuery(BaseModel):
    """Query configuration for searching memory."""
    query: str
    memory_types: Optional[List[MemoryType]] = None
    temporal_range: Optional[Tuple[datetime, datetime]] = None
    top_k: int = 5
    min_salience: float = 0.0
    min_retrievability: float = 0.05
    context_filter: Optional[Dict[str, Any]] = None

class MemoryResult(BaseModel):
    """Result from a memory query."""
    memory: Union[EpisodicMemory, SemanticMemory, ProceduralMemory]
    relevance_score: float
    tier: MemoryTier
    retrieval_latency_ms: float

class ConsolidationResult(BaseModel):
    """Result of a memory consolidation process."""
    consolidated_count: int
    evicted_count: int
    archived_count: int
    new_semantic_count: int
    duration_seconds: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BrainConfig(BaseModel):
    """Configuration for the Origin Brain."""
    agent_id: str
    working_memory_tokens: int = 128000
    episodic_capacity: int = 1_000_000
    consolidation_interval_seconds: int = 3600
    decay_model: Literal['ebbinghaus', 'ebbinghaus_weighted', 'power_law'] = 'ebbinghaus_weighted'
    initial_stability: float = 1.0
    survival_threshold: float = 0.05
    salience_threshold: float = 0.3
    novelty_threshold: float = 0.3
    spaced_repetition_alpha: float = 0.3
    spaced_repetition_w: float = -0.01
    consolidation_min_age_hours: int = 24
    archival_min_age_days: int = 90
    archival_backend: Literal['silicon', 'dna'] = 'silicon'
    embedding_model: str = 'text-embedding-3-small'
    embedding_dimension: int = 1536
    # v0.5: Persistent storage path (None = in-memory only)
    storage_path: Optional[str] = None

