"""
Temporal Invalidation Engine — Bi-Temporal Fact Management

Handles evolving facts in memory. When a new memory contradicts an older
one (e.g., "User switched from VSCode to Cursor"), the old memory gets
marked as superseded rather than keeping conflicting facts.

Inspired by:
  - Zep's bi-temporal edge tracking (Oct 2026)
  - CogMem's PEC2F schema (EMNLP 2026, arXiv:2610.11314)
  - Trace Transformation Theory (Moscovitch et al. 2005)

The key insight: facts have VALIDITY WINDOWS. "Bob works at Google" is
valid from time T1 to T2, then superseded by "Bob works at Meta" at T2.
This is fundamentally different from just storing both — it prevents
contradictions in recall.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import uuid
import re


class FactStatus(str, Enum):
    """Lifecycle of a temporal fact."""
    ACTIVE = "ACTIVE"            # Currently valid
    SUPERSEDED = "SUPERSEDED"    # Replaced by a newer fact
    INVALIDATED = "INVALIDATED"  # Explicitly contradicted
    EXPIRED = "EXPIRED"          # TTL expired


class TemporalFact(BaseModel):
    """
    A fact with a validity window.
    
    Bi-temporal model:
      - valid_from: when this fact became true in the real world
      - valid_to: when this fact stopped being true (None = still valid)
      - recorded_at: when we learned about this fact
      - invalidated_by: ID of the fact that superseded this one
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # The fact itself
    subject: str          # Entity the fact is about (e.g., "Bob")
    predicate: str        # Relationship type (e.g., "works_at")
    object: str           # Value (e.g., "Google")
    
    # Source
    source_memory_id: str = ""  # ID of the episodic memory this was extracted from
    confidence: float = 1.0
    
    # Bi-temporal timestamps
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None  # None = still valid
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Status tracking
    status: FactStatus = FactStatus.ACTIVE
    invalidated_by: Optional[str] = None  # ID of superseding fact
    
    # Metadata
    category: str = "general"  # career, health, preference, relationship, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def is_current(self) -> bool:
        """Is this fact currently valid?"""
        return self.status == FactStatus.ACTIVE and self.valid_to is None


class ConflictResolution(str, Enum):
    """Strategy for resolving conflicting facts."""
    TEMPORAL = "TEMPORAL"    # Newer fact wins (default)
    CONFIDENCE = "CONFIDENCE"  # Higher confidence wins
    MANUAL = "MANUAL"        # Flag for human review


# ─── Predicate patterns for entity-relation extraction ───────────────

# Maps natural language patterns to (subject_group, predicate, object_group)
EXTRACTION_PATTERNS = [
    # Work / Career
    (r"(\w+)\s+(?:works?|is working)\s+(?:at|for)\s+(.+)", "works_at"),
    (r"(\w+)\s+(?:got|received|has)\s+(?:a\s+)?(?:new\s+)?(?:job\s+offer|job|position|role|offer)\s+(?:at|from|with)\s+(.+)", "has_offer_from"),
    (r"(\w+)\s+(?:accepted|took|will start)\s+(?:the\s+)?(?:\w+\s+)?(?:job|position|role|offer)?\s*(?:at|from|with)?\s*(.+)", "works_at"),
    (r"(\w+)\s+(?:got promoted|was promoted)\s+to\s+(.+)", "role_is"),
    
    # Location / Travel
    (r"(\w+)\s+(?:lives?|moved?|is living)\s+(?:in|to)\s+(.+)", "lives_in"),
    (r"(\w+)\s+(?:is visiting|will visit|visited|planning to visit)\s+(.+)", "visiting"),
    
    # Preferences
    (r"(\w+)(?:'s)?\s+favorite\s+(\w+)\s+is\s+(.+)", "favorite"),
    (r"(\w+)\s+(?:prefers?|likes?)\s+(.+?)(?:\s+over|\s+more than|\s*$)", "prefers"),
    
    # Health
    (r"(\w+)\s+is\s+allergic\s+to\s+(.+)", "allergic_to"),
    (r"(\w+)\s+(?:injured|hurt)\s+(?:his|her|their)\s+(.+)", "injured"),
    
    # Relationships
    (r"(\w+)(?:'s)?\s+(?:sister|brother|mother|father|spouse|partner)\s+(?:is\s+)?(\w+)", "family_member"),
    
    # Pets
    (r"(\w+)\s+(?:adopted|got|has)\s+(?:a\s+)?(?:\w+\s+)*?(?:named|called)\s+(\w+)", "has_pet"),
]


class TemporalInvalidationEngine:
    """
    Manages bi-temporal facts with automatic contradiction detection
    and supersession.
    
    Key operations:
      1. extract_facts(content) - Extract subject-predicate-object triples
      2. register_fact(fact) - Add fact, auto-detect conflicts
      3. invalidate(fact_id, reason) - Mark a fact as superseded
      4. get_current_facts(subject) - Get all currently valid facts
      5. get_fact_history(subject, predicate) - Get temporal history
    """
    
    def __init__(self, resolution_strategy: ConflictResolution = ConflictResolution.TEMPORAL):
        self._facts: Dict[str, TemporalFact] = {}
        self._subject_index: Dict[str, List[str]] = {}  # subject -> [fact_ids]
        self._predicate_index: Dict[str, List[str]] = {}  # "subject:predicate" -> [fact_ids]
        self.resolution_strategy = resolution_strategy
        self._conflict_log: List[Dict[str, Any]] = []
    
    def extract_facts(self, content: str, source_memory_id: str = "") -> List[TemporalFact]:
        """
        Extract subject-predicate-object triples from natural language.
        
        This is a lightweight rule-based extractor. In production, this
        would use an LLM for more robust extraction.
        """
        facts = []
        content_lower = content.lower()
        
        for pattern, predicate in EXTRACTION_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    obj = groups[-1].strip().rstrip('.')
                    
                    fact = TemporalFact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        source_memory_id=source_memory_id,
                        category=self._infer_category(predicate),
                    )
                    facts.append(fact)
        
        return facts
    
    def register_fact(self, fact: TemporalFact) -> Tuple[bool, Optional[str]]:
        """
        Register a new fact. If it conflicts with an existing active fact
        on the same subject+predicate, automatically invalidate the old one.
        
        Returns:
            (was_conflict, invalidated_fact_id)
        """
        key = f"{fact.subject.lower()}:{fact.predicate}"
        conflict_id = None
        was_conflict = False
        
        # Check for conflicts
        existing_ids = self._predicate_index.get(key, [])
        for eid in existing_ids:
            existing = self._facts.get(eid)
            if existing and existing.is_current:
                # Same subject+predicate but different object = conflict
                if existing.object.lower() != fact.object.lower():
                    was_conflict = True
                    conflict_id = existing.id
                    
                    # Resolve based on strategy
                    if self.resolution_strategy == ConflictResolution.TEMPORAL:
                        # Newer wins — invalidate the old
                        self._invalidate_fact(existing.id, fact.id, 
                                            f"Superseded by newer fact: {fact.object}")
                    elif self.resolution_strategy == ConflictResolution.CONFIDENCE:
                        if fact.confidence >= existing.confidence:
                            self._invalidate_fact(existing.id, fact.id,
                                                f"Lower confidence ({existing.confidence:.2f} < {fact.confidence:.2f})")
                        else:
                            # Old fact has higher confidence — don't register new
                            self._conflict_log.append({
                                "type": "rejected",
                                "new_fact": fact.model_dump(),
                                "existing_fact": existing.model_dump(),
                                "reason": "Lower confidence than existing"
                            })
                            return True, None
                    
                    self._conflict_log.append({
                        "type": "supersession",
                        "old_fact_id": existing.id,
                        "new_fact_id": fact.id,
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "old_value": existing.object,
                        "new_value": fact.object,
                    })
                else:
                    # Same object — update confidence / metadata
                    existing.confidence = max(existing.confidence, fact.confidence)
                    existing.recorded_at = datetime.now(timezone.utc)
                    return False, None
        
        # Store the new fact
        self._facts[fact.id] = fact
        
        # Update indices
        subject_key = fact.subject.lower()
        if subject_key not in self._subject_index:
            self._subject_index[subject_key] = []
        self._subject_index[subject_key].append(fact.id)
        
        if key not in self._predicate_index:
            self._predicate_index[key] = []
        self._predicate_index[key].append(fact.id)
        
        return was_conflict, conflict_id
    
    def _invalidate_fact(self, fact_id: str, superseded_by: str, reason: str = ""):
        """Mark a fact as superseded."""
        fact = self._facts.get(fact_id)
        if fact:
            fact.status = FactStatus.SUPERSEDED
            fact.valid_to = datetime.now(timezone.utc)
            fact.invalidated_by = superseded_by
            fact.metadata["invalidation_reason"] = reason
    
    def get_current_facts(self, subject: Optional[str] = None) -> List[TemporalFact]:
        """Get all currently valid facts, optionally filtered by subject."""
        if subject:
            fact_ids = self._subject_index.get(subject.lower(), [])
            return [self._facts[fid] for fid in fact_ids 
                    if self._facts[fid].is_current]
        return [f for f in self._facts.values() if f.is_current]
    
    def get_fact_history(self, subject: str, predicate: str) -> List[TemporalFact]:
        """
        Get the complete temporal history of a fact.
        Returns facts in chronological order (oldest first).
        """
        key = f"{subject.lower()}:{predicate}"
        fact_ids = self._predicate_index.get(key, [])
        facts = [self._facts[fid] for fid in fact_ids]
        return sorted(facts, key=lambda f: f.valid_from)
    
    def get_conflicts(self) -> List[Dict[str, Any]]:
        """Get the log of all detected conflicts."""
        return self._conflict_log.copy()
    
    def query(self, subject: str, predicate: str, 
              at_time: Optional[datetime] = None) -> Optional[TemporalFact]:
        """
        Query for a fact at a specific point in time.
        If at_time is None, returns the currently valid fact.
        """
        key = f"{subject.lower()}:{predicate}"
        fact_ids = self._predicate_index.get(key, [])
        
        if at_time is None:
            # Return current active fact
            for fid in reversed(fact_ids):
                fact = self._facts[fid]
                if fact.is_current:
                    return fact
            return None
        
        # Find fact valid at the given time
        for fid in fact_ids:
            fact = self._facts[fid]
            if fact.valid_from <= at_time:
                if fact.valid_to is None or fact.valid_to > at_time:
                    return fact
        return None
    
    def _infer_category(self, predicate: str) -> str:
        """Infer category from predicate type."""
        category_map = {
            "works_at": "career", "has_offer_from": "career", "role_is": "career",
            "lives_in": "location", "visiting": "travel",
            "favorite": "preference", "prefers": "preference",
            "allergic_to": "health", "injured": "health",
            "family_member": "relationship", "has_pet": "relationship",
        }
        return category_map.get(predicate, "general")
    
    @property
    def total_facts(self) -> int:
        return len(self._facts)
    
    @property
    def active_facts(self) -> int:
        return sum(1 for f in self._facts.values() if f.is_current)
    
    @property
    def superseded_facts(self) -> int:
        return sum(1 for f in self._facts.values() if f.status == FactStatus.SUPERSEDED)
    
    def stats(self) -> Dict[str, Any]:
        return {
            "total_facts": self.total_facts,
            "active_facts": self.active_facts,
            "superseded_facts": self.superseded_facts,
            "conflicts_detected": len(self._conflict_log),
            "subjects": len(self._subject_index),
            "predicates": len(self._predicate_index),
        }
