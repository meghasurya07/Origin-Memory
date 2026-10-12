"""
Shared Memory Pool — Multi-Agent Transactive Memory

Implements the emerging 3-tier shared memory architecture for multi-agent systems:
1. Global Memory  — team-wide shared facts, goals, environmental state
2. Group Memory   — task-scoped partitions for collaborating agent clusters
3. Private Memory  — each agent's Brain (already built)

Based on:
- Transactive Memory Systems (Wegner 1987)
- MATM: Multi-Agent Transactive Memory (Kim et al. 2026)
- CoMem: Collective-Individual Memory Synergy (Yu et al. 2026)
- MemClaw failure taxonomy: leakage, staleness, contradiction, provenance

Key design decisions:
- Each agent's Brain publishes to the shared pool during consolidation
- Temporal supersession: latest write wins for conflicting facts
- Provenance tracking: every memory records which agent asserted it
- Scoped access: agents only see their group + global memories
- Conflict detection: flag contradictions for resolution

Brain analogue: Social/collective memory in human groups
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryScope(str, Enum):
    """Access scope for shared memories."""
    GLOBAL = "global"      # Visible to ALL agents
    GROUP = "group"        # Visible to agents in the same group
    PRIVATE = "private"    # Only visible to the owning agent


class ConflictStrategy(str, Enum):
    """How to resolve conflicting memories."""
    LATEST_WINS = "latest_wins"      # Temporal supersession
    HIGHER_CONFIDENCE = "confidence"  # Keep more confident assertion
    KEEP_BOTH = "keep_both"          # Store both, flag conflict
    MERGE = "merge"                  # Attempt semantic merge


class SharedMemoryConfig(BaseModel):
    """Configuration for the shared memory pool."""
    conflict_strategy: ConflictStrategy = ConflictStrategy.LATEST_WINS
    max_global_memories: int = 100_000
    max_group_memories: int = 50_000
    enable_provenance: bool = True
    enable_conflict_detection: bool = True
    staleness_threshold_seconds: int = 3600  # 1 hour


@dataclass
class SharedMemoryEntry:
    """A single entry in the shared memory pool."""
    id: str
    key: str
    value: str
    scope: MemoryScope
    group_id: Optional[str]  # Which group (None for global)
    
    # Provenance
    author_agent_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata
    confidence: float = 1.0
    category: str = "general"
    access_count: int = 0
    
    # Conflict tracking
    is_conflicted: bool = False
    conflicting_ids: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None


class SharedMemoryPool:
    """
    Multi-agent shared memory pool with scoped access and conflict resolution.
    
    Usage:
        pool = SharedMemoryPool()
        
        # Agent 1 publishes a global fact
        pool.publish("agent_1", "CEO_name", "Alice Smith", scope=MemoryScope.GLOBAL)
        
        # Agent 2 publishes to their group
        pool.publish("agent_2", "task_status", "complete", 
                     scope=MemoryScope.GROUP, group_id="research_team")
        
        # Agent 3 queries (sees global + their group)
        results = pool.query("agent_3", "CEO_name", groups=["research_team"])
        
        # Register agents into groups
        pool.join_group("agent_1", "research_team")
        pool.join_group("agent_2", "research_team")
    """
    
    def __init__(self, config: Optional[SharedMemoryConfig] = None):
        self.config = config or SharedMemoryConfig()
        
        # Memory stores by scope
        self._global: Dict[str, SharedMemoryEntry] = {}   # key → entry
        self._groups: Dict[str, Dict[str, SharedMemoryEntry]] = defaultdict(dict)  # group_id → {key → entry}
        
        # Agent group memberships
        self._agent_groups: Dict[str, Set[str]] = defaultdict(set)  # agent_id → {group_ids}
        self._group_agents: Dict[str, Set[str]] = defaultdict(set)  # group_id → {agent_ids}
        
        # Provenance index: agent_id → [memory_ids]
        self._provenance: Dict[str, List[str]] = defaultdict(list)
        
        # Conflict log
        self._conflicts: List[Dict[str, Any]] = []
        
        # Statistics
        self._total_publishes: int = 0
        self._total_queries: int = 0
        self._total_conflicts: int = 0
        
        logger.info("SharedMemoryPool initialized")
    
    def join_group(self, agent_id: str, group_id: str):
        """Register an agent as a member of a group."""
        self._agent_groups[agent_id].add(group_id)
        self._group_agents[group_id].add(agent_id)
        logger.debug(f"Agent {agent_id} joined group {group_id}")
    
    def leave_group(self, agent_id: str, group_id: str):
        """Remove an agent from a group."""
        self._agent_groups[agent_id].discard(group_id)
        self._group_agents[group_id].discard(agent_id)
    
    def publish(
        self,
        agent_id: str,
        key: str,
        value: str,
        scope: MemoryScope = MemoryScope.GLOBAL,
        group_id: Optional[str] = None,
        confidence: float = 1.0,
        category: str = "general",
    ) -> SharedMemoryEntry:
        """
        Publish a memory to the shared pool.
        
        Args:
            agent_id: ID of the publishing agent
            key: Memory key (e.g., "CEO_name", "project_status")
            value: Memory value
            scope: Visibility scope (GLOBAL, GROUP, PRIVATE)
            group_id: Required for GROUP scope
            confidence: 0.0-1.0 confidence in this assertion
            category: Semantic category
            
        Returns:
            The created/updated SharedMemoryEntry
        """
        if scope == MemoryScope.GROUP and not group_id:
            raise ValueError("group_id required for GROUP scope")
        
        entry = SharedMemoryEntry(
            id=str(uuid.uuid4()),
            key=key,
            value=value,
            scope=scope,
            group_id=group_id,
            author_agent_id=agent_id,
            confidence=confidence,
            category=category,
        )
        
        # Check for conflicts before storing
        existing = self._get_existing(key, scope, group_id)
        if existing and existing.value != value:
            entry = self._resolve_conflict(existing, entry)
        
        # Store in appropriate scope
        if scope == MemoryScope.GLOBAL:
            self._global[key] = entry
        elif scope == MemoryScope.GROUP:
            self._groups[group_id][key] = entry
        
        # Track provenance
        if self.config.enable_provenance:
            self._provenance[agent_id].append(entry.id)
        
        self._total_publishes += 1
        return entry
    
    def _get_existing(
        self,
        key: str,
        scope: MemoryScope,
        group_id: Optional[str]
    ) -> Optional[SharedMemoryEntry]:
        """Get existing entry for a key in the given scope."""
        if scope == MemoryScope.GLOBAL:
            return self._global.get(key)
        elif scope == MemoryScope.GROUP and group_id:
            return self._groups.get(group_id, {}).get(key)
        return None
    
    def _resolve_conflict(
        self,
        existing: SharedMemoryEntry,
        new: SharedMemoryEntry,
    ) -> SharedMemoryEntry:
        """Resolve a conflict between existing and new memory."""
        self._total_conflicts += 1
        
        conflict_record = {
            "existing_id": existing.id,
            "new_id": new.id,
            "key": existing.key,
            "existing_value": existing.value,
            "new_value": new.value,
            "existing_agent": existing.author_agent_id,
            "new_agent": new.author_agent_id,
            "strategy": self.config.conflict_strategy.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        if self.config.enable_conflict_detection:
            self._conflicts.append(conflict_record)
        
        if self.config.conflict_strategy == ConflictStrategy.LATEST_WINS:
            # Temporal supersession — new wins
            existing.superseded_by = new.id
            logger.debug(f"Conflict on '{existing.key}': latest wins ({new.author_agent_id})")
            return new
        
        elif self.config.conflict_strategy == ConflictStrategy.HIGHER_CONFIDENCE:
            if new.confidence >= existing.confidence:
                existing.superseded_by = new.id
                return new
            else:
                new.superseded_by = existing.id
                return existing
        
        elif self.config.conflict_strategy == ConflictStrategy.KEEP_BOTH:
            new.is_conflicted = True
            new.conflicting_ids.append(existing.id)
            existing.is_conflicted = True
            existing.conflicting_ids.append(new.id)
            return new
        
        elif self.config.conflict_strategy == ConflictStrategy.MERGE:
            # Simple merge: concatenate values
            merged = SharedMemoryEntry(
                id=new.id,
                key=new.key,
                value=f"{existing.value} | {new.value}",
                scope=new.scope,
                group_id=new.group_id,
                author_agent_id=new.author_agent_id,
                confidence=max(existing.confidence, new.confidence),
                category=new.category,
            )
            existing.superseded_by = merged.id
            return merged
        
        return new
    
    def query(
        self,
        agent_id: str,
        key: Optional[str] = None,
        groups: Optional[List[str]] = None,
        category: Optional[str] = None,
        include_global: bool = True,
    ) -> List[SharedMemoryEntry]:
        """
        Query the shared memory pool with scoped access.
        
        An agent can see:
        - All GLOBAL memories
        - GROUP memories for groups they belong to
        
        Args:
            agent_id: Querying agent's ID
            key: Optional specific key to look up
            groups: Override group list (default: agent's registered groups)
            category: Filter by category
            include_global: Include global memories in results
            
        Returns:
            List of matching SharedMemoryEntry objects
        """
        self._total_queries += 1
        results = []
        
        # Determine which groups this agent can see
        visible_groups = set(groups or []) | self._agent_groups.get(agent_id, set())
        
        # Global memories
        if include_global:
            for entry in self._global.values():
                if key and entry.key != key:
                    continue
                if category and entry.category != category:
                    continue
                if entry.superseded_by is None:
                    entry.access_count += 1
                    results.append(entry)
        
        # Group memories
        for gid in visible_groups:
            for entry in self._groups.get(gid, {}).values():
                if key and entry.key != key:
                    continue
                if category and entry.category != category:
                    continue
                if entry.superseded_by is None:
                    entry.access_count += 1
                    results.append(entry)
        
        return results
    
    def get_agent_contributions(self, agent_id: str) -> List[str]:
        """Get all memory IDs published by an agent (provenance)."""
        return self._provenance.get(agent_id, [])
    
    def get_conflicts(self) -> List[Dict[str, Any]]:
        """Get all recorded conflicts."""
        return self._conflicts
    
    def get_group_members(self, group_id: str) -> Set[str]:
        """Get all agents in a group."""
        return self._group_agents.get(group_id, set())
    
    def broadcast(
        self,
        agent_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
    ) -> List[SharedMemoryEntry]:
        """
        Broadcast a memory to all groups the agent belongs to + global.
        
        Useful for important discoveries that should be widely shared.
        """
        entries = []
        
        # Publish globally
        entry = self.publish(agent_id, key, value, MemoryScope.GLOBAL, confidence=confidence)
        entries.append(entry)
        
        # Publish to each group
        for gid in self._agent_groups.get(agent_id, set()):
            entry = self.publish(agent_id, key, value, MemoryScope.GROUP, group_id=gid, confidence=confidence)
            entries.append(entry)
        
        return entries
    
    def get_stale_memories(self) -> List[SharedMemoryEntry]:
        """Get memories that haven't been accessed within staleness threshold."""
        now = datetime.now(timezone.utc)
        threshold = self.config.staleness_threshold_seconds
        stale = []
        
        for entry in self._global.values():
            age = (now - entry.updated_at).total_seconds()
            if age > threshold and entry.access_count == 0:
                stale.append(entry)
        
        for group_memories in self._groups.values():
            for entry in group_memories.values():
                age = (now - entry.updated_at).total_seconds()
                if age > threshold and entry.access_count == 0:
                    stale.append(entry)
        
        return stale
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "global_memories": len([e for e in self._global.values() if not e.superseded_by]),
            "group_count": len(self._groups),
            "total_group_memories": sum(
                len([e for e in g.values() if not e.superseded_by])
                for g in self._groups.values()
            ),
            "total_agents": len(self._agent_groups),
            "total_publishes": self._total_publishes,
            "total_queries": self._total_queries,
            "total_conflicts": self._total_conflicts,
            "conflict_strategy": self.config.conflict_strategy.value,
        }
