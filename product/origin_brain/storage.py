"""
Storage Engine — Persistent Memory Backend

Provides pluggable storage backends for Origin Brain memories.
Supports:
- InMemoryStorage (default, for testing and development)
- SQLiteStorage (lightweight persistence, no external DB needed)
- PostgresStorage (production, with pgvector for embeddings) [future]

This is the bridge from "research prototype" to "production product".

Brain analogue: The physical substrate of memory — synaptic connections
(in-memory) vs consolidated cortical traces (persistent storage).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import EpisodicMemory, MemoryStatus, MemoryType

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_memory(self, memory: EpisodicMemory) -> None:
        """Persist a single episodic memory."""
        ...
    
    @abstractmethod
    def load_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Load a single memory by ID."""
        ...
    
    @abstractmethod
    def load_all_memories(self) -> List[EpisodicMemory]:
        """Load all memories."""
        ...
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        ...
    
    @abstractmethod
    def update_memory(self, memory: EpisodicMemory) -> None:
        """Update an existing memory."""
        ...
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of stored memories."""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Close the storage connection."""
        ...


class InMemoryStorage(StorageBackend):
    """
    In-memory storage backend. Fast, no persistence.
    Used for testing and development.
    """
    
    def __init__(self):
        self._store: Dict[str, EpisodicMemory] = {}
    
    def save_memory(self, memory: EpisodicMemory) -> None:
        self._store[memory.id] = memory
    
    def load_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        return self._store.get(memory_id)
    
    def load_all_memories(self) -> List[EpisodicMemory]:
        return list(self._store.values())
    
    def delete_memory(self, memory_id: str) -> bool:
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False
    
    def update_memory(self, memory: EpisodicMemory) -> None:
        self._store[memory.id] = memory
    
    def count(self) -> int:
        return len(self._store)
    
    def close(self) -> None:
        pass  # Nothing to close


class SQLiteStorage(StorageBackend):
    """
    SQLite-based persistent storage backend.
    
    No external database needed — just a file on disk.
    Perfect for self-hosted deployments and single-agent use cases.
    
    Schema:
    - memories table: all episodic memory fields
    - embeddings are stored as JSON arrays
    - context/metadata stored as JSON
    """
    
    def __init__(self, db_path: str = "origin_brain.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"SQLiteStorage initialized at {db_path}")
    
    def _create_tables(self):
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding TEXT,
                timestamp TEXT NOT NULL,
                context TEXT DEFAULT '{}',
                salience REAL DEFAULT 0.5,
                stability REAL DEFAULT 1.0,
                retrievability REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT NOT NULL,
                memory_type TEXT DEFAULT 'episodic',
                status TEXT DEFAULT 'active',
                associations TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Index on salience and status for efficient filtering
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_salience 
            ON episodic_memories(salience)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status 
            ON episodic_memories(status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON episodic_memories(timestamp)
        """)
        
        self._conn.commit()
    
    def _memory_to_row(self, memory: EpisodicMemory) -> Dict[str, Any]:
        """Convert EpisodicMemory to database row dict."""
        return {
            "id": memory.id,
            "content": memory.content,
            "embedding": json.dumps(memory.embedding) if memory.embedding else None,
            "timestamp": memory.timestamp.isoformat(),
            "context": json.dumps(memory.context),
            "salience": memory.salience,
            "stability": memory.stability,
            "retrievability": memory.retrievability,
            "access_count": memory.access_count,
            "last_accessed": memory.last_accessed.isoformat(),
            "memory_type": memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
            "status": memory.status.value if hasattr(memory.status, 'value') else str(memory.status),
            "associations": json.dumps(memory.associations),
            "metadata": json.dumps(memory.metadata),
        }
    
    def _row_to_memory(self, row: sqlite3.Row) -> EpisodicMemory:
        """Convert database row to EpisodicMemory."""
        return EpisodicMemory(
            id=row["id"],
            content=row["content"],
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            timestamp=datetime.fromisoformat(row["timestamp"]),
            context=json.loads(row["context"]),
            salience=row["salience"],
            stability=row["stability"],
            retrievability=row["retrievability"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            memory_type=MemoryType(row["memory_type"]) if row["memory_type"] else MemoryType.EPISODIC,
            status=MemoryStatus(row["status"]) if row["status"] else MemoryStatus.ACTIVE,
            associations=json.loads(row["associations"]) if row["associations"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
    
    def save_memory(self, memory: EpisodicMemory) -> None:
        """Insert or replace a memory."""
        row = self._memory_to_row(memory)
        self._conn.execute("""
            INSERT OR REPLACE INTO episodic_memories 
            (id, content, embedding, timestamp, context, salience, stability,
             retrievability, access_count, last_accessed, memory_type, status,
             associations, metadata)
            VALUES (:id, :content, :embedding, :timestamp, :context, :salience,
                    :stability, :retrievability, :access_count, :last_accessed,
                    :memory_type, :status, :associations, :metadata)
        """, row)
        self._conn.commit()
    
    def load_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        cursor = self._conn.execute(
            "SELECT * FROM episodic_memories WHERE id = ?", (memory_id,)
        )
        row = cursor.fetchone()
        return self._row_to_memory(row) if row else None
    
    def load_all_memories(self) -> List[EpisodicMemory]:
        cursor = self._conn.execute(
            "SELECT * FROM episodic_memories ORDER BY timestamp ASC"
        )
        return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def delete_memory(self, memory_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM episodic_memories WHERE id = ?", (memory_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0
    
    def update_memory(self, memory: EpisodicMemory) -> None:
        """Update an existing memory (same as save — uses INSERT OR REPLACE)."""
        self.save_memory(memory)
    
    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) FROM episodic_memories")
        return cursor.fetchone()[0]
    
    def search_by_salience(self, min_salience: float = 0.5) -> List[EpisodicMemory]:
        """Find memories with salience above threshold."""
        cursor = self._conn.execute(
            "SELECT * FROM episodic_memories WHERE salience >= ? ORDER BY salience DESC",
            (min_salience,)
        )
        return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def search_by_status(self, status: str = "active") -> List[EpisodicMemory]:
        """Find memories with a given status."""
        cursor = self._conn.execute(
            "SELECT * FROM episodic_memories WHERE status = ?", (status,)
        )
        return [self._row_to_memory(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total = self.count()
        active = self._conn.execute(
            "SELECT COUNT(*) FROM episodic_memories WHERE status = 'active'"
        ).fetchone()[0]
        avg_salience = self._conn.execute(
            "SELECT AVG(salience) FROM episodic_memories"
        ).fetchone()[0] or 0.0
        
        return {
            "backend": "sqlite",
            "db_path": self.db_path,
            "total_memories": total,
            "active_memories": active,
            "average_salience": round(avg_salience, 3),
        }
    
    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
        logger.info(f"SQLiteStorage connection closed ({self.db_path})")
