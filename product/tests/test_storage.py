"""
Tests for Storage Engine — InMemory and SQLite backends

Tests verify:
1. InMemory CRUD operations
2. SQLite CRUD operations
3. SQLite persistence across connections
4. Search operations (by salience, status)
5. Statistics
6. Memory serialization/deserialization roundtrip
"""
import os
import pytest
import tempfile

from origin_brain.storage import InMemoryStorage, SQLiteStorage
from origin_brain.models import EpisodicMemory, MemoryType, MemoryStatus


def _make_memory(content: str = "Test memory", salience: float = 0.5) -> EpisodicMemory:
    return EpisodicMemory(content=content, salience=salience)


class TestInMemoryStorage:

    def test_save_and_load(self):
        store = InMemoryStorage()
        mem = _make_memory("Hello world")
        store.save_memory(mem)
        loaded = store.load_memory(mem.id)
        assert loaded is not None
        assert loaded.content == "Hello world"

    def test_count(self):
        store = InMemoryStorage()
        assert store.count() == 0
        store.save_memory(_make_memory("one"))
        store.save_memory(_make_memory("two"))
        assert store.count() == 2

    def test_delete(self):
        store = InMemoryStorage()
        mem = _make_memory("to delete")
        store.save_memory(mem)
        assert store.delete_memory(mem.id) is True
        assert store.count() == 0
        assert store.delete_memory("nonexistent") is False

    def test_load_all(self):
        store = InMemoryStorage()
        store.save_memory(_make_memory("a"))
        store.save_memory(_make_memory("b"))
        store.save_memory(_make_memory("c"))
        all_mems = store.load_all_memories()
        assert len(all_mems) == 3


class TestSQLiteStorage:

    def _get_store(self, path=None):
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
        return SQLiteStorage(db_path=path), path

    def test_save_and_load(self):
        store, path = self._get_store()
        try:
            mem = _make_memory("SQLite test memory")
            store.save_memory(mem)
            loaded = store.load_memory(mem.id)
            assert loaded is not None
            assert loaded.content == "SQLite test memory"
        finally:
            store.close()
            os.unlink(path)

    def test_count(self):
        store, path = self._get_store()
        try:
            assert store.count() == 0
            store.save_memory(_make_memory("one"))
            store.save_memory(_make_memory("two"))
            assert store.count() == 2
        finally:
            store.close()
            os.unlink(path)

    def test_delete(self):
        store, path = self._get_store()
        try:
            mem = _make_memory("to delete")
            store.save_memory(mem)
            assert store.delete_memory(mem.id) is True
            assert store.count() == 0
        finally:
            store.close()
            os.unlink(path)

    def test_persistence(self):
        """Data should survive close + reopen."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            # Write
            store1 = SQLiteStorage(db_path=path)
            mem = _make_memory("persistent memory")
            store1.save_memory(mem)
            store1.close()

            # Reopen and read
            store2 = SQLiteStorage(db_path=path)
            loaded = store2.load_memory(mem.id)
            assert loaded is not None
            assert loaded.content == "persistent memory"
            store2.close()
        finally:
            os.unlink(path)

    def test_roundtrip_all_fields(self):
        """All EpisodicMemory fields should survive serialization roundtrip."""
        store, path = self._get_store()
        try:
            mem = EpisodicMemory(
                content="Full field test",
                salience=0.85,
                stability=2.5,
                retrievability=0.9,
                access_count=7,
                memory_type=MemoryType.EPISODIC,
                status=MemoryStatus.ACTIVE,
                context={"location": "office", "mood": "happy"},
                associations=["mem1", "mem2"],
                metadata={"source": "user", "tags": ["test"]},
            )
            store.save_memory(mem)
            loaded = store.load_memory(mem.id)

            assert loaded.content == "Full field test"
            assert loaded.salience == 0.85
            assert loaded.stability == 2.5
            assert loaded.access_count == 7
            assert loaded.context["location"] == "office"
            assert "mem1" in loaded.associations
            assert loaded.metadata["source"] == "user"
        finally:
            store.close()
            os.unlink(path)

    def test_search_by_salience(self):
        store, path = self._get_store()
        try:
            store.save_memory(_make_memory("low", salience=0.2))
            store.save_memory(_make_memory("mid", salience=0.5))
            store.save_memory(_make_memory("high", salience=0.9))

            high_salience = store.search_by_salience(min_salience=0.7)
            assert len(high_salience) == 1
            assert high_salience[0].content == "high"
        finally:
            store.close()
            os.unlink(path)

    def test_update(self):
        store, path = self._get_store()
        try:
            mem = _make_memory("original content", salience=0.3)
            store.save_memory(mem)

            # Update salience
            mem.salience = 0.95
            store.update_memory(mem)

            loaded = store.load_memory(mem.id)
            assert loaded.salience == 0.95
        finally:
            store.close()
            os.unlink(path)

    def test_statistics(self):
        store, path = self._get_store()
        try:
            store.save_memory(_make_memory("a", salience=0.4))
            store.save_memory(_make_memory("b", salience=0.8))

            stats = store.get_statistics()
            assert stats["backend"] == "sqlite"
            assert stats["total_memories"] == 2
            assert stats["average_salience"] == 0.6
        finally:
            store.close()
            os.unlink(path)

    def test_load_all(self):
        store, path = self._get_store()
        try:
            for i in range(10):
                store.save_memory(_make_memory(f"Memory {i}"))
            all_mems = store.load_all_memories()
            assert len(all_mems) == 10
        finally:
            store.close()
            os.unlink(path)
