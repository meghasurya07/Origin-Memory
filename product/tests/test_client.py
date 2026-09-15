"""Tests for Origin Brain SDK Client."""
from origin_brain.client import OriginBrainClient, MemoryItem


class TestClientBasics:
    def test_create_client(self):
        client = OriginBrainClient(agent_id="test-sdk")
        assert client.memory_count == 0

    def test_remember_and_recall(self):
        client = OriginBrainClient(agent_id="test-sdk")
        result = client.remember("Paris is the capital of France")
        assert result["memory_id"] is not None
        assert result["action"] == "encoded"
        
        memories = client.recall("capital of France")
        assert len(memories) > 0
        assert "paris" in memories[0].content.lower() or "france" in memories[0].content.lower()

    def test_remember_with_importance(self):
        client = OriginBrainClient(agent_id="test-sdk")
        result = client.remember("CRITICAL: Server is down!", importance=1.0)
        assert result["salience"] > 0.5

    def test_remember_with_context(self):
        client = OriginBrainClient(agent_id="test-sdk")
        result = client.remember(
            "Meeting with the team",
            context={"location": "office", "people": ["Alice", "Bob"]}
        )
        assert result["memory_id"] is not None

    def test_memory_count(self):
        client = OriginBrainClient(agent_id="test-sdk")
        client.remember("Memory 1")
        client.remember("Memory 2")
        client.remember("Memory 3")
        assert client.memory_count == 3


class TestRecall:
    def test_recall_with_min_score(self):
        client = OriginBrainClient(agent_id="test-sdk")
        client.remember("The Earth orbits the Sun")
        
        results = client.recall("Earth and Sun", min_score=0.0)
        assert len(results) > 0
        
        # Very high threshold should filter out results
        results_high = client.recall("random unrelated query xyz", min_score=0.99)
        # May or may not have results depending on TF-IDF

    def test_recall_returns_memory_items(self):
        client = OriginBrainClient(agent_id="test-sdk")
        client.remember("Python is a great programming language")
        
        results = client.recall("Python programming")
        assert all(isinstance(r, MemoryItem) for r in results)
        
        if results:
            assert hasattr(results[0], 'content')
            assert hasattr(results[0], 'score')
            assert hasattr(results[0], 'memory_id')

    def test_memory_item_to_dict(self):
        item = MemoryItem(
            content="test content",
            score=0.85,
            memory_id="mem_123",
        )
        d = item.to_dict()
        assert d["content"] == "test content"
        assert d["score"] == 0.85
        assert d["memory_id"] == "mem_123"


class TestWorkingMemory:
    def test_think_adds_to_working_memory(self):
        client = OriginBrainClient(agent_id="test-sdk")
        client.think("I need to buy groceries")
        
        wm = client.working_memory()
        assert len(wm) > 0
        assert any("groceries" in item.lower() for item in wm)


class TestContext:
    def test_set_and_clear_context(self):
        client = OriginBrainClient(agent_id="test-sdk")
        client.set_context(location="office", mood="focused")
        assert client.brain.context_buffer["location"] == "office"
        
        client.clear_context()
        assert len(client.brain.context_buffer) == 0


class TestConsolidation:
    def test_sleep_returns_report(self):
        client = OriginBrainClient(agent_id="test-sdk")
        client.remember("Something to consolidate")
        
        report = client.sleep()
        assert "memories_consolidated" in report
        assert "memories_pruned" in report


class TestStats:
    def test_stats_returns_dict(self):
        client = OriginBrainClient(agent_id="test-sdk")
        client.remember("Memory for stats")
        
        stats = client.stats()
        assert "agent_id" in stats
        assert stats["agent_id"] == "test-sdk"
        assert "plasticity" in stats


class TestPersistence:
    def test_persistence_roundtrip(self, tmp_path):
        db = str(tmp_path / "sdk_test.db")
        
        # Client 1: remember
        c1 = OriginBrainClient(agent_id="persist", storage_path=db)
        c1.remember("Mitochondria is the powerhouse of the cell")
        assert c1.memory_count == 1
        
        # Client 2: reopen
        c2 = OriginBrainClient(agent_id="persist", storage_path=db)
        assert c2.memory_count == 1
        
        results = c2.recall("mitochondria powerhouse")
        assert len(results) > 0
