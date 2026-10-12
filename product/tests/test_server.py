"""Tests for the Origin Brain REST API Server v0.7."""
import pytest
from fastapi.testclient import TestClient

from origin_brain.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthAndAgents:
    def test_health(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.7.0"
        assert data["hcsi"] == 1.0

    def test_list_agents_empty(self, client):
        resp = client.get("/v1/agents")
        assert resp.status_code == 200


class TestEncodeRecall:
    def test_encode(self, client):
        resp = client.post("/v1/encode", json={
            "agent_id": "test_agent",
            "content": "Meeting with Bob about Q3 budget",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["memory_id"] is not None

    def test_recall(self, client):
        # Encode first
        client.post("/v1/encode", json={
            "agent_id": "recall_agent",
            "content": "The Krebs cycle produces ATP in mitochondria",
        })
        # Recall
        resp = client.post("/v1/recall", json={
            "agent_id": "recall_agent",
            "query": "ATP energy production",
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] >= 1
        assert "confidence" in data

    def test_reconstruct(self, client):
        client.post("/v1/encode", json={
            "agent_id": "recon_agent",
            "content": "Alice presented revenue numbers at the board meeting",
        })
        resp = client.post("/v1/reconstruct", json={
            "agent_id": "recon_agent",
            "query": "board meeting",
            "emotional_arousal": 0.8,
            "emotional_valence": "positive",
        })
        assert resp.status_code == 200


class TestSleepAndStats:
    def test_sleep(self, client):
        client.post("/v1/encode", json={
            "agent_id": "sleep_agent",
            "content": "Test memory for sleep consolidation",
        })
        resp = client.post("/v1/sleep", json={"agent_id": "sleep_agent"})
        assert resp.status_code == 200
        data = resp.json()
        assert "memories_consolidated" in data

    def test_stats(self, client):
        client.post("/v1/encode", json={
            "agent_id": "stats_agent",
            "content": "Memory for stats test",
        })
        resp = client.get("/v1/stats/stats_agent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "stats_agent"

    def test_stats_404(self, client):
        resp = client.get("/v1/stats/nonexistent_xyz")
        assert resp.status_code == 404


class TestSharedMemory:
    def test_publish_global(self, client):
        resp = client.post("/v1/shared/publish", json={
            "agent_id": "agent_1",
            "key": "test_key",
            "value": "test_value",
            "scope": "global",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["scope"] == "global"

    def test_query_shared(self, client):
        client.post("/v1/shared/publish", json={
            "agent_id": "agent_1",
            "key": "query_key",
            "value": "query_value",
            "scope": "global",
        })
        resp = client.post("/v1/shared/query", json={
            "agent_id": "agent_2",
            "key": "query_key",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_join_group(self, client):
        resp = client.post("/v1/shared/join", json={
            "agent_id": "agent_1",
            "group_id": "team_alpha",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_shared_stats(self, client):
        resp = client.get("/v1/shared/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "global_memories" in data
        assert "total_publishes" in data


class TestSemanticAndSchema:
    def test_semantic_query(self, client):
        # First encode and sleep to build neocortex
        for _ in range(3):
            client.post("/v1/encode", json={
                "agent_id": "sem_agent",
                "content": "Meeting with Bob about budget planning",
            })
        client.post("/v1/sleep", json={"agent_id": "sem_agent"})
        
        resp = client.post("/v1/semantic/query", json={
            "agent_id": "sem_agent",
            "concept": "budget",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["concept"] == "budget"
        assert "associations" in data

    def test_schema_404(self, client):
        resp = client.get("/v1/schema/nonexistent_abc/meeting")
        assert resp.status_code == 404


class TestLegacyEndpoints:
    def test_legacy_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_legacy_encode(self, client):
        resp = client.post("/encode", json={
            "agent_id": "legacy_agent",
            "content": "Legacy test",
        })
        assert resp.status_code == 200
