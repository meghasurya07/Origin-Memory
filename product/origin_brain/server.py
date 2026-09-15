"""
Origin Brain — REST API Server v2

Production-ready API for the Origin Brain memory infrastructure.

Endpoints:
  POST /v1/encode         — Store a new memory
  POST /v1/recall         — Recall memories
  POST /v1/reconstruct    — Reconstructive recall (context-dependent)
  POST /v1/sleep          — Run sleep consolidation
  POST /v1/save           — Persist memories to storage
  GET  /v1/stats/{id}     — Get brain statistics
  GET  /v1/health         — Health check
  GET  /v1/agents         — List active agents
  DELETE /v1/agents/{id}  — Delete an agent's brain

Run:
  uvicorn origin_brain.server:app --host 0.0.0.0 --port 8000
"""
import logging
import os
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .brain import Brain
from .models import BrainConfig, MemoryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Origin Brain API",
    description="Human-Like Memory Infrastructure for AI Agents — by Origin AI",
    version="0.5.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Brain Registry ──────────────────────────────────────────────────

_brains: Dict[str, Brain] = {}
_storage_dir = os.environ.get("ORIGIN_BRAIN_STORAGE", "./brain_data")

def get_or_create_brain(agent_id: str, storage_path: Optional[str] = None) -> Brain:
    """Get existing brain or create a new one with optional persistence."""
    if agent_id not in _brains:
        db_path = storage_path or os.path.join(_storage_dir, f"{agent_id}.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        config = BrainConfig(agent_id=agent_id, storage_path=db_path)
        _brains[agent_id] = Brain(config=config)
        logger.info(f"Created brain for agent: {agent_id} (storage: {db_path})")
    return _brains[agent_id]

# ── Request/Response Models ─────────────────────────────────────────

class EncodeRequest(BaseModel):
    agent_id: str
    content: str
    context: Optional[Dict[str, Any]] = None
    salience: Optional[float] = None
    memory_type: str = "EPISODIC"

class RecallRequest(BaseModel):
    agent_id: str
    query: str
    top_k: int = 5

class ReconstructRequest(BaseModel):
    agent_id: str
    query: str
    top_k: int = 5
    emotional_arousal: float = 0.5
    emotional_valence: str = "neutral"

class SleepRequest(BaseModel):
    agent_id: str

class SaveRequest(BaseModel):
    agent_id: str

class MemoryResponse(BaseModel):
    content: str
    relevance_score: float
    memory_id: str = ""
    tier: str = "EPISODIC"

class EncodeResponse(BaseModel):
    success: bool
    memory_id: Optional[str] = None
    action: str
    salience: float = 0.0
    message: str = "Memory encoded"

class RecallResponse(BaseModel):
    results: List[MemoryResponse]
    confidence: float
    query: str
    total_results: int

class ReconstructResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str

class SleepResponse(BaseModel):
    memories_consolidated: int
    memories_pruned: int
    agent_id: str

class StatsResponse(BaseModel):
    agent_id: str
    episodic_count: int
    semantic_count: int
    procedural_count: int
    plasticity: Dict[str, Any] = {}
    details: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.5.1"
    active_brains: int
    hcsi: float = 1.00

# ── API Endpoints ───────────────────────────────────────────────────

@app.post("/v1/encode", response_model=EncodeResponse)
async def encode(request: EncodeRequest):
    """Store a new memory in the brain."""
    try:
        brain = get_or_create_brain(request.agent_id)
        mem_type = MemoryType(request.memory_type.upper())
        
        result = brain.encode(
            content=request.content,
            context=request.context,
            salience=request.salience,
            memory_type=mem_type,
        )
        
        memory_id = result.memory.id if result.memory else None
        
        return EncodeResponse(
            success=True,
            memory_id=memory_id,
            action=result.action,
            salience=result.memory.salience if result.memory else 0,
            message=f"Memory encoded with action: {result.action}",
        )
    except Exception as e:
        logger.error(f"Encode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/recall", response_model=RecallResponse)
async def recall(request: RecallRequest):
    """Recall memories matching a query."""
    try:
        brain = get_or_create_brain(request.agent_id)
        recall_result = brain.recall(query=request.query, top_k=request.top_k)
        
        results = []
        for r in recall_result.results:
            content = getattr(r.memory, "content", getattr(r.memory, "value", str(r.memory)))
            mem_id = getattr(r.memory, "id", "")
            tier = r.tier.value if hasattr(r, "tier") else "EPISODIC"
            
            results.append(MemoryResponse(
                content=content,
                relevance_score=r.relevance_score,
                memory_id=mem_id,
                tier=tier,
            ))
        
        return RecallResponse(
            results=results,
            confidence=recall_result.confidence,
            query=request.query,
            total_results=len(results),
        )
    except Exception as e:
        logger.error(f"Recall error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/reconstruct", response_model=ReconstructResponse)
async def reconstruct(request: ReconstructRequest):
    """Reconstructive recall — memory content changes based on current state."""
    try:
        brain = get_or_create_brain(request.agent_id)
        results = brain.reconstruct_recall(
            query=request.query,
            top_k=request.top_k,
            emotional_arousal=request.emotional_arousal,
            emotional_valence=request.emotional_valence,
        )
        
        reconstructed = []
        for r in results:
            reconstructed.append({
                "reconstructed_content": r.reconstructed_content,
                "original_memory_id": r.original_memory_id,
                "confidence": r.confidence,
                "fidelity": r.fidelity.value if hasattr(r.fidelity, "value") else str(r.fidelity),
            })
        
        return ReconstructResponse(results=reconstructed, query=request.query)
    except Exception as e:
        logger.error(f"Reconstruct error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/sleep", response_model=SleepResponse)
async def sleep(request: SleepRequest):
    """Run sleep consolidation cycle."""
    try:
        brain = get_or_create_brain(request.agent_id)
        report = brain.sleep()
        
        return SleepResponse(
            memories_consolidated=report.total_consolidated,
            memories_pruned=report.total_evicted,
            agent_id=request.agent_id,
        )
    except Exception as e:
        logger.error(f"Sleep error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/save")
async def save(request: SaveRequest):
    """Explicitly persist all memories to storage."""
    try:
        brain = get_or_create_brain(request.agent_id)
        count = brain.save()
        return {"success": True, "memories_saved": count, "agent_id": request.agent_id}
    except Exception as e:
        logger.error(f"Save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/stats/{agent_id}", response_model=StatsResponse)
async def get_stats(agent_id: str):
    """Get brain statistics for an agent."""
    if agent_id not in _brains:
        raise HTTPException(status_code=404, detail=f"Brain for {agent_id} not found")
    
    brain = _brains[agent_id]
    stats = brain.get_statistics()
    plasticity_stats = brain.plasticity.get_statistics()
    
    return StatsResponse(
        agent_id=agent_id,
        episodic_count=brain.episodic_count,
        semantic_count=brain.semantic_count,
        procedural_count=brain.procedural_count,
        plasticity=plasticity_stats,
        details=stats,
    )


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(active_brains=len(_brains))


@app.get("/v1/agents")
async def list_agents():
    """List all active agent brains."""
    agents = []
    for agent_id, brain in _brains.items():
        agents.append({
            "agent_id": agent_id,
            "episodic_count": brain.episodic_count,
        })
    return {"agents": agents, "total": len(agents)}


@app.delete("/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent's brain."""
    if agent_id not in _brains:
        raise HTTPException(status_code=404, detail=f"Brain for {agent_id} not found")
    del _brains[agent_id]
    return {"success": True, "message": f"Brain for {agent_id} deleted"}


# ── Legacy endpoints (backward compatible) ──────────────────────────

@app.post("/encode", include_in_schema=False)
async def encode_legacy(request: EncodeRequest):
    return await encode(request)

@app.post("/recall", include_in_schema=False)
async def recall_legacy(request: RecallRequest):
    return await recall(request)

@app.get("/health", include_in_schema=False)
async def health_legacy():
    return await health()

@app.get("/stats/{agent_id}", include_in_schema=False)
async def stats_legacy(agent_id: str):
    return await get_stats(agent_id)


# ── Entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
