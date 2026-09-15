"""
Origin Brain - REST API Server
"""
import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .brain import Brain
from .models import BrainConfig, MemoryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Origin Brain API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_brains: Dict[str, Brain] = {}

def get_or_create_brain(agent_id: str) -> Brain:
    if agent_id not in _brains:
        config = BrainConfig(agent_id=agent_id)
        _brains[agent_id] = Brain(config=config)
    return _brains[agent_id]

# --- Request Models ---

class EncodeRequest(BaseModel):
    agent_id: str
    content: str
    context: Optional[dict] = None
    salience: Optional[float] = None

class RecallRequest(BaseModel):
    agent_id: str
    query: str
    top_k: int = 5

class StoreFactRequest(BaseModel):
    agent_id: str
    key: str
    value: str
    confidence: float = 0.8
    category: str = "fact"

class StoreProcedureRequest(BaseModel):
    agent_id: str
    name: str
    description: str
    steps: List[str]
    triggers: Optional[List[str]] = None

class ReminderRequest(BaseModel):
    agent_id: str
    content: str
    trigger_condition: str
    trigger_time: Optional[str] = None

# --- Response Models ---

class EncodeResponse(BaseModel):
    success: bool
    memory_id: Optional[str] = None
    action: str
    emotional_valence: Optional[str] = None
    message: str

class RecallResponse(BaseModel):
    results: List[dict]
    confidence_level: str
    confidence_score: float
    recommendation: str
    triggered_intents: List[dict]

class StatsResponse(BaseModel):
    agent_id: str
    episodic_count: int
    semantic_count: int
    procedural_count: int
    details: dict

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    active_brains: int

# --- Endpoints ---

@app.post("/encode", response_model=EncodeResponse)
async def encode(request: EncodeRequest):
    try:
        brain = get_or_create_brain(request.agent_id)
        result = brain.encode(
            content=request.content,
            context=request.context,
            salience=request.salience,
        )
        
        # Handle fallback between EpisodicMemory and EncodeResult
        memory_id = None
        if hasattr(result, "memory_id"):
            memory_id = result.memory_id
        elif hasattr(result, "id"):
            memory_id = result.id
            
        action = getattr(result, "action", "encoded")
        emotional_valence = getattr(result, "emotional_valence", None)
        message = getattr(result, "message", "Successfully encoded memory")
        
        return EncodeResponse(
            success=True,
            memory_id=memory_id,
            action=action,
            emotional_valence=emotional_valence,
            message=message
        )
    except Exception as e:
        logger.error(f"Error encoding memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/recall", response_model=RecallResponse)
async def recall(request: RecallRequest):
    try:
        brain = get_or_create_brain(request.agent_id)
        results_raw = brain.recall(
            query=request.query,
            top_k=request.top_k
        )
        
        results = []
        if isinstance(results_raw, list):
            # Assume it's a list of MemoryResult
            for r in results_raw:
                mem = r.memory
                mem_type = getattr(mem, "memory_type", r.tier.value if hasattr(r, "tier") else "UNKNOWN")
                results.append({
                    "content": getattr(mem, "content", getattr(mem, "value", str(mem))),
                    "relevance_score": getattr(r, "relevance_score", 1.0),
                    "memory_type": str(mem_type),
                    "tier": str(r.tier.value) if hasattr(r, "tier") else "UNKNOWN"
                })
        elif hasattr(results_raw, "results"):
            # Assume it's a RecallResult object
            for r in getattr(results_raw, "results", []):
                results.append(r if isinstance(r, dict) else (r.model_dump() if hasattr(r, "model_dump") else str(r)))
        
        return RecallResponse(
            results=results,
            confidence_level="high" if len(results) > 0 else "low",
            confidence_score=0.9 if len(results) > 0 else 0.1,
            recommendation="Found relevant memories." if len(results) > 0 else "No relevant memories found.",
            triggered_intents=[]
        )
    except Exception as e:
        logger.error(f"Error recalling memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/consolidate")
async def consolidate(body: dict):
    agent_id = body.get("agent_id")
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")
    try:
        brain = get_or_create_brain(agent_id)
        result = brain.consolidate()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result
    except Exception as e:
        logger.error(f"Error consolidating memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/fact")
async def store_fact(request: StoreFactRequest):
    try:
        brain = get_or_create_brain(request.agent_id)
        result = brain.store_fact(
            key=request.key,
            value=request.value,
            confidence=request.confidence,
            category=request.category
        )
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result
    except Exception as e:
        logger.error(f"Error storing fact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/procedure")
async def store_procedure(request: StoreProcedureRequest):
    try:
        brain = get_or_create_brain(request.agent_id)
        result = brain.store_procedure(
            name=request.name,
            description=request.description,
            steps=request.steps,
            triggers=request.triggers
        )
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result
    except Exception as e:
        logger.error(f"Error storing procedure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/reminder")
async def create_reminder(request: ReminderRequest):
    try:
        # Fallback for reminder since it's not implemented in brain.py yet
        brain = get_or_create_brain(request.agent_id)
        if hasattr(brain, "store_reminder"):
            result = brain.store_reminder(
                content=request.content,
                trigger_condition=request.trigger_condition,
                trigger_time=request.trigger_time
            )
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result
        return {"success": True, "message": "Reminder endpoint hit, but store_reminder not implemented in Brain."}
    except Exception as e:
        logger.error(f"Error storing reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/stats/{agent_id}", response_model=StatsResponse)
async def get_stats(agent_id: str):
    if agent_id not in _brains:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Brain for agent_id {agent_id} not found."
        )
    try:
        brain = _brains[agent_id]
        stats = brain.get_statistics()
        return StatsResponse(
            agent_id=agent_id,
            episodic_count=brain.episodic_count,
            semantic_count=brain.semantic_count,
            procedural_count=brain.procedural_count,
            details=stats
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        version="1.0.0",
        active_brains=len(_brains)
    )

@app.get("/agents", response_model=List[str])
async def list_agents():
    return list(_brains.keys())

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    if agent_id not in _brains:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Brain for agent_id {agent_id} not found."
        )
    try:
        del _brains[agent_id]
        return {"success": True, "message": f"Brain for {agent_id} deleted."}
    except Exception as e:
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
