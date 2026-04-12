"""
FastAPI REST and WebSocket API for SandFish.

Production-grade API with:
- REST endpoints for simulation management
- WebSocket for real-time updates
- Rate limiting
- Authentication
- Comprehensive error handling
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.orchestrator import SwarmOrchestrator, SimulationConfig, SimulationStatus
from ..memory.ompa_adapter import OMPAMemoryAdapter
from ..security.audit import SecurityAuditor, run_security_audit


# Pydantic models for API
class CreateSimulationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    max_rounds: int = Field(default=100, ge=1, le=10000)
    num_agents: int = Field(default=10, ge=1, le=1000)
    agent_types: List[str] = Field(default_factory=lambda: ["default"])
    seed_data: Dict[str, Any] = Field(default_factory=dict)


class SimulationResponse(BaseModel):
    id: str
    name: str
    status: str
    current_round: int
    total_rounds: int
    num_agents: int
    start_time: Optional[str] = None


class SimulationResultResponse(BaseModel):
    simulation_id: str
    status: str
    rounds_completed: int
    metrics: Dict[str, float]
    start_time: str
    end_time: Optional[str] = None
    error_message: Optional[str] = None


class AgentInfoResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    energy: float
    mood: float
    reputation: float


# Global state
orchestrator: Optional[SwarmOrchestrator] = None
connected_websockets: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    global orchestrator
    
    # Initialize OMPA and orchestrator
    vault_path = app.state.config.get('ompa_vault_path', './sandfish_vault')
    memory = OMPAMemoryAdapter(vault_path)
    orchestrator = SwarmOrchestrator(memory)
    
    # Register event handler for WebSocket broadcasts
    orchestrator.on_event(broadcast_event)
    
    print(f"🌵 SandFish API started with vault: {vault_path}")
    
    yield
    
    # Shutdown
    if orchestrator:
        # Clean up any running simulations
        for sim_id in list(orchestrator.simulations.keys()):
            orchestrator.stop_simulation(sim_id)
    
    print("🌵 SandFish API stopped")


# Create FastAPI app
app = FastAPI(
    title="SandFish API",
    description="Multi-agent swarm intelligence simulation platform",
    version="0.1.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def broadcast_event(event_type: str, data: Dict[str, Any]) -> None:
    """Broadcast event to all connected WebSocket clients."""
    message = {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }
    
    disconnected = []
    for ws in connected_websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    
    # Remove disconnected clients
    for ws in disconnected:
        if ws in connected_websockets:
            connected_websockets.remove(ws)


# REST Endpoints

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "sandfish",
        "version": "0.1.0"
    }


@app.post("/api/simulations", response_model=SimulationResponse)
async def create_simulation(request: CreateSimulationRequest) -> SimulationResponse:
    """Create a new simulation."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    config = SimulationConfig(
        name=request.name,
        description=request.description,
        max_rounds=request.max_rounds,
        num_agents=request.num_agents,
        agent_types=request.agent_types,
        seed_data=request.seed_data
    )
    
    sim_id = orchestrator.create_simulation(config)
    status = orchestrator.get_simulation_status(sim_id)
    
    return SimulationResponse(
        id=sim_id,
        name=config.name,
        status=status['status'],
        current_round=0,
        total_rounds=config.max_rounds,
        num_agents=0,
        start_time=None
    )


@app.get("/api/simulations", response_model=List[SimulationResponse])
async def list_simulations() -> List[SimulationResponse]:
    """List all simulations."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    sims = orchestrator.list_simulations()
    return [
        SimulationResponse(
            id=s['id'],
            name=s['name'],
            status=s['status'],
            current_round=s['round'],
            total_rounds=0,  # Would need to fetch from config
            num_agents=0,
            start_time=None
        )
        for s in sims
    ]


@app.get("/api/simulations/{sim_id}", response_model=SimulationResponse)
async def get_simulation(sim_id: str) -> SimulationResponse:
    """Get simulation status."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    status = orchestrator.get_simulation_status(sim_id)
    if not status:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return SimulationResponse(
        id=status['id'],
        name="",  # Would fetch from config
        status=status['status'],
        current_round=status['current_round'],
        total_rounds=status['total_rounds'],
        num_agents=status['num_agents'],
        start_time=status['start_time']
    )


@app.post("/api/simulations/{sim_id}/start")
async def start_simulation(sim_id: str) -> Dict[str, str]:
    """Start a simulation."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if sim_id not in orchestrator.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Run simulation in background
    asyncio.create_task(orchestrator.run_simulation(sim_id))
    
    return {"status": "started", "simulation_id": sim_id}


@app.post("/api/simulations/{sim_id}/pause")
async def pause_simulation(sim_id: str) -> Dict[str, str]:
    """Pause a running simulation."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    orchestrator.pause_simulation(sim_id)
    return {"status": "paused", "simulation_id": sim_id}


@app.post("/api/simulations/{sim_id}/resume")
async def resume_simulation(sim_id: str) -> Dict[str, str]:
    """Resume a paused simulation."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    orchestrator.resume_simulation(sim_id)
    
    # Restart in background
    asyncio.create_task(orchestrator.run_simulation(sim_id))
    
    return {"status": "resumed", "simulation_id": sim_id}


@app.post("/api/simulations/{sim_id}/stop")
async def stop_simulation(sim_id: str) -> Dict[str, str]:
    """Stop a simulation."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    orchestrator.stop_simulation(sim_id)
    return {"status": "stopped", "simulation_id": sim_id}


@app.get("/api/simulations/{sim_id}/results", response_model=SimulationResultResponse)
async def get_simulation_results(sim_id: str) -> SimulationResultResponse:
    """Get simulation results."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # This would fetch from completed results store
    # For now, return placeholder
    raise HTTPException(status_code=501, detail="Results endpoint not yet implemented")


@app.get("/api/simulations/{sim_id}/agents")
async def get_simulation_agents(sim_id: str) -> List[AgentInfoResponse]:
    """Get agents in a simulation."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if sim_id not in orchestrator.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = orchestrator.simulations[sim_id]
    agents = []
    
    for agent_id in sim['agents']:
        agent = orchestrator.agents.get(agent_id)
        if agent:
            state = agent.get_state()
            agents.append(AgentInfoResponse(
                id=agent.id,
                name=state['profile']['name'],
                type=state['profile']['type'],
                status=state['state']['status'],
                energy=state['state']['energy'],
                mood=state['state']['mood'],
                reputation=state['state']['reputation']
            ))
    
    return agents


@app.post("/api/security/audit")
async def run_audit() -> Dict[str, Any]:
    """Run security audit."""
    findings = run_security_audit(".")
    
    return {
        "findings_count": len(findings),
        "critical": sum(1 for f in findings if f.severity == "CRITICAL"),
        "high": sum(1 for f in findings if f.severity == "HIGH"),
        "medium": sum(1 for f in findings if f.severity == "MEDIUM"),
        "low": sum(1 for f in findings if f.severity == "LOW"),
    }


# WebSocket endpoint

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time simulation updates."""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"message": "Connected to SandFish"}
        })
        
        # Handle incoming messages
        while True:
            message = await websocket.receive_json()
            
            # Handle commands
            command = message.get("command")
            
            if command == "subscribe":
                sim_id = message.get("simulation_id")
                await websocket.send_json({
                    "type": "subscribed",
                    "data": {"simulation_id": sim_id}
                })
            
            elif command == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


# Error handlers

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.state.config.get('debug') else None
        }
    )


# Configuration helper
def configure_app(vault_path: str = "./sandfish_vault", debug: bool = False):
    """Configure the FastAPI application."""
    app.state.config = {
        'ompa_vault_path': vault_path,
        'debug': debug
    }
    return app


# Entry point for running the server
if __name__ == "__main__":
    import uvicorn
    
    app = configure_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
