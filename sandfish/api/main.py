"""
FastAPI REST and WebSocket API for SandFish.

Endpoints:
- /health                            (no auth)
- POST /api/simulations              create
- GET  /api/simulations              list
- GET  /api/simulations/{id}         status
- POST /api/simulations/{id}/start   start (idempotent: 409 if already running)
- POST /api/simulations/{id}/pause   request a pause
- POST /api/simulations/{id}/resume  resume after a pause
- POST /api/simulations/{id}/stop    request a stop
- GET  /api/simulations/{id}/agents  list agent state
- GET  /api/simulations/{id}/results latest results
- POST /api/security/audit           run security auditor (auth required)
- WS   /ws                           real-time event stream

Auth:
- If SANDFISH_API_KEY is set, all /api/* endpoints require an
  `X-API-Key` header that matches. /health and /ws are unauthenticated.

Rate limiting:
- Per-client (X-API-Key or remote IP) sliding window.
- Configurable via SANDFISH_RATE_LIMIT_PER_MINUTE (default: 120).
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.orchestrator import (
    SimulationConfig,
    SwarmOrchestrator,
)
from ..memory.ompa_adapter import OMPAMemoryAdapter
from ..security.audit import run_security_audit


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- Pydantic schemas ----------


class CreateSimulationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    max_rounds: int = Field(default=100, ge=1, le=100_000)
    num_agents: int = Field(default=10, ge=1, le=10_000)
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
    end_time: Optional[str] = None
    error: Optional[str] = None


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


# ---------- Module-level state ----------


orchestrator: Optional[SwarmOrchestrator] = None
_websocket_clients: List[WebSocket] = []
_websocket_lock = asyncio.Lock()


# ---------- Auth + rate limiting ----------


def _api_key_setting() -> Optional[str]:
    """The configured API key (if any). None disables auth."""
    key = os.environ.get("SANDFISH_API_KEY")
    return key.strip() if key and key.strip() else None


async def require_api_key(request: Request) -> str:
    """FastAPI dependency that enforces SANDFISH_API_KEY when configured."""
    expected = _api_key_setting()
    if expected is None:
        return "anonymous"

    provided = request.headers.get("x-api-key")
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return provided


class _SlidingWindowLimiter:
    """In-process per-key sliding-window limiter."""

    def __init__(self, max_per_minute: int):
        self.max = max_per_minute
        self.window_seconds = 60.0
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> None:
        now = time.monotonic()
        async with self._lock:
            bucket = self._hits[key]
            cutoff = now - self.window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max:
                retry = max(0.0, self.window_seconds - (now - bucket[0]))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Retry in {retry:.1f}s.",
                )
            bucket.append(now)


_limiter = _SlidingWindowLimiter(
    max_per_minute=int(os.environ.get("SANDFISH_RATE_LIMIT_PER_MINUTE", "120"))
)


async def rate_limit(request: Request, _api_key: str = Depends(require_api_key)) -> None:
    """FastAPI dependency: applies rate limit after auth."""
    key = _api_key if _api_key != "anonymous" else (request.client.host if request.client else "unknown")
    await _limiter.check(key)


# ---------- App lifespan ----------


def _resolve_vault_path(app: FastAPI) -> str:
    cfg = getattr(app.state, "config", None) or {}
    return (
        cfg.get("ompa_vault_path")
        or os.environ.get("OMPA_VAULT_PATH")
        or "./sandfish_vault"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator

    vault_path = _resolve_vault_path(app)
    memory = OMPAMemoryAdapter(vault_path)
    orchestrator = SwarmOrchestrator(memory)
    orchestrator.on_event(_broadcast_event)

    # Track in-flight simulation tasks so we can cancel cleanly on shutdown.
    app.state.simulation_tasks: Dict[str, asyncio.Task] = {}

    yield

    # Shutdown: stop everything gracefully.
    if orchestrator is not None:
        for sim_id in list(orchestrator.simulations):
            orchestrator.stop_simulation(sim_id)
        for task in list(app.state.simulation_tasks.values()):
            task.cancel()
        await asyncio.gather(
            *app.state.simulation_tasks.values(), return_exceptions=True
        )

    async with _websocket_lock:
        for ws in list(_websocket_clients):
            try:
                await ws.close()
            except Exception:
                pass
        _websocket_clients.clear()


# ---------- App ----------


app = FastAPI(
    title="SandFish API",
    description="Multi-agent swarm intelligence simulation platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


def _allowed_origins() -> List[str]:
    raw = os.environ.get("SANDFISH_CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


_origins = _allowed_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
else:
    # Safe default: allow any origin but disable credentials so browsers
    # don't accidentally expose cookies cross-origin. Tighten via
    # SANDFISH_CORS_ORIGINS for production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


# ---------- Event broadcast ----------


async def _broadcast_event(event_type: str, data: Dict[str, Any]) -> None:
    """Broadcast an event to all connected websocket clients."""
    message = {
        "type": event_type,
        "timestamp": _utcnow_iso(),
        "data": data,
    }

    async with _websocket_lock:
        clients = list(_websocket_clients)

    dropped: List[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dropped.append(ws)

    if dropped:
        async with _websocket_lock:
            for ws in dropped:
                if ws in _websocket_clients:
                    _websocket_clients.remove(ws)


# ---------- Helpers ----------


def _require_orchestrator() -> SwarmOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


def _status_to_response(orch: SwarmOrchestrator, sim_id: str) -> SimulationResponse:
    status_dict = orch.get_simulation_status(sim_id)
    if status_dict is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return SimulationResponse(**status_dict)


# ---------- Public endpoints ----------


@app.get("/health", tags=["meta"])
async def health_check() -> Dict[str, str]:
    """Health probe (no auth)."""
    return {"status": "healthy", "service": "sandfish", "version": app.version}


# ---------- Simulation endpoints ----------


@app.post(
    "/api/simulations",
    response_model=SimulationResponse,
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def create_simulation(request: CreateSimulationRequest) -> SimulationResponse:
    """Create a new simulation."""
    orch = _require_orchestrator()
    config = SimulationConfig(
        name=request.name,
        description=request.description,
        max_rounds=request.max_rounds,
        num_agents=request.num_agents,
        agent_types=request.agent_types,
        seed_data=request.seed_data,
    )
    sim_id = orch.create_simulation(config)
    return _status_to_response(orch, sim_id)


@app.get(
    "/api/simulations",
    response_model=List[SimulationResponse],
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def list_simulations() -> List[SimulationResponse]:
    """List all known simulations."""
    orch = _require_orchestrator()
    return [_status_to_response(orch, summary["id"]) for summary in orch.list_simulations()]


@app.get(
    "/api/simulations/{sim_id}",
    response_model=SimulationResponse,
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def get_simulation(sim_id: str) -> SimulationResponse:
    """Get a single simulation's status."""
    return _status_to_response(_require_orchestrator(), sim_id)


@app.post(
    "/api/simulations/{sim_id}/start",
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def start_simulation(sim_id: str, request: Request) -> Dict[str, str]:
    """Start a simulation. Returns 409 if already running."""
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    tasks: Dict[str, asyncio.Task] = request.app.state.simulation_tasks
    existing = tasks.get(sim_id)
    if existing and not existing.done():
        raise HTTPException(status_code=409, detail="Simulation is already running")

    task = asyncio.create_task(orch.run_simulation(sim_id))
    tasks[sim_id] = task

    def _on_done(t: asyncio.Task) -> None:
        # Surface failures via logs; preserve task object for inspection.
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            # Logged at orchestrator level too; don't crash here.
            pass

    task.add_done_callback(_on_done)
    return {"status": "started", "simulation_id": sim_id}


@app.post(
    "/api/simulations/{sim_id}/pause",
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def pause_simulation(sim_id: str) -> Dict[str, str]:
    """Request a pause; takes effect after the in-flight round completes."""
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if not orch.pause_simulation(sim_id):
        raise HTTPException(status_code=409, detail="Simulation is not running")
    return {"status": "paused", "simulation_id": sim_id}


@app.post(
    "/api/simulations/{sim_id}/resume",
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def resume_simulation(sim_id: str, request: Request) -> Dict[str, str]:
    """Resume a paused simulation by spawning a new run loop."""
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if not orch.resume_simulation(sim_id):
        raise HTTPException(status_code=409, detail="Simulation is not paused")

    tasks: Dict[str, asyncio.Task] = request.app.state.simulation_tasks
    task = asyncio.create_task(orch.run_simulation(sim_id))
    tasks[sim_id] = task
    return {"status": "resumed", "simulation_id": sim_id}


@app.post(
    "/api/simulations/{sim_id}/stop",
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def stop_simulation(sim_id: str) -> Dict[str, str]:
    """Request a stop; takes effect after the in-flight round completes."""
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if not orch.stop_simulation(sim_id):
        raise HTTPException(status_code=409, detail="Simulation is not running")
    return {"status": "stopping", "simulation_id": sim_id}


@app.get(
    "/api/simulations/{sim_id}/agents",
    response_model=List[AgentInfoResponse],
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def get_simulation_agents(sim_id: str) -> List[AgentInfoResponse]:
    """List the agents in a simulation with their current state."""
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    sim = orch.simulations[sim_id]
    agents: List[AgentInfoResponse] = []
    for agent_id in sim["agents"]:
        agent = orch.agents.get(agent_id)
        if agent is None:
            continue
        snapshot = agent.get_state()
        agents.append(
            AgentInfoResponse(
                id=agent.id,
                name=snapshot["profile"]["name"],
                type=snapshot["profile"]["type"],
                status=snapshot["state"]["status"],
                energy=snapshot["state"]["energy"],
                mood=snapshot["state"]["mood"],
                reputation=snapshot["state"]["reputation"],
            )
        )
    return agents


@app.get(
    "/api/simulations/{sim_id}/results",
    response_model=SimulationResultResponse,
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def get_simulation_results(sim_id: str) -> SimulationResultResponse:
    """Return the latest result snapshot for a simulation."""
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    sim = orch.simulations[sim_id]
    metrics = orch._calculate_metrics(sim)  # internal but stable
    return SimulationResultResponse(
        simulation_id=sim_id,
        status=sim["status"].value,
        rounds_completed=sim["current_round"],
        metrics=metrics,
        start_time=(sim["start_time"] or datetime.now(timezone.utc)).isoformat(),
        end_time=sim["end_time"].isoformat() if sim["end_time"] else None,
        error_message=sim["error"],
    )


@app.post(
    "/api/security/audit",
    dependencies=[Depends(rate_limit)],
    tags=["security"],
)
async def run_audit() -> Dict[str, Any]:
    """Run the bundled security auditor against the working directory."""
    findings = run_security_audit(".")
    return {
        "findings_count": len(findings),
        "critical": sum(1 for f in findings if f.severity == "CRITICAL"),
        "high": sum(1 for f in findings if f.severity == "HIGH"),
        "medium": sum(1 for f in findings if f.severity == "MEDIUM"),
        "low": sum(1 for f in findings if f.severity == "LOW"),
    }


# ---------- WebSocket ----------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Real-time event stream. Optional API-key auth via query param `api_key`."""
    expected_key = _api_key_setting()
    if expected_key is not None:
        provided = websocket.query_params.get("api_key", "")
        if not secrets.compare_digest(provided, expected_key):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

    await websocket.accept()
    async with _websocket_lock:
        _websocket_clients.append(websocket)

    try:
        await websocket.send_json(
            {
                "type": "connected",
                "timestamp": _utcnow_iso(),
                "data": {"message": "Connected to SandFish"},
            }
        )

        while True:
            message = await websocket.receive_json()
            command = message.get("command")

            if command == "subscribe":
                await websocket.send_json(
                    {
                        "type": "subscribed",
                        "timestamp": _utcnow_iso(),
                        "data": {"simulation_id": message.get("simulation_id")},
                    }
                )
            elif command == "ping":
                await websocket.send_json(
                    {"type": "pong", "timestamp": _utcnow_iso()}
                )
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "timestamp": _utcnow_iso(),
                        "data": {"message": f"Unknown command: {command!r}"},
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception:
        # Don't crash the server on a single-client error.
        pass
    finally:
        async with _websocket_lock:
            if websocket in _websocket_clients:
                _websocket_clients.remove(websocket)


# ---------- Error handler ----------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    cfg = getattr(request.app.state, "config", None) or {}
    payload: Dict[str, Any] = {"error": "Internal server error"}
    if cfg.get("debug"):
        payload["detail"] = repr(exc)
    return JSONResponse(status_code=500, content=payload)


# ---------- Configuration helpers ----------


def configure_app(vault_path: str = "./sandfish_vault", debug: bool = False) -> FastAPI:
    """Configure the FastAPI app before launching it."""
    app.state.config = {"ompa_vault_path": vault_path, "debug": debug}
    return app


if __name__ == "__main__":
    import uvicorn

    configure_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
