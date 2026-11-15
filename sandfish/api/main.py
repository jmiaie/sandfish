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
- POST /api/security/audit           run the security auditor on the
                                     installed sandfish package (auth required)
- WS   /ws                           real-time event stream

Auth:
- If SANDFISH_API_KEY is set, all /api/* endpoints require an `X-API-Key`
  header that matches, and /ws requires the key (see websocket_endpoint).
  /health is always open.
- `sandfish api` binds 127.0.0.1 by default and refuses a non-loopback bind
  unless SANDFISH_API_KEY is set.
- Failed auth attempts are rate-limited per client IP
  (SANDFISH_AUTH_FAILURES_PER_MINUTE, default 10) before the key is checked.

Rate limiting:
- Per-client (X-API-Key or remote IP) sliding window.
- Configurable via SANDFISH_RATE_LIMIT_PER_MINUTE (default: 120).

Resource limits (0 = unlimited): SANDFISH_MAX_AGENTS (1000),
SANDFISH_MAX_ROUNDS (10000), SANDFISH_MAX_CONCURRENT_SIMULATIONS (4),
SANDFISH_MAX_FINISHED_SIMULATIONS (100, oldest finished are evicted).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
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
    SimulationLimitError,
    SimulationLimits,
    SimulationStatus,
    SwarmOrchestrator,
)
from ..memory.ompa_adapter import OMPAMemoryAdapter
from ..security.audit import run_security_audit

logger = logging.getLogger("sandfish.api")

# Seconds a WebSocket client has to send {"command": "auth", ...} when it
# didn't present the key in a header, and how many such not-yet-authenticated
# sockets may wait at once.
WS_AUTH_TIMEOUT_S = 10.0
WS_MAX_PENDING_AUTH = 32
_ws_pending_auth = 0


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
    start_date: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Simulated date of round 0 (YYYY-MM-DD); dates memory writes.",
    )
    days_per_round: int = Field(default=1, ge=0, le=365)
    seed: Optional[int] = Field(
        default=None,
        description="RNG seed; runs with the same seed and config are reproducible",
    )


class SimulationResponse(BaseModel):
    id: str
    name: str
    status: str
    current_round: int
    total_rounds: int
    num_agents: int
    rounds_completed: int = 0
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


def _keys_match(provided: Optional[str], expected: str) -> bool:
    """Constant-time compare that never raises.

    secrets.compare_digest raises TypeError for non-ASCII *str* input, which
    surfaced as a 500; comparing bytes handles any value. Starlette decodes
    header bytes as latin-1, so a UTF-8 key sent in a header is recovered by
    re-encoding as latin-1; keys from JSON (WebSocket auth message) are real
    text and compare as UTF-8.
    """
    if not provided:
        return False
    want = expected.encode("utf-8")
    candidates = [provided.encode("utf-8", "surrogatepass")]
    try:
        candidates.append(provided.encode("latin-1"))
    except UnicodeEncodeError:
        pass
    matched = False
    for candidate in candidates:
        matched |= secrets.compare_digest(candidate, want)
    return matched


def is_loopback_host(host: str) -> bool:
    """True for localhost / 127.0.0.0/8 / ::1 binds."""
    host = (host or "").strip().strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind_allowed(host: str) -> None:
    """Refuse to serve on a non-loopback address without an API key."""
    if not is_loopback_host(host) and _api_key_setting() is None:
        raise RuntimeError(
            f"Refusing to bind {host!r} without SANDFISH_API_KEY: the API would be "
            "reachable unauthenticated from the network. Set SANDFISH_API_KEY "
            "(e.g. `export SANDFISH_API_KEY=$(openssl rand -hex 32)`) or bind 127.0.0.1."
        )


def _client_ip(conn: Any) -> str:
    client = getattr(conn, "client", None)
    return client.host if client else "unknown"


class _SlidingWindowLimiter:
    """In-process per-key sliding-window limiter."""

    def __init__(self, max_per_minute: int):
        self.max = max_per_minute
        self.window_seconds = 60.0
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._last_sweep = float("-inf")

    def _trim(self, key: str, now: float) -> Deque[float]:
        bucket = self._hits[key]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        return bucket

    max_keys = 10_000

    def _evict_idle(self, now: float) -> None:
        # Keep memory bounded when many distinct clients/keys show up: drop
        # idle buckets at most once per window, then (if a flood of fresh
        # keys is still over the cap) the oldest ones. dicts keep insertion
        # order, so the first keys are the oldest.
        if len(self._hits) <= self.max_keys:
            return
        if now - self._last_sweep >= self.window_seconds:
            self._last_sweep = now
            cutoff = now - self.window_seconds
            for k in [k for k, b in self._hits.items() if not b or b[-1] < cutoff]:
                del self._hits[k]
        while len(self._hits) > self.max_keys:
            del self._hits[next(iter(self._hits))]

    async def check(self, key: str) -> None:
        now = time.monotonic()
        async with self._lock:
            bucket = self._trim(key, now)
            if len(bucket) >= self.max:
                retry = max(0.0, self.window_seconds - (now - bucket[0]))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Retry in {retry:.1f}s.",
                )
            bucket.append(now)
            self._evict_idle(now)

    async def blocked(self, key: str) -> Optional[float]:
        """Seconds until `key` may try again, or None if under the limit."""
        now = time.monotonic()
        async with self._lock:
            bucket = self._trim(key, now)
            if len(bucket) >= self.max:
                return max(0.0, self.window_seconds - (now - bucket[0]))
            return None

    async def record(self, key: str) -> None:
        now = time.monotonic()
        async with self._lock:
            self._trim(key, now).append(now)
            self._evict_idle(now)


_limiter = _SlidingWindowLimiter(
    max_per_minute=int(os.environ.get("SANDFISH_RATE_LIMIT_PER_MINUTE", "120"))
)
# Failed authentications per client IP. Checked *before* the key compare, so
# key guessing is throttled even though every guess is a distinct "client".
_auth_fail_limiter = _SlidingWindowLimiter(
    max_per_minute=int(os.environ.get("SANDFISH_AUTH_FAILURES_PER_MINUTE", "10"))
)


async def _authenticate(conn: Any, provided: Optional[str]) -> bool:
    """Check `provided` against the configured key with failure throttling.

    Returns True when auth is off or the key matches. Raises HTTPException
    429 while the client IP is over its failed-attempt budget.
    """
    expected = _api_key_setting()
    if expected is None:
        return True
    ip = _client_ip(conn)
    retry = await _auth_fail_limiter.blocked(ip)
    if retry is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed authentication attempts. Retry in {retry:.1f}s.",
        )
    if _keys_match(provided, expected):
        return True
    await _auth_fail_limiter.record(ip)
    return False


async def require_api_key(request: Request) -> str:
    """FastAPI dependency that enforces SANDFISH_API_KEY when configured."""
    if _api_key_setting() is None:
        return "anonymous"
    provided = request.headers.get("x-api-key")
    if not await _authenticate(request, provided):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return provided or ""


async def rate_limit(request: Request, _api_key: str = Depends(require_api_key)) -> None:
    """FastAPI dependency: applies rate limit after auth."""
    key = _api_key if _api_key != "anonymous" else _client_ip(request)
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
    orchestrator = SwarmOrchestrator(memory, limits=SimulationLimits.from_env())
    orchestrator.on_event(_broadcast_event)

    # Track in-flight simulation tasks so we can cancel cleanly on shutdown.
    app.state.simulation_tasks = {}  # Dict[str, asyncio.Task]
    app.state.audit_lock = asyncio.Lock()

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
        await asyncio.to_thread(orchestrator.close)

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


def _on_run_done(sim_id: str, task: "asyncio.Task[Any]") -> None:
    """Log background run failures instead of dropping them silently."""
    if task.cancelled():
        logger.info("Simulation %s run task cancelled", sim_id)
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Simulation %s background run failed: %s",
            sim_id,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return
    result = task.result()
    if getattr(result, "error_message", None):
        logger.error(
            "Simulation %s ended %s: %s",
            sim_id,
            result.status.value,
            result.error_message,
        )


def _spawn_run(request: Request, orch: SwarmOrchestrator, sim_id: str) -> None:
    """Claim a run slot synchronously, then run in a background task."""
    tasks: Dict[str, asyncio.Task] = request.app.state.simulation_tasks
    # Forget finished tasks (and evicted simulations) so the map stays bounded.
    for sid in [sid for sid, t in tasks.items() if t.done()]:
        del tasks[sid]
    existing = tasks.get(sim_id)
    if existing is not None and not existing.done():
        raise HTTPException(status_code=409, detail="Simulation is already running")
    try:
        orch.reserve_start(sim_id)
    except SimulationLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    task = asyncio.create_task(orch.run_simulation(sim_id))
    tasks[sim_id] = task
    task.add_done_callback(lambda t: _on_run_done(sim_id, t))


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
        start_date=request.start_date,
        days_per_round=request.days_per_round,
        seed=request.seed,
    )
    try:
        sim_id = orch.create_simulation(config)
    except SimulationLimitError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    """Start a pending simulation.

    409 if it isn't pending (running, pausing, paused -> use /resume, or
    finished); 429 if the concurrent-simulation cap is reached.
    """
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    _spawn_run(request, orch, sim_id)
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
    # The in-flight round finishes first; poll status until "paused".
    return {"status": "pausing", "simulation_id": sim_id}


@app.post(
    "/api/simulations/{sim_id}/resume",
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def resume_simulation(sim_id: str, request: Request) -> Dict[str, str]:
    """Resume a paused simulation from its next unexecuted round.

    If the pause hasn't taken effect yet (status "pausing"), the pause is
    cancelled and the existing run loop continues; no second loop starts.
    """
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    sim = orch.simulations[sim_id]
    if sim["status"] == SimulationStatus.PAUSED:
        # Check the concurrency cap before un-pausing, so a 429 leaves the
        # simulation paused rather than stranded in "pending".
        cap = orch.limits.max_concurrent
        if cap is not None and orch.active_count(exclude=sim_id) >= cap:
            raise HTTPException(
                status_code=429,
                detail=f"{cap} simulation(s) already running; the limit is {cap}",
            )
    was_paused = sim["status"] == SimulationStatus.PAUSED
    if not orch.resume_simulation(sim_id):
        raise HTTPException(status_code=409, detail="Simulation is not paused")
    if sim["status"] == SimulationStatus.RUNNING:
        return {"status": "running", "simulation_id": sim_id}
    try:
        _spawn_run(request, orch, sim_id)
    except HTTPException:
        if was_paused and sim["status"] == SimulationStatus.PENDING:
            sim["status"] = SimulationStatus.PAUSED  # don't strand it in pending
        raise
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
        rounds_completed=orch._rounds_completed(sim),
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
async def run_audit(request: Request) -> Dict[str, Any]:
    """Run the bundled security auditor on the installed sandfish package.

    The scan root is fixed (never the server's CWD), runs in a worker thread
    so the event loop keeps serving, and only one audit runs at a time.
    """
    lock: asyncio.Lock = request.app.state.audit_lock
    if lock.locked():
        raise HTTPException(status_code=409, detail="An audit is already running")
    async with lock:
        findings = await asyncio.to_thread(run_security_audit, str(_AUDIT_ROOT))
    return {
        "findings_count": len(findings),
        "critical": sum(1 for f in findings if f.severity == "CRITICAL"),
        "high": sum(1 for f in findings if f.severity == "HIGH"),
        "medium": sum(1 for f in findings if f.severity == "MEDIUM"),
        "low": sum(1 for f in findings if f.severity == "LOW"),
    }


_AUDIT_ROOT = Path(__file__).resolve().parents[1]


def _is_loopback_origin(origin: str) -> bool:
    from urllib.parse import urlsplit

    try:
        host = urlsplit(origin).hostname or ""
    except ValueError:
        return False
    return is_loopback_host(host)


def _ws_origin_allowed(origin: Optional[str], *, auth_on: bool) -> bool:
    """Origin policy for /ws.

    No Origin header (non-browser client): allowed. SANDFISH_CORS_ORIGINS
    set: the Origin must be listed. Otherwise: any origin when an API key is
    required, loopback origins only when auth is off.
    """
    if not origin:
        return True
    allowed = _allowed_origins()
    if allowed:
        return origin in allowed or "*" in allowed
    return auth_on or _is_loopback_origin(origin)


# ---------- WebSocket ----------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Real-time event stream.

    When SANDFISH_API_KEY is set, present the key in one of these ways:

    - an ``X-API-Key`` header on the upgrade request (non-browser clients);
    - the first message after connecting:
      ``{"command": "auth", "api_key": "..."}`` (browsers), within
      WS_AUTH_TIMEOUT_S seconds;
    - deprecated: ``?api_key=`` in the URL. Still accepted for backward
      compatibility, but the key ends up in access logs and proxies.

    Events are only delivered after authentication succeeds.
    """
    global _ws_pending_auth
    expected_key = _api_key_setting()
    if not _ws_origin_allowed(websocket.headers.get("origin"), auth_on=expected_key is not None):
        # Blocks cross-site WebSocket hijacking: browsers attach Origin, and
        # without a key any page could otherwise read the local event stream.
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    authed = expected_key is None
    if not authed:
        provided = websocket.headers.get("x-api-key")
        if provided is None and "api_key" in websocket.query_params:
            logger.warning(
                "WebSocket key passed in the URL (?api_key=) is deprecated; "
                "send an X-API-Key header or an auth message instead"
            )
            provided = websocket.query_params.get("api_key")
        if provided is not None:
            try:
                ok = await _authenticate(websocket, provided)
            except HTTPException:
                ok = False
            if not ok:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            authed = True

    if not authed:
        if _ws_pending_auth >= WS_MAX_PENDING_AUTH:
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
            return
        try:
            if await _auth_fail_limiter.blocked(_client_ip(websocket)) is not None:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        except Exception:
            return

    await websocket.accept()

    if not authed:
        _ws_pending_auth += 1
        ok = False
        try:
            first = await asyncio.wait_for(
                websocket.receive_json(), timeout=WS_AUTH_TIMEOUT_S
            )
            provided = first.get("api_key") if isinstance(first, dict) and first.get("command") == "auth" else None
            if isinstance(provided, str):
                ok = await _authenticate(websocket, provided)  # records failures
            else:
                await _auth_fail_limiter.record(_client_ip(websocket))
        except HTTPException:
            ok = False
        except Exception:  # timeout, malformed frame, disconnect
            ok = False
            await _auth_fail_limiter.record(_client_ip(websocket))
        finally:
            _ws_pending_auth -= 1
        if not ok:
            try:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            except Exception:
                pass
            return
        await websocket.send_json(
            {"type": "authenticated", "timestamp": _utcnow_iso(), "data": {}}
        )

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
    except Exception as exc:
        # Don't crash the server on a single-client error.
        logger.debug("WebSocket client error: %s", exc)
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

    host = os.environ.get("SANDFISH_HOST", "127.0.0.1")
    check_bind_allowed(host)
    configure_app()
    uvicorn.run(app, host=host, port=int(os.environ.get("SANDFISH_PORT", "8000")))
