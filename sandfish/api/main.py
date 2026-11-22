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
- GET  /api/simulations/{id}/agents  agent state (live, or the final snapshot)
- GET  /api/simulations/{id}/results metrics, totals and the top agents
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
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
from ..memory.ompa_adapter import OMPAMemoryAdapter, default_vault_path
from ..security.audit import run_security_audit

logger = logging.getLogger("sandfish.api")

# Seconds a WebSocket client has to send {"command": "auth", ...} when it
# didn't present the key in a header, and how many such not-yet-authenticated
# sockets may wait at once.
WS_AUTH_TIMEOUT_S = 10.0
WS_MAX_PENDING_AUTH = 32
_ws_pending_auth = 0
# Messages that may wait in one WebSocket client's outbox. A client that falls
# this far behind is disconnected (close code 1013) instead of being waited for.
WS_OUTBOX_MAX = 256
# Seconds allowed for the closing handshake with a client being disconnected.
WS_CLOSE_TIMEOUT_S = 5.0


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
    # What the run produced (see SwarmOrchestrator.results_summary).
    agent_types: Dict[str, int] = Field(default_factory=dict)
    total_actions: int = 0
    memory_targeted_actions: int = 0
    top_agents: List[Dict[str, Any]] = Field(default_factory=list)


class AgentInfoResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    energy: float
    mood: float
    reputation: float
    following: int = 0
    posts: int = 0
    actions: int = 0


# ---------- Module-level state ----------


class _EventClient:
    """One WebSocket subscriber with its own bounded outbox.

    Events used to be sent to every socket from inside the orchestrator's
    event callback, which the run loop awaits. A client that stopped reading
    made `send_json` wait, and every simulation waited with it. Now the
    callback only queues the message; one task per client does the sending,
    and it is the only writer on that socket. A client whose outbox fills up
    is disconnected rather than waited for.
    """

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.outbox: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=WS_OUTBOX_MAX)
        # None: events for every simulation (the default). A set: only these.
        self.subscriptions: Optional[set] = None
        self._sender: Optional["asyncio.Task[None]"] = None
        self._closer: Optional["asyncio.Task[None]"] = None

    def start(self) -> None:
        self._sender = asyncio.create_task(self._send_loop())

    def wants(self, data: Any) -> bool:
        if self.subscriptions is None:
            return True
        sim_id = data.get("simulation_id") if isinstance(data, dict) else None
        return sim_id is None or sim_id in self.subscriptions

    def offer(self, message: Dict[str, Any]) -> bool:
        """Queue a message without waiting. False means the outbox is full."""
        try:
            self.outbox.put_nowait(message)
        except asyncio.QueueFull:
            return False
        return True

    async def _send_loop(self) -> None:
        try:
            while True:
                await self.websocket.send_json(await self.outbox.get())
        except asyncio.CancelledError:
            raise
        except Exception:
            return  # socket is gone; the receive loop sees the disconnect

    def drop(self) -> None:
        """Disconnect a client that is not keeping up, without waiting on it."""
        self.stop()
        if self._closer is None:
            self._closer = asyncio.create_task(self._close(status.WS_1013_TRY_AGAIN_LATER))

    def stop(self) -> None:
        if self._sender is not None:
            self._sender.cancel()

    async def _close(self, code: int) -> None:
        try:
            await asyncio.wait_for(self.websocket.close(code=code), timeout=WS_CLOSE_TIMEOUT_S)
        except Exception:
            pass  # already closed, or the peer never answered


orchestrator: Optional[SwarmOrchestrator] = None
_websocket_clients: List[_EventClient] = []
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
    """Refuse to serve on a non-loopback address without an API key, or
    with an unsafe SANDFISH_CORS_ORIGINS (see cors_policy)."""
    cors_policy()
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
# Failed authentications per client IP.
_auth_fail_limiter = _SlidingWindowLimiter(
    max_per_minute=int(os.environ.get("SANDFISH_AUTH_FAILURES_PER_MINUTE", "10"))
)

# A key this long cannot be found by guessing over the network, whatever the
# rate. `openssl rand -hex 32`, the documented way to make one, gives 64.
MIN_STRONG_KEY_CHARS = 16


def api_key_is_guessable(key: Optional[str] = None) -> bool:
    """True when the configured key is short enough that guessing it online
    is the realistic attack, so failed attempts have to be throttled."""
    key = _api_key_setting() if key is None else key
    return key is not None and len(key) < MIN_STRONG_KEY_CHARS


async def _authenticate(conn: Any, provided: Optional[str]) -> bool:
    """Check `provided` against the configured key with failure throttling.

    Returns True when auth is off or the key matches. Raises HTTPException
    429 for a wrong key while the client IP is over its failed-attempt budget.

    The key is compared first, so the holder of the right key is not locked
    out by someone else's wrong guesses from the same address; behind a NAT
    or a reverse proxy that address is every client. Letting the right key
    through while the address is over budget means the budget no longer
    limits how fast keys can be tried, which only matters for a key short
    enough to guess. Such a key keeps the old order: an address over budget
    is refused before the key is looked at.
    """
    expected = _api_key_setting()
    if expected is None:
        return True
    ip = _client_ip(conn)
    throttle_first = api_key_is_guessable(expected)
    if not throttle_first and _keys_match(provided, expected):
        return True
    retry = await _auth_fail_limiter.blocked(ip)
    if retry is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed authentication attempts. Retry in {retry:.1f}s.",
        )
    if throttle_first and _keys_match(provided, expected):
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
    return cfg.get("ompa_vault_path") or default_vault_path(legacy_fallback=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator

    cors_policy()  # refuse to start with an unsafe/invalid CORS setting
    if api_key_is_guessable():
        logger.warning(
            "SANDFISH_API_KEY is shorter than %d characters. Failed attempts are "
            "throttled per client address, which also locks out the right key from "
            "an address that sent wrong ones. Use a longer key, e.g. "
            "`openssl rand -hex 32`, to remove both the risk and the lockout.",
            MIN_STRONG_KEY_CHARS,
        )
    # Off unless the operator set SANDFISH_SERVER_LLM_AGENTS; raises on a
    # half-configured opt-in rather than starting without it.
    from ..llm.agent import close_llm_client, disable_llm_agents, enable_llm_agents_from_env

    llm_budget = enable_llm_agents_from_env()
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
    if llm_budget is not None:
        await close_llm_client()
        disable_llm_agents()

    async with _websocket_lock:
        clients = list(_websocket_clients)
        _websocket_clients.clear()
    for client in clients:
        client.stop()
        await client._close(status.WS_1001_GOING_AWAY)


# ---------- App ----------


app = FastAPI(
    title="SandFish API",
    description="Multi-agent swarm intelligence simulation platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


_LOOPBACK_ORIGIN_RE = r"^https?://(localhost|127(\.\d{1,3}){3}|\[::1\])(:\d{1,5})?$"
_CORS_METHODS = ["GET", "POST"]
_CORS_HEADERS = ["X-API-Key", "Content-Type"]


@dataclass(frozen=True)
class CorsPolicy:
    """Which browser origins may call the API (and open /ws).

    Configured with SANDFISH_CORS_ORIGINS, a comma-separated list:

    - unset / empty / ``localhost``: loopback origins only
      (http(s)://localhost, 127.x.x.x, [::1], any port). The default.
    - ``none``: no cross-origin access at all.
    - explicit origins, e.g. ``https://app.example.com``; may be combined
      with ``localhost``.
    - ``*``: any origin. Refused unless SANDFISH_API_KEY is set, since
      without auth it would let any website drive the API.
    """

    origins: Tuple[str, ...] = ()
    loopback: bool = True
    any_origin: bool = False

    def allows(self, origin: Optional[str]) -> bool:
        if not origin:
            return True  # not a browser cross-origin request
        if self.any_origin or origin in self.origins:
            return True
        return self.loopback and _is_loopback_origin(origin)


def cors_policy(env: Optional[Dict[str, str]] = None) -> CorsPolicy:
    """Parse SANDFISH_CORS_ORIGINS; raises RuntimeError for `*` without auth."""
    env = os.environ if env is None else env
    raw = (env.get("SANDFISH_CORS_ORIGINS") or "").strip()
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return CorsPolicy()
    lowered = [item.lower() for item in items]
    if "none" in lowered:
        if len(items) > 1:
            raise RuntimeError("SANDFISH_CORS_ORIGINS: 'none' cannot be combined with other origins")
        return CorsPolicy(loopback=False)
    if "*" in items:
        key = (env.get("SANDFISH_API_KEY") or "").strip()
        if not key:
            raise RuntimeError(
                "SANDFISH_CORS_ORIGINS='*' is refused while auth is off: any website "
                "could drive this API. Set SANDFISH_API_KEY or list explicit origins."
            )
        return CorsPolicy(any_origin=True, loopback=True)
    origins = tuple(item.rstrip("/") for item, low in zip(items, lowered) if low != "localhost")
    for origin in origins:
        if not origin.startswith(("http://", "https://")):
            raise RuntimeError(
                f"SANDFISH_CORS_ORIGINS: {origin!r} is not an origin (expected http(s)://host[:port])"
            )
    return CorsPolicy(origins=origins, loopback="localhost" in lowered)


def _allowed_origins() -> List[str]:
    # Kept for backward compatibility with callers of the old helper.
    return list(cors_policy().origins)


class _DynamicCORS:
    """CORSMiddleware built from the policy in effect at request time, so the
    policy follows SANDFISH_CORS_ORIGINS/SANDFISH_API_KEY without re-importing
    the app. An invalid policy fails closed (no cross-origin access)."""

    def __init__(self, app: Any):
        self.app = app
        self._cache: Dict[CorsPolicy, CORSMiddleware] = {}

    def _middleware(self) -> CORSMiddleware:
        try:
            policy = cors_policy()
        except RuntimeError as exc:
            logger.error("CORS disabled: %s", exc)
            policy = CorsPolicy(loopback=False)
        mw = self._cache.get(policy)
        if mw is None:
            if len(self._cache) > 8:
                self._cache.clear()
            mw = CORSMiddleware(
                self.app,
                allow_origins=["*"] if policy.any_origin else list(policy.origins),
                allow_origin_regex=_LOOPBACK_ORIGIN_RE if policy.loopback and not policy.any_origin else None,
                allow_credentials=False,  # auth is a header, never cookies
                allow_methods=_CORS_METHODS,
                allow_headers=_CORS_HEADERS,
            )
            self._cache[policy] = mw
        return mw

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        await self._middleware()(scope, receive, send)


app.add_middleware(_DynamicCORS)


# ---------- Event broadcast ----------


async def _broadcast_event(event_type: str, data: Dict[str, Any]) -> None:
    """Queue an event for every WebSocket client that wants it.

    Registered as an orchestrator event callback, so it runs inside the
    simulation's run loop and must never wait on a client.
    """
    message = {
        "type": event_type,
        "timestamp": _utcnow_iso(),
        "data": data,
    }

    async with _websocket_lock:
        clients = list(_websocket_clients)

    slow = [c for c in clients if c.wants(data) and not c.offer(message)]
    if slow:
        async with _websocket_lock:
            for client in slow:
                if client in _websocket_clients:
                    _websocket_clients.remove(client)
        for client in slow:
            logger.warning("Disconnecting a WebSocket client that stopped reading events")
            client.drop()


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
    """List the agents in a simulation with their state.

    While the simulation can still run this is the live state. After it has
    finished, the agents themselves are released and this returns the state
    they ended in.
    """
    orch = _require_orchestrator()
    if sim_id not in orch.simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")

    agents: List[AgentInfoResponse] = []
    for agent_id, snapshot in orch.agent_states(sim_id).items():
        state = snapshot["state"]
        agents.append(
            AgentInfoResponse(
                id=snapshot.get("id", agent_id),
                name=snapshot["profile"]["name"],
                type=snapshot["profile"]["type"],
                status=state["status"],
                energy=state["energy"],
                mood=state["mood"],
                reputation=state["reputation"],
                following=len(state.get("connections", [])),
                posts=len(state.get("posts_created", [])),
                actions=int(snapshot.get("action_count", 0)),
            )
        )
    return agents


@app.get(
    "/api/simulations/{sim_id}/results",
    response_model=SimulationResultResponse,
    dependencies=[Depends(rate_limit)],
    tags=["simulations"],
)
async def get_simulation_results(
    sim_id: str,
    top_agents: int = Query(default=10, ge=0, le=100, description="Agents to list, by reputation"),
) -> SimulationResultResponse:
    """Return the latest result snapshot for a simulation: run metrics, action
    totals and the highest-reputation agents. Available during and after a run."""
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
        **orch.results_summary(sim_id, top_agents),
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


def _ws_origin_allowed(origin: Optional[str], *, auth_on: bool = False) -> bool:
    """Origin policy for /ws: the same CorsPolicy as the REST API.

    No Origin header (non-browser client) is allowed; an invalid policy
    fails closed. ``auth_on`` is accepted for backward compatibility.
    """
    try:
        return cors_policy().allows(origin)
    except RuntimeError:
        return not origin


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

    # From here on the sender task is the only writer on this socket: replies
    # to commands go through the same outbox as events, in order.
    client = _EventClient(websocket)
    client.offer(
        {
            "type": "connected",
            "timestamp": _utcnow_iso(),
            "data": {"message": "Connected to SandFish"},
        }
    )
    client.start()
    async with _websocket_lock:
        _websocket_clients.append(client)

    def reply(kind: str, data: Optional[Dict[str, Any]] = None) -> bool:
        message: Dict[str, Any] = {"type": kind, "timestamp": _utcnow_iso()}
        if data is not None:
            message["data"] = data
        return client.offer(message)

    try:
        while True:
            message = await websocket.receive_json()
            command = message.get("command")
            sim_id = message.get("simulation_id")

            if command == "subscribe":
                # No id (or "*"): every simulation, which is also the default.
                # Otherwise only events for the ids subscribed so far.
                if sim_id in (None, "*"):
                    client.subscriptions = None
                else:
                    if client.subscriptions is None:
                        client.subscriptions = set()
                    client.subscriptions.add(str(sim_id))
                ok = reply("subscribed", {"simulation_id": sim_id})
            elif command == "unsubscribe":
                if client.subscriptions is not None and sim_id is not None:
                    client.subscriptions.discard(str(sim_id))
                ok = reply("unsubscribed", {"simulation_id": sim_id})
            elif command == "ping":
                ok = reply("pong")
            else:
                ok = reply("error", {"message": f"Unknown command: {command!r}"})
            if not ok:
                break  # outbox full: the client is not reading

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        # Don't crash the server on a single-client error.
        logger.debug("WebSocket client error: %s", exc)
    finally:
        async with _websocket_lock:
            if client in _websocket_clients:
                _websocket_clients.remove(client)
        client.stop()
        await client._close(status.WS_1000_NORMAL_CLOSURE)


# ---------- Error handler ----------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    cfg = getattr(request.app.state, "config", None) or {}
    payload: Dict[str, Any] = {"error": "Internal server error"}
    if cfg.get("debug"):
        payload["detail"] = repr(exc)
    return JSONResponse(status_code=500, content=payload)


# ---------- Configuration helpers ----------


def configure_app(vault_path: Optional[str] = None, debug: bool = False) -> FastAPI:
    """Configure the FastAPI app before launching it.

    Without ``vault_path`` the vault is ``$SANDFISH_VAULT_PATH``, else
    ``./sandfish_vault`` (see ``default_vault_path``).
    """
    app.state.config = {"ompa_vault_path": vault_path or default_vault_path(), "debug": debug}
    return app


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SANDFISH_HOST", "127.0.0.1")
    check_bind_allowed(host)
    configure_app()
    uvicorn.run(app, host=host, port=int(os.environ.get("SANDFISH_PORT", "8000")))
