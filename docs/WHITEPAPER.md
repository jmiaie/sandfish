# SandFish: A Local-First Multi-Agent Simulation Engine

**Version**: 0.1 (draft)
**Status**: Early-stage research / simulation tool — not production software
**License**: MIT

## Abstract

SandFish is a small multi-agent simulation engine backed by a local OMPA
vault. It runs round-based simulations of heterogeneous agents, exposes a
FastAPI HTTP layer, and persists events and entities to the vault for later
analysis. It is explicitly scoped as a local, research-style tool: the
orchestrator is single-process, the API is minimally hardened, and there is
no distributed mode.

This paper describes the architecture, the simulation model, and the design
trade-offs. It deliberately avoids performance claims that have not been
reproduced.

## 1. Goals and non-goals

Goals:

- Give a reproducible, local sandbox for experimenting with small populations
  of interacting agents.
- Keep all state in one place (the OMPA vault) so a run can be inspected
  after the fact.
- Make it easy to add new agent types by subclassing `BaseAgent` and
  registering them with the factory.
- Ship a minimal HTTP API that a UI or notebook can poll during a run.

Non-goals:

- High-scale or distributed simulation.
- Hosting untrusted workloads or multi-tenant isolation.
- Beating any specific alternative on benchmarks. SandFish is not
  benchmarked against other frameworks in this repo; users who need that
  should run their own comparisons on their own hardware.

## 2. Architecture

```
┌──────────────────────────────────────────────┐
│           API (FastAPI + uvicorn)            │
│  REST endpoints · SSE event stream · auth    │
├──────────────────────────────────────────────┤
│              SwarmOrchestrator               │
│  Round loop · pause/resume/stop · callbacks  │
├──────────────────────────────────────────────┤
│                  Agents                      │
│  Default · Influencer · Lurker · custom      │
├──────────────────────────────────────────────┤
│              OMPAMemoryAdapter               │
│  Events · entities · relationships · search  │
└──────────────────────────────────────────────┘
```

### 2.1 Orchestrator

`SwarmOrchestrator` owns the simulation lifecycle: creating agents,
executing one round at a time, dispatching events to registered callbacks,
and exposing pause/resume/stop controls. Each simulation has a per-sim
`asyncio.Lock` so that duplicate `run_simulation` calls do not race.

A round:

1. Give each agent the current peer list.
2. Ask every agent for its next action concurrently (`asyncio.gather`).
3. Apply each action sequentially so shared state mutations are
   deterministic.
4. Emit a `round_complete` event and yield, then re-read status to support
   cooperative pause/stop.

### 2.2 Agents

`BaseAgent` owns an `AgentProfile` (traits, goals, backstory) and an
`AgentState` (energy, mood, action history bounded by a deque). Subclasses
override `decide_action`:

- `DefaultAgent` — trait-weighted random choice.
- `InfluencerAgent` — biased toward content creation.
- `LurkerAgent` — biased toward passive actions; forces rest at low energy.

New agent types can be registered via `register_agent_type(name, cls)` and
then created with `create_agent(name, ...)`.

### 2.3 Memory

`OMPAMemoryAdapter` is a thin wrapper over an OMPA vault with three main
surfaces:

- `record_event(event_type, description, metadata)` for orchestrator and
  agent lifecycle events.
- `add_entity` / `get_related_entities` for the agent knowledge graph.
- `search` for retrieval during agent decision-making.

All data is local to the vault directory; there is no mandatory remote
service.

### 2.4 API

`sandfish api` launches a FastAPI app with:

- `/health` — unauthenticated liveness probe.
- `/api/simulations/...` — CRUD and control endpoints for simulations.
- `/api/events` — server-sent events stream for orchestrator events.

If `SANDFISH_API_KEY` is set, every `/api/*` request must carry a matching
`X-API-Key` header. A sliding-window rate limit (default 120
requests/minute) applies per API key or client IP.

## 3. Security posture

SandFish is meant for local or trusted-network use. The shipped defaults
and guardrails:

- Non-root user in the Docker runtime image.
- Optional API-key auth and rate limiting on the HTTP layer.
- A static auditor (`sandfish security-audit`) that scans for common unsafe
  patterns (`eval`, `exec`, bare `subprocess shell=True`, hardcoded
  secrets), invokes `bandit` when available, and reports findings by
  severity. The audit exits non-zero on `CRITICAL` or `HIGH`.
- No eval or exec on untrusted content in the core.

Known limits:

- No multi-tenant isolation. All simulations run in the same process.
- No TLS termination; put SandFish behind a reverse proxy if exposed.
- The bundled agents do not call any LLM. If you plug one in, you own the
  egress policy.

## 4. Testing

The `tests/` directory covers agent behaviour (energy budget, action
selection, history bounds), orchestrator control flow (pause, resume, stop,
natural completion), edge cases (zero-agent / zero-round simulations,
Unicode names, nonexistent IDs), and a handful of integration tests that
run real small simulations against a temporary vault.

Run the suite with:

```bash
pip install -e '.[dev]'
pytest
```

## 5. Benchmarks

`benchmarks/run_all.py` measures orchestrator startup, agent-creation
throughput, full-simulation throughput, and (optional) process memory per
agent. Results depend on hardware, vault size, and Python version, so the
repo deliberately does not publish canonical numbers.

To run them:

```bash
pip install -e '.[dev]' psutil
python benchmarks/run_all.py
```

Each invocation writes `benchmark_report.json` in the working directory and
prints a summary.

## 6. Roadmap

Near-term candidates:

- Structured configuration for agent cohorts beyond the default/influencer/
  lurker mix.
- Optional persistence of agent state across runs.
- A minimal read-only dashboard backed by the event stream.
- Pluggable LLM-backed agents (with the policy work that implies).

Longer-term, out of current scope: distributed execution, custom OMPA
backends, or horizontal scale.

## 7. License

MIT. See `LICENSE`.
