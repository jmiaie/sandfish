# SandFish

Multi-agent swarm-simulation engine with local, OMPA-backed memory.

SandFish runs round-based simulations of populations of lightweight agents
(default, influencer, lurker) over a shared memory graph. Each agent picks an
action per round, the orchestrator executes and records it, and callbacks or
the HTTP API can observe the run in real time.

It is a small, local-first research/simulation tool. It is **not** a
production workload platform and should not be deployed in front of the public
internet without the auth and rate-limit settings below.

## Features

- Round-based orchestrator with pause, resume, and stop semantics.
- Pluggable agent types (`default`, `influencer`, `lurker`) with a factory
  hook for custom agents.
- OMPA vault as the single source of truth for events, entities, and
  relationships.
- Sync or async event callbacks for downstream dashboards.
- FastAPI HTTP server with optional API-key auth, per-client rate limiting,
  and an event-stream endpoint.
- Basic static security auditor over the source tree.
- Docker image and Compose stack for the API and a standalone orchestrator.

## Requirements

- Python 3.10 or newer
- `fastapi`, `uvicorn[standard]`, `pydantic`, `python-dotenv`, `ompa` (pulled
  in by `pip install .`)
- `pytest`, `pytest-asyncio`, `httpx`, `bandit`, `safety` for the `dev` extra
- `PyYAML` only if you pass a YAML config file to the CLI

## Install

```bash
git clone <this-repo>
cd sandfish
pip install .
# for tests and audits:
pip install -e '.[dev]'
```

The package is not on PyPI yet, so install from source.

## Quick start

### Run a simulation from the CLI

```bash
sandfish orchestrator \
  --vault ./sandfish_vault \
  --rounds 50 \
  --agents 20
```

Pass `--config path/to/config.json` (or `.yaml`) to load defaults from a file;
explicit `--rounds` / `--agents` flags still win. Use `--dry-run` to create
the simulation without executing it, and `--checkpoint-dir DIR` to persist
per-round checkpoints.

### Run the HTTP API

```bash
sandfish api --host 0.0.0.0 --port 8000 --vault ./sandfish_vault
```

Available endpoints include:

- `GET  /health` — liveness probe (no auth).
- `POST /api/simulations` — create a simulation.
- `POST /api/simulations/{id}/start` — run a simulation.
- `GET  /api/simulations/{id}` — status.
- `POST /api/simulations/{id}/pause|resume|stop` — control.
- `GET  /api/events` — server-sent events stream of orchestrator events.

### Run the security auditor

```bash
sandfish security-audit --path . --output audit.md
```

Exits non-zero when any `CRITICAL` or `HIGH` finding is reported.

## Configuration

Environment variables (see `.env.example` for the full list):

| Variable | Purpose | Default |
| --- | --- | --- |
| `OMPA_VAULT_PATH` | Directory for OMPA-backed memory | `./sandfish_vault` |
| `SANDFISH_API_KEY` | If set, `/api/*` requires an `X-API-Key` header | unset (auth off) |
| `SANDFISH_RATE_LIMIT_PER_MINUTE` | Sliding-window rate limit per key/IP | `120` |
| `SANDFISH_CORS_ORIGINS` | Comma-separated CORS origins | unset |
| `LLM_API_KEY` | Slot for user-provided LLM-backed agents | unset |

Leaving `SANDFISH_API_KEY` unset disables auth — fine for local work, **not**
fine for anything exposed to a network.

## Docker

```bash
# build and run the API
docker compose up --build

# run a long simulation alongside it
docker compose --profile orchestrator up
```

The API container runs as a non-root user and exposes a healthcheck against
`/health`. Mount a volume at `/app/vault` to persist data across runs.

## Project layout

```
sandfish/
  agents/       Agent definitions and factory.
  api/          FastAPI app, auth, rate limit, event stream.
  core/         SwarmOrchestrator and simulation config.
  memory/       OMPAMemoryAdapter wrapper around the OMPA vault.
  security/     Static security auditor.
  cli.py        `sandfish` entry point.
tests/          pytest suite (unit + edge cases + integration).
benchmarks/     Stand-alone benchmark runner.
docs/           Design and install notes.
```

## Tests

```bash
pip install -e '.[dev]'
pytest
```

The suite covers agent behavior, orchestrator lifecycle, and a handful of
integration and stress tests. Async tests run under `pytest-asyncio`.

## License

MIT. See [LICENSE](LICENSE).
