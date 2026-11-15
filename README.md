# SandFish

> **Public swarm demo / portfolio reference (v1).**  
> Deeper multi-agent product work lives in **AegisFlow**: private canonical [`jmiaie/af`](https://github.com/jmiaie/af), public mirror [`jmiaie/af_public`](https://github.com/jmiaie/af_public).  
> Lineage (v1 → AegisFlow) and dual-maintenance guidance: **[HISTORY.md](HISTORY.md)** · **[docs/POSITIONING.md](docs/POSITIONING.md)**.

Multi-agent swarm-simulation engine with local, OMPA-backed memory.

SandFish runs round-based simulations of populations of lightweight agents
(default, influencer, lurker) over a shared memory graph. Each agent picks an
action per round, the orchestrator executes and records it, and callbacks or
the HTTP API can observe the run in real time.

It is a small, local-first research/simulation tool kept as a **public demo**.
It is **not** a production workload platform and should not be deployed in
front of the public internet without the auth and rate-limit settings below.
For AegisFlow (stricter sandbox, lead/sub-agent delegation), see
[`jmiaie/af`](https://github.com/jmiaie/af) / [`jmiaie/af_public`](https://github.com/jmiaie/af_public).

## Features

- Round-based orchestrator with pause, resume, and stop semantics.
- Reproducible runs: set a `seed` and the same config replays identically.
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
- `pytest`, `pytest-asyncio`, `pytest-cov`, `httpx`, `bandit`, `pip-audit`, `ruff` for the `dev` extra
- `PyYAML` only if you pass a YAML config file to the CLI

## Install

```bash
pip install sandfish                 # core: engine, CLI, FastAPI service
pip install "sandfish[mcp]"          # + the sandfish-mcp server
pip install "sandfish[llm]"          # + Anthropic / Gemini / OpenRouter agents
```

From a checkout (for development, tests and audits):

```bash
git clone https://github.com/jmiaie/sandfish
cd sandfish
pip install -e '.[dev]'
```

Release notes are in [CHANGELOG.md](CHANGELOG.md).

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

### Reproducible runs

Pass `--seed` (or set `seed` in the config file, or `seed` in the
`POST /api/simulations` body) to make a run repeatable:

```bash
sandfish orchestrator --rounds 50 --agents 20 --seed 1234
```

Two runs with the same seed and config produce identical action sequences,
agent states, and post IDs. Each agent draws from its own stream derived from
the run seed, so results do not depend on how the agents' async decisions
interleave. Without a seed, every run is independently seeded from the OS.

Simulation and agent *identifiers* stay unique per run by design; only the
simulation dynamics reproduce.

Custom agents must draw randomness from `self.rng` (a `random.Random`) rather
than the `random` module, or their decisions will not replay under a seed.

### Checkpoints

Checkpoints are labelled by rounds *completed*, so a checkpoint at round 10
holds the state after 10 rounds. The last executed round is always
checkpointed, even when the round count is not a multiple of
`checkpoint_interval` or the run was stopped early. Set
`checkpoint_interval: 0` to disable checkpointing.

### Run the HTTP API

```bash
sandfish api --port 8000 --vault ./sandfish_vault          # binds 127.0.0.1
# Network-facing: an API key is required, or the server refuses to start.
SANDFISH_API_KEY=$(openssl rand -hex 32) sandfish api --host 0.0.0.0 --port 8000
```

Available endpoints include:

- `GET  /health` — liveness probe (no auth).
- `POST /api/simulations` — create a simulation (unknown `agent_types` → 422).
- `POST /api/simulations/{id}/start` — run a pending simulation.
- `GET  /api/simulations/{id}` — status.
- `POST /api/simulations/{id}/pause|resume|stop` — control. A pause reports
  `pausing` until the in-flight round finishes, then `paused`; resume
  continues from the next round (no round is re-run).
- `GET  /api/simulations/{id}/results` — rounds completed and metrics
  (including per-simulation LLM calls/tokens when LLM agents are used).
- `WS   /ws` — WebSocket stream of orchestrator events (`round_complete`, …).
  With an API key set, authenticate with an `X-API-Key` header or by sending
  `{"command": "auth", "api_key": "..."}` as the first message
  (`?api_key=` in the URL still works but is deprecated: URLs end up in logs).
  Browser connections follow the same origin policy as REST
  (`SANDFISH_CORS_ORIGINS`, default loopback origins only).

Failed authentication is throttled per client IP (default 10 per minute, then
429 for that IP, including requests with the right key). Behind a reverse
proxy every client shares the proxy's IP, so run uvicorn with
`--proxy-headers`/`--forwarded-allow-ips` or terminate auth at the proxy.

### Run the security auditor

```bash
sandfish security-audit --path . --output audit.md
```

Exits non-zero when any `CRITICAL` or `HIGH` finding is reported.

The dependency check runs [`pip-audit`](https://pypi.org/project/pip-audit/)
(in the `dev` extra) against the installed environment, which queries the
PyPA advisory service over the network. It is skipped when pip-audit is not
installed or when `SANDFISH_AUDIT_DEPENDENCIES=0`.

## Configuration

Environment variables (see `.env.example` for the full list):

| Variable | Purpose | Default |
| --- | --- | --- |
| `OMPA_VAULT_PATH` | Directory for OMPA-backed memory | `./sandfish_vault` |
| `SANDFISH_API_KEY` | If set, `/api/*` requires an `X-API-Key` header | unset (auth off) |
| `SANDFISH_RATE_LIMIT_PER_MINUTE` | Sliding-window rate limit per key/IP | `120` |
| `SANDFISH_AUTH_FAILURES_PER_MINUTE` | Failed auth attempts allowed per IP before 429 | `10` |
| `SANDFISH_MAX_AGENTS` / `SANDFISH_MAX_ROUNDS` | Per-simulation caps in the API (`0` = unlimited) | `1000` / `10000` |
| `SANDFISH_MAX_CONCURRENT_SIMULATIONS` | Simulations running at once in the API | `4` |
| `SANDFISH_MAX_FINISHED_SIMULATIONS` | Finished simulations kept; oldest evicted | `100` |
| `SANDFISH_CORS_ORIGINS` | Browser origins allowed for REST and `/ws`. Accepts `localhost`, `none`, explicit `https://…` origins, or `*`. `*` needs `SANDFISH_API_KEY`, or startup fails | unset = `localhost` (loopback origins only) |
| `LLM_API_KEY` | Slot for user-provided LLM-backed agents | unset |

Leaving `SANDFISH_API_KEY` unset disables auth — fine for local work, which
is why `sandfish api` binds `127.0.0.1` by default and refuses a non-loopback
`--host` without a key.

## Memory (OMPA 1.1)

SandFish requires `ompa>=1.1.1,<2` and uses it for real:

- **Events** are batched (one dated note per simulated day/round) via
  `Ompa.write(..., date=)`, so `memory.search()` returns hits (offline BM25 by
  default; hybrid embeddings with `pip install 'sandfish[semantic]'`).
- **Temporal ranking** is on: pass `start_date` (API/config) or
  `--start-date` (CLI) and each round is a simulated day, so queries like
  "what did agent_3 do yesterday" rank by the simulation's calendar.
- **Interactions** (follow / like / comment / repost) are dated
  knowledge-graph facts (`kg_add(valid_from=)`); `memory.get_interactions(id,
  as_of=...)` answers "who engaged with me, as of round N".
- **Memory drives behaviour:** bundled agents pick targets with a
  per-type `RECIPROCITY` (default 0.5, influencer 0.3, lurker 0.8) that
  prefers peers who previously engaged them, read from those facts.
  Set `agent.reciprocity = 0` for the old uniform-random targeting.

Set `SANDFISH_OMPA_SEMANTIC=0` to force offline keyword search.

## LLM agents (optional, off by default)

Bundled agents are heuristic and make no network calls. To let an LLM
choose actions with schema-constrained structured output:

```bash
pip install '.[llm]'
export SANDFISH_LLM_PROVIDER=anthropic   # or gemini | openrouter | mock
export ANTHROPIC_API_KEY=...             # GEMINI_API_KEY / OPENROUTER_API_KEY
sandfish orchestrator --agents 6 --rounds 5 --agent-types llm,default \
  --llm-max-calls 30
```

- Provider-neutral `LLMClient` protocol (`sandfish/llm/base.py`); clients for
  Anthropic (forced tool use, default `claude-sonnet-5-5`), Gemini
  (`responseJsonSchema`, default `gemini-3.5-flash-lite`) and OpenRouter
  (strict `json_schema`, default `anthropic/claude-sonnet-5.5`). Override
  with `SANDFISH_LLM_MODEL`.
- The decision schema constrains `action` to the action enum and `target`
  to visible peers, so models can't invent agents.
- Hard call budget per simulation (`--llm-max-calls`, or
  `enable_llm_agents(client, max_calls=..., max_total_calls=...)` for an
  overall cap too); per-simulation calls and tokens appear in the results'
  metrics. Retries on 429/5xx, and a heuristic fallback on any error; keys
  are read from the environment and redacted from errors.
- `MockLLMClient` (`--llm-provider mock`) for tests and dry runs.

## MCP server (optional)

`sandfish-mcp` lets an MCP client such as Claude Code or Cursor create, run and
inspect simulations. It uses stdio and the official Python MCP SDK, and its
tool names take an `sf_` prefix, the same convention as OMPA's `ao_` tools.

```bash
pip install "sandfish[mcp]"        # from PyPI
# or, from a checkout: pip install -e '.[mcp]'
sandfish-mcp --vault ./sandfish_vault
```

| Tool | What it does |
| --- | --- |
| `sf_create_simulation` | Create a simulation: name, agents, rounds, `agent_types`, `seed`, `start_date`, … |
| `sf_start` | Start it in the background; `wait_seconds` (≤300) waits for it to finish |
| `sf_pause` / `sf_resume` / `sf_stop` | Lifecycle control (`pausing` → `paused`; resume never re-runs a round) |
| `sf_status` / `sf_list` | Status of one simulation / of all of them |
| `sf_results` / `sf_report` | Metrics and top agents as JSON / as a Markdown report |
| `sf_memory_search` | Search the OMPA memory; `reference_date` anchors "yesterday" in simulated time |
| `sf_memory_entity` | One entity's facts and interactions, optionally `as_of` a date |

Resources:

- `sandfish://events` returns the recent event log, oldest first.
- `sandfish://events/{simulation_id}` returns one simulation's events.

The same `SANDFISH_MAX_*` caps as the HTTP API apply. Like other stdio MCP
servers it has no authentication: it runs as you and talks only to the client
that launched it.

**Claude Code**

```bash
claude mcp add sandfish -e SANDFISH_OMPA_SEMANTIC=0 -- sandfish-mcp --vault /abs/path/to/sandfish_vault
```

**Cursor** (`~/.cursor/mcp.json`, or `.cursor/mcp.json` in a project):

```json
{
  "mcpServers": {
    "sandfish": {
      "command": "sandfish-mcp",
      "args": ["--vault", "/abs/path/to/sandfish_vault"],
      "env": { "SANDFISH_OMPA_SEMANTIC": "0" }
    }
  }
}
```

Use an absolute vault path, because clients start servers from their own
working directory. If `sandfish-mcp` isn't on the client's `PATH`, use
`"command": "/path/to/venv/bin/python", "args": ["-m", "sandfish.mcp_server", ...]`
instead.

## Docker

```bash
# build and run the API
SANDFISH_API_KEY=$(openssl rand -hex 32) docker compose up --build   # key required

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
  mcp_server.py `sandfish-mcp` MCP server (stdio; `[mcp]` extra).
tests/          pytest suite (unit + edge cases + integration).
benchmarks/     Stand-alone benchmark runner.
docs/           Design and install notes.
```

## Tests

```bash
pip install -e '.[dev,llm,mcp]'
ruff check .
pytest            # offline; set SANDFISH_OMPA_SEMANTIC=0 to skip embeddings
pip-audit         # dependency advisories
```

CI (`.github/workflows/ci.yml`) runs the same ruff + pytest on Python 3.12
for pushes to `main` and for pull requests, with no secrets.

The suite covers agent behavior, orchestrator lifecycle, and a handful of
integration and stress tests. Async tests run under `pytest-asyncio`.

## Project history

- [HISTORY.md](HISTORY.md) — SandFish (v1) → AegisFlow (v2) lineage
- [docs/POSITIONING.md](docs/POSITIONING.md) — sandfish vs `af` vs `af_public`
- [docs/PORTFOLIO_SUMMARY.md](docs/PORTFOLIO_SUMMARY.md) — engineering scope for portfolio viewers

## License

MIT. See [LICENSE](LICENSE).
