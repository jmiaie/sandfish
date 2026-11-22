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
- Opt-in parameter search: researcher agents propose variants, an external
  evaluator plugin scores them, and an append-only, hash-chained trial ledger
  counts every evaluation (off by default; see
  [Parameter search](#parameter-search-opt-in)).
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

A running orchestrator keeps the five newest checkpoints of each simulation
in memory (`SwarmOrchestrator(..., keep_checkpoints=N)`, `None` for all);
`--checkpoint-dir` writes every one to disk. Checkpoints are snapshots for
inspection: there is no command that resumes a run from one.

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
- `GET  /api/simulations/{id}/agents` — every agent's state: live while the
  simulation can still run, and the state they ended in once it has finished.
- `GET  /api/simulations/{id}/results` — rounds completed, metrics (including
  per-simulation LLM calls/tokens when LLM agents are used), action totals,
  agents per type, and the `top_agents` (default 10, max 100) by reputation.
- `WS   /ws` — WebSocket stream of orchestrator events (`round_complete`, …).
  With an API key set, authenticate with an `X-API-Key` header or by sending
  `{"command": "auth", "api_key": "..."}` as the first message
  (`?api_key=` in the URL still works but is deprecated: URLs end up in logs).
  Browser connections follow the same origin policy as REST
  (`SANDFISH_CORS_ORIGINS`, default loopback origins only).
  A client receives events for every simulation until it sends
  `{"command": "subscribe", "simulation_id": "..."}`; after that it receives
  only the simulations it subscribed to (`unsubscribe` removes one, `subscribe`
  without an id goes back to everything). Events are queued per client and
  never hold up a simulation: a client that stops reading is disconnected
  with close code 1013 once 256 messages are waiting for it.

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
| `SANDFISH_VAULT_PATH` | Directory for Sandfish's OMPA vault (`--vault` overrides it). `OMPA_VAULT_PATH` is still read as a fallback by `sandfish-mcp`, and by the API when it is started without `sandfish api`, with a warning: OMPA's own MCP server uses that variable for an agent's real memory, so sharing it puts simulation notes there | `./sandfish_vault` |
| `SANDFISH_API_KEY` | If set, `/api/*` requires an `X-API-Key` header | unset (auth off) |
| `SANDFISH_RATE_LIMIT_PER_MINUTE` | Sliding-window rate limit per key/IP | `120` |
| `SANDFISH_AUTH_FAILURES_PER_MINUTE` | Wrong keys allowed per IP per minute before they get 429. The right key is never locked out, unless the key is shorter than 16 characters (then an IP over the limit is refused whatever key it sends) | `10` |
| `SANDFISH_MAX_AGENTS` / `SANDFISH_MAX_ROUNDS` | Per-simulation caps in the API (`0` = unlimited) | `1000` / `10000` |
| `SANDFISH_MAX_CONCURRENT_SIMULATIONS` | Simulations running at once in the API | `4` |
| `SANDFISH_MAX_FINISHED_SIMULATIONS` | Finished simulations kept; oldest evicted | `100` |
| `SANDFISH_MAX_SEARCH_EVALUATIONS` | Cap on evaluations per MCP search (grid size, or agents × rounds) | `10000` |
| `SANDFISH_CORS_ORIGINS` | Browser origins allowed for REST and `/ws`. Accepts `localhost`, `none`, explicit `https://…` origins, or `*`. `*` needs `SANDFISH_API_KEY`, or startup fails | unset = `localhost` (loopback origins only) |
| `LLM_API_KEY` | Slot for user-provided LLM-backed agents | unset |
| `SANDFISH_SERVER_LLM_AGENTS` | `1` lets callers of the API and MCP server run `llm` agents on the operator's provider key (needs `SANDFISH_LLM_PROVIDER`) | unset (off) |
| `SANDFISH_LLM_MAX_CALLS` / `SANDFISH_LLM_MAX_TOTAL_CALLS` | With the opt-in: provider requests per simulation / for the life of the process (`0` = unlimited) | `200` / `2000` |

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
  metrics. The budget counts requests sent to the provider: a retry on
  429/5xx or a failed connection takes another unit, so a simulation never
  sends more requests than its cap. Any error falls back to the heuristic;
  keys are read from the environment and redacted from errors.
- `MockLLMClient` (`--llm-provider mock`) for tests and dry runs.

**On a server.** Callers of the HTTP API and of `sandfish-mcp` choose the
agent types, and every `llm` agent spends the operator's provider key, so a
server offers them only when its operator opts in:

```bash
export SANDFISH_SERVER_LLM_AGENTS=1       # the opt-in; a provider alone is not enough
export SANDFISH_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...
export SANDFISH_LLM_MAX_CALLS=200         # requests per simulation (default)
export SANDFISH_LLM_MAX_TOTAL_CALLS=2000  # requests for the life of the process (default)
sandfish api        # or: sandfish-mcp
```

`0` lifts either cap. With the opt-in set and no provider, the server refuses
to start. Without it, `agent_types: ["llm"]` is rejected as an unknown type.

## MCP server (optional)

`sandfish-mcp` lets an MCP client such as Claude Code or Cursor create, run and
inspect simulations. It uses stdio and the official Python MCP SDK, and its
tool names take an `sf_` prefix, the same convention as OMPA's `ao_` tools.

```bash
pip install -e '.[mcp]'            # from a checkout
# or: pip install 'sandfish[mcp] @ git+https://github.com/jmiaie/sandfish'
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
| `sf_search_evaluators` | List evaluators usable by name (`toy` plus installed plugins) |
| `sf_search_run` | Start a parameter search: `strategy` `grid` or `swarm`, `evaluator`, `space`, `seed`, … |
| `sf_search_ledger` | Read trial-ledger rows (paged, optional `split`) with counts and the hash-chain check |
| `sf_search_summary` | Exact trial counts, integrity, best/top train variants and any test scores |
| `sf_search_validate` | Score finalists on the held-out `test` split (logged in the same ledger); `dedupe_metadata_fields` drops duplicate finalists |
| `sf_search_overfit` | Holdout-guard status, train-vs-test degradation, walk-forward runs and (for Sharpe scores) the Deflated Sharpe Ratio using the ledger's N |

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

## Parameter search (opt-in)

`sandfish.search` reuses the orchestrator for a different job: a swarm of
*researcher* agents proposes variants from a discrete parameter space, an
**external evaluator** you supply scores each one, and every evaluation is
written to an append-only trial ledger. None of it runs unless you call it;
the social-simulation agents and defaults behave exactly as before.

**What SandFish is and is not here.** SandFish is an orchestration and audit
layer. It is not a backtester, and it does not generate prices, forecasts or
base rates. All data access, modelling and scoring happen inside your
evaluator; SandFish only decides which variant to try next and records what
happened.

**Use a plain grid when you can.** If the space is small enough to enumerate,
evaluate all of it (`run_grid`, or `sf_search_run` with `strategy="grid"`).
A grid gives an exact trial count and has no path dependence. The swarm is
for spaces too large to enumerate, and it adds orchestration and logging,
not signal.

**Validate on held-out data, and count every trial.** A variant picked by
agents (or by you, from a grid) is selected on the training split, so its
training score is biased upward. Re-score finalists on held-out data
(`validate_holdout`, or `sf_search_validate`) before trusting them. Pick the
finalists first and validate once; every test-split evaluation is logged too,
so repeated peeking shows up. For a multiple-testing correction (Bonferroni,
deflated Sharpe ratio, and so on), use the ledger's count of distinct
variants tried on the training split, `counts.by_split.train.unique_variants`,
plus anything you tried outside this ledger, as the number of trials N, never
the number of finalists.

### Evaluator interface

An evaluator takes a proposed variant (a dict of parameter values) and a
split name (`"train"` or `"test"`) and returns a score plus metadata:

```python
from sandfish.search import EvaluationResult

class MyEvaluator:
    name = "my_eval"

    def evaluate(self, variant: dict, split: str) -> EvaluationResult:
        rows = load_rows(split)          # your data; keep "test" unseen during search
        score, n = my_metric(rows, **variant)
        return EvaluationResult(score=score, n=n, metadata={"note": "..."})
```

A plain function `fn(variant, split)` works too, and you may return a dict
with a numeric `score` (optional `n` and `metadata`) instead of an
`EvaluationResult`. The search always evaluates on `"train"`; only
`validate_holdout` uses `"test"`. An evaluator that raises is recorded as an
error row (a failed evaluation still counts as a trial).

To make an evaluator available by name (and so to the MCP server), register a
factory with `register_evaluator("my_eval", factory)` or ship it as a plugin in
any installed package:

```toml
[project.entry-points."sandfish.evaluators"]
my_eval = "my_pkg.evaluators:make_evaluator"   # factory(**options) -> evaluator
```

Only registered names and installed entry points can be loaded by name.
`toy` (`ToyQuadraticEvaluator`) is built in for tests and examples: a
synthetic quadratic with different deterministic noise on each split.

### Python

```python
import asyncio
from sandfish.core.orchestrator import create_orchestrator
from sandfish.search import ToyQuadraticEvaluator, TrialLedger, run_grid, run_search, validate_holdout

toy = ToyQuadraticEvaluator()
space = {"x": [-3, -2, -1, 0, 1, 2, 3], "y": [-3, -2, -1, 0, 1, 2, 3]}

grid = TrialLedger("ledgers/grid.jsonl")
run_grid(space, toy, grid)                       # 49 evaluations, exact

orch = create_orchestrator("./sandfish_vault")
swarm = TrialLedger("ledgers/swarm.jsonl")
summary = asyncio.run(run_search(orch, space=space, evaluator=toy, ledger=swarm,
                                 num_agents=4, max_rounds=10, seed=7))
print(summary["counts"])                         # every evaluation, repeats included
print(validate_holdout(swarm, toy, top_k=3))     # finalists re-scored on "test"
orch.close()
```

For full control, create a simulation with agent types `researcher`,
`researcher_explorer` (mostly random draws) or `researcher_exploiter` (mostly
one-parameter changes to the current best), call `register_search_agents()`
and `attach_search(orch, sim_id, space=..., evaluator=..., ledger=...)`, then
run it as usual. `repair=` maps a proposed variant to a valid one, `min_n=`
ignores small-sample results when choosing what to exploit, and
`remember=True` also writes each new variant to OMPA memory.

### Trial ledger

One JSON object per line, never rewritten. Each row has `seq`, `ts`,
`run_id`, `source` (agent id, `grid` or `holdout`), `round`, `split`,
`evaluator`, `variant`, `key`, `result` (`score`, `n`, `metadata`) or
`error`, `cached` (a repeat that reused the earlier score), `prev` and `hash`.
`hash` is the SHA-256 of the row without it, and `prev` links to the previous
row, so `verify_ledger_file(path)` detects edited, deleted, reordered or
corrupted rows, and an interrupted last write. Rows removed from the *end*
can't be detected from the file alone, so keep the `head` hash a summary
returns and pass it later as `verify_ledger_file(path, expected_head=...)`.
This is tamper evidence for an honest audit trail, not protection against
someone who rewrites the whole file. Reopening a ledger verifies it and
appends after the last row; use one writer process per ledger file, and one
ledger per evaluator configuration (repeats reuse an earlier score only for
the same split and evaluator name). `counts()` reports `n_trials` (every
row), `n_evaluator_calls`, `n_unique` and the same per split.

Over MCP, both strategies run in the background (`wait_seconds` ≤ 300 waits
for them), evaluator options from the client are passed to the evaluator's
factory (so plugin factories must treat them as untrusted), and ledgers are
written to `<vault>/search_ledgers/<search_id>.jsonl`, where
`sf_search_ledger` / `sf_search_summary` can still read them after a restart.
`sf_search_validate` refuses while a search is still running or paused.

### Robustness and overfitting checks

Everything here is opt-in and leaves the default paths unchanged.

- **Invalid combinations:** `run_grid(space, ev, ledger, valid=lambda v: v["fast"] < v["slow"])`
  skips combinations that make no sense. They are never evaluated or recorded,
  so the ledger's N counts hypotheses actually tested (the result reports
  `grid_size`, `valid_size` and `skipped_invalid`).
- **Not scorable:** return `EvaluationResult.not_scorable(n=3, reason="too few
  trades")` (or `{"score": None}`) when the sample is too small to score. The
  trial is recorded and counted in N, but `best()`/`top()`, the researcher
  agents and `validate_holdout` never rank it. Use this instead of a fake `0.0`.
- **Deterministic content head:** `ledger.content_head()` hashes each row
  without its timestamp, run id and chain hashes, so two identical, seeded
  runs give the same content head even though their `head` differs.
- **Duplicate finalists:** `validate_holdout(..., dedupe_key=dedupe_by_metadata("trade_set_hash"))`
  (or `dedupe_by_score(params=[...])`, or your own `row -> fingerprint`) keeps
  near-identical finalists from filling `top_k`; dropped ones are listed in
  `result.skipped`.
- **Locked holdout:** `ledger.lock_holdout()` (or `lock_holdout=True` on
  `run_grid`/`run_search`/`sf_search_run`) makes `validate_holdout` the only
  way to score the test split, once. Any other attempt raises
  `HoldoutLockedError` before the evaluator runs. The lock, each denied
  attempt and the validation are written to the ledger as *event rows* in the
  same hash chain (they are not trials and never count towards N). A second
  validation needs `allow_repeat=True` and is logged as a reopening.
- **Statistics** (`sandfish.search.overfit`, formulas cited in the docstrings;
  checked against independently computed reference values in the tests):
  `deflated_sharpe_ratio` / `ledger_deflated_sharpe` (Bailey & López de Prado,
  2014; give `score_is_sharpe=True`, since SandFish cannot know what a score
  means), `probability_of_backtest_overfitting` (CSCV, Bailey et al., 2017;
  needs a T x N returns matrix, so it is Python-only), `make_walk_forward_folds`
  and `walk_forward` (rolling or expanding windows, with an embargo; your
  evaluator implements `evaluate_fold(variant, fold)` on train-split data), and
  `holdout_report`. `summarize(ledger, overfit={"score_is_sharpe": True})` and
  `sf_search_overfit` bundle what a ledger alone can answer. These are
  diagnostics, not guarantees: they assume trials are roughly independent and
  N is honest.

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
  search/       Opt-in parameter search: evaluators, trial ledger, researcher agents.
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
