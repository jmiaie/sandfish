# Changelog

All notable changes to this project are listed here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-15

### Added
- **OMPA 1.1.1 memory.** Agents now actually use OMPA memory: they recall
  what they observed before acting, and each run writes its memory to a
  vault (`ompa>=1.1.1,<2`). Optional hybrid embeddings come with the
  `semantic` extra.
- **Optional LLM agents** (the `llm` extra): Anthropic, Gemini and OpenRouter
  clients, plus a mock client.
  - Each simulation gets its own call and token budget (`SANDFISH_MAX_*`).
  - Retries on 429/5xx, with a heuristic fallback on any error.
  - API keys are redacted from error messages.
- **`sandfish-mcp`** (the `mcp` extra): a stdio MCP server built on the official
  MCP SDK.
  - Tools: `sf_create_simulation`, `sf_start`, `sf_pause`, `sf_resume`,
    `sf_stop`, `sf_status`, `sf_list`, `sf_results`, `sf_report`,
    `sf_memory_search`, `sf_memory_entity`.
  - Resources: `sandfish://events` and `sandfish://events/{simulation_id}`.
- **Reproducible runs.** The same seed and the same config give the same
  results. The final checkpoint is always written.
- **CI** on GitHub Actions: `ruff check .` and `pytest` on Python 3.12.

### Changed
- **CORS** (`SANDFISH_CORS_ORIGINS`):
  - The default is loopback origins only. `none` blocks all cross-origin
    access, and explicit origins can be listed.
  - `*` is refused unless `SANDFISH_API_KEY` is set.
  - Credentials are never allowed.
  - `/ws` follows the same policy.
- **Security auditor:** the dependency check uses `pip-audit` instead of
  `safety`. `SANDFISH_AUDIT_DEPENDENCIES=0` skips it.
- **Pause/resume:** pause and resume are now real state transitions, and a
  resumed run continues from the round where it stopped.
- **API hardening:**
  - The server refuses to bind to a non-loopback address without an API key.
  - Failed logins are rate-limited per IP.
  - API keys are compared in constant time.
  - WebSocket clients authenticate with a header or a first `auth` message
    (`?api_key=` is deprecated).
  - There is a cap on pending WebSocket auths.
- **Package metadata:** authors and URLs now point at
  [jmiaie/sandfish](https://github.com/jmiaie/sandfish).

### Fixed
- A garbage-collector finalizer deadlock in the orchestrator. The orchestrator
  now has an explicit `close()`.
- Unclosed SQLite connections.
- Agent and context clean-up between runs.

## [0.1.0]

The first PyPI upload. Its metadata points at an old repository URL
(`github.com/micap/sandfish`). Use 0.2.0 or later.
