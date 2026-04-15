# SandFish — Portfolio Summary

## Overview

**SandFish** is a local-first multi-agent simulation engine built from scratch
in Python 3.10+. It runs round-based simulations of heterogeneous agents,
persists events and entities to a local OMPA vault, and exposes a FastAPI
HTTP layer with optional API-key auth and sliding-window rate limiting.

It is a small, transparent tool aimed at research and experimentation rather
than production workloads.

## Technical scope

### Architecture

- `SwarmOrchestrator` — round-based execution with pause / resume / stop
  semantics, per-simulation async locks, and pluggable event callbacks.
- Agent system — `BaseAgent` with pluggable subclasses (`DefaultAgent`,
  `InfluencerAgent`, `LurkerAgent`) and a factory for registering new
  types.
- `OMPAMemoryAdapter` — thin wrapper over an OMPA vault, unified surface for
  events, entities, and retrieval.
- FastAPI HTTP layer — REST + server-sent events; optional auth and rate
  limiting.
- CLI — `sandfish orchestrator`, `sandfish api`, `sandfish security-audit`.

### Engineering practices applied

- `async`/`await` throughout the orchestrator and agent loop.
- Typed data classes for configuration and results.
- Bounded `deque` for agent action history to prevent unbounded growth.
- pytest + pytest-asyncio test suite covering agents, orchestrator control
  flow, edge cases, and integration against a real (temporary) vault.
- Static security auditor with per-severity exit codes, driven by bandit
  when available.
- Docker build as a two-stage image running as a non-root user, with a
  `/health` probe.

## Tech stack

| Category | Tools |
| --- | --- |
| Language | Python 3.10+ |
| HTTP | FastAPI, uvicorn |
| Validation | pydantic v2 |
| Storage | OMPA vault (SQLite-backed) |
| Tests | pytest, pytest-asyncio |
| Security | bandit, safety |
| Packaging | hatchling |
| Container | Docker, Docker Compose |

## What I set out to build

A transparent, local-first sandbox for multi-agent simulations with:

1. No mandatory external services at runtime.
2. A small, readable codebase that someone can audit end-to-end.
3. A CLI and HTTP API that work the same way against the same engine.
4. Clear pause / resume / stop semantics so a long run can be stepped
   through rather than restarted.

## What I would not claim

- That it outperforms any specific alternative — no controlled benchmark
  comparison has been run for this repository.
- That it is production-ready. It is alpha-quality and single-process.
- That it provides hardened multi-tenant isolation. It does not.

## Current status

- Version 0.1.0; MIT licensed.
- Test suite runs green on Windows and Linux (54 tests, including
  async + integration).
- Not yet published to PyPI; install from source.
- No published Docker image; the Dockerfile and Compose file build locally.

## Links

- Source: this repository.
- License: [MIT](../LICENSE)
- Architecture write-up: [WHITEPAPER.md](WHITEPAPER.md)
- Install notes: [INSTALL.md](INSTALL.md)
- Performance notes: [PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md)
