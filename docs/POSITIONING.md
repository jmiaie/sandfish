# Positioning: SandFish vs AegisFlow (`af` / `af_public`)

This note exists so portfolio viewers (and future maintainers) do **not**
dual-maintain the same multi-agent ideas across three repos.

## One-line roles

| Repo | Visibility | Role |
|------|------------|------|
| [`jmiaie/sandfish`](https://github.com/jmiaie/sandfish) | public | **v1 swarm demo** — round-based OMPA-backed simulation + FastAPI. Portfolio / recruiting surface. Hygiene only. |
| [`jmiaie/af`](https://github.com/jmiaie/af) | private | **AegisFlow canonical** — production-minded lead/sub-agent harness, sandbox, memory vault. Where deeper work belongs. |
| [`jmiaie/af_public`](https://github.com/jmiaie/af_public) | public | **AegisFlow public mirror** — discoverability / release showcase for AegisFlow. Prefer aligning with `af`, not forking features back into SandFish. |

Related archive (not active): private [`jmiaie/sandfish-legacy`](https://github.com/jmiaie/sandfish-legacy).

## Lineage

```
SandFish (v1)  ──this repo──►  public demo / history
        │
        └──► AegisFlow (v2) ──► jmiaie/af (canonical)
                              └─► jmiaie/af_public (public mirror)
```

See [HISTORY.md](../HISTORY.md) for the commit-level audit of the brief
scaffold merge into this tree and the reset that restored SandFish as a
standalone portfolio reference.

## What belongs where

| Change type | Land in |
|-------------|---------|
| SandFish README / HISTORY / positioning / cheap test fixes | **sandfish** |
| New orchestration, sandbox, memory-bridge, agent harness features | **af** (then mirror to **af_public** if public) |
| Portfolio copy that implies SandFish is the product | Rewrite — funnel to AegisFlow |

## Explicit non-goals for sandfish

- Do **not** port AegisFlow lead/sub-agent APIs into this tree.
- Do **not** treat `af_public` and `sandfish` as interchangeable products.
- Do **not** invent a `jmiaie/aegisflow` URL; that repo does not exist.
