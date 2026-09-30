# Project History

## Version Timeline

- **SandFish (v1)** — Multi-agent swarm intelligence prototype. **This repository**
  ([jmiaie/sandfish](https://github.com/jmiaie/sandfish)).
  Python 3.10+, FastAPI, OMPA local-first memory, containerized deployment,
  automated security audit tooling, production-grade test suite.
  Kept as a **public swarm demo / portfolio reference**. Deeper product work
  does **not** continue here.

- **AegisFlow (v2)** — Architecture iteration: stricter sandbox boundaries,
  universal memory bridge (GBrain + OMPA), and explicit lead-agent / sub-agent
  delegation. Active homes:
  - Private canonical: [`jmiaie/af`](https://github.com/jmiaie/af) (AegisFlow)
  - Public mirror / showcase: [`jmiaie/af_public`](https://github.com/jmiaie/af_public)

- **sandfish-legacy** — Archived private snapshot from the v1→v2 transition.
  Not a maintenance target; the clean SandFish portfolio lives here.

> There is **no** `jmiaie/aegisflow` repository. Earlier notes that pointed there
> were wrong; use `af` / `af_public` above.

## Audit Note

During the v1 → v2 transition, two AegisFlow scaffold commits were briefly
merged into this repository before being moved to their intended home:

| SHA       | Summary                                             |
|-----------|-----------------------------------------------------|
| `fb168e6` | Merge PR #1: scaffold AegisFlow core architecture   |
| `a70f771` | Purge SandFish references from AegisFlow main       |

Those commits are superseded by work in `jmiaie/af` (and its public mirror
`jmiaie/af_public`) and are recorded here for transparency. This repository
was reset to preserve SandFish in its original, standalone form as a portfolio
reference — **not** as a second AegisFlow codebase to dual-maintain.

## Where to go next

| Goal | Repo |
|------|------|
| Browse / demo a small public swarm engine | **This repo** (`sandfish`) |
| Build on AegisFlow (canonical) | [`jmiaie/af`](https://github.com/jmiaie/af) (private) |
| Public AegisFlow surface | [`jmiaie/af_public`](https://github.com/jmiaie/af_public) |
| Lineage detail | [docs/POSITIONING.md](docs/POSITIONING.md) |
