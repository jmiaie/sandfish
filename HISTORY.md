# Project History

## Version Timeline

- **SandFish (v1)** — Multi-agent swarm intelligence prototype. This repository.
  Python 3.10+, FastAPI, OMPA local-first memory, containerized deployment,
  automated security audit tooling, production-grade test suite.

- **AegisFlow (v2)** — Architecture iteration exploring stricter sandbox boundaries,
  a universal memory bridge (GBrain + OMPA integration), and an explicit
  lead-agent / sub-agent delegation model.
  Separate repository: `jmiaie/aegisflow` *(archived; see AegisFlow skill in workspace)*.

## Audit Note

During the v1 → v2 transition, two AegisFlow scaffold commits were briefly
merged into this repository before being moved to their intended home:

| SHA       | Summary                                             |
|-----------|-----------------------------------------------------|
| `fb168e6` | Merge PR #1: scaffold AegisFlow core architecture   |
| `a70f771` | Purge SandFish references from AegisFlow main       |

Those commits are superseded by the ongoing work in `jmiaie/aegisflow` *(archived)*
and are recorded here for transparency. This repository was reset to preserve
SandFish in its original, standalone form as a portfolio reference.
