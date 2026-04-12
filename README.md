# SandFish

**A Clean-Room Multi-Agent Swarm Intelligence System**

Built from scratch with security-first principles, zero foreign dependencies, and native OMPA integration.

## Overview

SandFish is a distributed multi-agent simulation engine for prediction, scenario planning, and collective intelligence. Unlike foreign alternatives, SandFish is:

- **100% auditable** — All code in English, fully documented
- **Zero cloud lock-in** — OMPA-native memory (no Zep, no external APIs required)
- **Security-first** — Production-grade from day one
- **ZTB compliant** — Zero token burn, minimal operational cost

## Architecture

```
SandFish/
├── core/           # Simulation engine
├── memory/         # OMPA integration
├── agents/         # Swarm agent definitions
├── api/            # FastAPI REST/WebSocket
├── security/       # Audit, sandbox, crypto
└── tests/          # Comprehensive coverage
```

## Quick Start

```bash
# Install
pip install sandfish

# Configure
export OMPA_VAULT_PATH="/path/to/vault"
export LLM_API_KEY="your-key"

# Run
sandfish orchestrator --config config.yaml
```

## Security

See [SECURITY.md](docs/SECURITY.md) for audit reports and hardening guide.

## License

MIT — See [LICENSE](LICENSE)

---

Built with 🤖 in the desert.
