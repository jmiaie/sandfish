# AegisFlow

**The Universal, Security-First Multi-Agent Hybrid System**

AegisFlow is a clean-room implementation of a distributed, multi-agent swarm intelligence system. It brings together the persistent memory structures of **OMPA**, the secure swarm capabilities of **AegisFlow**, and the powerful sub-agent sandbox environment of **DeerFlow** into a single, cohesive framework.

## Core Principles

- **Universal & Agnostic:** Pluggable at every layer. Works with any LLM, any runtime, any toolset.
- **Security-First:** Built-in sandbox execution, granular permissions, and total auditability.
- **Persistent Memory:** Temporal knowledge graphs and human-readable vaults ensure your agents never forget.
- **Swarm Delegation:** Complex tasks are automatically decomposed and delegated to isolated sub-agents.

## Architecture Highlights

AegisFlow operates on three primary axes:
1. **Memory (OMPA):** Vaults, Palaces, and Temporal Knowledge Graphs.
2. **Execution (DeerFlow):** Strict, containerized sandbox environments for tool use.
3. **Orchestration (AegisFlow):** Multi-agent coordination with strict context boundaries.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deep dive into the system design.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/aegisflow.git
cd aegisflow

# Install dependencies (requires Python 3.10+)
pip install -e .
```

## Quick Start (Coming Soon)

```python
from aegisflow import Orchestrator, MemoryVault, Sandbox

# Initialize the universal memory layer
memory = MemoryVault(path="./my_vault")

# Initialize the secure execution sandbox
sandbox = Sandbox(isolation="container", workspace="./workspace")

# Create the swarm orchestrator
swarm = Orchestrator(memory=memory, sandbox=sandbox)

# Execute a complex task
result = swarm.execute("Analyze the logs in ./workspace and summarize the security incidents.")
```

## License

MIT
