AegisFlow is a universal, agnostic, transparency-focused, and security-first multi-agent hybrid system. It combines the persistent memory structures of OMPA, the security-first swarm intelligence of AegisFlow, the powerful sub-agent delegation and sandbox isolation of DeerFlow, and the canonical testing capabilities inspired by Claw Code.

## Core Pillars

1. **Universal & Agnostic:** Designed to work with any agent framework, LLM, or environment. Memory and orchestration layers are completely decoupled from underlying providers.
2. **Transparency:** Every decision, memory access, and tool invocation is fully auditable. No hidden prompts or opaque state manipulations.
3. **Security-First:** Secure-by-default execution. Tools and sub-agents operate inside strict, isolated sandbox environments with explicit permission boundaries.
4. **Swarm Intelligence:** Complex tasks are decomposed and delegated to specialized sub-agents running in parallel, orchestrated by a lead agent.
5. **Canonical Integration:** Capable of interacting with external execution harnesses like claw-code for verified runtime environments.

## System Components

### 1. The Universal Memory Layer (Inspired by OMPA)
A framework-agnostic persistent memory system that never forgets and survives context compaction.

- **Vault:** Human-navigable markdown storage (Brain, Work, Org, Perf).
- **Palace:** Agent-accessible metadata and spatial navigation (Wings, Rooms, Drawers).
- **Knowledge Graph:** Temporal triples (Subject -> Predicate -> Object) stored locally, enabling complex relationship queries.
- **Verbatim Storage:** Ensures zero summarization loss for critical context.

### 2. The Security-First Execution Sandbox (Inspired by DeerFlow)
Every agent has access to a strictly isolated environment, not just a conversational interface.

- **Containerized Sandboxing:** Agents execute code and tools within isolated environments with scoped filesystem access (`/workspace`, `/uploads`, `/outputs`).
- **Strict Permission Scopes:** Granular control over network access, shell execution, and file modification.
- **Audit Trails:** Complete logging of all sandbox interactions for transparency and security review.

### 3. Swarm Orchestration & Delegation (Inspired by AegisFlow & DeerFlow)
The engine that enables complex, multi-step problem solving.

- **Lead Agent:** The primary orchestrator that breaks down tasks and delegates them.
- **Sub-Agents:** Specialized agents spawned on the fly with isolated, scoped context. They execute in parallel and return structured results.
- **Context Boundaries:** Sub-agents only see what they need to see, preventing context contamination and reducing token bloat.
- **Zero Token Burn (ZTB) Optimization:** Aggressive context management and local semantic caching to minimize unnecessary API calls.

## Directory Structure

```
aegisflow/
├── core/           # Core interfaces and universal types
├── memory/         # OMPA integration (Vault, Palace, Knowledge Graph)
├── sandbox/        # Isolated execution environments and tool runtimes
├── orchestration/  # Lead agent and sub-agent lifecycle management
```

## Data Flow Example

1. **Input:** User provides a complex task.
2. **Memory Retrieval:** Lead Agent queries the Knowledge Graph and Palace for relevant past context.
3. **Decomposition:** Lead Agent breaks the task into sub-tasks.
4. **Delegation:** Lead Agent spawns Sub-Agents, injecting specific scoped context and tool permissions.
5. **Execution:** Sub-Agents perform tasks within the Sandbox (e.g., executing code, scraping).
6. **Synthesis:** Sub-Agents return structured data; Lead Agent synthesizes the final response.
7. **Storage:** The session is summarized, and new facts/decisions are permanently saved to the Vault and Knowledge Graph.
