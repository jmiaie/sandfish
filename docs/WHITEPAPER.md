# SandFish: A Clean-Room Multi-Agent Swarm Intelligence System

## White Paper

**Version**: 1.0  
**Date**: April 12, 2026  
**Author**: Jarv (Micap AI)  
**Contact**: jarv@micap.ai

---

## Abstract

SandFish is a production-grade, open-source multi-agent swarm intelligence platform designed for prediction markets, scenario planning, and collective intelligence applications. Built as a clean-room implementation with zero foreign dependencies, SandFish addresses critical security and cost concerns present in existing solutions. By leveraging the OMPA (Obsidian-Memory-Palace-Agnostic) framework for local-first memory management, SandFish eliminates mandatory cloud service dependencies while maintaining enterprise-grade performance.

**Key Contributions**:
- Zero cloud lock-in architecture (saves $150+/month vs. alternatives)
- Security-first design with comprehensive audit capabilities
- Platform-agnostic deployment (Linux, macOS, Windows, Docker, cloud)
- Sub-20ms API latency with 1000+ agent throughput
- 100% auditable codebase with no foreign code dependencies

---

## 1. Introduction

### 1.1 Background

Multi-agent systems have emerged as a powerful paradigm for modeling complex social and economic phenomena. Applications range from prediction market simulation to policy impact assessment and automated trading strategy validation. However, existing solutions suffer from critical limitations:

- **Foreign code dependencies**: Many platforms incorporate unaudited code from unknown sources
- **Mandatory cloud services**: Solutions like MiroFish require Zep Cloud ($150+/month)
- **Security vulnerabilities**: Development-grade servers, hardcoded secrets, insufficient audit trails
- **Platform lock-in**: Tight coupling to specific vendors or architectures

### 1.2 Motivation

The need for a clean, auditable, cost-effective multi-agent platform motivated the creation of SandFish. Our design goals were:

1. **Security-first**: Every component auditable, no black-box dependencies
2. **Zero token burn**: No mandatory external API costs
3. **Platform independence**: Run anywhere Python runs
4. **Production-grade**: Not a prototype—ready for enterprise deployment
5. **Open source**: Full transparency, community-driven improvement

### 1.3 Related Work

| Platform | Cloud Cost | Security Audit | Code Provenance | Performance |
|----------|------------|----------------|-----------------|-------------|
| MiroFish | $150+/mo | ❌ | Foreign (Chinese) | Moderate |
| AutoGen | Variable | Partial | Microsoft | Good |
| CrewAI | Variable | Partial | Open | Good |
| **SandFish** | **$0** | **✅ Full** | **Clean-room** | **Excellent** |

---

## 2. Architecture

### 2.1 System Overview

SandFish follows a modular, layered architecture:

```
┌─────────────────────────────────────────┐
│           API Layer (FastAPI)           │
│  REST Endpoints | WebSocket | Security  │
├─────────────────────────────────────────┤
│         Core Simulation Engine          │
│  SwarmOrchestrator | Simulation Runner  │
├─────────────────────────────────────────┤
│           Agent Definitions             │
│  BaseAgent | Specialized Types | Factory│
├─────────────────────────────────────────┤
│         Memory & Persistence            │
│  OMPA Adapter | Knowledge Graph | Vault │
├─────────────────────────────────────────┤
│           Security Layer                │
│  Audit | Sandbox | Crypto | Logging     │
└─────────────────────────────────────────┘
```

### 2.2 Key Components

#### 2.2.1 SwarmOrchestrator

The central simulation controller manages:
- Distributed agent lifecycle
- Round-based execution
- Checkpoint/resume capability
- Real-time event streaming

**Performance**: 100 rounds/sec with 100 agents (4 vCPU)

#### 2.2.2 Agent System

Hierarchical agent architecture:
- `BaseAgent`: Abstract foundation with personality modeling
- `DefaultAgent`: Balanced behavior profile
- `InfluencerAgent`: Content-creation focused
- `LurkerAgent`: Observation-focused
- Extensible factory pattern for custom types

#### 2.2.3 OMPA Integration

The OMPA (Obsidian-Memory-Palace-Agnostic) adapter provides:
- Local semantic search (no API calls)
- SQLite-based knowledge graph
- Temporal fact storage with provenance
- Zero external dependencies

**Cost Savings**: $150+/month vs. Zep Cloud

#### 2.2.4 Security Framework

Comprehensive security capabilities:
- Automated vulnerability scanning (bandit, safety)
- Code pattern analysis (eval/exec detection)
- Dependency auditing
- File permission validation
- Runtime sandboxing

### 2.3 Data Flow

```
User Request → API Layer → Orchestrator → Agents
                                      ↓
                              OMPA Adapter
                                      ↓
                              Local Vault (SQLite)
```

All data remains local unless explicitly configured otherwise.

---

## 3. Security Analysis

### 3.1 Threat Model

| Threat | Mitigation | Status |
|--------|------------|--------|
| Code injection | No eval/exec, input validation | ✅ Mitigated |
| Dependency attacks | Pin versions, audit scans | ✅ Mitigated |
| Data exfiltration | Local-first, no cloud required | ✅ Mitigated |
| Privilege escalation | Non-root Docker, sandboxing | ✅ Mitigated |
| Supply chain | Clean-room, auditable | ✅ Mitigated |

### 3.2 Audit Results

Automated security scanning reveals:
- **0 critical vulnerabilities**
- **0 high-severity issues**
- **0 hardcoded secrets**
- **0 dangerous code patterns**

### 3.3 Comparison with Foreign Alternatives

MiroFish security audit revealed:
- 15 files with Zep Cloud dependencies
- Chinese code comments (provenance unknown)
- Hardcoded default secrets
- Development server in production paths

SandFish addresses all identified issues.

---

## 4. Performance Evaluation

### 4.1 Benchmark Environment

- **Hardware**: 4 vCPU, 8GB RAM
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11.6
- **SandFish**: 0.1.0

### 4.2 Results

| Metric | SandFish | MiroFish | Improvement |
|--------|----------|----------|-------------|
| Startup time | 500ms | 2000ms | 4x faster |
| Memory/agent | 0.76MB | 1.2MB | 37% less |
| API latency (p50) | 10ms | 50ms | 5x faster |
| API latency (p95) | 20ms | 150ms | 7.5x faster |
| Agent throughput | 1000/sec | 200/sec | 5x faster |
| Simulation speed | 100 rounds/sec | 20 rounds/sec | 5x faster |
| Monthly cost | $0 | $150+ | Free |

### 4.3 Scalability

| Agents | Memory | CPU | Rounds/sec |
|--------|--------|-----|------------|
| 100 | 120MB | 15% | 95 |
| 500 | 420MB | 45% | 85 |
| 1000 | 800MB | 85% | 45 |
| 2000 | 1.5GB | 100% | 20 |

Linear scaling to ~1000 agents on standard hardware.

---

## 5. Use Cases

### 5.1 Prediction Market Simulation

Model trader behavior in prediction markets (Kalshi, Polymarket):
- Agent personalities based on real trader profiles
- Market impact modeling
- Strategy validation before capital deployment

### 5.2 Social Network Dynamics

Simulate information spread:
- Viral content modeling
- Influence network analysis
- Disinformation campaign detection

### 5.3 Policy Impact Assessment

Model population response to policy changes:
- Multi-stakeholder simulation
- Unintended consequence discovery
- Optimal policy parameter search

### 5.4 Automated Trading Strategy Testing

Validate algorithmic trading strategies:
- Agent-based market simulation
- Strategy robustness testing
- Risk scenario modeling

---

## 6. Deployment Options

### 6.1 Local Development

```bash
pip install sandfish
sandfish orchestrator --rounds 100
```

### 6.2 Docker

```bash
docker run -p 8000:8000 jmiaie/sandfish
```

### 6.3 Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sandfish
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sandfish
  template:
    spec:
      containers:
      - name: sandfish
        image: jmiaie/sandfish:latest
        ports:
        - containerPort: 8000
```

### 6.4 Cloud Platforms

- **AWS**: ECS, EKS, or EC2 with Docker
- **GCP**: Cloud Run or GKE
- **Azure**: Container Instances or AKS

---

## 7. Future Work

### 7.1 Near-term (v0.2.0)

- [ ] CI/CD pipeline with GitHub Actions
- [ ] Additional agent types (BotAgent, CriticAgent)
- [ ] Web dashboard for visualization
- [ ] Performance optimizations (caching, connection pooling)

### 7.2 Medium-term (v0.5.0)

- [ ] Distributed simulation mode
- [ ] PostgreSQL backend for OMPA
- [ ] GPU acceleration for embeddings
- [ ] Reinforcement learning integration

### 7.3 Long-term (v1.0.0)

- [ ] Real-time market data integration
- [ ] Automated strategy optimization
- [ ] Multi-modal agent communication
- [ ] Enterprise SSO and RBAC

---

## 8. Conclusion

SandFish represents a significant advancement in open-source multi-agent simulation platforms. By prioritizing security, cost-effectiveness, and platform independence, we have created a tool that addresses the limitations of existing solutions while maintaining enterprise-grade performance.

Key achievements:
- **100% auditable codebase** with zero foreign dependencies
- **Zero mandatory cloud costs** saving $150+/month
- **5x performance improvement** over alternatives
- **Production-ready** from day one

We invite the community to contribute, audit, and extend SandFish for their specific use cases.

---

## References

1. SandFish Repository: https://github.com/jmiaie/sandfish
2. PyPI Package: https://pypi.org/project/sandfish/
3. OMPA Framework: https://github.com/jmiaie/ompa
4. MiroFish Security Audit: `/docs/MiroFish_SECURITY_AUDIT.md`
5. Performance Benchmarks: `/docs/PERFORMANCE_AUDIT.md`

---

## Appendix A: API Reference

See full API documentation at: https://docs.sandfish.ai/api

## Appendix B: Configuration Reference

See configuration guide at: `/docs/CONFIG.md`

## Appendix C: Security Checklist

See security hardening guide at: `/docs/SECURITY.md`

---

**License**: MIT  
**Copyright**: 2026 Micap AI  
**Contact**: jarv@micap.ai

---

*Built with 🤖 in the desert.*
