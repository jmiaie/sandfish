"""
SandFish benchmark suite.

Runs basic performance benchmarks and writes a JSON report. Uses a temporary
directory for the vault so the suite works cross-platform (Windows included)
and leaves no artifacts behind.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from sandfish.agents.base_agent import create_agent
from sandfish.core.orchestrator import SimulationConfig, create_orchestrator
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


class BenchmarkRunner:
    """Runs performance benchmarks against an ephemeral vault."""

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = vault_root
        self.results: dict[str, Any] = {}

    def _vault(self, name: str) -> str:
        path = self.vault_root / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    async def run_all(self) -> dict[str, Any]:
        print("SandFish Benchmark Suite")
        print("=" * 50)
        self.results["startup"] = await self.benchmark_startup()
        self.results["agent_creation"] = await self.benchmark_agent_creation()
        self.results["simulation"] = await self.benchmark_simulation()
        self.results["memory"] = await self.benchmark_memory()
        return self.results

    async def benchmark_startup(self) -> dict[str, float]:
        print("\nBenchmarking startup...")
        times = []
        for i in range(5):
            start = time.perf_counter()
            create_orchestrator(self._vault(f"startup-{i}"))
            times.append(time.perf_counter() - start)
        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }

    async def benchmark_agent_creation(self) -> dict[str, Any]:
        print("\nBenchmarking agent creation...")
        memory = OMPAMemoryAdapter(self._vault("agents"))
        results: dict[str, Any] = {}
        for count in (10, 100, 500):
            start = time.perf_counter()
            for _ in range(count):
                agent = create_agent("default", memory_adapter=memory)
                await agent.initialize({})
            elapsed = time.perf_counter() - start
            results[f"{count}_agents"] = {
                "time": elapsed,
                "throughput": count / elapsed if elapsed > 0 else float("inf"),
            }
        return results

    async def benchmark_simulation(self) -> dict[str, float]:
        print("\nBenchmarking simulation...")
        orch = create_orchestrator(self._vault("simulation"))
        config = SimulationConfig(
            name="Benchmark",
            description="Performance test",
            max_rounds=100,
            num_agents=50,
        )
        sim_id = orch.create_simulation(config)

        start = time.perf_counter()
        result = await orch.run_simulation(sim_id)
        elapsed = time.perf_counter() - start

        return {
            "total_time": elapsed,
            "rounds_per_sec": result.rounds_completed / elapsed if elapsed > 0 else 0.0,
            "agents_per_sec": (result.rounds_completed * config.num_agents) / elapsed
            if elapsed > 0
            else 0.0,
        }

    async def benchmark_memory(self) -> dict[str, Any]:
        print("\nBenchmarking memory...")
        try:
            import os

            import psutil
        except ImportError:
            return {"note": "psutil not installed; skipping memory benchmark"}

        process = psutil.Process(os.getpid())
        baseline = process.memory_info().rss / 1024 / 1024  # MB

        memory = OMPAMemoryAdapter(self._vault("memory"))
        agents = []
        for _ in range(100):
            agent = create_agent("default", memory_adapter=memory)
            await agent.initialize({})
            agents.append(agent)

        with_agents = process.memory_info().rss / 1024 / 1024
        return {
            "baseline_mb": baseline,
            "with_100_agents_mb": with_agents,
            "per_agent_mb": (with_agents - baseline) / 100,
        }

    def generate_report(self) -> str:
        mem = self.results.get("memory", {})
        per_agent = mem.get("per_agent_mb")
        per_agent_line = (
            f"**Memory**: {per_agent:.2f} MB/agent" if per_agent is not None else "**Memory**: skipped"
        )
        lines = [
            "# SandFish Benchmark Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"**Startup Time**: {self.results['startup']['mean']:.3f}s (mean)",
            f"**Agent Creation**: {self.results['agent_creation']['100_agents']['throughput']:.1f} agents/sec",
            f"**Simulation**: {self.results['simulation']['rounds_per_sec']:.1f} rounds/sec",
            per_agent_line,
            "",
            "## Raw Results",
            "",
            "```json",
            json.dumps(self.results, indent=2),
            "```",
        ]
        return "\n".join(lines)


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sandfish-bench-") as tmp:
        runner = BenchmarkRunner(Path(tmp))
        await runner.run_all()

        report = runner.generate_report()
        print("\n" + "=" * 50)
        print(report)

        output_file = Path("benchmark_report.json")
        output_file.write_text(json.dumps(runner.results, indent=2), encoding="utf-8")
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
