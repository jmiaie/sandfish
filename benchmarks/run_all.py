"""
SandFish Benchmark Suite

Run all performance benchmarks and generate report.
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sandfish.core.orchestrator import create_orchestrator, SimulationConfig
from sandfish.agents.base_agent import create_agent
from sandfish.memory.ompa_adapter import OMPAMemoryAdapter


class BenchmarkRunner:
    """Runs performance benchmarks."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
        
    async def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("🌵 SandFish Benchmark Suite")
        print("=" * 50)
        
        self.results['startup'] = await self.benchmark_startup()
        self.results['agent_creation'] = await self.benchmark_agent_creation()
        self.results['simulation'] = await self.benchmark_simulation()
        self.results['memory'] = await self.benchmark_memory()
        
        return self.results
    
    async def benchmark_startup(self) -> Dict[str, float]:
        """Benchmark startup time."""
        print("\n📊 Benchmarking startup...")
        
        times = []
        for _ in range(5):
            start = time.time()
            orch = create_orchestrator("/tmp/bench-vault")
            elapsed = time.time() - start
            times.append(elapsed)
        
        return {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    async def benchmark_agent_creation(self) -> Dict[str, float]:
        """Benchmark agent creation throughput."""
        print("\n📊 Benchmarking agent creation...")
        
        memory = OMPAMemoryAdapter("/tmp/bench-vault")
        counts = [10, 100, 500]
        results = {}
        
        for count in counts:
            start = time.time()
            for i in range(count):
                agent = create_agent("default", memory_adapter=memory)
                await agent.initialize({})
            elapsed = time.time() - start
            
            results[f'{count}_agents'] = {
                'time': elapsed,
                'throughput': count / elapsed
            }
        
        return results
    
    async def benchmark_simulation(self) -> Dict[str, float]:
        """Benchmark simulation performance."""
        print("\n📊 Benchmarking simulation...")
        
        orch = create_orchestrator("/tmp/bench-vault")
        config = SimulationConfig(
            name="Benchmark",
            description="Performance test",
            max_rounds=100,
            num_agents=50
        )
        
        sim_id = orch.create_simulation(config)
        
        start = time.time()
        result = await orch.run_simulation(sim_id)
        elapsed = time.time() - start
        
        return {
            'total_time': elapsed,
            'rounds_per_sec': result.rounds_completed / elapsed,
            'agents_per_sec': (result.rounds_completed * config.num_agents) / elapsed
        }
    
    async def benchmark_memory(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        print("\n📊 Benchmarking memory...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline
        baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        # After creating agents
        memory = OMPAMemoryAdapter("/tmp/bench-vault")
        agents = []
        for i in range(100):
            agent = create_agent("default", memory_adapter=memory)
            await agent.initialize({})
            agents.append(agent)
        
        with_agents = process.memory_info().rss / 1024 / 1024
        
        return {
            'baseline_mb': baseline,
            'with_100_agents_mb': with_agents,
            'per_agent_mb': (with_agents - baseline) / 100
        }
    
    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "# SandFish Benchmark Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"**Startup Time**: {self.results['startup']['mean']:.3f}s (mean)",
            f"**Agent Creation**: {self.results['agent_creation']['100_agents']['throughput']:.1f} agents/sec",
            f"**Simulation**: {self.results['simulation']['rounds_per_sec']:.1f} rounds/sec",
            f"**Memory**: {self.results['memory']['per_agent_mb']:.2f} MB/agent",
            "",
            "## Raw Results",
            "",
            "```json",
            json.dumps(self.results, indent=2),
            "```"
        ]
        
        return '\n'.join(lines)


async def main():
    """Main entry point."""
    runner = BenchmarkRunner()
    results = await runner.run_all()
    
    # Print report
    report = runner.generate_report()
    print("\n" + "=" * 50)
    print(report)
    
    # Save to file
    output_file = Path("benchmark_report.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
