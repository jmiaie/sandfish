import time
import os
import csv
from typing import Dict, Any, List

class BenchmarkHarness:
    """
    A unified benchmarking harness for evaluating AegisFlow against
    industry standards (e.g., SandFish, Mirofish) on core KPIs.
    """

    def __init__(self, output_dir: str = "./benchmarks/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def measure_execution_time(self, func, *args, **kwargs) -> float:
        """Measures the wall-clock execution time of a function."""
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        return end - start

    def record_result(self, framework: str, metric_name: str, value: float, unit: str, notes: str = ""):
        self.results.append({
            "framework": framework,
            "metric": metric_name,
            "value": value,
            "unit": unit,
            "notes": notes
        })
        print(f"[{framework}] {metric_name}: {value:.4f} {unit}")

    def export_to_csv(self, filename: str = "kpi_results.csv"):
        filepath = os.path.join(self.output_dir, filename)
        keys = ["framework", "metric", "value", "unit", "notes"]
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"Results exported to {filepath}")
