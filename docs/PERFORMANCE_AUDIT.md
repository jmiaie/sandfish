# SandFish Performance Notes

**Status**: informal notes, not a formal audit.
**Version**: 0.1.0

SandFish has not been benchmarked in a controlled environment for this
repository. Rather than publish numbers we cannot defend, this document
describes what to measure, how to measure it, and the known hotspots worth
watching. If you want numbers for your use case, run the included benchmark
script on your target hardware.

## Running the benchmarks

```bash
pip install -e '.[dev]' psutil
python benchmarks/run_all.py
```

The runner creates a temporary vault, measures:

1. Orchestrator startup time (5 trials, mean/min/max).
2. Agent creation throughput at 10 / 100 / 500 agents.
3. A 100-round simulation with 50 agents.
4. Per-agent RSS delta for 100 agents (requires `psutil`; skipped otherwise).

Results are written to `benchmark_report.json` and printed as markdown.

## Profiling

Standard Python tools work. Two recipes:

```bash
# CPU profile a simulation run
python -m cProfile -o profile.stats -m sandfish.cli orchestrator --rounds 100 --agents 50
python -m pstats profile.stats

# Memory profile (requires `memory_profiler`)
mprof run sandfish orchestrator --agents 1000 --rounds 100
mprof plot
```

## Known hotspots

1. **OMPA search.** Agents that call `memory.search` during
   `decide_action` pay whatever the vault backend costs per call. In
   unindexed or semantic-search workloads this dominates the round loop.
2. **Serial action application.** `_execute_round` gathers decisions in
   parallel but applies them sequentially so that shared state stays
   deterministic. For very large agent populations this becomes the
   bottleneck; a batch-apply path would help if determinism can be
   relaxed.
3. **Event callbacks.** `_emit_event` iterates every registered callback
   per event. Slow or blocking callbacks serialise the whole round; keep
   them fast or off-load work.
4. **Checkpoint writes.** If `--checkpoint-dir` is set and
   `checkpoint_interval` is small, every Nth round does a JSON write. The
   write is best-effort (logged on failure), but a tight interval will
   still show up in wall-clock time.

## Known memory considerations

- `AgentState.action_history` uses a bounded `deque`; the default cap
  prevents unbounded growth even on long runs.
- The orchestrator keeps all agents for a simulation in `self.agents`
  keyed by `agent.id`. This is fine for hundreds of agents; for tens of
  thousands you will want to swap in lazy loading or split runs.
- OMPA's vault footprint scales with recorded events and entities. For
  long campaigns, rotate or compact the vault between runs.

## Suggested optimisations (not implemented)

The bits below are deferred until benchmark data justifies them.

- **Search cache.** Agents often repeat the same retrieval queries
  round-over-round. A small LRU cache with a short TTL in front of
  `memory.search` would help workloads dominated by retrieval.
- **Batch apply.** For simulations where action ordering does not matter,
  an opt-in batched-apply path could parallelise step 3 of the round loop.
- **Compact checkpoints.** Current checkpoints are JSON. Swapping to
  msgpack or zstd-compressed JSON would cut disk cost noticeably.
- **Distributed orchestrator.** Out of scope for v0.1.0. Would require a
  shared OMPA backend and an event bus.

## Scaling guidance

For a single process on commodity hardware, target populations in the
low-thousands of agents. Beyond that you are in territory where the
defaults (single-process asyncio loop, SQLite-backed OMPA vault) stop being
the right shape — consider a custom orchestrator subclass or a different
tool.

## What this document is not

- A comparison against other multi-agent frameworks. SandFish ships
  without such comparisons because we cannot reproduce them reliably here.
- A production SLA. Treat all numbers produced by `run_all.py` as rough
  signals, not guarantees.
