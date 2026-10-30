# RLHF Platform Execution & Simulation Scripts

This directory contains utilities for running and validating the distributed RLHF platform.

## Files

### `train.py` (repository root)

`train.py` exits with status 2. S0.4 removed `DistributedPPOEngine`, and this script does not call `PPOTrainer`. The PPO update is `rlhf_platform.ppo_engine.PPOTrainer`. Connecting a run is S0.11 in `ROADMAP.md`.

---

### `simulate_runtime.py` (Scripts)
**Local cluster emulation script.** Pytest does not collect it. No passing run is recorded in this repo. S0.9 decides whether it stays. The script tries to:

- Asymmetric process group creation without deadlocks
- Thread-safe rollout buffer operations with pinned CPU memory
- Non-blocking tensor transfers across ranks
- Communication hook integration in distributed context
- Barrier synchronization and rank ordering

#### Usage

**Basic (4 ranks, 10 iterations):**
```bash
python scripts/simulate_runtime.py
```

**Custom configuration:**
```bash
python scripts/simulate_runtime.py \
  --num-ranks 4 \
  --iterations 20 \
  --output-dir /tmp/rlhf_traces
```

#### Arguments

- `--num-ranks`: Number of simulated ranks (default: 4)
- `--iterations`: Iterations per rank (default: 10)
- `--output-dir`: Output directory for trace logs (default: temp directory)

#### Output

Generates JSON trace files for each rank (e.g., `trace_rank_0.json`, `trace_rank_1.json`, etc.) containing:

- Rank and world size metadata
- Timestamped trace entries for each operation (barrier, buffer ops, tensor transfers)
- Any errors or deadlock indicators

**Trace shape** (illustration of the JSON fields, not a recorded run):
```json
{
  "rank": 0,
  "world_size": 4,
  "trace": [
    "[T=0.001] Starting rank worker initialization.",
    "[T=0.002] Initializing distributed process group with gloo backend.",
    "[T=0.015] Distributed process group initialized successfully.",
    "[T=0.016] All ranks synchronized at barrier 1.",
    "[T=0.017] Loading mock asymmetric topology.",
    "[T=0.018] Topology groups initialized successfully.",
    "[T=0.019] All ranks synchronized at barrier 2.",
    "[T=0.020] Iteration 0: Generated 2 rollouts. Buffer size: 2."
  ]
}
```

#### Exit Codes

- `0`: All ranks completed successfully; no deadlocks detected.
- `1`: Simulation failed; rank-local trace logs available in output directory.

---

## What is not a run

`python train.py` prints a message and exits 2. It does not train.

`scripts/simulate_runtime.py` is not part of `pytest`. S0.9 is the item that tests the process-group split, the communication hooks, and the checkpoint writer, or removes them.
