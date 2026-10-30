# Distributed layout

This note is not a measured result. The module-by-module map, with tensor shapes and test status, is [CODE_MAP.md](../CODE_MAP.md).

`train.py` exits with status 2. S0.4 removed `DistributedPPOEngine`. The PPO update is `PPOTrainer` in `src/rlhf_platform/ppo_engine.py`. Connecting a run is S0.11.

What the remaining code does:

- `topology.py` stores actor, critic, reference, and reward placements. If `torch.distributed` is already initialized, it creates one process group per placement. It does not start FSDP or tensor parallel. `validate` rejects a rank that appears in both an active placement and an inference placement.
- `rollout.py` keeps generated tokens in a CPU list and can sample a response from a module that returns `.logits`. The background generator-thread class was removed in S0.4. Nothing tested it.
- `comm_hooks.py` and `async_io.py` import. No test runs a hook or writes a checkpoint. S0.9 decides whether those files stay.

An older draft of this page claimed a 40% GPU-memory reduction. That number was not a measurement. It is gone.

---

### Local cluster script (`scripts/simulate_runtime.py`)

Pytest does not collect this script. No passing run is recorded. S0.9 decides whether it stays. The script attempts the following:

**Simulation Architecture:**

- **Spawning Method:** `torch.multiprocessing.spawn` with 4 processes (default)
- **Backend:** Gloo (CPU-compatible; no NCCL/GPU required)
- **Network Simulation:** Localhost loopback with MASTER_ADDR=127.0.0.1, MASTER_PORT=29500

**Rank Workflow (Per Process):**

```
Rank Worker
├─ Initialize SimulationContext (rank, device, trace log)
├─ Init process group (gloo backend)
├─ Barrier 1: Synchronize all ranks
├─ Load mock asymmetric topology
│  ├─ Active ranks: 0, 1 (Actor/Critic)
│  └─ Inference ranks: 2, 3 (Reference/Reward)
├─ Initialize topology process groups
├─ Barrier 2: Synchronize after group creation
├─ Iteration loop:
│  ├─ If active rank:
│  │  ├─ Generate 2 mock rollouts (random tensors, pinned CPU)
│  │  ├─ Enqueue to RolloutBuffer
│  │  ├─ Sample 1 rollout from buffer
│  │  └─ Non-blocking device transfer: rollout.to(device)
│  ├─ Else (inference rank):
│  │  └─ Idle loop (simulates inference group)
│  └─ Every 3 iterations: dist.barrier()
├─ Final barrier
├─ Destroy process group
└─ Write trace log to JSON: trace_rank_<N>.json
```

**Validation Checks:**

1. **No Deadlocks:** All ranks reach all barriers consistently
2. **Correct Topology:** Active ranks (0,1) vs. inference ranks (2,3) properly segregated
3. **Thread Safety:** RolloutBuffer + lock mechanism prevents race conditions
4. **Non-Blocking Transfers:** Pinned tensors move to device without blocking other ranks
5. **Rank Ordering:** Trace logs show expected sequence (no out-of-order operations)

**Trace Log Output (JSON per rank):**

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
    "[T=0.020] Rank is in ACTIVE group; initializing rollout buffer.",
    "[T=0.021] Starting mock rollout generation loop.",
    "[T=0.022] Iteration 0: Generated 2 rollouts. Buffer size: 2.",
    "[T=0.023] Iteration 0: Sampled 1 rollout from buffer.",
    "[T=0.024] Iteration 0: Non-blocking transfer completed."
  ]
}
```

**Exit Codes:**

- `0`: All ranks completed without deadlocks; topology and buffers validated successfully
- `1`: Error detected (deadlock, group creation failure, or exception); traces written to output directory

**Usage:**

```bash
# Basic run
python scripts/simulate_runtime.py

# Custom parameters
python scripts/simulate_runtime.py --num-ranks 4 --iterations 20 --output-dir /tmp/traces

# Check traces
cat /tmp/rlhf_simulate_runtime/trace_rank_0.json
```

---

## What is not wired

`python train.py` exits 2. It does not load a topology, start a rollout, or call `PPOTrainer`.

`scripts/simulate_runtime.py` is a separate script. Pytest does not run it. A trace file from that script is not a recorded result unless it is committed under `benchmarks/results/`.

---

## Module Dependency Graph

```
train.py
  exits 2. It does not import the modules below.

The modules that still exist:
  ├─ rlhf_platform.distributed.topology
  │   └─ ClusterTopology, TopologyBuilder, load_topology_from_yaml
  │       └─ torch.distributed.new_group()
  │
  ├─ rlhf_platform.distributed.async_io
  │   └─ DistributedCheckpointManager, AsyncCheckpointWriter
  │       └─ Pinned CPU buffers, background thread I/O
  │
  ├─ rlhf_platform.distributed.comm_hooks
  │   └─ CommunicationHooks (overlap, clipping, NaN detection)
  │       └─ torch.distributed communication operations
  │
  ├─ rlhf_platform.alignment.rollout
  │   ├─ RolloutBuffer (CPU list, pin only if CUDA exists)
  │   └─ RolloutGenerator (autoregressive sampling)
  │
  ├─ rlhf_platform.ppo_engine
  │   └─ PPOTrainer (the only PPO update)
  │
  └─ rlhf_platform.utils.telemetry
      ├─ RankAwareTelemetry (structured JSON logging)
      ├─ NCCLProfiler (collective comm profiling)
      └─ DistributedMetricsCollector (rank aggregation)

scripts/simulate_runtime.py
  ├─ torch.multiprocessing.spawn (4 processes)
  ├─ rlhf_platform.distributed.topology
  ├─ rlhf_platform.alignment.rollout
  └─ rlhf_platform.utils.telemetry
```

---

## Configuration Files

### `configs/cluster_topology.yaml`
Defines hardware layout and model placement:

```yaml
cluster:
  name: "rlhf-cluster-v1"
  num_nodes: 2
  gpus_per_node: 8
  total_gpus: 16

models:
  actor:
    model_name: "opt-1.3b"
    num_gpus: 8
    parallelism_strategy: "fsdp_tp"
    tp_degree: 2
    dp_degree: 4
    precision: "bfloat16"
    devices: [0, 1, 2, 3, 4, 5, 6, 7]
    node_ids: [0, 1]
  
  critic:
    model_name: "opt-1.3b"
    num_gpus: 4
    parallelism_strategy: "fsdp_tp"
    tp_degree: 1
    dp_degree: 4
    precision: "bfloat16"
    devices: [0, 1, 2, 3]
    node_ids: [0]
  
  reference:
    model_name: "opt-1.3b"
    num_gpus: 2
    parallelism_strategy: "inference"
    precision: "float32"
    devices: [4, 5]
    node_ids: [0]
  
  reward:
    model_name: "deberta-v3-large"
    num_gpus: 2
    parallelism_strategy: "inference"
    precision: "float32"
    devices: [6, 7]
    node_ids: [0, 1]
```

### `configs/deepspeed_zero3.yaml`
ZeRO-3 optimizer configuration (optional):

```yaml
train_batch_size: 32
train_micro_batch_size_per_gpu: 1

optimizer:
  type: "AdamW"
  params:
    lr: 1e-5
    betas: [0.9, 0.999]
    eps: 1e-8
    weight_decay: 0.01

zero_optimization:
  stage: 3
  offload_optimizer:
    device: "cpu"
  offload_param:
    device: "cpu"
```

---

## Key Design Patterns

### 1. Asymmetric Topology
```python
# Active ranks and inference ranks must be disjoint. validate() asserts that.
topology.actor_placement.device_list      # example: [0, 1, 2, 3]
topology.reference_placement.device_list  # example: [4, 5]
```

### 2. Thread-Safe Async Buffer
```python
# Producer (background thread)
rollout = generator.generate_rollout(query)
buffer.add(rollout)  # Thread-safe with lock

# Consumer (training thread)
sampled = buffer.sample_batch(batch_size)  # list of Rollout, still on CPU
```

### 3. Communication Overlapping
```python
# Register hook: gradients fire all_reduce during backward
CommunicationHooks.register_overlap_hook(actor_model)

# During backward, gradient buckets trigger:
# all_reduce(bucket) → async_op=True → wait() later
```

### 4. Structured Telemetry
```python
# All events are JSON serializable
telemetry.log_ppo_step(
    step=100,
    actor_loss=0.52,
    kl_divergence=0.015,
    duration_ms=234.5
)
# Output: {"step": 100, "actor_loss": 0.52, ...}
```

---

## Next Steps for Users

1. **Validate Locally:**
   ```bash
   python scripts/simulate_runtime.py --num-ranks 4 --iterations 50
   ```

2. **Prepare Hardware:**
   - 2+ nodes with 8 GPUs per node (A100 / H100 recommended)
   - Low-latency InfiniBand fabric
   - Synchronized NTP clocks

3. **Configure Topology:**
   - Edit `configs/cluster_topology.yaml` to match your cluster layout

4. **Deploy:**
   ```bash
   # train.py exits 2. It is not a launch command. See S0.11.
   ```

5. **Monitor:**
   - Watch telemetry logs for KL divergence, loss trends
   - Check memory usage and communication latency in NCCL profiler output

---

## Files Summary

| File | Purpose |
|------|---------|
| `train.py` | Exits 2. Does not call `PPOTrainer` |
| `scripts/simulate_runtime.py` | Uncollected multi-rank script. S0.9 |
| `scripts/README.md` | Usage guide and troubleshooting |
| `ROADMAP.md` | Work status, notes, and change log |
| `src/rlhf_platform/distributed/topology.py` | Asymmetric topology definition and group creation |
| `src/rlhf_platform/distributed/async_io.py` | Non-blocking checkpoint streaming |
| `src/rlhf_platform/distributed/comm_hooks.py` | Communication overlapping and safety checks |
| `src/rlhf_platform/alignment/rollout.py` | CPU rollout buffer and token sampler |
| `src/rlhf_platform/ppo_engine.py` | `PPOTrainer`, the only PPO update |
| `src/rlhf_platform/utils/telemetry.py` | Structured observability and profiling |
| `configs/cluster_topology.yaml` | Hardware layout and model placement |
| `configs/deepspeed_zero3.yaml` | Optional ZeRO-3 configuration |

