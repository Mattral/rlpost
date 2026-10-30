# What the training code calls

This note is not a measured result. The module-by-module map, with tensor shapes and test status, is [CODE_MAP.md](../CODE_MAP.md).

`train.py` exits with status 2. S0.4 removed `DistributedPPOEngine`. The PPO update is `PPOTrainer` in `src/rlhf_platform/ppo_engine.py`. Connecting a run is S0.11.

What the remaining code does:

- `rollout.py` keeps generated tokens in a CPU list and can sample a response from a module that returns `.logits`. The background generator-thread class was removed in S0.4.
- `utils/telemetry.py` can log a JSON event. Nothing in `tests/` constructs it.

An older draft of this page claimed a 40% GPU-memory reduction. That number was not a measurement. It is gone.

S0.9 removed the process-group module, the communication hooks, the async checkpoint writer, the multi-rank script, the cluster-layout YAML, and `docs/design.md`. Stage 2 builds rollout and weight-sync distribution from scratch.

## What is not wired

`python train.py` exits 2. It does not start a rollout or call `PPOTrainer`.

## Module dependency graph

```
train.py
  exits 2. It does not import the modules below.

The modules that still exist:
  ├─ rlhf_platform.alignment.rollout
  │   ├─ RolloutBuffer (CPU list, pin only if CUDA exists)
  │   └─ RolloutGenerator (autoregressive sampling)
  │
  ├─ rlhf_platform.ppo_engine
  │   └─ PPOTrainer (the only PPO update)
  │
  └─ rlhf_platform.utils.telemetry
      ├─ RankAwareTelemetry (structured JSON logging)
      ├─ NCCLProfiler (no caller)
      └─ DistributedMetricsCollector (no caller)
```

## Configuration files

### `configs/deepspeed_zero3.yaml`

ZeRO-3 optimizer configuration. No trainer reads this file today.

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

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Exits 2. Does not call `PPOTrainer` |
| `scripts/README.md` | Note for this directory |
| `ROADMAP.md` | Work status, notes, and change log |
| `src/rlhf_platform/alignment/rollout.py` | CPU rollout buffer and token sampler |
| `src/rlhf_platform/ppo_engine.py` | `PPOTrainer`, the only PPO update |
| `src/rlhf_platform/utils/telemetry.py` | JSON logging. No test calls it |
| `configs/deepspeed_zero3.yaml` | Optional ZeRO-3 configuration. No trainer reads it |
