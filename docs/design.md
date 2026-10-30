# Design targets

These figures are design targets, not measured results. No run in this repo has produced them. Do not cite them as throughput, memory, or scale that this code has reached.

The old README printed a telemetry JSON sample with a VRAM total and a tokens-per-second figure. That sample was removed. It was not a recorded run.

## Cluster shape

| Metric / Layer | 8x GPU Node (Local Dev) | 512x GPU Cluster (Pod Scale) | 10,000x GPU Cluster (Petascale) |
| --- | --- | --- | --- |
| **Tensor Parallelism (TP)** | 1 | 8 (Intra-Chassis NVLink) | 8 (Intra-Chassis NVLink) |
| **Pipeline Parallelism (PP)** | 1 | 2 (Inter-Node InfiniBand) | 16 (Inter-Node Ring) |
| **Data Parallelism (DP)** | 8 (ZeRO-3) | 32 (FSDP + Sharding) | 780 (Hybrid FSDP / ZeRO) |
| **Gradient Overlap Bucket** | 25MB | 50MB | 128MB |
| **Target Context Length** | 4,096 | 16,384 | 65,536 |

`configs/cluster_topology.yaml` and `src/rlhf_platform/distributed/topology.py` are the code that corresponds to the asymmetric actor/critic and reference/reward split. Importing that package fails on torch 2.15 because `comm_hooks.py` imports a missing name. A CPU process-group test is item S0.9 in `ROADMAP.md`.
