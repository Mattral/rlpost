"""
telemetry.py: Structured observability for distributed RLHF clusters.

Implements OpenTelemetry-compatible structured JSON logging with rank-aware
tracing, NCCL profiling metrics, and per-rank performance tracking.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist


@dataclass
class TelemetryEvent:
    """Single structured telemetry event."""

    timestamp: float
    rank: int
    event_type: str  # e.g., "ppo_step", "communication", "checkpoint"
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), default=str)


class RankAwareTelemetry:
    """Telemetry tracker with rank-specific context."""

    def __init__(self, rank: int, world_size: int, log_file: Optional[str] = None):
        self.rank = rank
        self.world_size = world_size
        self.events: List[TelemetryEvent] = []
        self.log_file = log_file

        self.logger = logging.getLogger(f"rank_{rank}_telemetry")
        if log_file and rank == 0:
            handler = logging.FileHandler(log_file)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)

    def log_event(
        self,
        event_type: str,
        metrics: Dict[str, Any],
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a structured telemetry event."""
        event = TelemetryEvent(
            timestamp=time.time(),
            rank=self.rank,
            event_type=event_type,
            metrics=metrics,
            duration_ms=duration_ms,
            error=error,
        )
        self.events.append(event)

        # Log to file/console
        log_msg = event.to_json()
        if error:
            self.logger.error(log_msg)
        else:
            self.logger.info(log_msg)

    def log_ppo_step(
        self,
        step: int,
        actor_loss: float,
        value_loss: float,
        kl_divergence: float,
        policy_entropy: float,
        duration_ms: float,
    ) -> None:
        """Log a PPO training step."""
        self.log_event(
            event_type="ppo_step",
            metrics={
                "step": step,
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "kl_divergence": kl_divergence,
                "policy_entropy": policy_entropy,
            },
            duration_ms=duration_ms,
        )

    def log_communication(
        self,
        op_type: str,  # e.g., "all_reduce", "reduce_scatter"
        data_size_mb: float,
        duration_ms: float,
    ) -> None:
        """Log a collective communication operation."""
        throughput_gbps = (data_size_mb * 8) / (duration_ms / 1000.0) / 1000.0
        self.log_event(
            event_type="communication",
            metrics={
                "op_type": op_type,
                "data_size_mb": data_size_mb,
                "throughput_gbps": throughput_gbps,
            },
            duration_ms=duration_ms,
        )

    def log_checkpoint(
        self,
        step: int,
        model_name: str,
        checkpoint_size_mb: float,
        duration_ms: float,
    ) -> None:
        """Log a checkpoint save operation."""
        self.log_event(
            event_type="checkpoint",
            metrics={
                "step": step,
                "model_name": model_name,
                "checkpoint_size_mb": checkpoint_size_mb,
            },
            duration_ms=duration_ms,
        )

    def log_memory_usage(
        self,
        allocated_mb: float,
        reserved_mb: float,
        max_allocated_mb: float,
    ) -> None:
        """Log GPU memory usage snapshot."""
        self.log_event(
            event_type="memory",
            metrics={
                "allocated_mb": allocated_mb,
                "reserved_mb": reserved_mb,
                "max_allocated_mb": max_allocated_mb,
            },
        )


class NCCLProfiler:
    """Profile NCCL communication performance on this rank."""

    def __init__(self, rank: int):
        self.rank = rank
        self.all_reduce_times: List[float] = []
        self.reduce_scatter_times: List[float] = []
        self.broadcast_times: List[float] = []

    def record_all_reduce(self, duration_ms: float) -> None:
        """Record all_reduce latency."""
        self.all_reduce_times.append(duration_ms)

    def record_reduce_scatter(self, duration_ms: float) -> None:
        """Record reduce_scatter latency."""
        self.reduce_scatter_times.append(duration_ms)

    def record_broadcast(self, duration_ms: float) -> None:
        """Record broadcast latency."""
        self.broadcast_times.append(duration_ms)

    def summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        import statistics

        def stats(times: List[float]) -> Dict[str, float]:
            if not times:
                return {"count": 0, "mean_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}
            return {
                "count": len(times),
                "mean_ms": statistics.mean(times),
                "max_ms": max(times),
                "min_ms": min(times),
                "p99_ms": sorted(times)[-max(1, len(times) // 100)],
            }

        return {
            "rank": self.rank,
            "all_reduce": stats(self.all_reduce_times),
            "reduce_scatter": stats(self.reduce_scatter_times),
            "broadcast": stats(self.broadcast_times),
        }


class DistributedMetricsCollector:
    """Collect and aggregate metrics across all ranks."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.local_metrics: Dict[str, List[float]] = {}

    def record_metric(self, name: str, value: float) -> None:
        """Record a local metric."""
        if name not in self.local_metrics:
            self.local_metrics[name] = []
        self.local_metrics[name].append(value)

    def get_local_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of local metrics."""
        import statistics

        summary = {}
        for name, values in self.local_metrics.items():
            if values:
                summary[name] = {
                    "mean": statistics.mean(values),
                    "max": max(values),
                    "min": min(values),
                    "count": len(values),
                }
        return summary

    def allgather_metrics(self) -> List[Dict[str, Dict[str, float]]]:
        """Gather metrics from all ranks (rank 0 only)."""
        if self.world_size == 1:
            return [self.get_local_summary()]

        import pickle

        local_summary = self.get_local_summary()

        # Serialize
        local_bytes = pickle.dumps(local_summary)
        local_tensor = torch.tensor(list(local_bytes), dtype=torch.uint8)

        # Gather sizes
        size_tensor = torch.tensor(len(local_bytes), dtype=torch.long)
        sizes = [torch.tensor(0, dtype=torch.long) for _ in range(self.world_size)]
        dist.all_gather(sizes, size_tensor)

        # Gather data
        max_size = max([s.item() for s in sizes])
        padded = torch.zeros(max_size, dtype=torch.uint8)
        padded[: len(local_bytes)] = local_tensor

        all_data = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(self.world_size)]
        dist.all_gather(all_data, padded)

        # Deserialize (rank 0 only)
        all_metrics = []
        if self.rank == 0:
            for i, data in enumerate(all_data):
                actual_size = sizes[i].item()
                try:
                    metrics_dict = pickle.loads(data[:actual_size].cpu().numpy().tobytes())
                    all_metrics.append(metrics_dict)
                except Exception:
                    all_metrics.append({})

        return all_metrics
