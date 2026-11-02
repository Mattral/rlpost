"""
telemetry.py: Structured JSON logging with a rank on each event.

`RankAwareTelemetry` appends a `TelemetryEvent` and writes one JSON log line.
Nothing in the trainers constructs it.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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
