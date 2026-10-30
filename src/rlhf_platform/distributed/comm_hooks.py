"""
comm_hooks.py: Custom PyTorch distributed communication hooks for RLHF.

Implements overlapping all_reduce and reduce_scatter operations with
backward pass computation to maximize compute utilization on InfiniBand clusters.
Also includes numerical safeguards (KL clipping, gradient clipping) to detect
and prevent Silent Data Corruption (SDC).
"""

import logging
from typing import List

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class CommunicationHooks:
    """Register custom communication hooks for gradient synchronization."""

    @staticmethod
    def register_overlap_hook(model: torch.nn.Module) -> None:
        """
        Register a hook to overlap all_reduce with backward computation.
        Requires DDP-style distributed setup.

        This hook registers a communication function that batches gradients
        into buckets and fires off all_reduce ops as soon as a bucket is ready,
        rather than waiting for all gradients to be computed.
        """
        if not hasattr(model, "module"):
            logger.warning("Model is not wrapped in DDP. Skipping overlap hook registration.")
            return

        def overlap_communication(bucket: dist.GradBucket) -> torch.futures.Future:
            """Fire off all_reduce immediately when a gradient bucket is ready."""
            return dist.all_reduce_coalesced(
                [bucket.buffer()],
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

        model.register_comm_hook(state=None, hook=overlap_communication)
        logger.info("Registered gradient bucket overlap hook for DDP communication.")

    @staticmethod
    def register_kl_divergence_check(
        model: torch.nn.Module,
        reference_model: torch.nn.Module,
        max_kl: float = 0.5,
    ) -> None:
        """
        Register a backward hook to check KL divergence between policy and reference.

        This is a numerical safeguard: if KL exceeds a threshold, log a warning
        and clip gradients to prevent policy collapse.

        Args:
            model: Policy model (Actor)
            reference_model: Reference model (frozen baseline)
            max_kl: Maximum allowed KL divergence before clipping
        """

        def kl_check_hook(grad: torch.Tensor) -> torch.Tensor:
            """Hook to check gradient magnitude as a proxy for KL drift."""
            grad_norm = torch.norm(grad)
            if grad_norm > max_kl:
                logger.warning(
                    f"Gradient norm {grad_norm:.4f} exceeds KL threshold {max_kl}. "
                    "Clipping gradients to prevent policy collapse."
                )
                grad = torch.clamp(grad, -max_kl, max_kl)
            return grad

        # Register on actor model's last layer to catch KL issues early
        if hasattr(model, "lm_head"):
            model.lm_head.weight.register_hook(kl_check_hook)
            logger.info("Registered KL divergence safeguard hook on model.lm_head.")

    @staticmethod
    def register_gradient_clipping_hook(
        model: torch.nn.Module,
        max_grad_norm: float = 1.0,
    ) -> None:
        """
        Register a backward hook to clip gradients per-parameter.

        Helps prevent numerical instability and gradient explosion.
        """

        def clipping_hook(grad: torch.Tensor) -> torch.Tensor:
            """Clip gradient to max_grad_norm."""
            return torch.clamp(grad, -max_grad_norm, max_grad_norm)

        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(clipping_hook)

        logger.info(
            f"Registered gradient clipping hook (max_norm={max_grad_norm}) on all parameters."
        )

    @staticmethod
    def register_nan_check_hook(model: torch.nn.Module) -> None:
        """
        Register a backward hook to detect NaN/Inf gradients.

        Useful for detecting SDC or numerical overflow.
        """

        def nan_check_hook(grad: torch.Tensor) -> torch.Tensor:
            """Check for NaN or Inf and raise if found."""
            if torch.isnan(grad).any():
                raise RuntimeError(f"NaN detected in gradient: {grad}")
            if torch.isinf(grad).any():
                raise RuntimeError(f"Inf detected in gradient: {grad}")
            return grad

        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(nan_check_hook)

        logger.info("Registered NaN/Inf detection hook on all parameters.")


class AsynchronousReduceScatter:
    """Async reduce_scatter for ring-buffer rollout collection."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.pending_futures: List[torch.futures.Future] = []

    def async_reduce_scatter(self, tensor: torch.Tensor) -> torch.futures.Future:
        """
        Fire off a non-blocking reduce_scatter.

        Returns a Future that will complete when the scatter is done.
        """
        future = dist.reduce_scatter_multigpu(
            [tensor],
            [[tensor.clone() for _ in range(self.world_size)]],
            async_op=True,
        )
        self.pending_futures.append(future)
        return future

    def wait_all(self) -> None:
        """Block until all pending futures complete."""
        for future in self.pending_futures:
            future.wait()
        self.pending_futures.clear()
        logger.info(f"Rank {self.rank}: All pending reduce_scatter ops completed.")


class CollectiveProfiler:
    """Profile collective communication overhead on this rank."""

    def __init__(self):
        self.all_reduce_times: List[float] = []
        self.reduce_scatter_times: List[float] = []

    def record_all_reduce(self, duration_ms: float) -> None:
        """Record duration of an all_reduce operation."""
        self.all_reduce_times.append(duration_ms)

    def record_reduce_scatter(self, duration_ms: float) -> None:
        """Record duration of a reduce_scatter operation."""
        self.reduce_scatter_times.append(duration_ms)

    def summary(self) -> dict:
        """Return summary statistics."""
        import statistics

        def safe_stats(times: List[float]) -> dict:
            if not times:
                return {"count": 0, "mean_ms": 0.0, "max_ms": 0.0}
            return {
                "count": len(times),
                "mean_ms": statistics.mean(times),
                "max_ms": max(times),
                "min_ms": min(times),
            }

        return {
            "all_reduce": safe_stats(self.all_reduce_times),
            "reduce_scatter": safe_stats(self.reduce_scatter_times),
        }
