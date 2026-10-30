"""
async_io.py: Non-blocking checkpointing and asynchronous I/O for distributed RLHF.

Implements background-thread checkpointing that doesn't block training forward/backward passes.
Weights are streamed to pinned host memory, then asynchronously written to distributed storage.
"""

import json
import logging
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointBuffer:
    """Ring buffer for staging checkpoints before async write."""

    def __init__(self, capacity: int = 3):
        self.queue: Queue = Queue(maxsize=capacity)
        self.capacity = capacity
        self.total_written = 0

    def put(self, checkpoint_dict: Dict[str, Any], timeout: Optional[float] = None) -> None:
        """Place a checkpoint dict into the buffer."""
        try:
            self.queue.put(checkpoint_dict, timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to enqueue checkpoint: {e}")
            raise

    def get(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Retrieve next checkpoint from buffer."""
        try:
            return self.queue.get(timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to dequeue checkpoint: {e}")
            raise

    def empty(self) -> bool:
        """Check if buffer is empty."""
        return self.queue.empty()

    def size(self) -> int:
        """Current buffer occupancy."""
        return self.queue.qsize()


class AsyncCheckpointWriter:
    """Background thread that writes checkpoints without blocking training."""

    def __init__(
        self,
        checkpoint_dir: str,
        buffer_capacity: int = 3,
        save_frequency: int = 100,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = CheckpointBuffer(capacity=buffer_capacity)
        self.save_frequency = save_frequency
        self.step = 0

        self.writer_thread = threading.Thread(target=self._write_loop, daemon=False)
        self.writer_thread.start()

        self.stop_event = threading.Event()
        logger.info(f"Async checkpoint writer initialized. Target dir: {self.checkpoint_dir}")

    def _write_loop(self) -> None:
        """Run in background thread: continuously pop and write checkpoints."""
        while not self.stop_event.is_set():
            try:
                # Non-blocking get with timeout to check stop signal periodically
                checkpoint_dict = self.buffer.get(timeout=1.0)
                self._write_checkpoint(checkpoint_dict)
            except Exception:
                # queue.Empty on timeout, or a write error. Keep the writer alive.
                pass

    def _write_checkpoint(self, checkpoint_dict: Dict[str, Any]) -> None:
        """Actually write a checkpoint to disk."""
        try:
            step = checkpoint_dict.get("step", self.step)
            checkpoint_path = self.checkpoint_dir / f"checkpoint-step-{step}.pt"

            # Save state dict (weights) to pinned CPU memory first
            pinned_cpu_dict = {}
            for key, value in checkpoint_dict.items():
                if isinstance(value, torch.Tensor):
                    # Move to pinned CPU memory for faster I/O
                    pinned_cpu_dict[key] = value.cpu().pin_memory()
                else:
                    pinned_cpu_dict[key] = value

            # Now write to disk (this happens in background thread)
            torch.save(pinned_cpu_dict, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

            self.buffer.total_written += 1

        except Exception as e:
            logger.error(f"Error writing checkpoint: {e}", exc_info=True)

    def enqueue_checkpoint(
        self,
        step: int,
        model_state_dict: Dict[str, torch.Tensor],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Queue a checkpoint for asynchronous writing.

        This returns immediately (non-blocking); the actual write happens in the background.
        """
        checkpoint_dict = {
            "step": step,
            "model_state": model_state_dict,
            "optimizer_state": optimizer_state_dict,
            "metrics": metrics or {},
        }

        try:
            self.buffer.put(checkpoint_dict, timeout=5.0)
            logger.debug(f"Checkpoint step {step} enqueued (buffer size: {self.buffer.size()})")
        except Exception as e:
            logger.error(f"Failed to enqueue checkpoint at step {step}: {e}")
            raise

    def flush(self, timeout: float = 60.0) -> None:
        """Block until all queued checkpoints are written."""
        import time

        start_time = time.time()

        while not self.buffer.empty():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Checkpoint flush exceeded timeout {timeout}s")
            time.sleep(0.1)

        logger.info("All queued checkpoints flushed to disk.")

    def shutdown(self) -> None:
        """Gracefully shut down the writer thread."""
        self.stop_event.set()
        self.writer_thread.join(timeout=10.0)

        if self.writer_thread.is_alive():
            logger.warning("Writer thread did not terminate in time.")
        else:
            logger.info("Async checkpoint writer shutdown complete.")


class CheckpointMetadata:
    """Metadata about a checkpoint for easy retrieval and validation."""

    def __init__(
        self,
        step: int,
        model_name: str,
        precision: str,
        world_size: int,
        rank: int,
        metrics: Dict[str, float],
    ):
        self.step = step
        self.model_name = model_name
        self.precision = precision
        self.world_size = world_size
        self.rank = rank
        self.metrics = metrics
        self.timestamp = __import__("time").time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata."""
        return {
            "step": self.step,
            "model_name": self.model_name,
            "precision": self.precision,
            "world_size": self.world_size,
            "rank": self.rank,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    def save(self, path: str) -> None:
        """Write metadata to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CheckpointMetadata":
        """Load metadata from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class DistributedCheckpointManager:
    """Manage checkpoints across all ranks with metadata tracking."""

    def __init__(
        self,
        checkpoint_dir: str,
        world_size: int,
        rank: int,
        save_frequency: int = 100,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.world_size = world_size
        self.rank = rank
        self.save_frequency = save_frequency

        # Only rank 0 writes checkpoints
        if self.rank == 0:
            self.writer = AsyncCheckpointWriter(
                checkpoint_dir=str(self.checkpoint_dir),
                save_frequency=save_frequency,
            )
        else:
            self.writer = None

        logger.info(f"Rank {rank}/{world_size}: Checkpoint manager initialized.")

    def save_checkpoint(
        self,
        step: int,
        model_name: str,
        model_state_dict: Dict[str, torch.Tensor],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        precision: str = "fp16",
    ) -> None:
        """Save a checkpoint (asynchronously on rank 0)."""
        if self.rank == 0 and self.writer is not None:
            # Enqueue for async write
            self.writer.enqueue_checkpoint(
                step=step,
                model_state_dict=model_state_dict,
                optimizer_state_dict=optimizer_state_dict,
                metrics=metrics,
            )

            # Save metadata
            metadata = CheckpointMetadata(
                step=step,
                model_name=model_name,
                precision=precision,
                world_size=self.world_size,
                rank=self.rank,
                metrics=metrics or {},
            )
            metadata_path = self.checkpoint_dir / f"checkpoint-step-{step}-metadata.json"
            metadata.save(str(metadata_path))

    def flush_and_shutdown(self) -> None:
        """Flush all pending checkpoints and shutdown writers."""
        if self.rank == 0 and self.writer is not None:
            self.writer.flush()
            self.writer.shutdown()
            logger.info("Distributed checkpoint manager shutdown complete.")
