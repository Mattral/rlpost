"""Seed a run once, and save or load a checkpoint that can continue it.

``seed_everything`` seeds Python's ``random``, the global NumPy RNG, and
torch. When CUDA is present it seeds every CUDA device. A run calls it
once, at startup, before the first step. A resumed run calls it once and
then replaces that state with the checkpoint.

The checkpoint holds the models, the optimizer, the step counter, those
RNG states, and the config. The write is a temporary file in the same
directory and ``os.replace``, so a reader sees the old file or the new
one. ``checkpoint_path`` is where that file goes. It is not part of the
config hash: the hash is the run, and the path is only where the file
sits. A Drive path on Colab or a Volume path on Modal is that string.
Nothing here mounts either one.

``trainer_state`` is the trainer state the optimizer does not hold.
PPO stores ``kl_coefficient`` there. ``adaptive_kl_penalty`` changes it
after each step, and the next step's loss uses the new value.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from rlpost.config import TrainingConfig

_HASH_EXCLUDED = ("checkpoint_path",)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch. Seed CUDA when it is present.

    Call once at the start of a run. Do not call it between steps.
    A bool is rejected. ``True`` would otherwise seed as ``1``.

    Args:
        seed: Non-negative int. This is ``TrainingConfig.seed``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_payload(config: TrainingConfig) -> dict:
    """JSON-ready config, without ``checkpoint_path``.

    The path is where the checkpoint file is written. Moving the file
    to another directory does not make it a different run.
    """
    data = config.model_dump(mode="json")
    for key in _HASH_EXCLUDED:
        data.pop(key, None)
    return data


def config_hash(config: TrainingConfig) -> str:
    """SHA-256 of the canonical config JSON. ``checkpoint_path`` is left out."""
    blob = json.dumps(config_payload(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def capture_rng() -> dict:
    """Python, NumPy, and torch RNG states. CUDA is ``None`` when it is absent."""
    cuda = None
    if torch.cuda.is_available():
        cuda = torch.cuda.get_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda,
    }


def restore_rng(state: Mapping[str, object]) -> None:
    """Restore the states from ``capture_rng``.

    A checkpoint saved with CUDA RNG state does not load on a machine
    where CUDA is absent.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if not isinstance(torch_state, torch.Tensor):
        raise TypeError("torch RNG state must be a tensor")
    torch.set_rng_state(torch_state)
    cuda = state.get("cuda")
    if cuda is None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint has CUDA RNG state and CUDA is not available")
    torch.cuda.set_rng_state_all(cuda)


@dataclass
class LoadedCheckpoint:
    """What ``load_checkpoint`` restored, besides the modules and the optimizer.

    Attributes:
        step: Steps already completed. The next update is this step plus one.
        trainer_state: Trainer fields the optimizer does not hold.
            PPO writes ``kl_coefficient``.
    """

    step: int
    trainer_state: dict[str, float]


def save_checkpoint(
    path: str | Path,
    *,
    models: Mapping[str, nn.Module],
    optimizer: Optimizer,
    step: int,
    config: TrainingConfig,
    trainer_state: Mapping[str, float] | None = None,
) -> None:
    """Write a checkpoint atomically.

    The temporary file is created in the destination directory, filled,
    and then ``os.replace``d over ``path``. A crash leaves the previous
    file in place.

    Args:
        path: Destination file. Parent directories are created.
        models: Named modules. Each ``state_dict`` is stored.
        optimizer: Optimizer whose ``state_dict`` is stored.
        step: Steps completed before this write.
        config: Run config. The seed is this object's ``seed``.
        trainer_state: Extra trainer fields. ``None`` stores an empty dict.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be an int >= 0, got {step!r}")
    payload = {
        "seed": config.seed,
        "step": step,
        "models": {name: module.state_dict() for name, module in models.items()},
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(),
        "config": config.model_dump(mode="json"),
        "config_hash": config_hash(config),
        "trainer_state": dict(trainer_state or {}),
    }
    _atomic_torch_save(Path(path), payload)


def load_checkpoint(
    path: str | Path,
    *,
    models: Mapping[str, nn.Module],
    optimizer: Optimizer,
    config: TrainingConfig,
) -> LoadedCheckpoint:
    """Restore models, optimizer, and RNG. Return the step.

    The hash is checked before any weight is written. A mismatch raises
    ``ValueError`` and leaves ``models`` and ``optimizer`` as they were.
    ``checkpoint_path`` is not part of the hash.

    Args:
        path: File written by ``save_checkpoint``.
        models: Modules whose names match the checkpoint. Weights are
            copied in with ``load_state_dict``.
        optimizer: Optimizer built on those same parameters, in the same
            order, so its slots line up with the saved state.
        config: Config of this run. Its hash must equal the stored hash.

    Returns:
        The completed step and the saved trainer state.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"checkpoint not found: {file}")
    try:
        payload = torch.load(file, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(file, map_location="cpu")
    expected = config_hash(config)
    stored = payload.get("config_hash")
    if stored != expected:
        raise ValueError(f"checkpoint config hash {stored} does not match this config ({expected})")
    saved_models = payload["models"]
    if set(saved_models) != set(models):
        raise ValueError(f"checkpoint models {sorted(saved_models)} do not match {sorted(models)}")
    for name, module in models.items():
        module.load_state_dict(saved_models[name])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"checkpoint step must be an int, got {type(step).__name__}")
    trainer_state = payload.get("trainer_state", {})
    return LoadedCheckpoint(step=step, trainer_state=dict(trainer_state))


def _atomic_torch_save(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` via a temp file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
