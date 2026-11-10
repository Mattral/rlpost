"""Seed a run once, and save or load a checkpoint that can continue it.

``seed_everything`` seeds Python's ``random``, the global NumPy RNG, and
torch. When CUDA is present it seeds every CUDA device. Call it once, at
startup, before the first step. A resumed run calls it once and then
replaces that state with the checkpoint.

The checkpoint holds the models, the optimizer, the step counter, those
RNG states, and the config. NumPy's RNG state is a string, a list of ints,
and Python numbers. Every other non-tensor value is stored as a list, a
number, a string, or ``None``, so ``torch.load(..., weights_only=True)``
can read the file. The write is a temporary file in the same directory
and ``os.replace``, so a reader sees the old file or the new one.
``checkpoint_path`` is where that file goes. It is not part of the config
hash: the hash is the run, and the path is only where the file sits. A
Drive path on Colab or a Volume path on Modal is that string. Nothing
here mounts either one.

``trainer_state`` is the trainer state the optimizer does not hold.
PPO stores ``kl_coefficient`` there. ``adaptive_kl_penalty`` changes it
after each step, and the next step's loss uses the new value.

This module does not run a training loop. The synthetic PPO and DPO
steps that check resume on CPU live in the tests.
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
    """Python, NumPy, and torch RNG states. CUDA is ``None`` when it is absent.

    The NumPy state is ``(algorithm, keys, pos, has_gauss, cached_gaussian)``.
    ``keys`` is a list of ints. The algorithm is a string. The other three
    fields are Python numbers. Nothing in the result is a NumPy array.
    """
    cuda = None
    if torch.cuda.is_available():
        cuda = torch.cuda.get_rng_state_all()
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (str(kind), keys.tolist(), int(pos), int(has_gauss), float(cached)),
        "torch": torch.get_rng_state(),
        "cuda": cuda,
    }


def _numpy_state(plain: object) -> tuple:
    """Rebuild the tuple ``np.random.set_state`` expects."""
    if not isinstance(plain, (tuple, list)) or len(plain) != 5:
        raise TypeError("numpy RNG state must be a 5-item sequence of plain values")
    kind, keys, pos, has_gauss, cached = plain
    if not isinstance(kind, str):
        raise TypeError("numpy RNG algorithm must be a string")
    if not isinstance(keys, list) or any(
        isinstance(key, bool) or not isinstance(key, int) for key in keys
    ):
        raise TypeError("numpy RNG keys must be a list of ints")
    if isinstance(pos, bool) or not isinstance(pos, int):
        raise TypeError("numpy RNG position must be an int")
    if isinstance(has_gauss, bool) or not isinstance(has_gauss, int):
        raise TypeError("numpy RNG has_gauss must be an int")
    if isinstance(cached, bool) or not isinstance(cached, float):
        raise TypeError("numpy RNG cached gaussian must be a float")
    return (kind, np.array(keys, dtype=np.uint32), pos, has_gauss, cached)


def restore_rng(state: Mapping[str, object]) -> None:
    """Restore the states from ``capture_rng``.

    A checkpoint saved with CUDA RNG state does not load on a machine
    where CUDA is absent.
    """
    python_state = state["python"]
    if not isinstance(python_state, tuple):
        raise TypeError("python RNG state must be a tuple")
    random.setstate(python_state)
    np.random.set_state(_numpy_state(state["numpy"]))
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


def _plain_tree(value: object) -> object:
    """Return ``value`` as tensors and plain Python values.

    A tensor stays a tensor. A NumPy array becomes a nested list. A NumPy
    scalar becomes a Python scalar. A dict, list, or tuple is walked.
    Strings, ints, floats, bools, bytes, and ``None`` stay as they are.
    Anything else raises, so the file cannot hold a type that
    ``weights_only=True`` would refuse.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return _plain_tree(value.tolist())
    if isinstance(value, np.generic):
        return _plain_tree(value.item())
    if value is None or isinstance(value, (str, bytes, float, bool, int)):
        return value
    if isinstance(value, dict):
        return {_plain_tree(key): _plain_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_tree(item) for item in value)
    if isinstance(value, list):
        return [_plain_tree(item) for item in value]
    raise TypeError(
        f"checkpoint value of type {type(value).__name__} is not a tensor or a plain Python value"
    )


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
    payload = _plain_tree(
        {
            "seed": config.seed,
            "step": step,
            "models": {name: module.state_dict() for name, module in models.items()},
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng(),
            "config": config.model_dump(mode="json"),
            "config_hash": config_hash(config),
            "trainer_state": dict(trainer_state or {}),
        }
    )
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dict")
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
    ``checkpoint_path`` is not part of the hash. The file is read with
    ``weights_only=True``.

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
    payload = torch.load(file, map_location="cpu", weights_only=True)
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
