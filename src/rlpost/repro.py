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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from rlpost.config import TrainingConfig
from rlpost.dpo_engine import DPOTrainer
from rlpost.ops import gather_logprobs, padding_attention
from rlpost.ppo_engine import PPOTrainer

_HASH_EXCLUDED = ("checkpoint_path",)
_BATCH = 2
_LENGTH = 8


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


@dataclass
class RunOutput:
    """What one call of ``run_ppo`` or ``run_dpo`` did.

    Attributes:
        seed: ``TrainingConfig.seed``. The same value is written to the
            run JSON at ``output_path``.
        step: Trainer step after this call. A resumed call continues from
            the checkpoint, so this is the global step, not ``len(losses)``.
        losses: One loss per step taken in this call. PPO records
            ``policy_loss``. DPO records ``dpo_loss``.
        metrics: One dict per step. PPO is every ``PPOMetrics`` field.
            DPO is ``DPOMetrics.to_dict()``.
        output_path: JSON file for this call. It records the seed.
        checkpoint_path: Checkpoint file written at the end, or None when
            the config does not set one.
    """

    seed: int
    step: int
    losses: list[float]
    metrics: list[dict[str, float]]
    output_path: str
    checkpoint_path: str | None


def run_ppo(
    trainer: PPOTrainer,
    n_steps: int,
    *,
    resume_from: str | Path | None = None,
) -> RunOutput:
    """Run ``n_steps`` PPO updates.

    Seeds once, then draws each batch from Python, NumPy, and torch.
    The batch is ``[2, 8]``. Prompt length is 0 and the response length
    is 8, so every token is a response token on the right-padding layout.
    ``old_log_probs`` are the current policy's ``log_softmax(logits /
    temperature)`` at the sampled tokens, shape ``[2, 7]``. ``values``
    are the value model, detached, shape ``[2, 8]``. This is not
    ``generate_rollout``.

    A resumed call seeds once and then replaces the RNG, the weights,
    the optimizer, the step, and ``kl_coefficient`` from ``resume_from``.

    Args:
        trainer: PPO trainer. Policy and value are trained. The reference
            is stored in the checkpoint and is not stepped.
        n_steps: Updates to take in this call. At least 1.
        resume_from: Checkpoint to load before those updates. None starts
            from the trainer as it was constructed.

    Returns:
        Losses and metrics for the steps taken in this call.
    """
    _check_steps(n_steps)
    seed_everything(trainer.config.seed)
    models = {
        "policy": trainer.policy_model,
        "value": trainer.value_model,
        "reference": trainer.reference_model,
    }
    if resume_from is not None:
        loaded = load_checkpoint(
            resume_from,
            models=models,
            optimizer=trainer.optimizer,
            config=trainer.config,
        )
        trainer.global_step = loaded.step
        if "kl_coefficient" not in loaded.trainer_state:
            raise KeyError("PPO checkpoint trainer_state is missing kl_coefficient")
        trainer.kl_coefficient = float(loaded.trainer_state["kl_coefficient"])
    trainer.policy_model.train()
    trainer.value_model.train()
    trainer.reference_model.eval()
    losses: list[float] = []
    metrics: list[dict[str, float]] = []
    for _ in range(n_steps):
        batch, old_log_probs = _ppo_batch(trainer)
        step_metrics = trainer.train_step(batch, old_log_probs)
        losses.append(step_metrics.policy_loss)
        metrics.append(asdict(step_metrics))
    return _finish_run(
        trainer.config,
        models=models,
        optimizer=trainer.optimizer,
        step=trainer.global_step,
        trainer_state={"kl_coefficient": float(trainer.kl_coefficient)},
        losses=losses,
        metrics=metrics,
    )


def run_dpo(
    trainer: DPOTrainer,
    n_steps: int,
    *,
    resume_from: str | Path | None = None,
) -> RunOutput:
    """Run ``n_steps`` DPO updates on token batches.

    Seeds once. Each step draws a prompt length from Python, a token
    offset from NumPy, and the token ids from torch. Chosen and rejected
    rows are ``[2, 8]``. The completion mask is 0 on the prompt prefix.
    ``train_token_step`` sums ``completion_mask[:, 1:]``. This does not
    tokenize text.

    A resumed call seeds once and then replaces the RNG, the weights,
    the optimizer, and the step from ``resume_from``.

    Args:
        trainer: DPO trainer. The policy is trained. The reference is
            frozen and is still stored in the checkpoint.
        n_steps: Updates to take in this call. At least 1.
        resume_from: Checkpoint to load before those updates.

    Returns:
        Losses and metrics for the steps taken in this call.
    """
    _check_steps(n_steps)
    seed_everything(trainer.config.seed)
    models = {
        "policy": trainer.policy_model,
        "reference": trainer.reference_model,
    }
    if resume_from is not None:
        loaded = load_checkpoint(
            resume_from,
            models=models,
            optimizer=trainer.optimizer,
            config=trainer.config,
        )
        trainer.step_count = loaded.step
    losses: list[float] = []
    metrics: list[dict[str, float]] = []
    for _ in range(n_steps):
        batch = _dpo_batch(trainer)
        step_metrics = trainer.train_token_step(*batch)
        losses.append(step_metrics.dpo_loss)
        metrics.append(step_metrics.to_dict())
    return _finish_run(
        trainer.config,
        models=models,
        optimizer=trainer.optimizer,
        step=trainer.step_count,
        trainer_state={},
        losses=losses,
        metrics=metrics,
    )


def _check_steps(n_steps: int) -> None:
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be an int >= 1, got {n_steps!r}")


def _vocab_size(module: nn.Module) -> int:
    config = getattr(module, "config", None)
    vocab = getattr(config, "vocab_size", None)
    if isinstance(vocab, bool) or not isinstance(vocab, int):
        raise TypeError("the policy needs config.vocab_size")
    return vocab


def _ppo_batch(trainer: PPOTrainer) -> tuple[dict, torch.Tensor]:
    """One PPO batch. Python, NumPy, and torch each contribute a draw."""
    vocab = _vocab_size(trainer.policy_model)
    shift = random.randrange(0, 5)
    scale = 0.25 + float(np.random.random())
    input_ids = torch.randint(0, vocab, (_BATCH, _LENGTH))
    rewards = torch.randn(_BATCH, _LENGTH) * scale + shift
    prompt_lengths = torch.zeros(_BATCH, dtype=torch.long)
    response_lengths = torch.full((_BATCH,), _LENGTH, dtype=torch.long)
    attention, position_ids = padding_attention(prompt_lengths, response_lengths, _LENGTH, "right")
    temperature = trainer.alignment_config.temperature
    with torch.no_grad():
        policy_out = trainer.policy_model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
        )
        logits = policy_out.logits if hasattr(policy_out, "logits") else policy_out
        old_log_probs = gather_logprobs(logits / temperature, input_ids)
        value_out = trainer.value_model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
        )
        values = value_out.value if hasattr(value_out, "value") else value_out
        if values.ndim == input_ids.ndim + 1 and values.shape[-1] == 1:
            values = values.squeeze(-1)
    batch = {
        "input_ids": input_ids,
        "rewards": rewards,
        "values": values.detach(),
        "dones": torch.zeros(_BATCH, _LENGTH),
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
        "padding_side": "right",
        "temperature": temperature,
    }
    return batch, old_log_probs.detach()


def _dpo_batch(
    trainer: DPOTrainer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chosen and rejected rows. The prompt length comes from Python's RNG."""
    vocab = _vocab_size(trainer.policy_model)
    prompt_len = random.randrange(1, _LENGTH - 1)
    offset = int(np.random.randint(0, 7))
    chosen = (torch.randint(0, vocab, (_BATCH, _LENGTH)) + offset) % vocab
    rejected = torch.randint(0, vocab, (_BATCH, _LENGTH))
    attention = torch.ones(_BATCH, _LENGTH, dtype=torch.long)
    prompt_lengths = torch.full((_BATCH,), prompt_len, dtype=torch.long)
    positions = torch.arange(_LENGTH).unsqueeze(0)
    completion = (positions >= prompt_lengths.unsqueeze(1)).to(dtype=torch.long)
    return chosen, attention, completion, rejected, attention.clone(), completion.clone()


def _finish_run(
    config: TrainingConfig,
    *,
    models: Mapping[str, nn.Module],
    optimizer: Optimizer,
    step: int,
    trainer_state: Mapping[str, float],
    losses: list[float],
    metrics: list[dict[str, float]],
) -> RunOutput:
    checkpoint = config.checkpoint_path
    if checkpoint is not None:
        save_checkpoint(
            checkpoint,
            models=models,
            optimizer=optimizer,
            step=step,
            config=config,
            trainer_state=trainer_state,
        )
    output_path = Path(config.output_dir) / "run.json"
    output = RunOutput(
        seed=config.seed,
        step=step,
        losses=losses,
        metrics=metrics,
        output_path=str(output_path),
        checkpoint_path=checkpoint,
    )
    _atomic_text(
        output_path,
        json.dumps(
            {
                "checkpoint_path": output.checkpoint_path,
                "config_hash": config_hash(config),
                "losses": output.losses,
                "metrics": output.metrics,
                "seed": output.seed,
                "step": output.step,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return output


def _atomic_text(path: Path, text: str) -> None:
    """Write ``text`` via a temp file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
