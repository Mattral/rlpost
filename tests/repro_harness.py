"""Synthetic PPO and DPO loops for the CPU resume tests.

``rlpost.repro`` seeds a run and writes the checkpoint. The N-step loop,
the batches, and ``run.json`` live here. These batches are token ids.
The CLI loop that tokenizes preference text is
``rlpost.dpo_engine.run_dpo_training``.
"""

from __future__ import annotations

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

import rlpost.repro as repro
from rlpost.config import TrainingConfig
from rlpost.dpo_engine import DPOTrainer
from rlpost.ops import gather_logprobs, padding_attention
from rlpost.ppo_engine import PPOTrainer

_BATCH = 2
_LENGTH = 8


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
    repro.seed_everything(trainer.config.seed)
    models = {
        "policy": trainer.policy_model,
        "value": trainer.value_model,
        "reference": trainer.reference_model,
    }
    if resume_from is not None:
        loaded = repro.load_checkpoint(
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
    repro.seed_everything(trainer.config.seed)
    models = {
        "policy": trainer.policy_model,
        "reference": trainer.reference_model,
    }
    if resume_from is not None:
        loaded = repro.load_checkpoint(
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
        repro.save_checkpoint(
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
                "config_hash": repro.config_hash(config),
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
