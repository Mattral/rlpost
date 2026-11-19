"""Supervised fine-tuning.

The only training path is ``run_sft_training``. It trains a causal LM
the caller already built, on a fixed list of prompt and completion
strings. The loss is the mean negative log-likelihood of the completion
tokens. Prompt tokens are not targets. Logits at index ``t`` score the
token at index ``t + 1``, so the logit on the last prompt position is
kept: it predicts the first completion token.

S0.11e builds the real model and data path on this loop.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import AdamW

import rlpost.repro as repro
from rlpost.config import TrainingConfig
from rlpost.dpo_engine import collate_completions


def completion_cross_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean NLL of the completion tokens.

    Args:
        logits: ``[B, T, V]``. Index ``t`` scores ``input_ids`` at ``t + 1``.
        input_ids: ``[B, T]``, integer token ids.
        completion_mask: ``[B, T]``. 1 on a completion token, 0 on the
            prompt prefix and on padding.

    Returns:
        A scalar. The kept targets are ``completion_mask[:, 1:]``. A
        prompt token is not a target. The logit on the last prompt
        position is kept when it predicts a completion token.

    Raises:
        ValueError: The shapes disagree, ``T < 2``, the ids are not
            integers, or the shifted mask is empty.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}")
    if input_ids.ndim != 2 or completion_mask.ndim != 2:
        raise ValueError(
            "input_ids and completion_mask must be [B, T], got "
            f"{tuple(input_ids.shape)} and {tuple(completion_mask.shape)}"
        )
    if logits.shape[:2] != input_ids.shape or completion_mask.shape != input_ids.shape:
        raise ValueError(
            "logits, input_ids, and completion_mask must share [B, T], got "
            f"{tuple(logits.shape)}, {tuple(input_ids.shape)}, {tuple(completion_mask.shape)}"
        )
    if input_ids.shape[1] < 2:
        raise ValueError(f"T must be at least 2, got {input_ids.shape[1]}")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"input_ids must be int32 or int64, got {input_ids.dtype}")
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:].to(device=shift_logits.device)
    shift_mask = completion_mask[:, 1:].to(device=shift_logits.device, dtype=shift_logits.dtype)
    kept = shift_mask.sum()
    if int(kept.item()) == 0:
        raise ValueError("completion mask selects no tokens")
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
    return -(gathered * shift_mask).sum() / kept


# Fixed strings for ``train-sft --tiny``. Lengths differ so the collator
# right-pads. Each step draws a batch with the RNGs the checkpoint saves.
TINY_SFT_ROWS: tuple[dict[str, str], ...] = (
    {"prompt": "Q: hi ", "completion": "yes"},
    {"prompt": "Q: two ", "completion": "ok now"},
    {"prompt": "Go ", "completion": "left"},
    {"prompt": "Say ", "completion": "cat"},
    {"prompt": "Ask ", "completion": "alpha"},
    {"prompt": "Item ", "completion": "the box"},
)


@dataclass
class SFTRunOutput:
    """What one call of ``run_sft_training`` did.

    ``data_loss_before`` is the completion-token NLL of every tiny row
    before the updates in this call. ``data_loss_after`` is that NLL
    after them. A resumed call measures ``data_loss_before`` on the
    loaded weights.
    """

    seed: int
    step: int
    losses: list[float]
    metrics: list[dict[str, float]]
    output_path: str
    checkpoint_path: str | None
    data_loss_before: float
    data_loss_after: float


def tiny_sft_config(
    *,
    output_dir: str,
    seed: int,
    batch_size: int,
    checkpoint_path: str | None,
) -> TrainingConfig:
    """CPU config for ``train-sft --tiny``.

    Starts from ``TrainingConfig.toy_mode`` and then sets the fields this
    command uses. The learning rate is ``1e-2`` so a few CPU steps move
    this model. That is not a tuned rate. ``device`` is ``cpu``.
    ``log_to_wandb`` is false. ``max_seq_length`` is 32. The model id is
    the string ``tiny_lm`` and is not loaded.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")
    config = TrainingConfig.toy_mode()
    config.seed = seed
    config.device = "cpu"
    config.log_to_wandb = False
    config.output_dir = output_dir
    config.checkpoint_path = checkpoint_path
    config.model.policy_model_id = "tiny_lm"
    config.model.reference_model_id = "tiny_lm"
    config.optimization.learning_rate = 1e-2
    config.dataset.batch_size = batch_size
    config.dataset.max_seq_length = 32
    return config


def _pad_token_id(tokenizer: object) -> int:
    """Read ``pad_token_id``. A missing id is an error, not a default of 0."""
    pad = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad, bool) or not isinstance(pad, int):
        raise TypeError("tokenizer.pad_token_id must be an int")
    return pad


def sample_sft_batch(
    rows: Sequence[Mapping[str, str]],
    batch_size: int,
) -> dict[str, list[str]]:
    """Pick ``batch_size`` rows. Python, NumPy, and torch each draw once.

    The strings are fixed. Which rows appear is not. A resumed run
    restores those three RNGs, so the next batch matches the
    uninterrupted run. This is not a ``DataLoader``.
    """
    count = len(rows)
    if count < 1:
        raise ValueError("rows must be non-empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")
    start = random.randrange(count)
    stride = int(np.random.randint(1, count)) if count > 1 else 1
    shift = int(torch.randint(0, count, ()).item())
    indices = [(start + stride * i + shift) % count for i in range(batch_size)]
    return {
        "prompt": [str(rows[index]["prompt"]) for index in indices],
        "completion": [str(rows[index]["completion"]) for index in indices],
    }


def sft_data_loss(
    model: Module,
    rows: Sequence[Mapping[str, str]],
    tokenizer: object,
    *,
    max_length: int,
    pad_token_id: int,
) -> float:
    """Completion-token NLL of every row, with no gradient."""
    model.eval()
    batch = collate_completions(
        tokenizer,
        [str(row["prompt"]) for row in rows],
        [str(row["completion"]) for row in rows],
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = completion_cross_entropy(logits, batch["input_ids"], batch["completion_mask"])
    return float(loss.detach())


def run_sft_training(
    model: Module,
    optimizer: AdamW,
    config: TrainingConfig,
    n_steps: int,
    rows: Sequence[Mapping[str, str]],
    tokenizer: object,
    *,
    resume_from: str | Path | None = None,
) -> SFTRunOutput:
    """Run ``n_steps`` completion-masked updates on ``tiny_lm``.

    Calls ``seed_everything`` once. A resumed call then loads the
    checkpoint over ``model`` and ``optimizer``. Each step samples a
    batch and takes one gradient step on the completion-token NLL.
    ``optimization.max_grad_norm`` is the clip threshold. When
    ``checkpoint_path`` is set, the checkpoint is written once, at the
    end of this call. ``trainer_state`` is empty.

    Args:
        model: Policy. ``train-sft --tiny`` passes ``tiny_causal_lm``.
        optimizer: AdamW built on ``model.parameters()``.
        config: Tiny CPU config.
        n_steps: Updates in this call. At least 1.
        rows: Mappings with ``prompt`` and ``completion``.
        tokenizer: Callable returning ``input_ids``, with an int pad id.
        resume_from: Checkpoint to load before the updates. None starts
            from the model as constructed.

    Returns:
        Batch losses for this call, and the full-list NLL before and after.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be an int >= 1, got {n_steps!r}")
    repro.seed_everything(config.seed)
    models = {"policy": model}
    step = 0
    if resume_from is not None:
        loaded = repro.load_checkpoint(
            resume_from,
            models=models,
            optimizer=optimizer,
            config=config,
        )
        step = loaded.step
    pad_token_id = _pad_token_id(tokenizer)
    max_length = config.dataset.max_seq_length
    before = sft_data_loss(
        model,
        rows,
        tokenizer,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    losses: list[float] = []
    metrics: list[dict[str, float]] = []
    batch_size = config.dataset.batch_size
    clip = config.optimization.max_grad_norm
    for _ in range(n_steps):
        sampled = sample_sft_batch(rows, batch_size)
        batch = collate_completions(
            tokenizer,
            sampled["prompt"],
            sampled["completion"],
            max_length=max_length,
            pad_token_id=pad_token_id,
        )
        model.train()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = completion_cross_entropy(logits, batch["input_ids"], batch["completion_mask"])
        value = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        step += 1
        losses.append(value)
        metrics.append({"loss": value})
    after = sft_data_loss(
        model,
        rows,
        tokenizer,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    return _finish_sft_run(
        config,
        models=models,
        optimizer=optimizer,
        step=step,
        losses=losses,
        metrics=metrics,
        data_loss_before=before,
        data_loss_after=after,
    )


def _finish_sft_run(
    config: TrainingConfig,
    *,
    models: Mapping[str, Module],
    optimizer: AdamW,
    step: int,
    losses: list[float],
    metrics: list[dict[str, float]],
    data_loss_before: float,
    data_loss_after: float,
) -> SFTRunOutput:
    """Write the checkpoint, if any, and ``run.json``."""
    checkpoint = config.checkpoint_path
    if checkpoint is not None:
        repro.save_checkpoint(
            checkpoint,
            models=models,
            optimizer=optimizer,
            step=step,
            config=config,
            trainer_state={},
        )
    output_path = Path(config.output_dir) / "run.json"
    output = SFTRunOutput(
        seed=config.seed,
        step=step,
        losses=losses,
        metrics=metrics,
        output_path=str(output_path),
        checkpoint_path=checkpoint,
        data_loss_before=data_loss_before,
        data_loss_after=data_loss_after,
    )
    _atomic_text(
        output_path,
        json.dumps(
            {
                "checkpoint_path": output.checkpoint_path,
                "config_hash": repro.config_hash(config),
                "data_loss_after": output.data_loss_after,
                "data_loss_before": output.data_loss_before,
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
