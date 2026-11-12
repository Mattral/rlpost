"""Reward-model training.

The only training path is ``run_reward_training``. It trains a scorer
the caller already built, on a fixed list of preference pairs. The loss
is the Bradley-Terry term ``-log sigmoid(r_chosen - r_rejected)``,
averaged over the batch.

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


def bradley_terry_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
) -> torch.Tensor:
    """Mean of ``-log sigmoid(r_chosen - r_rejected)``.

    Both tensors are ``[B]``. Equal scores give ``ln 2``. This is the
    Bradley-Terry preference loss. It matches
    ``binary_cross_entropy_with_logits(chosen - rejected, 1)``.

    Raises:
        ValueError: The shapes differ, the rank is not 1, or the batch is empty.
    """
    if chosen_scores.shape != rejected_scores.shape:
        raise ValueError(
            "chosen and rejected scores must share a shape, got "
            f"{tuple(chosen_scores.shape)} and {tuple(rejected_scores.shape)}"
        )
    if chosen_scores.ndim != 1 or chosen_scores.numel() == 0:
        raise ValueError(f"scores must be a non-empty [B] tensor, got {tuple(chosen_scores.shape)}")
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()


# Fixed pairs for ``train-reward --tiny``. Chosen and rejected use
# different characters so a bag-of-embeddings scorer can separate them.
# Each step draws a batch with the RNGs the checkpoint saves.
TINY_REWARD_PAIRS: tuple[dict[str, str], ...] = (
    {"prompt": "Rate ", "chosen": "aaa", "rejected": "zzz"},
    {"prompt": "Rate ", "chosen": "bbb", "rejected": "yyy"},
    {"prompt": "Pick ", "chosen": "ccc", "rejected": "xxx"},
    {"prompt": "Pick ", "chosen": "ddd", "rejected": "www"},
    {"prompt": "Mark ", "chosen": "eee", "rejected": "vvv"},
    {"prompt": "Mark ", "chosen": "fff", "rejected": "uuu"},
)


@dataclass
class RewardRunOutput:
    """What one call of ``run_reward_training`` did.

    ``margin_before`` is the mean of ``r_chosen - r_rejected`` on every
    tiny pair before the updates in this call. ``margin_after`` is that
    mean after them.
    """

    seed: int
    step: int
    losses: list[float]
    metrics: list[dict[str, float]]
    output_path: str
    checkpoint_path: str | None
    margin_before: float
    margin_after: float


def tiny_reward_config(
    *,
    output_dir: str,
    seed: int,
    batch_size: int,
    checkpoint_path: str | None,
) -> TrainingConfig:
    """CPU config for ``train-reward --tiny``.

    Starts from ``TrainingConfig.toy_mode`` and then sets the fields this
    command uses. The learning rate is ``1e-1`` so the tiny scorer can
    separate the fixed pairs in a few steps. That is not a tuned rate.
    ``device`` is ``cpu``. ``log_to_wandb`` is false. ``max_seq_length``
    is 32. The model id is the string ``tiny_lm`` and is not loaded.
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
    config.model.reward_model_id = "tiny_lm"
    config.optimization.learning_rate = 1e-1
    config.dataset.batch_size = batch_size
    config.dataset.max_seq_length = 32
    return config


def _pad_token_id(tokenizer: object) -> int:
    """Read ``pad_token_id``. A missing id is an error, not a default of 0."""
    pad = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad, bool) or not isinstance(pad, int):
        raise TypeError("tokenizer.pad_token_id must be an int")
    return pad


def sample_reward_batch(
    pairs: Sequence[Mapping[str, str]],
    batch_size: int,
) -> dict[str, list[str]]:
    """Pick ``batch_size`` pairs. Python, NumPy, and torch each draw once.

    The strings are fixed. Which rows appear is not. A resumed run
    restores those three RNGs. This is not a ``DataLoader``.
    """
    count = len(pairs)
    if count < 1:
        raise ValueError("pairs must be non-empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")
    start = random.randrange(count)
    stride = int(np.random.randint(1, count)) if count > 1 else 1
    shift = int(torch.randint(0, count, ()).item())
    indices = [(start + stride * i + shift) % count for i in range(batch_size)]
    return {
        "prompt": [str(pairs[index]["prompt"]) for index in indices],
        "chosen": [str(pairs[index]["chosen"]) for index in indices],
        "rejected": [str(pairs[index]["rejected"]) for index in indices],
    }


def score_preference_pairs(
    model: Module,
    pairs: Sequence[Mapping[str, str]],
    tokenizer: object,
    *,
    max_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every pair. Returns ``(chosen, rejected)``, each ``[B]``.

    No gradient. The model is left in eval mode.
    """
    model.eval()
    prompts = [str(pair["prompt"]) for pair in pairs]
    chosen = collate_completions(
        tokenizer,
        prompts,
        [str(pair["chosen"]) for pair in pairs],
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    rejected = collate_completions(
        tokenizer,
        prompts,
        [str(pair["rejected"]) for pair in pairs],
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    with torch.no_grad():
        chosen_scores = model(
            input_ids=chosen["input_ids"],
            attention_mask=chosen["attention_mask"],
        )
        rejected_scores = model(
            input_ids=rejected["input_ids"],
            attention_mask=rejected["attention_mask"],
        )
    if not isinstance(chosen_scores, torch.Tensor) or not isinstance(rejected_scores, torch.Tensor):
        raise TypeError("the reward model must return a tensor of scores")
    return chosen_scores, rejected_scores


def run_reward_training(
    model: Module,
    optimizer: AdamW,
    config: TrainingConfig,
    n_steps: int,
    pairs: Sequence[Mapping[str, str]],
    tokenizer: object,
    *,
    resume_from: str | Path | None = None,
) -> RewardRunOutput:
    """Run ``n_steps`` Bradley-Terry updates on the tiny reward model.

    Calls ``seed_everything`` once. A resumed call then loads the
    checkpoint over ``model`` and ``optimizer``. Each step samples a
    batch, scores chosen and rejected, and steps
    ``bradley_terry_loss``. ``optimization.max_grad_norm`` is the clip
    threshold. When ``checkpoint_path`` is set, the checkpoint is written
    once, at the end. ``trainer_state`` is empty.

    Args:
        model: Scorer. ``train-reward --tiny`` passes ``tiny_reward_model``.
            ``forward(input_ids, attention_mask)`` returns ``[B]``.
        optimizer: AdamW built on ``model.parameters()``.
        config: Tiny CPU config.
        n_steps: Updates in this call. At least 1.
        pairs: Mappings with ``prompt``, ``chosen``, and ``rejected``.
        tokenizer: Callable returning ``input_ids``, with an int pad id.
        resume_from: Checkpoint to load before the updates.

    Returns:
        Batch losses for this call, and the full-list margin before and after.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be an int >= 1, got {n_steps!r}")
    repro.seed_everything(config.seed)
    models = {"reward": model}
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
    before_chosen, before_rejected = score_preference_pairs(
        model,
        pairs,
        tokenizer,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    margin_before = float((before_chosen - before_rejected).mean())
    losses: list[float] = []
    metrics: list[dict[str, float]] = []
    batch_size = config.dataset.batch_size
    clip = config.optimization.max_grad_norm
    for _ in range(n_steps):
        sampled = sample_reward_batch(pairs, batch_size)
        chosen = collate_completions(
            tokenizer,
            sampled["prompt"],
            sampled["chosen"],
            max_length=max_length,
            pad_token_id=pad_token_id,
        )
        rejected = collate_completions(
            tokenizer,
            sampled["prompt"],
            sampled["rejected"],
            max_length=max_length,
            pad_token_id=pad_token_id,
        )
        model.train()
        chosen_scores = model(
            input_ids=chosen["input_ids"],
            attention_mask=chosen["attention_mask"],
        )
        rejected_scores = model(
            input_ids=rejected["input_ids"],
            attention_mask=rejected["attention_mask"],
        )
        loss = bradley_terry_loss(chosen_scores, rejected_scores)
        value = float(loss.detach())
        accuracy = float((chosen_scores > rejected_scores).detach().float().mean())
        margin = float((chosen_scores - rejected_scores).detach().mean())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        step += 1
        losses.append(value)
        metrics.append({"accuracy": accuracy, "loss": value, "margin": margin})
    after_chosen, after_rejected = score_preference_pairs(
        model,
        pairs,
        tokenizer,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    margin_after = float((after_chosen - after_rejected).mean())
    return _finish_reward_run(
        config,
        models=models,
        optimizer=optimizer,
        step=step,
        losses=losses,
        metrics=metrics,
        margin_before=margin_before,
        margin_after=margin_after,
    )


def _finish_reward_run(
    config: TrainingConfig,
    *,
    models: Mapping[str, Module],
    optimizer: AdamW,
    step: int,
    losses: list[float],
    metrics: list[dict[str, float]],
    margin_before: float,
    margin_after: float,
) -> RewardRunOutput:
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
    output = RewardRunOutput(
        seed=config.seed,
        step=step,
        losses=losses,
        metrics=metrics,
        output_path=str(output_path),
        checkpoint_path=checkpoint,
        margin_before=margin_before,
        margin_after=margin_after,
    )
    _atomic_text(
        output_path,
        json.dumps(
            {
                "checkpoint_path": output.checkpoint_path,
                "config_hash": repro.config_hash(config),
                "losses": output.losses,
                "margin_after": output.margin_after,
                "margin_before": output.margin_before,
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
