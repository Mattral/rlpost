"""Reward-model training.

The Hugging Face path loads a sequence classifier and wraps it with LoRA.
That import of peft happens inside the loader. ``import rlpost.reward_engine``
does not import peft. This path does not construct a Hugging Face ``Trainer``.

``train-reward --tiny`` does not use that path. It trains ``TinyRewardModel``
on a fixed list of preference pairs. The loss is the Bradley-Terry term
``-log sigmoid(r_chosen - r_rejected)``, averaged over the batch.
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import AdamW

import rlpost.repro as repro
from rlpost.config import ModelConfig, TrainingConfig
from rlpost.dataset import PreferencePairDataset, ToyDatasetLoader
from rlpost.dpo_engine import collate_completions

logger = logging.getLogger(__name__)


def load_reward_model_with_lora(config: ModelConfig) -> Module:
    """Load reward model (sequence classifier) with LoRA.

    The reward model takes text as input and outputs a scalar reward score
    using a binary classification head.

    Args:
        config: ModelConfig with reward model ID and LoRA settings

    Returns:
        Model with LoRA applied, ready for preference pair training

    Raises:
        ValueError: If model not found
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification

    # Load sequence classification model (num_labels=1 for scalar output)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_id,
        num_labels=1,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Reward model trainable params: {trainable_params:,} / "
        f"{total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    return model


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


class RewardModelTrainer:
    """Trains a reward model with a Bradley-Terry loss on preference pairs.

    The loss is ``-log sigmoid(r_chosen - r_rejected)``. The reward model
    learns to score chosen above rejected.

    Example:
        >>> config = TrainingConfig.toy_mode()
        >>> trainer = RewardModelTrainer(config)
        >>> trainer.train("output/reward_model")
        >>> reward = trainer.score("Prompt here", "Response here")
    """

    def __init__(self, config: TrainingConfig, use_toy: bool = False):
        """Initialize reward model trainer.

        Args:
            config: TrainingConfig with training settings
            use_toy: If True, use 1K toy dataset

        Raises:
            ValueError: If model loading fails
        """
        self.config = config
        self.use_toy = use_toy

        from transformers import AutoTokenizer

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.reward_model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with LoRA
        self.model = load_reward_model_with_lora(config.model)

        self.use_wandb = getattr(config, "log_to_wandb", True)

    def _tokenize_preference_pair(self, prompt: str, response: str) -> dict:
        """Tokenize prompt and response together.

        Args:
            prompt: Prompt text
            response: Response text

        Returns:
            Dict with input_ids, attention_mask, token_type_ids
        """
        text = prompt + response
        tokens = self.tokenizer(
            text,
            max_length=self.config.dataset.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in tokens.items()}  # Remove batch dim

    def compute_loss(
        self,
        chosen_scores: torch.Tensor,
        rejected_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bradley-Terry loss and the fraction of pairs with chosen above rejected.

        The loss is ``bradley_terry_loss``. That is the mean of
        ``-log sigmoid(r_chosen - r_rejected)``. It is the same number as
        ``binary_cross_entropy_with_logits(r_chosen - r_rejected, 1)``.

        Args:
            chosen_scores: ``[B]`` scores for the chosen responses.
            rejected_scores: ``[B]`` scores for the rejected responses.

        Returns:
            ``(loss, accuracy)``. Accuracy is the fraction of rows with
            ``chosen > rejected``.
        """
        loss = bradley_terry_loss(chosen_scores, rejected_scores)
        accuracy = (chosen_scores > rejected_scores).float().mean()
        return loss, accuracy

    def train(
        self,
        output_dir: str,
        num_train_epochs: int = 3,
    ) -> dict:
        """Train reward model on preference pairs.

        Args:
            output_dir: Directory to save checkpoints
            num_train_epochs: Number of training epochs

        Returns:
            Dict with training metrics
        """
        # Load dataset
        if self.use_toy:
            pairs = ToyDatasetLoader.load()
        else:
            dataset = PreferencePairDataset(self.config.dataset)
            dataset.load()
            pairs = list(dataset)

        logger.info(f"Training reward model on {len(pairs)} pairs")

        # Create training loop (simplified - normally use Trainer)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimization.learning_rate,
        )

        total_loss = 0.0
        total_accuracy = 0.0
        step = 0

        for epoch in range(num_train_epochs):
            self.model.train()
            for i, pair in enumerate(pairs):
                # Tokenize
                chosen_tokens = self._tokenize_preference_pair(pair.prompt, pair.chosen)
                rejected_tokens = self._tokenize_preference_pair(pair.prompt, pair.rejected)

                # Forward pass
                chosen_output = self.model(
                    input_ids=chosen_tokens["input_ids"].unsqueeze(0),
                    attention_mask=chosen_tokens.get("attention_mask", None),
                )
                rejected_output = self.model(
                    input_ids=rejected_tokens["input_ids"].unsqueeze(0),
                    attention_mask=rejected_tokens.get("attention_mask", None),
                )

                chosen_score = chosen_output.logits.squeeze(-1)
                rejected_score = rejected_output.logits.squeeze(-1)

                # Compute loss
                loss, accuracy = self.compute_loss(chosen_score, rejected_score)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimization.max_grad_norm or 1.0,
                )
                optimizer.step()

                # Log metrics
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                step += 1

                if (i + 1) % 100 == 0:
                    avg_loss = total_loss / step
                    avg_acc = total_accuracy / step
                    logger.info(
                        f"Epoch {epoch + 1}, Step {i + 1}: "
                        f"Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}"
                    )

        avg_loss = total_loss / step
        avg_acc = total_accuracy / step
        logger.info(f"Training complete: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

        return {"loss": avg_loss, "accuracy": avg_acc}

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        """Score a response given a prompt.

        Args:
            prompt: Prompt text
            response: Response text

        Returns:
            Scalar reward score
        """
        self.model.eval()

        tokens = self._tokenize_preference_pair(prompt, response)
        output = self.model(
            input_ids=tokens["input_ids"].unsqueeze(0),
            attention_mask=tokens.get("attention_mask", None),
        )

        reward = output.logits.squeeze(-1).item()
        return reward

    @torch.no_grad()
    def score_batch(self, prompts: list[str], responses: list[str]) -> list[float]:
        """Score multiple responses in batch.

        Args:
            prompts: List of prompt texts
            responses: List of response texts

        Returns:
            List of scalar reward scores

        Raises:
            ValueError: If lengths don't match
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have same length")

        self.model.eval()

        rewards = []
        for prompt, response in zip(prompts, responses):
            tokens = self._tokenize_preference_pair(prompt, response)
            output = self.model(
                input_ids=tokens["input_ids"].unsqueeze(0),
                attention_mask=tokens.get("attention_mask", None),
            )
            reward = output.logits.squeeze(-1).item()
            rewards.append(reward)

        return rewards

    def save_model(self, output_dir: str) -> None:
        """Save reward model.

        Args:
            output_dir: Directory to save model
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Reward model saved to {output_dir}")

    def load_for_inference(self, model_dir: str) -> None:
        """Load saved reward model.

        Args:
            model_dir: Directory with saved model
        """
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.reward_model_id,
            num_labels=1,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, model_dir)


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


if __name__ == "__main__":
    """Quick test of reward model trainer."""
    config = TrainingConfig.toy_mode()
    trainer = RewardModelTrainer(config, use_toy=True)

    # Train for 1 epoch on toy dataset
    result = trainer.train(
        output_dir="output/reward_model_toy",
        num_train_epochs=1,
    )
    print(f"Training result: {result}")

    # Score sample responses
    reward_chosen = trainer.score(prompt="What is 2+2?", response="The answer is 4.")
    reward_rejected = trainer.score(prompt="What is 2+2?", response="The answer is 5.")
    print(f"Chosen reward: {reward_chosen:.4f}")
    print(f"Rejected reward: {reward_rejected:.4f}")
    print(f"Difference: {reward_chosen - reward_rejected:.4f}")

    # Save model
    trainer.save_model("output/reward_model_toy_checkpoint")
