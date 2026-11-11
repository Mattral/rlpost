"""Direct Preference Optimization (DPO). Reward-model-free, with a frozen reference policy.

This module provides:
- DPO trainer for direct preference learning on pair data
- DPO loss computation without explicit reward model
- Batch processing and W&B logging
- Comparable training loop to PPO
"""

import copy
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
from transformers import AutoModelForCausalLM, AutoTokenizer

import rlpost.repro as repro
from rlpost.config import TrainingConfig
from rlpost.dataset import ToyDatasetLoader
from rlpost.ops import gather_logprobs

logger = logging.getLogger(__name__)


def prompt_prefix_length(prompt_ids: list[int], full_ids: list[int]) -> int:
    """How many leading ids ``prompt_ids`` and ``full_ids`` share.

    The completion starts at this index. A BPE merge at the boundary makes
    the prompt not a prefix; those diverging prompt tokens are left out of
    the completion, which is the same cut TRL 1.17.1 documents.
    """
    length = 0
    for left, right in zip(prompt_ids, full_ids):
        if left != right:
            break
        length += 1
    return length


def completion_token_mask(
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """1 on completion tokens, 0 on the prompt prefix and on padding.

    Args:
        attention_mask: ``[B, T]``, 1 on real tokens.
        prompt_lengths: ``[B]``, count of leading prompt tokens in each row.

    Returns:
        ``[B, T]``, same dtype as ``attention_mask``. Index ``t`` is a
        completion token when ``t >= prompt_lengths`` and the attention
        mask is 1. The log-prob reduction shifts this mask by one, because
        the score at ``t`` belongs to token ``t + 1``.
    """
    if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != attention_mask.shape[0]:
        raise ValueError(
            f"prompt_lengths shape {tuple(prompt_lengths.shape)} must be "
            f"[{attention_mask.shape[0]}]"
        )
    positions = torch.arange(attention_mask.shape[-1], device=attention_mask.device)
    prompt = prompt_lengths.to(device=attention_mask.device).unsqueeze(1)
    return ((positions >= prompt) & attention_mask.bool()).to(dtype=attention_mask.dtype)


def _as_rows(value: object, name: str) -> list[str]:
    """One string is one row. A list or tuple of strings is a batch."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        if len(value) == 0:
            raise ValueError(f"{name} is empty")
        return list(value)
    raise TypeError(f"{name} must be a string or a sequence of strings")


def _pad_token_id(tokenizer: object) -> int:
    """Read ``pad_token_id``. A missing id is an error, not a default of 0."""
    pad = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad, bool) or not isinstance(pad, int):
        raise TypeError("tokenizer.pad_token_id must be an int")
    return pad


def _token_ids(tokenizer: object, text: str, max_length: int) -> list[int]:
    """Token ids for ``text``, truncated to ``max_length``.

    The tokenizer is called without ``return_tensors``. A tensor or a
    batched list is flattened to one row.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer must return a mapping with input_ids")
    ids = encoded["input_ids"]
    if isinstance(ids, torch.Tensor):
        flat = ids.detach().cpu().reshape(-1).tolist()
    elif isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        flat = list(ids[0])
    elif isinstance(ids, (list, tuple)):
        flat = list(ids)
    else:
        raise TypeError("input_ids must be a list or a tensor")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in flat):
        raise TypeError("input_ids must be ints")
    return [int(token) for token in flat[:max_length]]


def collate_completions(
    tokenizer: object,
    prompts: Sequence[str],
    responses: Sequence[str],
    *,
    max_length: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Right-pad one prompt and response per row.

    Each row is ``prompt + response``, truncated to ``max_length``. The
    prompt length is the shared prefix of the prompt ids and that row.
    Rows are then padded on the right to the longest row in the batch.
    The completion mask is 0 on the prompt prefix and on padding.

    Left padding is not implemented. A left pad is not part of the prompt
    prefix, so this mask would mark the wrong tokens.

    Args:
        tokenizer: Callable returning ``input_ids``. Must not left-pad.
        prompts: One prompt string per row.
        responses: One response string per row. Same length as ``prompts``.
        max_length: Truncation length for each row.
        pad_token_id: Written into the padded columns.

    Returns:
        ``input_ids`` ``[B, T]``, ``attention_mask`` ``[B, T]``,
        ``completion_mask`` ``[B, T]``, and ``prompt_lengths`` ``[B]``.
        ``T`` is the longest row after truncation.
    """
    if len(prompts) != len(responses):
        raise ValueError(
            "prompts and responses must have the same length, "
            f"got {len(prompts)} and {len(responses)}"
        )
    if len(prompts) == 0:
        raise ValueError("prompts is empty")
    rows: list[list[int]] = []
    prompt_lengths: list[int] = []
    for prompt, response in zip(prompts, responses, strict=True):
        prompt_ids = _token_ids(tokenizer, prompt, max_length)
        full_ids = _token_ids(tokenizer, prompt + response, max_length)
        if not full_ids:
            raise ValueError("a row has no tokens")
        rows.append(full_ids)
        prompt_lengths.append(prompt_prefix_length(prompt_ids, full_ids))
    width = max(len(row) for row in rows)
    batch = len(rows)
    input_ids = torch.full((batch, width), pad_token_id, dtype=torch.long)
    attention = torch.zeros(batch, width, dtype=torch.long)
    for index, row in enumerate(rows):
        length = len(row)
        input_ids[index, :length] = torch.tensor(row, dtype=torch.long)
        attention[index, :length] = 1
    lengths = torch.tensor(prompt_lengths, dtype=torch.long)
    completion = completion_token_mask(attention, lengths)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "completion_mask": completion,
        "prompt_lengths": lengths,
    }


def collate_preference_pairs(
    tokenizer: object,
    prompts: Sequence[str],
    chosen: Sequence[str],
    rejected: Sequence[str],
    *,
    max_length: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Right-pad a batch of preference pairs.

    Chosen and rejected are padded separately, so the two sides can have
    different widths. Each row keeps its own prompt length. The prompt
    text is the same string on both sides. The prefix length can still
    differ when tokenization merges across the boundary.

    Args:
        tokenizer: Callable returning ``input_ids``.
        prompts: One prompt per pair.
        chosen: Chosen response per pair.
        rejected: Rejected response per pair.
        max_length: Truncation length for each row.
        pad_token_id: Written into the padded columns.

    Returns:
        Chosen and rejected ``input_ids``, ``attention_mask``,
        ``completion_mask``, each ``[B, T]`` with that side's ``T``, and
        ``prompt_lengths`` ``[B]`` for each side.
    """
    count = len(prompts)
    if not (count == len(chosen) == len(rejected)):
        raise ValueError(
            "prompt, chosen, and rejected must have the same length, "
            f"got {count}, {len(chosen)}, {len(rejected)}"
        )
    chosen_side = collate_completions(
        tokenizer,
        prompts,
        chosen,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    rejected_side = collate_completions(
        tokenizer,
        prompts,
        rejected,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    return {
        "chosen_ids": chosen_side["input_ids"],
        "chosen_attention_mask": chosen_side["attention_mask"],
        "chosen_completion_mask": chosen_side["completion_mask"],
        "chosen_prompt_lengths": chosen_side["prompt_lengths"],
        "rejected_ids": rejected_side["input_ids"],
        "rejected_attention_mask": rejected_side["attention_mask"],
        "rejected_completion_mask": rejected_side["completion_mask"],
        "rejected_prompt_lengths": rejected_side["prompt_lengths"],
    }


class DPOMetrics:
    """Metrics container for DPO training step.

    Attributes:
        dpo_loss: Mean sigmoid DPO loss.
        policy_diff_mean: Mean of log π(y_c|x) - log π(y_r|x), not beta-scaled.
        margin_mean: Mean of chosen_reward - rejected_reward. Each reward is
            beta * (log π - log π_ref).
        accuracy: Fraction of rows with chosen_reward > rejected_reward.
    """

    def __init__(
        self,
        dpo_loss: float,
        policy_diff_mean: float,
        margin_mean: float,
        accuracy: float,
    ):
        self.dpo_loss = dpo_loss
        self.policy_diff_mean = policy_diff_mean
        self.margin_mean = margin_mean
        self.accuracy = accuracy

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "dpo_loss": self.dpo_loss,
            "policy_diff_mean": self.policy_diff_mean,
            "margin_mean": self.margin_mean,
            "accuracy": self.accuracy,
        }


class DPOTrainer:
    """Direct Preference Optimization trainer.

    DPO directly optimizes the policy on preference pairs without learning
    a separate reward model. Uses the log-likelihood ratio to compute
    preferences directly from policy predictions.

    Mathematical objective (Rafailov et al. 2023, the sigmoid DPO loss):

        chosen_reward   = β * (log π(y_c|x) - log π_ref(y_c|x))
        rejected_reward = β * (log π(y_r|x) - log π_ref(y_r|x))
        L_DPO = -E[log σ(chosen_reward - rejected_reward)]

    log π(y|x) is the sum of completion-token log-probabilities, not the mean.
    Prompt tokens are outside that sum. β is ``AlignmentConfig.dpo_beta``.

    Example:
        >>> config = TrainingConfig.toy_mode()
        >>> trainer = DPOTrainer(config)
        >>> metrics = trainer.train_step(batch)
        >>> print(f"DPO Loss: {metrics.dpo_loss:.4f}")
    """

    def __init__(
        self,
        config: TrainingConfig,
        use_toy: bool = False,
        policy_model: Module | None = None,
        reference_model: Module | None = None,
        tokenizer: object | None = None,
    ):
        """Initialize DPO trainer.

        Args:
            config: TrainingConfig with all training settings
            use_toy: If True, use 1K toy dataset for testing
            policy_model: Causal LM to train. When omitted, loaded from
                ``config.model.policy_model_id``.
            reference_model: Frozen reference. Required when ``policy_model``
                is passed. When omitted, loaded from
                ``config.model.reference_model_id``.
            tokenizer: Tokenizer for ``train_step``. Required for that path
                when the models are passed in. When the models are loaded
                here, the tokenizer is loaded with them.

        Raises:
            ValueError: If only one of the two models is passed, or they are
                the same module.
        """
        self.config = config
        self.use_toy = use_toy

        if (policy_model is None) != (reference_model is None):
            raise ValueError("Pass both policy_model and reference_model, or neither.")
        if policy_model is not None and policy_model is reference_model:
            raise ValueError("policy_model and reference_model must be different modules.")

        if policy_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model.policy_model_id,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.policy_model = AutoModelForCausalLM.from_pretrained(
                config.model.policy_model_id,
                device_map="auto",
                trust_remote_code=True,
            )
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                config.model.reference_model_id,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.tokenizer = tokenizer
            self.policy_model = policy_model
            self.reference_model = reference_model

        self.reference_model.requires_grad_(False)
        self.reference_model.eval()

        # Policy parameters only. The reference is frozen above.
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.optimization.learning_rate,
        )

        self.beta = config.alignment.dpo_beta
        self.step_count = 0

        logger.info(
            f"DPO Trainer initialized: policy={config.model.policy_model_id}, beta={self.beta}"
        )

    def _tokenize_for_loss(
        self, prompt: str, response: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize one prompt and response.

        The prompt is tokenized on its own and as a prefix of ``prompt + response``.
        The completion mask is 0 on that shared prefix and on padding, and 1 on
        the remaining tokens. One row is padded only up to its own length, so
        a single short pair has no pad columns. ``collate_completions`` is the
        implementation. It right-pads. A left-padded prompt is not a prefix,
        and this mask would then be wrong.

        Args:
            prompt: Prompt text.
            response: Response text.

        Returns:
            ``input_ids`` ``[1, T]``, ``attention_mask`` ``[1, T]``,
            ``completion_mask`` ``[1, T]``.
        """
        if self.tokenizer is None:
            raise RuntimeError("train_step requires a tokenizer.")
        collated = collate_completions(
            self.tokenizer,
            [prompt],
            [response],
            max_length=self.config.dataset.max_seq_length,
            pad_token_id=_pad_token_id(self.tokenizer),
        )
        return collated["input_ids"], collated["attention_mask"], collated["completion_mask"]

    def compute_dpo_loss(
        self,
        policy_logits_chosen: torch.Tensor,
        policy_logits_rejected: torch.Tensor,
        ref_logits_chosen: torch.Tensor,
        ref_logits_rejected: torch.Tensor,
    ) -> Tuple[torch.Tensor, DPOMetrics]:
        """Sigmoid DPO loss on sequence log-probabilities.

        ``label_smoothing`` is 0. TRL 1.17.1's sigmoid branch does not read
        that argument; smoothing lives on its other loss types.

            chosen_reward   = beta * (logp_policy_chosen - logp_ref_chosen)
            rejected_reward = beta * (logp_policy_rejected - logp_ref_rejected)
            loss            = -logsigmoid(chosen_reward - rejected_reward).mean()
            margin          = mean(chosen_reward - rejected_reward)
            accuracy        = mean(chosen_reward > rejected_reward)

        Args:
            policy_logits_chosen: Policy sequence log-probs, shape ``[B]``.
            policy_logits_rejected: Policy sequence log-probs, shape ``[B]``.
            ref_logits_chosen: Reference sequence log-probs, shape ``[B]``.
            ref_logits_rejected: Reference sequence log-probs, shape ``[B]``.

        Returns:
            Tuple of (loss, metrics). ``loss`` is a scalar tensor.
        """
        # Sequence log-probabilities, shape [B]. The names in the signature
        # say logits; the values are log π(y|x).
        policy_diff = policy_logits_chosen - policy_logits_rejected
        chosen_reward = self.beta * (policy_logits_chosen - ref_logits_chosen)
        rejected_reward = self.beta * (policy_logits_rejected - ref_logits_rejected)
        logits = chosen_reward - rejected_reward
        dpo_loss = -F.logsigmoid(logits).mean()

        # Margin is the beta-scaled, reference-adjusted gap. A tie is not a win.
        margin = logits.detach()
        accuracy = (chosen_reward.detach() > rejected_reward.detach()).float().mean().item()
        margin_mean = margin.mean().item()

        metrics = DPOMetrics(
            dpo_loss=dpo_loss.item(),
            policy_diff_mean=policy_diff.detach().mean().item(),
            margin_mean=margin_mean,
            accuracy=accuracy,
        )

        return dpo_loss, metrics

    def compute_log_probs(
        self,
        model: Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of completion-token log-probabilities. Shape ``[B]``.

        ``gather_logprobs`` returns ``[B, T - 1]``. Index ``t`` scores token
        ``t + 1``, so the mask that keeps those scores is
        ``completion_mask[:, 1:]``. Positions where that mask is 0, prompt
        tokens and padding, contribute nothing. The reduction is a sum.
        TRL 1.17.1 does the same: the fused head writes 0 wherever
        ``completion_mask`` is 0, and ``DPOTrainer._compute_loss`` sums the
        row when ``ld_alpha`` is ``None``. It does not average.

        ``attention_mask`` is forwarded to the model so padding is not
        attended to. It is not the reduction mask.

        The policy call must stay in the autograd graph. The reference call
        site wraps this method in ``torch.no_grad``.

        Args:
            model: Causal LM whose forward returns ``.logits`` ``[B, T, V]``.
            input_ids: Token ids, ``[B, T]``.
            attention_mask: ``[B, T]``. Passed through to ``model``.
            completion_mask: ``[B, T]``. 1 on completion tokens, 0 on prompt
                tokens and padding.

        Returns:
            Sequence log-probability of each completion, ``[B]``.
        """
        if completion_mask.shape != input_ids.shape:
            raise ValueError(
                f"completion_mask shape {tuple(completion_mask.shape)} must match "
                f"input_ids {tuple(input_ids.shape)}"
            )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_log_probs = gather_logprobs(outputs.logits, input_ids)
        shift_completion = completion_mask[:, 1:].to(dtype=token_log_probs.dtype)
        return (token_log_probs * shift_completion).sum(dim=1)

    def train_step(self, batch: dict) -> DPOMetrics:
        """One DPO update on one pair or a batch of pairs.

        ``prompt``, ``chosen``, and ``rejected`` are each a string or a
        sequence of strings. A string is a batch of one. The three sequences
        must have the same length. ``collate_preference_pairs`` right-pads
        each side, then ``train_token_step`` runs the update.

        Args:
            batch: Keys ``prompt``, ``chosen``, and ``rejected``.

        Returns:
            DPOMetrics for this step. ``accuracy`` is the mean over the batch,
            so a batch of more than one pair can land strictly between 0 and 1.
        """
        if self.tokenizer is None:
            raise RuntimeError("train_step requires a tokenizer.")
        prompts = _as_rows(batch["prompt"], "prompt")
        chosen = _as_rows(batch["chosen"], "chosen")
        rejected = _as_rows(batch["rejected"], "rejected")
        if not (len(prompts) == len(chosen) == len(rejected)):
            raise ValueError(
                "prompt, chosen, and rejected must have the same length, "
                f"got {len(prompts)}, {len(chosen)}, {len(rejected)}"
            )
        collated = collate_preference_pairs(
            self.tokenizer,
            prompts,
            chosen,
            rejected,
            max_length=self.config.dataset.max_seq_length,
            pad_token_id=_pad_token_id(self.tokenizer),
        )
        return self.train_token_step(
            collated["chosen_ids"],
            collated["chosen_attention_mask"],
            collated["chosen_completion_mask"],
            collated["rejected_ids"],
            collated["rejected_attention_mask"],
            collated["rejected_completion_mask"],
        )

    def train_token_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_completion_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_completion_mask: torch.Tensor,
    ) -> DPOMetrics:
        """One DPO update on token batches.

        Each tensor is ``[B, T]``. The completion masks are 1 on completion
        tokens. ``compute_log_probs`` sums ``completion_mask[:, 1:]``, because
        the score at ``t`` belongs to token ``t + 1``.

        Args:
            chosen_input_ids: Chosen token ids, ``[B, T]``.
            chosen_attention_mask: ``[B, T]``. Forwarded to the model.
            chosen_completion_mask: ``[B, T]``. 1 on chosen completion tokens.
            rejected_input_ids: Rejected token ids, ``[B, T]``.
            rejected_attention_mask: ``[B, T]``.
            rejected_completion_mask: ``[B, T]``. 1 on rejected completion tokens.

        Returns:
            DPOMetrics for this step.
        """
        self.policy_model.train()
        self.reference_model.eval()

        policy_chosen_log_probs = self.compute_log_probs(
            self.policy_model,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_completion_mask,
        )
        policy_rejected_log_probs = self.compute_log_probs(
            self.policy_model,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_completion_mask,
        )

        with torch.no_grad():
            ref_chosen_log_probs = self.compute_log_probs(
                self.reference_model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_completion_mask,
            )
            ref_rejected_log_probs = self.compute_log_probs(
                self.reference_model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_completion_mask,
            )

        loss, metrics = self.compute_dpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            ref_chosen_log_probs,
            ref_rejected_log_probs,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.optimization.max_grad_norm or 1.0,
        )
        self.optimizer.step()

        self.step_count += 1

        if getattr(self.config, "log_to_wandb", True):
            try:
                import wandb

                wandb.log(
                    {
                        "dpo/loss": metrics.dpo_loss,
                        "dpo/policy_diff_mean": metrics.policy_diff_mean,
                        "dpo/margin_mean": metrics.margin_mean,
                        "dpo/accuracy": metrics.accuracy,
                        "dpo/global_step": self.step_count,
                    }
                )
            except ImportError:
                pass

        return metrics

    def save_model(self, output_dir: str) -> None:
        """Save trained policy model.

        Args:
            output_dir: Directory to save model
        """
        self.policy_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")


# Short preference texts for ``run-dpo --tiny``. Lengths differ so the
# collator right-pads. This is not a DataLoader. S0.11d stores a DataLoader
# generator. This list is fixed, and each step picks rows with the RNGs
# the checkpoint already saves.
TINY_DPO_PAIRS: tuple[dict[str, str], ...] = (
    {"prompt": "Q: hi", "chosen": "yes", "rejected": "no"},
    {
        "prompt": "Q: longer prompt",
        "chosen": "short",
        "rejected": "a much longer rejected answer",
    },
    {"prompt": "P", "chosen": "one two three four", "rejected": "x"},
    {"prompt": "Ask", "chosen": "alpha beta gamma", "rejected": "nope"},
    {"prompt": "Go", "chosen": "stop now", "rejected": "keep going please"},
    {"prompt": "Item", "chosen": "the chosen completion", "rejected": "bad"},
)


@dataclass
class DPORunOutput:
    """What one call of ``run_dpo_training`` did.

    Attributes:
        seed: ``TrainingConfig.seed``. Also written to ``output_path``.
        step: Trainer step after this call. A resumed call continues, so
            this is the global step, not ``len(losses)``.
        losses: ``dpo_loss`` for each step taken in this call.
        metrics: ``DPOMetrics.to_dict()`` for each of those steps.
        output_path: ``{output_dir}/run.json``.
        checkpoint_path: File written at the end, or None when the config
            does not set one.
    """

    seed: int
    step: int
    losses: list[float]
    metrics: list[dict[str, float]]
    output_path: str
    checkpoint_path: str | None


def frozen_reference_copy(policy: Module) -> Module:
    """Frozen deep copy of the policy as it is right now.

    The training path uses this as π_ref. It is not a second random
    initialization. A resumed run does not call this: the checkpoint
    already holds the copy from the start of that run.
    """
    reference = copy.deepcopy(policy)
    reference.requires_grad_(False)
    reference.eval()
    return reference


def tiny_dpo_config(
    *,
    output_dir: str,
    seed: int,
    batch_size: int,
    checkpoint_path: str | None,
) -> TrainingConfig:
    """CPU config for ``run-dpo --tiny``.

    Starts from ``TrainingConfig.toy_mode`` and then sets the fields this
    command actually uses. The learning rate is ``1e-2`` so a few CPU
    steps move the tiny policy. That is not a tuned rate. ``device`` is
    ``cpu``. ``log_to_wandb`` is false. ``max_seq_length`` is 32, under
    the tiny Llama's position limit of 64. The policy and reference ids
    are ``tiny_lm``. Those ids are not loaded.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    config = TrainingConfig.toy_mode()
    config.seed = seed
    config.device = "cpu"
    config.log_to_wandb = False
    config.output_dir = output_dir
    config.checkpoint_path = checkpoint_path
    config.alignment.method = "dpo"
    config.model.policy_model_id = "tiny_lm"
    config.model.reference_model_id = "tiny_lm"
    config.optimization.learning_rate = 1e-2
    config.dataset.batch_size = batch_size
    config.dataset.max_seq_length = 32
    return config


def sample_preference_batch(
    pairs: Sequence[Mapping[str, str]],
    batch_size: int,
) -> dict[str, list[str]]:
    """Pick ``batch_size`` rows. Python, NumPy, and torch each draw once.

    The strings themselves are fixed. Which rows appear is not. A resumed
    run restores those three RNGs, so the next batch matches the
    uninterrupted run. This is not a ``DataLoader`` and does not keep a
    separate generator.
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


def run_dpo_training(
    trainer: DPOTrainer,
    n_steps: int,
    pairs: Sequence[Mapping[str, str]],
    *,
    resume_from: str | Path | None = None,
) -> DPORunOutput:
    """Run ``n_steps`` string DPO updates on ``tiny``-sized batches.

    Calls ``seed_everything`` once. On a fresh run the reference is then
    replaced with ``frozen_reference_copy`` of the policy, so π_ref is
    the initial policy and not whatever module the caller passed. A
    resumed call does not take that copy. ``load_checkpoint`` replaces
    the RNG, the policy, the reference, the optimizer, and
    ``step_count``. The saved reference is the frozen copy from the
    start of the run. Each step samples a batch and calls ``train_step``.
    When ``checkpoint_path`` is set, the checkpoint is written once, at
    the end of this call.

    Args:
        trainer: DPO trainer. The policy is trained. The reference stored
            in the checkpoint is a frozen copy of the initial policy.
        n_steps: Updates to take in this call. At least 1.
        pairs: Preference rows with ``prompt``, ``chosen``, and ``rejected``.
        resume_from: Checkpoint to load before those updates. None starts
            from the trainer as it was constructed.

    Returns:
        Losses and metrics for the steps taken in this call.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be an int >= 1, got {n_steps!r}")
    repro.seed_everything(trainer.config.seed)
    if resume_from is None:
        trainer.reference_model = frozen_reference_copy(trainer.policy_model)
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
    batch_size = trainer.config.dataset.batch_size
    for _ in range(n_steps):
        batch = sample_preference_batch(pairs, batch_size)
        step_metrics = trainer.train_step(batch)
        losses.append(step_metrics.dpo_loss)
        metrics.append(step_metrics.to_dict())
    return _finish_dpo_run(
        trainer.config,
        models=models,
        optimizer=trainer.optimizer,
        step=trainer.step_count,
        losses=losses,
        metrics=metrics,
    )


def _finish_dpo_run(
    config: TrainingConfig,
    *,
    models: Mapping[str, Module],
    optimizer: AdamW,
    step: int,
    losses: list[float],
    metrics: list[dict[str, float]],
) -> DPORunOutput:
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
    output = DPORunOutput(
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


if __name__ == "__main__":
    """Quick test of DPO trainer."""
    config = TrainingConfig.toy_mode()
    config.alignment.method = "dpo"

    trainer = DPOTrainer(config, use_toy=True)

    # Load toy dataset
    pairs = ToyDatasetLoader.load()

    # Train for a few steps
    for i, pair in enumerate(pairs[:5]):
        batch = {
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }
        metrics = trainer.train_step(batch)
        print(f"Step {i + 1}: DPO Loss={metrics.dpo_loss:.4f}, Accuracy={metrics.accuracy:.4f}")

    # Save model
    trainer.save_model("output/dpo_toy_checkpoint")
