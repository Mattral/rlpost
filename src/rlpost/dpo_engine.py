"""Direct Preference Optimization (DPO). Reward-model-free, with a frozen reference policy.

This module provides:
- DPO trainer for direct preference learning on pair data
- DPO loss computation without explicit reward model
- Batch processing and W&B logging
- Comparable training loop to PPO
"""

import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        the remaining tokens. Right padding is assumed: a left-padded prompt is
        not a prefix, and this mask would then be wrong. Batched left padding
        stays with the S0.11 collation note.

        Args:
            prompt: Prompt text.
            response: Response text.

        Returns:
            ``input_ids`` ``[1, T]``, ``attention_mask`` ``[1, T]``,
            ``completion_mask`` ``[1, T]``.
        """
        if self.tokenizer is None:
            raise RuntimeError("train_step requires a tokenizer.")
        max_length = self.config.dataset.max_seq_length
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        tokens = self.tokenizer(
            prompt + response,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        valid = int(attention_mask[0].sum().item())
        prompt_len = prompt_prefix_length(list(prompt_ids), input_ids[0, :valid].tolist())
        completion_mask = completion_token_mask(
            attention_mask,
            torch.tensor([prompt_len], dtype=torch.long),
        )
        return input_ids, attention_mask, completion_mask

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
        """Single DPO training step on one prompt and two response strings.

        Tokenizes ``prompt``, ``chosen``, and ``rejected``, then calls
        ``train_token_step``. Right padding is assumed.

        Args:
            batch: Dictionary with keys ``prompt``, ``chosen``, and ``rejected``.

        Returns:
            DPOMetrics for this step.
        """
        prompt = batch["prompt"]
        chosen_input_ids, chosen_attn, chosen_completion = self._tokenize_for_loss(
            prompt, batch["chosen"]
        )
        rejected_input_ids, rejected_attn, rejected_completion = self._tokenize_for_loss(
            prompt, batch["rejected"]
        )
        return self.train_token_step(
            chosen_input_ids,
            chosen_attn,
            chosen_completion,
            rejected_input_ids,
            rejected_attn,
            rejected_completion,
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
