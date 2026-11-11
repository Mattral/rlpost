"""PPO with GAE, a clipped surrogate, and an adaptive KL coefficient.

This is the only PPO implementation. The step is:

    delta_t = r_t + gamma * next_value_t * (1 - done_t) - value_t
    A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    ratio_t = exp(log_pi_new(a_t) - log_pi_old(a_t))
    L = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
        + c_v * L_v
        - c_ent * entropy
        + beta * KL

``train_step`` reads token log-probabilities from ``gather_logprobs``:
logits at position ``t`` score token ``t + 1``, and the result has shape
``[B, T - 1]``. The value head has the same shift. ``value_out[:, t]`` is
the critic after it has read token ``t`` and before it has read token
``t + 1``. Index ``t`` of the action layout is the action that emits
token ``t + 1``.

Advantages and returns stay fixed for every epoch on one rollout.
Standard PPO computes them once from the rollout values and holds them.
``train_step`` recomputes that GAE from ``batch["values"]`` on every
call, instead of storing advantages on the rollout. The tensor is
``V_old``. The critic's step does not change it, so two calls on the
same batch return the same advantages and the same returns. The
recursion stays in ``GeneralizedAdvantageEstimation.compute``. The
rollout does not emit advantages, and a second copy of this shift would
be a second place for the index to drift.

    values[:, t]       = V_old[:, t]          # V(s) before the action
    next_values[:, t]  = V_old[:, t + 1]      # V(s') after the action
    rewards[:, t]      = token_reward[:, t + 1]
    returns            = A + V_old

There is no ``next_values`` batch entry. The bootstrap is the next
column of the same rollout tensor, not the live head. The live head is
only the prediction the value loss trains:

    value_pred[:, t]   = value_out[:, t]

The reward written on a token moves back one column with the action.
TRL's PPO trainer (removed before 1.17.1; the 0.7 ``batched_forward_pass``
is the one that returned this) used the same shift: ``values[:, :-1]``
beside ``logprobs_from_logits(logits[:, :-1], input_ids[:, 1:])``. Those
stored values were the rollout forward. This is that alignment, not a
numeric match to a PPO loss TRL 1.17.1 does not ship.

Using the value at the sampled token as the baseline is wrong, whether
it is read from the live head or from ``V_old``. That position has
already read the action, so the baseline depends on the action. A
state-only baseline drops out of the policy gradient because
``sum_a ∇π(a|s) = 0``. An action-dependent baseline does not. A higher
value on the token that was just sampled makes ``A = r - V`` smaller,
so the surrogate reinforces that token less and punishes it more.
Tokens the critic scores low get the opposite push. With ``gamma = 0``
and zero rewards the advantage is ``-V(s)``, and editing the sampled
token must leave it unchanged. The state value in that test is ``-7``.
The value after the sampled token is ``-8``.

One response mask is built per batch from ``prompt_lengths`` and
``response_lengths``. ``response_token_mask`` documents the layout.
``"mixed"`` also takes ``prompt_width``, the padded query tensor width.
That width is not ``max(prompt_lengths)``.
``gathered_mask[:, t]`` is 1 when token ``t + 1`` is a generated response
token, and 0 for prompt and padding. Policy loss, entropy, KL, GAE, the
value loss, and advantage whitening all use that mask. The token-level
mask is only how the rollout finds the last response token. Every
reduction is ``sum(x * mask) / sum(mask)``. Advantage normalisation uses
that mean and the matching std on ``gathered_mask`` only.

The same lengths build ``padding_attention``. The policy, the reference,
and the value model each receive that attention mask and
``position_ids = clamp(cumsum(mask) - 1, min=0)``. A pad is not a token
they can read. A real token keeps the position it has with the pads removed.

``AlignmentConfig.temperature`` is the only temperature. Sampling reads
that field. ``generate_rollout`` records the number it used.
``train_step`` divides the policy and reference logits by the trainer's
field before the log-prob gather and before the entropy, and raises
when the recorded rollout temperature is a different number. Old
log-probs are already log-probabilities of that scaled distribution.
At the first step, before the weights move, the ratio is 1.

The KL term is added to the loss and differentiated with the sampled
token held fixed. ``gather_logprobs`` supplies ``log p`` at those tokens,
and ``lr = log p_ref - log p_policy``. ``kl_estimator`` selects

    k1 = -lr                 value unbiased, expected loss gradient 0
    k2 = 0.5 * lr^2          value biased, gradient of KL(policy || ref)
    k3 = exp(lr) - 1 - lr    value unbiased and >= 0, gradient of KL(ref || policy)

The default is ``k2``, because the loss is meant to follow
KL(policy || reference). ``k3`` stays available; its gradient is the
reverse KL, which is what TRL GRPO differentiates. ``k1`` is rejected
by config validation and again by the loss. Its value belongs in the
reward (reward minus beta * k1), and this trainer does not shape
rewards that way.

``V_clip`` is ``V_old + clip(V - V_old, -c, c)``. ``eps`` in the policy
clip is ``ppo_epsilon``. ``c`` is the ``value_clip`` config field, not
``ppo_epsilon``. Both value modes use the same one-half. When
``value_clip`` is set, ``L_v`` is the masked mean of
``0.5 * max((V - R)^2, (V_clip - R)^2)`` and ``train_step`` passes
``batch["values"][:, :-1]`` as ``V_old``. GAE reads that same rollout
tensor. When ``value_clip`` is unset, ``L_v`` is the masked mean of
``0.5 * (V - R)^2``. The target ``R`` is ``A + V_old`` in both modes.
"""

import json
import logging
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import wandb
except ImportError:
    wandb = None

import rlpost.repro as repro
from rlpost.alignment.rollout import (
    RolloutBatch,
    RolloutGenerator,
    response_logprobs_to_old,
)
from rlpost.config import (
    K1_LOSS_REJECTED,
    AlignmentConfig,
    OptimizationConfig,
    TrainingConfig,
)
from rlpost.dpo_engine import frozen_reference_copy
from rlpost.ops import gather_logprobs, padding_attention, response_token_mask

logger = logging.getLogger(__name__)


def _recorded_temperature(value: object) -> float:
    """Temperature stored on a rollout. A bool is not a temperature."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "batch['temperature'] must be the real number AlignmentConfig.temperature "
            f"the rollout sampled with, got {type(value).__name__}"
        )
    return float(value)


def masked_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """``sum(values * mask) / sum(mask)``.

    ``mask is None`` is the ordinary mean. ``mask`` must match ``values``.
    A 1 keeps an element and a 0 drops it. An all-zero mask returns 0.
    The denominator is clamped at 1 only so that empty mask stays 0
    instead of dividing by 0.
    """
    if mask is None:
        return values.mean()
    if mask.shape != values.shape:
        raise ValueError(f"shape mismatch: values {tuple(values.shape)}, mask {tuple(mask.shape)}")
    weight = mask.to(dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp(min=1)


def masked_whiten(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Whiten ``values`` on the masked positions. Elsewhere the result is 0.

    The mean is ``masked_mean``. The variance is the masked mean of squared
    deviations from that mean, so the divisor is ``sum(mask)`` and not
    ``sum(mask) - 1``. The returned std is the square root of that variance,
    before the ``1e-8`` stability term. Whitening divides by ``std + 1e-8``.
    """
    if mask.shape != values.shape:
        raise ValueError(f"shape mismatch: values {tuple(values.shape)}, mask {tuple(mask.shape)}")
    weight = mask.to(dtype=values.dtype)
    mean = masked_mean(values, weight)
    var = masked_mean((values - mean).square(), weight)
    std = var.sqrt()
    whitened = (values - mean) / (std + 1e-8) * weight
    return whitened, mean, std


@dataclass
class PPOMetrics:
    """Container for PPO training metrics (single step).

    Attributes:
        policy_loss: Clipped surrogate objective loss.
        value_loss: Critic network loss.
        entropy: Policy entropy (explorativity metric).
        kl_divergence: Mean per-token KL(policy || reference) under kl_estimator.
        explained_variance: Fraction of value function explained.
        adaptive_kl_coefficient: Current KL penalty coefficient (beta).
        reward_mean: Masked mean of action rewards. Index ``t`` is the
            reward for the action that emitted token ``t + 1``.
        advantage_mean: Masked mean of advantages before normalisation.
            The mask is the gathered response mask.
        gae_mean: Masked mean of advantages after normalisation.
        clip_fraction: Masked fraction of ratios outside
            ``(1 - ppo_epsilon, 1 + ppo_epsilon)``. A ratio of 1 counts
            as not clipped.
    """

    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    explained_variance: float
    adaptive_kl_coefficient: float
    reward_mean: float
    advantage_mean: float
    gae_mean: float
    clip_fraction: float


def per_token_kl(
    logp_policy: Tensor,
    logp_ref: Tensor,
    estimator: Literal["k1", "k2", "k3"],
) -> Tensor:
    """Per-token KL estimator. The token index is constant during backward.

    ``logp_policy`` and ``logp_ref`` are log-probabilities of the same
    sampled tokens and must share a shape. ``logp_ref`` is detached, so a
    backward reaches ``logp_policy`` only. Samples are drawn from the
    policy. The backward does not include the derivative through that
    sampling distribution, so the value and the gradient can name two
    different KLs.

    Let ``lr = logp_ref - logp_policy`` and write ``p`` for the policy and
    ``q`` for the reference.

    k1 = ``-lr``.
        Value: unbiased for KL(p || q). A token can be negative.
        Gradient: expected gradient w.r.t. the policy logits is 0.
        That makes k1 useless as a loss term. It is the InstructGPT
        penalty inside the reward, ``reward - beta * k1``, which needs
        the score-function gradient rather than this backward. This
        trainer does not implement that reward. Config validation and
        ``compute_kl_divergence`` both reject k1.

    k2 = ``0.5 * lr^2``.
        Value: always >= 0, and biased for KL(p || q).
        Gradient: expected gradient equals the gradient of KL(p || q).
        This is the loss-term default.

    k3 = ``exp(lr) - 1 - lr``.
        Value: unbiased for KL(p || q), and >= 0 at every token.
        ``f(x) = exp(x) - 1 - x`` is convex and its minimum is 0 at 0.
        Gradient: expected gradient equals the gradient of KL(q || p),
        the reverse direction. TRL 1.17.1 GRPO assigns this expression
        to its per-token KL, before the optional importance-ratio multiply.

    Args:
        logp_policy: Log-probability of the sampled token under the policy.
        logp_ref: Log-probability of that same token under the reference.
        estimator: ``k1``, ``k2``, or ``k3``.

    Returns:
        One estimate per token, same shape as the inputs.
    """
    if logp_policy.shape != logp_ref.shape:
        raise ValueError(
            "shape mismatch: "
            f"logp_policy {tuple(logp_policy.shape)}, logp_ref {tuple(logp_ref.shape)}"
        )
    if estimator not in ("k1", "k2", "k3"):
        raise ValueError(f"kl_estimator must be k1, k2, or k3, got {estimator!r}")

    log_ratio = logp_ref.detach() - logp_policy
    if estimator == "k1":
        return -log_ratio
    if estimator == "k2":
        return 0.5 * log_ratio.square()
    return torch.exp(log_ratio) - log_ratio - 1


class GeneralizedAdvantageEstimation:
    """Generalized Advantage Estimation (GAE) for trajectory advantage computation.

    GAE provides a trade-off between bias (low lambda) and variance (high lambda)
    in advantage estimation using the parameter lambda in [0, 1].

    Reference: Schulman et al., 2016 - High-Dimensional Continuous Control Using
    Generalized Advantage Estimation
    """

    def __init__(self: "GeneralizedAdvantageEstimation", gamma: float, lam: float) -> None:
        """Initialize GAE calculator.

        Args:
            gamma: Discount factor (typically 0.99).
            lam: GAE smoothing parameter lambda (typically 0.95).
                 0.0 = low variance (bootstrap), 1.0 = high variance (MC).
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam (lambda) must be in [0, 1], got {lam}")

        self.gamma = gamma
        self.lam = lam

    def compute(
        self: "GeneralizedAdvantageEstimation",
        rewards: Tensor,
        values: Tensor,
        next_values: Tensor,
        dones: Tensor,
        mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute advantages and returns using GAE.

        Axis 1 of every ``[B, T]`` tensor is time. The recursion does not
        know about tokens. PPO calls it on the action layout ``[B, T - 1]``:
        index ``t`` is the action that emits token ``t + 1``,
        ``rewards[:, t]`` is that action's reward, and ``values[:, t]`` is
        ``V(s)`` before the action. ``next_values[:, t]`` is ``V(s')`` and
        is not read from ``values``. The advantage after the last index is 0.

        ``mask[:, t] == 0`` is not a timestep. That index's advantage and
        return are 0. Its reward, value, and advantage do not enter any
        other index. ``dones[:, t] == 1`` stops the backup after a real
        index whose next index is also real. On real indices,
        ``returns = A + V``. ``train_step`` passes the rollout value
        head for both ``values`` and ``next_values``
        (``batch["values"][:, :-1]`` and ``batch["values"][:, 1:]``),
        so that sum is ``A + V_old``. The live critic is not an input.

        Args:
            rewards: Reward at each time index, shape ``[B, T]``.
            values: Critic estimates ``V`` at that index, shape ``[B, T]``.
            next_values: Critic estimates of the next state, shape ``[B, T]``.
            dones: 1 stops the backup after this index, shape ``[B, T]``.
            mask: 1 for a real index and 0 for one to drop, shape ``[B, T]``.
                ``None`` means every position is real. PPO passes the
                gathered response mask.

        Returns:
            advantages: GAE advantages, shape ``[B, T]``.
            returns: ``A + V`` on real indices and 0 where the mask is 0,
                shape ``[B, T]``.
        """
        if rewards.shape != values.shape:
            raise ValueError(
                f"rewards and values shape mismatch: {rewards.shape} vs {values.shape}"
            )
        if next_values.shape != values.shape or dones.shape != values.shape:
            raise ValueError(
                "shape mismatch: "
                f"values {tuple(values.shape)}, "
                f"next_values {tuple(next_values.shape)}, "
                f"dones {tuple(dones.shape)}"
            )
        if mask is None:
            active = torch.ones_like(values)
        else:
            if mask.shape != values.shape:
                raise ValueError(
                    f"mask and values shape mismatch: {tuple(mask.shape)} vs {tuple(values.shape)}"
                )
            active = mask.to(dtype=values.dtype)

        batch_size, seq_len = values.shape
        # The next state counts only when the next index is real. The value
        # after the last index is the caller's bootstrap, not a dropped index.
        next_real = torch.ones_like(active)
        if seq_len > 1:
            next_real[:, :-1] = active[:, 1:]
        continues = (1 - dones.to(dtype=values.dtype)) * next_real
        deltas = (rewards + self.gamma * next_values * continues - values) * active

        advantages = torch.zeros_like(values)
        gae = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
        for t in reversed(range(seq_len)):
            gae = (deltas[:, t] + self.gamma * self.lam * continues[:, t] * gae) * active[:, t]
            advantages[:, t] = gae

        returns = (advantages + values) * active
        return advantages, returns


class PPOTrainer:
    """PPO trainer with adaptive KL control and optional W&B logging.

    This trainer orchestrates policy gradient updates, value function training,
    and KL-based adaptive penalty coefficient adjustment. All hyperparameters
    are loaded from TrainingConfig for reproducibility.
    """

    def __init__(
        self: "PPOTrainer",
        config: TrainingConfig,
        policy_model: nn.Module,
        value_model: nn.Module,
        reference_model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Initialize PPO trainer.

        Args:
            config: Complete training configuration (TrainingConfig object).
            policy_model: Actor/policy network (differentiable).
            value_model: Critic network (differentiable).
            reference_model: Reference model (frozen) for KL divergence computation.
            optimizer: Optimizer for policy and value networks.
        """
        if not isinstance(config, TrainingConfig):
            raise TypeError(f"config must be TrainingConfig, got {type(config)}")

        self.config = config
        self.alignment_config: AlignmentConfig = config.alignment
        self.opt_config: OptimizationConfig = config.optimization

        # Models (assume on same device)
        self.policy_model = policy_model
        self.value_model = value_model
        self.reference_model = reference_model
        self.device = next(policy_model.parameters()).device

        # Optimizer
        self.optimizer = optimizer

        # GAE calculator
        self.gae = GeneralizedAdvantageEstimation(
            gamma=self.alignment_config.gamma,
            lam=self.alignment_config.gae_lambda,
        )

        # Adaptive KL control
        self.kl_coefficient = self.alignment_config.kl_coefficient
        self.target_kl = self.alignment_config.target_kl

        # Training state
        self.global_step = 0
        self.use_wandb = config.log_to_wandb and wandb is not None

        if self.use_wandb:
            wandb.define_metric("ppo/global_step")
            wandb.define_metric("ppo/*", step_metric="ppo/global_step")

        logger.info(
            f"PPO Trainer initialized: "
            f"eps={self.alignment_config.ppo_epsilon}, "
            f"beta={self.kl_coefficient}, "
            f"target_kl={self.target_kl}"
        )

    def compute_policy_loss(
        self: "PPOTrainer",
        log_probs_new: Tensor,
        log_probs_old: Tensor,
        advantages: Tensor,
        mask: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the clipped surrogate.

        ``r = exp(log_probs_new - log_probs_old)`` and

            L = -mean(min(r * A, clip(r, 1-eps, 1+eps) * A))

        The mean is ``masked_mean`` when ``mask`` is set. ``eps`` is
        ``ppo_epsilon``. ``log_probs_old`` and ``advantages`` are detached,
        so the gradient reaches ``log_probs_new`` only.

        Args:
            log_probs_new: Current action log-probs. Any shape that matches the
                other two arguments.
            log_probs_old: Rollout action log-probs, same shape.
            advantages: Advantages for those actions, same shape.
            mask: 1 keeps an action and 0 drops it. Same shape. ``None``
                averages every element. ``train_step`` passes the gathered
                response mask.

        Returns:
            loss: Scalar to minimize.
            ratio: ``r``, same shape as the inputs.
        """
        if log_probs_new.shape != log_probs_old.shape or advantages.shape != log_probs_new.shape:
            raise ValueError(
                "shape mismatch: "
                f"log_probs_new {tuple(log_probs_new.shape)}, "
                f"log_probs_old {tuple(log_probs_old.shape)}, "
                f"advantages {tuple(advantages.shape)}"
            )
        log_probs_old = log_probs_old.detach()
        advantages = advantages.detach()
        ratio = torch.exp(log_probs_new - log_probs_old)
        epsilon = self.alignment_config.ppo_epsilon
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages
        return masked_mean(-torch.min(unclipped, clipped), mask), ratio

    def compute_value_loss(
        self: "PPOTrainer",
        value_pred: Tensor,
        returns: Tensor,
        old_values: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the critic loss.

        ``returns`` and ``old_values`` are detached, so the gradient reaches
        ``value_pred`` only. The mean is ``masked_mean`` when ``mask`` is set.

        Both modes return ``value_loss_coefficient * L_v``. ``L_v`` always
        includes the one-half. When ``value_clip`` is unset, ``old_values``
        is not used and

            L_v = mean(0.5 * (V - R)^2)

        When ``value_clip`` is a radius ``c >= 0``, ``old_values`` is required
        and

            V_clip = V_old + clip(V - V_old, -c, c)
            L_v = mean(0.5 * max((V - R)^2, (V_clip - R)^2))

        A radius that does not bind matches the unset loss. ``c`` is not
        ``ppo_epsilon``.

        Args:
            value_pred: Current critic values ``V``.
            returns: Targets ``R``, same shape as ``value_pred``.
            old_values: Rollout values ``V_old``. Required when ``value_clip``
                is set. Same shape as ``value_pred``.
            mask: 1 keeps an action and 0 drops it. Same shape as ``value_pred``.
                ``None`` averages every element. ``train_step`` passes the
                gathered response mask, shape ``[B, T - 1]``.

        Returns:
            loss: Scalar to minimize.
        """
        if value_pred.shape != returns.shape:
            raise ValueError(
                "shape mismatch: "
                f"value_pred {tuple(value_pred.shape)}, returns {tuple(returns.shape)}"
            )
        returns = returns.detach()
        coefficient = self.alignment_config.value_loss_coefficient
        squared_error = (value_pred - returns).square()
        clip = self.alignment_config.value_clip
        if clip is not None:
            if old_values is None:
                raise ValueError("value_clip is set; old_values is required")
            if old_values.shape != value_pred.shape:
                raise ValueError(
                    "shape mismatch: "
                    f"value_pred {tuple(value_pred.shape)}, "
                    f"old_values {tuple(old_values.shape)}"
                )
            old_values = old_values.detach()
            value_clipped = old_values + (value_pred - old_values).clamp(-clip, clip)
            clipped = (value_clipped - returns).square()
            squared_error = torch.maximum(squared_error, clipped)
        return masked_mean(squared_error, mask) * 0.5 * coefficient

    def compute_entropy_bonus(
        self: "PPOTrainer",
        logits: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Mean per-row Shannon entropy of the action distribution.

        For each row, ``H = -sum_a softmax(logits)_a * log_softmax(logits)_a``.
        The scalar is the mean of those rows. That per-row value is the same
        quantity as ``trl.trainer.utils.entropy_from_logits`` in TRL 1.17.1.
        This module does not import TRL. ``mask`` selects rows. It must have
        the same shape as ``logits`` without the last axis. ``train_step``
        passes logits ``[:, :-1]`` and the gathered response mask, so only
        distributions that score a response token are averaged.

        Args:
            logits: Action logits, shape ``[..., num_actions]``.
            mask: 1 keeps a row and 0 drops it. ``None`` averages every row.

        Returns:
            entropy: Mean entropy (scalar, nats).
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        per_row = -(probs * log_probs).sum(dim=-1)
        return masked_mean(per_row, mask)

    def compute_kl_divergence(
        self: "PPOTrainer",
        logp_policy: Tensor,
        logp_ref: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Mean of the configured estimator. This scalar is the loss term.

        ``k2`` and ``k3`` only. ``k1`` raises. Under samples from the
        policy, the expected gradient of mean(k1) w.r.t. the policy logits
        is 0, so k1 does not constrain the policy when it is added to the
        loss. The value of k1 is still available from ``per_token_kl`` for
        a reward penalty. See that function for the value and the gradient
        of each estimator.

        The default estimator is ``k2``. Its gradient estimates
        KL(policy || reference). ``k3`` is legal here, and its gradient
        estimates KL(reference || policy).

        ``logp_ref`` is detached inside ``per_token_kl``. ``mask`` has the
        same shape as the log-probabilities. ``1`` keeps a token and ``0``
        drops it. The mean divides by the sum of ``mask``. An all-zero
        mask returns 0. ``mask is None`` averages every token.
        ``train_step`` passes the gathered response mask, so prompt and
        padding tokens stay out of the mean.

        Args:
            logp_policy: Log-probability of each sampled token under the policy.
            logp_ref: Log-probability of the same tokens under the reference.
            mask: Optional keep-mask, same shape. ``None`` keeps every token.

        Returns:
            kl_div: Scalar mean.
        """
        estimator = self.alignment_config.kl_estimator
        if estimator == "k1":
            raise ValueError(K1_LOSS_REJECTED)
        values = per_token_kl(logp_policy, logp_ref, estimator)
        return masked_mean(values, mask)

    def compute_explained_variance(
        self: "PPOTrainer",
        value_pred: Tensor,
        returns: Tensor,
        mask: Tensor | None = None,
    ) -> float:
        """Compute fraction of variance in returns explained by value function.

        EV = 1 - Var(returns - values) / Var(returns)

        Range: (-∞, 1], where 1 = perfect fit, 0 = mean baseline, <0 = worse than mean.
        Without ``mask``, variance is ``torch.var`` (divide by N - 1). With
        ``mask``, variance is the masked mean of squared deviations (divide
        by ``sum(mask)``). ``train_step`` passes the gathered response mask.

        Args:
            value_pred: Predicted values.
            returns: Returns (targets), same shape as ``value_pred``.
            mask: Optional response mask, same shape. ``None`` uses every element.

        Returns:
            explained_variance: Scalar in (-∞, 1]
        """
        if mask is None:
            variance_pred = torch.var(returns - value_pred)
            variance_returns = torch.var(returns)
        else:
            residual = returns - value_pred
            variance_pred = masked_mean((residual - masked_mean(residual, mask)).square(), mask)
            variance_returns = masked_mean((returns - masked_mean(returns, mask)).square(), mask)

        if variance_returns.item() == 0.0:
            return 0.0

        explained_var = 1.0 - (variance_pred / variance_returns)
        return float(explained_var.item())

    def adaptive_kl_penalty(self: "PPOTrainer", kl_div: float) -> None:
        """Adaptively adjust KL penalty coefficient based on KL divergence.

        If KL > target, increase beta (more penalty).
        If KL < target / 1.5, decrease beta (less penalty).
        Otherwise, keep beta constant.

        This ensures KL divergence stays close to target without manual tuning.

        Args:
            kl_div: Mean KL divergence in current batch.
        """
        if kl_div > self.target_kl * 1.5:
            # KL too high, increase penalty
            self.kl_coefficient *= 2.0
        elif kl_div < self.target_kl / 1.5:
            # KL too low, decrease penalty
            self.kl_coefficient *= 0.5

        # Clip to reasonable range [0.001, 10.0]
        self.kl_coefficient = max(0.001, min(10.0, self.kl_coefficient))

    def train_step(
        self: "PPOTrainer",
        batch: dict[str, Tensor | str | int],
        old_log_probs: Tensor,
    ) -> PPOMetrics:
        """Perform single PPO training step.

        This function computes:
        1. Policy loss (clipped surrogate on response tokens)
        2. Value loss (critic, response tokens)
        3. Entropy bonus (distributions that score a response token)
        4. KL penalty (``k2`` or ``k3`` on those tokens; ``k1`` raises)
        5. Combined loss = policy_loss + value_loss - entropy_coef * entropy + beta * KL

        The response mask is built once, from ``prompt_lengths`` and
        ``response_lengths``. See ``response_token_mask``. Losses, GAE, and
        whitening use ``gathered_mask``. Reductions use
        ``sum(x * mask) / sum(mask)``.

        Those same lengths build ``padding_attention``. The policy, the
        reference, and the value model are called with that
        ``attention_mask`` and with ``position_ids``. Log-probs and the
        entropy use ``logits / temperature``. ``temperature`` is
        ``AlignmentConfig.temperature``. Sampling reads that same field.
        ``batch["temperature"]`` is the number the rollout sampled with.
        A different number raises. ``old_log_probs`` are already
        log-probabilities of ``logits / temperature``. They are not
        divided again.

        Args:
            batch: Dict with keys:
                - "input_ids": Tokens, shape ``[B, T]``.
                - "rewards": Reward for the token at that index, shape
                  ``[B, T]``. The sequence score sits on the last real
                  response token. The action layout reads ``rewards[:, 1:]``,
                  so index ``t`` there is the reward for emitting token
                  ``t + 1``.
                - "values": Rollout value head at each token, shape ``[B, T]``.
                  ``values[:, t]`` was computed after reading token ``t``.
                  This is ``V_old``. GAE reads ``values[:, :-1]`` as the
                  baseline and ``values[:, 1:]`` as the bootstrap, and
                  holds the resulting advantages and returns fixed across
                  epochs. The clip reads ``values[:, :-1]`` as ``V_old``
                  when ``value_clip`` is set. The live value head is not
                  this tensor.
                - "dones": 1 stops the backup after that token, shape
                  ``[B, T]``. The action layout reads ``dones[:, 1:]``.
                  ``generate_rollout`` sets 1 only on an EOS token. A row
                  that hit ``max_response_length`` without EOS stays 0, so
                  the last action bootstraps from ``V``.
                - "prompt_lengths": Real prompt tokens per row, int64 ``[B]``.
                - "response_lengths": Real response tokens per row, int64 ``[B]``.
                - "padding_side": ``"right"`` (default) or ``"mixed"``.
                - "prompt_width": Padded query width. Required when
                  ``padding_side`` is ``"mixed"``. Not
                  ``max(prompt_lengths)``. Ignored for ``"right"``.
                - "temperature": The ``AlignmentConfig.temperature`` the
                  rollout sampled with. Must equal this trainer's field.
            old_log_probs: Log-probs of tokens ``1 .. T-1`` under the old policy,
                shape ``[B, T - 1]``, matching ``gather_logprobs``.

        Returns:
            metrics: PPOMetrics container with all training metrics
        """
        if "temperature" not in batch:
            raise KeyError(
                "batch needs temperature, the AlignmentConfig.temperature the rollout sampled with"
            )
        recorded = _recorded_temperature(batch["temperature"])
        temperature = self.alignment_config.temperature
        if recorded != temperature:
            raise ValueError(
                f"rollout sampled at temperature {recorded}, and "
                f"AlignmentConfig.temperature is {temperature}. "
                "Sampling and the train-step log-prob gather use that one field."
            )

        # The policy and the value head train. The reference stays in eval.
        self.policy_model.train()
        self.value_model.train()
        self.reference_model.eval()

        # Extract batch data
        input_ids = batch["input_ids"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        old_values_full = batch["values"].to(self.device)
        dones = batch["dones"].to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        if "prompt_lengths" not in batch or "response_lengths" not in batch:
            raise KeyError(
                "batch needs prompt_lengths and response_lengths to build the response mask"
            )
        prompt_lengths = batch["prompt_lengths"].to(self.device)
        response_lengths = batch["response_lengths"].to(self.device)
        padding_side = batch.get("padding_side", "right")
        if not isinstance(padding_side, str):
            raise TypeError(f"padding_side must be a str, got {type(padding_side)}")
        # The token half places the rollout reward. PPO keeps the gathered half.
        _, gathered_mask = response_token_mask(
            prompt_lengths,
            response_lengths,
            int(input_ids.shape[1]),
            padding_side,
            prompt_width=batch.get("prompt_width"),
        )
        # Same lengths as the loss mask. Prompt tokens stay visible here.
        attention_mask, position_ids = padding_attention(
            prompt_lengths,
            response_lengths,
            int(input_ids.shape[1]),
            padding_side,
            prompt_width=batch.get("prompt_width"),
        )
        # Forward passes (should not compute gradients for reference model)
        with torch.no_grad():
            # Reference log-probs of the sampled tokens. The shift matches
            # the policy gather below. The KL mean uses gathered_mask.
            ref_outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, "logits") else ref_outputs
            ref_logits = ref_logits / temperature
            ref_token_logprobs = gather_logprobs(ref_logits, input_ids)

        # Policy forward pass
        policy_outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        policy_logits = (
            policy_outputs.logits if hasattr(policy_outputs, "logits") else policy_outputs
        )
        policy_logits = policy_logits / temperature

        # Live value head. Position t has read tokens 0..t and has not read
        # token t+1. The value loss trains this prediction. GAE does not
        # read it: the baseline is V_old, frozen on the rollout.
        value_outputs = self.value_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        value_out = value_outputs.value if hasattr(value_outputs, "value") else value_outputs
        if value_out.ndim == input_ids.ndim + 1 and value_out.shape[-1] == 1:
            value_out = value_out.squeeze(-1)
        if value_out.shape != input_ids.shape:
            raise ValueError(
                f"value head shape {tuple(value_out.shape)} must match input_ids "
                f"{tuple(input_ids.shape)}"
            )
        if rewards.shape != input_ids.shape or dones.shape != input_ids.shape:
            raise ValueError(
                "rewards and dones must match input_ids, got "
                f"rewards {tuple(rewards.shape)}, dones {tuple(dones.shape)}, "
                f"input_ids {tuple(input_ids.shape)}"
            )
        if old_values_full.shape != input_ids.shape:
            raise ValueError(
                "batch values must be the value head at each token, shape "
                f"{tuple(input_ids.shape)}, got {tuple(old_values_full.shape)}"
            )

        # Action layout [B, T-1]. Index t emits token t+1.
        # V_old is frozen. Recomputing GAE from it is the same advantage a
        # rollout-time store would have kept, and the recursion stays here.
        # The live head is value_pred. Returns are A + V_old.
        action_values = value_out[:, :-1]
        action_rewards = rewards[:, 1:]
        action_dones = dones[:, 1:]
        old_action_values = old_values_full[:, :-1]
        old_next_values = old_values_full[:, 1:]

        advantages, returns = self.gae.compute(
            action_rewards,
            old_action_values.detach(),
            old_next_values.detach(),
            action_dones,
            gathered_mask,
        )
        advantage_mean = masked_mean(advantages, gathered_mask)

        # Normalise on response actions only. Prompt and padding stay 0.
        if self.alignment_config.advantage_normalization:
            advantages, _, _ = masked_whiten(advantages, gathered_mask)
        gae_mean = masked_mean(advantages, gathered_mask)

        # Token log-probs, shape [B, T-1]. Index t scores input_ids[:, t+1]
        # and is weighted by advantages[:, t], the return of that same action.
        token_logprobs = gather_logprobs(policy_logits, input_ids)
        if old_log_probs.shape != token_logprobs.shape:
            raise ValueError(
                f"old_log_probs shape {tuple(old_log_probs.shape)} must match "
                f"gathered logprobs {tuple(token_logprobs.shape)}"
            )
        if gathered_mask.shape != token_logprobs.shape:
            raise ValueError(
                f"gathered response mask {tuple(gathered_mask.shape)} must match "
                f"logprobs {tuple(token_logprobs.shape)}"
            )

        # One mask for every reduction. value_pred is the live head.
        # old_action_values is V_old for the clip. The returns are already
        # A + V_old.
        policy_loss, ratio = self.compute_policy_loss(
            token_logprobs, old_log_probs, advantages, gathered_mask
        )
        epsilon = self.alignment_config.ppo_epsilon
        outside = (ratio < (1.0 - epsilon)) | (ratio > (1.0 + epsilon))
        clip_fraction = masked_mean(outside.to(dtype=ratio.dtype), gathered_mask)
        value_loss = self.compute_value_loss(
            action_values, returns, old_action_values, gathered_mask
        )
        entropy = self.compute_entropy_bonus(policy_logits[:, :-1, :], gathered_mask)
        kl_div = self.compute_kl_divergence(token_logprobs, ref_token_logprobs, gathered_mask)

        # KL penalty term
        kl_penalty = self.kl_coefficient * kl_div

        # Combined loss
        total_loss = (
            policy_loss
            + value_loss
            - self.alignment_config.entropy_coefficient * entropy
            + kl_penalty
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.opt_config.max_grad_norm
        )
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.opt_config.max_grad_norm)
        self.optimizer.step()

        # Means are over response actions, not the padded token tensor.
        explained_var = self.compute_explained_variance(action_values, returns, gathered_mask)
        self.adaptive_kl_penalty(kl_div.item())

        metrics = PPOMetrics(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            kl_divergence=kl_div.item(),
            explained_variance=explained_var,
            adaptive_kl_coefficient=self.kl_coefficient,
            reward_mean=masked_mean(action_rewards, gathered_mask).item(),
            advantage_mean=advantage_mean.item(),
            gae_mean=gae_mean.item(),
            clip_fraction=clip_fraction.item(),
        )

        # Log to W&B
        if self.use_wandb:
            wandb.log(
                {
                    "ppo/policy_loss": metrics.policy_loss,
                    "ppo/value_loss": metrics.value_loss,
                    "ppo/entropy": metrics.entropy,
                    "ppo/kl_divergence": metrics.kl_divergence,
                    "ppo/explained_variance": metrics.explained_variance,
                    "ppo/adaptive_kl_coefficient": metrics.adaptive_kl_coefficient,
                    "ppo/reward_mean": metrics.reward_mean,
                    "ppo/advantage_mean": metrics.advantage_mean,
                    "ppo/clip_fraction": metrics.clip_fraction,
                    "ppo/global_step": self.global_step,
                },
                step=self.global_step,
            )

        self.global_step += 1

        return metrics


# Fixed prompt list for ``run-ppo --tiny``. Lengths differ so a batch
# left-pads. The strings are not a dataset split.
TINY_PPO_PROMPTS: tuple[str, ...] = (
    "a",
    "go",
    "cat",
    "stay",
    "left",
    "respond",
)
TINY_PPO_MAX_RESPONSE = 4
TINY_PPO_EOS_ID = 2
TINY_PPO_PAD_ID = 0


class ResponseTokenSumReward(nn.Module):
    """Sequence score is the sum of token ids in the response window.

    ``prompt_width`` is the left-padded query width. Columns at and after
    that index are the response window. Pad id 0 adds nothing, so right
    padding does not change the sum. EOS is included. The return is
    ``logits`` of shape ``[B, 1]``. ``generate_rollout`` treats that as
    the sequence score and writes it on the last real response token.

    This is not a learned reward model.
    """

    def __init__(self, prompt_width: int) -> None:
        super().__init__()
        if isinstance(prompt_width, bool) or not isinstance(prompt_width, int) or prompt_width < 1:
            raise ValueError(f"prompt_width must be an int >= 1, got {prompt_width!r}")
        self.prompt_width = prompt_width

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        if input_ids.ndim != 2 or input_ids.shape[1] < self.prompt_width:
            raise ValueError(
                "input_ids must be [B, T] with T >= "
                f"prompt_width {self.prompt_width}, got {tuple(input_ids.shape)}"
            )
        window = input_ids[:, self.prompt_width :].to(dtype=torch.float32)
        return SimpleNamespace(logits=window.sum(dim=-1, keepdim=True))


@dataclass
class PPORunOutput:
    """What one call of ``run_ppo_training`` did.

    Attributes:
        seed: ``TrainingConfig.seed``. Also written to ``output_path``.
        step: Trainer step after this call. A resumed call continues, so
            this is the global step, not ``len(losses)``.
        losses: ``policy_loss`` for each step taken in this call.
        metrics: ``asdict(PPOMetrics)`` for each of those steps.
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


def tiny_ppo_config(
    *,
    output_dir: str,
    seed: int,
    batch_size: int,
    checkpoint_path: str | None,
) -> TrainingConfig:
    """CPU config for ``run-ppo --tiny``.

    Starts from ``TrainingConfig.toy_mode`` and then sets the fields this
    command actually uses. The learning rate is ``1e-2`` so a few CPU
    steps move the tiny policy. That is not a tuned rate. ``device`` is
    ``cpu``. ``log_to_wandb`` is false. ``max_seq_length`` is 32, under
    the tiny Llama's position limit of 64. The policy and reference ids
    are ``tiny_lm``. Those ids are not loaded. ``method`` is ``ppo``.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")
    config = TrainingConfig.toy_mode()
    config.seed = seed
    config.device = "cpu"
    config.log_to_wandb = False
    config.output_dir = output_dir
    config.checkpoint_path = checkpoint_path
    config.alignment.method = "ppo"
    config.model.policy_model_id = "tiny_lm"
    config.model.reference_model_id = "tiny_lm"
    config.optimization.learning_rate = 1e-2
    config.dataset.batch_size = batch_size
    config.dataset.max_seq_length = 32
    return config


def sample_prompt_batch(prompts: Sequence[str], batch_size: int) -> list[str]:
    """Pick ``batch_size`` prompts. Python, NumPy, and torch each draw once.

    The strings themselves are fixed. Which rows appear is not. A resumed
    run restores those three RNGs, so the next batch matches the
    uninterrupted run. This is not a ``DataLoader`` and does not keep a
    separate generator.
    """
    count = len(prompts)
    if count < 1:
        raise ValueError("prompts must be non-empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")
    start = random.randrange(count)
    stride = int(np.random.randint(1, count)) if count > 1 else 1
    shift = int(torch.randint(0, count, ()).item())
    indices = [(start + stride * i + shift) % count for i in range(batch_size)]
    return [str(prompts[index]) for index in indices]


def collate_prompts(
    tokenizer: object,
    prompts: Sequence[str],
    *,
    max_length: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Left-pad prompts so the last column is the last real token.

    Returns ``query_tokens`` ``[B, prompt_width]`` and ``prompt_lengths``
    ``[B]``. ``prompt_width`` is the longest row after truncation, not
    ``max`` of a separate length list. An empty tokenization raises.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError(f"max_length must be an int >= 1, got {max_length!r}")
    rows: list[list[int]] = []
    for text in prompts:
        encoded = tokenizer(str(text), max_length=max_length)
        ids = [int(token) for token in encoded["input_ids"]]
        if not ids:
            raise ValueError("a prompt tokenized to no tokens")
        rows.append(ids[:max_length])
    width = max(len(row) for row in rows)
    query = torch.full((len(rows), width), int(pad_token_id), dtype=torch.long)
    lengths = torch.zeros(len(rows), dtype=torch.long)
    for index, row in enumerate(rows):
        query[index, width - len(row) :] = torch.tensor(row, dtype=torch.long)
        lengths[index] = len(row)
    return query, lengths


def _rollout_values(value_model: nn.Module, rollout: RolloutBatch) -> Tensor:
    """Detached value head on the finished rollout. Shape ``[B, T]``.

    The mask and the position ids are the same pair ``train_step`` builds.
    This tensor is ``V_old``. The live head is a later forward.
    """
    device = next(value_model.parameters()).device
    input_ids = rollout.input_ids.to(device)
    prompt_lengths = rollout.prompt_lengths.to(device)
    response_lengths = rollout.response_lengths.to(device)
    attention, position_ids = padding_attention(
        prompt_lengths,
        response_lengths,
        int(input_ids.shape[1]),
        rollout.padding_side,
        prompt_width=rollout.prompt_width,
    )
    value_model.eval()
    with torch.no_grad():
        value_out = value_model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
        )
    values = value_out.value if hasattr(value_out, "value") else value_out
    if values.ndim == input_ids.ndim + 1 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.shape != input_ids.shape:
        raise ValueError(
            f"value head shape {tuple(values.shape)} must match input_ids {tuple(input_ids.shape)}"
        )
    return values.detach().cpu()


def run_ppo_training(
    trainer: PPOTrainer,
    n_steps: int,
    prompts: Sequence[str],
    tokenizer: object,
    *,
    resume_from: str | Path | None = None,
    max_response_length: int = TINY_PPO_MAX_RESPONSE,
) -> PPORunOutput:
    """Run ``n_steps`` PPO updates: rollout, then ``train_step``.

    Calls ``seed_everything`` once. On a fresh run the reference is then
    replaced with ``frozen_reference_copy`` of the policy, so the
    reference is the initial policy and not whatever module the caller
    passed. A resumed call does not take that copy. ``load_checkpoint``
    replaces the RNG, the policy, the value head, the reference, the
    optimizer, ``global_step``, and ``kl_coefficient``. The saved
    reference is the frozen copy from the start of the run.

    Each step samples a prompt batch, left-pads it, and calls
    ``generate_rollout``. The toy reward is the sum of token ids in the
    response window (pad id 0 adds nothing, EOS is included), written on
    the last real response token. ``old_log_probs`` are the stored
    ``[B, R]`` policy log-probs placed into ``[B, T - 1]``. Sampling uses
    ``top_p`` 1, so the draw is the full softmax. EOS id is 2 and pad id
    is 0. When ``checkpoint_path`` is set, the checkpoint is written
    once, at the end of this call. ``trainer_state`` is
    ``kl_coefficient``.

    Args:
        trainer: PPO trainer. Policy and value are trained. The reference
            stored in the checkpoint is a frozen copy of the initial policy.
        n_steps: Updates to take in this call. At least 1.
        prompts: Fixed prompt strings.
        tokenizer: Maps one string to ``{"input_ids": [...]}``.
        resume_from: Checkpoint to load before those updates. None starts
            from the trainer as it was constructed.
        max_response_length: Response cap passed to ``generate_rollout``.

    Returns:
        Losses and metrics for the steps taken in this call.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"n_steps must be an int >= 1, got {n_steps!r}")
    if (
        isinstance(max_response_length, bool)
        or not isinstance(max_response_length, int)
        or max_response_length < 1
    ):
        raise ValueError(f"max_response_length must be an int >= 1, got {max_response_length!r}")
    repro.seed_everything(trainer.config.seed)
    if resume_from is None:
        trainer.reference_model = frozen_reference_copy(trainer.policy_model)
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
        trainer.reference_model.requires_grad_(False)
        trainer.reference_model.eval()
    losses: list[float] = []
    metrics: list[dict[str, float]] = []
    batch_size = trainer.config.dataset.batch_size
    pad_id = getattr(tokenizer, "pad_token_id", TINY_PPO_PAD_ID)
    if pad_id is None:
        pad_id = TINY_PPO_PAD_ID
    pad_id = int(pad_id)
    if pad_id == TINY_PPO_EOS_ID:
        raise ValueError("pad_token_id and eos_token_id must differ")
    for _ in range(n_steps):
        texts = sample_prompt_batch(prompts, batch_size)
        query, prompt_lengths = collate_prompts(
            tokenizer,
            texts,
            max_length=trainer.config.dataset.max_seq_length,
            pad_token_id=pad_id,
        )
        trainer.policy_model.eval()
        trainer.reference_model.eval()
        trainer.value_model.eval()
        reward = ResponseTokenSumReward(int(query.shape[1]))
        generator = RolloutGenerator(
            policy_model=trainer.policy_model,
            reference_model=trainer.reference_model,
            reward_model=reward,
            tokenizer=tokenizer,
            alignment_config=trainer.alignment_config,
            max_response_length=max_response_length,
            top_p=1.0,
            eos_token_id=TINY_PPO_EOS_ID,
            pad_token_id=pad_id,
        )
        rollout = generator.generate_rollout(query, prompt_lengths)
        old_log_probs = response_logprobs_to_old(
            rollout.logprobs_policy,
            prompt_width=rollout.prompt_width,
            seq_len=int(rollout.input_ids.shape[1]),
        )
        batch = {
            "input_ids": rollout.input_ids,
            "rewards": rollout.rewards,
            "values": _rollout_values(trainer.value_model, rollout),
            "dones": rollout.dones,
            "prompt_lengths": rollout.prompt_lengths,
            "response_lengths": rollout.response_lengths,
            "padding_side": rollout.padding_side,
            "prompt_width": rollout.prompt_width,
            "temperature": rollout.temperature,
        }
        step_metrics = trainer.train_step(batch, old_log_probs)
        losses.append(step_metrics.policy_loss)
        metrics.append(asdict(step_metrics))
    return _finish_ppo_run(
        trainer.config,
        models=models,
        optimizer=trainer.optimizer,
        step=trainer.global_step,
        trainer_state={"kl_coefficient": float(trainer.kl_coefficient)},
        losses=losses,
        metrics=metrics,
    )


def _finish_ppo_run(
    config: TrainingConfig,
    *,
    models: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    step: int,
    trainer_state: dict[str, float],
    losses: list[float],
    metrics: list[dict[str, float]],
) -> PPORunOutput:
    """Write the checkpoint, if any, and ``run.json``."""
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
    output = PPORunOutput(
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
