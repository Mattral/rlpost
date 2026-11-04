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
``[B, T - 1]``. Rewards, values, dones, and advantages are ``[B, T]``.
Axis 1 is time, and ``rewards[:, t]`` is the reward for token ``t``.
The policy ratio uses ``advantages[:, 1:]``, so that reward weights the
log-prob of token ``t``. A padding mask of 0 removes that index from the
backup. ``train_step`` does not pass one.

The KL term estimates KL(policy || reference) on the sampled tokens.
``gather_logprobs`` supplies ``log p`` at those tokens, and
``lr = log p_ref - log p_policy``. ``kl_estimator`` selects ``k1 = -lr``,
``k2 = 0.5 * lr^2``, or ``k3 = exp(lr) - 1 - lr``. The default is ``k3``:
unbiased for that KL, and >= 0 at every token. ``train_step`` averages
every gathered token. It does not pass a mask. That mask stays in S0.7.

``V_clip`` is ``V_old + clip(V - V_old, -c, c)``. ``eps`` in the policy
clip is ``ppo_epsilon``. ``c`` is the ``value_clip`` config field, not
``ppo_epsilon``. Both value modes use the same one-half. When
``value_clip`` is set, ``L_v`` is the mean of
``0.5 * max((V - R)^2, (V_clip - R)^2)`` and ``train_step`` passes the
batch values as ``V_old``. When ``value_clip`` is unset, ``L_v`` is
``mean(0.5 * (V - R)^2)``. The loss functions do not take a padding
mask. ``train_step`` still does not pass one. That mask stays in S0.7.
"""

import logging
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import wandb
except ImportError:
    wandb = None

from rlpost.config import AlignmentConfig, OptimizationConfig, TrainingConfig
from rlpost.ops import gather_logprobs

logger = logging.getLogger(__name__)


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
        reward_mean: Mean reward in batch.
        advantage_mean: Mean advantage estimate.
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


def per_token_kl(
    logp_policy: Tensor,
    logp_ref: Tensor,
    estimator: Literal["k1", "k2", "k3"],
) -> Tensor:
    """Per-token estimate of KL(policy || reference) at sampled tokens.

    ``logp_policy`` and ``logp_ref`` are the log-probabilities of the same
    sampled tokens and must share a shape. ``logp_ref`` is detached, so the
    gradient reaches ``logp_policy`` only.

    Let ``lr = logp_ref - logp_policy``.

    - ``k1 = -lr``. Unbiased for KL(policy || reference). A token can be negative.
    - ``k2 = 0.5 * lr^2``. Always >= 0. This is the quadratic term, so it is biased.
    - ``k3 = exp(lr) - 1 - lr``. Unbiased, and >= 0 at every token: ``f(x) =
      exp(x) - 1 - x`` is convex and its minimum is 0 at ``x = 0``.

    ``k3`` is the ``AlignmentConfig`` default for that reason. The expression
    is the one TRL 1.17.1 ``GRPOTrainer._compute_loss`` assigns to
    ``per_token_kl`` before the optional importance-ratio multiply.

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

        Axis 1 of every ``[B, T]`` tensor is time. Index ``t`` is the token
        ``input_ids[:, t]``. ``rewards[:, t]`` is that token's reward.
        ``values[:, t]`` is ``V_t``. ``next_values[:, t]`` is ``V_{t+1}`` and
        is not read from ``values``. The advantage after the last index is 0.

        ``mask[:, t] == 0`` is padding, not a timestep. That index's advantage
        and return are 0. Its reward, value, and advantage do not enter any
        other index. ``dones[:, t] == 1`` stops the backup after a real token
        whose next index is also real. On real tokens, ``returns = A + V``.

        Args:
            rewards: Token rewards ``r_t``, shape ``[B, T]``.
            values: Critic estimates ``V_t``, shape ``[B, T]``.
            next_values: Critic estimates ``V_{t+1}``, shape ``[B, T]``.
            dones: 1 stops the backup after this token, shape ``[B, T]``.
            mask: 1 for a real token and 0 for padding, shape ``[B, T]``.
                ``None`` means every position is real.

        Returns:
            advantages: GAE advantages, shape ``[B, T]``.
            returns: ``A + V`` on real tokens and 0 on padding, shape ``[B, T]``.
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
            token_mask = torch.ones_like(values)
        else:
            if mask.shape != values.shape:
                raise ValueError(
                    f"mask and values shape mismatch: {tuple(mask.shape)} vs {tuple(values.shape)}"
                )
            token_mask = mask.to(dtype=values.dtype)

        batch_size, seq_len = values.shape
        # V_{t+1} is a real token only when the next index is real. The value
        # after the last index is the caller's bootstrap, not a pad token.
        next_real = torch.ones_like(token_mask)
        if seq_len > 1:
            next_real[:, :-1] = token_mask[:, 1:]
        continues = (1 - dones.to(dtype=values.dtype)) * next_real
        deltas = (rewards + self.gamma * next_values * continues - values) * token_mask

        advantages = torch.zeros_like(values)
        gae = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
        for t in reversed(range(seq_len)):
            gae = (deltas[:, t] + self.gamma * self.lam * continues[:, t] * gae) * token_mask[:, t]
            advantages[:, t] = gae

        returns = (advantages + values) * token_mask
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
    ) -> Tuple[Tensor, Tensor]:
        """Compute the clipped surrogate.

        ``r = exp(log_probs_new - log_probs_old)`` and

            L = -mean(min(r * A, clip(r, 1-eps, 1+eps) * A))

        ``eps`` is ``ppo_epsilon``. ``log_probs_old`` and ``advantages`` are
        detached, so the gradient reaches ``log_probs_new`` only. No mask is
        applied. Masking stays in S0.7.

        Args:
            log_probs_new: Current action log-probs. Any shape that matches the
                other two arguments.
            log_probs_old: Rollout action log-probs, same shape.
            advantages: Advantages for those actions, same shape.

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
        return -torch.min(unclipped, clipped).mean(), ratio

    def compute_value_loss(
        self: "PPOTrainer",
        value_pred: Tensor,
        returns: Tensor,
        old_values: Tensor | None = None,
    ) -> Tensor:
        """Compute the critic loss.

        ``returns`` and ``old_values`` are detached, so the gradient reaches
        ``value_pred`` only. No mask is applied.

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
        return squared_error.mean() * 0.5 * coefficient

    def compute_entropy_bonus(self: "PPOTrainer", logits: Tensor) -> Tensor:
        """Mean per-row Shannon entropy of the action distribution.

        For each row, ``H = -sum_a softmax(logits)_a * log_softmax(logits)_a``.
        The scalar is the mean of those rows. That per-row value is the same
        quantity as ``trl.trainer.utils.entropy_from_logits`` in TRL 1.17.1.
        This module does not import TRL. No mask is applied.

        Args:
            logits: Action logits, shape ``[..., num_actions]``.

        Returns:
            entropy: Mean entropy (scalar, nats).
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

    def compute_kl_divergence(
        self: "PPOTrainer",
        logp_policy: Tensor,
        logp_ref: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Mean per-token KL(policy || reference) on the sampled tokens.

        The estimator is ``alignment.kl_estimator`` (see ``per_token_kl``).
        ``logp_ref`` is detached inside that function.

        ``mask`` has the same shape as the log-probabilities. ``1`` keeps a
        token and ``0`` drops it. The mean divides by the sum of ``mask``.
        An all-zero mask returns 0. ``mask is None`` averages every token.
        ``train_step`` does not pass a mask, so prompt and padding tokens
        that were gathered still enter the mean. Which tokens those are
        stays in S0.7.

        Args:
            logp_policy: Log-probability of each sampled token under the policy.
            logp_ref: Log-probability of the same tokens under the reference.
            mask: Optional keep-mask, same shape. ``None`` keeps every token.

        Returns:
            kl_div: Scalar mean.
        """
        values = per_token_kl(logp_policy, logp_ref, self.alignment_config.kl_estimator)
        if mask is None:
            return values.mean()
        if mask.shape != values.shape:
            raise ValueError(
                f"shape mismatch: kl values {tuple(values.shape)}, mask {tuple(mask.shape)}"
            )
        weight = mask.to(dtype=values.dtype)
        # An all-zero mask has nothing to average. The clamp yields 0.
        return (values * weight).sum() / weight.sum().clamp(min=1.0)

    def compute_explained_variance(
        self: "PPOTrainer",
        value_pred: Tensor,
        returns: Tensor,
    ) -> float:
        """Compute fraction of variance in returns explained by value function.

        EV = 1 - Var(returns - values) / Var(returns)

        Range: (-∞, 1], where 1 = perfect fit, 0 = mean baseline, <0 = worse than mean

        Args:
            value_pred: Predicted values, shape (batch_size,)
            returns: Returns (targets), shape (batch_size,)

        Returns:
            explained_variance: Scalar in (-∞, 1]
        """
        variance_pred = torch.var(returns - value_pred)
        variance_returns = torch.var(returns)

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
        batch: dict[str, Tensor],
        old_log_probs: Tensor,
    ) -> PPOMetrics:
        """Perform single PPO training step.

        This function computes:
        1. Policy loss (clipped surrogate objective)
        2. Value loss (critic training)
        3. Entropy bonus (exploration)
        4. KL penalty (per-token ``kl_estimator`` on the gathered tokens, no mask)
        5. Combined loss = policy_loss + value_loss - entropy_coef * entropy + beta * KL

        Args:
            batch: Dict with keys, time on axis 1:
                - "input_ids": Tokens, shape ``[B, T]``.
                - "rewards": Token reward ``r_t`` for ``input_ids[:, t]``,
                  shape ``[B, T]``.
                - "values": Critic ``V_t``, shape ``[B, T]``.
                - "next_values": Critic ``V_{t+1}``, shape ``[B, T]``.
                - "dones": 1 stops the backup after that token, shape ``[B, T]``.
            old_log_probs: Log-probs of tokens ``1 .. T-1`` under the old policy,
                shape ``[B, T - 1]``, matching ``gather_logprobs``.

        Returns:
            metrics: PPOMetrics container with all training metrics
        """
        # Extract batch data
        input_ids = batch["input_ids"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        values = batch["values"].to(self.device)
        next_values = batch["next_values"].to(self.device)
        dones = batch["dones"].to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # Forward passes (should not compute gradients for reference model)
        with torch.no_grad():
            # Reference log-probs of the sampled tokens. The shift matches
            # the policy gather below. No mask is applied. S0.7 owns that.
            ref_outputs = self.reference_model(input_ids)
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, "logits") else ref_outputs
            ref_token_logprobs = gather_logprobs(ref_logits, input_ids)

        # Policy forward pass
        policy_outputs = self.policy_model(input_ids)
        policy_logits = (
            policy_outputs.logits if hasattr(policy_outputs, "logits") else policy_outputs
        )

        # Value forward pass
        value_outputs = self.value_model(input_ids)
        value_pred = (
            value_outputs.value if hasattr(value_outputs, "value") else value_outputs.squeeze()
        )

        # Compute advantages using GAE
        advantages, returns = self.gae.compute(rewards, values, next_values, dones)

        # Normalize advantages (for stability)
        if self.alignment_config.advantage_normalization:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # Token log-probs, shape [B, T-1]. Index t scores input_ids[:, t+1].
        # GAE stays on input positions [B, T]. rewards[:, t] is r_t for token t,
        # and the ratio weight for that token is advantages[:, t], which sits
        # in advantages[:, 1:] for t >= 1. No padding mask is passed here.
        token_logprobs = gather_logprobs(policy_logits, input_ids)
        if old_log_probs.shape != token_logprobs.shape:
            raise ValueError(
                f"old_log_probs shape {tuple(old_log_probs.shape)} must match "
                f"gathered logprobs {tuple(token_logprobs.shape)}"
            )
        action_advantages = advantages[:, 1:]

        # Compute all losses
        policy_loss, ratio = self.compute_policy_loss(
            token_logprobs, old_log_probs, action_advantages
        )
        # batch["values"] is V_old. compute_value_loss uses it only when
        # value_clip is set. No padding mask is passed. That stays in S0.7.
        value_loss = self.compute_value_loss(value_pred, returns, values)
        entropy = self.compute_entropy_bonus(policy_logits)
        # Same gathered tokens as the policy ratio. No mask. S0.7.
        kl_div = self.compute_kl_divergence(token_logprobs, ref_token_logprobs)

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

        # Compute metrics
        explained_var = self.compute_explained_variance(value_pred, returns)
        self.adaptive_kl_penalty(kl_div.item())

        metrics = PPOMetrics(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            kl_divergence=kl_div.item(),
            explained_variance=explained_var,
            adaptive_kl_coefficient=self.kl_coefficient,
            reward_mean=rewards.mean().item(),
            advantage_mean=advantages.mean().item(),
            gae_mean=advantages.mean().item(),
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
                    "ppo/global_step": self.global_step,
                },
                step=self.global_step,
            )

        self.global_step += 1

        return metrics
