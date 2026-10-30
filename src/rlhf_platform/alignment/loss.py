"""
loss.py: Numerically stable loss functions for distributed PPO and reward modeling.

Implements KL penalties, clipped advantage functions, and value function regularization
with overflow/underflow safeguards suitable for cluster-scale training.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean of values where mask is True, handling empty masks."""
    if mask.sum() == 0:
        logger.warning("Empty mask in masked_mean; returning zero.")
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum()


def compute_kl_divergence(
    logits_policy: torch.Tensor,
    logits_reference: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference distributions.

    KL(policy || reference) = sum_i policy_i * (log(policy_i) - log(reference_i))

    Args:
        logits_policy: Policy model logits, shape (batch_size, vocab_size)
        logits_reference: Reference model logits, shape (batch_size, vocab_size)
        reduction: "mean", "sum", or "none"

    Returns:
        KL divergence (scalar or per-token if reduction="none")
    """
    # Convert logits to probabilities with numerical stability
    log_policy = F.log_softmax(logits_policy, dim=-1)
    log_reference = F.log_softmax(logits_reference, dim=-1)

    # KL = E[log p - log q] = sum(p * (log p - log q))
    policy_probs = torch.exp(log_policy)
    kl = torch.sum(policy_probs * (log_policy - log_reference), dim=-1)

    # Clamp to prevent NaN from numerical errors
    kl = torch.clamp(kl, min=0.0, max=1000.0)

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "none":
        return kl
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE provides a trade-off between bias and variance in advantage estimates.

    Args:
        rewards: Reward signal, shape (batch_size, seq_length)
        values: Value function estimates, shape (batch_size, seq_length)
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter (0=no lookahead, 1=full lookahead)
        normalize: Whether to normalize advantages

    Returns:
        advantages: GAE advantages, shape (batch_size, seq_length)
        returns: Discounted cumulative rewards, shape (batch_size, seq_length)
    """
    batch_size, seq_length = rewards.shape

    # Compute TD residuals
    deltas = (
        rewards
        + gamma * torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)
        - values
    )

    # Compute GAE backwards through time
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[:, 0:1])

    for t in reversed(range(seq_length)):
        gae = deltas[:, t : t + 1] + gamma * gae_lambda * gae
        advantages[:, t : t + 1] = gae

    # Compute returns
    returns = advantages + values

    # Normalize advantages (standard practice for stability)
    if normalize:
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

    # Clamp to prevent extreme values
    advantages = torch.clamp(advantages, -10.0, 10.0)

    return advantages, returns


def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    PPO objective with clipped probability ratio.

    L = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

    where r_t = exp(log_pi_new - log_pi_old)

    Args:
        old_log_probs: Log probabilities from old policy
        new_log_probs: Log probabilities from new policy
        advantages: Computed advantages
        epsilon: Clipping parameter (typically 0.2)

    Returns:
        PPO loss (scalar)
    """
    # Compute probability ratio
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # Return mean loss
    return loss.mean()


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    value_clip_epsilon: float = 0.2,
    regularization_lambda: float = 0.01,
) -> torch.Tensor:
    """
    Value function loss with optional clipping and L2 regularization.

    Args:
        values: Predicted values
        returns: Target returns
        old_values: Previous value estimates (for clipping)
        value_clip_epsilon: Clipping range for value updates
        regularization_lambda: L2 regularization coefficient

    Returns:
        Value loss (scalar)
    """
    # Base MSE loss
    value_loss_mse = F.mse_loss(values, returns)

    # Optional clipping for stability (like PPO)
    if old_values is not None:
        clipped_values = old_values + torch.clamp(
            values - old_values, -value_clip_epsilon, value_clip_epsilon
        )
        clipped_loss = F.mse_loss(clipped_values, returns)
        value_loss_mse = torch.max(value_loss_mse, clipped_loss)

    # L2 regularization on value predictions
    l2_reg = regularization_lambda * torch.sum(values**2)

    return value_loss_mse + l2_reg


def kl_penalty_loss(
    kl_divergence: torch.Tensor,
    kl_target: float = 0.01,
    kl_coef: float = 0.2,
) -> torch.Tensor:
    """
    Adaptive KL divergence penalty to prevent policy collapse.

    If KL > target, increase penalty coefficient to push policy back toward reference.

    Args:
        kl_divergence: Measured KL divergence (scalar or per-sample)
        kl_target: Target KL divergence (early stopping threshold)
        kl_coef: Penalty coefficient

    Returns:
        KL penalty loss term
    """
    # Adaptive penalty: increase if we exceed target
    penalty = kl_coef * torch.maximum(
        kl_divergence - kl_target, torch.tensor(0.0, device=kl_divergence.device)
    )

    return penalty.mean()


class PPOLossFn:
    """Aggregated PPO loss function with all components."""

    def __init__(
        self,
        epsilon: float = 0.2,
        value_clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        kl_target: float = 0.01,
        kl_coef: float = 0.2,
        value_weight: float = 1.0,
        entropy_weight: float = 0.01,
    ):
        self.epsilon = epsilon
        self.value_clip_epsilon = value_clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.kl_target = kl_target
        self.kl_coef = kl_coef
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

    def __call__(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        logits_policy: torch.Tensor,
        logits_reference: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total PPO loss with all components.

        Returns:
            total_loss: Scalar loss for backprop
            loss_dict: Dictionary of individual loss components for logging
        """
        # Compute advantages
        advantages, returns = compute_advantages(
            rewards,
            values,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize=True,
        )

        # PPO actor loss
        actor_loss = ppo_loss(
            old_log_probs,
            new_log_probs,
            advantages,
            epsilon=self.epsilon,
        )

        # Value function loss
        value_function_loss = value_loss(
            values,
            returns,
            old_values=old_values,
            value_clip_epsilon=self.value_clip_epsilon,
        )

        # KL divergence penalty
        kl_div = compute_kl_divergence(
            logits_policy,
            logits_reference,
            reduction="mean",
        )
        kl_penalty = kl_penalty_loss(kl_div, kl_target=self.kl_target, kl_coef=self.kl_coef)

        # Entropy bonus (discourage collapse to single action)
        policy_entropy = -torch.sum(torch.exp(new_log_probs) * new_log_probs, dim=-1).mean()

        # Total loss
        total_loss = (
            actor_loss
            + self.value_weight * value_function_loss
            + kl_penalty
            - self.entropy_weight * policy_entropy
        )

        loss_dict = {
            "actor_loss": actor_loss.item(),
            "value_loss": value_function_loss.item(),
            "kl_divergence": kl_div.item(),
            "kl_penalty": kl_penalty.item(),
            "policy_entropy": policy_entropy.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict
