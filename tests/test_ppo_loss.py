"""Clipped PPO policy loss, value loss, and entropy.

Policy and value parity is the closed form in this file. TRL 1.17.1 has no
PPO trainer and no value-loss function. ``SDPOTrainer._compute_policy_loss``
is a masked GRPO reduction, not this flat mean, and it is not called.

Entropy parity is ``trl.trainer.utils.entropy_from_logits`` (TRL 1.17.1,
``trainer/utils.py``). That helper returns one value per row. This loss
uses the mean of those rows. Triton is not used on this CPU.

``ppo_epsilon`` clips the probability ratio only. ``value_clip`` is the
value radius ``c``. Unset, the value loss is ``value_loss_coefficient *
mean((V - R)^2)``. Set, it is ``value_loss_coefficient * mean(0.5 *
max((V - R)^2, (V_clip - R)^2))`` with
``V_clip = V_old + clip(V - V_old, -c, c)``.

fp32 absolute tolerance is 1e-6. The hand numbers are dyadic.
"""

import pytest
import torch
from trl.trainer.utils import entropy_from_logits

from rlpost.config import TrainingConfig
from rlpost.ppo_engine import PPOTrainer

ATOL = 1e-6


def _trainer(epsilon: float, value_coef: float, value_clip: float | None) -> PPOTrainer:
    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    config.alignment.ppo_epsilon = epsilon
    config.alignment.value_loss_coefficient = value_coef
    config.alignment.value_clip = value_clip
    policy = torch.nn.Linear(1, 1)
    value = torch.nn.Linear(1, 1)
    reference = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()))
    return PPOTrainer(config, policy, value, reference, optimizer)


def _closed_form_policy(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.exp(log_probs_new - log_probs_old)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages
    return -torch.min(unclipped, clipped).mean(), ratio


def _closed_form_value(
    value_pred: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None,
    clip: float | None,
    coefficient: float,
) -> torch.Tensor:
    unclipped = (value_pred - returns).square()
    if clip is None:
        return unclipped.mean() * coefficient
    assert old_values is not None
    value_clipped = old_values + (value_pred - old_values).clamp(-clip, clip)
    clipped = (value_clipped - returns).square()
    return torch.maximum(unclipped, clipped).mean() * 0.5 * coefficient


def test_trl_1171_entropy_helper_and_no_ppo_value_loss() -> None:
    """The entropy helper is real. A PPO value clip is not in this TRL."""
    from pathlib import Path

    import trl

    assert trl.__version__ == "1.17.1"
    assert entropy_from_logits.__module__ == "trl.trainer.utils"
    root = Path(trl.__file__).resolve().parent
    needles = ("cliprange_value", "vf_loss", "class ppotrainer")
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        hits.extend(f"{path.name}:{needle}" for needle in needles if needle in text)
    assert hits == []


def test_policy_loss_matches_closed_form() -> None:
    """eps = 0.25. Ratios 1.5 and 0.5 bind, on A > 0 and A < 0.

    clip(r) is [[1, 1], [1.25, 0.75]].
    min(r*A, clip(r)*A) is [[1, -1], [1.25, -0.75]].
    The mean is 0.125. The loss is -0.125.
    The upper clip binds where A is 1. The lower clip binds where A is -1.
    """
    epsilon = 0.25
    trainer = _trainer(epsilon, value_coef=1.0, value_clip=None)
    log_probs_old = torch.zeros(2, 2)
    log_probs_new = torch.log(torch.tensor([[1.0, 1.0], [1.5, 0.5]]))
    advantages = torch.tensor([[1.0, -1.0], [1.0, -1.0]])

    loss, ratio = trainer.compute_policy_loss(log_probs_new, log_probs_old, advantages)
    spec_loss, spec_ratio = _closed_form_policy(log_probs_new, log_probs_old, advantages, epsilon)

    assert torch.allclose(ratio, torch.tensor([[1.0, 1.0], [1.5, 0.5]]), atol=ATOL, rtol=0)
    assert torch.allclose(loss, torch.tensor(-0.125), atol=ATOL, rtol=0)
    assert torch.allclose(loss, spec_loss, atol=ATOL, rtol=0)
    assert torch.allclose(ratio, spec_ratio, atol=ATOL, rtol=0)


def test_value_loss_clipped_matches_closed_form() -> None:
    """c = 0.25 and ppo_epsilon = 0.1, so the value radius is not the policy clip.

    Row 0 moves from 1 to 0 toward a return of 0. The clip stops it at 0.75.
    The unclipped square is 0. The clipped square is 0.5625. The max is 0.5625.
    Row 1 stays at 1, inside the clip, against a return of 0.5. The square is 0.25.
    The mean of the max is 0.40625. Times 0.5, with coefficient 1, is 0.203125.
    """
    clip = 0.25
    trainer = _trainer(epsilon=0.1, value_coef=1.0, value_clip=clip)
    value_pred = torch.tensor([[0.0], [1.0]])
    old_values = torch.tensor([[1.0], [1.0]])
    returns = torch.tensor([[0.0], [0.5]])

    loss = trainer.compute_value_loss(value_pred, returns, old_values)
    spec = _closed_form_value(value_pred, returns, old_values, clip, 1.0)

    assert torch.allclose(loss, torch.tensor(0.203125), atol=ATOL, rtol=0)
    assert torch.allclose(loss, spec, atol=ATOL, rtol=0)
    # A radius of 0.1, the policy epsilon, would stop row 0 at 0.9.
    wrong_radius = _closed_form_value(value_pred, returns, old_values, 0.1, 1.0)
    assert not torch.allclose(loss, wrong_radius, atol=ATOL, rtol=0)


def test_value_loss_inside_the_clip_is_half_mse() -> None:
    """A move smaller than c makes the clipped loss 0.5 * coefficient * MSE."""
    trainer = _trainer(epsilon=0.1, value_coef=0.5, value_clip=0.25)
    value_pred = torch.tensor([1.0, 0.5])
    old_values = torch.tensor([1.0, 0.5])
    returns = torch.tensor([0.0, 0.0])

    loss = trainer.compute_value_loss(value_pred, returns, old_values)
    half_mse = torch.nn.functional.mse_loss(value_pred, returns) * 0.5 * 0.5

    # MSE is 0.625. 0.5 * coefficient 0.5 * 0.625 = 0.15625.
    assert torch.allclose(loss, torch.tensor(0.15625), atol=ATOL, rtol=0)
    assert torch.allclose(loss, half_mse, atol=ATOL, rtol=0)


def test_value_loss_unclipped_matches_mse() -> None:
    """value_clip unset keeps coefficient * MSE, including when V_old is passed."""
    trainer = _trainer(epsilon=0.2, value_coef=0.5, value_clip=None)
    value_pred = torch.tensor([[0.0], [1.0]])
    old_values = torch.tensor([[1.0], [1.0]])
    returns = torch.tensor([[0.0], [0.5]])

    loss = trainer.compute_value_loss(value_pred, returns, old_values)
    omitted = trainer.compute_value_loss(value_pred, returns)
    spec = _closed_form_value(value_pred, returns, None, None, 0.5)
    clipped = _closed_form_value(value_pred, returns, old_values, 0.25, 0.5)

    assert torch.allclose(loss, torch.tensor(0.0625), atol=ATOL, rtol=0)
    assert torch.allclose(loss, omitted, atol=ATOL, rtol=0)
    assert torch.allclose(loss, spec, atol=ATOL, rtol=0)
    assert not torch.allclose(loss, clipped, atol=ATOL, rtol=0)


def test_value_clip_zero_uses_the_old_value() -> None:
    """c = 0 forces V_clip = V_old. ppo_epsilon does not."""
    trainer = _trainer(epsilon=0.2, value_coef=1.0, value_clip=0.0)
    value_pred = torch.tensor([0.0])
    old_values = torch.tensor([1.0])
    returns = torch.tensor([0.0])

    loss = trainer.compute_value_loss(value_pred, returns, old_values)

    assert torch.allclose(loss, torch.tensor(0.5), atol=ATOL, rtol=0)


def test_value_clip_requires_old_values() -> None:
    trainer = _trainer(epsilon=0.2, value_coef=1.0, value_clip=0.2)
    with pytest.raises(ValueError, match="old_values"):
        trainer.compute_value_loss(torch.zeros(2), torch.zeros(2))


def test_entropy_matches_trl_and_a_known_distribution() -> None:
    """Uniform logits have entropy log(n). A 3/4, 1/4 row has a hand entropy."""
    import trl

    assert trl.__version__ == "1.17.1"
    trainer = _trainer(epsilon=0.2, value_coef=0.5, value_clip=None)
    logits = torch.zeros(2, 3, 4)
    logits[1, :, 0] = 5.0
    logits[1, :, 1:] = -5.0

    got = trainer.compute_entropy_bonus(logits)
    per_row = entropy_from_logits(logits)
    assert per_row.shape == (2, 3)
    assert torch.allclose(got, per_row.mean(), atol=ATOL, rtol=0)

    two_way = torch.zeros(1, 2)
    two_entropy = trainer.compute_entropy_bonus(two_way)
    assert torch.allclose(two_entropy, torch.log(torch.tensor(2.0)), atol=ATOL, rtol=0)

    probs = torch.tensor([0.75, 0.25])
    hand = -(probs * probs.log()).sum()
    assert torch.allclose(
        trainer.compute_entropy_bonus(probs.log().view(1, 2)),
        hand,
        atol=ATOL,
        rtol=0,
    )
