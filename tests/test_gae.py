"""GAE advantages and returns.

Parity target: the backward recursion in ``backward_gae`` below, and the
literal T=4 values next to it. TRL 1.17.1 has no PPO trainer and no GAE
helper. The installed tree has no module whose name contains ``ppo`` or
``gae``. GRPO's advantage is ``reward - group mean``. RLOO's is
leave-one-out. Neither is this recursion, so there is no TRL parity call.

Time is axis 1 of ``[B, T]``. PPO passes the action layout ``[B, T-1]``:
index ``t`` is the action that emits token ``t + 1``, ``rewards[:, t]`` is
that action's reward, and ``values[:, t]`` is ``V`` before the action.
``next_values[:, t]`` is ``V`` of the next state. On real indices,
``returns = advantages + values``. The numbers below are the recursion.
They do not depend on that token alignment.

fp32 absolute tolerance is 1e-6. The T=4 numbers are dyadic and match to
the bit on this CPU.
"""

import torch

from rlpost.ppo_engine import GeneralizedAdvantageEstimation

ATOL = 1e-6

# gamma = 0.5, lambda = 0.5, dones = 0.
# rewards      [1, 2, 3, 4]
# values       [0.5, 1, 0.5, 0]
# next_values  [1, 0.5, 0, 0]   # V_1, V_2, V_3, and a zero bootstrap
# delta        [1, 1.25, 2.5, 4]
# A            [1.53125, 2.125, 3.5, 4]
# returns      [2.03125, 3.125, 4, 4]
REWARDS = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
VALUES = torch.tensor([[0.5, 1.0, 0.5, 0.0]])
NEXT_VALUES = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
DONES = torch.zeros(1, 4)
HAND_ADVANTAGES = torch.tensor([[1.53125, 2.125, 3.5, 4.0]])
HAND_RETURNS = torch.tensor([[2.03125, 3.125, 4.0, 4.0]])


def backward_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The specification. It does not call ``GeneralizedAdvantageEstimation``.

    delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
    A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    returns = A + V
    """
    delta = rewards + gamma * next_values * (1.0 - dones) - values
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[0], dtype=rewards.dtype)
    for t in reversed(range(rewards.shape[1])):
        gae = delta[:, t] + gamma * lam * (1.0 - dones[:, t]) * gae
        advantages[:, t] = gae
    return advantages, advantages + values


def _compute(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    gae = GeneralizedAdvantageEstimation(gamma=gamma, lam=lam)
    return gae.compute(rewards, values, next_values, dones, mask)


def test_trl_1171_has_no_gae_helper() -> None:
    """Record the search. A future TRL GAE helper should grow a parity test."""
    from pathlib import Path

    import trl

    assert trl.__version__ == "1.17.1"
    root = Path(trl.__file__).resolve().parent
    needles = ("generalized advantage", "compute_gae", "lastgaelam", "gae_lambda")
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        hits.extend(f"{path.name}:{needle}" for needle in needles if needle in text)
    assert hits == []


def test_hand_example_matches_literals_and_recursion() -> None:
    """T=4, two rows. Row 0 is the hand example. Row 1 is cut by done."""
    rewards = torch.cat([REWARDS, torch.tensor([[4.0, 0.0, 0.0, 1.0]])], dim=0)
    values = torch.cat([VALUES, torch.zeros(1, 4)], dim=0)
    next_values = torch.cat([NEXT_VALUES, torch.zeros(1, 4)], dim=0)
    dones = torch.cat([DONES, torch.ones(1, 4)], dim=0)

    advantages, returns = _compute(rewards, values, next_values, dones, gamma=0.5, lam=0.5)
    spec_advantages, spec_returns = backward_gae(
        rewards, values, next_values, dones, gamma=0.5, lam=0.5
    )

    assert advantages.shape == (2, 4)
    assert torch.allclose(advantages[0], HAND_ADVANTAGES[0], atol=ATOL, rtol=0)
    assert torch.allclose(returns[0], HAND_RETURNS[0], atol=ATOL, rtol=0)
    assert torch.allclose(advantages, spec_advantages, atol=ATOL, rtol=0)
    assert torch.allclose(returns, spec_returns, atol=ATOL, rtol=0)
    assert torch.equal(advantages[1], rewards[1])


def test_done_in_the_middle_stops_the_backup() -> None:
    """done at t=1 drops A_2 out of A_1. Later tokens still back up among themselves."""
    dones = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    advantages, returns = _compute(REWARDS, VALUES, NEXT_VALUES, dones, gamma=0.5, lam=0.5)
    spec_advantages, spec_returns = backward_gae(
        REWARDS, VALUES, NEXT_VALUES, dones, gamma=0.5, lam=0.5
    )

    # A_3 = 4, A_2 = 2.5 + 0.25 * 4 = 3.5, A_1 = 1, A_0 = 1 + 0.25 * 1 = 1.25.
    expect = torch.tensor([[1.25, 1.0, 3.5, 4.0]])
    assert torch.allclose(advantages, expect, atol=ATOL, rtol=0)
    assert torch.allclose(returns, expect + VALUES, atol=ATOL, rtol=0)
    assert torch.allclose(advantages, spec_advantages, atol=ATOL, rtol=0)
    assert torch.allclose(returns, spec_returns, atol=ATOL, rtol=0)
    assert advantages[0, 1] < HAND_ADVANTAGES[0, 1]


def test_lambda_zero_is_one_step_td() -> None:
    """lambda = 0 keeps delta_t and drops A_{t+1}."""
    advantages, returns = _compute(REWARDS, VALUES, NEXT_VALUES, DONES, gamma=0.5, lam=0.0)
    delta = REWARDS + 0.5 * NEXT_VALUES - VALUES
    assert torch.allclose(advantages, delta, atol=ATOL, rtol=0)
    assert torch.allclose(advantages, torch.tensor([[1.0, 1.25, 2.5, 4.0]]), atol=ATOL, rtol=0)
    assert torch.allclose(returns, advantages + VALUES, atol=ATOL, rtol=0)
    assert not torch.allclose(advantages, HAND_ADVANTAGES, atol=ATOL, rtol=0)


def test_lambda_one_is_monte_carlo_minus_value() -> None:
    """lambda = 1 and a zero terminal bootstrap is the discounted return minus V."""
    advantages, returns = _compute(REWARDS, VALUES, NEXT_VALUES, DONES, gamma=0.5, lam=1.0)

    monte_carlo = torch.empty_like(REWARDS)
    acc = torch.zeros(REWARDS.shape[0], dtype=REWARDS.dtype)
    for t in reversed(range(REWARDS.shape[1])):
        acc = REWARDS[:, t] + 0.5 * acc
        monte_carlo[:, t] = acc

    assert torch.allclose(returns, monte_carlo, atol=ATOL, rtol=0)
    assert torch.allclose(advantages, monte_carlo - VALUES, atol=ATOL, rtol=0)
    assert torch.allclose(advantages, torch.tensor([[2.75, 3.5, 4.5, 4.0]]), atol=ATOL, rtol=0)


def test_gamma_one() -> None:
    """gamma = 1 still mixes delta with the discounted advantage backup."""
    advantages, returns = _compute(REWARDS, VALUES, NEXT_VALUES, DONES, gamma=1.0, lam=0.5)
    spec_advantages, spec_returns = backward_gae(
        REWARDS, VALUES, NEXT_VALUES, DONES, gamma=1.0, lam=0.5
    )

    # delta = [1.5, 1.5, 2.5, 4]
    # A_3 = 4, A_2 = 2.5 + 0.5 * 4 = 4.5, A_1 = 1.5 + 0.5 * 4.5 = 3.75
    # A_0 = 1.5 + 0.5 * 3.75 = 3.375
    expect = torch.tensor([[3.375, 3.75, 4.5, 4.0]])
    assert torch.allclose(advantages, expect, atol=ATOL, rtol=0)
    assert torch.allclose(returns, expect + VALUES, atol=ATOL, rtol=0)
    assert torch.allclose(advantages, spec_advantages, atol=ATOL, rtol=0)
    assert torch.allclose(returns, spec_returns, atol=ATOL, rtol=0)


def test_right_padding_does_not_leak() -> None:
    """A pad suffix with a huge value must not change the real prefix.

    dones stay 0, so the cut is the mask. ``next_values`` at the last real
    token is the pad's value. The real advantages stay the T=4 example,
    whose bootstrap at the end is 0.
    """
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 100.0, 100.0]])
    values = torch.tensor([[0.5, 1.0, 0.5, 0.0, 80.0, 80.0]])
    next_values = torch.tensor([[1.0, 0.5, 0.0, 80.0, 80.0, 0.0]])
    dones = torch.zeros(1, 6)
    mask = torch.tensor([[True, True, True, True, False, False]])

    advantages, returns = _compute(
        rewards, values, next_values, dones, gamma=0.5, lam=0.5, mask=mask
    )

    assert torch.allclose(advantages[0, :4], HAND_ADVANTAGES[0], atol=ATOL, rtol=0)
    assert torch.allclose(returns[0, :4], HAND_RETURNS[0], atol=ATOL, rtol=0)
    assert torch.equal(advantages[0, 4:], torch.zeros(2))
    assert torch.equal(returns[0, 4:], torch.zeros(2))


def test_padding_between_tokens_does_not_leak() -> None:
    """A pad splits the sequence. The left side does not see the pad or the right side."""
    rewards = torch.tensor([[1.0, 2.0, 100.0, 3.0]])
    values = torch.tensor([[0.5, 1.0, 50.0, 0.25]])
    next_values = torch.tensor([[1.0, 50.0, 0.25, 0.0]])
    dones = torch.zeros(1, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    advantages, returns = _compute(
        rewards, values, next_values, dones, gamma=0.5, lam=0.5, mask=mask
    )

    # t=3 stands alone: 3 - 0.25 = 2.75. t=1 stops before the pad: 2 - 1 = 1.
    # t=0 backs up only through t=1: 1 + 0.25 * 1 = 1.25. The pad is 0.
    assert torch.allclose(advantages, torch.tensor([[1.25, 1.0, 0.0, 2.75]]), atol=ATOL, rtol=0)
    assert torch.allclose(returns, torch.tensor([[1.75, 2.0, 0.0, 3.0]]), atol=ATOL, rtol=0)
