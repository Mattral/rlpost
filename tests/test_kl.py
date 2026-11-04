"""Per-token KL(policy || reference) estimators k1, k2, and k3.

The closed form is ``sum_a p(a) * (log p(a) - log q(a))``. Monte Carlo
draws ``a ~ p`` and averages k1 and k3. Sample size is 1_000_000, seed 0,
absolute tolerance 1e-2. p = (0.8, 0.1, 0.1), q = (0.2, 0.2, 0.6). Those
two directions differ, so a swapped estimator misses KL(p || q).

TRL 1.17.1 ``GRPOTrainer._compute_loss`` (``trainer/grpo_trainer.py``)
assigns the k3 expression to ``per_token_kl``. ``use_bias_correction_kl``
defaults to True and then multiplies by the importance ratio. That product
is not k3. Parity is the uncorrected term on identical log-probabilities.
RLOO in the same release uses the k1 log ratio instead. It is not the
parity target.
"""

from pathlib import Path

import pytest
import torch
from trl.trainer.grpo_config import GRPOConfig

from rlpost.config import TrainingConfig
from rlpost.ppo_engine import PPOTrainer, per_token_kl

SAMPLE_SIZE = 1_000_000
SEED = 0
ATOL = 1e-2

# Not a swap of each other, so KL(p || q) and KL(q || p) differ.
POLICY = torch.tensor([0.8, 0.1, 0.1], dtype=torch.float64)
REFERENCE = torch.tensor([0.2, 0.2, 0.6], dtype=torch.float64)

# The expression GRPO assigns, copied so a source edit fails the scan below.
_GRPO_K3 = (
    "torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1"
)


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(p || q) for a categorical pair of probability vectors."""
    return torch.sum(p * (p.log() - q.log()))


def _trainer(estimator: str) -> PPOTrainer:
    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    config.alignment.kl_estimator = estimator
    policy = torch.nn.Linear(1, 1)
    value = torch.nn.Linear(1, 1)
    reference = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()))
    return PPOTrainer(config, policy, value, reference, optimizer)


def _sampled_logprobs() -> tuple[torch.Tensor, torch.Tensor]:
    """One draw of tokens from the policy, with both distributions' log-probs."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    index = torch.multinomial(POLICY, SAMPLE_SIZE, replacement=True, generator=generator)
    return POLICY.log()[index], REFERENCE.log()[index]


def test_monte_carlo_k1_and_k3_match_closed_form() -> None:
    """Means of k1 and k3 approach KL(policy || reference), not the swap."""
    kl_forward = _kl(POLICY, REFERENCE)
    kl_reverse = _kl(REFERENCE, POLICY)
    assert (kl_forward - kl_reverse).abs() > 5e-2

    logp_policy, logp_ref = _sampled_logprobs()
    for name in ("k1", "k3"):
        estimate = per_token_kl(logp_policy, logp_ref, name).mean()
        swapped = per_token_kl(logp_ref, logp_policy, name).mean()
        assert (estimate - kl_forward).abs() < ATOL
        assert (estimate - kl_reverse).abs() > ATOL
        assert (swapped - kl_forward).abs() > ATOL


def test_k3_nonnegative_and_k1_can_be_negative() -> None:
    """k3 stays >= 0. k1 takes a negative value when the policy is too high."""
    logp_policy = torch.tensor([0.0, -1.0, -0.5], dtype=torch.float64)
    logp_ref = torch.tensor([-1.0, -1.0, 0.5], dtype=torch.float64)

    k1 = per_token_kl(logp_policy, logp_ref, "k1")
    k2 = per_token_kl(logp_policy, logp_ref, "k2")
    k3 = per_token_kl(logp_policy, logp_ref, "k3")

    assert torch.any(k1 < 0)
    assert torch.all(k2 >= 0)
    assert torch.all(k3 >= 0)

    # The same draw used for the Monte Carlo mean.
    logp_policy_mc, logp_ref_mc = _sampled_logprobs()
    assert torch.any(per_token_kl(logp_policy_mc, logp_ref_mc, "k1") < 0)
    assert torch.all(per_token_kl(logp_policy_mc, logp_ref_mc, "k3") >= 0)


def test_configured_k1_scalar_can_be_negative() -> None:
    """The trainer reads kl_estimator. k1's mean can be negative. k3's cannot."""
    logp_policy = torch.tensor([0.0, 0.0])
    logp_ref = torch.tensor([1.0, 1.0])

    k3 = _trainer("k3").compute_kl_divergence(logp_policy, logp_ref)
    k1 = _trainer("k1").compute_kl_divergence(logp_policy, logp_ref)

    assert k3 > 0
    assert k1 < 0


@pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
def test_gradient_reaches_policy_logprobs_only(estimator: str) -> None:
    """Backprop through the mean reaches logp_policy and stops at logp_ref."""
    logp_policy = torch.tensor([0.0, -1.0, -0.5], dtype=torch.float64, requires_grad=True)
    logp_ref = torch.tensor([-1.0, -1.0, 0.5], dtype=torch.float64, requires_grad=True)

    loss = _trainer(estimator).compute_kl_divergence(logp_policy, logp_ref)
    loss.backward()

    assert logp_policy.grad is not None
    assert logp_policy.grad.abs().sum() > 0
    assert logp_ref.grad is None


def test_optional_mask_drops_tokens() -> None:
    """A keep-mask changes the mean. The all-zero mask is 0. None keeps all.

    ``train_step`` calls the estimator with no mask argument.
    """
    import inspect

    step = inspect.getsource(PPOTrainer.train_step)
    assert "compute_kl_divergence(token_logprobs, ref_token_logprobs)" in step
    assert "compute_kl_divergence(token_logprobs, ref_token_logprobs," not in step
    trainer = _trainer("k3")
    logp_policy = torch.tensor([0.0, -1.0, -2.0])
    logp_ref = torch.tensor([-1.0, -1.0, 0.0])
    mask = torch.tensor([1.0, 0.0, 1.0])

    values = per_token_kl(logp_policy, logp_ref, "k3")
    unmasked = trainer.compute_kl_divergence(logp_policy, logp_ref)
    masked = trainer.compute_kl_divergence(logp_policy, logp_ref, mask)
    zeros = trainer.compute_kl_divergence(logp_policy, logp_ref, torch.zeros(3))

    assert torch.allclose(unmasked, values.mean())
    assert torch.allclose(masked, (values[0] + values[2]) / 2)
    assert not torch.allclose(masked, unmasked)
    assert zeros.item() == 0.0

    with pytest.raises(ValueError):
        trainer.compute_kl_divergence(logp_policy, logp_ref, torch.ones(2))


def test_trl_1171_grpo_per_token_kl_is_k3() -> None:
    """GRPO's per-token KL is k3. The importance-ratio multiply is separate."""
    import dataclasses

    import trl
    import trl.trainer.grpo_trainer as grpo_trainer
    import trl.trainer.rloo_trainer as rloo_trainer

    assert trl.__version__ == "1.17.1"
    grpo_source = Path(grpo_trainer.__file__).read_text()
    rloo_source = Path(rloo_trainer.__file__).read_text()
    assert _GRPO_K3 in grpo_source
    assert "per_token_kl = per_token_kl * coef_1" in grpo_source
    assert "per_token_kl = old_per_token_logps - ref_per_token_logps" in rloo_source

    field = next(
        item for item in dataclasses.fields(GRPOConfig) if item.name == "use_bias_correction_kl"
    )
    assert field.default is True

    logp_policy = torch.tensor(
        [[-0.2, -1.4, -0.5], [-2.0, -0.3, -1.1]],
        dtype=torch.float64,
    )
    logp_ref = torch.tensor(
        [[-1.0, -0.2, -0.5], [-0.4, -1.7, -0.8]],
        dtype=torch.float64,
    )
    per_token_logps = logp_policy
    ref_per_token_logps = logp_ref
    trl_k3 = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps)
        - 1
    )
    ours = per_token_kl(logp_policy, logp_ref, "k3")
    torch.testing.assert_close(ours, trl_k3, atol=0, rtol=0)

    # float32, the dtype GRPO runs. Same expression, same bits.
    logp_policy32 = logp_policy.float()
    logp_ref32 = logp_ref.float()
    per_token_logps = logp_policy32
    ref_per_token_logps = logp_ref32
    trl_k3_32 = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps)
        - 1
    )
    torch.testing.assert_close(
        per_token_kl(logp_policy32, logp_ref32, "k3"),
        trl_k3_32,
        atol=0,
        rtol=0,
    )
