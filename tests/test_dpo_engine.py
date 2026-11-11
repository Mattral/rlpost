"""DPO loss, rewards, margin, and sequence log-probabilities.

Parity target: trl==1.17.1.
The loss lives on ``trl.trainer.dpo_trainer.DPOTrainer._compute_loss``.
That release has no free function for it. ``DPOTrainer.__init__`` also
builds a Hugging Face ``Trainer``, requires a tokenizer, and can load a
reference model from ``config._name_or_path``. The parity test skips that
constructor. It builds the instance with ``DPOTrainer.__new__`` and sets
the attributes ``_compute_loss`` reads, then calls the method.

Sigmoid DPO, ``label_smoothing`` 0. In this release the sigmoid branch does
not read ``label_smoothing``. Smoothing is on the robust, exo_pair, and aot
losses. The sigmoid loss is ``-logsigmoid(beta * delta).mean()`` with
``delta = (logp_policy - logp_ref)_chosen - (logp_policy - logp_ref)_rejected``.

Sequence log-probabilities are a sum over completion tokens. The fused head
writes 0 where ``completion_mask`` is 0 (prompt and padding), after shifting
labels by one. ``_compute_loss`` then sums the row when ``ld_alpha`` is
``None``. It does not average.

The parity tests run seeds 0 through 5. The policy is ``tiny_causal_lm(seed)``,
the reference is ``tiny_causal_lm(seed + 10)``, and the token batch uses that
same seed. Those two models are different initializations on purpose.
``run_dpo_training`` freezes a copy of the initial policy, so its first loss
is ln 2 and the gap is identically zero. This file keeps the models distinct
so the beta-scaled gap is what gets compared with TRL.
Each seed runs three times: the same masks on both sides, a batch
of 2 where chosen and rejected have different completion lengths and different
padding, and a batch of 4 with a different prompt length and completion
length on every row. Those masks are separate arguments on both sides.

fp32 absolute tolerance on the loss, the margin, and the rewards is 1e-6.
TRL logs the batch mean of those sums (``logps/chosen``, ``logps/rejected``).
On CPU, torch 2.15, this Llama, seeds 0-5, and the two batch-of-2 layouts,
the largest gap on that mean was 1.907e-6, one float32 ulp near magnitude 22.
The tolerance on the logged means is 1e-5. The batch of 4 passes that same
tolerance. A separate maximum for that layout was not recorded.

Policy-parameter gradients use absolute tolerance 1e-6. The two losses are
backwarded from identical initial weights, with no optimizer step. On the
batch-of-2 grid the largest |rlpost grad - TRL grad| was 9.686e-8. The batch
of 4 passes the same tolerance. A separate maximum for that layout was not
recorded. The reference parameters are frozen, and their grads stay None.
"""

import logging
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from tests.tiny_lm import VOCAB, tiny_causal_lm
from trl.trainer.dpo_trainer import DPOTrainer as TrlDPOTrainer
from trl.trainer.utils import add_fused_lm_head

from rlpost.config import TrainingConfig
from rlpost.dpo_engine import (
    DPOMetrics,
    DPOTrainer,
    collate_preference_pairs,
    completion_token_mask,
    prompt_prefix_length,
)
from rlpost.ops import gather_logprobs

FP32_ATOL = 1e-6
# One float32 ulp of a completion-token sum near magnitude 22 is 1.907e-6.
SUM_MEAN_ATOL = 1e-5
# Measured max |policy grad - TRL grad| on seeds 0-5 and both layouts: 9.686e-8.
GRAD_ATOL = 1e-6

logger = logging.getLogger(__name__)


class TestDPOMetrics:
    """Test DPOMetrics container."""

    def test_init(self):
        """Test DPOMetrics initialization."""
        metrics = DPOMetrics(
            dpo_loss=0.5,
            policy_diff_mean=0.1,
            margin_mean=0.2,
            accuracy=0.85,
        )
        assert metrics.dpo_loss == 0.5
        assert metrics.policy_diff_mean == 0.1
        assert metrics.margin_mean == 0.2
        assert metrics.accuracy == 0.85

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = DPOMetrics(
            dpo_loss=0.5,
            policy_diff_mean=0.1,
            margin_mean=0.2,
            accuracy=0.85,
        )
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["dpo_loss"] == 0.5
        assert d["accuracy"] == 0.85
        assert "explained_variance" not in d
        assert len(d) == 4


def _trainer(
    policy_seed: int = 0,
    reference_seed: int = 1,
    beta: float | None = None,
    learning_rate: float | None = None,
) -> DPOTrainer:
    # Distinct models. The training path copies the policy; this helper
    # does not, because a copy would make the TRL gap identically zero.
    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    if beta is not None:
        config.alignment.dpo_beta = beta
    if learning_rate is not None:
        config.optimization.learning_rate = learning_rate
    policy = tiny_causal_lm(policy_seed)
    reference = tiny_causal_lm(reference_seed)
    return DPOTrainer(config, policy_model=policy, reference_model=reference)


class TestDPOTrainerInit:
    """Test DPOTrainer initialization on a tiny causal LM."""

    def test_init_creates_trainer(self):
        """A real Llama has parameters, and beta is dpo_beta rather than the PPO coefficient."""
        config = TrainingConfig.toy_mode()
        config.log_to_wandb = False
        policy = tiny_causal_lm(0)
        reference = tiny_causal_lm(1)
        trainer = DPOTrainer(config, policy_model=policy, reference_model=reference)

        assert trainer.config is config
        assert trainer.policy_model is policy
        assert trainer.reference_model is reference
        assert trainer.beta == config.alignment.dpo_beta == 0.1
        assert trainer.beta != config.alignment.kl_coefficient
        assert trainer.step_count == 0
        assert any(p.requires_grad for p in trainer.policy_model.parameters())
        opt_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert opt_ids == {id(p) for p in policy.parameters()}

    def test_reference_model_frozen(self):
        """The reference starts trainable. The trainer freezes it and leaves it out of the loss."""
        config = TrainingConfig.toy_mode()
        config.log_to_wandb = False
        policy = tiny_causal_lm(0)
        reference = tiny_causal_lm(1)
        reference.requires_grad_(True)
        trainer = DPOTrainer(config, policy_model=policy, reference_model=reference)

        assert all(not p.requires_grad for p in trainer.reference_model.parameters())
        opt_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert opt_ids.isdisjoint(id(p) for p in trainer.reference_model.parameters())

        input_ids = torch.randint(0, VOCAB, (2, 8))
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        completion_mask = torch.zeros(2, 8, dtype=torch.long)
        completion_mask[:, 3:] = 1
        policy_logp = trainer.compute_log_probs(
            trainer.policy_model, input_ids, attention_mask, completion_mask
        )
        with torch.no_grad():
            ref_logp = trainer.compute_log_probs(
                trainer.reference_model, input_ids, attention_mask, completion_mask
            )
        rejected = torch.zeros_like(policy_logp)
        loss, _ = trainer.compute_dpo_loss(policy_logp, rejected, ref_logp, ref_logp)
        loss.backward()

        assert any(
            p.grad is not None and torch.count_nonzero(p.grad) > 0
            for p in trainer.policy_model.parameters()
        )
        assert all(p.grad is None for p in trainer.reference_model.parameters())


class TestDPOLossComputation:
    """Test DPO loss computation."""

    def test_compute_dpo_loss_shapes(self):
        """Test loss computation with correct tensor shapes."""
        # Create mock trainer
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1
        trainer.step_count = 0

        # Call actual compute_dpo_loss method
        policy_logits_chosen = torch.tensor([0.5, 0.6, 0.7, 0.8])
        policy_logits_rejected = torch.tensor([0.3, 0.4, 0.5, 0.6])
        ref_logits_chosen = torch.tensor([0.4, 0.5, 0.6, 0.7])
        ref_logits_rejected = torch.tensor([0.2, 0.3, 0.4, 0.5])

        # Use actual method
        loss, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_logits_chosen,
            policy_logits_rejected,
            ref_logits_chosen,
            ref_logits_rejected,
        )

        assert loss is not None
        assert isinstance(metrics, DPOMetrics)
        assert metrics.dpo_loss > 0  # Loss should be positive

    def test_dpo_loss_with_different_preferences(self):
        """Test loss when chosen > rejected (should be lower)."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Case 1: Chosen has higher logits (good preference alignment)
        policy_logits_chosen = torch.tensor([1.0, 1.0])
        policy_logits_rejected = torch.tensor([0.0, 0.0])
        ref_logits_chosen = torch.tensor([0.8, 0.8])
        ref_logits_rejected = torch.tensor([0.2, 0.2])

        loss_aligned, _ = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_logits_chosen,
            policy_logits_rejected,
            ref_logits_chosen,
            ref_logits_rejected,
        )

        # Case 2: Rejected has higher logits (bad preference alignment)
        policy_logits_chosen = torch.tensor([0.0, 0.0])
        policy_logits_rejected = torch.tensor([1.0, 1.0])

        loss_misaligned, _ = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_logits_chosen,
            policy_logits_rejected,
            ref_logits_chosen,
            ref_logits_rejected,
        )

        # Aligned loss should be lower (better)
        assert loss_aligned < loss_misaligned

    def test_dpo_loss_bounds(self):
        """Test that DPO loss stays within reasonable bounds."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Create random batches
        batch_size = 16
        policy_chosen = torch.randn(batch_size)
        policy_rejected = torch.randn(batch_size)
        ref_chosen = torch.randn(batch_size)
        ref_rejected = torch.randn(batch_size)

        loss, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Loss should be finite
        assert torch.isfinite(loss)
        assert loss.item() > 0

        # Metrics should be valid
        assert 0 <= metrics.accuracy <= 1
        assert torch.isfinite(torch.tensor(metrics.dpo_loss))


class TestDPOMetricsComputation:
    """Test metrics computation during DPO training."""

    def test_accuracy_computation(self):
        """Test accuracy metric (fraction where chosen > rejected)."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Perfect preference alignment: chosen always > rejected
        policy_chosen = torch.tensor([1.0, 2.0, 3.0])
        policy_rejected = torch.tensor([0.0, 0.5, 1.0])
        ref_chosen = torch.tensor([0.9, 1.8, 2.8])
        ref_rejected = torch.tensor([0.1, 0.7, 1.2])

        _, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        assert metrics.accuracy == 1.0  # All chosen > rejected

    def test_margin_computation(self):
        """Margin is mean(beta * (logp_policy - logp_ref) chosen minus rejected)."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        policy_chosen = torch.tensor([2.0, 3.0, 4.0])
        policy_rejected = torch.tensor([1.0, 1.5, 2.0])
        ref_chosen = torch.tensor([1.8, 2.8, 3.8])
        ref_rejected = torch.tensor([1.2, 1.7, 2.2])

        _, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # chosen log-ratio is 0.2, rejected log-ratio is -0.2, beta is 0.1.
        # Per-row margin is 0.04. The policy-only gap on this fixture is 1.5.
        chosen_reward = 0.1 * (policy_chosen - ref_chosen)
        rejected_reward = 0.1 * (policy_rejected - ref_rejected)
        expected_margin = (chosen_reward - rejected_reward).mean().item()
        assert expected_margin == pytest.approx(0.04, abs=1e-6)
        assert metrics.margin_mean == pytest.approx(expected_margin, abs=1e-6)


class TestDPOBetaParameter:
    """Test DPO beta (temperature) parameter."""

    def test_beta_affects_loss(self):
        """Beta scales a non-zero reference-adjusted margin.

        The previous fixture had policy_diff - ref_diff == 0, so every beta
        returned -logsigmoid(0). Here the unscaled margin is 1. A larger beta
        makes -logsigmoid(beta) smaller.
        """
        trainer_low_beta = MagicMock(spec=DPOTrainer)
        trainer_low_beta.beta = 0.01

        trainer_high_beta = MagicMock(spec=DPOTrainer)
        trainer_high_beta.beta = 1.0

        policy_chosen = torch.tensor([1.0, 1.0, 1.0])
        policy_rejected = torch.tensor([0.0, 0.0, 0.0])
        ref_chosen = torch.zeros(3)
        ref_rejected = torch.zeros(3)

        loss_low, _ = DPOTrainer.compute_dpo_loss(
            trainer_low_beta,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        loss_high, _ = DPOTrainer.compute_dpo_loss(
            trainer_high_beta,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        assert torch.allclose(loss_low, -F.logsigmoid(torch.tensor(0.01)), atol=FP32_ATOL)
        assert torch.allclose(loss_high, -F.logsigmoid(torch.tensor(1.0)), atol=FP32_ATOL)
        assert loss_high < loss_low


class TestDPONumericalStability:
    """Test numerical stability of DPO computation."""

    def test_large_logits_stability(self):
        """Test stability with large logit values."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Large logits can cause numerical issues with sigmoid
        policy_chosen = torch.tensor([100.0, 200.0, 150.0])
        policy_rejected = torch.tensor([50.0, 150.0, 100.0])
        ref_chosen = torch.tensor([90.0, 190.0, 140.0])
        ref_rejected = torch.tensor([40.0, 140.0, 90.0])

        loss, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Should not be NaN or Inf
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_small_logits_stability(self):
        """Test stability with very small logit differences."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        policy_chosen = torch.tensor([0.0001, 0.0001, 0.0001])
        policy_rejected = torch.tensor([0.0, 0.0, 0.0])
        ref_chosen = torch.tensor([0.00005, 0.00005, 0.00005])
        ref_rejected = torch.tensor([0.0, 0.0, 0.0])

        loss, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Should handle small differences gracefully
        assert torch.isfinite(loss)


class TestDPOBatchProcessing:
    """Test DPO batch processing."""

    def test_different_batch_sizes(self):
        """Test loss computation across different batch sizes."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        for batch_size in [1, 4, 8, 16, 32]:
            policy_chosen = torch.randn(batch_size)
            policy_rejected = torch.randn(batch_size)
            ref_chosen = torch.randn(batch_size)
            ref_rejected = torch.randn(batch_size)

            loss, metrics = DPOTrainer.compute_dpo_loss(
                trainer,
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
            )

            assert torch.isfinite(loss)
            assert 0 <= metrics.accuracy <= 1

    def test_metrics_batch_aggregation(self):
        """Test that metrics aggregate correctly across batch."""
        trainer = MagicMock(spec=DPOTrainer)
        trainer.beta = 0.1

        # Batch with known properties
        policy_chosen = torch.tensor([1.0, 2.0, 3.0, 4.0])
        policy_rejected = torch.tensor([0.0, 0.5, 1.0, 1.5])
        ref_chosen = torch.ones(4)
        ref_rejected = torch.zeros(4)

        _, metrics = DPOTrainer.compute_dpo_loss(
            trainer,
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
        )

        # Row 0 is a tie after the reference adjustment (both rewards are 0).
        # Strict > does not count a tie, so accuracy is 0.75. The mean margin
        # is beta * mean([0, 0.5, 1.0, 1.5]) = 0.075.
        assert metrics.accuracy == pytest.approx(0.75, abs=1e-6)
        chosen_reward = trainer.beta * (policy_chosen - ref_chosen)
        rejected_reward = trainer.beta * (policy_rejected - ref_rejected)
        expected_margin = (chosen_reward - rejected_reward).mean().item()
        assert expected_margin == pytest.approx(0.075, abs=1e-6)
        assert metrics.margin_mean == pytest.approx(expected_margin, abs=1e-6)


class TestDPOConfiguration:
    """Test DPO integration with TrainingConfig."""

    def test_config_integration(self):
        """The trainer reads the learning rate and the model ids off the config."""
        trainer = _trainer(policy_seed=2, reference_seed=3, learning_rate=3e-4)

        assert trainer.config.model.policy_model_id == "distilgpt2"
        assert trainer.config.model.reference_model_id == "distilgpt2"
        assert trainer.optimizer.param_groups[0]["lr"] == 3e-4

    def test_custom_beta_from_config(self):
        """dpo_beta is read off the field. There is no getattr fallback."""
        trainer = _trainer(policy_seed=4, reference_seed=5, beta=0.5)
        assert trainer.beta == 0.5
        assert trainer.config.alignment.dpo_beta == 0.5


def _pair_batch(seed: int = 0) -> dict[str, torch.Tensor]:
    """Chosen and rejected rows, prompt length 3, last column padded on row 1."""
    torch.manual_seed(seed)
    chosen = torch.randint(1, VOCAB, (2, 8))
    rejected = torch.randint(1, VOCAB, (2, 8))
    attention = torch.ones(2, 8, dtype=torch.long)
    attention[1, -1] = 0
    chosen[1, -1] = 0
    rejected[1, -1] = 0
    completion = torch.zeros(2, 8, dtype=torch.long)
    completion[:, 3:] = 1
    completion = completion * attention
    return {
        "chosen_ids": chosen,
        "rejected_ids": rejected,
        "attention_mask": attention,
        "completion_mask": completion,
    }


def _unequal_batch(seed: int) -> dict[str, torch.Tensor]:
    """Chosen and rejected differ in completion length and in padding.

    Both sides are width 8 so TRL can concatenate them. Chosen prompt length
    is 2, with completion lengths 4 and 2 (the tails are padding). Rejected
    prompt length is 3, with completion lengths 3 and 5 (row 1 has no pad).
    """
    torch.manual_seed(seed)
    width = 8
    chosen = torch.randint(1, VOCAB, (2, width))
    rejected = torch.randint(1, VOCAB, (2, width))

    chosen_attention = torch.zeros(2, width, dtype=torch.long)
    chosen_attention[0, :6] = 1
    chosen_attention[1, :4] = 1
    chosen_completion = torch.zeros(2, width, dtype=torch.long)
    chosen_completion[0, 2:6] = 1
    chosen_completion[1, 2:4] = 1

    rejected_attention = torch.zeros(2, width, dtype=torch.long)
    rejected_attention[0, :6] = 1
    rejected_attention[1, :] = 1
    rejected_completion = torch.zeros(2, width, dtype=torch.long)
    rejected_completion[0, 3:6] = 1
    rejected_completion[1, 3:] = 1

    chosen = chosen * chosen_attention
    rejected = rejected * rejected_attention
    return {
        "chosen_ids": chosen,
        "rejected_ids": rejected,
        "chosen_attention_mask": chosen_attention,
        "rejected_attention_mask": rejected_attention,
        "chosen_completion_mask": chosen_completion,
        "rejected_completion_mask": rejected_completion,
    }


def _fill_side(
    seed: int,
    specs: list[tuple[int, int]],
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Token ids, attention, and completion for one side.

    ``specs`` is ``(prompt_length, completion_length)`` per row. The tail
    past ``prompt + completion`` is padding. Ids on the pad are 0.
    """
    torch.manual_seed(seed)
    batch = len(specs)
    ids = torch.randint(1, VOCAB, (batch, width))
    attention = torch.zeros(batch, width, dtype=torch.long)
    completion = torch.zeros(batch, width, dtype=torch.long)
    for index, (prompt, length) in enumerate(specs):
        if prompt + length > width:
            raise ValueError(f"row {index} does not fit in width {width}")
        attention[index, : prompt + length] = 1
        completion[index, prompt : prompt + length] = 1
    return ids * attention, attention, completion


def _unequal_batch_of_4(seed: int) -> dict[str, torch.Tensor]:
    """Four rows. Prompt lengths and completion lengths differ on each side.

    Width 10. Chosen prompt lengths are 2, 4, 1, 3. Completion lengths are
    6, 2, 5, 7, so the last chosen row has no pad. Rejected prompt lengths
    are 3, 1, 5, 2. Completion lengths are 4, 9, 3, 5, so the second
    rejected row has no pad. The two sides are not the same mask.
    """
    width = 10
    chosen, chosen_attention, chosen_completion = _fill_side(
        seed,
        [(2, 6), (4, 2), (1, 5), (3, 7)],
        width,
    )
    rejected, rejected_attention, rejected_completion = _fill_side(
        seed + 1,
        [(3, 4), (1, 9), (5, 3), (2, 5)],
        width,
    )
    return {
        "chosen_ids": chosen,
        "rejected_ids": rejected,
        "chosen_attention_mask": chosen_attention,
        "rejected_attention_mask": rejected_attention,
        "chosen_completion_mask": chosen_completion,
        "rejected_completion_mask": rejected_completion,
    }


def _preference_batch(layout: str, seed: int) -> dict[str, torch.Tensor]:
    """One pair batch. Masks are per side, even when the values match."""
    if layout == "matched":
        batch = _pair_batch(seed)
        attention = batch["attention_mask"]
        completion = batch["completion_mask"]
        return {
            "chosen_ids": batch["chosen_ids"],
            "rejected_ids": batch["rejected_ids"],
            "chosen_attention_mask": attention,
            "rejected_attention_mask": attention.clone(),
            "chosen_completion_mask": completion,
            "rejected_completion_mask": completion.clone(),
        }
    if layout == "unequal":
        return _unequal_batch(seed)
    if layout == "unequal4":
        return _unequal_batch_of_4(seed)
    raise ValueError(f"unknown layout {layout}")


def _assert_sides_differ(batch: dict[str, torch.Tensor]) -> None:
    """The unequal case is not the same mask concatenated with itself."""
    assert not torch.equal(batch["chosen_attention_mask"], batch["rejected_attention_mask"])
    assert not torch.equal(batch["chosen_completion_mask"], batch["rejected_completion_mask"])
    assert not torch.equal(
        batch["chosen_completion_mask"].sum(dim=1),
        batch["rejected_completion_mask"].sum(dim=1),
    )


def _prompt_lengths(completion: torch.Tensor) -> list[int]:
    """Columns before the first completion token. That is the prompt prefix."""
    lengths = []
    for row in completion:
        ones = (row == 1).nonzero(as_tuple=False)
        lengths.append(int(ones[0]) if len(ones) else int(row.shape[0]))
    return lengths


def _assert_prompt_lengths_differ(completion: torch.Tensor) -> None:
    """Per-row prompt lengths are not one shared width."""
    lengths = _prompt_lengths(completion)
    assert len(set(lengths)) > 1


def _side_logprobs(
    trainer: DPOTrainer,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Policy or reference log-probs. Each side keeps its own masks."""
    chosen = trainer.compute_log_probs(
        model,
        batch["chosen_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_completion_mask"],
    )
    rejected = trainer.compute_log_probs(
        model,
        batch["rejected_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_completion_mask"],
    )
    return chosen, rejected


class _CharTokenizer:
    """Maps each character to an id. Prompt text is a prefix of prompt+response.

    ``pad_token_id`` is 0. Character ids start at 1, so a pad is not a letter.
    This tokenizer does not pad. ``collate_completions`` right-pads.
    """

    pad_token_id = 0

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: int = 32,
        padding: str | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict:
        del truncation, padding, return_tensors, add_special_tokens
        ids = [max(min(ord(ch), 120), 1) for ch in text][:max_length]
        return {"input_ids": ids}


class TestCompletionLogProb:
    """Sequence log-prob is the sum of completion tokens only."""

    def test_prompt_tokens_are_not_counted(self):
        """Fails if the reduction uses the attention mask, which still marks the prompt."""
        trainer = _trainer()
        batch = _pair_batch(seed=7)
        input_ids = batch["chosen_ids"]
        attention_mask = batch["attention_mask"]
        completion_mask = batch["completion_mask"]

        got = trainer.compute_log_probs(
            trainer.policy_model, input_ids, attention_mask, completion_mask
        )

        logits = trainer.policy_model(input_ids=input_ids, attention_mask=attention_mask).logits
        token_logps = gather_logprobs(logits, input_ids)
        completion_sum = (token_logps * completion_mask[:, 1:].to(token_logps.dtype)).sum(dim=1)
        prompt_and_completion = (token_logps * attention_mask[:, 1:].to(token_logps.dtype)).sum(
            dim=1
        )
        mean_over_completion = completion_sum / completion_mask[:, 1:].sum(dim=1).clamp(min=1)

        assert not torch.allclose(completion_sum, prompt_and_completion, atol=1e-3)
        assert not torch.allclose(completion_sum, mean_over_completion, atol=1e-3)
        assert torch.allclose(got, completion_sum, atol=FP32_ATOL, rtol=0)
        assert got.shape == (input_ids.shape[0],)

    def test_tokenize_masks_the_prompt_prefix(self):
        """train_step's mask is 0 on the prompt prefix and on padding."""
        config = TrainingConfig.toy_mode()
        config.log_to_wandb = False
        config.dataset.max_seq_length = 8
        trainer = DPOTrainer(
            config,
            policy_model=tiny_causal_lm(0),
            reference_model=tiny_causal_lm(1),
            tokenizer=_CharTokenizer(),
        )
        input_ids, attention_mask, completion_mask = trainer._tokenize_for_loss("ab", "cd")

        assert input_ids.shape == (1, 4)
        assert attention_mask.tolist() == [[1, 1, 1, 1]]
        assert completion_mask.tolist() == [[0, 0, 1, 1]]
        # Counting the prompt would mark the first two real tokens as well.
        assert completion_mask.tolist() != attention_mask.tolist()

    def test_prefix_length_stops_at_the_first_difference(self):
        assert prompt_prefix_length([1, 2, 3], [1, 2, 9, 4]) == 2
        mask = completion_token_mask(
            torch.tensor([[1, 1, 1, 1, 0]]),
            torch.tensor([2]),
        )
        assert mask.tolist() == [[0, 0, 1, 1, 0]]

    def test_batch_of_four_right_pads_each_row(self):
        """Unequal strings pad on the right. Each row has its own prompt length."""
        tokenizer = _CharTokenizer()
        prompts = ["ab", "xyz", "Q", "hello"]
        chosen = ["cd", "q", "one two", "yes"]
        rejected = ["e", "a much longer answer", "n", "nope"]
        collated = collate_preference_pairs(
            tokenizer,
            prompts,
            chosen,
            rejected,
            max_length=32,
            pad_token_id=tokenizer.pad_token_id,
        )

        assert collated["chosen_ids"].shape[0] == 4
        assert collated["rejected_ids"].shape[0] == 4
        assert collated["chosen_ids"].shape[1] != collated["rejected_ids"].shape[1]
        for side in ("chosen", "rejected"):
            ids = collated[f"{side}_ids"]
            attention = collated[f"{side}_attention_mask"]
            completion = collated[f"{side}_completion_mask"]
            lengths = collated[f"{side}_prompt_lengths"]
            assert torch.equal(completion, completion_token_mask(attention, lengths))
            assert int(attention[:, 0].sum()) == 4
            for row in range(4):
                valid = int(attention[row].sum())
                assert torch.equal(attention[row, :valid], torch.ones(valid, dtype=torch.long))
                assert int(attention[row, valid:].sum()) == 0
                assert int(ids[row, valid:].sum()) == 0
                assert int(completion[row, : int(lengths[row])].sum()) == 0
                assert int(completion[row, valid:].sum()) == 0
                assert int(completion[row].sum()) > 0
        assert len(set(int(v) for v in collated["chosen_prompt_lengths"])) > 1
        assert len(set(int(v) for v in collated["rejected_prompt_lengths"])) > 1

    def test_train_step_runs_a_batch_of_four(self):
        """The string path trains four unequal pairs. The reference gets no gradient."""
        config = TrainingConfig.toy_mode()
        config.log_to_wandb = False
        config.dataset.max_seq_length = 32
        config.optimization.learning_rate = 1e-2
        policy = tiny_causal_lm(0)
        reference = tiny_causal_lm(1)
        reference_before = [parameter.detach().clone() for parameter in reference.parameters()]
        trainer = DPOTrainer(
            config,
            policy_model=policy,
            reference_model=reference,
            tokenizer=_CharTokenizer(),
        )
        metrics = trainer.train_step(
            {
                "prompt": ["ab", "xyz", "Q", "hello"],
                "chosen": ["cd", "q", "one two", "yes"],
                "rejected": ["e", "a much longer answer", "n", "nope"],
            }
        )

        assert trainer.step_count == 1
        assert metrics.accuracy in {0.0, 0.25, 0.5, 0.75, 1.0}
        assert all(
            torch.equal(before, parameter)
            for before, parameter in zip(reference_before, reference.parameters(), strict=True)
        )
        assert all(parameter.grad is None for parameter in reference.parameters())

    def test_padded_row_matches_the_row_alone(self):
        """Right padding does not change the short row's completion log-prob."""
        config = TrainingConfig.toy_mode()
        config.log_to_wandb = False
        config.dataset.max_seq_length = 32
        trainer = DPOTrainer(
            config,
            policy_model=tiny_causal_lm(0),
            reference_model=tiny_causal_lm(1),
            tokenizer=_CharTokenizer(),
        )
        prompts = ["ab", "xyz"]
        chosen = ["cd", "a much longer chosen answer"]
        collated = collate_preference_pairs(
            trainer.tokenizer,
            prompts,
            chosen,
            ["n", "z"],
            max_length=32,
            pad_token_id=0,
        )
        batched = trainer.compute_log_probs(
            trainer.policy_model,
            collated["chosen_ids"],
            collated["chosen_attention_mask"],
            collated["chosen_completion_mask"],
        )
        short = collate_preference_pairs(
            trainer.tokenizer,
            prompts[:1],
            chosen[:1],
            ["n"],
            max_length=32,
            pad_token_id=0,
        )
        alone = trainer.compute_log_probs(
            trainer.policy_model,
            short["chosen_ids"],
            short["chosen_attention_mask"],
            short["chosen_completion_mask"],
        )
        assert collated["chosen_ids"].shape[1] > short["chosen_ids"].shape[1]
        assert torch.allclose(batched[:1], alone, atol=FP32_ATOL, rtol=0)

    def test_mismatched_pair_lengths_raise(self):
        tokenizer = _CharTokenizer()
        with pytest.raises(ValueError, match="same length"):
            collate_preference_pairs(
                tokenizer,
                ["ab", "cd"],
                ["x"],
                ["y", "z"],
                max_length=8,
                pad_token_id=0,
            )


def _trl_sigmoid_loss(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Call TRL 1.17.1 ``DPOTrainer._compute_loss`` on one concatenated batch.

    ``__init__`` is not used. It constructs a Hugging Face Trainer and a
    tokenizer, and with no reference model it loads one from the config
    name. ``__new__`` plus the attributes the method reads is enough to run
    the sigmoid loss on these tensors.

    Chosen and rejected masks are concatenated in that order. They are not
    the same tensor repeated.
    """
    add_fused_lm_head(policy, outputs=("log_probs", "entropy", "mean_logits", "is_top1"))
    add_fused_lm_head(reference, outputs=("log_probs", "entropy", "mean_logits", "is_top1"))

    class _Accel:
        device = torch.device("cpu")

        def gather(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

    trainer = TrlDPOTrainer.__new__(TrlDPOTrainer)
    trainer.model = policy
    trainer.ref_model = reference
    trainer.accelerator = _Accel()
    trainer.ld_alpha = None
    trainer.precompute_ref_logps = False
    trainer.f_divergence_type = "reverse_kl"
    trainer.loss_types = ["sigmoid"]
    trainer.loss_weights = [1.0]
    trainer.beta = beta
    trainer.use_weighting = False
    trainer.aux_loss_enabled = False
    trainer.label_smoothing = 0.0
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._total_train_tokens = 0
    trainer._tp_size = 1
    trainer.args = SimpleNamespace(gradient_checkpointing_kwargs=None)

    inputs = {
        "input_ids": torch.cat([batch["chosen_ids"], batch["rejected_ids"]], dim=0),
        "attention_mask": torch.cat(
            [batch["chosen_attention_mask"], batch["rejected_attention_mask"]],
            dim=0,
        ),
        "completion_mask": torch.cat(
            [batch["chosen_completion_mask"], batch["rejected_completion_mask"]],
            dim=0,
        ),
    }
    loss = trainer._compute_loss(policy, inputs, return_outputs=False)
    logged = {key: values[-1] for key, values in trainer._metrics["train"].items()}
    return loss, logged


def _weights_match(left: torch.nn.Module, right: torch.nn.Module) -> None:
    pairs = zip(left.parameters(), right.parameters(), strict=True)
    assert all(torch.equal(a, b) for a, b in pairs)


class TestDPOParity:
    """Loss, rewards, margin, accuracy, and policy gradients against TRL 1.17.1."""

    def test_trl_pin(self) -> None:
        import trl

        assert trl.__version__ == "1.17.1"

    @pytest.mark.parametrize("layout", ["matched", "unequal", "unequal4"])
    @pytest.mark.parametrize("seed", range(6))
    def test_loss_rewards_margin_match_trl(self, seed: int, layout: str) -> None:
        beta = 0.2
        trainer = _trainer(policy_seed=seed, reference_seed=seed + 10, beta=beta)
        batch = _preference_batch(layout, seed)
        if layout in {"unequal", "unequal4"}:
            _assert_sides_differ(batch)
        if layout == "unequal4":
            assert batch["chosen_ids"].shape == (4, 10)
            _assert_prompt_lengths_differ(batch["chosen_completion_mask"])
            _assert_prompt_lengths_differ(batch["rejected_completion_mask"])
        policy = trainer.policy_model
        reference = trainer.reference_model
        policy.train()

        policy_chosen, policy_rejected = _side_logprobs(trainer, policy, batch)
        with torch.no_grad():
            ref_chosen, ref_rejected = _side_logprobs(trainer, reference, batch)

        loss, metrics = trainer.compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected
        )
        # Paper equation, written here. It does not call rlpost or TRL.
        chosen_reward = beta * (policy_chosen - ref_chosen)
        rejected_reward = beta * (policy_rejected - ref_rejected)
        paper_loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
        assert torch.allclose(loss, paper_loss, atol=FP32_ATOL, rtol=0)

        trl_loss, logged = _trl_sigmoid_loss(policy, reference, batch, beta)
        assert torch.allclose(loss, trl_loss, atol=FP32_ATOL, rtol=0)
        assert metrics.margin_mean == pytest.approx(logged["rewards/margins"], abs=FP32_ATOL)
        assert metrics.accuracy == pytest.approx(logged["rewards/accuracies"], abs=FP32_ATOL)
        assert policy_chosen.detach().mean().item() == pytest.approx(
            logged["logps/chosen"], abs=SUM_MEAN_ATOL
        )
        assert policy_rejected.detach().mean().item() == pytest.approx(
            logged["logps/rejected"], abs=SUM_MEAN_ATOL
        )
        assert chosen_reward.detach().mean().item() == pytest.approx(
            logged["rewards/chosen"], abs=FP32_ATOL
        )
        assert rejected_reward.detach().mean().item() == pytest.approx(
            logged["rewards/rejected"], abs=FP32_ATOL
        )

    @pytest.mark.parametrize("layout", ["matched", "unequal", "unequal4"])
    @pytest.mark.parametrize("seed", range(6))
    def test_policy_gradients_match_trl(self, seed: int, layout: str) -> None:
        """Backward both losses from the same weights. Reference grads stay None."""
        beta = 0.2
        batch = _preference_batch(layout, seed)
        if layout in {"unequal", "unequal4"}:
            _assert_sides_differ(batch)
        if layout == "unequal4":
            assert batch["chosen_ids"].shape[0] == 4

        policy = tiny_causal_lm(seed)
        policy_trl = tiny_causal_lm(seed)
        reference = tiny_causal_lm(seed + 10)
        reference_trl = tiny_causal_lm(seed + 10)
        _weights_match(policy, policy_trl)
        _weights_match(reference, reference_trl)

        config = TrainingConfig.toy_mode()
        config.log_to_wandb = False
        config.alignment.dpo_beta = beta
        trainer = DPOTrainer(config, policy_model=policy, reference_model=reference)
        reference_trl.requires_grad_(False)
        reference_trl.eval()
        policy.train()
        policy_trl.train()

        policy_chosen, policy_rejected = _side_logprobs(trainer, policy, batch)
        with torch.no_grad():
            ref_chosen, ref_rejected = _side_logprobs(trainer, reference, batch)
        loss, _ = trainer.compute_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        loss.backward()

        trl_loss, _ = _trl_sigmoid_loss(policy_trl, reference_trl, batch, beta)
        trl_loss.backward()

        compared = 0
        for ours, theirs in zip(policy.parameters(), policy_trl.parameters(), strict=True):
            assert ours.grad is not None
            assert theirs.grad is not None
            assert torch.allclose(ours.grad, theirs.grad, atol=GRAD_ATOL, rtol=0)
            compared += 1
        assert compared > 0
        assert any(torch.count_nonzero(p.grad) > 0 for p in policy.parameters())
        assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
        assert all(not p.requires_grad and p.grad is None for p in reference_trl.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
