"""Bradley-Terry loss for ``train-reward --tiny``.

The loss is the mean of ``-log sigmoid(r_chosen - r_rejected)``. Equal
scores are ``ln 2``. The closed form uses ``math``:
``log1p(exp(-x))`` when ``x >= 0``, and ``-x + log1p(exp(x))`` otherwise.

N is 8. The seed is 7. The batch size is 4. After 8 updates every tiny
pair has ``r_chosen > r_rejected``, and the mean margin is greater than
it was before any update.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from rlpost.reward_engine import (
    TINY_REWARD_PAIRS,
    bradley_terry_loss,
    run_reward_training,
    score_preference_pairs,
    tiny_reward_config,
)
from rlpost.tiny_lm import TinyCharTokenizer, tiny_reward_model

N_STEPS = 8
SEED = 7
BATCH_SIZE = 4


def _neg_log_sigmoid(value: float) -> float:
    """``-log sigmoid(value)`` with ``math``, not ``torch.nn.functional``."""
    if value >= 0.0:
        return math.log1p(math.exp(-value))
    return -value + math.log1p(math.exp(value))


class TestBradleyTerry:
    def test_closed_form(self) -> None:
        chosen = torch.tensor([1.5, 0.0, -2.25, 4.0, -0.5])
        rejected = torch.tensor([0.0, 0.0, 1.0, 4.0, 1.5])
        diffs = (chosen - rejected).tolist()
        expected = sum(_neg_log_sigmoid(float(diff)) for diff in diffs) / len(diffs)
        loss = bradley_terry_loss(chosen, rejected)
        assert float(loss) == pytest.approx(expected, abs=1e-6)
        assert float(bradley_terry_loss(torch.zeros(4), torch.zeros(4))) == pytest.approx(
            math.log(2), abs=1e-6
        )

    def test_strict_win_fraction_is_half_this_batch(self) -> None:
        """One of two rows has chosen above rejected. The metric uses strict ``>``."""
        chosen = torch.tensor([1.0, 2.0])
        rejected = torch.tensor([0.0, 3.0])
        loss = bradley_terry_loss(chosen, rejected)
        accuracy = (chosen > rejected).float().mean()
        expected = (_neg_log_sigmoid(1.0) + _neg_log_sigmoid(-1.0)) / 2
        assert float(loss) == pytest.approx(expected, abs=1e-6)
        assert float(accuracy) == pytest.approx(0.5)

    def test_empty_and_wrong_rank_raise(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            bradley_terry_loss(torch.zeros(0), torch.zeros(0))
        with pytest.raises(ValueError, match="non-empty"):
            bradley_terry_loss(torch.zeros(2, 1), torch.zeros(2, 1))
        with pytest.raises(ValueError, match="share a shape"):
            bradley_terry_loss(torch.zeros(2), torch.zeros(3))


class TestTinyRewardSeparates:
    def test_chosen_scores_rise_above_rejected(self, tmp_path: Path) -> None:
        """N is 8. Every pair then has chosen above rejected, and the margin rises."""
        config = tiny_reward_config(
            output_dir=str(tmp_path),
            seed=SEED,
            batch_size=BATCH_SIZE,
            checkpoint_path=None,
        )
        model = tiny_reward_model(SEED)
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        optimizer = AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        result = run_reward_training(
            model,
            optimizer,
            config,
            N_STEPS,
            TINY_REWARD_PAIRS,
            TinyCharTokenizer(),
        )
        chosen, rejected = score_preference_pairs(
            model,
            TINY_REWARD_PAIRS,
            TinyCharTokenizer(),
            max_length=config.dataset.max_seq_length,
            pad_token_id=0,
        )
        assert len(result.losses) == N_STEPS
        assert result.margin_after > result.margin_before
        assert torch.all(chosen > rejected)
        assert any(not torch.equal(initial[key], model.state_dict()[key]) for key in initial)
