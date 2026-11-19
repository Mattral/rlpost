"""Completion-token NLL for ``train-sft --tiny``.

Logits at index ``t`` score the token at ``t + 1``. The loss keeps
``completion_mask[:, 1:]``. A prompt token is not a target. The logit
on the last prompt position is kept: it predicts the first completion
token.

N is 8. The seed is 7. The batch size is 4. The check is the
completion-token NLL of every tiny row after 8 updates, and it is
strictly less than that NLL before any update.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from rlpost.dpo_engine import collate_completions
from rlpost.sft_engine import (
    TINY_SFT_ROWS,
    completion_cross_entropy,
    run_sft_training,
    tiny_sft_config,
)
from rlpost.tiny_lm import TinyCharTokenizer, tiny_causal_lm

N_STEPS = 8
SEED = 7
BATCH_SIZE = 4


def test_importing_the_engines_does_not_import_peft() -> None:
    import rlpost.reward_engine
    import rlpost.sft_engine

    del rlpost.reward_engine, rlpost.sft_engine
    assert "peft" not in sys.modules


def _targets(
    batch: dict[str, torch.Tensor],
) -> tuple[
    list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]
]:
    """Logit positions ``(row, t)`` whose target is a prompt, completion, or pad.

    The fourth list is the last prompt position of each row, which
    predicts the first completion token.
    """
    ids = batch["input_ids"]
    attention = batch["attention_mask"]
    prompt_lengths = batch["prompt_lengths"]
    prompt: list[tuple[int, int]] = []
    completion: list[tuple[int, int]] = []
    pad: list[tuple[int, int]] = []
    boundary: list[tuple[int, int]] = []
    for row in range(ids.shape[0]):
        prompt_length = int(prompt_lengths[row])
        for time in range(ids.shape[1] - 1):
            target = time + 1
            if int(attention[row, target]) != 1:
                pad.append((row, time))
            elif target < prompt_length:
                prompt.append((row, time))
            else:
                completion.append((row, time))
                if time == prompt_length - 1:
                    boundary.append((row, time))
    return prompt, completion, pad, boundary


def _bump(
    logits: torch.Tensor,
    positions: list[tuple[int, int]],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Add 20 to the gathered class at ``positions``. Those logits score token ``t + 1``."""
    bumped = logits.clone()
    for row, time in positions:
        target = int(input_ids[row, time + 1])
        bumped[row, time, target] = bumped[row, time, target] + 20
    return bumped


class TestCompletionMask:
    def test_prompt_tokens_are_not_targets(self) -> None:
        """Bumping a prompt target leaves the loss unchanged.

        Bumping a completion target, including the logit on the last
        prompt position, changes it. Padding is not a target. This
        fails if the loss counts prompt tokens.
        """
        batch = collate_completions(
            TinyCharTokenizer(),
            [row["prompt"] for row in TINY_SFT_ROWS],
            [row["completion"] for row in TINY_SFT_ROWS],
            max_length=32,
            pad_token_id=0,
        )
        prompt, completion, pad, boundary = _targets(batch)
        assert prompt
        assert completion
        assert pad
        assert len(boundary) == len(TINY_SFT_ROWS)
        torch.manual_seed(0)
        logits = torch.randn(
            batch["input_ids"].shape[0],
            batch["input_ids"].shape[1],
            128,
        )
        base = completion_cross_entropy(logits, batch["input_ids"], batch["completion_mask"])
        assert torch.equal(
            completion_cross_entropy(
                _bump(logits, prompt, batch["input_ids"]),
                batch["input_ids"],
                batch["completion_mask"],
            ),
            base,
        )
        assert torch.equal(
            completion_cross_entropy(
                _bump(logits, pad, batch["input_ids"]),
                batch["input_ids"],
                batch["completion_mask"],
            ),
            base,
        )
        assert (
            completion_cross_entropy(
                _bump(logits, completion, batch["input_ids"]),
                batch["input_ids"],
                batch["completion_mask"],
            )
            < base
        )
        assert (
            completion_cross_entropy(
                _bump(logits, boundary, batch["input_ids"]),
                batch["input_ids"],
                batch["completion_mask"],
            )
            < base
        )

    def test_prompt_length_one_keeps_the_first_logit(self) -> None:
        """A one-token prompt has no prompt target. Logit 0 predicts the completion.

        Dropping that logit, which is what an unshifted prompt mask does,
        leaves this bump with no effect.
        """
        batch = collate_completions(
            TinyCharTokenizer(),
            ["A"],
            ["bc"],
            max_length=32,
            pad_token_id=0,
        )
        assert int(batch["prompt_lengths"][0]) == 1
        prompt, completion, _pad, boundary = _targets(batch)
        assert prompt == []
        assert boundary == [(0, 0)]
        assert (0, 0) in completion
        torch.manual_seed(1)
        logits = torch.randn(1, batch["input_ids"].shape[1], 128)
        base = completion_cross_entropy(logits, batch["input_ids"], batch["completion_mask"])
        assert (
            completion_cross_entropy(
                _bump(logits, boundary, batch["input_ids"]),
                batch["input_ids"],
                batch["completion_mask"],
            )
            < base
        )

    def test_mean_of_one_uniform_token_is_log_vocab(self) -> None:
        """Two kept tokens with uniform logits average to ``log(V)``, not twice that."""
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
        logits = torch.zeros(1, 3, 4)
        loss = completion_cross_entropy(logits, ids, mask)
        assert float(loss) == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-6)

    def test_int32_ids_match_int64(self) -> None:
        ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
        logits = torch.zeros(1, 3, 4)
        assert torch.equal(
            completion_cross_entropy(logits, ids.to(torch.int32), mask),
            completion_cross_entropy(logits, ids, mask),
        )

    def test_empty_mask_and_short_sequence_raise(self) -> None:
        ids = torch.tensor([[1, 2, 0]], dtype=torch.long)
        logits = torch.zeros(1, 3, 4)
        with pytest.raises(ValueError, match="no tokens"):
            completion_cross_entropy(logits, ids, torch.zeros(1, 3, dtype=torch.long))
        with pytest.raises(ValueError, match="at least 2"):
            completion_cross_entropy(
                torch.zeros(1, 1, 4),
                torch.zeros(1, 1, dtype=torch.long),
                torch.ones(1, 1, dtype=torch.long),
            )
        with pytest.raises(TypeError, match="int32 or int64"):
            completion_cross_entropy(logits, ids.to(torch.float32), torch.tensor([[0, 1, 0]]))


class TestTinySftDecreases:
    def test_full_set_loss_decreases_over_n_steps(self, tmp_path: Path) -> None:
        """N is 8. Full-set completion NLL after those updates is lower."""
        config = tiny_sft_config(
            output_dir=str(tmp_path),
            seed=SEED,
            batch_size=BATCH_SIZE,
            checkpoint_path=None,
        )
        model = tiny_causal_lm(SEED)
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        optimizer = AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        result = run_sft_training(
            model,
            optimizer,
            config,
            N_STEPS,
            TINY_SFT_ROWS,
            TinyCharTokenizer(),
        )
        assert len(result.losses) == N_STEPS
        assert result.data_loss_after < result.data_loss_before
        assert any(not torch.equal(initial[key], model.state_dict()[key]) for key in initial)
