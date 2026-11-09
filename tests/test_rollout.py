"""Per-row EOS stopping in a mixed rollout batch.

TRL 1.17.1 has no PPO trainer and no GAE helper. The missing-EOS
penalty matches ``OnlineDPOConfig.missing_eos_penalty`` (default
``None``) in ``experimental/online_dpo/online_dpo_config.py``.
``OnlineDPOTrainer.training_step`` does
``rewards[~contain_eos_token] -= missing_eos_penalty`` when the field
is set (``experimental/online_dpo/online_dpo_trainer.py``). The
docstring says the penalty must be positive. That release does not
check the sign. GRPO and RLOO ``mask_truncated_completions`` drop a
truncated completion from the loss. That is not this penalty.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rlpost.alignment.rollout import (
    RolloutGenerator,
)
from rlpost.ops import response_token_mask
from rlpost.ppo_engine import GeneralizedAdvantageEstimation

EOS = 2
PAD = 0
VOCAB = 16


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.head = nn.Linear(32, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        batch_size, seq_len = input_ids.shape
        features = torch.randn(batch_size, seq_len, 32, device=input_ids.device)
        logits = self.head(features)
        return SimpleNamespace(logits=logits)


class DummyRewardModel(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        batch_size = input_ids.shape[0]
        return SimpleNamespace(logits=torch.randn(batch_size, 1, device=input_ids.device))


def test_rollout_generator_creates_valid_rollouts():
    policy_model = DummyCausalLM(vocab_size=16)
    reference_model = DummyCausalLM(vocab_size=16)
    reward_model = DummyRewardModel()
    tokenizer = None

    generator = RolloutGenerator(
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        max_response_length=3,
        temperature=1.0,
        top_p=0.9,
    )

    query_batch = torch.randint(low=0, high=16, size=(2, 4))
    generated = generator.generate_rollout(query_batch)
    rollouts = generated.rollouts

    assert generated.padding_side == "mixed"
    assert generated.prompt_width == 4
    assert generated.response_lengths.tolist() == [3, 3]
    assert generated.dones.sum() == 0
    assert isinstance(rollouts, list)
    assert len(rollouts) == 2
    assert rollouts[0].response_tokens.shape == (3,)
    assert rollouts[0].logits_policy.shape == (3, 16)
    assert rollouts[0].logits_reference.shape == (3, 16)
    assert rollouts[0].reward.shape == () or rollouts[0].reward.shape == (1,)


def test_rank3_reward_is_the_last_token_not_the_time_mean():
    """No EOS id is set, so the row runs to the end and the last index is the response."""

    class Rank3Reward(nn.Module):
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            del attention_mask, position_ids
            batch, length = input_ids.shape
            scores = torch.arange(length, dtype=torch.float32).view(1, length, 1)
            return SimpleNamespace(logits=scores.expand(batch, length, 1).clone())

    generator = RolloutGenerator(
        policy_model=DummyCausalLM(vocab_size=16),
        reference_model=DummyCausalLM(vocab_size=16),
        reward_model=Rank3Reward(),
        tokenizer=None,
        max_response_length=3,
        temperature=1.0,
        top_p=1.0,
    )
    query = torch.randint(low=0, high=16, size=(2, 4))
    rollouts = generator.generate_rollout(query).rollouts

    # cat(query, response) has length 7. The time mean of 0..6 is 3.
    # No EOS id, so nothing stops early and the tail is the response.
    assert rollouts[0].reward.item() == pytest.approx(6.0)
    assert rollouts[0].reward.item() != pytest.approx(3.0)
    assert rollouts[1].reward.shape == ()


class ScriptedLM(nn.Module):
    """One predetermined token per row at each step. The chosen logit is 0."""

    def __init__(self, plan: torch.Tensor, vocab: int = VOCAB) -> None:
        super().__init__()
        self.plan = plan
        self.vocab = vocab
        self.step = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocab), -1.0e9)
        choice = self.plan[:, self.step]
        logits[torch.arange(batch), -1, choice] = 0.0
        self.step += 1
        return SimpleNamespace(logits=logits)


class ZeroLM(nn.Module):
    def __init__(self, vocab: int = VOCAB) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, self.vocab))


class IndexReward(nn.Module):
    """Per-token score is the column index. The sequence score is that index."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        _, length = input_ids.shape
        scores = torch.arange(length, dtype=torch.float32)
        return SimpleNamespace(logits=scores.expand(input_ids.shape[0], length).clone())


def _generate(
    plan: torch.Tensor,
    query: torch.Tensor,
    prompt_lengths: torch.Tensor,
    penalty: float | None = None,
):
    generator = RolloutGenerator(
        policy_model=ScriptedLM(plan),
        reference_model=ZeroLM(),
        reward_model=IndexReward(),
        tokenizer=None,
        max_response_length=int(plan.shape[1]),
        temperature=1.0,
        top_p=1.0,
        eos_token_id=EOS,
        pad_token_id=PAD,
        missing_eos_penalty=penalty,
    )
    return generator.generate_rollout(query, prompt_lengths)


def test_each_row_stops_at_its_own_eos_in_the_mixed_layout() -> None:
    """EOS counts. Tokens after it are padding. The response starts at prompt_width.

    ``prompt_width`` is the query tensor's width, 4. The longest real
    prompt is 3. Starting the response at that 3 would mark a prompt
    column as a response token. The EOS token is a done. The truncated
    row is 0 on every column.
    """
    query = torch.tensor(
        [
            [0, 0, 11, 12],
            [0, 13, 14, 15],
        ],
        dtype=torch.long,
    )
    prompt_lengths = torch.tensor([2, 3], dtype=torch.long)
    plan = torch.tensor(
        [
            [8, EOS, 9],
            [8, 8, 8],
        ],
        dtype=torch.long,
    )
    generated = _generate(plan, query, prompt_lengths)

    assert generated.padding_side == "mixed"
    assert generated.prompt_width == 4
    assert int(generated.prompt_lengths.max()) == 3
    assert generated.prompt_width != int(generated.prompt_lengths.max())
    assert generated.response_lengths.tolist() == [2, 3]
    assert generated.input_ids.shape == (2, 7)
    assert generated.input_ids[0].tolist() == [0, 0, 11, 12, 8, EOS, PAD]
    assert generated.input_ids[1].tolist() == [0, 13, 14, 15, 8, 8, 8]
    assert 9 not in generated.input_ids[0].tolist()
    assert generated.dones[0].tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert generated.dones[1].sum() == 0

    expected, _ = response_token_mask(
        generated.prompt_lengths,
        generated.response_lengths,
        generated.input_ids.shape[1],
        "mixed",
        prompt_width=generated.prompt_width,
    )
    assert torch.equal(generated.token_mask, expected)
    assert generated.token_mask[0, :4].sum() == 0
    assert generated.token_mask[0, 3] == 0
    assert generated.token_mask[0, 4:7].tolist() == [1, 1, 0]
    assert generated.token_mask[1, 4:].tolist() == [1, 1, 1]
    started_at_longest_prompt, _ = response_token_mask(
        prompt_lengths,
        generated.response_lengths,
        7,
        "mixed",
        prompt_width=int(prompt_lengths.max()),
    )
    assert not torch.equal(started_at_longest_prompt, generated.token_mask)

    assert generated.rollouts[0].response_tokens.tolist() == [8, EOS]
    assert generated.rollouts[1].response_tokens.tolist() == [8, 8, 8]
    assert generated.rollouts[0].logits_policy.shape == (2, VOCAB)
    assert generated.rollouts[1].logits_policy.shape == (3, VOCAB)


def test_sequence_reward_sits_on_eos_and_on_the_truncated_token() -> None:
    """One batch: the EOS row is scored on the EOS token, the truncated row on its last token.

    The column index is the per-token score. The EOS row's last real
    token is column 5, not the pad at column 6. The truncated row's
    last real token is column 6. A time mean, or a score on the final
    column of both rows, is a different reward.
    """
    query = torch.tensor(
        [
            [0, 0, 11, 12],
            [0, 13, 14, 15],
        ],
        dtype=torch.long,
    )
    prompt_lengths = torch.tensor([2, 3], dtype=torch.long)
    plan = torch.tensor(
        [
            [8, EOS, 9],
            [8, 8, 8],
        ],
        dtype=torch.long,
    )
    generated = _generate(plan, query, prompt_lengths)

    assert generated.rewards.shape == generated.input_ids.shape
    assert generated.rewards[0, 5] == pytest.approx(5.0)
    assert torch.count_nonzero(generated.rewards[0]) == 1
    assert generated.rewards[0, 6] == 0
    assert generated.rewards[1, 6] == pytest.approx(6.0)
    assert torch.count_nonzero(generated.rewards[1]) == 1
    assert generated.rollouts[0].reward.item() == pytest.approx(5.0)
    assert generated.rollouts[1].reward.item() == pytest.approx(6.0)
    # The time mean of columns 0..6 is 3. Neither sequence score is that.
    assert generated.rollouts[0].reward.item() != pytest.approx(3.0)
    assert generated.rollouts[1].reward.item() != pytest.approx(3.0)


def test_truncation_bootstraps_and_eos_does_not() -> None:
    """A full-length row without EOS is not terminal. An EOS on that same column is.

    Both rows fill ``max_response_length``, so the mask cannot tell them
    apart. ``dones`` does. gamma is 0.5 and lambda is 0, rewards are 0,
    and ``V`` after the last token is 4, so the truncated advantage is
    ``0.5 * 4 = 2``. Marking that token done drops it to 0. The EOS row
    stays at 0.
    """
    query = torch.tensor(
        [
            [0, 0, 11, 12],
            [0, 13, 14, 15],
        ],
        dtype=torch.long,
    )
    prompt_lengths = torch.tensor([2, 3], dtype=torch.long)
    plan = torch.tensor(
        [
            [8, 8, EOS],
            [8, 8, 8],
        ],
        dtype=torch.long,
    )
    generated = _generate(plan, query, prompt_lengths)
    seq_len = generated.input_ids.shape[1]
    assert generated.response_lengths.tolist() == [3, 3]
    assert generated.input_ids[0, -1].item() == EOS
    assert generated.input_ids[1, -1].item() == 8
    assert generated.dones[0, -1].item() == 1
    assert generated.dones[0, :-1].sum() == 0
    assert generated.dones[1].sum() == 0

    values = torch.zeros(2, seq_len)
    values[:, -1] = 4.0
    gathered = generated.token_mask[:, 1:]
    advantages, _returns = GeneralizedAdvantageEstimation(0.5, 0.0).compute(
        torch.zeros(2, seq_len - 1),
        values[:, :-1],
        values[:, 1:],
        generated.dones[:, 1:],
        gathered,
    )
    last_action = seq_len - 2
    assert advantages[1, last_action].item() == pytest.approx(2.0)
    assert advantages[0, last_action].item() == pytest.approx(0.0)

    forced_terminal = generated.dones.clone()
    forced_terminal[1, -1] = 1
    stopped, _returns = GeneralizedAdvantageEstimation(0.5, 0.0).compute(
        torch.zeros(2, seq_len - 1),
        values[:, :-1],
        values[:, 1:],
        forced_terminal[:, 1:],
        gathered,
    )
    assert stopped[1, last_action].item() == pytest.approx(0.0)


def test_missing_eos_penalty_hits_only_the_truncated_row() -> None:
    """``None`` leaves the score. A set penalty is subtracted from rows with no EOS.

    The EOS row's score stays the column index 5. The truncated row's
    score is 6 minus the penalty. This is the Online DPO subtraction,
    applied to the sequence score before it is written onto the token.
    """
    from rlpost.config import AlignmentConfig

    query = torch.tensor(
        [
            [0, 0, 11, 12],
            [0, 13, 14, 15],
        ],
        dtype=torch.long,
    )
    prompt_lengths = torch.tensor([2, 3], dtype=torch.long)
    plan = torch.tensor(
        [
            [8, EOS, 9],
            [8, 8, 8],
        ],
        dtype=torch.long,
    )
    plain = _generate(plan, query, prompt_lengths, penalty=None)
    assert plain.rollouts[0].reward.item() == pytest.approx(5.0)
    assert plain.rollouts[1].reward.item() == pytest.approx(6.0)

    penalty = AlignmentConfig(missing_eos_penalty=1.5).missing_eos_penalty
    penalized = _generate(plan, query, prompt_lengths, penalty=penalty)
    assert penalized.rewards[0, 5] == pytest.approx(5.0)
    assert torch.count_nonzero(penalized.rewards[0]) == 1
    assert penalized.rewards[1, 6] == pytest.approx(4.5)
    assert torch.count_nonzero(penalized.rewards[1]) == 1
    assert penalized.rollouts[0].reward.item() == pytest.approx(5.0)
    assert penalized.rollouts[1].reward.item() == pytest.approx(4.5)
