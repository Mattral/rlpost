"""Response-token mask for PPO.

The mask is built once from the rollout lengths. Index ``t`` of the gathered
mask is 1 when ``input_ids[:, t + 1]`` is a generated response token. Prompt
and padding are 0. These tests fail if a loss, the KL, or the advantage
stats still read those positions, and if a rank-3 reward is averaged over
time instead of landing on the last response token.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rlpost.alignment.rollout import (
    RolloutGenerator,
    last_response_index,
    place_sequence_reward,
)
from rlpost.config import TrainingConfig
from rlpost.ops import gather_logprobs, response_token_mask
from rlpost.ppo_engine import PPOTrainer, masked_whiten

PROMPT = torch.tensor([2, 1], dtype=torch.long)
RESPONSE = torch.tensor([3, 2], dtype=torch.long)
SEQ_LEN = 6
VOCAB = 6

# Right padding: [prompt | response | pad]. Left: [pad | prompt | response].
RIGHT_TOKEN = torch.tensor(
    [
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ]
)
LEFT_TOKEN = torch.tensor(
    [
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]
)
RIGHT_GATHERED = torch.tensor(
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
    ]
)
LEFT_GATHERED = torch.tensor(
    [
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0],
    ]
)


class _FixedLogits(nn.Module):
    """Logits do not depend on the token ids. The gather still uses the ids."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def forward(self, input_ids: torch.Tensor):
        del input_ids
        return SimpleNamespace(logits=self.logits)


class _FixedValue(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.values = values.clone()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        return self.values + self.bias


def _expected(padding_side: str) -> tuple[torch.Tensor, torch.Tensor]:
    if padding_side == "right":
        return RIGHT_TOKEN, RIGHT_GATHERED
    return LEFT_TOKEN, LEFT_GATHERED


def _ids(token_mask: torch.Tensor, prompt_id: int, pad_id: int) -> torch.Tensor:
    """Response tokens are 1. Prompt sits immediately before them. The rest is pad.

    The columns come from the mask the test just checked against the layout
    table, so a wrong helper fails before this function runs.
    """
    ids = torch.full(token_mask.shape, pad_id, dtype=torch.long)
    for row in range(token_mask.shape[0]):
        response_at = (token_mask[row] == 1).nonzero().flatten()
        start = int(response_at[0])
        stop = int(response_at[-1]) + 1
        prompt_len = int(PROMPT[row])
        ids[row, start - prompt_len : start] = prompt_id
        ids[row, start:stop] = 1
    return ids


def _run(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    value_pred: torch.Tensor,
    batch: dict,
    old_log_probs: torch.Tensor,
) -> object:
    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    policy = _FixedLogits(policy_logits)
    reference = _FixedLogits(ref_logits)
    value = _FixedValue(value_pred)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=1e-3)
    trainer = PPOTrainer(config, policy, value, reference, optimizer)
    return trainer.train_step(batch, old_log_probs)


def _batch(
    input_ids: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    padding_side: str,
) -> dict:
    return {
        "input_ids": input_ids,
        "rewards": rewards,
        "values": values,
        "next_values": next_values,
        "dones": torch.zeros(input_ids.shape),
        "prompt_lengths": PROMPT,
        "response_lengths": RESPONSE,
        "padding_side": padding_side,
    }


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_gathered_mask_matches_the_rollout_layout(padding_side: str) -> None:
    """Index t of the gathered mask marks token t+1, and only response tokens."""
    token_mask, gathered = response_token_mask(PROMPT, RESPONSE, SEQ_LEN, padding_side)
    expected_token, expected_gathered = _expected(padding_side)
    assert torch.equal(token_mask, expected_token)
    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(gathered, token_mask[:, 1:])


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_prompt_and_padding_do_not_change_losses_advantages_or_kl(padding_side: str) -> None:
    """Corrupting prompt tokens or padding leaves the PPO step unchanged.

    A response-scoring logit and the last-token reward do change it. An
    unmasked mean fails the first check. An empty mask fails the second.
    """
    token_mask, gathered = response_token_mask(PROMPT, RESPONSE, SEQ_LEN, padding_side)
    expected_token, expected_gathered = _expected(padding_side)
    assert torch.equal(token_mask, expected_token)
    assert torch.equal(gathered, expected_gathered)

    base_ids = _ids(token_mask, prompt_id=2, pad_id=0)
    corrupt_ids = _ids(token_mask, prompt_id=3, pad_id=4)
    assert not torch.equal(base_ids, corrupt_ids)
    assert torch.equal(base_ids[token_mask.bool()], corrupt_ids[token_mask.bool()])

    policy_logits = torch.arange(2 * SEQ_LEN * VOCAB, dtype=torch.float32).reshape(
        2, SEQ_LEN, VOCAB
    )
    policy_logits = policy_logits / 10
    ref_logits = -policy_logits

    outside = gathered == 0
    corrupt_policy = policy_logits.clone()
    corrupt_ref = ref_logits.clone()
    corrupt_policy[:, :-1, 0] = corrupt_policy[:, :-1, 0] + 5 * outside.to(corrupt_policy.dtype)
    corrupt_ref[:, :-1, 0] = corrupt_ref[:, :-1, 0] + 5 * outside.to(corrupt_ref.dtype)
    # The final logit scores no token in the gather. An unmasked entropy mean
    # over [B, T, V] still reads it.
    corrupt_policy[:, -1, 0] = corrupt_policy[:, -1, 0] + 5
    corrupt_ref[:, -1, 0] = corrupt_ref[:, -1, 0] + 5

    rewards = place_sequence_reward(torch.tensor([2.0, 2.0]), token_mask)
    corrupt_rewards = rewards.clone()
    corrupt_rewards[token_mask == 0] = 50

    values = torch.full((2, SEQ_LEN), 0.5)
    values[token_mask == 0] = 0
    corrupt_values = values.clone()
    corrupt_values[token_mask == 0] = 80

    next_values = torch.zeros(2, SEQ_LEN)
    corrupt_next = next_values.clone()
    corrupt_next[token_mask == 0] = 7

    value_pred = torch.full((2, SEQ_LEN), 0.25)
    corrupt_pred = value_pred.clone()
    corrupt_pred[token_mask == 0] = 9

    offset = torch.linspace(-0.3, 0.3, SEQ_LEN - 1).view(1, -1)
    old_log_probs = gather_logprobs(policy_logits, base_ids).detach() - offset

    base = _run(
        policy_logits,
        ref_logits,
        value_pred,
        _batch(base_ids, rewards, values, next_values, padding_side),
        old_log_probs,
    )
    corrupt = _run(
        corrupt_policy,
        corrupt_ref,
        corrupt_pred,
        _batch(corrupt_ids, corrupt_rewards, corrupt_values, corrupt_next, padding_side),
        old_log_probs,
    )

    assert base.kl_divergence > 0
    assert base.entropy > 0
    assert base.advantage_mean != pytest.approx(0.0, abs=1e-6)
    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "kl_divergence",
        "advantage_mean",
        "gae_mean",
        "reward_mean",
    ):
        assert getattr(corrupt, name) == pytest.approx(getattr(base, name), abs=1e-5), name

    # A response-scoring row is inside the mask, so this must move the KL.
    touched = policy_logits.clone()
    row, time = (gathered == 1).nonzero()[0].tolist()
    touched[row, time, 1] = touched[row, time, 1] + 5
    moved = _run(
        touched,
        ref_logits,
        value_pred,
        _batch(base_ids, rewards, values, next_values, padding_side),
        old_log_probs,
    )
    assert moved.kl_divergence != pytest.approx(base.kl_divergence, abs=1e-4)
    assert moved.policy_loss != pytest.approx(base.policy_loss, abs=1e-4)
    assert moved.entropy != pytest.approx(base.entropy, abs=1e-4)

    rewarded = rewards.clone()
    rewarded[0, int(last_response_index(token_mask)[0])] += 5
    shifted = _run(
        policy_logits,
        ref_logits,
        value_pred,
        _batch(base_ids, rewarded, values, next_values, padding_side),
        old_log_probs,
    )
    assert shifted.advantage_mean != pytest.approx(base.advantage_mean, abs=1e-4)


def test_advantage_mean_uses_response_tokens_only() -> None:
    """gamma 0 and V 0 make A = r. The mean divides by the response count.

    Two rewards of 4 over 5 response tokens is 1.6. The full [2, 6] tensor
    would divide by 12 and report 2/3. Prompt zeros would dilute it too.
    """
    token_mask, _ = response_token_mask(PROMPT, RESPONSE, SEQ_LEN, "right")
    rewards = place_sequence_reward(torch.tensor([4.0, 4.0]), token_mask)
    ids = torch.ones(2, SEQ_LEN, dtype=torch.long)
    zeros = torch.zeros(2, SEQ_LEN)
    config_logits = torch.zeros(2, SEQ_LEN, VOCAB)
    old = torch.zeros(2, SEQ_LEN - 1)

    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    config.alignment.gamma = 0.0
    config.alignment.advantage_normalization = True
    policy = _FixedLogits(config_logits)
    reference = _FixedLogits(config_logits)
    value = _FixedValue(zeros)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()))
    trainer = PPOTrainer(config, policy, value, reference, optimizer)
    metrics = trainer.train_step(
        _batch(ids, rewards, zeros, zeros, "right"),
        old,
    )

    assert metrics.advantage_mean == pytest.approx(1.6, abs=1e-6)
    assert metrics.gae_mean == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_sequence_reward_sits_on_the_last_response_token(padding_side: str) -> None:
    """A rank-3 head is not averaged over time. The score lands on the last 1.

    A sequence-level ``[B, 1]`` logit uses that same index. Left padding and
    right padding disagree about which column that is.
    """
    token_mask, _ = response_token_mask(PROMPT, RESPONSE, SEQ_LEN, padding_side)
    expected, _ = _expected(padding_side)
    assert torch.equal(token_mask, expected)

    time = torch.arange(SEQ_LEN, dtype=torch.float32)
    per_token = time.view(1, SEQ_LEN, 1).expand(2, SEQ_LEN, 1).clone()
    per_token[1] += 100
    placed = RolloutGenerator._decode_reward(SimpleNamespace(logits=per_token), token_mask)
    last = last_response_index(token_mask)
    for row in range(2):
        index = int(last[row])
        assert placed[row, index] == pytest.approx(per_token[row, index, 0])
        assert torch.count_nonzero(placed[row]) == 1
        assert placed[row, index] != pytest.approx(per_token[row, :, 0].mean())
    assert torch.all(placed[token_mask == 0] == 0)

    scalar = torch.tensor([[3.0], [5.0]])
    placed_scalar = RolloutGenerator._decode_reward(SimpleNamespace(logits=scalar), token_mask)
    assert torch.equal(placed_scalar.nonzero()[:, 1], last)
    assert placed_scalar[0, int(last[0])] == pytest.approx(3.0)
    assert placed_scalar[1, int(last[1])] == pytest.approx(5.0)

    # Feature axis is reduced at the chosen token. Other times stay out.
    featured = torch.ones(2, SEQ_LEN, 2)
    featured[0, int(last[0])] = torch.tensor([50.0, 70.0])
    featured[1, int(last[1])] = torch.tensor([8.0, 12.0])
    placed_featured = RolloutGenerator._decode_reward(SimpleNamespace(logits=featured), token_mask)
    assert placed_featured[0, int(last[0])] == pytest.approx(60.0)
    assert placed_featured[1, int(last[1])] == pytest.approx(10.0)
    assert torch.count_nonzero(placed_featured) == 2


def test_masked_whiten_divides_by_the_response_count() -> None:
    """Std is the masked root-mean-square, not torch.std over the whole row."""
    values = torch.tensor([[0.0, 0.0, 0.0, 4.0, 100.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0]])
    whitened, mean, std = masked_whiten(values, mask)

    assert mean == pytest.approx(4 / 3)
    assert std == pytest.approx((32 / 9) ** 0.5)
    assert whitened[0, 0] == 0
    assert whitened[0, -1] == 0
    assert mean != pytest.approx(values.mean())
