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
from rlpost.ppo_engine import GeneralizedAdvantageEstimation, PPOTrainer, masked_whiten

PROMPT = torch.tensor([2, 1], dtype=torch.long)
RESPONSE = torch.tensor([3, 2], dtype=torch.long)
SEQ_LEN = 6
VOCAB = 6
# This fixture's padded query width. It equals max(PROMPT). The width-8 test
# is the batch where the pad is wider than the longest prompt.
MIXED_WIDTH = 2

# Right: [prompt | response | pad].
# Mixed: prompts left-padded to MIXED_WIDTH, then response, then pad.
# Row 1 is the one that distinguishes them: right starts the response at column 1,
# mixed starts every response at column 2.
RIGHT_TOKEN = torch.tensor(
    [
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ]
)
MIXED_TOKEN = torch.tensor(
    [
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    ]
)
RIGHT_GATHERED = torch.tensor(
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0],
    ]
)
MIXED_GATHERED = torch.tensor(
    [
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
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
    return MIXED_TOKEN, MIXED_GATHERED


def _mask(
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
    seq_len: int,
    padding_side: str,
    prompt_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the mask. Mixed uses this fixture's width unless one is passed."""
    if padding_side == "mixed" and prompt_width is None:
        prompt_width = MIXED_WIDTH
    return response_token_mask(
        prompt_lengths,
        response_lengths,
        seq_len,
        padding_side,
        prompt_width=prompt_width,
    )


def _value_columns_in_use(gathered: torch.Tensor) -> torch.Tensor:
    """1 where rollout ``V_old[:, t]`` enters GAE.

    The state for the action that emits token ``t + 1`` is column ``t``, which
    can be the last prompt token. The bootstrap is column ``t + 1``, and only
    when the next action index is real. The live value head is trained on the
    state columns. A 0 here is unused by both. The value written on the
    action's own token is unused when the backup stops.
    """
    actions = gathered.shape[1]
    used = torch.zeros(gathered.shape[0], actions + 1)
    used[:, :-1] = gathered
    next_real = torch.ones_like(gathered)
    if actions > 1:
        next_real[:, :-1] = gathered[:, 1:]
    continues = gathered * next_real
    used[:, 1:] = torch.maximum(used[:, 1:], continues)
    return used


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
    batch = {
        "input_ids": input_ids,
        "rewards": rewards,
        "values": values,
        "next_values": next_values,
        "dones": torch.zeros(input_ids.shape),
        "prompt_lengths": PROMPT,
        "response_lengths": RESPONSE,
        "padding_side": padding_side,
    }
    if padding_side == "mixed":
        batch["prompt_width"] = MIXED_WIDTH
    return batch


@pytest.mark.parametrize("padding_side", ["right", "mixed"])
def test_gathered_mask_matches_the_rollout_layout(padding_side: str) -> None:
    """Index t of the gathered mask marks token t+1, and only response tokens."""
    token_mask, gathered = _mask(PROMPT, RESPONSE, SEQ_LEN, padding_side)
    expected_token, expected_gathered = _expected(padding_side)
    assert torch.equal(token_mask, expected_token)
    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(gathered, token_mask[:, 1:])


def test_suffix_left_layout_is_rejected() -> None:
    """A response flush with the end of the row is not a generation layout."""
    with pytest.raises(ValueError, match="right"):
        response_token_mask(PROMPT, RESPONSE, SEQ_LEN, "left")


def test_mixed_layout_rejects_a_response_past_the_common_prompt_width() -> None:
    """Each row fits on its own. The shared prompt width pushes row 0 off the end."""
    prompts = torch.tensor([1, 4], dtype=torch.long)
    responses = torch.tensor([4, 1], dtype=torch.long)
    token_mask, _ = response_token_mask(prompts, responses, 6, "right")
    assert token_mask.shape == (2, 6)
    with pytest.raises(ValueError, match="prompt_width \\+ response_length"):
        response_token_mask(prompts, responses, 6, "mixed", prompt_width=4)


def test_mixed_layout_requires_prompt_width() -> None:
    """Mixed does not fall back to max(prompt_lengths)."""
    with pytest.raises(ValueError, match="prompt_width"):
        response_token_mask(PROMPT, RESPONSE, SEQ_LEN, "mixed")


def test_mixed_response_starts_at_the_padded_query_width() -> None:
    """A query padded to 8 with a longest prompt of 5 opens the response at 8.

    max(prompt_lengths) would open it at 5 and count the extra left pad as
    response tokens. The loss uses that same window: a logit that scores
    column 5 is outside it, and a logit that scores column 8 is inside it.
    """
    prompts = torch.tensor([5, 3], dtype=torch.long)
    responses = torch.tensor([2, 2], dtype=torch.long)
    seq_len = 12
    width = 8
    assert int(prompts.max()) == 5

    token_mask, gathered = response_token_mask(
        prompts, responses, seq_len, "mixed", prompt_width=width
    )
    expected = torch.zeros(2, seq_len)
    expected[:, 8:10] = 1
    assert torch.equal(token_mask, expected)
    assert torch.equal(gathered, expected[:, 1:])
    assert torch.all(token_mask[:, :8] == 0)
    assert gathered[:, 4].eq(0).all()
    assert gathered[:, 7].eq(1).all()

    policy_logits = torch.arange(2 * seq_len * VOCAB, dtype=torch.float32).reshape(
        2, seq_len, VOCAB
    )
    policy_logits = policy_logits / 10
    ref_logits = -policy_logits
    ids = torch.ones(2, seq_len, dtype=torch.long)
    zeros = torch.zeros(2, seq_len)
    rewards = place_sequence_reward(torch.tensor([2.0, 2.0]), token_mask)
    old_log_probs = torch.zeros(2, seq_len - 1)
    batch = {
        "input_ids": ids,
        "rewards": rewards,
        "values": zeros,
        "dones": zeros,
        "prompt_lengths": prompts,
        "response_lengths": responses,
        "padding_side": "mixed",
        "prompt_width": width,
    }
    base = _run(policy_logits, ref_logits, zeros, batch, old_log_probs)

    outside = policy_logits.clone()
    outside[:, 4, 1] = outside[:, 4, 1] + 5
    held = _run(outside, ref_logits, zeros, batch, old_log_probs)
    assert held.kl_divergence == pytest.approx(base.kl_divergence, abs=1e-5)
    assert held.policy_loss == pytest.approx(base.policy_loss, abs=1e-5)

    inside = policy_logits.clone()
    inside[:, 7, 1] = inside[:, 7, 1] + 5
    moved = _run(inside, ref_logits, zeros, batch, old_log_probs)
    assert base.kl_divergence > 0
    assert moved.kl_divergence != pytest.approx(base.kl_divergence, abs=1e-4)
    assert moved.policy_loss != pytest.approx(base.policy_loss, abs=1e-4)


@pytest.mark.parametrize("padding_side", ["right", "mixed"])
def test_prompt_and_padding_do_not_change_losses_advantages_or_kl(padding_side: str) -> None:
    """Corrupting prompt ids, padding, or an unused value column leaves the step unchanged.

    The state for the first response action sits on the last prompt token.
    GAE reads that column from ``batch["values"]`` (``V_old``), so this test
    does not corrupt it. The live head is trained on the same state columns.
    A separate ``next_values`` entry is not read. A response-scoring logit
    and the last-token reward do change the step. An unmasked mean fails the
    first check. An empty mask fails the second.
    """
    token_mask, gathered = _mask(PROMPT, RESPONSE, SEQ_LEN, padding_side)
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

    in_use = _value_columns_in_use(gathered)
    values = torch.full((2, SEQ_LEN), 0.5)
    values[token_mask == 0] = 0
    corrupt_values = values.clone()
    # Includes the value on the last response token. The backup stops there,
    # so that column is not V_old for any response action.
    corrupt_values[in_use == 0] = 80

    next_values = torch.zeros(2, SEQ_LEN)
    corrupt_next = next_values.clone()
    corrupt_next[token_mask == 0] = 7

    value_pred = torch.full((2, SEQ_LEN), 0.25)
    corrupt_pred = value_pred.clone()
    # The last prompt column is the state of the first response action.
    # The value loss trains the live head there. GAE does not read it.
    corrupt_pred[in_use == 0] = 9

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


@pytest.mark.parametrize("padding_side", ["right", "mixed"])
def test_sequence_reward_sits_on_the_last_response_token(padding_side: str) -> None:
    """A rank-3 head is not averaged over time. The score lands on the last 1.

    A sequence-level ``[B, 1]`` logit uses that same index. Right padding and
    the mixed layout disagree about which column that is.
    """
    token_mask, _ = _mask(PROMPT, RESPONSE, SEQ_LEN, padding_side)
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


class _CausalValue(nn.Module):
    """``V[:, t]`` is the sum of token ids through ``t``. Later tokens are invisible."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(dtype=torch.float32).cumsum(dim=-1) + self.bias


def test_action_advantage_does_not_see_its_own_token() -> None:
    """gamma=0 and rewards=0 make ``A = -V(s)``.

    The action at index ``t`` emits ``input_ids[:, t + 1]``. Its state was
    read at position ``t``, so editing that token must not move the advantage.
    A baseline that has already seen the token is an action-dependent baseline.
    """
    seq_len = SEQ_LEN
    prompt = torch.tensor([2, 2], dtype=torch.long)
    response = torch.tensor([3, 3], dtype=torch.long)
    ids = torch.tensor(
        [
            [2, 5, 1, 4, 3, 0],
            [2, 5, 1, 4, 3, 0],
        ],
        dtype=torch.long,
    )
    edited = ids.clone()
    edited[:, 2] = 0
    action_index = 1  # logits here score token 2

    def run(token_ids: torch.Tensor) -> torch.Tensor:
        values = token_ids.to(dtype=torch.float32).cumsum(dim=-1)
        config = TrainingConfig.toy_mode()
        config.log_to_wandb = False
        config.alignment.gamma = 0.0
        config.alignment.advantage_normalization = False
        policy_logits = torch.zeros(2, seq_len, VOCAB)
        policy = _FixedLogits(policy_logits)
        reference = _FixedLogits(policy_logits)
        value = _CausalValue()
        optimizer = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()))
        trainer = PPOTrainer(config, policy, value, reference, optimizer)
        seen: dict[str, torch.Tensor] = {}
        compute_policy_loss = trainer.compute_policy_loss

        def capture(log_probs_new, log_probs_old, advantages, mask=None):
            seen["advantages"] = advantages.detach().clone()
            return compute_policy_loss(log_probs_new, log_probs_old, advantages, mask)

        trainer.compute_policy_loss = capture
        batch = {
            "input_ids": token_ids,
            "rewards": torch.zeros(2, seq_len),
            "values": values,
            "next_values": torch.zeros(2, seq_len),
            "dones": torch.zeros(2, seq_len),
            "prompt_lengths": prompt,
            "response_lengths": response,
            "padding_side": "right",
        }
        trainer.train_step(batch, torch.zeros(2, seq_len - 1))
        return seen["advantages"]

    base = run(ids)
    moved = run(edited)
    state_value = ids.to(dtype=torch.float32).cumsum(dim=-1)[:, action_index]
    token_value = ids.to(dtype=torch.float32).cumsum(dim=-1)[:, action_index + 1]
    edited_token_value = edited.to(dtype=torch.float32).cumsum(dim=-1)[:, action_index + 1]

    assert not torch.equal(token_value, edited_token_value)
    assert torch.allclose(base[:, action_index], -state_value, atol=1e-6)
    assert torch.allclose(moved[:, action_index], base[:, action_index], atol=1e-6)
    assert state_value[0].item() == pytest.approx(7.0)
    assert token_value[0].item() == pytest.approx(8.0)


def test_advantages_and_returns_stay_fixed_across_epochs() -> None:
    """Two train_step calls on one batch share advantages and returns.

    GAE reads ``batch["values"]`` (``V_old``). The live head is ``value_pred``
    and moves when the critic steps. Returns are ``A + V_old``. Recomputing
    GAE from the updated head makes the second call disagree. That is what
    the previous step did.
    """
    seq_len = 4
    prompt = torch.tensor([1], dtype=torch.long)
    response = torch.tensor([2], dtype=torch.long)
    v_old = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    rewards = torch.tensor([[0.0, 0.0, 3.0, 0.0]])
    dones = torch.zeros(1, seq_len)
    ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)

    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    config.alignment.gamma = 0.5
    config.alignment.gae_lambda = 0.5
    config.alignment.advantage_normalization = False
    policy_logits = torch.zeros(1, seq_len, VOCAB)
    policy = _FixedLogits(policy_logits)
    reference = _FixedLogits(policy_logits)
    value = _FixedValue(torch.zeros(1, seq_len))
    optimizer = torch.optim.SGD(list(policy.parameters()) + list(value.parameters()), lr=1.0)
    trainer = PPOTrainer(config, policy, value, reference, optimizer)

    seen: dict[str, torch.Tensor] = {}
    gae_compute = trainer.gae.compute
    value_loss_fn = trainer.compute_value_loss

    def capture_gae(action_rewards, values, next_values, action_dones, mask=None):
        seen["gae_values"] = values.detach().clone()
        seen["gae_next"] = next_values.detach().clone()
        advantages, returns = gae_compute(action_rewards, values, next_values, action_dones, mask)
        seen["advantages"] = advantages.detach().clone()
        seen["returns"] = returns.detach().clone()
        return advantages, returns

    def capture_value(value_pred, returns, old_values=None, mask=None):
        seen["value_pred"] = value_pred.detach().clone()
        return value_loss_fn(value_pred, returns, old_values, mask)

    trainer.gae.compute = capture_gae
    trainer.compute_value_loss = capture_value

    batch = {
        "input_ids": ids,
        "rewards": rewards,
        "values": v_old,
        "dones": dones,
        "prompt_lengths": prompt,
        "response_lengths": response,
        "padding_side": "right",
    }
    old_log_probs = torch.zeros(1, seq_len - 1)
    trainer.train_step(batch, old_log_probs)
    first = {key: tensor.clone() for key, tensor in seen.items()}
    trainer.train_step(batch, old_log_probs)
    second = {key: tensor.clone() for key, tensor in seen.items()}

    gathered = response_token_mask(prompt, response, seq_len, "right")[1]
    spec_adv, spec_ret = GeneralizedAdvantageEstimation(0.5, 0.5).compute(
        rewards[:, 1:],
        v_old[:, :-1],
        v_old[:, 1:],
        dones[:, 1:],
        gathered,
    )

    assert torch.equal(first["advantages"], second["advantages"])
    assert torch.equal(first["returns"], second["returns"])
    assert torch.equal(first["advantages"], spec_adv)
    assert torch.equal(first["returns"], spec_ret)
    assert torch.equal(first["gae_values"], v_old[:, :-1])
    assert torch.equal(second["gae_values"], v_old[:, :-1])
    assert torch.equal(first["gae_next"], v_old[:, 1:])
    assert torch.equal(second["gae_next"], v_old[:, 1:])
    assert torch.equal(first["returns"], (first["advantages"] + v_old[:, :-1]) * gathered)
    assert not torch.equal(first["value_pred"], second["value_pred"])
    assert not torch.equal(first["value_pred"], v_old[:, :-1])


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
