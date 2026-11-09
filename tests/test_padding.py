"""Padding stays out of PPO forwards, and the train-step ratio uses temperature.

A row's logits at its real tokens match that row run alone, with no left
or right padding. The comparison uses ``torch.testing.assert_close`` at
the default tolerance for the dtype. The batch is the mixed layout:
left pad, prompt, response, right pad. The model is ``tiny_causal_lm``.

Sampling and ``train_step`` both read ``AlignmentConfig.temperature``.
The rollout records that number. A train step whose config disagrees
raises. Log-probs are gathered from ``logits / temperature``, so the
first-step ratio is 1 when the temperature is not 1.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.tiny_lm import tiny_causal_lm
from torch import nn

from rlpost.alignment.rollout import RolloutGenerator
from rlpost.config import AlignmentConfig, TrainingConfig
from rlpost.ops import gather_logprobs, padding_attention
from rlpost.ppo_engine import PPOTrainer

EOS = 2


def test_position_ids_are_the_clamped_cumsum_of_the_attention_mask() -> None:
    """Position 0 is the first real token. Left pads stay 0. Right pads do not count."""
    prompt_lengths = torch.tensor([3, 4], dtype=torch.long)
    response_lengths = torch.tensor([2, 3], dtype=torch.long)
    attention, position_ids = padding_attention(
        prompt_lengths,
        response_lengths,
        9,
        "mixed",
        prompt_width=5,
    )
    assert attention[0].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert attention[1].tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 0]
    assert torch.equal(position_ids, (attention.cumsum(dim=-1) - 1).clamp(min=0))
    assert position_ids[0].tolist() == [0, 0, 0, 1, 2, 3, 4, 4, 4]

    right_attention, right_positions = padding_attention(
        prompt_lengths,
        response_lengths,
        9,
        "right",
    )
    assert right_attention[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0]
    assert torch.equal(right_positions, (right_attention.cumsum(dim=-1) - 1).clamp(min=0))

    empty_attention, empty_positions = padding_attention(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        3,
        "right",
    )
    assert empty_attention.tolist() == [[0, 0, 0]]
    assert empty_positions.tolist() == [[0, 0, 0]]


class _RecordingLM(nn.Module):
    """Forward the inner causal LM. Pass a mask through only when the caller did."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        kwargs: dict[str, torch.Tensor] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        out = self.inner(input_ids=input_ids, **kwargs)
        self.calls.append((input_ids.detach(), out.logits.detach()))
        return out


class _ZeroValue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask, position_ids
        return input_ids.new_zeros(input_ids.shape, dtype=torch.float32) + self.bias


def _solo_rows(
    inner: nn.Module,
    input_ids: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> list[torch.Tensor]:
    """Logits of each row's real tokens, run with the pads removed."""
    solos = []
    with torch.no_grad():
        for row in range(input_ids.shape[0]):
            lo = int(start[row])
            hi = int(end[row])
            real = input_ids[row, lo:hi].unsqueeze(0)
            solos.append(inner(input_ids=real).logits[0].clone())
    return solos


def _assert_matches_solo(
    logits: torch.Tensor,
    solos: list[torch.Tensor],
    start: torch.Tensor,
    end: torch.Tensor,
) -> None:
    for row, solo in enumerate(solos):
        lo = int(start[row])
        hi = int(end[row])
        torch.testing.assert_close(logits[row, lo:hi], solo)


def _assert_span_matches_unpadded(
    inner: nn.Module,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> None:
    """Logits on ``[start, end)`` match a forward of that slice alone."""
    _assert_matches_solo(logits, _solo_rows(inner, input_ids, start, end), start, end)


def test_mixed_batch_logits_match_the_unpadded_row() -> None:
    """Left and right padding do not move the logits on the real tokens.

    Row 0 is ``[pad, pad, 4, 5, 6, 7, 8, pad, pad]``. Row 1 is shorter
    on both pads. ``train_step`` runs the policy and the reference on
    that batch. Each row's logits on its real columns match the same
    tokens with the pads removed.
    """
    policy = _RecordingLM(tiny_causal_lm(0).eval())
    reference = _RecordingLM(tiny_causal_lm(1).eval())
    value = _ZeroValue()
    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=1e-3,
    )
    trainer = PPOTrainer(config, policy, value, reference, optimizer)

    prompt_width = 5
    seq_len = 9
    input_ids = torch.tensor(
        [
            [0, 0, 4, 5, 6, 7, 8, 0, 0],
            [0, 11, 12, 13, 14, 15, 16, 17, 0],
        ],
        dtype=torch.long,
    )
    prompt_lengths = torch.tensor([3, 4], dtype=torch.long)
    response_lengths = torch.tensor([2, 3], dtype=torch.long)
    batch = {
        "input_ids": input_ids,
        "rewards": torch.zeros(2, seq_len),
        "values": torch.zeros(2, seq_len),
        "dones": torch.zeros(2, seq_len),
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
        "padding_side": "mixed",
        "prompt_width": prompt_width,
        "temperature": 1.0,
    }
    # The step updates the policy. Snapshot the unpadded logits on the
    # weights the forward inside train_step still has.
    start = prompt_width - prompt_lengths
    end = prompt_width + response_lengths
    policy_solo = _solo_rows(policy.inner, input_ids, start, end)
    reference_solo = _solo_rows(reference.inner, input_ids, start, end)
    trainer.train_step(batch, torch.zeros(2, seq_len - 1))

    assert policy.calls, "policy forward did not run"
    assert reference.calls, "reference forward did not run"
    _assert_matches_solo(policy.calls[-1][1], policy_solo, start, end)
    _assert_matches_solo(reference.calls[-1][1], reference_solo, start, end)


def test_rollout_logits_match_the_unpadded_row() -> None:
    """Generation forwards hide the same pads.

    One row stops on EOS, so later columns are a right pad. The other
    row runs to the length limit. Queries are left-padded. Policy,
    reference, and reward logits on the real columns match the unpadded
    tokens.
    """
    policy = _RecordingLM(tiny_causal_lm(0).eval())
    reference = _RecordingLM(tiny_causal_lm(1).eval())
    reward = _RecordingLM(tiny_causal_lm(2).eval())
    prompt_width = 5
    prompt_lengths = torch.tensor([2, 4], dtype=torch.long)
    query = torch.tensor(
        [
            [0, 0, 0, 4, 5],
            [0, 11, 12, 13, 14],
        ],
        dtype=torch.long,
    )
    plan = torch.tensor(
        [
            [8, EOS, 9],
            [8, 8, 8],
        ],
        dtype=torch.long,
    )
    cursor = {"step": 0}

    def _planned(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        del num_samples
        choice = plan[:, cursor["step"]].to(device=probs.device)
        cursor["step"] += 1
        return choice.view(-1, 1)

    generator = RolloutGenerator(
        policy_model=policy,
        reference_model=reference,
        reward_model=reward,
        tokenizer=None,
        alignment_config=AlignmentConfig(),
        max_response_length=int(plan.shape[1]),
        top_p=1.0,
        eos_token_id=EOS,
        pad_token_id=0,
    )
    original = torch.multinomial
    torch.multinomial = _planned
    try:
        generated = generator.generate_rollout(query, prompt_lengths)
    finally:
        torch.multinomial = original

    assert generated.response_lengths.tolist() == [2, 3]
    assert int(generated.input_ids[0, -1]) == 0
    assert int(generated.input_ids[0, -2]) == EOS

    def _check(recorder: _RecordingLM) -> None:
        assert recorder.calls
        for seen_ids, seen_logits in recorder.calls:
            start = []
            end = []
            for row in range(seen_ids.shape[0]):
                lo = prompt_width - int(prompt_lengths[row])
                hi = prompt_width
                for token in seen_ids[row, prompt_width:].tolist():
                    hi += 1
                    if token == EOS:
                        break
                start.append(lo)
                end.append(hi)
            _assert_span_matches_unpadded(
                recorder.inner,
                seen_ids,
                seen_logits,
                torch.tensor(start),
                torch.tensor(end),
            )

    _check(policy)
    _check(reference)
    _check(reward)


def test_ratio_is_one_at_the_first_step_when_temperature_is_not_one() -> None:
    """Old log-probs were gathered from logits / temperature. The ratio is 1.

    The weights have not moved. A train step that gathers from the raw
    logits gets a ratio other than 1 when the temperature is not 1.
    """
    temperature = 0.5
    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    config.alignment.temperature = temperature
    policy = tiny_causal_lm(0).eval()
    value = _ZeroValue()
    reference = tiny_causal_lm(1).eval()
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=1e-2,
    )
    trainer = PPOTrainer(config, policy, value, reference, optimizer)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 128, (batch_size, seq_len))
    with torch.no_grad():
        logits = policy(input_ids=input_ids).logits
        old_log_probs = gather_logprobs(logits / temperature, input_ids)
    batch = {
        "input_ids": input_ids,
        "rewards": torch.zeros(batch_size, seq_len),
        "values": torch.zeros(batch_size, seq_len),
        "dones": torch.zeros(batch_size, seq_len),
        "prompt_lengths": torch.zeros(batch_size, dtype=torch.long),
        "response_lengths": torch.full((batch_size,), seq_len, dtype=torch.long),
        "padding_side": "right",
        "temperature": temperature,
    }
    seen: dict[str, torch.Tensor] = {}
    original = trainer.compute_policy_loss

    def _spy(
        log_probs_new: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss, ratio = original(log_probs_new, log_probs_old, advantages, mask)
        seen["ratio"] = ratio.detach()
        return loss, ratio

    trainer.compute_policy_loss = _spy
    trainer.train_step(batch, old_log_probs)

    ratio = seen["ratio"]
    assert torch.equal(ratio, torch.ones_like(ratio))


def test_rollout_sampled_at_another_temperature_cannot_train() -> None:
    """The generator has no temperature of its own. A 0.7 rollout cannot train at 1.0.

    ``RolloutGenerator`` reads ``AlignmentConfig.temperature`` and records
    it on the batch. Passing ``temperature=`` to the generator is not a
    parameter. ``train_step`` on a config whose field is 1.0 raises.
    """
    policy = tiny_causal_lm(0).eval()
    reference = tiny_causal_lm(1).eval()
    reward = tiny_causal_lm(2).eval()
    sample_config = AlignmentConfig(temperature=0.7)
    with pytest.raises(TypeError, match="temperature"):
        RolloutGenerator(
            policy_model=policy,
            reference_model=reference,
            reward_model=reward,
            tokenizer=None,
            alignment_config=sample_config,
            temperature=0.7,
        )

    generator = RolloutGenerator(
        policy_model=policy,
        reference_model=reference,
        reward_model=reward,
        tokenizer=None,
        alignment_config=sample_config,
        max_response_length=2,
        top_p=1.0,
        eos_token_id=EOS,
        pad_token_id=0,
    )
    assert generator.temperature == 0.7
    generated = generator.generate_rollout(torch.randint(0, 128, (2, 4)))
    assert generated.temperature == 0.7

    config = TrainingConfig.toy_mode()
    config.log_to_wandb = False
    assert config.alignment.temperature == 1.0
    value = _ZeroValue()
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()),
        lr=1e-3,
    )
    trainer = PPOTrainer(config, policy, value, reference, optimizer)
    rows, length = generated.input_ids.shape
    batch = {
        "input_ids": generated.input_ids,
        "rewards": generated.rewards,
        "values": torch.zeros(rows, length),
        "dones": generated.dones,
        "prompt_lengths": generated.prompt_lengths,
        "response_lengths": generated.response_lengths,
        "padding_side": generated.padding_side,
        "prompt_width": generated.prompt_width,
        "temperature": generated.temperature,
    }
    with pytest.raises(ValueError, match="0.7"):
        trainer.train_step(batch, torch.zeros(rows, length - 1))
