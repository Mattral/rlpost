from types import SimpleNamespace

import torch
import torch.nn as nn

from rlhf_platform.alignment.rollout import (
    Rollout,
    RolloutBuffer,
    RolloutCollator,
    RolloutGenerator,
)


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.head = nn.Linear(32, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        features = torch.randn(batch_size, seq_len, 32, device=input_ids.device)
        logits = self.head(features)
        return SimpleNamespace(logits=logits)


class DummyRewardModel(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        batch_size = input_ids.shape[0]
        return SimpleNamespace(logits=torch.randn(batch_size, 1, device=input_ids.device))


def test_rollout_buffer_sample_and_clear():
    buffer = RolloutBuffer(capacity=2, device="cpu")
    rollout_a = Rollout(
        query_tokens=torch.tensor([1, 2]),
        response_tokens=torch.tensor([3, 4]),
        reward=torch.tensor(0.5),
        logits_policy=torch.randn(2, 16),
        logits_reference=torch.randn(2, 16),
    )
    rollout_b = Rollout(
        query_tokens=torch.tensor([5]),
        response_tokens=torch.tensor([6, 7, 8]),
        reward=torch.tensor(1.0),
        logits_policy=torch.randn(3, 16),
        logits_reference=torch.randn(3, 16),
    )
    buffer.add(rollout_a)
    buffer.add(rollout_b)

    assert buffer.size() == 2
    sampled = buffer.sample_batch(batch_size=1)
    assert len(sampled) == 1
    assert isinstance(sampled[0], Rollout)

    buffer.clear()
    assert buffer.size() == 0


def test_rollout_collator_padding():
    rollouts = [
        Rollout(
            query_tokens=torch.tensor([10, 11]),
            response_tokens=torch.tensor([1, 2, 3]),
            reward=torch.tensor(0.1),
            logits_policy=torch.randn(3, 16),
            logits_reference=torch.randn(3, 16),
        ),
        Rollout(
            query_tokens=torch.tensor([20]),
            response_tokens=torch.tensor([4, 5]),
            reward=torch.tensor(0.2),
            logits_policy=torch.randn(2, 16),
            logits_reference=torch.randn(2, 16),
        ),
    ]
    collator = RolloutCollator(max_response_length=4)
    batch = collator(rollouts)

    assert batch["query_tokens"].shape == (2, 2)
    assert batch["response_tokens"].shape == (2, 4)
    assert batch["logits_policy"].shape == (2, 4, 16)
    assert batch["logits_reference"].shape == (2, 4, 16)
    assert batch["rewards"].shape == (2,)
    assert batch["query_lengths"].tolist() == [2, 1]
    assert batch["response_lengths"].tolist() == [3, 2]


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
    rollouts = generator.generate_rollout(query_batch)

    assert isinstance(rollouts, list)
    assert len(rollouts) == 2
    assert rollouts[0].response_tokens.shape == (3,)
    assert rollouts[0].logits_policy.shape == (3, 16)
    assert rollouts[0].logits_reference.shape == (3, 16)
    assert rollouts[0].reward.shape == () or rollouts[0].reward.shape == (1,)
