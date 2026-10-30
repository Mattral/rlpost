"""CPU rollout buffer and a token-at-a-time sampler.

``RolloutBuffer`` keeps completed rollouts in a Python list. Tensors are
moved to CPU. They are pinned only when CUDA is available.
``RolloutGenerator`` samples response tokens from a causal LM that returns
``.logits``. The async generator-thread class was removed in S0.4. Nothing
called it except ``train.py``, and it had no test.
"""

import logging
import threading
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    """Single PPO rollout trajectory."""

    query_tokens: torch.Tensor
    response_tokens: torch.Tensor
    reward: torch.Tensor
    logits_policy: torch.Tensor
    logits_reference: torch.Tensor


class RolloutBuffer:
    """In-memory ring buffer for collecting rollouts during generation phase."""

    def __init__(self, capacity: int = 1000, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.buffer: list = []
        self.ptr = 0
        self._lock = threading.Lock()

    def add(self, rollout: Rollout) -> None:
        """Add a rollout to the buffer (overwrites oldest if full)."""

        # Convert tensors to pinned CPU memory to enable non-blocking device copies
        def _pin(t: torch.Tensor) -> torch.Tensor:
            if not isinstance(t, torch.Tensor):
                return t
            out = t.detach().cpu()
            # pin_memory needs a CUDA driver. CPU-only torch raises if it is called.
            if torch.cuda.is_available():
                return out.pin_memory()
            return out

        pinned = Rollout(
            query_tokens=_pin(rollout.query_tokens),
            response_tokens=_pin(rollout.response_tokens),
            reward=_pin(rollout.reward),
            logits_policy=_pin(rollout.logits_policy),
            logits_reference=_pin(rollout.logits_reference),
        )

        with self._lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(pinned)
            else:
                self.buffer[self.ptr] = pinned
                self.ptr = (self.ptr + 1) % self.capacity

    def sample_batch(self, batch_size: int) -> list:
        """Sample a random batch of rollouts."""
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample from empty rollout buffer.")

        import random

        with self._lock:
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.ptr = 0

    def size(self) -> int:
        """Current number of rollouts in buffer."""
        return len(self.buffer)


class RolloutCollator:
    """Collate rollouts into batched tensors."""

    def __init__(self, max_response_length: int = 128):
        self.max_response_length = max_response_length

    def __call__(self, rollouts: list) -> dict:
        """Collate a list of Rollout objects."""
        # Pad response tokens
        response_tokens_list = [r.response_tokens for r in rollouts]
        padded_responses = self._pad_sequence(response_tokens_list, self.max_response_length)

        # Pad logits
        logits_policy_list = [r.logits_policy for r in rollouts]
        logits_reference_list = [r.logits_reference for r in rollouts]

        padded_logits_policy = self._pad_logits(logits_policy_list, self.max_response_length)
        padded_logits_reference = self._pad_logits(logits_reference_list, self.max_response_length)

        # Rewards
        rewards = torch.stack([r.reward for r in rollouts])

        query_tokens_list = [r.query_tokens for r in rollouts]
        max_query_length = max(token.shape[0] for token in query_tokens_list)
        padded_queries = self._pad_sequence(query_tokens_list, max_query_length)
        query_lengths = torch.tensor([t.shape[0] for t in query_tokens_list], dtype=torch.long)

        response_lengths = torch.tensor(
            [r.response_tokens.shape[0] for r in rollouts], dtype=torch.long
        )

        return {
            "query_tokens": padded_queries,
            "response_tokens": padded_responses,
            "logits_policy": padded_logits_policy,
            "logits_reference": padded_logits_reference,
            "rewards": rewards,
            "query_lengths": query_lengths,
            "response_lengths": response_lengths,
        }

    @staticmethod
    def _pad_sequence(sequences: list, max_length: int) -> torch.Tensor:
        """Pad token sequences to max_length."""
        padded = torch.zeros(len(sequences), max_length, dtype=torch.long)
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_length)
            padded[i, :seq_len] = seq[:seq_len]
        return padded

    @staticmethod
    def _pad_logits(logits_list: list, max_length: int) -> torch.Tensor:
        """Pad logits to (batch, max_length, vocab_size)."""
        vocab_size = logits_list[0].shape[-1]
        padded = torch.zeros(len(logits_list), max_length, vocab_size)
        for i, logits in enumerate(logits_list):
            seq_len = min(len(logits), max_length)
            padded[i, :seq_len] = logits[:seq_len]
        return padded


class RolloutGenerator:
    """Generate rollouts from policy model and score with reward model."""

    def __init__(
        self,
        policy_model: torch.nn.Module,
        reference_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        tokenizer,
        max_response_length: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.temperature = temperature
        self.top_p = top_p

    @torch.no_grad()
    def generate_rollout(self, query_tokens: torch.Tensor) -> list[Rollout]:
        """Generate one rollout per row of ``query_tokens`` ([B, T_query])."""
        # Generate response tokens autoregressively
        response_tokens_list = []
        logits_policy_list = []
        logits_reference_list = []

        current_tokens = query_tokens.clone()

        for _ in range(self.max_response_length):
            # Policy forward pass
            policy_out = self.policy_model(current_tokens)
            policy_logits = policy_out.logits[:, -1, :]

            # Reference forward pass
            ref_out = self.reference_model(current_tokens)
            ref_logits = ref_out.logits[:, -1, :]

            logits_policy_list.append(policy_logits)
            logits_reference_list.append(ref_logits)

            next_token = self._sample_token(
                policy_logits,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            response_tokens_list.append(next_token)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

        response_tokens = torch.cat(response_tokens_list, dim=1)
        logits_policy = torch.stack(logits_policy_list, dim=1)
        logits_reference = torch.stack(logits_reference_list, dim=1)

        with torch.no_grad():
            reward_input = torch.cat([query_tokens, response_tokens], dim=1)
            reward_out = self.reward_model(reward_input)
            rewards = self._decode_reward(reward_out)

        rollouts = []
        for index in range(query_tokens.shape[0]):
            rollouts.append(
                Rollout(
                    query_tokens=query_tokens[index].detach().cpu(),
                    response_tokens=response_tokens[index].detach().cpu(),
                    logits_policy=logits_policy[index].detach().cpu(),
                    logits_reference=logits_reference[index].detach().cpu(),
                    reward=rewards[index].detach().cpu(),
                )
            )

        return rollouts

    @staticmethod
    def _decode_reward(reward_out: object) -> torch.Tensor:
        if hasattr(reward_out, "logits"):
            logits = reward_out.logits
            if logits.dim() == 2:
                return logits.squeeze(-1)
            if logits.dim() == 3:
                return logits.mean(dim=-1).mean(dim=-1)

        if hasattr(reward_out, "pooler_output"):
            return reward_out.pooler_output.mean(dim=-1)

        raise RuntimeError("Unsupported reward model output shape")

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample the next token. Returns ids of shape [batch, 1]."""
        scaled_logits = logits / temperature
        if top_p < 1.0:
            probs = F.softmax(scaled_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumulative_probs > top_p
            cutoff_mask[..., 0] = False

            remove_mask = torch.zeros_like(logits, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=cutoff_mask)
            scaled_logits = scaled_logits.masked_fill(remove_mask, float("-inf"))

        token = torch.multinomial(F.softmax(scaled_logits, dim=-1), num_samples=1)
        return token
