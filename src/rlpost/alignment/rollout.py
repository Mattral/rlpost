"""Token-at-a-time sampler for one PPO rollout batch.

``RolloutGenerator`` samples response tokens from a causal LM that returns
``.logits``. The policy, the reference, and the reward model each receive
an attention mask and position ids from ``padding_attention``, built from
the same lengths as the response mask. Left and right padding are not
tokens those models can read. The async generator-thread class was removed
in S0.4.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rlpost.config import AlignmentConfig
from rlpost.ops import gather_logprobs, padding_attention, response_token_mask


@dataclass
class Rollout:
    """Single PPO rollout trajectory."""

    query_tokens: torch.Tensor
    response_tokens: torch.Tensor
    reward: torch.Tensor
    logprobs_policy: torch.Tensor
    logprobs_reference: torch.Tensor


@dataclass
class RolloutBatch:
    """One ``generate_rollout`` call, in the mixed layout.

    ``input_ids`` is ``[B, prompt_width + max_response_length]``:
    ``[pad | prompt | response | pad]``. ``prompt_width`` is
    ``query_tokens.shape[1]``, the padded query width, not
    ``max(prompt_lengths)``. Every response starts at that column.

    ``response_lengths`` counts real response tokens, including the EOS
    token when the row emitted one, and not the padding after it.
    ``dones`` is 1 on that EOS token and 0 everywhere else. A row that
    fills ``max_response_length`` without EOS stays 0, including on its
    last token, so GAE bootstraps from ``V`` there instead of treating
    the length limit as an episode end.

    ``rewards`` is ``[B, T]``. The sequence score sits on the last real
    response token, which is the EOS token when the row has one.
    ``token_mask`` is the ``[B, T]`` response mask from
    ``response_token_mask(..., "mixed", prompt_width=prompt_width)``.
    ``rollouts`` is the per-row view: ``response_tokens`` is the real
    response, EOS included, padding omitted.

    ``temperature`` is the ``AlignmentConfig.temperature`` this call
    sampled with. ``PPOTrainer.train_step`` raises when its config
    field is a different number. There is no second temperature.

    ``logprobs_policy`` and ``logprobs_reference`` are ``[B, R]``.
    ``R`` is ``max_response_length``. Column ``r`` is the log-prob of
    the token in response column ``r``, including the pad columns after
    EOS. The value is ``gather_logprobs(logits / temperature)`` on the
    finished sequence, the same gather ``train_step`` repeats. It is
    not a ``[B, R, V]`` logit tensor, and it is not renormalized inside
    the ``top_p`` nucleus.
    """

    rollouts: list[Rollout]
    input_ids: torch.Tensor
    prompt_lengths: torch.Tensor
    response_lengths: torch.Tensor
    prompt_width: int
    padding_side: str
    dones: torch.Tensor
    rewards: torch.Tensor
    token_mask: torch.Tensor
    temperature: float
    logprobs_policy: torch.Tensor
    logprobs_reference: torch.Tensor


def response_logprobs_to_old(
    response_logprobs: torch.Tensor,
    *,
    prompt_width: int,
    seq_len: int,
) -> torch.Tensor:
    """Place ``[B, R]`` response log-probs into the gathered layout ``[B, T - 1]``.

    Response column ``r`` is token column ``prompt_width + r``.
    ``gather_logprobs`` scores that token from logits at
    ``prompt_width + r - 1``. ``prompt_width`` must be at least 1, so
    the first response token has a logit. Columns before that window
    are 0. The gathered response mask drops them. ``R`` is
    ``seq_len - prompt_width``.

    Args:
        response_logprobs: Policy log-probs of the response window,
            shape ``[B, R]``.
        prompt_width: Padded query width. Not ``max(prompt_lengths)``.
        seq_len: ``T``, the padded length of ``input_ids``.

    Returns:
        ``old_log_probs`` of shape ``[B, T - 1]`` for ``train_step``.
    """
    if response_logprobs.ndim != 2:
        raise ValueError(f"response_logprobs must be [B, R], got {tuple(response_logprobs.shape)}")
    if isinstance(prompt_width, bool) or not isinstance(prompt_width, int):
        raise TypeError(f"prompt_width must be an int, got {type(prompt_width).__name__}")
    if prompt_width < 1:
        raise ValueError(
            f"prompt_width must be >= 1 so the first response token has a logit, got {prompt_width}"
        )
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= prompt_width:
        raise ValueError(f"seq_len must be an int > prompt_width ({prompt_width}), got {seq_len!r}")
    response_len = seq_len - prompt_width
    if response_logprobs.shape[1] != response_len:
        raise ValueError(
            f"response_logprobs width {response_logprobs.shape[1]} must be "
            f"seq_len - prompt_width = {response_len}"
        )
    old = response_logprobs.new_zeros(response_logprobs.shape[0], seq_len - 1)
    old[:, prompt_width - 1 :] = response_logprobs
    return old


def last_response_index(token_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last response token in each row. Shape ``[B]``.

    ``token_mask`` is ``[B, T]`` with 1 on generated response tokens.
    A row of all zeros has no response token and raises.
    """
    if token_mask.ndim != 2:
        raise ValueError(f"token_mask must have shape [B, T], got {tuple(token_mask.shape)}")
    positions = torch.arange(token_mask.shape[1], device=token_mask.device)
    filled = torch.where(
        token_mask.bool(),
        positions.unsqueeze(0).expand_as(token_mask),
        torch.full(token_mask.shape, -1, device=token_mask.device, dtype=torch.long),
    )
    last = filled.max(dim=1).values
    if torch.any(last < 0):
        raise ValueError("every row needs a response token")
    return last


def place_sequence_reward(scores: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Write each sequence score on the last real response token.

    Every other position is 0, including earlier response tokens, the prompt,
    and padding. ``scores`` is ``[B]``. ``token_mask`` is ``[B, T]``.
    The return is ``[B, T]``.

    This is the standard PPO placement: the reward model scores the whole
    sequence, and that scalar is the return of the action that ended it.
    The tensor is token-aligned. ``PPOTrainer.train_step`` reads index
    ``t - 1`` as the reward for the action that emitted token ``t``.
    """
    if scores.ndim != 1 or scores.shape[0] != token_mask.shape[0]:
        raise ValueError(
            "scores must be [B] and match token_mask rows, got "
            f"scores {tuple(scores.shape)}, token_mask {tuple(token_mask.shape)}"
        )
    last = last_response_index(token_mask).to(device=scores.device)
    placed = torch.zeros(token_mask.shape, dtype=scores.dtype, device=scores.device)
    placed.scatter_(1, last.unsqueeze(1), scores.unsqueeze(1))
    return placed


class RolloutGenerator:
    """Generate rollouts from policy model and score with reward model."""

    def __init__(
        self,
        policy_model: torch.nn.Module,
        reference_model: torch.nn.Module,
        reward_model: torch.nn.Module,
        tokenizer,
        alignment_config: AlignmentConfig,
        max_response_length: int = 128,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        missing_eos_penalty: float | None = None,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        if not isinstance(alignment_config, AlignmentConfig):
            raise TypeError(
                f"alignment_config must be AlignmentConfig, got {type(alignment_config)}"
            )
        self.alignment_config = alignment_config
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.top_p = top_p
        if eos_token_id is None and tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_token_id = int(eos_token_id)
        if pad_token_id is None and tokenizer is not None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        pad_token_id = int(pad_token_id)
        if eos_token_id is not None and pad_token_id == eos_token_id:
            raise ValueError("pad_token_id and eos_token_id must differ")
        if missing_eos_penalty is not None and (
            isinstance(missing_eos_penalty, bool) or missing_eos_penalty <= 0
        ):
            raise ValueError(
                "missing_eos_penalty must be > 0 when set, got "
                f"{missing_eos_penalty}. None leaves the sequence reward unchanged."
            )
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        # Same number as AlignmentConfig.missing_eos_penalty. None is off.
        self.missing_eos_penalty = missing_eos_penalty

    @property
    def temperature(self) -> float:
        """Sampling temperature. This is ``AlignmentConfig.temperature``."""
        return self.alignment_config.temperature

    @torch.no_grad()
    def generate_rollout(
        self,
        query_tokens: torch.Tensor,
        prompt_lengths: torch.Tensor | None = None,
    ) -> RolloutBatch:
        """Generate one rollout per row of ``query_tokens`` ([B, prompt_width]).

        ``query_tokens`` is already left-padded to ``prompt_width``. The
        last column is the last real prompt token, which is what
        ``logits[:, -1]`` reads. The returned batch is the mixed layout
        ``[pad | prompt | response | pad]`` with that same ``prompt_width``.
        It is not ``max(prompt_lengths)``.

        Each row stops when it samples ``eos_token_id``. That token counts
        in ``response_lengths`` and in the response mask. Later columns of
        that row are ``pad_token_id``, and the mask is 0 there. A row that
        reaches ``max_response_length`` without EOS is truncated: its
        ``dones`` stay 0, so the backup at the last action bootstraps from
        ``V``. An EOS token is the only 1 in ``dones``.

        ``missing_eos_penalty``, when set, is subtracted from the sequence
        score of rows that did not emit EOS, and from no other row. ``None``
        leaves the score unchanged. The score is then written on the last
        real response token.

        Each policy, reference, and reward forward receives
        ``padding_attention`` for the tokens written so far. A finished
        row's later columns are padding and are 0 in that mask.

        Sampling divides logits by ``alignment_config.temperature``,
        read once at the start of this call and stored on the returned
        batch. ``train_step`` gathers log-probs at that same number and
        raises when its config field differs. ``top_p`` only filters
        the tokens that can be drawn. The stored log-probs are
        ``gather_logprobs(logits / temperature)`` on the finished
        sequence, sliced to the response window ``[B, R]``. That is the
        gather ``train_step`` repeats, so at the first update the stored
        policy log-probs match the new ones. A prefix logit and that
        packed gather are not bit-identical, so the prefix logit is not
        what is stored. The nucleus is not applied to the stored value.
        """
        temperature = self.alignment_config.temperature
        if query_tokens.ndim != 2:
            raise ValueError(
                f"query_tokens must be [B, prompt_width], got {tuple(query_tokens.shape)}"
            )
        batch, prompt_width = query_tokens.shape
        device = query_tokens.device
        if prompt_lengths is None:
            prompt_lengths = torch.full((batch,), prompt_width, dtype=torch.long, device=device)
        else:
            prompt_lengths = prompt_lengths.to(device=device, dtype=torch.long)
        if prompt_lengths.shape != (batch,):
            raise ValueError(
                f"prompt_lengths must be [B], got {tuple(prompt_lengths.shape)} for batch {batch}"
            )
        if torch.any((prompt_lengths < 0) | (prompt_lengths > prompt_width)):
            raise ValueError(f"prompt_lengths must lie in [0, prompt_width={prompt_width}]")

        eos_id = self.eos_token_id
        pad_id = self.pad_token_id
        response = torch.full(
            (batch, self.max_response_length), pad_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch, dtype=torch.long, device=device)
        current = query_tokens.clone()

        for step in range(self.max_response_length):
            # `lengths` is the real response written so far. Pads are not in it.
            attention_mask, position_ids = padding_attention(
                prompt_lengths,
                lengths,
                int(current.shape[1]),
                "mixed",
                prompt_width=prompt_width,
            )
            policy_logits = self.policy_model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits[:, -1, :]
            # The reference is scored on the same prefix. Its logits are not
            # stored. The packed gather below is the log-prob train_step uses.
            self.reference_model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            sampled = self._sample_token(
                policy_logits,
                temperature=temperature,
                top_p=self.top_p,
            ).squeeze(1)
            token = torch.where(finished, torch.full_like(sampled, pad_id), sampled)
            response[:, step] = token
            # The EOS token itself is a response token. Padding after it is not.
            lengths = lengths + (~finished).to(dtype=torch.long)
            if eos_id is not None:
                finished = finished | (token == eos_id)
            current = torch.cat([current, token.unsqueeze(1)], dim=1)

        input_ids = current
        token_mask, _gathered = response_token_mask(
            prompt_lengths,
            lengths,
            int(input_ids.shape[1]),
            "mixed",
            prompt_width=prompt_width,
        )
        attention_mask, position_ids = padding_attention(
            prompt_lengths,
            lengths,
            int(input_ids.shape[1]),
            "mixed",
            prompt_width=prompt_width,
        )
        logprobs_policy = self._response_logprobs(
            self.policy_model,
            input_ids,
            attention_mask,
            position_ids,
            temperature,
            prompt_width,
        )
        logprobs_reference = self._response_logprobs(
            self.reference_model,
            input_ids,
            attention_mask,
            position_ids,
            temperature,
            prompt_width,
        )
        reward_out = self.reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        raw = self._decode_reward(reward_out, token_mask)
        last = last_response_index(token_mask).to(device=raw.device)
        scores = raw[torch.arange(batch, device=raw.device), last].clone()
        hit_eos = finished.to(device=scores.device)
        if self.missing_eos_penalty is not None:
            penalty = scores.new_tensor(self.missing_eos_penalty)
            scores = torch.where(hit_eos, scores, scores - penalty)
        rewards = place_sequence_reward(scores, token_mask.to(device=scores.device))

        dones = torch.zeros(rewards.shape, dtype=rewards.dtype, device=rewards.device)
        hit_eos = hit_eos.to(device=dones.device)
        last = last.to(device=dones.device)
        if eos_id is not None and bool(torch.any(hit_eos)):
            rows = torch.arange(batch, device=dones.device)[hit_eos]
            dones[rows, last[hit_eos]] = 1

        rollouts: list[Rollout] = []
        for index in range(batch):
            length = int(lengths[index].item())
            rollouts.append(
                Rollout(
                    query_tokens=query_tokens[index].detach().cpu(),
                    response_tokens=response[index, :length].detach().cpu(),
                    logprobs_policy=logprobs_policy[index, :length].detach().cpu(),
                    logprobs_reference=logprobs_reference[index, :length].detach().cpu(),
                    reward=scores[index].detach().cpu(),
                )
            )

        return RolloutBatch(
            rollouts=rollouts,
            input_ids=input_ids.detach().cpu(),
            prompt_lengths=prompt_lengths.detach().cpu(),
            response_lengths=lengths.detach().cpu(),
            prompt_width=prompt_width,
            padding_side="mixed",
            dones=dones.detach().cpu(),
            rewards=rewards.detach().cpu(),
            token_mask=token_mask.detach().cpu(),
            temperature=temperature,
            logprobs_policy=logprobs_policy.detach().cpu(),
            logprobs_reference=logprobs_reference.detach().cpu(),
        )

    def _response_logprobs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        temperature: float,
        prompt_width: int,
    ) -> torch.Tensor:
        """Gathered log-prob of the response window. Shape ``[B, R]``.

        This is the packed forward ``train_step`` repeats: logits divided
        by ``temperature``, then ``gather_logprobs``, then the columns
        that score response tokens. ``R`` is ``max_response_length``.
        """
        if prompt_width < 1:
            raise ValueError(
                "prompt_width must be >= 1 so the first response token has a logit, "
                f"got {prompt_width}"
            )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        gathered = gather_logprobs(logits / temperature, input_ids)
        response = gathered[:, prompt_width - 1 :]
        if response.shape[1] != self.max_response_length:
            raise ValueError(
                f"response log-probs width {response.shape[1]} must be "
                f"max_response_length {self.max_response_length}"
            )
        return response

    @staticmethod
    def _decode_reward(
        reward_out: object,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sequence score, placed on the last real response token when masked.

        The sequence score is one number per row.

        - Rank-2 ``logits`` of shape ``[B, 1]`` are already that score.
        - Rank-2 ``logits`` of shape ``[B, T]`` are per-token scores. The
          sequence score is the value at the last real response token, not
          the mean over time.
        - Rank-3 ``logits`` of shape ``[B, T, K]`` are per-token scores with
          a feature axis. The sequence score is the mean over ``K`` at the
          last real response token only. Averaging over time mixes padding
          into the reward, so this does not do that. ``K == 1`` is a squeeze.
        - ``pooler_output`` of shape ``[B, H]`` has no time axis. The score
          is the mean over ``H``.

        Without ``token_mask``, per-token scores use the last index. That
        index is the last response token only when the row has no padding
        after the response. ``generate_rollout`` passes a mask. Tokens
        after EOS are padding, and the mask ends on the EOS token.

        With ``token_mask`` ``[B, T]`` (1 on response tokens), the return is
        ``[B, T]``: the sequence score at the last 1, and 0 elsewhere. Right
        padding and the mixed layout both end the response at that 1.

        Without ``token_mask`` the return is the score ``[B]``.
        """
        score = RolloutGenerator._sequence_score(reward_out, token_mask)
        if token_mask is None:
            return score
        return place_sequence_reward(score, token_mask)

    @staticmethod
    def _sequence_score(reward_out: object, token_mask: torch.Tensor | None) -> torch.Tensor:
        if hasattr(reward_out, "logits"):
            logits = reward_out.logits
            if logits.dim() == 1:
                return logits
            if logits.dim() == 2 and logits.shape[-1] == 1:
                return logits.squeeze(-1)
            if logits.dim() == 2:
                return RolloutGenerator._score_at_last_response(logits, token_mask)
            if logits.dim() == 3:
                # Mean over the feature axis only. Time is selected, not averaged.
                per_token = logits.mean(dim=-1)
                return RolloutGenerator._score_at_last_response(per_token, token_mask)

        if hasattr(reward_out, "pooler_output"):
            return reward_out.pooler_output.mean(dim=-1)

        raise RuntimeError("Unsupported reward model output shape")

    @staticmethod
    def _score_at_last_response(
        per_token: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if token_mask is None:
            return per_token[:, -1]
        if token_mask.shape != per_token.shape:
            raise ValueError(
                "token_mask shape must match per-token scores, got "
                f"mask {tuple(token_mask.shape)}, scores {tuple(per_token.shape)}"
            )
        last = last_response_index(token_mask).to(device=per_token.device)
        return per_token.gather(1, last.unsqueeze(1)).squeeze(1)

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Sample the next token. Returns ids of shape ``[batch, 1]``.

        The distribution is ``softmax(logits / temperature)``.
        ``temperature`` is ``AlignmentConfig.temperature``, read by
        ``generate_rollout``. When ``top_p < 1``, tokens past the
        nucleus are dropped before the draw. The stored log-probs stay
        the full ``log_softmax(logits / temperature)`` of the finished
        sequence, not the nucleus.
        """
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
