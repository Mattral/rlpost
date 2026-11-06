"""Tensor ops shared by the trainers.

The log-probability gather lives here so PPO and DPO do not each keep a copy.
"""

import torch
import torch.nn.functional as F


def gather_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Log-probability of each next token under a causal language model.

    Args:
        logits: Unnormalized scores, shape ``[B, T, V]``. Position ``t``
            scores the token at index ``t + 1``.
        input_ids: Token ids, shape ``[B, T]``, same ``B`` and ``T`` as
            ``logits``. Values must be in ``[0, V)``.

    Returns:
        Per-token log-probabilities, shape ``[B, T - 1]``. Index ``t`` is
        ``log p(input_ids[:, t + 1] | logits[:, t, :])``.

    Masking is not applied. The caller multiplies by the mask of the tokens
    that should count. For PPO that mask is ``response_token_mask``'s
    gathered half: index ``t`` is 1 when ``input_ids[:, t + 1]`` is a
    generated response token, and 0 when it is prompt or padding.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if input_ids.ndim != 2 or input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"input_ids shape {tuple(input_ids.shape)} must match logits batch "
            f"and time {tuple(logits.shape[:2])}"
        )

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return gathered


def response_token_mask(
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
    seq_len: int,
    padding_side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Response-token masks for one rollout batch.

    A generated response token is 1. A prompt token is 0. Padding is 0.
    Lengths come from the rollout, not from an attention mask. An attention
    mask is 1 on the prompt as well, and that would keep prompt tokens in
    the loss.

    ``"right"`` lays each row out as ``[prompt | response | pad ...]``.
    The response starts at ``prompt_lengths`` and ends at
    ``prompt_lengths + response_lengths``.

    ``"mixed"`` is what a padded ``generate`` batch actually is:
    ``[pad | prompt | response | pad]``. Prompts are left-padded to one
    common width ``P = max(prompt_lengths)``, so the longest prompt has
    no leading pad and every response starts at column ``P``. The response
    ends at ``P + response_lengths``, which is where that row hit EOS, and
    the rest of the row is a right pad. A suffix layout
    ``[pad | prompt | response]`` (response flush with the end of the
    tensor) is not produced by that generation, and nothing in this
    repository reads it.

    ``token_mask`` has shape ``[B, seq_len]``. Index ``t`` is 1 when
    ``input_ids[:, t]`` is a response token. The rollout writes the
    sequence score on the last 1. PPO does not reduce with this mask.

    ``gathered_mask`` has shape ``[B, seq_len - 1]``. Index ``t`` is 1 when
    ``input_ids[:, t + 1]`` is a response token. That is the action layout,
    the same alignment as ``gather_logprobs``: logits at ``t`` score token
    ``t + 1``, and the value at ``t`` is the state before that token.
    Policy loss, entropy, KL, GAE, the value loss, and advantage whitening
    all use this mask. It equals ``token_mask[:, 1:]``. Token 0 is not an
    action in the gather, so it has no column.

    Args:
        prompt_lengths: Real prompt tokens per row, int64, shape ``[B]``.
        response_lengths: Real response tokens per row, int64, shape ``[B]``.
        seq_len: ``T``, the padded length of ``input_ids``.
        padding_side: ``"right"`` or ``"mixed"``.

    Returns:
        token_mask: float 0/1, shape ``[B, T]``.
        gathered_mask: float 0/1, shape ``[B, T - 1]``.
    """
    if prompt_lengths.ndim != 1 or response_lengths.shape != prompt_lengths.shape:
        raise ValueError(
            "prompt_lengths and response_lengths must both be [B], got "
            f"{tuple(prompt_lengths.shape)} and {tuple(response_lengths.shape)}"
        )
    if prompt_lengths.dtype != torch.long or response_lengths.dtype != torch.long:
        raise ValueError(
            "prompt_lengths and response_lengths must be int64, got "
            f"{prompt_lengths.dtype} and {response_lengths.dtype}"
        )
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if padding_side not in ("right", "mixed"):
        raise ValueError(f"padding_side must be 'right' or 'mixed', got {padding_side!r}")
    if torch.any(prompt_lengths < 0) or torch.any(response_lengths < 0):
        raise ValueError("prompt_lengths and response_lengths must be >= 0")
    if torch.any(prompt_lengths + response_lengths > seq_len):
        raise ValueError("prompt_length + response_length exceeds seq_len")

    batch = prompt_lengths.shape[0]
    positions = torch.arange(seq_len, device=prompt_lengths.device).unsqueeze(0).expand(batch, -1)
    if padding_side == "right":
        response_start = prompt_lengths
    else:
        # One prompt width for the batch. A shorter prompt has leading pads.
        prompt_width = prompt_lengths.max()
        if torch.any(prompt_width + response_lengths > seq_len):
            raise ValueError("mixed layout needs max(prompt_lengths) + response_length <= seq_len")
        response_start = prompt_width.expand_as(prompt_lengths)
    response_end = response_start + response_lengths
    token_mask = (positions >= response_start.unsqueeze(1)) & (
        positions < response_end.unsqueeze(1)
    )
    token_mask = token_mask.to(dtype=torch.float32)
    return token_mask, token_mask[:, 1:]
