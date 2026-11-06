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
    that should count. For PPO that mask is the gathered half of
    ``response_token_mask``: index ``t`` is 1 when ``input_ids[:, t + 1]``
    is a generated response token, and 0 when it is prompt or padding.
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

    Right padding lays each row out as ``[prompt | response | pad ...]``.
    Left padding lays each row out as ``[pad ... | prompt | response]``.
    The response is contiguous. On the left it is a suffix of length
    ``response_lengths``. On the right it starts at ``prompt_lengths``.

    ``token_mask`` has shape ``[B, seq_len]``. Index ``t`` is 1 when
    ``input_ids[:, t]`` is a response token. GAE and the value loss are
    indexed the same way: ``rewards[:, t]`` is the reward for that token.

    ``gathered_mask`` has shape ``[B, seq_len - 1]``. Index ``t`` is 1 when
    ``input_ids[:, t + 1]`` is a response token. That is the same alignment
    as ``gather_logprobs``: logits at ``t`` score token ``t + 1``, and the
    policy ratio, the KL, and the entropy of that distribution all sit on
    this index. ``gathered_mask`` equals ``token_mask[:, 1:]``. Token 0 is
    not an action in the gather, so it has no column. A non-empty prompt
    puts a prompt token or left padding at index 0, and ``token_mask[:, 0]``
    is 0 there.

    Args:
        prompt_lengths: Real prompt tokens per row, int64, shape ``[B]``.
        response_lengths: Real response tokens per row, int64, shape ``[B]``.
        seq_len: ``T``, the padded length of ``input_ids``.
        padding_side: ``"right"`` or ``"left"``.

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
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    if torch.any(prompt_lengths < 0) or torch.any(response_lengths < 0):
        raise ValueError("prompt_lengths and response_lengths must be >= 0")
    if torch.any(prompt_lengths + response_lengths > seq_len):
        raise ValueError("prompt_length + response_length exceeds seq_len")

    batch = prompt_lengths.shape[0]
    positions = torch.arange(seq_len, device=prompt_lengths.device).unsqueeze(0).expand(batch, -1)
    if padding_side == "right":
        response_start = prompt_lengths
    else:
        response_start = seq_len - response_lengths
    response_end = response_start + response_lengths
    token_mask = (positions >= response_start.unsqueeze(1)) & (
        positions < response_end.unsqueeze(1)
    )
    token_mask = token_mask.to(dtype=torch.float32)
    return token_mask, token_mask[:, 1:]
