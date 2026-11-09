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


def _prompt_width(prompt_width: int | torch.Tensor) -> int:
    """One padded-query width. A Python int, or a single int64 tensor."""
    if isinstance(prompt_width, torch.Tensor):
        if prompt_width.dtype != torch.long or prompt_width.numel() != 1:
            raise ValueError(
                "prompt_width must be one int64, got "
                f"dtype {prompt_width.dtype} and shape {tuple(prompt_width.shape)}"
            )
        width = int(prompt_width.reshape(()).item())
    elif isinstance(prompt_width, bool) or not isinstance(prompt_width, int):
        raise TypeError(f"prompt_width must be an int, got {type(prompt_width).__name__}")
    else:
        width = prompt_width
    if width < 0:
        raise ValueError(f"prompt_width must be >= 0, got {width}")
    return width


def response_token_mask(
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
    seq_len: int,
    padding_side: str = "right",
    *,
    prompt_width: int | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Response-token masks for one rollout batch.

    A generated response token is 1. A prompt token is 0. Padding is 0.
    Lengths come from the rollout, not from an attention mask. An attention
    mask is 1 on the prompt as well, and that would keep prompt tokens in
    the loss.

    ``"right"`` lays each row out as ``[prompt | response | pad ...]``.
    The response starts at ``prompt_lengths`` and ends at
    ``prompt_lengths + response_lengths``. ``prompt_width`` is not read.

    ``"mixed"`` is what a padded ``generate`` batch actually is:
    ``[pad | prompt | response | pad]``. ``prompt_width`` is the padded
    query tensor's width (``query_tokens.shape[1]``). Prompts are
    left-padded out to that width, so every response starts at column
    ``prompt_width`` and ends at ``prompt_width + response_lengths``,
    which is where that row hit EOS. The tail is a right pad. The width
    is not ``max(prompt_lengths)``. A batch can pad every prompt to 8
    while the longest real prompt is 5, and the response window still
    starts at 8. Using the longest prompt would treat the extra left pad
    as response tokens. A suffix layout ``[pad | prompt | response]``
    (response flush with the end of the tensor) is not produced by that
    generation, and nothing in this repository reads it.

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
            Each entry is ``<= prompt_width`` on the mixed layout.
        response_lengths: Real response tokens per row, int64, shape ``[B]``.
        seq_len: ``T``, the padded length of ``input_ids``.
        padding_side: ``"right"`` or ``"mixed"``.
        prompt_width: Padded query width. Required for ``"mixed"``. Not
            ``max(prompt_lengths)``. Ignored for ``"right"``.

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
        # The padded query width, not the longest prompt. A shorter prompt
        # has more leading pad, and every response still starts at `width`.
        if prompt_width is None:
            raise ValueError(
                "mixed layout needs prompt_width, the padded query tensor width, "
                "not max(prompt_lengths)"
            )
        width = _prompt_width(prompt_width)
        if torch.any(prompt_lengths > width):
            raise ValueError("prompt_lengths cannot exceed prompt_width, the padded query width")
        response_start = torch.full_like(prompt_lengths, width)
        if torch.any(response_start + response_lengths > seq_len):
            raise ValueError("mixed layout needs prompt_width + response_length <= seq_len")
    response_end = response_start + response_lengths
    token_mask = (positions >= response_start.unsqueeze(1)) & (
        positions < response_end.unsqueeze(1)
    )
    token_mask = token_mask.to(dtype=torch.float32)
    return token_mask, token_mask[:, 1:]
