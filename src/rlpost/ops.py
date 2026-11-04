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

    Masking is not applied. After this call, multiply by the mask of the
    scored tokens, ``attention_mask[:, 1:]``, to drop padding.
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
