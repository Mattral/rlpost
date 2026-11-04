"""Per-token logprob gather.

Parity target: trl==1.17.1.
The helper is ``trl.trainer.utils.selective_log_softmax``. It gathers
``log_softmax`` at ``index`` and does not shift. Callers that need the causal
offset pass ``logits[:, :-1]`` and ``input_ids[:, 1:]``.

TRL 1.17.1's ``DPOTrainer`` does not call that helper. It reads
``outputs.log_probs`` from a fused LM head and shifts the completion mask
with ``completion_mask[..., 1:]``. The helper is what the other trainers call,
and it is the logprob implementation in that release.

fp32 absolute tolerance is 1e-6. The bf16 case states its own tolerance.
"""

import torch
import torch.nn.functional as F
from tests.tiny_lm import VOCAB, tiny_causal_lm
from trl.trainer.utils import selective_log_softmax

from rlpost.ops import gather_logprobs

FP32_ATOL = 1e-6
# CPU, torch 2.15, this Llama: max |bf16 gather - fp32 gather| was 0.0173.
BF16_ATOL = 2e-2


def _batch(seed: int, batch: int = 2, length: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = tiny_causal_lm(seed)
    input_ids = torch.randint(0, VOCAB, (batch, length))
    logits = model(input_ids).logits
    return logits, input_ids


def _plain_shifted_gather(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Reference written here. It does not call rlpost."""
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:].unsqueeze(-1)
    return torch.gather(log_probs, -1, labels).squeeze(-1)


def test_trl_pin() -> None:
    import trl

    assert trl.__version__ == "1.17.1"


def test_parity_vs_plain_reference() -> None:
    logits, input_ids = _batch(0)
    got = gather_logprobs(logits, input_ids)
    ref = _plain_shifted_gather(logits, input_ids)
    assert got.shape == (input_ids.shape[0], input_ids.shape[1] - 1)
    assert torch.allclose(got, ref, atol=FP32_ATOL, rtol=0)


def test_parity_vs_trl_selective_log_softmax() -> None:
    logits, input_ids = _batch(1)
    got = gather_logprobs(logits, input_ids)
    trl_logps = selective_log_softmax(logits[:, :-1, :], input_ids[:, 1:])
    assert got.shape == trl_logps.shape
    assert torch.allclose(got, trl_logps, atol=FP32_ATOL, rtol=0)


def test_shift_fails_if_removed() -> None:
    """Position 0 is peaked on the *next* id. An unshifted gather reads the current id."""
    vocab = 16
    input_ids = torch.tensor([[1, 7, 2, 4]])
    logits = torch.full((1, 4, vocab), -30.0)
    logits[0, 0, input_ids[0, 1]] = 30.0
    logits[0, 1, input_ids[0, 2]] = 30.0
    logits[0, 2, input_ids[0, 3]] = 30.0

    got = gather_logprobs(logits, input_ids)
    next_from_first = F.log_softmax(logits[0, 0], dim=-1)[input_ids[0, 1]]
    current_from_first = F.log_softmax(logits[0, 0], dim=-1)[input_ids[0, 0]]

    assert got.shape == (1, 3)
    assert torch.allclose(got[0, 0], next_from_first, atol=FP32_ATOL, rtol=0)
    assert got[0, 0] > current_from_first + 10

    unshifted = _plain_unshifted(logits, input_ids)
    assert not torch.allclose(got, unshifted[:, : got.shape[1]], atol=1.0, rtol=0)


def _plain_unshifted(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)


def test_right_and_left_padding_mask_after_gather() -> None:
    logits, input_ids = _batch(2, batch=2, length=8)
    logps = gather_logprobs(logits, input_ids)
    right = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=logps.dtype,
    )
    left = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=logps.dtype,
    )

    right_masked = logps * right[:, 1:]
    left_masked = logps * left[:, 1:]

    assert torch.all(right_masked[right[:, 1:] == 0] == 0)
    assert torch.all(left_masked[left[:, 1:] == 0] == 0)
    assert torch.allclose(right_masked[right[:, 1:] == 1], logps[right[:, 1:] == 1])
    assert torch.allclose(left_masked[left[:, 1:] == 1], logps[left[:, 1:] == 1])

    # The two masks keep different positions, so the masked tensors differ.
    assert not torch.equal(right[:, 1:], left[:, 1:])
    assert not torch.allclose(right_masked, left_masked)


def test_bf16_against_fp32() -> None:
    logits, input_ids = _batch(3)
    fp32 = gather_logprobs(logits.float(), input_ids)
    bf16 = gather_logprobs(logits.to(torch.bfloat16), input_ids)
    assert bf16.dtype == torch.bfloat16
    assert torch.allclose(bf16.float(), fp32, atol=BF16_ATOL, rtol=0)
    # The tolerance is looser than fp32, and still tighter than a dropped shift.
    assert BF16_ATOL < 1.0
