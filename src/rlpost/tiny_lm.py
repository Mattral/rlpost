"""Tiny random-init Llama and a character tokenizer that fits its vocab.

CPU tests, ``run-dpo --tiny``, and ``run-ppo --tiny`` use this model.
It does not download weights. Vocab size is 128. Pad id is 0, bos is 1,
eos is 2. ``TinyValueModel`` is the PPO value head.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

VOCAB = 128


class TinyValueModel(torch.nn.Module):
    """Per-token value head for ``run-ppo --tiny``. Output shape ``[B, T]``.

    An embedding of the tiny vocab and a linear map to one value. No
    dropout. ``seed`` sets ``torch.manual_seed`` before the weights are
    created. A resumed run replaces those weights from the checkpoint.
    """

    def __init__(self, seed: int) -> None:
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be an int >= 0, got {seed!r}")
        torch.manual_seed(seed)
        self.embed = torch.nn.Embedding(VOCAB, 32)
        self.proj = torch.nn.Linear(32, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask, position_ids
        return self.proj(self.embed(input_ids)).squeeze(-1)


def tiny_value_model(seed: int) -> TinyValueModel:
    """Build ``TinyValueModel``. ``seed`` is the run seed."""
    return TinyValueModel(seed)


def tiny_causal_lm(seed: int) -> LlamaForCausalLM:
    """Llama with 2 layers, hidden 32, vocab 128. Weights come from ``seed``."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


class TinyCharTokenizer:
    """One character, one id, inside the tiny Llama vocab.

    ``pad_token_id`` is 0. A character id is ``min(ord(ch), 120)`` and at
    least 1, so a pad column is not a character. This tokenizer does not
    pad and does not insert BOS or EOS. ``collate_completions`` right-pads.
    """

    pad_token_id = 0

    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: int = 32,
        padding: str | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        del truncation, padding, return_tensors, add_special_tokens
        ids = [max(min(ord(ch), 120), 1) for ch in text][:max_length]
        return {"input_ids": ids}
