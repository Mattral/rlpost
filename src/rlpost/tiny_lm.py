"""Tiny random-init Llama and a character tokenizer that fits its vocab.

CPU tests and ``run-dpo --tiny`` both use this model. It does not download
weights. Vocab size is 128. Pad id is 0, bos is 1, eos is 2.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

VOCAB = 128


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
