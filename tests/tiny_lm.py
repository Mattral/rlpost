"""Tiny random-init causal LM for CPU tests.

Not collected by pytest (``python_files = test_*.py``).
"""

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
