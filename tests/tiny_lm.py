"""Re-export the tiny Llama used by the CPU tests.

Not collected by pytest (``python_files = test_*.py``). The builder lives
in ``rlpost.tiny_lm`` so ``run-dpo --tiny`` can construct the same model.
"""

from rlpost.tiny_lm import (
    VOCAB,
    TinyCharTokenizer,
    tiny_causal_lm,
    tiny_reward_model,
    tiny_value_model,
)

__all__ = [
    "VOCAB",
    "TinyCharTokenizer",
    "tiny_causal_lm",
    "tiny_reward_model",
    "tiny_value_model",
]
