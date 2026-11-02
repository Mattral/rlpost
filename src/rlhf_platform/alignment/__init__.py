"""Rollout storage and sampling.

PPO updates live in ``rlhf_platform.ppo_engine``.
This package re-exports the rollout buffer and the token sampler.
"""

from . import rollout

__all__ = ["rollout"]
