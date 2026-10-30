#!/usr/bin/env python3
"""Distributed training entry point.

S0.4 removed ``DistributedPPOEngine``. This script does not call ``PPOTrainer``.
The old body loaded four models, started rollout threads, and stepped that
engine. Nothing in this repo had run it. Wiring a real PPO run is S0.11.
"""

import sys


def main() -> int:
    """Exit. PPO updates are ``rlhf_platform.ppo_engine.PPOTrainer``."""
    print(
        "train.py does not run PPO. "
        "S0.4 removed DistributedPPOEngine. "
        "The update is rlhf_platform.ppo_engine.PPOTrainer. "
        "Connecting this script is S0.11.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
