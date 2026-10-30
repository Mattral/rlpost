# Scripts

`train.py` lives at the repository root, not in this directory.

`train.py` exits with status 2. S0.4 removed `DistributedPPOEngine`, and this script does not call `PPOTrainer`. The PPO update is `rlhf_platform.ppo_engine.PPOTrainer`. Connecting a run is S0.11 in `ROADMAP.md`.

`python train.py` prints a message and exits 2. It does not train.

S0.9 removed the multi-rank script that used to live in this directory. Nothing here is a training run.
