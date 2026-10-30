# rlpost

Code for language-model post-training: supervised fine-tuning, a reward model, PPO, and DPO.

The GitHub repository was renamed from [Improving-LLM-Models-with-RLHF-PPO-DPO](https://github.com/Mattral/Improving-LLM-Models-with-RLHF-PPO-DPO). The Python package is still `rlhf_platform`. Renaming the package is item S0.5 in [ROADMAP.md](ROADMAP.md).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](pyproject.toml)

## Measured scope

Unit-tested on CPU. No GPU benchmark yet.

Check on 2026-10-30, Python 3.12, torch 2.15.0+cpu, `PYTHONPATH=src`:

- `pytest -q -m "not slow and not gpu"` collects 82 tests: 69 passed, 13 failed.
- The failures are in DPO metrics, PPO `train_step`, and two rollout tests. Notes are in [ROADMAP.md](ROADMAP.md) under S0.6 and S0.9.
- `ruff check .` and `ruff format --check .` pass under the config in `pyproject.toml`.

Recount after you change tests:

```bash
PYTHONPATH=src pytest --collect-only -q
```

There is no coverage number in this README. A green coverage run has not been recorded.

## What is in the tree

| Path | What it is |
| --- | --- |
| `src/rlhf_platform/config.py` | Pydantic configs. `TrainingConfig.toy_mode()` and `from_yaml`. Tested in `tests/test_config.py`. |
| `src/rlhf_platform/ppo_engine.py` | `PPOTrainer`: GAE, a clipped policy loss, a value loss, entropy, and a KL term. Some of its tests fail. See S0.6. |
| `src/rlhf_platform/alignment/ppo_engine.py` | A second PPO implementation, `DistributedPPOEngine`. S0.4 merges the two. |
| `src/rlhf_platform/alignment/loss.py` | Categorical KL from full logits, GAE, and a clipped PPO loss. Not parity-tested against TRL yet. |
| `src/rlhf_platform/dpo_engine.py` | `DPOTrainer`. The CLI does not call it. |
| `src/rlhf_platform/sft_engine.py` | LoRA SFT through the Hugging Face `Trainer`. |
| `src/rlhf_platform/reward_engine.py` | Preference reward-model trainer. |
| `src/rlhf_platform/dataset.py` | Preference-pair loading. Toy mode slices Anthropic HH-RLHF to 1,000 rows when it builds the cache. |
| `src/rlhf_platform/cli.py` | Four Typer commands. Two of them do not train. See below. |
| `src/rlhf_platform/distributed/` | Topology, communication hooks, and async checkpoints. The package imports on torch 2.15. The hooks are not covered by a passing test. |
| `train.py` | Distributed training entrypoint. It has not been run on a cluster in this repo. |
| `configs/` | YAML for a toy run, a default run, a cluster topology, and a DeepSpeed ZeRO-3 config. |

Work still open is tracked in [ROADMAP.md](ROADMAP.md).

## DPO

DPO trains on preference pairs and does not train a reward model. The loss uses a frozen reference policy.

`DPOTrainer.compute_dpo_loss` takes sequence scores named `policy_logits_*` and `ref_logits_*` in the signature. The body treats them as log-probabilities:

```
logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
loss = -logsigmoid(logits).mean()
```

That is the DPO objective at sequence level. Parity with TRL, including the log-probability shift and the mask, is item S0.6. It is not done.

`python -m rlhf_platform.cli run-dpo` prints a placeholder and exits 0. It does not construct `DPOTrainer`.

## PPO

`PPOTrainer` in `ppo_engine.py` is the implementation the current unit tests call. The intended step is GAE, then a clipped probability ratio, a value loss, an entropy bonus, and a KL penalty. `train_step` currently gathers log-probabilities with the wrong tensor rank, so the step test raises. Details are in the S0.6 notes.

`alignment/loss.py` computes KL as the full-vocabulary categorical KL, `sum_i p_i (log p_i - log q_i)`, and clamps it to `[0, 1000]`. The sampled-token estimators usually called k1, k2, and k3 are not implemented here. Naming and testing that estimator is S0.6e.

## Command line

```bash
python -m rlhf_platform.cli train-sft --config configs/toy.yaml
python -m rlhf_platform.cli train-reward --config configs/toy.yaml
python -m rlhf_platform.cli run-ppo --toy
python -m rlhf_platform.cli run-dpo --toy
```

`train-sft` builds `SFTTrainer` and calls `train`. `train-reward` builds `RewardModelTrainer` and calls `train`. Both will download models and the HH-RLHF dataset. Neither command has a recorded runtime in this repo.

`run-ppo` checks that `alignment.method` is `ppo`, writes the config YAML, and returns. The function body says the training loop is not wired up.

`run-dpo` does not train.

`TrainingConfig.toy_mode()` sets the policy and the reference model to `distilgpt2`, the reward model to `prajjwal1/bert-tiny`, the device to `cuda`, and the dataset name to `Anthropic/hh-rlhf`. Those are config defaults. They are not a measured run.

## Install

`pyproject.toml` pins an old stack, including `transformers==4.32.0`, `trl==0.5.0`, `bitsandbytes==0.41.1`, and `triton==3.7.0`. `pip install -e .` was not run for the check in Measured scope.

That check used a virtualenv with the CPU wheel of torch 2.15.0 and current `pydantic`, `pytest`, `pyyaml`, `transformers`, and `datasets`, imported via `PYTHONPATH=src`.

## Distributed design

Process-group layout, bucket sizes, and context lengths for 8, 512, and 10,000 GPUs are written down as design targets in [docs/design.md](docs/design.md). They are not measurements. No run in this repo has produced a throughput, memory, or latency number.

## Layout

```
src/rlhf_platform/     package (import name until S0.5)
configs/               toy, default, topology, and DeepSpeed YAML
tests/                 CPU unit tests; the full run is not green
docs/                  design notes and older write-ups
train.py               distributed entrypoint
ROADMAP.md             status of the work
```

## Citation

```bibtex
@software{rlpost_2026,
  title  = {rlpost},
  author = {Mattral and contributors},
  year   = {2026},
  url    = {https://github.com/Mattral/rlpost}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
