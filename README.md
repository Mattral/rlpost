# rlpost

Code for language-model post-training: supervised fine-tuning, a reward model, PPO, and DPO.

The repository was formerly [Improving-LLM-Models-with-RLHF-PPO-DPO](https://github.com/Mattral/Improving-LLM-Models-with-RLHF-PPO-DPO).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](pyproject.toml)

## Measured scope

Unit-tested on CPU. No GPU benchmark yet.

Check on 2026-10-30, Python 3.12, torch 2.15.0+cpu, `PYTHONPATH=src`:

- `pytest -q -m "not slow and not gpu"` collects 80 tests: 69 passed, 11 failed.
- The 11 failures are DPO and PPO tests. A judgment on each one is in [ROADMAP.md](ROADMAP.md) under S0.6. They are not marked xfail.
- `ruff check .` and `ruff format --check .` pass under the config in `pyproject.toml`.

Recount after you change tests:

```bash
PYTHONPATH=src pytest --collect-only -q
```

There is no coverage number in this README. A green coverage run has not been recorded.

## What is in the tree

| Path | What it is |
| --- | --- |
| `src/rlpost/config.py` | Pydantic configs. `TrainingConfig.toy_mode()` and `from_yaml`. Tested in `tests/test_config.py`. |
| `src/rlpost/ppo_engine.py` | The only PPO implementation: `PPOTrainer`. GAE, a clipped policy loss, an unclipped value loss, entropy, and a KL term. `train_step` still gathers log-probabilities on the wrong axis. See S0.6. |
| `src/rlpost/dpo_engine.py` | `DPOTrainer`. The CLI does not call it. |
| `src/rlpost/sft_engine.py` | LoRA SFT through the Hugging Face `Trainer`. |
| `src/rlpost/reward_engine.py` | Preference reward-model trainer. |
| `src/rlpost/dataset.py` | Preference-pair loading. Toy mode slices Anthropic HH-RLHF to 1,000 rows when it builds the cache. |
| `src/rlpost/cli.py` | Four Typer commands. Two of them do not train. See below. |
| `train.py` | Exits with status 2. It does not call `PPOTrainer`. |
| `configs/` | YAML for a toy run and a default run. |

Work still open is tracked in [ROADMAP.md](ROADMAP.md).

## DPO

DPO trains on preference pairs and does not train a reward model. The loss uses a frozen reference policy.

`DPOTrainer.compute_dpo_loss` takes sequence scores named `policy_logits_*` and `ref_logits_*` in the signature. The body treats them as log-probabilities:

```
logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
loss = -logsigmoid(logits).mean()
```

That is the DPO objective at sequence level. Parity with TRL, including the log-probability shift and the mask, is item S0.6. It is not done.

`python -m rlpost.cli run-dpo` prints a placeholder and exits 0. It does not construct `DPOTrainer`.

## PPO

`PPOTrainer` in `ppo_engine.py` is the only PPO implementation. The step is GAE, then a clipped probability ratio, an unclipped value loss, an entropy bonus, and a KL penalty. `compute_kl_divergence(policy_logits, ref_logits)` evaluates the full-vocabulary KL(reference || policy), `sum_a p_ref(a) (log p_ref(a) - log p_policy(a))`. The sampled-token estimators usually called k1, k2, and k3 are not implemented. Naming that estimator is S0.6e.

`train_step` gathers log-probabilities with `gather` on dimension 1 and does not shift for the causal offset, so the step test raises. That bug is S0.6a. It lives in this one function.

## Command line

```bash
python -m rlpost.cli train-sft --config configs/toy.yaml
python -m rlpost.cli train-reward --config configs/toy.yaml
python -m rlpost.cli run-ppo --toy
python -m rlpost.cli run-dpo --toy
```

`train-sft` builds `SFTTrainer` and calls `train`. `train-reward` builds `RewardModelTrainer` and calls `train`. Both will download models and the HH-RLHF dataset. Neither command has a recorded runtime in this repo.

`run-ppo` checks that `alignment.method` is `ppo`, writes the config YAML, and returns. The function body says the training loop is not wired up.

`run-dpo` does not train.

`TrainingConfig.toy_mode()` sets the policy and the reference model to `distilgpt2`, the reward model to `prajjwal1/bert-tiny`, the device to `cuda`, and the dataset name to `Anthropic/hh-rlhf`. Those are config defaults. They are not a measured run.

## Install

Required packages are the ones the code imports: torch, transformers, datasets, pydantic, PyYAML, typer, and rich. Lower bounds are in `pyproject.toml`. They accept the versions in `requirements-cpu.txt`.

```bash
pip install -e ".[test]"
```

`[test]` pins pytest 9.2.0, ruff 0.17.3, and TRL 1.17.1. `[lora]` adds peft for the SFT and reward trainers. The CPU job installs `.[test]` under `requirements-cpu.txt` and the CPU torch index. See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

The measured-scope check above used a virtualenv with the CPU wheel of torch 2.15.0 and `PYTHONPATH=src`. It did not run `pip install -e .`.

## Layout

```
src/rlpost/            package
configs/               toy and default YAML
tests/                 CPU unit tests; the full run is not green
docs/                  notes and older write-ups
train.py               exits 2; does not call PPOTrainer
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
