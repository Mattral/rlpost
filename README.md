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
| `src/rlpost/ops.py` | `gather_logprobs`: logits at position `t` score token `t+1`. Shape `[B, T, V]` in, `[B, T-1]` out. |
| `src/rlpost/ppo_engine.py` | The only PPO implementation: `PPOTrainer`. GAE, a clipped policy loss, a value loss (unclipped unless `value_clip` is set), entropy, and a KL term. `train_step` uses `gather_logprobs`. |
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

`log π(y|x)` is the sum of completion-token log-probabilities. `compute_log_probs` calls `gather_logprobs` (position `t` scores token `t+1`) and multiplies by `completion_mask[:, 1:]`. Prompt tokens and padding contribute 0. The reduction is a sum, not a mean.

The sigmoid loss, label smoothing 0:

```
chosen_reward   = beta * (logp_policy_chosen - logp_ref_chosen)
rejected_reward = beta * (logp_policy_rejected - logp_ref_rejected)
loss            = -logsigmoid(chosen_reward - rejected_reward).mean()
```

`beta` is `AlignmentConfig.dpo_beta` (default 0.1, must be > 0). It is not the PPO `kl_coefficient`. The margin is the mean of `chosen_reward - rejected_reward`. Accuracy is the fraction of rows with `chosen_reward > rejected_reward`.

Parity is against TRL 1.17.1 `DPOTrainer._compute_loss` (`loss_type` sigmoid, `label_smoothing` 0). That method sums completion-token log-probs the same way. On CPU, torch 2.15, the tiny Llama, the loss matched within `1e-6`. The logged mean of the sums matched within `1e-5`. The measured gap on that mean was `1.907e-6`.

`python -m rlpost.cli run-dpo` prints a placeholder and exits 0. It does not construct `DPOTrainer`.

## PPO

`PPOTrainer` in `ppo_engine.py` is the only PPO implementation. The step is GAE, then a clipped probability ratio, a value loss, an entropy bonus, and a KL penalty. With `value_clip` unset, the value term is `value_loss_coefficient * mean((V - R)^2)`. With `value_clip = c` (`c >= 0`), it is `value_loss_coefficient * mean(0.5 * max((V - R)^2, (V_clip - R)^2))`, where `V_clip = V_old + clip(V - V_old, -c, c)`. `c` is not `ppo_epsilon`. `compute_kl_divergence(policy_logits, ref_logits)` evaluates the full-vocabulary KL(reference || policy), `sum_a p_ref(a) (log p_ref(a) - log p_policy(a))`. The sampled-token estimators usually called k1, k2, and k3 are not implemented. Naming that estimator is S0.6e.

`train_step` calls `gather_logprobs`. Logits at position `t` score token `t+1`, and the policy ratio uses those log-probabilities, shape `[B, T-1]`. The two step tests run on a tiny causal LM. The clipped surrogate, both value losses, and the entropy term are tested. The entropy scalar is the mean of TRL 1.17.1 `entropy_from_logits`.

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
