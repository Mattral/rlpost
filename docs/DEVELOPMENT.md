# Development

How to clone the repo and run the CPU checks. The status of the work is [ROADMAP.md](../ROADMAP.md). What each command actually does is in the [README](../README.md).

## Clone

```bash
git clone https://github.com/Mattral/rlpost.git
cd rlpost
```

`pyproject.toml` says `requires-python` is `>=3.11`. The CPU job uses Python 3.12, because `requirements-cpu.txt` pins `numpy==2.5.4`, which requires Python 3.12 or newer. `rlpost.repro` imports numpy. The project dependency is `numpy>=1.26`, which that pin satisfies. The pin stays in the lock so the job resolves one version.

## CPU tests

Install the package and the `[test]` extra under the CPU lock:

```bash
python -m pip install -e ".[test]" -c requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements-cpu.txt` pins `torch==2.15.0+cpu` from the CPU wheel index, plus pytest, ruff, pydantic, PyYAML, numpy, transformers, datasets, accelerate, and `trl==1.17.1`. The accelerate pin stays because TRL requires it. rlpost does not import accelerate.

Without the lock, the same extra is:

```bash
pip install -e ".[test]"
```

That command resolves torch from PyPI. The `[lora]` extra adds `peft`. No current trainer applies LoRA. There is no `[train]` extra. Weights & Biases is optional: the DPO and PPO engines import it only when it is installed and logging is on.

Then:

```bash
ruff check .
ruff format --check .
pytest -q
```

Ruff selects E, F, and I, and leaves line length to the formatter (`E501` is ignored). The pytest options in `pyproject.toml` skip tests marked `slow` or `gpu`.

The default CPU pytest run on this branch is green. Older red runs are recorded under S0.6 in [ROADMAP.md](../ROADMAP.md). They are not marked xfail.

## CI

`.github/workflows/ci.yml` has one job, `CPU torch, ruff, pytest`. It runs on pull requests and on pushes to `main`. The steps are Python 3.12, `pip install -e ".[test]" -c requirements-cpu.txt` with the CPU torch index, `ruff check`, `ruff format --check`, and `pytest -q -m "not slow and not gpu"`. The install pulls the dependencies declared on the package.

## Pascal GPUs

A Pascal GPU (compute capability sm_61) needs the cu126 PyTorch wheel. PyTorch builds for CUDA 12.8 and newer dropped Pascal.

The cu126 index has no `torch==2.15.0` wheel. The newest stable wheel on that index is `2.14.1`:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
```

The package floor stays `torch>=2.15`. That floor was checked on torch 2.15. `torch==2.14.1` is the Pascal wheel for fp32 smoke tests on a tiny model. It is not a benchmark. The default test run stays on CPU.

## Commands

`train-sft --tiny` and `train-reward --tiny` train `tiny_lm` on CPU and write `{output_dir}/run.json`. Without `--tiny` each exits non-zero and does not download a model. `run_sft_training` and `run_reward_training` are the only SFT and reward training paths.

`run-ppo --tiny` trains `tiny_lm` on CPU and writes `{output_dir}/run.json`. The reference is a frozen copy of the initial policy. Without `--tiny` it exits non-zero and does not download a model.

`run-dpo --tiny` trains `tiny_lm` on CPU and writes `{output_dir}/run.json`. Without `--tiny` it exits non-zero and does not download a model.

`python train.py` exits 2.

## Reporting a vulnerability

See [SECURITY.md](../SECURITY.md).
