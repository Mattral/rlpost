# Development

How to clone the repo and run the CPU checks. The status of the work is [ROADMAP.md](../ROADMAP.md). What each command actually does is in the [README](../README.md).

## Clone

```bash
git clone https://github.com/Mattral/rlpost.git
cd rlpost
```

`pyproject.toml` says `requires-python` is `>=3.11`. The CPU job uses Python 3.12, because `requirements-cpu.txt` pins `numpy==2.5.4`, which requires Python 3.12 or newer.

## CPU tests

The CI job installs the CPU file, then the package without pulling the older pins in `pyproject.toml`:

```bash
python -m pip install -r requirements-cpu.txt
pip install -e . --no-deps
```

`requirements-cpu.txt` pins `torch==2.15.0+cpu` from the CPU wheel index, plus pytest, ruff, pydantic, PyYAML, numpy, transformers, datasets, and accelerate.

Then:

```bash
ruff check .
ruff format --check .
pytest -q
```

Ruff selects E, F, and I, and leaves line length to the formatter (`E501` is ignored). The pytest options in `pyproject.toml` skip tests marked `slow` or `gpu`.

The failures that are still red are listed under S0.6 in [ROADMAP.md](../ROADMAP.md). They are not marked xfail.

## CI

`.github/workflows/ci.yml` has one job, `CPU torch, ruff, pytest`. It runs on pull requests and on pushes to `main`. The steps are Python 3.12, `requirements-cpu.txt`, `pip install -e . --no-deps`, `ruff check`, `ruff format --check`, and `pytest -q -m "not slow and not gpu"`.

## Pascal GPUs

A Pascal GPU (compute capability sm_61) needs the cu126 PyTorch wheel. PyTorch builds for CUDA 12.8 and newer dropped Pascal.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

That wheel is for fp32 smoke tests on a tiny model. It is not a benchmark. The default test run stays on CPU.

## Commands

`python -m rlhf_platform.cli train-sft` and `train-reward` construct their trainers and call `train`. Both will try to download models and data. Neither has a recorded CPU run.

`run-ppo` checks that `alignment.method` is `ppo`, writes the config YAML, and returns. It does not call `PPOTrainer`.

`run-dpo` prints a placeholder and exits 0. It does not construct `DPOTrainer`.

`python train.py` exits 2.

## Reporting a vulnerability

See [SECURITY.md](../SECURITY.md).
