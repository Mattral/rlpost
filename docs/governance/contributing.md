# Contributing

Branch from `main`. One item from [ROADMAP.md](../../ROADMAP.md) per pull request.
Commits use Conventional Commits: `fix:`, `test:`, `docs:`, `refactor:`, `feat:`, `chore:`, `ci:`.

Every pull request updates `ROADMAP.md`: status, notes, and a change-log line.

## Checks

The CI job is `CPU torch, ruff, pytest` in `.github/workflows/ci.yml`.
It installs `requirements-cpu.txt` (the CPU wheel of torch) and runs:

```bash
ruff check .
ruff format --check .
pytest -q
```

`pytest -q` skips tests marked `slow` or `gpu`. The job uses Python 3.12.

Failures listed under S0.6 in `ROADMAP.md` stay red. Do not mark them xfail or skip.

## Numbers

A throughput, memory, latency, reward, win-rate, or speedup figure in the README, docs, or code comments must come from a file committed under `benchmarks/results/`. Cite that file next to the number. Label a design target "design target, not measured".

A new public command-line option in `train.py` is documented in `scripts/README.md`.
