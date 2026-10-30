# Contributing Guide: Enterprise Engineering Gateways

## Purpose

This document defines the contribution and merge governance rules for the RLHF infrastructure framework. It is designed for core contributors and engineering teams operating under strict production constraints: deterministic tensor shapes, invariant regression checks, and fully auditable continuous integration.

---

## Code Quality & Pre-Commit Gatekeepers

All pull requests must pass the following automated gate checks before review:

1. **Formatting**
   - `black --line-length 88` for Python formatting.
   - `ruff check --fix` for syntax, style, and basic static issues.

2. **Static Type Enforcement**
   - `mypy src/rlhf_platform --strict` for type correctness.
   - All public APIs must expose explicit type hints; no `Any` unless justified.

3. **Security Linting**
   - `ruff check src/rlhf_platform --select=S` for security-sensitive patterns.
   - High-risk modules (`distributed`, `rollout`, `async_io`) must pass `bandit` or equivalent static analysis.

4. **Pre-Commit Hooks**
   - Contributors must install and use `.pre-commit-config.yaml`.
   - Required hooks:
     - `black`
     - `ruff`
     - `mypy`
     - `isort`
     - `end-of-file-fixer`
     - `trailing-whitespace`

5. **Release Branch Validation**
   - `main` and `release/*` branches require all checks green.
   - Feature branches may be merged only after at least one approval from a core maintainer.

---

## Tensor Shape Invariance & Regression Validation

The RLHF platform must remain robust under varying batch sizes, sequence lengths, and device placements. The following checks are mandatory:

### 1. Matrix Shape Assertion Tests

- Every tensor transformation in `src/rlhf_platform/` must include explicit shape assertions in unit tests.
- Examples:
  - `batch['input_ids'].shape == (batch_size, seq_len)`
  - `logits.shape == (batch_size, seq_len, vocab_size)`
  - `advantages.shape == (batch_size,)`

### 2. Shape Invariance Rules

- `RolloutCollator` pads queries to the longest query in the batch and responses to `max_response_length`.
- `ppo_engine` losses must preserve the same rank ordering as the input batch.
- `dist.all_reduce()` and `all_gather()` calls must use explicit rank lists or group locals to preserve invariant collective shapes.

### 3. Regression Tests

- New features touching model I/O, distributed plumbing, or checkpoint serialization must include a regression test.
- Regression tests must be written as deterministic units using `pytest` and avoid random seeds where possible.
- If randomness is required, fix seeds explicitly in the test using `torch.manual_seed(seed)` and `np.random.seed(seed)`.

---

## Continuous Integration Requirements

### CI Matrix

Every pull request must run the following matrix in CI:

| Job | Description |
| --- | --- |
| `format` | `black --check` and `ruff check` on all Python sources | 
| `type` | `mypy src/rlhf_platform --strict` |
| `unit` | `pytest tests/ --maxfail=1 --disable-warnings` |
| `security` | Static analysis for security-sensitive modules |

### Merge Conditions

- At least one review approval from a core maintainer.
- All CI jobs in the PR matrix must pass.
- No direct commits to `main` unless through an approved PR.
- No merge if the CPU pytest job fails. The known red tests are listed under S0.6 in `ROADMAP.md`. They are not hidden with xfail.

---

## Documentation & Specification Expectations

- All architectural changes must include updates to `ARCHITECTURE.md` or the relevant `docs/` deep dive.
- Code touching distributed execution must document process group invariants and race conditions.
- Any new public command-line option in `train.py` must appear in `scripts/README.md`.

---

## Operational Readiness Checklist

Before a new feature can be merged into `main` the contributor must verify:

- `python -m py_compile train.py` passes.
- `pytest tests/ -q` passes on the target Python runtime.
- New or modified code includes docstrings and inline comments for distributed semantics.
- `README.md` and the docs hub are updated if the change affects user-facing deployment or architectural assumptions.

---

## Review Guidelines for Core Maintainers

Core maintainers should prioritize:

- Correctness of communication group creation (`dist.new_group`) and barrier symmetry.
- Explicit device mapping and non-blocking transfer semantics.
- Shape invariance across all rank-local tensor operations.
- Security controls around rollout ingestion and checkpoint verification.
- Numerical stability: KL clipping, gradient clipping, and NaN/Inf handling.

Reviewers should reject changes that:

- Introduce implicit global process groups.
- Rely on unchecked tensor shape expansion or contraction.
- Disable pre-commit hooks or skip CI jobs.
- Add unvalidated checkpoint loading paths in training entrypoints.

---

## Contribution Workflow

1. Fork the repository.
2. Create a feature branch: `feature/<what-you-are-changing>`.
3. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```
4. Run local validation:
   ```bash
   black --check .
   ruff check .
   mypy src/rlhf_platform --strict
   pytest tests/ -q
   ```
5. Open a PR and request review from a core maintainer.
6. Address reviewer feedback and keep the branch rebased on `main`.

---

## Final Note

This framework is intended for high-assurance distributed training. Every contribution must be evaluated not only for functional correctness, but for operational stability, distributed determinism, and security resilience.
