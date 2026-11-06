# ROADMAP

The single source of truth for work in this repo, maintained by the owner and a coding agent.
Every item records its status, notes (including bugs found and null results) and a change-log line.

**Status markers:**
- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` dropped
- `[!]` blocked

**Who runs what:**
- **Agent** = CPU work on the i9 workstation.
- **Owner** = GPU runs on Colab L4 or Modal A100, plus final sign-off.

---

## Stage 0 — Make it true and make it owned (CPU only)

The exit criteria for Stage 0:
- Every claim in the README is supported by code or tests.
- The core math is parity-tested.
- The owner can explain every module.

### [x] S0.1 Repo hygiene
- Rename `gitignore` → `.gitignore`. Add `__pycache__/`, `*.pyc`, `*.egg-info/`, `.claude/`,
  `agent.md`, `CLAUDE.md`, `AGENTS.md`, `checkpoints/`, `wandb/`, `outputs/`.
- Remove committed `__pycache__/` and `src/rlhf_platform.egg-info/` from the index.
- Replace lowercase `roadmap.md` with this file.
- List these docs in the PR and propose delete or rewrite for each, then **ask the owner** before
  deleting: `docs/COMPLETION_SUMMARY.md`, `docs/PHASE_*.md`, `docs/RESEARCH_REPORT.md`,
  `results/benchmarks.md`.
- Notes:
  - Renamed `gitignore` to `.gitignore`. The ignore list covers bytecode, `*.egg-info/`,
    `.claude/`, `agent.md`, `CLAUDE.md`, `AGENTS.md`, `checkpoints/`, `wandb/`, `outputs/`,
    `.venv/`, tool caches, and `*:Zone.Identifier`.
  - Removed tracked `__pycache__/` and `src/rlhf_platform.egg-info/` from the index.
  - Replaced tracked `roadmap.md` with this file. `docs/core/ARCHITECTURE.md` now points here.
  - Docs were not deleted. Owner decision:
    - Propose delete: `docs/COMPLETION_SUMMARY.md`, `docs/PHASE_1_3_SUMMARY.md`,
      `docs/RESEARCH_REPORT.md`, `results/benchmarks.md`. They claim production-ready status,
      110+ tests, 100% type coverage, and a measured T4 run. None of that is backed by a
      result file or by the tests currently in `tests/`.
    - Propose rewrite into `docs/CODE_MAP.md` during S0.4, or delete: `docs/PHASE_1_CONFIG.md`,
      `docs/PHASE_2_PPO.md`, `docs/PHASE_3_CLI.md`, `docs/PHASE_4_BENCHMARKS.md`.
  - 2026-10-30, owner decision: delete all eight. They are deleted. `docs/CODE_MAP.md` will be
    written from the code in S0.4, not from those files. `ROADMAP.md` is a tracked file.
    `agent.md` is the only local instruction file, and it is gitignored.
  - Verification on this machine (CPU torch 2.15.0, Python 3.12): `ruff check` and
    `ruff format --check` already fail across the tree. There is no ruff config yet.
    `pytest` cannot collect `tests/test_topology.py` or `tests/test_rollout.py` because
    importing the package loads `comm_hooks.py`, which imports a name torch does not
    export (`ddp_comm_hook`). Of the 77 tests that do collect in config, DPO, and PPO,
    66 pass and 11 fail. Those failures were already in the code. Details are under
    S0.6 and S0.9. S0.1 does not change Python.

### [x] S0.2 README truth pass
- Fix the DPO description: it is reward-model-free, not reference-free.
- Replace hand-typed test counts and coverage badges with generated values, or remove them.
- Remove the telemetry JSON example (79 GB VRAM, 2,450 tok/s; never measured) and the
  8 / 512 / 10,000-GPU table. If kept at all, move them to `docs/design.md`, labelled
  "design target, not measured".
- Add a "Measured scope" section. Right now it should honestly read "unit-tested on CPU;
  no GPU benchmark yet".
- Notes:
  - README rewritten. DPO is described as using a frozen reference policy and not training
    a reward model. Test-count and coverage badges are gone. Measured scope states the
    2026-10-30 CPU result (77 collected tests in three modules, 66 passed, 11 failed;
    two modules do not collect) and points at `pytest --collect-only -q` to recount.
  - The telemetry JSON sample was deleted. It was a fabricated measurement, so it was
    not copied into `docs/design.md`. The 8 / 512 / 10,000-GPU table moved there and is
    labelled "design target, not measured".
  - README now also says what the CLI does: `train-sft` and `train-reward` call their
    trainers; `run-ppo` only writes a config; `run-dpo` exits 0 without calling
    `DPOTrainer`. CLI help strings still mention a T4 timing and "Phase 4". Those
    strings were left for a later code edit.
  - `pip install -e .` was not run. The measured-scope check used CPU torch 2.15 with
    `PYTHONPATH=src`.

### [x] S0.3 CI that tells the truth
- Remove jobs pointing at non-existent `tests/integration/` and `tests/benchmarks/`.
- Remove `|| true` from the CI steps.
- Make the default job: CPU torch, `ruff`, `pytest -q` (skipping `slow` and `gpu` markers).
- Pin versions.
- Notes:
  - Replaced `.github/workflows/ci_perf.yml` with `.github/workflows/ci.yml`. One job:
    Python 3.11, `requirements-cpu.txt` (torch 2.15.0+cpu and pinned pytest, ruff,
    pydantic, PyYAML, numpy, transformers, datasets, accelerate), `pip install -e . --no-deps`,
    then `ruff check`, `ruff format --check`, and `pytest -q -m "not slow and not gpu"`.
    No `|| true`. Integration, benchmark, bandit, mypy, and codecov jobs are gone.
  - `pyproject.toml` still pins the old training stack. CI does not install it.
  - Ruff is configured for E, F, and I, with E501 ignored because the formatter owns line length.
    `ruff check` and `ruff format --check` pass locally.
  - Pytest does not pass. After the import fix below, the same command collected 82 tests:
    69 passed, 13 failed. Two of those were later fixed as plain bugs (see S0.9). 11 remain.
    The job stays red. No xfail and no skip was added to hide them.
  - `comm_hooks.py` imported `ddp_comm_hook`, which torch 2.15 does not export, and nothing
    in the file used that name. The import was removed so the package imports and the two
    previously uncollected modules run. `telemetry.py` used `torch` without importing it.
    That import was added.
  - `tests/run_benchmark_comparison.py` still returns hardcoded throughput and VRAM figures
    (for example 1.9 steps/s and 3.5 GB) whether or not TRL imports. Pytest does not collect
    it (`python_files = test_*.py`). It is not a result.
  - 2026-10-30: the job never got past install. `numpy==2.5.4` requires Python >=3.12,
    and the workflow used 3.11. The workflow now uses 3.12, the same interpreter as the
    local 71 passed / 11 failed count. The 11 failures still fail the job. No xfail.

### [x] S0.4 Code ownership pass
- Merge `ppo_engine.py` and `alignment/ppo_engine.py` into one implementation. Delete dead code.
- Write `docs/CODE_MAP.md`: one paragraph per module covering what it does, its inputs and
  outputs with tensor shapes, and which tests cover it. The owner reads this before Stage 0.6.
- Flag any code from external contributors that the agent cannot explain clearly.
- Notes:
  - Owner: start this only after PRs 6, 7, and 8 are on main, and branch from main.
  - Choose the surviving PPO engine by correctness, including the `train_step` gather bug
    in `src/rlhf_platform/ppo_engine.py`, not by which file is longer. Document the choice
    in the PR. Write `docs/CODE_MAP.md` from the code, not from the deleted phase docs.
  - 2026-10-30: GitHub `main` is still the S0.1 merge (`764079a`, PR #6). PRs #7, #8, and #9
    are on the stack that ends at `fix/owner-decisions-docs-and-rollout` (`80d4aed`). This
    branch starts there. A branch from GitHub `main` would not contain the README, the CI
    job, or the 11 failures this item is not allowed to add to.
  - The survivor is `PPOTrainer` in `src/rlhf_platform/ppo_engine.py`. Deleted
    `alignment/ppo_engine.py` (`DistributedPPOEngine`) and `alignment/loss.py`. Only the
    deleted engine imported `loss.py`.
  - Why that file, not the longer one: `PPOTrainer` splits GAE, the clipped policy loss,
    the value loss, entropy, KL, the adaptive coefficient, and explained variance into
    methods the tests call, and those methods take per-action log-probs or rank-2 rewards
    and values. `DistributedPPOEngine.ppo_step` passed a full-vocabulary `log_softmax`
    (`[B, T, V]`) into a ratio that needs an action log-prob, discarded the reference
    forward, never called the reward model, left `collect_rollouts` empty, ignored the
    advantages it had just computed because `PPOLossFn` computed them again, called
    `optimizer.zero_grad()` at the start of every step, read critic values from `.logits`,
    set `old_values` to the detached current values so the value clip could not apply,
    and added `lambda * sum(values**2)` to the value loss. It also used
    `dist.ReduceOp.MEAN`, which torch 2.15 does not define. The `except` path returned
    the local metrics.
  - The gather stays in one place: `PPOTrainer.train_step`,
    `log_softmax(...).gather(1, input_ids.unsqueeze(-1))`. The line is unchanged.
    A comment above it names S0.6a. `input_ids` is `[B, T]`, so the index is `[B, T, 1]`.
    `DummyModel` logits are `[B, V]`. Causal-LM logits are `[B, T, V]`, and dimension 1
    is time, with no shift.
  - KL direction and the value loss were not edited. `compute_kl_divergence(policy_logits,
    ref_logits)` is the full-vocabulary KL(reference || policy). The value loss is
    `value_loss_coefficient * MSE`, with no clip. S0.6d and S0.6e own those.
  - Also removed, because nothing remaining called them: the old `train.py` loop (the
    file stays and exits 2; it does not call `PPOTrainer`); `sample_batch_to_device`,
    `RolloutDataset`, `AsyncRolloutPipeline`, `QuerySampler`, and `RolloutBuffer.is_full`;
    `PPOSample` and `async_load_dataset`; `create_default_topology` (actor and reference
    device lists overlapped, and `validate` would have rejected it; nothing called it);
    `log_config_to_wandb` and the `ppo_engine` `__main__` print. Docstrings that said
    production-grade, mixed precision, optional value clipping, a constructor
    `PPOTrainer(config)`, or reference-free DPO now match the code.
  - Left for S0.9: `distributed/comm_hooks.py`, `distributed/async_io.py`,
    `scripts/simulate_runtime.py`, and the unused `NCCLProfiler` and
    `DistributedMetricsCollector`. Config fields the tests construct stay, including
    `use_peft_lora`, `attn_implementation`, and `ppo_num_epochs`, even where no trainer
    reads them.
  - CPU pytest on this branch, `.venv`, torch 2.15.0+cpu, `PYTHONPATH=src`,
    `pytest -m "not slow and not gpu"`: 82 collected, 71 passed, 11 failed, in 18.72s.
    The 11 names are the ones already listed under S0.6. No other test failed.
    `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` pass.

### [x] S0.4b Remove fabricated and unexplainable code
- Depends on: S0.4.
- Branch: `refactor/S0.4b-remove-unexplainable`.
- Delete `tests/run_benchmark_comparison.py`. Its throughput and VRAM numbers are literals.
- Rename `LoRAMaskingCollator`. The body only sets padding labels to -100.
- Remove `EarlyStoppingCallback`. `SFTTrainer.train` passes `eval_dataset=None`, so
  `on_evaluate` never runs.
- Remove `DPOMetrics.explained_variance`. DPO has no value head, and the field is the
  constant `0.0`.
- Notes:
  - 2026-10-30 owner decision on the CODE_MAP list of code that cannot be explained.
    These four are this item. Undeclared config fields are S0.11. The KL direction is
    S0.6e. The rank-3 reward average is S0.7. The distributed files are S0.9, re-scoped
    to deletion.
  - GitHub `main` is still `764079a` (PR #6). This branch starts at the stack tip
    `03f74a1` (merge of PR #10). A branch from GitHub `main` would not contain S0.4 or
    `docs/CODE_MAP.md`.
  - Done. `IgnorePaddingLabelCollator` sets padding labels to -100.
    `EarlyStoppingCallback` is deleted. `DPOMetrics` no longer has
    `explained_variance`. `tests/run_benchmark_comparison.py` is deleted. It was not
    collected by pytest, so the failure set did not shrink.
  - `SFTTrainer.train` still sets `evaluation_strategy="steps"` and
    `load_best_model_at_end=True` with `eval_dataset=None`. No test calls `train`.
    S0.11 meets this when the command actually runs.
  - CPU pytest on this branch, `.venv`, torch 2.15.0+cpu, `PYTHONPATH=src`,
    `pytest -m "not slow and not gpu"`: 82 collected, 71 passed, 11 failed. The 11
    names are the ones already listed under S0.6. No other test failed.
    `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` pass.

### [x] S0.5 Package rename `rlhf_platform` → `rlpost`
- Depends on: the owner renaming the GitHub repo first.
- Update imports, `pyproject.toml`, the CLI entry point and docs. Keep a note in the README
  pointing to the old repo name.
- Notes:
  - 2026-11-02: branched from `origin/main` `aabce4f` (merge of PR #15).
    Branch `refactor/S0.5-package-rename`.
  - `60abd6f` lengthens the alignment package docstring. The old one-line
    docstring was most of that file, so changing the identifier on that line
    alone fell under git's 50% rename threshold. The new sentence says the
    package re-exports the rollout buffer and the token sampler.
  - `2bf145a` is `git mv src/rlhf_platform src/rlpost` plus the identifier
    edits. Every moved file is a rename at similarity 75% or higher.
    `git log --follow` reaches the old path. There is no compatibility shim.
  - Imports, tests, `train.py`, and `pyproject.toml` use `rlpost`. The project
    name is `rlpost`. Package data and the coverage source use that name.
    There was no setuptools console-script entry point. The CLI is
    `python -m rlpost.cli`. The Typer app name is `rlpost`.
  - README, `docs/CODE_MAP.md`, `docs/core/ARCHITECTURE.md`,
    `docs/DEVELOPMENT.md`, and `scripts/README.md` use the new path.
    `docs/governance/contributing.md` and `.github/workflows/ci.yml` do not
    name the package, so they are unchanged. CI still installs with
    `pip install -e .`.
  - `TrainingConfig.default_config` and `configs/default.yaml` use run name
    `rlpost_default_run` and output dir `./outputs/rlpost_default_run`.
    W&B project strings that were the old distribution name are `rlpost` and
    `rlpost-toy`. The field default `rlhf_alignment_run` and the toy run
    `rlhf_toy_run` stay. They are run names, not the package id.
  - The README says the repository was formerly
    Improving-LLM-Models-with-RLHF-PPO-DPO. RLHF, PPO, and DPO stay where
    they name the methods.
  - `git grep -n "rlhf_platform"` matches only this file.
  - CPU pytest on this branch, `.venv`, torch 2.15.0+cpu, `PYTHONPATH=src`,
    `pytest -q -m "not slow and not gpu"`: 79 collected, 68 passed, 11 failed.
    The 11 names are the ones already listed under S0.6. No other test failed.
    `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` pass.
  - Fresh venv, `pip install -e .`: the `rlpost` wheel built, then the install
    failed building `tokenizers==0.13.3`. rustc rejected a `&T` to `&mut T`
    cast in that pin. The pin was already in `pyproject.toml`. A second fresh
    venv, `pip install -e . --no-deps`, installed `rlpost` 0.1.0.
    `python -c "import rlpost"` loaded `src/rlpost/__init__.py`.

### [x] S0.5b One dependency truth
- Depends on: S0.5.
- `[project.dependencies]` lists only packages the code imports, with lower bounds the
  CPU lock satisfies. `pydantic` is declared. Packages nothing imports are not required.
- `[test]` pins pytest, ruff, and TRL. `[lora]` is peft. The three freeze files are
  deleted. CI installs `.[test]` under `requirements-cpu.txt`.
- Notes:
  - 2026-11-02: branched from `origin/main` `5f8b6e1` (merge of PR #16).
    Branch `chore/S0.5b-dependency-truth`.
  - Direct imports: torch, transformers, datasets, pydantic, PyYAML, typer, rich.
    peft is imported by `sft_engine.py` and `reward_engine.py`, so it is the `[lora]`
    extra (`peft>=0.17`). wandb is imported inside `try` / `except ImportError` and
    stays optional. numpy, accelerate, bitsandbytes, triton, safetensors, tensorboard,
    and tokenizers are not imported. `BitsAndBytesConfig` comes from transformers.
  - `[dev]`, `[eval]`, and `[docs]` named tools nothing in the tree imports. They are
    removed so the extras are not a second dependency list.
  - `[test]` pins `pytest==9.2.0`, `ruff==0.17.3`, and `trl==1.17.1`. TRL 1.17.1
    requires Python >=3.11, `transformers>=4.56.2` (no upper bound; 5.22.0 satisfies
    it), `torch>=2.6.0`, `datasets>=4.7.0`, `accelerate>=1.11.0`, and `numpy>=1.26`.
    The CPU lock already has transformers 5.22.0, torch 2.15.0+cpu, datasets 5.1.0,
    and accelerate 1.15.0.
  - `requirements.txt`, `requirements-fine-tune.txt`, and `requirements-reward.txt`
    are deleted. The S0.9c note that names them stays as history.
  - numpy stays pinned only in `requirements-cpu.txt` (`numpy==2.5.4`, Python >=3.12).
    That pin is why the CI job stays on 3.12. An unconstrained `pip install -e ".[test]"`
    does not force it, so Python 3.11 can resolve.
  - Fresh venvs, `pip install -e ".[test]"` (no constraints file):
    Python 3.11.18 resolved trl 1.17.1, transformers 5.22.1, torch 2.15.0+cu132,
    datasets 5.1.0, numpy 2.4.6. Python 3.12.3 resolved trl 1.17.1,
    transformers 5.22.1, torch 2.15.0+cu132, datasets 5.1.0, numpy 2.5.4.
    `python -c "import rlpost, trl"` loaded `src/rlpost/__init__.py` on both.
    Ruff check and ruff format check pass on both. Pytest collects 79
    (37 + 17 + 22 + 3), 68 passed, 11 failed. The 11 names are the ones
    already listed under S0.6.
  - The same install under `requirements-cpu.txt` and the CPU torch index, on
    Python 3.12.3, resolved trl 1.17.1, transformers 5.22.0, torch 2.15.0+cpu.
    Ruff passes. Pytest is the same 79 / 68 / 11.
  - `python -c "import rlpost.cli"` in the test extra raises
    `ModuleNotFoundError: peft`. The CLI imports the trainers at import time.
    That coupling is recorded under S0.11. It is not changed here.
  - typer and rich are required because `cli.py` imports them. They are not
    pinned in the CPU lock. The 3.12 CPU install resolved typer 0.27.3 and
    rich 15.0.0.
  - Project URLs are `https://github.com/Mattral/rlpost`, the repository URL
    with `.git`, and the issues URL. The Documentation URL is removed. There
    is no readthedocs site.
  - `pip download torch==2.15.0 --index-url https://download.pytorch.org/whl/cu126 --no-deps`
    finds no file. The index lists cu126 wheels from 2.6.0 through 2.14.1.
    The newest stable wheel is `torch==2.14.1` (cp311 and cp312 manylinux
    wheels answer HTTP 200). The package floor stays `torch>=2.15`. Tests
    were not re-run on 2.14.1. `docs/DEVELOPMENT.md` names that Pascal install.

### S0.6 Numerical parity tests (core correctness)
For each check below, compare against TRL (pinned version) or a closed-form computation,
on a tiny model on CPU, with an explicit tolerance. Split into S0.6a–f. A bug found
here is a result. S0.6f is optional and does not change the order of S0.6b–e.

### [x] S0.6a Per-token logprob gather
- Depends on: S0.5b.
- One `gather_logprobs` in `src/rlpost/ops.py`. Logits `[B, T, V]` in, log-probs
  `[B, T-1]` out. Position `t` scores token `t+1`. Masking stays outside the gather.
- PPO `train_step` and DPO `compute_log_probs` both call it. CPU parity against a
  hand-written `log_softmax` + `gather` and against TRL 1.17.1.
- Notes:
  - 2026-11-04: branch `test/S0.6a-logprob-gather` from `origin/main` `4ca23f4`
    (merge of PR #17).
  - TRL 1.17.1 helper is `trl.trainer.utils.selective_log_softmax`. It is not
    exported from `trl.__init__`. It gathers `log_softmax(logits)` at `index`
    and does not shift. The parity call passes `logits[:, :-1]` and
    `input_ids[:, 1:]`. `DPOTrainer` in that release does not call the helper.
    It reads `outputs.log_probs` from `fused_lm_head=True` and shifts the
    completion mask with `completion_mask[..., 1:]`.
  - fp32 absolute tolerance is `1e-6` against the hand-written reference and
    against `selective_log_softmax`. bf16 absolute tolerance is `2e-2` against
    the fp32 gather. On this CPU, torch 2.15, the tiny Llama, the max absolute
    gap was 0.0173.
  - The two `test_train_step_*` tests now build a 2-layer Llama (hidden 32,
    vocab 128) and pass. The other nine failures listed below are unchanged.
  - Pytest on this branch: 85 collected, 76 passed, 9 failed. Ruff check and
    ruff format check pass.

### [x] S0.6b DPO loss, rewards, and margins
- Depends on: S0.6a.
- DPO loss, chosen/rejected rewards and margins vs TRL's DPO loss on identical inputs.
- Notes:
  - 2026-10-30 judgments on the 11 failures still red after the rollout fixes. Neither side
    was edited to make the other pass.
  - 2026-10-30 owner triage, accepted and still to be confirmed by the S0.6b parity test
    against TRL. Do not implement these before that test:
    1. `test_margin_computation`: per TRL, implicit reward =
       `beta * (logp_policy - logp_ref)` per response, and
       `margin = chosen_reward - rejected_reward`. The margin includes the reference and
       is scaled by beta. The test uses the policy gap only (1.5 on its fixture). The code
       subtracts the reference gap but does not multiply by beta (0.4 on that fixture).
       Both are wrong. Fix both in S0.6b.
    2. `test_metrics_batch_aggregation`: accuracy is `chosen_reward > rejected_reward`
       after the reference adjustment. The code's 0.75 is the right convention. The test
       is wrong. It expects 1.0 from the policy scores alone.
    3. `dpo_beta`: add it as an explicit field on `AlignmentConfig`, with validation.
       Remove the silent `getattr(..., "dpo_beta", 0.1)` fallback in `DPOTrainer`.
       Do not reuse `kl_coefficient`. That coefficient belongs to PPO.
  - `test_beta_affects_loss`: on this fixture `policy_diff - ref_diff` is 0, so every beta
    gives `-logsigmoid(0)`. The fixture has no preference margin over the reference.
    Not changed. S0.6b covers the beta scaling.
  - Six test-side failures, not to be edited before S0.6 / S0.7. Rewrite them there with
    a tiny real `nn.Module` or a tiny random-init causal LM, not mocks:
    `test_init_creates_trainer`, `test_reference_model_frozen`, and
    `test_config_integration` (empty `parameters()` mocks);
    `test_policy_loss_computation` and `test_value_loss_computation` (`torch.randn`
    inputs that do not require grad);
    `test_train_step_updates_weights` and `test_train_step_returns_metrics`
    (`DummyModel` mean-pools the sequence to `[batch, vocab]`). The `train_step` gather
    was the S0.6a bug. S0.6a replaced it with `gather_logprobs` and rewrote those two
    tests. The other four test-side failures in this list are still open.
  - S0.6e: the current penalty is full-vocabulary KL(ref || policy); the expected target
    is KL(policy || ref) on sampled tokens with a named estimator.
  - S0.5b pins the parity target at `trl==1.17.1`. That release requires
    `transformers>=4.56.2` (no upper bound; transformers 5.22.0 satisfies it),
    `torch>=2.6.0`, `datasets>=4.7.0`, and `accelerate>=1.11.0`. Parity in S0.6a
    was measured against `selective_log_softmax` from that pin.
  - 2026-11-04: branch `test/S0.6b-dpo-loss` from `origin/main` `e96cb46`
    (merge of PR #19). Code commit `9cb030b`.
  - TRL 1.17.1 has no free function for the DPO loss. The sigmoid loss is
    `DPOTrainer._compute_loss` with `loss_types == ["sigmoid"]`.
    `DPOTrainer.__init__` builds a Hugging Face `Trainer` and a tokenizer,
    and with no reference model it loads one from `config._name_or_path`.
    The parity test does not call `__init__`. It builds the instance with
    `__new__`, sets the attributes `_compute_loss` reads, and calls that
    method on the same tiny Llama and the same token batch.
  - The sigmoid branch does not read `label_smoothing`. Smoothing is applied
    on the robust, exo_pair, and aot losses. Parity sets `label_smoothing`
    to 0, `f_divergence_type` to `reverse_kl`, and `ld_alpha` to `None`.
    The score is then the log-ratio, and the loss is
    `-logsigmoid(beta * (chosen_logratio - rejected_logratio)).mean()`.
  - Sequence log-probabilities are a sum, not a mean. `add_fused_lm_head`
    writes 0 where the label is -100. Labels are `input_ids` with
    `completion_mask == 0` set to -100, then shifted one position, so prompt
    tokens and padding add nothing. `_compute_loss` sums that row.
    `compute_log_probs` multiplies `gather_logprobs` by
    `completion_mask[:, 1:]` and sums. A test fails if the attention mask
    is used instead: that mask still includes the prompt.
  - On CPU, torch 2.15, the tiny Llama (2 layers, hidden 32, vocab 128),
    seeds 0 through 5: the largest |loss - TRL loss| was 1.8e-7. The largest
    gap on the batch mean of the completion-token sums (TRL logs this as
    `logps/chosen` and `logps/rejected`) was 1.907e-6, one float32 ulp near
    magnitude 22. Loss, margin, and reward absolute tolerance is 1e-6.
    The tolerance on those logged means is 1e-5.
  - The old margin fixture is 0.04: beta 0.1 times the 0.4
    reference-adjusted gap. The policy-only gap on that fixture was 1.5.
    Accuracy on the aggregation fixture stays 0.75. Row 0 is a tie after
    the reference adjustment, and the comparison is strict `>`.
  - The old beta fixture had `policy_diff - ref_diff == 0`, so every beta
    returned `-logsigmoid(0)`. The replacement has unscaled margin 1.
    A larger beta then makes `-logsigmoid(beta)` smaller.
  - `AlignmentConfig.dpo_beta` is a field, default 0.1, validated `> 0`.
    `DPOTrainer` reads `config.alignment.dpo_beta`. The `getattr` fallback
    is gone. `kl_coefficient` is unchanged and is still the PPO coefficient.
  - `test_init_creates_trainer`, `test_reference_model_frozen`, and
    `test_config_integration` now build the tiny Llama. The reference
    parameters do not require grad and are not in the optimizer. A backward
    through the policy log-probs leaves the reference grads as `None`.
  - `compute_log_probs` had `@torch.no_grad`. That also blocked policy
    gradients. The decorator is gone. `train_step` still wraps the
    reference call in `torch.no_grad`.
  - Pytest on this branch: 91 collected, 89 passed, 2 failed, 16.10s.
    The two failures are `test_policy_loss_computation` and
    `test_value_loss_computation` (S0.6d). Ruff check and ruff format
    check pass.
  - 2026-11-04 follow-up, same branch, test commit `c301a03` on top of
    `9cb030b`. The parity test
    is parametrized over seeds 0 through 5. The policy is
    `tiny_causal_lm(seed)`, the reference is `tiny_causal_lm(seed + 10)`,
    and the token batch uses that seed. Each seed runs a matched-mask
    batch and a batch whose chosen and rejected rows differ in completion
    length and in padding. Chosen completions are lengths 4 and 2 with
    prompt length 2. Rejected completions are lengths 3 and 5 with prompt
    length 3. The four masks are separate arguments to
    `compute_log_probs`, and TRL receives them concatenated per side.
    Reusing the chosen mask for the rejected side moves the loss by about
    2e-3 on seed 0 of that batch.
  - On that grid, CPU, torch 2.15, the same tiny Llama: the largest
    |loss - TRL loss| was 2.384e-7. The largest margin gap was 5.700e-7.
    The largest reward-mean gap was 5.737e-7. The accuracy gap was 0.
    The logged sum-mean gap stayed 1.907e-6. Tolerances are unchanged:
    1e-6 on the loss, the margin, and the rewards, and 1e-5 on the logged
    means.
  - Gradient parity builds a second policy and a second reference from
    the same seeds, so the starting weights match. Both losses are
    backwarded. The optimizer is not stepped. Every policy parameter has
    a gradient on both copies. Absolute tolerance is 1e-6. The largest
    |rlpost grad - TRL grad| was 9.686e-8. Both reference copies are
    frozen, and every reference grad stays None. The mock
    `test_gradients_not_through_reference` is deleted.
    `test_reference_model_frozen` still checks the freeze.
  - Pytest after that follow-up: 113 collected, 111 passed, 2 failed,
    16.86s. The two failures are the same S0.6d PPO loss tests. Ruff
    check and ruff format check pass.

### [x] S0.6c GAE advantages and returns
- Depends on: S0.6a.
- GAE advantages and returns vs a hand-computed small example.
- Notes:
  - 2026-11-04: branch `test/S0.6c-gae` from `origin/main` `8704671`
    (merge of PR #20). Code commit `3f875cf`.
  - TRL 1.17.1 has no PPO trainer and no GAE helper. A scan of the
    installed package's `.py` files found none of `generalized advantage`,
    `compute_gae`, `lastgaelam`, or `gae_lambda`. `trl.trainer` exports
    DPO, GRPO, RLOO, KTO, SFT, reward, and distillation. GRPO's advantage
    is `reward - group mean` (optional divide by the group std). RLOO's
    is leave-one-out. Neither is the delta recursion. There is no TRL
    parity test. The reference is `backward_gae` in `tests/test_gae.py`,
    plus the literal T=4 values in that file. `tests/test_gae.py` repeats
    the scan and fails if one of those strings appears.
  - Time is axis 1 of `[B, T]`. The backward pass starts at `t = T - 1`.
    The advantage after the last index is 0. S0.7 depends on this layout.
  - A token-level reward sits on the token it scores. `rewards[:, t]`,
    `values[:, t]`, and `dones[:, t]` belong to `input_ids[:, t]`.
    `next_values[:, t]` is `V_{t+1}`. The caller supplies it. It is not
    read from `values[:, t+1]`. For a token sequence,
    `next_values[:, t]` should be `values[:, t+1]` when `t < T - 1`, and
    `next_values[:, T - 1]` is the bootstrap after the last token.
  - `gather_logprobs` index `t` is the log-prob of token `t + 1`.
    `train_step` multiplies that by `advantages[:, 1:]`. The weight on
    the action "emit token t" is `advantages[:, t]`, for `t >= 1`. A
    reward written at index `t` enters the policy term for that same
    token, through the logit at `t - 1`. Token 0 is not an action in
    that gather.
  - `mask[:, t] == 0` is padding, not a timestep. That position's
    advantage and return are 0. Its reward and value do not enter any
    other position's delta, and its advantage does not enter the backup.
    The real token before a pad does not bootstrap from the pad's value.
    `done` is separate: `done_t == 1` stops the backup after a real
    token even when the next index is real. On real tokens,
    `returns = advantages + values`. The parameter is spelled `lam`
    because `lambda` is a Python keyword.
  - The T=4 example uses gamma 0.5, lambda 0.5, rewards `[1, 2, 3, 4]`,
    values `[0.5, 1, 0.5, 0]`, next values `[1, 0.5, 0, 0]`, dones 0.
    Advantages are `[1.53125, 2.125, 3.5, 4]`. Absolute tolerance is
    1e-6. Those values are dyadic and match to the bit in float32.
    Done at t=1 gives `[1.25, 1, 3.5, 4]`. Lambda 0 gives the deltas
    `[1, 1.25, 2.5, 4]`. Lambda 1 gives the discounted return minus V,
    `[2.75, 3.5, 4.5, 4]`. Gamma 1 with lambda 0.5 gives
    `[3.375, 3.75, 4.5, 4]`.
  - A right-pad suffix whose value is 80, with dones left at 0, leaves
    the real prefix on those same advantages. An internal pad splits
    the backup: the token before the pad does not see the pad's value
    or the token after it. Without the mask, that suffix moves the last
    real advantage from 4 to 60.25.
  - `train_step` still calls `compute` without a mask, and it still
    normalises advantages over the whole `[B, T]` tensor. Both stay in
    S0.7. The rollout reward is still one score per sequence.
    `RolloutCollator` stacks `rewards` of shape `[B]`. Nothing here
    scatters that score onto a token index.
  - Pytest on this branch, `.venv`, torch 2.15.0+cpu: 121 collected,
    119 passed, 2 failed, 16.27s. The two failures are
    `test_policy_loss_computation` and `test_value_loss_computation`
    (S0.6d). Ruff check and ruff format check pass.

### [x] S0.6d PPO clipped policy loss and value loss
- Depends on: S0.6a.
- PPO clipped policy loss, value loss (clipped) and entropy vs a reference implementation.
- `test_policy_loss_computation` and `test_value_loss_computation` take inputs that
  require grad. Gradients reach `logp_new` and `V` only.
- Notes:
  - 2026-11-04: branch `test/S0.6d-ppo-loss` from `origin/main` `a3cc67a`
    (merge of PR #22). Code commit `0107a29`. Follow-up `a1bb38e`
    puts the one-half on both value modes and adds the non-binding
    ratio test.
  - The recovered stash clipped the value loss with `ppo_epsilon` and no
    one-half, and its grad tests checked a `Linear` weight. That draft is
    not what landed. `value_clip` is an `AlignmentConfig` field, default
    `None`, and `>= 0` when set. Both modes use the same one-half:
    `value_loss_coefficient * mean(0.5 * squared error)`. `None` uses
    `(V - R)^2`. A set radius `c` uses
    `V_clip = V_old + clip(V - V_old, -c, c)` and the max of the two
    squares. `c = 1e9` matches the unset loss on the hand sample.
    `c` is not `ppo_epsilon`. `train_step` passes batch `values` as
    `V_old` and uses them only when `value_clip` is set.
  - The policy loss was already
    `-mean(min(r * A, clip(r, 1-eps, 1+eps) * A))` with
    `r = exp(logp_new - logp_old)` and `eps = ppo_epsilon`. `logp_old`
    and `A` are detached. The closed form in `tests/test_ppo_loss.py`
    uses eps 0.25. Ratios 1.5 and 0.5 bind when the clipped product is
    the smaller one (`A > 0` on the high ratio, `A < 0` on the low
    ratio). That loss is -0.125. The same ratios with the signs
    swapped do not bind: `1.5 * (-1) = -1.5` and `0.5 * 1 = 0.5` are
    the min terms, and the loss is 0.5. Always using the clipped ratio
    on that input gives 0.25, and the test requires those two losses
    to differ. Absolute tolerance is 1e-6.
  - Entropy is the mean of the per-row Shannon entropy. The per-row value
    matches `trl.trainer.utils.entropy_from_logits` in TRL 1.17.1
    (`trainer/utils.py`). A uniform row is `log(n)`. A distribution
    `[0.75, 0.25]` matches `-(p log p).sum()`. TRL 1.17.1 has no PPO
    trainer and no value-loss helper. A scan of the installed package
    found none of `cliprange_value`, `vf_loss`, or `class PPOTrainer`.
  - The loss functions do not take a mask. `train_step` still calls GAE
    and the losses without one, and it still normalises advantages over
    the whole tensor. Both stay in S0.7.
  - Pytest on this branch, `.venv`, torch 2.15.0+cpu: 132 passed, 19.34s.
    Ruff check and ruff format check pass.

### [x] S0.6e KL estimator
- Depends on: S0.6a.
- Name the estimator (k1/k2/k3) and test it. The penalty was full-vocabulary
  KL(reference || policy).
- Notes:
  - 2026-11-04: branch `test/S0.6e-kl-estimator` from `origin/main` `3532c95`
    (merge of PR #23). Code commit `3f0319f` added the sampled-token
    estimators and defaulted `kl_estimator` to `k3`, because that value is
    unbiased for KL(policy || reference) and >= 0. The loss holds the
    sampled token fixed, so the gradient of that default is the reverse KL.
    Code commit `d4bd358` changes the loss default to `k2`.
  - `AlignmentConfig.kl_estimator` is `k1`, `k2`, or `k3`, validated.
    Default is `k2`. `lr = log p_ref - log p_policy` at the sampled tokens.
    `k1 = -lr`. `k2 = 0.5 * lr^2`. `k3 = exp(lr) - 1 - lr`.
    `per_token_kl` detaches `log p_ref`. The sampled index is held fixed,
    so the backward does not go through which token was drawn. The value
    and the gradient can name different KLs.
  - Value, under samples from the policy. `k1` is unbiased for
    KL(policy || reference), and a token can be negative. `k2` is always
    >= 0 and biased for that KL. It is the quadratic term of the expansion
    of `k3`. `k3` is unbiased for KL(policy || reference) and >= 0 at every
    token:
    `E_{a ~ policy}[q(a) / p(a) - 1 - log(q(a) / p(a))] = KL(policy || reference)`.
    `f(x) = exp(x) - 1 - x` is convex, `f(0) = 0`, and `f'(0) = 0`, so the
    minimum is 0.
  - Gradient of the mean, same samples, token index held fixed, with
    respect to the policy logits. `k1`: the expected gradient is 0, so it
    does nothing as a loss term. `k2`: the expected gradient equals the
    gradient of KL(policy || reference). `k3`: the expected gradient equals
    the gradient of KL(reference || policy), the reverse direction. That is
    what GRPO differentiates.
  - The loss default is `k2`, because the loss is meant to follow
    KL(policy || reference). `k3` stays a legal config value. `k1` raises
    inside `compute_kl_divergence`. `per_token_kl(..., "k1")` still returns
    the value. That value belongs in the reward, as reward minus beta * k1
    (InstructGPT). This trainer does not implement that reward. Chosen
    policy: raise when `k1` is the loss term.
  - The old penalty was the full-vocabulary mean of
    `sum_a p_ref(a) (log p_ref(a) - log p_policy(a))`, which is
    KL(reference || policy). `train_step` gathers both models with
    `gather_logprobs` and averages the estimator on those tokens, shape
    `[B, T-1]`. `compute_kl_divergence` accepts a mask (`1` keeps a token,
    `0` drops it, all zeros return 0). `train_step` does not pass one.
    Prompt and padding tokens in that window stay in the mean. That stays
    in S0.7.
  - Value Monte Carlo. Closed form is `sum_a p(a) (log p(a) - log q(a))`
    on `p = (0.8, 0.1, 0.1)` and `q = (0.2, 0.2, 0.6)`. KL(p || q) is
    0.86054 and KL(q || p) is 0.93643, a gap of 0.07588. Draws 1,000,000
    tokens from `p`, seed 0. Absolute tolerance is `1e-2`. On that draw
    the k1 mean missed KL(p || q) by 0.00123 and the k3 mean missed it by
    0.00116. Swapping the two arguments misses KL(p || q). The k2 mean on
    the same draw was 0.95292 against KL 0.86054, a gap of 0.09238, outside
    that tolerance.
  - Gradient Monte Carlo. Policy logits `(1.0, 0.0, -0.5, 0.2)`, reference
    logits `(-0.4, 0.8, 0.3, -0.2)`, float64. The policy logits require
    grad. Analytic gradients are
    `grad_z KL(p || q) = p * (log(p/q) - KL(p || q))` and
    `grad_z KL(q || p) = p - q`. Autograd of the closed form matches those
    formulas at absolute tolerance `1e-12`. On these logits KL(p || q) is
    0.45165 and KL(q || p) is 0.40569. The two analytic gradients differ by
    a max absolute gap of 0.08995, so tolerance `1e-2` cannot accept the
    wrong direction. Draws 1,000,000 indices from softmax(policy logits),
    seed 0. On that draw the max absolute error was 0.000383 for k1 against
    0, 0.000259 for k2 against grad KL(p || q), and 0.000270 for k3 against
    grad KL(q || p). k2 is outside that tolerance of the reverse gradient,
    and k3 is outside it of the forward gradient.
  - TRL 1.17.1 `GRPOTrainer._compute_loss` (`trainer/grpo_trainer.py`) does
    compute a per-token KL. The assignment is
    `exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1`,
    which is k3. There is no k1/k2 switch. On identical log-probabilities,
    float64 and float32, `per_token_kl(..., "k3")` matches that expression
    bit for bit. Differentiating that term with the token held fixed applies
    the reverse-KL gradient. `GRPOConfig.use_bias_correction_kl` defaults to
    True and then multiplies by the importance ratio `coef_1`. That product
    is not k3, and this estimator does not apply it. RLOO in the same release
    uses `old_per_token_logps - ref_per_token_logps`, which is k1, as a
    reward-side penalty. Its comment calls that the first-order log ratio
    and contrasts it with GRPO's Schulman approximation. RLOO is not the
    parity target. The loss rejects k1.
  - Pytest on this branch, `.venv`, torch 2.15.0+cpu: 142 passed, 17.93s.
    Ruff check and ruff format check pass.

### [ ] S0.6f fp32 logsumexp gather (optional, before Stage 1)
- Depends on: S0.6a. Optional. It does not block S0.6b–e or S0.7. Land it before
  Stage 1.
- Inside `gather_logprobs`, upcast the shifted logits to fp32 and return
  `gathered_logit - logsumexp` over the vocabulary. Do not materialize the
  `[B, T-1, V]` log-probability tensor.
- Re-measure the bf16 tolerance against the fp32 gather, and record the peak
  memory difference on the tiny model (2 layers, hidden 32, vocab 128).
- Notes:
  - 2026-11-04, recorded only, from main `2200c69` (merge of PR #18). No code
    change. `gather_logprobs` still calls `F.log_softmax` on the shifted logits
    and gathers, which builds a `[B, T-1, V]` tensor. The S0.6a bf16 absolute
    tolerance is `2e-2`. On CPU, torch 2.15, that Llama, the max absolute gap
    was 0.0173. That figure stays the S0.6a result. S0.6f records a new bf16
    tolerance for the logsumexp gather, and the peak-memory difference against
    this gather. Those two numbers are not measured yet.

### [ ] S0.7 Classic RLHF edge-case tests
Cover:
- prompt-token masking in the loss
- logprob gather off-by-one
- padding side
- EOS and truncation
- reward position
- reference model gradients
- tokens the KL is computed over
- advantage normalisation with padding
- Notes:
  - `RolloutGenerator._decode_reward` averages a rank-3 tensor; the reward position test
    must settle what it should be.
  - The causal logprob shift is tested in S0.6a. This item still owns prompt-token
    masking, padding in the loss, EOS, reward position, reference gradients, which
    tokens the KL uses, and advantage normalisation over padding.
  - 2026-11-04, on main `2200c69`. `PPOTrainer.train_step` applies no mask.
    `ppo_engine.py` reads no attention mask, response mask, or prompt length.
    The policy loss is the mean of the clipped surrogate over every scored
    token, shape `[B, T-1]`. Entropy and the KL term are means over the full
    logits `[B, T, V]`. Prompt tokens and padding tokens are inside the policy
    loss, the entropy, and the KL.
  - The target is response-token-only masking. Write the tests so they require
    that mask and fail on the behavior above. Do not change `train_step` before
    those tests exist. DPO `compute_log_probs` already multiplies by
    `attention_mask[:, 1:]` and still averages prompt tokens; that prompt mask
    stays in this item.
  - 2026-11-04, S0.6b, code commit `9cb030b`. The sentence above describes
    main before that commit. DPO `compute_log_probs` now sums
    `completion_mask[:, 1:]`. Prompt tokens are outside that sum.
    `DPOTrainer._tokenize_for_loss` assumes right padding: the prompt ids
    are a prefix of `prompt + response`. The PPO `train_step` mask is still
    this item.
  - 2026-11-04, S0.6c, code commit `3f875cf`. Time is axis 1 of `[B, T]`.
    A token-level reward at index `t` is `r_t` for `input_ids[:, t]`.
    The policy ratio uses `advantages[:, 1:]`, so that reward weights
    the log-prob of token `t`, gathered from logits at `t - 1`.
    `compute` can take a padding mask. `train_step` does not pass one,
    and it still normalises advantages over every position, including
    pads. Those two stay in this item. The closed form is under S0.6c.
  - 2026-11-04, S0.6e, code commit `3f0319f`. The sentences above that
    call the KL term a mean over the full logits `[B, T, V]` describe
    main before this commit. `train_step` now averages the configured
    per-token estimator on `gather_logprobs` of the policy and the
    reference, shape `[B, T-1]`. It still passes no mask, so prompt and
    padding tokens in that window stay inside the KL mean. Entropy is
    still the mean over the full logits. Which tokens the KL uses, and
    the advantage normalisation over padding, stay in this item.
  - 2026-11-06, S0.6e, code commit `d4bd358`. The loss default is `k2`.
    `compute_kl_divergence` raises when `kl_estimator` is `k1`. `train_step`
    still passes no mask. Prompt and padding tokens in the gathered window
    stay inside the KL mean for `k2` and `k3`.

### [ ] S0.8 Determinism and resumable checkpoints
- Seeding gives identical losses on CPU for 2 runs.
- Kill and resume reproduces the uninterrupted run's step-N metrics (within tolerance).
- This is required before any Colab run.
- Notes:

### [ ] S0.11 CLI actually runs SFT/RM/DPO/PPO end-to-end on a tiny model on CPU
- Depends on: S0.8, because a resumable run is what the CLI should be able to finish.
- An end-to-end CPU test on a tiny model for all four commands.
- Today `run-dpo` prints a placeholder and exits 0. `run-ppo` writes the config YAML and
  returns. `train-sft` and `train-reward` call their trainers, and neither has a CPU
  end-to-end test.
- Notes:
  - SFT and the reward model read undeclared config fields (`lora_target_modules`,
    `dataset_size`). Declare and validate them as part of S0.11.
  - PEFT is the optional `[lora]` extra (`peft>=0.17`). `sft_engine.py` and
    `reward_engine.py` import it. No test calls those trainers, so the extra is
    unverified. `cli.py` imports both trainers at import time, so
    `python -m rlpost.cli` needs the extra. `import rlpost` does not.
  - Hugging Face `Trainer` needs `accelerate` at runtime. `sft_engine.py`
    constructs `Trainer`. The reward model is on the same SFT/reward path.
    S0.11 must declare `accelerate` for that path, in the `[lora]` extra or in
    a `[train]` extra. It is not a project dependency today.
  - 2026-11-04, review of merged PR #20 (`8704671`). Not a blocker.
    `DPOTrainer.train_step` takes one prompt string, one chosen string, and
    one rejected string. `_tokenize_for_loss` returns `input_ids`,
    `attention_mask`, and `completion_mask` with shape `[1, T]`. There is
    no batched collation. Accuracy is the mean of one comparison, so a
    per-step accuracy is 0 or 1. `compute_log_probs` and `compute_dpo_loss`
    already accept a batch. The missing piece is padding several pairs to
    `[B, T]` with a prompt length and a completion mask on each row.
    That collation has to exist before S1.2. Reuse the S0.6b parity test
    on a batch of 4.
  - Same review. Two silent fallbacks, the same pattern as the old
    `dpo_beta` getattr. `max_grad_norm or 1.0` turns 0 into 1.0.
    `getattr(config, "log_to_wandb", True)` turns logging on when the
    attribute is missing. Both are in `dpo_engine.py` `train_step`,
    `sft_engine.py`, and `reward_engine.py`. `PPOTrainer` reads
    `optimization.max_grad_norm` and `log_to_wandb` with no fallback.
    The fields already exist. `OptimizationConfig.max_grad_norm` defaults
    to 1.0 and is validated `gt=0`, so a `TrainingConfig` cannot store 0
    today. `TrainingConfig.log_to_wandb` defaults to True. S0.11 removes
    the fallbacks and reads the fields.
    In the same `TrainingArguments` block, `sft_engine.py` also uses
    `weight_decay or 0.01`, `warmup_ratio or 0.1`, and
    `gradient_accumulation_steps or 1`. `weight_decay` and `warmup_ratio`
    allow 0, and those `or` expressions replace a legal 0.

### [x] S0.9 Delete the orphaned distributed scaffolding
- Branch: `chore/S0.9-delete-distributed`.
- Nothing calls this code after S0.4. Stage 2 will build rollout and weight-sync
  distribution from scratch.
- Delete `src/rlhf_platform/distributed/topology.py`, `comm_hooks.py`, `async_io.py`,
  `scripts/simulate_runtime.py`, `configs/cluster_topology.yaml`, and `docs/design.md`.
- Remove the mock-topology sections of `docs/core/ARCHITECTURE.md` and `scripts/README.md`.
- Remove the README line that points at `docs/design.md`.
- Remove tests that only exist for the deleted code.
- Remove topology fields from config if nothing else reads them.
- After this item, `git grep -i "asymmetric\|topology\|10,000\|petascale"` returns only
  this file.
- Notes:
  - Previously scoped as "test or delete": a gloo process-group split on 2–4 CPU ranks,
    or remove `comm_hooks.py` and `async_io.py`, and delete `scripts/simulate_runtime.py`
    if real tests replaced it. 2026-10-30 owner decision: delete. Do not add a CPU
    process-group test for this scaffolding.
  - 2026-10-30: `comm_hooks.py` imported `ddp_comm_hook`, which torch 2.15 does not export.
    S0.3 removed that unused import. The module still has no test that runs a real hook.
  - 2026-10-30, once collection worked: `RolloutBuffer.add` called `pin_memory()` and raised
    `Cannot access accelerator device when none is available` on CPU torch.
    `RolloutGenerator._sample_token` returned `squeeze(-1)`, so each token was 1-D and
    `torch.cat(..., dim=1)` raised `IndexError`. Both are code bugs. Fixed: pin only when
    `torch.cuda.is_available()`, and keep sampled tokens shaped `[batch, 1]`.
    `tests/test_rollout.py` then passed (3 tests). `tests/test_topology.py` passed.
  - Done. Deleted `distributed/topology.py`, `comm_hooks.py`, `async_io.py`, and the
    package `__init__.py`. Also deleted `scripts/simulate_runtime.py`,
    `configs/cluster_topology.yaml`, `docs/design.md`, and `tests/test_topology.py`.
    `rlhf_platform` no longer lists `distributed`.
  - `TrainingConfig` has no field that only those modules read. `num_gpus`, `use_fsdp`,
    `use_deepspeed`, and `deepspeed_config` stay. `tests/test_config.py` and the config
    constructors set them. The cluster YAML was the only topology config.
  - `NCCLProfiler` and `DistributedMetricsCollector` remain in `utils/telemetry.py`.
    Nothing calls them. They were not in the delete list. Stage 2 does not extend them.
  - S0.9b removes those two classes, the four fields, and `configs/deepspeed_zero3.yaml`.
  - The two deleted tests were passing:
    `test_load_topology_from_yaml_loads_model_placements` and
    `test_topology_validate_rejects_conflicting_assignments`. The 11 failures under
    S0.6 are unchanged.
  - CPU pytest on this branch, `.venv`, torch 2.15.0+cpu, `PYTHONPATH=src`,
    `pytest -m "not slow and not gpu"`: 80 collected, 69 passed, 11 failed, in 18.20s.
    `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` pass.
  - `git grep -i "asymmetric\|topology\|10,000\|petascale"` returns only this file.

### [x] S0.9b Remove unused distributed config and telemetry
- Depends on: S0.9.
- Branch: `chore/S0.9b-drop-unused-distributed`.
- Remove `NCCLProfiler` and `DistributedMetricsCollector` from `utils/telemetry.py`.
- Remove `TrainingConfig` fields `num_gpus`, `use_fsdp`, `use_deepspeed`, and `deepspeed_config`.
- Delete `configs/deepspeed_zero3.yaml`.
- Notes:
  - 2026-10-30 caller check on `main` (`c6dd07d`) before deletion. Nothing outside
    `tests/test_config.py` reads the four fields. `TrainingConfig.toy_mode`,
    `TrainingConfig.default_config`, `configs/toy.yaml`, and `configs/default.yaml`
    set them. No trainer, CLI command, or engine reads the attributes.
    `NCCLProfiler` and `DistributedMetricsCollector` have no callers. Nothing opens
    `configs/deepspeed_zero3.yaml`. `default_config()` and `configs/default.yaml`
    only store that path.
  - Done. Removed the two classes and the `torch` and `torch.distributed` imports
    that only they used. `RankAwareTelemetry` stays. Removed the four fields,
    the same arguments in `toy_mode()` and `default_config()`, and the same keys
    in `configs/toy.yaml` and `configs/default.yaml`. Deleted
    `configs/deepspeed_zero3.yaml`.
  - `test_deepspeed_config_requirement` only asserted those fields, so it is
    gone. The YAML load tests pointed at `parent.parent.parent`
    (`/home/min/configs/`) and never opened the repo files. They now open
    `configs/toy.yaml` and `configs/default.yaml`.
  - CPU pytest on this branch, `.venv`, torch 2.15.0+cpu, `PYTHONPATH=src`,
    `pytest -m "not slow and not gpu"`: 79 collected, 68 passed, 11 failed, in
    21.00s. The 11 names are the ones already listed under S0.6. The passing
    test that left with the fields is `test_deepspeed_config_requirement`.
    `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` pass.
  - The optional `deepspeed` extra in `pyproject.toml` is unchanged. Nothing
    imports DeepSpeed. `docs/operations/setup.md` and
    `docs/operations/system_design.md` still describe an NCCL install. They do
    not name these classes or this YAML.

### [x] S0.9c Docs truth pass
- Depends on: S0.9b.
- Branch: `docs/S0.9c-docs-truth`.
- Remove the optional `deepspeed` extra from `pyproject.toml`.
- Delete `docs/operations/setup.md` and `docs/operations/system_design.md`.
  They describe multi-node SLURM, Kubernetes, and NCCL setup this repo never had.
- Fix links to those files in `docs/index.md`, `README.md`, and
  `docs/governance/contributing.md`.
- In the PR, list every production, scale, or hardware claim in
  `docs/core/philosophy.md`, `docs/governance/security.md`,
  `docs/educational_deepdive.md`, `docs/DEVELOPMENT.md`, and `docs/index.md`
  that no code or result file supports, and propose delete or rewrite per file.
  Do not edit those claims in this item.
- After the owner decision below, `git grep -i "production\|petascale\|multi-node\|slurm\|kubernetes\|infiniband\|deepspeed"`
  matches this file, and `LICENSE` where the Apache-2.0 text says "reproduction".
- Notes:
  - 2026-10-30: `origin/main` is `96a648c` (PR #14). S0.9b is on main. This branch
    starts there.
  - Done. `fbcc8a7` removes the optional extra. The same pin, `deepspeed==0.10.1`,
    was in `requirements.txt`, `requirements-fine-tune.txt`, and
    `requirements-reward.txt`. Nothing imports DeepSpeed, so those pins are gone
    too. The project description now matches the README scope. It does not say
    production-grade or multi-node.
  - `3cbd7b5` deletes the two operations docs. README had no link to them.
    `docs/index.md` drops the two table rows and nothing else.
    `docs/governance/contributing.md` no longer names `docs/operations/setup.md`.
    New command-line options are documented in `scripts/README.md` only.
  - `docs/core/ARCHITECTURE.md` no longer spells the ZeRO-3 filename. The S0.9b
    notes above still do. That keeps the grep off that page.
  - The claim text in the five files is not rewritten. `docs/index.md` changed
    only by the two deleted rows. `docs/DEVELOPMENT.md` still lists the deleted
    pages in its tree. The delete/rewrite proposal is in the PR.
  - `git grep -i "slurm\|kubernetes\|infiniband\|multi-node\|deepspeed"` returns
    this file and `docs/core/philosophy.md` only. The other four files in the
    claim list do not contain those strings. Their unsupported claims use other
    words, including production, CUDA, and 100% type coverage.
  - CPU pytest on this branch, `.venv`, torch 2.15.0+cpu, `PYTHONPATH=src`,
    `pytest -m "not slow and not gpu"`: 79 collected, 68 passed, 11 failed, in
    19.76s. The 11 names are the ones already listed under S0.6. No other test
    failed. `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` pass.
  - 2026-10-30 owner decision, done on this branch. Deleted
    `docs/core/philosophy.md` and `docs/governance/security.md` (`116dd75`).
    `docs/index.md` drops those two rows and nothing else. Added root
    `SECURITY.md`: report through GitHub private vulnerability reporting or a
    GitHub issue. Research codebase, no security guarantees, no feature claims.
  - `df564d9` rewrites `docs/educational_deepdive.md`. The literature sections
    and the four paper references stay. The notebook inventory, the cluster-scale
    section, and the sentence that called `src/rlhf_platform/` a production
    implementation are gone. `docs/DEVELOPMENT.md` now matches the one CPU CI
    job: Python 3.11+ in `pyproject.toml`, Python 3.12 in CI because of the
    numpy pin, CPU torch, ruff, `pytest -q`, and the cu126 wheel for Pascal
    (sm_61). The clone URL is `https://github.com/Mattral/rlpost`.
  - `587df62` removes the word production from the files that still had it
    outside this roadmap. `TrainingConfig.default_config` and
    `configs/default.yaml` use `rlhf_default_run` and `./outputs/default_run`.
    The module docstring, the `test_default_config` docstring, and the purpose
    sentence in `docs/governance/contributing.md` changed with them. Tests do
    not assert the old run name.
  - `git grep -i "production\|petascale\|multi-node\|slurm\|kubernetes\|infiniband\|deepspeed"`
    returns this file and `LICENSE`. Every `LICENSE` hit is the substring inside
    "reproduction" in the Apache-2.0 text. That file was not edited.
  - CPU pytest after the decision, same command: 79 collected, 68 passed,
    11 failed, in 20.40s. The 11 names are unchanged. Ruff check and format
    check pass.
  - `a90d4de` rewrites `docs/governance/contributing.md` (27 lines). It names
    the CPU job (`ruff check`, `ruff format --check`, `pytest -q`), branching
    from `main`, one roadmap item per PR, Conventional Commits, a `ROADMAP.md`
    update on every PR, and result files under `benchmarks/results/`. The same
    commit drops the unused mypy extra, the `[tool.mypy]` table, and the
    `.mypy_cache/` ignore. `git grep -i "enterprise\|gatekeep\|mypy\|bandit\|pre-commit\|release/"`
    matches only this file.
  - CPU pytest after that rewrite: 79 collected, 68 passed, 11 failed. The 11
    names are unchanged. Ruff check and format check pass.

### [ ] S0.10 P2000 CUDA smoke test (optional)
- Use the cu126 wheel, fp32, a tiny model, and run 20 steps each of SFT, DPO and PPO.
- This catches device and dtype bugs only. **It is not a benchmark.**
- Notes:
  - 2026-11-02: the cu126 index has no torch 2.15.0 wheel. The newest stable
    wheel is `torch==2.14.1`. `docs/DEVELOPMENT.md` names that install. The
    package floor stays `torch>=2.15`.

---

## Stage 1 — First measured result (owner GPU: Colab Pro L4)

The exit criterion for Stage 1 is one honest, reproducible comparison against TRL, including
whatever it shows.

### [ ] S1.1 Pre-registration
Write `benchmarks/PREREG.md` **before** any GPU run. It fixes:
- **Model:** Qwen2.5-0.5B-Instruct, unless the owner changes it.
- **Data:** an HH-RLHF slice with fixed held-out split.
- **Seeds:** 3.
- **Budget:** steps / tokens.
- **Metrics:**
  - held-out reward-model score
  - length-controlled win-rate vs SFT
  - KL to reference
  - mean response length
  - wall-clock time
  - peak memory
  - generation vs train time per step
- **Pass/fail criteria** for "ours matches TRL", stated in advance.
- Notes:

### [ ] S1.2 Benchmark harness
- Config-driven runner that writes one JSON per run to `benchmarks/results/`: git SHA and dirty flag,
  package versions, GPU, dtype, model ids, data slice, seeds, full config, per-step and final metrics.
- Resumable.
- A Colab notebook (Drive checkpoints) and a Modal script (volume checkpoints) that run the
  same config.
- A plotting script that reads JSON only.
- Notes:
  - DPO `train_step` is one pair, shape `[1, T]`. Batched collation is an
    S0.11 note. It has to land before this harness.

### [ ] S1.3 Evaluation module
- Held-out reward scoring, a length-controlled win-rate judge, and KL and length statistics.
- CPU-tested on a tiny model.
- Notes:

### [ ] S1.4 Run: DPO ours vs TRL DPO, 3 seeds
- Owner GPU run. The agent prepares the exact commands, then marks this `[!]`.
- Notes:

### [ ] S1.5 Run: PPO ours vs TRL PPO, 3 seeds
- Owner GPU run.
- Notes:

### [ ] S1.6 Results write-up
- README results section generated from the stored JSON, with full scope stated.
- Failures and null results are kept.
- Update "Measured scope".
- Notes:

---

## Stage 2 — Frontier-relevant work (owner GPU: Modal A100 if needed)

### [ ] S2.1 GRPO with a verifiable reward
- Group-relative advantages, no critic.
- GSM8K exact-match reward.
- CPU parity tests in the style of S0.6.
- Notes:

### [ ] S2.2 Run: PPO vs GRPO at equal budget
- Owner GPU run. Pre-register first.
- Notes:

### [ ] S2.3 vLLM rollouts and weight sync
- Move generation to vLLM.
- Implement trainer → inference weight sync.
- Measure the step-time breakdown (generation / scoring / training / sync).
- Notes:

### [ ] S2.4 Async / off-policy rollouts
- Overlap generation with training.
- Measure throughput gain against the quality cost of policy staleness.
- This is what turns the old "asymmetric process groups" design into a measured result.
- Notes:

### [ ] S2.5 (Optional) Speculative decoding for rollouts
- Try FlashSpec-style drafting to speed up rollout generation.
- Only claim a result if it is measured.
- Notes:

---

## Change log

Append one line per finished item: `YYYY-MM-DD · <item-id> · <outcome> · <PR/SHA>`

- 2026-10-30 · — · Roadmap created; replaces the old `roadmap.md` (whose phase "completion" statuses were not backed by measurements) · —
- 2026-10-30 · S0.1 · Gitignore, untracked caches and egg-info, ROADMAP.md replaces the old roadmap. Phase-completion docs kept pending an owner decision. · 33bed64
- 2026-10-30 · S0.2 · README states the CPU test result and drops unmeasured numbers. The GPU-count table moved to docs/design.md as a design target. The fake telemetry JSON was deleted. · 5f288b9
- 2026-10-30 · S0.3 · CI is one CPU job: pinned torch, ruff, and pytest, with no swallowed failures. Ruff passes. Pytest is 69 passed, 13 failed, and the job is expected to be red until those failures are fixed. · 42c71a9
- 2026-10-30 · owner · Deleted the eight unverified docs. Judged each of the 13 failures and left the 11 algorithm disagreements red. Fixed the CPU pin_memory crash and the 1-D rollout cat. Added S0.11. · a8db00d
- 2026-10-30 · S0.6 · Triage recorded, not implemented. Margin is beta-scaled and reference-adjusted (both test and code wrong). Accuracy uses that reward (code's 0.75 stands). `dpo_beta` becomes its own config field. S0.6b confirms against TRL. · 615cd83
- 2026-10-30 · S0.4 · One PPO trainer (`PPOTrainer`). The second engine and its loss module are gone. The logprob gather is unchanged and lives in `train_step` only. `docs/CODE_MAP.md` is written from the code. · 9bfc8a5
- 2026-10-30 · S0.4b · Removed the hardcoded benchmark script, renamed the padding-label collator, and dropped the dead early-stopping callback and the constant DPO explained-variance field. The 11 known failures are unchanged. · b500b17
- 2026-10-30 · S0.9 · Deleted the unused process-group, hook, and checkpoint modules, the multi-rank script, the cluster YAML, and docs/design.md. The 11 known failures are unchanged. Two passing tests went with the deleted module. · f8ce6d0
- 2026-10-30 · S0.9b · Removed the unused NCCL profiler, the distributed metrics collector, the four TrainingConfig device fields, and the unread ZeRO-3 YAML. The 11 known failures are unchanged. One passing config test went with the fields. · c72ccd7
- 2026-10-30 · S0.9c · Removed the unused DeepSpeed extra and deleted the operations docs that described SLURM, Kubernetes, and NCCL. Claim text in the five files is unchanged pending an owner decision. · 3cbd7b5
- 2026-10-30 · S0.9c · Owner decision: deleted philosophy.md and governance/security.md, added SECURITY.md, and rewrote the deep dive and DEVELOPMENT.md. The grep matches this file, and LICENSE on the substring inside "reproduction". · 116dd75
- 2026-10-30 · S0.9c · Rewrote contributing.md to the CPU CI job. The unused mypy config is gone, so the governance grep matches only this file. · a90d4de
- 2026-11-02 · S0.5 · The package, project name, and CLI path are rlpost. No shim. The default run is rlpost_default_run. · 2bf145a
- 2026-11-02 · S0.5b · Dependencies are the imported packages. The test extra pins TRL 1.17.1. CI installs that extra under the CPU lock. The 11 known failures are unchanged. · 87efbe5
- 2026-11-02 · S0.5b · Project URLs point at Mattral/rlpost. Pascal installs torch==2.14.1 from the cu126 index. The package floor stays torch>=2.15. · #17
- 2026-11-04 · S0.6a · One shifted logprob gather, parity-tested against TRL 1.17.1 selective_log_softmax. The two train_step tests pass. Nine older failures remain. · 48c6e03
- 2026-11-04 · S0.7 · Notes only. PPO train_step applies no mask. Policy loss, entropy, and KL include prompt and padding tokens. The target is response-token-only masking, with tests that fail on that behavior. Added optional S0.6f, an fp32 logsumexp gather before Stage 1. · b1abaff
- 2026-11-04 · S0.6b · Sigmoid DPO matches TRL 1.17.1 `_compute_loss`: beta-scaled rewards, completion-token sum, `dpo_beta` field. Seven DPO failures pass. Two PPO loss tests remain. · 9cb030b
- 2026-11-04 · S0.6b · Parity covers seeds 0-5, unequal completion masks, and policy gradients at atol 1e-6. Reference grads stay None. The mock gradient test is gone. Two PPO loss tests remain. · c301a03
- 2026-11-04 · S0.11 · Notes only. DPO train_step is one pair, shape [1, T]. Batched collation, with the S0.6b parity test on a batch of 4, has to land before S1.2. Drop the max_grad_norm and log_to_wandb fallbacks and read the existing fields. · 81b6834
- 2026-11-04 · S0.6c · GAE matches the T=4 backward recursion. Padding does not enter the backup. TRL 1.17.1 has no GAE helper. Two PPO loss tests remain. · 3f875cf
- 2026-11-04 · S0.6d · Clipped surrogate, optional value clip (`value_clip`, with the one-half), and entropy matched to TRL 1.17.1 `entropy_from_logits`. The CPU suite is green: 130 passed. · 0107a29
- 2026-11-04 · S0.6d · Both value modes use the one-half. Non-binding ratios keep r*A, and always clipping that input is a different loss. 132 passed. · a1bb38e
- 2026-11-04 · S0.6e · Sampled-token KL(policy || reference), default k3. k1 and k3 match the closed form. GRPO 1.17.1's per-token KL is that k3 term. 141 passed. · 3f0319f
- 2026-11-06 · S0.6e · Loss default is k2. Its gradient matches KL(policy || reference). k3 stays and its gradient is the reverse KL. The loss rejects k1. 142 passed. · d4bd358
