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

### [ ] S0.5 Package rename `rlhf_platform` → `rlpost`
- Depends on: the owner renaming the GitHub repo first.
- Update imports, `pyproject.toml`, the CLI entry point and docs. Keep a note in the README
  pointing to the old repo name.
- Notes:

### [ ] S0.6 Numerical parity tests (core correctness)
For each check below, compare against TRL (pinned version) or a closed-form computation,
on a tiny model on CPU, with an explicit tolerance:

- **a.** per-token logprob gather (shift / off-by-one, masking)
- **b.** DPO loss, chosen/rejected rewards and margins vs TRL's DPO loss on identical inputs
- **c.** GAE advantages and returns vs a hand-computed small example
- **d.** PPO clipped policy loss, value loss (clipped) and entropy vs a reference implementation
- **e.** KL estimator: name it (k1/k2/k3) and test it

Split into S0.6a–e if needed. Record any discrepancy found in Notes. A bug found here is a result.
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
    in `ppo_engine.py` is still a code bug for S0.6a. S0.4 should treat that gather as a
    reason not to keep that engine unchanged.
  - S0.6e: the current penalty is full-vocabulary KL(ref || policy); the expected target
    is KL(policy || ref) on sampled tokens with a named estimator.

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

### [ ] S0.10 P2000 CUDA smoke test (optional)
- Use the cu126 wheel, fp32, a tiny model, and run 20 steps each of SFT, DPO and PPO.
- This catches device and dtype bugs only. **It is not a benchmark.**
- Notes:

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
