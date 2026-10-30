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
  - Verification on this machine (CPU torch 2.15.0, Python 3.12): `ruff check` and
    `ruff format --check` already fail across the tree. There is no ruff config yet.
    `pytest` cannot collect `tests/test_topology.py` or `tests/test_rollout.py` because
    importing the package loads `comm_hooks.py`, which imports a name torch does not
    export (`ddp_comm_hook`). Of the 77 tests that do collect in config, DPO, and PPO,
    66 pass and 11 fail. Those failures were already in the code. Details are under
    S0.6 and S0.9. S0.1 does not change Python.

### [ ] S0.2 README truth pass
- Fix the DPO description: it is reward-model-free, not reference-free.
- Replace hand-typed test counts and coverage badges with generated values, or remove them.
- Remove the telemetry JSON example (79 GB VRAM, 2,450 tok/s; never measured) and the
  8 / 512 / 10,000-GPU table. If kept at all, move them to `docs/design.md`, labelled
  "design target, not measured".
- Add a "Measured scope" section. Right now it should honestly read "unit-tested on CPU;
  no GPU benchmark yet".
- Notes:

### [ ] S0.3 CI that tells the truth
- Remove jobs pointing at non-existent `tests/integration/` and `tests/benchmarks/`.
- Remove `|| true` from the CI steps.
- Make the default job: CPU torch, `ruff`, `pytest -q` (skipping `slow` and `gpu` markers).
- Pin versions.
- Notes:

### [ ] S0.4 Code ownership pass
- Merge `ppo_engine.py` and `alignment/ppo_engine.py` into one implementation. Delete dead code.
- Write `docs/CODE_MAP.md`: one paragraph per module covering what it does, its inputs and
  outputs with tensor shapes, and which tests cover it. The owner reads this before Stage 0.6.
- Flag any code from external contributors that the agent cannot explain clearly.
- Notes:

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
  - 2026-10-30, found while checking the tree on CPU torch 2.15, not yet fixed:
    `PPOTrainer.train_step` gathers log-probs with `log_probs_new.gather(1, input_ids.unsqueeze(-1))`.
    That raises `Index tensor must have the same number of dimensions as input tensor`
    (`test_train_step_updates_weights`, `test_train_step_returns_metrics`).
    `compute_policy_loss` and `compute_value_loss` return tensors that do not require grad.
    DPO: `test_margin_computation` gets a margin of about 1.10 against an expected value under 0.01.
    `test_beta_affects_loss` finds equal losses at two betas. `AlignmentConfig` has no `dpo_beta`
    field, so `test_custom_beta_from_config` fails validation. Two DPO init tests build `AdamW`
    on an empty parameter list, which torch 2.15 rejects.

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

### [ ] S0.8 Determinism and resumable checkpoints
- Seeding gives identical losses on CPU for 2 runs.
- Kill and resume reproduces the uninterrupted run's step-N metrics (within tolerance).
- This is required before any Colab run.
- Notes:

### [ ] S0.9 Distributed claims: test or delete
- Test the actor/critic vs reference/reward process-group split with `torch.distributed`
  (gloo backend) on 2–4 CPU ranks.
- Test `comm_hooks.py` and `async_io.py` for real, or remove them.
- Delete `scripts/simulate_runtime.py` if it is replaced by real tests.
- Notes:
  - 2026-10-30: `comm_hooks.py` line 15 is `from torch.distributed.algorithms.ddp_comm_hooks import ddp_comm_hook`.
    On torch 2.15 that module has `default_hooks` and `register_ddp_comm_hook`, not `ddp_comm_hook`.
    Importing `rlhf_platform.distributed` or `rlhf_platform.alignment` therefore fails, and
    `tests/test_topology.py` and `tests/test_rollout.py` do not collect.

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
- 2026-10-30 · S0.1 · Gitignore, untracked caches and egg-info, ROADMAP.md replaces the old roadmap. Phase-completion docs kept pending an owner decision. · branch chore/S0.1-repo-hygiene
