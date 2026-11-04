# Code map

Written from the code on this branch. Shapes below are what the functions use, not a promise that a training run has been executed. Status links go to [ROADMAP.md](../ROADMAP.md).

`tested` means a CPU test calls the behavior. `untested` means nothing in `tests/` calls it. `known-broken` means the code disagrees with the algorithm it claims, and a later item owns the fix. A module can be more than one of these.

## `src/rlpost/__init__.py`

Package marker. Lists `alignment` and `utils` in `__all__`. It does not import them. No tensors.

Status: untested. The version string is `"0.1.0"`.

## `src/rlpost/config.py`

Pydantic models for a run: `ModelConfig`, `OptimizationConfig`, `AlignmentConfig`, `DatasetConfig`, `TrainingConfig`. No tensors. YAML and JSON round-trip through `from_yaml`, `to_yaml`, `from_json_file`, and `to_json_file`. `TrainingConfig.toy_mode()` builds a small config the unit tests use.

`PPOTrainer` reads `alignment.gamma`, `gae_lambda`, `kl_coefficient`, `target_kl`, `ppo_epsilon`, `entropy_coefficient`, `value_loss_coefficient`, `advantage_normalization`, and `optimization.max_grad_norm`, plus `log_to_wandb`. It does not read `alignment.ppo_num_epochs`. SFT and the reward trainer read learning rate, weight decay, warmup, grad norm, mixed precision, and batch size. They also read `model.lora_target_modules`, which is not a field on `ModelConfig`. `PreferencePairDataset` reads `dataset.dataset_size`, which is not a field on `DatasetConfig`. Those two attributes raise `AttributeError` if that line runs. `model.use_peft_lora` and `model.attn_implementation` are stored. `tests/test_config.py` checks both. No loader reads `attn_implementation`, and the trainers apply LoRA without reading `use_peft_lora`.

Status: tested for construction, validation, and YAML load. `AlignmentConfig.dpo_beta` is a field, default `0.1`, and it must be `> 0`. It is not `kl_coefficient`. `model.lora_target_modules` and `dataset.dataset_size` are still missing. See [S0.11](../ROADMAP.md#s011-cli-actually-runs-sftrmdpoppo-end-to-end-on-a-tiny-model-on-cpu).

Tests: `tests/test_config.py`.

## `src/rlpost/dataset.py`

Loads preference pairs as strings: `prompt`, `chosen`, `rejected`, and optional reward floats. `PreferencePairDataset.load` prefers a local JSONL cache, then Hugging Face `datasets`. `get_dataloader` tokenizes `prompt + chosen` with `return_tensors="pt"` (each field is `[1, L]` before padding) and collates with `IgnorePaddingLabelCollator`.

The collator asks the tokenizer to pad to `max_seq_length`. It then sets `labels = input_ids` and `labels[attention_mask == 0] = -100`. After a successful collate the batch is `input_ids` `[B, max_seq]`, `attention_mask` `[B, max_seq]`, `labels` `[B, max_seq]`. No test builds that batch. The class only sets padding labels to -100. It does not look at a LoRA module.

`ToyDatasetLoader` and the HH-RLHF branch both require a `prompt` column. The public Anthropic HH-RLHF split is `chosen` and `rejected` dialogues, not `prompt` / `chosen` / `rejected`. A real load raises `KeyError` or `ValueError`. Nothing in `tests/` loads it.

Status: untested. The HH-RLHF column mismatch is known-broken. See [S0.11](../ROADMAP.md#s011-cli-actually-runs-sftrmdpoppo-end-to-end-on-a-tiny-model-on-cpu).

Tests: none.

## `src/rlpost/sft_engine.py`

`load_model_with_lora` loads `AutoModelForCausalLM` and wraps it with PEFT LoRA (`task_type="CAUSAL_LM"`). It always applies LoRA. It does not read `use_peft_lora`. Optional 4-bit or 8-bit quantization goes through `BitsAndBytesConfig`. `SFTTrainer.train` builds a Hugging Face `Trainer` on `get_dataloader(...).dataset`, with `eval_dataset=None`, `evaluation_strategy="steps"`, and `load_best_model_at_end=True`. There is no early-stopping callback. The return value is the Trainer result object (`training_loss` is an attribute). The annotation says `dict`.

No tensor is built in this file except inside the Hugging Face trainer, which is not called by tests.

Status: untested. See [S0.11](../ROADMAP.md#s011-cli-actually-runs-sftrmdpoppo-end-to-end-on-a-tiny-model-on-cpu).

Tests: none. `python -m rlpost.cli train-sft` constructs `SFTTrainer` and calls `train`.

## `src/rlpost/reward_engine.py`

`load_reward_model_with_lora` loads `AutoModelForSequenceClassification` with `num_labels=1` and PEFT LoRA (`task_type="SEQ_CLS"`).

`_tokenize_preference_pair` returns a dict of length-`max_seq_length` vectors (the batch dim from the tokenizer is squeezed). `train` runs one pair at a time: `input_ids` `[1, max_seq]`. The score is `logits.squeeze(-1)`. For a one-label head, `logits` is `[1, 1]` and the score is `[1]`.

`compute_loss(chosen_scores, rejected_scores)` expects both tensors to share a shape (the training loop uses `[1]`). It computes `BCEWithLogits(chosen - rejected, ones)`. That is `-log sigmoid(chosen - rejected)`. Accuracy is the fraction of rows with `chosen > rejected`, a 0-dim tensor. `train` returns `{"loss": float, "accuracy": float}`. `score` returns a Python float. `score_batch` returns `list[float]` and loops one row at a time.

Status: untested. See [S0.11](../ROADMAP.md#s011-cli-actually-runs-sftrmdpoppo-end-to-end-on-a-tiny-model-on-cpu).

Tests: none. `train-reward` calls `train` and reads `result["loss"]` and `result["accuracy"]`.

## `src/rlpost/dpo_engine.py`

Reward-model-free DPO with a frozen reference policy. The module used to say "reference-free". That was the wrong term. The reference model is loaded and frozen with `requires_grad_(False)`.

`compute_dpo_loss` takes four sequence log-probabilities of shape `[B]`. The parameter names still say logits.

```
chosen_reward   = beta * (policy_chosen - ref_chosen)       # [B]
rejected_reward = beta * (policy_rejected - ref_rejected)   # [B]
logits          = chosen_reward - rejected_reward           # [B]
loss            = -logsigmoid(logits).mean()                # scalar
margin          = mean(logits)                              # Python float
accuracy        = mean(chosen_reward > rejected_reward)     # Python float
```

`beta` is `config.alignment.dpo_beta`. `policy_diff_mean` is the mean of `policy_chosen - policy_rejected`. That one is not scaled by beta and does not subtract the reference.

`compute_log_probs(model, input_ids [B, T], attention_mask [B, T], completion_mask [B, T])` calls `gather_logprobs` (logits at `t` score token `t+1`, result `[B, T-1]`), multiplies by `completion_mask[:, 1:]`, and sums over that axis. The result is `[B]`. Prompt tokens and padding are 0 in the shifted mask, so they add nothing. This is a sum, not a mean. `attention_mask` is passed to the model. It is not the reduction mask. The method is not wrapped in `no_grad`. `train_step` calls the reference inside `torch.no_grad`.

TRL 1.17.1 `DPOTrainer._compute_loss`, sigmoid loss, `label_smoothing` 0, `ld_alpha` None, sums the same completion tokens. The fused head writes 0 where `completion_mask` is 0. The parity test calls that method. Loss absolute tolerance is `1e-6`. The logged batch mean of the sums uses absolute tolerance `1e-5`. On CPU, torch 2.15, the tiny Llama, the largest gap on that mean was 1.907e-6. See [S0.6b](../ROADMAP.md#s06b-dpo-loss-rewards-and-margins).

`_tokenize_for_loss` marks the shared prefix of the prompt ids and `prompt + response` as prompt, and marks padding as 0. It assumes right padding. A left-padded prompt is not a prefix. Padding side is [S0.7](../ROADMAP.md#s07-classic-rlhf-edge-case-tests).

`DPOMetrics` holds `dpo_loss`, `policy_diff_mean`, `margin_mean`, and `accuracy`. There is no value head, and no explained-variance field.

`train_step` tokenizes one pair, runs the policy forward with gradients, the reference forward under `no_grad`, then `loss.backward()`. No test calls `train_step`.

Status: the loss, the margin, the accuracy, `dpo_beta`, the completion-token sum, and construction on the tiny Llama are tested. `train_step` is untested.

Tests: `tests/test_dpo_engine.py`. The two PPO loss tests in `tests/test_ppo_engine.py` still fail. That is [S0.6d](../ROADMAP.md#s06d-ppo-clipped-policy-loss-and-value-loss).

## `src/rlpost/ppo_engine.py`

The only PPO implementation. `PPOTrainer` holds a policy, a value model, a frozen reference, and one optimizer.

`GeneralizedAdvantageEstimation.compute(rewards, values, next_values, dones)` requires rank-2 tensors of the same shape `[B, T]`.

```
delta_t = reward_t + gamma * next_value_t * (1 - done_t) - value_t
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}     # backward
return  = A + value
```

Outputs are `[B, T]`. There is no padding mask. `gamma == 0` and zero values make `A == reward`. That case is tested.

`compute_policy_loss(log_probs_new, log_probs_old, advantages)` uses matching shapes (the tests pass `[B]`):

```
ratio = exp(log_probs_new - log_probs_old)
loss  = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))    # scalar
```

`ratio` has the same shape as the inputs. The two loss tests fail because `torch.randn` inputs do not require grad. That is a test bug, left for S0.6. The formula is the clipped surrogate.

`compute_value_loss(value_pred, returns)` is `value_loss_coefficient * MSE`. Same shape in, scalar out. No clip. A clipped value loss is [S0.6d](../ROADMAP.md#s06-numerical-parity-tests-core-correctness).

`compute_entropy_bonus(logits [..., A])` is the categorical entropy, mean over every axis except the last. Scalar. Tested for `[B, A]`.

`compute_kl_divergence(logits_new, logits_old)` is the full-vocabulary categorical KL(old || new), then a mean. Scalar. `train_step` calls it as `(policy_logits, ref_logits)`, so the penalty is KL(reference || policy), not KL(policy || reference). k1, k2, and k3 are not implemented. Tested only for non-negativity and for identical inputs (about 0). Naming the estimator is [S0.6e](../ROADMAP.md#s06-numerical-parity-tests-core-correctness). The direction was not changed in S0.4.

`compute_explained_variance(value_pred, returns)` returns a Python float, `1 - Var(returns - values) / Var(returns)`, or `0` when the return variance is 0. Tested.

`adaptive_kl_penalty(kl: float)` multiplies `kl_coefficient` by 2 when `kl > 1.5 * target_kl`, by `0.5` when `kl < target_kl / 1.5`, and clips the coefficient to `[0.001, 10]`. Tested.

`train_step(batch, old_log_probs)` reads `input_ids`, `rewards`, `values`, `next_values`, and `dones`. The tests pass `[B, T]` for each. The reference forward is under `no_grad`. Advantages are normalized over the whole tensor when `advantage_normalization` is true, including positions that would be padding. The total loss is

```
policy_loss + value_loss - entropy_coefficient * entropy + kl_coefficient * kl
```

`train_step` calls `gather_logprobs(policy_logits, input_ids)`. Logits at `t` score token `t+1`. The result is `[B, T-1]`. `old_log_probs` must have that shape. GAE advantages stay `[B, T]`, and the policy ratio uses `advantages[:, 1:]`. No mask is applied in this step. The two `test_train_step_*` tests build a tiny random-init Llama and pass.

`value_pred` is used for the value loss. GAE uses the batch's `values` and `next_values`, not `value_pred`.

Status: GAE, entropy, KL-on-identical-inputs, explained variance, the adaptive coefficient, and `train_step` on a tiny causal LM are tested. The clipped loss and the value loss are known-broken only on the test side (no grad). See [S0.4](../ROADMAP.md#s04-code-ownership-pass), [S0.6](../ROADMAP.md#s06-numerical-parity-tests-core-correctness), and [S0.7](../ROADMAP.md#s07-classic-rlhf-edge-case-tests).

Tests: `tests/test_ppo_engine.py` and `tests/test_logprob_gather.py`. Still failing: `test_policy_loss_computation`, `test_value_loss_computation`.

## `src/rlpost/cli.py`

Typer commands. No tensors of its own.

`train-sft` and `train-reward` build the trainers and call `train`. `run-ppo` checks `alignment.method == "ppo"`, writes the config YAML, and returns. `run-dpo` prints a placeholder and exits 0. It does not construct `DPOTrainer`.

Help text still mentions a T4 timing and "Phase 4". Those strings are not measurements.

Status: untested. See [S0.11](../ROADMAP.md#s011-cli-actually-runs-sftrmdpoppo-end-to-end-on-a-tiny-model-on-cpu).

Tests: none.

## `src/rlpost/alignment/__init__.py`

Re-exports `rollout`. PPO is not in this package.

Status: untested as an import. The rollout pieces below are what the tests import directly.

## `src/rlpost/alignment/rollout.py`

`Rollout` is one trajectory. Per row after `RolloutGenerator`: `query_tokens` `[T_query]` long, `response_tokens` `[T_response]` long, `reward` a scalar (or `[1]` if a head squeezes that way), `logits_policy` `[T_response, V]`, `logits_reference` `[T_response, V]`. The generator's forward sees a batch. The stored row does not.

`RolloutBuffer.add` detaches tensors to CPU. It calls `pin_memory` only when `torch.cuda.is_available()`. `sample_batch` returns a Python list of up to `batch_size` rollouts. `clear` empties the list. Capacity overwrites the oldest slot.

`RolloutCollator` pads queries to the longest query in the batch and responses to `max_response_length`. Outputs: `query_tokens` `[B, max_query]` long, `response_tokens` `[B, max_response]` long, `logits_policy` and `logits_reference` `[B, max_response, V]`, `rewards` `[B]` if each reward is a scalar, `query_lengths` `[B]`, `response_lengths` `[B]`. Short rows are zero-padded. There is no padding mask in the dict. Lengths are returned so a caller could build one. No caller does.

`RolloutGenerator.generate_rollout(query_tokens [B, T_query])` samples `max_response_length` tokens. Each step reads `policy_out.logits[:, -1, :]` and the same slice from the reference, shape `[B, V]`, and appends a token of shape `[B, 1]`. `_sample_token` divides by `temperature` and, when `top_p < 1`, drops tokens whose sorted cumulative probability is already above `top_p`, always keeping the top token. The reward forward is `reward_model(cat(query, response))`. `_decode_reward` uses `logits.squeeze(-1)` when `logits` is rank 2 (`[B, 1]` becomes `[B]`). Rank 3 averages the last two axes. A `pooler_output` is averaged over its last axis. Anything else raises.

Status: buffer, collator, and generator are tested on CPU with stand-in modules. The rank-3 reward branch is untested. See [S0.9](../ROADMAP.md#s09-delete-the-orphaned-distributed-scaffolding) for the CPU pin fix, and [S0.7](../ROADMAP.md#s07-classic-rlhf-edge-case-tests) for reward position.

Tests: `tests/test_rollout.py` (3 tests, passing).

## `src/rlpost/utils/telemetry.py`

`RankAwareTelemetry.log_event` appends a `TelemetryEvent` and logs one JSON line. `log_ppo_step`, `log_communication`, `log_checkpoint`, and `log_memory_usage` are thin wrappers. `log_communication` sets `throughput_gbps = (data_size_mb * 8) / (duration_ms / 1000) / 1000`. No test checks that conversion. Nothing in `tests/` or `scripts/` constructs `RankAwareTelemetry`.

`NCCLProfiler` and `DistributedMetricsCollector` were removed in [S0.9b](../ROADMAP.md#s09b-remove-unused-distributed-config-and-telemetry). Nothing called them. Stage 2 builds distribution from scratch.

Status: untested.

Tests: none.

## `src/rlpost/utils/__init__.py`

Re-exports `telemetry`.

Status: untested. `tests/test_rollout.py` imports `alignment.rollout` directly.

## `train.py`

Prints one line to stderr and exits 2. It does not import a model or `PPOTrainer`. The previous body only existed to call `DistributedPPOEngine`.

Status: not a trainer. See [S0.4](../ROADMAP.md#s04-code-ownership-pass) and [S0.11](../ROADMAP.md#s011-cli-actually-runs-sftrmdpoppo-end-to-end-on-a-tiny-model-on-cpu).

Tests: none.

## Code I cannot explain confidently

External history on the current files: Kyle (`insiderzzz0005@gmail.com`) wrote the first versions of `config.py`, `dataset.py`, `sft_engine.py`, `reward_engine.py`, `dpo_engine.py`, `ppo_engine.py`, `cli.py`, and the config and PPO tests. thitsarhmwe-commits wrote `tests/test_dpo_engine.py`. Later edits on those files are from the repo owner. I can walk the PPO and DPO formulas above. The items below are the parts I would not defend.

S0.4b removed four of these. `IgnorePaddingLabelCollator` sets padding labels to -100. `EarlyStoppingCallback` is gone. `DPOMetrics` has no explained-variance field. `tests/run_benchmark_comparison.py` is deleted.

From Kyle, still unexplained as an API:

- `sft_engine.py` and `reward_engine.py` read `ModelConfig.lora_target_modules`. `dataset.py` reads `DatasetConfig.dataset_size`. Those fields are not declared. I can explain the `AttributeError`. I cannot explain the missing fields as an API. S0.11 declares and validates them.

From thitsarhmwe-commits:

- The DPO margin, beta, and empty stand-in failures were the S0.6 triage. S0.6b implemented them and those tests pass. The two PPO loss tests that still fail are S0.6d.

S0.9 deleted the process-group module, the communication hooks, the async checkpoint writer, and the multi-rank script. Those entries are gone with the files.

Still in the tree:

- `RolloutGenerator._decode_reward` averages a rank-3 logit tensor over the last two axes. I cannot say which reward head that shape is for. S0.7's reward-position test settles it.
