# Code map

Written from the code on this branch. Shapes below are what the functions use, not a promise that a training run has been executed. Status links go to [ROADMAP.md](../ROADMAP.md).

`tested` means a CPU test calls the behavior. `untested` means nothing in `tests/` calls it. `known-broken` means the code disagrees with the algorithm it claims, and a later item owns the fix. A module can be more than one of these.

## `src/rlpost/__init__.py`

Package marker. Lists `alignment` and `utils` in `__all__`. It does not import them. No tensors.

Status: untested. The version string is `"0.1.0"`.

## `src/rlpost/config.py`

Pydantic models for a run: `ModelConfig`, `OptimizationConfig`, `AlignmentConfig`, `DatasetConfig`, `TrainingConfig`. No tensors. YAML and JSON round-trip through `from_yaml`, `to_yaml`, `from_json_file`, and `to_json_file`. `TrainingConfig.toy_mode()` builds a small config the unit tests use.

`PPOTrainer` reads `alignment.gamma`, `gae_lambda`, `kl_coefficient`, `kl_estimator`, `target_kl`, `ppo_epsilon`, `entropy_coefficient`, `value_loss_coefficient`, `value_clip`, `advantage_normalization`, and `optimization.max_grad_norm`, plus `log_to_wandb`. It does not read `alignment.ppo_num_epochs`. SFT and the reward trainer read learning rate, weight decay, warmup, grad norm, mixed precision, and batch size. They also read `model.lora_target_modules`, which is not a field on `ModelConfig`. `PreferencePairDataset` reads `dataset.dataset_size`, which is not a field on `DatasetConfig`. Those two attributes raise `AttributeError` if that line runs. `model.use_peft_lora` and `model.attn_implementation` are stored. `tests/test_config.py` checks both. No loader reads `attn_implementation`, and the trainers apply LoRA without reading `use_peft_lora`.

Status: tested for construction, validation, and YAML load. `AlignmentConfig.dpo_beta` is a field, default `0.1`, and it must be `> 0`. It is not `kl_coefficient`. `AlignmentConfig.kl_estimator` names `k1`, `k2`, or `k3`, default `k2`. Validation rejects `k1` with the same message as `compute_kl_divergence`. `k2` and `k3` are legal. `AlignmentConfig.value_clip` defaults to `None` and, when set, must be `>= 0`. It is not `ppo_epsilon`. `model.lora_target_modules` and `dataset.dataset_size` are still missing. See [S0.11](../ROADMAP.md#s011-cli-actually-runs-sftrmdpoppo-end-to-end-on-a-tiny-model-on-cpu).

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

`_tokenize_for_loss` marks the shared prefix of the prompt ids and `prompt + response` as prompt, and marks padding as 0. It assumes right padding. A left-padded prompt is not a prefix. Batched left padding stays on the [S0.11](../ROADMAP.md#s011-cli-actually-runs-sftrmdpoppo-end-to-end-on-a-tiny-model-on-cpu) collation note.

`DPOMetrics` holds `dpo_loss`, `policy_diff_mean`, `margin_mean`, and `accuracy`. There is no value head, and no explained-variance field.

`train_step` tokenizes one pair, runs the policy forward with gradients, the reference forward under `no_grad`, then `loss.backward()`. No test calls `train_step`.

Status: the loss, the margin, the accuracy, `dpo_beta`, the completion-token sum, and construction on the tiny Llama are tested. `train_step` is untested.

Tests: `tests/test_dpo_engine.py`.

## `src/rlpost/ppo_engine.py`

The only PPO implementation. `PPOTrainer` holds a policy, a value model, a frozen reference, and one optimizer.

`GeneralizedAdvantageEstimation.compute(rewards, values, next_values, dones, mask=None)` requires rank-2 tensors of the same shape `[B, T]`. Axis 1 is time. The recursion does not know about tokens. PPO calls it on the action layout `[B, T-1]`: index `t` is the action that emits token `t+1`, `rewards[:, t]` is that action's reward, and `values[:, t]` is `V` before the action. `next_values[:, t]` is `V` of the next state, supplied by the caller, not read from `values[:, t+1]`.

```
delta_t = reward_t + gamma * next_value_t * (1 - done_t) - value_t
A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}     # backward
return  = A + value                                             # real indices
```

`mask[:, t] == 0` is not a timestep. That index's advantage and return are 0. Its reward, value, and advantage do not enter any other index, and the index before a 0 does not bootstrap from that index. In the GAE tests the 0s are padding. `train_step` passes the gathered response mask, so prompt tokens and padding are 0 there too. `mask is None` means every position is real. `done_t == 1` still stops the backup when the next index is real. `gamma == 0` and zero values make `A == reward` when nothing is masked. That case is tested. The T=4 closed form, done, lambda 0, lambda 1, gamma 1, and padding are in `tests/test_gae.py`. The numbers are the recursion. They do not depend on the token alignment. TRL 1.17.1 has no GAE helper.

`compute_policy_loss(log_probs_new, log_probs_old, advantages, mask=None)` uses matching shapes (the closed-form tests pass `[B]`). `mask` has that same shape. The mean is `sum(x * mask) / sum(mask)`. `None` averages every element.

```
ratio = exp(log_probs_new - log_probs_old)
loss  = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))    # scalar
```

`eps` is `ppo_epsilon`. `log_probs_old` and `advantages` are detached, so the gradient reaches `log_probs_new` only. `ratio` has the same shape as the inputs. The closed form in `tests/test_ppo_loss.py` clips on both sides, including `A < 0`. It also keeps `1.5 * A` when `A < 0` and `0.5 * A` when `A > 0`, and checks that always using the clipped ratio is a different loss on that input.

`compute_value_loss(value_pred, returns, old_values=None)` returns a scalar. `returns` and `old_values` are detached, so the gradient reaches `value_pred` only. Both modes use `value_loss_coefficient * mean(0.5 * squared error)`. When `alignment.value_clip` is `None`, the squared error is `(V - R)^2` and `old_values` is ignored. When `value_clip` is `c >= 0`, `old_values` is required:

```
V_clip = V_old + clip(V - V_old, -c, c)
L_v    = mean(0.5 * max((V - R)^2, (V_clip - R)^2))
loss   = value_loss_coefficient * L_v
```

`c` is not `ppo_epsilon`. A radius of `1e9` does not bind on the hand sample, and that loss equals the unset loss. Both forms are checked against a closed form in `tests/test_ppo_loss.py`. `compute_value_loss(..., mask=None)` uses the same masked mean. `train_step` passes the gathered response mask, shape `[B, T-1]`.

`compute_entropy_bonus(logits [..., A], mask=None)` is the mean of the per-row Shannon entropy. Each row matches `trl.trainer.utils.entropy_from_logits` in TRL 1.17.1. A uniform row is `log(n)`. A `[0.75, 0.25]` row is checked by hand. Scalar. `mask` selects rows and must match `logits` without the last axis. `None` averages every row. `train_step` passes `logits[:, :-1]` and the gathered response mask.

`per_token_kl(logp_policy, logp_ref, estimator)` is the per-token estimator. The two log-probability tensors share a shape. `logp_ref` is detached, so one backward reaches `logp_policy` only. The sampled index is held fixed, so the value and the loss gradient can name different KLs. `lr = logp_ref - logp_policy`.

`k1 = -lr`. Value: unbiased for KL(policy || reference), and a token can be negative. Gradient: the expected gradient w.r.t. the policy logits is 0. The loss does not use it. The value is the InstructGPT penalty inside the reward, `reward - beta * k1`. Reward shaping is not implemented.

`k2 = 0.5 * lr^2`. Value: always >= 0, biased for the KL. Gradient: expected gradient of KL(policy || reference). This is the loss default.

`k3 = exp(lr) - 1 - lr`. Value: unbiased for KL(policy || reference), and >= 0 at every token, because `exp(x) - 1 - x` is convex with minimum 0 at 0. Gradient: expected gradient of KL(reference || policy). TRL 1.17.1 `GRPOTrainer._compute_loss` assigns this expression before it multiplies by the importance ratio when `use_bias_correction_kl` is set. That flag defaults to True. The product is not this estimator. `tests/test_kl.py` checks the values, the sign, the direction, and both gradients.

`compute_kl_divergence(logp_policy, logp_ref, mask=None)` returns the mean of `per_token_kl` for `k2` or `k3`. Scalar, added to the loss. `k1` raises, and `AlignmentConfig` raises the same message when `kl_estimator` is `k1`. `mask` has the same shape. `1` keeps a token and `0` drops it. An all-zero mask returns 0. `None` averages every token. `train_step` passes the gathered response mask, so prompt and padding tokens stay out of the mean. Both log-probability tensors come from `gather_logprobs`, shape `[B, T-1]`.

`compute_explained_variance(value_pred, returns, mask=None)` returns a Python float, `1 - Var(returns - values) / Var(returns)`, or `0` when the return variance is 0. Without `mask`, variance is `torch.var`. With `mask`, variance is the masked mean of squared deviations. `train_step` passes the gathered response mask. Tested.

`adaptive_kl_penalty(kl: float)` multiplies `kl_coefficient` by 2 when `kl > 1.5 * target_kl`, by `0.5` when `kl < target_kl / 1.5`, and clips the coefficient to `[0.001, 10]`. Tested.

`train_step(batch, old_log_probs)` reads `input_ids`, `rewards`, `values`, and `dones`, each `[B, T]`, plus `prompt_lengths` and `response_lengths` (int64 `[B]`) and optional `padding_side` (`"right"` or `"mixed"`, default `"right"`). `"right"` is `[prompt | response | pad]`, and the response starts at `prompt_lengths`. `"mixed"` is a padded generate batch, `[pad | prompt | response | pad]`: prompts are left-padded to `P = max(prompt_lengths)`, and every response occupies `[P, P + response_length)`. A response flush with the end of the row is rejected. It builds one response mask and passes `gathered_mask` into GAE, the value loss, whitening, and the other reductions. The token-level mask is not used on this path. The reference forward is under `no_grad`. When `advantage_normalization` is true, the mean and the std are taken over response actions only. The total loss is

```
policy_loss + value_loss - entropy_coefficient * entropy + kl_coefficient * kl
```

`train_step` calls `gather_logprobs` on the policy logits and, under `no_grad`, on the reference logits. Logits at `t` score token `t+1`. Each result is `[B, T-1]`. `old_log_probs` must have that shape. The value head is shifted the same way: `value_out[:, t]` is the critic after it has read token `t` and before it has read token `t+1`. GAE, the value loss, and whitening run on that action layout, not on `[B, T]` followed by `advantages[:, 1:]`.

```
values[:, t]      = value_out[:, t]         # V(s) for the action at t
next_values[:, t] = value_out[:, t + 1]     # V(s') after that action
rewards[:, t]     = token_reward[:, t + 1]  # reward for emitting token t+1
```

The sequence score is still written on the last response token. In the action layout that column is index `last - 1`. `gathered_mask[:, t]` is 1 when token `t+1` is a response token. Policy loss, entropy, KL, GAE, the value loss, and whitening all use it. The mask is [S0.7](../ROADMAP.md#s07-classic-rlhf-edge-case-tests). The two `test_train_step_*` tests build a tiny random-init Llama whose whole row is the response, and pass.

`value_out[:, t+1]` is the wrong baseline for the action that emitted token `t+1`. That position has already read the sampled token, so the baseline depends on the action. A state-only baseline drops out of the policy gradient because the sum of `∇π(a|s)` over actions is zero. An action-dependent baseline does not. A higher value on the token that was just sampled makes `A = r - V` smaller, so the surrogate reinforces that token less and punishes it more. Tokens the critic scores low get the opposite push. With `gamma = 0` and rewards 0 the advantage is `-V(s)`, and editing `input_ids[:, t+1]` must leave the advantage of action `t+1` unchanged. That is `test_action_advantage_does_not_see_its_own_token`. TRL's PPO trainer, removed before 1.17.1, returned `values[:, :-1]` beside logprobs from `logits[:, :-1]` (the 0.7 `batched_forward_pass`). This is that shift. It is not a numeric match to a PPO loss 1.17.1 does not ship.

`action_values` is the current critic on `[B, T-1]` and is what the value loss trains. GAE reads the same tensor, detached. `batch["values"]` is the rollout value head at each token, shape `[B, T]`. `train_step` passes `values[:, :-1]` into the value loss as `V_old`. That argument is used only when `value_clip` is set. There is no `next_values` batch entry. The bootstrap is `value_out[:, 1:]`.

Status: GAE, the clipped surrogate, both value losses, entropy against the TRL helper, the k1/k2/k3 estimators (k3 matched to TRL 1.17.1 GRPO on identical log-probabilities), explained variance, the adaptive coefficient, and `train_step` on a tiny causal LM are tested. See [S0.4](../ROADMAP.md#s04-code-ownership-pass), [S0.6](../ROADMAP.md#s06-numerical-parity-tests-core-correctness), and [S0.7](../ROADMAP.md#s07-classic-rlhf-edge-case-tests).

Tests: `tests/test_ppo_engine.py`, `tests/test_ppo_loss.py`, `tests/test_gae.py`, `tests/test_logprob_gather.py`, `tests/test_kl.py`, and `tests/test_response_mask.py`.

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

`RolloutGenerator.generate_rollout(query_tokens [B, T_query])` samples `max_response_length` tokens. Each step reads `policy_out.logits[:, -1, :]` and the same slice from the reference, shape `[B, V]`, and appends a token of shape `[B, 1]`. `_sample_token` divides by `temperature` and, when `top_p < 1`, drops tokens whose sorted cumulative probability is already above `top_p`, always keeping the top token. The reward forward is `reward_model(cat(query, response))`. The row is `[prompt | response]` with no extra padding. `_decode_reward` reads a sequence score and `generate_rollout` stores that scalar. The token-level tensor, with the score on the last response token, is what `_decode_reward` returns when a `token_mask` is passed.

`_decode_reward(reward_out, token_mask=None)`:

- `logits` `[B, 1]` are the sequence score.
- `logits` `[B, T]` or `[B, T, K]` are per-token scores. The sequence score is the value at the last real response token. `K` is averaged at that token only. Time is not averaged.
- `pooler_output` `[B, H]` has no time axis. The score is the mean over `H`.
- With `token_mask` `[B, T]`, the return is `[B, T]`: the score at the last 1, and 0 elsewhere. Right padding and the mixed layout both use that index. `train_step` then reads it at index `last - 1`.
- Without `token_mask`, the return is the score `[B]`. Per-token scores then use the last index, which is the last response token only when the row is not right-padded.

Status: buffer, collator, generator, and the reward index are tested on CPU with stand-in modules. See [S0.9](../ROADMAP.md#s09-delete-the-orphaned-distributed-scaffolding) for the CPU pin fix, and [S0.7](../ROADMAP.md#s07-classic-rlhf-edge-case-tests) for the reward position.

Tests: `tests/test_rollout.py` (4 tests, passing).

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

- The DPO margin, beta, and empty stand-in failures were the S0.6 triage. S0.6b implemented them and those tests pass. S0.6d rewrote the two PPO loss tests so their inputs require grad, and those tests pass.

S0.9 deleted the process-group module, the communication hooks, the async checkpoint writer, and the multi-rank script. Those entries are gone with the files.

The rank-3 reward was the last open entry on this list. S0.7 settled it. The sequence score is the value at the last real response token, and that score is written on that token.
