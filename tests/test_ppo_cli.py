"""``run-ppo --tiny`` trains on CPU and a fresh process can resume it.

The command builds ``tiny_causal_lm`` and a value head. It does not
download a model. The reference is a frozen copy of the initial policy.
N is 2. One process takes 4 steps. Another takes 2 and writes a
checkpoint. A third process loads that file and takes 2 more. Losses
and metrics use ``==``. Weights use ``torch.equal``.

The toy reward is the sum of token ids in the response window. Pad id
is 0, so padding adds nothing. EOS is included. That sum is written on
the last real response token.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from rlpost.dpo_engine import frozen_reference_copy
from rlpost.ppo_engine import (
    TINY_PPO_PROMPTS,
    PPOTrainer,
    collate_prompts,
    run_ppo_training,
    tiny_ppo_config,
)
from rlpost.repro import seed_everything
from rlpost.tiny_lm import TinyCharTokenizer, tiny_causal_lm, tiny_value_model

ROOT = Path(__file__).resolve().parents[1]
N = 2
SEED = 7


def _run(output_dir: Path, *, steps: int, seed: int, checkpoint: Path, resume: bool) -> None:
    """One fresh ``python -m rlpost.cli run-ppo`` process."""
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "rlpost.cli",
        "run-ppo",
        "--tiny",
        "--output-dir",
        str(output_dir),
        "--steps",
        str(steps),
        "--batch-size",
        "4",
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
    ]
    if resume:
        command.append("--resume")
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"run-ppo failed\n{proc.stdout}\n{proc.stderr}")


def _read_run(output_dir: Path) -> dict:
    return json.loads((output_dir / "run.json").read_text())


def _state(checkpoint: Path, name: str) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return payload["models"][name]


def _trainer(
    tmp: Path,
    *,
    seed: int,
    estimator: str,
    reference: torch.nn.Module | None = None,
    checkpoint: Path | None = None,
) -> tuple[PPOTrainer, dict[str, torch.Tensor]]:
    config = tiny_ppo_config(
        output_dir=str(tmp),
        seed=seed,
        batch_size=4,
        checkpoint_path=None if checkpoint is None else str(checkpoint),
    )
    config.alignment.kl_estimator = estimator
    policy = tiny_causal_lm(seed)
    initial = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    value = tiny_value_model(seed)
    if reference is None:
        reference = tiny_causal_lm(seed + 1)
    optimizer = AdamW(
        list(policy.parameters()) + list(value.parameters()),
        lr=config.optimization.learning_rate,
    )
    trainer = PPOTrainer(config, policy, value, reference, optimizer)
    return trainer, initial


class TestTinyPrompts:
    def test_prompts_have_different_lengths(self) -> None:
        _query, lengths = collate_prompts(
            TinyCharTokenizer(),
            TINY_PPO_PROMPTS,
            max_length=32,
            pad_token_id=0,
        )
        assert len(set(int(value) for value in lengths)) > 1
        assert int(lengths.min()) >= 1


class TestRunPpoTraining:
    def test_seeds_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        real = seed_everything

        def spy(seed: int) -> None:
            calls.append(seed)
            real(seed)

        monkeypatch.setattr("rlpost.repro.seed_everything", spy)
        trainer, _initial = _trainer(tmp_path, seed=SEED, estimator="k2")
        run_ppo_training(trainer, N, TINY_PPO_PROMPTS, TinyCharTokenizer())
        assert calls == [SEED]

    @pytest.mark.parametrize("estimator", ["k2", "k3"])
    def test_step_zero_ratio_is_one_and_kl_is_zero(self, tmp_path: Path, estimator: str) -> None:
        """The copied reference matches the policy, so the first ratio is 1.

        Old log-probs are the stored response log-probs. New log-probs are
        the train-step gather. On the response mask they are equal, the
        ratio is 1, the KL is 0, and the clip fraction is 0. The caller
        passes a different initialization. ``run_ppo_training`` replaces
        it before the first step.
        """
        other = tiny_causal_lm(SEED + 3)
        trainer, initial = _trainer(tmp_path, seed=SEED, estimator=estimator, reference=other)
        seen: dict[str, torch.Tensor] = {}
        rewards: dict[str, torch.Tensor] = {}
        real_loss = trainer.compute_policy_loss
        real_step = trainer.train_step

        def spy_loss(
            log_probs_new: torch.Tensor,
            log_probs_old: torch.Tensor,
            advantages: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            loss, ratio = real_loss(log_probs_new, log_probs_old, advantages, mask)
            seen["new"] = log_probs_new.detach()
            seen["old"] = log_probs_old.detach()
            seen["ratio"] = ratio.detach()
            assert mask is not None
            seen["mask"] = mask.detach()
            return loss, ratio

        def spy_step(batch: dict, old_log_probs: torch.Tensor) -> object:
            rewards["ids"] = batch["input_ids"].detach()
            rewards["rewards"] = batch["rewards"].detach()
            rewards["width"] = torch.tensor(batch["prompt_width"])
            rewards["lengths"] = batch["response_lengths"].detach()
            return real_step(batch, old_log_probs)

        trainer.compute_policy_loss = spy_loss
        trainer.train_step = spy_step
        result = run_ppo_training(trainer, 1, TINY_PPO_PROMPTS, TinyCharTokenizer())

        mask = seen["mask"].bool()
        assert int(mask.sum()) > 0
        assert torch.equal(seen["new"][mask], seen["old"][mask])
        assert torch.equal(seen["ratio"][mask], torch.ones(int(mask.sum())))
        assert result.metrics[0]["kl_divergence"] == 0.0
        assert result.metrics[0]["clip_fraction"] == 0.0
        assert trainer.reference_model is not other
        assert all(
            not parameter.requires_grad for parameter in trainer.reference_model.parameters()
        )
        for key, value in trainer.reference_model.state_dict().items():
            assert torch.equal(value, initial[key]), key

        width = int(rewards["width"])
        ids = rewards["ids"]
        placed = rewards["rewards"]
        lengths = rewards["lengths"]
        for row in range(ids.shape[0]):
            end = width + int(lengths[row])
            expected = float(ids[row, width:end].sum())
            assert placed[row, end - 1].item() == expected
            assert int(torch.count_nonzero(placed[row])) == 1

    def test_resume_does_not_recopy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        real = frozen_reference_copy

        def spy(policy: torch.nn.Module) -> torch.nn.Module:
            calls.append(1)
            return real(policy)

        monkeypatch.setattr("rlpost.ppo_engine.frozen_reference_copy", spy)
        checkpoint = tmp_path / "ckpt.pt"
        trainer, initial = _trainer(tmp_path, seed=SEED, estimator="k2", checkpoint=checkpoint)
        run_ppo_training(trainer, 1, TINY_PPO_PROMPTS, TinyCharTokenizer())
        assert calls == [1]

        def boom(policy: torch.nn.Module) -> torch.nn.Module:
            del policy
            raise AssertionError("resumed run copied the reference")

        monkeypatch.setattr("rlpost.ppo_engine.frozen_reference_copy", boom)
        resumed, _ignored = _trainer(
            tmp_path,
            seed=SEED,
            estimator="k2",
            reference=tiny_causal_lm(SEED + 5),
            checkpoint=checkpoint,
        )
        run_ppo_training(
            resumed,
            1,
            TINY_PPO_PROMPTS,
            TinyCharTokenizer(),
            resume_from=checkpoint,
        )
        for key, value in resumed.reference_model.state_dict().items():
            assert torch.equal(value, initial[key]), key


class TestRunPpoCommand:
    def test_without_tiny_does_not_train(self, tmp_path: Path) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "rlpost.cli", "run-ppo", "--output-dir", str(tmp_path)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode != 0
        assert "tiny" in proc.stderr.lower()
        assert not (tmp_path / "run.json").exists()

    def test_resume_matches_the_uninterrupted_run(self, tmp_path: Path) -> None:
        full = tmp_path / "full"
        half = tmp_path / "half"
        other = tmp_path / "other"
        full.mkdir()
        half.mkdir()
        other.mkdir()
        full_ckpt = full / "ckpt.pt"
        half_ckpt = half / "ckpt.pt"
        other_ckpt = other / "ckpt.pt"

        _run(full, steps=2 * N, seed=SEED, checkpoint=full_ckpt, resume=False)
        _run(half, steps=N, seed=SEED, checkpoint=half_ckpt, resume=False)
        half_run = _read_run(half)
        _run(half, steps=N, seed=SEED, checkpoint=half_ckpt, resume=True)
        _run(other, steps=2 * N, seed=SEED + 1, checkpoint=other_ckpt, resume=False)

        full_run = _read_run(full)
        resume_run = _read_run(half)
        other_run = _read_run(other)

        assert full_run["seed"] == SEED
        assert full_run["step"] == 2 * N
        assert half_run["step"] == N
        assert resume_run["step"] == 2 * N
        assert half_run["losses"] == full_run["losses"][:N]
        assert half_run["metrics"] == full_run["metrics"][:N]
        assert resume_run["losses"] == full_run["losses"][N:]
        assert resume_run["metrics"] == full_run["metrics"][N:]
        assert other_run["losses"] != full_run["losses"]
        assert full_run["metrics"][0]["kl_divergence"] == 0.0
        assert full_run["metrics"][0]["clip_fraction"] == 0.0

        for name in ("policy", "value", "reference"):
            full_state = _state(full_ckpt, name)
            resume_state = _state(half_ckpt, name)
            assert full_state.keys() == resume_state.keys()
            for key in full_state:
                assert torch.equal(full_state[key], resume_state[key]), f"{name} {key}"

        initial_policy = tiny_causal_lm(SEED).state_dict()
        full_policy = _state(full_ckpt, "policy")
        full_reference = _state(full_ckpt, "reference")
        assert any(not torch.equal(initial_policy[key], full_policy[key]) for key in initial_policy)
        assert all(torch.equal(initial_policy[key], full_reference[key]) for key in initial_policy)
