"""``run-dpo --tiny`` trains on CPU and a fresh process can resume it.

The command builds ``tiny_causal_lm``. It does not download a model.
N is 2. One process takes 4 steps. Another takes 2 and writes a
checkpoint. A third process loads that file and takes 2 more. Losses
and metrics use ``==``. Weights use ``torch.equal``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rlpost.dpo_engine import (
    TINY_DPO_PAIRS,
    DPOTrainer,
    collate_preference_pairs,
    run_dpo_training,
    tiny_dpo_config,
)
from rlpost.repro import seed_everything
from rlpost.tiny_lm import TinyCharTokenizer, tiny_causal_lm

ROOT = Path(__file__).resolve().parents[1]
N = 2
SEED = 7


def _run(output_dir: Path, *, steps: int, seed: int, checkpoint: Path, resume: bool) -> None:
    """One fresh ``python -m rlpost.cli run-dpo`` process."""
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "rlpost.cli",
        "run-dpo",
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
        raise AssertionError(f"run-dpo failed\n{proc.stdout}\n{proc.stderr}")


def _read_run(output_dir: Path) -> dict:
    return json.loads((output_dir / "run.json").read_text())


def _state(checkpoint: Path, name: str) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return payload["models"][name]


class TestTinyPairs:
    def test_rows_have_different_prompt_lengths(self) -> None:
        prompts = [row["prompt"] for row in TINY_DPO_PAIRS]
        chosen = [row["chosen"] for row in TINY_DPO_PAIRS]
        rejected = [row["rejected"] for row in TINY_DPO_PAIRS]
        collated = collate_preference_pairs(
            TinyCharTokenizer(),
            prompts,
            chosen,
            rejected,
            max_length=32,
            pad_token_id=0,
        )
        chosen_lengths = {int(value) for value in collated["chosen_prompt_lengths"]}
        rejected_lengths = {int(value) for value in collated["rejected_prompt_lengths"]}
        assert len(chosen_lengths) > 1
        assert len(rejected_lengths) > 1
        assert (
            int(collated["rejected_attention_mask"].sum())
            < collated["rejected_attention_mask"].numel()
        )


class TestRunDpoTraining:
    def test_seeds_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        real = seed_everything

        def spy(seed: int) -> None:
            calls.append(seed)
            real(seed)

        monkeypatch.setattr("rlpost.repro.seed_everything", spy)
        config = tiny_dpo_config(
            output_dir=str(tmp_path),
            seed=SEED,
            batch_size=4,
            checkpoint_path=None,
        )
        trainer = DPOTrainer(
            config,
            policy_model=tiny_causal_lm(SEED),
            reference_model=tiny_causal_lm(SEED + 1),
            tokenizer=TinyCharTokenizer(),
        )
        run_dpo_training(trainer, N, TINY_DPO_PAIRS)
        assert calls == [SEED]

    def test_step_zero_loss_is_log_two(self, tmp_path: Path) -> None:
        """A copied reference makes the first loss ln 2.

        Absolute tolerance is 1e-6. The caller passes a different
        initialization. ``run_dpo_training`` replaces it with a frozen
        copy of the policy before the first step. Both reward tensors
        are 0, and the margin is 0. After the step the reference still
        matches the initial policy.
        """
        policy = tiny_causal_lm(SEED)
        initial = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        other = tiny_causal_lm(SEED + 1)
        config = tiny_dpo_config(
            output_dir=str(tmp_path),
            seed=SEED,
            batch_size=4,
            checkpoint_path=None,
        )
        trainer = DPOTrainer(
            config,
            policy_model=policy,
            reference_model=other,
            tokenizer=TinyCharTokenizer(),
        )
        seen: dict[str, torch.Tensor] = {}
        real_step = trainer.train_step

        def spy(batch: dict) -> object:
            collated = collate_preference_pairs(
                trainer.tokenizer,
                batch["prompt"],
                batch["chosen"],
                batch["rejected"],
                max_length=config.dataset.max_seq_length,
                pad_token_id=0,
            )
            trainer.policy_model.train()
            trainer.reference_model.eval()
            with torch.no_grad():
                policy_chosen = trainer.compute_log_probs(
                    trainer.policy_model,
                    collated["chosen_ids"],
                    collated["chosen_attention_mask"],
                    collated["chosen_completion_mask"],
                )
                policy_rejected = trainer.compute_log_probs(
                    trainer.policy_model,
                    collated["rejected_ids"],
                    collated["rejected_attention_mask"],
                    collated["rejected_completion_mask"],
                )
                ref_chosen = trainer.compute_log_probs(
                    trainer.reference_model,
                    collated["chosen_ids"],
                    collated["chosen_attention_mask"],
                    collated["chosen_completion_mask"],
                )
                ref_rejected = trainer.compute_log_probs(
                    trainer.reference_model,
                    collated["rejected_ids"],
                    collated["rejected_attention_mask"],
                    collated["rejected_completion_mask"],
                )
            seen["chosen"] = trainer.beta * (policy_chosen - ref_chosen)
            seen["rejected"] = trainer.beta * (policy_rejected - ref_rejected)
            return real_step(batch)

        trainer.train_step = spy
        result = run_dpo_training(trainer, 1, TINY_DPO_PAIRS)
        assert result.losses[0] == pytest.approx(math.log(2), abs=1e-6)
        assert result.metrics[0]["margin_mean"] == pytest.approx(0.0, abs=1e-6)
        assert seen["chosen"].shape[0] == 4
        assert torch.allclose(seen["chosen"], torch.zeros_like(seen["chosen"]), atol=1e-6, rtol=0)
        assert torch.allclose(
            seen["rejected"], torch.zeros_like(seen["rejected"]), atol=1e-6, rtol=0
        )
        assert trainer.reference_model is not other
        assert all(
            not parameter.requires_grad for parameter in trainer.reference_model.parameters()
        )
        for key, value in trainer.reference_model.state_dict().items():
            assert torch.equal(value, initial[key]), key


class TestRunDpoCommand:
    def test_without_tiny_does_not_train(self, tmp_path: Path) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "rlpost.cli", "run-dpo", "--output-dir", str(tmp_path)],
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

        full_policy = _state(full_ckpt, "policy")
        resume_policy = _state(half_ckpt, "policy")
        assert full_policy.keys() == resume_policy.keys()
        for key in full_policy:
            assert torch.equal(full_policy[key], resume_policy[key]), key
        full_reference = _state(full_ckpt, "reference")
        resume_reference = _state(half_ckpt, "reference")
        for key in full_reference:
            assert torch.equal(full_reference[key], resume_reference[key]), key

        initial_policy = tiny_causal_lm(SEED).state_dict()
        assert any(not torch.equal(initial_policy[key], full_policy[key]) for key in initial_policy)
        assert all(torch.equal(initial_policy[key], full_reference[key]) for key in initial_policy)
        assert full_run["losses"][0] == pytest.approx(math.log(2), abs=1e-6)
        assert full_run["metrics"][0]["margin_mean"] == pytest.approx(0.0, abs=1e-6)
