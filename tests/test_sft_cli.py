"""``train-sft --tiny`` trains on CPU and a fresh process can resume it.

The command builds ``tiny_causal_lm``. It does not download a model.
N is 2. One process takes 4 steps. Another takes 2 and writes a
checkpoint. A third process loads that file and takes 2 more. Losses
and metrics use ``==``. Weights use ``torch.equal``.
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

from rlpost.repro import seed_everything
from rlpost.sft_engine import TINY_SFT_ROWS, run_sft_training, tiny_sft_config
from rlpost.tiny_lm import TinyCharTokenizer, tiny_causal_lm

ROOT = Path(__file__).resolve().parents[1]
N = 2
SEED = 7


def _run(output_dir: Path, *, steps: int, seed: int, checkpoint: Path, resume: bool) -> None:
    """One fresh ``python -m rlpost.cli train-sft`` process."""
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "rlpost.cli",
        "train-sft",
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
        raise AssertionError(f"train-sft failed\n{proc.stdout}\n{proc.stderr}")


def _read_run(output_dir: Path) -> dict:
    return json.loads((output_dir / "run.json").read_text())


def _state(checkpoint: Path, name: str) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return payload["models"][name]


class TestRunSftTraining:
    def test_seeds_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        real = seed_everything

        def spy(seed: int) -> None:
            calls.append(seed)
            real(seed)

        monkeypatch.setattr("rlpost.repro.seed_everything", spy)
        config = tiny_sft_config(
            output_dir=str(tmp_path),
            seed=SEED,
            batch_size=4,
            checkpoint_path=None,
        )
        model = tiny_causal_lm(SEED)
        optimizer = AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        run_sft_training(model, optimizer, config, N, TINY_SFT_ROWS, TinyCharTokenizer())
        assert calls == [SEED]


class TestTrainSftCommand:
    def test_without_tiny_does_not_train(self, tmp_path: Path) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "rlpost.cli",
                "train-sft",
                "--output-dir",
                str(tmp_path),
            ],
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
        assert resume_run["data_loss_before"] == half_run["data_loss_after"]
        assert other_run["losses"] != full_run["losses"]
        assert set(full_run) == {
            "checkpoint_path",
            "config_hash",
            "data_loss_after",
            "data_loss_before",
            "losses",
            "metrics",
            "seed",
            "step",
        }

        full_policy = _state(full_ckpt, "policy")
        resume_policy = _state(half_ckpt, "policy")
        assert full_policy.keys() == resume_policy.keys()
        for key in full_policy:
            assert torch.equal(full_policy[key], resume_policy[key]), key

        initial_policy = tiny_causal_lm(SEED).state_dict()
        assert any(not torch.equal(initial_policy[key], full_policy[key]) for key in initial_policy)
