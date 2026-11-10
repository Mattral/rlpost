"""Same seed, same CPU losses. Resume in a fresh process matches the full run.

PPO and DPO each train tiny_lm for N steps. The batch is drawn from
Python, NumPy, and torch inside ``tests.repro_harness``. Losses are
compared with ``==``. Weights are compared with ``torch.equal``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tests.repro_harness import run_dpo, run_ppo
from tests.tiny_lm import VOCAB, tiny_causal_lm

from rlpost.config import TrainingConfig
from rlpost.dpo_engine import DPOTrainer
from rlpost.ppo_engine import PPOTrainer
from rlpost.repro import seed_everything

ROOT = Path(__file__).resolve().parents[1]
N = 2
SEED = 7


class _TokenValue(nn.Module):
    """Per-position value, shape [B, T]."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 32)
        self.proj = nn.Linear(32, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask, position_ids
        return self.proj(self.embed(input_ids)).squeeze(-1)


def _config(tmp: Path, seed: int, checkpoint: Path | None) -> TrainingConfig:
    config = TrainingConfig.toy_mode()
    config.seed = seed
    config.device = "cpu"
    config.log_to_wandb = False
    config.optimization.learning_rate = 1e-2
    config.output_dir = str(tmp)
    config.checkpoint_path = None if checkpoint is None else str(checkpoint)
    return config


def _build_ppo(config: TrainingConfig) -> PPOTrainer:
    policy = tiny_causal_lm(config.seed)
    policy.train()
    reference = tiny_causal_lm(config.seed + 1)
    reference.eval()
    reference.requires_grad_(False)
    torch.manual_seed(config.seed + 2)
    value = _TokenValue()
    optimizer = torch.optim.AdamW(
        list(policy.parameters()) + list(value.parameters()),
        lr=config.optimization.learning_rate,
    )
    return PPOTrainer(config, policy, value, reference, optimizer)


def _build_dpo(config: TrainingConfig) -> DPOTrainer:
    policy = tiny_causal_lm(config.seed)
    policy.train()
    reference = tiny_causal_lm(config.seed + 1)
    return DPOTrainer(
        config,
        policy_model=policy,
        reference_model=reference,
        tokenizer=None,
    )


def _weights(trainer: PPOTrainer | DPOTrainer, kind: str) -> dict[str, dict[str, torch.Tensor]]:
    modules: dict[str, nn.Module] = {
        "policy": trainer.policy_model,
        "reference": trainer.reference_model,
    }
    if kind == "ppo":
        modules["value"] = trainer.value_model
    return {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()
    }


def _assert_same_weights(
    left: dict[str, dict[str, torch.Tensor]],
    right: dict[str, dict[str, torch.Tensor]],
) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].keys() == right[name].keys()
        for key in left[name]:
            assert torch.equal(left[name][key], right[name][key]), f"{name} {key}"


def _any_differ(
    left: dict[str, dict[str, torch.Tensor]],
    right: dict[str, dict[str, torch.Tensor]],
) -> bool:
    for name in left:
        for key in left[name]:
            if not torch.equal(left[name][key], right[name][key]):
                return True
    return False


def resume_entry(argv: list[str]) -> None:
    """Load a checkpoint in this process and take ``n`` more steps.

    Invoked with ``python -c`` so the process does not share objects
    with the parent. Arguments: kind, checkpoint, n_steps, output file.
    """
    kind, ckpt, n_steps, out = argv
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    config = TrainingConfig(**payload["config"])
    if kind == "ppo":
        trainer = _build_ppo(config)
        result = run_ppo(trainer, int(n_steps), resume_from=ckpt)
    elif kind == "dpo":
        trainer = _build_dpo(config)
        result = run_dpo(trainer, int(n_steps), resume_from=ckpt)
    else:
        raise SystemExit(f"unknown kind {kind}")
    torch.save(
        {
            "losses": result.losses,
            "metrics": result.metrics,
            "step": result.step,
            "weights": _weights(trainer, kind),
        },
        out,
    )


def _resume_in_fresh_process(kind: str, ckpt: Path, n_steps: int, out: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tests.test_determinism import resume_entry; resume_entry(sys.argv[1:])",
            kind,
            str(ckpt),
            str(n_steps),
            str(out),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"resume process failed\n{proc.stdout}\n{proc.stderr}")
    return torch.load(out, map_location="cpu", weights_only=True)


def _assert_run_json(output_path: str, seed: int) -> None:
    payload = json.loads(Path(output_path).read_text())
    assert payload["seed"] == seed


class TestSameSeed:
    def test_two_ppo_runs_match_at_every_step(self, tmp_path: Path) -> None:
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        first_trainer = _build_ppo(_config(first_dir, SEED, None))
        before = _weights(first_trainer, "ppo")
        first = run_ppo(first_trainer, N)
        second = run_ppo(_build_ppo(_config(second_dir, SEED, None)), N)
        other = run_ppo(_build_ppo(_config(tmp_path / "other", SEED + 1, None)), N)

        assert first.seed == SEED
        assert first.losses == second.losses
        assert first.metrics == second.metrics
        assert first.losses != other.losses
        assert len(first.losses) == N
        _assert_run_json(first.output_path, SEED)
        assert _any_differ(before, _weights(first_trainer, "ppo"))

    def test_two_dpo_runs_match_at_every_step(self, tmp_path: Path) -> None:
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        (tmp_path / "other").mkdir()
        first = run_dpo(_build_dpo(_config(first_dir, SEED, None)), N)
        second = run_dpo(_build_dpo(_config(second_dir, SEED, None)), N)
        other = run_dpo(_build_dpo(_config(tmp_path / "other", SEED + 1, None)), N)

        assert first.seed == SEED
        assert first.losses == second.losses
        assert first.metrics == second.metrics
        assert first.losses != other.losses
        assert len(first.losses) == N
        _assert_run_json(first.output_path, SEED)

    def test_a_ppo_run_seeds_once(self, tmp_path: Path, monkeypatch: object) -> None:
        calls: list[int] = []
        real = seed_everything

        def spy(seed: int) -> None:
            calls.append(seed)
            real(seed)

        monkeypatch.setattr("rlpost.repro.seed_everything", spy)
        config = _config(tmp_path, SEED, None)
        run_ppo(_build_ppo(config), N)
        assert calls == [SEED]


class TestResume:
    def test_ppo_resume_matches_the_uninterrupted_run(self, tmp_path: Path) -> None:
        full_dir = tmp_path / "full"
        half_dir = tmp_path / "half"
        full_dir.mkdir()
        half_dir.mkdir()
        checkpoint = half_dir / "checkpoint.pt"
        full_trainer = _build_ppo(_config(full_dir, SEED, None))
        full = run_ppo(full_trainer, 2 * N)
        half = run_ppo(_build_ppo(_config(half_dir, SEED, checkpoint)), N)
        assert half.losses == full.losses[:N]
        assert half.metrics == full.metrics[:N]

        resumed = _resume_in_fresh_process("ppo", checkpoint, N, tmp_path / "resumed.pt")
        assert resumed["step"] == full.step == 2 * N
        assert resumed["losses"] == full.losses[N:]
        assert resumed["metrics"] == full.metrics[N:]
        assert resumed["metrics"][-1] == full.metrics[-1]
        _assert_same_weights(resumed["weights"], _weights(full_trainer, "ppo"))

    def test_dpo_resume_matches_the_uninterrupted_run(self, tmp_path: Path) -> None:
        full_dir = tmp_path / "full"
        half_dir = tmp_path / "half"
        full_dir.mkdir()
        half_dir.mkdir()
        checkpoint = half_dir / "checkpoint.pt"
        full_trainer = _build_dpo(_config(full_dir, SEED, None))
        full = run_dpo(full_trainer, 2 * N)
        half = run_dpo(_build_dpo(_config(half_dir, SEED, checkpoint)), N)
        assert half.losses == full.losses[:N]

        resumed = _resume_in_fresh_process("dpo", checkpoint, N, tmp_path / "resumed.pt")
        assert resumed["step"] == full.step == 2 * N
        assert resumed["losses"] == full.losses[N:]
        assert resumed["metrics"] == full.metrics[N:]
        assert resumed["metrics"][-1] == full.metrics[-1]
        _assert_same_weights(resumed["weights"], _weights(full_trainer, "dpo"))

    def test_ppo_resume_seeds_once(self, tmp_path: Path, monkeypatch: object) -> None:
        checkpoint = tmp_path / "checkpoint.pt"
        config = _config(tmp_path, SEED, checkpoint)
        run_ppo(_build_ppo(config), N)
        calls: list[int] = []
        real = seed_everything

        def spy(seed: int) -> None:
            calls.append(seed)
            real(seed)

        monkeypatch.setattr("rlpost.repro.seed_everything", spy)
        run_ppo(_build_ppo(config), N, resume_from=checkpoint)
        assert calls == [SEED]
