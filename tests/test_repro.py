"""Seed and checkpoint. CPU. The training-loop tests are test_determinism.py."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import AdamW

from rlpost.config import TrainingConfig
from rlpost.repro import (
    config_hash,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


def _config(tmp: Path, seed: int = 5) -> TrainingConfig:
    config = TrainingConfig.toy_mode()
    config.seed = seed
    config.device = "cpu"
    config.log_to_wandb = False
    config.output_dir = str(tmp)
    config.checkpoint_path = str(tmp / "checkpoint.pt")
    return config


def _pair() -> tuple[nn.Linear, AdamW, nn.Linear, AdamW]:
    torch.manual_seed(0)
    first = nn.Linear(2, 2)
    second = nn.Linear(2, 2)
    second.load_state_dict(first.state_dict())
    opt_first = AdamW(first.parameters(), lr=1e-2)
    opt_second = AdamW(second.parameters(), lr=1e-2)
    return first, opt_first, second, opt_second


def _step(model: nn.Linear, optimizer: AdamW, x: torch.Tensor) -> None:
    optimizer.zero_grad()
    model(x).sum().backward()
    optimizer.step()


def _draw() -> tuple[float, float, list[float]]:
    return (
        random.random(),
        float(np.random.random()),
        torch.rand(3).tolist(),
    )


class TestSeed:
    def test_same_seed_repeats_python_numpy_and_torch(self) -> None:
        seed_everything(5)
        first = _draw()
        seed_everything(5)
        second = _draw()
        assert first == second

    def test_a_different_seed_draws_different_numbers(self) -> None:
        seed_everything(5)
        first = _draw()
        seed_everything(6)
        second = _draw()
        assert first != second

    def test_a_bool_is_not_a_seed(self) -> None:
        with pytest.raises(TypeError, match="seed must be an int"):
            seed_everything(True)

    def test_a_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="seed must be >= 0"):
            seed_everything(-1)


class TestConfigHash:
    def test_checkpoint_path_is_not_part_of_the_hash(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        original = config_hash(config)
        config.checkpoint_path = str(tmp_path / "elsewhere.pt")
        assert config_hash(config) == original

    def test_a_blank_checkpoint_path_raises(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_path"):
            TrainingConfig(checkpoint_path="  ")

    def test_a_training_field_changes_the_hash(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        original = config_hash(config)
        config.optimization.learning_rate = 3e-4
        assert config_hash(config) != original


class TestCheckpoint:
    def test_round_trip_restores_weights_and_optimizer(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        first, opt_first, second, opt_second = _pair()
        x = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
        _step(first, opt_first, x)
        _step(second, opt_second, x)

        path = Path(config.checkpoint_path)
        save_checkpoint(
            path,
            models={"linear": first},
            optimizer=opt_first,
            step=4,
            config=config,
            trainer_state={"kl_coefficient": 0.2},
        )
        assert path.is_file()
        assert list(path.parent.glob("*.tmp")) == []

        with torch.no_grad():
            second.weight.add_(1)
        loaded = load_checkpoint(
            path,
            models={"linear": second},
            optimizer=opt_second,
            config=config,
        )

        assert loaded.step == 4
        assert loaded.trainer_state == {"kl_coefficient": 0.2}
        assert torch.equal(first.weight, second.weight)
        assert torch.equal(first.bias, second.bias)
        saved_avg = opt_first.state[first.weight]["exp_avg"]
        loaded_avg = opt_second.state[second.weight]["exp_avg"]
        assert torch.equal(saved_avg, loaded_avg)

    def test_round_trip_restores_the_rng(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        model = nn.Linear(2, 2)
        optimizer = AdamW(model.parameters(), lr=1e-2)
        path = tmp_path / "checkpoint.pt"

        seed_everything(config.seed)
        save_checkpoint(
            path,
            models={"linear": model},
            optimizer=optimizer,
            step=1,
            config=config,
        )
        expected = _draw()
        _draw()
        load_checkpoint(path, models={"linear": model}, optimizer=optimizer, config=config)
        assert _draw() == expected

    def test_payload_records_the_seed(self, tmp_path: Path) -> None:
        config = _config(tmp_path, seed=11)
        model = nn.Linear(2, 2)
        optimizer = AdamW(model.parameters(), lr=1e-2)
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            path,
            models={"linear": model},
            optimizer=optimizer,
            step=0,
            config=config,
        )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        assert payload["seed"] == 11
        assert payload["config"]["seed"] == 11
        if torch.cuda.is_available():
            assert payload["rng"]["cuda"] is not None
        else:
            assert payload["rng"]["cuda"] is None

    def test_a_different_config_hash_raises_and_does_not_write(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        model = nn.Linear(2, 2)
        other = nn.Linear(2, 2)
        other.load_state_dict(model.state_dict())
        optimizer = AdamW(model.parameters(), lr=1e-2)
        other_opt = AdamW(other.parameters(), lr=1e-2)
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            path,
            models={"linear": model},
            optimizer=optimizer,
            step=1,
            config=config,
        )
        before = other.weight.detach().clone()
        config.seed = config.seed + 1
        with pytest.raises(ValueError, match="config hash"):
            load_checkpoint(path, models={"linear": other}, optimizer=other_opt, config=config)
        assert torch.equal(other.weight, before)

    def test_moving_the_checkpoint_path_still_loads(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        model = nn.Linear(2, 2)
        optimizer = AdamW(model.parameters(), lr=1e-2)
        path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            path,
            models={"linear": model},
            optimizer=optimizer,
            step=2,
            config=config,
        )
        config.checkpoint_path = str(tmp_path / "drive" / "checkpoint.pt")
        loaded = load_checkpoint(
            path,
            models={"linear": model},
            optimizer=optimizer,
            config=config,
        )
        assert loaded.step == 2

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        model = nn.Linear(2, 2)
        optimizer = AdamW(model.parameters(), lr=1e-2)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                tmp_path / "missing.pt",
                models={"linear": model},
                optimizer=optimizer,
                config=config,
            )
