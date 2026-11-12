"""``dataset_size`` keeps a prefix of the loaded pairs.

The field is on ``DatasetConfig``. ``None`` keeps every row. This does
not download a dataset.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from rlpost.config import DatasetConfig
from rlpost.dataset import PreferencePairDataset


def _frame() -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": ["a", "b", "c"],
            "chosen": ["c1", "c2", "c3"],
            "rejected": ["r1", "r2", "r3"],
        }
    )


def test_dataset_size_keeps_the_leading_pairs(tmp_path: Path) -> None:
    limited = PreferencePairDataset(
        DatasetConfig(dataset_size=2, dataset_name="pairs"),
        cache_dir=str(tmp_path / "limited"),
    )
    pairs = limited._convert_to_preference_pairs(_frame())
    assert [pair.prompt for pair in pairs] == ["a", "b"]

    full = PreferencePairDataset(
        DatasetConfig(dataset_name="pairs"),
        cache_dir=str(tmp_path / "full"),
    )
    assert [pair.prompt for pair in full._convert_to_preference_pairs(_frame())] == [
        "a",
        "b",
        "c",
    ]
