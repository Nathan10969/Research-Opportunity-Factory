from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LosoFold:
    fold_id: str
    train_samples: tuple[str, ...]
    test_samples: tuple[str, ...]


def build_loso_folds(sample_ids: Iterable[str]) -> list[LosoFold]:
    unique = sorted(set(sample_ids))
    if len(unique) < 2:
        raise ValueError("LOSO requires at least two samples")
    folds = []
    for test_sample in unique:
        train = tuple(sample for sample in unique if sample != test_sample)
        folds.append(
            LosoFold(
                fold_id=f"test_{test_sample}",
                train_samples=train,
                test_samples=(test_sample,),
            )
        )
    return folds

