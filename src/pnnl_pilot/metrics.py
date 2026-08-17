from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def sample_balanced_mae(
    y_true: Iterable[float], y_pred: Iterable[float], sample_ids: Iterable[str]
) -> float:
    truth = list(y_true)
    prediction = list(y_pred)
    samples = list(sample_ids)
    if not truth or len(truth) != len(prediction) or len(truth) != len(samples):
        raise ValueError("truth, prediction, and sample_ids must have equal non-zero length")
    errors: dict[str, list[float]] = defaultdict(list)
    for expected, observed, sample_id in zip(truth, prediction, samples):
        errors[sample_id].append(abs(float(expected) - float(observed)))
    per_sample = [sum(values) / len(values) for values in errors.values()]
    return sum(per_sample) / len(per_sample)
