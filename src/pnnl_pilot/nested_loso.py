from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def elastic_l2_equivalent_ridge_alpha(alpha: float, n_samples: int) -> float:
    if alpha < 0 or n_samples <= 0:
        raise ValueError("alpha must be non-negative and n_samples must be positive")
    # ElasticNet(l1_ratio=0) minimizes ||e||^2/(2n) + alpha*||w||^2/2.
    # sklearn Ridge minimizes ||e||^2 + ridge_alpha*||w||^2.
    return float(alpha) * int(n_samples)


def coordinate_features(
    rows: Sequence[Mapping[str, str]],
    dimensions_by_sample: Mapping[str, tuple[int, int]],
) -> np.ndarray:
    coordinates = np.empty((len(rows), 2), dtype=np.float32)
    for index, row in enumerate(rows):
        sample_id = row["sample_id"]
        if sample_id not in dimensions_by_sample:
            raise KeyError(f"missing source dimensions for {sample_id}")
        width, height = dimensions_by_sample[sample_id]
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid source dimensions for {sample_id}: {(width, height)}")
        coordinates[index] = (
            float(row["center_u_px"]) / width,
            float(row["center_v_px"]) / height,
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("normalized coordinates contain non-finite values")
    return coordinates


def sample_balanced_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sample_ids = np.asarray(sample_ids)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or sample_ids.shape != y_true.shape:
        raise ValueError("targets, predictions, and sample IDs must be aligned vectors")
    if y_true.size == 0 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("targets and predictions must be non-empty and finite")
    per_sample_mae = []
    per_sample_rmse = []
    for sample_id in sorted(set(sample_ids.tolist())):
        mask = sample_ids == sample_id
        error = y_pred[mask] - y_true[mask]
        per_sample_mae.append(float(np.mean(np.abs(error))))
        per_sample_rmse.append(float(np.sqrt(np.mean(np.square(error)))))
    return {
        "sample_balanced_mae": float(np.mean(per_sample_mae)),
        "sample_balanced_rmse": float(np.mean(per_sample_rmse)),
    }


def select_config_inner_logo(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    configs: Sequence[Mapping[str, Any]],
    evaluate: Callable[[Mapping[str, Any], np.ndarray, np.ndarray], np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = np.asarray(x)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if x.shape[0] != y.shape[0] or y.shape != groups.shape:
        raise ValueError("features, targets, and groups must have aligned rows")
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        raise ValueError("inner LOGO requires at least two groups")
    if not configs:
        raise ValueError("at least one configuration is required")

    scores = []
    for trial_index, config in enumerate(configs):
        fold_mae = []
        for valid_group in unique_groups:
            valid_index = np.flatnonzero(groups == valid_group)
            train_index = np.flatnonzero(groups != valid_group)
            prediction = np.asarray(evaluate(config, train_index, valid_index))
            if prediction.shape != y[valid_index].shape or not np.isfinite(prediction).all():
                raise ValueError(
                    f"invalid inner prediction for group {valid_group}: {prediction.shape}"
                )
            fold_mae.append(float(np.mean(np.abs(prediction - y[valid_index]))))
        scores.append(
            {
                "trial_index": trial_index,
                "config": dict(config),
                "inner_sample_balanced_mae": float(np.mean(fold_mae)),
                "inner_fold_mae": fold_mae,
            }
        )
    winner = min(
        scores,
        key=lambda item: (item["inner_sample_balanced_mae"], item["trial_index"]),
    )
    return dict(winner["config"]), scores
