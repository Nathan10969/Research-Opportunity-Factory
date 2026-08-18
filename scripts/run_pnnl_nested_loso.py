from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pnnl_pilot.nested_loso import (
    coordinate_features,
    elastic_l2_equivalent_ridge_alpha,
    sample_balanced_regression_metrics,
    select_config_inner_logo,
)


SCHEMA_VERSION = "pnnl-nested-loso-v1"
REPRESENTATIONS = (
    "coordinate",
    "image_cls",
    "image_mean_patch",
    "fusion_cls",
    "fusion_mean_patch",
)
ALL_MODELS = ("ridge", "elastic_net", "random_forest")
RF_OUTER_SEEDS = (17, 29, 43)
TUNING_SEED = 1729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS)
    )
    parser.add_argument("--rf-jobs", type=int, default=16)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def configs_for(model_name: str) -> list[dict]:
    if model_name == "ridge":
        return [{"alpha": float(value)} for value in np.logspace(-4, 4, 6)]
    if model_name == "elastic_net":
        alphas = np.logspace(-4, 1, 6)
        l1_ratios = (0.0, 0.25, 0.5, 0.75, 1.0, 0.5)
        return [
            {"alpha": float(alpha), "l1_ratio": float(l1_ratio)}
            for alpha, l1_ratio in zip(alphas, l1_ratios)
        ]
    if model_name == "random_forest":
        return [
            {"n_estimators": 300, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 600, "min_samples_leaf": 1, "max_features": 0.33},
            {"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.5},
            {"n_estimators": 600, "min_samples_leaf": 4, "max_features": "sqrt"},
            {"n_estimators": 300, "min_samples_leaf": 8, "max_features": 0.33},
            {"n_estimators": 600, "min_samples_leaf": 8, "max_features": 0.5},
        ]
    raise ValueError(f"unknown model: {model_name}")


def build_estimator(
    model_name: str,
    config: dict,
    *,
    seed: int,
    rf_jobs: int,
    n_samples: int,
):
    if model_name == "ridge":
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=config["alpha"], solver="lsqr", tol=1e-4),
        )
    if model_name == "elastic_net":
        if config["l1_ratio"] == 0.0:
            return make_pipeline(
                StandardScaler(),
                Ridge(
                    alpha=elastic_l2_equivalent_ridge_alpha(
                        config["alpha"], n_samples
                    ),
                    solver="lsqr",
                    tol=1e-4,
                ),
            )
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=config["alpha"],
                l1_ratio=config["l1_ratio"],
                max_iter=20000,
                tol=1e-3,
                selection="random",
                random_state=TUNING_SEED,
            ),
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            **config,
            random_state=seed,
            n_jobs=rf_jobs,
        )
    raise ValueError(f"unknown model: {model_name}")


def load_inputs(feature_dir: Path, source_manifest: Path):
    rows = list(
        csv.DictReader((feature_dir / "feature_rows.csv").open("r", encoding="utf-8"))
    )
    source_rows = list(csv.DictReader(source_manifest.open("r", encoding="utf-8")))
    dimensions = {
        row["sample_id"]: (int(row["width_px"]), int(row["height_px"]))
        for row in source_rows
    }
    coords = coordinate_features(rows, dimensions)
    cls = np.load(feature_dir / "features_cls.npy", mmap_mode="r")
    mean_patch = np.load(feature_dir / "features_mean_patch.npy", mmap_mode="r")
    if cls.shape != mean_patch.shape or cls.shape[0] != len(rows):
        raise ValueError(
            f"feature arrays and rows are misaligned: {cls.shape}, {mean_patch.shape}, {len(rows)}"
        )
    y = np.asarray([float(row["hardness_hv"]) for row in rows], dtype=np.float32)
    groups = np.asarray([row["sample_id"] for row in rows])
    if len(set(groups.tolist())) != 9:
        raise ValueError(f"expected exactly nine samples, found {sorted(set(groups))}")
    if not np.isfinite(cls).all() or not np.isfinite(mean_patch).all():
        raise ValueError("DINO features contain non-finite values")
    representations = {
        "coordinate": coords,
        "image_cls": cls,
        "image_mean_patch": mean_patch,
        "fusion_cls": np.concatenate((cls, coords), axis=1),
        "fusion_mean_patch": np.concatenate((mean_patch, coords), axis=1),
    }
    return rows, y, groups, representations


def run_job(
    *,
    model_name: str,
    representation: str,
    test_sample: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    rf_jobs: int,
    provenance_hash: str,
) -> dict:
    outer_train = np.flatnonzero(groups != test_sample)
    outer_test = np.flatnonzero(groups == test_sample)
    if set(groups[outer_train]) & set(groups[outer_test]):
        raise RuntimeError("outer sample overlap detected")
    x_train = np.asarray(x[outer_train], dtype=np.float32)
    x_test = np.asarray(x[outer_test], dtype=np.float32)
    y_train = y[outer_train]
    inner_groups = groups[outer_train]
    configs = configs_for(model_name)

    def evaluate(config, inner_train, inner_valid):
        estimator = build_estimator(
            model_name,
            config,
            seed=TUNING_SEED,
            rf_jobs=rf_jobs,
            n_samples=len(inner_train),
        )
        estimator.fit(x_train[inner_train], y_train[inner_train])
        return estimator.predict(x_train[inner_valid])

    started = time.time()
    selected, scores = select_config_inner_logo(
        x_train, y_train, inner_groups, configs, evaluate
    )
    seeds = RF_OUTER_SEEDS if model_name == "random_forest" else (0,)
    seed_predictions = []
    for seed in seeds:
        estimator = build_estimator(
            model_name,
            selected,
            seed=seed,
            rf_jobs=rf_jobs,
            n_samples=len(outer_train),
        )
        estimator.fit(x_train, y_train)
        seed_predictions.append(estimator.predict(x_test).astype(float).tolist())
    ensemble = np.mean(np.asarray(seed_predictions, dtype=np.float64), axis=0)
    metrics = sample_balanced_regression_metrics(
        y[outer_test], ensemble, groups[outer_test]
    )
    return {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "provenance_hash": provenance_hash,
        "model": model_name,
        "representation": representation,
        "test_sample": test_sample,
        "outer_train_samples": sorted(set(groups[outer_train].tolist())),
        "outer_test_samples": [test_sample],
        "outer_sample_overlap": 0,
        "trial_budget": len(configs),
        "selected_config": selected,
        "inner_scores": scores,
        "outer_seeds": list(seeds),
        "outer_test_indices": outer_test.tolist(),
        "outer_y_true": y[outer_test].astype(float).tolist(),
        "outer_seed_predictions": seed_predictions,
        "outer_ensemble_prediction": ensemble.tolist(),
        "outer_metrics": metrics,
        "runtime_seconds": time.time() - started,
    }


def aggregate(
    *,
    run_dir: Path,
    final_report_path: Path,
    rows: list[dict[str, str]],
    y: np.ndarray,
    groups: np.ndarray,
    provenance: dict,
) -> dict:
    expected = {
        (sample, representation, model)
        for sample in sorted(set(groups.tolist()))
        for representation in REPRESENTATIONS
        for model in ALL_MODELS
    }
    jobs = []
    for path in sorted((run_dir / "jobs").glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        key = (job.get("test_sample"), job.get("representation"), job.get("model"))
        if (
            job.get("status") == "passed"
            and job.get("provenance_hash") == provenance["provenance_hash"]
            and key in expected
        ):
            jobs.append(job)
    completed = {
        (job["test_sample"], job["representation"], job["model"]): job
        for job in jobs
    }
    if len(completed) != len(jobs):
        raise RuntimeError("duplicate completed job keys detected")

    prediction_rows = []
    fold_rows = []
    trial_rows = []
    seed_rows = []
    for key in sorted(completed):
        job = completed[key]
        indices = job["outer_test_indices"]
        ensemble = job["outer_ensemble_prediction"]
        for local_index, (feature_index, prediction) in enumerate(
            zip(indices, ensemble)
        ):
            prediction_rows.append(
                {
                    "feature_row": feature_index,
                    "crop_id": rows[feature_index]["crop_id"],
                    "sample_id": rows[feature_index]["sample_id"],
                    "model": job["model"],
                    "representation": job["representation"],
                    "y_true": job["outer_y_true"][local_index],
                    "y_pred": prediction,
                }
            )
        fold_rows.append(
            {
                "test_sample": job["test_sample"],
                "model": job["model"],
                "representation": job["representation"],
                "mae": job["outer_metrics"]["sample_balanced_mae"],
                "rmse": job["outer_metrics"]["sample_balanced_rmse"],
                "selected_config": json.dumps(job["selected_config"], sort_keys=True),
                "runtime_seconds": job["runtime_seconds"],
            }
        )
        for score in job["inner_scores"]:
            trial_rows.append(
                {
                    "test_sample": job["test_sample"],
                    "model": job["model"],
                    "representation": job["representation"],
                    "trial_index": score["trial_index"],
                    "config": json.dumps(score["config"], sort_keys=True),
                    "inner_sample_balanced_mae": score[
                        "inner_sample_balanced_mae"
                    ],
                    "inner_fold_mae": json.dumps(score["inner_fold_mae"]),
                }
            )
        for seed, predictions in zip(
            job["outer_seeds"], job["outer_seed_predictions"]
        ):
            metric = sample_balanced_regression_metrics(
                np.asarray(job["outer_y_true"]),
                np.asarray(predictions),
                np.asarray([job["test_sample"]] * len(predictions)),
            )
            seed_rows.append(
                {
                    "test_sample": job["test_sample"],
                    "model": job["model"],
                    "representation": job["representation"],
                    "seed": seed,
                    "mae": metric["sample_balanced_mae"],
                    "rmse": metric["sample_balanced_rmse"],
                }
            )

    fields_by_name = {
        "predictions.csv": [
            "feature_row", "crop_id", "sample_id", "model", "representation",
            "y_true", "y_pred",
        ],
        "per_fold_results.csv": [
            "test_sample", "model", "representation", "mae", "rmse",
            "selected_config", "runtime_seconds",
        ],
        "inner_cv_trials.csv": [
            "test_sample", "model", "representation", "trial_index", "config",
            "inner_sample_balanced_mae", "inner_fold_mae",
        ],
        "per_seed_results.csv": [
            "test_sample", "model", "representation", "seed", "mae", "rmse",
        ],
    }
    for filename, data in (
        ("predictions.csv", prediction_rows),
        ("per_fold_results.csv", fold_rows),
        ("inner_cv_trials.csv", trial_rows),
        ("per_seed_results.csv", seed_rows),
    ):
        write_csv(run_dir / filename, data, fields_by_name[filename])

    overall = []
    for model in ALL_MODELS:
        for representation in REPRESENTATIONS:
            subset = [
                row
                for row in fold_rows
                if row["model"] == model and row["representation"] == representation
            ]
            if len(subset) == 9:
                overall.append(
                    {
                        "model": model,
                        "representation": representation,
                        "sample_balanced_mae": float(np.mean([r["mae"] for r in subset])),
                        "sample_balanced_rmse": float(np.mean([r["rmse"] for r in subset])),
                    }
                )
    primary_pairs = {}
    for sample in sorted(set(groups.tolist())):
        coord = completed.get((sample, "coordinate", "ridge"))
        fusion = completed.get((sample, "fusion_mean_patch", "ridge"))
        if coord is not None and fusion is not None:
            primary_pairs[sample] = {
                "coordinate_mae": coord["outer_metrics"]["sample_balanced_mae"],
                "fusion_mean_patch_mae": fusion["outer_metrics"]["sample_balanced_mae"],
                "improved": fusion["outer_metrics"]["sample_balanced_mae"]
                < coord["outer_metrics"]["sample_balanced_mae"],
            }
    improved = sum(item["improved"] for item in primary_pairs.values())
    complete = set(completed) == expected
    summary = {
        "status": "passed" if complete else "partial",
        "schema_version": SCHEMA_VERSION,
        "completed_jobs": len(completed),
        "expected_jobs": len(expected),
        "missing_jobs": [list(key) for key in sorted(expected - set(completed))],
        "overall_results": overall,
        "primary_gate": {
            "comparison": "ridge:fusion_mean_patch_vs_coordinate",
            "per_sample": primary_pairs,
            "improved_samples": improved,
            "required_improved_samples": 6,
            "passed": len(primary_pairs) == 9 and improved >= 6,
        },
        "provenance": provenance,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "configuration_provenance.json", provenance)

    if complete:
        final_report = {
            **summary,
            "inner_group_cv": True,
            "equal_trial_budget": len({job["trial_budget"] for job in jobs}) == 1,
            "outer_test_used_for_selection": False,
            "outer_sample_overlap": max(job["outer_sample_overlap"] for job in jobs),
            "per_fold_results_present": len(fold_rows) == 135,
            "per_sample_results_present": len(primary_pairs) == 9,
            "per_seed_results_present": len(seed_rows) > len(fold_rows),
            "configuration_provenance_present": True,
            "artifact_paths": {
                "run_summary": str((run_dir / "summary.json").resolve()),
                "per_fold_results": str((run_dir / "per_fold_results.csv").resolve()),
                "per_seed_results": str((run_dir / "per_seed_results.csv").resolve()),
                "inner_cv_trials": str((run_dir / "inner_cv_trials.csv").resolve()),
                "predictions": str((run_dir / "predictions.csv").resolve()),
            },
        }
        write_json(final_report_path, final_report)
    return summary


def main() -> None:
    args = parse_args()
    warnings.simplefilter("error", ConvergenceWarning)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = args.run_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    rows, y, groups, representations = load_inputs(
        args.feature_dir, args.source_manifest
    )
    feature_report_path = args.feature_dir / "feature_report.json"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "feature_report_sha256": sha256(feature_report_path),
        "feature_rows_sha256": sha256(args.feature_dir / "feature_rows.csv"),
        "features_cls_sha256": sha256(args.feature_dir / "features_cls.npy"),
        "features_mean_patch_sha256": sha256(
            args.feature_dir / "features_mean_patch.npy"
        ),
        "source_manifest_sha256": sha256(args.source_manifest),
        "runner_sha256": sha256(Path(__file__)),
        "nested_loso_module_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/pnnl_pilot/nested_loso.py"
        ),
        "models": list(ALL_MODELS),
        "representations": list(REPRESENTATIONS),
        "trial_budget_per_model": 6,
        "inner_split": "leave-one-Sample-ID-out on outer training samples",
        "outer_split": "leave-one-Sample-ID-out",
        "selection_metric": "sample-balanced MAE",
        "primary_gate": "ridge:fusion_mean_patch_vs_coordinate; improve >=6/9",
        "rf_tuning_seed": TUNING_SEED,
        "rf_outer_seeds": list(RF_OUTER_SEEDS),
        "rf_jobs": args.rf_jobs,
    }
    provenance_hash = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode("utf-8")
    ).hexdigest()
    provenance["provenance_hash"] = provenance_hash

    samples = sorted(set(groups.tolist()))
    requested = list(dict.fromkeys(args.models))
    total_requested = len(samples) * len(REPRESENTATIONS) * len(requested)
    request_index = 0
    for model_name in requested:
        for representation in REPRESENTATIONS:
            x = representations[representation]
            for test_sample in samples:
                request_index += 1
                job_path = jobs_dir / (
                    f"test_{test_sample}__{representation}__{model_name}.json"
                )
                if job_path.is_file():
                    existing = json.loads(job_path.read_text(encoding="utf-8"))
                    if (
                        existing.get("status") == "passed"
                        and existing.get("provenance_hash") == provenance_hash
                    ):
                        print(
                            f"[{request_index}/{total_requested}] resume {job_path.name}",
                            flush=True,
                        )
                        continue
                print(
                    f"[{request_index}/{total_requested}] run {model_name} "
                    f"{representation} test={test_sample}",
                    flush=True,
                )
                job = run_job(
                    model_name=model_name,
                    representation=representation,
                    test_sample=test_sample,
                    x=x,
                    y=y,
                    groups=groups,
                    rf_jobs=args.rf_jobs,
                    provenance_hash=provenance_hash,
                )
                write_json(job_path, job)
                print(
                    f"  selected={job['selected_config']} "
                    f"mae={job['outer_metrics']['sample_balanced_mae']:.4f} "
                    f"seconds={job['runtime_seconds']:.1f}",
                    flush=True,
                )
                aggregate(
                    run_dir=args.run_dir,
                    final_report_path=args.final_report,
                    rows=rows,
                    y=y,
                    groups=groups,
                    provenance=provenance,
                )
    summary = aggregate(
        run_dir=args.run_dir,
        final_report_path=args.final_report,
        rows=rows,
        y=y,
        groups=groups,
        provenance=provenance,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
