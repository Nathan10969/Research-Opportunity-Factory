from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--rf-jobs", type=int, required=True)
    return parser.parse_args()


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("pnnl_nested_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    runner = load_runner(args.runner)
    provenance = json.loads(
        (args.run_dir / "configuration_provenance.json").read_text(encoding="utf-8")
    )
    if provenance["rf_jobs"] != args.rf_jobs:
        raise ValueError("worker rf-jobs does not match frozen provenance")
    rows, y, groups, representations = runner.load_inputs(
        args.feature_dir, args.source_manifest
    )
    del rows
    all_keys = [
        (representation, sample)
        for representation in runner.REPRESENTATIONS
        for sample in sorted(set(groups.tolist()))
    ]
    shard_keys = all_keys[args.shard_index :: args.num_shards]
    jobs_dir = args.run_dir / "jobs"
    for index, (representation, test_sample) in enumerate(shard_keys, start=1):
        path = jobs_dir / (
            f"test_{test_sample}__{representation}__random_forest.json"
        )
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "passed"
                and existing.get("provenance_hash") == provenance["provenance_hash"]
            ):
                print(
                    f"[shard {args.shard_index} {index}/{len(shard_keys)}] "
                    f"resume {path.name}",
                    flush=True,
                )
                continue
        print(
            f"[shard {args.shard_index} {index}/{len(shard_keys)}] "
            f"run {representation} test={test_sample}",
            flush=True,
        )
        job = runner.run_job(
            model_name="random_forest",
            representation=representation,
            test_sample=test_sample,
            x=representations[representation],
            y=y,
            groups=groups,
            rf_jobs=args.rf_jobs,
            provenance_hash=provenance["provenance_hash"],
        )
        runner.write_json(path, job)
        print(
            f"  selected={job['selected_config']} "
            f"mae={job['outer_metrics']['sample_balanced_mae']:.4f} "
            f"seconds={job['runtime_seconds']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
