"""Command-line entry point for the resumable Idea Factory pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import pipeline


COMMANDS = (
    "init-run", "emit-corpus-jobs", "ingest-corpus-labels", "emit-card-jobs",
    "ingest-cards", "build-landscape", "emit-opportunity-jobs",
    "ingest-opportunities", "quality-gate", "internal-dedup", "emit-recon-pack",
    "ingest-recon", "emit-route-jobs", "ingest-routes", "emit-review-jobs",
    "ingest-reviews", "ingest-human-scores", "finalize", "status",
)


def _run_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="Exact run directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idea-factory", description="Auditable KV x Memory opportunity pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-run", help="Create one immutable-provenance run")
    _run_argument(init)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--config-repo-root", type=Path)
    init.add_argument("--mode", choices=("live", "offline-fixture"), default="live")
    init.add_argument("--new-run", action="store_true")

    status = subparsers.add_parser("status", help="Report honest run and external-handoff state")
    _run_argument(status)

    result_commands = {
        "ingest-corpus-labels", "ingest-cards", "ingest-opportunities",
        "ingest-recon", "ingest-routes", "ingest-reviews",
    }
    prompt_commands = {"emit-corpus-jobs", "ingest-corpus-labels", "emit-card-jobs", "emit-opportunity-jobs"}
    test_ready_commands = {
        "ingest-recon", "emit-route-jobs", "ingest-routes", "emit-review-jobs",
        "ingest-reviews", "ingest-human-scores", "finalize",
    }
    for command in COMMANDS:
        if command in {"init-run", "status"}:
            continue
        stage = subparsers.add_parser(command)
        _run_argument(stage)
        stage.add_argument("--config", type=Path)
        if command in result_commands:
            stage.add_argument("--results", type=Path, required=command != "ingest-recon")
        elif command in {"internal-dedup", "ingest-human-scores"}:
            stage.add_argument("--results", type=Path)
        if command in prompt_commands:
            stage.add_argument("--prompt", type=Path)
        if command == "build-landscape":
            stage.add_argument("--reviewed-assignments", type=Path)
        if command == "emit-recon-pack":
            stage.add_argument("--recon-context", type=Path)
        if command == "ingest-recon":
            stage.add_argument("--execution-bundle", type=Path)
        if command in {"emit-route-jobs", "ingest-routes", "emit-review-jobs", "ingest-reviews", "ingest-human-scores"}:
            stage.add_argument("--route-prompt", type=Path)
        if command in {"emit-review-jobs", "ingest-reviews", "ingest-human-scores"}:
            stage.add_argument("--review-prompt", type=Path)
        if command in test_ready_commands:
            stage.add_argument(
                "--allow-test-ready", action="store_true",
                help="Permit explicitly TEST_ONLY offline recon; never asserts live readiness",
            )
        if command == "finalize":
            stage.add_argument("--repository-root", type=Path)
            stage.add_argument("--route-prompt", type=Path)
            stage.add_argument("--review-prompt", type=Path)
        stage.add_argument(
            "--new-run", action="store_true",
            help="Never rewrites this run; initialize a distinct --run path first",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-run":
            run = pipeline.init_run(
                args.run, args.config, mode=args.mode, new_run=args.new_run,
                config_repo_root=args.config_repo_root,
            )
            print(json.dumps({"result": "INITIALIZED", "run": str(run)}, sort_keys=True))
            return 0
        if args.command == "status":
            print(json.dumps(pipeline.status(args.run), ensure_ascii=False, sort_keys=True))
            return 0
        if getattr(args, "new_run", False):
            raise pipeline.PipelineError(
                "--new-run never mutates an existing run; invoke init-run with a distinct --run path"
            )
        if args.command == "finalize":
            result = pipeline.finalize(
                args.run, args.config, allow_test_ready=args.allow_test_ready,
                repository_root=args.repository_root, route_prompt=args.route_prompt,
                review_prompt=args.review_prompt,
            )
        else:
            result = pipeline.execute(
                args.command, args.run, args.config,
                results=getattr(args, "results", None),
                prompt=getattr(args, "prompt", None),
                reviewed_assignments=getattr(args, "reviewed_assignments", None),
                allow_test_ready=getattr(args, "allow_test_ready", False),
                route_prompt=getattr(args, "route_prompt", None),
                review_prompt=getattr(args, "review_prompt", None),
                recon_context=getattr(args, "recon_context", None),
                execution_bundle=getattr(args, "execution_bundle", None),
            )
        print(json.dumps({"result": result, "run": str(Path(args.run).resolve())}, sort_keys=True))
        return 0
    except RuntimeError as exc:
        if not pipeline.is_expected_operational_runtime_error(exc):
            raise
        parser.error(str(exc))
    except (pipeline.PipelineError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
