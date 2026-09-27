"""Resumable, evidence-preserving orchestration for Idea Factory v1."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import read_jsonl, sha256_file, stable_id, write_json
from .models import (
    DecisiveTest,
    HumanScores,
    IdeaPack,
    IdeaStatus,
    NearestPrior,
    RUN_ORDER,
    RunStage,
    RunState,
)
from .stage_io import (
    encode_json, encode_jsonl, publish_cross_directory_transaction, publish_transaction,
    stage_mutation_lock,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = {
    "corpus": REPOSITORY_ROOT / "prompts" / "corpus_router.md",
    "corpus_v3": REPOSITORY_ROOT / "prompts" / "corpus_router_v3.md",
    "card": REPOSITORY_ROOT / "prompts" / "paper_card.md",
    "opportunity": REPOSITORY_ROOT / "prompts" / "opportunity_miner.md",
    "route": REPOSITORY_ROOT / "prompts" / "route_generator.md",
    "review": REPOSITORY_ROOT / "prompts" / "reviewer.md",
}


class PipelineError(ValueError):
    """A bounded operator error suitable for the CLI boundary."""


class AwaitingExternalResult(PipelineError):
    """A handoff was emitted successfully and is waiting, not failed."""


def is_expected_operational_runtime_error(error: RuntimeError) -> bool:
    """Recognize only runtime failures that are bounded operator conditions."""

    message = str(error)
    return message.startswith("stage mutation lock is busy:") or message.startswith((
        "finalize compensation",
        "finalize persistent ledger compensation",
        "finalize failed and exact local compensation",
        "finalize failed and exact compensation",
    ))


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid JSON artifact: {path}") from exc
    if type(value) is not dict:
        raise PipelineError(f"JSON artifact must be an object: {path}")
    return value


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _anchored_run(path: Path, *, create: bool = False) -> Path:
    lexical = Path(os.path.abspath(path))
    if create:
        lexical.mkdir(parents=True, exist_ok=True)
    if not lexical.is_dir() or lexical.resolve(strict=True) != lexical:
        raise PipelineError("run directory must have exact resolved identity")
    return lexical


def _state_path(run: Path) -> Path:
    return run / "state.json"


def _load_state(run: Path) -> RunState:
    return RunState.model_validate(_strict_json(_state_path(run)))


def _write_state(run: Path, state: RunState) -> None:
    write_json(_state_path(run), state.model_dump(mode="json"))


def _manifest(run: Path) -> dict[str, Any]:
    return _strict_json(run / "manifest.json")


def _config(run: Path, explicit: Path | None) -> tuple[Path, Any]:
    from .corpus import CorpusRouterConfig

    manifest = _manifest(run)
    configured = Path(str(manifest["config_path"]))
    path = Path(explicit).resolve() if explicit is not None else configured
    if path != configured or sha256_file(path) != manifest["config_sha256"]:
        raise PipelineError("config changed; initialize --new-run instead of rewriting evidence")
    return path, CorpusRouterConfig.from_json(path, repo_root=Path(str(manifest["config_repo_root"])))


def _path_hash(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    if resolved.is_file():
        return {"path": str(resolved), "sha256": sha256_file(resolved)}
    if resolved.is_dir():
        files = sorted(
            candidate for candidate in resolved.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()
        )
        if any(candidate.is_symlink() for candidate in resolved.rglob("*")):
            raise PipelineError(f"input bundle may not contain links: {resolved}")
        bundle = [
            {"path": candidate.relative_to(resolved).as_posix(), "sha256": sha256_file(candidate)}
            for candidate in files
        ]
        return {"path": str(resolved), "sha256": _canonical_hash(bundle)}
    raise PipelineError(f"input must be a regular file or anchored directory: {resolved}")


def _command_digest(command: str, config_path: Path, paths: Mapping[str, Path | None], options: Mapping[str, object]) -> str:
    payload = {
        "command": command,
        "config": _path_hash(config_path),
        "paths": {name: (_path_hash(path) if path is not None else None) for name, path in sorted(paths.items())},
        "options": dict(sorted(options.items())),
    }
    return _canonical_hash(payload)


def _stage_index(stage: RunStage) -> int:
    return RUN_ORDER.index(stage)


_OWNED_DIRECTORIES = {
    "emit-corpus-jobs": ("corpus",), "ingest-corpus-labels": ("corpus",),
    "emit-card-jobs": ("cards",), "ingest-cards": ("results",),
    "build-landscape": ("landscape",), "emit-opportunity-jobs": ("opportunities",),
    "ingest-opportunities": ("opportunities",), "quality-gate": ("quality",),
    "internal-dedup": ("ledger",), "emit-recon-pack": ("recon",),
    "ingest-recon": ("recon",), "emit-route-jobs": ("routes",),
    "ingest-routes": ("routes",), "emit-review-jobs": ("review",),
    "ingest-reviews": ("review",), "ingest-human-scores": ("ledger",),
    "finalize": ("idea_packs", "ledger"),
}


def _output_prefix(command: str) -> str:
    return f"output:{command}:"


def _owned_files(run: Path, command: str) -> list[Path]:
    files: list[Path] = []
    for rel_dir in _OWNED_DIRECTORIES[command]:
        directory = run / rel_dir
        if not directory.is_dir() or directory.resolve(strict=True) != directory:
            continue
        files.extend(
            path for path in directory.rglob("*")
            if path.is_file() and not path.is_symlink() and path.name != ".stage.lock"
        )
    if command == "finalize" and (run / "report.md").is_file():
        files.append(run / "report.md")
    return sorted(set(files), key=lambda path: path.relative_to(run).as_posix())


def _record_output_snapshots(run: Path, state: RunState, command: str) -> None:
    files = _owned_files(run, command)
    if not files:
        raise PipelineError(f"{command} produced no owned artifact bundle")
    for path in files:
        relpath = path.relative_to(run).as_posix()
        state.set_input_hash(_output_prefix(command) + relpath, sha256_file(path))


def _validate_output_snapshots(run: Path, state: RunState, command: str) -> None:
    prefix = _output_prefix(command)
    snapshots = {key[len(prefix):]: digest for key, digest in state.input_hashes.items() if key.startswith(prefix)}
    if not snapshots:
        raise PipelineError(f"{command} has no owned artifact snapshot; use --new-run")
    for relpath, digest in snapshots.items():
        path = run / Path(relpath)
        if not path.is_file() or path.resolve(strict=True) != path or not path.is_relative_to(run):
            raise PipelineError(f"{command} owned artifact is missing; use --new-run: {relpath}")
        if sha256_file(path) != digest:
            raise PipelineError(f"{command} owned artifact was modified; use --new-run: {relpath}")


def _assert_stage(state: RunState, expected: RunStage, command: str) -> None:
    if state.stage != expected:
        raise PipelineError(f"{command} requires RunState {expected.value}; current stage is {state.stage.value}")


def _idempotent_guard(run: Path, state: RunState, key: str, digest: str, *, expected: RunStage, target: RunStage | None) -> bool:
    previous = state.input_hashes.get(key)
    if previous is not None:
        if previous != digest:
            raise PipelineError("input hashes changed; use --new-run and preserve this run")
        if target is None:
            _validate_output_snapshots(run, state, key.removeprefix("command:"))
            return True
        if key == "command:ingest-human-scores" and state.stage == RunStage.REVIEWED:
            _validate_output_snapshots(run, state, "ingest-human-scores")
            return True
        if _stage_index(state.stage) >= _stage_index(target):
            _validate_output_snapshots(run, state, key.removeprefix("command:"))
            return True
    _assert_stage(state, expected, key.removeprefix("command:"))
    return False


def _record_success(run: Path, state: RunState, key: str, digest: str, target: RunStage | None) -> None:
    state.set_input_hash(key, digest)
    _record_output_snapshots(run, state, key.removeprefix("command:"))
    if target is not None:
        state.advance(target)
    _write_state(run, state)


def _sources(path: Path) -> list[str | Path]:
    resolved = Path(path).resolve(strict=True)
    if resolved.suffix.casefold() == ".jsonl":
        try:
            lines = [line for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]
        except UnicodeDecodeError as exc:
            raise PipelineError(f"result JSONL must be UTF-8: {resolved}") from exc
        return lines
    payload = _strict_json(resolved)
    values = payload.get("results")
    if type(values) is not list or any(type(value) is not str for value in values):
        raise PipelineError("result manifest must contain a string results array")
    return [(resolved.parent / value).resolve(strict=True) for value in values]


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(path).resolve(strict=True))


def _directory_snapshot(directory: Path) -> tuple[bool, dict[str, bytes], set[str]]:
    lexical = Path(os.path.abspath(directory))
    if not lexical.exists():
        return False, {}, set()
    if not lexical.is_dir() or lexical.resolve(strict=True) != lexical:
        raise PipelineError(f"snapshot directory is unsafe: {lexical}")
    files: dict[str, bytes] = {}
    directories = {"."}
    for child in lexical.rglob("*"):
        if child.is_symlink():
            raise PipelineError(f"snapshot directory contains a link: {child}")
        relative = child.relative_to(lexical).as_posix()
        if child.is_dir():
            directories.add(relative)
        elif child.is_file() and child.name != ".stage.lock":
            files[relative] = child.read_bytes()
        elif not child.is_file():
            raise PipelineError(f"snapshot directory contains an unsafe artifact: {child}")
    return True, files, directories


def _restore_directory_snapshot(
    directory: Path,
    snapshot: tuple[bool, dict[str, bytes], set[str]],
) -> None:
    existed, files, directories = snapshot
    lexical = Path(os.path.abspath(directory))
    lexical.mkdir(parents=True, exist_ok=True)
    if not lexical.is_dir() or lexical.resolve(strict=True) != lexical:
        raise PipelineError(f"restore directory is unsafe: {lexical}")
    with stage_mutation_lock(lexical):
        current = _directory_snapshot(lexical)[1]
        payloads: dict[Path, bytes | None] = {
            lexical / Path(relative): None
            for relative in current
            if relative not in files
        }
        payloads.update({lexical / Path(relative): payload for relative, payload in files.items()})
        if payloads:
            publish_cross_directory_transaction(payloads)
        for child in sorted(
            (path for path in lexical.rglob("*") if path.is_dir() and not path.is_symlink()),
            key=lambda path: len(path.parts), reverse=True,
        ):
            if child.relative_to(lexical).as_posix() not in directories:
                try:
                    child.rmdir()
                except OSError:
                    pass
    if not existed:
        try:
            lexical.rmdir()
        except OSError:
            pass


def _safe_optional_file_bytes(path: Path) -> bytes | None:
    lexical = Path(os.path.abspath(path))
    if not lexical.exists():
        return None
    if not lexical.is_file() or lexical.is_symlink() or lexical.resolve(strict=True) != lexical:
        raise PipelineError(f"snapshot file is unsafe: {lexical}")
    return lexical.read_bytes()


_PERSISTENT_COMPENSATION_UNRESOLVED = (
    "FINALIZE_PERSISTENT_LEDGER_COMPENSATION_UNRESOLVED"
)
_LOCAL_COMPENSATION_UNRESOLVED = "FINALIZE_LOCAL_COMPENSATION_UNRESOLVED"


def _ledger_records_by_id(
    records: Sequence[Mapping[str, Any]], *, context: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        event_id = record.get("ledger_update_id")
        if type(event_id) is not str or not event_id.strip():
            raise PipelineError(f"{context} contains an invalid ledger event ID")
        if event_id in indexed:
            raise PipelineError(f"{context} contains an ambiguous duplicate ledger event ID")
        indexed[event_id] = dict(record)
    return indexed


def _persistent_ledger_snapshot(repository_root: Path) -> dict[str, Any]:
    data = repository_root / "data"
    data_existed = data.exists()
    if data_existed and (not data.is_dir() or data.resolve(strict=True) != data):
        raise PipelineError("repository data path is unsafe")
    data.mkdir(parents=True, exist_ok=True)
    persistent = data / "idea_ledger.jsonl"
    with stage_mutation_lock(data):
        persistent_existed = persistent.exists()
        if persistent_existed:
            if (
                not persistent.is_file() or persistent.is_symlink()
                or persistent.resolve(strict=True) != persistent
            ):
                raise PipelineError("persistent ledger path is unsafe")
            records = read_jsonl(persistent)
            preexisting_ids = frozenset(
                _ledger_records_by_id(records, context="persistent ledger snapshot")
            )
        else:
            preexisting_ids = frozenset()
    if not data_existed:
        try:
            data.rmdir()
        except OSError:
            pass
    return {
        "data_existed": data_existed,
        "persistent_existed": persistent_existed,
        "preexisting_ids": preexisting_ids,
    }


def _finalization_snapshot(run: Path, repository_root: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(repository_root))
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise PipelineError("repository root must have exact resolved identity")
    state_bytes = (run / "state.json").read_bytes()
    return {
        "state": state_bytes,
        "run_id": str(json.loads(state_bytes)["run_id"]),
        "report": _safe_optional_file_bytes(run / "report.md"),
        "idea_packs": _directory_snapshot(run / "idea_packs"),
        "run_updates": _safe_optional_file_bytes(run / "ledger" / "updates.jsonl"),
        "persistent": _persistent_ledger_snapshot(root),
    }


def _restore_finalization_local_snapshot(run: Path, snapshot: Mapping[str, Any]) -> None:
    packs = run / "idea_packs"
    ledger = run / "ledger"
    pack_snapshot = snapshot["idea_packs"]
    pack_existed, pack_files, pack_directories = pack_snapshot
    packs.mkdir(parents=True, exist_ok=True)
    ledger.mkdir(parents=True, exist_ok=True)
    lock_dirs = [packs, ledger]
    with ExitStack() as stack:
        for directory in sorted(lock_dirs, key=lambda path: str(path).casefold()):
            stack.enter_context(stage_mutation_lock(directory))
        current_pack_files = _directory_snapshot(packs)[1]
        payloads: dict[Path, bytes | None] = {
            run / "state.json": snapshot["state"],
            run / "report.md": snapshot["report"],
            ledger / "updates.jsonl": snapshot["run_updates"],
        }
        payloads.update({
            packs / Path(relative): None
            for relative in current_pack_files
            if relative not in pack_files
        })
        payloads.update({packs / Path(relative): payload for relative, payload in pack_files.items()})
        publish_cross_directory_transaction(payloads)
        for child in sorted(
            (path for path in packs.rglob("*") if path.is_dir() and not path.is_symlink()),
            key=lambda path: len(path.parts), reverse=True,
        ):
            if child.relative_to(packs).as_posix() not in pack_directories:
                try:
                    child.rmdir()
                except OSError:
                    pass
    if not pack_existed:
        try:
            packs.rmdir()
        except OSError:
            pass


def _validate_failed_ledger_projection(
    run: Path, snapshot: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    updates = run / "ledger" / "updates.jsonl"
    if not updates.exists():
        return {}
    if not updates.is_file() or updates.is_symlink() or updates.resolve(strict=True) != updates:
        raise PipelineError("failed finalize run ledger projection is unsafe")
    records = read_jsonl(updates)
    indexed = _ledger_records_by_id(records, context="failed finalize run ledger projection")
    if any(record.get("run_id") != snapshot["run_id"] for record in indexed.values()):
        raise PipelineError("failed finalize run ledger projection has a foreign run ID")
    transaction_ids = {record.get("ledger_transaction_id") for record in indexed.values()}
    if len(transaction_ids) > 1 or any(type(value) is not str for value in transaction_ids):
        raise PipelineError("failed finalize run ledger projection is transaction-ambiguous")
    for record in indexed.values():
        recorded_hash = record.get("update_sha256")
        body = dict(record)
        body.pop("update_sha256", None)
        if type(recorded_hash) is not str or recorded_hash != _canonical_hash(body):
            raise PipelineError("failed finalize run ledger projection has an invalid event hash")
    return indexed


def _compensate_persistent_ledger(
    run: Path, repository_root: Path, snapshot: Mapping[str, Any],
) -> None:
    persistent_snapshot = snapshot["persistent"]
    preexisting_ids = set(persistent_snapshot["preexisting_ids"])
    projection = _validate_failed_ledger_projection(run, snapshot)
    newly_added = {
        event_id: record for event_id, record in projection.items()
        if event_id not in preexisting_ids
    }
    data = repository_root / "data"
    data.mkdir(parents=True, exist_ok=True)
    persistent = data / "idea_ledger.jsonl"
    with stage_mutation_lock(data):
        if persistent.exists():
            if (
                not persistent.is_file() or persistent.is_symlink()
                or persistent.resolve(strict=True) != persistent
            ):
                raise PipelineError("persistent ledger compensation target is unsafe")
            current = read_jsonl(persistent)
        else:
            current = []
        current_by_id = _ledger_records_by_id(
            current, context="persistent ledger compensation target",
        )
        unexpected_run_ids = {
            event_id for event_id, record in current_by_id.items()
            if record.get("run_id") == snapshot["run_id"]
            and event_id not in preexisting_ids and event_id not in newly_added
        }
        if unexpected_run_ids:
            raise PipelineError("persistent ledger compensation found an ambiguous run projection")
        for event_id, expected in newly_added.items():
            current_record = current_by_id.get(event_id)
            if current_record is None:
                raise PipelineError("persistent ledger compensation target event is missing")
            if current_record != expected:
                raise PipelineError("persistent ledger compensation target event conflicts")
        if newly_added:
            filtered = [
                record for record in current
                if str(record["ledger_update_id"]) not in newly_added
            ]
            payload = (
                encode_jsonl(filtered)
                if filtered or persistent_snapshot["persistent_existed"] else None
            )
            publish_transaction({persistent: payload})
    if not persistent_snapshot["data_existed"] and not persistent.exists():
        try:
            data.rmdir()
        except OSError:
            pass


def _record_compensation_unresolved(
    run: Path, snapshot: Mapping[str, Any], errors: Sequence[str],
) -> None:
    state = RunState.model_validate(json.loads(snapshot["state"]))
    for error in errors:
        if error not in state.unresolved_errors:
            state.add_unresolved_error(error)
    publish_transaction({run / "state.json": encode_json(state.model_dump(mode="json"))})


def _import_execution_bundle(run: Path, bundle_path: Path, config: Any) -> None:
    from .recon import validate_execution_jobs, validate_execution_receipts

    bundle = Path(bundle_path).resolve(strict=True)
    if not bundle.is_dir():
        raise PipelineError("execution bundle must be an anchored directory")
    jobs = list(validate_execution_jobs(run, config))
    receipt_source = bundle / "execution_receipts.jsonl"
    if not receipt_source.is_file() or receipt_source.resolve(strict=True).parent != bundle:
        raise PipelineError("execution bundle lacks anchored execution_receipts.jsonl")
    receipts = read_jsonl(receipt_source)
    if [row.get("job_id") for row in receipts] != [job["job_id"] for job in jobs]:
        raise PipelineError("execution bundle receipts do not exactly cover jobs")
    payloads: dict[Path, bytes | None] = {run / "recon" / "execution_receipts.jsonl": receipt_source.read_bytes()}
    for job, receipt in zip(jobs, receipts):
        relpath = Path(str(job["raw_output_path"]))
        if relpath.is_absolute() or ".." in relpath.parts or relpath.parts[:1] != ("recon",):
            raise PipelineError("execution job raw path is unsafe")
        external = bundle.joinpath(*relpath.parts[1:])
        target = run / relpath
        if receipt.get("status") in {"SUCCESS", "EMPTY"}:
            if not external.is_file() or not external.resolve(strict=True).is_relative_to(bundle):
                raise PipelineError(f"execution bundle raw artifact is missing: {relpath.as_posix()}")
            raw = external.read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt.get("raw_file_sha256"):
                raise PipelineError("execution bundle raw hash does not match receipt")
            target.parent.mkdir(parents=True, exist_ok=True)
            payloads[target] = raw
        elif receipt.get("status") in {"ERROR", "CREDENTIALS_PROTOCOL_REQUIRED"}:
            if receipt.get("raw_file_sha256") is not None or external.exists():
                raise PipelineError("failed execution receipt cannot carry a raw artifact")
            if target.exists():
                payloads[target] = None
        else:
            raise PipelineError("execution bundle receipt status is invalid")
    previous = {path: path.read_bytes() if path.is_file() else None for path in payloads}
    publish_cross_directory_transaction(payloads)
    try:
        validate_execution_receipts(run, config)
    except BaseException:
        publish_cross_directory_transaction(previous)
        raise


class RealBackend:
    """Thin adapter over the strict, replay-validating stage APIs."""

    def emit_corpus_jobs(self, run: Path, *, config: Any, prompt: Path, **_: object) -> None:
        from .corpus import enumerate_candidates, write_router_jobs
        write_router_jobs(enumerate_candidates(config), run, prompt_path=prompt, protocol_version=config.protocol_version)

    def ingest_corpus_labels(self, run: Path, *, config: Any, results: Path, prompt: Path, **_: object) -> None:
        from .corpus import enumerate_candidates, ingest_router_results, select_pilot
        candidates = enumerate_candidates(config)
        ingested = ingest_router_results(
            candidates, _jsonl_objects(results), prompt_sha256=sha256_file(prompt),
            protocol_version=config.protocol_version,
        )
        select_pilot(config, ingested, run)

    def emit_card_jobs(self, run: Path, *, prompt: Path, **_: object) -> None:
        from .cards import emit_card_jobs
        emit_card_jobs(run / "corpus" / "selection_manifest.jsonl", prompt, run / "cards" / "card_jobs.jsonl")

    def ingest_cards(self, run: Path, *, results: Path, **_: object) -> None:
        from .cards import ingest_card_results
        ingest_card_results(run / "cards" / "card_jobs.jsonl", _jsonl_objects(results))

    def build_landscape(self, run: Path, *, reviewed_assignments: Path | None = None, **_: object) -> None:
        from .landscape import build_landscape
        build_landscape(run, reviewed_assignments)

    def emit_opportunity_jobs(self, run: Path, *, prompt: Path, reviewed_local_entries: Path | None = None, **_: object) -> None:
        from .opportunities import emit_mining_jobs
        emit_mining_jobs(run, prompt, reviewed_local_entries=reviewed_local_entries)

    def ingest_opportunities(self, run: Path, *, results: Path, **_: object) -> None:
        from .opportunities import ingest_opportunity_results
        ingest_opportunity_results(run / "opportunities" / "mining_jobs.jsonl", _sources(results))

    def quality_gate(self, run: Path, **_: object) -> None:
        from .quality import publish_quality
        publish_quality(run)

    def internal_dedup(self, run: Path, *, config: Any, results: Path | None, **_: object) -> bool:
        from .ledger import emit_shape_assignment_jobs, ingest_shape_assignments, publish_internal_dedup
        from .legacy_ledger import import_legacy_ledger
        import_legacy_ledger(run, config)
        jobs = emit_shape_assignment_jobs(run, config)
        if read_jsonl(jobs) and results is None:
            return False
        ingest_shape_assignments(jobs, _sources(results) if results is not None else [])
        publish_internal_dedup(run, config)
        return True

    def emit_recon_pack(self, run: Path, *, config: Any, recon_context: Path | None, **_: object) -> bool:
        from .recon import (
            emit_recon_execution_job_templates, emit_recon_query_pack,
            materialize_execution_jobs, prepare_execution_context,
            record_operator_trust_attestation,
        )
        emit_recon_query_pack(run, config)
        templates_path = emit_recon_execution_job_templates(run, config)
        if not read_jsonl(templates_path):
            materialize_execution_jobs(run, config)
            return True
        if recon_context is None:
            return False
        context = _strict_json(Path(recon_context).resolve(strict=True))
        expected = {"workspace_root", "skill_roots", "prepared_at", "uv_executable", "allow_test_attestation", "operator_attestation"}
        if set(context) != expected or type(context["skill_roots"]) is not dict or type(context["allow_test_attestation"]) is not bool or type(context["operator_attestation"]) is not dict:
            raise PipelineError("recon context schema is not exact")
        record_operator_trust_attestation(run, context["operator_attestation"])
        prepare_execution_context(
            run, workspace_root=Path(str(context["workspace_root"])),
            skill_roots=context["skill_roots"], prepared_at=context["prepared_at"],
            uv_executable=context["uv_executable"],
            allow_test_attestation=context["allow_test_attestation"],
        )
        materialize_execution_jobs(run, config)
        return True

    def ingest_recon(self, run: Path, *, config: Any, results: Path | None, execution_bundle: Path | None, **_: object) -> bool:
        from .recon import emit_recon_report_jobs, ingest_recon_reports, normalize_recon_raw
        report_jobs = run / "recon" / "report_jobs.jsonl"
        if not report_jobs.exists():
            from .recon import validate_execution_jobs
            if validate_execution_jobs(run, config):
                if execution_bundle is None:
                    raise PipelineError("ingest-recon requires --execution-bundle for materialized jobs")
                snapshot = _directory_snapshot(run / "recon")
                try:
                    _import_execution_bundle(run, execution_bundle, config)
                    normalize_recon_raw(run, config)
                except BaseException:
                    _restore_directory_snapshot(run / "recon", snapshot)
                    raise
            else:
                normalize_recon_raw(run, config)
            report_jobs = emit_recon_report_jobs(run, config)
        if read_jsonl(report_jobs) and results is None:
            return False
        ingest_recon_reports(report_jobs, _sources(results) if results is not None else [], config)
        return True

    def emit_route_jobs(self, run: Path, *, config: Any, route_prompt: Path, allow_test_ready: bool, **_: object) -> None:
        from .routes import emit_route_jobs
        emit_route_jobs(run, config, prompt_path=route_prompt, allow_test_ready=allow_test_ready)

    def ingest_routes(self, run: Path, *, config: Any, results: Path, route_prompt: Path, allow_test_ready: bool, **_: object) -> None:
        from .routes import ingest_route_results
        ingest_route_results(run / "routes" / "jobs.jsonl", _sources(results), config, prompt_path=route_prompt, allow_test_ready=allow_test_ready)

    def emit_review_jobs(self, run: Path, *, config: Any, review_prompt: Path, route_prompt: Path, allow_test_ready: bool, **_: object) -> None:
        from .review import emit_review_jobs
        emit_review_jobs(run, config, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)

    def ingest_reviews(self, run: Path, *, config: Any, results: Path, review_prompt: Path, route_prompt: Path, allow_test_ready: bool, **_: object) -> bool:
        from .review import emit_review_jobs, ingest_review_results, validate_review_bundle
        pending = run / "review" / "round2_jobs.jsonl"
        jobs = pending if pending.is_file() else run / "review" / "jobs.jsonl"
        ingest_review_results(jobs, _sources(results), config, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        bundle = validate_review_bundle(run, config, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        narrowed = {
            str(row["route_id"]) for row in bundle["results"]
            if row.get("decision") == "NARROW" and int(row.get("round", 0)) == 1
        }
        round_two_done = {
            str(row["route_id"]) for row in bundle["results"]
            if int(row.get("round", 0)) == 2
        }
        if narrowed - round_two_done:
            round_two = emit_review_jobs(run, config, round_number=2, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
            if read_jsonl(round_two):
                return False
        return True

    def ingest_human_scores(self, run: Path, *, config: Any, results: Path | None, review_prompt: Path, route_prompt: Path, allow_test_ready: bool, **_: object) -> bool | str:
        from .ledger import emit_human_scoring_jobs, ingest_human_scores
        jobs = emit_human_scoring_jobs(run, config, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        human_jobs = read_jsonl(jobs)
        if human_jobs and results is None:
            return False
        ingest_human_scores(jobs, _sources(results) if results is not None else [], config, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        return True if human_jobs else "ZERO_SURVIVOR"

    @staticmethod
    def _completion_pack(raw: Mapping[str, Any]) -> IdeaPack:
        priors = tuple(NearestPrior.model_validate({key: prior[key] for key in ("paper", "exact_overlap", "residual_difference", "evidence_url_or_id")}) for prior in raw["nearest_priors"])
        score = HumanScores.model_validate(raw["human_scores"])
        return IdeaPack(
            idea_id=raw["idea_id"], status=IdeaStatus(raw["status"]),
            opportunity=json.dumps(raw["opportunity"], ensure_ascii=False, sort_keys=True),
            core_hypothesis=raw["core_hypothesis"], why_now=raw["why_now"],
            supporting_observations=tuple(raw["supporting_observations"]),
            inference_flags=tuple(raw["evidence_flags"]), nearest_priors=priors,
            proposed_mechanism=raw["proposed_mechanism"], source_of_gain=raw["source_of_gain"],
            cheapest_decisive_test=DecisiveTest.model_validate(raw["cheapest_decisive_test"]),
            strongest_baseline=json.dumps(raw["strongest_baseline"], ensure_ascii=False, sort_keys=True),
            kill_condition=raw["kill_condition"], main_uncertainty=raw["main_uncertainty"],
            expected_reviewer_2_objection=json.dumps(raw["expected_reviewer_2_objection"], ensure_ascii=False, sort_keys=True),
            response_to_objection=raw["response_to_objection"],
            evidence_that_would_make_reviewer_correct=json.dumps(raw["evidence_that_would_make_reviewer_correct"], ensure_ascii=False, sort_keys=True),
            human_scores=score, human_reason_codes=tuple(raw["human_reason_codes"]),
        )

    def finalize(self, run: Path, *, config: Any, review_prompt: Path, route_prompt: Path, allow_test_ready: bool, repository_root: Path, **_: object) -> dict[str, object]:
        from .ledger import append_ledger_updates, validate_human_bundle, validate_ledger_updates
        from .render import render_idea_packs, validate_idea_pack_bundle
        validate_human_bundle(run, config, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        render_idea_packs(run, config, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        ledger_paths = append_ledger_updates(run, config, repository_root=repository_root, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        validate_ledger_updates(run, config, repository_root=repository_root, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        manifest = validate_idea_pack_bundle(run, config, prompt_path=review_prompt, route_prompt_path=route_prompt, allow_test_ready=allow_test_ready)
        ready = []
        for idea_id in manifest["idea_ids"]:
            raw = _strict_json(run / "idea_packs" / f"{idea_id}.json")
            if raw["status"] == IdeaStatus.READY_FOR_CHEAP_TEST.value:
                ready.append(self._completion_pack(raw))
        return {
            "pack_count": manifest["pack_count"], "ready_count": manifest["ready_count"],
            "ready_packs": ready, "persistent_ledger": ledger_paths["persistent_ledger"],
        }


BACKEND: Any = RealBackend()


_TRANSITIONS: dict[str, tuple[RunStage, RunStage | None]] = {
    "emit-corpus-jobs": (RunStage.CREATED, None),
    "ingest-corpus-labels": (RunStage.CREATED, RunStage.CORPUS_ROUTED),
    "emit-card-jobs": (RunStage.CORPUS_ROUTED, None),
    "ingest-cards": (RunStage.CORPUS_ROUTED, RunStage.CARDS_READY),
    "build-landscape": (RunStage.CARDS_READY, RunStage.LANDSCAPE_READY),
    "emit-opportunity-jobs": (RunStage.LANDSCAPE_READY, None),
    "ingest-opportunities": (RunStage.LANDSCAPE_READY, RunStage.OPPORTUNITIES_READY),
    "quality-gate": (RunStage.OPPORTUNITIES_READY, RunStage.QUALITY_GATED),
    "internal-dedup": (RunStage.QUALITY_GATED, None),
    "emit-recon-pack": (RunStage.QUALITY_GATED, None),
    "ingest-recon": (RunStage.QUALITY_GATED, RunStage.RECON_READY),
    "emit-route-jobs": (RunStage.RECON_READY, None),
    "ingest-routes": (RunStage.RECON_READY, RunStage.ROUTES_READY),
    "emit-review-jobs": (RunStage.ROUTES_READY, None),
    "ingest-reviews": (RunStage.ROUTES_READY, RunStage.REVIEWED),
    "ingest-human-scores": (RunStage.REVIEWED, RunStage.HUMAN_SCORED),
}


_UPSTREAM_FILES: dict[str, tuple[str, ...]] = {
    "ingest-corpus-labels": ("corpus/router_jobs.jsonl",),
    "emit-card-jobs": ("corpus/selection_manifest.jsonl", "corpus/rejected_manifest.jsonl"),
    "ingest-cards": ("cards/card_jobs.jsonl",),
    "build-landscape": ("results/paper_cards.jsonl", "results/card_job_outcomes.jsonl"),
    "emit-opportunity-jobs": ("landscape/assumptions.json", "landscape/failures.json", "landscape/mechanisms.json", "landscape/evaluations.json", "landscape/card_concept_edges.csv"),
    "ingest-opportunities": ("opportunities/mining_jobs.jsonl", "opportunities/cluster_audit.jsonl"),
    "quality-gate": ("opportunities/opportunity_candidates.jsonl", "opportunities/rejected_results.jsonl", "opportunities/result_outcomes.jsonl"),
    "internal-dedup": ("quality/ready_for_internal_dedup.jsonl", "quality/rejected.jsonl", "quality/opportunity_job_outcomes.jsonl"),
    "emit-recon-pack": ("ledger/ready_after_internal_dedup.jsonl", "ledger/dedup_outcomes.jsonl"),
    "ingest-recon": ("recon/query_pack.jsonl", "recon/query_pack_manifest.json", "recon/execution_job_templates.jsonl", "recon/execution_jobs.jsonl"),
    "emit-route-jobs": ("recon/reports.jsonl", "recon/outcomes.jsonl", "recon/ready_for_routes.jsonl"),
    "ingest-routes": ("routes/jobs.jsonl",),
    "emit-review-jobs": ("routes/results.jsonl", "routes/outcomes.jsonl", "routes/bundle_manifest.json"),
    "ingest-reviews": ("review/jobs.jsonl",),
    "ingest-human-scores": ("review/results.jsonl", "review/outcomes.jsonl", "review/bundle_manifest.json"),
    "finalize": ("ledger/human_scores.jsonl", "ledger/human_scoring_outcomes.jsonl", "ledger/human_bundle_manifest.json"),
}


def _effective_input_paths(run: Path, command: str, config: Any) -> dict[str, Path]:
    from .corpus import enumerate_candidates

    paths: dict[str, Path] = {}
    for index, path in enumerate(config.candidate_lists):
        paths[f"candidate_list_{index}"] = Path(path)
    paths["legacy_ledger"] = Path(config.legacy_ledger)
    if config.source_primary_manifest is not None:
        paths["primary_source_manifest"] = Path(config.source_primary_manifest)
    for candidate in enumerate_candidates(config):
        paths[f"note_{candidate.slug}"] = candidate.note_path
        if candidate.source_pdf_path is not None:
            paths[f"primary_pdf_{candidate.slug}"] = Path(candidate.source_pdf_path)
        if candidate.source_adjudication_binding is not None:
            paths[f"source_adjudication_{candidate.slug}"] = Path(candidate.source_adjudication_binding[0])
    for relpath in _UPSTREAM_FILES.get(command, ()):
        path = run / relpath
        if not path.is_file():
            raise PipelineError(f"{command} upstream artifact is missing: {relpath}")
        paths[f"upstream_{relpath}"] = path
    return paths


def init_run(run_dir: Path, config_path: Path, *, mode: str = "live", new_run: bool = False, config_repo_root: Path | None = None) -> Path:
    from .corpus import CorpusRouterConfig

    lexical = Path(os.path.abspath(run_dir))
    if lexical.exists() and any(lexical.iterdir()):
        if new_run:
            raise PipelineError("--new-run requires a new empty --run path; existing evidence is never overwritten")
        manifest_path = lexical / "manifest.json"
        state_path = lexical / "state.json"
        if manifest_path.is_file() and state_path.is_file():
            current = _strict_json(manifest_path)
            resolved_config = Path(config_path).resolve(strict=True)
            if current.get("config_path") == str(resolved_config) and current.get("config_sha256") == sha256_file(resolved_config) and current.get("mode") == mode:
                return lexical.resolve(strict=True)
        raise PipelineError("run path already contains different evidence; use --new-run with a new path")
    resolved_config = Path(config_path).resolve(strict=True)
    if config_repo_root is not None:
        root = Path(config_repo_root).resolve(strict=True)
        CorpusRouterConfig.from_json(resolved_config, repo_root=root)
    else:
        payload = _strict_json(resolved_config)
        expected_config_keys = {
            "notes_root", "candidate_lists", "legacy_ledger", "target_min",
            "target_max", "allowed_labels", "bridge_regression_slugs",
        }
        valid_config_keys = {
            frozenset(expected_config_keys),
            frozenset(expected_config_keys | {"protocol_version"}),
            frozenset(expected_config_keys | {"protocol_version", "source_primary_manifest"}),
        }
        if frozenset(payload) not in valid_config_keys or type(payload.get("candidate_lists")) is not list:
            raise PipelineError("corpus router config has missing or unexpected fields")
        path_values = (
            Path(str(payload["notes_root"])), Path(str(payload["legacy_ledger"])),
            *(Path(str(item)) for item in payload["candidate_lists"]),
        )
        roots = (resolved_config.parent, *resolved_config.parent.parents)
        root = next((
            candidate for candidate in roots
            if (candidate / path_values[0]).resolve().is_dir()
            and (candidate / path_values[1]).resolve().is_file()
            and all((candidate / value).resolve().is_file() for value in path_values[2:])
        ), None)
        if root is None:
            raise PipelineError("no coherent config base found; pass --config-repo-root")
        CorpusRouterConfig.from_json(resolved_config, repo_root=root)
    run = _anchored_run(lexical, create=True)
    run_id = stable_id("run", str(run), sha256_file(resolved_config), mode)
    manifest = {
        "schema_version": "idea_factory.run_manifest.v1", "run_id": run_id,
        "mode": mode, "config_path": str(resolved_config), "config_sha256": sha256_file(resolved_config),
        "config_repo_root": str(root), "created_at": datetime.now(timezone.utc).isoformat(),
        "truth_boundary": "OFFLINE_FIXTURE_NOT_LIVE_RECON" if mode == "offline-fixture" else "LIVE_RECON_REQUIRES_VALIDATED_RECEIPTS",
        "cheap_test_executed": False, "subsequent_result": None,
    }
    state = RunState(run_id=run_id)
    with stage_mutation_lock(run):
        publish_transaction({run / "manifest.json": encode_json(manifest), run / "state.json": encode_json(state.model_dump(mode="json"))})
    return run


def _execute_unlocked(command: str, run_dir: Path, config_path: Path | None, *, results: Path | None = None, prompt: Path | None = None, reviewed_assignments: Path | None = None, reviewed_local_entries: Path | None = None, allow_test_ready: bool = False, repository_root: Path | None = None, route_prompt: Path | None = None, review_prompt: Path | None = None, recon_context: Path | None = None, execution_bundle: Path | None = None) -> str:
    if command not in _TRANSITIONS:
        raise PipelineError(f"unsupported stage command: {command}")
    run = _anchored_run(run_dir)
    if reviewed_local_entries is not None and command != "emit-opportunity-jobs":
        raise PipelineError("--reviewed-local-entries is only valid for emit-opportunity-jobs")
    resolved_config_path, config = _config(run, config_path)
    expected, target = _TRANSITIONS[command]
    state = _load_state(run)
    prompt_key = {
        "emit-corpus-jobs": "corpus", "ingest-corpus-labels": "corpus",
        "emit-card-jobs": "card", "emit-opportunity-jobs": "opportunity",
    }.get(command)
    selected_prompt = Path(prompt).resolve(strict=True) if prompt is not None else (
        DEFAULT_PROMPTS["corpus_v3"] if prompt_key == "corpus" and config.protocol_version == "v3"
        else DEFAULT_PROMPTS[prompt_key] if prompt_key else None
    )
    route_prompt = Path(route_prompt).resolve(strict=True) if route_prompt else DEFAULT_PROMPTS["route"]
    review_prompt = Path(review_prompt).resolve(strict=True) if review_prompt else DEFAULT_PROMPTS["review"]
    paths: dict[str, Path | None] = {"results": results, "prompt": selected_prompt, "reviewed_assignments": reviewed_assignments, "recon_context": recon_context, "execution_bundle": execution_bundle}
    if reviewed_local_entries is not None:
        paths["reviewed_local_entries"] = reviewed_local_entries
    paths.update(_effective_input_paths(run, command, config))
    if command == "ingest-recon":
        for relpath in (
            "recon/execution_receipts.jsonl", "recon/normalized_manifest.json",
            "recon/report_jobs.jsonl",
        ):
            artifact = run / relpath
            if artifact.is_file():
                paths[f"upstream_{relpath}"] = artifact
    if command in {"emit-route-jobs", "ingest-routes", "emit-review-jobs", "ingest-reviews", "ingest-human-scores"}:
        paths["route_prompt"] = route_prompt
    if command in {"emit-review-jobs", "ingest-reviews", "ingest-human-scores"}:
        paths["review_prompt"] = review_prompt
    options = {"allow_test_ready": allow_test_ready, "repository_root": str(Path(repository_root).resolve()) if repository_root else None}
    digest = _command_digest(command, resolved_config_path, paths, options)
    key = f"command:{command}"
    if _idempotent_guard(run, state, key, digest, expected=expected, target=target):
        return "NO_OP"
    kwargs = {
        "config": config, "results": results, "prompt": selected_prompt,
        "reviewed_assignments": reviewed_assignments, "allow_test_ready": allow_test_ready,
        "reviewed_local_entries": reviewed_local_entries,
        "route_prompt": route_prompt, "review_prompt": review_prompt,
        "repository_root": Path(repository_root).resolve() if repository_root else REPOSITORY_ROOT,
        "recon_context": recon_context, "execution_bundle": execution_bundle,
    }
    completed = getattr(BACKEND, command.replace("-", "_"))(run, **kwargs)
    if completed is False:
        return "AWAITING_EXTERNAL_RESULT"
    if completed == "ZERO_SURVIVOR":
        _record_success(run, state, key, digest, None)
        return "COMPLETED_ZERO_SURVIVOR_HANDOFF"
    _record_success(run, state, key, digest, target)
    return "COMPLETED"


def execute(command: str, run_dir: Path, config_path: Path | None, *, results: Path | None = None, prompt: Path | None = None, reviewed_assignments: Path | None = None, reviewed_local_entries: Path | None = None, allow_test_ready: bool = False, repository_root: Path | None = None, route_prompt: Path | None = None, review_prompt: Path | None = None, recon_context: Path | None = None, execution_bundle: Path | None = None) -> str:
    """Execute one stage under the run-wide mutation lock."""

    run = _anchored_run(run_dir)
    with stage_mutation_lock(run):
        return _execute_unlocked(
            command, run, config_path, results=results, prompt=prompt,
            reviewed_assignments=reviewed_assignments,
            reviewed_local_entries=reviewed_local_entries,
            allow_test_ready=allow_test_ready, repository_root=repository_root,
            route_prompt=route_prompt, review_prompt=review_prompt,
            recon_context=recon_context, execution_bundle=execution_bundle,
        )


def _report_bytes(*, status: Mapping[str, Any]) -> bytes:
    text = (
        f"# Idea Factory run {status['run_id']}\n\n"
        f"Stage: {status['stage']}\n\n"
        f"Operational completion: {str(status['operational_completion']).lower()}\n\n"
        f"Idea production success: {str(status['idea_pack_produced']).lower()}\n\n"
        f"Live recon executed: {str(status['live_recon_executed']).lower()}\n\n"
        f"Human review completed: {str(status['human_review_completed']).lower()}\n\n"
        "Cheap test executed: false\n\n"
        "Subsequent result: none\n\n"
        "This report does not claim that any experiment or cheapest falsification test has run.\n"
    )
    return text.encode("utf-8")


_PERSISTENT_LEDGER_OUTPUT_KEY = "external-output:finalize:persistent-current-run"
_PERSISTENT_LEDGER_PATH_KEY = "external-output:finalize:persistent-path"
_FINALIZE_BINDING_VERSION_KEY = "external-input:finalize:binding-version"
_FINALIZE_BINDING_VERSION = "idea_factory.finalize_binding.v1"
_FINALIZE_ALLOW_TEST_READY_KEY = "external-input:finalize:option:allow-test-ready"
_FINALIZE_PROMPT_BINDING_KEYS = {
    "route": (
        "external-input:finalize:route-prompt-path",
        "external-input:finalize:route-prompt-sha256",
    ),
    "review": (
        "external-input:finalize:review-prompt-path",
        "external-input:finalize:review-prompt-sha256",
    ),
}


def _bind_finalization_prompt(
    state: RunState, *, kind: str, prompt: Path,
) -> None:
    path_key, hash_key = _FINALIZE_PROMPT_BINDING_KEYS[kind]
    if not prompt.is_file() or prompt.is_symlink() or prompt.resolve(strict=True) != prompt:
        raise PipelineError(f"finalize {kind} prompt is missing or unsafe")
    state.set_input_hash(path_key, str(prompt))
    state.set_input_hash(hash_key, sha256_file(prompt))


def _validate_finalization_replay(
    run: Path,
    state: RunState,
    config: Any,
    *,
    repository_root: Path,
    route_prompt: Path,
    review_prompt: Path,
    allow_test_ready: bool,
) -> None:
    _validate_output_snapshots(run, state, "finalize")
    persistent_digest = state.input_hashes.get(_PERSISTENT_LEDGER_OUTPUT_KEY)
    if persistent_digest is not None:
        persistent = repository_root / "data" / "idea_ledger.jsonl"
        if state.input_hashes.get(_PERSISTENT_LEDGER_PATH_KEY) != str(persistent):
            raise PipelineError("finalize persistent ledger path binding was modified")
        if not persistent.is_file() or persistent.is_symlink() or persistent.resolve(strict=True) != persistent:
            raise PipelineError("finalize persistent ledger is missing or unsafe")
        run_updates = run / "ledger" / "updates.jsonl"
        expected_binding = _canonical_hash({
            "persistent_path": str(persistent),
            "current_run_updates_sha256": sha256_file(run_updates),
        })
        if persistent_digest != expected_binding:
            raise PipelineError("finalize persistent ledger binding was modified; use --new-run")
        from .ledger import validate_ledger_updates
        from .render import validate_idea_pack_bundle

        validate_idea_pack_bundle(
            run, config, prompt_path=review_prompt, route_prompt_path=route_prompt,
            allow_test_ready=allow_test_ready,
        )
        validate_ledger_updates(
            run, config, repository_root=repository_root, prompt_path=review_prompt,
            route_prompt_path=route_prompt, allow_test_ready=allow_test_ready,
        )
    elif isinstance(BACKEND, RealBackend):
        raise PipelineError("finalize replay lacks a persistent ledger snapshot")
    report = run / "report.md"
    if report.read_bytes() != _report_bytes(status=status(run, _skip_complete_replay=True)):
        raise PipelineError("finalize report does not replay from current validated state")


def _finalize_unlocked(run_dir: Path, config_path: Path | None, *, allow_test_ready: bool = False, repository_root: Path | None = None, route_prompt: Path | None = None, review_prompt: Path | None = None) -> str:
    run = _anchored_run(run_dir)
    resolved_config_path, config = _config(run, config_path)
    state = _load_state(run)
    route_prompt = Path(route_prompt).resolve(strict=True) if route_prompt else DEFAULT_PROMPTS["route"]
    review_prompt = Path(review_prompt).resolve(strict=True) if review_prompt else DEFAULT_PROMPTS["review"]
    paths: dict[str, Path | None] = {"review_prompt": review_prompt, "route_prompt": route_prompt}
    paths.update(_effective_input_paths(run, "finalize", config))
    root = Path(repository_root).resolve() if repository_root else REPOSITORY_ROOT
    digest = _command_digest("finalize", resolved_config_path, paths, {"allow_test_ready": allow_test_ready, "repository_root": str(root)})
    key = "command:finalize"
    previous = state.input_hashes.get(key)
    if state.stage == RunStage.COMPLETE:
        if previous == digest:
            _validate_finalization_replay(
                run, state, config, repository_root=root, route_prompt=route_prompt,
                review_prompt=review_prompt, allow_test_ready=allow_test_ready,
            )
            return "NO_OP"
        raise PipelineError("input hashes changed; use --new-run and preserve this run")
    if state.stage not in {RunStage.REVIEWED, RunStage.HUMAN_SCORED}:
        raise PipelineError(
            f"finalize requires RunState REVIEWED or HUMAN_SCORED; current stage is {state.stage.value}"
        )
    if previous is not None:
        if previous != digest:
            raise PipelineError("input hashes changed; use --new-run and preserve this run")
        if (run / "report.md").is_file():
            _validate_finalization_replay(
                run, state, config, repository_root=root, route_prompt=route_prompt,
                review_prompt=review_prompt, allow_test_ready=allow_test_ready,
            )
            return "NO_OP"
    snapshot = _finalization_snapshot(run, root)
    try:
        outcome = BACKEND.finalize(
            run, config=config, review_prompt=review_prompt, route_prompt=route_prompt,
            allow_test_ready=allow_test_ready, repository_root=root,
        )
        ready_packs = list(outcome.get("ready_packs", []))
        state.set_input_hash(key, digest)
        persistent_output = outcome.get("persistent_ledger")
        if persistent_output is not None:
            persistent = Path(persistent_output)
            expected_persistent = root / "data" / "idea_ledger.jsonl"
            if persistent != expected_persistent or not persistent.is_file():
                raise PipelineError("finalize backend returned an invalid persistent ledger path")
            state.set_input_hash(_PERSISTENT_LEDGER_OUTPUT_KEY, _canonical_hash({
                "persistent_path": str(persistent),
                "current_run_updates_sha256": sha256_file(run / "ledger" / "updates.jsonl"),
            }))
            state.set_input_hash(_PERSISTENT_LEDGER_PATH_KEY, str(persistent))
        elif isinstance(BACKEND, RealBackend):
            raise PipelineError("real finalize did not return its persistent ledger output")
        state.set_input_hash(_FINALIZE_BINDING_VERSION_KEY, _FINALIZE_BINDING_VERSION)
        state.set_input_hash(
            _FINALIZE_ALLOW_TEST_READY_KEY, "true" if allow_test_ready else "false",
        )
        _bind_finalization_prompt(state, kind="route", prompt=route_prompt)
        _bind_finalization_prompt(state, kind="review", prompt=review_prompt)
        _record_output_snapshots(run, state, "finalize")
        predicted_stage = RunStage.COMPLETE if ready_packs or state.stage == RunStage.REVIEWED else state.stage
        current = status(run, _state_override=state, _skip_complete_replay=True)
        current = dict(current) | {
            "stage": predicted_stage.value,
            "completion_kind": (
                "IDEA_YIELD" if ready_packs else
                "AUDITABLE_ZERO_SURVIVOR" if state.stage == RunStage.REVIEWED else None
            ),
            "operational_completion": True,
            "idea_pack_produced": bool(outcome.get("ready_count", 0)),
        }
        report_bytes = _report_bytes(status=current)
        state.set_input_hash("output:finalize:report.md", hashlib.sha256(report_bytes).hexdigest())
        if ready_packs:
            if state.stage != RunStage.HUMAN_SCORED:
                raise PipelineError("ready Idea Packs require a completed human-scoring stage")
            state.finalize_idea_yield(ready_packs)
        elif state.stage == RunStage.REVIEWED:
            state.advance(RunStage.COMPLETE, zero_survivor=True)
        publish_transaction({
            run / "state.json": encode_json(state.model_dump(mode="json")),
            run / "report.md": report_bytes,
        })
    except BaseException as finalize_error:
        persistent_rollback_error: BaseException | None = None
        local_rollback_error: BaseException | None = None
        try:
            _compensate_persistent_ledger(run, root, snapshot)
        except BaseException as rollback_error:
            persistent_rollback_error = rollback_error
        try:
            _restore_finalization_local_snapshot(run, snapshot)
        except BaseException as rollback_error:
            local_rollback_error = rollback_error
        compensation_markers = []
        if persistent_rollback_error is not None:
            compensation_markers.append(_PERSISTENT_COMPENSATION_UNRESOLVED)
        if local_rollback_error is not None:
            compensation_markers.append(_LOCAL_COMPENSATION_UNRESOLVED)
        if compensation_markers:
            try:
                _record_compensation_unresolved(run, snapshot, compensation_markers)
            except BaseException as signal_error:
                raise RuntimeError(
                    "finalize compensation failed and durable recovery signalling failed; "
                    f"original={type(finalize_error).__name__}: {finalize_error}; "
                    f"persistent={type(persistent_rollback_error).__name__}: {persistent_rollback_error}; "
                    f"local={type(local_rollback_error).__name__}: {local_rollback_error}; "
                    f"signal={type(signal_error).__name__}: {signal_error}"
                ) from signal_error
            raise RuntimeError(
                "finalize compensation is unresolved; "
                f"original={type(finalize_error).__name__}: {finalize_error}; "
                f"persistent={type(persistent_rollback_error).__name__}: {persistent_rollback_error}; "
                f"local={type(local_rollback_error).__name__}: {local_rollback_error}"
            ) from (persistent_rollback_error or local_rollback_error)
        raise
    return "COMPLETED"


def finalize(run_dir: Path, config_path: Path | None, *, allow_test_ready: bool = False, repository_root: Path | None = None, route_prompt: Path | None = None, review_prompt: Path | None = None) -> str:
    """Finalize atomically with respect to all other run-level commands."""

    run = _anchored_run(run_dir)
    with stage_mutation_lock(run):
        return _finalize_unlocked(
            run, config_path, allow_test_ready=allow_test_ready,
            repository_root=repository_root, route_prompt=route_prompt,
            review_prompt=review_prompt,
        )


def _count_rows(path: Path) -> int:
    try:
        return len(read_jsonl(path)) if path.is_file() else 0
    except ValueError:
        return 0


def _count_unfinished_jobs(jobs_path: Path, outcomes_path: Path) -> int:
    jobs = read_jsonl(jobs_path) if jobs_path.is_file() else []
    if not jobs:
        return 0
    outcomes = read_jsonl(outcomes_path) if outcomes_path.is_file() else []
    completed = {
        str(row["job_id"])
        for row in outcomes
        if type(row.get("job_id")) is str and row.get("status") in {"ACCEPTED", "REJECTED"}
    }
    return sum(str(job.get("job_id", "")) not in completed for job in jobs)


def _repository_root_from_finalization_state(state: RunState) -> Path:
    raw = state.input_hashes.get(_PERSISTENT_LEDGER_PATH_KEY)
    if raw is None:
        raise PipelineError("finalize status lacks its persistent ledger path binding")
    persistent = Path(os.path.abspath(raw))
    if (
        persistent.name != "idea_ledger.jsonl" or persistent.parent.name != "data"
        or not persistent.is_file() or persistent.is_symlink()
        or persistent.resolve(strict=True) != persistent
    ):
        raise PipelineError("finalize status persistent ledger path is missing or unsafe")
    root = persistent.parent.parent
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise PipelineError("finalize status repository root is missing or unsafe")
    return root


def _finalization_prompt_paths(
    state: RunState,
) -> tuple[Path, Path]:
    bound_values = {
        kind: (
            state.input_hashes.get(path_key), state.input_hashes.get(hash_key),
        )
        for kind, (path_key, hash_key) in _FINALIZE_PROMPT_BINDING_KEYS.items()
    }
    present = [value is not None for pair in bound_values.values() for value in pair]
    if any(present) and not all(present):
        raise PipelineError("finalize prompt provenance binding is incomplete")
    if all(present):
        prompts: dict[str, Path] = {}
        for kind, (raw_path, expected_hash) in bound_values.items():
            assert raw_path is not None and expected_hash is not None
            prompt = Path(os.path.abspath(raw_path))
            if (
                str(prompt) != raw_path or not prompt.is_file() or prompt.is_symlink()
                or prompt.resolve(strict=True) != prompt
            ):
                raise PipelineError(f"finalize bound {kind} prompt is missing or unsafe")
            if sha256_file(prompt) != expected_hash:
                raise PipelineError(f"finalize bound {kind} prompt was modified")
            prompts[kind] = prompt
        route_prompt, review_prompt = prompts["route"], prompts["review"]
    else:
        route_prompt, review_prompt = DEFAULT_PROMPTS["route"], DEFAULT_PROMPTS["review"]
    return route_prompt, review_prompt


def _finalization_allow_test_ready(
    run: Path, state: RunState, config_path: Path, config: Any,
    *, repository_root: Path, route_prompt: Path, review_prompt: Path,
) -> bool:
    version = state.input_hashes.get(_FINALIZE_BINDING_VERSION_KEY)
    raw_option = state.input_hashes.get(_FINALIZE_ALLOW_TEST_READY_KEY)
    if version is None and raw_option is None:
        candidates = (False, True)
    else:
        if version != _FINALIZE_BINDING_VERSION or raw_option not in {"false", "true"}:
            raise PipelineError("finalize option provenance binding is incomplete or invalid")
        candidates = (raw_option == "true",)
    paths: dict[str, Path | None] = {
        "review_prompt": review_prompt, "route_prompt": route_prompt,
    }
    paths.update(_effective_input_paths(run, "finalize", config))
    matches = [
        candidate for candidate in candidates
        if state.input_hashes.get("command:finalize") == _command_digest(
            "finalize", config_path, paths,
            {"allow_test_ready": candidate, "repository_root": str(repository_root)},
        )
    ]
    if len(matches) != 1:
        if version is None and raw_option is None:
            raise PipelineError("legacy finalize option provenance is ambiguous or unrecoverable")
        raise PipelineError("finalize option or prompt binding does not match command provenance")
    return matches[0]


def _validated_pack_counts(
    run: Path, state: RunState, *, skip_complete_replay: bool,
) -> tuple[int, int, int, bool]:
    packs = run / "idea_packs"
    try:
        has_pack_artifacts = packs.is_dir() and any(
            path.name != ".stage.lock" for path in packs.iterdir()
        )
    except OSError:
        has_pack_artifacts = True
    finalized = "command:finalize" in state.input_hashes
    eligible_stage = state.stage in {
        RunStage.REVIEWED, RunStage.HUMAN_SCORED, RunStage.COMPLETE,
    }
    if not has_pack_artifacts and not finalized:
        return 0, 0, 0, False
    if not finalized or not eligible_stage:
        return 0, 0, 0, True
    try:
        _validate_output_snapshots(run, state, "finalize")
        config_path, config = _config(run, None)
        repository_root = _repository_root_from_finalization_state(state)
        route_prompt, review_prompt = _finalization_prompt_paths(state)
        allow_test_ready = _finalization_allow_test_ready(
            run, state, config_path, config, repository_root=repository_root,
            route_prompt=route_prompt, review_prompt=review_prompt,
        )
        from .render import validate_idea_pack_bundle

        validated = validate_idea_pack_bundle(
            run, config, prompt_path=review_prompt,
            route_prompt_path=route_prompt, allow_test_ready=allow_test_ready,
        )
        if state.stage == RunStage.COMPLETE and not skip_complete_replay:
            _validate_finalization_replay(
                run, state, config, repository_root=repository_root,
                route_prompt=route_prompt, review_prompt=review_prompt,
                allow_test_ready=allow_test_ready,
            )
    except RuntimeError as exc:
        if not is_expected_operational_runtime_error(exc):
            raise
        return 0, 0, 0, True
    except (OSError, ValueError, PipelineError):
        return 0, 0, 0, True
    return (
        int(validated["pack_count"]), int(validated["ready_count"]),
        int(validated["hold_count"]), False,
    )


def status(
    run_dir: Path, *, _skip_complete_replay: bool = False,
    _state_override: RunState | None = None,
) -> dict[str, Any]:
    run = _anchored_run(run_dir)
    persisted_state = _load_state(run)
    state = _state_override or persisted_state
    if state.run_id != persisted_state.run_id:
        raise PipelineError("status state override does not belong to this run")
    manifest = _manifest(run)
    pack_count, ready_count, hold_count, pack_invalid = _validated_pack_counts(
        run, state, skip_complete_replay=_skip_complete_replay,
    )
    live_rows = read_jsonl(run / "recon" / "ready_for_routes.jsonl") if (run / "recon" / "ready_for_routes.jsonl").is_file() else []
    live_ready = bool(live_rows) and all(row.get("live_ready") is True for row in live_rows)
    live_recon = False
    receipts = run / "recon" / "execution_receipts.jsonl"
    if manifest["mode"] == "live" and receipts.is_file() and _stage_index(state.stage) >= _stage_index(RunStage.RECON_READY):
        try:
            from .recon import validate_execution_receipts
            _config_path, validated_config = _config(run, None)
            live_recon = bool(validate_execution_receipts(run, validated_config))
        except (OSError, ValueError):
            live_recon = False
    human_complete = (run / "ledger" / "human_bundle_manifest.json").is_file() or "command:ingest-human-scores" in state.input_hashes
    waits = 0
    for jobs, result in (
        ("corpus/router_jobs.jsonl", "corpus/selection_manifest.jsonl"),
        ("cards/card_jobs.jsonl", "results/paper_cards.jsonl"),
        ("opportunities/mining_jobs.jsonl", "opportunities/opportunity_candidates.jsonl"),
        ("ledger/shape_assignment_jobs.jsonl", "ledger/ready_after_internal_dedup.jsonl"),
        ("recon/execution_job_templates.jsonl", "recon/execution_jobs.jsonl"),
        ("recon/execution_jobs.jsonl", "recon/execution_receipts.jsonl"),
        ("recon/report_jobs.jsonl", "recon/reports.jsonl"),
        ("routes/jobs.jsonl", "routes/results.jsonl"),
        ("ledger/human_scoring_jobs.jsonl", "ledger/human_scores.jsonl"),
    ):
        if (run / jobs).is_file() and not (run / result).is_file():
            waits += _count_rows(run / jobs)
    waits += _count_unfinished_jobs(run / "review" / "jobs.jsonl", run / "review" / "outcomes.jsonl")
    waits += _count_unfinished_jobs(run / "review" / "round2_jobs.jsonl", run / "review" / "outcomes.jsonl")
    killed = sum(_count_rows(run / path) for path in (
        "quality/rejected.jsonl", "ledger/blocked.jsonl", "recon/killed.jsonl",
    ))
    if (run / "review" / "results.jsonl").is_file():
        killed += sum(row.get("decision") == "KILL" for row in read_jsonl(run / "review" / "results.jsonl"))
    unresolved = len(state.unresolved_errors) + int(pack_invalid)
    for relpath, bad_statuses in (
        ("results/card_job_outcomes.jsonl", {"INVALID"}),
        ("opportunities/result_outcomes.jsonl", {"INVALID"}),
        ("recon/execution_receipts.jsonl", {"ERROR", "CREDENTIALS_PROTOCOL_REQUIRED"}),
        ("recon/outcomes.jsonl", {"REJECTED"}),
        ("routes/outcomes.jsonl", {"REJECTED"}),
        ("review/outcomes.jsonl", {"REJECTED"}),
        ("ledger/human_scoring_outcomes.jsonl", {"REJECTED"}),
    ):
        artifact = run / relpath
        if artifact.is_file():
            unresolved += sum(row.get("status") in bad_statuses for row in read_jsonl(artifact))
    result = {
        "schema_version": "idea_factory.status.v1", "run_id": state.run_id,
        "stage": state.stage.value, "completion_kind": state.completion_kind.value if state.completion_kind else None,
        "counts": {
            "completed": int(state.stage == RunStage.COMPLETE or "command:finalize" in state.input_hashes), "awaiting-external-result": waits,
            "killed": killed, "surviving": ready_count, "unresolved-error": unresolved,
        },
        "offline_tests_ready": True,
        "live_recon_executed": live_recon,
        "human_review_completed": human_complete,
        "idea_pack_artifact_count": pack_count,
        "hold_count": hold_count,
        "idea_pack_produced": ready_count > 0,
        "real_idea_pack_produced": ready_count > 0 and live_recon and live_ready,
        "operational_completion": state.stage == RunStage.COMPLETE or "command:finalize" in state.input_hashes,
        "cheap_test_executed": False,
        "subsequent_result": None,
        "truth_boundary": manifest["truth_boundary"],
    }
    return result
