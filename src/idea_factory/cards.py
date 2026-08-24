"""Bounded, evidence-linked paper-card job emission and result ingestion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError

from .artifacts import read_jsonl, sha256_file, stable_id, write_jsonl, write_jsonl_bundle
from .corpus import validate_corpus_selection_bundle
from .models import PaperCard, Sha256Hex, StrictModel


CARD_PROMPT_VERSION = "idea_factory.paper_card_prompt.v1"
CARD_JOB_SCHEMA_VERSION = "idea_factory.paper_card_job.v1"
CARD_RESULT_SCHEMA_VERSION = "idea_factory.paper_card_result.v1"
PAPER_CARD_SCHEMA_VERSION = "idea_factory.paper_card.v1"
CARD_OUTCOME_SCHEMA_VERSION = "idea_factory.card_job_outcome.v2"
ACCEPTED_CARD_SCHEMA_VERSION = "idea_factory.accepted_paper_card.v2"
REJECTED_CARD_SCHEMA_VERSION = "idea_factory.rejected_paper_card.v1"

SUPPORT_TOKENS = frozenset(
    {
        "problem", "assumption", "mechanism", "failure_observation",
        "failure_mechanism", "limitation", "evaluation.measurement",
        "evaluation.regime", "scope.object", "scope.time_horizon", "scope.setting",
    }
)

REQUIRED_CARD_SCHEMA: dict[str, object] = {
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "job_id", "slug", "note_sha256", "prompt_sha256", "cards"],
    "properties": {
        "schema_version": {"const": CARD_RESULT_SCHEMA_VERSION},
        "job_id": {"type": "string", "minLength": 1},
        "slug": {"type": "string", "minLength": 1},
        "note_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "prompt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "cards": {"type": "array"},
    },
}


class CardResultWrapper(StrictModel):
    schema_version: str
    job_id: str
    slug: str
    note_sha256: Sha256Hex
    prompt_sha256: Sha256Hex
    cards: list[dict[str, Any]]


def card_job_id(slug: str, note_sha256: str, prompt_sha256: str) -> str:
    """Bind a paper-card job to source, prompt, and exact result schema."""

    return stable_id("paper_card_job", slug, note_sha256, prompt_sha256, CARD_RESULT_SCHEMA_VERSION)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _canonical_hash(value: object) -> str:
    return _hash_bytes(_canonical_json(value).encode("utf-8"))


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _unsafe_note_path(path: Path, legacy_ledger: Path) -> bool:
    lower_name = path.name.lower()
    return (
        lower_name.endswith(".audit.md")
        or lower_name.endswith(".atom.md")
        or path == legacy_ledger
        or any(part.lower() == "_atoms" for part in path.parts)
    )


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current.resolve(strict=True)


def _prepare_run_directory(run: Path, name: str) -> Path:
    expected = run / name
    ancestor = _nearest_existing(expected)
    if not _within(ancestor, run):
        raise ValueError(f"{name} directory escapes the same active run")
    expected.mkdir(parents=False, exist_ok=True)
    resolved = expected.resolve(strict=True)
    if str(resolved) != str(expected):
        raise ValueError(f"{name} directory escapes the same active run")
    return expected


def _validate_optional_run_directory(run: Path, name: str) -> Path:
    expected = run / name
    ancestor = _nearest_existing(expected)
    if not _within(ancestor, run):
        raise ValueError(f"{name} directory escapes the same active run")
    if expected.exists() and str(expected.resolve(strict=True)) != str(expected):
        raise ValueError(f"{name} directory escapes the same active run")
    return expected


def _owned_paths(selection_manifest: Path, output_path: Path) -> dict[str, Path]:
    selection = Path(selection_manifest).resolve(strict=True)
    if selection.name != "selection_manifest.jsonl" or selection.parent.name != "corpus":
        raise ValueError("selection manifest must be the active run corpus selection_manifest.jsonl")
    run = selection.parent.parent
    output = _absolute_lexical(Path(output_path))
    expected = run / "cards" / "card_jobs.jsonl"
    if str(output) != str(expected):
        raise ValueError("card jobs output must use exact active run cards/card_jobs.jsonl")
    cards_dir = _prepare_run_directory(run, "cards")
    results_dir = _validate_optional_run_directory(run, "results")
    return {
        "run": run,
        "jobs": cards_dir / "card_jobs.jsonl",
        "accepted": results_dir / "paper_cards.jsonl",
        "rejected": results_dir / "rejected_paper_cards.jsonl",
        "outcomes": results_dir / "card_job_outcomes.jsonl",
    }


def _invalidate(paths: dict[str, Path], *names: str) -> None:
    for name in names:
        paths[name].unlink(missing_ok=True)


def _read_utf8(path: Path, message: str) -> tuple[bytes, str]:
    try:
        content = path.read_bytes()
        return content, content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(message) from exc


def _selection_records(selection_manifest: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection = Path(selection_manifest).resolve(strict=True)
    sibling = selection.with_name("rejected_manifest.jsonl")
    if not sibling.is_file():
        raise ValueError("complete selection bundle requires selection_manifest and sibling rejected_manifest")
    try:
        selected, rejected = validate_corpus_selection_bundle(selection)
    except (OSError, ValueError) as exc:
        raise ValueError(f"complete selection bundle is invalid: {exc}") from exc
    required = {
        "slug", "note_path", "note_sha256", "source_lists", "job_id",
        "result_schema_version", "result_note_sha256", "prompt_sha256", "label",
        "confidence", "raw_result_sha256", "reason_code", "allowed_note_root",
        "allowed_papers_root", "legacy_ledger_path",
    }
    all_rows = [*selected, *rejected]
    if any(set(row) < required for row in all_rows):
        raise ValueError("complete selection bundle has missing required bindings")
    slugs = [row["slug"] for row in all_rows]
    if any(type(slug) is not str or not slug for slug in slugs) or len(set(slugs)) != len(slugs):
        raise ValueError("complete selection bundle has duplicate or invalid slugs")
    if any(row["reason_code"] != "ELIGIBLE" for row in selected):
        raise ValueError("selected records must all be eligible")
    if any(row["reason_code"] == "ELIGIBLE" for row in rejected):
        raise ValueError("selected and rejected records overlap")
    if not selected:
        raise ValueError("complete selection bundle has no selected records")
    for key in ("allowed_note_root", "allowed_papers_root", "legacy_ledger_path"):
        values = [row[key] for row in all_rows]
        if any(type(value) is not str or not value.strip() for value in values):
            raise ValueError("complete selection bundle has invalid root provenance")
        canonical = str(Path(values[0]).resolve())
        expected_kind = Path(canonical).is_file() if key == "legacy_ledger_path" else Path(canonical).is_dir()
        if any(value != canonical for value in values) or not expected_kind:
            raise ValueError("complete selection bundle has mixed or noncanonical root provenance")
    return selected, rejected


def _validated_note(row: dict[str, Any]) -> tuple[Path, str]:
    values = (row["note_path"], row["note_sha256"], row["allowed_note_root"], row["allowed_papers_root"], row["legacy_ledger_path"])
    if any(type(value) is not str or not value.strip() for value in values):
        raise ValueError("selected record has invalid note bindings")
    raw = Path(row["note_path"])
    resolved = raw.resolve()
    roots = (Path(row["allowed_note_root"]).resolve(), Path(row["allowed_papers_root"]).resolve())
    legacy = Path(row["legacy_ledger_path"]).resolve()
    if str(resolved) != row["note_path"] or _unsafe_note_path(resolved, legacy) or not any(_within(resolved, root) for root in roots):
        raise ValueError(f"selected note is audit, atom, legacy, or outside provenance roots: {raw}")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"selected note is not a regular source file: {raw}")
    content, text = _read_utf8(resolved, f"selected note must be readable UTF-8: {resolved}")
    if _hash_bytes(content) != row["note_sha256"]:
        raise ValueError(f"selected note changed after selection: {row['slug']}")
    return resolved, text


def emit_card_jobs(selection_manifest: Path, prompt_path: Path, output_path: Path) -> Path:
    """Emit one fully bound card extraction job per selected paper.

    Regeneration invalidates every downstream result member before input reads;
    job replacement is atomic, while card result publication is a separate bundle.
    """

    paths = _owned_paths(selection_manifest, output_path)
    _invalidate(paths, "accepted", "rejected", "outcomes")
    try:
        selected, _ = _selection_records(selection_manifest)
        selection = Path(selection_manifest).resolve(strict=True)
        rejected_manifest = selection.with_name("rejected_manifest.jsonl")
        selection_sha = sha256_file(selection)
        rejected_sha = sha256_file(rejected_manifest)
        prompt = Path(prompt_path).resolve(strict=True)
        prompt_bytes, prompt_text = _read_utf8(prompt, "paper-card prompt must be readable UTF-8")
        prompt_sha = _hash_bytes(prompt_bytes)
        jobs: list[dict[str, Any]] = []
        for row in sorted(selected, key=lambda item: item["slug"]):
            note, note_text = _validated_note(row)
            job_id = card_job_id(row["slug"], row["note_sha256"], prompt_sha)
            jobs.append({
                "record_id": job_id, "schema_version": CARD_JOB_SCHEMA_VERSION, "job_id": job_id,
                "slug": row["slug"], "note_path": str(note), "note_sha256": row["note_sha256"],
                "prompt_id": "paper_card", "prompt_version": CARD_PROMPT_VERSION,
                "prompt_path": str(prompt), "prompt_sha256": prompt_sha, "prompt_text": prompt_text,
                "note_text": note_text, "result_schema_version": CARD_RESULT_SCHEMA_VERSION,
                "selection_manifest_sha256": selection_sha,
                "rejected_manifest_sha256": rejected_sha,
                "required_output_schema": REQUIRED_CARD_SCHEMA,
            })
        write_jsonl(paths["jobs"], jobs)
        return paths["jobs"]
    except BaseException:
        _invalidate(paths, "jobs", "accepted", "rejected", "outcomes")
        raise


def _job_rows(jobs_path: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    jobs = Path(jobs_path).resolve(strict=True)
    if jobs.name != "card_jobs.jsonl" or jobs.parent.name != "cards":
        raise ValueError("card jobs must use exact active run cards/card_jobs.jsonl")
    run = jobs.parent.parent
    if str(jobs) != str(run / "cards" / "card_jobs.jsonl"):
        raise ValueError("card jobs must use exact active run cards/card_jobs.jsonl")
    selection = run / "corpus" / "selection_manifest.jsonl"
    rejected_manifest = run / "corpus" / "rejected_manifest.jsonl"
    try:
        selected, _ = _selection_records(selection)
    except (OSError, ValueError) as exc:
        raise ValueError(f"card jobs require a complete active-run selection bundle: {exc}") from exc
    paths = {
        "run": run, "jobs": jobs,
        "accepted": run / "results" / "paper_cards.jsonl",
        "rejected": run / "results" / "rejected_paper_cards.jsonl",
        "outcomes": run / "results" / "card_job_outcomes.jsonl",
    }
    rows = read_jsonl(jobs)
    required = {
        "record_id", "schema_version", "job_id", "slug", "note_path",
        "note_sha256", "note_text", "prompt_id", "prompt_version",
        "prompt_path", "prompt_sha256", "prompt_text", "result_schema_version",
        "selection_manifest_sha256", "rejected_manifest_sha256",
        "required_output_schema",
    }
    if not rows or any(set(row) < required or row["schema_version"] != CARD_JOB_SCHEMA_VERSION or row["result_schema_version"] != CARD_RESULT_SCHEMA_VERSION for row in rows):
        raise ValueError("invalid card jobs artifact")
    if len({row["job_id"] for row in rows}) != len(rows) or len({row["slug"] for row in rows}) != len(rows):
        raise ValueError("card jobs must have unique job IDs and slugs")
    selected_by_slug = {row["slug"]: row for row in selected}
    if {row["slug"] for row in rows} != set(selected_by_slug):
        raise ValueError("card job set does not exactly match selection")
    selection_sha = sha256_file(selection)
    rejected_sha = sha256_file(rejected_manifest)
    for job in rows:
        selected_row = selected_by_slug[job["slug"]]
        note, note_text = _validated_note(selected_row)
        try:
            prompt = Path(job["prompt_path"]).resolve(strict=True)
        except (OSError, TypeError) as exc:
            raise ValueError(f"card job prompt is missing: {job['slug']}") from exc
        prompt_bytes, prompt_text = _read_utf8(prompt, f"card job prompt must be readable UTF-8: {job['slug']}")
        prompt_sha = _hash_bytes(prompt_bytes)
        if prompt_sha != job["prompt_sha256"]:
            raise ValueError(f"card job prompt hash mismatch for {job['slug']}")
        expected_job_id = card_job_id(job["slug"], selected_row["note_sha256"], prompt_sha)
        expected = {
            "record_id": expected_job_id,
            "job_id": expected_job_id,
            "note_path": str(note),
            "note_sha256": selected_row["note_sha256"],
            "note_text": note_text,
            "prompt_id": "paper_card",
            "prompt_version": CARD_PROMPT_VERSION,
            "prompt_path": str(prompt),
            "prompt_sha256": prompt_sha,
            "prompt_text": prompt_text,
            "selection_manifest_sha256": selection_sha,
            "rejected_manifest_sha256": rejected_sha,
            "required_output_schema": REQUIRED_CARD_SCHEMA,
        }
        mismatch = next((key for key, value in expected.items() if job[key] != value), None)
        if mismatch is not None:
            raise ValueError(f"card job binding mismatch for {job['slug']}: {mismatch}")
    return paths, sorted(rows, key=lambda row: row["slug"])


def _tokens(value: str) -> set[str]:
    return {token.strip() for token in value.split(",") if token.strip()}


def _is_strict_json_value(value: object) -> bool:
    if value is None or type(value) in {str, bool, int}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_is_strict_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _is_strict_json_value(item) for key, item in value.items())
    return False


def _owned_json_snapshot(value: object, *, path: str = "$") -> object:
    """Copy one exact native-JSON tree in a single traversal of caller data."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"external result contains a non-finite number at {path}")
        return value
    if type(value) is list:
        return [
            _owned_json_snapshot(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        snapshot: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"external result contains a non-string object key at {path}")
            snapshot[key] = _owned_json_snapshot(item, path=f"{path}.{key}")
        return snapshot
    raise TypeError(
        f"external result must contain only exact native JSON containers and types; invalid value at {path}"
    )


def _snapshot_result(raw: object) -> tuple[dict[str, Any], str]:
    if type(raw) is not dict:
        raise TypeError("each external result must be an exact native dict JSON object")
    snapshot = _owned_json_snapshot(raw)
    if type(snapshot) is not dict:  # pragma: no cover - guarded above
        raise TypeError("external result snapshot must be a JSON object")
    encoded = _canonical_json(snapshot)
    return snapshot, _hash_bytes(encoded.encode("utf-8"))


def _same_json_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_json_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _validate_card(card_data: object, job: dict[str, Any]) -> PaperCard:
    if type(card_data) is not dict or not _is_strict_json_value(card_data):
        raise ValueError("PaperCard input must be a strict JSON object without tuple coercions")
    raw_card_json = _canonical_json(card_data)
    try:
        card = PaperCard.model_validate_json(raw_card_json, strict=True)
    except ValidationError as exc:
        raise ValueError(f"invalid PaperCard: {exc.errors()[0]['loc']}") from exc
    raw_card_value = json.loads(raw_card_json)
    if not _same_json_value(raw_card_value, card.model_dump(mode="json")):
        raise ValueError("PaperCard validation normalizes or coerces the raw card")
    if not card.card_id.strip():
        raise ValueError("card_id must be nonblank")
    if card.paper.source_path != job["note_path"]:
        raise ValueError("paper.source_path must canonically equal job note_path")
    if not card.paper.title.strip():
        raise ValueError("paper.title must be nonblank")
    if not card.paper.venue.strip():
        raise ValueError("paper.venue must be nonblank")
    if card.paper.year <= 0:
        raise ValueError("paper.year must be positive")
    coherent_fields = {
        "problem": card.problem,
        "assumption": card.assumption.text,
        "mechanism": card.mechanism,
        "failure_observation": card.failure_observation.text,
        "evaluation.measurement": card.evaluation.measurement,
        "evaluation.regime": card.evaluation.regime,
        "scope.object": card.scope.object,
        "scope.time_horizon": card.scope.time_horizon,
        "scope.setting": card.scope.setting,
    }
    blank = [name for name, value in coherent_fields.items() if not value.strip()]
    if blank:
        raise ValueError(
            "incoherent PaperCard: blank required fields: "
            + ", ".join(blank)
            + "; return an empty cards list instead"
        )
    failure_text = card.failure_mechanism.text.strip()
    failure_status = card.failure_mechanism.status.value
    if failure_status == "UNKNOWN" and failure_text:
        raise ValueError("failure_mechanism UNKNOWN must have empty text")
    if failure_status != "UNKNOWN" and not failure_text:
        raise ValueError("empty failure_mechanism requires UNKNOWN status")
    if not card.evidence_pointers:
        raise ValueError("coherent PaperCard requires at least one evidence pointer")
    supports: set[str] = set()
    prefix = job["note_path"] + "#"
    for pointer in card.evidence_pointers:
        if not pointer.locator.startswith(prefix) or not pointer.locator[len(prefix):].strip():
            raise ValueError("evidence locator must use canonical-note-path#nonblank-locator")
        pointer_tokens = _tokens(pointer.supports)
        if not pointer_tokens <= SUPPORT_TOKENS:
            raise ValueError("evidence supports contains unknown token")
        supports.update(pointer_tokens)
    requirements = {
        "problem": card.problem,
        "assumption": card.assumption.text,
        "mechanism": card.mechanism,
        "failure_observation": card.failure_observation.text,
        "failure_mechanism": card.failure_mechanism.text,
        "limitation": card.limitation,
        "evaluation.measurement": card.evaluation.measurement,
        "evaluation.regime": card.evaluation.regime,
        "scope.object": card.scope.object,
        "scope.time_horizon": card.scope.time_horizon,
        "scope.setting": card.scope.setting,
    }
    missing = [token for token, text in requirements.items() if text.strip() and token not in supports]
    if missing:
        raise ValueError("missing evidence supports: " + ", ".join(missing))
    return card


def _validate_result_run_anchor(
    jobs_path: Path,
    expected_run_dir: Path | None,
    *,
    require_result_bundle: bool = True,
) -> tuple[Path, Path]:
    jobs = _absolute_lexical(Path(jobs_path))
    run = (
        _absolute_lexical(Path(expected_run_dir))
        if expected_run_dir is not None
        else jobs.parent.parent
    )
    if jobs != run / "cards" / "card_jobs.jsonl":
        raise ValueError(
            "card jobs must use exact active run cards/card_jobs.jsonl and be anchored to the expected run"
        )
    try:
        resolved_run = run.resolve(strict=True)
    except OSError as exc:
        raise ValueError("expected run directory is missing") from exc
    if not run.is_dir() or resolved_run != run:
        raise ValueError("expected run directory must have exact resolved identity")

    directory_names = ["cards", "corpus"]
    if require_result_bundle or (run / "results").exists():
        directory_names.append("results")
    directories = {name: run / name for name in directory_names}
    for name, directory in directories.items():
        try:
            resolved = directory.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{name} directory is missing from the expected run") from exc
        if not directory.is_dir() or resolved != resolved_run / name:
            raise ValueError(
                f"{name} directory is not anchored to the expected run and escapes the same active run"
            )
        for child in directory.iterdir():
            try:
                resolved_child = child.resolve(strict=True)
            except OSError as exc:
                raise ValueError(f"{name} contains an unresolved artifact: {child.name}") from exc
            if not resolved_child.is_relative_to(resolved):
                raise ValueError(f"{name} artifact is not anchored to the expected run: {child.name}")

    anchored_files = [
        jobs,
        *(run / "corpus" / name for name in (
            "selection_manifest.jsonl",
            "rejected_manifest.jsonl",
            "router_jobs.jsonl",
            "router_results.jsonl",
            "selection_policy.jsonl",
        )),
    ]
    if require_result_bundle:
        anchored_files.extend(
            run / "results" / name
            for name in (
                "paper_cards.jsonl",
                "rejected_paper_cards.jsonl",
                "card_job_outcomes.jsonl",
            )
        )
    for path in anchored_files:
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"required artifact is missing from the expected run: {path.name}") from exc
        expected = resolved_run / path.relative_to(run)
        if not path.is_file() or resolved != expected:
            raise ValueError(f"artifact is not anchored to the expected run: {path.name}")
    return jobs, resolved_run


def validate_card_result_bundle(
    jobs_path: Path,
    expected_run_dir: Path | None = None,
) -> tuple[dict[str, Path], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Revalidate the complete durable card-result bundle for read-only consumers.

    This deliberately replays the active corpus/job chain through ``_job_rows``
    rather than treating ``paper_cards.jsonl`` as an authority.  Invalid jobs
    are not a usable landscape input: they leave the accepted set incomplete.
    """

    anchored_jobs, anchored_run = _validate_result_run_anchor(
        jobs_path,
        expected_run_dir,
    )
    paths, jobs = _job_rows(anchored_jobs)
    if paths["run"] != anchored_run:
        raise ValueError("card result bundle resolved to a different active run")
    _validate_optional_run_directory(paths["run"], "results")
    for name in ("accepted", "rejected", "outcomes"):
        if not paths[name].is_file():
            raise ValueError("complete card result bundle requires all three result files")
    accepted = read_jsonl(paths["accepted"])
    rejected = read_jsonl(paths["rejected"])
    outcomes = read_jsonl(paths["outcomes"])
    by_job = {job["job_id"]: job for job in jobs}
    required_outcome = {
        "schema_version", "job_id", "slug", "note_sha256", "prompt_sha256",
        "result_schema_version", "raw_result_sha256", "status", "accepted_card_ids",
        "rejected_card_ids", "accepted_count", "rejected_count",
        "accepted_cards_sha256", "rejected_projection_sha256",
        "raw_wrapper_reconstructable",
    }
    if len(outcomes) != len(jobs) or any(set(row) < required_outcome for row in outcomes):
        raise ValueError("card result bundle has incomplete job outcomes")
    if len({row.get("job_id") for row in outcomes}) != len(outcomes):
        raise ValueError("card result bundle has duplicate job outcomes")
    if {row.get("job_id") for row in outcomes} != set(by_job):
        raise ValueError("card result bundle outcomes do not exactly cover card jobs")

    outcome_by_job: dict[str, dict[str, Any]] = {}
    accepted_ids_from_outcomes: set[str] = set()
    for row in outcomes:
        job_id = row["job_id"]
        job = by_job[job_id]
        expected = {
            "schema_version": CARD_OUTCOME_SCHEMA_VERSION,
            "job_id": job_id,
            "slug": job["slug"],
            "note_sha256": job["note_sha256"],
            "prompt_sha256": job["prompt_sha256"],
            "result_schema_version": CARD_RESULT_SCHEMA_VERSION,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("card result bundle outcome binding mismatch")
        status = row["status"]
        accepted_ids = row["accepted_card_ids"]
        rejected_ids = row["rejected_card_ids"]
        if (
            status not in {"ACCEPTED", "PARTIAL_VALID", "VALID_EMPTY", "INVALID"}
            or not isinstance(accepted_ids, list)
            or not isinstance(rejected_ids, list)
            or any(not isinstance(value, str) or not value for value in accepted_ids + rejected_ids)
            or len(set(accepted_ids)) != len(accepted_ids)
            or len(set(rejected_ids)) != len(rejected_ids)
            or type(row["accepted_count"]) is not int
            or row["accepted_count"] != len(accepted_ids)
            or type(row["rejected_count"]) is not int
            or row["rejected_count"] < len(rejected_ids)
            or not isinstance(row["raw_result_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["raw_result_sha256"])
            or not isinstance(row["accepted_cards_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["accepted_cards_sha256"])
            or not isinstance(row["rejected_projection_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["rejected_projection_sha256"])
            or type(row["raw_wrapper_reconstructable"]) is not bool
        ):
            raise ValueError("card result bundle has invalid outcome counts")
        if status == "VALID_EMPTY" and (accepted_ids or rejected_ids or row["rejected_count"]):
            raise ValueError("VALID_EMPTY outcome must have no cards")
        if status == "ACCEPTED" and (not accepted_ids or rejected_ids or row["rejected_count"]):
            raise ValueError("ACCEPTED outcome has inconsistent card coverage")
        if status == "PARTIAL_VALID" and (not accepted_ids or row["rejected_count"] < 1):
            raise ValueError("PARTIAL_VALID outcome has inconsistent card coverage")
        if status == "INVALID" and (accepted_ids or row["rejected_count"] < 1):
            raise ValueError("INVALID outcome has inconsistent card coverage")
        if row["raw_wrapper_reconstructable"] != (status in {"ACCEPTED", "VALID_EMPTY"}):
            raise ValueError("card result bundle misstates raw wrapper reconstructability")
        if accepted_ids_from_outcomes.intersection(accepted_ids):
            raise ValueError("card result bundle has duplicate accepted card IDs")
        accepted_ids_from_outcomes.update(accepted_ids)
        outcome_by_job[job_id] = row

    required_accepted = {
        "schema_version", "record_id", "job_id", "slug", "index", "note_path",
        "note_sha256", "prompt_sha256", "result_schema_version", "raw_result_sha256", "card",
    }
    if any(set(row) < required_accepted for row in accepted):
        raise ValueError("card result bundle has malformed accepted record")
    accepted_ids: set[str] = set()
    accepted_by_job: dict[str, list[dict[str, Any]]] = {job_id: [] for job_id in by_job}
    for row in accepted:
        job = by_job.get(row["job_id"])
        if job is None:
            raise ValueError("card result bundle has orphan accepted record")
        outcome = outcome_by_job[row["job_id"]]
        expected = {
            "schema_version": ACCEPTED_CARD_SCHEMA_VERSION,
            "slug": job["slug"], "note_path": job["note_path"],
            "note_sha256": job["note_sha256"], "prompt_sha256": job["prompt_sha256"],
            "result_schema_version": CARD_RESULT_SCHEMA_VERSION,
            "raw_result_sha256": outcome["raw_result_sha256"],
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("card result bundle accepted record binding mismatch")
        if type(row["index"]) is not int or row["index"] < 0:
            raise ValueError("accepted card index must bind a raw wrapper position")
        card = _validate_card(row["card"], job)
        if row["record_id"] != card.card_id or card.card_id in accepted_ids:
            raise ValueError("card result bundle has duplicate or mismatched accepted card ID")
        accepted_ids.add(card.card_id)
        accepted_by_job[job["job_id"]].append(row)
    if accepted_ids != accepted_ids_from_outcomes:
        raise ValueError("accepted cards do not exactly match job outcomes")
    for job_id, outcome in outcome_by_job.items():
        rows = _accepted_projection(accepted_by_job[job_id])
        indexes = [row["index"] for row in rows]
        if len(set(indexes)) != len(indexes):
            raise ValueError("accepted card indexes must be unique per job")
        actual_ids = [row["record_id"] for row in rows]
        if actual_ids != outcome["accepted_card_ids"]:
            raise ValueError("accepted cards projection order does not match outcome")
        if _canonical_hash(rows) != outcome["accepted_cards_sha256"]:
            raise ValueError("accepted cards projection hash mismatch")
        if not outcome["raw_wrapper_reconstructable"]:
            continue
        job = by_job[job_id]
        wrapper = {
            "schema_version": CARD_RESULT_SCHEMA_VERSION,
            "job_id": job_id,
            "slug": job["slug"],
            "note_sha256": job["note_sha256"],
            "prompt_sha256": job["prompt_sha256"],
            "cards": [row["card"] for row in rows],
        }
        if _canonical_hash(wrapper) != outcome["raw_result_sha256"]:
            raise ValueError("card result bundle raw result hash does not bind accepted cards")

    required_rejected = {"schema_version", "result_schema_version", "raw_result_sha256", "error"}
    if any(set(row) < required_rejected for row in rejected):
        raise ValueError("card result bundle has malformed rejected record")
    rejected_by_job: dict[str, list[dict[str, Any]]] = {job_id: [] for job_id in by_job}
    for row in rejected:
        if row["schema_version"] != REJECTED_CARD_SCHEMA_VERSION or row["result_schema_version"] != CARD_RESULT_SCHEMA_VERSION:
            raise ValueError("card result bundle rejected record schema mismatch")
        if "job_id" not in row:
            continue
        job = by_job.get(row["job_id"])
        if job is None:
            raise ValueError("card result bundle has orphan rejected record")
        outcome = outcome_by_job[job["job_id"]]
        if outcome["status"] == "PARTIAL_VALID" and row["raw_result_sha256"] != outcome["raw_result_sha256"]:
            raise ValueError("card result bundle rejected record raw hash mismatch")
        for key in ("slug", "note_sha256", "prompt_sha256"):
            if row.get(key) != job[key]:
                raise ValueError("card result bundle rejected record binding mismatch")
        rejected_by_job[job["job_id"]].append(row)
    for job_id, outcome in outcome_by_job.items():
        rows = _rejected_projection(rejected_by_job[job_id])
        if len(rows) != outcome["rejected_count"]:
            raise ValueError("rejected records do not exactly match outcome count")
        card_indexes = [row["index"] for row in rows if "index" in row]
        if (
            any(type(index) is not int or index < 0 for index in card_indexes)
            or len(set(card_indexes)) != len(card_indexes)
        ):
            raise ValueError("rejected card indexes must uniquely bind raw wrapper positions")
        card_ids = [row["card_id"] for row in rows if isinstance(row.get("card_id"), str)]
        if len(set(card_ids)) != len(card_ids) or card_ids != outcome["rejected_card_ids"]:
            raise ValueError("rejected records do not exactly match outcome card IDs")
        if _canonical_hash(rows) != outcome["rejected_projection_sha256"]:
            raise ValueError("rejected records projection hash mismatch")
        if outcome["status"] != "INVALID":
            accepted_indexes = [row["index"] for row in accepted_by_job[job_id]]
            all_indexes = accepted_indexes + card_indexes
            if sorted(all_indexes) != list(range(len(all_indexes))):
                raise ValueError("card projections do not exactly cover raw wrapper indexes")
    if any(outcome["status"] == "INVALID" for outcome in outcomes):
        raise ValueError("card result bundle contains unresolved INVALID outcomes")
    return paths, accepted, rejected, outcomes


def _rejection(
    job: dict[str, Any] | None,
    error: str,
    raw_result_sha256: str,
    *,
    index: int | None = None,
    card_id: object = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": REJECTED_CARD_SCHEMA_VERSION,
        "result_schema_version": CARD_RESULT_SCHEMA_VERSION,
        "raw_result_sha256": raw_result_sha256,
        "error": error,
    }
    if job is not None:
        record |= {"job_id": job["job_id"], "slug": job["slug"], "note_sha256": job["note_sha256"], "prompt_sha256": job["prompt_sha256"]}
    if index is not None:
        record["index"] = index
    if isinstance(card_id, str):
        record["card_id"] = card_id
    return record


def _accepted_projection(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order accepted records by their immutable raw-wrapper card index."""

    return sorted(records, key=lambda record: record["index"])


def _rejected_projection(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order card rejections by raw index and other errors by stable metadata."""

    def order(record: dict[str, Any]) -> tuple[int, int, str]:
        index = record.get("index")
        if type(index) is int:
            return (0, index, _canonical_json(record))
        return (1, 0, _canonical_json(record))

    return sorted(records, key=order)


def _outcome(
    job: dict[str, Any],
    status: str,
    raw_result_sha256: str,
    accepted_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted_projection = _accepted_projection(accepted_records)
    rejected_projection = _rejected_projection(rejected_records)
    accepted_ids = [record["record_id"] for record in accepted_projection]
    rejected_ids = [
        record["card_id"]
        for record in rejected_projection
        if isinstance(record.get("card_id"), str)
    ]
    return {
        "schema_version": CARD_OUTCOME_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "slug": job["slug"],
        "note_sha256": job["note_sha256"],
        "prompt_sha256": job["prompt_sha256"],
        "result_schema_version": CARD_RESULT_SCHEMA_VERSION,
        "raw_result_sha256": raw_result_sha256,
        "status": status,
        "accepted_card_ids": accepted_ids,
        "rejected_card_ids": rejected_ids,
        "accepted_count": len(accepted_ids),
        "rejected_count": len(rejected_projection),
        "accepted_cards_sha256": _canonical_hash(accepted_projection),
        "rejected_projection_sha256": _canonical_hash(rejected_projection),
        "raw_wrapper_reconstructable": status in {"ACCEPTED", "VALID_EMPTY"},
    }


def ingest_card_results(jobs_path: Path, results: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate raw JSON-object results and publish one auditable result bundle.

    ``results`` must contain exact native ``dict`` objects from the external JSON boundary;
    pre-instantiated Pydantic models are rejected because their original input
    types and normalization history are no longer observable.
    A crash between replacement operations can leave no valid bundle: consumers
    must require all three files and verify each job has exactly one outcome.
    """

    snapshots = [_snapshot_result(raw) for raw in results]
    anchored_jobs, anchored_run = _validate_result_run_anchor(
        jobs_path,
        None,
        require_result_bundle=False,
    )
    paths, jobs = _job_rows(anchored_jobs)
    if paths["run"] != anchored_run:
        raise ValueError("card ingestion resolved to a different active run")
    _prepare_run_directory(paths["run"], "results")
    _invalidate(paths, "accepted", "rejected", "outcomes")
    by_id = {row["job_id"]: row for row in jobs}
    supplied: dict[str, CardResultWrapper] = {}
    invalid_jobs: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    hashes_by_job: dict[str, list[str]] = {job_id: [] for job_id in by_id}
    try:
        for raw_payload, raw_hash in snapshots:
            raw_job_id = raw_payload.get("job_id")
            if isinstance(raw_job_id, str) and raw_job_id in hashes_by_job:
                hashes_by_job[raw_job_id].append(raw_hash)
            try:
                if type(raw_payload.get("cards")) is not list:
                    raise TypeError("result cards must be a JSON array")
                wrapper = CardResultWrapper.model_validate(raw_payload)
            except (ValidationError, TypeError):
                job = by_id.get(raw_job_id) if isinstance(raw_job_id, str) else None
                if job is not None:
                    invalid_jobs[job["job_id"]] = "invalid result wrapper"
                rejected.append(_rejection(job, "invalid result wrapper", raw_hash))
                continue
            job = by_id.get(wrapper.job_id)
            if job is None:
                rejected.append(_rejection(None, "unexpected result", raw_hash))
                continue
            if wrapper.job_id in supplied or wrapper.job_id in invalid_jobs:
                invalid_jobs[wrapper.job_id] = "duplicate result for job"
                rejected.append(_rejection(job, "duplicate result for job", raw_hash))
                continue
            expected = {"schema_version": CARD_RESULT_SCHEMA_VERSION, "job_id": job["job_id"], "slug": job["slug"], "note_sha256": job["note_sha256"], "prompt_sha256": job["prompt_sha256"]}
            mismatch = next((field for field, value in expected.items() if getattr(wrapper, field) != value), None)
            if mismatch is not None:
                invalid_jobs[job["job_id"]] = f"result binding mismatch: {mismatch}"
                rejected.append(_rejection(job, f"result binding mismatch: {mismatch}", raw_hash))
                continue
            supplied[wrapper.job_id] = wrapper

        accepted: list[dict[str, Any]] = []
        outcome_states: list[tuple[dict[str, Any], str, str]] = []
        seen_cards: set[str] = set()
        for job in jobs:
            job_id = job["job_id"]
            raw_hashes = sorted(hashes_by_job[job_id])
            job_raw_hash = (
                raw_hashes[0]
                if len(raw_hashes) == 1
                else _canonical_hash(raw_hashes) if raw_hashes else _hash_bytes(b"")
            )
            if job_id in invalid_jobs:
                outcome_states.append((job, "INVALID", job_raw_hash))
                continue
            wrapper = supplied.get(job_id)
            if wrapper is None:
                rejected.append(_rejection(job, "missing result for job", job_raw_hash))
                outcome_states.append((job, "INVALID", job_raw_hash))
                continue
            if not wrapper.cards:
                outcome_states.append((job, "VALID_EMPTY", job_raw_hash))
                continue
            accepted_ids: list[str] = []
            local_ids: set[str] = set()
            for index, raw_card in enumerate(wrapper.cards):
                supplied_id = raw_card.get("card_id") if isinstance(raw_card, dict) else None
                try:
                    card = _validate_card(raw_card, job)
                    if card.card_id in local_ids or card.card_id in seen_cards:
                        raise ValueError(f"duplicate card_id: {card.card_id}")
                except ValueError as exc:
                    rejected.append(_rejection(job, str(exc), job_raw_hash, index=index, card_id=supplied_id))
                    continue
                local_ids.add(card.card_id)
                seen_cards.add(card.card_id)
                accepted.append({
                    "schema_version": ACCEPTED_CARD_SCHEMA_VERSION,
                    "record_id": card.card_id, "job_id": job_id, "slug": job["slug"],
                    "index": index,
                    "note_path": job["note_path"], "note_sha256": job["note_sha256"],
                    "prompt_sha256": job["prompt_sha256"],
                    "result_schema_version": CARD_RESULT_SCHEMA_VERSION,
                    "raw_result_sha256": job_raw_hash,
                    "card": card.model_dump(mode="json"),
                })
                accepted_ids.append(card.card_id)
            status = "ACCEPTED" if len(accepted_ids) == len(wrapper.cards) else ("PARTIAL_VALID" if accepted_ids else "INVALID")
            outcome_states.append((job, status, job_raw_hash))
        outcomes = [
            _outcome(
                job,
                status,
                raw_hash,
                [record for record in accepted if record["job_id"] == job["job_id"]],
                [record for record in rejected if record.get("job_id") == job["job_id"]],
            )
            for job, status, raw_hash in outcome_states
        ]
        write_jsonl_bundle({paths["accepted"]: accepted, paths["rejected"]: rejected, paths["outcomes"]: outcomes})
        return accepted, rejected, outcomes
    except BaseException:
        _invalidate(paths, "accepted", "rejected", "outcomes")
        raise
