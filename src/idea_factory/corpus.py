"""Deterministic, auditable routing and selection for the KV-memory pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Literal, Sequence

from pydantic import ValidationError

from .artifacts import read_jsonl, sha256_file, stable_id, write_jsonl, write_jsonl_bundle
from .models import CorpusLabel, FrozenStrictModel, NonEmptyStr, Sha256Hex


Confidence = Literal["HIGH", "MEDIUM", "LOW"]
PROMPT_VERSION = "idea_factory.corpus_router_prompt.v2"
ROUTER_JOB_SCHEMA_VERSION = "idea_factory.corpus_router_job.v2"
ROUTER_RESULT_SCHEMA_VERSION = "idea_factory.corpus_router_result.v2"
V3_PROMPT_VERSION = "idea_factory.corpus_router_prompt.v3"
V3_ROUTER_JOB_SCHEMA_VERSION = "idea_factory.corpus_router_job.v3"
V3_ROUTER_RESULT_SCHEMA_VERSION = "idea_factory.corpus_router_result.v3"
V2_LABELS = [
    "KV_CACHE", "LONG_MEMORY", "BRIDGE", "HUMAN_SUPERVISION",
    "SHIFT_ROBUSTNESS", "SUPERVISION_SHIFT_BRIDGE", "OTHER",
]
ROUTER_JOB_FIELDS = {
    "record_id", "slug", "note_path", "source_lists", "source_list_bindings",
    "candidate_list_manifest", "allowed_note_root", "allowed_papers_root",
    "legacy_ledger_path", "note_sha256", "schema_version", "job_id",
    "note_text", "prompt_id", "prompt_version", "prompt_path", "prompt_sha256",
    "prompt_text", "result_schema_version", "required_output_schema",
}
SELECTION_POLICY_SCHEMA_VERSION = "idea_factory.selection_policy.v1"
STRATIFICATION_VERSION = "round_robin_allowed_label_then_slug.v1"
ACCEPTED_CONFIDENCES = ("HIGH", "MEDIUM")

REQUIRED_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "job_id", "slug", "note_sha256", "prompt_sha256",
        "label", "core_mechanism", "scope_reason", "evidence_locator", "confidence",
    ],
    "properties": {
        "schema_version": {"const": ROUTER_RESULT_SCHEMA_VERSION},
        "job_id": {"type": "string", "minLength": 1},
        "slug": {"type": "string", "minLength": 1},
        "note_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "prompt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "label": {"enum": V2_LABELS},
        "core_mechanism": {"type": "string", "minLength": 1},
        "scope_reason": {"type": "string", "minLength": 1},
        "evidence_locator": {"type": "string", "minLength": 1},
        "confidence": {"enum": ["HIGH", "MEDIUM", "LOW"]},
    },
}
V3_REQUIRED_OUTPUT_SCHEMA: dict[str, object] = {
    **REQUIRED_OUTPUT_SCHEMA,
    "properties": {
        **REQUIRED_OUTPUT_SCHEMA["properties"],
        "schema_version": {"const": V3_ROUTER_RESULT_SCHEMA_VERSION},
        "label": {"enum": ["GENERAL_RESEARCH", "OTHER"]},
    },
}


@dataclass(frozen=True)
class CorpusRouterConfig:
    """Resolved configuration; paths are never implicitly tied to a machine root."""

    notes_root: Path
    candidate_lists: tuple[Path, ...]
    legacy_ledger: Path
    target_min: int
    target_max: int
    allowed_labels: tuple[str, ...]
    bridge_regression_slugs: tuple[str, ...]
    protocol_version: str = "v2"
    source_primary_manifest: Path | None = None

    def __post_init__(self) -> None:
        if (
            type(self.target_min) is not int
            or type(self.target_max) is not int
            or self.target_min < 1
            or self.target_max < 1
        ):
            raise ValueError("target_min and target_max must be exact positive integers")
        if self.target_max < self.target_min:
            raise ValueError("target bounds must satisfy 1 <= target_min <= target_max")
        if not isinstance(self.allowed_labels, (list, tuple)):
            raise ValueError("allowed_labels must be a list or tuple")
        allowed = tuple(self.allowed_labels)
        if self.protocol_version not in {"v2", "v3"}:
            raise ValueError("protocol_version must be v2 or v3")
        if self.protocol_version == "v3" and self.source_primary_manifest is None:
            raise ValueError("v3 requires an explicit primary source manifest")
        if self.protocol_version == "v2" and self.source_primary_manifest is not None:
            raise ValueError("primary source manifest is v3-only")
        eligible_labels = (
            {"GENERAL_RESEARCH"} if self.protocol_version == "v3" else set(V2_LABELS) - {"OTHER"}
        )
        if (
            not allowed
            or any(type(label) is not str or label not in eligible_labels for label in allowed)
            or len(set(allowed)) != len(allowed)
        ):
            raise ValueError("allowed_labels must be nonempty, unique exact allowed enum strings")
        if not isinstance(self.candidate_lists, (list, tuple)) or not self.candidate_lists:
            raise ValueError("candidate_lists must be nonempty")
        if any(
            not isinstance(path, (str, os.PathLike))
            or (isinstance(path, str) and not path.strip())
            for path in self.candidate_lists
        ):
            raise ValueError("candidate_lists entries must be nonblank paths")
        resolved_lists = tuple(Path(path).resolve() for path in self.candidate_lists)
        normalized_lists = tuple(os.path.normcase(str(path)) for path in resolved_lists)
        if len(set(normalized_lists)) != len(normalized_lists):
            raise ValueError("candidate_lists must be unique after resolution")
        if not isinstance(self.bridge_regression_slugs, (list, tuple)):
            raise ValueError("bridge_regression_slugs must be a list or tuple")
        normalized_bridges: list[str] = []
        for slug in self.bridge_regression_slugs:
            if type(slug) is not str or not slug.strip():
                raise ValueError("bridge_regression_slugs must be nonblank unique strings")
            normalized_bridges.append(slug.strip().lower())
        if len(set(normalized_bridges)) != len(normalized_bridges):
            raise ValueError("bridge_regression_slugs must be unique after normalization")
        object.__setattr__(self, "notes_root", Path(self.notes_root).resolve())
        object.__setattr__(self, "candidate_lists", resolved_lists)
        object.__setattr__(self, "legacy_ledger", Path(self.legacy_ledger).resolve())
        object.__setattr__(self, "allowed_labels", allowed)
        object.__setattr__(self, "bridge_regression_slugs", tuple(normalized_bridges))
        if self.source_primary_manifest is not None:
            raw_manifest = Path(self.source_primary_manifest)
            if raw_manifest.is_symlink():
                raise ValueError("primary source manifest must not be a symlink")
            try:
                resolved_manifest = raw_manifest.resolve(strict=True)
            except OSError as exc:
                raise ValueError("primary source manifest is missing") from exc
            if not resolved_manifest.is_file():
                raise ValueError("primary source manifest must be a regular file")
            object.__setattr__(self, "source_primary_manifest", resolved_manifest)

    @classmethod
    def from_json(cls, path: Path, *, repo_root: Path | None = None) -> "CorpusRouterConfig":
        """Load JSON config using an explicit or coherently discovered repo root."""

        config_path = Path(path).resolve()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid corpus router config: {config_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("corpus router config must be an object")
        expected = {
            "notes_root", "candidate_lists", "legacy_ledger", "target_min", "target_max",
            "allowed_labels", "bridge_regression_slugs",
        }
        accepted_keys = {
            frozenset(expected),
            frozenset(expected | {"protocol_version"}),
            frozenset(expected | {"protocol_version", "source_primary_manifest"}),
        }
        if frozenset(payload) not in accepted_keys:
            raise ValueError("corpus router config has missing or unexpected fields")
        if payload.get("protocol_version", "v2") == "v3" and "source_primary_manifest" not in payload:
            raise ValueError("v3 config must explicitly bind source_primary_manifest")

        lists = payload["candidate_lists"]
        if not isinstance(lists, list) or not lists:
            raise ValueError("candidate_lists must be a nonempty array")
        allowed_labels = payload["allowed_labels"]
        if not isinstance(allowed_labels, list):
            raise ValueError("allowed_labels must be an array")
        bridge_slugs = payload["bridge_regression_slugs"]
        if not isinstance(bridge_slugs, list):
            raise ValueError("bridge_regression_slugs must be an array")
        path_values = [payload["notes_root"], payload["legacy_ledger"], *lists]
        if "source_primary_manifest" in payload:
            path_values.append(payload["source_primary_manifest"])
        if any(not isinstance(value, str) or not value.strip() for value in path_values):
            raise ValueError("corpus router config paths must be nonblank strings")

        def resolve_from(root: Path, value: object) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        if repo_root is not None:
            root = Path(repo_root).resolve()
        else:
            config_repo_root = config_path.parent.parent
            root = next(
                (
                    base
                    for base in (config_repo_root, *config_repo_root.parents)
                    if resolve_from(base, payload["notes_root"]).is_dir()
                    and resolve_from(base, payload["legacy_ledger"]).parent.is_dir()
                    and all(resolve_from(base, item).is_file() for item in lists)
                ),
                None,
            )
            if root is None:
                raise ValueError(
                    f"no coherent config base found in ancestor chain from {config_repo_root}"
                )
        return cls(
            notes_root=resolve_from(root, payload["notes_root"]),
            candidate_lists=tuple(resolve_from(root, item) for item in lists),
            legacy_ledger=resolve_from(root, payload["legacy_ledger"]),
            target_min=payload["target_min"],
            target_max=payload["target_max"],
            allowed_labels=tuple(allowed_labels),
            bridge_regression_slugs=tuple(bridge_slugs),
            protocol_version=payload.get("protocol_version", "v2"),
            source_primary_manifest=resolve_from(root, payload["source_primary_manifest"]) if payload.get("source_primary_manifest") else None,
        )


@dataclass(frozen=True)
class CorpusCandidate:
    slug: str
    note_path: Path
    source_lists: tuple[str, ...]
    note_sha256: str
    source_list_bindings: tuple[tuple[str, str], ...] = ()
    candidate_list_manifest: tuple[tuple[str, str], ...] = ()
    allowed_note_root: str = ""
    allowed_papers_root: str = ""
    legacy_ledger_path: str = ""
    protocol_version: str = "v2"
    source_primary_manifest_binding: tuple[str, str] | None = None
    source_pdf_path: str | None = None
    source_pdf_sha256: str | None = None
    source_record_aliases: tuple[str, ...] = ()
    source_status: str = "NOT_ADJUDICATED"
    source_version: str | None = None
    source_adjudication_binding: tuple[str, str] | None = None
    primary_source_record: dict[str, object] | None = None


class RouterResult(FrozenStrictModel):
    schema_version: Literal[
        "idea_factory.corpus_router_result.v2", "idea_factory.corpus_router_result.v3"
    ]
    job_id: NonEmptyStr
    slug: NonEmptyStr
    note_sha256: Sha256Hex
    prompt_sha256: Sha256Hex
    label: CorpusLabel
    core_mechanism: NonEmptyStr
    scope_reason: NonEmptyStr
    evidence_locator: NonEmptyStr
    confidence: Confidence


@dataclass(frozen=True)
class IngestedRouterResult:
    candidate: CorpusCandidate
    result: RouterResult
    raw_result_sha256: str


@dataclass(frozen=True)
class RejectedCandidate:
    candidate: CorpusCandidate
    result: RouterResult
    raw_result_sha256: str
    reason_code: str


def _is_audit(path: Path) -> bool:
    return path.name.lower().endswith(".audit.md")


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _slug_for(path: Path) -> str:
    if path.suffix.lower() != ".md":
        raise ValueError(f"candidate note must be a .md file: {path}")
    return path.name[:-3].lower()


def _read_candidate_paths(source_list: Path) -> Iterable[Path]:
    if not source_list.is_file():
        raise ValueError(f"candidate list is missing or not a file: {source_list}")
    for line in source_list.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        item = Path(raw)
        yield item.resolve() if item.is_absolute() else (source_list.parent / item).resolve()


_ADJUDICATION_FIELDS = {
    "schema_version", "reviewed_at_utc", "review_scope", "human_approved", "event_id",
    "canonical_arxiv_id", "source_version", "source_url", "source_status",
    "scientific_admission", "evidence_summary", "original_download_status",
    "original_download_error", "retry_decision", "silent_version_fallback_allowed",
    "original_artifacts_changed", "graph_ingested", "downstream_requirement",
}


def _load_primary_source_manifest(path: Path) -> tuple[str, str, dict[str, dict[str, object]]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("primary source manifest is missing or unsafe")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("primary source manifest is not valid UTF-8 JSON") from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "records"} or data.get("schema_version") != "idea_factory.primary_source_manifest.v1" or not isinstance(data["records"], list):
        raise ValueError("primary source manifest has unexpected or missing fields")
    required = {
        "slug", "canonical_note_path", "note_sha256", "primary_pdf_path", "primary_pdf_sha256",
        "source_record_ids", "source_status", "source_version", "adjudication_path", "adjudication_sha256",
    }
    records: dict[str, dict[str, object]] = {}
    allowed_statuses = {
        "PRIMARY_SOURCE_VERIFIED", "SOURCE_WITHDRAWN", "SOURCE_REJECTED",
        "SOURCE_UNAVAILABLE", "SOURCE_STATUS_PENDING",
    }
    for row in data["records"]:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("primary source record has unexpected or missing fields")
        slug = row["slug"]
        note_path = row["canonical_note_path"]
        note_sha = row["note_sha256"]
        pdf_path, pdf_sha = row["primary_pdf_path"], row["primary_pdf_sha256"]
        ids, status, version = row["source_record_ids"], row["source_status"], row["source_version"]
        adj_path, adj_sha = row["adjudication_path"], row["adjudication_sha256"]
        if (
            type(slug) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", slug)
            or type(note_path) is not str or str(Path(note_path).resolve()) != note_path
            or type(note_sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", note_sha)
            or not isinstance(ids, list) or not ids or any(type(item) is not str or not item.strip() for item in ids)
            or len(set(ids)) != len(ids) or type(status) is not str or status not in allowed_statuses
            or (version is not None and (type(version) is not str or not re.fullmatch(r"v[1-9][0-9]*", version)))
        ):
            raise ValueError("primary source record is malformed")
        for source_id in ids:
            versioned_arxiv = re.fullmatch(r"(?:arxiv|arxiv_existing):\d{4}\.\d{4,5}(v[1-9][0-9]*)", source_id.lower())
            if versioned_arxiv and version != versioned_arxiv.group(1):
                raise ValueError("primary source record version does not match its exact source alias")
        if (pdf_path is None) != (pdf_sha is None):
            raise ValueError("primary PDF path and hash must be bound together")
        if pdf_path is not None:
            if type(pdf_path) is not str or str(Path(pdf_path).resolve()) != pdf_path or type(pdf_sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", pdf_sha):
                raise ValueError("primary PDF binding is malformed")
        elif status == "PRIMARY_SOURCE_VERIFIED":
            raise ValueError("verified primary source requires an actual primary PDF binding")
        if (adj_path is None) != (adj_sha is None):
            raise ValueError("adjudication path and hash must be bound together")
        if status in {"SOURCE_WITHDRAWN", "SOURCE_REJECTED"} and adj_path is None:
            raise ValueError("withdrawn or rejected source requires an explicit adjudication binding")
        if adj_path is not None and (type(adj_path) is not str or str(Path(adj_path).resolve()) != adj_path or type(adj_sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", adj_sha)):
            raise ValueError("adjudication binding is malformed")
        if note_path in records:
            raise ValueError("primary source manifest contains duplicate canonical notes")
        if any(existing["slug"] == slug for existing in records.values()):
            raise ValueError("primary source manifest contains duplicate stable slugs")
        records[note_path] = row
    source_identity_rows: dict[str, list[dict[str, object]]] = {}
    for row in records.values():
        for source_id in row["source_record_ids"]:
            value = str(source_id).lower()
            match = re.fullmatch(r"(?:arxiv|arxiv_existing):(\d{4}\.\d{4,5})(v[1-9][0-9]*)?", value)
            identity = f"arxiv:{match.group(1)}" if match else value
            source_identity_rows.setdefault(identity, []).append(row)
    for rows in source_identity_rows.values():
        pdf_hashes = {row["primary_pdf_sha256"] for row in rows if row["primary_pdf_sha256"] is not None}
        if len(pdf_hashes) > 1 and any(row["source_status"] != "SOURCE_STATUS_PENDING" for row in rows):
            raise ValueError("source identity aliases across distinct primary PDFs must remain SOURCE_STATUS_PENDING")
    return str(path.resolve()), sha256_file(path), records


def _validate_primary_source_row(row: dict[str, object], *, note_path: Path, note_sha256: str, papers_root: Path) -> None:
    if row["canonical_note_path"] != str(note_path) or row["note_sha256"] != note_sha256:
        raise ValueError("primary source manifest note hash/path binding mismatch")
    pdf_path, pdf_sha = row["primary_pdf_path"], row["primary_pdf_sha256"]
    if pdf_path is not None:
        pdf = Path(str(pdf_path))
        if pdf.is_symlink() or not pdf.is_file() or not _is_within(pdf.resolve(), papers_root):
            raise ValueError("primary source manifest PDF path is unsafe or outside the allowed papers root")
        if sha256_file(pdf) != pdf_sha:
            raise ValueError("primary source manifest PDF hash mismatch")
    if row["source_status"] == "PRIMARY_SOURCE_VERIFIED" and pdf_path is None:
        raise ValueError("verified primary source requires a bound primary PDF")
    adj_path, adj_sha = row["adjudication_path"], row["adjudication_sha256"]
    if adj_path is not None:
        path = Path(str(adj_path))
        if path.is_symlink() or not path.is_file() or sha256_file(path) != adj_sha:
            raise ValueError("primary source adjudication hash binding mismatch")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("primary source adjudication is not valid UTF-8 JSON") from exc
        if not isinstance(data, dict) or set(data) != _ADJUDICATION_FIELDS or data.get("schema_version") != "source_review_adjudication.v1":
            raise ValueError("primary source adjudication has unexpected or missing fields")
        if (
            type(data.get("canonical_arxiv_id")) is not str
            or not re.fullmatch(r"\d{4}\.\d{4,5}", data["canonical_arxiv_id"])
            or type(data.get("source_version")) is not str
            or not re.fullmatch(r"v[1-9][0-9]*", data["source_version"])
            or row["source_version"] != data["source_version"]
            or data.get("source_status") != {
                "SOURCE_WITHDRAWN": "WITHDRAWN_SOURCE",
                "SOURCE_REJECTED": "REJECTED_SOURCE",
            }.get(str(row["source_status"]), row["source_status"])
            or (data.get("source_status") == "WITHDRAWN_SOURCE" and data.get("scientific_admission") != "QUARANTINE_FOR_SCIENTIFIC_USE")
            or (data.get("source_status") == "REJECTED_SOURCE" and data.get("scientific_admission") != "REJECT_FOR_SCIENTIFIC_USE")
        ):
            raise ValueError("primary source adjudication identity/status mismatch")
        exact_id = f"arxiv:{data['canonical_arxiv_id'].lower()}{data['source_version']}"
        aliases = {str(value).lower() for value in row["source_record_ids"]}
        if exact_id not in aliases and f"arxiv_existing:{exact_id.split(':', 1)[1]}" not in aliases:
            raise ValueError("primary source adjudication identity is not bound to this canonical note")


def enumerate_candidates(config: CorpusRouterConfig) -> tuple[CorpusCandidate, ...]:
    """Read list membership once, excluding audits before any content is read."""

    corpus_root = config.legacy_ledger.parent.parent.resolve()
    allowed_roots = (config.notes_root.resolve(), corpus_root)
    preflighted_lists: list[Path] = []
    seen_lists: set[str] = set()
    for configured_path in config.candidate_lists:
        try:
            candidate_list = Path(configured_path).resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"candidate list is missing or not a regular file: {configured_path}") from exc
        identity = os.path.normcase(str(candidate_list))
        if identity in seen_lists:
            raise ValueError("candidate lists collide after canonical resolution")
        if not _is_within(candidate_list, corpus_root):
            raise ValueError(f"candidate list must remain within configured corpus root: {candidate_list}")
        if not candidate_list.is_file():
            raise ValueError(f"candidate list is missing or not a regular file: {candidate_list}")
        seen_lists.add(identity)
        preflighted_lists.append(candidate_list)
    preflighted_lists.sort()
    list_manifest = tuple(
        (str(path), sha256_file(path)) for path in preflighted_lists
    )
    list_hashes = dict(list_manifest)
    primary_binding: tuple[str, str] | None = None
    primary_records: dict[str, dict[str, object]] = {}
    if config.protocol_version == "v3":
        if config.source_primary_manifest is None:
            raise ValueError("v3 requires a primary source manifest")
        manifest_path, manifest_sha, primary_records = _load_primary_source_manifest(config.source_primary_manifest)
        primary_binding = (manifest_path, manifest_sha)
    discovered: dict[Path, tuple[str, list[str]]] = {}
    for source_list in preflighted_lists:
        source = str(source_list)
        for path in _read_candidate_paths(source_list):
            if _is_audit(path):
                continue
            if path == config.legacy_ledger:
                continue
            if not any(_is_within(path, root) for root in allowed_roots):
                raise ValueError(f"candidate is outside configured candidate roots: {path}")
            if not path.is_file():
                if config.protocol_version == "v2":
                    raise ValueError(f"missing base note for {_slug_for(path)}: {path}")
                raise ValueError(f"missing base note: {path}")
            if config.protocol_version == "v3":
                record = primary_records.get(str(path))
                if record is None:
                    raise ValueError(f"primary source manifest is missing canonical note binding: {path}")
                slug = str(record["slug"])
                _validate_primary_source_row(
                    record, note_path=path, note_sha256=sha256_file(path), papers_root=corpus_root
                )
            else:
                slug = _slug_for(path)
            existing = discovered.get(path)
            if existing is None:
                if any(previous_slug == slug and previous_path != path for previous_path, (previous_slug, _) in discovered.items()):
                    if config.protocol_version == "v2":
                        raise ValueError(f"duplicate slug {slug} points to different base files")
                    raise ValueError(f"duplicate stable candidate slug {slug}")
                discovered[path] = (slug, [source])
            else:
                previous_slug, memberships = existing
                if previous_slug != slug:
                    raise ValueError(f"candidate path changed stable slug: {path}")
                if source not in memberships:
                    memberships.append(source)
    if config.protocol_version == "v3" and set(primary_records) != {str(path) for path in discovered}:
        raise ValueError("primary source manifest must exactly cover candidate notes")
    slugs = {slug for slug, _ in discovered.values()}
    missing_controls = sorted(set(config.bridge_regression_slugs) - slugs)
    if missing_controls:
        raise ValueError(
            "missing configured bridge regression candidates: "
            + ", ".join(missing_controls)
        )
    candidates: list[CorpusCandidate] = []
    for path, (slug, memberships) in sorted(discovered.items(), key=lambda item: item[1][0]):
        record = primary_records.get(str(path)) if config.protocol_version == "v3" else None
        candidates.append(CorpusCandidate(
            slug=slug,
            note_path=path,
            source_lists=tuple(sorted(memberships)),
            note_sha256=sha256_file(path),
            source_list_bindings=tuple((source, list_hashes[source]) for source in sorted(memberships)),
            candidate_list_manifest=list_manifest,
            allowed_note_root=str(config.notes_root),
            allowed_papers_root=str(corpus_root),
            legacy_ledger_path=str(config.legacy_ledger),
            protocol_version=config.protocol_version,
            source_primary_manifest_binding=primary_binding,
            source_pdf_path=record["primary_pdf_path"] if record is not None else None,
            source_pdf_sha256=record["primary_pdf_sha256"] if record is not None else None,
            source_record_aliases=tuple(record["source_record_ids"]) if record is not None else (),
            source_status=str(record["source_status"]) if record is not None else "NOT_ADJUDICATED",
            source_version=record["source_version"] if record is not None else None,
            source_adjudication_binding=(
                (str(record["adjudication_path"]), str(record["adjudication_sha256"]))
                if record is not None and record["adjudication_path"] is not None else None
            ),
            primary_source_record=record,
        ))
    return tuple(candidates)


def _candidate_record(candidate: CorpusCandidate) -> dict[str, object]:
    record = {
        "record_id": f"corpus-{candidate.slug}",
        "slug": candidate.slug,
        "note_path": str(candidate.note_path),
        "source_lists": list(candidate.source_lists),
        "source_list_bindings": [
            {"path": path, "sha256": digest}
            for path, digest in candidate.source_list_bindings
        ],
        "candidate_list_manifest": [
            {"path": path, "sha256": digest}
            for path, digest in candidate.candidate_list_manifest
        ],
        "allowed_note_root": candidate.allowed_note_root,
        "allowed_papers_root": candidate.allowed_papers_root,
        "legacy_ledger_path": candidate.legacy_ledger_path,
        "note_sha256": candidate.note_sha256,
    }
    if candidate.protocol_version == "v3":
        record.update({
            "primary_source_manifest_binding": {
                "path": candidate.source_primary_manifest_binding[0],
                "sha256": candidate.source_primary_manifest_binding[1],
            },
            "primary_source_record": candidate.primary_source_record,
            "source_record_aliases": list(candidate.source_record_aliases),
        })
    return record


def _owned_corpus_paths(active_run_dir: Path) -> dict[str, Path]:
    requested_run_dir = Path(active_run_dir)
    requested_run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = requested_run_dir.resolve(strict=True)
    requested_corpus_dir = run_dir / "corpus"
    requested_corpus_dir.mkdir(exist_ok=True)
    corpus_dir = requested_corpus_dir.resolve(strict=True)
    if corpus_dir.parent != run_dir:
        raise ValueError("corpus output directory must resolve inside active run directory")
    return {
        "jobs": corpus_dir / "router_jobs.jsonl",
        "router_results": corpus_dir / "router_results.jsonl",
        "policy": corpus_dir / "selection_policy.jsonl",
        "selection": corpus_dir / "selection_manifest.jsonl",
        "rejected": corpus_dir / "rejected_manifest.jsonl",
    }


def router_job_id(slug: str, note_sha256: str, prompt_sha256: str) -> str:
    """Bind a job to paper content, prompt content, and the result schema."""

    return stable_id(
        "router_job", slug, note_sha256, prompt_sha256, ROUTER_RESULT_SCHEMA_VERSION
    )


def _protocol_values(protocol_version: str) -> tuple[str, str, str, dict[str, object]]:
    if protocol_version == "v2":
        return PROMPT_VERSION, ROUTER_JOB_SCHEMA_VERSION, ROUTER_RESULT_SCHEMA_VERSION, REQUIRED_OUTPUT_SCHEMA
    if protocol_version == "v3":
        return V3_PROMPT_VERSION, V3_ROUTER_JOB_SCHEMA_VERSION, V3_ROUTER_RESULT_SCHEMA_VERSION, V3_REQUIRED_OUTPUT_SCHEMA
    raise ValueError(f"unsupported corpus routing protocol: {protocol_version}")


def _protocol_job_id(slug: str, note_sha256: str, prompt_sha256: str, protocol_version: str) -> str:
    _, _, result_schema, _ = _protocol_values(protocol_version)
    if protocol_version == "v2":
        return router_job_id(slug, note_sha256, prompt_sha256)
    return stable_id("router_job", slug, note_sha256, prompt_sha256, result_schema)


def write_router_jobs(
    candidates: Sequence[CorpusCandidate],
    active_run_dir: Path,
    *,
    prompt_path: Path,
    protocol_version: str = "v2",
) -> Path:
    """Replace the run-local job file, never modifying caller-owned source lists."""

    if any(getattr(candidate, "protocol_version", "v2") != protocol_version for candidate in candidates):
        raise ValueError("router candidates do not match the selected protocol version")
    owned_paths = _owned_corpus_paths(active_run_dir)
    owned_paths["router_results"].unlink(missing_ok=True)
    owned_paths["policy"].unlink(missing_ok=True)
    owned_paths["selection"].unlink(missing_ok=True)
    owned_paths["rejected"].unlink(missing_ok=True)
    resolved_prompt = Path(prompt_path).resolve()
    try:
        prompt_bytes = resolved_prompt.read_bytes()
        prompt_text = prompt_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"router prompt must be a readable UTF-8 file: {resolved_prompt}"
        ) from exc
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    prompt_version, job_schema_version, result_schema_version, output_schema = _protocol_values(protocol_version)
    records: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: item.slug):
        try:
            note_bytes = candidate.note_path.read_bytes()
            note_text = note_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"candidate note must be a readable UTF-8 file: {candidate.note_path}"
            ) from exc
        if hashlib.sha256(note_bytes).hexdigest() != candidate.note_sha256:
            raise ValueError(f"candidate note changed after enumeration: {candidate.slug}")
        job_id = _protocol_job_id(candidate.slug, candidate.note_sha256, prompt_sha256, protocol_version)
        records.append(
            _candidate_record(candidate)
            | {
                "record_id": job_id,
                "schema_version": job_schema_version,
                "job_id": job_id,
                "note_text": note_text,
                "prompt_id": "corpus_router",
                "prompt_version": prompt_version,
                "prompt_path": str(resolved_prompt),
                "prompt_sha256": prompt_sha256,
                "prompt_text": prompt_text,
                "result_schema_version": result_schema_version,
                "required_output_schema": output_schema,
            }
        )
    path = owned_paths["jobs"]
    write_jsonl(path, records)
    return path


def _binding_pairs(value: object, field_name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a canonical binding list")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError(f"{field_name} contains an invalid binding")
        path, digest = item["path"], item["sha256"]
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError(f"{field_name} contains an invalid binding")
        pairs.append((path, digest))
    if pairs != sorted(pairs) or len({path for path, _ in pairs}) != len(pairs):
        raise ValueError(f"{field_name} must be ordered and unique")
    return tuple(pairs)


def _validate_router_jobs(path: Path) -> list[dict[str, object]]:
    jobs_path = Path(path).resolve(strict=True)
    if jobs_path.name != "router_jobs.jsonl" or jobs_path.parent.name != "corpus":
        raise ValueError("router jobs must use exact active-run corpus path")
    jobs = read_jsonl(jobs_path)
    if not jobs:
        raise ValueError("router jobs must be nonempty")
    if any(
        set(job) != ROUTER_JOB_FIELDS | ({
            "primary_source_manifest_binding", "primary_source_record", "source_record_aliases",
        } if job.get("schema_version") == V3_ROUTER_JOB_SCHEMA_VERSION else set())
        for job in jobs
    ):
        raise ValueError("router jobs contain unexpected or missing fields")
    if len({job.get("slug") for job in jobs}) != len(jobs) or len({job.get("job_id") for job in jobs}) != len(jobs):
        raise ValueError("router jobs contain duplicate slugs or job IDs")
    first = jobs[0]
    roots = (first.get("allowed_note_root"), first.get("allowed_papers_root"), first.get("legacy_ledger_path"))
    if any(not isinstance(value, str) or str(Path(value).resolve()) != value for value in roots):
        raise ValueError("router jobs contain noncanonical provenance roots")
    note_root, papers_root, legacy = (Path(value) for value in roots)
    manifest = _binding_pairs(first.get("candidate_list_manifest"), "candidate_list_manifest")
    if not manifest:
        raise ValueError("candidate_list_manifest must be nonempty")
    for list_path, digest in manifest:
        candidate_list = Path(list_path)
        if str(candidate_list.resolve()) != list_path or not _is_within(candidate_list, papers_root) or not candidate_list.is_file():
            raise ValueError(f"candidate list is missing or outside allowed papers root: {list_path}")
        if sha256_file(candidate_list) != digest:
            raise ValueError(f"candidate list changed after routing: {list_path}")
    protocols = {
        "v2" if job.get("schema_version") == ROUTER_JOB_SCHEMA_VERSION else
        "v3" if job.get("schema_version") == V3_ROUTER_JOB_SCHEMA_VERSION else
        "unsupported"
        for job in jobs
    }
    if len(protocols) != 1 or "unsupported" in protocols:
        raise ValueError("router job schema binding mismatch")
    protocol_version = next(iter(protocols))
    prompt_version, job_schema_version, result_schema_version, output_schema = _protocol_values(protocol_version)
    primary_manifest_binding: tuple[str, str] | None = None
    primary_records: dict[str, dict[str, object]] = {}
    if protocol_version == "v3":
        raw_binding = jobs[0].get("primary_source_manifest_binding")
        if not isinstance(raw_binding, dict) or set(raw_binding) != {"path", "sha256"}:
            raise ValueError("primary source manifest binding is malformed")
        manifest_path, manifest_sha = raw_binding["path"], raw_binding["sha256"]
        if type(manifest_path) is not str or type(manifest_sha) is not str:
            raise ValueError("primary source manifest binding is malformed")
        loaded_path, loaded_sha, primary_records = _load_primary_source_manifest(Path(manifest_path))
        if (loaded_path, loaded_sha) != (manifest_path, manifest_sha):
            raise ValueError("primary source manifest hash binding mismatch")
        primary_manifest_binding = (manifest_path, manifest_sha)
    routed_primary_notes: set[str] = set()
    for job in jobs:
        if job.get("schema_version") != job_schema_version or job.get("result_schema_version") != result_schema_version:
            raise ValueError("router job schema binding mismatch")
        if tuple(job.get(key) for key in ("allowed_note_root", "allowed_papers_root", "legacy_ledger_path")) != roots:
            raise ValueError("router jobs contain mixed provenance roots")
        if _binding_pairs(job.get("candidate_list_manifest"), "candidate_list_manifest") != manifest:
            raise ValueError("router jobs contain mixed candidate-list manifests")
        bindings = _binding_pairs(job.get("source_list_bindings"), "source_list_bindings")
        if list(job.get("source_lists", [])) != [path for path, _ in bindings]:
            raise ValueError("router job source-list set mismatch")
        note = Path(str(job.get("note_path"))).resolve()
        if str(note) != job.get("note_path") or not note.is_file() or note == legacy or _is_audit(note) or note.name.lower().endswith(".atom.md") or any(part.lower() == "_atoms" for part in note.parts):
            raise ValueError("router job note path is unsafe")
        if not (_is_within(note, note_root) or _is_within(note, papers_root)):
            raise ValueError("router job note is outside provenance roots")
        note_bytes = note.read_bytes()
        try:
            note_text = note_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("router job note is not UTF-8") from exc
        if sha256_file(note) != job.get("note_sha256") or note_text != job.get("note_text"):
            raise ValueError("router job note binding mismatch")
        if job.get("schema_version") == V3_ROUTER_JOB_SCHEMA_VERSION:
            if job.get("primary_source_manifest_binding") != {
                "path": primary_manifest_binding[0], "sha256": primary_manifest_binding[1],
            }:
                raise ValueError("router jobs contain mixed primary source manifests")
            if job.get("primary_source_record") != primary_records.get(str(note)):
                raise ValueError("router job primary source identity binding mismatch")
            primary = job["primary_source_record"]
            if primary.get("slug") != job.get("slug") or job.get("source_record_aliases") != primary.get("source_record_ids"):
                raise ValueError("router job stable slug does not match primary source manifest")
            _validate_primary_source_row(primary, note_path=note, note_sha256=str(job["note_sha256"]), papers_root=papers_root)
            routed_primary_notes.add(str(note))
        for list_path, digest in bindings:
            if (list_path, digest) not in manifest or note not in set(_read_candidate_paths(Path(list_path))):
                raise ValueError("router job source list does not contain canonical note path")
        prompt = Path(str(job.get("prompt_path"))).resolve()
        prompt_bytes = prompt.read_bytes()
        try:
            prompt_text = prompt_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("router prompt is not UTF-8") from exc
        prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
        expected_id = _protocol_job_id(str(job.get("slug")), str(job.get("note_sha256")), prompt_sha, protocol_version)
        if (
            job.get("prompt_sha256") != prompt_sha
            or job.get("prompt_text") != prompt_text
            or job.get("prompt_id") != "corpus_router"
            or job.get("prompt_version") != prompt_version
            or job.get("required_output_schema") != output_schema
            or job.get("job_id") != expected_id
            or job.get("record_id") != expected_id
        ):
            raise ValueError("router job prompt or ID binding mismatch")
    if protocol_version == "v3" and routed_primary_notes != set(primary_records):
        raise ValueError("primary source manifest does not exactly match routed candidate notes")
    return jobs


def _replay_router_results(
    path: Path, jobs: Sequence[dict[str, object]]
) -> tuple[IngestedRouterResult, ...]:
    records = read_jsonl(path)
    if [record.get("slug") for record in records] != sorted(job["slug"] for job in jobs):
        raise ValueError("router results must exactly and canonically match router jobs")
    jobs_by_slug = {job["slug"]: job for job in jobs}
    replayed: list[IngestedRouterResult] = []
    for record in records:
        if set(record) != set(RouterResult.model_fields) | {"raw_result_sha256"}:
            raise ValueError("router results contain missing or unexpected fields")
        raw_hash = record["raw_result_sha256"]
        result = RouterResult.model_validate(
            {key: value for key, value in record.items() if key != "raw_result_sha256"}
        )
        if raw_hash != _canonical_result_hash(result):
            raise ValueError("router result raw hash mismatch")
        job = jobs_by_slug[result.slug]
        if (
            result.job_id != job["job_id"]
            or result.note_sha256 != job["note_sha256"]
            or result.prompt_sha256 != job["prompt_sha256"]
            or result.schema_version != job["result_schema_version"]
            or (
                job["result_schema_version"] == V3_ROUTER_RESULT_SCHEMA_VERSION
                and result.label.value not in {"GENERAL_RESEARCH", "OTHER"}
            )
        ):
            raise ValueError("router result binding mismatch")
        candidate = CorpusCandidate(
            slug=result.slug,
            note_path=Path(str(job["note_path"])),
            source_lists=tuple(job["source_lists"]),
            note_sha256=str(job["note_sha256"]),
            source_list_bindings=_binding_pairs(job["source_list_bindings"], "source_list_bindings"),
            candidate_list_manifest=_binding_pairs(job["candidate_list_manifest"], "candidate_list_manifest"),
            allowed_note_root=str(job["allowed_note_root"]),
            allowed_papers_root=str(job["allowed_papers_root"]),
            legacy_ledger_path=str(job["legacy_ledger_path"]),
            protocol_version=("v3" if job["result_schema_version"] == V3_ROUTER_RESULT_SCHEMA_VERSION else "v2"),
            source_primary_manifest_binding=(
                (str(job["primary_source_manifest_binding"]["path"]), str(job["primary_source_manifest_binding"]["sha256"]))
                if isinstance(job.get("primary_source_manifest_binding"), dict) else None
            ),
            source_pdf_path=(job["primary_source_record"]["primary_pdf_path"] if isinstance(job.get("primary_source_record"), dict) else None),
            source_pdf_sha256=(job["primary_source_record"]["primary_pdf_sha256"] if isinstance(job.get("primary_source_record"), dict) else None),
            source_record_aliases=tuple(job["primary_source_record"]["source_record_ids"]) if isinstance(job.get("primary_source_record"), dict) else (),
            source_status=str(job["primary_source_record"]["source_status"]) if isinstance(job.get("primary_source_record"), dict) else "NOT_ADJUDICATED",
            source_version=job["primary_source_record"].get("source_version") if isinstance(job.get("primary_source_record"), dict) else None,
            source_adjudication_binding=(
                (str(job["primary_source_record"]["adjudication_path"]), str(job["primary_source_record"]["adjudication_sha256"]))
                if isinstance(job.get("primary_source_record"), dict) and job["primary_source_record"].get("adjudication_path") is not None else None
            ),
            primary_source_record=(job.get("primary_source_record") if isinstance(job.get("primary_source_record"), dict) else None),
        )
        replayed.append(IngestedRouterResult(candidate, result, str(raw_hash)))
    return tuple(replayed)


def _replay_policy(path: Path) -> dict[str, object]:
    records = read_jsonl(path)
    if len(records) != 1:
        raise ValueError("selection policy artifact must contain exactly one record")
    policy = records[0]
    required = {
        "schema_version", "allowed_labels", "accepted_confidences", "target_min",
        "target_max", "mandatory_bridge_slugs", "stratification_version",
        "selection_policy_sha256",
    }
    if set(policy) != required:
        raise ValueError("selection policy has missing or unexpected fields")
    body = {key: value for key, value in policy.items() if key != "selection_policy_sha256"}
    expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if policy["selection_policy_sha256"] != expected:
        raise ValueError("selection policy hash mismatch")
    if (
        policy["schema_version"] != SELECTION_POLICY_SCHEMA_VERSION
        or policy["stratification_version"] != STRATIFICATION_VERSION
        or policy["accepted_confidences"] != list(ACCEPTED_CONFIDENCES)
    ):
        raise ValueError("unsupported selection policy")
    return policy


def validate_corpus_selection_bundle(selection_manifest: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Independently replay the current Task4 routing and selection provenance."""

    selection = Path(selection_manifest).resolve(strict=True)
    if selection.name != "selection_manifest.jsonl" or selection.parent.name != "corpus":
        raise ValueError("selection must use exact active-run corpus path")
    rejected_path = selection.with_name("rejected_manifest.jsonl")
    jobs_path = selection.with_name("router_jobs.jsonl")
    router_results_path = selection.with_name("router_results.jsonl")
    policy_path = selection.with_name("selection_policy.jsonl")
    if not all(path.is_file() for path in (rejected_path, jobs_path, router_results_path, policy_path)):
        raise ValueError("complete corpus selection bundle is missing")
    jobs = _validate_router_jobs(jobs_path)
    replayed = _replay_router_results(router_results_path, jobs)
    policy = _replay_policy(policy_path)
    selected, rejected = read_jsonl(selection), read_jsonl(rejected_path)
    rows = [*selected, *rejected]
    slugs = [row.get("slug") for row in rows]
    if len(slugs) != len(set(slugs)) or set(slugs) != {job["slug"] for job in jobs}:
        raise ValueError("selected and rejected union must exactly match router jobs")
    jobs_sha = sha256_file(jobs_path)
    results_sha = sha256_file(router_results_path)
    selected_expected, rejected_expected = _selection_decisions(replayed, policy)
    expected_selected_rows, expected_rejected_rows = _manifest_rows(
        selected_expected,
        rejected_expected,
        router_jobs_sha256=jobs_sha,
        router_results_sha256=results_sha,
        selection_policy_sha256=str(policy["selection_policy_sha256"]),
    )
    if selected != expected_selected_rows or rejected != expected_rejected_rows:
        raise ValueError("selection manifests do not match replayed decisions")
    return selected, rejected


def _canonical_result_hash(result: RouterResult) -> str:
    raw = json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _router_result_record(item: IngestedRouterResult) -> dict[str, object]:
    return item.result.model_dump(mode="json") | {
        "raw_result_sha256": item.raw_result_sha256
    }


def _selection_policy(config: CorpusRouterConfig) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": SELECTION_POLICY_SCHEMA_VERSION,
        "allowed_labels": list(config.allowed_labels),
        "accepted_confidences": list(ACCEPTED_CONFIDENCES),
        "target_min": config.target_min,
        "target_max": config.target_max,
        "mandatory_bridge_slugs": sorted(config.bridge_regression_slugs),
        "stratification_version": STRATIFICATION_VERSION,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return body | {
        "selection_policy_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    }


def _policy_value(policy: object, name: str) -> object:
    if isinstance(policy, dict) and name == "bridge_regression_slugs":
        return policy["mandatory_bridge_slugs"]
    return policy[name] if isinstance(policy, dict) else getattr(policy, name)


def ingest_router_results(
    candidates: Sequence[CorpusCandidate], results: Sequence[RouterResult | dict[str, object]],
    *,
    prompt_sha256: str,
    protocol_version: str = "v2",
) -> tuple[IngestedRouterResult, ...]:
    """Validate a complete, one-to-one router response batch against candidates."""

    candidate_by_slug = {candidate.slug: candidate for candidate in candidates}
    if len(candidate_by_slug) != len(candidates):
        raise ValueError("candidate slugs must be unique")
    if any(getattr(candidate, "protocol_version", "v2") != protocol_version for candidate in candidates):
        raise ValueError("router candidates do not match the selected protocol version")
    parsed: list[RouterResult] = []
    _, _, expected_result_schema, _ = _protocol_values(protocol_version)
    seen: set[str] = set()
    for supplied in results:
        try:
            payload = (
                supplied.model_dump(mode="python")
                if isinstance(supplied, RouterResult)
                else supplied
            )
            item = RouterResult.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("invalid router result") from exc
        if item.slug in seen:
            raise ValueError(f"duplicate slug in router results: {item.slug}")
        if item.slug not in candidate_by_slug:
            raise ValueError(f"unexpected slug in router results: {item.slug}")
        candidate = candidate_by_slug[item.slug]
        if item.schema_version != expected_result_schema:
            raise ValueError(f"router result protocol mismatch for {item.slug}")
        if protocol_version == "v3" and item.label.value not in {"GENERAL_RESEARCH", "OTHER"}:
            raise ValueError(f"v3 router result label is not a corpus admission decision: {item.slug}")
        if protocol_version == "v2" and item.label.value not in V2_LABELS:
            raise ValueError(f"v2 router result label is not part of the frozen v2 enum: {item.slug}")
        expected_bindings = {
            "note_sha256": candidate.note_sha256,
            "prompt_sha256": prompt_sha256,
            "job_id": _protocol_job_id(item.slug, candidate.note_sha256, prompt_sha256, protocol_version),
        }
        for field_name, expected_value in expected_bindings.items():
            if getattr(item, field_name) != expected_value:
                raise ValueError(
                    f"router result binding mismatch for {item.slug}: {field_name}"
                )
        seen.add(item.slug)
        parsed.append(item)
    if seen != set(candidate_by_slug):
        missing = sorted(set(candidate_by_slug) - seen)
        raise ValueError(f"router ingestion requires exactly one result per candidate; missing: {', '.join(missing)}")
    return tuple(
        IngestedRouterResult(
            candidate=candidate_by_slug[item.slug], result=item, raw_result_sha256=_canonical_result_hash(item)
        )
        for item in sorted(parsed, key=lambda item: item.slug)
    )


def _selection_record(item: IngestedRouterResult | RejectedCandidate) -> dict[str, object]:
    ingested = item if isinstance(item, IngestedRouterResult) else item
    record = _candidate_record(ingested.candidate) | {
        "job_id": ingested.result.job_id,
        "result_schema_version": ingested.result.schema_version,
        "result_note_sha256": ingested.result.note_sha256,
        "prompt_sha256": ingested.result.prompt_sha256,
        "label": ingested.result.label.value,
        "confidence": ingested.result.confidence,
        "core_mechanism": ingested.result.core_mechanism,
        "scope_reason": ingested.result.scope_reason,
        "evidence_locator": ingested.result.evidence_locator,
        "raw_result_sha256": ingested.raw_result_sha256,
    }
    if isinstance(item, RejectedCandidate):
        record["reason_code"] = item.reason_code
    return record


def _choose_stratified(
    eligible: Sequence[IngestedRouterResult], config: CorpusRouterConfig | dict[str, object]
) -> tuple[IngestedRouterResult, ...]:
    mandatory_slugs = set(_policy_value(config, "bridge_regression_slugs"))
    mandatory = [item for item in eligible if item.candidate.slug in mandatory_slugs]
    target_max = int(_policy_value(config, "target_max"))
    allowed_labels = tuple(_policy_value(config, "allowed_labels"))
    if len(mandatory) > target_max:
        raise ValueError("bridge regression items exceed target_max")
    selected = sorted(mandatory, key=lambda item: item.candidate.slug)
    remaining = [item for item in eligible if item.candidate.slug not in mandatory_slugs]
    groups = {
        label: sorted(
            [item for item in remaining if item.result.label.value == label],
            key=lambda item: item.candidate.slug,
        )
        for label in allowed_labels
    }
    while len(selected) < target_max and any(groups.values()):
        made_progress = False
        for label in allowed_labels:
            if len(selected) >= target_max:
                break
            if groups[label]:
                selected.append(groups[label].pop(0))
                made_progress = True
        if not made_progress:
            break
    return tuple(sorted(selected, key=lambda item: item.candidate.slug))


def _selection_decisions(
    ingested: Sequence[IngestedRouterResult],
    policy: CorpusRouterConfig | dict[str, object],
) -> tuple[tuple[IngestedRouterResult, ...], tuple[RejectedCandidate, ...]]:
    mandatory_slugs = set(_policy_value(policy, "bridge_regression_slugs"))
    by_slug = {item.candidate.slug: item for item in ingested}
    missing_controls = sorted(mandatory_slugs - set(by_slug))
    if missing_controls:
        raise ValueError("missing configured bridge regression candidates: " + ", ".join(missing_controls))
    for slug in mandatory_slugs:
        item = by_slug[slug]
        if item.result.label != CorpusLabel.BRIDGE or item.result.confidence not in ACCEPTED_CONFIDENCES:
            raise ValueError(f"bridge regression candidate is invalid: {slug}")
    allowed_labels = set(_policy_value(policy, "allowed_labels"))
    accepted_confidences = set(
        _policy_value(policy, "accepted_confidences")
        if isinstance(policy, dict)
        else ACCEPTED_CONFIDENCES
    )
    eligible: list[IngestedRouterResult] = []
    rejected: list[RejectedCandidate] = []
    for item in ingested:
        if item.candidate.source_status in {"SOURCE_WITHDRAWN", "SOURCE_REJECTED", "SOURCE_UNAVAILABLE", "SOURCE_STATUS_PENDING"}:
            reason = {
                "SOURCE_WITHDRAWN": "SOURCE_WITHDRAWN",
                "SOURCE_REJECTED": "SOURCE_REJECTED",
                "SOURCE_UNAVAILABLE": "SOURCE_UNAVAILABLE",
                "SOURCE_STATUS_PENDING": "SOURCE_STATUS_PENDING",
            }[item.candidate.source_status]
        elif item.result.label == CorpusLabel.OTHER:
            reason = "LABEL_OTHER"
        elif item.result.label.value not in allowed_labels:
            reason = "LABEL_NOT_ALLOWED"
        elif item.result.confidence not in accepted_confidences:
            reason = "CONFIDENCE_LOW"
        else:
            eligible.append(item)
            continue
        rejected.append(RejectedCandidate(item.candidate, item.result, item.raw_result_sha256, reason))
    unique_eligible: list[IngestedRouterResult] = []
    occupied_content_ids: set[str] = set()
    for item in sorted(eligible, key=lambda row: row.candidate.slug):
        identity = item.candidate.source_pdf_sha256 if item.candidate.source_status == "PRIMARY_SOURCE_VERIFIED" else None
        if identity and identity in occupied_content_ids:
            rejected.append(RejectedCandidate(item.candidate, item.result, item.raw_result_sha256, "SOURCE_ALIAS_DUPLICATE"))
            continue
        if identity:
            occupied_content_ids.add(identity)
        unique_eligible.append(item)
    eligible = unique_eligible
    target_min = int(_policy_value(policy, "target_min"))
    if len(eligible) < target_min:
        raise ValueError(f"eligible corpus ({len(eligible)}) is below target_min ({target_min})")
    selected = _choose_stratified(eligible, policy)
    selected_slugs = {item.candidate.slug for item in selected}
    for item in eligible:
        if item.candidate.slug not in selected_slugs:
            rejected.append(RejectedCandidate(item.candidate, item.result, item.raw_result_sha256, "TARGET_MAX"))
    if len(selected) < target_min:
        raise ValueError(f"selected corpus ({len(selected)}) is below target_min ({target_min})")
    return selected, tuple(sorted(rejected, key=lambda entry: entry.candidate.slug))


def _manifest_rows(
    selected: Sequence[IngestedRouterResult],
    rejected: Sequence[RejectedCandidate],
    *,
    router_jobs_sha256: str,
    router_results_sha256: str,
    selection_policy_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    bindings = {
        "router_jobs_sha256": router_jobs_sha256,
        "router_results_sha256": router_results_sha256,
        "selection_policy_sha256": selection_policy_sha256,
    }
    return (
        [_selection_record(item) | {"reason_code": "ELIGIBLE"} | bindings for item in selected],
        [_selection_record(item) | bindings for item in rejected],
    )


def select_pilot(
    config: CorpusRouterConfig,
    ingested: Sequence[IngestedRouterResult],
    active_run_dir: Path,
) -> tuple[tuple[IngestedRouterResult, ...], tuple[RejectedCandidate, ...]]:
    """Select and publish a staged two-manifest bundle.

    Cross-file replacement cannot be crash-atomic. Consumers must require both
    manifests and validate their job/note/prompt bindings before using either.
    """

    owned_paths = _owned_corpus_paths(active_run_dir)
    selection_path = owned_paths["selection"]
    rejected_path = owned_paths["rejected"]
    selection_path.unlink(missing_ok=True)
    rejected_path.unlink(missing_ok=True)
    if not owned_paths["jobs"].is_file():
        raise ValueError("selection requires current run-local router_jobs.jsonl")
    validated_jobs = _validate_router_jobs(owned_paths["jobs"])
    jobs_by_slug = {job["slug"]: job for job in validated_jobs}
    if set(jobs_by_slug) != {item.candidate.slug for item in ingested}:
        raise ValueError("router jobs must exactly match ingested candidates")
    for item in ingested:
        if item.raw_result_sha256 != _canonical_result_hash(item.result):
            raise ValueError("router result raw hash mismatch")
        expected_candidate = _candidate_record(item.candidate)
        if any(
            key != "record_id" and jobs_by_slug[item.candidate.slug].get(key) != value
            for key, value in expected_candidate.items()
        ):
            raise ValueError("router job candidate provenance mismatch")
        job = jobs_by_slug[item.candidate.slug]
        if (
            item.result.job_id != job["job_id"]
            or item.result.note_sha256 != job["note_sha256"]
            or item.result.prompt_sha256 != job["prompt_sha256"]
            or item.result.schema_version != job["result_schema_version"]
        ):
            raise ValueError("router result does not bind current router job")
    router_jobs_sha256 = sha256_file(owned_paths["jobs"])
    router_result_records = [
        _router_result_record(item)
        for item in sorted(ingested, key=lambda entry: entry.candidate.slug)
    ]
    policy = _selection_policy(config)
    write_jsonl_bundle(
        {
            owned_paths["router_results"]: router_result_records,
            owned_paths["policy"]: [policy],
        }
    )
    router_results_sha256 = sha256_file(owned_paths["router_results"])
    selected, rejected = _selection_decisions(ingested, policy)
    selection_rows, rejected_rows = _manifest_rows(
        selected,
        rejected,
        router_jobs_sha256=router_jobs_sha256,
        router_results_sha256=router_results_sha256,
        selection_policy_sha256=str(policy["selection_policy_sha256"]),
    )
    write_jsonl_bundle(
        {
            selection_path: selection_rows,
            rejected_path: rejected_rows,
        }
    )
    return selected, rejected
