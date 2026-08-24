"""Deterministic, auditable routing and selection for the KV-memory pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Literal, Sequence

from pydantic import ValidationError

from .artifacts import read_jsonl, sha256_file, stable_id, write_jsonl, write_jsonl_bundle
from .models import CorpusLabel, FrozenStrictModel, NonEmptyStr, Sha256Hex


Confidence = Literal["HIGH", "MEDIUM", "LOW"]
PROMPT_VERSION = "idea_factory.corpus_router_prompt.v1"
ROUTER_JOB_SCHEMA_VERSION = "idea_factory.corpus_router_job.v1"
ROUTER_RESULT_SCHEMA_VERSION = "idea_factory.corpus_router_result.v1"
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
        "label": {"enum": ["KV_CACHE", "LONG_MEMORY", "BRIDGE", "OTHER"]},
        "core_mechanism": {"type": "string", "minLength": 1},
        "scope_reason": {"type": "string", "minLength": 1},
        "evidence_locator": {"type": "string", "minLength": 1},
        "confidence": {"enum": ["HIGH", "MEDIUM", "LOW"]},
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
        eligible_labels = {
            CorpusLabel.KV_CACHE.value,
            CorpusLabel.LONG_MEMORY.value,
            CorpusLabel.BRIDGE.value,
        }
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
        if set(payload) != expected:
            raise ValueError("corpus router config has missing or unexpected fields")

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


class RouterResult(FrozenStrictModel):
    schema_version: Literal["idea_factory.corpus_router_result.v1"]
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
    discovered: dict[str, tuple[Path, list[str]]] = {}
    for source_list in preflighted_lists:
        source = str(source_list)
        for path in _read_candidate_paths(source_list):
            if _is_audit(path):
                continue
            if path == config.legacy_ledger:
                continue
            if not any(_is_within(path, root) for root in allowed_roots):
                raise ValueError(f"candidate is outside configured candidate roots: {path}")
            slug = _slug_for(path)
            if not path.is_file():
                raise ValueError(f"missing base note for {slug}: {path}")
            existing = discovered.get(slug)
            if existing is None:
                discovered[slug] = (path, [source])
            else:
                previous_path, memberships = existing
                if previous_path != path:
                    raise ValueError(f"duplicate slug {slug} points to different base files")
                if source not in memberships:
                    memberships.append(source)
    missing_controls = sorted(set(config.bridge_regression_slugs) - set(discovered))
    if missing_controls:
        raise ValueError(
            "missing configured bridge regression candidates: "
            + ", ".join(missing_controls)
        )
    return tuple(
        CorpusCandidate(
            slug=slug,
            note_path=path,
            source_lists=tuple(sorted(memberships)),
            note_sha256=sha256_file(path),
            source_list_bindings=tuple((source, list_hashes[source]) for source in sorted(memberships)),
            candidate_list_manifest=list_manifest,
            allowed_note_root=str(config.notes_root),
            allowed_papers_root=str(corpus_root),
            legacy_ledger_path=str(config.legacy_ledger),
        )
        for slug, (path, memberships) in sorted(discovered.items())
    )


def _candidate_record(candidate: CorpusCandidate) -> dict[str, object]:
    return {
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


def write_router_jobs(
    candidates: Sequence[CorpusCandidate],
    active_run_dir: Path,
    *,
    prompt_path: Path,
) -> Path:
    """Replace the run-local job file, never modifying caller-owned source lists."""

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
        job_id = router_job_id(candidate.slug, candidate.note_sha256, prompt_sha256)
        records.append(
            _candidate_record(candidate)
            | {
                "record_id": job_id,
                "schema_version": ROUTER_JOB_SCHEMA_VERSION,
                "job_id": job_id,
                "note_text": note_text,
                "prompt_id": "corpus_router",
                "prompt_version": PROMPT_VERSION,
                "prompt_path": str(resolved_prompt),
                "prompt_sha256": prompt_sha256,
                "prompt_text": prompt_text,
                "result_schema_version": ROUTER_RESULT_SCHEMA_VERSION,
                "required_output_schema": REQUIRED_OUTPUT_SCHEMA,
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
    for job in jobs:
        if job.get("schema_version") != ROUTER_JOB_SCHEMA_VERSION or job.get("result_schema_version") != ROUTER_RESULT_SCHEMA_VERSION:
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
        expected_id = router_job_id(str(job.get("slug")), str(job.get("note_sha256")), prompt_sha)
        if (
            job.get("prompt_sha256") != prompt_sha
            or job.get("prompt_text") != prompt_text
            or job.get("prompt_id") != "corpus_router"
            or job.get("prompt_version") != PROMPT_VERSION
            or job.get("required_output_schema") != REQUIRED_OUTPUT_SCHEMA
            or job.get("job_id") != expected_id
            or job.get("record_id") != expected_id
        ):
            raise ValueError("router job prompt or ID binding mismatch")
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
) -> tuple[IngestedRouterResult, ...]:
    """Validate a complete, one-to-one router response batch against candidates."""

    candidate_by_slug = {candidate.slug: candidate for candidate in candidates}
    if len(candidate_by_slug) != len(candidates):
        raise ValueError("candidate slugs must be unique")
    parsed: list[RouterResult] = []
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
        expected_bindings = {
            "note_sha256": candidate.note_sha256,
            "prompt_sha256": prompt_sha256,
            "job_id": router_job_id(item.slug, candidate.note_sha256, prompt_sha256),
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
        if item.result.label == CorpusLabel.OTHER:
            reason = "LABEL_OTHER"
        elif item.result.label.value not in allowed_labels:
            reason = "LABEL_NOT_ALLOWED"
        elif item.result.confidence not in accepted_confidences:
            reason = "CONFIDENCE_LOW"
        else:
            eligible.append(item)
            continue
        rejected.append(RejectedCandidate(item.candidate, item.result, item.raw_result_sha256, reason))
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
