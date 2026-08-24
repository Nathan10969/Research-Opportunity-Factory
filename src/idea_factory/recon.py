"""Task-9 external reconnaissance handoff and strict evidence ingestion.

This module deliberately emits commands instead of making network requests.  A
separate runtime may use the jobs only after notification preflight passes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import ValidationError, field_validator, model_validator

from .artifacts import read_jsonl, sha256_file, stable_id, write_json, write_jsonl_bundle
from .corpus import CorpusRouterConfig
from .ledger import DEDUP_POLICY_VERSION, validate_dedup_bundle
from .models import FrozenStrictModel, Opportunity, ReconDecision, ReconQuery
from .opportunities import _anchored_run, _owned_dir, _owned_file, _safe_unlink


QUERY_POLICY_VERSION = "idea_factory.recon_query_policy.v1"
METHOD_FIREWALL_VERSION = "idea_factory.method_model_firewall.v1"
COMMAND_POLICY_VERSION = "idea_factory.recon_command_policy.v4"
RECON_PROTOCOL_VERSION = "idea_factory.recon_protocol.v3"
EXECUTION_CONTEXT_POLICY_VERSION = "idea_factory.recon_execution_context.v5"
UV_EXECUTABLE_POLICY_VERSION = "idea_factory.uv_executable_identity.v2"
UNSIGNED_NOTICE_RECORD_KIND = "UNSIGNED_TERMS_NOTIFICATION_RECORD"
UNSIGNED_CONTEXT_RECORD_KIND = "UNSIGNED_NOTIFICATION_INTEGRITY"
UNSIGNED_AUTHENTICITY = "NOT_AUTHENTICATED"
ARXIV_SCRIPT = "literature_search_arxiv/scripts/search_arxiv.py"
OPENALEX_SCRIPT = "literature_search_openalex/scripts/openalex_cli.py"
ARXIV_URL = "https://info.arxiv.org/help/api/index.html"
OPENALEX_URL = "https://developers.openalex.org/"
TERMS_ACKNOWLEDGEMENT = "I acknowledge the API terms and paper-license restrictions."
NO_DIRECT_COVERAGE_REASON = "No direct coverage found under this query pack. This result is bounded to the supplied queries and execution receipts."
OPERATOR_ATTESTATION_USE = "IDEA_FACTORY_TASK9_RECON_EXECUTION"
ARXIV_OUTPUT_CONTRACT = "CUMULATIVE_WRAPPERS_OR_EMPTY_STDOUT_ON_ZERO_V1"
OPENALEX_OUTPUT_CONTRACT = "OPENALEX_JSON_OBJECT_WRAPPER_V1"
TOOL_OUTPUT_CONTRACTS = {"ARXIV": ARXIV_OUTPUT_CONTRACT, "OPENALEX": OPENALEX_OUTPUT_CONTRACT}
_LANES = ("CONCEPT", "MECHANISM", "FAILURE", "EVALUATION")
_VARIANTS = ("CURRENT_TERMS", "GENERIC_SHAPE")
_MEASURE = re.compile(r"\b(measure|metric|recall|accuracy|latency|loss|rate|error|throughput|evaluate|test)\w*\b", re.I)
_NAMED_METHOD = re.compile(r"\b(?:called|named|dubbed)\s+[A-Za-z][A-Za-z0-9_-]*", re.I)
_METHOD_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?![A-Za-z0-9])")
_CONTROLLED_MODEL_STEM = re.compile(r"^(?:transformer(?:xl|base|large|small|tiny|\d+)|bert(?:base|large|small|tiny|\d+)|gpt\d+|llama\d+|lora\d+|qlora\d+|dpo\d+|ppo\d+)$", re.I)
_CONTROLLED_LETTER_NUMBER_MODEL = re.compile(r"^t\d+(?:base|small|large|xl)?$", re.I)
_CONTROLLED_METHOD_TERMS = (
    "adam", "bert", "claude", "deepseek", "dpo", "flashattention", "gemini",
    "gpt", "gpt-2", "gpt-3", "gpt-4", "llama", "lora", "mamba", "mistral",
    "phi", "ppo", "qlora", "qwen", "roberta", "transformer",
)
_DOMAIN_TERM_ALLOWLIST = ("kv", "rag")
METHOD_FIREWALL_POLICY = {
    "version": METHOD_FIREWALL_VERSION,
    "policy_kind": "deterministic-controlled-vocabulary-and-style-rules",
    "controlled_method_terms": list(_CONTROLLED_METHOD_TERMS),
    "domain_term_allowlist": list(_DOMAIN_TERM_ALLOWLIST),
    "named_method_pattern": _NAMED_METHOD.pattern,
    "token_pattern": _METHOD_TOKEN.pattern,
    "controlled_model_stem_pattern": _CONTROLLED_MODEL_STEM.pattern,
    "controlled_letter_number_model_pattern": _CONTROLLED_LETTER_NUMBER_MODEL.pattern,
    "style_rules": ["unallowlisted-uppercase-acronym-length-at-least-3", "mixed-case-token", "net-or-former-suffix"],
    "boundary": "deterministic lexical firewall; not semantic omniscience",
}
METHOD_FIREWALL_POLICY_SHA256 = hashlib.sha256(
    json.dumps(METHOD_FIREWALL_POLICY, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON object key: {key}")
        out[key] = value
    return out


def _strict_load(raw: str | bytes, *, root: type = dict) -> Any:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    if type(text) is not str:
        raise TypeError("external result must be JSON text or bytes")
    try:
        value = json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("external result must be strict JSON") from exc
    if type(value) is not root:
        raise ValueError(f"external result must be one JSON {root.__name__}")
    return value


def _strict_json_stream(raw: str | bytes) -> list[dict[str, Any]]:
    """Decode adjacent native JSON objects while retaining strict object hooks."""
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise ValueError("external result must be UTF-8 JSON") from exc
    if type(text) is not str:
        raise TypeError("external result must be JSON text or bytes")
    if not text.strip():
        raise ValueError("arXiv raw output is empty")
    decoder = json.JSONDecoder(
        object_pairs_hook=_strict_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number {value}")
        ),
    )
    values: list[dict[str, Any]] = []
    offset = 0
    try:
        while offset < len(text):
            while offset < len(text) and text[offset].isspace():
                offset += 1
            if offset == len(text):
                break
            value, offset = decoder.raw_decode(text, offset)
            if type(value) is not dict:
                raise ValueError("arXiv output requires wrapper objects")
            values.append(value)
    except (json.JSONDecodeError, ValueError) as exc:
        if "wrapper objects" in str(exc):
            raise
        raise ValueError("arXiv output must be a strict JSON wrapper stream") from exc
    return values


def _safe_text(value: object) -> str:
    if type(value) is not str:
        raise ValueError("metadata text must be a native string")
    return " ".join(value.split())


def _validated_url(value: object) -> str:
    url = _safe_text(value)
    if not re.match(r"^https?://[^\s]+$", url, re.I):
        raise ValueError("evidence URL must be an absolute HTTP(S) URL")
    return url


def _evidence_id(*, source: str, source_id: str, title: str, year: int | None, doi: str | None, url: str | None, authors: Sequence[str], abstract_text: str) -> str:
    content = {"source": source, "source_id": source_id, "title": title, "year": year, "doi": doi, "url": url, "authors": list(authors), "abstract_text": abstract_text}
    return stable_id("recon_evidence", _hash(content))


def _date_year(value: object) -> int | None:
    if type(value) is int and 1000 <= value <= 9999:
        return value
    if type(value) is str and re.match(r"^\d{4}", value):
        return int(value[:4])
    return None


def _canonicalize_doi(value: str) -> str:
    if type(value) is not str:
        raise ValueError("DOI must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"^doi\s*:\s*", "", normalized, flags=re.I)
    parsed = urlsplit(normalized)
    if parsed.scheme.casefold() in {"http", "https"}:
        if (parsed.hostname or "").casefold() not in {"doi.org", "www.doi.org", "dx.doi.org"}:
            raise ValueError("DOI URL must use a recognized doi.org resolver")
        normalized = parsed.path.lstrip("/")
    normalized = unicodedata.normalize("NFKC", normalized).strip().casefold()
    if not normalized:
        raise ValueError("DOI must be nonblank")
    return normalized


class ReconEvidence(FrozenStrictModel):
    schema_version: Literal["idea_factory.recon_evidence.v1"] = "idea_factory.recon_evidence.v1"
    evidence_id: str
    source: Literal["ARXIV", "OPENALEX"]
    source_id: str
    title: str
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    authors: tuple[str, ...] = ()
    abstract_text: str = ""
    query_id: str
    raw_file_sha: str
    source_aliases: tuple[str, ...] = ()
    query_ids: tuple[str, ...] = ()

    @field_validator("evidence_id", "source_id", "title", "query_id", "raw_file_sha")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("required evidence strings must be nonblank")
        return value.strip()

    @field_validator("doi")
    @classmethod
    def _canonical_doi(cls, value: str | None) -> str | None:
        return _canonicalize_doi(value) if value is not None else None


class _BoundNearestPrior(FrozenStrictModel):
    """A report-facing prior must point at a record, never free-form prose."""

    paper: str
    exact_overlap: str
    residual_difference: str
    evidence_url_or_id: str

    @field_validator("paper", "exact_overlap", "residual_difference", "evidence_url_or_id")
    @classmethod
    def _required(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("nearest-prior fields must be nonblank strings")
        return value.strip()


class _ReportInput(FrozenStrictModel):
    schema_version: Literal["idea_factory.recon_report_result.v1"]
    job_id: str
    opportunity_id: str
    query_pack_sha256: str
    normalized_evidence_manifest_sha256: str
    protocol_hash: str
    cache_key: str
    searched_query_ids: list[str]
    searched_at: datetime
    nearest_priors: list[_BoundNearestPrior]
    decision: ReconDecision
    decision_reason: str

    @model_validator(mode="after")
    def _timezone(self) -> "_ReportInput":
        if self.searched_at.tzinfo is None or self.searched_at.utcoffset() is None:
            raise ValueError("searched_at must be timezone-aware")
        return self


def normalize_openalex_abstract(index: Mapping[str, object] | None) -> str:
    """Rebuild an OpenAlex inverted abstract without guessing missing tokens."""
    if index is None:
        return ""
    if not isinstance(index, Mapping):
        raise ValueError("abstract_inverted_index must be an object or null")
    words: dict[int, str] = {}
    for word, positions in index.items():
        if type(word) is not str or not word or type(positions) is not list:
            raise ValueError("invalid OpenAlex abstract index")
        for position in positions:
            if type(position) is not int or position < 0 or position in words:
                raise ValueError("OpenAlex abstract positions must be unique nonnegative integers")
            words[position] = word
    if not words:
        return ""
    if set(words) != set(range(max(words) + 1)):
        raise ValueError("OpenAlex abstract positions must be contiguous")
    return " ".join(words[position] for position in range(max(words) + 1))


def _terms_text(source: str, timestamp: str, acknowledgement: str) -> str:
    url = ARXIV_URL if source == "ARXIV" else OPENALEX_URL
    return (
        f"Recon live-search terms notification\nRecord kind: {UNSIGNED_NOTICE_RECORD_KIND}\nAuthenticity: {UNSIGNED_AUTHENTICITY}\nSource: {source}\nAPI terms: {url}\n"
        "Paper-license restrictions: retrieved papers and metadata remain subject to their source licenses; do not redistribute or train on restricted text.\n"
        "Boundary: runtime orchestration is responsible for actually notifying the user; this unsigned file only records notification integrity and does not authenticate identity or consent.\n"
        f"User acknowledgement: {acknowledgement}\nNotified at: {timestamp}\n"
    )


def _exact_workspace_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    if not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("workspace root must have exact resolved identity")
    return root


def _licenses_directory(root: Path, *, create: bool) -> Path | None:
    workspace = _exact_workspace_root(root)
    licenses = workspace / ".licenses"
    is_junction = getattr(licenses, "is_junction", lambda: False)
    if licenses.is_symlink() or is_junction():
        raise ValueError(".licenses must not be a symlink or junction")
    if not licenses.exists():
        if not create:
            return None
        licenses.mkdir()
    if not licenses.is_dir() or licenses.resolve(strict=True) != licenses or licenses.parent.resolve(strict=True) != workspace:
        raise ValueError(".licenses directory escapes workspace or is not anchored")
    for file_name in ("literature_search_arxiv_LICENSE.txt", "literature_search_openalex_LICENSE.txt"):
        path = licenses / file_name
        if path.is_symlink():
            raise ValueError("notice path must not be a symlink")
        if path.exists() and (not path.is_file() or path.resolve(strict=True) != path):
            raise ValueError("notice path escapes .licenses or is not an anchored file")
    return licenses


def record_terms_notification(run_or_repo_root: Path, *, user_notified_at: str | datetime, acknowledgement: str) -> dict[str, Path]:
    """Write an unsigned notification record after orchestration notified the user.

    This integrity artifact does not authenticate user identity or authorization.
    """
    timestamp = datetime.fromisoformat(user_notified_at) if type(user_notified_at) is str else user_notified_at
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("user_notified_at must be a timezone-aware ISO-8601 timestamp")
    if type(acknowledgement) is not str or acknowledgement.strip() != TERMS_ACKNOWLEDGEMENT:
        raise ValueError("exact terms acknowledgement is required")
    root = _exact_workspace_root(Path(run_or_repo_root))
    notices = _licenses_directory(root, create=True)
    if notices is None:  # pragma: no cover - create=True guarantees a directory
        raise ValueError(".licenses directory was not created")
    stamp = timestamp.isoformat()
    paths = {
        "ARXIV": notices / "literature_search_arxiv_LICENSE.txt",
        "OPENALEX": notices / "literature_search_openalex_LICENSE.txt",
    }
    staged: dict[str, Path] = {}
    try:
        for source, path in paths.items():
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=notices, prefix=f".{path.name}.", suffix=".tmp", delete=False, newline="\n") as handle:
                handle.write(_terms_text(source, stamp, acknowledgement)); handle.flush(); os.fsync(handle.fileno())
                staged[source] = Path(handle.name)
        for source, path in paths.items():
            current = _licenses_directory(root, create=False)
            if current != notices or staged[source].resolve(strict=True).parent != notices:
                raise ValueError(".licenses changed during notice publication")
            os.replace(staged.pop(source), path)
    finally:
        for path in staged.values(): path.unlink(missing_ok=True)
    return {source.lower(): path for source, path in paths.items()}


def _notice_timestamp(path: Path) -> datetime:
    matches = re.findall(r"^Notified at: (.+)$", path.read_text(encoding="utf-8"), re.M)
    if len(matches) != 1:
        raise ValueError("notice must contain exactly one Notified at timestamp")
    try:
        timestamp = datetime.fromisoformat(matches[0])
    except ValueError as exc:
        raise ValueError("notice Notified at timestamp is invalid") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("notice Notified at timestamp must be timezone-aware")
    return timestamp


def _notice_status(root: Path) -> tuple[dict[str, str], list[str]]:
    try:
        root = _exact_workspace_root(root)
        notices = _licenses_directory(root, create=False)
    except (OSError, ValueError):
        return {}, ["LICENSES_PATH_INVALID"]
    hashes: dict[str, str] = {}
    missing: list[str] = []
    if notices is None:
        return {}, ["NOTICE_ARXIV", "NOTICE_OPENALEX"]
    for source, file_name, url in (("ARXIV", "literature_search_arxiv_LICENSE.txt", ARXIV_URL), ("OPENALEX", "literature_search_openalex_LICENSE.txt", OPENALEX_URL)):
        path = notices / file_name
        if not path.is_file():
            missing.append(f"NOTICE_{source}")
            continue
        if path.resolve(strict=True) != path:
            missing.append(f"NOTICE_{source}_PATH_INVALID")
            continue
        content = path.read_text(encoding="utf-8")
        try:
            _notice_timestamp(path)
            timestamp_valid = True
        except ValueError:
            timestamp_valid = False
        if url not in content or TERMS_ACKNOWLEDGEMENT not in content or f"Record kind: {UNSIGNED_NOTICE_RECORD_KIND}" not in content or f"Authenticity: {UNSIGNED_AUTHENTICITY}" not in content or "Paper-license restrictions:" not in content or not timestamp_valid:
            missing.append(f"NOTICE_{source}_MISMATCH")
            continue
        hashes[source] = sha256_file(path)
    return hashes, missing


def _skill_path(skill_roots: Mapping[str, Path | str], source: str) -> Path | None:
    key = "arxiv" if source == "ARXIV" else "openalex"
    root = skill_roots.get(key) or skill_roots.get(source) or skill_roots.get(f"literature_search_{key}")
    return Path(root) if root is not None else None


def _uv_policy(path: Path) -> str | None:
    name = path.name.casefold()
    if name in {"uv", "uv.exe"}:
        return "NATIVE_UV_EXECUTABLE"
    return None


def _run_uv_version(path: Path, policy: str) -> str:
    if policy != "NATIVE_UV_EXECUTABLE":
        raise ValueError("uv must be a native uv/uv.exe executable")
    completed = subprocess.run([str(path), "--version"], capture_output=True, text=True, shell=False, timeout=10)
    version = completed.stdout.strip()
    match = re.fullmatch(
        r"uv\s+\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?"
        r"(?: \((?P<commit>[0-9a-f]{7,40}) (?P<date>\d{4}-\d{2}-\d{2}) "
        r"(?P<target>[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+){2,})\))?",
        version,
    )
    if completed.returncode or match is None:
        raise ValueError("uv --version did not return strict uv semver identity")
    if match.group("date") is not None:
        try:
            datetime.strptime(match.group("date"), "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("uv --version did not return strict uv semver identity") from exc
    return version


def check_live_recon_preflight(run_or_repo_root: Path, skill_roots: Mapping[str, Path | str] | None = None, *, uv_executable: Path | str | None = None) -> dict[str, Any]:
    """Return all blockers without mutation or installing ``uv``."""
    root = Path(os.path.abspath(run_or_repo_root))
    roots = skill_roots or {}
    missing: list[str] = []
    uv_candidate = Path(uv_executable) if uv_executable is not None else Path(found) if (found := shutil.which("uv")) else None
    uv_path: str | None = None
    uv_version: str | None = None
    uv_sha256: str | None = None
    uv_policy: str | None = None
    if uv_candidate is None or not uv_candidate.is_file():
        missing.append("UV_NOT_FOUND")
    else:
        try:
            resolved_uv = uv_candidate.resolve(strict=True)
            uv_policy = _uv_policy(resolved_uv)
            if uv_policy is None:
                missing.append("UV_IDENTITY_INVALID")
                raise ValueError("uv executable basename is invalid")
            uv_version = _run_uv_version(resolved_uv, uv_policy)
            uv_path, uv_sha256 = str(resolved_uv), sha256_file(resolved_uv)
        except (OSError, subprocess.SubprocessError, ValueError):
            if "UV_IDENTITY_INVALID" not in missing:
                missing.append("UV_VERSION_UNAVAILABLE")
    notice_hashes, notice_missing = _notice_status(root)
    missing.extend(notice_missing)
    scripts: dict[str, str | None] = {}
    for source, rel in (("ARXIV", ARXIV_SCRIPT), ("OPENALEX", OPENALEX_SCRIPT)):
        skill = _skill_path(roots, source)
        script = skill / "scripts" / Path(rel).name if skill else None
        # accept either the skill root or the parent that contains the documented relative path
        if skill and not script.is_file():
            candidate = skill / rel
            script = candidate if candidate.is_file() else script
        scripts[source] = str(script.resolve(strict=True)) if script and script.is_file() else None
        if scripts[source] is None:
            missing.append(f"SCRIPT_{source}")
    return {"ok": not missing, "missing": sorted(missing), "uv": uv_path, "uv_version": uv_version, "uv_sha256": uv_sha256, "uv_executable_policy": uv_policy, "notice_hashes": notice_hashes, "scripts": scripts}


def _aware_timestamp(value: str | datetime, field: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value) if type(value) is str else value
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware ISO-8601 timestamp") from exc
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware ISO-8601 timestamp")
    return timestamp


def _directory_manifest_sha256(root: Path) -> str:
    anchored = Path(os.path.abspath(root))
    if not anchored.is_dir() or anchored.resolve(strict=True) != anchored:
        raise ValueError("attested skill root must have exact resolved identity")
    records: list[dict[str, str]] = []
    for path in sorted(anchored.rglob("*")):
        is_junction = getattr(path, "is_junction", lambda: False)
        if path.is_symlink() or is_junction() or not path.resolve(strict=True).is_relative_to(anchored):
            raise ValueError("attested skill root contains a link or escaping path")
        relative = path.relative_to(anchored).as_posix()
        if path.is_dir():
            records.append({"path": relative, "type": "directory"})
        elif path.is_file():
            records.append({"path": relative, "type": "file", "sha256": sha256_file(path)})
        else:
            raise ValueError("attested skill root contains an unsupported entry")
    return _hash(records)


def _validate_operator_attestation_record(
    record: Mapping[str, Any],
    *,
    allow_test_attestation: bool,
    expected_workspace_root: Path | None = None,
    expected_skill_roots: Mapping[str, Path | str] | None = None,
    expected_uv_executable: Path | str | None = None,
    require_hash: bool,
) -> dict[str, Any]:
    body_keys = {"schema_version", "record_kind", "attestation_scope", "authenticity", "boundary", "allowed_use", "acknowledged_at", "workspace_root", "uv", "scripts"}
    expected_keys = body_keys | ({"operator_trust_attestation_sha256"} if require_hash else set())
    if type(record) is not dict or set(record) != expected_keys:
        raise ValueError("operator trust attestation schema is not exact")
    data = dict(record)
    if require_hash:
        claimed_hash = data.pop("operator_trust_attestation_sha256")
        if type(claimed_hash) is not str or claimed_hash != _hash(data):
            raise ValueError("operator trust attestation hash mismatch")
    if data.get("schema_version") != "idea_factory.operator_trust_attestation.v1" or data.get("record_kind") != "OPERATOR_TRUST_ATTESTATION":
        raise ValueError("operator trust attestation policy mismatch")
    scope = data.get("attestation_scope")
    if scope not in {"LIVE_OPERATOR_ATTESTED", "TEST_ONLY"}:
        raise ValueError("operator trust attestation scope is invalid")
    expected_boundary = "TEST_ONLY/NOT_AUTHENTICATED" if scope == "TEST_ONLY" else "LIVE_OPERATOR_ATTESTED/NOT_AUTHENTICATED"
    if data.get("authenticity") != "NOT_AUTHENTICATED" or data.get("boundary") != expected_boundary:
        raise ValueError("operator trust attestation authenticity boundary mismatch")
    if scope == "TEST_ONLY" and not allow_test_attestation:
        raise ValueError("TEST_ONLY operator trust attestation requires explicit opt-in")
    if data.get("allowed_use") != OPERATOR_ATTESTATION_USE:
        raise ValueError("operator trust attestation allowed_use mismatch")
    _aware_timestamp(data.get("acknowledged_at"), "acknowledged_at")
    workspace = Path(str(data.get("workspace_root")))
    if not workspace.is_dir() or workspace.resolve(strict=True) != workspace:
        raise ValueError("operator trust attestation workspace root mismatch")
    if expected_workspace_root is not None and workspace != Path(os.path.abspath(expected_workspace_root)):
        raise ValueError("operator trust attestation workspace root mismatch")
    uv = data.get("uv")
    if type(uv) is not dict or set(uv) != {"path", "sha256", "executable_policy"}:
        raise ValueError("operator trust attestation uv schema mismatch")
    uv_path = Path(str(uv.get("path")))
    if not uv_path.is_file() or uv_path.resolve(strict=True) != uv_path or _uv_policy(uv_path) != "NATIVE_UV_EXECUTABLE" or uv.get("executable_policy") != "NATIVE_UV_EXECUTABLE" or uv.get("sha256") != sha256_file(uv_path):
        raise ValueError("operator trust attestation uv binding mismatch")
    if expected_uv_executable is not None and uv_path != Path(expected_uv_executable).resolve(strict=True):
        raise ValueError("operator trust attestation uv binding mismatch")
    scripts = data.get("scripts")
    if type(scripts) is not dict or set(scripts) != {"ARXIV", "OPENALEX"}:
        raise ValueError("operator trust attestation script schema mismatch")
    for source, package_name, script_name in (
        ("ARXIV", "literature_search_arxiv", "search_arxiv.py"),
        ("OPENALEX", "literature_search_openalex", "openalex_cli.py"),
    ):
        binding = scripts.get(source)
        if type(binding) is not dict or set(binding) != {"package_name", "skill_root", "skill_root_sha256", "path", "sha256"} or binding.get("package_name") != package_name:
            raise ValueError(f"operator trust attestation {source} script schema mismatch")
        root, script = Path(str(binding.get("skill_root"))), Path(str(binding.get("path")))
        if not root.is_dir() or root.resolve(strict=True) != root or not script.is_file() or script.resolve(strict=True) != script or not script.is_relative_to(root) or script.name != script_name:
            raise ValueError(f"operator trust attestation {source} path mismatch")
        if scope == "LIVE_OPERATOR_ATTESTED" and root.name != package_name:
            raise ValueError("live operator trust attestation requires exact installed package names")
        if binding.get("skill_root_sha256") != _directory_manifest_sha256(root) or binding.get("sha256") != sha256_file(script):
            raise ValueError(f"operator trust attestation {source} root hash mismatch")
        if expected_skill_roots is not None:
            expected_root = _skill_path(expected_skill_roots, source)
            if expected_root is None or root != Path(expected_root).resolve(strict=True):
                raise ValueError(f"operator trust attestation {source} root mismatch")
    return data | ({"operator_trust_attestation_sha256": _hash(data)} if require_hash else {})


def record_operator_trust_attestation(run_dir: Path, attestation: Mapping[str, Any]) -> dict[str, Any]:
    run = _anchored_run(Path(run_dir)); recon = _recon_dir(run, create=False)
    body = _validate_operator_attestation_record(attestation, allow_test_attestation=True, require_hash=False)
    record = body | {"operator_trust_attestation_sha256": _hash(body)}
    write_json(recon / "operator_trust_attestation.json", record)
    return record


def validate_operator_trust_attestation(
    recon_or_run_dir: Path,
    *,
    allow_test_attestation: bool = False,
    expected_workspace_root: Path | None = None,
    expected_skill_roots: Mapping[str, Path | str] | None = None,
    expected_uv_executable: Path | str | None = None,
) -> dict[str, Any]:
    candidate = Path(recon_or_run_dir)
    recon = _recon_dir(_anchored_run(candidate), create=False) if candidate.name != "recon" else candidate
    if not (recon / "operator_trust_attestation.json").exists():
        raise ValueError("operator trust attestation artifact is missing")
    record = _strict_load(_owned_file(recon, "operator_trust_attestation.json").read_bytes())
    return _validate_operator_attestation_record(
        record,
        allow_test_attestation=allow_test_attestation,
        expected_workspace_root=expected_workspace_root,
        expected_skill_roots=expected_skill_roots,
        expected_uv_executable=expected_uv_executable,
        require_hash=True,
    )


def prepare_execution_context(
    run_dir: Path,
    *,
    workspace_root: Path,
    skill_roots: Mapping[str, Path | str],
    prepared_at: str | datetime,
    uv_executable: Path | str | None = None,
    allow_test_attestation: bool = False,
) -> dict[str, Any]:
    """Publish a portable-by-explicit-rebinding context only after live preflight passes.

    Absolute roots are intentional. Replay may provide ``expected_workspace_root``;
    moving a run requires preparing a fresh unsigned notification context.
    """
    run = _anchored_run(Path(run_dir)); recon = _recon_dir(run, create=False)
    workspace = Path(os.path.abspath(workspace_root))
    if not workspace.is_dir() or workspace.resolve(strict=True) != workspace:
        raise ValueError("workspace_root must have exact resolved identity")
    timestamp = _aware_timestamp(prepared_at, "prepared_at")
    preflight = check_live_recon_preflight(workspace, skill_roots, uv_executable=uv_executable)
    if not preflight["ok"]:
        raise ValueError("live recon preflight failed: " + ", ".join(preflight["missing"]))
    notices: dict[str, dict[str, str]] = {}
    notified_at: dict[str, str] = {}
    scripts: dict[str, dict[str, str]] = {}
    for source, file_name in (("ARXIV", "literature_search_arxiv_LICENSE.txt"), ("OPENALEX", "literature_search_openalex_LICENSE.txt")):
        notice = (workspace / ".licenses" / file_name).resolve(strict=True)
        if not notice.is_relative_to(workspace):
            raise ValueError("notice path escapes workspace root")
        notice_time = _notice_timestamp(notice)
        if notice_time > timestamp:
            raise ValueError("notified_at must be no later than prepared_at")
        notices[source] = {"path": str(notice), "sha256": sha256_file(notice)}
        notified_at[source] = notice_time.isoformat()
        skill_root = _skill_path(skill_roots, source)
        if skill_root is None:
            raise ValueError("live recon preflight omitted a skill root")
        skill_root = Path(os.path.abspath(skill_root))
        if not skill_root.is_dir() or skill_root.resolve(strict=True) != skill_root:
            raise ValueError("skill root must have exact resolved identity")
        script = Path(preflight["scripts"][source])
        if not script.resolve(strict=True).is_relative_to(skill_root):
            raise ValueError("skill script escapes its bound root")
        scripts[source] = {"skill_root": str(skill_root), "path": str(script), "sha256": sha256_file(script)}
    attestation = validate_operator_trust_attestation(
        recon,
        allow_test_attestation=allow_test_attestation,
        expected_workspace_root=workspace,
        expected_skill_roots=skill_roots,
        expected_uv_executable=Path(str(preflight["uv"])),
    )
    if _aware_timestamp(attestation["acknowledged_at"], "acknowledged_at") > timestamp:
        raise ValueError("operator trust attestation acknowledged_at must be no later than prepared_at")
    body = {
        "schema_version": "idea_factory.recon_execution_context.v4",
        "policy_version": EXECUTION_CONTEXT_POLICY_VERSION,
        "record_kind": UNSIGNED_CONTEXT_RECORD_KIND,
        "authenticity": UNSIGNED_AUTHENTICITY,
        "claim_boundary": "Current notification, operator trust attestation, script, and uv files match this unsigned execution record. User identity and consent are not authenticated; adversarial repository writers are outside scope.",
        "prepared_at": timestamp.isoformat(),
        "notified_at": notified_at,
        "operator_trust_attestation_sha256": attestation["operator_trust_attestation_sha256"],
        "operator_trust_scope": attestation["attestation_scope"],
        "tool_output_contracts": TOOL_OUTPUT_CONTRACTS,
        "workspace_root": str(workspace),
        "uv": {"path": preflight["uv"], "version": preflight["uv_version"], "sha256": preflight["uv_sha256"], "executable_policy": preflight["uv_executable_policy"], "policy_version": UV_EXECUTABLE_POLICY_VERSION},
        "terms_urls": {"ARXIV": ARXIV_URL, "OPENALEX": OPENALEX_URL},
        "notices": notices,
        "scripts": scripts,
    }
    context = body | {"execution_context_sha256": _hash(body)}
    write_json(recon / "execution_context.json", context)
    return context


def validate_execution_context(recon_or_run_dir: Path, *, expected_workspace_root: Path | None = None) -> dict[str, Any]:
    candidate = Path(recon_or_run_dir)
    recon = _recon_dir(_anchored_run(candidate), create=False) if candidate.name != "recon" else candidate
    if not recon.is_dir() or recon.resolve(strict=True) != recon:
        raise ValueError("recon directory is not anchored")
    context = _strict_load(_owned_file(recon, "execution_context.json").read_bytes())
    expected_keys = {"schema_version", "policy_version", "record_kind", "authenticity", "claim_boundary", "prepared_at", "notified_at", "operator_trust_attestation_sha256", "operator_trust_scope", "tool_output_contracts", "workspace_root", "uv", "terms_urls", "notices", "scripts", "execution_context_sha256"}
    if set(context) != expected_keys or type(context.get("uv")) is not dict or type(context.get("notices")) is not dict or type(context.get("scripts")) is not dict:
        raise ValueError("execution context schema is not exact")
    context_hash = context.get("execution_context_sha256")
    body = {key: value for key, value in context.items() if key != "execution_context_sha256"}
    if type(context_hash) is not str or context_hash != _hash(body):
        raise ValueError("execution context hash mismatch")
    if context.get("schema_version") != "idea_factory.recon_execution_context.v4" or context.get("policy_version") != EXECUTION_CONTEXT_POLICY_VERSION:
        raise ValueError("execution context policy mismatch")
    if context.get("record_kind") != UNSIGNED_CONTEXT_RECORD_KIND or context.get("authenticity") != UNSIGNED_AUTHENTICITY or "not authenticated" not in str(context.get("claim_boundary", "")).lower():
        raise ValueError("execution context unsigned authenticity boundary mismatch")
    if context.get("tool_output_contracts") != TOOL_OUTPUT_CONTRACTS:
        raise ValueError("execution context tool output contract mismatch")
    prepared_at = _aware_timestamp(context.get("prepared_at"), "prepared_at")
    if type(context.get("notified_at")) is not dict or set(context["notified_at"]) != {"ARXIV", "OPENALEX"}:
        raise ValueError("execution context notified_at schema mismatch")
    workspace = Path(str(context.get("workspace_root")))
    if not workspace.is_dir() or workspace.resolve(strict=True) != workspace:
        raise ValueError("execution context workspace root is missing or unanchored")
    if expected_workspace_root is not None and workspace != Path(os.path.abspath(expected_workspace_root)):
        raise ValueError("execution context workspace root differs from expected replay root")
    notice_hashes, missing = _notice_status(workspace)
    if missing:
        raise ValueError("execution context notices are missing or mismatched: " + ", ".join(missing))
    for source, file_name in (("ARXIV", "literature_search_arxiv_LICENSE.txt"), ("OPENALEX", "literature_search_openalex_LICENSE.txt")):
        expected_path = (workspace / ".licenses" / file_name).resolve(strict=True)
        if not expected_path.is_relative_to(workspace):
            raise ValueError("execution context notice path escapes workspace")
        notice = context.get("notices", {}).get(source, {})
        if notice.get("path") != str(expected_path) or notice.get("sha256") != notice_hashes[source]:
            raise ValueError(f"execution context {source} notice hash mismatch")
        notice_time = _notice_timestamp(expected_path)
        if context["notified_at"].get(source) != notice_time.isoformat() or notice_time > prepared_at:
            raise ValueError("execution context requires notified_at <= prepared_at")
    for source in ("ARXIV", "OPENALEX"):
        binding = context.get("scripts", {}).get(source, {})
        root, script = Path(str(binding.get("skill_root"))), Path(str(binding.get("path")))
        if not root.is_dir() or root.resolve(strict=True) != root or not script.is_file() or script.resolve(strict=True) != script or not script.is_relative_to(root):
            raise ValueError(f"execution context {source} script path is missing or unanchored")
        if binding.get("sha256") != sha256_file(script):
            raise ValueError(f"execution context {source} script hash mismatch")
    uv = context.get("uv", {}); uv_path = Path(str(uv.get("path")))
    if set(uv) != {"path", "version", "sha256", "executable_policy", "policy_version"} or uv.get("policy_version") != UV_EXECUTABLE_POLICY_VERSION:
        raise ValueError("execution context uv identity schema mismatch")
    if not uv_path.is_file() or uv_path.resolve(strict=True) != uv_path:
        raise ValueError("execution context uv path is missing or unanchored")
    current_policy = _uv_policy(uv_path)
    if current_policy is None or current_policy != uv.get("executable_policy"):
        raise ValueError("execution context uv executable identity mismatch")
    if sha256_file(uv_path) != uv.get("sha256"):
        raise ValueError("execution context uv executable hash mismatch")
    if _run_uv_version(uv_path, current_policy) != uv.get("version"):
        raise ValueError("execution context uv version mismatch")
    trust_scope = context.get("operator_trust_scope")
    if trust_scope not in {"LIVE_OPERATOR_ATTESTED", "TEST_ONLY"}:
        raise ValueError("execution context operator trust scope mismatch")
    attestation = validate_operator_trust_attestation(
        recon,
        allow_test_attestation=trust_scope == "TEST_ONLY",
        expected_workspace_root=workspace,
        expected_skill_roots={source: binding["skill_root"] for source, binding in context["scripts"].items()},
        expected_uv_executable=uv_path,
    )
    if context.get("operator_trust_attestation_sha256") != attestation["operator_trust_attestation_sha256"] or trust_scope != attestation["attestation_scope"] or _aware_timestamp(attestation["acknowledged_at"], "acknowledged_at") > prepared_at:
        raise ValueError("execution context operator trust attestation binding mismatch")
    return context


def _query_text(opportunity: Opportunity, lane: str, variant: str) -> str:
    x, y, z, f, w, a, test = (opportunity.assumption_x, opportunity.observation_y, opportunity.condition_z, opportunity.failure_f, opportunity.missing_capability_w, opportunity.alternative_explanation_a, opportunity.decisive_experiment)
    if lane == "CONCEPT":
        current, generic = f"{x} {w} {z}", f"{x} {z}"
    elif lane == "MECHANISM":
        current, generic = f"{f} {w} relation", f"{f} capability relation"
    elif lane == "FAILURE":
        current, generic = f"{y} {f} {z}", f"{y} {f}"
    else:
        measure = " ".join(_MEASURE.findall(test)) or "evaluation"
        current, generic = f"{test} {a}", f"{measure} {a} evaluation"
    return " ".join((current if variant == "CURRENT_TERMS" else generic).split())


def generate_recon_queries(opportunity: Opportunity) -> tuple[ReconQuery, ...]:
    """Pure exact-4x2 generation after enforcing the Task-7 naming firewall."""
    fields = (
        opportunity.assumption_x, opportunity.observation_y, opportunity.condition_z,
        opportunity.failure_f, opportunity.missing_capability_w,
        opportunity.alternative_explanation_a, opportunity.decisive_experiment,
        opportunity.scope_compatibility, *opportunity.inference_flags,
    )
    denied = set(_CONTROLLED_METHOD_TERMS); allowed = set(_DOMAIN_TERM_ALLOWLIST)
    offending: set[str] = set()
    for value in fields:
        if _NAMED_METHOD.search(value):
            offending.add(_NAMED_METHOD.search(value).group(0))  # type: ignore[union-attr]
        for token in _METHOD_TOKEN.findall(value):
            folded = token.casefold()
            if folded in allowed:
                continue
            uppercase_count = sum(character.isupper() for character in token)
            mixed_case = uppercase_count >= 2 and any(character.islower() for character in token)
            if folded in denied or _CONTROLLED_MODEL_STEM.fullmatch(token) or _CONTROLLED_LETTER_NUMBER_MODEL.fullmatch(token) or (token.isupper() and len(token) >= 3) or mixed_case or folded.endswith(("net", "former")):
                offending.add(token)
    if offending:
        raise ValueError("controlled deterministic firewall rejects method names or model acronyms: " + ", ".join(sorted(offending)))
    queries: list[ReconQuery] = []
    seen: set[str] = set()
    for lane in _LANES:
        for variant in _VARIANTS:
            query = _query_text(opportunity, lane, variant)
            if not query or query.casefold() in seen:
                raise ValueError("recon query must be nonblank and unique per opportunity")
            seen.add(query.casefold())
            query_id = stable_id("recon_query", opportunity.opportunity_id, lane, variant, QUERY_POLICY_VERSION, query)
            queries.append(ReconQuery(query_id=query_id, opportunity_id=opportunity.opportunity_id, lane=lane, variant=variant, query=query, max_results=10))
    return tuple(queries)


def _opportunity_rows(run: Path, config: CorpusRouterConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dedup = validate_dedup_bundle(run, config)
    # Task-8 output has no full opportunity body, so bind it back to the exact Task-7 gate row.
    quality = _owned_dir(run, "quality", create=False)
    ready = read_jsonl(_owned_file(quality, "ready_for_internal_dedup.jsonl"))
    by_id = {str(row.get("record_id")): row for row in ready}
    rows: list[dict[str, Any]] = []
    for row in dedup["ready"]:
        if row.get("status") not in {"CLEAR", "INTERNAL_ADJACENT"} or row.get("blocked") is not False:
            raise ValueError("dedup ready projection contains a blocked or non-residual record")
        source_id = str(row.get("record_id"))
        quality_row = by_id.get(source_id)
        if quality_row is None or not isinstance(quality_row.get("opportunity"), dict):
            raise ValueError("dedup ready record lacks its validated opportunity body")
        opportunity = Opportunity.model_validate_json(_canonical_json(quality_row["opportunity"]), strict=True)
        if opportunity.opportunity_id != source_id:
            raise ValueError("dedup opportunity binding mismatch")
        rows.append({"dedup": row, "quality": quality_row, "opportunity": opportunity})
    return sorted(rows, key=lambda row: row["opportunity"].opportunity_id), dedup


def _recon_dir(run: Path, *, create: bool) -> Path:
    return _owned_dir(run, "recon", create=create)


_DOWNSTREAM = ("execution_job_templates.jsonl", "execution_jobs.jsonl", "execution_receipts.jsonl", "normalized_manifest.json", "report_jobs.jsonl", "reports.jsonl", "rejected.jsonl", "outcomes.jsonl", "ready_for_routes.jsonl", "killed.jsonl")


def _assert_anchored_tree(recon: Path, name: str) -> Path | None:
    path = recon / name
    if not path.exists():
        return None
    if not path.is_dir() or path.resolve(strict=True) != path or not path.resolve().is_relative_to(recon):
        raise ValueError(f"recon {name} path is not anchored")
    for child in path.rglob("*"):
        try:
            resolved = child.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"recon {name} contains an unresolved artifact") from exc
        if not resolved.is_relative_to(path):
            raise ValueError(f"recon {name} contains an artifact that escapes the anchored tree")
    return path


def _remove_anchored_tree(recon: Path, name: str) -> None:
    """Invalidate a known recon-owned tree only after proving every child is owned."""
    path = _assert_anchored_tree(recon, name)
    if path is None:
        return
    shutil.rmtree(path)


def _query_projections(rows: Sequence[dict[str, Any]], dedup: Mapping[str, Sequence[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pure deterministic query generation from the already validated survivors."""
    variants = _VARIANTS
    records: list[dict[str, Any]] = []
    for item in rows:
        opportunity = item["opportunity"]
        for model in generate_recon_queries(opportunity):
            records.append({
                "schema_version": "idea_factory.recon_query_job.v1", **model.model_dump(),
                "opportunity": opportunity.model_dump(mode="json"), "opportunity_sha256": _hash(opportunity.model_dump(mode="json")),
                "dedup_record_sha256": _hash(item["dedup"]), "dedup_bundle_sha256": _hash(list(dedup["ready"])),
                    "dedup_policy_version": DEDUP_POLICY_VERSION, "query_generation_policy_version": QUERY_POLICY_VERSION,
                    "method_firewall_version": METHOD_FIREWALL_VERSION, "method_firewall_policy_sha256": METHOD_FIREWALL_POLICY_SHA256,
            })
    records.sort(key=lambda row: (row["opportunity_id"], _LANES.index(row["lane"]), variants.index(row["variant"])))
    manifest = {"schema_version": "idea_factory.recon_query_pack_manifest.v1", "query_generation_policy_version": QUERY_POLICY_VERSION, "method_firewall_version": METHOD_FIREWALL_VERSION, "method_firewall_policy_sha256": METHOD_FIREWALL_POLICY_SHA256, "method_firewall_policy": METHOD_FIREWALL_POLICY, "dedup_policy_version": DEDUP_POLICY_VERSION, "dedup_bundle_sha256": _hash(list(dedup["ready"])), "query_pack_sha256": _hash(records), "opportunity_ids": [row["opportunity"].opportunity_id for row in rows], "query_ids": [row["query_id"] for row in records], "opportunity_count": len(rows), "query_count": len(records), "lanes": list(_LANES), "variants": list(variants)}
    return records, manifest


@dataclass(frozen=True)
class QueryPackBundle:
    run: Path
    queries: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def emit_recon_query_pack(run_dir: Path, config: CorpusRouterConfig) -> Path:
    """Create the deterministic eight-query pack from only Task-8 survivors."""
    run = _anchored_run(Path(run_dir))
    rows, dedup = _opportunity_rows(run, config)
    records, manifest = _query_projections(rows, dedup)
    recon = _recon_dir(run, create=True)
    for name in ("raw", "normalized"):
        _assert_anchored_tree(recon, name)
    _safe_unlink(recon, ("query_pack.jsonl", "query_pack_manifest.json", *_DOWNSTREAM))
    _remove_anchored_tree(recon, "raw")
    _remove_anchored_tree(recon, "normalized")
    try:
        write_jsonl_bundle({recon / "query_pack.jsonl": records})
        write_json(recon / "query_pack_manifest.json", manifest)
    except BaseException:
        _safe_unlink(recon, ("query_pack.jsonl", "query_pack_manifest.json"))
        raise
    return recon / "query_pack.jsonl"


emit_query_pack = emit_recon_query_pack


def validate_query_pack(run_dir: Path, config: CorpusRouterConfig) -> QueryPackBundle:
    run = _anchored_run(Path(run_dir))
    recon = _recon_dir(run, create=False)
    pack = read_jsonl(_owned_file(recon, "query_pack.jsonl"))
    manifest = _strict_load(_owned_file(recon, "query_pack_manifest.json").read_bytes())
    rows, dedup = _opportunity_rows(run, config)
    expected_pack, expected_manifest = _query_projections(rows, dedup)
    if pack != expected_pack or manifest != expected_manifest:
        raise ValueError("query pack replay mismatch")
    return QueryPackBundle(run, tuple(pack), manifest)


def _expected_execution_job_templates(pack: Sequence[dict[str, Any]], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for query in pack:
        if query.get("variant") not in _VARIANTS:
            raise ValueError("SYNONYM is forbidden by the v1 recon policy")
        for source in ("ARXIV", "OPENALEX"):
            raw_rel = f"recon/raw/{source.lower()}/{query['query_id']}.json"
            if source == "ARXIV":
                args = ["--query", query["query"], "--max_results", "10"]
                command = {"required_skill": "literature_search_arxiv", "max_results": 10, "rate_limit_seconds": 3}
            else:
                args = ["filter", "works", "--search", query["query"], "--select", "id,doi,display_name,publication_year,authorships,abstract_inverted_index,primary_location", "--per-page", "10"]
                command = {"required_skill": "literature_search_openalex", "per_page": 10, "rate_limit_seconds": 0}
            template_id = stable_id("recon_execution_template", source.lower(), query["query_id"], COMMAND_POLICY_VERSION)
            templates.append({"schema_version": "idea_factory.recon_execution_job_template.v2", "template_id": template_id, "source": source, "script_key": source, "tool_output_contract": TOOL_OUTPUT_CONTRACTS[source], "query_id": query["query_id"], "opportunity_id": query["opportunity_id"], "query": query["query"], "args": args, "raw_output_path": raw_rel, "query_pack_sha256": manifest["query_pack_sha256"], "command_policy_version": COMMAND_POLICY_VERSION, **command})
    return sorted(templates, key=lambda row: (row["query_id"], row["source"]))


def emit_recon_execution_job_templates(run_dir: Path, config: CorpusRouterConfig) -> Path:
    bundle = validate_query_pack(run_dir, config); run = bundle.run
    recon = _recon_dir(run, create=False); pack, manifest = bundle.queries, bundle.manifest
    _safe_unlink(recon, _DOWNSTREAM)
    templates = _expected_execution_job_templates(pack, manifest)
    write_jsonl_bundle({recon / "execution_job_templates.jsonl": templates})
    return recon / "execution_job_templates.jsonl"


def validate_execution_job_templates(run_dir: Path, config: CorpusRouterConfig) -> tuple[dict[str, Any], ...]:
    bundle = validate_query_pack(run_dir, config); recon = _recon_dir(bundle.run, create=False)
    templates = read_jsonl(_owned_file(recon, "execution_job_templates.jsonl"))
    if templates != _expected_execution_job_templates(bundle.queries, bundle.manifest):
        raise ValueError("execution job template replay mismatch")
    return tuple(templates)


def _expected_execution_jobs(templates: Sequence[dict[str, Any]], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for template in templates:
        source = template["source"]; script = context["scripts"][source]
        if template.get("tool_output_contract") != context["tool_output_contracts"].get(source):
            raise ValueError("execution template and context tool output contracts differ")
        argv = [context["uv"]["path"], "run", script["path"], *template["args"]]
        job_id = stable_id("recon_execution", template["template_id"], context["execution_context_sha256"], COMMAND_POLICY_VERSION)
        jobs.append({
            "schema_version": "idea_factory.recon_execution_job.v3", "job_id": job_id,
            "template_id": template["template_id"], "template_sha256": _hash(template),
            "source": source, "query_id": template["query_id"], "opportunity_id": template["opportunity_id"],
            "query": template["query"], "argv": argv, "argv_sha256": _hash(argv), "cwd": script["skill_root"],
            "raw_output_path": template["raw_output_path"], "query_pack_sha256": template["query_pack_sha256"],
            "execution_context_sha256": context["execution_context_sha256"],
            "operator_trust_attestation_sha256": context["operator_trust_attestation_sha256"],
            "operator_trust_scope": context["operator_trust_scope"],
            "tool_output_contract": template["tool_output_contract"],
            "command_policy_version": COMMAND_POLICY_VERSION,
            **{key: template[key] for key in ("required_skill", "rate_limit_seconds")},
            **({"max_results": template["max_results"]} if source == "ARXIV" else {"per_page": template["per_page"]}),
        })
    return sorted(jobs, key=lambda row: (row["query_id"], row["source"]))


def materialize_execution_jobs(run_dir: Path, config: CorpusRouterConfig) -> Path:
    bundle = validate_query_pack(run_dir, config); recon = _recon_dir(bundle.run, create=False)
    templates = validate_execution_job_templates(bundle.run, config)
    context = validate_execution_context(recon) if templates else {"execution_context_sha256": "", "uv": {}, "scripts": {}}
    jobs = _expected_execution_jobs(templates, context) if templates else []
    if len({job["job_id"] for job in jobs}) != len(jobs) or len({job["raw_output_path"] for job in jobs}) != len(jobs):
        raise ValueError("execution jobs must have unique IDs and raw paths")
    if any(Path(job["raw_output_path"]).is_absolute() or ".." in Path(job["raw_output_path"]).parts for job in jobs):
        raise ValueError("execution job raw path escapes run")
    downstream = tuple(name for name in _DOWNSTREAM if name != "execution_job_templates.jsonl")
    _safe_unlink(recon, downstream)
    write_jsonl_bundle({recon / "execution_jobs.jsonl": jobs})
    return recon / "execution_jobs.jsonl"


emit_recon_execution_jobs = materialize_execution_jobs
emit_execution_jobs = materialize_execution_jobs


def validate_execution_jobs(run_dir: Path, config: CorpusRouterConfig) -> tuple[dict[str, Any], ...]:
    bundle = validate_query_pack(run_dir, config)
    recon = _recon_dir(bundle.run, create=False)
    templates = validate_execution_job_templates(bundle.run, config)
    if not templates and not (recon / "execution_jobs.jsonl").exists():
        return ()
    context = validate_execution_context(recon) if templates else {"execution_context_sha256": "", "uv": {}, "scripts": {}}
    jobs = read_jsonl(_owned_file(recon, "execution_jobs.jsonl"))
    if jobs != (_expected_execution_jobs(templates, context) if templates else []):
        raise ValueError("execution job replay mismatch")
    return tuple(jobs)


def _read_job_receipts(recon: Path, jobs: Sequence[dict[str, Any]], *, expected_workspace_root: Path | None = None) -> list[dict[str, Any]]:
    receipts = read_jsonl(_owned_file(recon, "execution_receipts.jsonl"))
    if [row.get("job_id") for row in receipts] != [job["job_id"] for job in jobs]:
        raise ValueError("execution receipts must follow canonical job order")
    by_job = {row.get("job_id"): row for row in receipts}
    if len(by_job) != len(receipts) or set(by_job) != {job["job_id"] for job in jobs}:
        raise ValueError("execution receipts must exactly cover execution jobs")
    context = validate_execution_context(recon, expected_workspace_root=expected_workspace_root) if jobs else None
    prepared_at = _aware_timestamp(context["prepared_at"], "prepared_at") if context else None
    arxiv_request_times: list[datetime] = []
    for job in jobs:
        receipt = by_job[job["job_id"]]
        expected_keys = {
            "schema_version", "job_id", "source", "query_id", "argv_sha256",
            "raw_output_path", "query_pack_sha256", "command_policy_version",
            "started_at", "request_started_at", "completed_at", "exit_code", "status", "http_status", "raw_file_sha256",
            "execution_context_sha256", "tool_stdout_summary", "tool_stderr_summary", "error_reason",
        }
        if set(receipt) != expected_keys or receipt.get("schema_version") != "idea_factory.recon_execution_receipt.v3":
            raise ValueError("execution receipt schema is not exact")
        if any(receipt.get(key) != job.get(key) for key in ("job_id", "source", "query_id", "argv_sha256", "raw_output_path", "query_pack_sha256", "command_policy_version")):
            raise ValueError("execution receipt binding mismatch")
        if receipt.get("execution_context_sha256") != context["execution_context_sha256"]:
            raise ValueError("execution receipt context hash binding mismatch")
        if receipt.get("status") not in {"SUCCESS", "EMPTY", "ERROR", "CREDENTIALS_PROTOCOL_REQUIRED"}:
            raise ValueError("invalid execution receipt status")
        exit_code = receipt.get("exit_code")
        if type(exit_code) is not int or (receipt["status"] in {"SUCCESS", "EMPTY"}) != (exit_code == 0):
            raise ValueError("execution receipt exit_code and status are inconsistent")
        parsed_times: dict[str, datetime] = {}
        for timestamp_key in ("started_at", "request_started_at", "completed_at"):
            timestamp = receipt.get(timestamp_key)
            if type(timestamp) is not str:
                raise ValueError("execution receipt requires ISO timestamps")
            try:
                parsed = datetime.fromisoformat(timestamp)
            except ValueError as exc:
                raise ValueError("execution receipt timestamp is invalid") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("execution receipt timestamp must be timezone-aware")
            parsed_times[timestamp_key] = parsed
        if parsed_times["request_started_at"] < parsed_times["started_at"]:
            raise ValueError("execution receipt request_started_at is before started_at")
        if parsed_times["completed_at"] < parsed_times["request_started_at"]:
            raise ValueError("execution receipt completed_at is before request_started_at")
        if prepared_at is not None and parsed_times["started_at"] < prepared_at:
            raise ValueError("prepared_at must be no later than execution_started_at")
        if receipt["source"] == "ARXIV":
            arxiv_request_times.append(parsed_times["request_started_at"])
        for summary_key in ("tool_stdout_summary", "tool_stderr_summary"):
            summary = receipt.get(summary_key)
            if type(summary) is not str or re.search(r"(?:api[_-]?key|bearer\s+|sk-[A-Za-z0-9])", summary, re.I):
                raise ValueError("execution receipt summary is missing or may contain a secret")
        http_status = receipt.get("http_status")
        if http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599):
            raise ValueError("execution receipt http_status is invalid")
        credentials_status = receipt["source"] == "OPENALEX" and http_status in {401, 429}
        if credentials_status:
            if receipt["status"] != "CREDENTIALS_PROTOCOL_REQUIRED" or "credentials protocol" not in str(receipt.get("error_reason", "")).lower():
                raise ValueError("OpenAlex 401/429 requires CREDENTIALS_PROTOCOL_REQUIRED and an actionable credentials protocol reason")
        elif receipt["status"] == "CREDENTIALS_PROTOCOL_REQUIRED":
            raise ValueError("CREDENTIALS_PROTOCOL_REQUIRED is only valid for OpenAlex 401/429")
        raw_sha = receipt.get("raw_file_sha256")
        if receipt["status"] in {"SUCCESS", "EMPTY"}:
            if type(raw_sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", raw_sha):
                raise ValueError("successful receipt requires an exact raw hash")
            raw = recon.parent / str(receipt["raw_output_path"])
            if not raw.is_file() or raw.resolve(strict=True) != raw or not raw.resolve().is_relative_to(recon):
                raise ValueError("successful receipt raw file is missing or escapes recon")
            if sha256_file(raw) != raw_sha:
                raise ValueError("execution receipt raw hash does not match current file")
        elif raw_sha is not None:
            raise ValueError("error receipts cannot carry a success raw hash")
    ordered_arxiv_requests = sorted(arxiv_request_times)
    for previous, current in zip(ordered_arxiv_requests, ordered_arxiv_requests[1:]):
        if current - previous < timedelta(seconds=3):
            raise ValueError("arXiv request spacing must be at least 3 seconds across jobs")
    return receipts


def validate_execution_receipts(run_dir: Path, config: CorpusRouterConfig, *, expected_workspace_root: Path | None = None) -> tuple[dict[str, Any], ...]:
    jobs = validate_execution_jobs(run_dir, config)
    run = _anchored_run(Path(run_dir))
    return tuple(_read_job_receipts(_recon_dir(run, create=False), jobs, expected_workspace_root=expected_workspace_root))


def parse_arxiv_raw(raw: str | bytes, *, query_id: str, raw_file_sha: str, max_results: int = 10) -> list[ReconEvidence]:
    if type(max_results) is not int or max_results <= 0:
        raise ValueError("max_results must be a positive integer")
    wrappers = _strict_json_stream(raw)
    previous: list[dict[str, Any]] = []
    for wrapper in wrappers:
        if wrapper.get("status") != "success":
            raise ValueError("arXiv wrapper status must be success")
        values = wrapper.get("papers")
        count = wrapper.get("results_count")
        if type(values) is not list or type(count) is not int or count != len(values):
            raise ValueError("arXiv wrapper results_count and papers are inconsistent")
        if count > max_results:
            raise ValueError("arXiv results exceed materialized max_results")
        if any(type(item) is not dict for item in values):
            raise ValueError("arXiv wrapper papers must be objects")
        if len(values) < len(previous) or values[: len(previous)] != previous:
            raise ValueError("arXiv cumulative stream is not prefix-monotonic")
        previous = values
    values = previous
    out: list[ReconEvidence] = []
    for item in values:
        if type(item) is not dict:
            raise ValueError("arXiv result must be an object")
        source_id = _safe_text(item.get("id", item.get("entry_id", "")))
        title = _safe_text(item.get("title", ""))
        doi = item.get("doi")
        if doi is not None: doi = _canonicalize_doi(_safe_text(doi))
        url = item.get("pdf_url") or item.get("url") or item.get("link")
        if url is None and re.match(r"^https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/", source_id):
            url = source_id
        if url is not None: url = _validated_url(url)
        authors_raw = item.get("authors", [])
        if type(authors_raw) is not list: raise ValueError("arXiv authors must be a list")
        authors = tuple(_safe_text(author.get("name")) if type(author) is dict else _safe_text(author) for author in authors_raw)
        year = _date_year(item.get("published", item.get("year")))
        abstract = _safe_text(item.get("summary", item.get("abstract", "")))
        evidence_id = _evidence_id(source="ARXIV", source_id=source_id, title=title, year=year, doi=doi, url=url, authors=authors, abstract_text=abstract)
        out.append(ReconEvidence(evidence_id=evidence_id, source="ARXIV", source_id=source_id, title=title, year=year, doi=doi, url=url, authors=authors, abstract_text=abstract, query_id=query_id, raw_file_sha=raw_file_sha, source_aliases=(source_id,), query_ids=(query_id,)))
    return out


def parse_openalex_raw(raw: str | bytes, *, query_id: str, raw_file_sha: str, per_page: int = 10) -> list[ReconEvidence]:
    if type(per_page) is not int or per_page <= 0:
        raise ValueError("per_page must be a positive integer")
    body = _strict_load(raw)
    values = body.get("results")
    if type(values) is not list: raise ValueError("OpenAlex raw payload requires a results array")
    if len(values) > per_page:
        raise ValueError("OpenAlex results exceed materialized per_page")
    meta = body.get("meta")
    if not values:
        if type(meta) is not dict or type(meta.get("count")) is not int or meta["count"] != 0 or type(meta.get("per_page")) is not int or meta["per_page"] != per_page:
            raise ValueError("OpenAlex empty results require metadata-confirmed genuine zero")
    elif meta is not None:
        if type(meta) is not dict or type(meta.get("count")) is not int or meta["count"] < len(values) or ("per_page" in meta and (type(meta.get("per_page")) is not int or meta["per_page"] != per_page)):
            raise ValueError("OpenAlex metadata is inconsistent with results and per_page")
    out: list[ReconEvidence] = []
    for item in values:
        if type(item) is not dict: raise ValueError("OpenAlex result must be an object")
        source_id, title = _safe_text(item.get("id", "")), _safe_text(item.get("display_name", ""))
        doi = item.get("doi"); doi = _canonicalize_doi(_safe_text(doi)) if doi is not None else None
        location = item.get("primary_location")
        if location is not None and type(location) is not dict: raise ValueError("OpenAlex primary_location must be object or null")
        url = location.get("landing_page_url") if location else None
        url = _validated_url(url) if url is not None else None
        authorships = item.get("authorships", [])
        if type(authorships) is not list: raise ValueError("OpenAlex authorships must be a list")
        authors: list[str] = []
        for entry in authorships:
            if type(entry) is not dict or type(entry.get("author")) is not dict: raise ValueError("invalid OpenAlex authorship")
            authors.append(_safe_text(entry["author"].get("display_name", "")))
        year = _date_year(item.get("publication_year")); abstract = normalize_openalex_abstract(item.get("abstract_inverted_index")); author_tuple = tuple(authors)
        evidence_id = _evidence_id(source="OPENALEX", source_id=source_id, title=title, year=year, doi=doi, url=url, authors=author_tuple, abstract_text=abstract)
        out.append(ReconEvidence(evidence_id=evidence_id, source="OPENALEX", source_id=source_id, title=title, year=year, doi=doi, url=url, authors=author_tuple, abstract_text=abstract, query_id=query_id, raw_file_sha=raw_file_sha, source_aliases=(source_id,), query_ids=(query_id,)))
    return out


def _dedup_evidence(records: Sequence[ReconEvidence]) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: (
            row.evidence_id,
            row.source,
            row.source_id,
            row.query_id,
            row.raw_file_sha,
        ),
    )
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # The smaller root is stable because ``ordered`` is stable.
            parent[max(left_root, right_root)] = min(left_root, right_root)

    doi_owner: dict[str, int] = {}
    source_owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(ordered):
        if record.doi:
            doi = _canonicalize_doi(record.doi)
            if doi in doi_owner:
                union(index, doi_owner[doi])
            else:
                doi_owner[doi] = index
        source_key = (record.source, record.source_id)
        if source_key in source_owner:
            union(index, source_owner[source_key])
        else:
            source_owner[source_key] = index

    title_groups: dict[tuple[str, int | None], list[int]] = {}
    for index, record in enumerate(ordered):
        title_key = (" ".join(record.title.casefold().split()), record.year)
        title_groups.setdefault(title_key, []).append(index)

    # Title/year is a deliberately weaker edge. It may bridge records with no DOI
    # to one DOI-bearing component, but it must never collapse conflicting DOIs.
    for indices in title_groups.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                left_root, right_root = find(left), find(right)
                if left_root == right_root:
                    continue
                component_dois = {
                    _canonicalize_doi(row.doi)
                    for member, row in enumerate(ordered)
                    if find(member) in {left_root, right_root} and row.doi
                }
                if len(component_dois) <= 1:
                    union(left_root, right_root)

    components: dict[int, list[ReconEvidence]] = {}
    for index, record in enumerate(ordered):
        components.setdefault(find(index), []).append(record)

    out: list[dict[str, Any]] = []
    for members in components.values():
        primary = members[0]
        aliases = tuple(sorted({f"{row.source}:{row.source_id}" for row in members}))
        queries = tuple(
            sorted(
                {
                    query_id
                    for row in members
                    for query_id in (*row.query_ids, row.query_id)
                }
            )
        )
        out.append(
            primary.model_copy(
                update={"source_aliases": aliases, "query_ids": queries}
            ).model_dump(mode="json")
        )
    return sorted(out, key=lambda row: (row["evidence_id"], row["source"], row["source_id"]))


def _jsonl_bytes(records: Sequence[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(record) + "\n" for record in records).encode("utf-8")


def _normalized_projection(
    run: Path,
    pack: QueryPackBundle,
    jobs: Sequence[dict[str, Any]],
    receipts: Sequence[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    recon = _recon_dir(run, create=False)
    records: list[ReconEvidence] = []
    per_file: dict[str, list[dict[str, Any]]] = {}
    receipts_by_job = {row["job_id"]: row for row in receipts}
    raw_bindings: list[dict[str, Any]] = []
    for job in jobs:
        receipt = receipts_by_job[job["job_id"]]
        raw_bindings.append({"job_id": job["job_id"], "query_id": job["query_id"], "source": job["source"], "status": receipt["status"], "raw_output_path": job["raw_output_path"], "raw_file_sha256": receipt["raw_file_sha256"]})
        if receipt["status"] not in {"SUCCESS", "EMPTY"}: continue
        raw = recon.parent / job["raw_output_path"]
        arxiv_empty_stdout = job["source"] == "ARXIV" and raw.is_file() and raw.stat().st_size == 0
        if arxiv_empty_stdout:
            empty_sha = hashlib.sha256(b"").hexdigest()
            if job.get("tool_output_contract") != ARXIV_OUTPUT_CONTRACT or receipt.get("status") != "EMPTY" or receipt.get("exit_code") != 0 or receipt.get("raw_file_sha256") != empty_sha or sha256_file(raw) != empty_sha:
                raise ValueError("empty arXiv stdout is genuine zero only with the bound contract and EMPTY receipt")
            continue
        raw_sha = sha256_file(raw)
        raw_bytes = raw.read_bytes()
        parsed = (
            parse_arxiv_raw(raw_bytes, query_id=job["query_id"], raw_file_sha=raw_sha, max_results=job["max_results"])
            if job["source"] == "ARXIV"
            else parse_openalex_raw(raw_bytes, query_id=job["query_id"], raw_file_sha=raw_sha, per_page=job["per_page"])
        )
        if job["source"] == "ARXIV" and not parsed:
            raise ValueError("synthetic zero wrapper is forbidden; official arXiv zero uses empty stdout")
        if (receipt["status"] == "EMPTY") != (not parsed):
            raise ValueError("receipt status is inconsistent with parsed result count")
        records.extend(parsed)
    deduped = _dedup_evidence(records)
    # materialize source/query views from the full conservative dedup projection.
    for item in deduped:
        for query_id in item["query_ids"]:
            relative = f"normalized/{item['source'].lower()}/{query_id}.jsonl"
            per_file.setdefault(relative, []).append(item)
    per_file = {path: sorted(rows, key=lambda row: row["evidence_id"]) for path, rows in sorted(per_file.items())}
    file_hashes = {path: hashlib.sha256(_jsonl_bytes(rows)).hexdigest() for path, rows in per_file.items()}
    normalized_manifest = {"schema_version": "idea_factory.recon_normalized_manifest.v1", "query_pack_sha256": pack.manifest["query_pack_sha256"], "execution_jobs_sha256": _hash(list(jobs)), "execution_receipts_sha256": _hash(list(receipts)), "raw_bindings": raw_bindings, "normalized_file_hashes": file_hashes, "evidence": deduped, "evidence_count": len(deduped)}
    normalized_manifest["normalized_evidence_manifest_sha256"] = _hash({key: value for key, value in normalized_manifest.items() if key != "normalized_evidence_manifest_sha256"})
    return per_file, normalized_manifest


@dataclass(frozen=True)
class NormalizedEvidenceBundle:
    run: Path
    files: Mapping[str, tuple[dict[str, Any], ...]]
    manifest: dict[str, Any]
    evidence: tuple[dict[str, Any], ...]


def normalize_recon_raw(run_dir: Path, config: CorpusRouterConfig) -> Path:
    pack = validate_query_pack(run_dir, config); run = pack.run
    jobs = validate_execution_jobs(run, config)
    receipts = validate_execution_receipts(run, config)
    recon = _recon_dir(run, create=False)
    _remove_anchored_tree(recon, "normalized")
    _safe_unlink(recon, ("normalized_manifest.json", "report_jobs.jsonl", "reports.jsonl", "rejected.jsonl", "outcomes.jsonl", "ready_for_routes.jsonl", "killed.jsonl"))
    try:
        files, normalized_manifest = _normalized_projection(run, pack, jobs, receipts)
        targets = {recon / relative: rows for relative, rows in files.items()}
        if targets: write_jsonl_bundle(targets)
        write_json(recon / "normalized_manifest.json", normalized_manifest)
        return recon / "normalized_manifest.json"
    except BaseException:
        _remove_anchored_tree(recon, "normalized")
        _safe_unlink(recon, ("normalized_manifest.json", "report_jobs.jsonl", "reports.jsonl", "rejected.jsonl", "outcomes.jsonl", "ready_for_routes.jsonl", "killed.jsonl"))
        raise


def validate_normalized_evidence_bundle(run_dir: Path, config: CorpusRouterConfig) -> NormalizedEvidenceBundle:
    pack = validate_query_pack(run_dir, config); run = pack.run
    jobs = validate_execution_jobs(run, config); receipts = validate_execution_receipts(run, config)
    expected_files, expected_manifest = _normalized_projection(run, pack, jobs, receipts)
    recon = _recon_dir(run, create=False)
    actual_manifest = _strict_load(_owned_file(recon, "normalized_manifest.json").read_bytes())
    normalized = recon / "normalized"
    actual_files: dict[str, list[dict[str, Any]]] = {}
    if normalized.exists():
        if not normalized.is_dir() or normalized.resolve(strict=True) != normalized:
            raise ValueError("normalized directory escapes recon")
        for path in sorted(normalized.rglob("*")):
            if path.is_dir(): continue
            if path.resolve(strict=True) != path or not path.resolve().is_relative_to(normalized) or path.suffix != ".jsonl":
                raise ValueError("normalized artifact escapes or has unexpected type")
            actual_files[path.relative_to(recon).as_posix()] = read_jsonl(path)
    if actual_files != expected_files or actual_manifest != expected_manifest:
        raise ValueError("normalized evidence replay mismatch")
    return NormalizedEvidenceBundle(run, {path: tuple(rows) for path, rows in actual_files.items()}, actual_manifest, tuple(actual_manifest["evidence"]))


normalize_raw_evidence = normalize_recon_raw


def _report_job_projection(pack: QueryPackBundle, normalized: NormalizedEvidenceBundle) -> list[dict[str, Any]]:
    evidence = normalized.manifest["evidence"]
    protocol = {"version": RECON_PROTOCOL_VERSION, "query_pack_sha256": pack.manifest["query_pack_sha256"], "normalized_evidence_manifest_sha256": normalized.manifest["normalized_evidence_manifest_sha256"], "raw_json_forbidden": True, "no_direct_coverage_reason": NO_DIRECT_COVERAGE_REASON}
    protocol_hash = _hash(protocol); jobs: list[dict[str, Any]] = []
    for opportunity_id in pack.manifest["opportunity_ids"]:
        query_ids = [row["query_id"] for row in pack.queries if row["opportunity_id"] == opportunity_id]
        lane_query_ids = {lane: [row["query_id"] for row in pack.queries if row["opportunity_id"] == opportunity_id and row["lane"] == lane] for lane in _LANES}
        slim = []
        for row in evidence:
            if not any(qid in row.get("query_ids", []) for qid in query_ids):
                continue
            projected = {key: row.get(key) for key in ("evidence_id", "source", "source_id", "title", "year", "doi", "url", "authors", "abstract_text", "query_ids")}
            projected["source_aliases"] = sorted(row.get("source_aliases", []))
            slim.append(projected)
        cache_key = _hash({"opportunity_id": opportunity_id, "query_ids": query_ids, "evidence_ids": [row["evidence_id"] for row in slim], "protocol_hash": protocol_hash})
        job_id = stable_id("recon_report", opportunity_id, pack.manifest["query_pack_sha256"], normalized.manifest["normalized_evidence_manifest_sha256"])
        jobs.append({"schema_version": "idea_factory.recon_report_job.v1", "result_schema_version": "idea_factory.recon_report_result.v1", "job_id": job_id, "opportunity_id": opportunity_id, "query_pack_sha256": pack.manifest["query_pack_sha256"], "normalized_evidence_manifest_sha256": normalized.manifest["normalized_evidence_manifest_sha256"], "protocol": protocol, "protocol_hash": protocol_hash, "cache_key": cache_key, "searched_query_ids": query_ids, "lane_query_ids": lane_query_ids, "evidence": slim, "prompt": f"Use only the supplied normalized evidence. Return one strict recon report object; raw JSON is forbidden. NO_DIRECT_COVERAGE_FOUND requires this exact reason: {NO_DIRECT_COVERAGE_REASON}", "raw_json_forbidden": True})
    return jobs


def emit_recon_report_jobs(run_dir: Path, config: CorpusRouterConfig) -> Path:
    pack = validate_query_pack(run_dir, config)
    normalized = validate_normalized_evidence_bundle(run_dir, config)
    recon = _recon_dir(pack.run, create=False)
    _safe_unlink(recon, ("report_jobs.jsonl", "reports.jsonl", "rejected.jsonl", "outcomes.jsonl", "ready_for_routes.jsonl", "killed.jsonl"))
    write_jsonl_bundle({recon / "report_jobs.jsonl": _report_job_projection(pack, normalized)})
    return recon / "report_jobs.jsonl"


def validate_report_jobs(run_dir: Path, config: CorpusRouterConfig) -> tuple[dict[str, Any], ...]:
    pack = validate_query_pack(run_dir, config)
    normalized = validate_normalized_evidence_bundle(run_dir, config)
    recon = _recon_dir(pack.run, create=False)
    jobs = read_jsonl(_owned_file(recon, "report_jobs.jsonl"))
    if jobs != _report_job_projection(pack, normalized):
        raise ValueError("report job replay mismatch")
    return tuple(jobs)


_REPORT_REJECTION_REASONS = {
    "stale recon report binding": "STALE_REPORT_BINDING",
    "each recon lane requires a SUCCESS or EMPTY receipt": "LANE_COVERAGE_INCOMPLETE",
    "searched_at precedes execution completion": "SEARCH_TIMESTAMP_INVALID",
    "nearest prior URL or ID is not evidence-bound": "PRIOR_REFERENCE_UNBOUND",
    "nearest prior paper must equal the canonical title": "PRIOR_TITLE_MISMATCH",
    "NO_DIRECT_COVERAGE_FOUND must use the controlled literal reason exactly": "NO_DIRECT_REASON_INVALID",
    "covered and near-prior decisions require evidence": "PRIOR_EVIDENCE_REQUIRED",
    "covered or near-prior decision reason must name its evidence-bound prior": "PRIOR_REASON_UNBOUND",
}


def _bounded_report_rejection(exc: ValidationError | ValueError) -> tuple[str, str]:
    if isinstance(exc, ValidationError):
        return "REPORT_SCHEMA_INVALID", "recon report schema is invalid"
    reason = str(exc)
    code = _REPORT_REJECTION_REASONS.get(reason)
    return (code, reason) if code is not None else ("REPORT_VALIDATION_FAILED", "recon report validation failed")


def _report_projection(jobs: Sequence[dict[str, Any]], raw_by_job: Mapping[str, str], receipts: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if set(raw_by_job) != {job["job_id"] for job in jobs}: raise ValueError("reports must exactly cover report jobs")
    successful_queries = {row["query_id"] for row in receipts if row.get("status") in {"SUCCESS", "EMPTY"}}
    latest_search = max((datetime.fromisoformat(row["completed_at"]) for row in receipts if row.get("status") in {"SUCCESS", "EMPTY"}), default=None)
    reports: list[dict[str, Any]] = []; rejected: list[dict[str, Any]] = []; outcomes: list[dict[str, Any]] = []; routes: list[dict[str, Any]] = []; killed: list[dict[str, Any]] = []
    for job in jobs:
        raw = raw_by_job[job["job_id"]]; raw_sha = hashlib.sha256(raw.encode()).hexdigest()
        try:
            item = _ReportInput.model_validate_json(_canonical_json(_strict_load(raw)), strict=True)
            for key in ("job_id", "opportunity_id", "query_pack_sha256", "normalized_evidence_manifest_sha256", "protocol_hash", "cache_key", "searched_query_ids"):
                if getattr(item, key) != job[key]: raise ValueError("stale recon report binding")
            lane_status = job.get("lane_query_ids", {})
            if set(lane_status) != set(_LANES) or any(not any(qid in successful_queries for qid in qids) for qids in lane_status.values()): raise ValueError("each recon lane requires a SUCCESS or EMPTY receipt")
            if latest_search is not None and item.searched_at < latest_search:
                raise ValueError("searched_at precedes execution completion")
            permitted = {row["evidence_id"]: row for row in job["evidence"]}
            priors = item.nearest_priors
            for prior in priors:
                reference = prior.evidence_url_or_id
                matched = next((row for evidence_id, row in permitted.items() if reference in {evidence_id, row.get("url"), *row.get("source_aliases", [])}), None)
                if matched is None: raise ValueError("nearest prior URL or ID is not evidence-bound")
                if prior.paper != matched.get("title"):
                    raise ValueError("nearest prior paper must equal the canonical title")
            if item.decision == ReconDecision.NO_DIRECT_COVERAGE_FOUND:
                if item.decision_reason != NO_DIRECT_COVERAGE_REASON:
                    raise ValueError("NO_DIRECT_COVERAGE_FOUND must use the controlled literal reason exactly")
            elif not priors:
                raise ValueError("covered and near-prior decisions require evidence")
            elif not any(prior.paper.casefold() in item.decision_reason.casefold() or prior.evidence_url_or_id.casefold() in item.decision_reason.casefold() for prior in priors):
                raise ValueError("covered or near-prior decision reason must name its evidence-bound prior")
            report = {"schema_version": "idea_factory.recon_report.v1", "record_id": item.opportunity_id, "job_id": item.job_id, "raw_result_sha256": raw_sha, **item.model_dump(mode="json")}
            reports.append(report); target = killed if item.decision == ReconDecision.COVERED else routes
            target.append(report)
            outcomes.append({"schema_version": "idea_factory.recon_outcome.v2", "job_id": job["job_id"], "status": "ACCEPTED", "raw_result_json": raw, "raw_result_sha256": raw_sha, "report_id": item.opportunity_id, "decision": item.decision.value, "accepted_projection_sha256": _hash(report), "rejected_projection_sha256": _hash([])})
        except (ValidationError, ValueError) as exc:
            error_code, error = _bounded_report_rejection(exc)
            rejection = {"schema_version": "idea_factory.rejected_recon_report.v2", "record_id": stable_id("rejected_recon", job["job_id"], raw_sha), "job_id": job["job_id"], "raw_result_sha256": raw_sha, "error_code": error_code, "error": error}
            rejected.append(rejection)
            outcomes.append({"schema_version": "idea_factory.recon_outcome.v2", "job_id": job["job_id"], "status": "REJECTED", "raw_result_sha256": raw_sha, "error_code": error_code, "error": error, "report_id": None, "decision": None, "accepted_projection_sha256": _hash([]), "rejected_projection_sha256": _hash(rejection)})
    return reports, rejected, outcomes, routes + killed, routes, killed


_REPORT_OUTPUTS = ("reports.jsonl", "rejected.jsonl", "outcomes.jsonl", "ready_for_routes.jsonl", "killed.jsonl")


def ingest_recon_reports(report_jobs_path: Path, results: Sequence[str | bytes | Path], config: CorpusRouterConfig) -> dict[str, Path]:
    run = _anchored_run(Path(report_jobs_path).parent.parent); recon = _recon_dir(run, create=False)
    _safe_unlink(recon, _REPORT_OUTPUTS)
    try:
        if Path(os.path.abspath(report_jobs_path)) != _owned_file(recon, "report_jobs.jsonl"): raise ValueError("report jobs must use active recon path")
        jobs = validate_report_jobs(run, config); receipts = validate_execution_receipts(run, config)
        raw_by_job: dict[str, str] = {}
        for source in results:
            raw = source.read_bytes() if isinstance(source, Path) else source.encode() if type(source) is str else source
            data = _strict_load(raw); job_id = data.get("job_id")
            if type(job_id) is not str or job_id not in {job["job_id"] for job in jobs} or job_id in raw_by_job: raise ValueError("unexpected or duplicate report job")
            raw_by_job[job_id] = raw.decode("utf-8")
        reports, rejected, outcomes, _all, routes, killed = _report_projection(jobs, raw_by_job, receipts)
        paths = {"reports": recon / "reports.jsonl", "rejected": recon / "rejected.jsonl", "outcomes": recon / "outcomes.jsonl", "ready": recon / "ready_for_routes.jsonl", "killed": recon / "killed.jsonl"}
        write_jsonl_bundle({paths["reports"]: reports, paths["rejected"]: rejected, paths["outcomes"]: outcomes, paths["ready"]: routes, paths["killed"]: killed})
        return paths
    except BaseException:
        _safe_unlink(recon, _REPORT_OUTPUTS)
        raise


def validate_recon_bundle(run_dir: Path, config: CorpusRouterConfig) -> dict[str, tuple[dict[str, Any], ...]]:
    run = _anchored_run(Path(run_dir)); recon = _recon_dir(run, create=False)
    reports = read_jsonl(_owned_file(recon, "reports.jsonl")); rejected = read_jsonl(_owned_file(recon, "rejected.jsonl")); outcomes = read_jsonl(_owned_file(recon, "outcomes.jsonl")); ready = read_jsonl(_owned_file(recon, "ready_for_routes.jsonl")); killed = read_jsonl(_owned_file(recon, "killed.jsonl"))
    jobs = validate_report_jobs(run, config); receipts = validate_execution_receipts(run, config)
    if [row.get("job_id") for row in outcomes] != [job["job_id"] for job in jobs]:
        raise ValueError("recon outcomes must follow canonical report job order")
    rejected_by_job = {row.get("job_id"): row for row in rejected}
    if len(rejected_by_job) != len(rejected):
        raise ValueError("rejected recon reports contain duplicate jobs")
    allowed_errors = {
        ("REPORT_SCHEMA_INVALID", "recon report schema is invalid"),
        ("REPORT_VALIDATION_FAILED", "recon report validation failed"),
        *((code, reason) for reason, code in _REPORT_REJECTION_REASONS.items()),
    }
    expected_reports: list[dict[str, Any]] = []
    expected_rejected: list[dict[str, Any]] = []
    expected_outcomes: list[dict[str, Any]] = []
    expected_ready: list[dict[str, Any]] = []
    expected_killed: list[dict[str, Any]] = []
    for job, outcome in zip(jobs, outcomes):
        if outcome.get("status") == "ACCEPTED":
            raw, raw_sha = outcome.get("raw_result_json"), outcome.get("raw_result_sha256")
            if type(raw) is not str or hashlib.sha256(raw.encode()).hexdigest() != raw_sha:
                raise ValueError("accepted recon outcome raw replay binding mismatch")
            projected = _report_projection((job,), {job["job_id"]: raw}, receipts)
            if projected[1] or projected[2] != [outcome]:
                raise ValueError("accepted recon outcome replay mismatch")
            expected_reports.extend(projected[0]); expected_outcomes.extend(projected[2]); expected_ready.extend(projected[4]); expected_killed.extend(projected[5])
            continue
        rejected_keys = {"schema_version", "job_id", "status", "raw_result_sha256", "error_code", "error", "report_id", "decision", "accepted_projection_sha256", "rejected_projection_sha256"}
        if set(outcome) != rejected_keys or outcome.get("schema_version") != "idea_factory.recon_outcome.v2" or outcome.get("status") != "REJECTED" or (outcome.get("error_code"), outcome.get("error")) not in allowed_errors or not re.fullmatch(r"[0-9a-f]{64}", str(outcome.get("raw_result_sha256", ""))):
            raise ValueError("rejected recon outcome schema or bounded reason mismatch")
        rejection = rejected_by_job.get(job["job_id"])
        raw_sha = outcome["raw_result_sha256"]
        expected_rejection = {"schema_version": "idea_factory.rejected_recon_report.v2", "record_id": stable_id("rejected_recon", job["job_id"], raw_sha), "job_id": job["job_id"], "raw_result_sha256": raw_sha, "error_code": outcome["error_code"], "error": outcome["error"]}
        if rejection != expected_rejection or outcome.get("report_id") is not None or outcome.get("decision") is not None or outcome.get("accepted_projection_sha256") != _hash([]) or outcome.get("rejected_projection_sha256") != _hash(expected_rejection):
            raise ValueError("rejected recon outcome replay mismatch")
        expected_rejected.append(expected_rejection); expected_outcomes.append(outcome)
    if (reports, rejected, outcomes, ready, killed) != (expected_reports, expected_rejected, expected_outcomes, expected_ready, expected_killed): raise ValueError("recon bundle replay mismatch")
    return {"reports": tuple(reports), "rejected": tuple(rejected), "outcomes": tuple(outcomes), "ready": tuple(ready), "killed": tuple(killed)}
