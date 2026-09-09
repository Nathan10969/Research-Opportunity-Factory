from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from idea_factory.corpus import (
    REQUIRED_OUTPUT_SCHEMA,
    CorpusRouterConfig,
    PROMPT_VERSION,
    ROUTER_JOB_SCHEMA_VERSION,
    ROUTER_RESULT_SCHEMA_VERSION,
    RouterResult,
    CorpusCandidate,
    IngestedRouterResult,
    _selection_decisions,
    enumerate_candidates,
    ingest_router_results,
    write_router_jobs,
)
from idea_factory.models import CorpusLabel


ROOT = Path(__file__).parents[1]


def test_d09_d07_labels_are_explicit_and_schema_is_derived() -> None:
    assert CorpusLabel.HUMAN_SUPERVISION.value == "HUMAN_SUPERVISION"
    assert CorpusLabel.SHIFT_ROBUSTNESS.value == "SHIFT_ROBUSTNESS"
    assert CorpusLabel.SUPERVISION_SHIFT_BRIDGE.value == "SUPERVISION_SHIFT_BRIDGE"
    assert REQUIRED_OUTPUT_SCHEMA["properties"]["label"]["enum"] == [
        label.value for label in CorpusLabel
    ]


def test_d09_d07_config_allows_three_labels_but_rejects_other_invalid_duplicate() -> None:
    common = dict(
        notes_root=ROOT / "notes",
        candidate_lists=(ROOT / "tests" / "fixtures" / "candidates" / "list-one.txt",),
        legacy_ledger=ROOT / "tests" / "fixtures" / "legacy-ledger.md",
        target_min=1,
        target_max=3,
        bridge_regression_slugs=(),
    )
    config = CorpusRouterConfig(
        **common,
        allowed_labels=("HUMAN_SUPERVISION", "SHIFT_ROBUSTNESS", "SUPERVISION_SHIFT_BRIDGE"),
    )
    assert config.allowed_labels == (
        "HUMAN_SUPERVISION", "SHIFT_ROBUSTNESS", "SUPERVISION_SHIFT_BRIDGE"
    )
    for labels in (("OTHER",), ("NOPE",), ("HUMAN_SUPERVISION", "HUMAN_SUPERVISION")):
        with pytest.raises(ValueError):
            CorpusRouterConfig(**common, allowed_labels=labels)


def test_d09_d07_prompt_requires_mechanisms_and_off_topic_other() -> None:
    prompt = (ROOT / "prompts" / "corpus_router_d09_d07.md").read_text(encoding="utf-8")
    assert "HUMAN_SUPERVISION" in prompt and "human feedback" in prompt
    assert "SHIFT_ROBUSTNESS" in prompt and "regime" in prompt
    assert "SUPERVISION_SHIFT_BRIDGE" in prompt and "both" in prompt
    assert "OTHER" in prompt and "incidental" in prompt
    assert '"schema_version":"idea_factory.corpus_router_result.v2"' in prompt


def test_d09_d07_config_uses_portable_source_lists_and_bounded_targets() -> None:
    config_path = ROOT / "config" / "pilot_d09_d07.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["notes_root"] == "../notes"
    assert payload["candidate_lists"] == [
        "../papers/_domain_rounds/D09_annotation_human/note_paths.txt",
        "../papers/_domain_rounds/D07_robustness_ood/note_paths.txt",
    ]
    assert payload["target_min"] == 30 and payload["target_max"] == 40
    assert payload["allowed_labels"] == [
        "HUMAN_SUPERVISION", "SHIFT_ROBUSTNESS", "SUPERVISION_SHIFT_BRIDGE"
    ]
    assert payload["bridge_regression_slugs"] == []


def test_d09_d07_config_resolves_hermetic_domain_layout(tmp_path: Path) -> None:
    repo_root = tmp_path / "idea_factory"
    notes = tmp_path / "notes"
    papers = tmp_path / "papers"
    (papers / "_ideas").mkdir(parents=True)
    notes.mkdir()
    (papers / "_ideas" / "idea_ledger.md").write_text("legacy", encoding="utf-8")
    for name in ("D09_annotation_human", "D07_robustness_ood"):
        domain = papers / "_domain_rounds" / name
        domain.mkdir(parents=True)
        (domain / "note_paths.txt").write_text("", encoding="utf-8")
    payload = json.loads((ROOT / "config" / "pilot_d09_d07.json").read_text(encoding="utf-8"))
    config_path = tmp_path / "config" / "pilot.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    config = CorpusRouterConfig.from_json(
        config_path, repo_root=repo_root
    )
    assert all(path.is_file() for path in config.candidate_lists)


def test_d09_d07_jobs_preserve_bindings_and_versions(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("human feedback and regime shift", encoding="utf-8")
    source = tmp_path / "list.txt"
    source.write_text(f"{note}\n", encoding="utf-8")
    legacy = tmp_path / "ledger.md"
    legacy.write_text("legacy", encoding="utf-8")
    config = CorpusRouterConfig(
        notes_root=tmp_path,
        candidate_lists=(source,),
        legacy_ledger=legacy,
        target_min=1,
        target_max=1,
        allowed_labels=("HUMAN_SUPERVISION",),
        bridge_regression_slugs=(),
    )
    jobs_path = write_router_jobs(enumerate_candidates(config), tmp_path, prompt_path=ROOT / "prompts" / "corpus_router_d09_d07.md")
    job = json.loads(jobs_path.read_text(encoding="utf-8").splitlines()[0])
    prompt_bytes = (ROOT / "prompts" / "corpus_router_d09_d07.md").read_bytes()
    assert job["schema_version"] == ROUTER_JOB_SCHEMA_VERSION
    assert job["result_schema_version"] == ROUTER_RESULT_SCHEMA_VERSION
    assert job["prompt_version"] == PROMPT_VERSION
    assert job["note_sha256"] == hashlib.sha256(note.read_bytes()).hexdigest()
    assert job["prompt_sha256"] == hashlib.sha256(prompt_bytes).hexdigest()
    assert job["source_lists"] == [str(source.resolve())]
    ingested = ingest_router_results(
        enumerate_candidates(config),
        [{
            "schema_version": ROUTER_RESULT_SCHEMA_VERSION,
            "job_id": job["job_id"], "slug": job["slug"],
            "note_sha256": job["note_sha256"], "prompt_sha256": job["prompt_sha256"],
            "label": "HUMAN_SUPERVISION", "core_mechanism": "human feedback",
            "scope_reason": "supervision mechanism", "evidence_locator": "note:1",
            "confidence": "HIGH",
        }],
        prompt_sha256=str(job["prompt_sha256"]),
    )
    assert ingested[0].result.label is CorpusLabel.HUMAN_SUPERVISION


def test_d09_d07_v1_router_result_is_rejected_after_v2_bump() -> None:
    with pytest.raises(ValidationError):
        RouterResult.model_validate({
            "schema_version": "idea_factory.corpus_router_result.v1",
            "job_id": "job", "slug": "slug", "note_sha256": "0" * 64,
            "prompt_sha256": "0" * 64, "label": "KV_CACHE",
            "core_mechanism": "mechanism", "scope_reason": "scope",
            "evidence_locator": "note:1", "confidence": "HIGH",
        })


def test_d09_d07_rejects_other_and_low_confidence_from_selection(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("evidence", encoding="utf-8")
    digest = hashlib.sha256(note.read_bytes()).hexdigest()
    def item(slug: str, label: str, confidence: str) -> IngestedRouterResult:
        candidate = CorpusCandidate(slug, note, (), digest)
        result = RouterResult(
            schema_version=ROUTER_RESULT_SCHEMA_VERSION,
            job_id="job", slug=slug, note_sha256=digest,
            prompt_sha256="0" * 64, label=label,
            core_mechanism="mechanism", scope_reason="scope",
            evidence_locator="note:1", confidence=confidence,
        )
        return IngestedRouterResult(candidate, result, "0" * 64)

    config = CorpusRouterConfig(
        notes_root=tmp_path, candidate_lists=(tmp_path / "list.txt",),
        legacy_ledger=tmp_path / "ledger.md", target_min=1, target_max=3,
        allowed_labels=("HUMAN_SUPERVISION",),
        bridge_regression_slugs=(),
    )
    selected, rejected = _selection_decisions(
        [item("other", "OTHER", "HIGH"), item("low", "HUMAN_SUPERVISION", "LOW"), item("unconfigured", "SHIFT_ROBUSTNESS", "HIGH"), item("good", "HUMAN_SUPERVISION", "HIGH")], config
    )
    assert [entry.candidate.slug for entry in selected] == ["good"]
    assert {entry.reason_code for entry in rejected} == {"LABEL_OTHER", "CONFIDENCE_LOW", "LABEL_NOT_ALLOWED"}


def test_d09_d07_tampered_prompt_binding_is_rejected(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("evidence", encoding="utf-8")
    source = tmp_path / "list.txt"
    source.write_text(f"{note}\n", encoding="utf-8")
    ledger = tmp_path / "ledger.md"
    ledger.write_text("legacy", encoding="utf-8")
    config = CorpusRouterConfig(tmp_path, (source,), ledger, 1, 1, ("HUMAN_SUPERVISION",), ())
    candidate = enumerate_candidates(config)[0]
    with pytest.raises(ValueError, match="prompt_sha256"):
        ingest_router_results(
            (candidate,),
            [{"schema_version": ROUTER_RESULT_SCHEMA_VERSION, "job_id": "job", "slug": candidate.slug,
              "note_sha256": candidate.note_sha256, "prompt_sha256": "f" * 64,
              "label": "HUMAN_SUPERVISION", "core_mechanism": "feedback", "scope_reason": "scope",
              "evidence_locator": "note:1", "confidence": "HIGH"}],
            prompt_sha256="0" * 64,
        )
