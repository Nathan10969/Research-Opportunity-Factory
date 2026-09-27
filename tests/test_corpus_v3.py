from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import idea_factory.corpus as corpus_module

from idea_factory.artifacts import read_jsonl
from idea_factory.corpus import (
    PROMPT_VERSION,
    REQUIRED_OUTPUT_SCHEMA,
    router_job_id,
    V3_REQUIRED_OUTPUT_SCHEMA,
    _validate_router_jobs,
    enumerate_candidates,
    ingest_router_results,
    select_pilot,
    validate_corpus_selection_bundle,
    write_router_jobs,
)
from idea_factory.models import CorpusLabel
from test_corpus import config as fixture_config, scoped_config


FROZEN_V2_LABELS = [
    "KV_CACHE",
    "LONG_MEMORY",
    "BRIDGE",
    "HUMAN_SUPERVISION",
    "SHIFT_ROBUSTNESS",
    "SUPERVISION_SHIFT_BRIDGE",
    "OTHER",
]


def _primary_source_manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"schema_version": "idea_factory.primary_source_manifest.v1", "records": records}, sort_keys=True), encoding="utf-8")
    return path


def _primary_record(note: Path, pdf: Path | None, *, aliases: list[str], status: str = "PRIMARY_SOURCE_VERIFIED", adjudication: Path | None = None, slug: str | None = None, source_version: str | None = None) -> dict[str, object]:
    return {
        "slug": slug or (note.parent.name if note.name == "note.md" else note.stem.lower()),
        "canonical_note_path": str(note.resolve()),
        "note_sha256": hashlib.sha256(note.read_bytes()).hexdigest(),
        "primary_pdf_path": str(pdf.resolve()) if pdf is not None else None,
        "primary_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf is not None else None,
        "source_record_ids": aliases,
        "source_status": {
            "ACTIVE": "PRIMARY_SOURCE_VERIFIED",
            "WITHDRAWN_SOURCE": "SOURCE_WITHDRAWN",
            "REJECTED_SOURCE": "SOURCE_REJECTED",
        }.get(status, status),
        "source_version": source_version,
        "adjudication_path": str(adjudication.resolve()) if adjudication is not None else None,
        "adjudication_sha256": hashlib.sha256(adjudication.read_bytes()).hexdigest() if adjudication is not None else None,
    }


def _router_validation_config(tmp_path: Path):
    notes_root = tmp_path / "notes"
    notes_root.mkdir()
    note_names = ("paper-a", "paper-kv", "paper-memory", "paper-bridge", "paper-keyword")
    notes = []
    for name in note_names:
        note = notes_root / f"{name}.md"
        note.write_text(f"# {name}\n\nFixture content.\n", encoding="utf-8")
        notes.append(note)
    base = scoped_config(tmp_path, notes)
    list_root = base.legacy_ledger.parent.parent / "candidate-lists"
    list_root.mkdir()
    first = list_root / "list-one.txt"
    second = list_root / "list-two.txt"
    first.write_text("".join(f"{note}\n" for note in notes[:4]), encoding="utf-8")
    second.write_text("".join(f"{note}\n" for note in (notes[0], notes[3], notes[4])), encoding="utf-8")
    return replace(base, candidate_lists=(first, second))


def test_general_research_label_does_not_mutate_frozen_v2_schema() -> None:
    assert CorpusLabel.GENERAL_RESEARCH.value == "GENERAL_RESEARCH"
    assert REQUIRED_OUTPUT_SCHEMA["properties"]["label"]["enum"] == FROZEN_V2_LABELS
    assert V3_REQUIRED_OUTPUT_SCHEMA["properties"]["label"]["enum"] == ["GENERAL_RESEARCH", "OTHER"]


def test_v2_ingest_rejects_new_v3_label() -> None:
    candidate = enumerate_candidates(fixture_config())[0]
    prompt_sha = "a" * 64
    result = {
        "schema_version": "idea_factory.corpus_router_result.v2",
        "job_id": router_job_id(candidate.slug, candidate.note_sha256, prompt_sha),
        "slug": candidate.slug,
        "note_sha256": candidate.note_sha256,
        "prompt_sha256": prompt_sha,
        "label": "GENERAL_RESEARCH",
        "core_mechanism": "new v3-only label",
        "scope_reason": "must not widen the v2 enum",
        "evidence_locator": "note:1",
        "confidence": "HIGH",
    }
    with pytest.raises(ValueError, match="v2 router result label is not part of the frozen v2 enum"):
        ingest_router_results([candidate], [result], prompt_sha256=prompt_sha)


def test_v2_job_replay_rejects_a_dynamically_appended_enum(tmp_path: Path) -> None:
    from idea_factory.corpus import enumerate_candidates

    jobs_path = write_router_jobs(
        enumerate_candidates(fixture_config()),
        tmp_path / "run",
        prompt_path=Path(__file__).parents[1] / "prompts" / "corpus_router.md",
    )
    jobs = read_jsonl(jobs_path)
    for job in jobs:
        job["required_output_schema"]["properties"]["label"]["enum"].append("GENERAL_RESEARCH")
    jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8")
    with pytest.raises(ValueError, match="router job prompt or ID binding mismatch"):
        _validate_router_jobs(jobs_path)


def test_prechange_v2_job_fixture_replays_with_original_prompt_schema_and_ids(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "corpus_router_jobs_v2_frozen.jsonl"
    root = Path(__file__).parents[1].resolve()
    frozen_rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    frozen_prompt = tmp_path / "frozen-v2-prompt.md"
    frozen_prompt.write_bytes(frozen_rows[0]["prompt_text"].encode("utf-8"))
    frozen_text = "".join(
        json.dumps({
            **row,
            **{
                key: value.replace("__REPO_ROOT__", str(root)) if isinstance(value, str)
                else [item.replace("__REPO_ROOT__", str(root)) for item in value] if key == "source_lists"
                else value
                for key, value in row.items()
                if key in {"allowed_note_root", "allowed_papers_root", "legacy_ledger_path", "note_path", "source_lists"}
            },
            "prompt_path": str(frozen_prompt.resolve()),
            "candidate_list_manifest": [
                {**entry, "path": entry["path"].replace("__REPO_ROOT__", str(root))}
                for entry in row["candidate_list_manifest"]
            ],
            "source_list_bindings": [
                {**entry, "path": entry["path"].replace("__REPO_ROOT__", str(root))}
                for entry in row["source_list_bindings"]
            ],
        }, sort_keys=True) + "\n"
        for row in frozen_rows
    )
    run = tmp_path / "fixture-run"
    jobs_path = run / "corpus" / "router_jobs.jsonl"
    jobs_path.parent.mkdir(parents=True)
    jobs_path.write_text(frozen_text, encoding="utf-8")
    replayed = _validate_router_jobs(jobs_path)
    assert len(replayed) == 5
    assert all(job["prompt_version"] == PROMPT_VERSION == "idea_factory.corpus_router_prompt.v2" for job in replayed)
    assert all(job["result_schema_version"] == "idea_factory.corpus_router_result.v2" for job in replayed)
    assert all(job["required_output_schema"] == REQUIRED_OUTPUT_SCHEMA for job in replayed)
    assert all(job["job_id"] == router_job_id(job["slug"], job["note_sha256"], job["prompt_sha256"]) for job in replayed)

    current_path = write_router_jobs(
        __import__("idea_factory.corpus", fromlist=["enumerate_candidates"]).enumerate_candidates(fixture_config()),
        tmp_path / "current-run",
        prompt_path=root / "prompts" / "corpus_router.md",
    )
    current_jobs = read_jsonl(current_path)
    assert len(current_jobs) == len(replayed)
    assert all(job["schema_version"] == "idea_factory.corpus_router_job.v2" for job in current_jobs)
    assert all(job["required_output_schema"] == REQUIRED_OUTPUT_SCHEMA for job in current_jobs)
    assert all(job["job_id"] == router_job_id(job["slug"], job["note_sha256"], job["prompt_sha256"]) for job in current_jobs)


def test_router_validation_expands_each_candidate_list_once_per_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from idea_factory.corpus import enumerate_candidates

    prompt = Path(__file__).parents[1] / "prompts" / "corpus_router.md"
    jobs_path = write_router_jobs(
        enumerate_candidates(_router_validation_config(tmp_path)), tmp_path / "run", prompt_path=prompt,
    )
    jobs = read_jsonl(jobs_path)
    expected_paths = {
        entry["path"]
        for job in jobs
        for entry in job["source_list_bindings"]
    }
    original = corpus_module._read_candidate_paths
    expanded: list[str] = []

    def tracked(source_list: Path):
        expanded.append(str(source_list))
        return original(source_list)

    monkeypatch.setattr(corpus_module, "_read_candidate_paths", tracked)
    assert len(_validate_router_jobs(jobs_path)) == len(jobs)
    assert len(expanded) == len(expected_paths)
    assert set(expanded) == expected_paths
    assert len(_validate_router_jobs(jobs_path)) == len(jobs)
    assert len(expanded) == 2 * len(expected_paths)


def test_router_validation_rejects_candidate_list_changed_during_membership_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from idea_factory.corpus import enumerate_candidates

    prompt = Path(__file__).parents[1] / "prompts" / "corpus_router.md"
    jobs_path = write_router_jobs(
        enumerate_candidates(_router_validation_config(tmp_path)), tmp_path / "run", prompt_path=prompt,
    )
    jobs = read_jsonl(jobs_path)
    list_path = Path(jobs[0]["candidate_list_manifest"][0]["path"])
    original = corpus_module._read_candidate_paths
    changed = False

    def mutate_after_first_scan(source_list: Path):
        nonlocal changed
        paths = original(source_list)
        if source_list == list_path and not changed:
            source_list.write_bytes(source_list.read_bytes() + b"\n")
            changed = True
        return paths

    monkeypatch.setattr(corpus_module, "_read_candidate_paths", mutate_after_first_scan)
    with pytest.raises(ValueError, match="candidate list changed during router job validation"):
        _validate_router_jobs(jobs_path)


def test_v3_admission_jobs_select_general_research_but_reject_other(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    admitted = notes / "acl-paper.md"
    rejected = notes / "unrelated.md"
    admitted.write_text("# ACL paper\n\nSource: https://aclanthology.org/2025.acl-long.1/\n\nWe evaluate retrieval under distribution shift.", encoding="utf-8")
    rejected.write_text("# Unsupported item\n\nNo source or research content.", encoding="utf-8")
    base = scoped_config(tmp_path, [admitted, rejected])
    paper_root = base.legacy_ledger.parent.parent / "_source"
    paper_root.mkdir()
    pdf1, pdf2 = paper_root / "acl.pdf", paper_root / "other.pdf"
    pdf1.write_bytes(b"%PDF ACL paper")
    pdf2.write_bytes(b"%PDF unrelated record")
    primary_manifest = _primary_source_manifest(tmp_path / "primary-sources.json", [
        _primary_record(admitted, pdf1, aliases=["acl:2025.acl-long.1"]),
        _primary_record(rejected, pdf2, aliases=["fixture:unrelated"]),
    ])
    v3 = replace(base, target_max=10000, allowed_labels=("GENERAL_RESEARCH",), protocol_version="v3", source_primary_manifest=primary_manifest)
    prompt = Path(__file__).parents[1] / "prompts" / "corpus_router_v3.md"
    candidates = enumerate_candidates(v3)
    run = tmp_path / "run"
    jobs_path = write_router_jobs(candidates, run, prompt_path=prompt, protocol_version="v3")
    jobs = read_jsonl(jobs_path)
    assert all(job["prompt_version"] == "idea_factory.corpus_router_prompt.v3" for job in jobs)
    assert all(job["result_schema_version"] == "idea_factory.corpus_router_result.v3" for job in jobs)
    assert all(job["required_output_schema"] == V3_REQUIRED_OUTPUT_SCHEMA for job in jobs)
    results = [
        {
            "schema_version": job["result_schema_version"],
            "job_id": job["job_id"],
            "slug": job["slug"],
            "note_sha256": job["note_sha256"],
            "prompt_sha256": job["prompt_sha256"],
            "label": "GENERAL_RESEARCH" if job["slug"] == "acl-paper" else "OTHER",
            "core_mechanism": "retrieval evaluation under distribution shift" if job["slug"] == "acl-paper" else "no supported mechanism",
            "scope_reason": "source and substantive research content are present" if job["slug"] == "acl-paper" else "source evidence absent",
            "evidence_locator": "note:3" if job["slug"] == "acl-paper" else "note:1",
            "confidence": "HIGH",
        }
        for job in jobs
    ]
    ingested = ingest_router_results(
        candidates, results, prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest(), protocol_version="v3"
    )
    selected, rejected_items = select_pilot(v3, ingested, run)
    selected_rows, rejected_rows = validate_corpus_selection_bundle(run / "corpus" / "selection_manifest.jsonl")
    assert [item.candidate.slug for item in selected] == ["acl-paper"]
    assert [item.candidate.slug for item in rejected_items] == ["unrelated"]
    assert [row["label"] for row in selected_rows] == ["GENERAL_RESEARCH"]
    assert [row["reason_code"] for row in rejected_rows] == ["LABEL_OTHER"]


def test_v3_missing_candidate_source_binding_fails_closed(tmp_path: Path) -> None:
    note = tmp_path / "notes" / "unbound.md"
    note.parent.mkdir()
    note.write_text("# Unbound note\n", encoding="utf-8")
    base = scoped_config(tmp_path, [note])
    manifest = _primary_source_manifest(tmp_path / "primary-sources.json", [])
    config = replace(base, allowed_labels=("GENERAL_RESEARCH",), protocol_version="v3", source_primary_manifest=manifest)
    with pytest.raises(ValueError, match="primary source manifest is missing canonical note binding"):
        enumerate_candidates(config)
    with pytest.raises(ValueError, match="v3 requires an explicit primary source manifest"):
        replace(config, source_primary_manifest=None)


def test_v3_public_cli_runs_through_paper_card_landscape_and_opportunity_jobs(tmp_path: Path) -> None:
    from idea_factory.artifacts import read_jsonl
    from idea_factory.cli import main
    from idea_factory.cards import validate_card_result_bundle
    from idea_factory.landscape import validate_landscape_bundle

    fixture_root = Path(os.environ.get("GENERAL_CORPUS_E2E_DIR", tmp_path)).resolve()
    fixture_root.mkdir(parents=True, exist_ok=True)
    note = fixture_root / "notes" / "workers" / "w22" / "items" / "acl-retrieval-1" / "note.md"
    second_note = fixture_root / "notes" / "workers" / "w22" / "items" / "acl-retrieval-2" / "note.md"
    empty_note = fixture_root / "notes" / "workers" / "w22" / "items" / "acl-retrieval-3" / "note.md"
    third_card_note = fixture_root / "notes" / "workers" / "w22" / "items" / "acl-retrieval-4" / "note.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    second_note.parent.mkdir(parents=True, exist_ok=True)
    empty_note.parent.mkdir(parents=True, exist_ok=True)
    third_card_note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# ACL retrieval study\n\n"
        "Source: https://aclanthology.org/2025.acl-long.1/\n\n"
        "## Evidence\n\n"
        "The authors evaluate retrieval under distribution shift and report failures when the working set changes.\n",
        encoding="utf-8",
    )
    second_note.write_text(
        "# ACL retrieval evaluation\n\n"
        "Source: https://aclanthology.org/2025.acl-long.2/\n\n"
        "## Evidence\n\n"
        "The study measures failures in a separate retrieval setting under changing query distributions.\n",
        encoding="utf-8",
    )
    empty_reason = "No schema-valid PaperCard facts extracted; retain the constructive source note for future synthesis."
    empty_note.write_text(
        "# ACL retrieval note retained without a PaperCard\n\n"
        "Source: https://aclanthology.org/2025.acl-long.3/\n\n"
        "## Constructive note\n\n"
        "The source reports a separate retrieval benchmark and describes where its evidence is insufficiently specified. "
        "This observation remains available for later landscape review even though no PaperCard fields were extracted.\n\n"
        f"Migration provenance (fixture): empty_reason={empty_reason}\n",
        encoding="utf-8",
    )
    third_card_note.write_text(
        "# ACL retrieval study third Card\n\n"
        "Source: https://aclanthology.org/2025.acl-long.4/\n\n"
        "## Evidence\n\n"
        "The study evaluates retrieval under distribution shift and reports degraded results when the active working set changes.\n",
        encoding="utf-8",
    )
    resolved_notes = scoped_config(fixture_root, [note, second_note, empty_note, third_card_note])
    config = fixture_root / "v3-config.json"
    pdf1 = resolved_notes.legacy_ledger.parent.parent / "_source" / "acl-1.pdf"
    pdf2 = resolved_notes.legacy_ledger.parent.parent / "_source" / "acl-2.pdf"
    pdf3 = resolved_notes.legacy_ledger.parent.parent / "_source" / "acl-3.pdf"
    pdf4 = resolved_notes.legacy_ledger.parent.parent / "_source" / "acl-4.pdf"
    pdf1.parent.mkdir(parents=True, exist_ok=True)
    pdf1.write_bytes(b"%PDF ACL retrieval study one")
    pdf2.write_bytes(b"%PDF ACL retrieval study two")
    pdf3.write_bytes(b"%PDF ACL retrieval note three")
    pdf4.write_bytes(b"%PDF ACL retrieval study four")
    primary_manifest = _primary_source_manifest(fixture_root / "primary-sources.json", [
        _primary_record(note, pdf1, aliases=["acl:2025.acl-long.1"]),
        _primary_record(second_note, pdf2, aliases=["acl:2025.acl-long.2"]),
        _primary_record(empty_note, pdf3, aliases=["acl:2025.acl-long.3"]),
        _primary_record(third_card_note, pdf4, aliases=["acl:2025.acl-long.4"]),
    ])
    config.write_text(json.dumps({
        "notes_root": str(resolved_notes.notes_root),
        "candidate_lists": [str(path) for path in resolved_notes.candidate_lists],
        "legacy_ledger": str(resolved_notes.legacy_ledger),
        "target_min": 1,
        "target_max": 10000,
        "allowed_labels": ["GENERAL_RESEARCH"],
        "bridge_regression_slugs": [],
        "protocol_version": "v3",
        "source_primary_manifest": str(primary_manifest),
    }), encoding="utf-8")
    run = fixture_root / "run"
    assert main(["init-run", "--run", str(run), "--config", str(config), "--mode", "offline-fixture", "--config-repo-root", str(fixture_root)]) == 0
    assert main(["emit-corpus-jobs", "--run", str(run), "--config", str(config)]) == 0
    router_jobs = read_jsonl(run / "corpus" / "router_jobs.jsonl")
    assert len(router_jobs) == 4
    assert len({job["slug"] for job in router_jobs}) == 4
    assert {job["note_path"] for job in router_jobs} == {str(note.resolve()), str(second_note.resolve()), str(empty_note.resolve()), str(third_card_note.resolve())}
    labels = fixture_root / "labels.jsonl"
    labels.write_text("".join(json.dumps({
        "schema_version": router_job["result_schema_version"],
        **{key: router_job[key] for key in ("job_id", "slug", "note_sha256", "prompt_sha256")},
        "label": "GENERAL_RESEARCH", "core_mechanism": "retrieval fails under working-set shift",
        "scope_reason": "the source note contains bibliographic provenance and substantive evidence",
        "evidence_locator": "note:Evidence", "confidence": "HIGH",
    }) + "\n" for router_job in router_jobs), encoding="utf-8")
    assert main(["ingest-corpus-labels", "--run", str(run), "--config", str(config), "--results", str(labels)]) == 0
    assert main(["emit-card-jobs", "--run", str(run), "--config", str(config)]) == 0
    card_jobs = read_jsonl(run / "cards" / "card_jobs.jsonl")
    cards_payload = []
    for index, card_job in enumerate(card_jobs):
        if card_job["note_path"] == str(empty_note.resolve()):
            cards_payload.append(None)
            continue
        cards_payload.append({
            "schema_version": "idea_factory.paper_card.v1", "card_id": f"acl-evidence-card-{index + 1}",
            "paper": {"title": f"Retrieval under distribution shift {index + 1}", "year": 2025, "venue": "ACL", "source_path": card_job["note_path"]},
            "problem": "retrieval quality degrades under distribution shift",
            "assumption": {"text": "the working set remains stable", "status": "REPORTED"},
            "mechanism": "retrieval over a changing working set",
            "failure_observation": {"text": "performance drops when the working set changes", "status": "OBSERVED"},
            "failure_mechanism": {"text": "the retrieved context no longer covers the active distribution", "status": "INFERRED"},
            "limitation": "one evaluated retrieval setting",
            "evaluation": {"measurement": "retrieval quality", "regime": "distribution shift"},
            "scope": {"object": "retrieval system", "time_horizon": "evaluation window", "setting": "document retrieval"},
            "evidence_pointers": [{
                "source": "NOTE", "locator": card_job["note_path"] + "#Evidence",
                "supports": "problem,assumption,mechanism,failure_observation,failure_mechanism,limitation,evaluation.measurement,evaluation.regime,scope.object,scope.time_horizon,scope.setting",
            }],
            "extraction_confidence": "HIGH",
        })
    cards = fixture_root / "cards.jsonl"
    cards.write_text("".join(json.dumps({
        "schema_version": "idea_factory.paper_card_result.v1",
        **{key: card_job[key] for key in ("job_id", "slug", "note_sha256", "prompt_sha256")},
        "cards": [] if card is None else [card],
    }) + "\n" for card_job, card in zip(card_jobs, cards_payload)), encoding="utf-8")
    assert main(["ingest-cards", "--run", str(run), "--config", str(config), "--results", str(cards)]) == 0
    assert main(["build-landscape", "--run", str(run), "--config", str(config)]) == 0
    card_bundle = validate_card_result_bundle(run / "cards" / "card_jobs.jsonl", expected_run_dir=run)
    outcomes = card_bundle[3]
    empty_outcome = next(row for row in outcomes if row["slug"] == "acl-retrieval-3")
    assert empty_outcome["status"] == "VALID_EMPTY"
    assert empty_outcome["accepted_count"] == 0
    assert empty_outcome["note_sha256"] == hashlib.sha256(empty_note.read_bytes()).hexdigest()
    assert empty_reason in empty_note.read_text(encoding="utf-8")
    assert len(outcomes) == 4
    assert len(read_jsonl(run / "corpus" / "selection_manifest.jsonl")) == 4
    validated_landscape = validate_landscape_bundle(run)
    assert len(validated_landscape.cards) == 3
    assert str(empty_note.resolve()) not in {card.paper.source_path for card in validated_landscape.cards.values()}
    assert main(["emit-opportunity-jobs", "--run", str(run), "--config", str(config)]) == 0
    from idea_factory.opportunities import validate_mining_job_bundle
    mining_bundle = validate_mining_job_bundle(run)
    assert len(mining_bundle.jobs) >= 1
    assert (run / "corpus" / "selection_manifest.jsonl").is_file()
    assert (run / "results" / "paper_cards.jsonl").is_file()
    assert (run / "landscape" / "assumptions.json").is_file()
    assert (run / "opportunities" / "mining_jobs.jsonl").is_file()
    cards_ingested = read_jsonl(run / "results" / "paper_cards.jsonl")
    assert {row["card"]["paper"]["source_path"] for row in cards_ingested} == {str(note.resolve()), str(second_note.resolve()), str(third_card_note.resolve())}
    assert all(row["card"]["evidence_pointers"][0]["locator"].startswith(row["card"]["paper"]["source_path"] + "#") for row in cards_ingested)


def test_v3_source_aliases_do_not_duplicate_occupancy_and_job_extras_fail_closed(tmp_path: Path) -> None:
    note = tmp_path / "notes" / "same-source.md"
    note.parent.mkdir()
    note.write_text("# One source\n\nSource: https://arxiv.org/abs/2601.12345\n", encoding="utf-8")
    base = scoped_config(tmp_path, [note])
    pdf = base.legacy_ledger.parent.parent / "_source" / "same.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF same source")
    primary = _primary_source_manifest(tmp_path / "primary-sources.json", [_primary_record(note, pdf, aliases=["arxiv:2601.12345v1"], source_version="v1")])
    alias_list = base.candidate_lists[0].with_name("alias-list.txt")
    alias_list.write_text(str(note.resolve()) + "\n", encoding="utf-8")
    config = replace(base, candidate_lists=(*base.candidate_lists, alias_list), allowed_labels=("GENERAL_RESEARCH",), protocol_version="v3", source_primary_manifest=primary)
    candidates = enumerate_candidates(config)
    assert len(candidates) == 1
    assert len(candidates[0].source_lists) == 2
    run = tmp_path / "run"
    prompt = Path(__file__).parents[1] / "prompts" / "corpus_router_v3.md"
    jobs_path = write_router_jobs(candidates, run, prompt_path=prompt, protocol_version="v3")
    jobs = read_jsonl(jobs_path)
    assert len(jobs) == 1
    assert len(jobs[0]["source_lists"]) == 2
    jobs[0]["unbound_extra"] = "must reject"
    jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        _validate_router_jobs(jobs_path)


def test_v3_identical_pdf_source_aliases_keep_provenance_but_one_occupancy(tmp_path: Path) -> None:
    first = tmp_path / "notes" / "149160-nips.md"
    second = tmp_path / "notes" / "2609.27747v1-arxiv.md"
    first.parent.mkdir()
    first.write_text("# Conference record\n\nSource record: nips:149160\n", encoding="utf-8")
    second.write_text("# arXiv record\n\nSource record: arxiv_existing:2609.27747v1\n", encoding="utf-8")
    base = scoped_config(tmp_path, [first, second])
    pdf_root = base.legacy_ledger.parent.parent / "_source"
    pdf_root.mkdir()
    pdf1, pdf2 = pdf_root / "first.pdf", pdf_root / "second.pdf"
    pdf1.write_bytes(b"%PDF identical primary source")
    pdf2.write_bytes(b"%PDF identical primary source")
    alias_manifest = _primary_source_manifest(tmp_path / "primary-sources.json", [
        _primary_record(first, pdf1, aliases=["nips:149160"]),
        _primary_record(second, pdf2, aliases=["arxiv_existing:2609.27747v1"], source_version="v1"),
    ])
    config = replace(base, target_max=10000, allowed_labels=("GENERAL_RESEARCH",), protocol_version="v3", source_primary_manifest=alias_manifest)
    candidates = enumerate_candidates(config)
    assert len(candidates) == 2
    assert {candidate.source_pdf_sha256 for candidate in candidates} == {hashlib.sha256(pdf1.read_bytes()).hexdigest()}
    prompt = Path(__file__).parents[1] / "prompts" / "corpus_router_v3.md"
    run = tmp_path / "run"
    jobs = read_jsonl(write_router_jobs(candidates, run, prompt_path=prompt, protocol_version="v3"))
    results = [{
        "schema_version": job["result_schema_version"], "job_id": job["job_id"],
        "slug": job["slug"], "note_sha256": job["note_sha256"],
        "prompt_sha256": job["prompt_sha256"], "label": "GENERAL_RESEARCH",
        "core_mechanism": "same source PDF identity", "scope_reason": "two aliases of identical bytes",
        "evidence_locator": "note:2", "confidence": "HIGH",
    } for job in jobs]
    ingested = ingest_router_results(candidates, results, prompt_sha256=jobs[0]["prompt_sha256"], protocol_version="v3")
    selected, rejected = select_pilot(config, ingested, run)
    rows, rejected_rows = validate_corpus_selection_bundle(run / "corpus" / "selection_manifest.jsonl")
    assert len(selected) == len(rows) == 1
    assert len(rejected) == len(rejected_rows) == 1
    assert rejected[0].reason_code == "SOURCE_ALIAS_DUPLICATE"
    retained_aliases = {alias for row in [*rows, *rejected_rows] for alias in row["source_record_aliases"]}
    assert retained_aliases == {"arxiv_existing:2609.27747v1", "nips:149160"}


def test_v3_primary_source_manifest_ignores_referenced_withdrawn_identity(tmp_path: Path) -> None:
    active = tmp_path / "notes" / "active.md"
    withdrawn = tmp_path / "notes" / "withdrawn.md"
    active.parent.mkdir()
    active.write_text("# Active\n\nReferences arXiv:2510.23594v4 as prior work.\n", encoding="utf-8")
    withdrawn.write_text("# Withdrawn source\n\nSource record: arxiv:2510.23594v4\n", encoding="utf-8")
    adjudication = tmp_path / "withdrawal.json"
    adjudication.write_text(json.dumps({
        "schema_version": "source_review_adjudication.v1",
        "reviewed_at_utc": "2026-09-27T11:31:52Z",
        "review_scope": "MODEL_REVIEW_ONLY_PRIMARY_SOURCE_STATUS",
        "human_approved": False,
        "event_id": "139756",
        "canonical_arxiv_id": "2510.23594", "source_version": "v4",
        "source_url": "https://arxiv.org/abs/2510.23594v4",
        "source_status": "WITHDRAWN_SOURCE", "scientific_admission": "QUARANTINE_FOR_SCIENTIFIC_USE",
        "evidence_summary": "author withdrawal; PDF absent",
        "original_download_status": "failed", "original_download_error": "missing PDF",
        "retry_decision": "DO_NOT_REPEAT_NORMAL_V4_PDF_RETRY",
        "silent_version_fallback_allowed": False, "original_artifacts_changed": False,
        "graph_ingested": False, "downstream_requirement": "retain version alias and quarantine",
    }), encoding="utf-8")
    base = scoped_config(tmp_path, [active, withdrawn])
    active_pdf = base.legacy_ledger.parent.parent / "_source" / "active.pdf"
    active_pdf.parent.mkdir(parents=True, exist_ok=True)
    active_pdf.write_bytes(b"%PDF active primary source")
    manifest = _primary_source_manifest(tmp_path / "primary-sources.json", [
        _primary_record(active, active_pdf, aliases=["acl:active"]),
        _primary_record(withdrawn, None, aliases=["arxiv:2510.23594v4"], status="WITHDRAWN_SOURCE", adjudication=adjudication, source_version="v4"),
    ])
    config = replace(base, target_max=10000, allowed_labels=("GENERAL_RESEARCH",), protocol_version="v3", source_primary_manifest=manifest)
    candidates = enumerate_candidates(config)
    assert next(item for item in candidates if item.slug == "active").source_status == "PRIMARY_SOURCE_VERIFIED"
    assert next(item for item in candidates if item.slug == "withdrawn").source_status == "SOURCE_WITHDRAWN"
    prompt = Path(__file__).parents[1] / "prompts" / "corpus_router_v3.md"
    run = tmp_path / "run"
    jobs = read_jsonl(write_router_jobs(candidates, run, prompt_path=prompt, protocol_version="v3"))
    responses = [{
        "schema_version": job["result_schema_version"], "job_id": job["job_id"],
        "slug": job["slug"], "note_sha256": job["note_sha256"],
        "prompt_sha256": job["prompt_sha256"], "label": "GENERAL_RESEARCH",
        "core_mechanism": "research mechanism", "scope_reason": "primary source manifest bound",
        "evidence_locator": "note:1", "confidence": "HIGH",
    } for job in jobs]
    ingested = ingest_router_results(candidates, responses, prompt_sha256=jobs[0]["prompt_sha256"], protocol_version="v3")
    selected, rejected = select_pilot(config, ingested, run)
    assert [item.candidate.slug for item in selected] == ["active"]
    assert [(item.candidate.slug, item.reason_code) for item in rejected] == [("withdrawn", "SOURCE_WITHDRAWN")]

    mismatched_manifest = _primary_source_manifest(tmp_path / "mismatched-primary-sources.json", [
        _primary_record(active, active_pdf, aliases=["acl:active"]),
        _primary_record(withdrawn, None, aliases=["arxiv:2510.23594v4"], status="WITHDRAWN_SOURCE", adjudication=adjudication, source_version="v3"),
    ])
    with pytest.raises(ValueError, match="version does not match its exact source alias"):
        enumerate_candidates(replace(config, source_primary_manifest=mismatched_manifest))


def test_v3_primary_pdf_identity_not_reference_alias_deduplicates_unrelated_sources(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    first, second = notes / "first.md", notes / "second.md"
    first.write_text("# First\n\nReference: nips:149160\n", encoding="utf-8")
    second.write_text("# Second\n\nReference: nips:149160\n", encoding="utf-8")
    base = scoped_config(tmp_path, [first, second])
    pdf_root = base.legacy_ledger.parent.parent / "_source"
    pdf_root.mkdir(parents=True)
    pdf1, pdf2 = pdf_root / "first.pdf", pdf_root / "second.pdf"
    pdf1.write_bytes(b"%PDF first distinct paper")
    pdf2.write_bytes(b"%PDF second distinct paper")
    manifest = _primary_source_manifest(tmp_path / "primary-sources.json", [
        _primary_record(first, pdf1, aliases=["acl:first"]),
        _primary_record(second, pdf2, aliases=["acl:second"]),
    ])
    config = replace(base, target_max=10000, allowed_labels=("GENERAL_RESEARCH",), protocol_version="v3", source_primary_manifest=manifest)
    candidates = enumerate_candidates(config)
    assert {item.source_pdf_sha256 for item in candidates} == {
        hashlib.sha256(pdf1.read_bytes()).hexdigest(), hashlib.sha256(pdf2.read_bytes()).hexdigest(),
    }
    prompt = Path(__file__).parents[1] / "prompts" / "corpus_router_v3.md"
    run = tmp_path / "run"
    jobs = read_jsonl(write_router_jobs(candidates, run, prompt_path=prompt, protocol_version="v3"))
    responses = [{
        "schema_version": job["result_schema_version"], "job_id": job["job_id"],
        "slug": job["slug"], "note_sha256": job["note_sha256"],
        "prompt_sha256": job["prompt_sha256"], "label": "GENERAL_RESEARCH",
        "core_mechanism": "independent source", "scope_reason": "primary PDF differs",
        "evidence_locator": "note:1", "confidence": "HIGH",
    } for job in jobs]
    ingested = ingest_router_results(candidates, responses, prompt_sha256=jobs[0]["prompt_sha256"], protocol_version="v3")
    selected, rejected = select_pilot(config, ingested, run)
    assert len(selected) == 2
    assert rejected == ()


def test_v3_cross_version_source_alias_with_different_pdf_requires_pending_adjudication(tmp_path: Path) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    first, second = notes / "arxiv-v1.md", notes / "arxiv-v2.md"
    first.write_text("# Version one\n", encoding="utf-8")
    second.write_text("# Version two\n", encoding="utf-8")
    base = scoped_config(tmp_path, [first, second])
    pdf_root = base.legacy_ledger.parent.parent / "_source"
    pdf_root.mkdir(parents=True)
    pdf1, pdf2 = pdf_root / "v1.pdf", pdf_root / "v2.pdf"
    pdf1.write_bytes(b"%PDF v1 bytes")
    pdf2.write_bytes(b"%PDF v2 bytes")
    records = [
        _primary_record(first, pdf1, aliases=["arxiv:2510.23594v4"], source_version="v4"),
        _primary_record(second, pdf2, aliases=["arxiv_existing:2510.23594v5"], source_version="v5"),
    ]
    active_manifest = _primary_source_manifest(tmp_path / "active-aliases.json", records)
    config = replace(base, target_max=10000, allowed_labels=("GENERAL_RESEARCH",), protocol_version="v3", source_primary_manifest=active_manifest)
    with pytest.raises(ValueError, match="source identity aliases across distinct primary PDFs.*SOURCE_STATUS_PENDING"):
        enumerate_candidates(config)

    for record in records:
        record["source_status"] = "SOURCE_STATUS_PENDING"
    pending_manifest = _primary_source_manifest(tmp_path / "pending-aliases.json", records)
    candidates = enumerate_candidates(replace(config, source_primary_manifest=pending_manifest))
    assert {candidate.source_status for candidate in candidates} == {"SOURCE_STATUS_PENDING"}


@pytest.mark.parametrize("changed_file", ["note", "pdf"])
def test_v3_primary_source_manifest_hash_change_fails_closed(tmp_path: Path, changed_file: str) -> None:
    note = tmp_path / "notes" / "bound.md"
    note.parent.mkdir()
    note.write_text("# Bound source\n", encoding="utf-8")
    base = scoped_config(tmp_path, [note])
    pdf = base.legacy_ledger.parent.parent / "_source" / "bound.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF bound source")
    manifest = _primary_source_manifest(tmp_path / "primary-sources.json", [_primary_record(note, pdf, aliases=["test:bound"])])
    config = replace(base, target_max=10000, allowed_labels=("GENERAL_RESEARCH",), protocol_version="v3", source_primary_manifest=manifest)
    target = note if changed_file == "note" else pdf
    target.write_bytes(target.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="primary source manifest.*(note|PDF).*hash"):
        enumerate_candidates(config)
