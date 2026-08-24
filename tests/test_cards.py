from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import idea_factory.artifacts as artifacts_module
from idea_factory.artifacts import read_jsonl
from idea_factory.cards import (
    CARD_JOB_SCHEMA_VERSION,
    CARD_RESULT_SCHEMA_VERSION,
    CardResultWrapper,
    card_job_id,
    emit_card_jobs,
    ingest_card_results,
    validate_card_result_bundle,
)
from idea_factory.corpus import (
    PROMPT_VERSION as ROUTER_PROMPT_VERSION,
    REQUIRED_OUTPUT_SCHEMA as ROUTER_OUTPUT_SCHEMA,
    ROUTER_JOB_SCHEMA_VERSION,
    ROUTER_RESULT_SCHEMA_VERSION,
    SELECTION_POLICY_SCHEMA_VERSION,
    STRATIFICATION_VERSION,
    router_job_id,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_bundle(tmp_path: Path, names: tuple[str, ...] = ("alpha",)) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    notes = tmp_path / "notes"
    papers = tmp_path / "papers"
    legacy = papers / "_ideas" / "idea_ledger.md"
    corpus = run / "corpus"
    corpus.mkdir(parents=True)
    notes.mkdir()
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    notes_by_name: dict[str, Path] = {}
    for name in names:
        note = notes / f"{name}.md"
        note.write_text(f"# {name}\nMeasured failure in a table.", encoding="utf-8")
        notes_by_name[name] = note.resolve()
    candidate_list = papers / "list.txt"
    candidate_list.write_text("".join(str(notes_by_name[name]) + "\n" for name in names), encoding="utf-8")
    list_binding = {"path": str(candidate_list.resolve()), "sha256": _sha(candidate_list)}
    router_prompt = tmp_path / "corpus_router.md"
    router_prompt.write_text("route strictly", encoding="utf-8")
    router_prompt_sha = _sha(router_prompt)
    selected = []
    router_jobs = []
    for name in names:
        note = notes_by_name[name]
        job_id = router_job_id(name, _sha(note), router_prompt_sha)
        selected.append({
            "record_id": f"corpus-{name}", "slug": name, "note_path": str(note.resolve()),
            "note_sha256": _sha(note), "source_lists": [list_binding["path"]],
            "source_list_bindings": [list_binding], "candidate_list_manifest": [list_binding],
            "job_id": job_id, "result_schema_version": ROUTER_RESULT_SCHEMA_VERSION,
            "result_note_sha256": _sha(note), "prompt_sha256": router_prompt_sha,
            "label": "KV_CACHE", "confidence": "HIGH", "core_mechanism": "m",
            "scope_reason": "s", "evidence_locator": "note#heading", "raw_result_sha256": "b" * 64,
            "reason_code": "ELIGIBLE", "allowed_note_root": str(notes.resolve()),
            "allowed_papers_root": str(papers.resolve()),
            "legacy_ledger_path": str(legacy.resolve()),
        })
        router_jobs.append({
            "record_id": job_id, "schema_version": ROUTER_JOB_SCHEMA_VERSION,
            "job_id": job_id, "slug": name, "note_path": str(note), "note_sha256": _sha(note),
            "note_text": note.read_bytes().decode("utf-8"), "source_lists": [list_binding["path"]],
            "source_list_bindings": [list_binding], "candidate_list_manifest": [list_binding],
            "allowed_note_root": str(notes.resolve()), "allowed_papers_root": str(papers.resolve()),
            "legacy_ledger_path": str(legacy.resolve()), "prompt_id": "corpus_router",
            "prompt_version": ROUTER_PROMPT_VERSION, "prompt_path": str(router_prompt.resolve()),
            "prompt_sha256": router_prompt_sha, "prompt_text": router_prompt.read_text(encoding="utf-8"),
            "result_schema_version": ROUTER_RESULT_SCHEMA_VERSION,
            "required_output_schema": ROUTER_OUTPUT_SCHEMA,
        })
    selection = corpus / "selection_manifest.jsonl"
    rejected = corpus / "rejected_manifest.jsonl"
    router_jobs_path = corpus / "router_jobs.jsonl"
    router_jobs_path.write_text("".join(json.dumps(row) + "\n" for row in router_jobs), encoding="utf-8")
    router_results = []
    for row in selected:
        result_body = {
            "schema_version": ROUTER_RESULT_SCHEMA_VERSION, "job_id": row["job_id"],
            "slug": row["slug"], "note_sha256": row["note_sha256"],
            "prompt_sha256": row["prompt_sha256"], "label": row["label"],
            "core_mechanism": row["core_mechanism"], "scope_reason": row["scope_reason"],
            "evidence_locator": row["evidence_locator"], "confidence": row["confidence"],
        }
        raw_hash = hashlib.sha256(json.dumps(result_body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        row["raw_result_sha256"] = raw_hash
        router_results.append(result_body | {"raw_result_sha256": raw_hash})
    router_results_path = corpus / "router_results.jsonl"
    router_results_path.write_text("".join(json.dumps(row) + "\n" for row in router_results), encoding="utf-8")
    policy_body = {
        "schema_version": SELECTION_POLICY_SCHEMA_VERSION, "allowed_labels": ["KV_CACHE"],
        "accepted_confidences": ["HIGH", "MEDIUM"], "target_min": 1,
        "target_max": len(names), "mandatory_bridge_slugs": [],
        "stratification_version": STRATIFICATION_VERSION,
    }
    policy_hash = hashlib.sha256(json.dumps(policy_body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    (corpus / "selection_policy.jsonl").write_text(json.dumps(policy_body | {"selection_policy_sha256": policy_hash}) + "\n", encoding="utf-8")
    for row in selected:
        row["router_jobs_sha256"] = _sha(router_jobs_path)
        row["router_results_sha256"] = _sha(router_results_path)
        row["selection_policy_sha256"] = policy_hash
    selection.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    rejected.write_text("", encoding="utf-8")
    prompt = tmp_path / "paper_card.md"
    prompt.write_text("strict prompt", encoding="utf-8")
    return run, selection, prompt


def _job_bundle(tmp_path: Path, names: tuple[str, ...] = ("alpha",)) -> tuple[Path, list[dict[str, object]]]:
    run, selection, prompt = _selection_bundle(tmp_path, names)
    jobs = emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")
    return jobs, read_jsonl(jobs)


def _card(job: dict[str, object], card_id: str = "card-1", **overrides: object) -> dict[str, object]:
    note = str(job["note_path"])
    fields: dict[str, object] = {
        "schema_version": "idea_factory.paper_card.v1", "card_id": card_id,
        "paper": {"title": "Alpha", "year": 2026, "venue": "Test", "source_path": note},
        "problem": "slow retrieval", "assumption": {"text": "cache stays warm", "status": "REPORTED"},
        "mechanism": "eviction policy", "failure_observation": {"text": "latency rose", "status": "OBSERVED"},
        "failure_mechanism": {"text": "working set shifted", "status": "INFERRED"},
        "limitation": "single workload", "evaluation": {"measurement": "latency", "regime": "long context"},
        "scope": {"object": "cache", "time_horizon": "hours", "setting": "serving"},
        "evidence_pointers": [{"source": "NOTE", "locator": note + "# Results", "supports": "problem,assumption,mechanism,failure_observation,failure_mechanism,limitation,evaluation.measurement,evaluation.regime,scope.object,scope.time_horizon,scope.setting"}],
        "extraction_confidence": "HIGH",
    }
    fields.update(overrides)
    return fields


def _result(job: dict[str, object], cards: list[dict[str, object]]) -> dict[str, object]:
    return {key: job[key] for key in ("schema_version", "job_id", "slug", "note_sha256", "prompt_sha256")} | {"schema_version": CARD_RESULT_SCHEMA_VERSION, "cards": cards}


def test_emits_bound_jobs_and_ingests_two_cards_deterministically(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    job = jobs[0]
    assert job["schema_version"] == CARD_JOB_SCHEMA_VERSION
    assert job["job_id"] == card_job_id(str(job["slug"]), str(job["note_sha256"]), str(job["prompt_sha256"]))
    assert job["note_text"] == Path(str(job["note_path"])).read_bytes().decode("utf-8")
    assert job["selection_manifest_sha256"] == _sha(tmp_path / "run" / "corpus" / "selection_manifest.jsonl")
    assert job["rejected_manifest_sha256"] == _sha(tmp_path / "run" / "corpus" / "rejected_manifest.jsonl")

    accepted, rejected, outcomes = ingest_card_results(jobs_path, [_result(job, [_card(job, "a"), _card(job, "b")])])

    assert [row["card"]["card_id"] for row in accepted] == ["a", "b"]
    assert rejected == []
    assert outcomes[0]["status"] == "ACCEPTED"
    assert len(read_jsonl(tmp_path / "run" / "results" / "paper_cards.jsonl")) == 2


def test_zero_cards_is_valid_durable_empty_outcome_and_inferred_is_preserved(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    raw = _result(jobs[0], [])
    accepted, _, outcomes = ingest_card_results(jobs_path, [raw])
    assert accepted == []
    raw_hash = hashlib.sha256(json.dumps(raw, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert outcomes == [{
        "schema_version": "idea_factory.card_job_outcome.v2", "job_id": jobs[0]["job_id"],
        "slug": "alpha", "note_sha256": jobs[0]["note_sha256"], "prompt_sha256": jobs[0]["prompt_sha256"],
        "result_schema_version": CARD_RESULT_SCHEMA_VERSION, "raw_result_sha256": raw_hash,
        "status": "VALID_EMPTY", "accepted_card_ids": [], "rejected_card_ids": [],
        "accepted_count": 0, "rejected_count": 0,
        "accepted_cards_sha256": hashlib.sha256(b"[]").hexdigest(),
        "rejected_projection_sha256": hashlib.sha256(b"[]").hexdigest(),
        "raw_wrapper_reconstructable": True,
    }]

    accepted, _, outcomes = ingest_card_results(jobs_path, [_result(jobs[0], [_card(jobs[0])])])
    assert accepted[0]["card"]["failure_mechanism"]["status"] == "INFERRED"
    assert outcomes[0]["status"] == "ACCEPTED"


@pytest.mark.parametrize("empty", [False, True])
def test_read_only_bundle_validator_accepts_accepted_and_valid_empty_projections(tmp_path: Path, empty: bool) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    cards = [] if empty else [_card(jobs[0])]
    ingest_card_results(jobs_path, [_result(jobs[0], cards)])
    _, accepted, rejected, outcomes = validate_card_result_bundle(jobs_path)
    assert len(accepted) == (0 if empty else 1)
    assert rejected == []
    assert outcomes[0]["status"] == ("VALID_EMPTY" if empty else "ACCEPTED")
    assert outcomes[0]["raw_wrapper_reconstructable"] is True


@pytest.mark.parametrize("mutate, match", [
    (lambda job, card: card["paper"].__setitem__("source_path", str(job["note_path"]) + ".alias"), "paper.source_path must canonically equal job note_path"),
    (lambda job, card: card["evidence_pointers"][0].__setitem__("locator", "other.md#Results"), "evidence locator must use canonical-note-path#nonblank-locator"),
    (lambda job, card: card["evidence_pointers"][0].__setitem__("supports", "problem"), "missing evidence supports: assumption, mechanism, failure_observation, failure_mechanism, limitation, evaluation.measurement, evaluation.regime, scope.object, scope.time_horizon, scope.setting"),
])
def test_card_evidence_source_and_coverage_are_enforced(tmp_path: Path, mutate, match: str) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    card = _card(jobs[0])
    mutate(jobs[0], card)
    accepted, rejected, outcomes = ingest_card_results(jobs_path, [_result(jobs[0], [card])])
    assert accepted == []
    assert rejected[0]["error"] == match
    assert outcomes[0]["status"] == "INVALID"


def test_stale_missing_duplicate_and_unexpected_results_are_invalid_outcomes(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path, ("alpha", "beta"))
    stale = _result(jobs[0], [])
    stale["prompt_sha256"] = "f" * 64
    rogue = _result(jobs[0], [])
    rogue["job_id"] = "rogue"
    _, rejected, outcomes = ingest_card_results(jobs_path, [stale, _result(jobs[0], []), rogue])
    assert {row["error"] for row in rejected} >= {"result binding mismatch: prompt_sha256", "duplicate result for job", "unexpected result"}
    assert {row["status"] for row in outcomes} == {"INVALID"}


def test_duplicate_card_across_jobs_keeps_first_and_rejects_second(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path, ("alpha", "beta"))
    accepted, rejected, outcomes = ingest_card_results(jobs_path, [_result(jobs[1], [_card(jobs[1], "same")]), _result(jobs[0], [_card(jobs[0], "same")])])
    assert len(accepted) == 1
    assert rejected[0]["error"] == "duplicate card_id: same"
    assert [row["status"] for row in outcomes] == ["ACCEPTED", "INVALID"]


def test_partial_card_failure_publishes_all_three_files_and_regeneration_invalidates(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    bad = _card(jobs[0], "bad")
    bad["evidence_pointers"] = []
    accepted, rejected, outcomes = ingest_card_results(jobs_path, [_result(jobs[0], [_card(jobs[0], "ok"), bad])])
    assert [row["card"]["card_id"] for row in accepted] == ["ok"]
    assert rejected[0]["index"] == 1 and rejected[0]["card_id"] == "bad"
    assert outcomes[0]["status"] == "PARTIAL_VALID"
    assert outcomes[0]["accepted_card_ids"] == ["ok"]
    assert outcomes[0]["rejected_card_ids"] == ["bad"]
    assert outcomes[0]["accepted_count"] == outcomes[0]["rejected_count"] == 1
    assert accepted[0]["result_schema_version"] == CARD_RESULT_SCHEMA_VERSION
    assert rejected[0]["result_schema_version"] == CARD_RESULT_SCHEMA_VERSION
    assert accepted[0]["raw_result_sha256"] == rejected[0]["raw_result_sha256"] == outcomes[0]["raw_result_sha256"]
    assert outcomes[0]["accepted_cards_sha256"] == hashlib.sha256(
        json.dumps(accepted, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert outcomes[0]["rejected_projection_sha256"] == hashlib.sha256(
        json.dumps(rejected, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert outcomes[0]["raw_wrapper_reconstructable"] is False

    run = jobs_path.parent.parent
    selection = run / "corpus" / "selection_manifest.jsonl"
    new_prompt = jobs_path.parent / "new-prompt.md"
    new_prompt.write_text("new prompt", encoding="utf-8")
    emit_card_jobs(selection, new_prompt, jobs_path)
    results = run / "results"
    assert not (results / "paper_cards.jsonl").exists()
    assert not (results / "rejected_paper_cards.jsonl").exists()
    assert not (results / "card_job_outcomes.jsonl").exists()


def test_refuses_incomplete_bundle_and_unsafe_note_or_owned_path(tmp_path: Path) -> None:
    run, selection, prompt = _selection_bundle(tmp_path)
    (run / "corpus" / "rejected_manifest.jsonl").unlink()
    with pytest.raises(ValueError, match="complete selection bundle"):
        emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")

    run, selection, prompt = _selection_bundle(tmp_path / "second")
    row = json.loads(selection.read_text(encoding="utf-8"))
    audit = Path(row["note_path"]).with_name("alpha.AUDIT.md")
    audit.write_text("secret", encoding="utf-8")
    row["note_path"] = str(audit.resolve())
    row["note_sha256"] = _sha(audit)
    selection.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection bundle|candidate provenance"):
        emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")


def test_refuses_mixed_root_provenance_and_changed_note_hash(tmp_path: Path) -> None:
    run, selection, prompt = _selection_bundle(tmp_path, ("alpha", "beta"))
    rows = [json.loads(line) for line in selection.read_text(encoding="utf-8").splitlines()]
    rows[1]["allowed_note_root"] = str((tmp_path / "other").resolve())
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="replayed decisions"):
        emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")

    run, selection, prompt = _selection_bundle(tmp_path / "hash")
    note = Path(json.loads(selection.read_text(encoding="utf-8"))["note_path"])
    note.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="note binding mismatch"):
        emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")


def test_bundle_publish_failure_cleans_every_result_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    real_replace = artifacts_module.os.replace
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_second)
    with pytest.raises(OSError, match="simulated publish failure"):
        ingest_card_results(jobs_path, [_result(jobs[0], [])])
    assert not (tmp_path / "run" / "results" / "paper_cards.jsonl").exists()
    assert not (tmp_path / "run" / "results" / "rejected_paper_cards.jsonl").exists()
    assert not (tmp_path / "run" / "results" / "card_job_outcomes.jsonl").exists()


def test_owned_path_junction_escape_is_refused_without_touching_external_file(tmp_path: Path) -> None:
    run, selection, prompt = _selection_bundle(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "card_jobs.jsonl"
    sentinel.write_text("sentinel", encoding="utf-8")
    link = run / "cards"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="same active run"):
        emit_card_jobs(selection, prompt, link / "card_jobs.jsonl")
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.parametrize("action", ["emit", "ingest"])
def test_result_junction_escape_never_deletes_external_results(tmp_path: Path, action: str) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    run = tmp_path / "run"
    outside = tmp_path / "outside-results"
    outside.mkdir()
    sentinel = outside / "paper_cards.jsonl"
    sentinel.write_text("sentinel", encoding="utf-8")
    link = run / "results"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="same active run"):
        if action == "emit":
            emit_card_jobs(run / "corpus" / "selection_manifest.jsonl", Path(str(jobs[0]["prompt_path"])), jobs_path)
        else:
            ingest_card_results(jobs_path, [_result(jobs[0], [])])
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


def test_external_and_forged_in_run_job_artifacts_are_rejected(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    external = tmp_path / "external" / "card_jobs.jsonl"
    external.parent.mkdir()
    external.write_bytes(jobs_path.read_bytes())
    with pytest.raises(ValueError, match="active run cards"):
        ingest_card_results(external, [_result(jobs[0], [])])

    forged = read_jsonl(jobs_path)
    forged[0]["note_sha256"] = "f" * 64
    forged[0]["job_id"] = card_job_id("alpha", "f" * 64, str(forged[0]["prompt_sha256"]))
    jobs_path.write_text("".join(json.dumps(row) + "\n" for row in forged), encoding="utf-8")
    with pytest.raises(ValueError, match="job.*selection|binding"):
        ingest_card_results(jobs_path, [_result(forged[0], [])])


@pytest.mark.parametrize("mutation", ["selection", "rejected", "prompt", "note"])
def test_ingestion_revalidates_current_selection_prompt_and_note(tmp_path: Path, mutation: str) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    if mutation == "selection":
        selection = tmp_path / "run" / "corpus" / "selection_manifest.jsonl"
        selection.write_text(selection.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif mutation == "rejected":
        rejected = tmp_path / "run" / "corpus" / "rejected_manifest.jsonl"
        rejected.write_text("\n", encoding="utf-8")
    elif mutation == "prompt":
        Path(str(jobs[0]["prompt_path"])).write_text("changed prompt", encoding="utf-8")
    else:
        Path(str(jobs[0]["note_path"])).write_text("changed note", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|prompt|note"):
        ingest_card_results(jobs_path, [_result(jobs[0], [])])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_ingestion_requires_exact_job_set(tmp_path: Path, mutation: str) -> None:
    jobs_path, jobs = _job_bundle(tmp_path, ("alpha", "beta"))
    records = read_jsonl(jobs_path)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[0])
    else:
        records.append({**records[0], "slug": "rogue", "job_id": "rogue"})
    jobs_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="job"):
        ingest_card_results(jobs_path, [_result(jobs[0], [])])


def test_outside_new_output_path_has_no_creation_side_effect(tmp_path: Path) -> None:
    _, selection, prompt = _selection_bundle(tmp_path)
    outside_parent = tmp_path / "outside" / "new"
    with pytest.raises(ValueError, match="active run cards"):
        emit_card_jobs(selection, prompt, outside_parent / "card_jobs.jsonl")
    assert not outside_parent.exists()


def test_exact_legacy_ledger_is_rejected_but_ledger_named_note_is_allowed(tmp_path: Path) -> None:
    run, selection, prompt = _selection_bundle(tmp_path)
    row = json.loads(selection.read_text(encoding="utf-8"))
    ledger = Path(row["legacy_ledger_path"])
    row["note_path"] = str(ledger)
    row["note_sha256"] = _sha(ledger)
    selection.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="replayed decisions"):
        emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")

    run, selection, prompt = _selection_bundle(tmp_path / "named", ("ledger-analysis",))
    emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")


def test_changed_candidate_list_blocks_emission_and_ingestion(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    selection = tmp_path / "run" / "corpus" / "selection_manifest.jsonl"
    candidate_list = Path(json.loads(selection.read_text(encoding="utf-8"))["source_lists"][0])
    candidate_list.write_text(candidate_list.read_text(encoding="utf-8") + "missing.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate list changed"):
        emit_card_jobs(selection, Path(str(jobs[0]["prompt_path"])), jobs_path)

    jobs_path, jobs = _job_bundle(tmp_path / "ingest")
    selection = tmp_path / "ingest" / "run" / "corpus" / "selection_manifest.jsonl"
    candidate_list = Path(json.loads(selection.read_text(encoding="utf-8"))["source_lists"][0])
    candidate_list.write_text(candidate_list.read_text(encoding="utf-8") + "missing.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate list changed"):
        ingest_card_results(jobs_path, [_result(jobs[0], [])])


@pytest.mark.parametrize("mutation", ["removed_job", "extra_job", "union_mismatch"])
def test_router_job_set_and_selection_union_must_remain_exact(tmp_path: Path, mutation: str) -> None:
    run, selection, prompt = _selection_bundle(tmp_path, ("alpha", "beta"))
    router_jobs = run / "corpus" / "router_jobs.jsonl"
    jobs = read_jsonl(router_jobs)
    if mutation == "removed_job":
        jobs.pop()
        router_jobs.write_text("".join(json.dumps(row) + "\n" for row in jobs), encoding="utf-8")
    elif mutation == "extra_job":
        jobs.append({**jobs[0], "slug": "rogue", "job_id": "rogue", "record_id": "rogue"})
        router_jobs.write_text("".join(json.dumps(row) + "\n" for row in jobs), encoding="utf-8")
    else:
        rows = [json.loads(line) for line in selection.read_text(encoding="utf-8").splitlines()]
        selection.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="router jobs|union|snapshot|prompt or ID"):
        emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")


def test_unlisted_note_with_nonexistent_source_list_fails_closed(tmp_path: Path) -> None:
    run, selection, prompt = _selection_bundle(tmp_path)
    row = json.loads(selection.read_text(encoding="utf-8"))
    router_jobs_path = run / "corpus" / "router_jobs.jsonl"
    job = read_jsonl(router_jobs_path)[0]
    unlisted = Path(row["allowed_note_root"]) / "unlisted.md"
    unlisted.write_text("# unlisted", encoding="utf-8")
    missing = Path(row["allowed_papers_root"]) / "missing-list.txt"
    binding = {"path": str(missing.resolve()), "sha256": "f" * 64}
    for target in (row, job):
        target["note_path"] = str(unlisted.resolve())
        target["note_sha256"] = _sha(unlisted)
        target["source_lists"] = [binding["path"]]
        target["source_list_bindings"] = [binding]
        target["candidate_list_manifest"] = [binding]
    job["note_text"] = unlisted.read_text(encoding="utf-8")
    job_id = router_job_id("alpha", _sha(unlisted), str(job["prompt_sha256"]))
    job["job_id"] = job["record_id"] = row["job_id"] = job_id
    row["result_note_sha256"] = _sha(unlisted)
    router_jobs_path.write_text(json.dumps(job) + "\n", encoding="utf-8")
    row["router_jobs_sha256"] = _sha(router_jobs_path)
    selection.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate list is missing"):
        emit_card_jobs(selection, prompt, run / "cards" / "card_jobs.jsonl")


def test_contentless_card_is_rejected_instead_of_becoming_valid_empty(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    card = _card(jobs[0], "blank")
    card.update({
        "problem": " ", "assumption": {"text": "\t", "status": "UNKNOWN"},
        "mechanism": "", "failure_observation": {"text": " ", "status": "UNKNOWN"},
        "failure_mechanism": {"text": "", "status": "UNKNOWN"},
        "limitation": "", "evaluation": {"measurement": " ", "regime": ""},
        "scope": {"object": "", "time_horizon": " ", "setting": "\t"},
        "evidence_pointers": [],
    })
    accepted, rejected, outcomes = ingest_card_results(jobs_path, [_result(jobs[0], [card])])
    assert accepted == []
    assert rejected[0]["error"] == (
        "incoherent PaperCard: blank required fields: problem, assumption, mechanism, "
        "failure_observation, evaluation.measurement, evaluation.regime, scope.object, "
        "scope.time_horizon, scope.setting; return an empty cards list instead"
    )
    assert outcomes[0]["status"] == "INVALID"


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("problem", lambda card: card.__setitem__("problem", " ")),
        ("assumption", lambda card: card.__setitem__("assumption", {"text": "", "status": "UNKNOWN"})),
        ("mechanism", lambda card: card.__setitem__("mechanism", "\t")),
        ("failure_observation", lambda card: card.__setitem__("failure_observation", {"text": " ", "status": "UNKNOWN"})),
        ("evaluation.measurement", lambda card: card["evaluation"].__setitem__("measurement", "")),
        ("evaluation.regime", lambda card: card["evaluation"].__setitem__("regime", " ")),
        ("scope.object", lambda card: card["scope"].__setitem__("object", "")),
        ("scope.time_horizon", lambda card: card["scope"].__setitem__("time_horizon", "\t")),
        ("scope.setting", lambda card: card["scope"].__setitem__("setting", " ")),
    ],
)
def test_partial_chain_scope_or_evaluation_blanks_are_rejected(tmp_path: Path, field: str, mutate) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    card = _card(jobs[0])
    mutate(card)
    _, rejected, outcomes = ingest_card_results(jobs_path, [_result(jobs[0], [card])])
    assert rejected[0]["error"] == f"incoherent PaperCard: blank required fields: {field}; return an empty cards list instead"
    assert outcomes[0]["status"] == "INVALID"


def test_unknown_failure_mechanism_requires_empty_text_and_empty_requires_unknown(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    valid_unknown = _card(jobs[0], "unknown", failure_mechanism={"text": " ", "status": "UNKNOWN"})
    accepted, rejected, outcomes = ingest_card_results(jobs_path, [_result(jobs[0], [valid_unknown])])
    assert [row["card"]["card_id"] for row in accepted] == ["unknown"]
    assert rejected == [] and outcomes[0]["status"] == "ACCEPTED"

    contradictory = _card(jobs[0], "contradictory", failure_mechanism={"text": "maybe drift", "status": "UNKNOWN"})
    _, rejected, _ = ingest_card_results(jobs_path, [_result(jobs[0], [contradictory])])
    assert rejected[0]["error"] == "failure_mechanism UNKNOWN must have empty text"

    unexplained = _card(jobs[0], "unexplained", failure_mechanism={"text": "", "status": "INFERRED"})
    _, rejected, _ = ingest_card_results(jobs_path, [_result(jobs[0], [unexplained])])
    assert rejected[0]["error"] == "empty failure_mechanism requires UNKNOWN status"


@pytest.mark.parametrize("field", ["title", "venue"])
def test_blank_paper_identity_is_rejected(tmp_path: Path, field: str) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    card = _card(jobs[0])
    card["paper"][field] = " "
    _, rejected, _ = ingest_card_results(jobs_path, [_result(jobs[0], [card])])
    assert rejected[0]["error"] == f"paper.{field} must be nonblank"


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        (lambda card: card["paper"].__setitem__("year", "2026"), "invalid PaperCard"),
        (lambda card: card["evidence_pointers"][0].__setitem__("supports", " problem,assumption,mechanism,failure_observation,failure_mechanism,limitation,evaluation.measurement,evaluation.regime,scope.object,scope.time_horizon,scope.setting "), "normalizes or coerces"),
    ],
)
def test_card_ingestion_rejects_type_and_whitespace_coercion(tmp_path: Path, mutation, expected_fragment: str) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    card = _card(jobs[0])
    mutation(card)
    accepted, rejected, outcomes = ingest_card_results(jobs_path, [_result(jobs[0], [card])])
    assert accepted == []
    assert expected_fragment in rejected[0]["error"]
    assert outcomes[0]["status"] == "INVALID"


def test_result_wrapper_rejects_tuple_cards_instead_of_coercing_to_list(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    raw = _result(jobs[0], [_card(jobs[0])])
    raw["cards"] = tuple(raw["cards"])
    with pytest.raises(TypeError, match="native JSON"):
        ingest_card_results(jobs_path, [raw])
    assert not (jobs_path.parent.parent / "results").exists()


def test_nested_tuple_is_rejected_before_publication(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    raw = _result(jobs[0], [_card(jobs[0])])
    raw["cards"][0]["evidence_pointers"] = tuple(raw["cards"][0]["evidence_pointers"])
    with pytest.raises(TypeError, match="native JSON"):
        ingest_card_results(jobs_path, [raw])
    assert not (jobs_path.parent.parent / "results").exists()


def test_card_canonical_round_trip_ignores_json_object_key_order(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)

    def reverse_objects(value):
        if isinstance(value, dict):
            return {key: reverse_objects(item) for key, item in reversed(tuple(value.items()))}
        if isinstance(value, list):
            return [reverse_objects(item) for item in value]
        return value

    accepted, rejected, outcomes = ingest_card_results(
        jobs_path,
        [_result(jobs[0], [reverse_objects(_card(jobs[0]))])],
    )
    assert len(accepted) == 1
    assert rejected == []
    assert outcomes[0]["status"] == "ACCEPTED"


def test_ingestion_rejects_prevalidated_result_model_even_after_tuple_was_lost(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    raw = _result(jobs[0], [_card(jobs[0])])
    raw["cards"] = tuple(raw["cards"])
    wrapper = CardResultWrapper.model_validate(raw)
    assert type(wrapper.cards) is list
    with pytest.raises(TypeError, match="exact native dict"):
        ingest_card_results(jobs_path, [wrapper])
    assert not (jobs_path.parent.parent / "results").exists()


def test_stateful_mapping_is_rejected_without_iteration_or_bundle_replacement(tmp_path: Path) -> None:
    from collections.abc import Mapping

    jobs_path, jobs = _job_bundle(tmp_path)
    ingest_card_results(jobs_path, [_result(jobs[0], [])])
    results_dir = jobs_path.parent.parent / "results"
    before = {path.name: path.read_bytes() for path in results_dir.iterdir()}
    canonical = _result(jobs[0], [_card(jobs[0])])

    class StatefulResult(Mapping):
        def __init__(self) -> None:
            self.card_reads = 0

        def __iter__(self):
            return iter(canonical)

        def __len__(self) -> int:
            return len(canonical)

        def __getitem__(self, key):
            if key == "cards":
                self.card_reads += 1
                return canonical[key] if self.card_reads == 1 else []
            return canonical[key]

    stateful = StatefulResult()
    with pytest.raises(TypeError, match="exact native dict"):
        ingest_card_results(jobs_path, [stateful])
    assert stateful.card_reads == 0
    assert before == {path.name: path.read_bytes() for path in results_dir.iterdir()}


@pytest.mark.parametrize("container", ["dict", "list"])
def test_nested_container_subclasses_are_rejected_before_bundle_replacement(tmp_path: Path, container: str) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    ingest_card_results(jobs_path, [_result(jobs[0], [])])
    results_dir = jobs_path.parent.parent / "results"
    before = {path.name: path.read_bytes() for path in results_dir.iterdir()}
    raw = _result(jobs[0], [_card(jobs[0])])
    if container == "dict":
        class CustomDict(dict):
            pass
        raw["cards"][0]["paper"] = CustomDict(raw["cards"][0]["paper"])
    else:
        class CustomList(list):
            pass
        raw["cards"] = CustomList(raw["cards"])
    with pytest.raises(TypeError, match="native JSON"):
        ingest_card_results(jobs_path, [raw])
    assert before == {path.name: path.read_bytes() for path in results_dir.iterdir()}


def test_ingestion_snapshot_is_not_aliased_to_mutated_caller_data(tmp_path: Path) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    raw = _result(jobs[0], [_card(jobs[0])])
    ingest_card_results(jobs_path, [raw])
    raw["cards"][0]["assumption"]["text"] = "mutated after ingestion"
    _, accepted, _, _ = validate_card_result_bundle(jobs_path)
    assert accepted[0]["card"]["assumption"]["text"] == "cache stays warm"


@pytest.mark.parametrize("linked_name", ["cards", "results", "corpus"])
def test_ingestion_rejects_upstream_link_before_any_result_invalidation(tmp_path: Path, linked_name: str) -> None:
    jobs_path, jobs = _job_bundle(tmp_path)
    ingest_card_results(jobs_path, [_result(jobs[0], [_card(jobs[0])])])
    run = jobs_path.parent.parent
    outside = tmp_path / "outside" / linked_name
    outside.parent.mkdir()
    (run / linked_name).rename(outside)
    sentinel = outside / "outside-sentinel.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    link = run / linked_name
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            pytest.skip("directory links unavailable")
    result_location = outside if linked_name == "results" else run / "results"
    before = {path.name: path.read_bytes() for path in result_location.iterdir()}
    with pytest.raises(ValueError, match="anchored|expected run"):
        ingest_card_results(run / "cards" / "card_jobs.jsonl", [_result(jobs[0], [])])
    after = {path.name: path.read_bytes() for path in result_location.iterdir()}
    assert after == before
    assert sentinel.read_text(encoding="utf-8") == "untouched"
