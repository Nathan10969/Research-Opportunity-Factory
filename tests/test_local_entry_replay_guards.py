import copy
import json
import os
from pathlib import Path

import pytest

from idea_factory.artifacts import read_jsonl


def _run(tmp_path: Path, cards_per_note: int = 1):
    from test_opportunities import _build_run
    from idea_factory.landscape import validate_landscape_bundle
    run, prompt = _build_run(tmp_path, cards_per_note=cards_per_note)
    return run, prompt, validate_landscape_bundle(run)


def _doc(bundle, *, status="APPROVED", kind="PAIR_RELATION"):
    from test_local_entries import _document, _entry
    row = _entry(bundle, kind)
    row["review"]["status"] = status
    return _document(bundle, [row])


def _emit(run, prompt, bundle, path, **kwargs):
    from idea_factory.opportunities import emit_mining_jobs
    path.write_text(json.dumps(_doc(bundle, **kwargs)), encoding="utf-8")
    emit_mining_jobs(run, prompt, reviewed_local_entries=path)


@pytest.mark.parametrize("status", ["APPROVED", "REJECTED"])
def test_missing_both_local_sidecars_never_downgrades_opt_in_run(tmp_path, status):
    from idea_factory.opportunities import emit_mining_jobs, validate_mining_job_bundle
    run, prompt, bundle = _run(tmp_path)
    local = tmp_path / "local.json"
    _emit(run, prompt, bundle, local, status=status)
    (run / "opportunities" / "reviewed_local_entries.json").unlink()
    (run / "opportunities" / "local_entry_audit.jsonl").unlink()
    with pytest.raises(ValueError, match="snapshot|marker|local entry"):
        validate_mining_job_bundle(run)


def test_empty_opt_in_run_requires_sidecars_and_marker(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs, validate_mining_job_bundle
    run, prompt, bundle = _run(tmp_path)
    from test_local_entries import _document
    local = tmp_path / "empty.json"
    local.write_text(json.dumps(_document(bundle, [])), encoding="utf-8")
    emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    audit = run / "opportunities" / "cluster_audit.jsonl"
    rows = read_jsonl(audit)
    assert any(row.get("schema_version") == "idea_factory.local_entry_mode.v1" for row in rows)
    audit.write_text("\n".join(json.dumps(row) for row in rows if row.get("schema_version") != "idea_factory.local_entry_mode.v1") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marker|snapshot|local entry"):
        validate_mining_job_bundle(run)


def test_zero_card_empty_opt_in_replays_from_marker_prompt(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs, validate_mining_job_bundle
    run, prompt, bundle = _run(tmp_path, cards_per_note=0)
    from test_local_entries import _document
    local = tmp_path / "empty.json"
    local.write_text(json.dumps(_document(bundle, [])), encoding="utf-8")
    emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    assert read_jsonl(run / "opportunities" / "mining_jobs.jsonl") == []
    validate_mining_job_bundle(run)


@pytest.mark.parametrize("which", ["snapshot", "audit"])
def test_single_local_sidecar_deletion_fails_replay(tmp_path, which):
    from idea_factory.opportunities import emit_mining_jobs, validate_mining_job_bundle
    run, prompt, bundle = _run(tmp_path)
    local = tmp_path / "local.json"
    _emit(run, prompt, bundle, local)
    (run / "opportunities" / ("reviewed_local_entries.json" if which == "snapshot" else "local_entry_audit.jsonl")).unlink()
    with pytest.raises(ValueError, match="snapshot|marker|local entry"):
        validate_mining_job_bundle(run)


def test_tampered_marker_fails_replay(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs, validate_mining_job_bundle
    run, prompt, bundle = _run(tmp_path)
    local = tmp_path / "local.json"
    _emit(run, prompt, bundle, local)
    audit = run / "opportunities" / "cluster_audit.jsonl"
    rows = read_jsonl(audit)
    rows[-1]["artifact_sha256"] = "0" * 64
    audit.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marker|replay"):
        validate_mining_job_bundle(run)


@pytest.mark.parametrize("status", ["APPROVED", "REJECTED"])
def test_marker_removed_from_audit_rejects_nonempty_opt_in(tmp_path, status):
    from idea_factory.opportunities import validate_mining_job_bundle
    run, prompt, bundle = _run(tmp_path)
    local = tmp_path / "local.json"
    _emit(run, prompt, bundle, local, status=status)
    audit = run / "opportunities" / "cluster_audit.jsonl"
    rows = [row for row in read_jsonl(audit) if row.get("schema_version") != "idea_factory.local_entry_mode.v1"]
    audit.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marker|local entry"):
        validate_mining_job_bundle(run)


def test_rehashed_invalid_input_preserves_existing_owned_bytes(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs
    run, prompt, bundle = _run(tmp_path)
    emit_mining_jobs(run, prompt)
    before = {p.name: p.read_bytes() for p in (run / "opportunities").iterdir()}
    from test_local_entries import _document, _entry
    local = tmp_path / "bad.json"
    doc = _document(bundle, [_entry(bundle)])
    doc["entries"][0]["anchors"][0]["raw_text"] = "tampered"
    from test_local_entries import _rehash
    local.write_text(json.dumps(_rehash(doc)), encoding="utf-8")
    with pytest.raises(ValueError, match="anchor|binding"):
        emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    assert before == {p.name: p.read_bytes() for p in (run / "opportunities").iterdir()}


@pytest.mark.parametrize("sidecar", ["reviewed_local_entries.json", "local_entry_audit.jsonl"])
def test_sidecar_symlink_refused_and_outside_sentinel_preserved(tmp_path, sidecar):
    from idea_factory.opportunities import emit_mining_jobs
    run, prompt, bundle = _run(tmp_path)
    local = tmp_path / "local.json"
    _emit(run, prompt, bundle, local)
    target = tmp_path / "sentinel.txt"
    target.write_text("do not touch", encoding="utf-8")
    path = run / "opportunities" / sidecar
    path.unlink()
    try:
        os.symlink(target, path)
    except OSError:
        pytest.skip("symlink creation unavailable")
    before = (run / "opportunities" / "mining_jobs.jsonl").read_bytes()
    with pytest.raises(ValueError):
        emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    assert target.read_text(encoding="utf-8") == "do not touch"
    assert (run / "opportunities" / "mining_jobs.jsonl").read_bytes() == before


def test_hardlink_to_quality_output_refused_before_delete(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs
    run, prompt, bundle = _run(tmp_path)
    emit_mining_jobs(run, prompt)
    quality = run / "quality"
    quality.mkdir()
    target = quality / "rejected.jsonl"
    target.write_text("sentinel", encoding="utf-8")
    local = tmp_path / "local.json"
    try:
        local.unlink(missing_ok=True)
        os.link(target, local)
    except OSError:
        pytest.skip("hardlink creation unavailable")
    before = (run / "opportunities" / "mining_jobs.jsonl").read_bytes()
    with pytest.raises(ValueError, match="external|regular|alias"):
        emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    assert target.read_text(encoding="utf-8") == "sentinel"
    assert (run / "opportunities" / "mining_jobs.jsonl").read_bytes() == before


def test_lexical_symlink_is_rejected_before_resolve_and_preserves_jobs(tmp_path, monkeypatch):
    import idea_factory.opportunities as opportunities
    run, prompt, bundle = _run(tmp_path)
    opportunities.emit_mining_jobs(run, prompt)
    before = (run / "opportunities" / "mining_jobs.jsonl").read_bytes()
    local = tmp_path / "lexical-link.json"
    local.write_text(json.dumps(_doc(bundle)), encoding="utf-8")
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    def fake_is_symlink(path):
        return path == local or original_is_symlink(path)
    def forbidden_resolve(path, *args, **kwargs):
        if path == local:
            raise AssertionError("lexical local input was resolved before symlink rejection")
        return original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    with pytest.raises(ValueError, match="external regular file"):
        opportunities.emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    assert (run / "opportunities" / "mining_jobs.jsonl").read_bytes() == before


def test_local_audit_write_failure_cleans_partial_owned_outputs(tmp_path, monkeypatch):
    import idea_factory.opportunities as opportunities
    run, prompt, bundle = _run(tmp_path)
    local = tmp_path / "approved.json"
    local_bytes = json.dumps(_doc(bundle)).encode("utf-8")
    local.write_bytes(local_bytes)
    original_bundle_writer = opportunities.write_jsonl_bundle
    def fail_audit(payload):
        if any(path.name == "local_entry_audit.jsonl" for path in payload):
            raise OSError("injected local audit failure")
        return original_bundle_writer(payload)
    monkeypatch.setattr(opportunities, "write_jsonl_bundle", fail_audit)
    with pytest.raises(OSError, match="injected local audit failure"):
        opportunities.emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    owned = run / "opportunities"
    assert not any((owned / name).exists() for name in ("reviewed_local_entries.json", "local_entry_audit.jsonl", "mining_jobs.jsonl", "cluster_audit.jsonl"))
    assert local.read_bytes() == local_bytes


def test_normal_jobs_and_cross_facet_local_job_coexist_with_prior_neighbors(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs
    run, prompt, bundle = _run(tmp_path, cards_per_note=2)
    local = tmp_path / "cross.json"
    _emit(run, prompt, bundle, local, kind="CROSS_FACET")
    jobs = read_jsonl(run / "opportunities" / "mining_jobs.jsonl")
    local_job = next(row for row in jobs if row.get("local_entry_id"))
    assert len([row for row in jobs if not row.get("local_entry_id")]) >= 3
    assert len(local_job["card_ids"]) == 2
    assert all(anchor["raw_text"] in {edge["raw_text"] for edge in bundle.edges} for anchor in local_job["local_entry"]["anchors"])
    assert set(local_job["neighbor_ids"]) == {row["neighbor_id"] for row in local_job["neighbor_summaries"]}
    assert local_job["local_entry_sha256"]


@pytest.mark.parametrize("variant,expected", [("reverse", "ACCEPTED"), ("original", "ACCEPTED"), ("missing", "INVALID"), ("duplicate", "INVALID"), ("flag", "INVALID"), ("empty", "VALID_EMPTY")])
def test_local_result_contract_variants(tmp_path, variant, expected):
    from idea_factory.opportunities import emit_mining_jobs, ingest_opportunity_results, validate_opportunity_result_bundle
    run, prompt, bundle = _run(tmp_path)
    local = tmp_path / "local.json"
    _emit(run, prompt, bundle, local)
    jobs_path = run / "opportunities" / "mining_jobs.jsonl"
    job = next(row for row in read_jsonl(jobs_path) if row.get("local_entry_id"))
    cards = list(job["card_ids"])
    if variant == "reverse": cards.reverse()
    opportunities = []
    if variant != "empty":
        flags = [] if variant == "flag" else ["LOCAL_RELATION_HYPOTHESIS"]
        supporting = cards if variant != "missing" else [cards[0]]
        if variant == "duplicate": supporting = [cards[0], cards[0]]
        opportunities = [{"schema_version":"idea_factory.opportunity.v1","opportunity_id":"local-x","operator":job["operator"],"assumption_x":"a","observation_y":"b","condition_z":"c","failure_f":"d","missing_capability_w":"e","alternative_explanation_a":"f","decisive_experiment":"g","supporting_card_ids":supporting,"nearest_internal_neighbors":[],"scope_compatibility":"s","inference_flags":flags}]
    result = {"schema_version":"idea_factory.opportunity_result.v1","job_id":job["job_id"],"cluster_id":job["cluster_id"],"operator":job["operator"],"prompt_sha256":job["prompt_sha256"],"landscape_hashes":job["landscape_hashes"],"card_ids":job["card_ids"],"neighbor_ids":job["neighbor_ids"],"opportunities":opportunities}
    ingest_opportunity_results(jobs_path, [json.dumps(result)])
    outcome = next(row for row in read_jsonl(run / "opportunities" / "result_outcomes.jsonl") if row["job_id"] == job["job_id"])
    assert outcome["status"] == expected
    validate_opportunity_result_bundle(run)
