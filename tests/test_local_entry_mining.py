import copy
import json
from pathlib import Path

import pytest

from idea_factory.artifacts import read_jsonl


def _run(tmp_path: Path, cards_per_note: int = 2):
    from test_opportunities import _build_run
    from idea_factory.landscape import validate_landscape_bundle
    run, prompt = _build_run(tmp_path, cards_per_note=cards_per_note)
    return run, prompt, validate_landscape_bundle(run)


def _local_doc(bundle, *, rejected=False):
    from test_local_entries import _document, _entry
    row = _entry(bundle)
    if rejected:
        row["review"] = {"status": "REJECTED", "reviewer_id": "pm", "reviewer_role": "AI_PM", "reason": "not admitted"}
    return _document(bundle, [row])


def test_opt_in_emits_one_local_job_and_rejected_audit(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs
    run, prompt, bundle = _run(tmp_path, cards_per_note=1)
    local = tmp_path / "local.json"
    local.write_text(json.dumps(_local_doc(bundle)), encoding="utf-8")
    emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    jobs = read_jsonl(run / "opportunities" / "mining_jobs.jsonl")
    audit = read_jsonl(run / "opportunities" / "local_entry_audit.jsonl")
    local_jobs = [job for job in jobs if job.get("local_entry_id") == "entry-1"]
    assert len(local_jobs) == 1
    assert local_jobs[0]["card_ids"] == [a["card_id"] for a in _local_doc(bundle)["entries"][0]["anchors"]]
    assert "LOCAL_RELATION_HYPOTHESIS" in local_jobs[0]["prompt_text"]
    assert {row["status"] for row in audit} == {"APPROVED"}


def test_rejected_entry_is_audit_only_and_empty_input_is_zero_job(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs
    run, prompt, bundle = _run(tmp_path)
    local = tmp_path / "local.json"
    local.write_text(json.dumps(_local_doc(bundle, rejected=True)), encoding="utf-8")
    emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    assert not [job for job in read_jsonl(run / "opportunities" / "mining_jobs.jsonl") if job.get("local_entry_id")]
    assert read_jsonl(run / "opportunities" / "local_entry_audit.jsonl")[0]["status"] == "REJECTED"


def test_default_two_card_lane_bytes_and_sidecars_are_preserved_or_removed(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs
    run, prompt, _bundle = _run(tmp_path)
    emit_mining_jobs(run, prompt)
    before = {p.name: p.read_bytes() for p in (run / "opportunities").iterdir()}
    local = tmp_path / "empty.json"
    from test_local_entries import _document
    from idea_factory.landscape import validate_landscape_bundle
    local.write_text(json.dumps(_document(validate_landscape_bundle(run), [])), encoding="utf-8")
    emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    emit_mining_jobs(run, prompt)
    assert before == {p.name: p.read_bytes() for p in (run / "opportunities").iterdir()}
    assert not (run / "opportunities" / "reviewed_local_entries.json").exists()


def test_local_snapshot_or_audit_tamper_fails_replay(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs, validate_mining_job_bundle
    run, prompt, bundle = _run(tmp_path, cards_per_note=1)
    local = tmp_path / "local.json"
    local.write_text(json.dumps(_local_doc(bundle)), encoding="utf-8")
    emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    (run / "opportunities" / "local_entry_audit.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_mining_job_bundle(run)


def test_local_result_requires_both_cards_and_inference_flag(tmp_path):
    from idea_factory.opportunities import emit_mining_jobs, ingest_opportunity_results
    run, prompt, bundle = _run(tmp_path, cards_per_note=1)
    local = tmp_path / "local.json"
    local.write_text(json.dumps(_local_doc(bundle)), encoding="utf-8")
    jobs_path = emit_mining_jobs(run, prompt, reviewed_local_entries=local)
    job = read_jsonl(jobs_path)[0]
    result = {"schema_version": "idea_factory.opportunity_result.v1", "job_id": job["job_id"], "cluster_id": job["cluster_id"], "operator": job["operator"], "prompt_sha256": job["prompt_sha256"], "landscape_hashes": job["landscape_hashes"], "card_ids": job["card_ids"], "neighbor_ids": job["neighbor_ids"], "opportunities": [{"schema_version":"idea_factory.opportunity.v1", "opportunity_id":"x", "operator":job["operator"], "assumption_x":"a", "observation_y":"b", "condition_z":"c", "failure_f":"d", "missing_capability_w":"e", "alternative_explanation_a":"f", "decisive_experiment":"g", "supporting_card_ids":job["card_ids"], "nearest_internal_neighbors":[], "scope_compatibility":"s", "inference_flags":[]}]}
    ingest_opportunity_results(jobs_path, [json.dumps(result)])
    assert "MISSING_LOCAL_RELATION_HYPOTHESIS_FLAG" in read_jsonl(run / "opportunities" / "rejected_results.jsonl")[0]["reason_codes"]
    result["opportunities"][0]["inference_flags"] = ["LOCAL_RELATION_HYPOTHESIS"]
    ingest_opportunity_results(jobs_path, [json.dumps(result)])
    assert len(read_jsonl(run / "opportunities" / "opportunity_candidates.jsonl")) == 1
