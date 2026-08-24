from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from idea_factory.artifacts import read_jsonl
from idea_factory.models import OpportunityOperator


def _build_run(tmp_path: Path, *, cards_per_note: int = 2) -> tuple[Path, Path]:
    from idea_factory.cards import emit_card_jobs, ingest_card_results
    from idea_factory.landscape import build_landscape
    from test_cards import _card, _result, _selection_bundle

    run, selection, card_prompt = _selection_bundle(tmp_path, ("one", "two"))
    card_jobs = read_jsonl(emit_card_jobs(selection, card_prompt, run / "cards" / "card_jobs.jsonl"))
    wrappers = []
    for job in card_jobs:
        cards = [_card(job, f"{job['slug']}-{index}") for index in range(cards_per_note)]
        wrappers.append(_result(job, cards))
    ingest_card_results(run / "cards" / "card_jobs.jsonl", wrappers)
    build_landscape(run)
    prompt = tmp_path / "opportunity.md"
    prompt.write_text("strict opportunity json", encoding="utf-8")
    return run, prompt


def _empty_result(job: dict[str, object]) -> str:
    return json.dumps({
        "schema_version": "idea_factory.opportunity_result.v1",
        "job_id": job["job_id"], "cluster_id": job["cluster_id"],
        "operator": job["operator"], "prompt_sha256": job["prompt_sha256"],
        "landscape_hashes": job["landscape_hashes"], "card_ids": job["card_ids"],
        "neighbor_ids": job["neighbor_ids"], "opportunities": [],
    })


def test_basis_clusters_do_not_transitively_union_scientific_groups() -> None:
    from idea_factory.opportunities import cluster_cards

    scope = {"object": "cache", "time_horizon": "long", "setting": "serving"}
    cards = [
        {"card_id": "a", "scope": scope, "assumption": {"text": "shared assumption"}, "failure_observation": {"text": "only a"}, "failure_mechanism": {"text": "m-a", "status": "OBSERVED"}},
        {"card_id": "b", "scope": scope, "assumption": {"text": "shared assumption"}, "failure_observation": {"text": "shared failure"}, "failure_mechanism": {"text": "m-b", "status": "OBSERVED"}},
        {"card_id": "c", "scope": scope, "assumption": {"text": "shared assumption"}, "failure_observation": {"text": "shared failure"}, "failure_mechanism": {"text": "m-c", "status": "OBSERVED"}},
        {"card_id": "d", "scope": scope, "assumption": {"text": "only d"}, "failure_observation": {"text": "shared failure"}, "failure_mechanism": {"text": "m-d", "status": "OBSERVED"}},
    ]

    clusters, audit = cluster_cards(list(reversed(cards)))

    eligible_sets = {tuple(cluster["card_ids"]) for cluster in clusters}
    assert eligible_sets == {("a", "b", "c"), ("b", "c", "d")}
    assert all(cluster["cluster_basis"]["source_card_ids"] == cluster["card_ids"] for cluster in clusters)
    assert ("a", "b", "c", "d") not in eligible_sets
    assert any(row["status"] == "SKIPPED_TOO_SMALL" for row in audit)


def test_balanced_basis_split_has_no_small_tail_and_exact_scope_separation() -> None:
    from idea_factory.opportunities import cluster_cards

    cards = []
    for index in range(11):
        cards.append({"card_id": f"a-{index:02}", "scope": {"object": "cache", "time_horizon": "long", "setting": "serving"}, "assumption": {"text": "same"}, "failure_observation": {"text": f"f-{index}"}, "failure_mechanism": {"text": "", "status": "UNKNOWN"}})
    for index in range(3):
        cards.append({"card_id": f"b-{index}", "scope": {"object": "cache", "time_horizon": "short", "setting": "serving"}, "assumption": {"text": "same"}, "failure_observation": {"text": f"g-{index}"}, "failure_mechanism": {"text": "", "status": "UNKNOWN"}})

    clusters, _audit = cluster_cards(cards)
    assumption = [row for row in clusters if row["cluster_basis"]["dimension"] == "assumptions"]
    assert sorted(len(row["card_ids"]) for row in assumption) == [3, 5, 6]
    assert len({row["cluster_basis"]["scope_key"] for row in assumption}) == 2
    assert all(3 <= len(row["card_ids"]) <= 8 for row in clusters)


def test_jobs_have_one_basis_neighbors_cost_audit_and_are_byte_deterministic(tmp_path: Path) -> None:
    from idea_factory.opportunities import emit_mining_jobs

    run, prompt = _build_run(tmp_path)
    path = emit_mining_jobs(run, prompt)
    first = {child.name: child.read_bytes() for child in (run / "opportunities").iterdir()}
    jobs = read_jsonl(path)
    audit = read_jsonl(run / "opportunities" / "cluster_audit.jsonl")

    assert {job["operator"] for job in jobs} == {operator.value for operator in OpportunityOperator}
    assert all(job["cluster_basis"] and len(job["card_ids"]) in range(3, 9) for job in jobs)
    assert all(job["neighbor_ids"] and {row["neighbor_id"] for row in job["neighbor_summaries"]} == set(job["neighbor_ids"]) for job in jobs)
    assert all(job["estimated_external_calls"] == 1 and job["calls_per_cluster"] == 8 for job in jobs)
    assert {row["status"] for row in audit} == {"ELIGIBLE"}
    assert sum(row["jobs_emitted"] for row in audit) == len(jobs)
    assert all(row["operator_schedule_version"] == "idea_factory.operator_schedule.v1" for row in audit)
    assert all(row["prompt_sha256"] == jobs[0]["prompt_sha256"] and row["landscape_hashes"] == jobs[0]["landscape_hashes"] for row in audit)

    emit_mining_jobs(run, prompt)
    assert first == {child.name: child.read_bytes() for child in (run / "opportunities").iterdir()}


def test_landscape_validator_rejects_duplicate_edge_and_aggregate_tamper(tmp_path: Path) -> None:
    from idea_factory.opportunities import validate_landscape_bundle

    run, _prompt = _build_run(tmp_path)
    edge_path = run / "landscape" / "card_concept_edges.csv"
    original = edge_path.read_text(encoding="utf-8")
    edge_path.write_text(original + original.splitlines(keepends=True)[1], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate|ordered|projection"):
        validate_landscape_bundle(run)

    from idea_factory.landscape import build_landscape
    build_landscape(run)
    map_path = run / "landscape" / "assumptions.json"
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    payload["entries"][0]["raw_phrases"] = ["tampered"]
    map_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate|projection"):
        validate_landscape_bundle(run)


def test_result_and_quality_bundles_are_separate_replayable_and_idempotent(tmp_path: Path) -> None:
    from idea_factory.opportunities import (
        emit_mining_jobs, ingest_opportunity_results,
        validate_opportunity_result_bundle,
    )
    from idea_factory.quality import publish_quality, validate_quality_bundle

    run, prompt = _build_run(tmp_path)
    jobs = read_jsonl(emit_mining_jobs(run, prompt))
    results = [_empty_result(job) for job in jobs]
    outputs = ingest_opportunity_results(run / "opportunities" / "mining_jobs.jsonl", results)

    assert set(outputs) == {"candidates", "rejected", "outcomes"}
    assert all(path.parent == run / "opportunities" for path in outputs.values())
    validated = validate_opportunity_result_bundle(run)
    assert len(validated.outcomes) == len(jobs)
    assert {row["status"] for row in validated.outcomes} == {"VALID_EMPTY"}

    quality = publish_quality(run)
    first = {name: path.read_bytes() for name, path in quality.items()}
    quality_bundle = validate_quality_bundle(run)
    assert all("decision_reason_codes" in row for row in quality_bundle.outcomes)
    publish_quality(run)
    assert first == {name: path.read_bytes() for name, path in quality.items()}


def test_result_bundle_rejects_unknown_neighbor_and_detects_projection_tamper(tmp_path: Path) -> None:
    from idea_factory.opportunities import emit_mining_jobs, ingest_opportunity_results, validate_opportunity_result_bundle

    run, prompt = _build_run(tmp_path)
    jobs = read_jsonl(emit_mining_jobs(run, prompt))
    results = [_empty_result(job) for job in jobs]
    target = jobs[0]
    wrapper = json.loads(results[0])
    wrapper["opportunities"] = [{
        "schema_version": "idea_factory.opportunity.v1", "opportunity_id": "bad-neighbor",
        "operator": target["operator"], "assumption_x": "The cache remains resident during serving.",
        "observation_y": "Cards report retrieval degrades after eviction.", "condition_z": "under long serving traces",
        "failure_f": "cache eviction causes retrieval loss", "missing_capability_w": "eviction-aware control",
        "alternative_explanation_a": "routing error", "decisive_experiment": "Compare recall against routing error.",
        "supporting_card_ids": [target["card_ids"][0]], "nearest_internal_neighbors": ["unknown-neighbor"],
        "scope_compatibility": "same serving scope", "inference_flags": [],
    }]
    results[0] = json.dumps(wrapper)
    ingest_opportunity_results(run / "opportunities" / "mining_jobs.jsonl", results)
    rejected = read_jsonl(run / "opportunities" / "rejected_results.jsonl")
    assert rejected[0]["reason_codes"] == ["UNKNOWN_NEIGHBOR_ID"]

    outcome_path = run / "opportunities" / "result_outcomes.jsonl"
    rows = read_jsonl(outcome_path); rows[0]["accepted_projection_sha256"] = "0" * 64
    outcome_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="projection|replay"):
        validate_opportunity_result_bundle(run)


def test_opportunities_junction_escape_preserves_outside_sentinel(tmp_path: Path) -> None:
    from idea_factory.opportunities import emit_mining_jobs

    run, prompt = _build_run(tmp_path)
    outside = tmp_path / "outside"; outside.mkdir()
    sentinel = outside / "mining_jobs.jsonl"; sentinel.write_text("sentinel", encoding="utf-8")
    link = run / "opportunities"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="anchored|escapes"):
        emit_mining_jobs(run, prompt)
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


def test_unsafe_quality_link_is_rejected_before_downstream_delete(tmp_path: Path) -> None:
    from idea_factory.opportunities import emit_mining_jobs

    run, prompt = _build_run(tmp_path)
    emit_mining_jobs(run, prompt)
    candidate = run / "opportunities" / "opportunity_candidates.jsonl"
    candidate.write_text("preserve", encoding="utf-8")
    outside = tmp_path / "quality-outside"; outside.mkdir()
    sentinel = outside / "rejected.jsonl"; sentinel.write_text("sentinel", encoding="utf-8")
    link = run / "quality"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            pytest.skip("directory links unavailable")
    with pytest.raises(ValueError, match="anchored|escapes"):
        emit_mining_jobs(run, prompt)
    assert candidate.read_text(encoding="utf-8") == "preserve"
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


def test_prompt_declares_exact_wrapper_and_forbids_methods_acronyms_and_markdown() -> None:
    prompt = (Path(__file__).parents[1] / "prompts" / "opportunity_miner.md").read_text(encoding="utf-8")
    for token in ("schema_version", "job_id", "cluster_id", "operator", "landscape_hashes", "prompt_sha256", "neighbor_ids", "opportunities"):
        assert token in prompt
    assert "all method names" in prompt.lower()
    assert "model acronyms" in prompt.lower()
    assert "markdown" in prompt.lower()
    assert '"additionalProperties": false' in prompt
    assert '"required"' in prompt
