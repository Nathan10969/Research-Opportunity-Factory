"""Render replayable per-idea JSON/Markdown packs from validated human scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .corpus import CorpusRouterConfig
from .ledger import _SCORE_FIELDS, _canonical_hash, validate_human_bundle
from .opportunities import _anchored_run, _owned_dir, _owned_file
from .stage_io import encode_json, publish_transaction, stage_mutation_lock
from .artifacts import stable_id


def _nonempty(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _readiness_gates(job: Mapping[str, Any], score: Mapping[str, Any]) -> dict[str, bool]:
    route = job["route"]
    test = route.get("cheapest_decisive_test")
    priors = job.get("nearest_priors")
    return {
        "reviewer_pass": job["review"].get("decision") == "PASS_TO_HUMAN",
        "want_to_test_now": "WANT_TO_TEST_NOW" in score["reason_codes"],
        "evidence_or_inference_flags": (
            type(opportunity_flags := job["opportunity"].get("inference_flags")) is list
            and bool(opportunity_flags)
            and all(_nonempty(flag) for flag in opportunity_flags)
        ),
        "nearest_prior_exact_difference": (
            type(priors) is list and bool(priors)
            and all(_nonempty(prior.get("residual_difference")) for prior in priors)
            and _nonempty(route.get("why_nearest_priors_cannot"))
        ),
        "source_of_gain": _nonempty(route.get("source_of_gain")),
        "strongest_baseline": (
            type(job["review"].get("strongest_baseline")) is dict
            and _nonempty(job["review"]["strongest_baseline"].get("evidence_id"))
            and _nonempty(job["review"]["strongest_baseline"].get("comparison"))
        ),
        "decisive_test": (
            type(test) is dict
            and all(_nonempty(test.get(key)) for key in ("setup", "discriminates_against", "expected_runtime_or_cost"))
        ),
        "kill_condition": _nonempty(route.get("kill_condition")),
    }


def _idea_pack(job: Mapping[str, Any], score: Mapping[str, Any]) -> dict[str, Any]:
    route = job["route"]
    opportunity = job["opportunity"]
    review = job["review"]
    gates = _readiness_gates(job, score)
    status = "READY_FOR_CHEAP_TEST" if all(gates.values()) else "HOLD"
    idea_id = stable_id(
        "idea_pack", job["route_id"], score["human_score_id"], job["review_record_sha256"],
        job["authenticity_sha256"],
    )
    body: dict[str, Any] = {
        "schema_version": "idea_factory.idea_pack.v1",
        "idea_id": idea_id,
        "status": status,
        "opportunity": opportunity,
        "core_hypothesis": route["new_assumption"],
        "why_now": f"{opportunity['observation_y']} Under {opportunity['condition_z']}.",
        "evidence_flags": opportunity.get("inference_flags", []),
        "supporting_observations": [opportunity["observation_y"], opportunity["failure_f"]],
        "nearest_priors": job["nearest_priors"],
        "nearest_prior_exact_difference": route["why_nearest_priors_cannot"],
        "proposed_mechanism": route["new_mechanism"],
        "source_of_gain": route["source_of_gain"],
        "cheapest_decisive_test": route["cheapest_decisive_test"],
        "strongest_baseline": review["strongest_baseline"],
        "kill_condition": route["kill_condition"],
        "main_uncertainty": opportunity["alternative_explanation_a"],
        "expected_reviewer_2_objection": review["a_plus_b_objection"],
        "response_to_objection": route["why_it_addresses_failure"],
        "evidence_that_would_make_reviewer_correct": {
            "discriminates_against": route["cheapest_decisive_test"]["discriminates_against"],
            "kill_condition": route["kill_condition"],
        },
        "human_scores": {field: score[field] for field in _SCORE_FIELDS},
        "human_reason_codes": score["reason_codes"],
        "human_identity": score["human_identity"],
        "human_attested_at": score["attested_at"],
        "authenticity_boundary": job["authenticity_binding"],
        "readiness_gates": gates,
        "provenance": {
            key: job[key]
            for key in (
                "human_job_sha256", "opportunity_sha256", "route_sha256", "recon_provenance_sha256",
                "nearest_priors_sha256", "review_record_sha256", "review_bundle_manifest_sha256",
                "authenticity_sha256",
            )
        } | {"human_score_id": score["human_score_id"], "human_score_raw_sha256": score["raw_result_sha256"]},
    }
    return body | {"idea_pack_sha256": _canonical_hash(body)}


def _markdown(pack: Mapping[str, Any]) -> bytes:
    canonical = json.dumps(pack, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    text = (
        f"# IdeaPack {pack['idea_id']}\n\n"
        f"Status: {pack['status']}\n\n"
        f"IdeaPack SHA-256: {pack['idea_pack_sha256']}\n\n"
        "## Canonical JSON provenance\n\n"
        "```json\n"
        f"{canonical}\n"
        "```\n"
    )
    return text.encode("utf-8")


def _classification(human: Mapping[str, Any], packs: Sequence[Mapping[str, Any]]) -> str:
    if packs:
        return "HAS_SURVIVORS"
    eligibility = human["eligibility"]
    if not eligibility:
        return str(human["report"]["classification"])
    statuses = {str(row["status"]) for row in eligibility}
    if "UNRESOLVED" in statuses or any(row["status"] == "REJECTED" for row in human["outcomes"]):
        return "UNRESOLVED"
    if "PENDING_REVIEW" in statuses or "PENDING_HUMAN" in statuses:
        return "PENDING"
    return "ZERO_SURVIVOR"


def _projection(human: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any]]:
    scores = {str(row["job_id"]): row for row in human["results"]}
    packs = [_idea_pack(job, scores[str(job["job_id"])]) for job in human["jobs"] if str(job["job_id"]) in scores]
    packs.sort(key=lambda row: str(row["idea_id"]))
    files: dict[str, bytes] = {}
    file_index: dict[str, dict[str, str]] = {}
    for pack in packs:
        idea_id = str(pack["idea_id"])
        json_bytes = encode_json(pack)
        markdown_bytes = _markdown(pack)
        files[f"{idea_id}.json"] = json_bytes
        files[f"{idea_id}.md"] = markdown_bytes
        file_index[idea_id] = {
            "json_sha256": hashlib.sha256(json_bytes).hexdigest(),
            "markdown_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
            "idea_pack_sha256": str(pack["idea_pack_sha256"]),
        }
    classification = _classification(human, packs)
    report = {
        "schema_version": "idea_factory.zero_survivor_report.v1",
        "classification": classification,
        "source": "VALIDATED_REVIEW_AND_HUMAN_BUNDLES",
        "source_classification": human["report"]["classification"],
        "opportunity_input_count": human["report"]["opportunity_input_count"],
        "source_counts": {
            status: sum(row["status"] == status for row in human["eligibility"])
            for status in ("PENDING_HUMAN", "KILLED_REVIEW", "PENDING_REVIEW", "UNRESOLVED")
        },
        "pack_count": len(packs),
        "operational_completion_is_idea_yield": False,
    }
    report_bytes = encode_json(report)
    files["zero_survivor_report.json"] = report_bytes
    body = {
        "schema_version": "idea_factory.idea_pack_manifest.v1",
        "classification": classification,
        "idea_ids": [pack["idea_id"] for pack in packs],
        "pack_count": len(packs),
        "ready_count": sum(pack["status"] == "READY_FOR_CHEAP_TEST" for pack in packs),
        "hold_count": sum(pack["status"] == "HOLD" for pack in packs),
        "human_bundle_manifest_sha256": human["manifest"]["bundle_manifest_sha256"],
        "zero_survivor_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "files": file_index,
    }
    manifest = body | {"manifest_sha256": _canonical_hash(body)}
    files["manifest.json"] = encode_json(manifest)
    return files, manifest, report


def render_idea_packs(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> Path:
    run = _anchored_run(Path(run_dir))
    directory = _owned_dir(run, "idea_packs", create=True)
    with stage_mutation_lock(directory):
        human = validate_human_bundle(
            run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
            allow_test_ready=allow_test_ready,
        )
        files, _manifest, _report = _projection(human)
        expected_names = set(files)
        transaction: dict[Path, bytes | None] = {directory / name: payload for name, payload in files.items()}
        for path in directory.iterdir():
            if path.name == ".stage.lock":
                continue
            if path.name not in expected_names:
                if not path.is_file() or path.is_symlink():
                    raise ValueError("idea pack directory contains an unsafe artifact")
                transaction[path] = None
        publish_transaction(transaction)
        return directory


def validate_idea_pack_bundle(
    run_dir: Path,
    config: CorpusRouterConfig,
    *,
    prompt_path: Path | None = None,
    route_prompt_path: Path | None = None,
    allow_test_ready: bool = False,
) -> dict[str, Any]:
    run = _anchored_run(Path(run_dir))
    directory = _owned_dir(run, "idea_packs", create=False)
    human = validate_human_bundle(
        run, config, prompt_path=prompt_path, route_prompt_path=route_prompt_path,
        allow_test_ready=allow_test_ready,
    )
    files, manifest, _report = _projection(human)
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != set(files):
        raise ValueError("idea pack bundle file set mismatch")
    for name, payload in files.items():
        if _owned_file(directory, name).read_bytes() != payload:
            raise ValueError("idea pack JSON/Markdown provenance mismatch")
    return manifest
