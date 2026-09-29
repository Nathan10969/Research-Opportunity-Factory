from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from idea_factory.private_router_aggregate import aggregate
from idea_factory.corpus import RouterResult, _canonical_result_hash


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(path: Path, rows: list[dict]) -> bytes:
    data = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    path.write_bytes(data)
    return data


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    run = tmp_path / "run"
    run.mkdir()
    old = tmp_path / "old"
    old.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_bytes(b"pinned router prompt\n")
    jobs, allow = [], []
    for index, slug in enumerate(("paper-a", "paper-b"), 1):
        note = tmp_path / f"{slug}.md"
        note.write_bytes(f"# {slug}\n".encode())
        job = {"schema_version": "idea_factory.corpus_router_job.v3",
               "job_id": f"router_job-{index}", "slug": slug,
               "note_path": str(note), "note_sha256": _sha(note.read_bytes()),
               "prompt_path": str(prompt), "prompt_sha256": _sha(prompt.read_bytes())}
        jobs.append(job)
        result = {"schema_version": "idea_factory.corpus_router_result.v3",
                  "job_id": job["job_id"], "slug": slug,
                  "note_sha256": job["note_sha256"], "prompt_sha256": job["prompt_sha256"],
                  "label": "GENERAL_RESEARCH", "core_mechanism": "A concrete mechanism",
                  "scope_reason": "A bounded relevant scope", "evidence_locator": "note:1",
                  "confidence": "HIGH"}
        old_job = old / f"{slug}.job.jsonl"
        old_result = old / f"{slug}.result.jsonl"
        old_job_bytes = _jsonl(old_job, [job])
        result_bytes = _jsonl(old_result, [result])
        qa = old / f"{slug}.qa.json"
        qa.write_text(json.dumps({"verdict": "PASS_PRIVATE_ONLY",
                                   "batch": {"results_sha256": _sha(result_bytes)}, "per_item":
                                   [{"slug": slug, "status": "PASS"}]}) + "\n")
        allow.append({"schema_version": "engineering.old811_card_pilot100_strict_v3_input_allowlist.v1",
                      "ordinal": index, "slug": slug,
                      "status": "INPUT_PRECHECK_PASS_SCIENCE_PENDING", "hold_reason": None,
                      "router_job_id": job["job_id"], "note_path": str(note),
                      "note_sha256": job["note_sha256"], "router_prompt_path": str(prompt),
                      "router_prompt_sha256": job["prompt_sha256"],
                      "old_router_job_path": str(old_job),
                      "old_router_job_file_sha256": _sha(old_job_bytes),
                      "old_router_job_line": 1,
                      "old_router_job_row_sha256": _sha(old_job_bytes[:-1]),
                      "private_router_result_path": str(old_result),
                      "private_router_result_file_sha256": _sha(result_bytes),
                      "private_router_result_line": 1,
                      "private_router_result_row_sha256": _sha(result_bytes[:-1]),
                      "private_qa_path": str(qa), "private_qa_sha256": _sha(qa.read_bytes()),
                      "private_qa_scope": "PASS"})
    jobs_path = run / "router_jobs.jsonl"
    _jsonl(jobs_path, jobs)
    allowlist = tmp_path / "allowlist.jsonl"
    _jsonl(allowlist, allow)
    return allowlist, jobs_path, tmp_path / "private-candidate", allow


def _run(allowlist: Path, jobs: Path, output: Path):
    return aggregate(allowlist, _sha(allowlist.read_bytes()),
                     jobs, _sha(jobs.read_bytes()), output)


def _repin_qa_result(row: dict) -> None:
    qa_path = Path(row["private_qa_path"])
    qa = json.loads(qa_path.read_text())
    qa["batch"]["results_sha256"] = row["private_router_result_file_sha256"]
    qa_path.write_text(json.dumps(qa) + "\n")
    row["private_qa_sha256"] = _sha(qa_path.read_bytes())


def test_aggregates_exact_pinned_results_without_editing_inputs(tmp_path: Path) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    before = {p: p.read_bytes() for row in rows for p in
              (Path(row["old_router_job_path"]), Path(row["private_router_result_path"]),
               Path(row["private_qa_path"]), Path(row["note_path"]))}
    receipt = _run(allowlist, jobs, output)
    candidate = (output / "router_results.candidate.v3.jsonl").read_bytes()
    assert receipt["candidate_count"] == 2
    assert receipt["candidate_sha256"] == _sha(candidate)
    assert [json.loads(line)["slug"] for line in candidate.splitlines()] == ["paper-a", "paper-b"]
    assert candidate == b"".join(Path(row["private_router_result_path"]).read_bytes() for row in rows)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert receipt["ingested"] is False and receipt["scientific_approval"] is False


@pytest.mark.parametrize("field,value", [("router_job_id", "wrong"),
                                         ("note_sha256", "0" * 64),
                                         ("router_prompt_sha256", "0" * 64)])
def test_job_or_binding_drift_fails_before_output(tmp_path: Path, field: str, value: str) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    rows[0][field] = value
    _jsonl(allowlist, rows)
    with pytest.raises(ValueError):
        _run(allowlist, jobs, output)
    assert not output.exists()


def test_qa_or_physical_row_drift_fails_before_output(tmp_path: Path) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    result = Path(rows[0]["private_router_result_path"])
    result.write_bytes(result.read_bytes().replace(b"A concrete", b"A changed "))
    with pytest.raises(ValueError, match="SHA-256"):
        _run(allowlist, jobs, output)
    assert not output.exists()

    rows[0]["private_router_result_file_sha256"] = _sha(result.read_bytes())
    _jsonl(allowlist, rows)
    with pytest.raises(ValueError, match="row SHA-256"):
        _run(allowlist, jobs, output)
    assert not output.exists()

    result.write_bytes(result.read_bytes().replace(b"A changed ", b"A concrete"))
    rows[0]["private_router_result_file_sha256"] = _sha(result.read_bytes())
    _jsonl(allowlist, rows)
    qa = Path(rows[0]["private_qa_path"])
    qa.write_bytes(qa.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        _run(allowlist, jobs, output)
    assert not output.exists()


def test_missing_job_or_unsafe_output_fails(tmp_path: Path) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    _jsonl(allowlist, rows[:1])
    with pytest.raises(ValueError, match="job set"):
        _run(allowlist, jobs, output)
    assert not output.exists()

    _jsonl(allowlist, rows)
    with pytest.raises(ValueError, match="outside run"):
        _run(allowlist, jobs, jobs.parent / "private-candidate")


def test_qa_slug_mention_without_pass_is_not_approval(tmp_path: Path) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    qa = Path(rows[0]["private_qa_path"])
    qa.write_text(json.dumps({"per_item": [{"slug": rows[0]["slug"]}]}) + "\n")
    rows[0]["private_qa_sha256"] = _sha(qa.read_bytes())
    _jsonl(allowlist, rows)
    with pytest.raises(ValueError, match="QA is not PASS"):
        _run(allowlist, jobs, output)
    assert not output.exists()


def test_itemized_findings_with_exact_result_pin_is_qa_pass(tmp_path: Path) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    qa = Path(rows[0]["private_qa_path"])
    qa.write_text(json.dumps({"pinned_artifacts": {"results": {
        "sha256": rows[0]["private_router_result_file_sha256"], "rows": 1}},
        "itemized_findings": [{"slug": rows[0]["slug"], "status": "PASS"}],
        "disposition": "BATCH_013_QA_PASS_PENDING_ROOT_ACCEPTANCE"}) + "\n")
    rows[0]["private_qa_sha256"] = _sha(qa.read_bytes())
    _jsonl(allowlist, rows)
    receipt = _run(allowlist, jobs, output)
    assert receipt["candidate_count"] == 2


def test_qa_pass_with_wrong_result_file_hash_fails_closed(tmp_path: Path) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    qa = Path(rows[0]["private_qa_path"])
    qa.write_text(json.dumps({"verdict": "PASS_PRIVATE_ONLY",
        "batch": {"results_sha256": "0" * 64},
        "per_item": [{"slug": rows[0]["slug"], "status": "PASS"}]}) + "\n")
    rows[0]["private_qa_sha256"] = _sha(qa.read_bytes())
    _jsonl(allowlist, rows)
    with pytest.raises(ValueError, match="QA is not PASS"):
        _run(allowlist, jobs, output)
    assert not output.exists()


def test_legacy_canonical_hash_moves_to_provenance_without_changing_science(tmp_path: Path) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    path = Path(rows[0]["private_router_result_path"])
    original = json.loads(path.read_text())
    legacy_hash = _canonical_result_hash(RouterResult.model_validate(original))
    _jsonl(path, [{**original, "raw_result_sha256": legacy_hash}])
    rows[0]["private_router_result_file_sha256"] = _sha(path.read_bytes())
    rows[0]["private_router_result_row_sha256"] = _sha(path.read_bytes()[:-1])
    _repin_qa_result(rows[0])
    _jsonl(allowlist, rows)
    receipt = _run(allowlist, jobs, output)
    candidate = json.loads((output / "router_results.candidate.v3.jsonl").read_text().splitlines()[0])
    assert candidate == original
    assert receipt["custody"][0]["legacy_router_canonical_sha256"] == legacy_hash
    assert receipt["custody"][0]["source_row_sha256"] == rows[0]["private_router_result_row_sha256"]


@pytest.mark.parametrize("change", ["wrong_hash", "other_extra_key"])
def test_legacy_adapter_rejects_wrong_hash_or_any_other_extra_key(tmp_path: Path, change: str) -> None:
    allowlist, jobs, output, rows = _fixture(tmp_path)
    path = Path(rows[0]["private_router_result_path"])
    original = json.loads(path.read_text())
    value = {**original, "raw_result_sha256": _canonical_result_hash(RouterResult.model_validate(original))}
    if change == "wrong_hash":
        value["raw_result_sha256"] = "0" * 64
    else:
        value["unrelated"] = "not allowed"
    _jsonl(path, [value])
    rows[0]["private_router_result_file_sha256"] = _sha(path.read_bytes())
    rows[0]["private_router_result_row_sha256"] = _sha(path.read_bytes()[:-1])
    _repin_qa_result(rows[0])
    _jsonl(allowlist, rows)
    with pytest.raises(ValueError, match="legacy|extra"):
        _run(allowlist, jobs, output)
    assert not output.exists()
