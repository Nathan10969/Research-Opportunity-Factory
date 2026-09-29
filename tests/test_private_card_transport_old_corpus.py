from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from idea_factory.private_card_transport import _check_old_corpus_row, transport
from idea_factory.cards import _job_rows
from test_transport_private_card_results import _fixture
from idea_factory.private_card_allowlist_builder import _row, build_old_corpus_allowlist


STAGING = Path(r"F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\audits\old811-card-pilot100-strict-v3-input-staging-20260929-v1\allowlist.jsonl")
RUN = Path(r"F:\LLM_Evoke\runs\old811-card-pilot100-v3-20260929T0237Z")


def _pin(path: str | Path) -> dict[str, str]:
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.skipif(not STAGING.is_file(), reason="frozen old811 staging is unavailable")
def test_real_first_old_worker_card_rewraps_without_science_approval() -> None:
    source = json.loads(STAGING.read_bytes().splitlines()[0])
    assert source["slug"] == "2026-aaai-37168"
    task = json.loads(Path(source["task_path"]).read_bytes())
    note = Path(source["note_path"])
    prompt = Path(task["prompt_path"])
    row = {
        "origin_format": "old_worker_v1", "job_id": "new-card-job", "slug": source["slug"],
        "old_task": _pin(source["task_path"]), "old_raw": _pin(source["raw_path"]),
        "old_review": _pin(source["review_path"]),
        "old_schema_validated": _pin(Path(source["task_path"]).with_name("schema_validated.json")),
        "old_receipt": _pin(source["card_receipt_path"]),
        "note": _pin(note), "prompt": _pin(prompt), "pdf": _pin(source["pdf_path"]),
        "source_record_id": source["source_record_ids"][0], "source_version": None,
        "science_gate": "UNREVIEWED", "science_hold_reason": "",
    }
    job = {
        "job_id": "new-card-job", "slug": source["slug"], "note_path": str(note),
        "note_sha256": source["note_sha256"], "note_text": note.read_bytes().decode("utf-8"),
        "prompt_sha256": task["prompt_sha256"], "prompt_text": prompt.read_bytes().decode("utf-8"),
    }
    selected = {"primary_source_record": {
        "primary_pdf_path": source["pdf_path"], "primary_pdf_sha256": source["pdf_sha256"],
        "source_record_ids": source["source_record_ids"], "source_version": None,
    }}
    data = {name: Path(row[name]["path"]).read_bytes() for name in
            ("old_task", "old_raw", "old_review", "old_schema_validated", "old_receipt", "note", "prompt", "pdf")}
    wrapper, card_hash = _check_old_corpus_row(row, job, selected, source, data)
    old_card = json.loads(data["old_raw"])["cards"][0]
    assert wrapper["cards"] == [old_card]
    assert card_hash == hashlib.sha256(json.dumps(old_card, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()
    assert wrapper["job_id"] == job["job_id"]


@pytest.mark.skipif(not STAGING.is_file() or not (RUN / "cards" / "card_jobs.jsonl").is_file(),
                    reason="frozen old811 run is unavailable")
def test_real_100_builder_emits_exact_two_origin_formats_without_science_pass(tmp_path: Path) -> None:
    digest = hashlib.sha256(STAGING.read_bytes()).hexdigest()
    receipt = build_old_corpus_allowlist(RUN, STAGING, digest, tmp_path / "built")
    manifest = json.loads((tmp_path / "built" / "transport_allowlist.v2.json").read_bytes())
    rows = manifest["rows"]
    assert receipt["input_count"] == len(rows) == 100
    assert {kind: sum(row["origin_format"] == kind for row in rows)
            for kind in ("old_worker_v1", "reused_old_corpus_v1")} == {
                "old_worker_v1": 98, "reused_old_corpus_v1": 2,
            }
    assert all(row["science_gate"] == "UNREVIEWED" for row in rows)
    assert len({row["job_id"] for row in rows}) == 100
    assert receipt["scientific_entailment_audited"] is False
    assert receipt["ready_unreviewed_count"] == 98
    assert receipt["hold_count"] == 2


@pytest.mark.skipif(not STAGING.is_file() or not (RUN / "cards" / "card_jobs.jsonl").is_file(),
                    reason="frozen old811 run is unavailable")
def test_real_reused_card_uses_reuse_and_qa_pins_without_schema_validated() -> None:
    lines = STAGING.read_bytes().splitlines(keepends=True)
    source = next(json.loads(line) for line in lines if json.loads(line)["slug"] == "2026-aaai-37195")
    jobs = [json.loads(line) for line in (RUN / "cards" / "card_jobs.jsonl").read_bytes().splitlines()]
    selected = [json.loads(line) for line in (RUN / "corpus" / "selection_manifest.jsonl").read_bytes().splitlines()]
    job = next(job for job in jobs if job["slug"] == source["slug"])
    selection = next(item for item in selected if item["slug"] == source["slug"])
    line_no = next(i for i, line in enumerate(lines, 1) if json.loads(line)["slug"] == source["slug"])
    row = _row(source, job, line_no, hashlib.sha256(lines[line_no - 1]).hexdigest())
    assert row["origin_format"] == "reused_old_corpus_v1"
    assert "old_schema_validated" not in row
    data = {name: Path(row[name]["path"]).read_bytes() for name in
            ("old_task", "old_raw", "old_review", "old_receipt", "note", "prompt", "pdf", "old_reuse", "old_reuse_qa")}
    with pytest.raises(ValueError, match="reused QA"):
        _check_old_corpus_row(row, job, selection, source, data)


@pytest.mark.skipif(not STAGING.is_file() or not (RUN / "cards" / "card_jobs.jsonl").is_file(),
                    reason="frozen old811 run is unavailable")
def test_builder_rejects_wrong_staging_pin_before_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="staging SHA"):
        build_old_corpus_allowlist(RUN, STAGING, "0" * 64, tmp_path / "built")
    assert not (tmp_path / "built").exists()


@pytest.mark.skipif(not STAGING.is_file() or not (RUN / "cards" / "card_jobs.jsonl").is_file(),
                    reason="frozen old811 run is unavailable")
def test_real_100_private_transport_keeps_all_card_objects_and_no_ingest(tmp_path: Path) -> None:
    results = RUN / "results"
    before_results = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in results.rglob("*") if path.is_file()}
    digest = hashlib.sha256(STAGING.read_bytes()).hexdigest()
    build_old_corpus_allowlist(RUN, STAGING, digest, tmp_path / "built")
    allowlist = tmp_path / "built" / "transport_allowlist.v2.json"
    receipt = transport(RUN, allowlist, hashlib.sha256(allowlist.read_bytes()).hexdigest(), tmp_path / "candidate")
    assert receipt["ready_count"] == 98 and receipt["hold_count"] == 2
    candidates = [json.loads(line) for line in (tmp_path / "candidate" / "paper_card_results.candidate.v1.jsonl").read_bytes().splitlines()]
    manifest = json.loads(allowlist.read_bytes())
    by_job = {row["job_id"]: row for row in manifest["rows"]}
    assert len(candidates) == 98
    for candidate in candidates:
        raw = json.loads(Path(by_job[candidate["job_id"]]["old_raw"]["path"]).read_bytes())
        assert candidate["cards"] == raw["cards"]
    assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in results.rglob("*") if path.is_file()} == before_results


def test_portable_old_worker_receipt_status_tamper_is_hold(tmp_path: Path) -> None:
    run, _, _, old = _fixture(tmp_path)
    job = _job_rows(run / "cards" / "card_jobs.jsonl")[1][0]
    task_path = Path(old["old_task"]["path"])
    raw_path = Path(old["old_raw"]["path"])
    review_path = Path(old["old_review"]["path"])
    validated_path = Path(old["old_schema_validated"]["path"])
    task = json.loads(task_path.read_bytes()); task.update(schema_version="idea_factory.bulk_card_task.v1", slug=job["slug"])
    task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
    raw = json.loads(raw_path.read_bytes()); raw["slug"] = job["slug"]
    raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    review = {"schema_version": "idea_factory.old_note_review.v1", "item_id": job["slug"],
              "status": "CREATED", "note_path": job["note_path"], "note_sha256": job["note_sha256"],
              "reviewed_raw_path": str(raw_path)}
    review_path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    validated = json.loads(validated_path.read_bytes()); validated.update(
        slug=job["slug"], raw_result_sha256=_pin(raw_path)["sha256"])
    validated_path.write_text(json.dumps(validated) + "\n", encoding="utf-8")
    receipt_path = task_path.with_name("receipt.json")
    receipt = {"slug": job["slug"], "shard_id": task["shard_id"], "status": "SCHEMA_VALID",
               "errors": [], "accepted_cards": 1, "raw_result_sha256": _pin(raw_path)["sha256"]}
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    row = {**old, "origin_format": "old_worker_v1", "old_task": _pin(task_path),
           "old_raw": _pin(raw_path), "old_review": _pin(review_path),
           "old_schema_validated": _pin(validated_path), "old_receipt": _pin(receipt_path),
           "source_record_id": "synthetic:alpha", "source_version": None,
           "science_gate": "UNREVIEWED"}
    staging = {"status": "INPUT_PRECHECK_PASS_SCIENCE_PENDING", "science_approved": False,
               "slug": job["slug"], "pdf_path": row["pdf"]["path"],
               "pdf_sha256": row["pdf"]["sha256"], "source_record_ids": ["synthetic:alpha"],
               "source_version": None}
    for name, prefix in (("old_task", "task"), ("old_raw", "raw"), ("old_review", "review"),
                         ("old_receipt", "card_receipt"), ("note", "note"), ("pdf", "pdf")):
        staging[f"{prefix}_path"] = row[name]["path"]
        staging[f"{prefix}_sha256"] = row[name]["sha256"]
    selected = {"primary_source_record": {
        "primary_pdf_path": staging["pdf_path"], "primary_pdf_sha256": staging["pdf_sha256"],
        "source_record_ids": staging["source_record_ids"], "source_version": None,
    }}
    data = {name: Path(row[name]["path"]).read_bytes() for name in
            ("old_task", "old_raw", "old_review", "old_schema_validated", "old_receipt", "note", "prompt", "pdf")}
    _check_old_corpus_row(row, job, selected, staging, data)
    receipt["status"] = "INVALID"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    row["old_receipt"] = _pin(receipt_path)
    staging["card_receipt_sha256"] = row["old_receipt"]["sha256"]
    data["old_receipt"] = receipt_path.read_bytes()
    with pytest.raises(ValueError, match="structural receipt"):
        _check_old_corpus_row(row, job, selected, staging, data)
