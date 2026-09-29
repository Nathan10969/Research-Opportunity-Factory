from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import idea_factory.private_card_transport as transport_module
from test_cards import _card, _job_bundle
from idea_factory.private_card_transport import transport


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _fixture(tmp_path: Path, *, hold: bool = False) -> tuple[Path, Path, Path, dict]:
    jobs_path, jobs = _job_bundle(tmp_path)
    job = jobs[0]
    run = jobs_path.parent.parent
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nsource\n")
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_prompt = old_dir / "paper_card.md"
    old_prompt.write_bytes(Path(job["prompt_path"]).read_bytes())
    card = _card(job)
    raw = _write(old_dir / "raw.json", {
        "schema_version": "idea_factory.bulk_card_response.v1", "slug": "worker-" + job["slug"],
        "note_sha256": job["note_sha256"], "prompt_sha256": job["prompt_sha256"],
        "cards": [card], "empty_reason": "",
    })
    task = _write(old_dir / "task.json", {
        "slug": "worker-" + job["slug"], "shard_id": "worker", "note_path": job["note_path"],
        "note_sha256": job["note_sha256"], "prompt_path": str(old_prompt),
        "prompt_sha256": job["prompt_sha256"], "source_pdf_path": str(pdf.resolve()),
        "source_pdf_sha256": _sha(pdf), "output_path": str(raw.resolve()), "card_id": card["card_id"],
        "source_aliases": [{"arxiv_id": "testv1"}],
    })
    review = _write(old_dir / "review.json", {
        "raw_path": str(raw.resolve()), "raw_sha256": _sha(raw),
        "note_path": job["note_path"], "note_sha256": job["note_sha256"],
        "source_pdf": {"path": str(pdf.resolve()), "sha256": _sha(pdf)},
        "terminal_disposition": "SCHEMA_VALID",
    })
    validated = _write(old_dir / "schema_validated.json", {
        "schema_version": "idea_factory.bulk_schema_validated_card.v1", "card": card,
        "slug": "worker-" + job["slug"], "note_path": job["note_path"],
        "note_sha256": job["note_sha256"], "prompt_sha256": job["prompt_sha256"],
        "raw_result_sha256": _sha(raw), "record_id": card["card_id"],
    })
    row = {
        "job_id": job["job_id"], "slug": job["slug"], "old_raw": _pin(raw),
        "old_task": _pin(task), "old_review": _pin(review),
        "old_schema_validated": _pin(validated), "note": _pin(Path(job["note_path"])),
        "prompt": _pin(old_prompt), "pdf": _pin(pdf),
        "source_record_id": "arxiv:testv1", "source_version": "v1",
        "science_gate": "HOLD" if hold else "PASS",
        "science_hold_reason": "mechanism unverified" if hold else "",
    }
    run_files = {name: _pin(run / rel) for name, rel in {
        "jobs": "cards/card_jobs.jsonl", "selection": "corpus/selection_manifest.jsonl",
        "rejected": "corpus/rejected_manifest.jsonl", "router_jobs": "corpus/router_jobs.jsonl",
        "router_results": "corpus/router_results.jsonl", "selection_policy": "corpus/selection_policy.jsonl",
    }.items()}
    allowlist = _write(tmp_path / "allowlist.json", {
        "schema_version": "private_card_transport_allowlist.v1", "run_path": str(run.resolve()),
        "run_files": run_files, "rows": [row],
    })
    return run, allowlist, tmp_path / "candidate", row


def test_rewraps_card_without_changing_body_and_never_ingests(tmp_path: Path) -> None:
    run, allowlist, output, row = _fixture(tmp_path)
    source_before = Path(row["old_raw"]["path"]).read_bytes()
    receipt = transport(run, allowlist, _sha(allowlist), output)
    candidate = json.loads((output / "paper_card_results.candidate.v1.jsonl").read_text().splitlines()[0])
    old = json.loads(source_before)
    assert candidate["cards"] == old["cards"]
    canonical_card = json.dumps(old["cards"][0], ensure_ascii=False, allow_nan=False,
                                sort_keys=True, separators=(",", ":")).encode("utf-8")
    ledger = json.loads((output / "hold_ledger.v1.jsonl").read_text().splitlines()[0])
    assert ledger["card_body_sha256"] == hashlib.sha256(canonical_card).hexdigest()
    assert candidate["slug"] == row["slug"]
    assert candidate["job_id"] == row["job_id"]
    assert receipt["ready_count"] == 1 and receipt["hold_count"] == 0
    assert receipt["human_approved"] is False and receipt["graph_ingested"] is False
    assert Path(row["old_raw"]["path"]).read_bytes() == source_before
    assert not (run / "results").exists()


def test_science_hold_has_no_candidate_or_fake_empty(tmp_path: Path) -> None:
    run, allowlist, output, _ = _fixture(tmp_path, hold=True)
    receipt = transport(run, allowlist, _sha(allowlist), output)
    assert (output / "paper_card_results.candidate.v1.jsonl").read_bytes() == b""
    ledger = json.loads((output / "hold_ledger.v1.jsonl").read_text().splitlines()[0])
    assert ledger["status"] == "HOLD" and "mechanism" in ledger["reason"]
    assert receipt["ready_count"] == 0 and receipt["hold_count"] == 1


def test_bad_pin_fails_before_output(tmp_path: Path) -> None:
    run, allowlist, output, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="allowlist SHA"):
        transport(run, allowlist, "0" * 64, output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    run, allowlist, output, _ = _fixture(tmp_path)
    output.mkdir()
    with pytest.raises(FileExistsError):
        transport(run, allowlist, _sha(allowlist), output)


def _rewrite_allowlist(allowlist: Path, change: object) -> None:
    value = json.loads(allowlist.read_text(encoding="utf-8"))
    change(value)
    _write(allowlist, value)


def test_tampered_raw_fails_before_output_and_preserves_source(tmp_path: Path) -> None:
    run, allowlist, output, row = _fixture(tmp_path)
    raw = Path(row["old_raw"]["path"])
    raw.write_bytes(raw.read_bytes() + b" ")
    changed = raw.read_bytes()
    with pytest.raises(ValueError, match="pinned SHA-256 mismatch"):
        transport(run, allowlist, _sha(allowlist), output)
    assert raw.read_bytes() == changed and not output.exists()


def test_old_empty_result_is_explicit_hold_not_candidate(tmp_path: Path) -> None:
    run, allowlist, output, row = _fixture(tmp_path)
    raw = Path(row["old_raw"]["path"])
    old = json.loads(raw.read_text(encoding="utf-8"))
    old["cards"] = []
    old["empty_reason"] = "unknown"
    _write(raw, old)
    _rewrite_allowlist(allowlist, lambda value: value["rows"][0]["old_raw"].update(sha256=_sha(raw)))
    receipt = transport(run, allowlist, _sha(allowlist), output)
    assert receipt["ready_count"] == 0 and receipt["hold_count"] == 1
    assert (output / "paper_card_results.candidate.v1.jsonl").read_bytes() == b""


def test_source_identity_mismatch_is_hold(tmp_path: Path) -> None:
    run, allowlist, output, _ = _fixture(tmp_path)
    _rewrite_allowlist(allowlist, lambda value: value["rows"][0].update(source_record_id="arxiv:other-v1"))
    receipt = transport(run, allowlist, _sha(allowlist), output)
    assert receipt["ready_count"] == 0 and receipt["hold_count"] == 1


def test_unreviewed_science_can_be_technical_candidate_without_approval(tmp_path: Path) -> None:
    run, allowlist, output, _ = _fixture(tmp_path)
    _rewrite_allowlist(allowlist, lambda value: value["rows"][0].update(science_gate="UNREVIEWED"))
    receipt = transport(run, allowlist, _sha(allowlist), output)
    ledger = json.loads((output / "hold_ledger.v1.jsonl").read_text().splitlines()[0])
    assert receipt["ready_count"] == 1 and receipt["scientific_entailment_audited"] is False
    assert ledger["status"] == "READY" and ledger["science_gate"] == "UNREVIEWED"


def test_missing_source_identity_evidence_is_hold(tmp_path: Path) -> None:
    run, allowlist, output, row = _fixture(tmp_path)
    task = Path(row["old_task"]["path"])
    value = json.loads(task.read_text(encoding="utf-8"))
    value.pop("source_aliases")
    _write(task, value)
    _rewrite_allowlist(allowlist, lambda manifest: manifest["rows"][0]["old_task"].update(sha256=_sha(task)))
    receipt = transport(run, allowlist, _sha(allowlist), output)
    assert receipt["ready_count"] == 0 and receipt["hold_count"] == 1


def test_old_slug_must_be_shard_prefixed_new_slug(tmp_path: Path) -> None:
    run, allowlist, output, row = _fixture(tmp_path)
    raw_path = Path(row["old_raw"]["path"])
    task_path = Path(row["old_task"]["path"])
    review_path = Path(row["old_review"]["path"])
    validated_path = Path(row["old_schema_validated"]["path"])
    wrong = "unrelated-private-slug"
    raw = json.loads(raw_path.read_text(encoding="utf-8")); raw["slug"] = wrong; _write(raw_path, raw)
    task = json.loads(task_path.read_text(encoding="utf-8")); task["slug"] = wrong; _write(task_path, task)
    review = json.loads(review_path.read_text(encoding="utf-8")); review["raw_sha256"] = _sha(raw_path); _write(review_path, review)
    validated = json.loads(validated_path.read_text(encoding="utf-8")); validated["slug"] = wrong; validated["raw_result_sha256"] = _sha(raw_path); _write(validated_path, validated)
    def repin(value: dict) -> None:
        for name in ("old_raw", "old_task", "old_review", "old_schema_validated"):
            value["rows"][0][name]["sha256"] = _sha(Path(value["rows"][0][name]["path"]))
    _rewrite_allowlist(allowlist, repin)
    receipt = transport(run, allowlist, _sha(allowlist), output)
    assert receipt["ready_count"] == 0 and receipt["hold_count"] == 1


def test_path_alias_rejected_before_output(tmp_path: Path) -> None:
    run, allowlist, output, _ = _fixture(tmp_path)
    _rewrite_allowlist(allowlist, lambda value: value["rows"][0]["old_raw"].update(path=str(tmp_path / "old" / ".." / "old" / "raw.json")))
    with pytest.raises(ValueError, match="path alias"):
        transport(run, allowlist, _sha(allowlist), output)
    assert not output.exists()


def test_output_write_failure_never_publishes_candidate_or_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, allowlist, output, _ = _fixture(tmp_path)
    original = transport_module._write_verified

    def fail_ledger(path: Path, data: bytes) -> None:
        if path.name == "hold_ledger.v1.jsonl":
            raise OSError("injected ledger write failure")
        original(path, data)

    monkeypatch.setattr(transport_module, "_write_verified", fail_ledger)
    with pytest.raises(OSError, match="injected ledger"):
        transport(run, allowlist, _sha(allowlist), output)
    assert not output.exists()
