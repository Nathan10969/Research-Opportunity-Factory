from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from idea_factory.private_old_corpus_staging import (
    expand_census_positions,
    select_remaining,
    qa_binds_result,
    write_staging,
    choose_unique_qa_result,
    load_frozen_parent,
    build_remaining,
    validate_worker_card_links,
    select_accepted_subset,
    build_accepted_subset,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_expand_exact_census_positions_and_reject_overlap() -> None:
    report = """Exact latest-QA-PASS position allowlist (1-based):
- Shard 1 engineering: `1,3-4`.
- Shard 1 ACL09: `7-8`.
- Shard 2 ACL03: `2`.
For the 1,232 result-bearing slugs, the 811 PASS rows are latest.
"""
    assert expand_census_positions(report, expected_count=6) == {
        (1, 1), (1, 3), (1, 4), (1, 7), (1, 8), (2, 2)
    }
    with pytest.raises(ValueError, match="overlap"):
        expand_census_positions(report.replace("`7-8`", "`4,7-8`"), expected_count=7)


def test_remaining_set_excludes_formal_pilot_and_hard_hold_without_replacement() -> None:
    rows = [
        {"slug": "formal", "assignment_input_binding": {"status": "EXACT_WORKER_QUEUE_BOUND"}},
        {"slug": "pilot", "assignment_input_binding": {"status": "EXACT_WORKER_QUEUE_BOUND"}},
        {"slug": "hard", "assignment_input_binding": {"status": "RETAINED_REVIEW_REFERENCE_BINDING_MISMATCH_HOLD"}},
        {"slug": "ready", "assignment_input_binding": {"status": "EXACT_WORKER_QUEUE_BOUND"}},
    ]
    selected, excluded = select_remaining(rows, {"formal"}, {"pilot"})
    assert [row["slug"] for row in selected] == ["ready"]
    assert excluded == {"FORMAL_CARD": 1, "PRIOR_PILOT100": 1, "FROZEN_HARD_HOLD": 1}


def test_qa_requires_pinned_result_hash_and_pass_not_slug_prose(tmp_path: Path) -> None:
    result = tmp_path / "router-results.jsonl"
    result.write_bytes(b'{"slug":"alpha"}\n')
    digest = _sha(result.read_bytes())
    qa = {"verdict": "PASS_PRIVATE_ONLY", "batch": {"results_sha256": digest},
          "items": [{"slug": "alpha", "status": "PASS"}]}
    assert qa_binds_result(qa, digest, "alpha")
    assert not qa_binds_result(qa, "0" * 64, "alpha")
    assert not qa_binds_result({"verdict": "HOLD", "batch": qa["batch"],
                                "items": qa["items"]}, digest, "alpha")
    assert not qa_binds_result({"verdict": "PASS_PRIVATE_ONLY", "scope": "alpha",
                                "batch": qa["batch"]}, digest, "alpha")


def test_private_staging_contains_only_ready_candidates_and_hold_ledger(tmp_path: Path) -> None:
    note = tmp_path / "paper.md"
    pdf = tmp_path / "paper.pdf"
    ledger = tmp_path / "legacy.md"
    for path in (note, pdf, ledger):
        path.write_bytes(b"evidence\n")
    ready = [{"slug": "2026-aaai-123", "venue": "AAAI", "note_path": str(note),
              "note_sha256": _sha(note.read_bytes()), "pdf_path": str(pdf),
              "pdf_sha256": _sha(pdf.read_bytes()),
              "source_record_ids": ["aaai2026-123"],
              "source_status": "PRIMARY_SOURCE_VERIFIED", "source_version": None}]
    holds = [{"slug": "2026-aaai-124", "status": "HOLD", "reason": "QA pin missing"}]
    output = tmp_path / "private-stage"
    receipt = write_staging(output, ready, holds, notes_root=tmp_path,
                            legacy_ledger=ledger, input_pins={"census_sha256": "0" * 64})
    assert receipt["ready_count"] == 1 and receipt["hold_count"] == 1
    assert (output / "candidate-list.txt").read_text().splitlines() == [str(note)]
    assert len((output / "allowlist.jsonl").read_text().splitlines()) == 1
    assert len((output / "hold_ledger.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "corpus-router-v3-config.json").read_text())["target_max"] == 1
    with pytest.raises(FileExistsError):
        write_staging(output, ready, holds, notes_root=tmp_path,
                      legacy_ledger=ledger, input_pins={"census_sha256": "0" * 64})


def test_corrected_pin_beats_original_without_mtime_and_ambiguous_pins_hold() -> None:
    original, corrected = "1" * 64, "2" * 64
    candidates = [{"file_sha256": original, "row_sha256": "a" * 64, "path": "original"},
                  {"file_sha256": corrected, "row_sha256": "b" * 64, "path": "corrected"}]
    qa = {"qa": {"verdict": "PASS_PRIVATE_ONLY", "batch": {
        "original_results_sha256": original, "corrected_results_sha256": corrected},
        "items": [{"slug": "paper", "status": "PASS"}]}, "path": "qa"}
    selected, _ = choose_unique_qa_result(candidates, [qa], "paper")
    assert selected["file_sha256"] == corrected
    qa_with_implicit_corrected = {"qa": {"verdict": "PASS_PRIVATE_ONLY", "batch": {
        "original_results_sha256": original, "results_sha256": corrected},
        "items": [{"slug": "paper", "status": "PASS"}]}, "path": "qa-implicit"}
    selected, _ = choose_unique_qa_result(candidates, [qa_with_implicit_corrected], "paper")
    assert selected["file_sha256"] == corrected
    ambiguous = [
        {"qa": {"verdict": "PASS_PRIVATE_ONLY", "result_sha256": digest,
                 "items": [{"slug": "paper", "status": "PASS"}]}, "path": f"qa-{i}"}
        for i, digest in enumerate((original, corrected))
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        choose_unique_qa_result(candidates, ambiguous, "paper")


def test_frozen_parent_uses_only_pinned_physical_positions(tmp_path: Path) -> None:
    shard = tmp_path / "shard.jsonl"
    _jsonl = lambda rows: b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    raw = _jsonl([{"slug": "a"}, {"slug": "b"}])
    shard.write_bytes(raw)
    summary = {"shards": [{"file": shard.name, "sha256": _sha(raw), "rows": 2}]}
    selected = load_frozen_parent(tmp_path, summary, {(1, 2)})
    assert selected[0][0]["slug"] == "b"
    assert selected[0][1:] == (shard, _sha(raw), 2, _sha(raw.splitlines()[1]))
    with pytest.raises(ValueError, match="shard SHA-256"):
        load_frozen_parent(tmp_path, {"shards": [{**summary["shards"][0],
                                                  "sha256": "0" * 64}]}, {(1, 2)})


def test_builder_rejects_unpinned_census_before_output(tmp_path: Path) -> None:
    census = tmp_path / "census.md"
    census.write_text("not the frozen report")
    with pytest.raises(ValueError, match="SHA-256"):
        build_remaining(tmp_path, tmp_path, census, "0" * 64,
                        tmp_path / "pilot.jsonl", "0" * 64,
                        tmp_path / "output", expected_parent_count=1)
    assert not (tmp_path / "output").exists()


def test_old_worker_review_created_is_not_terminal_card_status() -> None:
    frozen = {"slug": "paper", "canonical_note_path": "note.md",
              "canonical_note_sha256": "1" * 64, "raw_path": "raw.json",
              "raw_sha256": "2" * 64}
    review = {"item_id": "paper", "status": "CREATED", "note_path": "note.md",
              "note_sha256": "1" * 64, "reviewed_raw_path": "raw.json"}
    receipt = {"slug": "paper", "status": "SCHEMA_VALID", "accepted_cards": 1,
               "raw_result_sha256": "2" * 64}
    validate_worker_card_links(frozen, review, receipt)
    parallel_review = {"item_id": "paper", "status": "CREATED",
                       "canonical_note_path": "note.md", "note_sha256": "1" * 64}
    validate_worker_card_links(frozen, parallel_review, receipt)
    validate_worker_card_links(frozen, {"status": "ACCEPT", "source_note_path": "note.md",
                                        "source_note_sha256": "1" * 64}, receipt)
    validate_worker_card_links(frozen, {"bindings": {"note_path": "note.md",
                                                    "note_sha256": "1" * 64}}, receipt)
    with pytest.raises(ValueError, match="review"):
        validate_worker_card_links(frozen, {"status": "CREATED"}, receipt)
    with pytest.raises(ValueError, match="receipt"):
        validate_worker_card_links(frozen, review, {**receipt, "status": "VALID_EMPTY"})


def test_accepted_subset_is_exact_outcome_join_and_keeps_invalid_as_hold() -> None:
    original = [{"slug": "a", "note_sha256": "1" * 64},
                {"slug": "b", "note_sha256": "2" * 64}]
    jobs = [{"slug": "a", "job_id": "job-a", "note_sha256": "1" * 64},
            {"slug": "b", "job_id": "job-b", "note_sha256": "2" * 64}]
    outcomes = [{"slug": "a", "job_id": "job-a", "note_sha256": "1" * 64,
                 "status": "ACCEPTED", "accepted_count": 1, "accepted_card_ids": ["card-a"]},
                {"slug": "b", "job_id": "job-b", "note_sha256": "2" * 64,
                 "status": "INVALID", "accepted_count": 0, "accepted_card_ids": []}]
    cards = [{"slug": "a", "job_id": "job-a", "record_id": "card-a"}]
    ready, holds = select_accepted_subset(original, jobs, outcomes, cards)
    assert [row["slug"] for row in ready] == ["a"]
    assert holds == [{"slug": "b", "status": "HOLD", "reason": "prior Card outcome INVALID"}]
    with pytest.raises(ValueError, match="job identity"):
        select_accepted_subset(original, jobs, [{**outcomes[0], "job_id": "wrong"}, outcomes[1]], cards)


def test_accepted_subset_builder_rejects_unpinned_original(tmp_path: Path) -> None:
    original = tmp_path / "allowlist.jsonl"
    original.write_bytes(b'{}\n')
    with pytest.raises(ValueError, match="SHA-256"):
        build_accepted_subset(original, "0" * 64, tmp_path / "old-run", tmp_path / "out")
    assert not (tmp_path / "out").exists()
