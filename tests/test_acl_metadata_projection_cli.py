import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


WORKTREE = Path(__file__).resolve().parents[1]
PYTHON = Path(r"F:\LLM_Evoke\idea_factory\.venv\Scripts\python.exe")
CLI = WORKTREE / "scripts" / "build_acl_metadata_projection.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _write_jsonl(path: Path, rows) -> bytes:
    data = b"".join(_json_bytes(row) + b"\r\n" for row in rows)
    path.write_bytes(data)
    return data


def _make_source(root: Path, item_id: str, index: int):
    source_dir = root / "sources"
    source_dir.mkdir(exist_ok=True)
    queue_path = source_dir / f"queue-{index}.jsonl"
    queue_row = {"item_id": item_id, "expected_title": "bib"}
    queue_bytes = _write_jsonl(queue_path, [queue_row])
    manifest_path = source_dir / f"manifest-{index}.jsonl"
    manifest_bytes = _write_jsonl(manifest_path, [{"item_id": item_id, "title": "bib"}])
    html_path = source_dir / f"official-{index}.html"
    html_path.write_bytes(f"<html><body>{item_id}</body></html>".encode())
    pdf_path = source_dir / f"{item_id}.pdf"
    pdf_path.write_bytes(f"pinned-pdf-{item_id}".encode())
    cache_receipt = source_dir / f"{item_id}.receipt.json"
    cache_receipt.write_bytes(b'{"text_sha256":"dummy"}')
    cache_text = source_dir / f"{item_id}.txt"
    cache_text.write_bytes(b"cached source text")
    proof = {
        "item_id": item_id,
        "queue_path": str(queue_path),
        "queue_file_sha256": _sha(queue_bytes),
        "queue_line_number": 1,
        "queue_raw_line_sha256": _sha(queue_bytes.splitlines(keepends=True)[0]),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _sha(manifest_bytes),
        "manifest_line_number": 1,
        "manifest_raw_line_sha256": _sha(manifest_bytes.splitlines(keepends=True)[0]),
        "official_html_path": str(html_path),
        "official_html_sha256": _sha(html_path.read_bytes()),
        "pdf_path": str(pdf_path),
        "pdf_sha256": _sha(pdf_path.read_bytes()),
        "cache_receipt_path": str(cache_receipt),
        "cache_receipt_sha256": _sha(cache_receipt.read_bytes()),
        "cache_text_path": str(cache_text),
        "cache_text_sha256": _sha(cache_text.read_bytes()),
        "current_expected_title": "bib",
        "current_source_title": "bib",
        "official_html_witness": {
            "outcome": "UNIQUE_WITNESS",
            "title": f"Paper {index}: Exact ACL Title",
            "authors": [f"Author {index} A", f"Author {index} B"],
            "publication_section_witness": {
                "text": "Findings of the Association for Computational Linguistics: ACL 2026",
                "year": 2026,
            },
        },
        "outcome": "UNIQUE_WITNESS",
        "overlay_is_approval": False,
        "source_status_mutation": False,
    }
    page1_text = f"Paper {index}: Exact ACL Title\nAuthor {index} A; Author {index} B\nACL 2026"
    page1 = {
        "item_id": item_id,
        "pdf_sha256": proof["pdf_sha256"],
        "page1_text_sha256": _sha(page1_text.encode("utf-8")),
        "page1_text": page1_text,
        "pdftotext_version": "pdftotext version test-fixture",
        "witness_author_id": f"builder-{index}",
    }
    review = {
        "item_id": item_id,
        "proof_row_sha256": _sha(_json_bytes(proof)),
        "pdf_sha256": proof["pdf_sha256"],
        "page1_text_sha256": page1["page1_text_sha256"],
        "decision": "SUPPORTED",
        "reviewer_id": f"reviewer-{index}",
        "reviewed_at_utc": "2026-09-28T01:02:03Z",
        "reason": "Independent native page-1 identity confirmed.",
        "native_pdf_identity_verified": True,
        "field_verdicts": {"title": "PASS", "ordered_authors": "PASS", "venue": "PASS", "year": "PASS"},
    }
    return proof, page1, review, [queue_path, manifest_path, html_path, pdf_path, cache_receipt, cache_text]


@pytest.fixture
def projection_inputs(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    ids = [f"2026.findings-acl.{n}" for n in (1001, 1002, 1003)]
    built = [_make_source(root, item_id, i + 1) for i, item_id in enumerate(ids)]
    proofs = [row[0] for row in built]
    page1_rows = [row[1] for row in built]
    reviews = [row[2] for row in built]
    proof_path = root / "proof.jsonl"
    proof_data = _write_jsonl(proof_path, proofs)
    proof_receipt_path = root / "proof-receipt.json"
    proof_receipt = {
        "schema_version": "acl670_metadata_proof_only_run_receipt.v1",
        "scope": "3 frozen ACL Findings queue rows with expected_title and source_record.title equal to bib",
        "final_proof_only_run": {
            "rows_path": str(proof_path),
            "rows_sha256": _sha(proof_data),
            "row_count": 3,
            "unique_item_ids": 3,
        },
    }
    proof_receipt_data = _json_bytes(proof_receipt)
    proof_receipt_path.write_bytes(proof_receipt_data)
    page1_path = root / "page1.jsonl"
    page1_data = _write_jsonl(page1_path, page1_rows)
    reviews_path = root / "reviews.jsonl"
    reviews_data = _write_jsonl(reviews_path, reviews)
    input_paths = [proof_path, proof_receipt_path, page1_path, reviews_path]
    input_paths += [path for row in built for path in row[3]]
    snapshots = {path: _sha(path.read_bytes()) for path in input_paths}
    out = root / "runs" / "projection-1"
    return {
        "root": root,
        "ids": ids,
        "proofs": proofs,
        "page1_rows": page1_rows,
        "reviews": reviews,
        "proof_path": proof_path,
        "proof_receipt_path": proof_receipt_path,
        "page1_path": page1_path,
        "reviews_path": reviews_path,
        "out": out,
        "snapshots": snapshots,
    }


def _args(data, *, out=None, proof_sha=None, expected_count=3):
    proof_bytes = data["proof_path"].read_bytes()
    page1_bytes = data["page1_path"].read_bytes()
    reviews_bytes = data["reviews_path"].read_bytes()
    receipt_bytes = data["proof_receipt_path"].read_bytes()
    return [
        "--proof", str(data["proof_path"]),
        "--proof-sha256", proof_sha or _sha(proof_bytes),
        "--proof-receipt", str(data["proof_receipt_path"]),
        "--proof-receipt-sha256", _sha(receipt_bytes),
        "--page1-witnesses", str(data["page1_path"]),
        "--page1-sha256", _sha(page1_bytes),
        "--reviews", str(data["reviews_path"]),
        "--reviews-sha256", _sha(reviews_bytes),
        "--corpus-root", str(data["root"]),
        "--output-dir", str(out or data["out"]),
        "--expected-count", str(expected_count),
    ]


def _run(data, **kwargs):
    env = os.environ.copy()
    src = str(WORKTREE / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [str(PYTHON), "-X", "utf8", str(CLI), *_args(data, **kwargs)],
        cwd=WORKTREE,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _assert_inputs_unchanged(data):
    for path, digest in data["snapshots"].items():
        assert _sha(path.read_bytes()) == digest, path


def test_projection_conserves_every_source_row_and_holds_missing_review(projection_inputs):
    data = projection_inputs
    data["reviews"][1]["decision"] = "METADATA_HOLD"
    data["reviews"][1]["reason"] = "The native PDF author order is ambiguous."
    data["reviews"][1]["field_verdicts"]["ordered_authors"] = "UNCLEAR"
    _write_jsonl(data["reviews_path"], data["reviews"][:2])
    data["snapshots"][data["reviews_path"]] = _sha(data["reviews_path"].read_bytes())
    result = _run(data)

    assert result.returncode == 0, result.stderr
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    diagnostic = [json.loads(line) for line in (data["out"] / "acl_metadata_witness.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(projection) == len(diagnostic) == 3
    assert len({row["item_id"] for row in projection}) == 3
    assert {row["status"] for row in projection} == {"SUPPORTED", "METADATA_HOLD"}
    missing_id = data["ids"][2]
    assert next(row for row in projection if row["item_id"] == missing_id)["hold_reason"] == "REVIEW_MISSING"
    held = next(row for row in projection if row["item_id"] == data["ids"][1])
    assert held["status"] == "METADATA_HOLD"
    assert all(field["approved"] is None for field in held["fields"].values())
    receipt = json.loads((data["out"] / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["counts"] == {"SUPPORTED": 1, "METADATA_HOLD": 2}
    assert receipt["source_admission_approved"] is False
    assert receipt["human_approved"] is False
    assert receipt["graph_ingested"] is False
    _assert_inputs_unchanged(data)


@pytest.mark.parametrize("duplicate_kind", ["proof", "reviews", "page1"])
def test_duplicate_item_ids_are_rejected_without_outputs_or_source_mutation(projection_inputs, duplicate_kind):
    data = projection_inputs
    key = {"proof": "proof_path", "reviews": "reviews_path", "page1": "page1_path"}[duplicate_kind]
    rows_key = {"proof": "proofs", "reviews": "reviews", "page1": "page1_rows"}[duplicate_kind]
    rows = list(data[rows_key]) + [dict(data[rows_key][0])]
    new_data = _write_jsonl(data[key], rows)
    if duplicate_kind == "proof":
        receipt = json.loads(data["proof_receipt_path"].read_text(encoding="utf-8"))
        receipt["final_proof_only_run"]["rows_sha256"] = _sha(new_data)
        data["proof_receipt_path"].write_bytes(_json_bytes(receipt))
        data["snapshots"][data["proof_receipt_path"]] = _sha(data["proof_receipt_path"].read_bytes())
    data["snapshots"][data[key]] = _sha(data[key].read_bytes())
    result = _run(data)

    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()
    assert not data["out"].exists()
    assert not (data["out"] / "receipt.json").exists()
    _assert_inputs_unchanged(data)


def test_duplicate_review_item_is_rejected_before_any_output(projection_inputs):
    data = projection_inputs
    _write_jsonl(data["reviews_path"], data["reviews"] + [data["reviews"][0]])
    data["snapshots"][data["reviews_path"]] = _sha(data["reviews_path"].read_bytes())
    result = _run(data)

    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()
    assert not data["out"].exists()
    assert not (data["out"] / "receipt.json").exists()
    _assert_inputs_unchanged(data)


def test_altered_proof_file_pin_is_rejected_without_success_receipt(projection_inputs):
    data = projection_inputs
    result = _run(data, proof_sha="0" * 64)

    assert result.returncode != 0
    assert "proof" in result.stderr.lower()
    assert not data["out"].exists()
    assert not (data["out"] / "receipt.json").exists()
    _assert_inputs_unchanged(data)


def test_altered_queue_row_hash_is_rejected_without_success_receipt(projection_inputs):
    data = projection_inputs
    data["proofs"][0]["queue_raw_line_sha256"] = "0" * 64
    proof_bytes = _write_jsonl(data["proof_path"], data["proofs"])
    receipt = json.loads(data["proof_receipt_path"].read_text(encoding="utf-8"))
    receipt["final_proof_only_run"]["rows_sha256"] = _sha(proof_bytes)
    data["proof_receipt_path"].write_bytes(_json_bytes(receipt))
    data["snapshots"][data["proof_path"]] = _sha(proof_bytes)
    data["snapshots"][data["proof_receipt_path"]] = _sha(data["proof_receipt_path"].read_bytes())
    result = _run(data)

    assert result.returncode != 0
    assert "queue" in result.stderr.lower()
    assert not data["out"].exists()
    assert not (data["out"] / "receipt.json").exists()
    _assert_inputs_unchanged(data)


def test_output_directory_is_exclusive_and_existing_bytes_are_preserved(projection_inputs):
    data = projection_inputs
    data["out"].mkdir(parents=True)
    sentinel = data["out"] / "sentinel.txt"
    sentinel.write_bytes(b"do not overwrite")
    result = _run(data)

    assert result.returncode != 0
    assert "exist" in result.stderr.lower() or "exclusive" in result.stderr.lower()
    assert sentinel.read_bytes() == b"do not overwrite"
    assert not (data["out"] / "receipt.json").exists()
    _assert_inputs_unchanged(data)


def test_output_directory_must_stay_under_explicit_corpus_run_root(projection_inputs, tmp_path):
    data = projection_inputs
    result = _run(data, out=tmp_path / "outside" / "projection")

    assert result.returncode != 0
    assert "corpus" in result.stderr.lower() or "root" in result.stderr.lower()
    assert not (tmp_path / "outside").exists()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_symlink_output_path_component_is_rejected(projection_inputs):
    data = projection_inputs
    runs = data["root"] / "runs"
    runs.mkdir()
    target = data["root"] / "real-runs"
    target.mkdir()
    alias = runs / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    result = _run(data, out=alias / "projection")

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower() or "reparse" in result.stderr.lower()
    assert not (target / "projection").exists()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)
