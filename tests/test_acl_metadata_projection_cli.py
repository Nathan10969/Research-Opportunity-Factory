import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest


WORKTREE = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CLI = WORKTREE / "scripts" / "build_acl_metadata_projection.py"
_SPEC = importlib.util.spec_from_file_location("acl_metadata_projection_cli", CLI)
BUILDER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BUILDER
_SPEC.loader.exec_module(BUILDER)
_PRODUCTION_PARENT_PINS = (
    BUILDER._PARENT_RECEIPT_RELATIVE_PATH.as_posix(),
    BUILDER._PARENT_RECEIPT_SHA256,
    BUILDER._PARENT_PROOF_RELATIVE_PATH.as_posix(),
    BUILDER._PARENT_PROOF_SHA256,
)

TASK4_IDS = [
    "2026.findings-acl.1077", "2026.findings-acl.1105", "2026.findings-acl.1174",
    "2026.findings-acl.1266", "2026.findings-acl.1320", "2026.findings-acl.135",
    "2026.findings-acl.1371", "2026.findings-acl.1388", "2026.findings-acl.1412",
    "2026.findings-acl.1530",
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _producer_raw_lines(data: bytes) -> list[bytes]:
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    return lines


def test_physical_line_hash_bytes_match_frozen_producer_for_mixed_lf_and_crlf():
    data = b'{"item_id":"queue"}\r\n{"item_id":"manifest"}\n'

    assert BUILDER._physical_line(data, 1, "queue") == b'{"item_id":"queue"}\r'
    assert BUILDER._physical_line(data, 2, "manifest") == b'{"item_id":"manifest"}'
    assert BUILDER._json_physical_line(data, 1, "queue") == {"item_id": "queue"}
    assert BUILDER._json_physical_line(data, 2, "manifest") == {"item_id": "manifest"}


@pytest.mark.parametrize("data", [b'{"item_id":"x\ry"}\n', b'{"item_id":"x"}\rZ\n'])
def test_physical_line_rejects_embedded_carriage_returns(data):
    with pytest.raises(ValueError, match="embedded carriage return"):
        BUILDER._physical_line(data, 1, "fixture")


@pytest.mark.parametrize("data", [b"\n", b"\r\n"])
def test_physical_line_rejects_blank_rows(data):
    with pytest.raises(ValueError, match="blank JSONL row.*line 1"):
        BUILDER._physical_line(data, 1, "fixture")


def test_physical_line_numbers_follow_lf_delimited_physical_rows():
    data = b'{"item_id":"first"}\n{"item_id":"second"}'

    with pytest.raises(ValueError, match="physical line 3 is absent"):
        BUILDER._physical_line(data, 3, "fixture")
    with pytest.raises(ValueError, match="positive one-based integer"):
        BUILDER._physical_line(data, 0, "fixture")


def _pdftotext_path() -> str:
    executable = shutil.which("pdftotext")
    if executable is None:
        pytest.skip("pdftotext is required for native-page projection tests")
    return executable


def _minimal_pdf(index: int) -> bytes:
    stream = (
        "BT\n/F1 12 Tf\n72 720 Td\n"
        f"(Paper {index}: Exact ACL Title) Tj\n0 -20 Td\n"
        f"(Author {index} A; Author {index} B) Tj\n0 -20 Td\n"
        "(ACL 2026) Tj\nET\n"
    ).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def _extract_page1(pdf_path: Path) -> tuple[bytes, str]:
    executable = _pdftotext_path()
    version_result = subprocess.run([executable, "-v"], capture_output=True, check=False)
    assert version_result.returncode == 0
    version = (version_result.stdout + version_result.stderr).decode("utf-8").splitlines()[0]
    extraction = subprocess.run(
        [executable, "-f", "1", "-l", "1", "-layout", str(pdf_path), "-"],
        capture_output=True,
        check=False,
    )
    assert extraction.returncode == 0, extraction.stderr.decode("utf-8", errors="replace")
    return extraction.stdout, version


def _write_jsonl(path: Path, rows) -> bytes:
    data = b"".join(_json_bytes(row) + b"\r\n" for row in rows)
    path.write_bytes(data)
    return data


def _task4_page1_record(proof, text, text_bytes, version, index):
    return {
        "schema_version": "acl_metadata_page1_witness_input.v1",
        "item_id": proof["item_id"],
        "pdf": {"path": proof["pdf_path"], "sha256": proof["pdf_sha256"]},
        "pdf_page1": {
            "text": text,
            "text_sha256": _sha(text_bytes),
            "text_utf8_bytes": len(text_bytes),
            "pdftotext_version": version,
            "pdftotext_executable_sha256": "b" * 64,
        },
        "witness_author_id": f"builder-{index}",
    }


@pytest.fixture
def projection_inputs(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    ids = TASK4_IDS + [f"2026.findings-acl.{9000 + i}" for i in range(660)]
    source_dir = root / "sources"
    source_dir.mkdir()
    queue_path = source_dir / "queue.jsonl"
    manifest_path = source_dir / "manifest.jsonl"
    queue_rows = [{"item_id": item_id, "expected_title": "bib"} for item_id in ids]
    manifest_rows = [{"item_id": item_id, "title": "bib"} for item_id in ids]
    queue_data = _write_jsonl(queue_path, queue_rows)
    manifest_data = _write_jsonl(manifest_path, manifest_rows)
    html_path = source_dir / "official.html"
    html_path.write_bytes(b"<html><body>official ACL witness fixture</body></html>")
    pdf_paths = []
    extracted_pages = []
    pdftotext_version = None
    for index in range(1, 11):
        pdf_path = source_dir / f"selected-{index}.pdf"
        pdf_path.write_bytes(_minimal_pdf(index))
        extracted_page, pdftotext_version = _extract_page1(pdf_path)
        pdf_paths.append(pdf_path)
        extracted_pages.append(extracted_page)
    extra_pdf_path = source_dir / "unselected.pdf"
    extra_pdf_path.write_bytes(_minimal_pdf(1))
    cache_receipt = source_dir / "cache.receipt.json"
    cache_receipt.write_bytes(b'{"text_sha256":"fixture"}')
    cache_text = source_dir / "cache.txt"
    cache_text.write_bytes(b"cached source text")
    proofs = []
    for index, item_id in enumerate(ids, 1):
        proofs.append({
            "item_id": item_id,
            "queue_path": str(queue_path), "queue_file_sha256": _sha(queue_data),
            "queue_line_number": index,
            "queue_raw_line_sha256": _sha(_producer_raw_lines(queue_data)[index - 1]),
            "manifest_path": str(manifest_path), "manifest_file_sha256": _sha(manifest_data),
            "manifest_line_number": index,
            "manifest_raw_line_sha256": _sha(_producer_raw_lines(manifest_data)[index - 1]),
            "official_html_path": str(html_path), "official_html_sha256": _sha(html_path.read_bytes()),
            "pdf_path": str(pdf_paths[index - 1] if index <= 10 else extra_pdf_path),
            "pdf_sha256": _sha((pdf_paths[index - 1] if index <= 10 else extra_pdf_path).read_bytes()),
            "cache_receipt_path": str(cache_receipt), "cache_receipt_sha256": _sha(cache_receipt.read_bytes()),
            "cache_text_path": str(cache_text), "cache_text_sha256": _sha(cache_text.read_bytes()),
            "current_expected_title": "bib", "current_source_title": "bib",
            "official_html_witness": {
                "outcome": "UNIQUE_WITNESS", "title": "Paper 1: Exact ACL Title",
                "authors": ["Author 1 A", "Author 1 B"],
                "publication_section_witness": {
                    "text": "Findings of the Association for Computational Linguistics: ACL 2026", "year": 2026,
                },
            },
            "outcome": "UNIQUE_WITNESS", "overlay_is_approval": False, "source_status_mutation": False,
        })
    page1_rows = []
    reviews = []
    for index, proof in enumerate(proofs[:10], 1):
        page1_text = extracted_pages[index - 1].decode("utf-8")
        page1 = _task4_page1_record(
            proof, page1_text, extracted_pages[index - 1], pdftotext_version, index
        )
        page1_rows.append(page1)
        reviews.append({
            "item_id": proof["item_id"], "proof_row_sha256": _sha(_json_bytes(proof)),
            "pdf_sha256": proof["pdf_sha256"], "page1_text_sha256": page1["pdf_page1"]["text_sha256"],
            "decision": "SUPPORTED", "reviewer_id": f"reviewer-{index}",
            "reviewed_at_utc": "2026-09-28T01:02:03Z", "reason": "Independent native page-1 identity confirmed.",
            "native_pdf_identity_verified": True,
            "field_verdicts": {"title": "PASS", "ordered_authors": "PASS", "venue": "PASS", "year": "PASS"},
        })
    proof_path = root / "proof.jsonl"
    proof_data = _write_jsonl(proof_path, proofs)
    proof_receipt_path = root / "proof-receipt.json"
    proof_receipt = {
        "schema_version": "acl670_metadata_proof_only_run_receipt.v1",
        "scope": "670 frozen ACL Findings queue rows with expected_title and source_record.title equal to bib",
        "final_proof_only_run": {
            "rows_path": str(proof_path),
            "rows_sha256": _sha(proof_data),
        "row_count": 670,
        "unique_item_ids": 670,
        },
    }
    proof_receipt_data = _json_bytes(proof_receipt)
    proof_receipt_path.write_bytes(proof_receipt_data)
    page1_path = root / "page1.jsonl"
    page1_data = _write_jsonl(page1_path, page1_rows)
    reviews_path = root / "reviews.jsonl"
    reviews_data = _write_jsonl(reviews_path, reviews)
    selection_path = root / "frozen_real_item_ids.v1.txt"
    selection_path.write_text("".join(item_id + "\n" for item_id in TASK4_IDS), encoding="utf-8", newline="")
    input_paths = [proof_path, proof_receipt_path, page1_path, reviews_path]
    input_paths += [queue_path, manifest_path, html_path, *pdf_paths, extra_pdf_path, cache_receipt, cache_text, selection_path]
    snapshots = {path: _sha(path.read_bytes()) for path in input_paths}
    out = root / "runs" / "projection-1"
    BUILDER._PARENT_RECEIPT_RELATIVE_PATH = Path("proof-receipt.json")
    BUILDER._PARENT_RECEIPT_SHA256 = _sha(proof_receipt_data)
    BUILDER._PARENT_PROOF_RELATIVE_PATH = Path("proof.jsonl")
    BUILDER._PARENT_PROOF_SHA256 = _sha(proof_data)
    BUILDER._FROZEN_SUBSET_IDS = tuple(TASK4_IDS)
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
        "selection_path": selection_path,
        "out": out,
        "snapshots": snapshots,
    }


def _args(data, *, out=None, proof_sha=None, mode="full", selection_path=None, selection_sha=None,
          proof_receipt_path=None, expected_count=None):
    proof_bytes = data["proof_path"].read_bytes()
    page1_bytes = data["page1_path"].read_bytes()
    reviews_bytes = data["reviews_path"].read_bytes()
    receipt_bytes = data["proof_receipt_path"].read_bytes()
    args = [
        "--proof", str(data["proof_path"]),
        "--proof-sha256", proof_sha or _sha(proof_bytes),
        "--proof-receipt", str(proof_receipt_path or data["proof_receipt_path"]),
        "--proof-receipt-sha256", _sha(receipt_bytes),
        "--page1-witnesses", str(data["page1_path"]),
        "--page1-sha256", _sha(page1_bytes),
        "--reviews", str(data["reviews_path"]),
        "--reviews-sha256", _sha(reviews_bytes),
        "--corpus-root", str(data["root"]),
        "--output-dir", str(out or data["out"]),
        "--mode", mode,
    ]
    if mode == "pilot-subset":
        selection_bytes = Path(selection_path or data["selection_path"]).read_bytes()
        args += ["--selection-list", str(selection_path or data["selection_path"]),
                 "--selection-sha256", selection_sha or _sha(selection_bytes)]
    if expected_count is not None:
        args += ["--expected-count", str(expected_count)]
    return args


def _repin_test_parent(data):
    BUILDER._PARENT_PROOF_SHA256 = _sha(data["proof_path"].read_bytes())
    BUILDER._PARENT_RECEIPT_SHA256 = _sha(data["proof_receipt_path"].read_bytes())


def _run(data, **kwargs):
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = BUILDER.main(_args(data, **kwargs))
        except SystemExit as exc:
            code = int(exc.code or 0)
    return SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())


def _run_public_cli(data, **kwargs):
    env = os.environ.copy()
    src = str(WORKTREE / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [str(PYTHON), "-X", "utf8", str(CLI), *_args(data, **kwargs)],
        cwd=WORKTREE, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
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
    assert [row["item_id"] for row in projection] == sorted(data["ids"])
    assert [row["item_id"] for row in diagnostic] == sorted(data["ids"])
    assert len(projection) == len(diagnostic) == 670
    assert len({row["item_id"] for row in projection}) == 670
    assert {row["status"] for row in projection} == {"SUPPORTED", "METADATA_HOLD"}
    assert all(row["review_authentication"] == "EXTERNAL_QA_REQUIRED" for row in projection)
    assert all(row["downstream_card_use_approved"] is False for row in projection)
    missing_id = data["ids"][2]
    assert next(row for row in projection if row["item_id"] == missing_id)["hold_reason"] == "REVIEW_MISSING"
    held = next(row for row in projection if row["item_id"] == data["ids"][1])
    assert held["status"] == "METADATA_HOLD"
    assert all(field["approved"] is None for field in held["fields"].values())
    receipt = json.loads((data["out"] / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["counts"] == {"SUPPORTED": 1, "METADATA_HOLD": 669}
    assert receipt["run_mode"] == "FULL"
    assert receipt["parent_row_count"] == 670
    assert receipt["output_row_count"] == 670
    assert receipt["source_admission_approved"] is False
    assert receipt["human_approved"] is False
    assert receipt["graph_ingested"] is False
    assert receipt["review_authentication"] == "EXTERNAL_QA_REQUIRED"
    assert receipt["downstream_card_use_approved"] is False
    assert receipt["toolchain"]["pdftotext_version"].startswith("pdftotext version ")
    assert len(receipt["toolchain"]["pdftotext_sha256"]) == 64
    _assert_inputs_unchanged(data)


def test_projection_uses_frozen_producer_row_hashes_for_lf_manifest_and_crlf_queue(projection_inputs):
    data = projection_inputs
    queue_path = data["proofs"][0]["queue_path"]
    manifest_path = Path(data["proofs"][0]["manifest_path"])
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b"\r\n", b"\n"))
    manifest_data = manifest_path.read_bytes()
    manifest_lines = _producer_raw_lines(manifest_data)
    queue_data = Path(queue_path).read_bytes()
    queue_lines = _producer_raw_lines(queue_data)
    for index, proof in enumerate(data["proofs"]):
        proof["queue_raw_line_sha256"] = _sha(queue_lines[index])
        proof["manifest_file_sha256"] = _sha(manifest_data)
        proof["manifest_raw_line_sha256"] = _sha(manifest_lines[index])
    proof_data = _write_jsonl(data["proof_path"], data["proofs"])
    for index, review in enumerate(data["reviews"]):
        review["proof_row_sha256"] = _sha(_json_bytes(data["proofs"][index]))
    reviews_data = _write_jsonl(data["reviews_path"], data["reviews"])
    receipt = json.loads(data["proof_receipt_path"].read_text(encoding="utf-8"))
    receipt["final_proof_only_run"]["rows_sha256"] = _sha(proof_data)
    data["proof_receipt_path"].write_bytes(_json_bytes(receipt))
    _repin_test_parent(data)
    for path in (manifest_path, data["proof_path"], data["reviews_path"], data["proof_receipt_path"]):
        data["snapshots"][path] = _sha(path.read_bytes())

    result = _run(data)

    assert result.returncode == 0, result.stderr
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(projection) == 670
    assert sum(row["status"] == "SUPPORTED" for row in projection) == 10
    assert sum(row["status"] == "METADATA_HOLD" for row in projection) == 660
    _assert_inputs_unchanged(data)


def test_ten_id_subset_conserves_nine_supported_and_one_missing_review(projection_inputs):
    data = projection_inputs
    data["reviews"] = data["reviews"][:9]
    _write_jsonl(data["reviews_path"], data["reviews"])
    data["snapshots"][data["reviews_path"]] = _sha(data["reviews_path"].read_bytes())
    result = _run(data, mode="pilot-subset")

    assert result.returncode == 0, result.stderr
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    diagnostic = [json.loads(line) for line in (data["out"] / "acl_metadata_witness.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_ids = sorted(TASK4_IDS)
    assert [row["item_id"] for row in projection] == expected_ids
    assert [row["item_id"] for row in diagnostic] == expected_ids
    assert {row["status"] for row in projection} == {"SUPPORTED", "METADATA_HOLD"}
    assert sum(row["status"] == "SUPPORTED" for row in projection) == 9
    assert sum(row["status"] == "METADATA_HOLD" for row in projection) == 1
    held_id = TASK4_IDS[9]
    assert next(row for row in projection if row["item_id"] == held_id)["hold_reason"] == "REVIEW_MISSING"
    receipt = json.loads((data["out"] / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["run_mode"] == "PILOT_SUBSET"
    assert receipt["parent_row_count"] == 670
    assert receipt["selected_row_count"] == receipt["output_row_count"] == 10
    assert receipt["selected_ids"] == expected_ids
    assert receipt["selection_list_sha256"] == _sha(data["selection_path"].read_bytes())
    assert receipt["page1_adapter"]["protocol_version"] == "acl_task4_page1_normalization.v1"
    assert receipt["page1_adapter"]["protocol_sha256"] == _sha(
        (WORKTREE / "src" / "idea_factory" / "acl_task4_evidence_adapter.py").read_bytes()
    )
    assert receipt["admission_module_sha256"] == _sha(
        (WORKTREE / "src" / "idea_factory" / "acl_metadata_admission.py").read_bytes()
    )
    assert receipt["page1_adapter"]["raw_input_path"] == str(data["page1_path"])
    assert receipt["page1_adapter"]["raw_input_sha256"] == _sha(data["page1_path"].read_bytes())
    assert len(receipt["page1_adapter"]["raw_input_row_sha256"]) == 10
    assert receipt["source_admission_approved"] is False
    assert receipt["human_approved"] is False
    assert receipt["graph_ingested"] is False
    assert receipt["downstream_card_use_approved"] is False
    _assert_inputs_unchanged(data)


@pytest.mark.parametrize("mutation, message", [
    ("wrong", "frozen"),
    ("duplicate", "duplicate"),
    ("missing", "ten"),
    ("additional", "ten"),
    ("noncanonical", "canonical"),
    ("test_only", "canonical"),
])
def test_invalid_selection_ids_fail_before_output_and_preserve_inputs(projection_inputs, mutation, message):
    data = projection_inputs
    ids = list(TASK4_IDS)
    if mutation == "wrong":
        ids[-1] = "2026.findings-acl.999999"
    elif mutation == "duplicate":
        ids[-1] = ids[0]
    elif mutation == "missing":
        ids.pop()
    elif mutation == "additional":
        ids.append("2026.findings-acl.999999")
    elif mutation == "noncanonical":
        ids[-1] = "2026.findings-acl.01320"
    else:
        ids[-1] = "2026.findings-acl.TEST_ONLY"
    data["selection_path"].write_text("".join(item_id + "\n" for item_id in ids), encoding="utf-8", newline="")
    data["snapshots"][data["selection_path"]] = _sha(data["selection_path"].read_bytes())
    result = _run(data, mode="pilot-subset")

    assert result.returncode != 0
    assert message in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_selection_list_outside_root_is_rejected_without_output(projection_inputs, tmp_path):
    data = projection_inputs
    outside = tmp_path / "outside.txt"
    outside.write_text("".join(item_id + "\n" for item_id in TASK4_IDS), encoding="utf-8")
    result = _run(data, mode="pilot-subset", selection_path=outside)
    assert result.returncode != 0
    assert "inside" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_selection_list_reparse_path_is_rejected_without_output(projection_inputs):
    data = projection_inputs
    alias = data["root"] / "selection-alias.txt"
    try:
        alias.symlink_to(data["selection_path"])
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    result = _run(data, mode="pilot-subset", selection_path=alias)
    assert result.returncode != 0
    assert "reparse" in result.stderr.lower() or "symlink" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


@pytest.mark.parametrize("case", ["wrong_hash", "wrong_count", "ten_rows"])
def test_forged_or_cropped_parent_receipt_is_rejected_before_output(projection_inputs, case):
    data = projection_inputs
    receipt = json.loads(data["proof_receipt_path"].read_text(encoding="utf-8"))
    if case == "wrong_hash":
        receipt["final_proof_only_run"]["rows_sha256"] = "0" * 64
    elif case == "wrong_count":
        receipt["final_proof_only_run"]["row_count"] = 10
        receipt["final_proof_only_run"]["unique_item_ids"] = 10
    else:
        proof_bytes = _write_jsonl(data["proof_path"], data["proofs"][:10])
        receipt["final_proof_only_run"].update({"rows_sha256": _sha(proof_bytes), "row_count": 10, "unique_item_ids": 10})
        receipt["scope"] = "10 frozen ACL Findings queue rows with expected_title and source_record.title equal to bib"
        data["snapshots"][data["proof_path"]] = _sha(proof_bytes)
    data["proof_receipt_path"].write_bytes(_json_bytes(receipt))
    data["snapshots"][data["proof_receipt_path"]] = _sha(data["proof_receipt_path"].read_bytes())
    _repin_test_parent(data)
    result = _run(data)
    assert result.returncode != 0
    assert "receipt" in result.stderr.lower() or "670" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_selected_id_must_exist_in_parent_even_when_parent_has_670_unique_rows(projection_inputs):
    data = projection_inputs
    data["proofs"][0]["item_id"] = "2026.findings-acl.999998"
    proof_bytes = _write_jsonl(data["proof_path"], data["proofs"])
    receipt = json.loads(data["proof_receipt_path"].read_text(encoding="utf-8"))
    receipt["final_proof_only_run"]["rows_sha256"] = _sha(proof_bytes)
    data["proof_receipt_path"].write_bytes(_json_bytes(receipt))
    data["snapshots"][data["proof_path"]] = _sha(proof_bytes)
    data["snapshots"][data["proof_receipt_path"]] = _sha(data["proof_receipt_path"].read_bytes())
    _repin_test_parent(data)
    result = _run(data, mode="pilot-subset")
    assert result.returncode != 0
    assert "absent" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_child_receipt_cannot_be_used_as_parent_receipt(projection_inputs):
    data = projection_inputs
    child = data["root"] / "child-receipt.json"
    child.write_bytes(_json_bytes({"schema_version": "acl_metadata_projection_run_receipt.v1"}))
    data["snapshots"][child] = _sha(child.read_bytes())
    result = _run(data, proof_receipt_path=child)
    assert result.returncode != 0
    assert "authoritative" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


@pytest.mark.parametrize("evidence_kind", ["page1", "review"])
def test_evidence_id_outside_parent_is_rejected_without_output(projection_inputs, evidence_kind):
    data = projection_inputs
    if evidence_kind == "page1":
        rows = data["page1_rows"] + [dict(data["page1_rows"][0], item_id="2026.findings-acl.888888")]
        path = data["page1_path"]
    else:
        rows = data["reviews"] + [dict(data["reviews"][0], item_id="2026.findings-acl.888888")]
        path = data["reviews_path"]
    _write_jsonl(path, rows)
    data["snapshots"][path] = _sha(path.read_bytes())
    result = _run(data, mode="pilot-subset")
    assert result.returncode != 0
    assert "broader than proof scope" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


@pytest.mark.parametrize("mismatch", ["page1_pdf", "review_pdf", "review_proof", "review_page1"])
def test_selected_evidence_with_wrong_hash_binding_is_rejected_before_output(projection_inputs, mismatch):
    data = projection_inputs
    if mismatch == "page1_pdf":
        data["page1_rows"][0]["pdf"]["sha256"] = data["proofs"][1]["pdf_sha256"]
        path = data["page1_path"]
    elif mismatch == "review_pdf":
        data["reviews"][0]["pdf_sha256"] = data["proofs"][1]["pdf_sha256"]
        path = data["reviews_path"]
    elif mismatch == "review_proof":
        data["reviews"][0]["proof_row_sha256"] = "0" * 64
        path = data["reviews_path"]
    else:
        data["reviews"][0]["page1_text_sha256"] = "0" * 64
        path = data["reviews_path"]
    _write_jsonl(path, data["page1_rows"] if mismatch == "page1_pdf" else data["reviews"])
    data["snapshots"][path] = _sha(path.read_bytes())
    result = _run(data, mode="pilot-subset")
    assert result.returncode != 0
    assert "hash" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_135_author_conflict_hold_remains_visible_in_subset_fixture(projection_inputs):
    data = projection_inputs
    index = TASK4_IDS.index("2026.findings-acl.135")
    data["reviews"][index]["decision"] = "METADATA_HOLD"
    data["reviews"][index]["reason"] = "Independent reviewer found author-order conflict."
    data["reviews"][index]["field_verdicts"]["ordered_authors"] = "CONFLICT"
    _write_jsonl(data["reviews_path"], data["reviews"])
    data["snapshots"][data["reviews_path"]] = _sha(data["reviews_path"].read_bytes())
    result = _run(data, mode="pilot-subset")

    assert result.returncode == 0, result.stderr
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    held = next(row for row in projection if row["item_id"] == "2026.findings-acl.135")
    assert held["status"] == "METADATA_HOLD"
    assert held["hold_reason"] == "Independent reviewer found author-order conflict."
    assert len(projection) == 10
    assert all(field["approved"] is None for field in held["fields"].values())
    _assert_inputs_unchanged(data)


def test_missing_nested_page1_field_becomes_item_visible_hold(projection_inputs):
    data = projection_inputs
    del data["page1_rows"][0]["witness_author_id"]
    _write_jsonl(data["page1_path"], data["page1_rows"])
    data["snapshots"][data["page1_path"]] = _sha(data["page1_path"].read_bytes())
    result = _run(data, mode="pilot-subset")

    assert result.returncode == 0, result.stderr
    diagnostic = [json.loads(line) for line in (data["out"] / "acl_metadata_witness.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    held = next(row for row in projection if row["item_id"] == TASK4_IDS[0])
    raw = next(row for row in diagnostic if row["item_id"] == TASK4_IDS[0])
    assert len(projection) == len(diagnostic) == 10
    assert held["status"] == "METADATA_HOLD"
    assert held["hold_reason"] == "PAGE1_WITNESS_AUTHOR_MISSING"
    assert raw["page1_witness"] is None
    assert raw["page1_witness_raw_sha256"] == _sha(
        data["page1_path"].read_bytes().splitlines(keepends=True)[0]
    )
    _assert_inputs_unchanged(data)


def test_aggregate_task4_review_decision_cannot_substitute_for_exact_review_row(projection_inputs):
    data = projection_inputs
    aggregate = {
        "schema_version": "acl_metadata_independent_reviewer_decision.v1",
        "item_id": TASK4_IDS[0],
        "decision": "SUPPORTED_METADATA_EVIDENCE_ONLY",
        "overall": "metadata evidence reviewed",
    }
    _write_jsonl(data["reviews_path"], [aggregate])
    data["snapshots"][data["reviews_path"]] = _sha(data["reviews_path"].read_bytes())
    result = _run(data, mode="pilot-subset")

    assert result.returncode == 0, result.stderr
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    diagnostic = [json.loads(line) for line in (data["out"] / "acl_metadata_witness.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(projection) == len(diagnostic) == 10
    held = next(row for row in projection if row["item_id"] == TASK4_IDS[0])
    assert held["status"] == "METADATA_HOLD"
    assert held["hold_reason"] == "REVIEW_EXACT_RECORD_MISSING"
    assert all(field["approved"] is None for field in held["fields"].values())
    assert not any(row["item_id"] == TASK4_IDS[0] and row["status"] == "SUPPORTED" for row in projection)
    _assert_inputs_unchanged(data)


def test_duplicate_review_json_keys_are_rejected_before_any_output(projection_inputs):
    data = projection_inputs
    rows = data["reviews_path"].read_bytes().splitlines(keepends=True)
    rows[0] = rows[0].replace(b'"decision":"SUPPORTED"', b'"decision":"METADATA_HOLD","decision":"SUPPORTED"')
    data["reviews_path"].write_bytes(b"".join(rows))
    data["snapshots"][data["reviews_path"]] = _sha(data["reviews_path"].read_bytes())

    result = _run(data, mode="pilot-subset")

    assert result.returncode != 0
    assert "duplicate JSON key" in result.stderr
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_page1_record_without_item_id_fails_without_guessing_selected_identity(projection_inputs):
    data = projection_inputs
    del data["page1_rows"][0]["item_id"]
    _write_jsonl(data["page1_path"], data["page1_rows"])
    data["snapshots"][data["page1_path"]] = _sha(data["page1_path"].read_bytes())

    result = _run(data, mode="pilot-subset")

    assert result.returncode != 0
    assert "cannot safely assign evidence without guessing" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_stale_selection_hash_is_rejected_without_output(projection_inputs):
    data = projection_inputs
    result = _run(data, mode="pilot-subset", selection_sha="0" * 64)
    assert result.returncode != 0
    assert "selection list hash mismatch" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_arbitrary_expected_count_is_not_a_production_cli_mode(projection_inputs):
    data = projection_inputs
    result = _run(data, expected_count=3)
    assert result.returncode != 0
    assert "expected-count" in result.stderr.lower()
    assert not data["out"].exists()
    _assert_inputs_unchanged(data)


def test_production_trust_root_is_fixed_and_rejects_synthetic_parent(projection_inputs):
    data = projection_inputs
    assert _PRODUCTION_PARENT_PINS == (
        "runs/parallel24-20260927-1340/engineering/ACL670_WITNESS_FINAL_RUN_RECEIPT.json",
        "2cf9982c0f39c7d82098baaf7ed41c759b83f45fd46b61fd752c899a0a669205",
        "runs/parallel24-20260927-1340/engineering/acl670-metadata-witness-proofonly-v5-20260927/acl_metadata_witness.jsonl",
        "375b75a511042b5e617a69649814d4c599a9b5d6a411d8f076c13ac48796cb8a",
    )
    result = _run_public_cli(data)
    assert result.returncode != 0
    assert "authoritative frozen" in result.stderr.lower()
    assert not data["out"].exists()
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
    if duplicate_kind == "proof":
        _repin_test_parent(data)
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
    _repin_test_parent(data)
    result = _run(data)

    assert result.returncode != 0
    assert "queue" in result.stderr.lower()
    assert not data["out"].exists()
    assert not (data["out"] / "receipt.json").exists()
    _assert_inputs_unchanged(data)


def test_synthetic_unrelated_page1_text_cannot_be_supported_by_recomputed_hashes(projection_inputs):
    data = projection_inputs
    page1 = data["page1_rows"][0]
    page1["pdf_page1"]["text"] = "UNRELATED DOCUMENT BY UNKNOWN AUTHORS / NOT ACL 2026"
    page1["pdf_page1"]["text_sha256"] = _sha(page1["pdf_page1"]["text"].encode("utf-8"))
    page1["pdf_page1"]["text_utf8_bytes"] = len(page1["pdf_page1"]["text"].encode("utf-8"))
    review = data["reviews"][0]
    review["page1_text_sha256"] = page1["pdf_page1"]["text_sha256"]
    review["native_pdf_identity_verified"] = True
    review["field_verdicts"] = {"title": "PASS", "ordered_authors": "PASS", "venue": "PASS", "year": "PASS"}
    page1_data = _write_jsonl(data["page1_path"], data["page1_rows"])
    reviews_data = _write_jsonl(data["reviews_path"], data["reviews"])
    data["snapshots"][data["page1_path"]] = _sha(page1_data)
    data["snapshots"][data["reviews_path"]] = _sha(reviews_data)
    _repin_test_parent(data)
    result = _run(data)

    assert result.returncode == 0, result.stderr
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected = next(row for row in projection if row["item_id"] == data["ids"][0])
    assert rejected["status"] == "METADATA_HOLD"
    assert rejected["hold_reason"] == "PAGE1_EXTRACTION_MISMATCH"
    assert all(field["approved"] is None for field in rejected["fields"].values())
    _assert_inputs_unchanged(data)


def test_recorded_pdftotext_version_must_match_the_extractor(projection_inputs):
    data = projection_inputs
    data["page1_rows"][0]["pdf_page1"]["pdftotext_version"] = "pdftotext version 0.0.0-forged"
    page1_data = _write_jsonl(data["page1_path"], data["page1_rows"])
    data["snapshots"][data["page1_path"]] = _sha(page1_data)
    result = _run(data)

    assert result.returncode == 0, result.stderr
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected = next(row for row in projection if row["item_id"] == data["ids"][0])
    assert rejected["status"] == "METADATA_HOLD"
    assert rejected["hold_reason"] == "PDFTOTEXT_VERSION_MISMATCH"
    assert all(field["approved"] is None for field in rejected["fields"].values())
    _assert_inputs_unchanged(data)


def test_non_pdf_bytes_cannot_be_supported_even_when_all_declared_hashes_are_recomputed(projection_inputs):
    data = projection_inputs
    proof = data["proofs"][0]
    page1 = data["page1_rows"][0]
    review = data["reviews"][0]
    pdf_path = Path(proof["pdf_path"])
    pdf_path.write_bytes(b"not a PDF document")
    proof["pdf_sha256"] = _sha(pdf_path.read_bytes())
    page1["pdf"]["sha256"] = proof["pdf_sha256"]
    review["pdf_sha256"] = proof["pdf_sha256"]
    review["proof_row_sha256"] = _sha(_json_bytes(proof))
    proof_data = _write_jsonl(data["proof_path"], data["proofs"])
    receipt = json.loads(data["proof_receipt_path"].read_text(encoding="utf-8"))
    receipt["final_proof_only_run"]["rows_sha256"] = _sha(proof_data)
    data["proof_receipt_path"].write_bytes(_json_bytes(receipt))
    page1_data = _write_jsonl(data["page1_path"], data["page1_rows"])
    reviews_data = _write_jsonl(data["reviews_path"], data["reviews"])
    data["snapshots"][pdf_path] = _sha(pdf_path.read_bytes())
    data["snapshots"][data["proof_path"]] = _sha(proof_data)
    data["snapshots"][data["proof_receipt_path"]] = _sha(data["proof_receipt_path"].read_bytes())
    data["snapshots"][data["page1_path"]] = _sha(page1_data)
    data["snapshots"][data["reviews_path"]] = _sha(reviews_data)
    _repin_test_parent(data)
    result = _run(data)

    assert result.returncode == 0, result.stderr
    projection = [json.loads(line) for line in (data["out"] / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    rejected = next(row for row in projection if row["item_id"] == data["ids"][0])
    assert rejected["status"] == "METADATA_HOLD"
    assert rejected["hold_reason"] == "PAGE1_EXTRACTION_FAILED"
    assert all(field["approved"] is None for field in rejected["fields"].values())
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


def test_output_write_failure_leaves_no_partial_final_directory(projection_inputs, monkeypatch):
    data = projection_inputs
    calls = 0

    def fail_fsync(_fd):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected receipt fsync failure")

    monkeypatch.setattr(BUILDER.os, "fsync", fail_fsync)
    result = _run(data, mode="pilot-subset")

    assert result.returncode != 0
    assert "injected receipt fsync failure" in result.stderr
    assert calls == 3
    assert not data["out"].exists()
    staging = list(data["out"].parent.glob(f".{data['out'].name}.staging-*")) if data["out"].parent.exists() else []
    assert staging == []
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
