"""Contract tests for mechanical ACL witness/review batch custody."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest


CLI = Path(__file__).resolve().parents[1] / "scripts" / "assemble_acl_metadata_batch_inputs.py"
SCHEMA = "acl_metadata_batch_inputs.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return _sha(data)


def _lines(rows, ending=b"\r\n") -> bytes:
    return b"".join(json.dumps(row, separators=(",", ":")).encode() + ending for row in rows)


def _fixture(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    ids = [f"2026.findings-acl.{1000 + i}" for i in range(20)]
    proof = root / "proof.jsonl"
    proof_sha = _write(proof, _lines([{"item_id": item_id} for item_id in ids], b"\n"))
    witnesses = []
    reviews = []
    sources = [proof]
    for batch in (1, 2):
        batch_ids = ids[(batch - 1) * 10:batch * 10]
        witness = root / f"witness-{batch}.jsonl"
        witness_sha = _write(witness, _lines([{"item_id": item_id, "batch": batch} for item_id in batch_ids]))
        witnesses.append({"batch_number": batch, "path": str(witness), "sha256": witness_sha})
        sources.append(witness)
        review = root / f"review-{batch}.jsonl"
        review_sha = _write(review, _lines([{"item_id": batch_ids[0], "decision": "METADATA_HOLD"}]))
        qa = root / f"qa-{batch}.json"
        qa_sha = _write(qa, b'{"independent_qa":true}\n')
        reviews.append({"batch_number": batch, "path": str(review), "sha256": review_sha,
                        "qa_report_path": str(qa), "qa_report_sha256": qa_sha})
        sources.extend((review, qa))
    manifest = root / "manifest.json"
    value = {"schema_version": SCHEMA, "witnesses": witnesses, "reviews": reviews}
    _write(manifest, json.dumps(value, separators=(",", ":")).encode())
    sources.append(manifest)
    return root, proof, proof_sha, manifest, value, sources, ids


def _save_manifest(path, value):
    _write(path, json.dumps(value, separators=(",", ":")).encode())


def _run(root, proof, proof_sha, manifest, output=None):
    out = output or root / "assembled"
    result = subprocess.run(
        [sys.executable, str(CLI), "--manifest", str(manifest),
         "--manifest-sha256", _sha(manifest.read_bytes()), "--proof", str(proof),
         "--proof-sha256", proof_sha, "--corpus-root", str(root), "--output-dir", str(out),
         "--expected-batches", "2", "--expected-total", "20"],
        capture_output=True, text=True,
    )
    return result, out


def test_assembles_raw_lines_in_batch_order_and_writes_nonapproval_receipt(tmp_path):
    root, proof, proof_sha, manifest, value, sources, ids = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in sources}

    result, out = _run(root, proof, proof_sha, manifest)

    assert result.returncode == 0, result.stderr
    assert (out / "page1-witnesses.v1.jsonl").read_bytes() == b"".join(
        Path(row["path"]).read_bytes() for row in value["witnesses"]
    )
    assert (out / "reviews.v1.jsonl").read_bytes() == b"".join(
        Path(row["path"]).read_bytes() for row in value["reviews"]
    )
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["inputs"]["manifest"] == {"path": str(manifest), "sha256": _sha(before[manifest])}
    assert receipt["counts"]["witness_rows"] == 20
    assert receipt["counts"]["review_rows"] == 2
    assert receipt["proof_id_set_covered"] is True
    assert receipt["review_authentication"] == "EXTERNAL_QA_REQUIRED"
    assert receipt["source_admission_approved"] is False
    assert receipt["downstream_card_use_approved"] is False
    assert receipt["human_approved"] is False
    assert receipt["graph_ingested"] is False
    for name in ("page1-witnesses.v1.jsonl", "reviews.v1.jsonl"):
        data = (out / name).read_bytes()
        assert receipt["outputs"][name] == {"sha256": _sha(data), "bytes": len(data)}
    assert all(path.read_bytes() == data for path, data in before.items())


def test_swapped_but_valid_manifest_fails_its_external_pin(tmp_path):
    root, proof, proof_sha, manifest, value, sources, _ = _fixture(tmp_path)
    frozen_manifest_sha = _sha(manifest.read_bytes())
    value["reviews"] = []  # Still a valid assembly manifest, but not the frozen one.
    _save_manifest(manifest, value)
    before = {path: path.read_bytes() for path in sources}
    output = root / "assembled"
    spec = importlib.util.spec_from_file_location("acl_batch_assembler_for_manifest_pin", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = Namespace(manifest=str(manifest), manifest_sha256=frozen_manifest_sha,
                     proof=str(proof), proof_sha256=proof_sha, corpus_root=str(root),
                     output_dir=str(output), expected_batches=2, expected_total=20)

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        module.assemble(args)

    assert not output.exists()
    assert all(path.read_bytes() == data for path, data in before.items())


def test_cli_requires_manifest_sha256(tmp_path):
    root, proof, proof_sha, manifest, _, _, _ = _fixture(tmp_path)
    output = root / "assembled"
    result = subprocess.run(
        [sys.executable, str(CLI), "--manifest", str(manifest), "--proof", str(proof),
         "--proof-sha256", proof_sha, "--corpus-root", str(root),
         "--output-dir", str(output), "--expected-batches", "2", "--expected-total", "20"],
        capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert "--manifest-sha256" in result.stderr
    assert not output.exists()


def test_malformed_manifest_sha256_is_rejected_before_output(tmp_path):
    root, proof, proof_sha, manifest, _, _, _ = _fixture(tmp_path)
    output = root / "assembled"
    spec = importlib.util.spec_from_file_location("acl_batch_assembler_for_bad_manifest_pin", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = Namespace(manifest=str(manifest), manifest_sha256="ABC",
                     proof=str(proof), proof_sha256=proof_sha, corpus_root=str(root),
                     output_dir=str(output), expected_batches=2, expected_total=20)

    with pytest.raises(ValueError, match="manifest-sha256 must be a lowercase SHA-256 hash"):
        module.assemble(args)

    assert not output.exists()


@pytest.mark.parametrize("failure", ["write", "close"])
def test_receipt_io_failure_never_publishes_success_receipt(tmp_path, monkeypatch, failure):
    root, proof, proof_sha, manifest, _, sources, _ = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in sources}
    output = root / "assembled"
    spec = importlib.util.spec_from_file_location("acl_batch_assembler_for_fault_test", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_open = Path.open

    class FaultyReceiptFile:
        def __init__(self, real):
            self.real = real

        def __getattr__(self, name):
            return getattr(self.real, name)

        def __enter__(self):
            return self

        def write(self, data):
            if failure == "write":
                self.real.write(data[:1])
                raise OSError("injected receipt write failure")
            return self.real.write(data)

        def __exit__(self, exc_type, exc, tb):
            self.real.__exit__(exc_type, exc, tb)
            if failure == "close" and exc_type is None:
                raise OSError("injected receipt close failure")

    def open_with_fault(path, mode="r", *args, **kwargs):
        real = original_open(path, mode, *args, **kwargs)
        if path.name.startswith("receipt") and mode == "xb":
            return FaultyReceiptFile(real)
        return real

    monkeypatch.setattr(Path, "open", open_with_fault)
    args = Namespace(manifest=str(manifest), manifest_sha256=_sha(manifest.read_bytes()),
                     proof=str(proof), proof_sha256=proof_sha,
                     corpus_root=str(root), output_dir=str(output), expected_batches=2,
                     expected_total=20)

    with pytest.raises(OSError, match="injected receipt"):
        module.assemble(args)

    assert output.is_dir()
    assert not (output / "receipt.json").exists()
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("output_name", ["page1-witnesses.v1.jsonl", "reviews.v1.jsonl"])
def test_short_output_write_never_publishes_receipt(tmp_path, monkeypatch, output_name):
    root, proof, proof_sha, manifest, _, sources, _ = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in sources}
    output = root / "assembled"
    spec = importlib.util.spec_from_file_location("acl_batch_assembler_for_short_write", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_open = Path.open

    class ShortWriter:
        def __init__(self, real):
            self.real = real

        def __enter__(self):
            return self

        def write(self, data):
            self.real.write(data[:1])
            return 1

        def __exit__(self, exc_type, exc, tb):
            return self.real.__exit__(exc_type, exc, tb)

    def open_with_short_write(path, mode="r", *args, **kwargs):
        real = original_open(path, mode, *args, **kwargs)
        return ShortWriter(real) if path.name == output_name and mode == "xb" else real

    monkeypatch.setattr(Path, "open", open_with_short_write)
    args = Namespace(manifest=str(manifest), manifest_sha256=_sha(manifest.read_bytes()),
                     proof=str(proof), proof_sha256=proof_sha,
                     corpus_root=str(root), output_dir=str(output), expected_batches=2,
                     expected_total=20)

    with pytest.raises(OSError, match="short output write"):
        module.assemble(args)

    assert output.is_dir()
    assert not (output / "receipt.json").exists()
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("fault", ["changed_bytes", "read_error"])
def test_output_readback_failure_never_publishes_receipt(tmp_path, monkeypatch, fault):
    root, proof, proof_sha, manifest, _, sources, _ = _fixture(tmp_path)
    before = {path: path.read_bytes() for path in sources}
    output = root / "assembled"
    spec = importlib.util.spec_from_file_location("acl_batch_assembler_for_readback", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_read = Path.read_bytes

    def read_with_fault(path):
        data = original_read(path)
        if path.name == "page1-witnesses.v1.jsonl":
            if fault == "read_error":
                raise OSError("injected output readback failure")
            return b"X" + data[1:]
        return data

    monkeypatch.setattr(Path, "read_bytes", read_with_fault)
    args = Namespace(manifest=str(manifest), manifest_sha256=_sha(manifest.read_bytes()),
                     proof=str(proof), proof_sha256=proof_sha,
                     corpus_root=str(root), output_dir=str(output), expected_batches=2,
                     expected_total=20)

    with pytest.raises((OSError, ValueError), match="output readback"):
        module.assemble(args)

    assert output.is_dir()
    assert not (output / "receipt.json").exists()
    assert all(path.read_bytes() == data for path, data in before.items())


def test_missing_review_batch_is_not_approval(tmp_path):
    root, proof, proof_sha, manifest, value, _, _ = _fixture(tmp_path)
    value["reviews"].pop()
    _save_manifest(manifest, value)

    result, out = _run(root, proof, proof_sha, manifest)

    assert result.returncode == 0, result.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["counts"]["witness_rows"] == 20
    assert receipt["counts"]["review_rows"] == 1
    assert receipt["counts"]["review_batches"] == 1
    assert receipt["human_approved"] is False


def test_no_review_batches_still_assembles_all_witnesses(tmp_path):
    root, proof, proof_sha, manifest, value, _, _ = _fixture(tmp_path)
    value["reviews"] = []
    _save_manifest(manifest, value)

    result, out = _run(root, proof, proof_sha, manifest)

    assert result.returncode == 0, result.stderr
    assert (out / "reviews.v1.jsonl").read_bytes() == b""
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["counts"]["witness_rows"] == 20
    assert receipt["counts"]["review_rows"] == 0
    assert receipt["review_authentication"] == "EXTERNAL_QA_REQUIRED"


@pytest.mark.parametrize("mutation,expected", [
    ("duplicate_witness_id", "duplicate"),
    ("missing_proof_id", "proof"),
    ("bad_sha", "hash mismatch"),
    ("bad_qa_sha", "hash mismatch"),
    ("review_cross_batch", "witness batch"),
    ("duplicate_review_id", "duplicate"),
    ("duplicate_review_across_batches", "duplicate"),
    ("duplicate_witness_across_batches", "duplicate"),
    ("missing_batch", "missing"),
    ("duplicate_batch", "duplicate"),
    ("unexpected_batch", "unexpected"),
    ("malformed_jsonl", "JSONL"),
    ("unterminated_jsonl", "newline"),
    ("unknown_manifest_key", "keys"),
    ("unknown_entry_key", "keys"),
    ("outside_root", "corpus root"),
])
def test_rejects_invalid_batch_inputs_without_touching_sources(tmp_path, mutation, expected):
    root, proof, proof_sha, manifest, value, sources, ids = _fixture(tmp_path)
    first = Path(value["witnesses"][0]["path"])
    second = Path(value["witnesses"][1]["path"])
    review = Path(value["reviews"][0]["path"])
    if mutation in {"duplicate_witness_id", "missing_proof_id"}:
        rows = [json.loads(line) for line in first.read_bytes().splitlines()]
        rows[1]["item_id"] = rows[0]["item_id"] if mutation == "duplicate_witness_id" else "2026.findings-acl.9999"
        value["witnesses"][0]["sha256"] = _write(first, _lines(rows))
    elif mutation == "bad_sha":
        value["witnesses"][0]["sha256"] = "0" * 64
    elif mutation == "bad_qa_sha":
        value["reviews"][0]["qa_report_sha256"] = "0" * 64
    elif mutation in {"review_cross_batch", "duplicate_review_id"}:
        rows = [json.loads(line) for line in review.read_bytes().splitlines()]
        rows[0]["item_id"] = ids[10] if mutation == "review_cross_batch" else ids[0]
        if mutation == "duplicate_review_id":
            rows.append(dict(rows[0]))
        value["reviews"][0]["sha256"] = _write(review, _lines(rows))
    elif mutation == "missing_batch":
        value["witnesses"].pop()
    elif mutation == "duplicate_review_across_batches":
        rows = [json.loads(line) for line in Path(value["reviews"][1]["path"]).read_bytes().splitlines()]
        rows[0]["item_id"] = ids[0]
        value["reviews"][1]["sha256"] = _write(Path(value["reviews"][1]["path"]), _lines(rows))
    elif mutation == "duplicate_witness_across_batches":
        rows = [json.loads(line) for line in second.read_bytes().splitlines()]
        rows[0]["item_id"] = ids[0]
        value["witnesses"][1]["sha256"] = _write(second, _lines(rows))
    elif mutation == "duplicate_batch":
        value["witnesses"][1]["batch_number"] = 1
    elif mutation == "unexpected_batch":
        value["witnesses"][1]["batch_number"] = 3
    elif mutation == "malformed_jsonl":
        value["witnesses"][0]["sha256"] = _write(first, b'{bad}\n')
    elif mutation == "unterminated_jsonl":
        value["witnesses"][0]["sha256"] = _write(first, first.read_bytes().rstrip(b"\n"))
    elif mutation == "unknown_entry_key":
        value["witnesses"][0]["extra"] = True
    elif mutation == "outside_root":
        value["witnesses"][0]["path"] = str(tmp_path / "outside.jsonl")
    else:
        value["other"] = True
    _save_manifest(manifest, value)
    before = {path: path.read_bytes() for path in sources}

    result, out = _run(root, proof, proof_sha, manifest)

    assert result.returncode != 0
    assert expected.lower() in result.stderr.lower()
    assert not out.exists()
    assert all(path.read_bytes() == data for path, data in before.items())


def test_existing_output_directory_is_untouched(tmp_path):
    root, proof, proof_sha, manifest, _, sources, _ = _fixture(tmp_path)
    out = root / "assembled"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("untouched")
    before = {path: path.read_bytes() for path in sources}

    result, _ = _run(root, proof, proof_sha, manifest, out)

    assert result.returncode != 0
    assert marker.read_text() == "untouched"
    assert not (out / "receipt.json").exists()
    assert all(path.read_bytes() == data for path, data in before.items())


def test_symlinked_source_is_rejected(tmp_path):
    root, proof, proof_sha, manifest, value, _, _ = _fixture(tmp_path)
    link = root / "linked-witness.jsonl"
    try:
        link.symlink_to(Path(value["witnesses"][0]["path"]))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    value["witnesses"][0]["path"] = str(link)
    _save_manifest(manifest, value)

    result, out = _run(root, proof, proof_sha, manifest)

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower() or "reparse" in result.stderr.lower()
    assert not out.exists()


def test_noncanonical_parent_traversal_in_source_path_is_rejected(tmp_path):
    root, proof, proof_sha, manifest, value, _, _ = _fixture(tmp_path)
    witness = Path(value["witnesses"][0]["path"])
    value["witnesses"][0]["path"] = str(root / "unused" / ".." / witness.name)
    _save_manifest(manifest, value)

    result, out = _run(root, proof, proof_sha, manifest)

    assert result.returncode != 0
    assert "parent traversal" in result.stderr.lower()
    assert not out.exists()
