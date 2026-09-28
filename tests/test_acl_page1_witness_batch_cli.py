"""Fail-closed checks for the evidence-only ACL page-1 batch builder."""

import hashlib
import importlib.util
import json
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(r"F:\LLM_Evoke")
ENGINEERING = ROOT / "runs/parallel24-20260927-1340/engineering"
SELECTION = ENGINEERING / "acl-metadata-priority640-20260928-v1/acl_metadata_priority640_selection.v1.json"
PROOF = ENGINEERING / "acl670-metadata-witness-proofonly-v5-20260927/acl_metadata_witness.jsonl"
RECEIPT = ENGINEERING / "ACL670_WITNESS_FINAL_RUN_RECEIPT.json"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_acl_page1_witness_batch.py"


def minimal_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def args(tmp_path, **changes):
    values = dict(selection=SELECTION, selection_sha256=digest(SELECTION), proof=PROOF,
                  proof_sha256=digest(PROOF), proof_receipt=RECEIPT,
                  proof_receipt_sha256=digest(RECEIPT), batch_number=5,
                  corpus_root=ROOT, output_dir=tmp_path / "batch5",
                  pdftotext=Path(r"C:\Users\Jony\tools\poppler24\poppler-24.08.0\Library\bin\pdftotext.exe"),
                  pdftoppm=Path(r"C:\Users\Jony\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe"))
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("acl_page1_witness_batch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch5_preflight_is_exact_ten_and_read_only(builder, tmp_path):
    originals = {path: digest(path) for path in (SELECTION, PROOF, RECEIPT)}
    selected = builder.validate_inputs(args(tmp_path))
    assert [row["item_id"] for row in selected] == json.loads(SELECTION.read_text())["batches"][1]["item_ids"]
    assert len(selected) == 10
    assert {path: digest(path) for path in originals} == originals
    assert not (tmp_path / "batch5").exists()


@pytest.mark.parametrize("batch", [4, 68, 0])
def test_out_of_scope_batch_rejected(builder, tmp_path, batch):
    with pytest.raises(ValueError, match="batch number"):
        builder.validate_inputs(args(tmp_path, batch_number=batch))


def test_altered_selection_hash_rejected(builder, tmp_path):
    changed = tmp_path / "selection.json"
    changed.write_bytes(SELECTION.read_bytes() + b" ")
    with pytest.raises(ValueError, match="selection SHA-256"):
        builder.validate_inputs(args(tmp_path, selection=changed))


def test_altered_proof_hash_rejected(builder, tmp_path):
    changed = tmp_path / "proof.jsonl"
    changed.write_bytes(PROOF.read_bytes() + b" ")
    with pytest.raises(ValueError, match="proof SHA-256"):
        builder.validate_inputs(args(tmp_path, proof=changed))


@pytest.mark.parametrize("field,source,label", [
    ("selection", SELECTION, "selection"),
    ("proof", PROOF, "proof"),
    ("proof_receipt", RECEIPT, "proof receipt"),
])
def test_self_consistently_rehashed_replacement_is_not_frozen_authority(builder, tmp_path, field, source, label):
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match=f"approved {label} SHA-256"):
        builder.validate_inputs(args(tmp_path, **{field: changed, field + "_sha256": digest(changed)}))


def test_duplicate_selected_id_rejected_after_authority_gate(builder, tmp_path, monkeypatch):
    changed = tmp_path / "selection.json"
    data = json.loads(SELECTION.read_text(encoding="utf-8"))
    data["batches"][1]["item_ids"][1] = data["batches"][1]["item_ids"][0]
    changed.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(builder, "APPROVED_SELECTION_SHA256", digest(changed))
    with pytest.raises(ValueError, match="duplicate|mismatch"):
        builder.validate_inputs(args(tmp_path, selection=changed, selection_sha256=digest(changed)))


def test_missing_selected_id_rejected_after_authority_gate(builder, tmp_path, monkeypatch):
    changed = tmp_path / "selection.json"
    data = json.loads(SELECTION.read_text(encoding="utf-8"))
    data["batches"][1]["item_ids"][0] = "2026.findings-acl.absent"
    changed.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(builder, "APPROVED_SELECTION_SHA256", digest(changed))
    with pytest.raises(ValueError, match="missing|mismatch"):
        builder.validate_inputs(args(tmp_path, selection=changed, selection_sha256=digest(changed)))


def test_pdf_hash_mismatch_rejected_without_output(builder, tmp_path, monkeypatch):
    original = builder.file_sha

    def changed_pdf(path):
        if str(path).endswith("2026.findings-acl.1611__bib.pdf"):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(builder, "file_sha", changed_pdf)
    with pytest.raises(ValueError, match="pdf_path.*SHA-256"):
        builder.validate_inputs(args(tmp_path))
    assert not (tmp_path / "batch5").exists()


def test_existing_output_directory_rejected(builder, tmp_path):
    (tmp_path / "batch5").mkdir()
    with pytest.raises(FileExistsError, match="output directory"):
        builder.validate_inputs(args(tmp_path))


def test_priority30_source_hash_mismatch_rejected(builder, tmp_path, monkeypatch):
    priority = json.loads(SELECTION.read_text(encoding="utf-8"))["source_priority30_selection_path"]
    original = builder.file_sha
    monkeypatch.setattr(builder, "file_sha", lambda path: "0" * 64 if str(path) == priority else original(path))
    with pytest.raises(ValueError, match="priority30 selection SHA-256"):
        builder.validate_inputs(args(tmp_path))


def test_symlink_output_ancestor_rejected(builder, tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink/reparse"):
        builder.validate_inputs(args(tmp_path, output_dir=link / "batch5"))


def test_receipt_hash_closes_upstream_priority_and_proof_receipts(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "_version", lambda executable, name: f"{name} version test")

    def produce(command, item_id):
        if command[-1] == "-":
            return f"Page one {item_id}\n".encode()
        Path(command[-1] + ".png").write_bytes(minimal_png())
        return b""

    monkeypatch.setattr(builder, "_run", produce)
    receipt_path = builder.build(args(tmp_path))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    priority = json.loads(SELECTION.read_text(encoding="utf-8"))["source_priority30_selection_path"]
    outer = json.loads(RECEIPT.read_text(encoding="utf-8"))
    inner = outer["final_proof_only_run"]["receipt_path"]
    pins = outer["input_authority"]["pm_pins_path"]
    for path in (priority, inner, pins, str(SCRIPT), str(args(tmp_path).pdftotext), str(args(tmp_path).pdftoppm)):
        assert receipt["input_source_file_hashes"][path] == digest(Path(path))
    assert receipt["counts"] == {"witness_records": 10, "text_files": 10, "png_files": 10, "reviewer_decisions": 0}
    assert len(receipt["outputs"]) == 21
    assert all(value is False for key, value in receipt.items() if key.endswith("approved") or key.endswith("written") or key.endswith("ingested") or key.endswith("performed") or key == "card_created")


@pytest.mark.parametrize("payload", [
    b"not a PNG despite exit zero",
    b"\x89PNG\r\n\x1a\n",
    minimal_png()[:8] + struct.pack(">I", 12) + minimal_png()[12:],
    minimal_png()[:16] + b"\x00\x00\x00\x00" + minimal_png()[20:],
    minimal_png()[:29] + b"\x00\x00\x00\x00" + minimal_png()[33:],
], ids=["wrong-signature", "missing-ihdr", "wrong-ihdr-length", "zero-width", "wrong-ihdr-crc"])
def test_renderer_success_with_invalid_png_has_no_success_receipt(builder, tmp_path, monkeypatch, payload):
    monkeypatch.setattr(builder, "_version", lambda executable, name: f"{name} version test")

    def emit_non_png(command, item_id):
        if command[-1] == "-":
            return f"Page one {item_id}\n".encode()
        Path(command[-1] + ".png").write_bytes(payload)
        return b""

    monkeypatch.setattr(builder, "_run", emit_non_png)
    with pytest.raises(ValueError, match="PNG"):
        builder.build(args(tmp_path))
    assert not (tmp_path / "batch5" / "receipt.json").exists()


def test_mid_build_pdf_drift_leaves_no_success_receipt(builder, tmp_path, monkeypatch):
    selected = json.loads(PROOF.read_text(encoding="utf-8").splitlines()[10])
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(b"%PDF-1.4 synthetic pinned bytes")
    selected["pdf_path"] = str(pdf)
    selected["pdf_sha256"] = digest(pdf)
    selected.update(_proof_line_number=11, _proof_line_sha256="0" * 64,
                    _proof_canonical_sha256="0" * 64)
    monkeypatch.setattr(builder, "validate_inputs", lambda options: [selected])
    monkeypatch.setattr(builder, "_version", lambda executable, name: f"{name} version test")

    def drift_during_render(command, item_id):
        if command[-1] == "-":
            return b"Page one synthetic text\n"
        Path(command[-1] + ".png").write_bytes(b"test png bytes")
        pdf.write_bytes(b"%PDF-1.4 changed during render")
        return b""

    monkeypatch.setattr(builder, "_run", drift_during_render)
    with pytest.raises(ValueError, match="PDF.*SHA-256|pdf_path.*SHA-256"):
        builder.build(args(tmp_path))
    assert not (tmp_path / "batch5" / "receipt.json").exists()
    assert pdf.read_bytes() == b"%PDF-1.4 changed during render"
