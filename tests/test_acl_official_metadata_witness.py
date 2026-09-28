import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import idea_factory.acl_official_metadata_witness as witness_module

from idea_factory.acl_official_metadata_witness import (
    _safe_path,
    build_acl_witness_overlay,
    extract_official_record_witness,
    validate_source_binding,
    verify_input_snapshots,
    verify_input_pins,
    write_overlay_new,
)


PAPER_ID = "2026.findings-acl.1077"
PDF = b"%PDF-1.7\nfixture bytes\n"
CACHE_TEXT = b"fixture parsed cache\n"


def test_extracts_exact_permalink_with_inline_text_and_same_card_authors():
    html = f'''<h1>ACL 2026 landing page</h1>
    <a href="#2026findings-acl">Findings of the Association for Computational Linguistics: ACL 2026</a>
    <div class="d-sm-flex align-items-stretch">
      <a href="/{PAPER_ID}.bib">bib</a>
      <a href="/{PAPER_ID}/"><span>ATAAT</span>: Adaptive <em>Threat-Aware</em> Tuning</a>
      <a href="/people/first-author/">First Author</a> | <a href="/people/second-author/">Second Author</a>
      <a href="/2026.findings-acl.10770/">not this item</a>
    </div>
    <div class="d-sm-flex align-items-stretch">
      <a href="/2026.findings-acl.1078/">Adjacent paper</a>
      <a href="/people/wrong-person/">Wrong Person</a>
    </div>'''

    result = extract_official_record_witness(html, PAPER_ID)

    assert result["canonical_href"] == f"/{PAPER_ID}/"
    assert result["title"] == "ATAAT: Adaptive Threat-Aware Tuning"
    assert result["authors"] == ["First Author", "Second Author"]
    assert result["parser_version"]
    assert result["outcome"] == "UNIQUE_WITNESS"
    assert result["publication_section_witness"] == {
        "href": "#2026findings-acl",
        "text": "Findings of the Association for Computational Linguistics: ACL 2026",
        "year": 2026,
    }


@pytest.mark.parametrize(
    "href",
    [
        f"/{PAPER_ID}.bib",
        f"/{PAPER_ID}.pdf",
        f"/{PAPER_ID}/?download=1",
        f"/{PAPER_ID}/#abstract",
        f"https://evil.example/{PAPER_ID}/",
        f"/prefix-{PAPER_ID}/",
    ],
)
def test_rejects_noncanonical_or_nonunique_permalink(href):
    html = f'<div class="d-sm-flex"><a href="{href}">Wrong link</a></div>'

    with pytest.raises(ValueError, match="exact canonical permalink"):
        extract_official_record_witness(html, PAPER_ID)


def test_rejects_duplicate_exact_permalink():
    html = "".join(
        f'<div class="d-sm-flex"><a href="/{PAPER_ID}/">Paper {i}</a></div>'
        for i in range(2)
    )

    with pytest.raises(ValueError, match="exactly one"):
        extract_official_record_witness(html, PAPER_ID)


def _source_rows():
    pdf_sha = hashlib.sha256(PDF).hexdigest()
    queue = {
        "item_id": PAPER_ID,
        "expected_title": "bib",
        "expected_venue": "ACL Anthology event ACL 2026",
        "expected_year": 2026,
        "pdf_path": "paper.pdf",
        "pdf_sha256": pdf_sha,
        "page_count": 16,
        "cache_path": "cache.txt",
        "cache_sha256": hashlib.sha256(CACHE_TEXT).hexdigest(),
        "source_record": {
            "anthology_id": PAPER_ID,
            "title": "bib",
            "url": f"https://aclanthology.org/{PAPER_ID}/",
            "pdf_url": f"https://aclanthology.org/{PAPER_ID}.pdf",
            "local_file": "paper.pdf",
            "sha256": pdf_sha,
            "actual_sha256": pdf_sha,
        },
    }
    manifest = {
        "anthology_id": PAPER_ID,
        "title": "bib",
        "url": f"https://aclanthology.org/{PAPER_ID}/",
        "pdf_url": f"https://aclanthology.org/{PAPER_ID}.pdf",
        "local_file": "paper.pdf",
        "sha256": pdf_sha,
        "authors": [],
        "track": "Findings",
        "booktitle": "ACL Anthology event ACL 2026",
        "page_count": 16,
    }
    receipt = {
        "anthology_id": PAPER_ID,
        "pdf_sha256": pdf_sha,
        "status": "CACHE_READY",
        "page_count": 16,
        "manifest_page_count": 16,
        "page_count_matches_manifest": True,
        "text_path": "cache.txt",
        "text_sha256": hashlib.sha256(CACHE_TEXT).hexdigest(),
    }
    return queue, manifest, receipt


def _write_input_pins(root: Path, manifest_path: Path, queue_paths: list[Path], html_path: Path) -> Path:
    control_path = root / "pm-control.json"
    manifest_v2_path = root / "manifest-v2.json"
    download_manifest_path = root / "download-manifest.jsonl"
    inventory_path = root / "source-inventory.jsonl"
    cache_ledger_path = root / "cache-ledger.jsonl"
    original_html_path = root / "original-event.html"
    original_manifest_path = root / "original-manifest.jsonl"
    control_path.write_bytes(b'{"stage":"approved fixture"}\n')
    manifest_v2_path.write_bytes(b'{"run":"fixture"}\n')
    download_manifest_path.write_bytes(manifest_path.read_bytes())
    inventory_path.write_bytes(b'{"inventory":"fixture"}\n')
    cache_ledger_path.write_bytes(b'{"cache":"fixture"}\n')
    original_manifest_path.write_bytes(manifest_path.read_bytes())
    original_html_path.write_bytes(html_path.read_bytes())
    pin_entry = lambda path: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    value = {
        "schema_version": "research_opportunity_factory.pm_acl_authority_pins.v1",
        "captured_at_utc": "2026-09-27T00:00:00Z",
        "assignment_manifest": pin_entry(manifest_v2_path),
        "inputs": {"acl_inventory": pin_entry(inventory_path), "acl_cache": pin_entry(cache_ledger_path)},
        "queues": [
            {"worker_id": path.parent.name, "queue_path": str(path),
             "queue_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "item_count": len([line for line in path.read_bytes().split(bytes([10])) if line.strip()])}
            for path in queue_paths
        ],
        "proof_only": True,
        "source_admission_approved": False,
        "human_approved": False,
        "graph_ingested": False,
        "original_download_manifest": pin_entry(download_manifest_path),
        "download_manifest_mirror": pin_entry(manifest_path),
        "saved_official_html": pin_entry(original_html_path),
        "saved_official_html_mirror": pin_entry(html_path),
        "verified_source_conservation": {
            "original_download_manifest_rows": 1,
            "source_inventory_rows": 1,
            "all_original_fields_unchanged": True,
            "assigned_queue_rows": 1,
            "queue_source_records_exactly_match_inventory": True,
        },
    }
    pins_path = root / "input-pins.json"
    pins_path.write_text(json.dumps(value), encoding="utf-8")
    return pins_path


def test_source_binding_accepts_exact_ids_urls_and_current_pdf_bytes():
    queue, manifest, receipt = _source_rows()

    result = validate_source_binding(queue, manifest, PDF, receipt)

    assert result["item_id"] == PAPER_ID
    assert result["pdf_sha256"] == hashlib.sha256(PDF).hexdigest()
    assert result["source_status_mutation"] is False


@pytest.mark.parametrize("change", ["id", "url", "recorded_sha", "receipt_sha", "pdf_bytes"])
def test_source_binding_fails_closed_on_identity_url_or_hash_drift(change):
    queue, manifest, receipt = _source_rows()
    pdf = PDF
    if change == "id":
        queue["item_id"] = "2026.findings-acl.1078"
    elif change == "url":
        manifest["url"] += "?wrong=1"
    elif change == "recorded_sha":
        queue["pdf_sha256"] = "0" * 64
    elif change == "receipt_sha":
        receipt["pdf_sha256"] = "0" * 64
    elif change == "pdf_bytes":
        pdf = b"different bytes"

    with pytest.raises(ValueError):
        validate_source_binding(queue, manifest, pdf, receipt)


def test_overlay_writer_refuses_overwrite_and_keeps_existing_bytes(tmp_path: Path):
    out = tmp_path / "overlay"
    out.mkdir()
    sentinel = out / "existing.jsonl"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_overlay_new(out, [{"item_id": PAPER_ID}], {"overlay_is_approval": False}, corpus_root=tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_overlay_writer_rejects_output_outside_explicit_root(tmp_path: Path):
    with pytest.raises(ValueError, match="outside the bound corpus root"):
        write_overlay_new(
            tmp_path / "outside" / "overlay",
            [{"item_id": PAPER_ID}],
            {"overlay_is_approval": False},
            corpus_root=tmp_path / "bound",
        )


def test_output_guard_rejects_reparse_flag_independent_of_os_symlink_permission(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    reparse_dir = root / "junction-like"
    reparse_dir.mkdir(parents=True)
    original_lstat = Path.lstat

    def fake_lstat(path: Path):
        if path == reparse_dir:
            return SimpleNamespace(st_file_attributes=0x400)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="symlink or reparse"):
        _safe_path(reparse_dir / "output", root, must_exist=False)


def test_file_builder_binds_whole_inputs_and_writes_proof_only_row(tmp_path: Path):
    root = tmp_path / "root"
    queue_dir = root / "workers" / "acl-01"
    receipt_dir = root / "cache"
    pdf_dir = root / "papers"
    for directory in (queue_dir, receipt_dir, pdf_dir):
        directory.mkdir(parents=True)
    manifest_path = root / "manifest.jsonl"
    queue_path = queue_dir / "queue.jsonl"
    html_path = root / "anthology_event.html"
    pdf_path = pdf_dir / f"{PAPER_ID}__bib.pdf"
    cache_path = receipt_dir / f"{PAPER_ID}.txt"
    receipt_path = receipt_dir / f"{PAPER_ID}.receipt.json"
    queue, manifest, receipt = _source_rows()
    queue["pdf_path"] = str(pdf_path)
    manifest["local_file"] = f"papers/{PAPER_ID}__bib.pdf"
    queue["source_record"]["local_file"] = f"papers/{PAPER_ID}__bib.pdf"
    queue["cache_path"] = str(cache_path)
    queue["cache_sha256"] = hashlib.sha256(CACHE_TEXT).hexdigest()
    receipt["text_path"] = str(cache_path)
    receipt["text_sha256"] = hashlib.sha256(CACHE_TEXT).hexdigest()
    manifest["sha256"] = hashlib.sha256(PDF).hexdigest()
    queue["pdf_sha256"] = hashlib.sha256(PDF).hexdigest()
    queue["source_record"]["sha256"] = hashlib.sha256(PDF).hexdigest()
    queue["source_record"]["actual_sha256"] = hashlib.sha256(PDF).hexdigest()
    receipt["pdf_sha256"] = hashlib.sha256(PDF).hexdigest()
    manifest["page_count"] = 16
    queue["page_count"] = 16
    manifest_path.write_bytes((json.dumps(manifest) + "\r\n").encode("utf-8"))
    queue_path.write_bytes((json.dumps(queue) + "\r\n").encode("utf-8"))
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    pdf_path.write_bytes(PDF)
    cache_path.write_bytes(CACHE_TEXT)
    html_path.write_text(
        f'<a href="#2026findings-acl">Findings of the Association for Computational Linguistics: ACL 2026</a>'
        f'<div class="d-sm-flex"><a href="/{PAPER_ID}/">'
        '<span>ATAAT</span>: Adaptive Tuning</a><a href="/people/a/">Author A</a></div>',
        encoding="utf-8",
    )
    out = root / "proof-overlay"
    pins_path = _write_input_pins(root, manifest_path, [queue_path], html_path)

    result = build_acl_witness_overlay(
        manifest_path=manifest_path,
        queue_paths=[queue_path],
        html_path=html_path,
        receipt_dir=receipt_dir,
        corpus_root=root,
        input_pins_path=pins_path,
        expected_input_pins_sha256=hashlib.sha256(pins_path.read_bytes()).hexdigest(),
        output_dir=out,
        expected_count=1,
    )

    rows = [json.loads(line) for line in result["rows_path"].read_text(encoding="utf-8").splitlines()]
    saved_receipt = json.loads(result["receipt_path"].read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["item_id"] == PAPER_ID
    assert rows[0]["official_html_witness"]["title"] == "ATAAT: Adaptive Tuning"
    assert rows[0]["official_html_witness"]["authors"] == ["Author A"]
    assert rows[0]["official_html_witness"]["outcome"] == "UNIQUE_WITNESS"
    assert rows[0]["official_html_witness"]["publication_section_witness"]["year"] == 2026
    assert rows[0]["parser_version"]
    assert rows[0]["cache_text_sha256"] == hashlib.sha256(CACHE_TEXT).hexdigest()
    assert rows[0]["overlay_is_approval"] is False
    assert "replacement_title" not in rows[0]
    assert saved_receipt["overlay_is_approval"] is False
    assert saved_receipt["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert saved_receipt["queue_files"][str(queue_path)]["sha256"] == hashlib.sha256(queue_path.read_bytes()).hexdigest()
    assert manifest_path.read_text(encoding="utf-8").find('"title": "bib"') >= 0


@pytest.mark.parametrize("drift", ["pdf", "cache"])
def test_file_builder_rejects_pdf_or_cache_hash_drift_before_writing(tmp_path: Path, drift: str):
    root = tmp_path / "root"
    (root / "workers").mkdir(parents=True)
    (root / "cache").mkdir()
    (root / "papers").mkdir()
    queue, manifest, receipt = _source_rows()
    queue_path = root / "workers" / "queue.jsonl"
    manifest_path = root / "manifest.jsonl"
    receipt_path = root / "cache" / f"{PAPER_ID}.receipt.json"
    html_path = root / "event.html"
    pdf_path = root / "papers" / f"{PAPER_ID}__bib.pdf"
    cache_path = root / "cache" / f"{PAPER_ID}.txt"
    manifest["local_file"] = f"papers/{PAPER_ID}__bib.pdf"
    queue["source_record"]["local_file"] = f"papers/{PAPER_ID}__bib.pdf"
    queue["cache_path"] = str(cache_path)
    receipt["text_path"] = str(cache_path)
    queue["pdf_path"] = str(pdf_path)
    queue_path.write_bytes((json.dumps(queue) + "\r\n").encode("utf-8"))
    manifest_path.write_bytes((json.dumps(manifest) + "\r\n").encode("utf-8"))
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    pdf_path.write_bytes(b"changed bytes" if drift == "pdf" else PDF)
    cache_path.write_bytes(b"changed cache" if drift == "cache" else CACHE_TEXT)
    html_path.write_text(f'<div class="d-sm-flex"><a href="/{PAPER_ID}/">Title</a></div>', encoding="utf-8")
    out = root / "must-not-exist"
    pins_path = _write_input_pins(root, manifest_path, [queue_path], html_path)

    with pytest.raises(ValueError, match="PDF bytes|cache text bytes"):
        build_acl_witness_overlay(
            manifest_path=manifest_path,
            queue_paths=[queue_path],
            html_path=html_path,
            receipt_dir=root / "cache",
            corpus_root=root,
            input_pins_path=pins_path,
            expected_input_pins_sha256=hashlib.sha256(pins_path.read_bytes()).hexdigest(),
            output_dir=out,
            expected_count=1,
        )
    assert not out.exists()


def test_file_builder_rejects_pdf_alias_path_even_when_bytes_match(tmp_path: Path):
    root = tmp_path / "root"
    for name in ("workers", "cache", "papers"):
        (root / name).mkdir(parents=True)
    queue, manifest, receipt = _source_rows()
    pdf_path = root / "papers" / "other-name.pdf"
    manifest["local_file"] = f"papers/{PAPER_ID}__bib.pdf"
    queue["source_record"]["local_file"] = f"papers/{PAPER_ID}__bib.pdf"
    (root / "papers" / f"{PAPER_ID}__bib.pdf").write_bytes(PDF)
    pdf_path.write_bytes(PDF)
    queue["pdf_path"] = str(pdf_path)
    queue_path = root / "workers" / "queue.jsonl"
    manifest_path = root / "manifest.jsonl"
    queue_path.write_bytes((json.dumps(queue) + "\n").encode("utf-8"))
    manifest_path.write_bytes((json.dumps(manifest) + "\n").encode("utf-8"))
    (root / "cache" / f"{PAPER_ID}.receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (root / "event.html").write_text(
        f'<a href="#2026findings-acl">Findings of the Association for Computational Linguistics: ACL 2026</a>'
        f'<div class="d-sm-flex"><a href="/{PAPER_ID}/">Title</a></div>',
        encoding="utf-8",
    )
    pins_path = _write_input_pins(root, manifest_path, [queue_path], root / "event.html")
    with pytest.raises(ValueError, match="canonical local_file path"):
        build_acl_witness_overlay(
            manifest_path=manifest_path,
            queue_paths=[queue_path],
            html_path=root / "event.html",
            receipt_dir=root / "cache",
            corpus_root=root,
            input_pins_path=pins_path,
            expected_input_pins_sha256=hashlib.sha256(pins_path.read_bytes()).hexdigest(),
            output_dir=root / "must-not-exist",
            expected_count=1,
        )


def test_input_pins_reject_changed_original_manifest_or_wrong_queue_set(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    manifest_path = root / "manifest.jsonl"
    queue_path = root / "queue.jsonl"
    html_path = root / "event.html"
    for path, value in ((manifest_path, b"{}\n"), (queue_path, b"{}\n"), (html_path, b"<html></html>")):
        path.write_bytes(value)
    pins_path = _write_input_pins(root, manifest_path, [queue_path], html_path)
    manifest_path.write_bytes(b"{\"changed\":true}\n")

    with pytest.raises(ValueError, match="pinned input hash mismatch: download_manifest_mirror"):
        verify_input_pins(
            pins_path=pins_path,
            corpus_root=root,
            manifest_path=manifest_path,
            queue_paths=[queue_path],
            html_path=html_path,
            expected_pins_sha256=hashlib.sha256(pins_path.read_bytes()).hexdigest(),
        )


def test_input_pins_reject_wrong_pm_pin_receipt_sha(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    manifest_path = root / "manifest.jsonl"
    queue_path = root / "queue.jsonl"
    html_path = root / "event.html"
    for path, value in ((manifest_path, b"{}\n"), (queue_path, b"{}\n"), (html_path, b"<html></html>")):
        path.write_bytes(value)
    pins_path = _write_input_pins(root, manifest_path, [queue_path], html_path)

    with pytest.raises(ValueError, match="PM input-pin receipt hash mismatch"):
        verify_input_pins(
            pins_path=pins_path,
            corpus_root=root,
            manifest_path=manifest_path,
            queue_paths=[queue_path],
            html_path=html_path,
            expected_pins_sha256="0" * 64,
        )


def test_output_writer_rejects_reparse_symlink_ancestor_when_supported(tmp_path: Path):
    root = tmp_path / "root"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("test environment does not permit directory symlink creation")

    with pytest.raises(ValueError, match="symlink or reparse"):
        write_overlay_new(
            link / "overlay",
            [{"item_id": PAPER_ID}],
            {"overlay_is_approval": False},
            corpus_root=root,
        )


def test_input_snapshot_gate_detects_changes_after_initial_reads(tmp_path: Path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"before")
    expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    path.write_bytes(b"after")

    with pytest.raises(ValueError, match="input changed while proof overlay was being built"):
        verify_input_snapshots(expected, corpus_root=tmp_path)


def test_builder_rejects_manifest_html_or_queue_drift_after_pin_validation(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    queue_dir = root / "workers" / "acl-01"
    receipt_dir = root / "cache"
    pdf_dir = root / "papers"
    for directory in (queue_dir, receipt_dir, pdf_dir):
        directory.mkdir(parents=True)
    manifest_path = root / "manifest.jsonl"
    queue_path = queue_dir / "queue.jsonl"
    html_path = root / "anthology_event.html"
    pdf_path = pdf_dir / f"{PAPER_ID}__bib.pdf"
    cache_path = receipt_dir / f"{PAPER_ID}.txt"
    receipt_path = receipt_dir / f"{PAPER_ID}.receipt.json"
    queue, manifest, receipt = _source_rows()
    for row in (manifest, queue["source_record"]):
        row["local_file"] = f"papers/{PAPER_ID}__bib.pdf"
    queue["pdf_path"] = str(pdf_path)
    queue["cache_path"] = str(cache_path)
    queue["cache_sha256"] = hashlib.sha256(CACHE_TEXT).hexdigest()
    receipt["text_path"] = str(cache_path)
    receipt["text_sha256"] = hashlib.sha256(CACHE_TEXT).hexdigest()
    manifest_path.write_bytes((json.dumps(manifest) + "\r\n").encode("utf-8"))
    queue_path.write_bytes((json.dumps(queue) + "\r\n").encode("utf-8"))
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    pdf_path.write_bytes(PDF)
    cache_path.write_bytes(CACHE_TEXT)
    html_path.write_text(
        f'<a href="#2026findings-acl">Findings of the Association for Computational Linguistics: ACL 2026</a>'
        f'<div class="d-sm-flex"><a href="/{PAPER_ID}/">Title</a></div>',
        encoding="utf-8",
    )
    pins_path = _write_input_pins(root, manifest_path, [queue_path], html_path)
    original_pin_verifier = witness_module.verify_input_pins

    def mutate_after_pins(**kwargs):
        result = original_pin_verifier(**kwargs)
        html_path.write_text(html_path.read_text(encoding="utf-8") + "<!-- concurrent drift -->", encoding="utf-8")
        return result

    monkeypatch.setattr(witness_module, "verify_input_pins", mutate_after_pins)
    out = root / "must-not-exist"

    with pytest.raises(ValueError, match="pinned|input changed"):
        build_acl_witness_overlay(
            manifest_path=manifest_path,
            queue_paths=[queue_path],
            html_path=html_path,
            receipt_dir=receipt_dir,
            corpus_root=root,
            input_pins_path=pins_path,
            expected_input_pins_sha256=hashlib.sha256(pins_path.read_bytes()).hexdigest(),
            output_dir=out,
            expected_count=1,
        )
    assert not out.exists()
