"""Read-only, proof-only witnesses for exact ACL Anthology metadata rows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from html.parser import HTMLParser
from pathlib import Path


PARSER_VERSION = "acl_html_record_witness.v2"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def _normal_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class _RecordHTMLParser(HTMLParser):
    """Capture anchors with nearest paper-card identity and heading witnesses."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, int | None]] = []
        self._next_card = 0
        self._anchor: dict | None = None
        self.anchors: list[dict] = []
        self._heading: dict | None = None
        self.headings: list[str] = []

    def _current_card(self) -> int | None:
        for _, card in reversed(self._stack):
            if card is not None:
                return card
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key.lower(): value or "" for key, value in attrs}
        card = None
        if tag.lower() == "div" and "d-sm-flex" in attr.get("class", "").split():
            self._next_card += 1
            card = self._next_card
        self._stack.append((tag.lower(), card))
        if tag.lower() == "a":
            self._anchor = {"href": attr.get("href", ""), "card": self._current_card(), "text": []}
        if tag.lower() in {"h1", "h2", "h3"}:
            self._heading = {"tag": tag.lower(), "text": []}

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._anchor is not None:
            self._anchor["text"].append(data)
        if self._heading is not None:
            self._heading["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._anchor is not None:
            self._anchor["text"] = _normal_text("".join(self._anchor["text"]))
            self.anchors.append(self._anchor)
            self._anchor = None
        if tag in {"h1", "h2", "h3"} and self._heading is not None:
            heading = _normal_text("".join(self._heading["text"]))
            if heading:
                self.headings.append(heading)
            self._heading = None
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                break


def _parse_html(html_text: str) -> _RecordHTMLParser:
    parser = _RecordHTMLParser()
    parser.feed(html_text)
    parser.close()
    return parser


def _witness_from_parser(parser: _RecordHTMLParser, anthology_id: str) -> dict:
    if not re.fullmatch(r"20\d{2}\.findings-acl\.\d+", anthology_id):
        raise ValueError("invalid canonical ACL Findings ID")
    exact_href = f"/{anthology_id}/"
    matching = [a for a in parser.anchors if a["href"] == exact_href]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one exact canonical permalink {exact_href!r}; found {len(matching)}")
    anchor = matching[0]
    if anchor["card"] is None:
        raise ValueError("exact canonical permalink is not inside a paper card")
    title = anchor["text"]
    if not title:
        raise ValueError("exact canonical permalink has empty title text")
    authors = [
        a["text"] for a in parser.anchors
        if a["card"] == anchor["card"] and a["href"].startswith("/people/") and a["text"]
    ]
    year = int(anthology_id[:4])
    expected_section_href = f"#{year}findings-acl"
    section_matches = [a for a in parser.anchors if a["href"] == expected_section_href]
    if len(section_matches) != 1:
        raise ValueError(f"expected exactly one Findings section witness {expected_section_href!r}")
    expected_section_text = f"Findings of the Association for Computational Linguistics: ACL {year}"
    if section_matches[0]["text"] != expected_section_text:
        raise ValueError("Findings section heading text/year does not match the source ID")
    section_witness = {
        "href": expected_section_href,
        "text": section_matches[0]["text"],
        "year": year,
    }
    return {
        "anthology_id": anthology_id,
        "canonical_href": exact_href,
        "title": title,
        "title_sha256": _sha256(title.encode("utf-8")),
        "authors": authors,
        "authors_sha256": _canonical_json_sha256(authors),
        "publication_section_witness": section_witness,
        "publication_section_witness_sha256": _canonical_json_sha256(section_witness),
        "witness_kind": "SAVED_OFFICIAL_HTML_METADATA_ONLY",
        "parser_version": PARSER_VERSION,
        "outcome": "UNIQUE_WITNESS",
    }


def extract_official_record_witness(html_text: str, anthology_id: str) -> dict:
    """Extract one exact permalink's text and same-card author links; no joins."""
    return _witness_from_parser(_parse_html(html_text), anthology_id)


def validate_source_binding(queue_row: dict, manifest_row: dict, pdf_bytes: bytes, receipt: dict) -> dict:
    """Fail closed unless queue, manifest, receipt and current PDF bytes agree."""
    if not all(isinstance(item, dict) for item in (queue_row, manifest_row, receipt)):
        raise ValueError("queue, manifest, and cache receipt must be JSON objects")
    item_id = queue_row.get("item_id")
    if not isinstance(item_id, str) or not re.fullmatch(r"20\d{2}\.findings-acl\.\d+", item_id):
        raise ValueError("queue item_id is not a canonical ACL Findings ID")
    source = queue_row.get("source_record")
    if not isinstance(source, dict):
        raise ValueError("queue source_record must be an object")
    expected_url = f"https://aclanthology.org/{item_id}/"
    expected_pdf_url = f"https://aclanthology.org/{item_id}.pdf"
    if queue_row.get("expected_title") != "bib" or manifest_row.get("title") != "bib":
        raise ValueError("row is outside the exact bib-title repair scope")
    if source.get("anthology_id") != item_id or manifest_row.get("anthology_id") != item_id:
        raise ValueError("queue, source_record, and manifest IDs differ")
    for row in (source, manifest_row):
        if row.get("url") != expected_url or row.get("pdf_url") != expected_pdf_url:
            raise ValueError("source URL is not the exact canonical Anthology URL")
    if source.get("local_file") != manifest_row.get("local_file"):
        raise ValueError("queue and manifest local PDF identities differ")
    if not isinstance(queue_row.get("pdf_path"), str) or not queue_row["pdf_path"]:
        raise ValueError("queue has no PDF path")
    actual_sha = _sha256(pdf_bytes)
    recorded_hashes = [
        queue_row.get("pdf_sha256"), source.get("sha256"), source.get("actual_sha256"),
        manifest_row.get("sha256"), receipt.get("pdf_sha256"),
    ]
    if any(not isinstance(value, str) or value != actual_sha for value in recorded_hashes):
        raise ValueError("current PDF bytes do not match every queue/manifest/cache hash")
    if receipt.get("anthology_id") != item_id or receipt.get("status") != "CACHE_READY":
        raise ValueError("cache receipt is not CACHE_READY for this exact ID")
    page_count = receipt.get("page_count")
    if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count <= 0:
        raise ValueError("cache receipt page_count is invalid")
    if receipt.get("page_count_matches_manifest") is not True:
        raise ValueError("cache receipt page count does not match manifest")
    if queue_row.get("page_count") != page_count or manifest_row.get("page_count") != page_count:
        raise ValueError("queue, manifest, and cache page counts differ")
    return {
        "item_id": item_id,
        "canonical_url": expected_url,
        "pdf_url": expected_pdf_url,
        "pdf_path": queue_row["pdf_path"],
        "pdf_sha256": actual_sha,
        "page_count": page_count,
        "source_metadata_sha256": _canonical_json_sha256(manifest_row),
        "source_status_mutation": False,
    }


def write_overlay_new(
    output_dir: str | Path,
    rows: list[dict],
    receipt: dict,
    *,
    corpus_root: str | Path,
) -> tuple[Path, Path]:
    """Create a new output directory and write the overlay/receipt exclusively."""
    output_dir = _safe_path(output_dir, corpus_root, must_exist=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if receipt.get("overlay_is_approval") is not False:
        raise ValueError("proof-only receipt must bind overlay_is_approval=false")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("overlay rows must be a list of JSON objects")
    ids = [row.get("item_id") for row in rows]
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("overlay item IDs must be present and unique")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "acl_metadata_witness.jsonl"
    receipt_path = output_dir / "receipt.json"
    rows_bytes = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    receipt_value = dict(receipt)
    receipt_value["rows_sha256"] = _sha256(rows_bytes)
    receipt_value["row_count"] = len(rows)
    with rows_path.open("xb") as stream:
        stream.write(rows_bytes)
    with receipt_path.open("xb") as stream:
        stream.write(json.dumps(receipt_value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    return rows_path, receipt_path


def _safe_path(path: str | Path, corpus_root: str | Path, *, must_exist: bool = True) -> Path:
    root = Path(os.path.abspath(str(corpus_root)))
    target = Path(os.path.abspath(str(path)))
    if not target.is_relative_to(root):
        raise ValueError(f"path is outside the bound corpus root: {target}")
    chain = [target, *target.parents]
    for candidate in chain:
        if candidate == root.parent:
            break
        if not candidate.exists() and not candidate.is_symlink():
            continue
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise ValueError(f"cannot inspect path component: {candidate}") from exc
        if getattr(info, "st_file_attributes", 0) & 0x400 or candidate.is_symlink():
            raise ValueError(f"symlink or reparse path is not allowed: {candidate}")
        if candidate == root:
            break
    if must_exist and not target.is_file():
        raise ValueError(f"expected a regular file: {target}")
    return target


def _read_jsonl(path: Path) -> tuple[list[tuple[int, bytes, dict]], bytes]:
    data = path.read_bytes()
    raw_lines = data.split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines.pop()
    records = []
    for line_number, raw in enumerate(raw_lines, start=1):
        json_bytes = raw[:-1] if raw.endswith(b"\r") else raw
        if b"\r" in json_bytes:
            raise ValueError(f"embedded carriage return in JSONL row: {path}:{line_number}")
        if not json_bytes.strip():
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        try:
            value = json.loads(json_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid UTF-8 JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        records.append((line_number, raw, value))
    return records, data


def verify_input_pins(
    *,
    pins_path: str | Path,
    corpus_root: str | Path,
    manifest_path: str | Path,
    queue_paths: list[str | Path],
    html_path: str | Path,
    expected_pins_sha256: str,
) -> dict:
    """Verify an exact input-pin sidecar against frozen authorities/current bytes."""
    pins_path = _safe_path(pins_path, corpus_root)
    pins_bytes = pins_path.read_bytes()
    try:
        pins = json.loads(pins_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input pin sidecar is not valid UTF-8 JSON") from exc
    actual_pins_sha = _sha256(pins_bytes)
    if not isinstance(expected_pins_sha256, str) or actual_pins_sha != expected_pins_sha256:
        raise ValueError("PM input-pin receipt hash mismatch")
    required = {
        "schema_version", "assignment_manifest", "inputs", "queues", "proof_only",
        "source_admission_approved", "human_approved", "graph_ingested",
        "original_download_manifest", "download_manifest_mirror",
        "saved_official_html", "saved_official_html_mirror", "verified_source_conservation",
        "captured_at_utc",
    }
    if not isinstance(pins, dict) or set(pins) != required:
        raise ValueError("PM input-pin receipt does not match the closed required schema")
    if pins.get("schema_version") != "research_opportunity_factory.pm_acl_authority_pins.v1":
        raise ValueError("unsupported input pin sidecar schema")
    if pins.get("proof_only") is not True or pins.get("source_admission_approved") is not False:
        raise ValueError("PM pin receipt must be proof-only and must not approve source admission")
    if pins.get("human_approved") is not False or pins.get("graph_ingested") is not False:
        raise ValueError("PM pin receipt must retain human/graph approval as false")
    pin_snapshots: dict[str, str] = {str(pins_path): actual_pins_sha}
    verified: dict[str, dict] = {}
    def verify_entry(name: str, entry: object) -> dict:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"pin entry {name} must have exactly path and sha256")
        path = _safe_path(entry["path"], corpus_root)
        actual_sha = _sha256(path.read_bytes())
        if not isinstance(entry["sha256"], str) or actual_sha != entry["sha256"]:
            raise ValueError(f"pinned input hash mismatch: {name}")
        pin_snapshots[str(path)] = actual_sha
        return {"path": str(path), "sha256": actual_sha}

    norm = lambda value: os.path.normcase(os.path.abspath(str(value)))
    verified["assignment_manifest"] = verify_entry("assignment_manifest", pins["assignment_manifest"])
    pin_inputs = pins["inputs"]
    if not isinstance(pin_inputs, dict) or set(pin_inputs) != {"acl_inventory", "acl_cache"}:
        raise ValueError("PM pins must include exactly ACL inventory and cache ledger")
    verified["acl_inventory"] = verify_entry("inputs.acl_inventory", pin_inputs["acl_inventory"])
    verified["acl_cache"] = verify_entry("inputs.acl_cache", pin_inputs["acl_cache"])
    verified["original_download_manifest"] = verify_entry("original_download_manifest", pins["original_download_manifest"])
    verified["download_manifest_mirror"] = verify_entry("download_manifest_mirror", pins["download_manifest_mirror"])
    verified["saved_official_html"] = verify_entry("saved_official_html", pins["saved_official_html"])
    verified["saved_official_html_mirror"] = verify_entry("saved_official_html_mirror", pins["saved_official_html_mirror"])
    if verified["original_download_manifest"]["sha256"] != verified["download_manifest_mirror"]["sha256"]:
        raise ValueError("download manifest and its mirror hash differ")
    if verified["saved_official_html"]["sha256"] != verified["saved_official_html_mirror"]["sha256"]:
        raise ValueError("saved official HTML and its mirror hash differ")
    if norm(verified["download_manifest_mirror"]["path"]) != norm(manifest_path):
        raise ValueError("metadata manifest path differs from its pinned path")
    if norm(verified["saved_official_html_mirror"]["path"]) != norm(html_path):
        raise ValueError("official HTML path differs from its pinned path")
    conservation = pins.get("verified_source_conservation")
    conservation_keys = {
        "original_download_manifest_rows", "source_inventory_rows", "all_original_fields_unchanged",
        "assigned_queue_rows", "queue_source_records_exactly_match_inventory",
    }
    if not isinstance(conservation, dict) or set(conservation) != conservation_keys:
        raise ValueError("PM source conservation receipt has unexpected shape")
    if conservation.get("all_original_fields_unchanged") is not True or conservation.get("queue_source_records_exactly_match_inventory") is not True:
        raise ValueError("PM source conservation flags are not literally true")
    for name in ("original_download_manifest_rows", "source_inventory_rows", "assigned_queue_rows"):
        value = conservation.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"PM source conservation count is invalid: {name}")
    if conservation["original_download_manifest_rows"] != conservation["source_inventory_rows"]:
        raise ValueError("PM original download manifest and inventory row counts differ")
    if norm(verified["original_download_manifest"]["path"]) != norm(manifest_path):
        original_rows, _ = _read_jsonl(Path(verified["original_download_manifest"]["path"]))
    else:
        original_rows, _ = _read_jsonl(Path(manifest_path))
    inventory_rows, _ = _read_jsonl(Path(verified["acl_inventory"]["path"]))
    if len(original_rows) != conservation["original_download_manifest_rows"] or len(inventory_rows) != conservation["source_inventory_rows"]:
        raise ValueError("PM source conservation counts do not match current manifest/inventory rows")
    queue_entries = pins.get("queues")
    if not isinstance(queue_entries, list) or any(
        not isinstance(item, dict) or set(item) != {"worker_id", "queue_path", "queue_sha256", "item_count"}
        for item in queue_entries
    ):
        raise ValueError("queues pins must be an exact list of worker/path/hash/count objects")
    if len(queue_entries) != len(queue_paths):
        raise ValueError("pinned queue file count differs from CLI queue file count")
    actual_queue_paths = {norm(path) for path in queue_paths}
    pinned_queue_paths: set[str] = set()
    verified_queues: dict[str, dict] = {}
    for entry in queue_entries:
        if not isinstance(entry["worker_id"], str) or not isinstance(entry["item_count"], int) or isinstance(entry["item_count"], bool) or entry["item_count"] < 1:
            raise ValueError("queue worker/count pin is invalid")
        path = _safe_path(entry["queue_path"], corpus_root)
        if path.parent.name != entry["worker_id"]:
            raise ValueError("queue worker ID does not match its path")
        key = norm(path)
        if key in pinned_queue_paths:
            raise ValueError("duplicate queue path in input pin sidecar")
        pinned_queue_paths.add(key)
        actual_sha = _sha256(path.read_bytes())
        if not isinstance(entry["queue_sha256"], str) or actual_sha != entry["queue_sha256"]:
            raise ValueError(f"pinned queue hash mismatch: {path}")
        queue_rows, _ = _read_jsonl(path)
        if len(queue_rows) != entry["item_count"]:
            raise ValueError(f"pinned queue item count mismatch: {path}")
        pin_snapshots[str(path)] = actual_sha
        verified_queues[str(path)] = {"path": str(path), "sha256": actual_sha, "item_count": len(queue_rows)}
    if pinned_queue_paths != actual_queue_paths:
        raise ValueError("CLI queue paths do not exactly equal the pinned queue set")
    if sum(item["item_count"] for item in verified_queues.values()) != conservation["assigned_queue_rows"]:
        raise ValueError("PM assigned queue row count does not match current queues")
    return {
        "pins_path": str(pins_path),
        "pins_sha256": actual_pins_sha,
        "verified_inputs": verified,
        "queue_files": verified_queues,
        "snapshots": pin_snapshots,
    }


def verify_input_snapshots(expected: dict[str, str], *, corpus_root: str | Path) -> None:
    """Rehash every admitted input immediately before writing derived output."""
    if not isinstance(expected, dict) or not expected:
        raise ValueError("input snapshots must be a non-empty path/hash mapping")
    for input_path, expected_hash in expected.items():
        path = _safe_path(input_path, corpus_root)
        current_hash = _sha256(path.read_bytes())
        if current_hash != expected_hash:
            raise ValueError(f"input changed while proof overlay was being built: {input_path}")


def _bind_snapshot(snapshots: dict[str, str], path: Path, observed_hash: str) -> None:
    key = str(path)
    expected_hash = snapshots.get(key)
    if expected_hash is not None and expected_hash != observed_hash:
        raise ValueError(f"input differs from its previously pinned snapshot: {path}")
    snapshots[key] = observed_hash


def build_acl_witness_overlay(
    *,
    manifest_path: str | Path,
    queue_paths: list[str | Path],
    html_path: str | Path,
    receipt_dir: str | Path,
    corpus_root: str | Path,
    output_dir: str | Path,
    input_pins_path: str | Path,
    expected_input_pins_sha256: str,
    expected_count: int = 670,
) -> dict:
    """Build a new proof-only overlay bound to all original inputs and bytes."""
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    output_dir = Path(os.path.abspath(str(output_dir)))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    pins = verify_input_pins(
        pins_path=input_pins_path,
        corpus_root=corpus_root,
        manifest_path=manifest_path,
        queue_paths=queue_paths,
        html_path=html_path,
        expected_pins_sha256=expected_input_pins_sha256,
    )
    manifest_path = _safe_path(manifest_path, corpus_root)
    html_path = _safe_path(html_path, corpus_root)
    receipt_dir = _safe_path(receipt_dir, corpus_root, must_exist=False)
    if not receipt_dir.is_dir():
        raise ValueError("receipt_dir must be an existing directory")
    queue_paths = [_safe_path(path, corpus_root) for path in queue_paths]
    if not queue_paths:
        raise ValueError("at least one queue file is required")
    manifest_rows, manifest_bytes = _read_jsonl(manifest_path)
    manifest_hash = _sha256(manifest_bytes)
    _bind_snapshot(pins["snapshots"], manifest_path, manifest_hash)
    manifest_by_id: dict[str, list[tuple[int, bytes, dict]]] = {}
    for row in manifest_rows:
        item_id = row[2].get("anthology_id")
        if isinstance(item_id, str):
            manifest_by_id.setdefault(item_id, []).append(row)
    html_bytes = html_path.read_bytes()
    html_hash = _sha256(html_bytes)
    _bind_snapshot(pins["snapshots"], html_path, html_hash)
    try:
        html_text = html_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("saved official HTML is not UTF-8") from exc
    html_parser = _parse_html(html_text)
    queue_records: list[tuple[Path, int, bytes, dict]] = []
    queue_file_pins: dict[str, dict] = {}
    input_snapshots: dict[str, str] = dict(pins["snapshots"])
    seen_ids: set[str] = set()
    for queue_path in queue_paths:
        rows, queue_bytes = _read_jsonl(queue_path)
        queue_sha = _sha256(queue_bytes)
        _bind_snapshot(input_snapshots, queue_path, queue_sha)
        queue_file_pins[str(queue_path)] = {"sha256": queue_sha, "row_count": len(rows)}
        for line_number, raw, row in rows:
            source = row.get("source_record")
            source_title = source.get("title") if isinstance(source, dict) else None
            if row.get("expected_title") != "bib" and source_title != "bib":
                continue
            if row.get("expected_title") != "bib" or source_title != "bib":
                raise ValueError(f"queue title fields disagree at {queue_path}:{line_number}")
            item_id = row.get("item_id")
            if not isinstance(item_id, str) or item_id in seen_ids:
                raise ValueError(f"missing or duplicate item_id at {queue_path}:{line_number}")
            seen_ids.add(item_id)
            queue_records.append((queue_path, line_number, raw, row))
    if len(queue_records) != expected_count:
        raise ValueError(f"expected {expected_count} bib-title rows, found {len(queue_records)}")
    entries = []
    for queue_path, queue_line_number, queue_raw, queue_row in queue_records:
        item_id = queue_row["item_id"]
        candidates = manifest_by_id.get(item_id, [])
        if len(candidates) != 1:
            raise ValueError(f"expected one manifest row for {item_id}; found {len(candidates)}")
        manifest_line_number, manifest_raw, manifest_row = candidates[0]
        receipt_path = _safe_path(receipt_dir / f"{item_id}.receipt.json", corpus_root)
        receipt_bytes = receipt_path.read_bytes()
        try:
            cache_receipt = json.loads(receipt_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid cache receipt for {item_id}") from exc
        if not isinstance(cache_receipt, dict):
            raise ValueError(f"cache receipt must be an object for {item_id}")
        source_local_file = manifest_row.get("local_file")
        if not isinstance(source_local_file, str) or not source_local_file or Path(source_local_file).is_absolute() or ".." in Path(source_local_file).parts:
            raise ValueError(f"manifest local_file is not a safe relative path for {item_id}")
        expected_pdf_path = _safe_path(Path(corpus_root) / source_local_file, corpus_root)
        pdf_path = _safe_path(queue_row.get("pdf_path", ""), corpus_root)
        if expected_pdf_path.name != f"{item_id}__bib.pdf" or pdf_path.name != expected_pdf_path.name:
            raise ValueError(f"PDF does not use the exact canonical local_file path for {item_id}")
        try:
            same_pdf = os.path.samefile(expected_pdf_path, pdf_path)
        except OSError as exc:
            raise ValueError(f"cannot verify canonical local_file path for {item_id}") from exc
        if not same_pdf:
            raise ValueError(f"PDF path is not the canonical local_file path for {item_id}")
        pdf_bytes = pdf_path.read_bytes()
        cache_path_raw = queue_row.get("cache_path")
        receipt_text_path_raw = cache_receipt.get("text_path")
        if not isinstance(cache_path_raw, str) or not isinstance(receipt_text_path_raw, str):
            raise ValueError(f"queue/receipt cache text path is missing for {item_id}")
        cache_path = _safe_path(cache_path_raw, corpus_root)
        receipt_text_path = _safe_path(receipt_text_path_raw, corpus_root)
        if cache_path != receipt_text_path or not os.path.samefile(cache_path, receipt_text_path):
            raise ValueError(f"queue cache path is not the exact receipt cache path for {item_id}")
        cache_bytes = cache_path.read_bytes()
        cache_sha = _sha256(cache_bytes)
        if queue_row.get("cache_sha256") != cache_sha or cache_receipt.get("text_sha256") != cache_sha:
            raise ValueError(f"current cache text bytes do not match queue/receipt for {item_id}")
        source_binding = validate_source_binding(queue_row, manifest_row, pdf_bytes, cache_receipt)
        _bind_snapshot(input_snapshots, receipt_path, _sha256(receipt_bytes))
        _bind_snapshot(input_snapshots, pdf_path, _sha256(pdf_bytes))
        _bind_snapshot(input_snapshots, cache_path, cache_sha)
        official_witness = _witness_from_parser(html_parser, item_id)
        if official_witness["title"].casefold() == "bib":
            raise ValueError(f"official exact permalink title is still bib for {item_id}")
        entries.append({
            "item_id": item_id,
            "queue_path": str(queue_path),
            "queue_file_sha256": queue_file_pins[str(queue_path)]["sha256"],
            "queue_line_number": queue_line_number,
            "queue_raw_line_sha256": _sha256(queue_raw),
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": manifest_hash,
            "manifest_line_number": manifest_line_number,
            "manifest_raw_line_sha256": _sha256(manifest_raw),
            "official_html_path": str(html_path),
            "official_html_sha256": html_hash,
            "pdf_path": str(pdf_path),
            "pdf_sha256": source_binding["pdf_sha256"],
            "cache_receipt_path": str(receipt_path),
            "cache_receipt_sha256": _sha256(receipt_bytes),
            "cache_text_path": str(cache_path),
            "cache_text_sha256": cache_sha,
            "source_metadata_sha256": source_binding["source_metadata_sha256"],
            "current_expected_title": queue_row["expected_title"],
            "current_source_title": manifest_row["title"],
            "official_html_witness": official_witness,
            "outcome": "UNIQUE_WITNESS",
            "parser_version": PARSER_VERSION,
            "overlay_is_approval": False,
            "source_status_mutation": False,
        })
    receipt = {
        "schema_version": "acl_metadata_proof_only_overlay.v1",
        "overlay_is_approval": False,
        "source_status_mutation": False,
        "metadata_projection_performed": False,
        "scope": "exact_queue_rows_with_expected_title_and_source_title_bib",
        "expected_count": expected_count,
        "actual_count": len(entries),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "html_path": str(html_path),
        "html_sha256": html_hash,
        "queue_files": queue_file_pins,
        "queue_file_count": len(queue_file_pins),
        "distinct_item_ids": len(seen_ids),
        "input_snapshot_hashes": input_snapshots,
        "input_pins_path": pins["pins_path"],
        "input_pins_sha256": pins["pins_sha256"],
        "verified_authority_pins": pins["verified_inputs"],
    }
    verify_input_snapshots(input_snapshots, corpus_root=corpus_root)
    rows_path, receipt_path = write_overlay_new(output_dir, entries, receipt, corpus_root=corpus_root)
    return {"rows_path": rows_path, "receipt_path": receipt_path, "row_count": len(entries)}
