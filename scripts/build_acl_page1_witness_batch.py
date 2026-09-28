"""Build one evidence-only ACL Task 5 page-1 witness batch (numbers 5..67).

This performs mechanical extraction and rendering, not visual review, reviewer
decision, metadata admission, Card promotion, projection, or graph ingestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess


SCHEMA = "acl_metadata_page1_witness_input.v1"
APPROVED_SELECTION_SHA256 = "6d94272fe91d84983504da54a34f95027578ee2824788244e87f5cb312125557"
FROZEN_PROOF_SHA256 = "375b75a511042b5e617a69649814d4c599a9b5d6a411d8f076c13ac48796cb8a"
FROZEN_PROOF_RECEIPT_SHA256 = "2cf9982c0f39c7d82098baaf7ed41c759b83f45fd46b61fd752c899a0a669205"
_ID = re.compile(r"2026\.(?:findings-acl|acl)\.\d+\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_PINS = (
    ("pdf_path", "pdf_sha256"),
    ("queue_path", "queue_file_sha256"),
    ("manifest_path", "manifest_file_sha256"),
    ("official_html_path", "official_html_sha256"),
    ("cache_receipt_path", "cache_receipt_sha256"),
    ("cache_text_path", "cache_text_sha256"),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: pathlib.Path) -> str:
    return sha(path.read_bytes())


def canonical_sha(value: object) -> str:
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _checked_hash(path: pathlib.Path, expected: str, label: str) -> None:
    if not isinstance(expected, str) or not _SHA.fullmatch(expected) or file_sha(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")


def _inside(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    absolute = path.absolute()
    if not absolute.is_relative_to(root.absolute()):
        raise ValueError(f"path outside corpus root: {path}")
    return absolute


def _no_reparse_ancestors(path: pathlib.Path) -> None:
    for node in (path, *path.parents):
        if node.exists() or node.is_symlink():
            stat = node.lstat()
            if node.is_symlink() or stat.st_file_attributes & 0x400:
                raise ValueError(f"output path has symlink/reparse ancestor: {node}")


def _physical_lines(path: pathlib.Path) -> list[bytes]:
    lines = path.read_bytes().split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return lines


def _line(lines: list[bytes], number: int, label: str) -> bytes:
    if type(number) is not int or number < 1 or number > len(lines):
        raise ValueError(f"{label} physical line number invalid: {number}")
    raw = lines[number - 1]
    if not raw or b"\r" in raw.rstrip(b"\r"):
        raise ValueError(f"{label} physical line invalid: {number}")
    return raw


def _load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not object: {path}")
    return value


def _validate_selection(selection: dict, proof_rows: list[dict], batch_number: int) -> tuple[list[str], list[int]]:
    if type(batch_number) is not int or not 5 <= batch_number <= 67:
        raise ValueError("batch number must be 5..67 (batch 4 is the existing pilot)")
    if selection.get("schema_version") != "research_opportunity_factory.acl_metadata_remainder_selection.v1" or selection.get("authority") != "DETERMINISTIC_REMAINDER_SELECTION_ONLY_NOT_APPROVAL":
        raise ValueError("selection schema/authority mismatch")
    proof_ids = [row["item_id"] for row in proof_rows]
    priority_ids = selection.get("priority30_batch_ids")
    remaining = selection.get("remaining_selected_ids")
    batches = selection.get("batches")
    if not isinstance(priority_ids, list) or len(priority_ids) != 30 or len(set(priority_ids)) != 30:
        raise ValueError("priority30 IDs invalid")
    if not isinstance(remaining, list) or len(remaining) != 640 or len(set(remaining)) != 640:
        raise ValueError("remaining IDs duplicate/missing")
    if set(priority_ids) & set(remaining) or set(priority_ids) | set(remaining) != set(proof_ids):
        raise ValueError("selection/proof ID partition mismatch")
    if remaining != [item for item in proof_ids if item not in set(priority_ids)]:
        raise ValueError("selection/proof physical order mismatch")
    if not isinstance(batches, list) or len(batches) != 64:
        raise ValueError("selection batch count mismatch")
    details = selection.get("remaining_selected_rows")
    if not isinstance(details, list) or len(details) != 640:
        raise ValueError("selection proof row details missing")
    proof_lines = selection["_proof_lines"]
    for index, detail in enumerate(details):
        number = detail.get("proof_physical_line_number")
        if detail.get("item_id") != remaining[index] or _line(proof_lines, number, "proof") != proof_lines[proof_ids.index(remaining[index])]:
            raise ValueError("selection/proof row mismatch")
        if detail.get("proof_physical_line_sha256") != sha(proof_lines[number - 1]) or detail.get("proof_canonical_row_sha256") != canonical_sha(proof_rows[number - 1]):
            raise ValueError("selection/proof row hash mismatch")
    for offset, batch in enumerate(batches):
        part = details[offset * 10:(offset + 1) * 10]
        if batch.get("batch") != offset + 4 or batch.get("item_ids") != [r["item_id"] for r in part] or batch.get("proof_physical_line_numbers") != [r["proof_physical_line_number"] for r in part]:
            raise ValueError("selection batch duplicate/missing/mismatch")
    selected = batches[batch_number - 4]
    return selected["item_ids"], selected["proof_physical_line_numbers"]


def validate_inputs(args: argparse.Namespace) -> list[dict]:
    """Validate all source bytes and physical lines before creating output."""
    root = pathlib.Path(args.corpus_root).absolute()
    output = pathlib.Path(args.output_dir).absolute()
    _no_reparse_ancestors(output)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    # Reject an out-of-scope batch before touching heavyweight frozen inputs.
    if type(args.batch_number) is not int or not 5 <= args.batch_number <= 67:
        raise ValueError("batch number must be 5..67")
    for label, supplied, frozen in (("selection", args.selection_sha256, APPROVED_SELECTION_SHA256),
                                    ("proof", args.proof_sha256, FROZEN_PROOF_SHA256),
                                    ("proof receipt", args.proof_receipt_sha256, FROZEN_PROOF_RECEIPT_SHA256)):
        if supplied != frozen:
            raise ValueError(f"approved {label} SHA-256 mismatch")
    for path, expected, label in ((args.selection, args.selection_sha256, "selection"),
                                  (args.proof, args.proof_sha256, "proof"),
                                  (args.proof_receipt, args.proof_receipt_sha256, "proof receipt")):
        _checked_hash(pathlib.Path(path), expected, label)
    selection = _load_json(pathlib.Path(args.selection))
    priority_path = pathlib.Path(selection["source_priority30_selection_path"])
    _checked_hash(priority_path, selection["source_priority30_selection_sha256"], "priority30 selection")
    priority_selection = _load_json(priority_path)
    if [item for batch in priority_selection["batches"] for item in batch["item_ids"]] != selection.get("priority30_batch_ids"):
        raise ValueError("priority30 selection IDs mismatch")
    outer = _load_json(pathlib.Path(args.proof_receipt))
    run = outer.get("final_proof_only_run", {})
    if (run.get("rows_sha256") != args.proof_sha256 or pathlib.Path(run.get("rows_path", "")) != pathlib.Path(args.proof)
            or run.get("row_count") != 670 or run.get("unique_item_ids") != 670):
        raise ValueError("proof receipt/rows pin mismatch")
    if (selection.get("source_proof_sha256") != args.proof_sha256 or pathlib.Path(selection.get("source_proof_path", "")) != pathlib.Path(args.proof)
            or selection.get("source_proof_rows") != 670 or selection.get("source_proof_unique_ids") != 670):
        raise ValueError("selection/proof pin mismatch")
    inner = pathlib.Path(run["receipt_path"])
    _checked_hash(inner, run["receipt_sha256"], "proof run receipt")
    inner_data = _load_json(inner)
    if inner_data.get("rows_sha256") != args.proof_sha256 or inner_data.get("row_count") != 670:
        raise ValueError("proof run receipt content mismatch")
    authority = outer["input_authority"]
    _checked_hash(pathlib.Path(authority["pm_pins_path"]), authority["pm_pins_sha256"], "authority pins")
    if inner_data.get("input_pins_sha256") != authority["pm_pins_sha256"]:
        raise ValueError("authority pin cross-check mismatch")
    proof_lines = _physical_lines(pathlib.Path(args.proof))
    if len(proof_lines) != 670:
        raise ValueError("proof rows not exactly 670")
    proof_rows = [json.loads(_line(proof_lines, i, "proof")) for i in range(1, 671)]
    ids = [row.get("item_id") for row in proof_rows]
    if any(not isinstance(item, str) or not _ID.fullmatch(item) for item in ids) or len(set(ids)) != 670:
        raise ValueError("proof IDs duplicate/missing/malformed")
    selection["_proof_lines"] = proof_lines
    selected_ids, selected_lines = _validate_selection(selection, proof_rows, args.batch_number)
    selected = []
    file_cache: dict[pathlib.Path, bytes] = {}
    for item_id, number in zip(selected_ids, selected_lines):
        row = proof_rows[number - 1]
        if row["item_id"] != item_id or row.get("overlay_is_approval") is not False or row.get("source_status_mutation") is not False:
            raise ValueError(f"{item_id}: proof identity/approval contradiction")
        for path_key, hash_key in _SOURCE_PINS:
            source = _inside(pathlib.Path(row[path_key]), root)
            if source not in file_cache:
                file_cache[source] = source.read_bytes()
            if file_sha(source) != row[hash_key]:
                raise ValueError(f"{item_id}: {path_key} SHA-256 mismatch")
        for kind in ("queue", "manifest"):
            source = pathlib.Path(row[f"{kind}_path"]).absolute()
            raw_lines = file_cache[source].split(b"\n")
            raw = _line(raw_lines, row[f"{kind}_line_number"], kind)
            if sha(raw) != row[f"{kind}_raw_line_sha256"]:
                raise ValueError(f"{item_id}: {kind} physical line SHA-256 mismatch")
            physical = json.loads(raw)
            key = "item_id" if kind == "queue" else "anthology_id"
            if physical.get(key) != item_id or physical.get("sha256" if kind == "manifest" else "pdf_sha256") != row["pdf_sha256"]:
                raise ValueError(f"{item_id}: {kind} physical line identity mismatch")
        row = dict(row)
        row["_proof_line_sha256"] = sha(proof_lines[number - 1])
        row["_proof_canonical_sha256"] = canonical_sha(proof_rows[number - 1])
        row["_proof_line_number"] = number
        selected.append(row)
    return selected


def _version(executable: pathlib.Path, name: str) -> str:
    run = subprocess.run([str(executable), "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    text = (run.stdout + run.stderr).decode("utf-8", errors="replace").splitlines()
    if run.returncode or not text or not text[0].startswith(name + " version "):
        raise ValueError(f"{name} version unavailable")
    return text[0]


def _run(command: list[str], item_id: str) -> bytes:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"{item_id}: command failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}")
    return result.stdout


def build(args: argparse.Namespace) -> pathlib.Path:
    selected = validate_inputs(args)
    text_exe, render_exe = pathlib.Path(args.pdftotext), pathlib.Path(args.pdftoppm)
    text_version, render_version = _version(text_exe, "pdftotext"), _version(render_exe, "pdftoppm")
    selection_data = _load_json(pathlib.Path(args.selection))
    proof_receipt_data = _load_json(pathlib.Path(args.proof_receipt))
    output = pathlib.Path(args.output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / "page1-text").mkdir()
    (output / "page1-render").mkdir()
    rows = []
    output_hashes = {}
    source_hashes = {str(pathlib.Path(args.selection)): args.selection_sha256,
                     str(pathlib.Path(args.proof)): args.proof_sha256,
                     str(pathlib.Path(args.proof_receipt)): args.proof_receipt_sha256}
    for path in (selection_data["source_priority30_selection_path"],
                 proof_receipt_data["final_proof_only_run"]["receipt_path"],
                 proof_receipt_data["input_authority"]["pm_pins_path"],
                 str(pathlib.Path(__file__).absolute()), str(text_exe), str(render_exe)):
        source_hashes[path] = file_sha(pathlib.Path(path))
    for proof in selected:
        item_id = proof["item_id"]
        pdf = pathlib.Path(proof["pdf_path"])
        text_path = output / "page1-text" / f"{item_id}.page1.txt"
        png_path = output / "page1-render" / f"{item_id}.page1.png"
        text_bytes = _run([str(text_exe), "-f", "1", "-l", "1", "-layout", str(pdf), "-"], item_id)
        _checked_hash(pdf, proof["pdf_sha256"], f"{item_id}: PDF after extraction")
        text = text_bytes.decode("utf-8", errors="strict")
        if not text.strip():
            raise ValueError(f"{item_id}: empty page-1 text")
        text_path.write_bytes(text_bytes)
        _run([str(render_exe), "-f", "1", "-l", "1", "-singlefile", "-r", "120", "-png", str(pdf), str(png_path.with_suffix(""))], item_id)
        _checked_hash(pdf, proof["pdf_sha256"], f"{item_id}: PDF after render")
        if not png_path.is_file() or png_path.stat().st_size == 0:
            raise ValueError(f"{item_id}: missing/empty page-1 PNG")
        source_pins = {
            "proof": {"path": str(args.proof), "file_sha256": args.proof_sha256, "physical_line_number": proof["_proof_line_number"], "physical_line_sha256": proof["_proof_line_sha256"], "canonical_row_sha256": proof["_proof_canonical_sha256"]},
            "queue": {"path": proof["queue_path"], "file_sha256": proof["queue_file_sha256"], "physical_line_number": proof["queue_line_number"], "physical_line_sha256": proof["queue_raw_line_sha256"]},
            "manifest": {"path": proof["manifest_path"], "file_sha256": proof["manifest_file_sha256"], "physical_line_number": proof["manifest_line_number"], "physical_line_sha256": proof["manifest_raw_line_sha256"]},
            "official_html": {"path": proof["official_html_path"], "file_sha256": proof["official_html_sha256"], "witness": proof["official_html_witness"]},
            "cache_receipt": {"path": proof["cache_receipt_path"], "sha256": proof["cache_receipt_sha256"]},
            "cache_text": {"path": proof["cache_text_path"], "sha256": proof["cache_text_sha256"]},
        }
        for path_key, hash_key in _SOURCE_PINS:
            source_hashes[proof[path_key]] = proof[hash_key]
        text_hash, png_hash = sha(text_bytes), file_sha(png_path)
        output_hashes[str(text_path)] = text_hash
        output_hashes[str(png_path)] = png_hash
        rows.append({"schema_version": SCHEMA, "item_id": item_id,
                     "pdf": {"path": str(pdf), "sha256": proof["pdf_sha256"]},
                     "pdf_page1": {"text": text, "text_sha256": text_hash, "text_utf8_bytes": len(text_bytes),
                                   "pdftotext_version": text_version, "extraction_command": "pdftotext -f 1 -l 1 -layout <pinned-pdf-path> -",
                                   "text_evidence": {"path": str(text_path), "sha256": text_hash, "bytes": len(text_bytes)},
                                   "render_evidence": {"path": str(png_path), "sha256": png_hash, "bytes": png_path.stat().st_size, "renderer": render_version, "page": 1, "resolution_dpi": 120}},
                     "witness_author_id": "codex-agent-acl670-page1-batch-generator",
                     "source_pins": source_pins,
                     "witness_observations": {"title_visual_observation": "NOT_REVIEWED", "ordered_author_visual_observation": "NOT_REVIEWED", "proceedings_footer_visual_observation": "NOT_REVIEWED", "uncertainty_or_conflict": "NOT_REVIEWED"}})
    witness = output / "acl_metadata_page1_witness_input.v1.jsonl"
    witness.write_bytes(b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n" for row in rows))
    output_hashes[str(witness)] = file_sha(witness)
    for path, expected in source_hashes.items():
        _checked_hash(pathlib.Path(path), expected, "source input before receipt")
    for path, expected in output_hashes.items():
        _checked_hash(pathlib.Path(path), expected, "output before receipt")
    receipt = {"schema_version": "acl_metadata_task5_page1_batch_generator_receipt.v1", "scope": "Task 5 Step 1 evidence-only mechanical page-1 witness batch",
               "batch_number": args.batch_number, "item_ids": [r["item_id"] for r in rows],
               "selection_authority": "DETERMINISTIC_REMAINDER_SELECTION_ONLY_NOT_APPROVAL",
               "selection_path": str(args.selection), "selection_sha256": args.selection_sha256,
               "proof_rows_path": str(args.proof), "proof_rows_sha256": args.proof_sha256, "proof_row_count": 670, "proof_unique_item_ids": 670,
               "proof_run_receipt_path": str(args.proof_receipt), "proof_run_receipt_sha256": args.proof_receipt_sha256,
               "tool_versions": {"pdftotext": text_version, "pdftoppm": render_version},
               "tool_paths": {"pdftotext": str(text_exe), "pdftoppm": str(render_exe)},
               "input_source_file_hashes": source_hashes, "outputs": output_hashes,
               "counts": {"witness_records": 10, "text_files": 10, "png_files": 10, "reviewer_decisions": 0},
               "visual_review_performed": False, "source_admission_approved": False, "human_approved": False,
               "independent_reviewer_decisions_written": False, "metadata_projection_written": False, "card_created": False, "graph_ingested": False}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return output / "receipt.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "proof", "proof-receipt", "corpus-root", "output-dir", "pdftotext", "pdftoppm"):
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    for name in ("selection-sha256", "proof-sha256", "proof-receipt-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch-number", type=int, required=True)
    args = parser.parse_args()
    try:
        receipt = build(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    print(json.dumps({"receipt": str(receipt), "sha256": file_sha(receipt), "batch": args.batch_number}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
