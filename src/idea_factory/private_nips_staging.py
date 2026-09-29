"""Pin heterogeneous private NeurIPS Card evidence without creating run results.

READY means mechanical custody only. It does not approve source identity, science,
human review, or graph ingestion, and it does not invent a new Card job ID.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from .cards import _validate_card
from .private_card_transport import _canonical, _json, _read_pin, _row_bytes, _sha, _write_verified


SHA = re.compile(r"[0-9a-f]{64}\Z")
RAW_KEYS = {"schema_version", "slug", "note_sha256", "prompt_sha256", "cards", "empty_reason"}
SOURCE_FLAGS = {"science_gate": "UNREVIEWED", "scientific_entailment_audited": False,
                "human_approved": False, "graph_ingested": False,
                "source_admission_approved": False,
                "router_result_authenticated": False,
                "router_qa_authenticated": False,
                "card_job_bound": False}


def _lines(data: bytes) -> list[tuple[dict[str, Any], str]]:
    if not data.endswith(b"\n"):
        raise ValueError("JSONL must have newline-terminated rows")
    return [(_json(line), _sha(line + b"\n")) for line in data.splitlines()]


def _file(path: Path, root: Path) -> bytes:
    path = _canonical(path)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"MISSING_OR_OUTSIDE_ROOT: {path}")
    return path.read_bytes()


def _pin(path: Path, root: Path, expected_sha: str | None = None) -> dict[str, str]:
    data = _file(path, root)
    digest = _sha(data)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"SHA_CONFLICT: {path}")
    return {"path": str(path), "sha256": digest}


def _alias_matches(aliases: object, manifest: dict[str, Any], pdf_path: str) -> bool:
    return type(aliases) is list and any(
        type(alias) is dict and alias.get("cohort") == "neurips" and
        str(alias.get("event_id")) == str(manifest["event_id"]) and
        alias.get("arxiv_id") == manifest["arxiv_id"] and
        alias.get("pdf_path", pdf_path) == pdf_path
        for alias in aliases
    )


def _task_alias_matches(aliases: object, queue: dict[str, Any],
                        manifest: dict[str, Any], pdf_path: str) -> bool:
    queue_aliases = [a for a in queue["source_aliases"] if type(a) is dict and
                     a.get("cohort") == "neurips" and
                     str(a.get("event_id")) == str(manifest["event_id"]) and
                     a.get("arxiv_id") == manifest["arxiv_id"]]
    if len(queue_aliases) != 1 or type(aliases) is not list or not aliases:
        return False
    anchor = queue_aliases[0]
    if all(type(a) is dict for a in aliases):
        return any(
            str(a.get("event_id")) == str(manifest["event_id"]) and
            a.get("arxiv_id") == manifest["arxiv_id"] and
            a.get("source_key") == anchor.get("source_key") and
            a.get("source_manifest_sha256") == anchor.get("source_manifest_sha256") and
            a.get("cohort", "neurips") == "neurips" and
            a.get("pdf_path", pdf_path) == pdf_path
            for a in aliases
        )
    if all(type(a) is str for a in aliases):
        expected = {
            "arXiv:" + manifest["arxiv_id"],
            f"NeurIPS virtual event {manifest['event_id']} (event alias only; accepted-version equivalence not established)",
            "source_key: " + anchor["source_key"],
            anchor["official_event_url"],
            "source manifest SHA-256: " + anchor["source_manifest_sha256"],
        }
        return len(aliases) == len(expected) and set(aliases) == expected
    return False


def _review_conflicts(review: dict[str, Any], *, queue: dict[str, Any], task: dict[str, Any],
                      pins: dict[str, dict[str, str]]) -> None:
    expected = {
        "card_id": queue["card_id"], "raw_path": pins["raw"]["path"],
        "raw_sha256": pins["raw"]["sha256"],
        "reviewed_raw_path": pins["raw"]["path"],
        "note_path": pins["note"]["path"],
        "note_sha256": pins["note"]["sha256"],
        "source_pdf_sha256": pins["pdf"]["sha256"],
    }
    for key, value in expected.items():
        if key in review and review[key] not in (None, value):
            raise ValueError(f"REVIEW_{key.upper()}_CONFLICT")
    if "item_id" in review and review["item_id"] not in (None, queue["item_id"], task["slug"]):
        raise ValueError("REVIEW_ITEM_ID_CONFLICT")
    if "slug" in review and review["slug"] not in (None, task["slug"]):
        raise ValueError("REVIEW_SLUG_CONFLICT")
    source_pdf = review.get("source_pdf")
    if type(source_pdf) is dict:
        if source_pdf.get("path", pins["pdf"]["path"]) != pins["pdf"]["path"] or \
           source_pdf.get("sha256", pins["pdf"]["sha256"]) != pins["pdf"]["sha256"]:
            raise ValueError("REVIEW_PDF_CONFLICT")
    if review.get("human_approved") is True or review.get("graph_ingested") is True:
        raise ValueError("REVIEW_APPROVAL_CONFLICT")


def inspect_item(queue: dict[str, Any], manifest: dict[str, Any], worker: Path,
                 *, corpus_root: Path, queue_row_sha256: str | None = None,
                 manifest_row_sha256: str | None = None) -> dict[str, Any]:
    """Classify one exact queue/manifest pair; any item defect becomes HOLD."""
    base = {
        "event_id": str(manifest.get("event_id", "")),
        "source_version": manifest.get("arxiv_id"),
        "item_id": queue.get("item_id"),
        "worker": worker.name,
        **SOURCE_FLAGS,
    }
    try:
        root = _canonical(corpus_root)
        worker = _canonical(worker)
        if not worker.is_relative_to(root):
            raise ValueError("WORKER_OUTSIDE_ROOT")
        digest = queue.get("pdf_sha256")
        if (type(digest) is not str or not SHA.fullmatch(digest) or
            queue.get("item_id") != "pdf-content-" + digest or
            manifest.get("download_status") != "downloaded" or manifest.get("sha256") != digest):
            raise ValueError("QUEUE_MANIFEST_IDENTITY_CONFLICT")
        pdf_path = queue.get("pdf_path")
        if (type(pdf_path) is not str or
            pdf_path != manifest.get("pdf_path") or pdf_path != manifest.get("local_pdf") or
            not _alias_matches(queue.get("source_aliases"), manifest, pdf_path)):
            raise ValueError("QUEUE_MANIFEST_SOURCE_CONFLICT")
        item = worker / "items" / queue["item_id"]
        if (item / "progress/raw.json").is_file():
            item /= "progress"
        names = {"task": "task.json", "raw": "raw.json", "review": "review.json",
                 "validated": "schema_validated.json", "receipt": "receipt.json", "note": "note.md"}
        pins = {name: _pin(item / filename, root) for name, filename in names.items()}
        pins["pdf"] = _pin(Path(pdf_path), root, digest)
        task, raw, review, validated, receipt = (
            _json(_read_pin(pins[name]))
            for name in ("task", "raw", "review", "validated", "receipt")
        )
        slug = task.get("slug")
        if (type(slug) is not str or raw.get("slug") != slug or
            task.get("card_id") != queue.get("card_id")):
            raise ValueError("CARD_ID_OR_SLUG_CONFLICT")
        if (task.get("note_path") != pins["note"]["path"] or
            task.get("note_sha256") != pins["note"]["sha256"] or
            task.get("output_path") != pins["raw"]["path"]):
            raise ValueError("TASK_NOTE_OR_RAW_CONFLICT")
        prompt_path = task.get("prompt_path")
        if type(prompt_path) is not str:
            raise ValueError("TASK_PROMPT_MISSING")
        pins["prompt"] = _pin(Path(prompt_path), root, task.get("prompt_sha256"))
        for path_key, sha_key in (("source_pdf_path", "source_pdf_sha256"),
                                  ("pdf_path", "pdf_sha256")):
            if path_key in task or sha_key in task:
                if task.get(path_key) != pdf_path or task.get(sha_key) != digest:
                    raise ValueError("TASK_PDF_CONFLICT")
        if not (task.get("source_pdf_path") or task.get("pdf_path")):
            raise ValueError("TASK_PDF_MISSING")
        aliases = task.get("source_aliases")
        if aliases is not None and not _task_alias_matches(aliases, queue, manifest, pdf_path):
            raise ValueError("TASK_SOURCE_ALIAS_CONFLICT")
        if (set(raw) != RAW_KEYS or raw.get("schema_version") != "idea_factory.bulk_card_response.v1" or
            raw.get("empty_reason") != "" or type(raw.get("cards")) is not list or
            len(raw["cards"]) != 1 or raw.get("note_sha256") != pins["note"]["sha256"] or
            raw.get("prompt_sha256") != pins["prompt"]["sha256"]):
            raise ValueError("RAW_CARD_ENVELOPE_CONFLICT")
        card = raw["cards"][0]
        _validate_card(card, {"note_path": pins["note"]["path"]})
        if card["card_id"] != queue["card_id"]:
            raise ValueError("CARD_ID_CONFLICT")
        if (receipt.get("status") != "SCHEMA_VALID" or receipt.get("accepted_cards") != 1 or
            receipt.get("errors") != [] or receipt.get("slug") != slug or
            receipt.get("shard_id") != task.get("shard_id") or
            receipt.get("raw_result_sha256") != pins["raw"]["sha256"]):
            raise ValueError("RECEIPT_CONFLICT")
        if validated.get("schema_version") == "idea_factory.bulk_schema_validated_bundle.v1":
            items = validated.get("items")
            if (validated.get("worker_id") != worker.name or
                type(items) is not list or len(items) != 1 or type(items[0]) is not dict):
                raise ValueError("SCHEMA_VALIDATED_BUNDLE_CONFLICT")
            validated = items[0]
        if (validated.get("schema_version") != "idea_factory.bulk_schema_validated_card.v1" or
            validated.get("slug") != slug or validated.get("card") != card or
            validated.get("record_id") != card["card_id"] or
            validated.get("raw_result_sha256") != pins["raw"]["sha256"] or
            validated.get("note_path") != pins["note"]["path"] or
            validated.get("note_sha256") != pins["note"]["sha256"] or
            validated.get("prompt_sha256") != pins["prompt"]["sha256"]):
            raise ValueError("SCHEMA_VALIDATED_CONFLICT")
        _review_conflicts(review, queue=queue, task=task, pins=pins)
        card_bytes = json.dumps(card, ensure_ascii=False, allow_nan=False,
                                sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {**base, "status": "READY", "card_id": card["card_id"],
                "card_sha256": _sha(card_bytes), "source_record_id": "arxiv:" + manifest["arxiv_id"],
                "pdf_sha256": digest, "pdf_path": pdf_path,
                "queue_row_sha256": queue_row_sha256,
                "manifest_row_sha256": manifest_row_sha256,
                "pins": pins, "review_authentication": "HASH_BOUND_NOT_SCIENCE_QA"}
    except FileNotFoundError:
        return {**base, "status": "HOLD", "reason": "MISSING_PRIVATE_ARTIFACT",
                "queue_row_sha256": queue_row_sha256,
                "manifest_row_sha256": manifest_row_sha256}
    except (ValueError, KeyError, TypeError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        reason = str(exc).split(":", 1)[0] or type(exc).__name__
        return {**base, "status": "HOLD", "reason": reason,
                "queue_row_sha256": queue_row_sha256,
                "manifest_row_sha256": manifest_row_sha256}


def stage(manifest_pin: dict[str, str], queue_pins: list[tuple[str, dict[str, str]]],
          output_dir: Path, *, corpus_root: Path, expected_downloaded: int = 615) -> dict[str, Any]:
    """Create a new private custody package; no results or run are initialized."""
    root = _canonical(corpus_root)
    manifest_path = _canonical(Path(manifest_pin["path"]))
    if not manifest_path.is_relative_to(root):
        raise ValueError("manifest outside corpus root")
    manifest = _lines(_read_pin(manifest_pin))
    downloaded = [(row, row_sha) for row, row_sha in manifest if row.get("download_status") == "downloaded"]
    if len(downloaded) != expected_downloaded:
        raise ValueError("downloaded manifest count mismatch")
    identity = [(str(row.get("event_id")), row.get("arxiv_id"), row.get("sha256"))
                for row, _ in downloaded]
    if len(set(identity)) != len(identity):
        raise ValueError("duplicate downloaded manifest identity")
    if len(queue_pins) != 5 or {name for name, _ in queue_pins} != {f"nips-arxiv-{i:02d}" for i in range(1, 6)}:
        raise ValueError("expected five distinct NeurIPS worker queue pins")
    index: dict[tuple[str, str, str], list[tuple[dict[str, Any], str, Path]]] = {}
    for name, pin in queue_pins:
        queue_path = _canonical(Path(pin["path"]))
        worker = queue_path.parent
        if worker.name != name or queue_path.name != "queue.jsonl" or not worker.is_relative_to(root):
            raise ValueError("queue worker/path mismatch")
        for row, row_sha in _lines(_read_pin(pin)):
            for alias in row.get("source_aliases", []):
                if type(alias) is dict and alias.get("cohort") == "neurips":
                    key = (str(alias.get("event_id")), alias.get("arxiv_id"), row.get("pdf_sha256"))
                    index.setdefault(key, []).append((row, row_sha, worker))
    ready: list[dict[str, Any]] = []
    holds: list[dict[str, Any]] = []
    for row, row_sha in downloaded:
        key = (str(row["event_id"]), row["arxiv_id"], row["sha256"])
        matches = index.get(key, [])
        if len(matches) != 1:
            holds.append({"event_id": key[0], "source_version": key[1], "status": "HOLD",
                          "reason": "MISSING_QUEUE" if not matches else "DUPLICATE_QUEUE_MATCH",
                          "manifest_row_sha256": row_sha, **SOURCE_FLAGS})
            continue
        q, q_sha, worker = matches[0]
        result = inspect_item(q, row, worker, corpus_root=root,
                              queue_row_sha256=q_sha, manifest_row_sha256=row_sha)
        (ready if result["status"] == "READY" else holds).append(result)
    # Recheck every selected source before publishing; a changed item becomes HOLD.
    still_ready = []
    for row in ready:
        try:
            for pin in row["pins"].values():
                _read_pin(pin)
            still_ready.append(row)
        except (ValueError, OSError) as exc:
            holds.append({key: row[key] for key in ("event_id", "source_version", "item_id", "worker")}
                         | {"status": "HOLD", "reason": "SOURCE_DRIFT_BEFORE_PUBLISH",
                            "manifest_row_sha256": row["manifest_row_sha256"],
                            "queue_row_sha256": row["queue_row_sha256"], **SOURCE_FLAGS})
    ready = still_ready
    _read_pin(manifest_pin)
    for _, pin in queue_pins:
        _read_pin(pin)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not output_dir.parent.is_dir():
        raise ValueError("output parent missing")
    os.mkdir(output_dir)
    ready_bytes, hold_bytes = _row_bytes(ready), _row_bytes(holds)
    _write_verified(output_dir / "ready.v1.jsonl", ready_bytes)
    _write_verified(output_dir / "holds.v1.jsonl", hold_bytes)
    receipt = {"schema_version": "idea_factory.private_nips_card_staging_receipt.v1",
               "manifest": manifest_pin, "queues": [{"worker": name, **pin} for name, pin in queue_pins],
               "downloaded_count": len(downloaded), "ready_count": len(ready), "hold_count": len(holds),
               "ready_sha256": _sha(ready_bytes), "holds_sha256": _sha(hold_bytes),
               **SOURCE_FLAGS}
    receipt_bytes = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    temp_receipt = output_dir / "receipt.json.partial"
    _write_verified(temp_receipt, receipt_bytes)
    os.rename(temp_receipt, output_dir / "receipt.json")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--queue-pin", nargs=3, action="append", metavar=("WORKER", "PATH", "SHA256"), required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-downloaded", type=int, default=615)
    args = parser.parse_args(argv)
    if not SHA.fullmatch(args.manifest_sha256):
        parser.error("invalid manifest SHA-256")
    queue_pins = []
    for name, path, digest in args.queue_pin:
        if not SHA.fullmatch(digest):
            parser.error("invalid queue SHA-256")
        queue_pins.append((name, {"path": path, "sha256": digest}))
    receipt = stage({"path": str(args.manifest), "sha256": args.manifest_sha256}, queue_pins,
                    args.output_dir, corpus_root=args.corpus_root,
                    expected_downloaded=args.expected_downloaded)
    print(json.dumps({"ready": receipt["ready_count"], "hold": receipt["hold_count"],
                      "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
