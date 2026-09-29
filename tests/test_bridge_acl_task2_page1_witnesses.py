"""Exact-ID custody tests for the ten legacy flat ACL page-1 witnesses."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from idea_factory.acl_task4_evidence_adapter import (
    load_task4_page1_witnesses,
    normalize_task4_page1_witness,
)


WORKTREE = Path(__file__).resolve().parents[1]
CLI = WORKTREE / "scripts" / "bridge_acl_task2_page1_witnesses.py"
OUTPUT_NAME = "page1-witnesses.task4-compatible.v1.jsonl"
FLAT_IDS = {
    "2026.findings-acl.1077", "2026.findings-acl.1105", "2026.findings-acl.1174",
    "2026.findings-acl.1266", "2026.findings-acl.1320", "2026.findings-acl.135",
    "2026.findings-acl.1371", "2026.findings-acl.1388", "2026.findings-acl.1412",
    "2026.findings-acl.1530",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(rows):
    return b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n" for row in rows)


@pytest.fixture
def bridge_inputs(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    ids = [f"2026.findings-acl.{2000 + index}" for index in range(670)]
    for position, item_id in zip(range(0, 670, 67), sorted(FLAT_IDS), strict=True):
        ids[position] = item_id
    proof_rows = [{"item_id": item_id, "pdf_sha256": _sha(item_id.encode("utf-8"))} for item_id in ids]
    proof_path = root / "proof.jsonl"
    proof_bytes = _jsonl(proof_rows)
    proof_path.write_bytes(proof_bytes)
    records = []
    raw_lines = []
    for index, proof in enumerate(proof_rows):
        item_id = proof["item_id"]
        text = f"Page 1 for {item_id} — Author"
        digest = _sha(text.encode("utf-8"))
        if item_id in FLAT_IDS:
            row = {"item_id": item_id, "pdf_sha256": proof["pdf_sha256"],
                   "page1_text_sha256": digest, "page1_text": text,
                   "pdftotext_version": "pdftotext version fixture",
                   "witness_author_id": "fixture-author"}
        else:
            row = {"schema_version": "acl_metadata_page1_witness_input.v1", "item_id": item_id,
                   "pdf": {"sha256": proof["pdf_sha256"]},
                   "pdf_page1": {"text": text, "text_sha256": digest,
                                 "pdftotext_version": "pdftotext version fixture"},
                   "witness_author_id": "fixture-author",
                   "source_pins": {"fixture": index}, "witness_observations": ["untouched"]}
        records.append(row)
        # Mixed CRLF/LF and formatting make raw-line preservation observable.
        ending = b"\r\n" if index % 2 else b"\n"
        raw_lines.append(json.dumps(row, ensure_ascii=True, separators=(", ", ": ")).encode("utf-8") + ending)
    source_path = root / "page1-input.jsonl"
    source_bytes = b"".join(raw_lines)
    source_path.write_bytes(source_bytes)
    return {"root": root, "source": source_path, "source_bytes": source_bytes,
            "source_sha": _sha(source_bytes), "proof": proof_path, "proof_bytes": proof_bytes,
            "proof_sha": _sha(proof_bytes), "proof_rows": proof_rows, "rows": records,
            "raw_lines": raw_lines, "ids": ids, "out": root / "bridged"}


def _cli_args(data, **overrides):
    return ["--input", str(overrides.get("input_path", data["source"])),
            "--input-sha256", overrides.get("input_sha", data["source_sha"]),
            "--proof", str(overrides.get("proof_path", data["proof"])),
            "--proof-sha256", overrides.get("proof_sha", data["proof_sha"]),
            "--corpus-root", str(data["root"]), "--output-dir", str(data["out"]),
            "--expected-count", "670"]


def _run_public(data, **overrides):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKTREE / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, str(CLI), *_cli_args(data, **overrides)],
                          cwd=WORKTREE, env=env, capture_output=True, text=True)


def _run(data, **overrides):
    module = _module(data)
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = module.main(_cli_args(data, **overrides))
    return SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())


def test_production_cli_rejects_self_consistent_synthetic_sources(bridge_inputs):
    data = bridge_inputs
    result = _run_public(data)

    assert result.returncode != 0
    assert "authoritative" in result.stderr.lower()
    assert not data["out"].exists()
    assert data["source"].read_bytes() == data["source_bytes"]
    assert data["proof"].read_bytes() == data["proof_bytes"]


@pytest.mark.parametrize("target", ["input", "proof"])
def test_same_bytes_at_alias_path_are_not_authoritative(bridge_inputs, target):
    data = bridge_inputs
    original = data["source"] if target == "input" else data["proof"]
    alias = data["root"] / f"alias-{target}.jsonl"
    alias.write_bytes(original.read_bytes())

    result = _run(data, **{f"{target}_path": alias})

    assert result.returncode != 0
    assert f"authoritative {target} path" in result.stderr.lower()
    assert not data["out"].exists()


def _module(data=None):
    spec = importlib.util.spec_from_file_location("acl_task2_bridge_test", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if data is not None:
        module._AUTHORITATIVE_INPUT_PATH = data["source"]
        module._AUTHORITATIVE_INPUT_SHA256 = data["source_sha"]
        module._AUTHORITATIVE_PROOF_PATH = data["proof"]
        module._AUTHORITATIVE_PROOF_SHA256 = data["proof_sha"]
    return module


def _namespace(data):
    return Namespace(input=str(data["source"]), input_sha256=data["source_sha"],
                     proof=str(data["proof"]), proof_sha256=data["proof_sha"],
                     corpus_root=str(data["root"]), output_dir=str(data["out"]),
                     expected_count=670)


def _repin(data):
    data["source_bytes"] = data["source"].read_bytes()
    data["source_sha"] = _sha(data["source_bytes"])


def test_only_ten_exact_flat_ids_are_converted_and_660_raw_lines_survive(bridge_inputs):
    data = bridge_inputs
    result = _run(data)

    assert result.returncode == 0, result.stderr
    output_bytes = (data["out"] / OUTPUT_NAME).read_bytes()
    output_lines = output_bytes.splitlines(keepends=True)
    assert len(output_lines) == 670
    for index, row in enumerate(data["rows"]):
        if row["item_id"] not in FLAT_IDS:
            assert output_lines[index] == data["raw_lines"][index]
    output_rows = load_task4_page1_witnesses(output_bytes)
    assert [row["item_id"] for row in output_rows] == data["ids"]
    for source, output, proof in zip(data["rows"], output_rows, data["proof_rows"], strict=True):
        normalized, hold = normalize_task4_page1_witness(
            output, expected_item_id=source["item_id"], expected_pdf_sha256=proof["pdf_sha256"])
        assert hold is None
        if source["item_id"] in FLAT_IDS:
            assert normalized == source
        else:
            source_normalized, source_hold = normalize_task4_page1_witness(
                source, expected_item_id=source["item_id"], expected_pdf_sha256=proof["pdf_sha256"])
            assert source_hold is None
            assert normalized == source_normalized
    receipt = json.loads((data["out"] / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["source"]["sha256"] == data["source_sha"]
    assert receipt["proof"]["sha256"] == data["proof_sha"]
    assert receipt["output"]["sha256"] == _sha(output_bytes)
    assert receipt["converted_flat_count"] == 10
    assert receipt["unchanged_nested_count"] == 660
    assert set(receipt["converted_flat_ids"]) == FLAT_IDS
    assert receipt["source_admission_approved"] is False
    assert receipt["downstream_card_use_approved"] is False
    assert receipt["human_approved"] is False
    assert receipt["graph_ingested"] is False
    assert data["source"].read_bytes() == data["source_bytes"]
    assert data["proof"].read_bytes() == data["proof_bytes"]
    assert {path.name for path in data["out"].iterdir()} == {OUTPUT_NAME, "receipt.json"}


@pytest.mark.parametrize("mutation,expected", [
    ("wrong_input_pin", "authoritative frozen pin"),
    ("wrong_proof_pin", "authoritative frozen pin"),
    ("wrong_proof_count", "exactly 670"),
    ("duplicate_proof_id", "duplicate"),
    ("missing_flat", "flat ID set"),
    ("extra_flat", "flat ID set"),
    ("extra_flat_key", "unsupported"),
    ("wrong_flat_id", "proof ID set"),
    ("bad_text_hash", "page1 text hash"),
    ("bad_pdf_hash", "PDF hash"),
    ("bad_nested_pdf_hash", "PDF hash"),
    ("duplicate_id", "duplicate"),
])
def test_invalid_scope_or_binding_never_publishes(bridge_inputs, mutation, expected):
    data = bridge_inputs
    overrides = {}
    if mutation == "wrong_input_pin":
        overrides["input_sha"] = "0" * 64
    elif mutation == "wrong_proof_pin":
        overrides["proof_sha"] = "0" * 64
    elif mutation in {"wrong_proof_count", "duplicate_proof_id"}:
        proof_rows = list(data["proof_rows"])
        if mutation == "wrong_proof_count":
            proof_rows.pop()
        else:
            proof_rows[-1] = dict(proof_rows[0])
        data["proof_bytes"] = _jsonl(proof_rows)
        data["proof_sha"] = _sha(data["proof_bytes"])
        data["proof"].write_bytes(data["proof_bytes"])
    else:
        rows = data["rows"]
        flat_index = next(index for index, row in enumerate(rows) if row["item_id"] in FLAT_IDS)
        nested_index = next(index for index, row in enumerate(rows) if row["item_id"] not in FLAT_IDS)
        if mutation == "missing_flat":
            flat = rows[flat_index]
            rows[flat_index] = {"schema_version": "acl_metadata_page1_witness_input.v1",
                                "item_id": flat["item_id"], "pdf": {"sha256": flat["pdf_sha256"]},
                                "pdf_page1": {"text": flat["page1_text"], "text_sha256": flat["page1_text_sha256"],
                                              "pdftotext_version": flat["pdftotext_version"]},
                                "witness_author_id": flat["witness_author_id"]}
        elif mutation == "extra_flat":
            nested = rows[nested_index]
            rows[nested_index] = {"item_id": nested["item_id"], "pdf_sha256": nested["pdf"]["sha256"],
                                  "page1_text_sha256": nested["pdf_page1"]["text_sha256"],
                                  "page1_text": nested["pdf_page1"]["text"],
                                  "pdftotext_version": nested["pdf_page1"]["pdftotext_version"],
                                  "witness_author_id": nested["witness_author_id"]}
        elif mutation == "extra_flat_key":
            rows[flat_index] = dict(rows[flat_index], extra="not an exact Task2 row")
        elif mutation == "wrong_flat_id":
            rows[flat_index] = dict(rows[flat_index], item_id="2026.findings-acl.999999")
        elif mutation == "bad_text_hash":
            rows[flat_index] = dict(rows[flat_index], page1_text_sha256="0" * 64)
        elif mutation == "bad_pdf_hash":
            rows[flat_index] = dict(rows[flat_index], pdf_sha256="0" * 64)
        elif mutation == "bad_nested_pdf_hash":
            nested = dict(rows[nested_index])
            nested["pdf"] = {"sha256": "0" * 64}
            rows[nested_index] = nested
        else:
            rows[-1] = dict(rows[flat_index])
        data["source"].write_bytes(_jsonl(rows))
        _repin(data)

    result = _run(data, **overrides)

    assert result.returncode != 0
    assert expected.lower() in result.stderr.lower()
    assert not data["out"].exists()
    assert data["source"].read_bytes() == data["source_bytes"]
    assert data["proof"].read_bytes() == data["proof_bytes"]


def test_existing_output_is_untouched(bridge_inputs):
    data = bridge_inputs
    data["out"].mkdir()
    marker = data["out"] / "keep.txt"
    marker.write_text("keep")
    result = _run(data)
    assert result.returncode != 0
    assert marker.read_text() == "keep"
    assert not (data["out"] / "receipt.json").exists()


@pytest.mark.parametrize("target", [OUTPUT_NAME, "receipt.json"])
def test_short_staged_write_leaves_no_published_output(bridge_inputs, monkeypatch, target):
    data = bridge_inputs
    module = _module(data)
    original_open = Path.open

    class ShortWriter:
        def __init__(self, real):
            self.real = real

        def __getattr__(self, name):
            return getattr(self.real, name)

        def __enter__(self):
            return self

        def write(self, payload):
            self.real.write(payload[:1])
            return 1

        def __exit__(self, exc_type, exc, tb):
            return self.real.__exit__(exc_type, exc, tb)

    def open_with_short_write(path, mode="r", *args, **kwargs):
        real = original_open(path, mode, *args, **kwargs)
        if path.name == target and mode == "xb" and ".staging-" in path.parent.name:
            return ShortWriter(real)
        return real

    monkeypatch.setattr(Path, "open", open_with_short_write)
    with pytest.raises(OSError, match="short output write"):
        module.bridge(_namespace(data))
    assert not data["out"].exists()
    assert list(data["root"].glob(f".{data['out'].name}.staging-*")) == []
    assert data["source"].read_bytes() == data["source_bytes"]


def test_staged_readback_corruption_leaves_no_published_output(bridge_inputs, monkeypatch):
    data = bridge_inputs
    module = _module(data)
    original_read = Path.read_bytes

    def corrupt_staged_read(path):
        payload = original_read(path)
        if path.name == OUTPUT_NAME and ".staging-" in path.parent.name:
            return b"X" + payload[1:]
        return payload

    monkeypatch.setattr(Path, "read_bytes", corrupt_staged_read)
    with pytest.raises(ValueError, match="output readback mismatch"):
        module.bridge(_namespace(data))
    assert not data["out"].exists()
    assert list(data["root"].glob(f".{data['out'].name}.staging-*")) == []
