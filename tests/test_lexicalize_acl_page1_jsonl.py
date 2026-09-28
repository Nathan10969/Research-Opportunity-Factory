"""A byte-only bridge for the one literal U+2028 in ACL page-1 JSONL."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from idea_factory.acl_task4_evidence_adapter import load_task4_page1_witnesses


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "lexicalize_acl_page1_jsonl.py"
OUTPUT_NAME = "page1-witnesses.adapter-safe.v1.jsonl"
LITERAL = "\u2028".encode("utf-8")
ESCAPED = b"\\u2028"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def lexical_inputs(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    source = root / "page1-witnesses.v1.jsonl"
    rows = []
    for index in range(670):
        item_id = "2026.findings-acl.192" if index == 134 else f"2026.findings-acl.{1000 + index}"
        page_text = "A title\u2028and an author" if index == 134 else f"Page one {index}"
        rows.append({
            "schema_version": "acl_metadata_page1_witness_input.v1",
            "item_id": item_id,
            "pdf": {"sha256": "a" * 64},
            "pdf_page1": {"text": page_text, "text_sha256": _sha(page_text.encode("utf-8"))},
            "witness_author_id": "fixture-author",
        })
    raw = b"".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n" for row in rows)
    assert raw.count(LITERAL) == 1
    source.write_bytes(raw)
    return {"root": root, "source": source, "raw": raw, "sha256": _sha(raw),
            "out": root / "lexical-bridge", "rows": rows}


def _args(data, *, sha=None, count="670", literals="1", output=None):
    return [sys.executable, str(CLI), "--input", str(data["source"]),
            "--input-sha256", sha or data["sha256"], "--corpus-root", str(data["root"]),
            "--output-dir", str(output or data["out"]), "--expected-count", count,
            "--expected-literal-u2028", literals]


def _run(data, **kwargs):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(_args(data, **kwargs), cwd=ROOT, env=env, capture_output=True, text=True)


def _module():
    spec = importlib.util.spec_from_file_location("acl_page1_lexical_bridge_test", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _namespace(data):
    return Namespace(input=str(data["source"]), input_sha256=data["sha256"],
                     corpus_root=str(data["root"]), output_dir=str(data["out"]),
                     expected_count=670, expected_literal_u2028=1)


def test_one_literal_u2028_becomes_only_lexical_escape(lexical_inputs):
    data = lexical_inputs
    with pytest.raises(ValueError, match="LF/CRLF"):
        load_task4_page1_witnesses(data["raw"])

    result = _run(data)

    assert result.returncode == 0, result.stderr
    output = (data["out"] / OUTPUT_NAME).read_bytes()
    assert output == data["raw"].replace(LITERAL, ESCAPED)
    assert len(load_task4_page1_witnesses(output)) == 670
    assert [json.loads(line) for line in data["raw"].split(b"\n")[:-1]] == load_task4_page1_witnesses(output)
    receipt = json.loads((data["out"] / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["source"]["sha256"] == data["sha256"]
    assert receipt["output"]["sha256"] == _sha(output)
    assert receipt["output"]["bytes"] == len(output)
    assert receipt["literal_u2028_replacements"] == 1
    assert receipt["byte_length_delta"] == 3
    assert receipt["source_admission_approved"] is False
    assert receipt["downstream_card_use_approved"] is False
    assert receipt["human_approved"] is False
    assert receipt["graph_ingested"] is False
    assert data["source"].read_bytes() == data["raw"]
    assert {path.name for path in data["out"].iterdir()} == {OUTPUT_NAME, "receipt.json"}


@pytest.mark.parametrize("mutation,expected", [
    ("wrong_pin", "hash mismatch"),
    ("wrong_rows", "row count"),
    ("wrong_literals", "literal U+2028"),
    ("duplicate_id", "duplicate"),
])
def test_mismatched_pins_or_scope_create_no_output(lexical_inputs, mutation, expected):
    data = lexical_inputs
    kwargs = {}
    if mutation == "wrong_pin":
        kwargs["sha"] = "0" * 64
    elif mutation == "wrong_rows":
        raw = b"\n".join(data["raw"].split(b"\n")[:-2]) + b"\n"
        data["source"].write_bytes(raw)
        data["raw"] = raw
        data["sha256"] = _sha(raw)
    elif mutation == "wrong_literals":
        kwargs["literals"] = "2"
    else:
        rows = list(data["rows"])
        rows[-1] = dict(rows[0])
        raw = b"".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n" for row in rows)
        data["source"].write_bytes(raw)
        data["raw"] = raw
        data["sha256"] = _sha(raw)

    result = _run(data, **kwargs)

    assert result.returncode != 0
    assert expected.lower() in result.stderr.lower()
    assert not data["out"].exists()
    assert data["source"].read_bytes() == data["raw"]


def test_existing_output_directory_is_preserved(lexical_inputs):
    data = lexical_inputs
    data["out"].mkdir()
    marker = data["out"] / "keep.txt"
    marker.write_text("keep")

    result = _run(data)

    assert result.returncode != 0
    assert marker.read_text() == "keep"
    assert not (data["out"] / "receipt.json").exists()
    assert data["source"].read_bytes() == data["raw"]


def test_output_directory_outside_root_is_rejected(lexical_inputs, tmp_path):
    data = lexical_inputs
    outside = tmp_path / "outside"

    result = _run(data, output=outside)

    assert result.returncode != 0
    assert "inside corpus root" in result.stderr
    assert not outside.exists()
    assert data["source"].read_bytes() == data["raw"]


@pytest.mark.parametrize("target", [OUTPUT_NAME, "receipt.json"])
def test_short_staged_write_never_publishes(lexical_inputs, monkeypatch, target):
    data = lexical_inputs
    module = _module()
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
        module.lexicalize(_namespace(data))

    assert not data["out"].exists()
    assert list(data["root"].glob(f".{data['out'].name}.staging-*")) == []
    assert data["source"].read_bytes() == data["raw"]


@pytest.mark.parametrize("fault", ["changed_bytes", "read_error"])
def test_staged_readback_fault_never_publishes(lexical_inputs, monkeypatch, fault):
    data = lexical_inputs
    module = _module()
    original_read = Path.read_bytes

    def read_with_fault(path):
        payload = original_read(path)
        if path.name == OUTPUT_NAME and ".staging-" in path.parent.name:
            if fault == "read_error":
                raise OSError("injected staged read error")
            return b"X" + payload[1:]
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_with_fault)
    with pytest.raises((OSError, ValueError), match="output readback"):
        module.lexicalize(_namespace(data))

    assert not data["out"].exists()
    assert list(data["root"].glob(f".{data['out'].name}.staging-*")) == []
    assert data["source"].read_bytes() == data["raw"]


def test_input_drift_before_publish_does_not_publish(lexical_inputs, monkeypatch):
    data = lexical_inputs
    module = _module()
    original_read = Path.read_bytes

    def read_then_drift(path):
        payload = original_read(path)
        if path.name == "receipt.json" and ".staging-" in path.parent.name:
            data["source"].write_bytes(data["raw"] + b" ")
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_drift)
    with pytest.raises(ValueError, match="input hash changed before publish"):
        module.lexicalize(_namespace(data))

    assert not data["out"].exists()
    assert list(data["root"].glob(f".{data['out'].name}.staging-*")) == []
