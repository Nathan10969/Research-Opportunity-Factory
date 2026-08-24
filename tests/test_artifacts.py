import json
import os
from pathlib import Path

import pytest

import idea_factory.artifacts as artifact_module
from idea_factory.artifacts import (
    append_jsonl,
    read_jsonl,
    sha256_file,
    stable_id,
    write_json,
)


def test_jsonl_round_trip_preserves_unicode_and_two_records(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "events.jsonl"

    append_jsonl(path, {"event": "开始", "count": 1})
    append_jsonl(path, {"event": "完了", "count": 2})

    assert read_jsonl(path) == [
        {"event": "开始", "count": 1},
        {"event": "完了", "count": 2},
    ]
    assert "开始" in path.read_text(encoding="utf-8")


def test_write_json_creates_parent_unicode_and_no_temp_leftover(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"

    write_json(path, {"message": "路线已就绪", "count": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "message": "路线已就绪",
        "count": 2,
    }
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("writer", [write_json, append_jsonl])
def test_json_writers_reject_non_object_payloads(tmp_path: Path, writer: object) -> None:
    with pytest.raises(TypeError, match="dict"):
        writer(tmp_path / "record.json", ["not", "an object"])  # type: ignore[operator]


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('\n{"one":1}\n  \n{"two":2}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"one": 1}, {"two": 2}]


def test_read_jsonl_reports_exact_malformed_line_and_path(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"one": 1}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        read_jsonl(path)

    assert str(path) in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_read_jsonl_reports_non_object_line_and_path(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"one": 1}\n["not", "object"]\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        read_jsonl(path)

    assert str(path) in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_stable_id_is_deterministic_and_parts_change_it() -> None:
    assert stable_id("run", "corpus", "abc") == stable_id("run", "corpus", "abc")
    assert stable_id("run", "corpus", "abc") != stable_id("run", "corpus", "def")


@pytest.mark.parametrize(
    ("prefix", "parts"),
    [("", ("part",)), ("run", ("",)), ("run", ("a\x1fb",)), ("bad-prefix!", ("part",))],
)
def test_stable_id_rejects_ambiguous_or_blank_inputs(
    prefix: str, parts: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        stable_id(prefix, *parts)


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.parametrize("writer", [write_json, append_jsonl])
def test_json_writers_reject_nonfinite_numbers(
    tmp_path: Path, writer: object
) -> None:
    path = tmp_path / "record.json"
    path.write_text('{"preserved":true}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        writer(path, {"score": float("nan")})  # type: ignore[operator]

    assert path.read_text(encoding="utf-8") == '{"preserved":true}\n'
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_read_jsonl_rejects_nonfinite_constants_with_path_and_line(
    tmp_path: Path, constant: str
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"ok":1}\n{"value":' + constant + "}\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        read_jsonl(path)

    assert str(path) in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_read_jsonl_reports_invalid_utf8_path_and_exact_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"ok":1}\n{"value":"\xff"}\n')

    with pytest.raises(ValueError) as exc_info:
        read_jsonl(path)

    assert str(path) in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


def test_append_jsonl_uses_one_append_write_and_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    real_write = os.write
    real_fsync = os.fsync
    calls = {"write": 0, "fsync": 0}

    def tracked_write(fd: int, data: bytes) -> int:
        calls["write"] += 1
        return real_write(fd, data)

    def tracked_fsync(fd: int) -> None:
        calls["fsync"] += 1
        real_fsync(fd)

    monkeypatch.setattr(artifact_module.os, "write", tracked_write)
    monkeypatch.setattr(artifact_module.os, "fsync", tracked_fsync)

    append_jsonl(path, {"event": "durable"})

    assert calls == {"write": 1, "fsync": 1}
    assert read_jsonl(path) == [{"event": "durable"}]
