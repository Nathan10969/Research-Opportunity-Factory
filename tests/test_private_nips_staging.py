"""Private, read-only NeurIPS staging contract tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from idea_factory.private_nips_staging import inspect_item, stage


CORPUS = Path("F:/LLM_Evoke")
WORKERS = CORPUS / "runs/parallel24-20260927-1340/workers"
MANIFEST = CORPUS / "papers/nips2026/.metadata/manifest.jsonl"

# Two structurally distinct task/review signatures from each of five workers.
SAMPLES = [
    ("01", "00c1918f"), ("01", "3856382b"),
    ("02", "520c27aa"), ("02", "03899b36"),
    ("03", "3caca338"), ("03", "9e4f3860"),
    ("04", "024a32c9"), ("04", "c01fdf3e"),
    ("05", "63bc84f0"), ("05", "01f896b2"),
]


def _real_pair(worker_number: str, pdf_prefix: str) -> tuple[dict, dict, Path]:
    worker = WORKERS / f"nips-arxiv-{worker_number}"
    queue = [json.loads(line) for line in (worker / "queue.jsonl").read_text(encoding="utf-8").splitlines()]
    q = next(row for row in queue if row["item_id"].startswith("pdf-content-" + pdf_prefix))
    manifest = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    m = next(row for row in manifest if row.get("download_status") == "downloaded" and
             row["sha256"] == q["pdf_sha256"] and
             any(str(alias.get("event_id")) == str(row["event_id"]) and
                 alias.get("arxiv_id") == row["arxiv_id"]
                 for alias in q["source_aliases"]))
    return q, m, worker


@pytest.mark.skipif(not MANIFEST.is_file(), reason="live private NeurIPS fixture unavailable")
@pytest.mark.parametrize("worker_number,pdf_prefix", SAMPLES)
def test_ten_real_format_worker_samples_preserve_card_body(
    worker_number: str, pdf_prefix: str,
) -> None:
    q, m, worker = _real_pair(worker_number, pdf_prefix)
    item = worker / "items" / q["item_id"]
    if (item / "progress/raw.json").is_file():
        item /= "progress"
    before = {name: hashlib.sha256((item / name).read_bytes()).hexdigest()
              for name in ("task.json", "raw.json", "review.json", "receipt.json", "schema_validated.json")}
    raw_card = json.loads((item / "raw.json").read_text(encoding="utf-8"))["cards"][0]
    result = inspect_item(q, m, worker, corpus_root=CORPUS)
    assert result["status"] == "READY", result
    assert result["card_sha256"] == hashlib.sha256(
        json.dumps(raw_card, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert result["source_version"] == m["arxiv_id"]
    assert result["science_gate"] == "UNREVIEWED"
    assert result["scientific_entailment_audited"] is False
    assert result["human_approved"] is False
    assert result["router_result_authenticated"] is False
    assert result["router_qa_authenticated"] is False
    assert result["card_job_bound"] is False
    assert before == {name: hashlib.sha256((item / name).read_bytes()).hexdigest() for name in before}


@pytest.mark.skipif(not MANIFEST.is_file(), reason="live private NeurIPS fixture unavailable")
def test_queue_card_id_conflict_is_item_hold() -> None:
    q, m, worker = _real_pair("01", "00c1918f")
    broken = copy.deepcopy(q)
    broken["card_id"] = "card-conflict"
    result = inspect_item(broken, m, worker, corpus_root=CORPUS)
    assert result["status"] == "HOLD"
    assert "CARD_ID" in result["reason"]


@pytest.mark.skipif(not MANIFEST.is_file(), reason="live private NeurIPS fixture unavailable")
@pytest.mark.parametrize("worker_number,pdf_prefix", [
    ("02", "8c0c25d3"), ("02", "8c01efbb"), ("02", "8e212475"),
    ("05", "51041a4b"), ("05", "5194fc84"), ("05", "53085129"),
])
def test_sparse_but_exact_task_alias_variants_are_ready(worker_number: str, pdf_prefix: str) -> None:
    q, m, worker = _real_pair(worker_number, pdf_prefix)
    result = inspect_item(q, m, worker, corpus_root=CORPUS)
    assert result["status"] == "READY", result


@pytest.mark.skipif(not MANIFEST.is_file(), reason="live private NeurIPS fixture unavailable")
@pytest.mark.parametrize("pdf_prefix", ["0d5d505e", "08fb2bdb"])
def test_single_item_schema_bundle_is_ready(pdf_prefix: str) -> None:
    q, m, worker = _real_pair("04", pdf_prefix)
    result = inspect_item(q, m, worker, corpus_root=CORPUS)
    assert result["status"] == "READY", result


@pytest.mark.skipif(not MANIFEST.is_file(), reason="live private NeurIPS fixture unavailable")
def test_task_alias_with_conflicting_pdf_path_is_hold() -> None:
    q, m, worker = _real_pair("03", "9731cd61")
    result = inspect_item(q, m, worker, corpus_root=CORPUS)
    assert result["status"] == "HOLD"
    assert result["reason"] == "TASK_SOURCE_ALIAS_CONFLICT"


def test_missing_private_artifacts_is_item_hold(tmp_path: Path) -> None:
    digest = "a" * 64
    q = {"item_id": "pdf-content-" + digest, "pdf_sha256": digest,
         "pdf_path": str(tmp_path / "missing.pdf"), "card_id": "card-1",
         "source_aliases": [{"cohort": "neurips", "event_id": "1", "arxiv_id": "2601.00001v1"}]}
    m = {"event_id": "1", "arxiv_id": "2601.00001v1", "sha256": digest,
         "pdf_path": str(tmp_path / "missing.pdf"), "download_status": "downloaded"}
    result = inspect_item(q, m, tmp_path / "worker", corpus_root=tmp_path)
    assert result["status"] == "HOLD"
    assert "MISSING" in result["reason"]


def _tiny_stage_inputs(tmp_path: Path) -> tuple[Path, dict, list]:
    root = tmp_path / "corpus"
    root.mkdir()
    manifest = root / "manifest.jsonl"
    manifest.write_bytes((json.dumps({"event_id": "1", "arxiv_id": "2601.00001v1",
                                      "sha256": "a" * 64, "download_status": "downloaded"}) + "\n").encode())
    queues = []
    for number in range(1, 6):
        worker = root / f"nips-arxiv-{number:02d}"
        worker.mkdir()
        queue = worker / "queue.jsonl"
        queue.write_bytes(b"{}\n")
        queues.append((worker.name, {"path": str(queue), "sha256": _sha(queue.read_bytes())}))
    return root, {"path": str(manifest), "sha256": _sha(manifest.read_bytes())}, queues


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_stage_writes_one_hold_and_receipt_without_invented_card(tmp_path: Path) -> None:
    root, manifest, queues = _tiny_stage_inputs(tmp_path)
    output = root / "private-staging"
    result = stage(manifest, queues, output, corpus_root=root, expected_downloaded=1)
    assert result["ready_count"] == 0 and result["hold_count"] == 1
    assert result["router_result_authenticated"] is False
    assert result["router_qa_authenticated"] is False
    assert result["card_job_bound"] is False
    assert json.loads((output / "holds.v1.jsonl").read_text())["reason"] == "MISSING_QUEUE"
    assert not (output / "ready.v1.jsonl").read_bytes()
    assert (output / "receipt.json").is_file()
    with pytest.raises(FileExistsError):
        stage(manifest, queues, output, corpus_root=root, expected_downloaded=1)


def test_stage_rejects_manifest_outside_corpus_root(tmp_path: Path) -> None:
    root, manifest, queues = _tiny_stage_inputs(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(Path(manifest["path"]).read_bytes())
    manifest = {"path": str(outside), "sha256": _sha(outside.read_bytes())}
    with pytest.raises(ValueError, match="manifest outside corpus root"):
        stage(manifest, queues, root / "private-staging", corpus_root=root, expected_downloaded=1)
    assert not (root / "private-staging").exists()
