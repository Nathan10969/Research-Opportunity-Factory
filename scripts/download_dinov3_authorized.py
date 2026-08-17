from __future__ import annotations

import argparse
import csv
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from pnnl_pilot.authorized_download import (
    DownloadItem,
    download_item,
    extract_authorized_items,
    redacted_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attachment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--expected-count", type=int, default=17)
    parser.add_argument("--free-floor-gib", type=float, default=200.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = extract_authorized_items(args.attachment.read_text(encoding="utf-8"))
    if len(items) != args.expected_count:
        raise RuntimeError(
            f"authorized inventory count {len(items)} != {args.expected_count}"
        )
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    progress_tmp = args.output_root / "progress.json.tmp"
    state = {
        "status": "running",
        "scheduled": len(items),
        "items": {
            item.filename: {
                "category": item.category,
                "hash_prefix": item.hash_prefix,
                "status": "queued",
                "downloaded_bytes": 0,
                "expected_bytes": None,
            }
            for item in items
        },
    }
    lock = threading.Lock()
    last_write = {"time": 0.0}

    def write_state(force: bool = False) -> None:
        now = time.monotonic()
        with lock:
            if not force and now - last_write["time"] < 2.0:
                return
            progress_tmp.write_text(
                json.dumps(state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            progress_tmp.replace(progress_path)
            last_write["time"] = now

    def progress(filename: str, downloaded: int, expected: int | None) -> None:
        with lock:
            state["items"][filename]["status"] = "downloading"
            state["items"][filename]["downloaded_bytes"] = downloaded
            state["items"][filename]["expected_bytes"] = expected
        write_state()

    def run_one(item: DownloadItem) -> dict[str, object]:
        free_gib = shutil.disk_usage(args.output_root).free / (1024**3)
        if free_gib < args.free_floor_gib:
            return {
                "filename": item.filename,
                "category": item.category,
                "status": "free_space_floor",
                "bytes": 0,
                "sha256": "",
                "prefix_verified": False,
                "resumed_from": 0,
                "error_type": "FreeSpaceFloor",
            }
        try:
            result = download_item(
                item, args.output_root / item.category, progress=progress
            )
            return {**asdict(result), "status": "passed", "error_type": ""}
        except Exception as error:
            return {
                "filename": item.filename,
                "category": item.category,
                "status": "failed",
                "bytes": 0,
                "sha256": "",
                "prefix_verified": False,
                "resumed_from": 0,
                "error_type": type(error).__name__,
            }

    write_state(force=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, item): item for item in items}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with lock:
                entry = state["items"][result["filename"]]
                entry["status"] = result["status"]
                entry["downloaded_bytes"] = result["bytes"]
                entry["sha256"] = result["sha256"]
                entry["prefix_verified"] = result["prefix_verified"]
                entry["error_type"] = result["error_type"]
            write_state(force=True)
            print(
                f"{result['status']} {result['filename']} bytes={result['bytes']}",
                flush=True,
            )

    results.sort(key=lambda row: (row["category"], row["filename"]))
    manifest_path = args.output_root / "download_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    checksum_lines = [
        f"{row['sha256']}  {row['category']}/{row['filename']}"
        for row in results
        if row["status"] == "passed"
    ]
    (args.output_root / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + ("\n" if checksum_lines else ""),
        encoding="utf-8",
    )
    passed = sum(row["status"] == "passed" for row in results)
    state["status"] = "passed" if passed == len(items) else "partial"
    state["passed"] = passed
    state["failed"] = len(items) - passed
    state["redacted_inventory"] = redacted_inventory(items)
    write_state(force=True)
    print(
        f"FINAL status={state['status']} passed={passed}/{len(items)}",
        flush=True,
    )
    return 0 if passed == len(items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
