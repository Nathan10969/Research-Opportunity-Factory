from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


AUTHORIZED_HOST = "dinov3.llamameta.net"
URL_PATTERN = re.compile(r"https://[^\s\]\)>\"']+")
HASH_PATTERN = re.compile(r"-([0-9a-fA-F]{8})\.pth$")


@dataclass(frozen=True)
class DownloadItem:
    url: str
    filename: str
    category: str
    hash_prefix: str


@dataclass(frozen=True)
class DownloadResult:
    filename: str
    category: str
    bytes: int
    sha256: str
    prefix_verified: bool
    resumed_from: int


def classify_filename(filename: str) -> str:
    lowered = filename.lower()
    return "heads" if "head" in lowered or "encoder" in lowered else "backbones"


def hash_prefix_from_filename(filename: str) -> str:
    match = HASH_PATTERN.search(filename)
    if match is None:
        raise ValueError("official filename lacks an eight-hex hash suffix")
    return match.group(1).lower()


def extract_authorized_items(text: str) -> list[DownloadItem]:
    items = []
    seen = set()
    for url in URL_PATTERN.findall(text):
        url = url.rstrip(",;")
        if url in seen:
            continue
        parsed = urlparse(url)
        filename = unquote(Path(parsed.path).name)
        if parsed.scheme != "https" or parsed.hostname != AUTHORIZED_HOST:
            raise ValueError("URL is outside the authorized DINOv3 host")
        if not filename.endswith(".pth") or Path(filename).name != filename:
            raise ValueError("authorized URL does not have a safe .pth basename")
        items.append(
            DownloadItem(
                url=url,
                filename=filename,
                category=classify_filename(filename),
                hash_prefix=hash_prefix_from_filename(filename),
            )
        )
        seen.add(url)
    if not items:
        raise ValueError("no authorized DINOv3 URLs were found")
    return sorted(items, key=lambda item: (item.category, item.filename))


def redacted_inventory(items: list[DownloadItem]) -> list[dict[str, str]]:
    return [
        {
            "filename": item.filename,
            "category": item.category,
            "hash_prefix": item.hash_prefix,
        }
        for item in items
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_item(
    item: DownloadItem,
    destination: Path,
    progress: Callable[[str, int, int | None], None] | None = None,
) -> DownloadResult:
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / item.filename
    part_path = destination / f"{item.filename}.part"
    if final_path.is_file():
        digest = _sha256(final_path)
        if not digest.startswith(item.hash_prefix):
            raise ValueError("existing final file fails official hash-prefix check")
        return DownloadResult(
            item.filename,
            item.category,
            final_path.stat().st_size,
            digest,
            True,
            final_path.stat().st_size,
        )

    resumed_from = part_path.stat().st_size if part_path.is_file() else 0
    headers = {"User-Agent": "Mozilla/5.0 DINOv3-authorized-downloader"}
    if resumed_from:
        headers["Range"] = f"bytes={resumed_from}-"
    response = urlopen(Request(item.url, headers=headers), timeout=120)
    try:
        status = getattr(response, "status", response.getcode())
        append = resumed_from > 0 and status == 206
        if resumed_from > 0 and not append:
            resumed_from = 0
        content_range = response.headers.get("Content-Range")
        expected_total = None
        if content_range and "/" in content_range:
            total_text = content_range.rsplit("/", 1)[1]
            if total_text.isdigit():
                expected_total = int(total_text)
        elif response.headers.get("Content-Length", "").isdigit():
            expected_total = resumed_from + int(response.headers["Content-Length"])
        mode = "ab" if append else "wb"
        downloaded = resumed_from
        with part_path.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(item.filename, downloaded, expected_total)
    finally:
        response.close()
    if expected_total is not None and part_path.stat().st_size != expected_total:
        raise IOError("downloaded byte count does not match the server total")
    digest = _sha256(part_path)
    prefix_verified = digest.startswith(item.hash_prefix)
    if not prefix_verified:
        raise ValueError("downloaded file fails official hash-prefix check")
    part_path.replace(final_path)
    return DownloadResult(
        item.filename,
        item.category,
        final_path.stat().st_size,
        digest,
        prefix_verified,
        resumed_from,
    )
