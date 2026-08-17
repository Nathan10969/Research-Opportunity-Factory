import hashlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pnnl_pilot.authorized_download import (
    DownloadItem,
    classify_filename,
    download_item,
    extract_authorized_items,
    hash_prefix_from_filename,
    redacted_inventory,
)


class AuthorizedInventoryTests(unittest.TestCase):
    def test_markdown_duplicates_collapse_to_one_authorized_item(self):
        url = "https://dinov3.llamameta.net/model-1234abcd.pth"
        items = extract_authorized_items(f"[{url}]({url})\n{url}")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].filename, "model-1234abcd.pth")

    def test_non_meta_hosts_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_authorized_items("https://example.com/model-1234abcd.pth")

    def test_filename_classification_and_hash_prefix(self):
        self.assertEqual(classify_filename("dinov3_vitb16_pretrain-1234abcd.pth"), "backbones")
        self.assertEqual(classify_filename("dinov3_vit7b16_coco_detr_head-1234abcd.pth"), "heads")
        self.assertEqual(hash_prefix_from_filename("model-1234abcd.pth"), "1234abcd")

    def test_redacted_inventory_never_contains_authorized_urls(self):
        url = "https://dinov3.llamameta.net/model-1234abcd.pth"
        records = redacted_inventory(extract_authorized_items(url))
        self.assertNotIn("url", records[0])
        self.assertNotIn("http", str(records).lower())


class RangeHandler(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):
        start = 0
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            self.send_response(206)
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}",
            )
        else:
            self.send_response(200)
        body = self.payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class ResumableDownloadTests(unittest.TestCase):
    def test_resumes_partial_file_and_atomically_finishes(self):
        payload = (b"DINOv3-material-image" * 4096) + b"end"
        digest = hashlib.sha256(payload).hexdigest()
        filename = f"fixture-{digest[:8]}.pth"
        RangeHandler.payload = payload
        server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                part = root / f"{filename}.part"
                part.write_bytes(payload[:1000])
                item = DownloadItem(
                    url=f"http://127.0.0.1:{server.server_port}/weight",
                    filename=filename,
                    category="backbones",
                    hash_prefix=digest[:8],
                )
                result = download_item(item, root)
                final = root / filename
                self.assertEqual(final.read_bytes(), payload)
                self.assertFalse(part.exists())
                self.assertEqual(result.resumed_from, 1000)
                self.assertEqual(result.sha256, digest)
                self.assertTrue(result.prefix_verified)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
