# DINOv3 Authorized Full Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably download and validate all 17 unique official DINOv3 files from the user's authorized URL list without persisting the URLs.

**Architecture:** A small tested library parses and validates the authorized list, classifies files, extracts the official hash prefix, and performs resumable atomic transfers. A CLI runs two workers, writes redacted progress, and produces checksum evidence under the external model-weight root.

**Tech Stack:** Python standard library, `urllib`, `hashlib`, `concurrent.futures`, `unittest`.

---

### Task 1: Redacted URL inventory

**Files:**
- Create: `src/pnnl_pilot/authorized_download.py`
- Create: `tests/test_authorized_download.py`

- [ ] Write failing tests proving Markdown duplicates collapse to one URL,
  non-Meta hosts are rejected, head/backbone classification is deterministic,
  and the eight-hex filename suffix is extracted.
- [ ] Run `python -m unittest tests.test_authorized_download -v`; expect import
  failure because the module does not exist.
- [ ] Implement `extract_authorized_items`, `classify_filename`, and
  `hash_prefix_from_filename` without logging URL values.
- [ ] Re-run the focused tests; expect all tests to pass.

### Task 2: Resumable atomic transfer

**Files:**
- Modify: `src/pnnl_pilot/authorized_download.py`
- Modify: `tests/test_authorized_download.py`

- [ ] Add a failing local HTTP-server test starting from a partial file and
  asserting byte-exact completion plus atomic final naming.
- [ ] Implement Range resume, safe restart when Range is ignored, SHA-256, and
  filename-prefix verification.
- [ ] Re-run focused tests and the complete suite.

### Task 3: Background downloader and evidence

**Files:**
- Create: `scripts/download_dinov3_authorized.py`

- [ ] Implement a two-worker CLI with a 200 GiB free-space floor, redacted
  `progress.json`, `download_manifest.csv`, and `SHA256SUMS.txt`.
- [ ] Compile all source and script files and run the complete unit-test suite.
- [ ] Start it with the user attachment and the frozen D-drive output root in a
  hidden process; save only its PID and redacted stdout/stderr.
- [ ] Verify the process is active and that the progress artifact contains 17
  unique scheduled filenames and no `http` text.

### Task 4: Freeze the experiment matrix

**Files:**
- Modify: `docs/experiments/pnnl_compute_and_ablation_protocol.md`

- [ ] Record DINOv3 ViT S/S+/B/L/H+/7B and ConvNeXt T/S/B/L as backbone-scale
  ablations, with DINOv2 matched-capacity controls.
- [ ] Keep native backbone width separate from released subspace dimensions
  `4/8/16/32/64/128/raw` and from CLS/mean-patch pooling.
- [ ] Commit the design, implementation, tests, and protocol without any URL or
  weight file.
