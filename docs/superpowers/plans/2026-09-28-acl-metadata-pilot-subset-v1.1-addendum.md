# ACL metadata pilot subset mode — implementation plan addendum v1.1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Emit a parent-linked ten-item ACL pilot from the authoritative pinned 670-row proof without cropping or forging its proof receipt, and accept only correctly normalized page-1 evidence plus independently authored exact per-item reviews.

**Architecture:** Keep the full 670-row proof and receipt as the sole parent authority. Add an explicit hash-pinned selection list for the pre-frozen ten IDs; validate all 670 parent source pins before projecting only those ten. Normalize nested Task 4 page-1 evidence through a versioned, schema-explicit adapter. Do not derive approval from aggregate reviewer decisions: require exact reviewer-authored per-item records. `SUPPORTED` remains provisional behind ROOT's independent receipt/task-provenance and native visual QA gate.

**Tech Stack:** Python 3.12, stdlib JSON/hashlib/subprocess/pathlib, Poppler `pdftotext`, pytest, existing `idea_factory` validator.

---

Design authority: `docs/superpowers/specs/2026-09-28-acl-derived-metadata-admission-design.md` plus addendum `docs/superpowers/specs/2026-09-28-acl-metadata-pilot-subset-v1.1-addendum.md`. Execute only in `F:\LLM_Evoke\idea_factory\.worktrees\full-corpus-20260927-1040`. This plan does not authorize a real pilot run or any Card/public/graph promotion. Never modify the frozen proof, source queues/manifests/HTML/PDFs, prior receipts, or existing Card/task/raw/note files.

## File map and boundaries

- Modify `scripts/build_acl_metadata_projection.py`: add a strict optional pilot-selection input while keeping full-mode count and parent receipt validation at 670; validate all 670 source pins, then emit only selected rows in subset mode. Add parent/selection provenance to the child receipt.
- Create `src/idea_factory/acl_task4_evidence_adapter.py`: after first inventorying actual Task 4 schemas, provide deterministic, versioned normalization from nested page-1 records to the exact Task 2 witness schema. It may copy only explicit unambiguous per-ID evidence; missing or contradictory values become HOLD.
- Do not auto-convert aggregate `independent_reviewer_decisions.v1.jsonl` into an approval record. Continue to consume exact Task 2 review rows authored per item by an independent reviewer. Aggregate decision data may be referenced as provenance only.
- Extend `tests/test_acl_metadata_projection_cli.py`; add `tests/test_acl_task4_evidence_adapter.py` for the normalization contract.
- Do not edit the original design or Task 1–3 plan text; this addendum is versioned and parent-linked.

### Task A: Pin the actual Task 4 evidence schemas before coding

**Files:** Read-only inspection of the versioned Task 4 page-1 witness, aggregate reviewer decision, and receipt artifacts named in the approved workflow; document the exact schemas and per-ID paths in adapter tests/docstrings.

- [ ] **Step 1: Inventory representative, missing, conflicting, and negative-control records.** Record exact JSON key paths and distinguish values copied from values recomputed from pinned bytes. Do not write to the input artifacts.
- [ ] **Step 2: Define the normalization map.** Map only available per-ID values for `item_id`, `pdf_sha256`, `page1_text_sha256`, `page1_text`, `pdftotext_version`, and `witness_author_id`. Require the canonical item ID and PDF SHA to join to the parent proof. Recompute the text SHA from exact UTF-8 text. If any value cannot be derived unambiguously from a pinned record, specify a reasoned HOLD and require new witness evidence; do not infer from path/name or adjacent records.
- [ ] **Step 3: Lock reviewer-row provenance.** Record how the reviewer-authored exact rows bind proof hash, PDF hash, and normalized page-1 hash. The aggregate decision artifact alone must be tested as insufficient to create a SUPPORTED review row.

### Task B: Implement strict full-parent selection semantics

**Files:** Modify `scripts/build_acl_metadata_projection.py`; extend `tests/test_acl_metadata_projection_cli.py`.

- [ ] **Step 1: Add selection-list test fixtures.** Keep the proof fixture's parent row count/receipt at 670 in a scalable fixture. The selected-list fixture contains the ten exact Task 4 IDs, one per line. The expected output in pilot mode is 10 rows; full mode remains 670.
- [ ] **Step 2: Add failing tests for parent-linked behavior.** Reject a ten-row proof file or derived ten-row receipt, wrong parent receipt/hash/count, wrong/duplicate/missing/additional/noncanonical/`TEST_ONLY` selection IDs, selection-list path outside the root or through a reparse point, selected IDs absent from the parent, and a stale selection-list hash. Assert no output directory/success receipt and unchanged originals on each failure. Assert production CLI cardinalities are only `670→670` and `670→10`; no arbitrary expected count may make a three-row or other partial run look valid.
- [ ] **Step 3: Implement the mode boundary.** First read back the authoritative frozen 670 proof-only receipt and establish the expected parent receipt/proof hashes from that run, not by hashing an arbitrary candidate file. Without a selection list, require that exact original 670-row proof/receipt and emit 670. With a selection list, still require and validate that same 670-row parent and rehash every referenced source file and physical queue/manifest row. Validate that the selection list is under the explicit corpus root with no reparse path component; parse strict UTF-8 with exactly the frozen ten canonical unique IDs; verify its caller-supplied SHA before reading and again before writing; confirm each selected ID exists once in the parent; then project exactly the selected ten. Never accept a cropped proof/receipt as the parent.
- [ ] **Step 4: Keep all selected rows visible.** Resolve per-selected-row page-1/review evidence after full-parent validation. Missing or held evidence creates a reasoned HOLD row; no selected ID can disappear. Reject evidence IDs outside the parent, duplicates, and mismatched proof/PDF/page-1 hashes. Sort output by canonical item ID and require exact selected-ID set equality in both JSONL artifacts.
- [ ] **Step 5: Bind the child receipt.** Include parent receipt and proof path/hash/count, selected list path/hash, selected IDs/count, `run_mode`, output count/hashes, disposition counts, toolchain pins, and false promotion flags. Preserve exclusive output creation. The pilot receipt is explicitly a child of the 670 parent and is never accepted as a proof-only parent receipt.
- [ ] **Step 6: Add the nine/one conservation test.** With 10 selected real-shaped fixture IDs, provide nine exact valid reviews and one missing or HOLD review; assert ten diagnostic rows, ten projection rows, nine `SUPPORTED`, one `METADATA_HOLD`, selected-ID set equality, and no synthetic/unselected IDs. This fixture count is not a prediction for the real cohort.
- [ ] **Step 7: Run Task 3 regressions.** Run the existing Task 3 CLI/admission/witness suites plus new selection tests; check `git diff --check`. Verify both full mode (670 rows) and subset mode (10 rows) preserve source bytes.

### Task C: Add nested Task 4 page-1 normalization without review inference

**Files:** Create `src/idea_factory/acl_task4_evidence_adapter.py`; create `tests/test_acl_task4_evidence_adapter.py`; integrate adapter in `scripts/build_acl_metadata_projection.py` only after Task A's schema inventory.

- [ ] **Step 1: Test actual nested artifacts.** Use redacted exact-shape fixtures from the pinned Task 4 artifacts. Cover correct per-ID mapping, wrong ID, wrong PDF hash, malformed/missing page text, incorrect supplied text hash, conflicting duplicate representations, and missing/ambiguous witness author or extractor version. Missing evidence must resolve to HOLD, not invented values.
- [ ] **Step 2: Prohibit automatic reviewer approval conversion.** Test that the aggregate reviewer-decision artifact by itself never satisfies Task 2's exact review schema and never yields SUPPORTED. Require a separate reviewer-authored per-item row with exact proof-row hash, PDF hash, page-1 text hash, reviewer ID, UTC review time, reason, explicit native-PDF identity verification, and exact four field verdicts. Reject stale/cross-item/cross-PDF/cross-witness joins and reviewer/witness-author collision.
- [ ] **Step 3: Normalize deterministically.** Copy only the documented per-ID witness fields, compute the page-1 text hash from exact UTF-8 bytes, and retain raw input hashes plus adapter protocol hash in the run receipt. Do not derive human intent/consent or field verdicts from aggregate labels. If a required witness or review value is unavailable, emit the corresponding HOLD.
- [ ] **Step 4: Run the focused adapter and CLI tests plus the Task 3 regression suite.** Check diff and confirm no input artifact was edited.

### Task D: Pilot execution remains a separate approval gate

No real pilot is part of this documentation/implementation addendum. Before a future real run, freeze and hash the selection list before inspecting projected outcomes; inventory and pin the actual nested witness and aggregate decision artifacts; obtain exact independent reviewer rows where existing records do not already contain them; then run into a fresh, exclusive, versioned output directory. ROOT must independently verify reviewer receipt/task provenance and native visual PDF checks per batch before any downstream Card use. Report 10/10 row conservation, dispositions, unresolved holds, parent/child hashes, unchanged original bytes, and `downstream_card_use_approved=false` until that review is complete. Synthetic `TEST_ONLY` controls stay outside both the 670-row parent and the real ten-row output.
