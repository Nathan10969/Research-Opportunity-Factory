# ACL Derived Metadata Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an exact-ID, evidence-bound ACL 2026 metadata correction path that leaves frozen intake bytes untouched and makes reviewed rows usable by new Card tasks.

**Architecture:** Reuse the already-tested proof-only official-HTML witness, add a separate native-PDF/reviewer admission record, and emit a reviewed field-level projection into a distinct run. The frozen 2026 ACL queues and Card tasks remain immutable. Conflicts stay visible as HOLD; neither metadata correction nor Card format validity implies scientific or graph approval.

**Tech Stack:** Python 3.12, stdlib HTML/JSON/hashlib/subprocess, Poppler `pdftotext`, pytest, existing `idea_factory` and pinned `validate_cards.py` adapter; Windows PowerShell commands below.

---

Design authority: `docs/superpowers/specs/2026-09-28-acl-derived-metadata-admission-design.md` at commit `5b6c5edcdc4db6e8cf679c0693dcdbdb835fb3c6`. Execute in `F:\LLM_Evoke\idea_factory\.worktrees\full-corpus-20260927-1040`; do not edit the proof-only artifacts under `F:\LLM_Evoke\runs\parallel24-20260927-1340`. A stage is complete only when its exact tests and readback pass, not when code exists.

## File map and boundaries

- Create `src/idea_factory/acl_official_metadata_witness.py` by byte-identical import of the frozen proof-only helper (original SHA-256 `d76fbe4ae8a4f8e8ddb1fc2ce6860c30e21d4c06c07ea15b7a3b9daca72e2cd5`). It owns HTML parsing, exact-ID/source binding, and proof-only input pin checks; no admission flags.
- Create `tests/test_acl_official_metadata_witness.py` from the existing 25-pass/1-skip proof suite, changing imports only. It locks the imported helper's behavior.
- Create `src/idea_factory/acl_metadata_admission.py` and `tests/test_acl_metadata_admission.py`. It validates one independent reviewer decision against one immutable proof row and one native-PDF page-1 evidence record, then constructs a narrow projection row. It does not create Cards or touch sources.
- Create `scripts/build_acl_metadata_projection.py` and `tests/test_acl_metadata_projection_cli.py`. It reads the 670 proof rows, page-1 witnesses, and reviewed decisions; emits a joined diagnostic `acl_metadata_witness.v1.jsonl` and exactly one `SUPPORTED` or reasoned `METADATA_HOLD` row per source row in `acl_metadata_projection.v1.jsonl` into a new exclusive directory; and writes a receipt with all input/output hashes.
- Existing external `F:\LLM_Evoke\runs\full-corpus-paper-cards-20260926\tools\validate_cards.py` remains pinned; its `audit_task` checks `task.title/venue/year` against new raw Card content. Do **not** modify general `src/idea_factory/cards.py`: that module's Card jobs bind note/prompt, not ACL source metadata.

### Task 1: Import the frozen proof-only witness without changing semantics

**Files:** Create `src/idea_factory/acl_official_metadata_witness.py`; create `tests/test_acl_official_metadata_witness.py`.

- [ ] **Step 1: Verify the exact frozen helper and tests.**

```powershell
Get-FileHash -LiteralPath 'F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\tools\acl_official_metadata_witness.py' -Algorithm SHA256
& 'F:\LLM_Evoke\idea_factory\.venv\Scripts\python.exe' -X utf8 -m pytest -q 'F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\tests\test_acl_official_metadata_witness.py'
```

Expected: helper SHA above; `25 passed, 1 skipped`. If the source hash changed, stop instead of copying a moving dependency.

- [ ] **Step 2: Copy the two existing files mechanically and adjust only the test import with `apply_patch`.**

```powershell
Copy-Item -LiteralPath 'F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\tools\acl_official_metadata_witness.py' -Destination 'src\idea_factory\acl_official_metadata_witness.py'
Copy-Item -LiteralPath 'F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\tests\test_acl_official_metadata_witness.py' -Destination 'tests\test_acl_official_metadata_witness.py'
```

In the copied test, replace both `import acl_official_metadata_witness as witness_module` and `from acl_official_metadata_witness import (` with the corresponding `idea_factory.acl_official_metadata_witness` imports. No production helper edits in this task.

- [ ] **Step 3: Prove byte-preserving helper import and run tests.**

```powershell
Get-FileHash -LiteralPath 'src\idea_factory\acl_official_metadata_witness.py' -Algorithm SHA256
& 'F:\LLM_Evoke\idea_factory\.venv\Scripts\python.exe' -X utf8 -m pytest -q tests\test_acl_official_metadata_witness.py
git diff --check
```

Expected: imported helper SHA matches original; 25 pass/1 skip; diff check clean. Commit only these two files:

```powershell
git add -- src/idea_factory/acl_official_metadata_witness.py tests/test_acl_official_metadata_witness.py
git commit -m "test: pin ACL official metadata witness"
```

### Task 2: Define an admission row and its fail-closed validator

**Files:** Create `src/idea_factory/acl_metadata_admission.py`; create `tests/test_acl_metadata_admission.py`.

Admission input is a proof row, a PDF-page witness, and an **independent** reviewer record. The PDF-page witness has exactly `item_id`, `pdf_sha256`, `page1_text_sha256`, `page1_text`, `pdftotext_version`, and `witness_author_id`. Its text is obtained from the already-pinned PDF with `pdftotext -f 1 -l 1 -layout <pdf> -`, not from a filename. The reviewer record has exactly `item_id`, `proof_row_sha256`, `pdf_sha256`, `page1_text_sha256`, `decision`, `reviewer_id`, `reviewed_at_utc`, `reason`, `native_pdf_identity_verified`, and `field_verdicts`; `decision` is `SUPPORTED` or `METADATA_HOLD`. The validator must reject `reviewer_id == witness_author_id`. A `SUPPORTED` record requires `native_pdf_identity_verified: true` and `field_verdicts` equal to `{"title":"PASS","ordered_authors":"PASS","venue":"PASS","year":"PASS"}`. A reviewer record can be written only after the independent native-page check; an LLM vote alone is not a reviewer record. The frozen proof row actually contains `current_expected_title/current_source_title`, `official_html_witness.title/authors/publication_section_witness`, queue path/file hash/one-based line/raw-line hash, manifest path/file hash/line/raw-line hash, HTML path/hash, PDF path/hash, cache receipt/text path/hash, and `outcome`; it does **not** carry pages. Thus pages remain unknown unless a separately reviewed official page-range witness is added in a later version, and v1 must not invent them.

- [ ] **Step 1: Write failing tests for exact joins and negative cases.** Tests must construct one synthetic proof row and mutate each of: item ID, proof hash, PDF hash, page-1 text hash, empty reviewer, `reviewer_id == witness_author_id`, `native_pdf_identity_verified=false`, title disagreement, author disagreement, and missing venue/year witness. Every mutation must return a reasoned HOLD or raise before projection; none may return `SUPPORTED`. Duplicate-reviewer rows are tested in Task 3, where the collection is visible. Include one passing case whose only approved replacement fields are title, ordered authors, venue, and year; pages must be absent in v1.

Use this exact positive assertion in `tests/test_acl_metadata_admission.py`, with the fixture builders supplying the frozen proof-row keys listed above:

```python
def test_supported_projection_has_no_unwitnessed_fields(proof_row, page1_row, review_row):
    row = validate_admission(proof_row, page1_row, review_row)
    assert row["status"] == "SUPPORTED"
    assert set(row["fields"]) == {"title", "ordered_authors", "venue", "year"}
    assert row["fields"]["title"]["before"] == "bib"
    assert row["fields"]["title"]["approved"] == proof_row["official_html_witness"]["title"]
    assert "pages" not in row["fields"]
    assert row["source_admission_approved"] is False
    assert row["human_approved"] is False
    assert row["graph_ingested"] is False

def test_reviewer_cannot_approve_own_witness(proof_row, page1_row, review_row):
    review_row["reviewer_id"] = page1_row["witness_author_id"]
    with pytest.raises(ValueError, match="independent reviewer"):
        validate_admission(proof_row, page1_row, review_row)
```
- [ ] **Step 2: Run the exact new test file and confirm RED.**

```powershell
& 'F:\LLM_Evoke\idea_factory\.venv\Scripts\python.exe' -X utf8 -m pytest -q tests\test_acl_metadata_admission.py
```

Expected: import/function-not-found failure, not a fixture setup failure.

- [ ] **Step 3: Implement `validate_admission(proof: dict, page1: dict, review: dict) -> dict`.** Use canonical UTF-8 JSON hashing (`sort_keys=True`, `ensure_ascii=False`, compact separators) for the proof-row and review-row digests. Reject non-exact key sets, ID/hash mismatch, non-`bib` proof scope, absent unique official witness, missing page-1 text, and a reviewer status that is not `SUPPORTED` or `METADATA_HOLD`. Return an immutable-style plain dict with original queue path/file hash/physical row/row hash, exact ACL ID and PDF hash, proof/review hashes, field-level **before/proposed/approved/disposition** for title, ordered authors, venue, and year, plus overall status. v1 never outputs pages because the frozen proof row has no page-range witness. A HOLD row must carry `hold_reason`, must carry no approved replacement fields, and must never be omitted. `SUPPORTED` requires the independently verified PDF flag and the four field verdicts; it cannot set scientific, human, source, or graph approval.
- [ ] **Step 4: Run GREEN and mutation tests.**

```powershell
& 'F:\LLM_Evoke\idea_factory\.venv\Scripts\python.exe' -X utf8 -m pytest -q tests\test_acl_metadata_admission.py tests\test_acl_official_metadata_witness.py
git diff --check
```

Expected: all new cases pass and the frozen witness suite remains 25 pass/1 skip. Commit this component and its test: `git commit -m "feat: validate ACL metadata admission rows"`.

### Task 3: Exclusive projection builder with all-row conservation

**Files:** Create `scripts/build_acl_metadata_projection.py`; create `tests/test_acl_metadata_projection_cli.py`.

- [ ] **Step 1: Write failing CLI tests.** A three-row fixture contains one `SUPPORTED` row, one reviewer HOLD, and one missing review. Assert three diagnostic witness rows and three projection rows, with the missing review converted to `METADATA_HOLD: REVIEW_MISSING`; no source row is dropped. Reject duplicate source IDs, duplicate review IDs, altered proof file hash, altered queue-row hash, output directory that exists, output path outside the explicit run root, and symlink/reparse path components. Verify that original fixture bytes are unchanged after both success and failure. The core conservation assertion is:

```python
projection = [json.loads(line) for line in (out / "acl_metadata_projection.v1.jsonl").read_text(encoding="utf-8").splitlines()]
diagnostic = [json.loads(line) for line in (out / "acl_metadata_witness.v1.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(projection) == len(diagnostic) == 3
assert len({row["item_id"] for row in projection}) == 3
assert {row["status"] for row in projection} == {"SUPPORTED", "METADATA_HOLD"}
assert next(row for row in projection if row["item_id"] == missing_id)["hold_reason"] == "REVIEW_MISSING"
```
- [ ] **Step 2: Run RED.**

```powershell
& 'F:\LLM_Evoke\idea_factory\.venv\Scripts\python.exe' -X utf8 -m pytest -q tests\test_acl_metadata_projection_cli.py
```

Expected: missing builder/module, not a broken fixture.

- [ ] **Step 3: Implement CLI with required arguments `--proof`, `--proof-sha256`, `--proof-receipt`, `--proof-receipt-sha256`, `--page1-witnesses`, `--page1-sha256`, `--reviews`, `--reviews-sha256`, `--corpus-root`, `--output-dir`, `--expected-count`.** The proof receipt must identify the exact proof-row file/hash and 670-row scope. Rehash every input before reading and again before output. For every proof row, independently rehash its `queue_path`/`queue_file_sha256`, the physical line with the frozen CRLF convention, `manifest_path`/`manifest_file_sha256`, `official_html_path`/`official_html_sha256`, and `pdf_path`/`pdf_sha256`; reject any drift before creating output. Require `--expected-count` exact unique proof IDs; map at most one page-1 witness and one review to each proof ID; call `validate_admission` only when both exist; materialize explicit HOLD rows for missing evidence or review; sort by exact ACL ID; create output directory with exclusive creation; write `acl_metadata_witness.v1.jsonl`, `acl_metadata_projection.v1.jsonl`, and `receipt.json` using exclusive file creation. The diagnostic witness row binds the proof, page-1 evidence or its absence, review reference or its absence, and each field's before/proposed/approved/disposition; it never acts as approval. Receipt records protocol hash, source/page-1/reviewer/witness/projection hashes, counts by disposition, and `source_admission_approved=false`, `human_approved=false`, `graph_ingested=false`. On exception, do not overwrite or delete any existing output; do not emit a success receipt.
- [ ] **Step 4: Run GREEN and full local regression.**

```powershell
& 'F:\LLM_Evoke\idea_factory\.venv\Scripts\python.exe' -X utf8 -m pytest -q tests\test_acl_metadata_projection_cli.py tests\test_acl_metadata_admission.py tests\test_acl_official_metadata_witness.py
git diff --check
```

Expected: all new tests pass; original helper suite unchanged. Commit: `git commit -m "feat: build reviewed ACL metadata projection"`.

### Task 4: Ten-item real pilot and independent review

**Files:** New versioned outputs only under `F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\acl-metadata-admission-pilot-20260928-*`; do not edit the original 670-row proof, queues, or Cards.

- [ ] **Step 1: Freeze ten real IDs before viewing projected outcomes:** `2026.findings-acl.1077`, `.1105`, `.1174`, `.1266`, `.1320`, `.135`, `.1371`, `.1388`, `.1412`, and `.1530` (all with the same `2026.findings-acl` prefix). These are exact rows in the frozen 670-row proof artifact. Exercise absent/duplicate/wrong anchor, PDF/title conflict, ordered-author conflict, and an already-correct title with separate synthetic `TEST_ONLY` negative controls; synthetic controls never enter the 670-row real projection.
- [ ] **Step 2: For each real pilot ID, independently read the pinned PDF page 1 natively and run `pdftotext -f 1 -l 1 -layout <pdf> -`.** Hash the exact extracted bytes, record the ordered author check and each field verdict, and independently rehash proof/queue/PDF/cache/note/task. A reviewer cannot approve their own witness construction.
- [ ] **Step 3: Build the ten-row pilot projection into a new output directory, then independently QA the receipt and all ten rows.** Expected: 10/10 source-row conservation, 0 silently dropped items, every negative control HOLD, no original hash changed. If any case fails, stop; repair code in a new commit and rerun into a new versioned directory, never overwrite a failed pilot.
- [ ] **Step 4: Run the pinned Card adapter on a **new** pilot task/raw pair for each approved ID.** New task title/venue/year come only from the approved projection; note and PDF identities stay bound to the original item; raw Card content is newly authored against that task and must pass `validate_cards.py::audit_task`. No old task/raw/receipt is copied over with altered metadata, and no held item is misrepresented as a valid Card.

### Task 5: Cover all 670 candidates and close the ACL remainder

- [ ] **Step 1: Read back the frozen proof-only run receipt** `F:\LLM_Evoke\runs\parallel24-20260927-1340\engineering\ACL670_WITNESS_FINAL_RUN_RECEIPT.json`; verify its input pins and 670 unique rows. Work in ≤10-item reviewer batches with immutable decision files and independent QA. Include the currently open ACL11/ACL13 `bib` items explicitly; keep all conflict cases as HOLD.
- [ ] **Step 2: Produce 670 diagnostic witness rows and a 670-row projection with 670 distinct exact IDs and reasoned disposition for every row.** Independently rehash original 13 queues, ACL manifest, official HTML, every referenced PDF, and original Card/task/raw files after the build. The count `SUPPORTED + METADATA_HOLD` must equal 670; the projection must have no other disposition.
- [ ] **Step 3: For the strict checkpoint's 31 ACL nonterminal items, separately reconcile transfer ownership and build new exact Card tasks only where a reviewed projection and source-content artifacts are complete.** A missing owner receipt, abstract numerical conflict, or source identity conflict remains HOLD even when title metadata is fixed. Re-run the pinned `audit_task` for each new task and then the 3,908-item PM checker; do not infer scientific approval from `SCHEMA_VALID`.
- [ ] **Step 4: Record a versioned final report** listing 670 projection dispositions, 31 original nonterminal IDs and their new statuses, unchanged original hashes, all unresolved HOLD reasons, strict Card replay, and explicit `formal_graph_ingested=false` until the separate graph gate runs. Publish the branch only after tests and `git status`/remote SHA verification; do not declare the ACL cohort complete while an eligible recoverable item remains unresolved.

## Plan self-review gates

- Spec coverage: exact ID and physical-row identity, native PDF/reviewer evidence, fail-closed HOLD, immutable originals, negative tests, ten-item pilot, full 670 conservation, and separate Card/graph gate are mapped to Tasks 1–5.
- This plan does not resolve the unrelated NeurIPS #138 source alias, the 146 old Router hard HOLDs, or the 231-subset run. They remain separate goal workstreams.
- Before any Task 1 code change, use a fresh isolated worktree or confirm this existing worktree is clean and that the preceding design-spec commit is preserved.
