# ACL 2026 derived metadata admission — design v1

Status: design reviewed and approved by the user for implementation planning on 2026-09-28; **not an approval for corpus promotion**. Code execution follows the separately reviewed implementation plan and its gates.

## Purpose and boundary

The frozen ACL 2026 intake sometimes records the `.bib` link text `bib` as a paper title. The existing proof-only census identifies 670 potentially affected queue rows across 13 ACL queues; the 2026-09-28 07:19 UTC strict Card checkpoint still has 31 ACL items without a format terminal state. Neither number is a count of scientifically empty papers. We need an exact-paper-ID, evidence-bound comparison layer that can recover source metadata for later Card-job construction while preserving every original queue, manifest, Card, note, PDF, task, raw result, and review byte.

This design does **not** approve paper identity by title similarity, repair Paper Cards in place, clear existing holds, assert source-science approval, or ingest a high-dimensional graph. The independently approved full-corpus and 231-paper subset runs remain separate.

## Options considered

1. Edit the frozen ACL queues and existing Card artifacts. This is short operationally but destroys the original error evidence and invalidates downstream hashes. Rejected.
2. Put an ACL-specific title exception into the existing Card validator. This hides source ambiguity inside a general validator and cannot retain a field-by-field witness. Rejected.
3. **Recommended: immutable witness sidecar, then separately reviewed derived job metadata.** The original intake remains authoritative as a historical record; the derived layer is explicitly a proposed correction with its own hashes and admission status.

## Architecture and data flow

There are two versioned, separately gated artifacts:

1. `acl_metadata_witness.v1.jsonl` contains one row per exact frozen queue item in scope. It binds the original physical queue row and ACL paper ID; official cached event HTML; the unique canonical `/id/<paper-id>/` paper anchor; a source-span/DOM witness for title, ordered authors, venue, year, and pages when available; the selected native PDF's exact path and SHA-256; page-1 title/author/year identity evidence; and all parser/version/input hashes. It also records each field's original value, proposed value, and disposition (`SUPPORTED`, `CONFLICT`, or `MISSING`). This artifact is diagnostic only and cannot change a Card job.
2. `acl_metadata_projection.v1.jsonl` is produced **only after independent item-level review** of the witness. It contains a unique exact queue-row key, the approved field-level replacements, the witness row hash and reviewer receipt, plus a fail-closed disposition. A new run may use this projection to create new Card jobs; it never rewrites old jobs or Cards. A row with conflict or missing identity remains `METADATA_HOLD` and remains visible in the candidate set.

The join key is the tuple of canonical ACL paper ID, frozen queue path and file hash, one-based physical row number and row-byte hash, and pinned PDF hash. Neither normalized title nor a fuzzy match is a join key. The HTML authority must be the expected official ACL source and exact canonical ID path; `.bib`, `/pdf/`, foreign hosts, duplicate anchors, query/fragment variants, and ambiguous author lists do not authorize a replacement. The parser must concatenate adjacent inline text nodes without inventing spaces, then apply only documented whitespace normalization.

For a projected field, the official record establishes the proposed metadata while the native PDF independently checks document identity. HTML title alone is insufficient to bind a PDF; a matching PDF alone does not establish ACL venue, year, pages, or proceedings version. Unknown fields stay unknown. Existing `VALID_EMPTY` and hold reasons remain historical facts, not a scientific judgment about the paper.

## Admission gates and errors

- Before parsing, pin the frozen manifest, all relevant queue files, HTML cache files, source-inventory rows, candidate PDFs, and affected task/raw/note artifacts. Rehash immediately before writing each new sidecar. A changed input aborts the affected row and records `INPUT_DRIFT`.
- Emit one witness for every target row, including unsuccessful ones. Missing or duplicate canonical anchors, title or ordered-author disagreement, PDF mismatch, year/venue/page ambiguity, alias/version conflict, or unavailable primary evidence yields a reasoned HOLD. Do not turn a HOLD into `VALID_EMPTY` by omission.
- A different person or independent agent reviews the exact witness row and native PDF identity before projection. Reviewer decision is bound to the witness hash and exact source row. The builder must reject an unreviewed, stale, duplicated, or broader-than-approved projection.
- Only source metadata fields of a **new** exact Card job may differ from the frozen job; all source IDs, note/PDF paths and hashes, raw/review references, original queue bytes, and prompt bindings remain traceable. A projection does not set `human_approved`, `scientific_entailment_global_approved`, `graph_ingested`, or source-identity approval.
- All products carry a version, creation time, protocol hash, input hashes, reviewer receipt, and before/after field values. Corrections create a new version; old witness, projection, and error artifacts are retained.

## Verification before use

The first pilot is ten exact IDs sampled to include a straightforward `bib` title, inline-tag adjacency, missing anchor, duplicate anchor, wrong ID, PDF/title mismatch, ordered-author mismatch, and an already-correct control. The pilot must prove that only supported fields appear in the projection; every negative control remains HOLD. Independent QA checks native page-1 identity and all pinned byte hashes, not just parsed HTML.

After the pilot, cover all 670 census rows with explicit `SUPPORTED` or reasoned HOLD witness records and independently audit the currently open ACL11/ACL13 `bib` holds. A new run's Card-job generator must show exact one-to-one source-row coverage, no duplicate or missing items, unchanged original-file hashes, and strict Card adapter replay. Format acceptance remains separate from scientific/source approval. Only then may a separately reviewed ingestion step consume approved new jobs; no automatic graph promotion is part of this design.

## Existing evidence and non-goals

- Proof-only root-cause and negative-test proposal: `F:\\LLM_Evoke\\runs\\parallel24-20260927-1340\\engineering\\ACL670_DERIVED_METADATA_ADMISSION_SPEC.md`.
- Strict Card checkpoint: `F:\\LLM_Evoke\\runs\\parallel24-20260927-1340\\pm\\checkpoint-20260928-071710\\progress.json`.
- Frozen queue/source files and their hashes are read from the run's pinned manifests at execution time; this design does not change those pins.

Out of scope: PDF redownload, arXiv/NeurIPS source repair, general cross-venue metadata normalization, Card semantic rewriting, Router label generation, ontology assignment, and graph/idea generation.
