# Strict private Card transport into v3 jobs — design

Status: design for user review. Scope: transport already-written private Card objects into newly emitted, source-bound v3 Card jobs without changing their scientific text. This is a throughput bridge, **not** a scientific approval system.

## Decision and alternatives

We considered (1) strict envelope rewrapping of existing private results, (2) regenerating every Card with a model, and (3) copying old raw JSON directly into a v3 run. Choose **(1)**. It reuses the 2,856 structurally valid private results where exact identities match, costs no model calls, and preserves the original Card object. (2) would repeat costly reading and introduce new content drift; (3) violates v3 job/result binding and is forbidden. The user approved the direction on 2026-09-29.

## Inputs and outputs

Input is an immutable, run-specific allowlist. Each row names exactly one actual `cards/card_jobs.jsonl` job and one prior private raw result; it pins their file SHA-256 values, the original task/accepted-wrapper and QA receipts, canonical note path/hash, Card prompt hash, source ID/version and local PDF path/hash. The v3 run must already have a validated Router selection bundle and emitted Card jobs. The transport never infers a match from title, filename similarity, directory order or mtime.

Output is a **private candidate** JSONL of `idea_factory.paper_card_result.v1` wrappers plus a row-level `READY/HOLD` ledger and receipt. No CLI ingest, formal Card, graph, human approval or scientific approval occurs in this step. The result wrapper has exactly `{schema_version,job_id,slug,note_sha256,prompt_sha256,cards}` from the actual new job. `cards` is copied as a strict JSON value from the pinned old raw; its canonical JSON hash must be identical before and after. The original raw, task, note, PDF and existing run files are never edited. The old wrapper's `slug` and `empty_reason` are not copied into the new envelope.

## Acceptance algorithm

1. Verify the run path is the exact expected isolated run, the selection/rejected bundle is valid, and the job set equals the selected set. Rehash all allowlist and source artifacts before reading them; reject path aliases or changed bytes.
2. For each row, require one-to-one actual job/old raw matching by canonical note path and note SHA, prompt SHA, source ID/version, PDF SHA, old task/raw/QA pins, and the old raw's own note/prompt bindings. Verify the source's original queue/manifest identity where supplied. No missing, duplicate, or extra job is silently accepted.
3. Require old raw schema `idea_factory.bulk_card_response.v1`, exactly one nonempty `cards` array for this initial nonempty-Card path, and no native type coercion. Preserve every Card object unchanged; require each Card's `paper.source_path` to equal the new job's canonical note path and pass the existing strict `_validate_card` check against that job. Require globally unique `card_id` values in the candidate batch.
4. Produce a `READY` candidate only if every deterministic check passes. Any mismatch yields an explicit item-level `HOLD` with a reason and all original hashes. For an already emitted job that is HOLD, do **not** manufacture `cards=[]` or claim `VALID_EMPTY`; later ingest may report the missing result honestly, or a fresh run may be constructed from the READY subset. No batch is ingested automatically.

`READY` means *eligible for a separate science/source QA and PM ingest decision*, not that the paper's claims are true. The exact old and new prompt bytes must match; a future prompt change requires a new extraction rather than transport. Positive-only private `VALID_EMPTY` items remain outside this path and keep their original reason.

## Error and custody behavior

Preflight a whole requested batch before writing the candidate JSONL. Publish the candidate/ledger/receipt atomically to a new private staging directory; never overwrite a prior candidate or a public run result. A failed preflight produces a HOLD ledger without a fabricated candidate. The receipt records run/job/selection/input/output hashes, per-row source and body hash, exclusions, and `scientific_entailment_audited=false`, `human_approved=false`, `graph_ingested=false`. Importing staged results into a run requires a distinct action after independent QA. The transport performs no network or model call.

## Verification

Test-first implementation: prove a real private Card with matching new job rewraps and the Card body canonical hash stays equal; then negative cases for note/prompt/PDF/slug/job drift, duplicate IDs, tampered raw/QA, path alias, old empty result, extra keys, and changed body. Exercise the existing `_validate_card` and v3 selection-bundle validators, not mocked acceptance. Run focused and full regression suites in the linked worktree's own virtual environment. Pilot on the current eight-article isolated run with private outputs only, independently review it, then apply the same contract to a larger frozen allowlist. Do not call eight provisional Cards a completed corpus.
