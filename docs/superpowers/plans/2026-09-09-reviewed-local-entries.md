# Reviewed Local Opportunity Entries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking. The user selected Luna implementation and main-agent PM review.

**Goal:** Admit explicitly reviewed two-paper/cross-facet hypotheses without changing the existing three-card lane or downstream scientific gates.

**Architecture:** A small `local_entries.py` owns strict input models, validation and canonical hashing. `opportunities.py` compiles its approved entries alongside unchanged normal jobs and replays their owned snapshots. CLI/pipeline forward and hash an opt-in argument only.

**Tech Stack:** Existing Python, Pydantic v2, pytest, strict JSON artifact utilities, PowerShell and Git. No new dependencies.

## Fixed workspace and runtime

Worktree: the existing isolated `fix-20260909-recon-and-routes` linked worktree.
Base: `7a1be89122133f74315cecb3771570647b5e9284`.
New branch: `feat/20260909-reviewed-local-opportunities`.
Runtime: the existing project Python environment; do not install new dependencies.
Set `PYTHONPATH` to the active worktree's `src` for CLI calls; verify imported module path. Do not edit the original dirty worktree.

The exact field contract, ownership rules and scientific boundaries are in the companion design. Main-agent responsibilities are specification, semantic review, independent test/readback checks and release approval. Luna owns implementation and Git mutations. This overrides generic plan templates asking the PM to supply all implementation code.

## Task 1: Strict entry contract

Files: create `src/idea_factory/local_entries.py` and `tests/test_local_entries.py`.

- [x] Verify the existing opportunities/models regression boundary; the final PM run includes both full test files. Do not interpret this as a separately retained pre-change baseline run.
- [x] Create the named feature branch from the verified clean base (carry only these PM design/plan documents).
- [x] Write focused failing tests using the existing real landscape fixture: an APPROVED same-relation two-paper entry, APPROVED cross-facet entry, REJECTED entry, and an empty document.
- [x] Add refusal tests: same card twice; two cards from one paper; wrong facet/dimension/raw text/scope; unknown prior ID; stale landscape hash; duplicate IDs and JSON keys; blank required text; unknown fields/status/role/operator; changed document hash.
- [x] Implement strict models and validation. Do not normalize/coerce user records into validity. For PAIR_RELATION compare the actual normalized relation and reviewed cluster scopes; for CROSS_FACET require different facets.
- [x] Run new tests and record RED/GREEN evidence. Main reviews spec compliance before integration.

Example rejection contract (test constructs a valid fixture document before the mutation):

```python
document['entries'][0]['anchors'][1] = document['entries'][0]['anchors'][0].copy()
document = with_recomputed_artifact_hash(document)
with pytest.raises(ValueError):
    validate_local_entries(document, landscape)
```

The test helper is implemented in the same test file. `validate_local_entries(document, landscape)` returns validated JSON records or raises; it must not write files.

## Task 2: Emission, strict replay and CLI integration

Files: modify `src/idea_factory/opportunities.py`, `src/idea_factory/pipeline.py`, `src/idea_factory/cli.py`; create `tests/test_local_entry_mining.py`; extend narrowly scoped CLI tests only where needed.

- [x] Write failing integration tests for one local job per approved entry, rejected audit without jobs, preserved source scopes, real neighbor IDs, and separate local provenance in job ID/hash.
- [x] Test byte-for-byte unchanged normal jobs/audit when option absent, including the existing two-card zero-job behavior.
- [x] Test same-facet pair and cross-facet mode coexist with normal >=3 clusters without transitive merging.
- [x] Test tampering/deletion of local snapshot or audit fails replay; stale/orphan or rehashed-but-invalid anchors fail; outside sentinels survive junction/symlink rejection.
- [x] Test invalid local input preserves an existing valid bundle and input/output alias is rejected before mutation.
- [x] Implement optional emission and snapshot/audit replay. Keep `_partition_bases` and the normal eight-operator schedule unchanged. A local entry selects exactly one existing operator.
- [x] Attach a local-only prompt contract requiring both cards and `LOCAL_RELATION_HYPOTHESIS`; reject local outputs omitting a card, duplicating support, or omitting the inference flag. Test both valid and invalid external wrappers through real ingestion.
- [x] Forward `--reviewed-local-entries` only on `emit-opportunity-jobs`. Bind non-null input into pipeline command hashing; old null-option command digests must be unchanged. Test CLI help/forwarding, unrelated-command rejection, NO_OP on identical input, and refusal to reuse a run after changed input.
- [x] Run entry/mining/opportunities/models/CLI subsets, then existing quality tests to prove no gate relaxation. Main performs spec review; a separate Luna performs read-only code-quality review. Fix substantive findings with regression tests.

## Task 3: Frozen-data replay, reporting and publication

Files: add concise README usage/schema pointer; local research artifacts stay outside the published repository; add code-verification report under `docs/reviews/`.

- [x] Create `pipeline_v4_local_entries_retry` under the existing private D09/D07 pilot directory through CLI initialization; preserve v3 byte hashes.
- [x] Reuse the original local config/intake, router judgments and approved card semantics, mechanically rebind against freshly emitted immutable jobs. Do not generate new science just to satisfy parser fields.
- [x] Rebuild the same 20-card landscape with the two approved taxonomy overrides. Record derivation hashes and exact counts.
- [x] Draft at most two entry reviews from the prior local relationships. PM approves only entry compatibility, not novelty. Emit jobs and obtain actual Luna outputs, permitting no candidate when priors already explain the question.
- [x] Ingest complete job coverage and execute unchanged quality gating. Proceed through dedup/recon preparation only for real survivors; do not fabricate empty downstream stages, external receipts or human scores.
- [x] Main independently validates result bundles and compares v3 hashes. The private Chinese handoff is maintained separately from this public code release and distinguishes entry admission, scientific candidates and pending gates.
- [ ] Publication receipt: after PM approval, commit only scoped source/tests/docs and push the new date branch. Record independent remote-SHA verification in the private release handoff after the commit exists; this pre-publication plan does not assert push completion. No PR/merge/force push.

## Acceptance tests and stopping conditions

Required: default-lane byte compatibility; strict bad-entry refusal; local-result binding; CLI input provenance; real two-paper replay; existing downstream gate unchanged; preserved v3; verified remote branch. No survivor quota. Missing recon credentials pause recon, not evidence extraction. A novel entry admits investigation but does not imply a novel research contribution.

## Execution record

Code accepted after the PM independently ran 192 passing tests with two platform symlink skips; see [verification report](../../reviews/2026-09-09-reviewed-local-entry-verification.md). The canonical first replay is `pipeline_v4_local_entries_retry`. Earlier path/mapping attempts remain preserved and are not the accepted replay. Its two jobs returned VALID_EMPTY; the PM requested a separate scientific reassessment because unknown data availability must not be treated as a novelty rejection. Any reassessed run is separately named and must not overwrite the first result.
