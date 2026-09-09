# Reviewed local opportunity entries

Approved scope: the user authorized the proposed minimal change with "继续" on 2026-09-09. Codex main is requirements owner and reviewer; GPT-5.6-Luna implements. This document makes that approved scope operational, not a new research architecture.

## Outcome

Retain the existing three-to-eight-card, same-relation mining lane unchanged. Add an opt-in, reviewed two-paper entry lane. A reviewed entry permits asking a bounded research question; it does not certify novelty, scientific validity, human approval, or readiness for an experiment.

Alternatives considered: globally reducing three to two would weaken the existing lane without explaining compatibility; replacing the graph with a joint-occupancy engine exceeds this task. The separate reviewed lane is selected.

## Interface

Add `--reviewed-local-entries PATH` to `emit-opportunity-jobs` only, and the corresponding optional `reviewed_local_entries: Path | None` argument in the backend/emitter. Keep old prompts, job IDs, audit rows, and output bytes unchanged when the option is absent. Reject use of the new argument with unrelated stages.

The strict JSON document has exactly `schema_version`, `landscape_hashes`, `entries`, and `artifact_sha256`. Version is `idea_factory.reviewed_local_entries.v1`. Hash the canonical JSON of the document without `artifact_sha256` (sorted keys, UTF-8, compact separators, no NaN). `landscape_hashes` must equal the replay-validated current landscape hashes. Reject duplicate JSON keys, coercion, unknown fields, duplicate IDs, stale anchors, and mismatched hashes.

Each entry has exactly:

- `entry_id`: nonblank string.
- `entry_kind`: `PAIR_RELATION` or `CROSS_FACET`.
- `anchors`: exactly two records, from distinct card IDs and distinct source-paper paths. Each anchor has `card_id`, `dimension`, `facet`, `raw_text`, `source_scope_key`, and `role`. The first five fields must exactly identify an existing validated landscape edge; role is a nonblank explanation of that source's role.
- `shared_object`, `interface_relation`, `compatibility_reason`, `scope_differences`, `proposed_hypothesis`, `alternative_explanation`, `decisive_test`, `novelty_boundary`: nonblank strings, independently reviewed for meaning. These express a proposal, not a new observation attributed to either paper.
- `nearest_prior_card_ids`: nonempty unique IDs from the validated landscape, allowed to include the anchor papers.
- `operator`: exactly one existing OpportunityOperator, explicitly selected to bound cost.
- `review`: exactly `status` (`APPROVED` or `REJECTED`), `reviewer_id`, `reviewer_role` (`AI_PM` or `HUMAN`), and `reason`; strings are nonblank. Preserve AI_PM identity; this review never advances the Human Gate.

`PAIR_RELATION` requires the two anchors to share facet, normalized relation, and cluster scope in the validated landscape. `CROSS_FACET` requires different facets; source scopes stay separate and compatibility is a reviewed claim, not an assertion of equivalence. Whitespace-only text does not satisfy a field.

## Emission and replay

Only APPROVED entries emit, exactly one job per entry/operator. Rejected entries remain in a local-entry audit with their reason. Use a new local job version/lane and bind the complete entry, its hash, and input artifact hash into its identity. Retain the existing external opportunity result wrapper; job ID and subsequent job hash bind the new provenance.

Persist an owned, hash-verified input snapshot at `opportunities/reviewed_local_entries.json`, and an owned `opportunities/local_entry_audit.jsonl`. Replay must rebuild both regular and local jobs from the validated landscape and snapshot. Deleting/changing the snapshot or audit, tampering with anchors/review/hash, and changing a job must fail replay. Default runs must not acquire these files. An explicit direct-API re-emission without the option must remove only its owned local sidecars, not reuse stale local inputs. CLI state hashing must include the provided file only when non-null so old command digests remain stable.

Local jobs contain the two source cards and applicable existing neighbor summaries, including summaries associated with explicitly named nearest-prior cards. Do not invent neighbor IDs or flatten source scopes. Attach local-entry instructions to the local job prompt only, leaving the default prompt unchanged. Require generated local opportunities to cite both source cards exactly once and carry `LOCAL_RELATION_HYPOTHESIS` in `inference_flags`. Empty output is allowed when no defensible opportunity follows.

Validate new input before mutating previously valid generated outputs. Use existing strict artifact ownership guards. Do not allow the snapshot input to alias an output that would be overwritten. Cleanup concerns only explicit generated files under the validated run, never source notes or arbitrary paths.

## Scientific gates and limits

Keep quality, deduplication, recon, route/reviewer, Human Gate, and experiment requirements unchanged. In particular, the existing UNKNOWN-failure-mechanism quality rejection still applies; a new entry does not promise downstream acceptance. No OpenAlex credentials are fabricated, no paid API/GPU/human experiment is launched, and no legacy killed hypothesis is revived by renaming it.

This is a reviewed candidate-entry feature atop a typed landscape, not an automatic high-dimensional empty-cell detector. Review text and hashes are accountability records, not cryptographic proof of who authored a review or automated proof of semantic compatibility.

## Pilot and release

Work from published base `7a1be89122133f74315cecb3771570647b5e9284` on a new branch `feat/20260909-reviewed-local-opportunities`, using the existing isolated worktree. Preserve `pipeline_v3` and all historical runs. Rebuild a new `pipeline_v4_local_entries` from the same frozen intake and already-reviewed card semantics, rebinding wrappers to newly emitted jobs mechanically. Record derivation and source hashes.

Use the two previously identified local relationships as entry-review material, not preapproved scientific ideas. The PM must review each filled entry. Then emit jobs, have Luna evaluate them, run result/quality gates, and record genuine disposition, including zero candidates. Progress to dedup/recon only if candidates actually survive. Code acceptance and research acceptance are separate. Publish reviewed code/docs to the named branch only; no merge, force push, or automatic PR.
