# Reviewed local entry verification — 2026-09-09

## Verdict and scope

The PM accepts the code change for a scoped branch release. GPT-5.6-Luna implemented it; the PM supplied requirements and independently checked source contracts, tests, and real pilot artifacts. A separate Luna code-quality review identified issues that were fixed before acceptance.

Base commit: `7a1be89122133f74315cecb3771570647b5e9284`.
Release branch: `feat/20260909-reviewed-local-opportunities`.
This report is a pre-publication acceptance record, not a remote-push receipt.

The change adds an opt-in reviewed two-paper/cross-facet entry lane. It does not lower the normal three-card threshold, implement automatic high-dimensional empty-cell discovery, or relax quality, deduplication, recon, reviewer, or Human Gate rules. No dependencies were added.

## Independent verification

The PM ran the following targeted regression set from the active worktree with the existing project Python environment:

```text
python -B -m pytest tests/test_local_entries.py tests/test_local_entry_mining.py tests/test_local_entry_replay_guards.py tests/test_local_entry_cli.py tests/test_opportunities.py tests/test_models.py tests/test_quality.py tests/test_pipeline.py::test_all_explicit_subcommands_are_registered tests/test_pipeline.py::test_cli_bounds_expected_runtime_errors_without_hiding_programming_bugs -q --tb=short -rs
192 passed, 2 skipped in 12.49s
```

`git diff --check` passed. This is a targeted regression result, not a claim that the entire repository test suite was run. Both skips are physical symlink creation unavailable on this Windows host. Lexical symlink rejection is separately regression-tested; the hardlink alias test passed. Physical symlink behavior still needs a host permitting those tests.

Covered contracts include strict fields and hashes, exact source-edge anchors, distinct papers, pair/cross-facet semantics, default-lane byte compatibility, mixed lanes, rejected/empty entry documents, snapshot/audit/marker replay, tamper refusal, external input preservation, both-source result binding, CLI command scope, absent-option digest compatibility, and repeated-input NO_OP behavior.

Code-quality review fixes verified by regression tests:

1. Reject an input symlink before resolving its path.
2. Replay an explicit empty local document on a zero-card/zero-edge landscape using the marker prompt snapshot.
3. Clean only owned generated artifacts after an injected partial-emission I/O failure; retain the external input.

## Real-data replay

The bounded D09 human-supervision × D07 distribution-shift replay reused 20 approved card records and 114 typed edges, including two reviewed taxonomy assignments. The PM verified exact card-semantic equality and all five primary landscape hashes against the frozen predecessor. Router judgments were mechanically rebound, not represented as newly generated model judgments.

The unchanged normal lane emitted zero jobs from 53 small bases. Two reviewed local entries emitted two real jobs; the combined cluster audit contained 54 rows including one local-mode marker. The identical second CLI invocation returned NO_OP. The PM successfully replay-validated the mining, result, and quality bundles. Both first-round generation wrappers were VALID_EMPTY, with zero ready and zero rejected opportunities.

An empty generation result is neither a falsification of the proposed relation nor a completed novelty check. In particular, an inferred mechanism or unverified data availability does not by itself justify discarding a diagnostic opportunity. The PM requested a separate scientific reassessment; the first-round artifacts remain frozen. Research revisions and private corpus material are intentionally not part of this code release.

## Remaining boundaries

- AI_PM entry approval is not human approval, independent novelty evidence, or permission to run experiments.
- Hashes bind review records; they do not authenticate the claimed reviewer or prove semantic correctness.
- This pilot validates entry plumbing and artifact custody. It does not establish a novel idea, a survivor rate, or field-wide coverage.
- Formal recon and later scientific gates must operate on real accepted candidates and genuine search receipts. Missing credentials are unresolved coverage, not empty search results.
- No paid API, GPU, training, inference scoring, human study, or cheap experiment was run for this feature verification.

See the [design](../superpowers/specs/2026-09-09-reviewed-local-entries-design.md) and [implementation plan](../superpowers/plans/2026-09-09-reviewed-local-entries.md).
