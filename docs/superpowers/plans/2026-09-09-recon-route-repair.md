# Recon relevance and open mechanisms — implementation requirements

> For agentic workers: use subagent-driven-development and test-driven-development. The user explicitly assigns the main agent to PM/review only and implementation to GPT-5.6-Luna. Do not add reviewer agents or upgrade models.

**Goal:** Stop transport-success-only novelty clearance, and allow evidence-bound mechanisms that are not limited to six templates.

**Architecture:** Keep the existing pipeline, artifacts and replay checks. Add an explicit evidence-bound retrieval-quality contract and version the changed contracts. Keep generation content open while preserving source, scope, budget and test constraints.

**Stack:** Existing Python 3.12+, Pydantic 2 and pytest. No new runtime dependencies, GPU experiments, paid APIs or broad refactoring.

## Workspace and publication

- Work only on `fix/20260909-recon-and-routes`, in a separate worktree under the existing ignored `.worktrees/` directory.
- Base on verified `origin/main`; the currently observed published tree is `e49e0f5`, identical to committed feature tree `3db0d35`.
- Leave all pre-existing tracked and untracked changes in the old worktrees untouched. Do not copy data, notes, PDFs, run archives, secrets, `.omx`, or unrelated docs.
- Do not rewrite historical runs or reinterpret their machine PASS as scientific approval. New protocol artifacts must have new policy/schema identities; stale inputs fail explicitly.
- After PM approval, commit only named in-scope paths, push the new branch, and verify remote SHA equals local HEAD. No merge, force-push, or automatic PR.

## Task A — retrieval quality (reviewed first)

Primary paths: `src/idea_factory/recon.py`, `tests/test_recon.py`; update narrow CLI/model/prompt callers only when required by the contract.

- [ ] Confirm the reproduction: a successful query receipt with unrelated physics results currently permits a no-direct-prior report. Add a regression that fails on the committed baseline.
- [ ] Replace paragraph concatenation with bounded, lane-specific search-term candidates. Any deterministic term extraction is explicitly a lexical draft, not a semantic relevance decision. Preserve useful established method/domain terms for retrieval instead of applying idea-generation naming restrictions to searches.
- [ ] Render arXiv candidates with explicit title/abstract fields and Boolean grouping; do not search giant author lists through unrestricted fields. Keep OpenAlex query syntax provider-appropriate. Preserve deterministic IDs, bound query/argv provenance, caps and rate limits.
- [ ] Require report results to include one explicit relevance assessment for every bound query: status, supporting evidence IDs and a nonblank rationale. Distinguish relevant, irrelevant, genuinely empty and unavailable results. Assessments are model/operator judgments, not authenticated or automatically proven semantic truth.
- [ ] Validate complete and unique query coverage, evidence-to-query membership, successful execution for cited results, and nonempty evidence for a relevant judgment. Reject foreign IDs, invented references and inconsistent empty/error claims.
- [ ] A lane without any evidenced relevant result cannot support route-ready `NO_DIRECT_COVERAGE_FOUND` or `NEAR_PRIOR_WITH_RESIDUAL`. Reject with an actionable retrieval-quality-incomplete error; do not turn this into an idea-level KILL. Transport SUCCESS/EMPTY by itself never establishes semantic coverage.
- [ ] Bind the assessment to persisted report/job/replay hashes. Old results without the new contract must not silently pass the new validator.
- [ ] Keep all original source/scope/receipt/timestamp and nearest-prior evidence checks. No hidden model calls and no hardcoded D02/physics-domain blacklist.

Required tests: unrelated-result regression; related-result positive control; partial relevant lane coverage; genuine zero results; provider error; absent/duplicate/foreign query assessment; foreign/cross-query evidence; replay tampering; legitimate method names allowed in retrieval; bounded provider-correct query generation.

## Task B — open mechanism content (after Task A review)

Primary paths: `src/idea_factory/routes.py`, `prompts/route_generator.md`, `tests/test_routes_review.py`; update only necessary downstream fixtures/docs.

- [ ] Add red tests showing a grounded non-enumerated mechanism and a single valid route are rejected by the old implementation.
- [ ] Retain structured mechanism facets, but make values nonblank descriptive text instead of closed scientific-content enums. The schema constrains evidence and structure, not which scientific mechanisms may exist.
- [ ] Make `new_mechanism` an actual explanatory description; remove compulsory canonical-enum-string equality. Keep a deterministic serialization/fingerprint for provenance, not a novelty certificate.
- [ ] Permit one to three routes with deterministic IDs and ordering. Do not require three filler alternatives. Update job metadata and prompt to express the same range.
- [ ] Reject exact/normalized clones of mechanisms and causal-gain descriptions. Do not claim different words or axes establish semantic diversity; flag semantic diversity as requiring review.
- [ ] Preserve exact opportunity, scope, nearest-prior, residual, budget and decisive-test bindings. Include newly free-text mechanism fields in the existing secret/global-priority checks while preserving legitimate frozen quoted evidence.
- [ ] Version changed policy/schema contracts so old frozen jobs are never silently reinterpreted. Do not bulk migrate the historical runs.

Required tests: 1/2/3 valid routes; 0/4 invalid; non-catalog mechanism accepted; blank facets rejected; normalized clones rejected; stale/tampered bindings rejected; budget and counter-explanation unchanged; forbidden authored claims checked inside new free text; route-to-review handoff still works.

## Acceptance and cost limits

- [ ] Luna records red and green test commands and outcomes; PM reads the diff and reviews requirements first, then code quality.
- [ ] Run focused suites during iteration. Run the complete local suite once for final acceptance, unless a failure requires another targeted cycle. No repeated full-suite polling.
- [ ] A very small live metadata smoke is allowed after offline tests: at most three public arXiv requests, using the installed official skill script and its rate limit/terms protocol. Test recovery of a known nearby paper and inspect relevance, not just HTTP status. Do not upload private texts or credentials. An unavailable service is reported as unverified, never as a successful scientific search.
- [ ] Final report states branch, remote SHA, test counts, live-smoke boundary, changed contracts, and untouched historical evidence. It must not claim any existing research idea has become novel or experimentally validated.

Publication note: the two reviewed repairs may be committed together because the route integration fixtures depend on the new recon contract. Do not publish an intermediate commit whose retained route fixtures still expect transport-only clearance.

## Explicitly deferred

Full high-dimensional joint-occupancy discovery; historical semantic dedup repair; corpus retagging; new idea generation; human review automation; any training experiment; importing another AI-scientist framework.
