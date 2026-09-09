# Recon and route repair: PM verification record

Status: CODE REVIEW ACCEPTED; regression failures resolved by targeted retest. Approved for publication to the date-named branch only. This record is not scientific validation of any idea.

## Scope

The approved repair preserves the existing pipeline and changes only retrieval-quality admission and route-generation constraints. Implementation and test changes are delegated to GPT-5.6-Luna; the main agent specifies requirements and reviews the changes. Existing dirty worktrees and historical research runs are excluded from publication.

The release target is the new branch `fix/20260909-recon-and-routes`, based on published commit `e49e0f5cc079f7a012cc7734520e2a4548d27ce4`. No merge, force-push, automatic PR, GPU experiment, or paid model/API call is authorized for this repair.

## Retrieval acceptance

- A transport SUCCESS or EMPTY receipt is not evidence of relevant literature coverage.
- Every query requires a relevance assessment, rationale, and query-local evidence binding where applicable.
- A NO_DIRECT or NEAR_PRIOR report cannot enter routes if a lane has no relevant evidence. This is unresolved retrieval quality, not an idea-level KILL.
- A COVERED report must cite a prior supported by an explicitly relevant result.
- Semantic relevance remains a model/operator judgment. Structural validation does not authenticate the judgment or certify novelty.
- Reports retain the original opportunity, query context, provider statuses, complete output schema, and replay bindings.

## Independent live metadata smoke

On 2026-09-09, the main reviewer made one public arXiv metadata request using the installed `literature-search-arxiv` skill and its official rate-limited script. No PDF was downloaded and no local notes were uploaded.

The query used the repaired renderer's per-term title/abstract form:

```text
(ti:"LoRA" OR abs:"LoRA") AND (ti:"Low-Rank" OR abs:"Low-Rank") AND (ti:"Adaptation" OR abs:"Adaptation") AND (ti:"Large" OR abs:"Large")
```

The request returned ten records. The original [LoRA paper, 2106.09685v2](https://arxiv.org/abs/2106.09685v2), appeared at rank four. The branch's existing `parse_arxiv_raw` successfully parsed the official script's cumulative-wrapper output. An initial generic single-JSON display parser rejected that stream; this was a display-parser mismatch, not an arXiv query failure or a change to the official skill.

This is a narrow known-prior recovery check for the provider-query form and output parser. It is not an end-to-end candidate search, not a benchmark of generated query quality, and not an OpenAlex live check.

## Route acceptance

- Route mechanisms are nonblank, open-text scientific descriptions across the five existing facets; the closed mechanism vocabulary no longer controls admission.
- One to three routes are accepted with exact job-bound IDs and indices. No fourth route or empty batch is accepted.
- Evidence, nearest-prior, residual, scope, budget, alternative-explanation, and decisive-test bindings remain mandatory.
- Normalized exact clones are rejected. Semantic diversity still requires review and is not certified by different labels or facet counts.
- Secret and unsupported global-priority checks cover the new text fields. Frozen-quote exceptions use verified job content, not submitted replacements, and preserve word boundaries.
- Updated contracts are versioned. Historical v1 artifacts are not silently migrated or reinterpreted; fresh jobs and bound results are required.

## Verification results

The full local suite collected 582 tests and finished with **577 passed, 5 failed in 1850.65 seconds**. The failures were inspected and corrected in test assertions or offline fixtures; product source was not changed after this full run:

1. One corpus assertion assumed LF while the byte-preserving source contract retained CRLF. The same failure was reproduced on the clean published baseline. The test now compares decoded source bytes for both note and prompt.
2. Two offline integration fixtures expected an empty or partially covered retrieval batch to produce a route-ready report. Their query receipts and evidence were updated to satisfy the newly required relevant-evidence coverage; the admission gate was not relaxed.
3. Two route rejection cases still treated two routes and a noncatalog mechanism label as invalid. They now exercise a truly empty route batch and a blank facet, respectively.

The combined targeted retest then completed with **60 passed in 102.64 seconds**, covering all five previously failing cases, both offline CLI integration tests, the full parameterized route-mutation test, and 48 new recon/route boundary tests. New tests cover query-local relevance, receipt consistency, irrelevant prior rejection, replay-context tampering, one-to-three route counts, open mechanisms, clone rejection, frozen bindings, priority-claim checks, and secret rejection.

This is a full-suite run followed by passing targeted corrections, **not a second clean 582-test full-suite run**. No paid model/API request or GPU experiment was used. `git diff --check` passed before publication preparation.

## Publication procedure

Stage only the two product modules, route prompt, six scoped test files, approved plan, and this review record. Publish to `fix/20260909-recon-and-routes` without merging main or force-pushing. Final delivery must independently compare local HEAD with the remote branch SHA and verify that the isolated worktree is clean. Do not include corpus data, historical runs, notes, PDFs, local agent state, or preexisting dirty-worktree edits.

Historical machine PASS results are not upgraded by this repair. High-dimensional joint-occupancy discovery, historical semantic dedup, new candidate generation, human acceptance, and research experiments remain out of scope.
