# Research Opportunity Factory

An evidence-preserving pipeline for turning a bounded research corpus into
falsifiable research opportunities, prior-art residuals, and human-approved
Idea Packs.

The current pilot targets KV-cache and long-term-memory research. The repository
name describes the system's actual role: it is an opportunity miner and
false-positive firewall, not a claim to be a general autonomous scientist or a
paradigm-discovery engine.

## What this repository is

The implemented path is opportunity-first rather than method-first:

```text
Corpus selection -> Paper Cards -> reviewed research landscape
-> opportunity mining -> quality gate -> internal deduplication
-> four-lane prior-art recon -> route generation -> reviewer gate
-> human gate -> Idea Pack
```

Failure-aware fields are used to identify which assumption breaks under which
condition and what capability is missing. They are a problem-formation language,
not an instruction to generate only limitation repairs.

The following upstream discovery fuels are important but **not implemented in
v1**: foundation/theorem cards, mathematical object construction, deployment
incidents, experiment anomalies, and other exogenous observations. They belong
before the Opportunity schema and are tracked as roadmap work rather than being
presented as existing capability.

## Current state

The offline orchestration, strict artifact validators, resumable CLI, human gate,
Idea Pack renderer, and immutable ledger update path are implemented. This is a
tool for producing usable research opportunities; it is not a claim that the
pipeline itself is a publishable research contribution.

| Capability | Status |
|---|---|
| Resumable offline pipeline and artifact validation | Implemented |
| Failure-aware Paper Cards and reviewed landscape assignments | Implemented |
| Opportunity quality gate and internal deduplication | Implemented |
| Operator-mediated live recon with validated receipts | Implemented |
| Route/reviewer/human handoffs and Idea Pack rendering | Implemented |
| Full 1,806-paper corpus run | Not run |
| Autonomous cheapest-test execution | Not implemented |
| Foundations/exogenous-fuel intake and Primitive Gate | Not implemented |
| Research execution graph, protected validation, and UI | Not implemented |

## Install and test

```powershell
uv sync --extra dev
uv run pytest -v
```

Alternatively, install with a Python 3.12 virtual environment and
`pip install -e ".[dev]"`.

No test makes a live network or LLM request.

## Resumable CLI

Initialize one evidence-preserving run, then execute the explicit handoffs in
order. External model, recon, reviewer, and human results enter only through
their corresponding `ingest-*` commands.

```powershell
idea-factory init-run --run D:\runs\kv-memory-001 --config config\pilot_kv_memory.json
idea-factory emit-corpus-jobs --run D:\runs\kv-memory-001
idea-factory status --run D:\runs\kv-memory-001
```

The complete command sequence is:

```text
init-run -> emit-corpus-jobs -> ingest-corpus-labels -> emit-card-jobs
-> ingest-cards -> build-landscape -> emit-opportunity-jobs
-> ingest-opportunities -> quality-gate -> internal-dedup -> emit-recon-pack
-> ingest-recon -> emit-route-jobs -> ingest-routes -> emit-review-jobs
-> ingest-reviews -> ingest-human-scores -> finalize
```

Every command checks the current `RunState`. Exact replay with identical input
hashes is a no-op only when every recorded owned artifact still exists with its
recorded hash. Deleted or modified outputs and changed prompts, upstream bundles,
or repository roots are rejected and require a distinct `--new-run` path;
existing evidence is never silently rewritten. A command that has emitted jobs
but lacks external results is `awaiting-external-result`, not failed.

Human scoring distinguishes an evaluated route from an unselected route. A
reviewer may submit an `idea_factory.human_skip.v1` result bound to the exact
human job with reason `NOT_MY_RESEARCH_PRIORITY`; skipped routes carry no
fabricated scalar scores and produce no Idea Pack.

## Reviewed landscape assignments

Paper Cards preserve the paper's detailed source scope. Free-text source scopes
must not be silently treated as an ontology: in a real corpus they are usually
unique and cannot form an opportunity basis. `build-landscape` therefore accepts
an optional human-reviewed assignment artifact.

Legacy `idea_factory.concept_assignments.v1` rows may normalize synonymous
concept text only inside the exact source scope. For a reviewed cross-paper
basis, use `idea_factory.concept_assignments.v2`: each row binds to the exact
`source_scope_key` while separately supplying a structured `cluster_scope` and
a nonblank reviewer rationale. The landscape retains both scopes in its edge
projection. Mechanical or disputed rows can never change cluster scope.

```powershell
idea-factory build-landscape `
  --run D:\runs\kv-memory-001 `
  --reviewed-assignments D:\runs\kv-memory-001\external\concept_assignments.json
```

This reviewed overlay is required evidence, not an automatic semantic merge.
Near-miss papers should remain unassigned rather than be forced into a cluster.

## Recon handoffs

The real recon path is operator-mediated and makes no hidden network call. For a
non-empty query pack it has three explicit handoffs:

1. `emit-recon-pack` without `--recon-context` emits
   `recon/execution_job_templates.jsonl` and waits.
2. Supply a strict context JSON with `workspace_root`, `skill_roots`,
   `prepared_at`, `uv_executable`, `allow_test_attestation`, and the independently
   prepared `operator_attestation`. Re-run `emit-recon-pack --recon-context ...`
   to validate the context and materialize `recon/execution_jobs.jsonl`.
3. Execute those jobs outside the pipeline. Put `execution_receipts.jsonl` at the
   root of a bounded bundle and mirror each job's `recon/raw/...` output below
   `<bundle>/raw/...`. Run `ingest-recon --execution-bundle <bundle>` without
   `--results`; the pipeline validates and imports the exact receipts/raw files,
   normalizes them, emits `recon/report_jobs.jsonl`, and waits. After the report
   jobs are completed externally, run `ingest-recon --results <reports.jsonl>`.

All bundle paths, contents, receipts, raw hashes, execution context, and report
jobs are replay-validated. `TEST_ONLY` attestation remains test-only even after a
complete recon bundle.

Each opportunity is searched through four semantic lanes: concept, mechanism,
failure, and evaluation. Each lane may use current terminology, generic-shape
queries, and synonyms. `NO_DIRECT_COVERAGE_FOUND` is deliberately bounded to the
frozen query pack and receipts; it is never promoted to a global novelty claim.

`status` reports `completed`, `awaiting-external-result`, `killed`, `surviving`,
and `unresolved-error` counts. It also separates these facts:

- code/offline-test readiness;
- whether validated live recon actually ran;
- whether human review completed;
- whether an Idea Pack exists;
- whether that pack is backed by live rather than `TEST_ONLY` recon.

For offline fixtures, initialize with `--mode offline-fixture` and use
`--allow-test-ready` only on the commands that expose it. Those artifacts remain
explicitly test-only and can never be reported as live recon.

`finalize` validates human scores, renders Idea Packs, appends and revalidates
the ledger, and writes `report.md`. It always states that the cheap falsification
test has not run and that no subsequent experimental result exists.

Zero survivors and HOLD-only packs are valid operational completion with an
auditable report, but are not successful idea production. Only
`READY_FOR_CHEAP_TEST` contributes to `surviving` or `idea_pack_produced`.

## v1 non-goals

- No full 1806 corpus.
- No live Science Skills execution in the offline suite.
- No autonomous experiments.
- No UI.

## Public repository boundary

This repository contains code, prompts, configurations, tests, and synthetic
fixtures only. It intentionally excludes copyrighted paper corpora, private
notes, live recon raw results, execution receipts, human scoring data, run
directories, credentials, and locally generated Idea Packs.

## Roadmap

1. Compile a human-approved Idea Pack into a bounded cheapest-test contract.
2. Separate exploratory feedback from protected validation evidence.
3. Add recoverable executable state and a minimal
   Hypothesis-Experiment-Finding-Claim evidence graph.
4. Add an upstream foundations/exogenous lane for theorem cards, mathematical
   objects, deployment failures, benchmark contradictions, and experiment
   anomalies.
5. Evaluate whether the added lane increases new-object discovery rather than
   merely producing mathematically decorated recombinations.

## License

Apache-2.0. See `LICENSE`.
