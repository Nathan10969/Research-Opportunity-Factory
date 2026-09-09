# D09 x D07 code verification (2026-09-09)

This is a bounded implementation verification record, not a full-suite or
scientific acceptance claim.

- Branch: `feat/20260909-d09-d07-pilot`
- New focused tests: `tests/test_corpus_d09_d07.py`, 9 passed after the final
  hermetic-test and prompt/README corrections.
- Earlier focused verification (before these final corrections): models,
  corpus, and D09/D07 tests 145 passed; recon-quality and routes-open tests 48
  passed. These were not rerun in this correction pass.
- Real local-source smoke (manual, not a pytest): with
  `repo_root=F:\LLM_Evoke\idea_factory`, the two configured source lists
  resolved to `D09_annotation_human` and `D07_robustness_ood`; enumeration found
  341 candidates and 341 unique slugs.
- Version boundary: `idea_factory.corpus_router_prompt.v2`,
  `idea_factory.corpus_router_job.v2`, and
  `idea_factory.corpus_router_result.v2`; the focused test confirms a v1 result
  is rejected.

The hermetic pytest creates the same relative `notes`, `papers/_ideas`, and
`papers/_domain_rounds` layout under `tmp_path`; it does not depend on the
developer's local corpus. No commit or push was performed.

## Independent PM verification

On 2026-09-09, an independent run executed:

```text
python -B -m pytest tests/test_corpus_d09_d07.py tests/test_models.py tests/test_recon_quality.py tests/test_routes_open.py -q --tb=short
```

Result: **147 passed in 0.43s**; `git diff --check` passed. The full test suite
was not run in this verification.
