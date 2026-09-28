# Nips03 positions 134–137: isolated formal Card admission

Status on 2026-09-28: four PaperCards accepted in a separate live v3 run. This is Card admission only, not graph ingestion, human approval, idea generation, or completion of the full corpus.

The evidence lives outside this Git repository at `F:\LLM_Evoke\runs\nips03-four-formal-v3-20260928-061331\formal-card-receipt.json` (SHA-256 `3b67532df12a0165efd225e28c52f9753f093d6f3621d79c014141016fc5e4cb`). Its additive audit sidecar is `formal-card-receipt-addendum-v1.json` (SHA-256 `f6dce94739573acda3633fe6b2fe5a95907a2544c52b9ed5239172e5bd0620af`). The actual CLI run is the sibling `F:\LLM_Evoke\runs\nips03-four-formal-v3-20260928-061331-run`, run ID `run-3e800048e10e68c4`, mode `live`, stage `CARDS_READY`.

The new primary-source manifest contains exactly four versioned arXiv IDs: `2608.00409v1`, `2606.22352v1`, `2605.09806v1`, and `2602.21515v2`. These `source_record_ids` were newly bound from source evidence; they were absent from the frozen Nips03 queue. Positions 134, 136, and 137 are treated as preprints. Position 135's PDF prints NeurIPS 2026, while its downloaded source remains an arXiv preprint, not a claimed camera-ready version. Position 138 remains `HOLD_SOURCE_IDENTITY` and is excluded from all candidates, jobs, and results.

Four RouterResults were freshly authored against the canonical notes and v3 prompt (`FRESH_AUTHORED`). Four previously reviewed Card contents were re-bound to the new Card jobs (`REPLAY_ONLY`), with old raw Card, new wrapper, and accepted Card contents independently confirmed equal. The CLI and read-only bundle validators report four selected, zero Router rejects, four accepted Cards, zero Card rejects. An independent spec review and separate quality review both passed with bounded claims. The code baseline was 797 tests passed, 5 skipped.

The original 1,534-job run remains `CREATED` with zero formal results; no original queue, progress, note, PDF, or private handoff was intentionally rewritten. The receipt only proves the current Nips03 progress hash, not byte-level immutability over the entire interval, because no pre-run progress pin exists. No high-dimensional graph or Idea Pack has been generated from this four-item run.

The latest original PM checkpoint still counts 4,276 papers as 3,223 private `SCHEMA_VALID` Cards, 1,017 `VALID_EMPTY`, and 36 nonterminal. The four new formal Cards overlap that original corpus and must not be added to those PM counts until a versioned cross-run reconciliation exists.
