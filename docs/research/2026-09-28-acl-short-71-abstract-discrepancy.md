# ACL 2026 `2026.acl-short.71`: abstract count discrepancy

## Finding

This is an abstract-content discrepancy, not a source-identity mismatch. The official [ACL Anthology record](https://aclanthology.org/2026.acl-short.71/) says it treats PDFs as authoritative and directs metadata corrections to match the PDF. Its displayed/exported abstract says “sixteen models.” The linked [official proceedings PDF](https://aclanthology.org/2026.acl-short.71.pdf) instead says “twenty-one models” in the printed abstract (p. 1), repeats 21 in the introduction and methodology, and itemizes the total as eight open-source MRMs, eight backbone MLMs, and five proprietary models (pp. 1–2; 8 + 8 + 5 = 21). The main text names the same thirteen spatial datasets in the abstract.

The 16/21 difference is localized to the abstract metadata text: the Anthology page and frozen queue/source record say 16, while the linked PDF’s abstract and body agree on 21. The identity fields do agree: Anthology ID `2026.acl-short.71`, title, ordered four-author list, 2026 ACL Short Papers venue, and pages 862–876 match between the official record, PDF p. 1, and the frozen local item. The Anthology’s own correction notice makes the PDF the authority for this comparison; this report records that policy and evidence, but does not assert that the external metadata has been corrected.

## Card check

The existing generated Card at `F:\LLM_Evoke\runs\parallel24-20260927-1340\workers\acl-13\items\2026.acl-short.71\raw.json` follows the PDF on the disputed count: its mechanism and evaluation regime state 21 models across 13 datasets, and its result/limitation summaries match the PDF’s benchmark and No-Image++ evidence. The associated note explicitly preserves the 16-versus-21 discrepancy rather than silently rewriting the frozen source abstract. Thus, the evidence supports binding this Card to the identified PDF source and its 21-model contents; it does not make the queue’s 16-model abstract accurate to that PDF.

## Frozen evidence and separate admission gate

- Local proceedings PDF: `F:\LLM_Evoke\papers\acl2026\2026.acl-short.71__Chain_of_Thought_Degrades_Visual_Spatial_Reasoning_Capabilities_of_Multimodal_LLMs.pdf`; 333,550 bytes; SHA-256 `7db57fa8221bf1d21002009cff01ede7dee3a396548eed2a9afc5e5f3aa9fa0f`.
- Frozen item note: `F:\LLM_Evoke\runs\parallel24-20260927-1340\workers\acl-13\items\2026.acl-short.71\note.md`; SHA-256 `440d9cd94b07bf25dcd04b926a8938b7601707a801a0d6206cf5cc6875d34ab2`.
- Generated Card: `F:\LLM_Evoke\runs\parallel24-20260927-1340\workers\acl-13\items\2026.acl-short.71\raw.json`; SHA-256 `cb15b4831f59a2569c264357f4f3712b47b89020a4436c97f71d68b63a9a6d43`.
- Frozen review: `F:\LLM_Evoke\runs\parallel24-20260927-1340\workers\acl-13\items\2026.acl-short.71\review.json`; SHA-256 `f90877d18e526d9324b9c7d41cf61e2a304e1c13b2a483de7deb579ea3350867`. Its disposition is `SCHEMA_VALID_FULL_SOURCE_READ_NATIVE_SPOT_CHECKED`; this is a schema/content review result, not PM admission.
- Identity witness: `F:\LLM_Evoke\runs\parallel24-20260927-1340\workers\acl-13\items\2026.acl-short.71\admission-witness-v1.json`; SHA-256 `97ff5c03a4e9236d7306bfff9c519312dff9574f6af54c60a317118151ddecad`. It records all five identity comparisons as matching and the abstract conflict as `PRESERVE_BOTH; NO_IDENTITY_REPAIR_PROPOSED`.
- The witness’s pinned PM snapshot, `F:\LLM_Evoke\runs\parallel24-20260927-1340\pm\checkpoint-20260928-084304\progress.json` (SHA-256 `514bcdf65273abbd56484e6b9af20ae38d59cb0f63f7e126dec96f03e7583a83`), lists this item among the 16 `PUBLICATION_IN_PROGRESS` rows. The witness also records `human_approved=false`, `graph_ingested=false`, `source_admission_approved=false`, and `added_to_corpus_denominator=false`. Accordingly, the evidence supports source identity, but formal PM admission is not authorized or established by this finding.

This report is additive evidence only. No queue, source record, note, Card, review, PM state, manifest, or promotion flag was changed.
