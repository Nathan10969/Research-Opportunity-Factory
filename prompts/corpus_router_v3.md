# General research corpus admission router (v3)

This protocol decides only whether a source-grounded research note belongs in the selected research corpus. It does not assign a scientific topic, method family, or ontology label. Those judgments belong to the unchanged PaperCard v1 evidence extraction and the existing landscape pipeline.

Choose `GENERAL_RESEARCH` only when the supplied note itself contains identifiable bibliographic/source evidence and substantive research content relevant to the user's explicitly selected research scope. Use `OTHER` for irrelevant, unsupported, empty, duplicate-as-a-distinct-work, or otherwise inadmissible notes. `OTHER` is never eligible for automatic corpus selection.

Use the full note text. Bind every factual statement in `core_mechanism`, `scope_reason`, and `evidence_locator` to this note. Do not infer admission from title keywords, venue, or filename alone. Do not turn ACL, VLM, RL, geometry, or any other field into an old KV/memory/supervision label. Do not invent source identities or scientific claims. For uncertain source status, explain the uncertainty in `scope_reason` and choose `OTHER` unless the source is independently supported.

Return exactly one JSON object with the job-bound identifiers and schema in the supplied output contract. No additional keys or prose.
