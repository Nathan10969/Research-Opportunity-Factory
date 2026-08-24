Classify each candidate by its paper's core mechanism and scope, not by keyword occurrences such as “memory”, “compression”, or “retrieval”.

Use `KV_CACHE` for work whose core mechanism directly concerns transformer KV/cache representation, eviction, compression, reuse, or management. Use `LONG_MEMORY` for persistent or cross-query memory as the core mechanism. Use `BRIDGE` only when the paper substantively couples KV/cache and persistent/cross-query memory; incidental mentions of either concept are not sufficient. Use `OTHER` for all remaining work.

Echo the supplied binding fields exactly. Output exactly one JSON object and no prose or Markdown:
{"schema_version":"idea_factory.corpus_router_result.v1","job_id":"string","slug":"string","note_sha256":"string","prompt_sha256":"string","label":"KV_CACHE|LONG_MEMORY|BRIDGE|OTHER","core_mechanism":"string","scope_reason":"string","evidence_locator":"string","confidence":"HIGH|MEDIUM|LOW"}
