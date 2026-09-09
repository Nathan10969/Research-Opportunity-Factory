# Route generator v2

Return only one native strict JSON object matching the replay-bound result
schema embedded in the job. Do not add Markdown, duplicate keys, non-finite
numbers, secrets, credentials, or fields outside that schema.
The job's `result_schema_sha256` is part of the replay binding.

Generate between 1 and 3 routes for the exact supplied opportunity. Use the
first N route IDs in order, with route_index 1..N; never emit filler routes.
Do not rename the opportunity, change its research question, substitute another
target, or weaken its scope and constraints. Copy every binding field, ID, hash,
and prompt hash exactly from the job.

Each route must contain all eight scientific fields: old assumption changed,
new assumption, new mechanism, why it addresses the bound failure, why the
nearest priors cannot resolve the residual, source of gain, cheapest decisive
test, and kill condition. The old assumption is copied exactly. Every other
field must explicitly retain the supplied failure, residual, scope, missing
capability, alternative explanation, or decisive experiment binding.

MechanismSpec has five required facets: intervention_site, operation,
target_state, learning_signal, and state_representation. Each facet is a
nonblank descriptive explanation; mechanisms are open-ended and need not use a
catalog or canonical rendering. `new_mechanism` is a concise explanatory
description and need not equal a serialization of MechanismSpec. The system
derives a deterministic fingerprint for provenance only; it is not a novelty
certificate. Duplicate normalized mechanism facets or duplicate normalized
mechanism descriptions are rejected. Semantic diversity is
`REQUIRES_REVIEW`, not proven by different words or axes.

Provide a grounded `causal_gain_hypothesis` for each route and include it in
`source_of_gain`. Normalized causal gain descriptions and normalized
source-of-gain descriptions must not be clones.

Copy `frozen_opportunity`, complete `scope_constraints`,
`resource_constraints`, `nearest_prior_bindings`, and `residual` exactly. A
NEAR_PRIOR_WITH_RESIDUAL route must retain at least one exact evidence-bound
prior. `required_resources` may contain only IDs from `allowed_resources`;
private data, root access, credentials, and unlimited compute are forbidden.

Reconnaissance is bounded. Never make an absolute novelty or global-priority
claim, including world-first, first-ever, never attempted, no prior work,
no one, nobody, unprecedented, globally novel, or highest priority. Do not
emit sensitive keys such as `access_token`, `access_secret`, `client_secret`,
`api_key`, `authorization`, `password`, or `private_key`, or credential-shaped
content. Legitimate
quoted text copied from the verified frozen opportunity remains evidence, not a
model-authored global claim. Recon authenticity is always
`NOT_AUTHENTICATED`; `operator_attested_execution_ready` is an operator scope,
not cryptographic verification. A `NO_DIRECT_COVERAGE_FOUND` result means only
that the supplied bounded query pack found no direct coverage.
The phrase `previously attempted baseline` is ordinary procedural text, not a
novelty claim.
