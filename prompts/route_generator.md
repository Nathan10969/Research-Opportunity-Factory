# Route generator v1

Return only one native strict JSON object matching the result schema embedded in
the job. Do not add Markdown, comments, prose outside JSON, duplicate keys,
non-finite numbers, secrets, credentials, or fields not present in the schema.
Treat the embedded `result_schema` and `result_schema_sha256` as the complete,
replay-bound handoff contract; use only its allowed enums and the supplied
controlled catalog.

Generate exactly 3 routes for the exact supplied opportunity. Do not rename the
opportunity, change its research question, substitute another target, or weaken
its scope and constraints. Copy every binding field, route ID, opportunity ID,
hash, and prompt hash exactly from the job.

Each route must contain all eight scientific fields: old assumption changed,
new assumption, new mechanism, why it addresses the bound failure, why the
nearest priors cannot resolve the residual, source of gain, cheapest decisive
test, and kill condition. The old assumption is copied exactly. Every other
field must explicitly retain the supplied failure, residual, scope, missing
capability, alternative explanation, or decisive experiment binding requested
by the job.

For every route, provide one strict `MechanismSpec` using only the controlled IDs
supplied by the job for `intervention_site`, `operation`, `target_state`,
`learning_signal`, and `state_representation`. Copy its canonical rendering into
`new_mechanism`. Do not provide a free-text fingerprint; the system derives the
fingerprint from the five IDs. Every route pair must differ on at least two IDs.
Labels such as adaptive, dynamic, or hierarchical are not controlled IDs and
cannot establish distinctness.

Use only the generic IDs in `mechanism_vocabulary` to propose three
`MechanismSpec` objects. The vocabulary proves only structured representation;
it does not certify domain compatibility. Supply a grounded
`causal_gain_hypothesis` for each route and include it in `source_of_gain`.
All three normalized source-of-gain fingerprints and causal gain hypotheses
must differ; changing only MechanismSpec IDs is insufficient.

Copy `frozen_opportunity`, complete `scope_constraints`,
`resource_constraints`, `nearest_prior_bindings`, and `residual` exactly. A
NEAR_PRIOR_WITH_RESIDUAL route must retain at least one exact evidence-bound
prior. `required_resources` may contain only IDs from the job's
`allowed_resources`; private data, root access, credentials, and unlimited
compute are forbidden.

Reconnaissance is bounded. Never make an absolute novelty or global-priority
claim. Never say world-first, first-ever, never attempted, no prior work, no
one, nobody, unprecedented, globally novel, highest priority, or that the route
is novel. The conservative firewall rejects absolute phrases such as first in
the world, first-ever, never attempted, not been attempted, globally novel, and
unprecedented. It does not reject ordinary phrases such as first stage, a new
control policy, or a previously attempted baseline.

Do not emit sensitive keys such as `access_token`, `access_secret`, `client_secret`, `api_key`,
`authorization`, `password`, or `private_key`, and do not emit Bearer, GitHub,
Slack, AWS, Google, or private-key credential shapes. Recon authenticity is
`NOT_AUTHENTICATED`; `operator_attested_execution_ready` is an operator scope,
not cryptographic verification. A
NO_DIRECT_COVERAGE_FOUND result means only that the supplied query pack found no
direct coverage. A nearest prior and its residual must be preserved exactly.
