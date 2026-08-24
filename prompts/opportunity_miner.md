# Opportunity miner v2

Return only one native strict JSON object: no prose, comments, code fences, or
Markdown. The wrapper must have exactly these fields and bindings from the job:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "job_id", "cluster_id", "operator", "prompt_sha256", "landscape_hashes", "card_ids", "neighbor_ids", "opportunities"],
  "properties": {
    "schema_version": {"const": "idea_factory.opportunity_result.v1"},
    "job_id": {"type": "string", "minLength": 1},
    "cluster_id": {"type": "string", "minLength": 1},
    "operator": {"enum": ["ASSUMPTION_BREAK", "FAILURE_TRANSFER", "BOUNDARY_CONDITION", "MEASUREMENT_GAP", "EVALUATION_MISMATCH", "OBJECTIVE_CONFLICT", "DYNAMIC_MISMATCH", "MECHANISM_TRANSPLANT"]},
    "prompt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "landscape_hashes": {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
    "card_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
    "neighbor_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
    "opportunities": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["schema_version", "opportunity_id", "operator", "assumption_x", "observation_y", "condition_z", "failure_f", "missing_capability_w", "alternative_explanation_a", "decisive_experiment", "supporting_card_ids", "nearest_internal_neighbors", "scope_compatibility", "inference_flags"],
        "properties": {
          "schema_version": {"const": "idea_factory.opportunity.v1"},
          "opportunity_id": {"type": "string"},
          "operator": {"type": "string"},
          "assumption_x": {"type": "string"},
          "observation_y": {"type": "string"},
          "condition_z": {"type": "string"},
          "failure_f": {"type": "string"},
          "missing_capability_w": {"type": "string"},
          "alternative_explanation_a": {"type": "string"},
          "decisive_experiment": {"type": "string"},
          "supporting_card_ids": {"type": "array", "items": {"type": "string"}},
          "nearest_internal_neighbors": {"type": "array", "items": {"type": "string"}},
          "scope_compatibility": {"type": "string"},
          "inference_flags": {"type": "array", "items": {"type": "string"}}
        }
      }
    }
  }
}
```

Every `opportunities` item must have exactly:
`schema_version`, `opportunity_id`, `operator`, `assumption_x`, `observation_y`,
`condition_z`, `failure_f`, `missing_capability_w`, `alternative_explanation_a`,
`decisive_experiment`, `supporting_card_ids`, `nearest_internal_neighbors`,
`scope_compatibility`, and `inference_flags`.

Each item is an opportunity, never explanatory prose. Existing work assumes X.
Observation Y suggests X fails under Z, producing F. State missing capability W
and alternative A. The decisive experiment must literally name or quote A and
distinguish it with a measurable contrast.

Use only supplied `card_ids` for support and supplied `neighbor_ids` for nearest
internal neighbors. Unsupported IDs are forbidden. All method names and model acronyms
are forbidden in the proposed opportunity. Do not say to combine A and
B. Do not output invented claims, Markdown, or any field outside the exact schema.
