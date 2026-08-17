# Autoresearch sandbox

Allowed to modify:

- `experiments/pnnl_leakage_safe_pilot/configs/search/`
- `experiments/pnnl_leakage_safe_pilot/runs/inner_cv/`
- model/head implementation under `src/pnnl_pilot/models/`
- tuning reports under `evidence/pnnl_autoresearch/`

Read-only/frozen:

- source and crop manifests;
- eligible/quarantined sample lists;
- outer LOSO split manifest;
- target and primary metrics;
- crop-scale sensitivity set;
- model-family trial budgets;
- Gate 1–4 criteria.

Forbidden:

- reading the outer test sample during fitting or selection;
- pooling crops across Sample IDs before splitting;
- changing labels, coordinate transforms, or exclusions after seeing results;
- reporting best-of-search outer-test scores;
- treating crop count as independent sample count.
