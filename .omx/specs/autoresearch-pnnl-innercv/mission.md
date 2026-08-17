# PNNL inner-CV autoresearch mission

Objective: within each frozen outer LOSO training fold, search a fair and
bounded hyperparameter space for utility heads and later release encoders,
without accessing the held-out Sample ID or modifying any data gate.

Completion requires a machine validator to confirm:

1. Gate A is `passed` and the crop/split manifests match their frozen hashes.
2. Every selected configuration was chosen only from group-aware inner CV.
3. Competing heads used the same declared trial budget.
4. All outer-fold predictions were generated exactly once after selection.
5. Per-fold, per-sample, per-seed results and configuration provenance exist.
6. No outer-test result was used to select crop scale, dimension, head, loss,
   attacker, early stopping point, or hyperparameter.

The loop is allowed to return a negative result. It is not allowed to redefine
the metric, sample list, split, or kill criterion to obtain a positive result.
