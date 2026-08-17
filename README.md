# Leakage-Safe Materials Imaging Pilot

This repository contains reproducible code, configuration, tests, and public
evidence for a leakage-safe materials-image/property pipeline. The first pilot
uses PNNL MCPC Round 1 optical micrographs and spatial Vickers hardness.

The verified 43.27 GB source TAR is intentionally not versioned. The default
local path is configured in `configs/pnnl_gate_a.json`; all derived pixels,
embeddings, checkpoints, and run caches remain outside Git. Frozen source
contract hashes and the exact nine-image whitelist are versioned under
`data_contracts/pnnl_binding_contract/`.

Current gate: **PNNL Gate A — extraction, contamination audit, coordinate QA,
and Sample-ID-isolated split construction. No model result is claimed yet.**

See `docs/superpowers/plans/2026-08-17-pnnl-gate-a.md` for the execution plan.
