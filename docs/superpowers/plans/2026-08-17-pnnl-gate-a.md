# PNNL Gate A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-safe PNNL MCPC Round 1 extraction, cleaning, spatial-QA, crop-manifest, and Sample-ID split gates required before any model training.

**Architecture:** Keep the 43.27 GB verified TAR read-only on `D:` and keep all executable code, configuration, tests, contract snapshots, and public evidence in this Git repository. A standard-library/Pillow pipeline selects exactly nine whitelisted optical members from the frozen binding contract, rejects overlays and quarantined samples, produces deterministic cleaned images and spatial QA, and freezes LOSO split metadata. Model code is deliberately excluded until Gate A passes.

**Tech Stack:** Python 3.10, standard library (`csv`, `hashlib`, `json`, `tarfile`, `unittest`), Pillow, NumPy; pytest-compatible unittest tests; Git.

---

### Task 1: Freeze repository and path contracts

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `configs/pnnl_gate_a.json`
- Create: `data_contracts/pnnl_binding_contract/README.md`
- Create: `data_contracts/pnnl_binding_contract/source_files.sha256`
- Create: `data_contracts/pnnl_binding_contract/eligible_optical_assets.csv`
- Create: `data_contracts/pnnl_binding_contract/specimens.csv`

- [ ] Copy only the frozen specimen registry, nine-member whitelist, and source-file hashes into Git; keep the full normalized label tables external and hash-pinned because they are generated evidence rather than source code.
- [ ] Store the external TAR and output roots in `configs/pnnl_gate_a.json`; make command-line overrides possible so paths are not hard-coded in Python.
- [ ] Add ignore rules for `data/raw/`, `data/derived/`, `runs/`, weights, embeddings, archives, and secrets.
- [ ] Verify the copied contract SHA-256 values against the source contract before proceeding.
- [ ] Commit the repository contract checkpoint.

### Task 2: Implement whitelist and extraction validation with TDD

**Files:**
- Create: `tests/test_pnnl_gate_a.py`
- Create: `src/pnnl_pilot/__init__.py`
- Create: `src/pnnl_pilot/contracts.py`
- Create: `src/pnnl_pilot/extract.py`
- Create: `scripts/run_pnnl_gate_a.py`

- [ ] **Step 1: Write failing whitelist tests**

```python
def test_selects_exactly_nine_eligible_10x_data_members(self):
    rows = load_assets(self.assets_csv)
    selected = select_optical_sources(rows)
    self.assertEqual({r.sample_id for r in selected}, {
        "SS01", "SS02", "SS03", "SS04", "SS05",
        "SS06", "SS07", "SS08", "SS31",
    })
    self.assertTrue(all("/DATA/" in r.member_path for r in selected))
    self.assertTrue(all(r.member_path.endswith("_10X_Nugget-Region-Etched.jpg") for r in selected))
```

- [ ] **Step 2: Write failing blacklist tests**

```python
def test_rejects_overlay_and_quarantined_members(self):
    for forbidden in ("/VISUALIZATION/MARKER/", "/VISUALIZATION/crop-7mm/"):
        with self.assertRaises(ContractViolation):
            validate_member_path(forbidden + "bad.jpg", "SS01")
    for sample_id in ("SS28", "SS32", "SS36"):
        with self.assertRaises(ContractViolation):
            validate_member_path("/DATA/x_10X_Nugget-Region-Etched.jpg", sample_id)
```

- [ ] Run `python -m unittest tests.test_pnnl_gate_a -v`; confirm RED because the package does not exist.
- [ ] Implement immutable dataclasses, CSV loaders, exact whitelist matching, duplicate/member uniqueness checks, and streaming SHA-256 extraction.
- [ ] Run the tests and confirm GREEN.
- [ ] Run the extractor on the real TAR and require exactly 9 extracted files, 9 source hashes, and zero blacklisted paths.
- [ ] Commit the extraction checkpoint.

### Task 3: Implement deterministic image cleaning and contamination checks with TDD

**Files:**
- Modify: `tests/test_pnnl_gate_a.py`
- Create: `src/pnnl_pilot/cleaning.py`
- Create: `src/pnnl_pilot/image_checks.py`
- Generate outside Git: `data/derived/pnnl_gate_a/clean_images/`
- Generate in Git: `evidence/pnnl_gate_a/source_manifest.csv`
- Generate in Git: `evidence/pnnl_gate_a/cleaning_manifest.csv`

- [ ] Write a synthetic-image test proving that a fixed metadata band is removed without consulting hardness labels.
- [ ] Write a loader test that rejects filenames or provenance paths containing `MARKER`, `crop-7mm`, overlay QA, or quarantined sample IDs.
- [ ] Write a test proving every cleaned image records source hash, cleaning-policy version, crop box, output dimensions, and output hash.
- [ ] Run the focused tests and confirm RED.
- [ ] Implement a versioned, position-only cleaning policy and contamination validator; do not implement hardness-conditioned masking.
- [ ] Run focused and full tests and confirm GREEN.
- [ ] Generate the nine cleaned candidates and machine-readable manifests; keep image pixels outside Git.
- [ ] Commit the cleaning checkpoint.

### Task 4: Implement coordinate projection, crop geometry, QA, and LOSO split freezing with TDD

**Files:**
- Modify: `tests/test_pnnl_gate_a.py`
- Create: `src/pnnl_pilot/spatial.py`
- Create: `src/pnnl_pilot/crops.py`
- Create: `src/pnnl_pilot/splits.py`
- Generate in Git: `evidence/pnnl_gate_a/projection_audit.csv`
- Generate in Git: `evidence/pnnl_gate_a/crop_manifest.csv`
- Generate in Git: `evidence/pnnl_gate_a/split_manifest.csv`
- Generate outside Git: `data/derived/pnnl_gate_a/qa_overlays/`
- Generate outside Git: `data/derived/pnnl_gate_a/crops/`

- [ ] Write tests for physical-to-pixel projection direction, stir-centred coordinates, and out-of-bounds rejection.
- [ ] Write tests for 0.5/1.0/2.0 mm crop sizes and deterministic padding/rejection behavior.
- [ ] Write tests proving all records from one Sample ID occur in exactly one LOSO outer fold and train/test overlap is zero.
- [ ] Write tests for sample-balanced metrics and sample-equal sampling weights.
- [ ] Run focused tests and confirm RED.
- [ ] Implement the minimal projection, crop, split, and metric functions; keep approximate transforms explicitly labelled.
- [ ] Run focused and full tests and confirm GREEN.
- [ ] Generate global and stratified local QA overlays for human review; overlays must never be accepted by the training loader.
- [ ] Freeze split metadata, but keep `eligible_for_model=false` until human indentation/spatial QA is recorded.
- [ ] Commit the spatial and split checkpoint.

### Task 5: Gate A validator and CPU/GPU execution handoff

**Files:**
- Create: `src/pnnl_pilot/validate_gate_a.py`
- Create: `tests/test_gate_a_validation.py`
- Create: `docs/experiments/pnnl_compute_and_ablation_protocol.md`
- Generate: `evidence/pnnl_gate_a/validation_report.json`
- Generate: `evidence/pnnl_gate_a/SHA256_MANIFEST.txt`

- [ ] Write failing tests for exact sample count, hash coverage, forbidden-path count, quarantined-sample count, split overlap, and unresolved human-QA status.
- [ ] Implement a validator that returns `passed`, `partial`, or `failed`; `partial` is mandatory while indentation/spatial QA is unresolved.
- [ ] Document compute placement: local CPU for contract/crops/splits/classical baselines and A100 for frozen-backbone extraction, neural heads, release encoder, and adaptive decoders.
- [ ] Freeze ablation axes before GPU runs: input modality, crop scale, contamination masking, representation compression, prediction head, backbone, privacy loss, capacity match, and attacker strength.
- [ ] Run all tests, regenerate hashes after the validation report, and verify the evidence manifest.
- [ ] Commit and push the Gate A branch only after local verification.

## Stop conditions

- Stop before feature extraction if fewer or more than nine eligible whitelisted sources resolve.
- Stop before modelling if any training image contains overlay/scale-bar text or an unresolved visible indentation.
- Stop the optical point-level route if most samples cannot isolate post-hardness contamination.
- Do not tune crop scale, selector, model, or attacker on an outer held-out sample.
- Do not start the release encoder unless image+coordinate improves sample-balanced MAE over coordinate-only on at least 6/9 LOSO samples.

## Self-review

- The plan covers Gate A, crop geometry, split isolation, compute placement, ablation freezing, and a machine-readable stop state.
- Full utility/privacy sweeps are intentionally excluded until Gate A and the raw-DINO minimum falsification pass.
- No result values or positive outcomes are assumed.
