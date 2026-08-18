# PNNL DINOv3-B nested-LOSO result

Status: **COMPLETE / validator passed**

Experiment date: 2026-08-18

Primary target: local Vickers hardness (`hardness_hv`)

Unit of outer generalization: held-out `Sample-ID`

## Hard verdict

The pre-registered primary gate passed: Ridge with DINOv3-B mean-patch plus
normalized coordinates beat coordinate-only Ridge on 8 of 9 held-out samples
(required: at least 6 of 9). Its sample-balanced MAE fell from 26.5331 to
21.1136 HV.

The stronger nonlinear spatial baseline changes the interpretation. Coordinate-
only random forest reached 17.6890 HV MAE, so image-only DINO features were not
competitive (20.6221--21.1969 HV; only 1 of 9 samples beat coordinate-only RF).
However, combining DINO features with coordinates reduced RF MAE to 15.5040 HV
(CLS) or 15.5209 HV (mean-patch), and both fusion variants improved 8 of 9
held-out samples. Therefore this pilot supports **conditional incremental image
utility beyond the nonlinear spatial field**, not a claim that image alone is a
strong hardness predictor.

## Leakage-safe protocol

- 6,680 registered 1.0 mm crops from 9 physical samples.
- Frozen official DINOv3 ViT-B/16 features, width 768, with both CLS and
  mean-patch pooling.
- Five representations: normalized coordinate-only, CLS-only, mean-patch-only,
  CLS+coordinate, and mean-patch+coordinate.
- Three regressors: Ridge, Elastic Net, and random forest.
- Outer evaluation: 9-fold leave-one-`Sample-ID`-out.
- Hyperparameter selection: leave-one-`Sample-ID`-out inner CV on the eight
  outer-training samples only.
- Equal frozen search budget: 6 configurations per model and representation.
- Selection metric: sample-balanced MAE. Outer test data were never used for
  model or hyperparameter selection.
- Random forest outer fits used seeds 17, 29, and 43 and were ensembled. Linear
  estimators are deterministic.

## Overall held-out results

| Model | Representation | MAE (HV) | RMSE (HV) |
|---|---|---:|---:|
| Ridge | coordinate | 26.5331 | 31.0591 |
| Ridge | image CLS | 21.4833 | 27.1984 |
| Ridge | image mean-patch | 21.3065 | 27.1381 |
| Ridge | fusion CLS | 20.9179 | 26.1895 |
| Ridge | fusion mean-patch | 21.1136 | 26.4510 |
| Elastic Net | coordinate | 26.5332 | 31.0590 |
| Elastic Net | image CLS | 21.0918 | 26.9612 |
| Elastic Net | image mean-patch | 20.8093 | 26.7935 |
| Elastic Net | fusion CLS | 20.6565 | 25.7466 |
| Elastic Net | fusion mean-patch | 21.0636 | 26.1748 |
| Random forest | coordinate | 17.6890 | 22.0279 |
| Random forest | image CLS | 21.1969 | 26.8266 |
| Random forest | image mean-patch | 20.6221 | 26.1821 |
| Random forest | fusion CLS | **15.5040** | 19.5367 |
| Random forest | fusion mean-patch | 15.5209 | **19.4333** |

Relative to coordinate-only RF, fusion CLS reduces sample-balanced MAE by
12.35%, and fusion mean-patch reduces it by 12.26%.

## Paired held-out-sample check against coordinate-only RF

| Representation | Samples with lower MAE | Non-improving sample(s) |
|---|---:|---|
| image CLS | 1/9 | SS01, SS02, SS03, SS04, SS05, SS06, SS07, SS31 |
| image mean-patch | 1/9 | SS01, SS02, SS03, SS04, SS05, SS06, SS07, SS31 |
| fusion CLS | 8/9 | SS06 (+0.6257 HV) |
| fusion mean-patch | 8/9 | SS06 (+0.5769 HV) |

## Evidence and integrity

- Completed jobs: 135/135; unique job keys: 135; duplicate keys: 0.
- Inner-CV trial rows: 810; per-fold rows: 135; per-seed rows: 225;
  held-out prediction rows: 100,200.
- One provenance hash across every job:
  `4b13b6761e2832d6e322637ab30c79b6d3a005945b3d5beb663d608c77ec3431`.
- Maximum outer train/test `Sample-ID` overlap: 0.
- Final report SHA-256:
  `c76fa7b47343fcc64483858bfb3565c363535bec36184ca9220913233a829b98`.
- Mission validator: `status=passed`, with no failed checks.
- Feature extraction provenance: official `dinov3_vitb16`, commit
  `6876159a11b4df116f30f667f8c9888617df0751`, weights SHA-256
  `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`,
  local RTX 4090, CUDA 12.4. Regression and inner CV were CPU-only; no A100 was
  used.

Machine-readable result: `final_report.json`. Full predictions and trial-level
artifacts remain under the git-ignored derived-data run directory
`data/derived/pnnl_gate_a/experiments/nested_loso_dinov3b_1mm/`.

## Claim boundary and next ablations

This is a positive pilot for the specific 1.0 mm, unmasked, raw-crop DINOv3-B
setting. It does not yet establish robustness to printed coordinates or labels,
crop registration perturbations, crop scale, center masking, alternative DINOv3
backbones, DINOv2, or privacy transformations. Those are the next ablations; the
current experiment must remain frozen as the primary Gate-A result.
