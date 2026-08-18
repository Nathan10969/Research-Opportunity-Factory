# PNNL compute, tuning, and ablation protocol

Status: frozen before feature extraction. Unknown result cells must remain empty
until produced by the matching LOSO protocol.

## Compute placement

| Work | Execution host | Reason |
| --- | --- | --- |
| Contract/hash validation, TAR extraction, JPEG inventory | Local CPU | Sequential disk and parsing workload |
| Coordinate projection, material/background checks, crop generation | Local CPU | Large-image decode and file I/O; no useful CUDA gain |
| Duplicate-point handling, LOSO manifests, metrics, bootstrap/statistics | Local CPU | Small tabular workload |
| Mean/median, coordinate-only, Ridge/Elastic Net, PCA/random projection | Local CPU | Small classical models |
| RF/XGBoost feature selection and regression | Local CPU | A100 host has weak CPU; GPU transfer gives no benefit for sklearn RF |
| Frozen DINO feature extraction and second-backbone extraction | A100 GPU | Batched tensor inference |
| MLP, LoTRA-KAN, AE/VIB, release encoder, DP variants | A100 GPU | Neural training workload |
| Reconstruction/retrieval/attribute adaptive attackers | A100 GPU | Decoder and contrastive training workload |

Only verified manifests and clean crops are transferred to `/opt/Jony`.
Classical-model runs remain local; returned GPU embeddings/results carry hashes
and the same `crop_id`, `sample_id`, `outer_fold`, and cleaning-policy version.

### Runtime placement override (2026-08-18)

The A100 is reserved for the Ahui project. PNNL frozen-backbone extraction is
therefore running in isolated WSL2 on the local RTX 4090 (24 GB). This changes
only the execution host, not the frozen data, model, split, tuning, or ablation
protocol. The first complete primary extraction used official DINOv3-B/16,
batch size 64, PyTorch 2.6.0+cu124, and records the GPU, official-code commit,
weight SHA-256, manifest SHA-256, and output hashes in `feature_report.json`.

## Permanently frozen before modelling

- eligible samples: SS01–SS08 and SS31;
- quarantined/control-only samples: SS28, SS32, SS36;
- nine-member `DATA/*_10X_Nugget-Region-Etched.jpg` whitelist;
- forbidden MARKER and crop-7mm directories;
- target: local Vickers hardness;
- outer evaluation: 9-fold leave-one-Sample-ID-out;
- primary utility metrics: sample-balanced MAE and RMSE;
- coordinate-only comparison and Gate 1 kill criterion;
- primary crop scale: 1.0 mm, frozen before feature extraction because the
  source registration is region-level rather than pixel-exact;
- crop scales 0.5/2.0 mm as reported sensitivity axes;
- coordinate-jitter robustness at 0.25/0.5 mm in four cardinal directions;
- outer test samples never select preprocessing, dimensions, heads, or attacks.

## Inner-CV tuning budget

All tuning occurs on the eight outer-training samples with group-aware inner CV.
Every competing head receives the same number of evaluated configurations.

| Component | Inner-CV search |
| --- | --- |
| Ridge | alpha in `10^-4 ... 10^4` on a fixed log grid |
| Elastic Net | alpha log grid; l1 ratio 0/0.25/0.5/0.75/1 |
| RF | trees 300/600; min leaf 1/2/4/8; max features sqrt/0.33/0.5 |
| XGBoost/LightGBM | shallow depth 2/3/4; fixed small learning-rate/trees budget |
| Small MLP | 1–2 layers, width 32/64/128, learning rate and weight decay |
| LoTRA-KAN | rank ratio, regularization, learning rate; same trial budget as MLP |
| Attacker | capacity, training data fraction, augmentation, learning rate, convergence epoch |

Early stopping and normalization are fitted without access to the outer test
sample. Random models use at least three seeds in the pilot and five in final
evidence.

## Required ablations

| Axis | Frozen variants | Question |
| --- | --- | --- |
| Input | coordinate-only / image-only / image+coordinate | Does the image add real utility? |
| Contamination | raw diagnostic / metadata-safe / center-mask | Is performance caused by text, scale bars, or indents? |
| Physical scale | 0.5 / 1.0 / 2.0 mm | Which spatial scale carries utility/leakage? |
| Registration | no jitter / 0.25 mm / 0.5 mm cardinal jitter | Does utility survive source registration uncertainty? |
| Representation | raw DINO / PCA / random projection / random deletion / RF-topK / AE / VIB / proposed | Is the method better than ordinary compression? |
| Released dimension | 4 / 8 / 16 / 32 / 64 | Full privacy–utility Pareto curve |
| Capacity | matched dimension and matched rank | Is any gain more than capacity reduction? |
| Prediction head | Ridge / RF / MLP / LoTRA-KAN | Is the conclusion head-independent? |
| Backbone | DINO plus at least one non-DINO frozen backbone | Is the conclusion DINO-specific? |
| Privacy loss | remove reconstruction / retrieval / attribute / joint | Which mechanism contributes? |
| Attacker | weak/strong, data-size sweep, same/cross source | Does privacy rely on a weak attacker? |

## Frozen backbone and representation-width matrix

Backbone size and released representation width are separate axes. A larger
native vector is not counted as a privacy bottleneck, and a comparison across
different backbone sizes is not used to claim a benefit from compression.

| Family | Frozen variants | Native output width |
| --- | --- | ---: |
| DINOv3 ViT | S/16, S+/16, B/16, L/16, H+/16, 7B/16 | 384, 384, 768, 1024, 1280, 4096 |
| DINOv3 ConvNeXt | Tiny, Small, Base, Large | 768, 768, 1024, 1536 |
| DINOv3 pretraining domain | LVD-1689M; SAT-493M for L and 7B | matched architecture where available |
| DINOv2 control | S/14, B/14, L/14, G/14 | 384, 768, 1024, 1536 |

The pilot runs matched-capacity pairs first: DINOv2-B versus DINOv3-B, then
DINOv2-L versus DINOv3-L. S/S+, H+, 7B, ConvNeXt, and SAT variants are
backbone-scale/domain ablations and cannot be selected from outer-test scores.
For transformer backbones, CLS and mean patch-token pooling are both reported.
For every selected frozen backbone, the privacy-release axis is
`4/8/16/32/64/128/raw` dimensions with identical inner-CV trial budgets.

## Gates

1. Gate A remains partial until all nine samples have spatial and visible-indent
   QA and every accepted crop excludes metadata/background contamination.
2. Stop the release encoder if image+coordinate does not improve
   sample-balanced MAE over coordinate-only on at least 6/9 held-out samples.
3. A privacy operating point may lose at most 5% utility relative to raw DINO
   and must reduce conditional incremental leakage for two attack families.
4. If matched-capacity PCA/random/bottleneck is equally good, report compression
   rather than a new privacy mechanism.

For the first Gate 1 utility screen, the pre-registered primary comparison is
Ridge on DINOv3-B mean-patch plus normalized image coordinates versus Ridge on
normalized image coordinates alone. Improvement is evaluated separately on
each held-out Sample ID and must occur on at least 6/9 samples. CLS pooling,
Elastic Net, and Random Forest are reported as robustness analyses and cannot
replace the primary comparison after outer-fold results are observed.
