# Project State

This file records the current factual state of the project.

Keep it short. Update it only when the project state actually changes.

## Current Objective

Build and validate a weakly supervised appliance-localization system on cleaned REFIT data, then add only the engineering layers justified by the validated ML result.

## Current Authorized Work

Data audit completed, Kettle target policy locked, and regular timebase/window policy locked.

Model training is not yet authorized.

## Dataset

- Name: REFIT: Electrical Load Measurements (Cleaned)
- Source: University of Strathclyde / Zenodo
- DOI: 10.5281/zenodo.5063428
- Local status: Fully audited in this workspace (20 households, 119,495,879 rows, 6.47 GB uncompressed)

## Appliance & Target Policy

- Approved first appliance: **Kettle** (Decision D-005).
- Ground-truth target policy: **Locked** (Decision D-005)
  - ON threshold: `Kettle power >= 1500 W` (pre-model evaluation target definition).
  - Event continuity: gap `<= 20 s` merges into the same event.
  - Duration ceiling: candidate events `> 600 s` (10 min) excluded from localization ground truth as sensor-freeze anomalies.
- Household handling:
  - H12: Excluded from primary Kettle localization evaluation because the monitored Kettle channel provides insufficient and internally inconsistent target evidence.
  - H3: Retained (replacement on 16 Apr 2014 documented).
  - H13: Retained (sensor freezes > 600 s filtered).
  - H17: Retained (shared toaster IAM documented).
  - H19: Retained (observed kettle vs aggregate discrepancy documented).
  - H11: Retained (solar-affected aggregate documented).
  - H21: Retained (shared toaster IAM and solar aggregate documented).
  - H1, H10, H15, H16, H18: Unmonitored Kettle channels in metadata.

## Timebase & Window Policy

- Regular timebase: **Locked** (Decision D-006)
  - Grid resolution: **8-second regular grid**.
  - Gap bridging: Gaps `<= 16 s` forward-filled with zero-order hold.
  - Segment boundary: Gaps `> 16 s` start a new contiguous recording segment; never bridge or interpolate across a `> 16 s` boundary.
- Model window policy: **Locked** (Decision D-006)
  - Reference window size: **510 grid points** (temporal span: (510 - 1) * 8 = 4,072 seconds / ~67.87 minutes).
  - Window structure: Non-overlapping windows for initial reference dataset.
  - Boundary constraint: A window is valid only when all 510 grid points reside strictly within the same contiguous segment (no cross-segment windows).
- Target projection: Same 8-second grid and contiguous segment boundaries applied to ground-truth localization targets.

## Evaluation & Split

- Primary principle: strict household-level separation.
- Primary household split: **Locked & Frozen** (Decision D-007)
  - **TRAIN (16 households):** H1, H3, H5, H6, H7, H8, H9, H10, H11, H12, H15, H16, H18, H19, H20, H21 (141,003 windows / 81.00%)
  - **VALIDATION (2 households):** H4, H17 (18,561 windows / 10.66%)
  - **TEST (2 households):** H2, H13 (14,508 windows / 8.33%)
- Weak supervision: Derived strictly from metadata-level presence (1 = indicated presence, 0 = unmonitored). Timestamp-level sub-meter measurements are excluded from model training.
- Test split protection: H2 and H13 are frozen holdouts reserved solely for final evaluation.

## Models

- Simple 1D CNN Baseline (`src/models/cnn_baseline.py` & `scripts/train_cnn_baseline.py`):
  - Completed 6-epoch reference baseline training.
  - Result: Negative sanity baseline (Precision 19.92%, Recall 100.0%, F1 33.23%, Balanced Accuracy 50.0%).
- 1D ResNet Reference Model (`src/models/resnet.py` & `scripts/train_resnet.py`):
  - Architecture: 3 residual blocks with filter groups `{64, 128, 128}` and kernel pattern `{8, 5, 3}`, Global Average Pooling, and Linear classifier producing one logit per window.
  - Architectural Adaptation Note: The published CamAL paper describes a 2-class softmax classification head. Our project intentionally implements a single output logit with `BCEWithLogitsLoss` and sigmoid-equivalent binary decision at threshold 0.5 as defined by our project model-training specification. It is an intentional binary-classification implementation adaptation, not byte-for-byte or mathematically identical to the paper's 2-class softmax head.
  - Experiment Scope: This experiment evaluates a single ResNet reference model. It is NOT yet the multi-scale CamAL ensemble, and Class Activation Map (CAM) localization is NOT yet implemented in this experiment.
  - Training policy: Identical experimental controls to CNN baseline (BCE loss on weak labels, Adam lr=1e-3, batch size 256, max epochs 30, patience 5, train-derived standardization, validation strong targets at threshold 0.5, test split H2/H13 sealed).
  - Status: Implemented, smoke-tested, preflight-verified, awaiting real training run.

## Artifacts

- Dataset forensic audit report: [`artifacts/reports/refit_data_audit.md`](file:///d:/SmartMeter/artifacts/reports/refit_data_audit.md) / [`.json`](file:///d:/SmartMeter/artifacts/reports/refit_data_audit.json)
- Kettle target quality audit report: [`artifacts/reports/kettle_target_audit.md`](file:///d:/SmartMeter/artifacts/reports/kettle_target_audit.md) / [`.json`](file:///d:/SmartMeter/artifacts/reports/kettle_target_audit.json)
- Timebase window index manifest: [`artifacts/manifests/timebase_window_index.md`](file:///d:/SmartMeter/artifacts/manifests/timebase_window_index.md) / [`.json`](file:///d:/SmartMeter/artifacts/manifests/timebase_window_index.json) (174,072 total windows)
- Structural window descriptors: [`artifacts/manifests/timebase_windows.jsonl`](file:///d:/SmartMeter/artifacts/manifests/timebase_windows.jsonl)
- Kettle evaluation targets: [`artifacts/manifests/kettle_evaluation_targets.jsonl`](file:///d:/SmartMeter/artifacts/manifests/kettle_evaluation_targets.jsonl) (15,837 active target windows)
- Post-build integrity audit: [`artifacts/manifests/post_build_integrity_audit.md`](file:///d:/SmartMeter/artifacts/manifests/post_build_integrity_audit.md) / [`.json`](file:///d:/SmartMeter/artifacts/manifests/post_build_integrity_audit.json)
- Household split specification: [`artifacts/manifests/kettle_household_split.md`](file:///d:/SmartMeter/artifacts/manifests/kettle_household_split.md) / [`.json`](file:///d:/SmartMeter/artifacts/manifests/kettle_household_split.json)
- Model dataset specification: [`artifacts/manifests/kettle_model_dataset.md`](file:///d:/SmartMeter/artifacts/manifests/kettle_model_dataset.md) / [`.json`](file:///d:/SmartMeter/artifacts/manifests/kettle_model_dataset.json)
- Simple 1D CNN Baseline experiment report: [`artifacts/experiments/cnn_baseline/cnn_baseline_report.md`](file:///d:/SmartMeter/artifacts/experiments/cnn_baseline/cnn_baseline_report.md) / [`.json`](file:///d:/SmartMeter/artifacts/experiments/cnn_baseline/metrics.json)
- 1D ResNet Reference configuration: [`configs/resnet_reference.yaml`](file:///d:/SmartMeter/configs/resnet_reference.yaml)

## Known Constraints

- small REFIT household count (19 usable evaluation households, 14 positive for Kettle localization);
- CPU/runtime limits;
- Colab free-tier GPU/runtime limits;
- strict final-test protection (H2, H13 untouched);
- weak-supervision label constraint (metadata presence proxy, not physical absence claim);
- validation split has single weak class (H4, H17 both positive; validation model selection requires strong evaluation metrics at fixed threshold).

## Current Technical Decision

Decision D-008 completed: Simple 1D CNN baseline training logged. 1D ResNet reference model implemented with distinct artifact namespace, strict test isolation, and shared training infrastructure.

## Next Technical Decision

Execution of the full 1D ResNet reference model training run and logging of validation strong-target metrics.

## Last Updated

1D ResNet reference model implemented, smoke-tested, preflight-verified, and validated via 62/62 passing unit and smoke tests.


