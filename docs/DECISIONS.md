# Technical Decisions

Record only material technical decisions.

Do not use this file as a task log.

## Decision Format

```text
## D-XXX — Short decision title

Date:
Decision:
Reason:
Evidence:
Alternatives considered:
Impact:
Status:
```

## D-001 — Use cleaned REFIT as the primary dataset

Date: Initial setup

Decision: Use the cleaned REFIT Electrical Load Measurements dataset as the primary project dataset.

Reason: It provides aggregate and appliance-level measurements for household-safe evaluation and is directly relevant to the project objective.

Evidence: Official University of Strathclyde and Zenodo dataset records.

Alternatives considered: None selected.

Impact: All data preparation and evaluation work starts from the cleaned REFIT release.

Status: Approved

## D-002 — Preserve weak supervision for the main model

Date: Initial setup

Decision: Train the main weakly supervised model from household-level appliance-presence information rather than timestamp-level appliance labels.

Reason: This preserves the central research setting and prevents the project from becoming a conventional fully supervised localization system.

Evidence: Project research foundation and execution specification.

Alternatives considered: Fully supervised timestamp-level training.

Impact: Strong appliance measurements remain evaluation/audit data.

Status: Approved

## D-003 — Protect final household holdout

Date: Initial setup

Decision: Use household-level separation and protect final test households from development decisions.

Reason: Window-level random splitting can expose the model to the consumption patterns of the same household and produce misleadingly optimistic results.

Evidence: Project evaluation policy.

Alternatives considered: Random window splitting.

Impact: Split must be defined before final development decisions and stored in versioned configuration.

Status: Approved

## D-004 — Delay full MLOps implementation

Date: Initial setup

Decision: Build serving, tracking, monitoring, and retraining layers only after the core ML system demonstrates credible localization behavior.

Reason: Infrastructure cannot compensate for a weak model.

Evidence: Project quality policy.

Alternatives considered: Building the full stack immediately.

Impact: Early work focuses on data quality, model validity, and evaluation.

Status: Approved

## D-005 — Lock Kettle ground-truth target policy

Date: Current release

Decision: Lock the Kettle ground-truth target policy for the current project release as follows:

- Target appliance: Kettle.
- ON threshold: Kettle power >= 1500 W (pre-model evaluation target definition selected from the target-quality audit, not model-tuned).
- Event continuity: Consecutive active Kettle samples belong to the same event when the timestamp gap is <= 20 seconds.
- Maximum event duration: 600 seconds (10 minutes). Candidate events longer than 600 seconds are treated as target-quality anomalies and excluded from localization ground truth.
- Household handling:
  - H12: Exclude from primary Kettle localization evaluation because the monitored Kettle channel provides insufficient and internally inconsistent target evidence.
  - H3: Retain; record documented Kettle replacement date of 16 Apr 2014.
  - H13: Retain; exclude candidate events longer than 600 seconds from localization ground truth.
  - H17: Retain; document shared Kettle/toaster IAM contamination.
  - H19: Retain; record observed Kettle-versus-Aggregate discrepancy without claiming unsubstantiated causes.
  - H11: Retain; document solar-affected aggregate measurements as a distribution/data-quality consideration.
  - H21: Retain; document both shared Kettle/toaster IAM contamination and solar-affected aggregate behavior.
  - H1, H10, H15, H16, H18: Record as households with no monitored Kettle IAM in the dataset metadata (do not describe as proof of physical absence).

Reason: The target-quality audit found clear, short Kettle activation patterns in comparison households and identifiable sensor/shared-channel anomalies in specific households. The approved policy provides a consistent pre-model evaluation target without using model results to tune it.

Evidence: `artifacts/reports/kettle_target_audit.md` and `artifacts/reports/kettle_target_audit.json`.

Alternatives considered: 500 W generic threshold, per-household dynamic thresholds, fully supervised window labeling.

Impact: Establishes a fixed ground-truth target definition for all future localization evaluation, error analysis, and metric calculation while keeping weak supervision intact for model training.

Status: Approved

## D-006 — Lock regular timebase and window generation policy

Date: Current release

Decision: Lock the regular timebase resampling and reference window generation policy as follows:
- Regular timebase: 8-second regular grid.
- Contiguous segments:
  - Each contiguous recording segment anchors its 8-second grid at the segment's first valid observation.
  - Grid timestamps advance deterministically by exactly 8 seconds.
  - Gaps <= 16 seconds are bridged using zero-order hold (forward-fill from the last valid observation).
  - A gap > 16 seconds starts a new contiguous segment.
  - Never interpolate, extrapolate, or forward-fill across a > 16-second boundary.
- Model window definition:
  - Initial/reference model window length: 510 grid points (temporal span: (510 - 1) * 8 = 4,072 seconds / ~67.87 minutes).
  - Non-overlapping windows for the initial reference dataset.
  - A window is valid only if all 510 grid points belong to the same contiguous recording segment (never create a window across segment boundaries).
- Target alignment:
  - When the locked Kettle evaluation target is projected onto the model grid, it uses the exact same 8-second grid and contiguous-segment boundaries.

Reason: The full dataset audit demonstrates no duplicate or reversed timestamps, most observed gaps are <= 15 seconds, and large multi-day recording gaps exist. A deterministic regular timebase and segment boundary policy is required before model-window generation.

Evidence: Existing REFIT forensic audit artifact (`artifacts/reports/refit_data_audit.md`) and documented project target policy.

Alternatives considered: Linear interpolation across large gaps, variable-length windows, windowing across recording outages.

Impact: Establishes deterministic input tensor shapes and strict leakage-free segment boundaries for baseline CNN and ResNet training.

Status: Approved

## D-007 — Lock primary household train/validation/test split

Date: Current release

Decision: Freeze and lock the primary household-isolated train/validation/test split across all 20 processed REFIT households as follows:
- **TRAIN (16 households):** H1, H3, H5, H6, H7, H8, H9, H10, H11, H12, H15, H16, H18, H19, H20, H21 (141,003 windows / 81.00%)
- **VALIDATION (2 households):** H4, H17 (18,561 windows / 10.66%)
- **TEST (2 households):** H2, H13 (14,508 windows / 8.33%)

Enforce strict household-level isolation rules:
1. Every household is assigned strictly to a single split. All 510-point windows unconditionally inherit their household's split assignment.
2. Zero random window splitting, zero segment-level splitting, and zero timestamp-level splitting.
3. Weak training labels are derived exclusively from metadata-level Kettle presence (1 = indicated presence, 0 = unmonitored), strictly separated from strong sub-meter targets.
4. Strong timestamp-level Kettle targets are never used to decide, tune, or modify the split.
5. The holdout test split (H2, H13) is frozen before model training and must never be accessed for model training, architecture selection, hyperparameter tuning, threshold selection, or early stopping.
6. H12 provides weak presence supervision (1) during training, but remains strictly excluded from primary localization evaluation (`is_evaluation_eligible=False`).
7. Unmonitored households (H1, H10, H15, H16, H18) reside in TRAIN to provide negative weak supervision without fabricating timestamp-level absence labels.

Reason: In NILM and weak-supervised appliance localization, random window-level splitting causes catastrophic data leakage due to contiguous background load patterns and repeated appliance signatures. Strict household isolation is mandatory for generalization evaluation.

Evidence: Documented REFIT household metadata (`data/raw/MetaData_Tables.xlsx`) and audited 174,072-window manifest (`artifacts/manifests/kettle_household_split.md`).

Alternatives considered: Random window splitting (rejected due to severe temporal leakage), leave-one-house-out cross-validation (deferred to future benchmark phase).

Impact: Freezes the training, validation, and holdout test partitions before model construction.

Status: Approved

## D-008 — Simple 1D CNN Baseline Architecture & Validation Policy

Date: Current release

Decision: Implement a deliberately minimal 1D CNN baseline for weakly supervised Kettle presence detection on 510-point aggregate electricity consumption windows, defined as follows:

1. **Architecture:**
   - Input: `(B, 1, 510)` (aggregate power in Watts, float32)
   - Layer 1: `Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=0)`
   - Layer 2: `ReLU`
   - Layer 3: `MaxPool1d(kernel_size=2)`
   - Layer 4: `Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding=0)`
   - Layer 5: `ReLU`
   - Layer 6: `AdaptiveAvgPool1d(output_size=1)`
   - Layer 7: `Flatten`
   - Layer 8: `Linear(in_features=64, out_features=1)`
   - Output: Raw scalar logit per window; Sigmoid applied only for probability estimation / reporting.
   - Prohibitions: Strictly NO BatchNorm, Dropout, attention mechanisms, residual blocks, multi-branch kernels, CAM/localization logic, class weighting, or learning rate schedulers in this baseline.

2. **Training Supervision & Loss:**
   - Loss function: `torch.nn.BCEWithLogitsLoss()`.
   - Supervision target: Binary household-level weak label $y \in \{0, 1\}$ from REFIT metadata (1 = monitored Kettle presence, 0 = unmonitored proxy).
   - Weak label 0 is strictly an unmonitored proxy and NOT a claim of physical appliance absence.
   - Strong sub-meter measurements and window evaluation targets are strictly excluded from model inputs and training loss.

3. **Normalization:**
   - Global train-derived standardization: $x_{\text{normalized}} = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$.
   - Mean and standard deviation are computed exclusively by streaming across the 16 TRAIN households.
   - Validation and test data are strictly excluded from normalization fitting.

4. **Validation Evaluation & Fixed Threshold:**
   - Because validation weak labels are single-class (100% positive for H4, H17), weak-label accuracy is non-discriminative and forbidden for model selection.
   - Validation diagnosis is computed against the delayed strong Kettle evaluation target manifest (`kettle_evaluation_targets.jsonl`) keyed by `window_id`.
   - Reported metrics: Precision, Recall, F1-Score, Balanced Accuracy, Accuracy.
   - Decision threshold is fixed at $0.5$ (default) without validation-based threshold optimization in this baseline.
   - A non-ML majority-class reference baseline (predicting inactive for all validation windows) is reported as a benchmark reference.

5. **Test Set Isolation:**
   - Holdout test households H2 and H13 are sealed and forbidden from all training, normalization, hyperparameter selection, threshold tuning, and early stopping.

Reason: A minimal sanity baseline is essential to verify that the weakly supervised learning setup produces non-trivial signal on aggregate power before introducing complex architectures (ResNet), activation maps (CAM), or ensembling.

Evidence: Documented REFIT split manifest (`artifacts/manifests/kettle_household_split.json`) and verified test suite.

Alternatives considered: Direct implementation of ResNet-1D / CAM (deferred to subsequent milestone after baseline validation).

Impact: Establishes the first reproducible ML benchmark for weakly supervised kettle detection on REFIT.

Status: Approved



