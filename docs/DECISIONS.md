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

## D-009 — 1D ResNet Reference Model Implementation & Binary Adaptation

Date: 2026-09-24

Decision: Implement and evaluate a 1D ResNet reference model for weakly supervised Kettle presence detection, defined with 3 residual blocks (`[64, 128, 128]` channels, `{8, 5, 3}` kernel pattern), Global Average Pooling, and a single output logit trained via `BCEWithLogitsLoss`.

Reason: Establishes a standard deep convolutional time-series reference baseline with expanded receptive field prior to temporal multi-scale modeling and localization extraction.

Evidence: Completed Colab GPU training run on 2026-09-24 (`artifacts/experiments/resnet_reference/metrics.json`), achieving best validation window-level F1 of 0.3488 and balanced accuracy of 0.5399 at epoch 4 (Precision 21.36%, Recall 94.97%).

Alternatives considered: Direct multi-scale ensemble without single-model baseline (rejected to maintain controlled experimental step).

Impact: Confirmed baseline window classification performance and verified compatibility of training pipeline on GPU.

Status: Approved

## D-010 — Canonical Source Control & Three-Tier Environment Architecture

Date: 2026-09-24

Decision: Establish strict three-tier project environment boundaries:
1. **Canonical Source Tree:** `D:\SmartMeter` synchronized to GitHub (`nayanojwal0810/smartmeter.git`). Normal Git tracks code, configs, tests, documentation, and lightweight metadata.
2. **Compute Environment:** Google Colab is strictly an ephemeral GPU execution engine, running canonical Git commits against verified data snapshots.
3. **Large Binary Storage:** Processed `.npy` arrays, large `.jsonl` target manifests, and PyTorch model checkpoints (`*.pt`) reside on local disk and Google Drive, strictly excluded from Git.

Reason: Prevents code divergence across environments, eliminates accidental repository bloat, and provides immutable experiment reproducibility.

Evidence: Verified repository `.gitignore` and clean extraction validation from data snapshots.

Alternatives considered: Git LFS (deferred to avoid vendor lock-in and quota limits), committing processed datasets directly to Git (rejected due to size).

Impact: Governs all future experiment execution, data sharing, and code synchronization.

Status: Approved

## D-011 — Sealed Kettle Data Snapshot & Validation Target Isolation

Date: 2026-09-24

Decision: Package an immutable, self-contained data snapshot (`kettle_train_val_snapshot_2026-09-24.zip`) containing exactly the 18 `.npy` arrays (159,564 windows) for TRAIN and VALIDATION households, accompanied by a validation-only evaluation target manifest (`kettle_validation_targets.jsonl`, 18,561 records for H4 and H17).

Reason: Colab training runs do not require the 6.47 GB raw REFIT archive. Filtering evaluation targets to H4 and H17 ensures sealed test households H2 and H13 cannot be exposed in compute environments.

Evidence: Clean sandbox extraction verification and automated sha256 checksum validation (`dd41f144ceafa516d7b2ba2c0bb42959f7a6e7e7da3cf6a48d96b78da3b8e36a`).

Alternatives considered: Uploading full raw CSVs to Colab (rejected due to transfer overhead), packaging full 38 MB targets manifest (rejected to enforce test isolation).

Impact: Standardizes all subsequent training runs against an immutable dataset snapshot.

Status: Approved

## D-012 — Project Identity & MLOps Platform Scope

Date: 2026-09-24

Decision: Define the project identity as **SmartMeter Appliance Intelligence**, an end-to-end MLOps platform for weakly supervised appliance detection and temporal localization from smart-meter active power. Position research literature (e.g., CamAL) as technical foundation and academic references rather than project branding or a paper reproduction claim.

Reason: Focuses the project on delivering an end-to-end operational software and ML engineering system rather than an academic reproduction, emphasizing engineering rigor, explainability, and production viability.

Evidence: Operational execution plan and updated project architecture.

Alternatives considered: Positioning as an academic CamAL benchmark reproduction (rejected).

Impact: Directs all documentation, README, and resume positioning toward end-to-end MLOps engineering.

Status: Approved

## D-013 — Two-Stage Quality Gates: Gate A (ML Viability) and Gate B (MLOps Completeness)

Date: 2026-09-24

Decision: Structure the remaining project delivery into two sequential, independent quality gates:
- **Gate A (Core ML Viability):** The temporal localization model must achieve validation Localization F1 $\ge 0.50 - 0.60$ with non-degenerate predictions across validation households (H4, H17) before proceeding to production infrastructure.
- **Gate B (MLOps Completeness):** Complete the full MLOps lifecycle (experiment tracking, model registry, FastAPI serving, Docker containerization, CI/CD, monitoring/drift detection, and automated retraining) only around a validated model.

Reason: A polished MLOps layer around an unviable or non-functioning ML model is engineering failure. Verifying ML quality first ensures that the production system delivers genuine utility.

Evidence: Methodological audit of window-level vs. point-level localization metrics.

Alternatives considered: Building the full MLOps stack in parallel with early baselines (rejected to avoid wasted infrastructure work if model fails).

Impact: Governs project milestones and formal halt/proceed criteria.

Status: Approved

## D-014 — Core Model Improvement Policy & Focused Campaign

Date: 2026-09-24

Decision: Limit the core ML development to a focused 5-experiment campaign:
1. Exp 1: Simple 1D CNN baseline (completed).
2. Exp 2: 1D ResNet reference baseline (completed).
3. Exp 3: Multi-scale temporal ResNet ensemble + CAM-based localization ($\text{ResNet}(k)$ with $k \in \{5, 7, 9, 15, 25\}$, CAM extraction, normalization/aggregation, temporal localization).
4. Exp 4: Context-aware / attention weak supervision (temporal feature extraction + attention / MIL pooling over internal temporal regions).
5. Exp 5: Targeted ablation & model freeze.

Reason: Prevents arbitrary hyperparameter hunting. The context-aware attention extension represents an understandable, engineering-driven design improvement grounded in weak supervision literature, keeping every component fully explainable in a technical interview.

Evidence: Time-series multiple-instance learning and temporal convolutional literature.

Alternatives considered: Arbitrary architecture search or hyperparameter sweeps (rejected).

Impact: Freezes the experimental roadmap for Gate A execution.

Status: Approved

## D-015 — Finalized Gate A / Gate B Operational Criteria and Metric Standards

Date: 2026-09-24

Decision: Lock the operational execution criteria and metric standards for the project:
1. **Explicit Metric Distinction:** Window classification F1 evaluates binary window-level detection for baseline diagnostics. Temporal Localization F1 evaluates point-by-point and event-level timeline segmentation against strong ground truth and is the exclusive criterion for Gate A.
2. **Gate A Decision Rules:**
   - PASS: Validation Localization F1 $\ge 0.50$ (Preferred: $\approx 0.55 - 0.60+$).
   - BORDERLINE: $0.40 \le \text{Localization F1} < 0.50$ (NOT a pass; authorizes at most one targeted diagnostic based on empirical evidence).
   - FAIL: Localization F1 $< 0.40$ (halts development; no full MLOps built around an unviable model).
3. **MLOps Promotion Policy:** Automated retraining and challenger evaluation are supported, but production promotion is strictly gated by measurable validation criteria. A drift alert alone does not deploy a model. Rollback and rejection are required.
4. **Latency Measurement Standard:** Remove arbitrary fixed latency thresholds. Measure and report empirical inference latency (p50 and p95) on documented test hardware and representative request sizes.

Reason: Establishes clear, objective engineering quality gates and prevents building complex production infrastructure around sub-standard ML models.

Evidence: NILM evaluation literature, MLOps operational standards, and project review decisions.

Alternatives considered: Parallel MLOps build before model validation (rejected), arbitrary latency thresholds (rejected).

Impact: Defines exact acceptance criteria for Experiment 3 through Gate B completion.

Status: Approved

## D-016 — Lock Reference CamAL Localization Pipeline and Point-Level Localization F1 Standard for Experiment 3

Date: 2026-09-24 (Revised 2026-09-25)

Decision: Formally lock the reference temporal localization extraction pipeline and metric evaluation protocol for Experiment 3:
1. **Multi-Scale Temporal ResNet Ensemble:** Train independent 1D ResNet models across kernel scales $k \in \{5, 7, 9, 15, 25\}$.
2. **Distinct Input Representations:**
   - **Neural-Network Model Input:** $x_{\text{model}}(t) = \frac{X(t) - \mu_{\text{train}}}{\sigma_{\text{train}}}$ using train-only z-score standardization (Decision D-007) for stable ResNet detection and CAM extraction.
   - **CAM Attention Input:** $x_{\text{attn}}(t) = \frac{X(t)}{1000}$ using nonnegative aggregate power scaled to Kilowatts ($X(t) \ge 0$), following the input scaling described by CamAL (*Petralia et al., ICDE 2025, Section V-B*).
   - *Rationale:* Using z-score input in the attention stage would allow negative CAM $\times$ negative input to produce positive activations, distorting localization. Using nonnegative $x_{\text{attn}}(t) \ge 0$ preserves the intended attention property.
   - *Attribution:* This is an explicit project implementation adaptation.
3. **Reference CamAL Localization Sequence:**
   a. Execute each $\text{ResNet}(k)$ on the 510-point standardized input window $x_{\text{model}}(t)$ (4,072-second timestamp span: $(510-1) \times 8 = 4,072\text{ s}$).
   b. Compute the ensemble detection probability: $P_{\text{ens}} = \frac{1}{M}\sum_{m=1}^M \sigma(\text{logit}_m)$.
   c. Apply the ensemble detection gate at fixed threshold $\tau_{\text{det}} = 0.50$.
   d. For detected windows ($P_{\text{ens}} \ge \tau_{\text{det}}$), extract class-1 CAM from the final convolutional layer of each model before Global Average Pooling: $CAM^{(m)}(t) = \sum_{c=1}^C w_c^{(m)} A_c^{(m)}(t)$.
   e. Normalize each CAM using the reference division: $CAM_{\text{norm}}^{(m)}(t) = \frac{CAM^{(m)}(t)}{\max_t CAM^{(m)}(t)}$. If $\max_t CAM^{(m)}(t) \le 0$, the normalized CAM is strictly zero across all timestamps. (The published normalization formula does not introduce an explicit ReLU/clamping step. Because the published method describes the normalized CAM as $[0, 1]$, the implementation must preserve the intended nonnegative attention behavior; our project therefore does not add an unverified ReLU operation).
   f. Average normalized CAMs across the ensemble: $CAM_{\text{ens}}(t) = \frac{1}{M} \sum_{m=1}^M CAM_{\text{norm}}^{(m)}(t)$.
   g. Apply attention multiplication on nonnegative kW-scaled power $x_{\text{attn}}(t)$ and compute reference sigmoid activation: $S(t) = \sigma\left(CAM_{\text{ens}}(t) \cdot x_{\text{attn}}(t)\right)$.
   h. Convert continuous activation $S(t)$ to binary temporal states with deterministic zero-power suppression:
      - If $X(t) == 0 \implies \hat{y}_t = 0$.
      - For $X(t) > 0 \implies \hat{y}_t = 1 \text{ if } S(t) \ge 0.50 \text{ else } 0$.
      *(Edge-case rationale: Since $\sigma(0) = 0.50$, timestamps with $X(t) = 0$ would otherwise produce false positive localizations at threshold $\ge 0.50$ regardless of CAM attribution; zero-power points are deterministically suppressed).*
   i. Undetected windows ($P_{\text{ens}} < \tau_{\text{det}}$) produce all-zero localization states ($\hat{y}_t = 0$ for all $t \in [1, 510]$).
4. **CAM Alignment & Resolution:** All 1D ResNet blocks use `stride=1` and `padding="same"`, ensuring the final feature map $A_c(t)$ naturally retains the exact 510-point temporal resolution ($T=510$). No temporal interpolation or resizing is required.
5. **Segment-Bounded Timeline Stitching:** Window predictions stitch contiguously within continuous recording segments without crossing forbidden segment boundaries (raw gaps $>16$s). Unwindowed tail points ($<510$ points) receive default inactive state 0.
6. **Primary Metric (Gate A Acceptance):** Point-Level Localization F1 computed across all 9,466,110 validation points ($18,561 \times 510$) against strong ground-truth Kettle target masks ($P_{\text{kettle}} \ge 1500\text{ W}$). Gate A threshold: $\text{Localization F1} \ge 0.50$.
7. **Secondary Metric (Operational Reporting):** Segment-level stitched deterministic 1-to-1 event matching at Temporal $\text{IoU} \ge 0.50$.

Reason: As documented in the Weak-Supervision Formulation Audit (`AUDIT-E3-001`), 80.08% of weak-positive validation windows contain no Kettle activity, and the active Kettle boil accounts for only 3.02% of an active window. Standard Global Average Pooling creates BCE pressure on generic background power, whereas the reference CamAL pipeline leverages multi-scale feature maps, CAM attention masking on nonnegative $x_{\text{attn}}$, and sigmoid activation to achieve sub-meter temporal localization.

Evidence: Pre-E3 Weak-Supervision & Formulation Audit (`docs/WEAK_SUPERVISION_FORMULATION_AUDIT.md`), CamAL ICDE 2025 publication (arXiv:2506.05895), and empirical baseline findings.

Alternatives considered: Direct window-level classification thresholding (rejected due to 80% background false positives), longer temporal context (rejected due to further burst dilution), explicit forward-graph MIL/attention pooling (reserved for Experiment 4).

Impact: Governs Experiment 3 implementation, evaluation pipeline, and Gate A acceptance.

Status: Approved (Formulation Locked for E3 Implementation)









