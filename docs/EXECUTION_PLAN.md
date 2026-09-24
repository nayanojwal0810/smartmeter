# SmartMeter Appliance Intelligence & Localization Platform

## Project Objective

Build an independently implemented ML system that uses aggregate household electricity consumption to estimate when a selected appliance was active, using weak household-level appliance-presence information for model training and timestamp-level appliance measurements only for evaluation and audit.

The project is inspired by the published CamAL research approach. We do not claim algorithmic novelty.

## Research Foundation

Petralia, A., Boniol, P., Charpentier, P., Palpanas, T. (2025).
*Few Labels are All You Need: A Weakly Supervised Framework for Appliance Localization in Smart-Meter Series.*
IEEE ICDE 2025.

- Paper: https://arxiv.org/abs/2506.05895
- Official implementation: https://github.com/adrienpetralia/CamAL
- DOI: 10.1109/ICDE65448.2025.00329

The implementation is independent. The paper and repository are technical references, not source code to copy wholesale.

## Dataset

Primary dataset: **REFIT: Electrical Load Measurements (Cleaned)**.

- University of Strathclyde record:
  https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/
- Zenodo record:
  https://zenodo.org/records/5063428
- DOI: 10.5281/zenodo.5063428
- License: CC BY 4.0

The cleaned release contains aggregate and appliance-level electricity measurements for 20 households at 8-second sampling.

The dataset must be audited before model development. The audit determines the usable appliance candidate, household coverage, sampling behavior, missingness, and other data-quality constraints.

## ML Objective

The core ML story is:

```text
aggregate household power
        ↓
weakly supervised classification
        ↓
time-series 1D ResNet
        ↓
ensemble
        ↓
CAM-based temporal localization
        ↓
appliance activity timeline
```

The first appliance is **Kettle**, approved and locked under Decision D-005 following the forensic data audit.

The pre-model evaluation ground truth is locked as:
- ON threshold: `Kettle power >= 1500 W`
- Event continuity: timestamp gap `<= 20 s`
- Duration ceiling: candidate events `> 600 s` (10 min) excluded as sensor anomalies
- Household eligibility: H12 excluded from primary localization evaluation because the monitored Kettle channel provides insufficient and internally inconsistent target evidence; H3, H13, H17, H19, H11, H21 retained with documented constraints.

## Timebase and Window Policy

The regular timebase and reference model window are locked under Decision D-006:

- **Regular Timebase:** 8-second regular grid.
- **Contiguous Segments:**
  - Anchor 8-second grid at the first valid timestamp of each contiguous segment.
  - Gaps `<= 16 s` are forward-filled using zero-order hold.
  - Gaps `> 16 s` start a new contiguous recording segment; never interpolate across segment boundaries.
- **Model Windows:**
  - Length: **510 grid points** (4,080 seconds / 68 minutes).
  - Structure: Non-overlapping windows for initial reference dataset.
  - Validity: All 510 grid points must reside strictly within the same contiguous segment.

## Label Policy

Timestamp-level appliance measurements are reserved for:

- data auditing;
- evaluation-target construction;
- localization evaluation;
- error analysis;
- final reporting.

They must not be used to train the main weakly supervised model, tune thresholds, select architectures, fit normalization statistics, or make model-selection decisions.

This separation must be visible in the data manifests and code.

## Evaluation Policy

Primary evaluation is household-safe.

- Split by household before generating windows.
- Freeze the household split before model development decisions begin.
- Keep final test households isolated from tuning and repeated experimentation.
- Use validation data for architecture, hyperparameter, threshold, and calibration decisions.
- Report window-level detection metrics and timestamp-level localization metrics separately.
- Retain per-household results so aggregate metrics do not hide failure cases.

At minimum, report:

- Precision
- Recall
- F1
- Balanced Accuracy
- timestamp-level localization Precision/Recall/F1 where applicable

Infrastructure measurements such as latency, throughput, model size, and training time are reported separately from ML quality.

## Model Development

The intended model progression is:

### Baseline CNN

Purpose: establish that the data formulation contains learnable signal and provide a reference point.

Acceptance evidence:

- reproducible training;
- sensible learning behavior;
- performance above a trivial baseline;
- no leakage;
- inspectable predictions.

### Single ResNet

Implement a 1D ResNet-style classifier based on the published design:

- three residual blocks;
- three convolutional blocks per residual block;
- filter groups `{64, 128, 128}`;
- convolution pattern `{8, 5, 3}`;
- global average pooling;
- linear classification layer.

The implementation must be independently written and understood.

### Ensemble and Localization

Start with a small ensemble to validate the mechanism, then expand only when compute and results justify it.

Target ensemble:

```text
k = {5, 7, 9, 15, 25}
```

Detection probabilities are combined across models.

CAMs are extracted, normalized, aggregated, and passed through the agreed localization processing before producing the activity timeline.

The reference 0.5 threshold is a baseline reference only. The final threshold must be selected using development evidence and frozen before final test evaluation.

## Controlled Experiments

Experiments must answer a defined question.

Core questions include:

- Does the CNN learn useful signal?
- Does the ResNet improve over the CNN?
- Does an ensemble improve localization?
- Do different receptive fields help?
- Does the attention/sigmoid localization processing help?
- Does temporal context affect results?
- Does threshold calibration improve the validation result?

Do not run broad parameter sweeps without a clear research question.

Each meaningful experiment should preserve:

```text
question
setup
result
interpretation
artifact
```

## Production-Oriented Extension

The engineering target, after the core ML system is credible, is:

```text
validated data
    ↓
reproducible preprocessing
    ↓
training + experiment tracking
    ↓
versioned model
    ↓
batch inference
    ↓
API serving
    ↓
monitoring
    ↓
drift/performance review
    ↓
candidate retraining
```

MLOps components are not built early just for appearance. Core ML quality comes first.

## Colab / GPU Strategy

CPU and quota limits are project constraints.

When training or another operation is likely to exceed practical local runtime:

1. Prepare the exact code and configuration.
2. Package a clean Colab bundle under `colab/` using the same project source components.
3. Provide the exact command/notebook execution instructions.
4. Stop.
5. The User runs the expensive job in Colab and returns the output.
6. Continue from the returned evidence.

Do not create a second, diverging implementation for Colab.

## Project Quality Standard

The project is successful when it has:

- a defensible weak-supervision formulation;
- strong leakage controls;
- credible unseen-household localization results;
- justified experimental improvements;
- reproducible artifacts;
- clear documentation;
- a practical serving path;
- traceable model/version history;
- monitoring and retraining evidence where implemented;
- resume claims backed by stored experiment evidence.

The goal is a strong, defensible project, not maximum feature count.

## Repository Direction

The repository should grow only as needed.

Expected areas:

```text
configs/
data/
src/
scripts/
tests/
experiments/
artifacts/
docs/
colab/
```

Raw data remains outside version control unless explicitly required.

Generated experiment evidence must not be overwritten.

## Completion Standard

Final project claims must be traceable to:

```text
dataset version
+ household split
+ evaluation-target definition
+ configuration
+ code version
+ seed
+ model artifact
+ reported result
```

No final metric is valid without an inspectable source artifact.

## Immediate Authorized Work

Following completion of the forensic data audit and locking of the Kettle ground-truth target policy (D-005), immediate work progresses to:

- define and test timestamp resampling and contiguous windowing strategy;
- formally establish and freeze the household train/validation/test split;
- prepare the baseline 1D CNN pipeline;
- stop for technical review before model training execution.

The first model is **not** authorized until the windowing policy and household split have been formally reviewed and accepted.
